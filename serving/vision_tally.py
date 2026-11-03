import importlib
import os

ENV = "TAAF_VLLM_VISION_TALLY"
EVERY_ENV = "TAAF_VLLM_VISION_TALLY_EVERY"
EVERY = 100
SEEN_CAP = 1_000_000
RUNNER_MODULE = "vllm.v1.worker.gpu.mm.encoder_runner"
RUNNER_CLASS = "EncoderRunner"
STATE_MODULE = "vllm.v1.worker.gpu.model_states.default"
STATE_CLASS = "DefaultModelState"
MARK = "_taaf_vision_tally"
MASK = (1 << 64) - 1
_STATE = {"tally": None}


def enabled():
    return str(os.environ.get(ENV) or "").strip() == "1"


def every():
    try:
        value = int(str(os.environ.get(EVERY_ENV) or EVERY).strip())
    except ValueError:
        return EVERY
    return value if value > 0 else EVERY


def say(line):
    print(line, flush=True)


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _capturing(torch):
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _mixed(input_batch, scheduled):
    try:
        return bool((input_batch.num_computed_tokens_np < input_batch.prefill_len_np).any())
    except Exception:
        return bool(scheduled)


def _tokens(outputs):
    total = 0
    for output in outputs or ():
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape):
            total += int(shape[0])
    return total


class Tally:
    def __init__(self, every_steps=None, out=say, cap=SEEN_CAP):
        self.every = every_steps or every()
        self.out = out
        self.cap = cap
        self.window = 0
        self.seen = {}
        self.pending = None
        self.events = []
        self.reset()

    def reset(self):
        self.steps = 0
        self.mixed_steps = 0
        self.calls = 0
        self.items = 0
        self.tokens = 0
        self.repeats = 0
        self.unhashed_calls = 0
        self.unhashed_items = 0
        self.gpu_ms = 0.0
        self.timed_calls = 0

    def remember(self, identifier):
        key = hash(identifier) & MASK
        if key in self.seen:
            self.seen.pop(key)
            self.seen[key] = True
            return True
        self.seen[key] = True
        if len(self.seen) > self.cap:
            self.seen.pop(next(iter(self.seen)))
        return False

    def prepared(self, hashes):
        self.pending = list(hashes or ())

    def encoded(self, outputs, timing=None):
        hashes, self.pending = self.pending, None
        items = len(outputs or ())
        self.calls += 1
        self.items += items
        self.tokens += _tokens(outputs)
        if hashes is None:
            self.unhashed_calls += 1
            self.unhashed_items += items
        else:
            for identifier in hashes:
                if self.remember(identifier):
                    self.repeats += 1
        if timing is not None:
            self.events.append(timing)

    def drain(self):
        held = []
        for start, end in self.events:
            try:
                done = end.query()
            except Exception:
                continue
            if done:
                try:
                    self.gpu_ms += float(start.elapsed_time(end))
                    self.timed_calls += 1
                except Exception:
                    pass
            else:
                held.append((start, end))
        self.events = held

    def step(self, mixed):
        self.steps += 1
        if mixed:
            self.mixed_steps += 1
        if self.steps >= self.every:
            self.report()

    def line(self):
        items = max(self.items, 1)
        mixed = max(self.mixed_steps, 1)
        return (f"vision_tally window={self.window} steps={self.steps} mixed_steps={self.mixed_steps} "
                f"encoder_calls={self.calls} items={self.items} tokens={self.tokens} repeats={self.repeats} "
                f"repeat_share={self.repeats / items:.4f} unhashed_calls={self.unhashed_calls} "
                f"unhashed_items={self.unhashed_items} items_per_mixed_step={self.items / mixed:.3f} "
                f"tokens_per_item={self.tokens / items:.1f} encoder_ms={self.gpu_ms:.2f} "
                f"timed_calls={self.timed_calls} pending_timings={len(self.events)} "
                f"encoder_ms_per_mixed_step={self.gpu_ms / mixed:.3f} seen={len(self.seen)}")

    def report(self):
        self.drain()
        self.out(self.line())
        self.window += 1
        self.reset()


def _timed(torch):
    if torch is None or not torch.cuda.is_available() or _capturing(torch):
        return None
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end
    except Exception:
        return None


def wrap_runner(runner_class, tally):
    prepare = runner_class.prepare_mm_inputs
    execute = runner_class.execute_mm_encoder
    torch = _torch()

    def prepare_mm_inputs(self, *args, **kwargs):
        result = prepare(self, *args, **kwargs)
        try:
            tally.prepared(result[0])
        except Exception:
            tally.prepared(None)
        return result

    def execute_mm_encoder(self, *args, **kwargs):
        timing = _timed(torch)
        outputs = execute(self, *args, **kwargs)
        if timing is not None:
            try:
                timing[1].record()
            except Exception:
                timing = None
        tally.encoded(outputs, timing)
        return outputs

    runner_class.prepare_mm_inputs = prepare_mm_inputs
    runner_class.execute_mm_encoder = execute_mm_encoder


def wrap_state(state_class, tally):
    original = state_class.prepare_inputs_embeds

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, *args, **kwargs):
        out = original(self, scheduled_encoder_inputs, input_batch, *args, **kwargs)
        tally.step(_mixed(input_batch, scheduled_encoder_inputs))
        return out

    state_class.prepare_inputs_embeds = prepare_inputs_embeds


def install(runner_class=None, state_class=None, out=say):
    if not enabled():
        return None
    if _STATE["tally"] is not None:
        return _STATE["tally"]
    if runner_class is None:
        runner_class = getattr(importlib.import_module(RUNNER_MODULE), RUNNER_CLASS)
    if state_class is None:
        state_class = getattr(importlib.import_module(STATE_MODULE), STATE_CLASS)
    if getattr(runner_class, MARK, False) or getattr(state_class, MARK, False):
        return _STATE["tally"]
    tally = Tally(out=out)
    wrap_runner(runner_class, tally)
    wrap_state(state_class, tally)
    setattr(runner_class, MARK, True)
    setattr(state_class, MARK, True)
    _STATE["tally"] = tally
    out(f"VISION_TALLY armed: {RUNNER_CLASS}.prepare_mm_inputs/execute_mm_encoder and "
        f"{STATE_CLASS}.prepare_inputs_embeds wrapped; one vision_tally line every {tally.every} steps counts the "
        f"images the encoder runs, the ones whose hash it already encoded earlier in the run, their tokens and the "
        f"encoder's GPU time from events read without a sync")
    return tally
