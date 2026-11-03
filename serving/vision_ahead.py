import os
import time
from collections import OrderedDict

ENV = "TAAF_VLLM_VISION_AHEAD"
CACHE_ENV = "TAAF_VLLM_VISION_CACHE"
EVERY_S = 60.0
MIB = 1 << 20
TOKENS_CAP = 1_000_000
HELD_CAP = 100_000
_STATE = {"sched": None, "work": None}


def enabled():
    return str(os.environ.get(ENV) or "").strip() == "1"


def cache_mib():
    raw = str(os.environ.get(CACHE_ENV) or "").strip()
    try:
        mib = int(raw) if raw else 0
    except ValueError:
        return 0
    return mib if mib > 0 else 0


def say(line):
    print(line, flush=True)


class Mirror:
    def __init__(self, budget_tokens):
        self.budget = int(budget_tokens)
        self.held = OrderedDict()
        self.tokens = 0

    def store(self, key, tokens):
        tokens = int(tokens)
        if key in self.held:
            self.tokens -= self.held.pop(key)
        if tokens <= 0 or tokens > self.budget:
            return False
        while self.held and self.tokens + tokens > self.budget:
            _, old = self.held.popitem(last=False)
            self.tokens -= old
        self.held[key] = tokens
        self.tokens += tokens
        return True

    def take(self, key):
        tokens = self.held.pop(key, None)
        if tokens is None:
            return False
        self.tokens -= tokens
        return True


class Scheduling:
    def __init__(self, mirror=None, out=say, clock=time.monotonic, every_s=EVERY_S):
        self.mirror = mirror
        self.out = out
        self.clock = clock
        self.every_s = every_s
        self.last = clock()
        self.window = 0
        self.embeds = OrderedDict()
        self.held = OrderedDict()
        self.reset()

    def reset(self):
        self.counts = {"holds": 0, "held_tokens": 0, "fallbacks": 0, "mirror_hits": 0, "scheduled": 0,
                       "freed": 0, "repeats": 0}

    def remember(self, key, tokens):
        self.embeds[key] = int(tokens)
        self.embeds.move_to_end(key)
        if len(self.embeds) > TOKENS_CAP:
            self.embeds.popitem(last=False)

    def hold(self, features, inputs, num_computed, num_new, shift, request_id=None):
        starts = []
        for index in inputs:
            feature = features[index]
            key = feature.identifier
            self.remember(key, feature.mm_position.get_num_embeds())
            self.counts["scheduled"] += 1
            if self.mirror is not None and self.mirror.take(key):
                self.counts["mirror_hits"] += 1
                continue
            starts.append(int(feature.mm_position.offset))
        self.tick()
        if not starts:
            return num_new
        keep = min(starts) - (int(num_computed) + int(shift))
        if keep <= 0:
            self.counts["fallbacks"] += 1
            return num_new
        if keep >= num_new:
            return num_new
        mark = (int(num_computed), min(starts))
        if request_id is not None:
            if self.held.get(request_id) == mark:
                self.counts["repeats"] += 1
                self.held.pop(request_id, None)
                return num_new
            self.held[request_id] = mark
            self.held.move_to_end(request_id)
            if len(self.held) > HELD_CAP:
                self.held.popitem(last=False)
        self.counts["holds"] += 1
        self.counts["held_tokens"] += num_new - keep
        return keep

    def freed(self, hashes):
        for key in hashes:
            self.counts["freed"] += 1
            if self.mirror is not None:
                tokens = self.embeds.get(key)
                if tokens is not None:
                    self.mirror.store(key, tokens)
        return hashes

    def line(self):
        mirror = self.mirror
        return (f"vision_ahead_sched window={self.window} scheduled={self.counts['scheduled']} "
                f"holds={self.counts['holds']} held_tokens={self.counts['held_tokens']} "
                f"fallbacks={self.counts['fallbacks']} mirror_hits={self.counts['mirror_hits']} "
                f"repeats={self.counts['repeats']} freed={self.counts['freed']} mirror_entries={len(mirror.held) if mirror else 0} "
                f"mirror_tokens={mirror.tokens if mirror else 0}")

    def tick(self):
        now = self.clock()
        if now - self.last >= self.every_s:
            self.out(self.line())
            self.window += 1
            self.last = now
            self.reset()


def _embed_bytes(scheduler):
    config = getattr(getattr(scheduler, "vllm_config", None), "model_config", None)
    try:
        width = int(config.get_inputs_embeds_size())
    except Exception:
        width = int(getattr(getattr(config, "hf_text_config", None), "hidden_size", 0) or 0)
    try:
        item = int(config.dtype.itemsize)
    except Exception:
        item = 2
    return width * item


def scheduling(scheduler):
    if not enabled():
        return None
    held = _STATE["sched"]
    if held is None:
        mirror = None
        mib = cache_mib()
        per = _embed_bytes(scheduler)
        if mib and per > 0:
            mirror = Mirror(mib * MIB // per)
        held = _STATE["sched"] = Scheduling(mirror)
        say(f"VISION_AHEAD armed in the scheduler: a prefill chunk that reaches an image no cache holds stops "
            f"just before it and the image is encoded this step, so the next step prefills it from the encoder "
            f"cache; a mirror of the {mib} MiB worker cache "
            f"({'off' if mirror is None else f'{mirror.budget} tokens at {per} bytes a token'}) skips the hold "
            f"for an image the worker will restore; one vision_ahead_sched line a minute")
    return held


def hold(scheduler, request, num_computed, num_new, shift, inputs):
    if not inputs or getattr(scheduler, "is_encoder_decoder", False):
        return num_new
    state = scheduling(scheduler)
    if state is None:
        return num_new
    return state.hold(request.mm_features, inputs, num_computed, num_new, shift,
                      getattr(request, "request_id", None))


def freed(scheduler, hashes):
    state = scheduling(scheduler)
    if state is not None:
        state.freed(hashes)
    return hashes


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _on_device(tensor):
    return bool(getattr(tensor, "is_cuda", False))


def _tensors(value):
    if isinstance(value, dict):
        for one in value.values():
            yield from _tensors(one)
    elif isinstance(value, (list, tuple)):
        for one in value:
            yield from _tensors(one)
    elif hasattr(value, "record_stream") and _on_device(value):
        yield value


def _key(tensor):
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return id(tensor)


class Working:
    def __init__(self, torch=None, stream=None, out=say, clock=time.monotonic, every_s=EVERY_S):
        self.torch = torch
        self.stream = stream
        self.out = out
        self.clock = clock
        self.every_s = every_s
        self.last = clock()
        self.window = 0
        self.pending = {}
        self.timings = []
        self.marks = []
        self.reset()

    def reset(self):
        self.counts = {"steps": 0, "ahead_steps": 0, "ahead_items": 0, "now_items": 0, "waits": 0,
                       "inline": 0, "encoder_ms": 0.0, "timed": 0}
        self.step_ms = {True: [0.0, 0], False: [0.0, 0]}

    def split(self, encoder_cache, scheduled, batch):
        rows = {req_id: index for index, req_id in enumerate(batch.req_ids)}
        now, ahead = {}, {}
        for req_id, ids in scheduled.items():
            index = rows.get(req_id)
            if index is None:
                now[req_id] = list(ids)
                continue
            end = int(batch.num_computed_tokens_np[index]) + int(batch.num_scheduled_tokens[index])
            features = encoder_cache.mm_features[req_id]
            for input_id in ids:
                target = ahead if int(features[input_id].mm_position.offset) >= end else now
                target.setdefault(req_id, []).append(input_id)
        return now, ahead

    def encode_ahead(self, state, ahead):
        runner = state.encoder_runner
        hashes, kwargs = runner.prepare_mm_inputs(ahead)
        if not kwargs:
            return 0
        torch = self.torch
        side = self.stream
        if torch is None or side is None:
            outputs = runner.execute_mm_encoder(kwargs)
            state.encoder_cache.encoder_outputs.update(zip(hashes, outputs))
            self.counts["inline"] += len(hashes)
            return len(hashes)
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        for tensor in _tensors(kwargs):
            tensor.record_stream(side)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            start.record(side)
            outputs = runner.execute_mm_encoder(kwargs)
            end.record(side)
        for key, output in zip(hashes, outputs):
            state.encoder_cache.encoder_outputs[key] = output
            self.pending[_key(output)] = (end, output)
        self.timings.append((start, end))
        return len(hashes)

    def settle_tensor(self, tensor):
        if not self.pending or tensor is None:
            return False
        held = self.pending.pop(_key(tensor), None)
        if held is None:
            return False
        event, output = held
        main = self.torch.cuda.current_stream()
        main.wait_event(event)
        output.record_stream(main)
        self.counts["waits"] += 1
        return True

    def settle(self, embeds):
        if not self.pending:
            return 0
        settled = 0
        for embed in embeds or ():
            settled += self.settle_tensor(embed)
        return settled

    def mark(self, carried):
        torch = self.torch
        if torch is None or self.stream is None:
            return
        try:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        except Exception:
            return
        self.marks.append((event, carried))

    def drain(self):
        held = []
        for start, end in self.timings:
            try:
                done = end.query()
            except Exception:
                continue
            if done:
                self.counts["encoder_ms"] += float(start.elapsed_time(end))
                self.counts["timed"] += 1
            else:
                held.append((start, end))
        self.timings = held
        kept = []
        for index in range(len(self.marks) - 1):
            first, carried = self.marks[index]
            second, _ = self.marks[index + 1]
            try:
                done = second.query()
            except Exception:
                continue
            if not done:
                kept = self.marks[index:]
                break
            slot = self.step_ms[bool(carried)]
            slot[0] += float(first.elapsed_time(second))
            slot[1] += 1
        else:
            kept = self.marks[-1:]
        self.marks = kept

    def line(self):
        steps = max(self.counts["steps"], 1)
        ahead = self.step_ms[True]
        plain = self.step_ms[False]
        return (f"vision_ahead window={self.window} steps={self.counts['steps']} "
                f"ahead_steps={self.counts['ahead_steps']} ahead_items={self.counts['ahead_items']} "
                f"now_items={self.counts['now_items']} inline_items={self.counts['inline']} "
                f"hidden_encoder_ms={self.counts['encoder_ms']:.2f} timed_encodes={self.counts['timed']} "
                f"waits={self.counts['waits']} pending={len(self.pending)} "
                f"ahead_step_share={self.counts['ahead_steps'] / steps:.4f} "
                f"step_ms_with_ahead={ahead[0] / max(ahead[1], 1):.2f} (n={ahead[1]}) "
                f"step_ms_without={plain[0] / max(plain[1], 1):.2f} (n={plain[1]})")

    def tick(self):
        now = self.clock()
        if now - self.last >= self.every_s:
            self.drain()
            self.out(self.line())
            self.window += 1
            self.last = now
            self.reset()


def _side(torch):
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        least, _ = torch.cuda.Stream.priority_range()
        return torch.cuda.Stream(priority=least)
    except Exception:
        return None


def working():
    if not enabled():
        return None
    held = _STATE["work"]
    if held is None:
        torch = _torch()
        stream = _side(torch)
        held = _STATE["work"] = Working(torch, stream)
        priority = getattr(stream, "priority", None)
        say(f"VISION_AHEAD armed in the worker: an image whose tokens start past this step's scheduled tokens "
            f"is encoded on a side stream (priority {priority}, the least the device offers) that waits for the "
            f"step's prep and runs beside its forward; the step that gathers it waits on its event, never on "
            f"the host; one vision_ahead line a minute")
    return held


def _capturing(torch):
    try:
        return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def execute(state, scheduled, batch):
    work = working()
    if work is None or _capturing(work.torch):
        return state.execute_mm_encoder(scheduled)
    work.counts["steps"] += 1
    done = 0
    if scheduled:
        now, ahead = work.split(state.encoder_cache, scheduled, batch)
        work.counts["now_items"] += sum(len(ids) for ids in now.values())
        if now:
            state.execute_mm_encoder(now)
        done = work.encode_ahead(state, ahead) if ahead else 0
        if done:
            work.counts["ahead_steps"] += 1
            work.counts["ahead_items"] += done
    work.mark(bool(done))
    work.tick()
    return None


def settle(embeds):
    work = _STATE["work"]
    if work is not None:
        work.settle(embeds)


def released(tensor):
    work = _STATE["work"]
    if work is not None:
        work.settle_tensor(tensor)
