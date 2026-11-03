import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

NO_BLANK_ENV = "TAAF_VLLM_NO_BLANK_LINES"
TALLY_S = 15.0
CAP = 16
WINDOW = 4096
SEARCH = 256
BLOCK = 4096
PPM = 1_000_000
STOP_LINE = -1

OFF, TOOL, THINK_OPEN, AFTER_THINK, CONTENT_SEP = 0, 1, 2, 3, 4
REASONING, CONTENT = 8, 16
START, FRESH, BLANKISH, TEXT, FRESH_AFTER_SPACE, TEXT_SPACE = 0, 1, 2, 3, 4, 5
LINES = 6
STATES = 24
LINE_NAMES = {START: "start", FRESH: "fresh", BLANKISH: "blankish", TEXT: "text",
              FRESH_AFTER_SPACE: "fresh_after_space", TEXT_SPACE: "text_space"}
NAMES = {OFF: "off", TOOL: "tool", THINK_OPEN: "think_open", AFTER_THINK: "after_think", CONTENT_SEP: "content_sep",
         **{REASONING + line: "r_" + name for line, name in LINE_NAMES.items()},
         **{CONTENT + line: "c_" + name for line, name in LINE_NAMES.items()}}

PLAIN, THINK, END_THINK, TOOL_CALL, END_TOOL_CALL, BOUNDARY, OTHER = range(7)
KINDS = {"<think>": THINK, "</think>": END_THINK, "<tool_call>": TOOL_CALL, "</tool_call>": END_TOOL_CALL,
         "<|im_start|>": BOUNDARY, "<|im_end|>": BOUNDARY, "<|endoftext|>": BOUNDARY}
BREAKS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")

STAT_ROWS, STAT_ACTIVE, STAT_SOME, STAT_HALF, STAT_MASS, STAT_PEAK, STAT_PROPOSED, STAT_REJECTED = range(8)
STAT_WIDTH = 8

_SAID = set()
_STATE = {"ban": None, "broken": "", "added": [], "at": 0.0, "pending": None, "steps": 0, "tail": 0, "eager": 0}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def armed():
    value = str(os.environ.get(NO_BLANK_ENV) or "").strip().lower()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{NO_BLANK_ENV}={value!r} is not one of '', '0', '1'.")


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def byte_decoder():
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    codes = keep[:]
    extra = 0
    for value in range(256):
        if value not in keep:
            keep.append(value)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): value for value, code in zip(keep, codes)}


def token_texts(tokenizer_json):
    record = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    decoder = byte_decoder()
    vocab = record["model"]["vocab"]
    size = max([*vocab.values(), *(item["id"] for item in record.get("added_tokens", ()))]) + 1
    texts = [""] * size
    kinds = [PLAIN] * size
    for piece, index in vocab.items():
        texts[index] = bytes(decoder[ch] for ch in piece).decode("utf-8", errors="replace")
    for item in record.get("added_tokens", ()):
        texts[item["id"]] = item["content"]
        kinds[item["id"]] = KINDS.get(item["content"], OTHER)
    return texts, kinds


def char_class(ch):
    if ch == "\n":
        return "E"
    if ch in BREAKS:
        return "X"
    if ch.isspace():
        return "W"
    return "N"


def line_walk(text, line):
    for ch in text:
        kind = char_class(ch)
        if kind == "N":
            line = TEXT
        elif kind == "W":
            if line == START:
                return STOP_LINE
            line = TEXT_SPACE if line in (TEXT, TEXT_SPACE) else BLANKISH
        elif kind == "E":
            if line not in (TEXT, TEXT_SPACE):
                return STOP_LINE
            line = FRESH if line == TEXT else FRESH_AFTER_SPACE
        else:
            return STOP_LINE
    return line


def separator(text, line):
    if line not in (TEXT, TEXT_SPACE) or not text.endswith("\n\n"):
        return False
    return line_walk(text[:-2], line) == TEXT if text[:-2] else line == TEXT


def step(state, text, kind):
    if kind == BOUNDARY:
        return True, OFF
    if state == CONTENT_SEP:
        return (True, TOOL) if kind == TOOL_CALL else (False, state)
    if state == TOOL:
        return True, (OFF if kind == END_TOOL_CALL else TOOL)
    if state >= REASONING and (state - (CONTENT if state >= CONTENT else REASONING)) < LINES:
        content = state >= CONTENT
        base = CONTENT if content else REASONING
        line = state - base
        if kind == PLAIN:
            if content and separator(text, line):
                return True, CONTENT_SEP
            after = line_walk(text, line)
            return (False, state) if after == STOP_LINE else (True, base + after)
        if kind == THINK:
            return False, state
        if kind == END_THINK:
            return (False, state) if content or line not in (START, FRESH) else (True, AFTER_THINK)
        if kind == TOOL_CALL:
            return (False, state) if content and line != START else (True, TOOL)
        return True, OFF
    if state == THINK_OPEN and kind == PLAIN and text == "\n":
        return True, REASONING + START
    if state == AFTER_THINK and kind == PLAIN and text == "\n\n":
        return True, CONTENT + START
    if kind == THINK:
        return True, THINK_OPEN
    if kind == END_THINK:
        return True, AFTER_THINK
    if kind == TOOL_CALL:
        return True, TOOL
    return True, OFF


def signature(text, kind):
    if kind != PLAIN:
        return ("kind", kind)
    if text in ("\n", "\n\n"):
        return ("exact", text)
    return ("plain", tuple(line_walk(text, line) for line in range(LINES)),
            tuple(separator(text, line) for line in range(LINES)))


def tables(texts, kinds, width):
    if width < len(texts):
        raise ValueError(f"the logits are {width} wide, narrower than the {len(texts)}-token vocabulary")
    nxt = np.zeros((width, STATES), dtype=np.int8)
    banned = np.zeros((STATES, width), dtype=np.uint8)
    groups = {}
    for index in range(width):
        text = texts[index] if index < len(texts) else "\ufffd"
        kind = kinds[index] if index < len(kinds) else PLAIN
        key = signature(text, kind)
        held = groups.get(key)
        if held is None:
            row = [step(state, text, kind) for state in range(STATES)]
            held = groups[key] = (np.array([after for _, after in row], dtype=np.int8),
                                  np.array([0 if ok else 1 for ok, _ in row], dtype=np.uint8))
        nxt[index] = held[0]
        banned[:, index] = held[1]
    return nxt, banned


def fold(nxt, state, tokens):
    for token in tokens:
        state = int(nxt[token, state])
    return state


def structural(kinds, width):
    held = np.zeros(width, dtype=np.uint8)
    for index, kind in enumerate(kinds):
        held[index] = kind != PLAIN
    return held


def cold_state(nxt, kinds, tokens):
    tokens = list(tokens)[-WINDOW:]
    marks = [at for at, token in enumerate(tokens) if token < len(kinds) and kinds[token] != PLAIN]
    return fold(nxt, OFF, tokens[marks[-1]:]) if marks else OFF


@triton.jit
def _commit_kernel(idx_ptr, real_ptr, all_ptr, all_stride, total_ptr, state_ptr, scan_ptr, next_ptr, marks_ptr,
                   HAS_REAL: tl.constexpr, CAP_: tl.constexpr, WINDOW_: tl.constexpr, SEARCH_: tl.constexpr,
                   STATES_: tl.constexpr):
    r = tl.program_id(0)
    live_req = r >= 0
    if HAS_REAL:
        live_req = r < tl.load(real_ptr)
    if live_req:
        slot = tl.load(idx_ptr + r)
        if slot >= 0:
            total = tl.load(total_ptr + slot)
            scan = tl.load(scan_ptr + slot)
            state = tl.load(state_ptr + slot)
            cold = (scan <= 0) | (scan > total) | (total - scan > CAP_)
            row = all_ptr + slot.to(tl.int64) * all_stride
            state = tl.where(cold, 0, state)
            floor = tl.where(cold, tl.maximum(total - WINDOW_, 0), total)
            found = tl.zeros([], dtype=tl.int32) - 1
            top = total
            while (top > floor) & (found < 0):
                low = tl.maximum(top - SEARCH_, floor)
                offs = low + tl.arange(0, SEARCH_)
                inside = offs < top
                tokens = tl.load(row + offs, mask=inside, other=0).to(tl.int64)
                marked = (tl.load(marks_ptr + tokens, mask=inside, other=0) != 0) & inside
                found = tl.max(tl.where(marked, offs, -1), axis=0)
                top = low
            first = tl.where(cold, tl.where(found >= 0, found, total), scan)
            for at in tl.range(first, total):
                token = tl.load(row + at).to(tl.int64)
                state = tl.load(next_ptr + token * STATES_ + state).to(tl.int32)
            tl.store(state_ptr + slot, state)
            tl.store(scan_ptr + slot, total)


@triton.jit
def _mask_kernel(logits_ptr, logits_stride, ex_idx_ptr, ex_pos_ptr, inputs_ptr, real_rows_ptr, rows, state_ptr,
                 next_ptr, banned_ptr, banned_stride, temperature_ptr, flags_ptr, stats_ptr, hist_ptr, vocab,
                 HAS_REAL: tl.constexpr, STEPS: tl.constexpr, STATES_: tl.constexpr, BLOCK_: tl.constexpr):
    row = tl.program_id(0)
    real_rows = rows
    if HAS_REAL:
        real_rows = tl.load(real_rows_ptr)
    if row < real_rows:
        slot = tl.load(ex_idx_ptr + row)
        local = tl.load(ex_pos_ptr + row)
        first = row - local
        state = tl.load(state_ptr + slot)
        for j in tl.static_range(1, STEPS + 1):
            live = j <= local
            token = tl.load(inputs_ptr + first + j, mask=live, other=0).to(tl.int64)
            after = tl.load(next_ptr + token * STATES_ + state, mask=live, other=0).to(tl.int32)
            state = tl.where(live, after, state)
        tl.atomic_add(hist_ptr + state, 1)
        tl.atomic_add(stats_ptr + 0, 1)
        follows = row + 1 < real_rows
        follows = follows & (tl.load(ex_idx_ptr + row + 1, mask=follows, other=-1) == slot)
        follows = follows & (tl.load(ex_pos_ptr + row + 1, mask=follows, other=-1) == local + 1)
        draft = tl.load(inputs_ptr + row + 1, mask=follows, other=0).to(tl.int64)
        hit = tl.load(banned_ptr + state.to(tl.int64) * banned_stride + draft, mask=follows, other=0) != 0
        hit = hit & follows
        tl.store(flags_ptr + row, hit.to(tl.int8))
        tl.atomic_add(stats_ptr + 6, hit.to(tl.int64))
        if (state == 4) | ((state >= 8) & (state < 14)) | ((state >= 16) & (state < 22)):
            temperature = tl.load(temperature_ptr + slot).to(tl.float32)
            scale = tl.where(temperature > 0, 1.0 / tl.maximum(temperature, 1e-4), 1e4)
            base = logits_ptr + row.to(tl.int64) * logits_stride
            ban_row = banned_ptr + state.to(tl.int64) * banned_stride
            peak = -1e30
            total = 0.0
            removed = 0.0
            for start in range(0, vocab, BLOCK_):
                offs = start + tl.arange(0, BLOCK_)
                inside = offs < vocab
                raw = tl.load(base + offs, mask=inside, other=float("-inf"))
                x = raw.to(tl.float32) * scale
                drop = (tl.load(ban_row + offs, mask=inside, other=0) != 0) & inside
                top = tl.maximum(peak, tl.max(x, axis=0))
                shrink = tl.exp(peak - top)
                e = tl.exp(x - top)
                total = total * shrink + tl.sum(e, axis=0)
                removed = removed * shrink + tl.sum(tl.where(drop, e, 0.0), axis=0)
                peak = top
                tl.store(base + offs, tl.full([BLOCK_], float("-inf"), tl.float32).to(raw.dtype), mask=drop)
            share = removed / tl.maximum(total, 1e-30)
            ppm = (share * 1000000.0).to(tl.int64)
            tl.atomic_add(stats_ptr + 1, 1)
            tl.atomic_add(stats_ptr + 2, (share > 0.001).to(tl.int64))
            tl.atomic_add(stats_ptr + 3, (share > 0.5).to(tl.int64))
            tl.atomic_add(stats_ptr + 4, ppm)
            tl.atomic_max(stats_ptr + 5, ppm)


@triton.jit
def _settle_kernel(cu_ptr, num_sampled_ptr, real_reqs_ptr, reqs, flags_ptr, stats_ptr, HAS_REAL: tl.constexpr):
    r = tl.program_id(0)
    real = reqs
    if HAS_REAL:
        real = tl.load(real_reqs_ptr)
    if r < real:
        start = tl.load(cu_ptr + r)
        count = tl.load(cu_ptr + r + 1) - start
        taken = tl.load(num_sampled_ptr + r).to(tl.int32)
        rejected = (taken >= 1) & (taken < count)
        at = start + tl.maximum(taken - 1, 0)
        hit = tl.load(flags_ptr + at, mask=rejected, other=0) != 0
        tl.atomic_add(stats_ptr + 7, (hit & rejected).to(tl.int64))


def tokenizer_json(runner):
    config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    for name in ("tokenizer", "model"):
        held = getattr(config, name, None)
        if held and (Path(str(held)) / "tokenizer.json").is_file():
            return Path(str(held)) / "tokenizer.json"
    raise RuntimeError("no tokenizer.json beside the served model; the ban needs the model's own vocabulary")


class Ban:
    def __init__(self, texts, kinds, width, slots, device, source=""):
        nxt, banned = tables(texts, kinds, width)
        self.width = width
        self.device = device
        self.next = torch.from_numpy(nxt).to(device)
        self.banned = torch.from_numpy(banned).to(device)
        self.host_next = nxt
        self.marks = torch.from_numpy(structural(kinds, width)).to(device)
        self.state = torch.zeros(slots, dtype=torch.int32, device=device)
        self.scan = torch.zeros(slots, dtype=torch.int32, device=device)
        self.stats = torch.zeros(STAT_WIDTH, dtype=torch.int64, device=device)
        self.hist = torch.zeros(STATES, dtype=torch.int64, device=device)
        self.flags = {}
        counts = banned.sum(axis=1)
        print(f"NO_BLANK armed: {source or 'vocabulary'} {len(texts)} tokens, logits {width} wide; banned per state "
              + ", ".join(f"{NAMES[s]} {int(counts[s])}" for s in sorted(NAMES) if counts[s])
              + "; free text is the reasoning body and the content body, a line may not be blank or whitespace "
                "only, no line break but \\n, no leading whitespace, </think> only at the start of a line, the "
                "content's blank line only right before <tool_call>; tool-call bodies and the drafter are not "
                "masked", flush=True)

    def flag_buffer(self, rows):
        held = self.flags.get(rows)
        if held is None:
            held = self.flags[rows] = torch.zeros(rows, dtype=torch.int8, device=self.device)
        return held

    def prepare(self):
        added = _STATE["added"]
        if not added:
            return
        held = torch.tensor(sorted(set(added)), dtype=torch.int64)
        added.clear()
        if self.scan.is_cuda:
            held = held.pin_memory().to(self.device, non_blocking=True)
        self.scan.index_fill_(0, held, 0)

    def commit(self, idx, real, reqs, all_token_ids, total_len):
        _commit_kernel[(reqs,)](
            idx, idx if real is None else real, all_token_ids, all_token_ids.stride(0), total_len, self.state,
            self.scan, self.next, self.marks, HAS_REAL=real is not None, CAP_=CAP, WINDOW_=WINDOW,
            SEARCH_=SEARCH, STATES_=STATES)

    def mask(self, logits, ex_idx, ex_pos, inputs, real_rows, temperature, steps):
        rows = logits.shape[0]
        if logits.shape[1] != self.width:
            say(f"width-{logits.shape[1]}", f"NO_BLANK stands aside for steps whose logits are {logits.shape[1]} wide; "
                                            f"its tables are {self.width} wide")
            return None
        flags = self.flag_buffer(rows)
        _mask_kernel[(rows,)](
            logits, logits.stride(0), ex_idx, ex_pos, inputs, ex_idx if real_rows is None else real_rows, rows,
            self.state, self.next, self.banned, self.banned.stride(0), temperature, flags, self.stats, self.hist,
            self.width, HAS_REAL=real_rows is not None, STEPS=steps, STATES_=STATES, BLOCK_=BLOCK)
        return flags

    def settle(self, cu, num_sampled, real_reqs, reqs, flags):
        _settle_kernel[(reqs,)](cu, num_sampled, cu if real_reqs is None else real_reqs, reqs, flags, self.stats,
                                HAS_REAL=real_reqs is not None)

    def snapshot(self):
        return torch.cat((self.stats, self.hist))


def describe(values):
    stats, hist = values[:STAT_WIDTH], values[STAT_WIDTH:]
    rows, active = int(stats[STAT_ROWS]), int(stats[STAT_ACTIVE])
    mean = stats[STAT_MASS] / PPM / active if active else 0.0
    states = {NAMES[s]: int(hist[s]) for s in sorted(NAMES) if int(hist[s])}
    return (f"NO_BLANK tally: {rows} sampled positions seen, {active} in free text or the content separator; "
            f"banned mass > 0.001 at {int(stats[STAT_SOME])}, > 0.5 at {int(stats[STAT_HALF])}; removed mass mean "
            f"{mean:.5f} max {stats[STAT_PEAK] / PPM:.5f}; drafts proposed into the ban {int(stats[STAT_PROPOSED])}, "
            f"rejected at a banned draft {int(stats[STAT_REJECTED])}; positions by state {states}")


def tick(force=False):
    ban = _STATE["ban"]
    if ban is None:
        return
    pending = _STATE["pending"]
    if pending is not None and pending[1].query():
        print(describe([int(x) for x in pending[0].tolist()]), flush=True)
        _STATE["pending"] = None
        pending = None
    now = time.monotonic()
    if pending is not None or (not force and now - _STATE["at"] < TALLY_S):
        return
    _STATE["at"] = now
    if not ban.stats.is_cuda:
        print(describe([int(x) for x in ban.snapshot().tolist()]), flush=True)
        return
    host = torch.empty(STAT_WIDTH + STATES, dtype=torch.int64, pin_memory=True)
    host.copy_(ban.snapshot(), non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    _STATE["pending"] = (host, event)


def selfcheck(ban, steps=3):
    if not ban.stats.is_cuda:
        return False
    device = ban.state.device
    ints = {"dtype": torch.int32, "device": device}
    idx, real = torch.zeros(1, **ints), torch.ones(1, **ints)
    all_ids, total = torch.zeros(1, 8, **ints), torch.ones(1, **ints)
    ex_idx, ex_pos, inputs = torch.zeros(1, **ints), torch.zeros(1, **ints), torch.zeros(1, **ints)
    cu, sampled = torch.tensor([0, 1], **ints), torch.ones(1, **ints)
    temperature = torch.ones(ban.state.shape[0], dtype=torch.float32, device=device)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.zeros(1, ban.width, dtype=dtype, device=device)
        for given in (None, real):
            ban.commit(idx, given, 1, all_ids, total)
            flags = ban.mask(logits, ex_idx, ex_pos, inputs, given, temperature, steps)
            ban.settle(cu, sampled, given, 1, flags)
    torch.cuda.synchronize(device)
    for tensor in (ban.state, ban.scan, ban.stats, ban.hist):
        tensor.zero_()
    ban.flags.clear()
    print("NO_BLANK self-check: the commit, mask and settle kernels ran once on a dummy row for both logits dtypes "
          "and both row layouts before any capture", flush=True)
    return True


def get(runner, width):
    ban = _STATE["ban"]
    if ban is not None or _STATE["broken"]:
        return ban
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the blank-line ban was first asked for inside a CUDA graph capture; an eager step must "
                           "build it before any capture")
    try:
        source = tokenizer_json(runner)
        texts, kinds = token_texts(source)
        slots = int(getattr(getattr(runner, "req_states", None), "max_num_reqs", 0) or runner.max_num_reqs)
        ban = Ban(texts, kinds, int(width), slots, runner.device, str(source))
        selfcheck(ban)
    except Exception as failure:
        _STATE["broken"] = f"{type(failure).__name__}: {failure}"
        selfcheck_fail(NO_BLANK_ENV, _STATE["broken"])
        print(f"NO_BLANK stands aside for the whole run: {_STATE['broken']}", flush=True)
        return None
    _STATE["ban"] = ban
    return ban


def added(runner, req_index):
    if armed():
        _STATE["added"].append(int(req_index))


def eager(runner, logits, input_batch):
    if not armed() or input_batch.num_reqs == 0:
        return None
    ban = get(runner, logits.shape[1])
    if ban is None or ban.width != logits.shape[1]:
        return None
    ban.prepare()
    states = runner.req_states
    ban.commit(input_batch.idx_mapping, None, input_batch.num_reqs, states.all_token_ids.gpu, states.total_len.gpu)
    inputs = input_batch.input_ids[input_batch.logits_indices]
    steps = getattr(runner.rejection_sampler, "num_speculative_steps", 0) if runner.rejection_sampler else 0
    flags = ban.mask(logits, input_batch.expanded_idx_mapping, input_batch.expanded_local_pos, inputs, None,
                     runner.sampler.sampling_states.temperature.gpu, max(int(steps), 1))
    _STATE["eager"] += 1
    say("live-eager", f"NO_BLANK live: the eager sampler masked its first step ({logits.shape[0]} logits rows)")
    tick()
    return flags


def after(runner, flags, sampler_output, input_batch):
    ban = _STATE["ban"]
    if ban is None or flags is None or sampler_output is None:
        return
    if input_batch.num_draft_tokens == 0:
        return
    ban.settle(input_batch.cu_num_logits, sampler_output.num_sampled, None, input_batch.num_reqs, flags)


class TailBan:
    def __init__(self, runner, geom):
        self.runner = runner
        self.geom = geom
        self.ban = None
        self.flags = None

    def stage(self):
        if _STATE["ban"] is not None:
            _STATE["ban"].prepare()

    def apply(self, logits, layout, inputs, temperature):
        self.ban = get(self.runner, logits.shape[1])
        self.flags = None
        if self.ban is None:
            return
        states = self.runner.req_states
        self.ban.commit(layout.idx_safe, layout.real_reqs, self.geom.reqs, states.all_token_ids.gpu,
                        states.total_len.gpu)
        self.flags = self.ban.mask(logits, layout.ex_idx, layout.ex_pos, inputs, layout.real_rows, temperature,
                                   self.geom.steps)

    def settle(self, layout, num_sampled):
        if self.ban is None or self.flags is None:
            return
        self.ban.settle(layout.cu, num_sampled, layout.real_reqs, self.geom.reqs, self.flags)


def for_tail(runner, geom):
    if not armed():
        return None
    if _STATE["broken"]:
        return None
    held = TailBan(runner, geom)
    say("live-tail", f"NO_BLANK live: the captured sampling tail masks the target logits of every sampled and "
                     f"verified position ({geom.reqs}-request bucket, {geom.steps} draft steps)")
    return held


def after_tail():
    if _STATE["ban"] is not None:
        _STATE["tail"] += 1
        tick()
