import json
import math
import os
import time
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton

ENV = "TAAF_VLLM_ACCEPT_LAMBDA"
GUARD_ENV = "TAAF_VLLM_ACCEPT_LAMBDA_GUARD"
TOOL_IDS_ENV = "TAAF_VLLM_ACCEPT_LAMBDA_TOOL_IDS"
EVERY_ENV = "TAAF_VLLM_ACCEPT_LAMBDA_LIVE_S"
XARG = "taaf_accept_lambda"
GUARD_XARG = "taaf_accept_lambda_guard"
DEFAULT_TOOL_IDS = (248058, 248059)
TOOL_TEXTS = ("<tool_call>", "</tool_call>")
MAX_LAMBDA = 16.0
DEFAULT_EVERY_S = 60.0
CHUNK = 64
LOOK = 4096
HEAD = ("rows", "relaxed", "tool", "guarded")
ARMS = 32
ARM_FIELDS = ("verified", "accepted", "offered")
FIELDS = HEAD + tuple(f"arm{arm}_{name}" for arm in range(ARMS) for name in ARM_FIELDS)
_state = {"tables": None, "states": None, "ids": None, "tool": None, "tally": None, "host": None, "pending": None,
          "last": None, "started": None, "fired": False, "seen": [0] * len(FIELDS), "said": set(), "bad_xargs": 0}


def say(key, line):
    if key not in _state["said"]:
        _state["said"].add(key)
        print(f"ACCEPT_LAMBDA {line}", flush=True)


def parse_lambda(said):
    said = str(said if said is not None else "").strip()
    if not said:
        return None
    value = float(said)
    if not 1.0 <= value <= MAX_LAMBDA or value != value:
        raise ValueError(f"{ENV}={said!r} must be a number in 1..{MAX_LAMBDA:g}")
    return value


def parse_guard(said):
    said = str(said if said is not None else "").strip()
    if said in ("", "0"):
        return 0, 0
    parts = [part.strip() for part in said.split(",")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{GUARD_ENV}={said!r} must read 'n,window', for example '6,256'")
    gram, window = int(parts[0]), int(parts[1])
    if not 2 <= gram <= 8 or not 16 <= window <= 1024 or window & (window - 1):
        raise ValueError(f"{GUARD_ENV}={said!r}: n must be 2..8 and the window a power of two in 16..1024")
    return gram, window


def global_lambda():
    return parse_lambda(os.environ.get(ENV)) or 1.0


def guard():
    return parse_guard(os.environ.get(GUARD_ENV))


def every_s():
    said = str(os.environ.get(EVERY_ENV) or "").strip()
    return max(1.0, float(said)) if said else DEFAULT_EVERY_S


def capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _tokenizer_ids(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    found = {item["content"]: int(item["id"]) for item in record.get("added_tokens", ())}
    if all(text in found for text in TOOL_TEXTS):
        return tuple(found[text] for text in TOOL_TEXTS)
    return None


def _model_dirs():
    dirs = []
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        model = getattr(config, "model_config", None)
        dirs.extend(str(getattr(model, name, "") or "") for name in ("tokenizer", "model"))
    except Exception:
        pass
    dirs.append(str(os.environ.get("TAAF_MODEL_PATH") or ""))
    return [one for one in dirs if one]


def tool_ids():
    held = _state["tool"]
    if held is not None:
        return held
    said = str(os.environ.get(TOOL_IDS_ENV) or "").strip()
    if said:
        parts = [int(part) for part in said.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{TOOL_IDS_ENV}={said!r} must read 'start,end'")
        held, source = tuple(parts), TOOL_IDS_ENV
    else:
        held, source = DEFAULT_TOOL_IDS, "the Qwen3.8 defaults"
        for one in _model_dirs():
            path = Path(one) / "tokenizer.json"
            if path.is_file():
                found = _tokenizer_ids(path)
                if found is not None:
                    held, source = found, str(path)
                    break
    _state["tool"] = held
    say("tool", f"tool-call ids {held[0]} / {held[1]} from {source}; every draft inside a tool call, and the "
                f"marker tokens themselves, is verified strictly")
    return held


def bind(states):
    slots = int(states.max_num_reqs)
    device = states.device
    tables = {name: torch.zeros(slots, dtype=dtype, device=device) for name, dtype in
              (("lam", torch.float32), ("state", torch.int32), ("scan", torch.int32), ("start", torch.int32),
               ("guard", torch.int32), ("arm", torch.int32))}
    _state["tables"] = tables
    _state["states"] = states
    _state["tally"] = torch.zeros(len(FIELDS), dtype=torch.int64, device=device)
    tool_ids()
    gram, window = guard()
    say("bound", f"bound to the request state: {slots} slots; global lambda {global_lambda():g}, per request "
                 f"through vllm_xargs {{'{XARG}': x}}; loop guard "
                 + (f"on: a draft that would complete a {gram}-gram already in the last {window} tokens of the "
                    f"reply is verified strictly, a request's {{'{GUARD_XARG}': 0}} turns it off"
                    if gram else "off"))


def _last(tokens, value):
    try:
        return len(tokens) - 1 - tokens[::-1].index(value)
    except ValueError:
        return -1


def added(slot, prompt_len, all_token_ids):
    tables = _state["tables"]
    if tables is None:
        return
    start, end = tool_ids()
    reply = list(all_token_ids[prompt_len:])[-LOOK:]
    inside = int(_last(reply, start) > _last(reply, end))
    tables["state"][slot] = inside
    tables["scan"][slot] = len(all_token_ids)
    tables["start"][slot] = int(prompt_len)


def arm_of(value, guarded):
    return min(ARMS // 2 - 1, max(0, round((value - 1.0) * 4))) + (ARMS // 2 if guarded else 0)


def set_slot(slot, value, guarded):
    tables = _state["tables"]
    tables["lam"][slot] = math.log(value) if value > 1.0 else 0.0
    tables["guard"][slot] = int(guarded)
    tables["arm"][slot] = arm_of(value, guarded)


def sampling_added(slot, sampling_params):
    tables = _state["tables"]
    if tables is None:
        return
    value = global_lambda()
    extra = getattr(sampling_params, "extra_args", None) or {}
    guarded = guard()[0] > 0
    if GUARD_XARG in extra:
        guarded = guarded and str(extra[GUARD_XARG]).strip() not in ("0", "False", "false")
    if XARG in extra:
        try:
            asked = float(extra[XARG])
        except (TypeError, ValueError):
            asked = -1.0
        if 1.0 <= asked <= MAX_LAMBDA:
            value = asked
        else:
            _state["bad_xargs"] += 1
            say("bad-xarg", f"ignored a request's {XARG}={extra[XARG]!r} (must be 1..{MAX_LAMBDA:g}); it runs at the "
                            f"global {value:g}")
    set_slot(slot, value, guarded)
    tick()


def use_ids(ids):
    held = _state["ids"]
    _state["ids"] = ids
    return held


@triton.jit
def _rows_kernel(out_ptr, ids_ptr, cu_ptr, idx_ptr, temp_ptr, lam_ptr, state_ptr, scan_ptr, start_ptr, guard_ptr,
                 all_ptr, all_stride, total_ptr, span_ptr, tally_ptr, slots, START: tl.constexpr, END: tl.constexpr,
                 GRAM: tl.constexpr, WINDOW: tl.constexpr, STEPS: tl.constexpr, SPAN: tl.constexpr,
                 CHUNK_: tl.constexpr, LOOK_: tl.constexpr):
    r = tl.program_id(0)
    first = tl.load(cu_ptr + r).to(tl.int64)
    drafts = tl.load(cu_ptr + r + 1).to(tl.int64) - first - 1
    slot = tl.load(idx_ptr + r).to(tl.int64)
    if (slot >= 0) & (slot < slots) & (drafts > 0):
        temp = tl.load(temp_ptr + slot).to(tl.float32)
        lam = tl.load(lam_ptr + slot)
        total = tl.load(total_ptr + slot).to(tl.int64)
        scan = tl.load(scan_ptr + slot).to(tl.int64)
        state = tl.load(state_ptr + slot)
        row = all_ptr + slot * all_stride
        cold = (scan > total) | (total - scan > CHUNK_)
        floor = tl.where(cold, tl.maximum(total - LOOK_, 0), scan)
        found = tl.zeros((), tl.int64) - 1
        kind = tl.zeros((), tl.int32)
        top = total
        while (top > floor) & (found < 0):
            low = tl.maximum(top - CHUNK_, floor)
            offs = low + tl.arange(0, CHUNK_)
            inside = offs < top
            tokens = tl.load(row + offs, mask=inside, other=-1)
            at_start = tl.max(tl.where(inside & (tokens == START), offs, -1), axis=0)
            at_end = tl.max(tl.where(inside & (tokens == END), offs, -1), axis=0)
            found = tl.maximum(at_start, at_end)
            kind = (at_start > at_end).to(tl.int32)
            top = low
        state = tl.where(found >= 0, kind, tl.where(cold, 0, state))
        tl.store(state_ptr + slot, state)
        tl.store(scan_ptr + slot, total.to(tl.int32))
        if GRAM > 0:
            base = span_ptr + r.to(tl.int64) * SPAN
            j = tl.arange(0, WINDOW)
            at = total - WINDOW + j
            reply = (at >= tl.load(start_ptr + slot).to(tl.int64)) & (at >= 0)
            seen = tl.load(row + at, mask=reply, other=0).to(tl.int64)
            tl.store(base + j, tl.where(reply, seen, -2 - j.to(tl.int64)))
            k = tl.arange(0, STEPS)
            own = k < drafts
            mine = tl.load(ids_ptr + first + 1 + k, mask=own, other=0).to(tl.int64)
            tl.store(base + WINDOW + k, tl.where(own, mine, -100000 - k.to(tl.int64)))
            tl.debug_barrier()
        for i in range(drafts):
            token = tl.load(ids_ptr + first + 1 + i).to(tl.int64)
            plain = (state == 0) & (token != START) & (token != END) & (token >= 0)
            sampled = temp != 0.0
            wanted = sampled & (lam > 0.0)
            tool = wanted & (plain == 0)
            hit = tl.zeros((), tl.int32)
            if GRAM > 0:
                base = span_ptr + r.to(tl.int64) * SPAN
                cand = tl.arange(0, SPAN)
                valid = cand + GRAM - 1 < WINDOW + i
                match = valid
                for g in tl.static_range(GRAM):
                    want = tl.load(base + WINDOW + i - (GRAM - 1) + g)
                    have = tl.load(base + cand + g, mask=valid, other=-7)
                    match = match & (have == want)
                hit = tl.max(match.to(tl.int32), axis=0) * (tl.load(guard_ptr + slot) != 0).to(tl.int32)
            guarded = wanted & plain & (hit != 0)
            relaxed = wanted & plain & (hit == 0)
            tl.store(out_ptr + first + i, tl.where(relaxed, lam, 0.0))
            tl.atomic_add(tally_ptr + 0, sampled.to(tl.int64))
            tl.atomic_add(tally_ptr + 1, relaxed.to(tl.int64))
            tl.atomic_add(tally_ptr + 2, tool.to(tl.int64))
            tl.atomic_add(tally_ptr + 3, guarded.to(tl.int64))
            state = tl.where(token == START, 1, tl.where(token == END, 0, state))


@triton.jit
def _settle_kernel(accepted_ptr, cu_ptr, idx_ptr, temp_ptr, arm_ptr, tally_ptr, slots, HEAD_: tl.constexpr):
    r = tl.program_id(0)
    drafts = tl.load(cu_ptr + r + 1).to(tl.int64) - tl.load(cu_ptr + r).to(tl.int64) - 1
    slot = tl.load(idx_ptr + r).to(tl.int64)
    if (slot >= 0) & (slot < slots) & (drafts > 0):
        if tl.load(temp_ptr + slot) != 0.0:
            at = tally_ptr + HEAD_ + 3 * tl.load(arm_ptr + slot).to(tl.int64)
            tl.atomic_add(at, tl.full((), 1, tl.int64))
            tl.atomic_add(at + 1, tl.load(accepted_ptr + r).to(tl.int64))
            tl.atomic_add(at + 2, drafts)


def rows(draft_sampled, cu_num_logits, idx_mapping, temperature, num_logits, num_speculative_steps):
    out = torch.zeros(num_logits, dtype=torch.float32, device=draft_sampled.device)
    tables, states = _state["tables"], _state["states"]
    if tables is None:
        say("unbound", "stands aside: no request state is bound, so every draft is verified strictly")
        return out
    ids = _state["ids"]
    if ids is None or ids.shape != draft_sampled.shape:
        ids = draft_sampled
    num_reqs = int(cu_num_logits.shape[0]) - 1
    if num_reqs <= 0:
        return out
    steps = triton.next_power_of_2(max(1, int(num_speculative_steps)))
    gram, window = guard()
    span_width = triton.next_power_of_2(window + steps) if gram else 1
    span = torch.empty((num_reqs if gram else 1, span_width), dtype=torch.int64, device=draft_sampled.device)
    all_token_ids, total_len = states.all_token_ids.gpu, states.total_len.gpu
    start, end = tool_ids()
    _rows_kernel[(num_reqs,)](
        out, ids, cu_num_logits, idx_mapping, temperature, tables["lam"], tables["state"], tables["scan"],
        tables["start"], tables["guard"], all_token_ids, all_token_ids.stride(0), total_len, span, _state["tally"],
        int(tables["lam"].shape[0]), START=start, END=end, GRAM=gram, WINDOW=max(window, 16), STEPS=steps,
        SPAN=span_width, CHUNK_=CHUNK, LOOK_=LOOK, num_warps=1)
    return out


def settle(accepted, cu_num_logits, idx_mapping, temperature):
    tables = _state["tables"]
    num_reqs = int(cu_num_logits.shape[0]) - 1
    if tables is None or num_reqs <= 0:
        return
    _settle_kernel[(num_reqs,)](accepted, cu_num_logits, idx_mapping, temperature, tables["arm"], _state["tally"],
                                int(tables["lam"].shape[0]), HEAD_=len(HEAD), num_warps=1)
    tick()


def arm_name(arm):
    return f"lambda {1 + (arm % (ARMS // 2)) / 4:g}" + (" guarded" if arm >= ARMS // 2 else "")


def _line(values, label):
    rows_, relaxed, tool, guarded = values[:len(HEAD)]
    if not rows_:
        return f"{label}: no sampled draft row"
    parts = [f"{label}: {rows_} sampled draft rows, {relaxed / rows_:.1%} relaxed, {tool / rows_:.1%} strict in a tool "
             f"call, {guarded / rows_:.1%} strict by the loop guard"]
    for arm in range(ARMS):
        verified, accepted, offered = values[len(HEAD) + 3 * arm:len(HEAD) + 3 * arm + 3]
        if verified:
            parts.append(f"{arm_name(arm)} {1 + accepted / verified:.4f} tokens a step over {verified} verifications "
                         f"({accepted / max(offered, 1):.1%} of offered drafts)")
    return "; ".join(parts)


def report(values, now):
    seen = _state["seen"]
    window = [one - two for one, two in zip(values, seen)]
    _state["seen"] = list(values)
    if not _state["fired"]:
        if not values[1]:
            return
        _state["fired"] = True
        print(f"{ENV} fired: drafts accepted at u < min(1, lambda * p / q) outside tool calls; "
              + _line(values, "first read") + f"; a live line follows every {every_s():.0f} s", flush=True)
        return
    minutes = (now - (_state["started"] or now)) / 60
    print(f"{ENV} live [{minutes:.0f} min]: " + _line(window, "window") + " | " + _line(values, "since start"),
          flush=True)


def tick():
    held = _state["tally"]
    if held is None or capturing():
        return
    now = time.monotonic()
    if _state["started"] is None:
        _state["started"] = now
        _state["last"] = now - every_s()
    if not held.is_cuda:
        if now - _state["last"] >= every_s() or not _state["fired"]:
            _state["last"] = now
            report([int(one) for one in held.tolist()], now)
        return
    pending = _state["pending"]
    if pending is not None and pending.query():
        _state["pending"] = None
        report([int(one) for one in _state["host"].tolist()], now)
    if _state["pending"] is None and (now - _state["last"] >= every_s() or not _state["fired"]):
        _state["last"] = now
        if _state["host"] is None:
            _state["host"] = torch.zeros(len(FIELDS), dtype=torch.int64).pin_memory()
        _state["host"].copy_(held, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        _state["pending"] = event


def install():
    try:
        from vllm.v1.worker.gpu import states as request_module
        from vllm.v1.worker.gpu.sample import states as sampling_module
    except Exception as failure:
        say("no-install", f"could not reach the request states ({type(failure).__name__}: {failure}); every draft "
                          f"is verified strictly")
        return False
    request_cls, sampling_cls = request_module.RequestState, sampling_module.SamplingStates
    if getattr(request_cls, "_taaf_accept_lambda", False):
        return True
    init, add, sampling_add = request_cls.__init__, request_cls.add_request, sampling_cls.add_request

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        bind(self)

    def add_request(self, req_id, prompt_len, all_token_ids, *args, **kwargs):
        add(self, req_id, prompt_len, all_token_ids, *args, **kwargs)
        slot = self.req_id_to_index.get(req_id)
        if slot is not None:
            added(slot, prompt_len, all_token_ids)

    def sampling_add_request(self, req_idx, sampling_params, *args, **kwargs):
        sampling_add(self, req_idx, sampling_params, *args, **kwargs)
        sampling_added(req_idx, sampling_params)

    request_cls.__init__ = __init__
    request_cls.add_request = add_request
    sampling_cls.add_request = sampling_add_request
    request_cls._taaf_accept_lambda = True
    say("install", f"armed in pid {os.getpid()}: RequestState and SamplingStates carry each request's lambda, "
                   f"reply start and tool-call state")
    return True


install()
