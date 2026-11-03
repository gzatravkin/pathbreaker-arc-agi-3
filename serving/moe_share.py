import json
import os
import re
import threading
import time

import torch
import triton
import triton.language as tl


_PDL_STATE = {"on": None}


def pdl_on():
    if _PDL_STATE["on"] is None:
        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()
        forced = said == "interpret"
        asked = forced or said in ("1", "true", "yes", "on")
        on = forced or bool(
            asked
            and os.environ.get("TRITON_INTERPRET") != "1"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )
        _PDL_STATE["on"] = on
        if asked:
            print(f"PDL {__name__} enabled={int(on)}", flush=True)
    return _PDL_STATE["on"]

FILE_ENV = "TAAF_VLLM_MOE_ROUTE_FILE"
MTP_ENV = "TAAF_VLLM_MOE_SHARE_MTP"
CUT_ENV = "TAAF_VLLM_MOE_CUT"
READ_EVERY_S = 1.0
STATS_EVERY_S = 5.0
STOCK_PREFERENCE = 1e-5
MODES = ("off", "share")
MAIN_LAYERS = 48
INSIDE_WARPS = 1
REPICK_WARPS = 4
LAYER = re.compile(r"layers\.(\d+)\.")
OFF = {"mode": "off", "k_keep": 0, "ratio": 1.0, "cap": 0, "topp": 0.0, "group": 0,
       "prefill_topk": 0, "mtp_topk": 0, "cut_file": "", "cut_keep": 0}
STATS = 10

_lock = threading.Lock()
_staged = dict(OFF)
_raw: dict = {"text": None, "path": ""}
_controls: dict = {}
_stats: dict = {}
_said: set = set()
_reported_at = {"t": float("-inf")}
_started = {"watcher": False}
_staged_copies: dict = {}
_reported: dict = {}
_deferred: dict = {}
_capture_depth = {"n": 0}
_rows: dict = {}
_scratch: dict = {}
_cuda_gate = threading.RLock()
_cuts: dict = {}
_cut_rows: dict = {}


def say(text) -> None:
    print(f"MOE_SHARE {text}", flush=True)


def say_once(key, text) -> None:
    if key in _said:
        return
    _said.add(key)
    say(text)


def main_layer(name) -> bool:
    found = LAYER.search(str(name))
    return bool(found) and int(found.group(1)) < MAIN_LAYERS


def mtp_on() -> bool:
    value = str(os.environ.get(MTP_ENV) or "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def unshareable(router, experts: int) -> str:
    scoring = str(getattr(router, "scoring_func", "softmax"))
    if scoring != "softmax":
        return f"the router scores with {scoring}, and the re-pick compares softmax scores"
    held = int(getattr(router, "global_num_experts", experts) or experts)
    if held != experts:
        return f"the router logits have {experts} columns while the router holds {held} experts"
    return ""


def _parse(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("the control file is not a JSON object")
    mode = str(data.get("mode") or "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    out = {"mode": mode, "k_keep": int(data.get("k_keep", OFF["k_keep"])),
           "ratio": float(data.get("ratio", OFF["ratio"])),
           "cap": int(data.get("cap", OFF["cap"])),
           "topp": float(data.get("topp", OFF["topp"])),
           "group": int(data.get("group", OFF["group"])),
           "prefill_topk": int(data.get("prefill_topk", OFF["prefill_topk"])),
           "mtp_topk": int(data.get("mtp_topk", OFF["mtp_topk"])),
           "cut_file": str(data.get("cut_file") or ""),
           "cut_keep": int(data.get("cut_keep", OFF["cut_keep"]))}
    if not 0 <= out["cut_keep"] <= 4096 or (out["cut_keep"] and not out["cut_file"]):
        raise ValueError("cut_keep must be 0, or a positive count with the cut_file that ranks the experts")
    if not 0.0 <= out["ratio"] <= 1.0:
        raise ValueError(f"ratio {out['ratio']} is outside [0, 1]")
    if not 0.0 <= out["topp"] <= 1.0:
        raise ValueError(f"topp {out['topp']} is outside [0, 1]")
    if min(out["k_keep"], out["cap"], out["group"], out["prefill_topk"], out["mtp_topk"]) < 0:
        raise ValueError("k_keep, cap, group, prefill_topk and mtp_topk must not be negative")
    return out


def _vector(fields: dict) -> list:
    armed = fields["mode"] == "share" and fields["cap"] > 0
    return [1.0 if armed else 0.0, float(fields["k_keep"]), float(fields["ratio"]),
            float(fields["cap"]), float(fields["topp"]), float(fields["group"]),
            float(fields["prefill_topk"]), float(fields["mtp_topk"])]


def _vector_now() -> list:
    with _lock:
        return _vector(_staged)


def _staging(tensor: torch.Tensor) -> dict:
    with _lock:
        held = _staged_copies.get(id(tensor))
    if held is not None:
        return held
    if tensor.is_cuda:
        made = {"host": torch.zeros(tensor.shape, dtype=tensor.dtype, pin_memory=True),
                "stream": torch.cuda.Stream(tensor.device)}
    else:
        made = {"host": torch.zeros(tensor.shape, dtype=tensor.dtype), "stream": None}
    with _lock:
        return _staged_copies.setdefault(id(tensor), made)


def _not_capturing(what: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"MOE_SHARE {what} was first asked for inside a CUDA graph capture; it "
                           f"must be made by an eager forward first, or the graph would re-create "
                           f"it on every replay")


def _push(tensor: torch.Tensor, vector: list) -> None:
    held = _staging(tensor)
    host = held["host"]
    if held["stream"] is None:
        with torch.inference_mode():
            host.copy_(torch.tensor(vector, dtype=tensor.dtype))
            tensor.copy_(host)
        return
    with _cuda_gate:
        if _capture_depth["n"]:
            _deferred["push"] = True
            say_once("deferred", "a settings change waits for the CUDA graph capture in progress "
                                 "and is copied in once it ends")
            return
        with torch.inference_mode():
            host.copy_(torch.tensor(vector, dtype=tensor.dtype))
            with torch.cuda.stream(held["stream"]):
                tensor.copy_(host, non_blocking=True)
        held["stream"].synchronize()


def cut_on() -> bool:
    return str(os.environ.get(CUT_ENV) or "").strip() not in ("", "0", "false", "off")


def _cut_plan(fields: dict, experts: int) -> tuple:
    keep, source = int(fields["cut_keep"]), str(fields["cut_file"])
    if not keep:
        return [[0.0] * experts for _ in range(MAIN_LAYERS)], 0.0, "off"
    key = (source, keep, experts)
    with _lock:
        held = _cut_rows.get(key)
    if held is not None:
        return held
    try:
        order = json.loads(open(source, encoding="utf-8").read())
        rows = []
        for layer in range(MAIN_LAYERS):
            ranked = [int(one) for one in order[str(layer)]]
            if sorted(ranked) != list(range(experts)):
                raise ValueError(f"layer {layer} ranks {len(ranked)} experts, not a permutation of {experts}")
            if keep > experts:
                raise ValueError(f"cut_keep {keep} exceeds the {experts} experts")
            kept = set(ranked[:keep])
            rows.append([0.0 if expert in kept else float("-inf") for expert in range(experts)])
        made = (rows, 1.0, f"cut={keep}:{os.path.basename(source)} on {MAIN_LAYERS} layers")
    except (OSError, ValueError, KeyError, TypeError) as failure:
        say(f"cut refused: {source} keep {keep}: {type(failure).__name__}: {failure}; routing stays uncut")
        made = ([[0.0] * experts for _ in range(MAIN_LAYERS)], 0.0, "refused")
    with _lock:
        return _cut_rows.setdefault(key, made)


def _push_cuts() -> None:
    with _lock:
        held = list(_cuts.values())
        fields = dict(_staged)
    for bias, flag in held:
        rows, on, _ = _cut_plan(fields, int(bias.shape[-1]))
        _push(bias, rows)
        _push(flag, [on])


def cut_tensors(device, experts: int) -> tuple:
    key = (device, experts)
    with _lock:
        held = _cuts.get(key)
    if held is not None:
        return held
    _not_capturing("the expert cut")
    made = (torch.zeros(MAIN_LAYERS, experts, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.float32, device=device))
    with _lock:
        _staged_fields = dict(_staged)
    rows, on, label = _cut_plan(_staged_fields, experts)
    _push(made[0], rows)
    _push(made[1], [on])
    say(f"cut armed on {experts} experts a layer: the main layers' top-k is taken over the kept experts "
        f"only, from a device mask the control file refills; now {label}")
    with _lock:
        return _cuts.setdefault(key, made)


def cut(index: int, logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
        renormalize: bool) -> tuple:
    bias, flag = cut_tensors(logits.device, int(logits.shape[-1]))
    masked = logits.float() + bias[index]
    values, picked = torch.topk(masked, int(ids.shape[1]), dim=-1)
    if renormalize:
        fresh = torch.softmax(values, dim=-1)
    else:
        fresh = torch.softmax(masked, dim=-1).gather(1, picked)
    on = flag[0] > 0
    return (torch.where(on, fresh.to(weights.dtype), weights),
            torch.where(on, picked.to(ids.dtype), ids))


def _push_all() -> None:
    vector = _vector_now()
    with _lock:
        held = list(_controls.values())
    for tensor in held:
        _push(tensor, vector)
    _push_cuts()


def poll(now: float | None = None) -> bool:
    path = str(os.environ.get(FILE_ENV) or "")
    try:
        text = open(path, encoding="utf-8").read() if path else ""
    except OSError:
        text = ""
    with _lock:
        same = text == _raw["text"]
        if not same:
            _raw["text"] = text
            _raw["path"] = path
            try:
                fresh = _parse(text) if text.strip() else dict(OFF)
            except (ValueError, TypeError):
                fresh = None
            changed = fresh is not None and fresh != _staged
            if fresh is not None:
                _staged.clear()
                _staged.update(fresh)
            held = dict(_staged)
    if same:
        if _deferred.pop("push", False):
            _push_all()
        return False
    if fresh is None:
        say(f"the control file {path} is unreadable; routing stays as last armed")
        return False
    if changed:
        _push_all()
    say(f"settings mode={held['mode']} k_keep={held['k_keep']} ratio={held['ratio']} "
        f"cap={held['cap']} topp={held['topp']} group={held['group']} "
        f"prefill_topk={held['prefill_topk']} mtp_topk={held['mtp_topk']}"
        f"{_cut_said(held)}, read from "
        f"{path or '(no file)'}{'' if changed else ', the settings already in force'}")
    return changed


def _cut_said(held: dict) -> str:
    if not held["cut_keep"]:
        return ""
    experts = [key[1] for key in _cuts] or [0]
    return " " + _cut_plan(held, experts[0])[2] if experts[0] else f" cut={held['cut_keep']}:{os.path.basename(held['cut_file'])} pending"


def control(device) -> torch.Tensor:
    with _lock:
        tensor = _controls.get(device)
    if tensor is not None:
        return tensor
    _not_capturing("the control tensor")
    poll()
    tensor = torch.zeros(len(_vector(OFF)), dtype=torch.float32, device=device)
    _push(tensor, _vector_now())
    with _lock:
        tensor = _controls.setdefault(device, tensor)
    return tensor


def stats_tensor(layer: str, device) -> torch.Tensor:
    key = (layer, device)
    with _lock:
        tensor = _stats.get(key)
    if tensor is not None:
        return tensor
    _not_capturing(f"the {layer} counter")
    tensor = torch.zeros(STATS, dtype=torch.float64, device=device)
    _staging(tensor)
    with _lock:
        return _stats.setdefault(key, tensor)


def _snapshot(tensor: torch.Tensor) -> list | None:
    held = _staging(tensor)
    if held["stream"] is None:
        return tensor.detach().tolist()
    with _cuda_gate:
        if _capture_depth["n"]:
            return None
        with torch.inference_mode(), torch.cuda.stream(held["stream"]):
            held["host"].copy_(tensor, non_blocking=True)
        held["stream"].synchronize()
        return held["host"].tolist()


def _report_stats(force: bool = False) -> None:
    now = time.monotonic()
    if not force and now - _reported_at["t"] < STATS_EVERY_S:
        return
    _reported_at["t"] = now
    with _lock:
        items = list(_stats.items())
    for (layer, _device), tensor in sorted(items, key=lambda row: row[0][0]):
        held = _snapshot(tensor)
        if held is None:
            return
        last = _reported.get((layer, _device), [0.0] * len(held))
        _reported[(layer, _device)] = held
        step = [now_value - then for now_value, then in zip(held, last)]
        forwards = step[0]
        if forwards <= 0:
            continue
        line = f"stats {layer} forwards={forwards:.0f}"
        if step[1] > 0 or step[2] > 0:
            line += (f" stock={step[1] / forwards:.2f} routed={step[2] / forwards:.2f} "
                     f"moved={step[3] / forwards:.2f}")
        for name, rows_at in (("topp", 4), ("prefill_topk", 6), ("mtp_topk", 8)):
            if step[rows_at] > 0:
                line += (f" {name}_kept={step[rows_at + 1] / step[rows_at]:.2f} over "
                         f"{step[rows_at]:.0f} rows")
        say(line)


def _gate_captures() -> None:
    graph = torch.cuda.CUDAGraph
    if getattr(graph, "_taaf_moe_share_gated", False):
        return
    begin, end = graph.capture_begin, graph.capture_end

    def capture_begin(self, *args, **kwargs):
        with _cuda_gate:
            _capture_depth["n"] += 1
        try:
            return begin(self, *args, **kwargs)
        except BaseException:
            with _cuda_gate:
                _capture_depth["n"] -= 1
            raise

    def capture_end(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            with _cuda_gate:
                _capture_depth["n"] -= 1

    graph.capture_begin = capture_begin
    graph.capture_end = capture_end
    graph._taaf_moe_share_gated = True


def _watch() -> None:
    while True:
        time.sleep(READ_EVERY_S)
        try:
            poll()
            _report_stats()
        except Exception as failure:
            say(f"watcher failed: {type(failure).__name__}: {failure}")


def arm() -> None:
    with _lock:
        if _started["watcher"]:
            return
        _started["watcher"] = True
    if torch.cuda.is_available():
        _gate_captures()
    poll()
    thread = threading.Thread(target=_watch, daemon=True, name="moe-share-watch")
    thread.start()
    say(f"armed: every main MoE layer forward hands its top-k to the batch-aware re-pick under "
        f"CUDA graphs; settings come from {os.environ.get(FILE_ENV) or '(unset)'}, read at most "
        f"once a second and copied into a device tensor so a load needs no server restart")


@triton.jit
def _row_live(ids_ptr, w_ptr, qsl_ptr, t, stride_ids, stride_w, E: tl.constexpr,
              K: tl.constexpr, KPAD: tl.constexpr, ROWS: tl.constexpr, NREQ: tl.constexpr):
    offs = tl.arange(0, KPAD)
    mask = offs < K
    raw_id = tl.load(ids_ptr + t * stride_ids + offs, mask=mask, other=0)
    raw_w = tl.load(w_ptr + t * stride_w + offs, mask=mask, other=0.0)
    w = raw_w.to(tl.float32)
    good = (raw_id >= 0) & (raw_id < E) & (w * 0.0 == 0.0)
    live = tl.sum(tl.where(mask & (~good), 1, 0), axis=0) == 0
    if ROWS:
        live = live & (t < tl.load(qsl_ptr + NREQ))
    return live, raw_id, raw_w


@triton.jit
def _mark_kernel(ids_ptr, w_ptr, control_ptr, qsl_ptr, demand_ptr, core_ptr, before_ptr,
                  kept_ptr, stats_ptr, stride_ids, stride_w, stride_kept,
                  E: tl.constexpr, K: tl.constexpr, KPAD: tl.constexpr, ROWS: tl.constexpr,
                  NREQ: tl.constexpr, PDL: tl.constexpr = False) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    batch_on = tl.load(control_ptr + 0) > 0.5
    group_on = tl.load(control_ptr + 5) >= 2.0
    if batch_on | group_on:
        offs = tl.arange(0, KPAD)
        mask = offs < K
        live, raw_id, raw_w = _row_live(ids_ptr, w_ptr, qsl_ptr, t, stride_ids, stride_w, E, K,
                                        KPAD, ROWS, NREQ)
        e = raw_id.to(tl.int32)
        w = tl.where(mask, raw_w.to(tl.float32), -1.0e30)
        keep = tl.load(control_ptr + 1)
        wi = w[:, None]
        wj = w[None, :]
        oi = offs[:, None]
        oj = offs[None, :]
        greater = (wj > wi) | ((wj == wi) & (oj < oi))
        rank = tl.sum(tl.where(mask[None, :], greater.to(tl.int32), 0), axis=1)
        is_kept = (rank.to(tl.float32) < keep) & mask
        tl.store(kept_ptr + t * stride_kept + offs, is_kept.to(tl.int32), mask=mask)
        if live:
            if batch_on:
                tl.atomic_add(demand_ptr + e, tl.where(mask, w, 0.0), mask=mask)
                tl.atomic_max(core_ptr + e, is_kept.to(tl.float32), mask=is_kept)
            seen = tl.atomic_xchg(before_ptr + e, tl.full((KPAD,), 1, tl.int32), mask=mask)
            fresh = tl.sum(tl.where(mask & (seen == 0), 1, 0), axis=0)
            tl.atomic_add(stats_ptr + 1, fresh.to(tl.float64))


@triton.jit
def _inside_kernel(demand_ptr, core_ptr, before_ptr, after_ptr, control_ptr, inside_ptr,
                    E: tl.constexpr, EPAD: tl.constexpr, K: tl.constexpr,
                    BITS: tl.constexpr, PDL: tl.constexpr = False) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    offs = tl.arange(0, EPAD)
    mask = offs < E
    batch_on = tl.load(control_ptr + 0) > 0.5
    group_on = tl.load(control_ptr + 5) >= 2.0
    if batch_on:
        demand = tl.load(demand_ptr + offs, mask=mask, other=0.0)
        core = tl.load(core_ptr + offs, mask=mask, other=0.0)
        total = tl.sum(tl.where(mask, demand, 0.0), axis=0)
        score = demand + core * (total + 1.0)
        bits = (score + 0.0).to(tl.int32, bitcast=True)
        ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        key = (ordered.to(tl.int64) + 2147483648) * EPAD + (EPAD - 1 - offs)
        key = tl.where(mask, key, 0)
        cap = tl.load(control_ptr + 3)
        width = tl.maximum(cap, float(K))
        one = tl.full((), 1, tl.int64)
        floor = tl.full((), 0, tl.int64)
        for step in tl.static_range(BITS):
            trial = floor | (one << (BITS - 1 - step))
            count = tl.sum(tl.where(mask & (key >= trial), 1, 0), axis=0)
            floor = tl.where(count.to(tl.float32) >= width, trial, floor)
        inside = (key >= floor) & mask & (demand > 0.0)
        tl.store(inside_ptr + offs, inside.to(tl.int32), mask=mask)
    if batch_on | group_on:
        zeros = tl.zeros((EPAD,), dtype=tl.int32)
        tl.store(demand_ptr + offs, zeros.to(tl.float32), mask=mask)
        tl.store(core_ptr + offs, zeros.to(tl.float32), mask=mask)
        tl.store(before_ptr + offs, zeros, mask=mask)
        tl.store(after_ptr + offs, zeros, mask=mask)


@triton.jit
def _trim(out_id, out_w, k_offs, k_mask, keep_by_mass, mass, keep_top, RENORM: tl.constexpr):
    ki = k_offs[:, None]
    kj = k_offs[None, :]
    wi = out_w[:, None]
    wj = out_w[None, :]
    ahead = k_mask[None, :] & ((wj > wi) | ((wj == wi) & (kj < ki)))
    rank = tl.sum(tl.where(ahead, 1, 0), axis=1)
    before = tl.sum(tl.where(ahead, wj, 0.0), axis=1)
    total = tl.sum(tl.where(k_mask, out_w, 0.0), axis=0)
    by_mass = (before < mass * total) | (rank == 0)
    by_top = rank.to(tl.float32) < keep_top
    keep = tl.where(keep_by_mass, by_mass, by_top) & k_mask
    count = tl.sum(tl.where(keep, 1, 0), axis=0)
    top1 = tl.sum(tl.where(k_mask & (rank == 0), out_id, 0), axis=0)
    kept_total = tl.sum(tl.where(keep, out_w, 0.0), axis=0)
    if RENORM:
        scaled = out_w / kept_total
    else:
        scaled = out_w * (total / kept_total)
    trimmed = count < tl.sum(tl.where(k_mask, 1, 0), axis=0)
    new_id = tl.where(trimmed, tl.where(keep, out_id, top1), out_id)
    new_w = tl.where(trimmed, tl.where(keep, scaled, 0.0), out_w)
    return new_id, new_w, count


@triton.jit
def _count_after(out_id, k_offs, k_mask, after_ptr, stats_ptr):
    ki = k_offs[:, None]
    kj = k_offs[None, :]
    repeated = tl.sum(tl.where(k_mask[None, :] & (kj < ki) & (out_id[None, :] == out_id[:, None]),
                               1, 0), axis=1) > 0
    first = k_mask & (~repeated)
    seen = tl.atomic_xchg(after_ptr + out_id, tl.full(out_id.shape, 1, tl.int32), mask=first)
    fresh = tl.sum(tl.where(first & (seen == 0), 1, 0), axis=0)
    tl.atomic_add(stats_ptr + 2, fresh.to(tl.float64))


@triton.jit
def _repick_kernel(logits_ptr, ids_ptr, w_ptr, kept_ptr, inside_ptr, control_ptr, after_ptr,
                    qsl_ptr, out_ids_ptr, out_w_ptr, stats_ptr,
                    stride_logits, stride_ids, stride_w, stride_kept,
                    stride_out_ids, stride_out_w,
                    E: tl.constexpr, EPAD: tl.constexpr, K: tl.constexpr, KPAD: tl.constexpr,
                    RENORM: tl.constexpr, STOCK_PREF: tl.constexpr, ROWS: tl.constexpr,
                    NREQ: tl.constexpr, NREQ_PAD: tl.constexpr, DQL: tl.constexpr,
                    DRAFT: tl.constexpr, PDL: tl.constexpr = False) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    k_offs = tl.arange(0, KPAD)
    k_mask = k_offs < K
    enabled = tl.load(control_ptr + 0)
    group = tl.load(control_ptr + 5)
    topp = tl.load(control_ptr + 4)
    prefill_top = tl.load(control_ptr + 6)
    draft_top = tl.load(control_ptr + 7)
    if DRAFT:
        any_on = (enabled > 0.5) | ((draft_top > 0.0) & (draft_top < float(K)))
    else:
        any_on = ((enabled > 0.5) | (group >= 2.0) | ((topp > 0.0) & (topp < 1.0))
                  | (prefill_top > 0.0))
    if any_on:
        if t == 0:
            tl.atomic_add(stats_ptr + 0, 1.0)
        live, raw_id, raw_w = _row_live(ids_ptr, w_ptr, qsl_ptr, t, stride_ids, stride_w, E, K,
                                        KPAD, ROWS, NREQ)
        oid = raw_id.to(tl.int32)
        ow = raw_w.to(tl.float32)
        batch_on = enabled > 0.5

        is_decode = t < 0
        is_prefill = t < 0
        start = t * 0
        end = t * 0
        if ROWS:
            q_offs = tl.arange(0, NREQ_PAD)
            ends = tl.load(qsl_ptr + 1 + q_offs, mask=q_offs < NREQ, other=2147483647)
            req = tl.sum(tl.where(ends <= t, 1, 0), axis=0)
            real = req < NREQ
            start = tl.load(qsl_ptr + req, mask=real, other=0)
            end = tl.load(qsl_ptr + req + 1, mask=real, other=0)
            length = end - start
            is_decode = real & (length >= 1) & (length <= DQL)
            is_prefill = real & (length > DQL)

        size = tl.minimum(tl.maximum(group, 1.0), float(DQL)).to(tl.int32)
        first_row = start + ((t - start) // size) * size
        last_row = tl.minimum(end, first_row + size)
        group_on = live & is_decode & (size >= 2) & (last_row - first_row >= 2)
        pick_row = live & (batch_on | group_on)

        out_id = oid
        out_w = ow
        changed = t < 0
        if pick_row:
            e_offs = tl.arange(0, EPAD)
            e_mask = e_offs < E
            logits = tl.load(logits_ptr + t * stride_logits + e_offs, mask=e_mask,
                             other=-1.0e30).to(tl.float32)
            row_max = tl.max(logits, axis=0)
            num = tl.exp(logits - row_max)
            num = tl.where(e_mask, num, 0.0)
            denom = tl.sum(num, axis=0)
            probs = num / denom
            kept = tl.load(kept_ptr + t * stride_kept + k_offs, mask=k_mask, other=0) != 0
            inside = tl.load(inside_ptr + e_offs, mask=e_mask, other=0) != 0
            ratio = tl.load(control_ptr + 2)

            inside_g = e_offs < 0
            if group_on:
                demand_g = tl.zeros((EPAD,), dtype=tl.float32)
                core_g = e_offs < 0
                for member in tl.static_range(DQL):
                    row = first_row + member
                    in_run = row < last_row
                    for slot in tl.static_range(K):
                        slot_id = tl.load(ids_ptr + row * stride_ids + slot, mask=in_run,
                                          other=-1).to(tl.int32)
                        slot_w = tl.load(w_ptr + row * stride_w + slot, mask=in_run,
                                         other=0.0).to(tl.float32)
                        slot_kept = tl.load(kept_ptr + row * stride_kept + slot, mask=in_run,
                                            other=0) != 0
                        hit = e_offs == slot_id
                        demand_g += tl.where(hit, slot_w, 0.0)
                        core_g = core_g | (hit & slot_kept)
                total_g = tl.sum(tl.where(e_mask, demand_g, 0.0), axis=0)
                score_g = demand_g + tl.where(core_g, total_g + 1.0, 0.0)
                bits_g = (score_g + 0.0).to(tl.int32, bitcast=True)
                ordered_g = tl.where(bits_g >= 0, bits_g, bits_g ^ 0x7FFFFFFF)
                key_g = (ordered_g.to(tl.int64) + 2147483648) * EPAD + (EPAD - 1 - e_offs)
                key_g = tl.where(e_mask & (demand_g > 0.0), key_g, -1)
                for pick in tl.static_range(K):
                    best_g = tl.max(key_g, axis=0)
                    chosen_g = (EPAD - 1 - (best_g % EPAD)).to(tl.int32)
                    chosen_g = tl.where(best_g >= 0, chosen_g, -1)
                    inside_g = inside_g | (e_offs == chosen_g)
                    key_g = tl.where(e_offs == chosen_g, -1, key_g)

            is_stock_col = e_offs < 0
            kept_col = e_offs < 0
            for slot in tl.static_range(K):
                slot_id = tl.load(ids_ptr + t * stride_ids + slot).to(tl.int32)
                slot_kept = tl.load(kept_ptr + t * stride_kept + slot) != 0
                hit = e_offs == slot_id
                is_stock_col = is_stock_col | hit
                kept_col = kept_col | (hit & slot_kept)

            factor = tl.where(~inside, tl.where(batch_on, ratio, 1.0), 1.0)
            factor = factor * tl.where(~inside_g, tl.where(group_on, ratio, 1.0), 1.0)
            adjusted = probs * factor
            adjusted = tl.where(is_stock_col, adjusted * (1.0 + STOCK_PREF), adjusted)
            adjusted = tl.where(kept_col, -1.0, adjusted)
            bits = (adjusted + 0.0).to(tl.int32, bitcast=True)
            ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
            key = (ordered.to(tl.int64) + 2147483648) * EPAD + (EPAD - 1 - e_offs)
            key = tl.where(e_mask, key, -1)

            not_kept = (~kept) & k_mask
            ki = k_offs[:, None]
            kj = k_offs[None, :]
            prefix = tl.sum(tl.where((kj < ki) & not_kept[None, :], 1, 0), axis=1)
            final_id = oid
            for rank in tl.static_range(K):
                best = tl.max(key, axis=0)
                chosen = (EPAD - 1 - (best % EPAD)).to(tl.int32)
                final_id = tl.where(not_kept & (prefix == rank), chosen, final_id)
                key = tl.where(e_offs == chosen, -1, key)

            picked = tl.load(logits_ptr + t * stride_logits + final_id, mask=k_mask,
                             other=-1.0e30).to(tl.float32)
            picked_prob = tl.where(k_mask, tl.exp(picked - row_max) / denom, 0.0)
            if RENORM:
                total_picked = tl.sum(picked_prob, axis=0)
                fresh_w = picked_prob / total_picked
            else:
                fresh_w = picked_prob
            found = tl.sum(tl.where(k_mask[None, :] & (final_id[:, None] == oid[None, :]), 1, 0),
                           axis=1) > 0
            use_new = tl.sum(tl.where(k_mask & (~found), 1, 0), axis=0) > 0
            out_id = tl.where(use_new, final_id, oid)
            out_w = tl.where(use_new, fresh_w, ow)
            changed = use_new

        if DRAFT:
            mass_on = t < 0
            keep_top = tl.where(draft_top > 0.0, draft_top, float(K))
        else:
            mass_on = is_decode & (topp > 0.0) & (topp < 1.0)
            keep_top = tl.where(is_prefill & (prefill_top > 0.0), prefill_top, float(K))
        trim_row = live & (mass_on | (keep_top < float(K)))
        count = tl.sum(tl.where(k_mask, 1, 0), axis=0)
        if trim_row:
            out_id, out_w, count = _trim(out_id, out_w, k_offs, k_mask, mass_on, topp, keep_top,
                                         RENORM)
            changed = changed | (count < K)
        tl.store(out_ids_ptr + t * stride_out_ids + k_offs, tl.where(changed, out_id, raw_id),
                 mask=k_mask)
        tl.store(out_w_ptr + t * stride_out_w + k_offs, tl.where(changed, out_w, raw_w),
                 mask=k_mask)

        if pick_row | trim_row:
            moved = tl.sum(tl.where(k_mask[None, :] & (out_id[:, None] == oid[None, :]), 1, 0),
                           axis=1)
            moved = tl.sum(tl.where(k_mask & (moved == 0), 1, 0), axis=0)
            tl.atomic_add(stats_ptr + 3, moved.to(tl.float64))
            _count_after(out_id, k_offs, k_mask, after_ptr, stats_ptr)
            if DRAFT:
                if draft_top > 0.0:
                    tl.atomic_add(stats_ptr + 8, 1.0)
                    tl.atomic_add(stats_ptr + 9, count.to(tl.float64))
            else:
                if mass_on:
                    tl.atomic_add(stats_ptr + 4, 1.0)
                    tl.atomic_add(stats_ptr + 5, count.to(tl.float64))
                if is_prefill & (prefill_top > 0.0):
                    tl.atomic_add(stats_ptr + 6, 1.0)
                    tl.atomic_add(stats_ptr + 7, count.to(tl.float64))
    else:
        raw_id = tl.load(ids_ptr + t * stride_ids + k_offs, mask=k_mask, other=0)
        raw_w = tl.load(w_ptr + t * stride_w + k_offs, mask=k_mask, other=0.0)
        tl.store(out_ids_ptr + t * stride_out_ids + k_offs, raw_id, mask=k_mask)
        tl.store(out_w_ptr + t * stride_out_w + k_offs, raw_w, mask=k_mask)


@triton.jit
def _draft_trim_kernel(ids_ptr, w_ptr, control_ptr, stats_ptr, stride_ids, stride_w,
                        E: tl.constexpr, K: tl.constexpr, KPAD: tl.constexpr,
                        RENORM: tl.constexpr, PDL: tl.constexpr = False) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    draft_top = tl.load(control_ptr + 7)
    if (draft_top > 0.0) & (draft_top < float(K)):
        if t == 0:
            tl.atomic_add(stats_ptr + 0, 1.0)
        k_offs = tl.arange(0, KPAD)
        k_mask = k_offs < K
        live, raw_id, raw_w = _row_live(ids_ptr, w_ptr, ids_ptr, t, stride_ids, stride_w, E, K,
                                        KPAD, False, 0)
        if live:
            out_id, out_w, count = _trim(raw_id.to(tl.int32), raw_w.to(tl.float32), k_offs,
                                         k_mask, t < 0, 0.0, draft_top, RENORM)
            tl.store(ids_ptr + t * stride_ids + k_offs, out_id, mask=k_mask)
            tl.store(w_ptr + t * stride_w + k_offs, out_w, mask=k_mask)
            tl.atomic_add(stats_ptr + 8, 1.0)
            tl.atomic_add(stats_ptr + 9, count.to(tl.float64))


def _rows_args(rows, device):
    if rows is None:
        return torch.zeros(2, dtype=torch.int32, device=device), False, 1, 1, 1
    starts, decode_len = rows
    requests = int(starts.shape[0]) - 1
    return starts, True, requests, triton.next_power_of_2(max(requests, 1)), int(decode_len)


def make_scratch(experts: int, device) -> torch.Tensor:
    return torch.zeros(4, experts, dtype=torch.int32, device=device)


def repick(logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
           control_vector: torch.Tensor, renormalize: bool,
           stats: torch.Tensor | None = None, rows: tuple | None = None,
           draft: bool = False, scratch: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, width = int(ids.shape[0]), int(ids.shape[1])
    if tokens == 0 or width == 0:
        return weights, ids
    experts = int(logits.shape[-1])
    device = logits.device
    kpad = triton.next_power_of_2(width)
    epad = triton.next_power_of_2(experts)
    ids_c = ids.contiguous()
    w_c = weights.contiguous()
    logits_c = logits.contiguous()
    if stats is None:
        stats = torch.zeros(STATS, dtype=torch.float64, device=device)
    if scratch is None:
        scratch = make_scratch(experts, device)
    starts, known, requests, requests_pad, decode_len = _rows_args(rows, device)
    demand = scratch[0].view(torch.float32)
    core = scratch[1].view(torch.float32)
    before = scratch[2]
    after = scratch[3]
    kept = torch.empty(tokens, width, dtype=torch.int32, device=device)
    _mark_kernel[(tokens,)](ids_c, w_c, control_vector, starts, demand, core, before, kept, stats,
                            ids_c.stride(0), w_c.stride(0), kept.stride(0),
                            E=experts, K=width, KPAD=kpad, ROWS=known, NREQ=requests, PDL=pdl_on(), launch_pdl=pdl_on())
    inside = torch.empty(experts, dtype=torch.int32, device=device)
    _inside_kernel[(1,)](demand, core, before, after, control_vector, inside, E=experts,
                         EPAD=epad, K=width, BITS=32 + epad.bit_length() - 1,
                         num_warps=INSIDE_WARPS, PDL=pdl_on(), launch_pdl=pdl_on())
    out_ids = torch.empty_like(ids_c)
    out_w = torch.empty_like(w_c)
    _repick_kernel[(tokens,)](
        logits_c, ids_c, w_c, kept, inside, control_vector, after, starts, out_ids, out_w, stats,
        logits_c.stride(0), ids_c.stride(0), w_c.stride(0), kept.stride(0),
        out_ids.stride(0), out_w.stride(0),
        E=experts, EPAD=epad, K=width, KPAD=kpad,
        RENORM=bool(renormalize), STOCK_PREF=STOCK_PREFERENCE, ROWS=known, NREQ=requests,
        NREQ_PAD=requests_pad, DQL=decode_len, DRAFT=bool(draft), num_warps=REPICK_WARPS, PDL=pdl_on(), launch_pdl=pdl_on())
    return out_w, out_ids


def draft_trim(weights: torch.Tensor, ids: torch.Tensor, experts: int,
               control_vector: torch.Tensor, renormalize: bool,
               stats: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, width = int(ids.shape[0]), int(ids.shape[1])
    if tokens == 0 or width == 0:
        return weights, ids
    ids_c = ids.contiguous()
    w_c = weights.contiguous()
    if stats is None:
        stats = torch.zeros(STATS, dtype=torch.float64, device=ids.device)
    _draft_trim_kernel[(tokens,)](ids_c, w_c, control_vector, stats, ids_c.stride(0),
                                  w_c.stride(0), E=experts, K=width,
                                  KPAD=triton.next_power_of_2(width), RENORM=bool(renormalize), PDL=pdl_on(), launch_pdl=pdl_on())
    return w_c, ids_c


def _find_rows(device):
    import gc

    found = []
    for held in gc.get_objects():
        kind = type(held)
        if kind.__name__ != "GPUModelRunner" or "worker.gpu" not in kind.__module__:
            continue
        starts = getattr(getattr(held, "input_buffers", None), "query_start_loc", None)
        decode_len = getattr(held, "decode_query_len", None)
        if (isinstance(starts, torch.Tensor) and starts.dim() == 1 and starts.dtype == torch.int32
                and starts.device == torch.device(device) and isinstance(decode_len, int)
                and decode_len >= 1):
            found.append((starts, decode_len))
    if len(found) != 1:
        return None, f"{len(found)} model runners with a request-start buffer were found, not one"
    return found[0], ""


def rows_for(device):
    with _lock:
        if device in _rows:
            return _rows[device]
    _not_capturing("the request-boundary lookup")
    held, why = _find_rows(device)
    with _lock:
        _rows[device] = held
    if held is None:
        say(f"row boundaries unknown ({why}): topp, group and prefill_topk stay off on every row; "
            f"rows whose ids or weights are not real experts are passed through untouched")
    else:
        say(f"rows from the model runner's request starts ({int(held[0].shape[0]) - 1} request "
            f"slots, {held[1]} decode rows a request): topp and group act on decode rows, "
            f"prefill_topk on prefill rows, padding rows on neither")
    return held


def scratch_for(layer: str, experts: int, device) -> torch.Tensor:
    key = (layer, device)
    with _lock:
        held = _scratch.get(key)
    if held is not None:
        return held
    _not_capturing(f"the {layer} scratch")
    made = make_scratch(experts, device)
    with _lock:
        return _scratch.setdefault(key, made)


def route(layer, router, logits, weights, ids):
    if ids.dim() != 2 or int(ids.shape[0]) == 0:
        return weights, ids
    experts = int(logits.shape[-1])
    reason = unshareable(router, experts)
    if reason:
        say_once(("unshareable", str(layer)), f"share skipped on {layer}: {reason}")
        return weights, ids
    arm()
    vector = control(logits.device)
    stats = stats_tensor(str(layer), logits.device)
    renormalize = bool(getattr(router, "renormalize", True))
    if cut_on() and main_layer(layer):
        weights, ids = cut(int(LAYER.search(str(layer)).group(1)), logits, weights, ids, renormalize)
    if not main_layer(layer):
        if not mtp_on():
            return draft_trim(weights, ids, experts, vector, renormalize, stats)
        return repick(logits, weights, ids, vector, renormalize, stats, draft=True,
                      scratch=scratch_for(str(layer), experts, logits.device))
    rows = rows_for(logits.device)
    return repick(logits, weights, ids, vector, renormalize, stats, rows,
                  scratch=scratch_for(str(layer), experts, logits.device))
