# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import contextlib
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

if __import__("os").environ.get("TAAF_VLLM_MARKS", "").strip() == "1":
    _marked = __import__("importlib").import_module("vllm.taaf_marks").marked
else:
    def _marked(site):
        return lambda fn: fn


_PDL_STATE = {"on": None}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


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

CACHE_DTYPE = "int8_per_token_head"
INT4_DTYPE = "int4_per_token_head"
CACHE_DTYPES = (CACHE_DTYPE, INT4_DTYPE)
LAYOUTS = ("int8", "int4", "k4v8")
SCALE_BYTES = 4
ROW_ALIGN = 16
PINNED_OPS = "vllm.models.qwen4_exp.nvidia.ops.qsa"
PINNED_INT4 = "vllm.v1.attention.ops.int4_per_token_head"
SETTINGS = Path(__file__).with_name("taaf_qsa_kv8.json")
FAST_ENV = "TAAF_VLLM_QSA_KV8_FAST"
FAST_PREFILL_BLOCK_N = 64
FAST_MIN_WARPS = 4
TUNE_FILE_ENV = "TAAF_VLLM_QSA_KV8_TUNE_FILE"
TUNE_READ_EVERY_S = 1.0
TUNE_DEFAULT = {"block_n": FAST_PREFILL_BLOCK_N, "num_warps": FAST_MIN_WARPS, "num_splits": 0}
PREFILL_ENV = "TAAF_VLLM_QSA_PREFILL"
PREFILL_FILE_ENV = "TAAF_VLLM_QSA_PREFILL_FILE"
PREFILL_CONFIG_ENV = "TAAF_VLLM_QSA_PREFILL_CONFIG"
PREFILL_MIN_PROGRAMS = 2048
PREFILL_MODES = ("stock", "fused")
PREFILL_DEFAULT = {"mode": "fused", "block_n": 32, "num_warps": 4, "num_stages": 2, "overlap": 0}
PREFILL_ROTATE_CHUNK = 32
DECODE_FUSED_ENV = "TAAF_VLLM_QSA_DECODE_FUSED"
DECODE_WARPS_ENV = "TAAF_VLLM_QSA_DECODE_WARPS"
DECODE_FUSED_WARPS = 4
SPLITK_FUSED_ENV = "TAAF_VLLM_QSA_SPLITK_FUSED"
SPLITK_COUNTERS = 1 << 16
_SPLITK = {"counters": {}, "said": set()}
SPILL_GATE_ENV = "TAAF_VLLM_HOST_KV_SPILL_FLAG"
SPILL_GATE_ROWS = 512
DECODE_WARPS_COLUMNS = 32
DECODE_CONFIG_ENV = "TAAF_VLLM_QSA_DECODE_CONFIG"
DECODE_CONFIG_MIN_PROGRAMS = 512
_DCONFIG = {"raw": None, "parsed": None}
INT8_FAST_ENV = "TAAF_VLLM_INT8_FAST"
OVERLAP_GROUPS = (2, 4, 8, 16, 32, 64, 128, 256)
OVERLAP_FULL_SHARE = 0.25
OVERLAP_TOKEN_SPAN = 1 << 20
OVERLAP_ENV = "TAAF_VLLM_QSA_OVERLAP"
OVERLAP_SAMPLES = 12
REUSE_ENV = "TAAF_VLLM_QSA_REUSE"
REUSE_REQUESTS = 8
REUSE_HISTORY = (1, 4, 16, 64)
REUSE_RECENT = (1024, 4096, 8192)
REUSE_MIX = (16, 4096)
REUSE_BANDS = (8192, 32768, 65536)
REUSE_MIN_CONTEXT = 16384
REUSE_REPORT_S = 60.0
TILE_ENV = "TAAF_VLLM_QSA_TILE"
TILE_UNION_ENV = "TAAF_VLLM_QSA_TILE_UNION_FUSED"
TILE_SKIP_ENV = "TAAF_VLLM_QSA_TILE_SKIP_EMPTY"
TILE_UNION_CHUNK = 1024
TILE_SIZES = (2, 4)
TILE_WARPS = {2: 4, 4: 8}
TILE_BLOCK_N = 32
TILE_CHECKS = 3
TILE_TOLERANCE = 1.0e-2
TILE_SPAN = 1 << 18
TILE_EMPTY = (1 << 31) - 1
TILE_TALLY_S = 15.0
LEAN_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN"
LEAN_CHECKED_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN_CHECKED"
LEAN_CHECKS = 3
LEAN_TOLERANCE = 1.0e-2
LEAN_PAIRS = 64
LEAN_COLUMNS = 128
LEAN_WARPS = 4
LEAN_SLOT_LIMIT = 1 << 31
LEAN_TIMED = 5
LEAN_MAXNREG_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN_MAXNREG"
LEAN_CAPS = (0, 168, 128)
_LEAN = {"on": None, "checked": 0, "aside": "", "said": set(), "tables": {}, "maxnreg": None}
DECODE_LEAN_ENV = "TAAF_VLLM_QSA_DECODE_LEAN"
DECODE_LEAN_CHECKED_ENV = "TAAF_VLLM_QSA_DECODE_LEAN_CHECKED"
DECODE_LEAN_CHECKS = 3
DECODE_LEAN_TOLERANCE = 1.0e-2
DECODE_STAGES_ENV = "TAAF_VLLM_QSA_DECODE_STAGES"
DECODE_STAGES_MAX = 4
_DLEAN = {"on": None, "checked": 0, "aside": "", "said": set()}
FAST_TALLY_FIELDS = ("decode_lean_launches", "decode_lean_rows", "prefill_lean_launches", "prefill_lean_rows")
_FAST_TALLY = {"tensors": {}, "registered": set()}
FAST_TALLY_COUNTER = "qsa_fast"
TIMERS_MODULE = "vllm.taaf_step_timers"
_STATE = {"layout": None}
_ROWS = {"last": None}
_ANNOUNCED = set()
_TUNE = {"read_at": float("-inf"), "raw": None, "settings": dict(TUNE_DEFAULT)}
_PREFILL = {"read_at": float("-inf"), "raw": None, "settings": dict(PREFILL_DEFAULT), "measured": 0}
_TILE = {"rows": None, "checked": 0, "aside": "", "at": 0.0, "calls": 0, "said": set()}
_REUSE = {"budget": None, "steps": 0, "layers": {}, "keys": {}, "history": {}, "tally": {}, "reported": None,
          "closed": False}


def pdl_args(kernel):
    if pdl_on() and "PDL" in getattr(kernel, "arg_names", ()):
        return {"PDL": True, "launch_pdl": True}
    return {}


def announce(event, **fields):
    if event in _ANNOUNCED:
        return
    _ANNOUNCED.add(event)
    detail = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"QSA_KV8 {event} {detail}".rstrip(), flush=True)


def allowed(cache_dtype):
    return cache_dtype in CACHE_DTYPES


def fast_enabled():
    return os.environ.get(FAST_ENV) == "1"


def decode_fused_enabled():
    return os.environ.get(DECODE_FUSED_ENV) == "1"


def int8_fast():
    return os.environ.get(INT8_FAST_ENV) == "1"


def splitk_fused():
    return os.environ.get(SPLITK_FUSED_ENV) == "1"


def splitk_counters(device, slots):
    if slots > SPLITK_COUNTERS:
        return None
    key = (device.type, device.index)
    held = _SPLITK["counters"].get(key)
    if held is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        held = torch.zeros(SPLITK_COUNTERS, dtype=torch.int32, device=device)
        _SPLITK["counters"][key] = held
    return held


def splitk_aside(reason, **fields):
    if reason not in _SPLITK["said"]:
        _SPLITK["said"].add(reason)
        detail = " ".join(f"{name}={value}" for name, value in fields.items())
        print(f"QSA_KV8 splitk_fused_aside reason={reason} {detail}".rstrip(), flush=True)


def follows_int4(chosen):
    return chosen == "int4" or (chosen == "int8" and int8_fast())


def decode_warps():
    said = str(os.environ.get(DECODE_WARPS_ENV) or "").strip()
    if not said:
        return 0
    warps = int(said)
    if warps < 1 or warps & (warps - 1):
        raise ValueError(f"{DECODE_WARPS_ENV} must be a power of 2 >= 1, found {said!r}")
    return warps


def decode_stages():
    said = str(os.environ.get(DECODE_STAGES_ENV) or "").strip()
    if not said:
        return 0
    if not said.isdigit() or not 1 <= int(said) <= DECODE_STAGES_MAX:
        raise ValueError(f"{DECODE_STAGES_ENV} must be a stage count from 1 to {DECODE_STAGES_MAX}, found {said!r}")
    return int(said)


def decode_lean_enabled():
    if _DLEAN["on"] is None:
        said = str(os.environ.get(DECODE_LEAN_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{DECODE_LEAN_ENV}={said!r} is not '', '0' or '1'")
        _DLEAN["on"] = said == "1"
    return _DLEAN["on"]


def decode_lean_ready():
    return _DLEAN["checked"] >= DECODE_LEAN_CHECKS or os.environ.get(DECODE_LEAN_CHECKED_ENV) == "1"


def decode_lean_say(key, line):
    if key in _DLEAN["said"]:
        return
    _DLEAN["said"].add(key)
    print(line, flush=True)


def fast_tally(device):
    key = (device.type, device.index)
    held = _FAST_TALLY["tensors"].get(key)
    if held is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        held = torch.zeros(len(FAST_TALLY_FIELDS), dtype=torch.int64, device=device)
        _FAST_TALLY["tensors"][key] = held
    if key not in _FAST_TALLY["registered"]:
        timers = sys.modules.get(TIMERS_MODULE)
        if callable(getattr(timers, "counter", None)):
            timers.counter(FAST_TALLY_COUNTER, FAST_TALLY_FIELDS, held)
            _FAST_TALLY["registered"].add(key)
            print(f"QSA_KV8 tally {FAST_TALLY_COUNTER} registered with the step timers: "
                  f"{', '.join(FAST_TALLY_FIELDS)}", flush=True)
    return held


def fast_tally_totals():
    columns = zip(*(held.tolist() for held in _FAST_TALLY["tensors"].values()))
    return dict(zip(FAST_TALLY_FIELDS, (sum(column) for column in columns)))


def int4_config(pinned, num_rows, num_kv_heads, use_prefill_config, num_columns):
    block_n, num_warps, num_tiles, num_splits = pinned._select_config(
        num_rows, num_kv_heads, use_prefill_config, num_columns)
    widen = decode_warps() * max(1, block_n // DECODE_WARPS_COLUMNS)
    if widen and not use_prefill_config and widen > num_warps:
        announce("decode_warps", rows=num_rows, block_n=block_n, num_warps=f"{num_warps}->{widen}",
                 num_splits=num_splits)
        num_warps = widen
    chosen = decode_config_override()
    if chosen and not use_prefill_config and num_rows * num_kv_heads > DECODE_CONFIG_MIN_PROGRAMS:
        block_n, num_warps, wanted = chosen
        num_tiles = triton.cdiv(num_columns, block_n)
        num_splits = min(wanted, num_tiles)
        announce("decode_config", knob=DECODE_CONFIG_ENV, rows=num_rows, block_n=block_n, num_warps=num_warps,
                 num_tiles=num_tiles, num_splits=num_splits)
    return block_n, num_warps, num_tiles, num_splits


def decode_config_override():
    raw = str(os.environ.get(DECODE_CONFIG_ENV) or "").strip()
    if raw == _DCONFIG["raw"]:
        return _DCONFIG["parsed"]
    parsed = None
    if raw:
        parts = raw.split(",")
        if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
            raise ValueError(f"{DECODE_CONFIG_ENV}={raw!r} is not 'block_n,num_warps,num_splits'")
        block_n, num_warps, num_splits = (int(part) for part in parts)
        if block_n < 16 or block_n & (block_n - 1):
            raise ValueError(f"{DECODE_CONFIG_ENV}: block_n {block_n} must be a power of 2 >= 16")
        if not 1 <= num_warps <= 16 or num_warps & (num_warps - 1):
            raise ValueError(f"{DECODE_CONFIG_ENV}: num_warps {num_warps} must be a power of 2 from 1 to 16")
        if num_splits < 1:
            raise ValueError(f"{DECODE_CONFIG_ENV}: num_splits {num_splits} must be at least 1")
        parsed = (block_n, num_warps, num_splits)
    _DCONFIG["raw"], _DCONFIG["parsed"] = raw, parsed
    return parsed


def parsed_tune(raw):
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("the QSA_KV8 tune file is not a JSON object")
    out = dict(TUNE_DEFAULT)
    for key in ("block_n", "num_warps", "num_splits"):
        if key in data:
            out[key] = int(data[key])
    if out["block_n"] < 16 or out["block_n"] & (out["block_n"] - 1):
        raise ValueError(f"block_n {out['block_n']} must be a power of 2 >= 16")
    if out["num_warps"] < 1 or out["num_warps"] & (out["num_warps"] - 1):
        raise ValueError(f"num_warps {out['num_warps']} must be a power of 2 >= 1")
    if out["num_splits"] < 0:
        raise ValueError(f"num_splits {out['num_splits']} must not be negative")
    return out


def tune_settings(now=None):
    now = time.monotonic() if now is None else now
    if now - _TUNE["read_at"] < TUNE_READ_EVERY_S:
        return _TUNE["settings"]
    _TUNE["read_at"] = now
    path = str(os.environ.get(TUNE_FILE_ENV) or "")
    try:
        raw = open(path, encoding="utf-8").read() if path else ""
    except OSError:
        raw = ""
    if raw == _TUNE["raw"]:
        return _TUNE["settings"]
    _TUNE["raw"] = raw
    try:
        fresh = parsed_tune(raw) if raw.strip() else dict(TUNE_DEFAULT)
    except (ValueError, TypeError) as failure:
        print(f"QSA_KV8 tune file {path} is unreadable ({failure}); keeping the last prefill tuning",
              flush=True)
        fresh = dict(_TUNE["settings"])
    if fresh != _TUNE["settings"]:
        _TUNE["settings"] = fresh
        print(f"QSA_KV8 tune now block_n={fresh['block_n']} num_warps={fresh['num_warps']} "
              f"num_splits={fresh['num_splits'] or 'auto'}, read from {path or '(unset)'}", flush=True)
    return _TUNE["settings"]


def prefill_enabled():
    return os.environ.get(PREFILL_ENV) == "1"


def prefill_launch(num_rows, num_kv_heads, use_prefill_config):
    return bool(use_prefill_config) and num_rows * num_kv_heads > PREFILL_MIN_PROGRAMS


def overlap_depth():
    said = str(os.environ.get(OVERLAP_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def overlap_budget():
    return OVERLAP_SAMPLES if overlap_depth() else 1


def default_prefill():
    return {**PREFILL_DEFAULT, **prefill_config(), "overlap": overlap_depth()}


def prefill_config():
    said = str(os.environ.get(PREFILL_CONFIG_ENV) or "").strip()
    if not said:
        return {}
    parts = said.split(",")
    if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"{PREFILL_CONFIG_ENV}={said!r} is not 'block_n,num_warps,num_stages'")
    block_n, num_warps, num_stages = (int(part) for part in parts)
    if block_n < 16 or block_n & (block_n - 1):
        raise ValueError(f"{PREFILL_CONFIG_ENV}: block_n {block_n} must be a power of 2 >= 16")
    if not 1 <= num_warps <= 16 or num_warps & (num_warps - 1):
        raise ValueError(f"{PREFILL_CONFIG_ENV}: num_warps {num_warps} must be a power of 2 from 1 to 16")
    if not 1 <= num_stages <= 4:
        raise ValueError(f"{PREFILL_CONFIG_ENV}: num_stages {num_stages} must be from 1 to 4")
    announce("prefill_config", knob=PREFILL_CONFIG_ENV, block_n=block_n, num_warps=num_warps, num_stages=num_stages)
    return {"block_n": block_n, "num_warps": num_warps, "num_stages": num_stages}


def parsed_prefill(raw):
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("the QSA prefill file is not a JSON object")
    out = default_prefill()
    if "mode" in data:
        out["mode"] = str(data["mode"])
    for key in ("block_n", "num_warps", "num_stages", "overlap"):
        if key in data:
            out[key] = int(data[key])
    if out["mode"] not in PREFILL_MODES:
        raise ValueError(f"mode {out['mode']!r} is not one of {PREFILL_MODES}")
    if out["block_n"] < 16 or out["block_n"] & (out["block_n"] - 1):
        raise ValueError(f"block_n {out['block_n']} must be a power of 2 >= 16")
    if out["num_warps"] < 1 or out["num_warps"] & (out["num_warps"] - 1):
        raise ValueError(f"num_warps {out['num_warps']} must be a power of 2 >= 1")
    if out["num_stages"] < 1:
        raise ValueError(f"num_stages {out['num_stages']} must be at least 1")
    return out


def prefill_settings(now=None):
    now = time.monotonic() if now is None else now
    if now - _PREFILL["read_at"] < TUNE_READ_EVERY_S:
        return _PREFILL["settings"]
    _PREFILL["read_at"] = now
    path = str(os.environ.get(PREFILL_FILE_ENV) or "")
    try:
        raw = open(path, encoding="utf-8").read() if path else ""
    except OSError:
        raw = ""
    if raw == _PREFILL["raw"]:
        return _PREFILL["settings"]
    _PREFILL["raw"] = raw
    _PREFILL["measured"] = 0
    try:
        fresh = parsed_prefill(raw) if raw.strip() else default_prefill()
    except (ValueError, TypeError) as failure:
        print(f"QSA_PREFILL file {path} is unreadable ({failure}); keeping the last prefill settings",
              flush=True)
        fresh = dict(_PREFILL["settings"])
    if fresh != _PREFILL["settings"]:
        _PREFILL["settings"] = fresh
        print(f"QSA_PREFILL now mode={fresh['mode']} block_n={fresh['block_n']} "
              f"num_warps={fresh['num_warps']} num_stages={fresh['num_stages']} "
              f"overlap={fresh['overlap']}, read from {path or '(unset)'}", flush=True)
    return _PREFILL["settings"]


def overlap(logical_indices, token_to_req, groups=OVERLAP_GROUPS, depth=0):
    width = logical_indices.shape[1] - 1
    rows = logical_indices.shape[0]
    tokens = logical_indices[:, :width]
    counts = logical_indices[:, width].clamp(0, width)
    columns = torch.arange(width, device=logical_indices.device)
    valid = (tokens >= 0) & (columns[None, :] < counts[:, None]) & (tokens < OVERLAP_TOKEN_SPAN)
    requests = token_to_req[:rows].to(torch.int32)
    keys = torch.where(valid, requests[:, None] * OVERLAP_TOKEN_SPAN + tokens, torch.full_like(tokens, -1))
    full = (counts == width) & (tokens.max(dim=1).values >= depth)
    out = {}
    for group in groups:
        usable = rows // group * group
        if not usable:
            continue
        owners = requests[:usable].reshape(-1, group)
        same = (owners == owners[:, :1]).all(dim=1) & (owners[:, 0] >= 0)
        same &= full[:usable].reshape(-1, group).all(dim=1)
        tiles = keys[:usable].reshape(-1, group * width)[same]
        if not tiles.shape[0]:
            continue
        ordered = tiles.sort(dim=1).values
        fresh = torch.ones_like(ordered, dtype=torch.bool)
        fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        held = ordered >= 0
        out[group] = (int(tiles.shape[0]), int((fresh & held).sum()), int(held.sum()))
    return out


def overlap_ready(logical_indices, depth=0):
    if logical_indices.is_cuda and torch.cuda.is_current_stream_capturing():
        announce("overlap_capture", knob=OVERLAP_ENV,
                 skipped="the overlap count reads the picks on the host and is never taken inside a CUDA graph capture")
        return False
    width = logical_indices.shape[1] - 1
    deep = logical_indices[:, :width].max(dim=1).values >= depth
    full = int(((logical_indices[:, width] >= width) & deep).sum())
    return full >= OVERLAP_FULL_SHARE * logical_indices.shape[0]


def announce_overlap(logical_indices, token_to_req, depth=0):
    found = overlap(logical_indices, token_to_req, depth=depth)
    parts = [f"G={group}: {tiles} full tiles of one request, the union is {union / max(selected, 1):.3f} "
             f"of the rows' selections ({union * group / max(selected, 1):.2f}x one row's)"
             for group, (tiles, union, selected) in found.items()]
    print(f"QSA_PREFILL overlap rows={logical_indices.shape[0]} width={logical_indices.shape[1] - 1} "
          f"depth={depth}; "
          + ("; ".join(parts) if parts else "no full tile of one request"), flush=True)
    return found


def reuse_budget():
    if _REUSE["budget"] is None:
        said = str(os.environ.get(REUSE_ENV) or "").strip()
        budget = int(said) if said else 0
        if budget < 0:
            raise ValueError(f"{REUSE_ENV} must be a count of decode steps >= 0, found {said!r}")
        _REUSE["budget"] = budget
        if budget:
            print(f"QSA_REUSE armed: the first {budget} decode steps outside CUDA graphs that hold a request "
                  f"whose newest selected token is at {REUSE_MIN_CONTEXT} or beyond record, for up to "
                  f"{REUSE_REQUESTS} of the longest such requests in every QSA layer, how much of each step's "
                  f"selection the "
                  f"previous {', '.join(map(str, REUSE_HISTORY))} steps' selections and the newest "
                  f"{', '.join(map(str, REUSE_RECENT))} tokens already hold", flush=True)
    return _REUSE["budget"]


def reuse_tally(history, chosen):
    if not chosen:
        return None
    newest = max(chosen)
    found = {"selected": len(chosen), "newest": newest, "history": {}, "recent": {}}
    union = set()
    past = list(history)
    for depth in range(1, max(REUSE_HISTORY) + 1):
        if depth > len(past):
            break
        union |= past[-depth]
        if depth in REUSE_HISTORY:
            found["history"][depth] = (len(chosen & union), len(union | chosen))
    for window in REUSE_RECENT:
        found["recent"][window] = sum(1 for token in chosen if token > newest - window)
    steps, window = REUSE_MIX
    if len(past) >= steps:
        held = set().union(*past[-steps:])
        found["mix"] = (sum(1 for token in chosen if token in held or token > newest - window),
                        window + sum(1 for token in held | chosen if token <= newest - window))
    return found


def reuse_band(newest):
    for index, edge in enumerate(REUSE_BANDS):
        if newest < edge:
            return index
    return len(REUSE_BANDS)


def reuse_add(tally, layer, found):
    band = tally.setdefault(reuse_band(found["newest"]), {"samples": 0, "selected": 0, "history": {}, "recent": {},
                                                          "mix": [0, 0, 0], "layers": {}})
    band["samples"] += 1
    band["selected"] += found["selected"]
    for depth, (hit, footprint) in found["history"].items():
        cell = band["history"].setdefault(depth, [0, 0, 0, 0])
        cell[0] += hit
        cell[1] += found["selected"]
        cell[2] += footprint
        cell[3] += 1
    for window, hit in found["recent"].items():
        band["recent"][window] = band["recent"].get(window, 0) + hit
    if "mix" in found:
        band["mix"][0] += found["mix"][0]
        band["mix"][1] += found["selected"]
        band["mix"][2] += found["mix"][1]
    depth = REUSE_MIX[0]
    if depth in found["history"]:
        cell = band["layers"].setdefault(layer, [0, 0])
        cell[0] += found["history"][depth][0]
        cell[1] += found["selected"]


def reuse_lines(tally, steps):
    names = [f"newest token below {REUSE_BANDS[0]}"] + [
        f"newest token {low} to {high}" for low, high in zip(REUSE_BANDS, REUSE_BANDS[1:])] + [
        f"newest token at {REUSE_BANDS[-1]} or more"]
    lines = []
    for index in sorted(tally):
        band = tally[index]
        parts = []
        for depth in REUSE_HISTORY:
            cell = band["history"].get(depth)
            if cell and cell[1]:
                parts.append(f"last {depth} step(s) hold {100 * cell[0] / cell[1]:.1f}% "
                             f"({cell[2] / cell[3]:,.0f} tokens kept)")
        recent = ", ".join(f"{window} {100 * band['recent'].get(window, 0) / max(band['selected'], 1):.1f}%"
                           for window in REUSE_RECENT)
        mix = band["mix"]
        mixed = (f"; last {REUSE_MIX[0]} steps or the newest {REUSE_MIX[1]} tokens hold "
                 f"{100 * mix[0] / mix[1]:.1f}% ({mix[2] / max(band['samples'], 1):,.0f} tokens kept)"
                 if mix[1] else "")
        layers = [hit / total for hit, total in band["layers"].values() if total]
        spread = (f"; layers' last-{REUSE_MIX[0]} share {100 * min(layers):.1f}-{100 * max(layers):.1f}%"
                  if layers else "")
        lines.append(f"QSA_REUSE after {steps} decode steps, {names[index]}, {len(band['layers'])} layer(s): "
                     f"{band['samples']} request-layer "
                     f"samples, {band['selected'] / max(band['samples'], 1):,.0f} tokens selected each; "
                     + "; ".join(parts) + f"; the newest tokens hold {recent}" + mixed + spread)
    return lines


def reuse_report(final=False):
    now = time.monotonic()
    if _REUSE["reported"] is None:
        _REUSE["reported"] = now
    if not final and now - _REUSE["reported"] < REUSE_REPORT_S:
        return
    _REUSE["reported"] = now
    lines = reuse_lines(_REUSE["tally"], _REUSE["steps"])
    for line in lines or [f"QSA_REUSE after {_REUSE['steps']} decode steps: no sampled request had a history yet"]:
        print(line, flush=True)


def probe_reuse(kv_cache, logical_indices, block_table, token_to_req):
    budget = reuse_budget()
    if not budget or _REUSE["closed"]:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    layers = _REUSE["layers"]
    layer = layers.setdefault(int(kv_cache.data_ptr()), len(layers))
    rows = logical_indices.shape[0]
    width = logical_indices.shape[1] - 1
    owners = token_to_req[:rows].to(torch.int64)
    row_keys = block_table[owners, 0].to(torch.int64).cpu().tolist()
    if layer == 0 and _REUSE["steps"] >= budget:
        _REUSE["closed"] = True
        reuse_report(final=True)
        return
    newest = logical_indices[:, :width].max(dim=1).values.cpu().tolist()
    longest = {}
    for key, top in zip(row_keys, newest):
        longest[key] = max(longest.get(key, -1), int(top))
    held = _REUSE["keys"].get(layer, [])
    kept = [key for key in held if longest.get(key, -1) >= REUSE_MIN_CONTEXT]
    fresh = sorted((key for key, top in longest.items() if top >= REUSE_MIN_CONTEXT and key not in kept),
                   key=lambda key: -longest[key])
    kept += fresh[:REUSE_REQUESTS - len(kept)]
    _REUSE["keys"][layer] = kept
    for stale in [one for one in _REUSE["history"] if one[0] == layer and one[1] not in kept]:
        del _REUSE["history"][stale]
    if layer == 0 and kept:
        _REUSE["steps"] += 1
    sample = set(kept)
    picked = [row for row, key in enumerate(row_keys) if key in sample]
    if not picked:
        return
    chosen_rows = logical_indices.index_select(0, torch.tensor(picked, device=logical_indices.device)).cpu()
    chosen = {}
    for position, row in enumerate(picked):
        count = max(0, min(int(chosen_rows[position, width]), width))
        tokens = chosen_rows[position, :count]
        chosen.setdefault(row_keys[row], set()).update(int(token) for token in tokens[tokens >= 0].tolist())
    for key, tokens in chosen.items():
        history = _REUSE["history"].setdefault((layer, key), collections.deque(maxlen=max(REUSE_HISTORY)))
        found = reuse_tally(history, tokens)
        if found is not None and found["history"]:
            reuse_add(_REUSE["tally"], layer, found)
        history.append(tokens)
    reuse_report()


def select_config(pinned, num_rows, num_kv_heads, use_prefill_config, num_columns, fast):
    block_n, num_warps, num_tiles, num_splits = pinned._select_config(
        num_rows, num_kv_heads, use_prefill_config, num_columns)
    if not fast:
        return block_n, num_warps, num_tiles, num_splits
    widen = use_prefill_config and num_rows * num_kv_heads > 2048
    if widen:
        tune = tune_settings()
        block_n = tune["block_n"]
        num_tiles = triton.cdiv(num_columns, block_n)
        num_splits = min(tune["num_splits"] or num_splits, num_tiles)
        num_warps = tune["num_warps"]
    else:
        num_warps = max(num_warps, FAST_MIN_WARPS)
    announce("fast_config_prefill" if widen else "fast_config_decode",
             block_n=block_n, num_warps=num_warps, num_tiles=num_tiles, num_splits=num_splits)
    return block_n, num_warps, num_tiles, num_splits


def int8_layout():
    if _STATE["layout"] is None:
        held = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.is_file() else {}
        chosen = held.get("layout", "int8")
        if chosen not in ("int8", "k4v8"):
            raise ValueError(f"QSA_KV8 settings name layout {chosen!r}; an int8 cache serves int8 or k4v8")
        _STATE["layout"] = chosen
    return _STATE["layout"]


def layout(cache_dtype):
    if cache_dtype == INT4_DTYPE:
        return "int4"
    if cache_dtype == CACHE_DTYPE:
        return int8_layout()
    raise ValueError(f"QSA_KV8 serves {CACHE_DTYPES}, not {cache_dtype!r}")


def widths(chosen, head_size, head_size_v=None):
    head_size_v = head_size if head_size_v is None else head_size_v
    if chosen == "int4":
        return head_size // 2, head_size_v // 2
    if chosen == "k4v8":
        return head_size // 2, head_size_v
    return head_size, head_size_v


def customize_spec(spec):
    from vllm.v1.kv_cache_interface import KVQuantMode

    if spec.state_content_bytes is not None:
        return spec
    if spec.kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
        chosen = int8_layout()
    elif spec.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        chosen = "int4"
    else:
        return spec
    key_width, value_width = widths(chosen, spec.head_size, spec.head_size_v)
    return dataclasses.replace(spec, state_content_bytes=content_bytes(key_width, value_width))


def content_bytes(key_width, value_width):
    used = key_width + value_width + 2 * SCALE_BYTES
    return -(-used // ROW_ALIGN) * ROW_ALIGN


def views(holder, kv_cache, head_size, chosen="int8"):
    key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()),
           kv_cache.storage_offset(), head_size, chosen)
    held = getattr(holder, "_taaf_kv8_views", None)
    if held is not None and held[0] == key:
        return held[1]
    if chosen not in LAYOUTS:
        raise ValueError(f"QSA_KV8 has no layout {chosen!r}")
    storage_dtype = torch.uint8 if chosen == "int4" else torch.int8
    key_width, value_width = widths(chosen, head_size)
    if kv_cache.ndim != 4 or kv_cache.dtype != storage_dtype:
        raise ValueError(f"QSA_KV8 expects a 4-D {storage_dtype} cache, got {kv_cache.dtype} "
                         f"{tuple(kv_cache.shape)}")
    blocks, heads, tokens, content = kv_cache.shape
    value_start = key_width
    scale_start = key_width + value_width
    if content != content_bytes(key_width, value_width):
        raise ValueError(f"QSA_KV8 cache content {content} does not hold a {chosen} head of {head_size}: "
                         f"codes, two {SCALE_BYTES}-byte scales, padded to {ROW_ALIGN} bytes")
    strides = kv_cache.stride()
    offset = kv_cache.storage_offset()
    if (any(stride % SCALE_BYTES for stride in strides[:3]) or offset % SCALE_BYTES
            or key_width % SCALE_BYTES or value_width % SCALE_BYTES):
        raise ValueError(f"QSA_KV8 scales are not {SCALE_BYTES}-byte aligned: strides {strides}, offset {offset}")
    tokens_first = kv_cache.transpose(1, 2)
    keys = tokens_first[..., :key_width]
    values = tokens_first[..., value_start:value_start + value_width]
    if chosen == "k4v8":
        keys = keys.view(torch.uint8)
    base = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    shape = (blocks, tokens, heads)
    scale_strides = (strides[0] // SCALE_BYTES, strides[2] // SCALE_BYTES, strides[1] // SCALE_BYTES)
    key_scales = torch.as_strided(base, shape, scale_strides, (offset + scale_start) // SCALE_BYTES)
    value_scales = torch.as_strided(base, shape, scale_strides,
                                    (offset + scale_start + SCALE_BYTES) // SCALE_BYTES)
    made = (keys, values, key_scales, value_scales)
    holder._taaf_kv8_views = (key, made)
    return made


def write(impl, key, value, kv_cache, slot_mapping):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    chosen = layout(impl.kv_cache_dtype)
    keys, values, key_scales, value_scales = views(impl, kv_cache, impl.head_size, chosen)
    rows = slot_mapping.shape[0]
    announce("write", layout=chosen, cache=tuple(kv_cache.shape), rows=rows)
    if chosen == "k4v8":
        write_k4v8(key[:rows], value[:rows], keys, values, key_scales, value_scales, slot_mapping)
        return
    triton_reshape_and_cache_flash_per_token_head_quant(
        key[:rows], value[:rows], keys, values, key_scales, value_scales, slot_mapping,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD if chosen == "int4" else KVQuantMode.INT8_PER_TOKEN_HEAD)


@contextlib.contextmanager
def last_rows_scope(rows):
    held = _ROWS["last"]
    _ROWS["last"] = rows
    try:
        yield
    finally:
        _ROWS["last"] = held


def full_batch_config(chosen, pinned, num_rows, num_kv_heads, use_prefill_config, num_columns):
    if chosen == "int8" and not int8_fast():
        return select_config(pinned, num_rows, num_kv_heads, use_prefill_config, num_columns, fast_enabled())
    return pinned._select_config(num_rows, num_kv_heads, use_prefill_config, num_columns)


def attention_rows(rows, impl, query, kv_cache, logical_indices, block_table, token_to_req,
                   use_prefill_config, out, output_gate):
    chosen = layout(impl.kv_cache_dtype)
    keys, values, key_scales, value_scales = views(impl, kv_cache, impl.head_size, chosen)
    full = query.shape[0]
    rows = rows.clamp(0, full - 1)
    announce("attention_rows", layout=chosen, rows=full, kept=rows.shape[0],
             prefill=bool(use_prefill_config))
    sub_query = query.index_select(0, rows)
    sub_indices = logical_indices.index_select(0, rows)
    sub_requests = token_to_req.index_select(0, rows)
    sub_gate = output_gate.index_select(0, rows)
    sub_out = torch.empty_like(sub_query)
    fused = None
    if prefill_enabled() and prefill_launch(full, keys.shape[2], use_prefill_config):
        settings = prefill_settings()
        if settings["mode"] == "fused":
            fused = settings
    if fused is not None:
        prefill(chosen, fused, sub_query, keys, values, key_scales, value_scales, sub_indices,
                block_table, sub_requests, sub_out, sub_gate)
    else:
        config = full_batch_config(chosen, _pinned(), full, keys.shape[2], use_prefill_config,
                                   logical_indices.shape[1] - 1)
        run = {"int8": sparse_attention, "int4": sparse_attention_int4,
               "k4v8": sparse_attention_k4v8}[chosen]
        run(sub_query, keys, values, key_scales, value_scales, sub_indices, block_table, sub_requests,
            use_prefill_config, sub_out, output_gate=sub_gate, config=config)
    out.index_copy_(0, rows, sub_out)
    return out


def attention(impl, query, kv_cache, logical_indices, block_table, token_to_req,
              use_prefill_config, out, output_gate, work=None):
    held = _ROWS["last"]
    bucketed = bool(getattr(use_prefill_config, "tokens", 0))
    if held is not None and use_prefill_config:
        if bucketed:
            from vllm.taaf_qsa_graph import aside

            aside("host_kv_rows", "the host-KV last-rows path chooses rows on the host")
        return attention_rows(held, impl, query, kv_cache, logical_indices, block_table, token_to_req,
                              use_prefill_config, out, output_gate)
    chosen = layout(impl.kv_cache_dtype)
    keys, values, key_scales, value_scales = views(impl, kv_cache, impl.head_size, chosen)
    announce("attention", layout=chosen, rows=query.shape[0],
             width=logical_indices.shape[1] - 1, prefill=bool(use_prefill_config))
    if not use_prefill_config:
        probe_reuse(kv_cache, logical_indices, block_table, token_to_req)
    if prefill_enabled() and prefill_launch(query.shape[0], keys.shape[2], use_prefill_config):
        settings = prefill_settings()
        depth = settings["overlap"]
        if (depth and not bucketed and _PREFILL["measured"] < overlap_budget()
                and not torch.cuda.is_current_stream_capturing() and overlap_ready(logical_indices, depth)):
            _PREFILL["measured"] += 1
            announce_overlap(logical_indices, token_to_req, depth)
        if settings["mode"] == "fused":
            return prefill(chosen, settings, query, keys, values, key_scales, value_scales,
                           logical_indices, block_table, token_to_req, out, output_gate)
    run = {"int8": sparse_attention, "int4": sparse_attention_int4, "k4v8": sparse_attention_k4v8}[chosen]
    extra = {"work": work} if work is not None and follows_int4(chosen) and not use_prefill_config else {}
    return run(query, keys, values, key_scales, value_scales, logical_indices,
               block_table, token_to_req, use_prefill_config, out, output_gate=output_gate, **extra)


def prefill(chosen, settings, query, keys, values, key_scales, value_scales, logical_indices,
            block_table, token_to_req, out, output_gate):
    block_n = settings["block_n"]
    num_warps = settings["num_warps"]
    num_stages = settings["num_stages"]
    num_tiles = triton.cdiv(logical_indices.shape[1] - 1, block_n)
    announce("prefill", layout=chosen, rows=query.shape[0], block_n=block_n, num_warps=num_warps,
             num_stages=num_stages, rotation={"int8": "none", "int4": "in-kernel", "k4v8": "torch"}[chosen])
    if follows_int4(chosen) and lean_enabled():
        return prefill_leaned(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                              token_to_req, out, output_gate, block_n=block_n, num_warps=num_warps,
                              num_stages=num_stages, chosen=chosen)
    if follows_int4(chosen) and tile_rows():
        return prefill_tiled(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                             token_to_req, out, output_gate, block_n=block_n, num_warps=num_warps,
                             num_stages=num_stages, chosen=chosen)
    if chosen == "int4":
        return prefill_int4(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                            token_to_req, out, output_gate, block_n=block_n, num_warps=num_warps,
                            num_stages=num_stages)
    run = sparse_attention if chosen == "int8" else sparse_attention_k4v8
    return run(query, keys, values, key_scales, value_scales, logical_indices, block_table, token_to_req,
               True, out, output_gate=output_gate, config=(block_n, num_warps, num_tiles, 1),
               num_stages=num_stages)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_kv8_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    work_ptr=None,
    GATED: tl.constexpr = False,
    PDL: tl.constexpr = False,
    counter_ptr=None,
    BLOCK_SPLITS: tl.constexpr = 1,
    FUSED: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    valid_count = tl.where((request >= 0) & (request < num_requests), valid_count, 0)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_codes = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_codes = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = (
            safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        )
        key_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0)
        value_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
        scores = tl.dot(query, key_codes.to(query.dtype))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weights = probabilities * value_scale[None, :]
        accumulator = tl.dot(
            weights.to(query.dtype),
            value_codes.to(query.dtype),
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )
        if FUSED:
            tl.debug_barrier()
            slot = row * tl.num_programs(1) + kv_head
            seen = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
            if seen == NUM_SPLITS - 1:
                for member in range(GROUP_SIZE):
                    _kv8_merge_head(
                        partial_output_ptr, partial_lse_ptr, output_ptr, output_gate_ptr, stride_output_row,
                        stride_output_head, stride_output_gate_row, stride_output_gate_head, num_rows, row,
                        first_head + member, HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, BLOCK_SPLITS, True,
                    )
                tl.atomic_xchg(counter_ptr + slot, 0, sem="relaxed", scope="gpu")


def _pinned():
    import importlib

    return importlib.import_module(PINNED_OPS)


def sparse_attention(q, key_codes, value_codes, key_scales, value_scales, logical_indices,
                     block_table, token_to_req, use_prefill_config, out=None, *, output_gate,
                     config=None, num_stages=2, work=None):
    pinned = _pinned()
    if q.ndim != 3 or key_codes.ndim != 4 or value_codes.shape != key_codes.shape:
        raise ValueError("QSA_KV8 attention received invalid Q/K/V shapes")
    if key_scales.shape != key_codes.shape[:3] or value_scales.shape != key_scales.shape:
        raise ValueError("QSA_KV8 scales must be [blocks, page, kv_heads]")
    if key_scales.stride() != value_scales.stride():
        raise ValueError("QSA_KV8 key and value scales must share one layout")
    if key_codes.stride() != value_codes.stride():
        raise ValueError("QSA_KV8 key and value codes must share one layout")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA_KV8 indices must have one row per query")
    if logical_indices.shape[1] < 2:
        raise ValueError("QSA_KV8 packed indices need selection columns plus the count column")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA_KV8 metadata has invalid shapes")
    if not all(key_codes.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA_KV8 cache and block table must be nonempty")
    if q.shape[2] != key_codes.shape[3] or q.shape[1] % key_codes.shape[2]:
        raise ValueError("QSA_KV8 attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert key_codes.dtype == value_codes.dtype == torch.int8
    assert key_scales.dtype == value_scales.dtype == torch.float32
    assert logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert q.stride(2) == key_codes.stride(3) == value_codes.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // key_codes.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1
    fast = int8_fast()
    if config is None and fast:
        config = int4_config(pinned, q.shape[0], key_codes.shape[2], use_prefill_config, selection_width)
    block_n, partial_warps, num_tiles, num_splits = config or select_config(
        pinned, q.shape[0], key_codes.shape[2], use_prefill_config, selection_width, fast_enabled()
    )
    fused = fast and decode_fused_enabled()
    gated = {"work_ptr": work, "GATED": True} if fused and work is not None else {}
    in_kernel = {}
    if splitk_fused() and num_splits > 1:
        held = splitk_counters(q.device, q.shape[0] * key_codes.shape[2])
        if held is None:
            splitk_aside("counters", rows=q.shape[0], kv_heads=key_codes.shape[2], layout="int8")
        else:
            in_kernel = {"counter_ptr": held, "BLOCK_SPLITS": triton.next_power_of_2(num_splits), "FUSED": True}
    if fused:
        announce("decode_fused_int8_gated" if gated else "decode_fused_int8", rows=q.shape[0], block_n=block_n,
                 num_warps=partial_warps, num_splits=num_splits,
                 kernels="split+merge in one" if in_kernel else "split+merge")
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32,
                                  device=q.device)

    _qsa_sparse_kv8_splitk_kernel[(q.shape[0], key_codes.shape[2], num_splits)](
        q,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=num_stages,
        **gated,
        **in_kernel,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    if num_splits == 1 or in_kernel:
        return out

    merge = _qsa_kv8_merge_kernel if fused else pinned._qsa_merge_splitk_kernel
    merge[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
        **gated,
        **pdl_args(merge),
    )
    return out


@triton.jit
def _kv8_merge_head(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    row,
    head,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    CG: tl.constexpr,
):
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    if CG:
        lse = tl.load(
            partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
            mask=split_mask,
            other=-float("inf"),
            cache_modifier=".cg",
        )
    else:
        lse = tl.load(
            partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
            mask=split_mask,
            other=-float("inf"),
        )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(has_values, tl.where(split_mask, lse - lse_max, -float("inf")), -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    if CG:
        partial_output = tl.load(
            partial_output_ptr
            + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
            + dim_offsets[None, :],
            mask=split_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        partial_output = tl.load(
            partial_output_ptr
            + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
            + dim_offsets[None, :],
            mask=split_mask[:, None],
            other=0.0,
        )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets, merged)


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_kv8_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    work_ptr=None,
    GATED: tl.constexpr = False,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    _kv8_merge_head(
        partial_output_ptr, partial_lse_ptr, output_ptr, output_gate_ptr, stride_output_row, stride_output_head,
        stride_output_gate_row, stride_output_gate_head, num_rows, tl.program_id(0), tl.program_id(1),
        HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, BLOCK_SPLITS, False,
    )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_kv4_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    work_ptr=None,
    GATED: tl.constexpr = False,
    PDL: tl.constexpr = False,
    counter_ptr=None,
    signs_ptr=None,
    final_ptr=None,
    gate_ptr=None,
    stride_final_row=0,
    stride_final_head=0,
    stride_gate_row=0,
    stride_gate_head=0,
    CHUNK: tl.constexpr = 32,
    FUSED: tl.constexpr = False,
    LEAN: tl.constexpr = False,
    PARTIAL_SLOT: tl.constexpr = -1,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    HALF: tl.constexpr = HEAD_DIM // 2
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    valid_count = tl.where((request >= 0) & (request < num_requests), valid_count, 0)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    query_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    query_even = tl.load(query_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
    query_odd = tl.load(query_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 / HEAD_DIM

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        if LEAN:
            slot = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
            valid = slot >= 0
            safe_slot = tl.maximum(slot, 0)
            safe_page = (safe_slot // PAGE_SIZE).to(tl.int64)
            page_offset = safe_slot % PAGE_SIZE
        else:
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_offset = safe_token % PAGE_SIZE
            valid = (
                (request >= 0)
                & (request < num_requests)
                & (logical_token >= 0)
                & (logical_page < PAGE_TABLE_WIDTH)
            )
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = (
            safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        )
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        if LEAN:
            key_zero = 1.0 + (key_bits & 0xF).to(tl.float32) * 0.0625
            key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
            value_zero = 1.0 + (value_bits & 0xF).to(tl.float32) * 0.0625
            value_scale = (value_bits & -16).to(tl.float32, bitcast=True) * 16.0
            key_low, key_high = _lean_nibbles(key_packed)
            raw = _bf16_dot(query_even, key_low, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
            raw = _bf16_dot(query_odd, key_high, raw)
            scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * (softmax_scale_log2 * 16.0))
        else:
            key_zero = (key_bits & 0xF).to(tl.float32)
            key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
            value_zero = (value_bits & 0xF).to(tl.float32)
            value_scale = (value_bits & -16).to(tl.float32, bitcast=True)
            key_low = (key_packed & 0xF).to(query_even.dtype)
            key_high = ((key_packed >> 4) & 0xF).to(query_even.dtype)
            raw = tl.dot(query_even, key_low) + tl.dot(query_odd, key_high)
            scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weights = (probabilities * value_scale[None, :]).to(query_even.dtype)
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        if LEAN:
            value_low, value_high = _lean_nibbles(value_packed)
            accumulator_even = _bf16_dot(weights, value_low, accumulator_even * alpha[:, None]) - weighted_zero[:, None]
            accumulator_odd = _bf16_dot(weights, value_high, accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        else:
            value_low = (value_packed & 0xF).to(query_even.dtype)
            value_high = ((value_packed >> 4) & 0xF).to(query_even.dtype)
            accumulator_even = tl.dot(weights, value_low, acc=accumulator_even * alpha[:, None]) - weighted_zero[:, None]
            accumulator_odd = tl.dot(weights, value_high, acc=accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    output_even = tl.where(has_values[:, None], accumulator_even / denominator, 0.0)
    output_odd = tl.where(has_values[:, None], accumulator_odd / denominator, 0.0)
    if NUM_SPLITS == 1 and PARTIAL_SLOT < 0:
        output_rows = output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
        tl.store(output_rows + 2 * half_offsets[None, :], output_even, mask=head_mask)
        tl.store(output_rows + 2 * half_offsets[None, :] + 1, output_odd, mask=head_mask)
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = ((split_id if PARTIAL_SLOT < 0 else PARTIAL_SLOT) * num_rows + row).to(tl.int64)
        partial_rows = partial_output_ptr + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None]) * HEAD_DIM
        tl.store(partial_rows + 2 * half_offsets[None, :], output_even, mask=head_mask)
        tl.store(partial_rows + 2 * half_offsets[None, :] + 1, output_odd, mask=head_mask)
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )
    if FUSED:
        tl.debug_barrier()
        if NUM_SPLITS == 1:
            _kv4_merge_restore(
                partial_output_ptr, partial_lse_ptr, signs_ptr, final_ptr, gate_ptr, stride_final_row,
                stride_final_head, stride_gate_row, stride_gate_head, num_rows, row, kv_head, GROUP_SIZE,
                HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, BLOCK_M, CHUNK, True,
            )
        else:
            slot = row * tl.num_programs(1) + kv_head
            seen = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
            if seen == NUM_SPLITS - 1:
                _kv4_merge_restore(
                    partial_output_ptr, partial_lse_ptr, signs_ptr, final_ptr, gate_ptr, stride_final_row,
                    stride_final_head, stride_gate_row, stride_gate_head, num_rows, row, kv_head, GROUP_SIZE,
                    HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, BLOCK_M, CHUNK, True,
                )
                tl.atomic_xchg(counter_ptr + slot, 0, sem="relaxed", scope="gpu")


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_kv4_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(has_values, tl.where(split_mask, lse - lse_max, -float("inf")), -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets, merged)


def _rht():
    import importlib

    return importlib.import_module(PINNED_INT4).single_rht


def check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req):
    if q.ndim != 3 or key_codes.ndim != 4 or value_codes.shape != key_codes.shape:
        raise ValueError("QSA_KV8 int4 attention received invalid Q/K/V shapes")
    if key_scales.shape != key_codes.shape[:3] or value_scales.shape != key_scales.shape:
        raise ValueError("QSA_KV8 int4 scales must be [blocks, page, kv_heads]")
    if key_scales.stride() != value_scales.stride() or key_codes.stride() != value_codes.stride():
        raise ValueError("QSA_KV8 int4 keys and values must share one layout")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0] or logical_indices.shape[1] < 2:
        raise ValueError("QSA_KV8 int4 indices must have one row per query plus the count column")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA_KV8 int4 metadata has invalid shapes")
    if not all(key_codes.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA_KV8 int4 cache and block table must be nonempty")
    if q.shape[2] != 2 * key_codes.shape[3] or q.shape[1] % key_codes.shape[2]:
        raise ValueError("QSA_KV8 int4 attention requires two packed values per byte and grouped heads")
    head_dim = q.shape[2]
    assert head_dim >= 32 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert key_codes.dtype == value_codes.dtype == torch.uint8
    assert key_scales.dtype == value_scales.dtype == torch.float32
    assert logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert key_codes.stride(3) == value_codes.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1


def sparse_attention_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices,
                          block_table, token_to_req, use_prefill_config, out=None, *, output_gate,
                          config=None, num_stages=2, work=None):
    pinned = _pinned()
    single_rht = _rht()
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 int4 output must match its query")
    assert out.dtype == q.dtype
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // key_codes.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1
    block_n, partial_warps, num_tiles, num_splits = config or int4_config(
        pinned, q.shape[0], key_codes.shape[2], use_prefill_config, selection_width
    )
    if decode_fused_enabled() and q.stride(2) == 1 and out.stride(2) == 1:
        return decode_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices,
                           block_table, token_to_req, out, output_gate_view,
                           (block_n, partial_warps, num_tiles, num_splits), num_stages, work=work)
    rotated = single_rht(q.float()).to(q.dtype).contiguous()
    rotated_out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    if num_splits == 1:
        partial_output = rotated_out
        partial_lse = rotated_out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32,
                                  device=q.device)

    _qsa_sparse_kv4_splitk_kernel[(q.shape[0], key_codes.shape[2], num_splits)](
        rotated,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        rotated_out,
        rotated.stride(0),
        rotated.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        rotated_out.stride(0),
        rotated_out.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=num_stages,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    if num_splits > 1:
        _qsa_kv4_merge_kernel[(q.shape[0], q.shape[1])](
            partial_output,
            partial_lse,
            rotated_out,
            rotated_out.stride(0),
            rotated_out.stride(1),
            q.shape[0],
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=2,
            num_stages=1,
            PDL=pdl_on(), launch_pdl=pdl_on(),
        )
    restored = (single_rht(rotated_out, inverse=True) / head_dim).to(q.dtype)
    gated = restored.float() * torch.sigmoid(output_gate_view.float())
    out.copy_(gated.to(out.dtype))
    return out


@triton.jit
def _hadamard(rows, columns):
    bits = rows & columns
    bits = bits ^ (bits >> 16)
    bits = bits ^ (bits >> 8)
    bits = bits ^ (bits >> 4)
    bits = bits ^ (bits >> 2)
    bits = bits ^ (bits >> 1)
    return (1.0 - 2.0 * (bits & 1).to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16)


@triton.jit
def _bf16_dot(a, b, acc):
    return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc=acc)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_prefill_kv4_kernel(
    q_ptr,
    signs_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HALF: tl.constexpr = HEAD_DIM // 2
    NUM_CHUNKS: tl.constexpr = HEAD_DIM // CHUNK
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    valid_count = tl.where((request >= 0) & (request < num_requests), valid_count, 0)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    chunk_offsets = tl.arange(0, CHUNK)
    column_offsets = tl.arange(0, BLOCK_N)
    even_columns = 2 * half_offsets
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    query_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head

    rotated_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    rotated_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        signs = tl.load(signs_ptr + dims)
        part = tl.load(query_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        part = part * signs[None, :]
        rotated_even = _bf16_dot(part, _hadamard(dims[:, None], even_columns[None, :]), rotated_even)
        rotated_odd = _bf16_dot(part, _hadamard(dims[:, None], even_columns[None, :] + 1), rotated_odd)
    query_even = _bf16(rotated_even)
    query_odd = _bf16(rotated_odd)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 / HEAD_DIM

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(0, tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = (
            safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        )
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = (key_bits & 0xF).to(tl.float32)
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = (value_bits & 0xF).to(tl.float32)
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True)
        raw = _bf16_dot(query_even, key_packed & 0xF, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, (key_packed >> 4) & 0xF, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weights = _bf16(probabilities * value_scale[None, :])
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        accumulator_even = _bf16_dot(weights, value_packed & 0xF,
                                     accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, (value_packed >> 4) & 0xF,
                                    accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    output_even = _bf16(tl.where(has_values[:, None], accumulator_even / denominator, 0.0))
    output_odd = _bf16(tl.where(has_values[:, None], accumulator_odd / denominator, 0.0))
    output_rows = output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
    gate_rows = (output_gate_ptr + row * stride_output_gate_row
                 + (first_head + head_offsets[:, None]) * stride_output_gate_head)
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        restored = _bf16_dot(output_even, _hadamard(even_columns[:, None], dims[None, :]),
                             tl.zeros((BLOCK_M, CHUNK), dtype=tl.float32))
        restored = _bf16_dot(output_odd, _hadamard(even_columns[:, None] + 1, dims[None, :]), restored)
        signs = tl.load(signs_ptr + dims)
        restored = _bf16(_bf16(restored).to(tl.float32) * signs[None, :] / HEAD_DIM)
        gate = tl.load(gate_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        tl.store(output_rows + dims[None, :], _bf16(restored.to(tl.float32) * tl.sigmoid(gate)),
                 mask=head_mask)


def rotation_signs(head_dim, device):
    import importlib

    held = importlib.import_module(PINNED_INT4)._get_rht_signs(head_dim, 0, device)
    return held.to(torch.float32).contiguous()


def prefill_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                 token_to_req, out, output_gate, *, block_n, num_warps, num_stages):
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]
    assert q.stride(2) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 int4 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out
    signs = rotation_signs(head_dim, q.device)
    group_size = q.shape[1] // key_codes.shape[2]
    selection_width = logical_indices.shape[1] - 1
    _qsa_prefill_kv4_kernel[(q.shape[0], key_codes.shape[2])](
        q,
        signs,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        logical_indices,
        block_table,
        token_to_req,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_TILES=triton.cdiv(selection_width, block_n),
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=block_n,
        CHUNK=min(PREFILL_ROTATE_CHUNK, head_dim),
        num_warps=num_warps,
        num_stages=num_stages,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


def tile_rows():
    if _TILE["rows"] is None:
        said = str(os.environ.get(TILE_ENV) or "").strip()
        if said in ("", "0"):
            value = 0
        elif said.isdigit() and int(said) in TILE_SIZES:
            value = int(said)
        else:
            raise ValueError(f"{TILE_ENV}={said!r} is not '', '0' or one of {TILE_SIZES}")
        _TILE["rows"] = value
    return _TILE["rows"]


def tile_say(key, line):
    if key in _TILE["said"]:
        return
    _TILE["said"].add(key)
    print(line, flush=True)


def tile_union_fused():
    return os.environ.get(TILE_UNION_ENV) == "1"


def tile_skip_empty():
    return os.environ.get(TILE_SKIP_ENV) == "1"


@triton.jit
def _popcount32(word):
    word = word - ((word >> 1) & 0x55555555)
    word = (word & 0x33333333) + ((word >> 2) & 0x33333333)
    word = (word + (word >> 4)) & 0x0F0F0F0F
    return ((word * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _tile_set_bits(logical_ptr, stride_logical, words, row, reach, BOUND: tl.constexpr, WIDTH: tl.constexpr,
                   CHUNK: tl.constexpr):
    for start in range(0, WIDTH, CHUNK):
        columns = start + tl.arange(0, CHUNK)
        token = tl.load(logical_ptr + row.to(tl.int64) * stride_logical + columns, mask=columns < WIDTH, other=-1)
        held = (columns < reach) & (token >= 0) & (token < BOUND)
        safe = tl.where(held, token, 0)
        flag = (tl.full((CHUNK,), 1, tl.uint32) << (safe & 31).to(tl.uint32)).to(tl.int32, bitcast=True)
        tl.atomic_or(words + (safe >> 5), flag, mask=held)


@triton.jit(do_not_specialize=["logical_ptr", "stride_logical", "token_to_req_ptr", "union_ptr", "bits_ptr",
                               "count_ptr", "picked_ptr", "words_ptr", "ranks_ptr", "rows", "num_requests"])
def _tile_union_kernel(logical_ptr, stride_logical, token_to_req_ptr, union_ptr, bits_ptr, count_ptr, picked_ptr,
                       words_ptr, ranks_ptr, rows, num_requests, WIDTH: tl.constexpr, TILE: tl.constexpr,
                       BLOCK_N: tl.constexpr, SPAN: tl.constexpr, BOUND: tl.constexpr, WORDS: tl.constexpr,
                       CHUNK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tile = tl.program_id(0)
    out_width = TILE * WIDTH
    place = tl.arange(0, TILE)
    lane_rows = tile * TILE + place
    inside = lane_rows < rows
    request = tl.load(token_to_req_ptr + lane_rows, mask=inside, other=-1).to(tl.int32)
    counts = tl.load(logical_ptr + lane_rows.to(tl.int64) * stride_logical + WIDTH, mask=inside, other=0)
    counts = tl.minimum(tl.maximum(counts, 0), WIDTH)
    reach = (counts + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    usable = inside & (request >= 0) & (request < num_requests) & (counts > 0)
    base = tl.sum(tl.zeros((TILE,), tl.int32), 0)
    taken = tl.sum(tl.zeros((TILE,), tl.int32), 0)
    if tl.sum(usable.to(tl.int32), 0) > 0:
        words = words_ptr + tile.to(tl.int64) * WORDS
        ranks = ranks_ptr + tile.to(tl.int64) * WORDS
        bits_row = bits_ptr + tile.to(tl.int64) * out_width
        union_row = union_ptr + tile.to(tl.int64) * out_width
        for out_at in range(0, out_width, CHUNK):
            at = out_at + tl.arange(0, CHUNK)
            tl.store(bits_row + at, tl.zeros((CHUNK,), tl.int32), mask=at < out_width)
        remaining = usable
        for slot in tl.static_range(TILE):
            current = tl.min(tl.where(remaining, request, 2147483647), 0)
            members = remaining & (request == current)
            remaining = remaining & (request != current)
            if current < 2147483647:
                for start in range(0, WORDS, CHUNK):
                    w = start + tl.arange(0, CHUNK)
                    tl.store(words + w, tl.zeros((CHUNK,), tl.int32), mask=w < WORDS)
                tl.debug_barrier()
                for member in tl.static_range(TILE):
                    mine = tl.sum(tl.where(place == member, members.to(tl.int32), 0), 0) > 0
                    if mine:
                        reach_m = tl.sum(tl.where(place == member, reach, 0), 0)
                        _tile_set_bits(logical_ptr, stride_logical, words, tile * TILE + member, reach_m, BOUND,
                                       WIDTH, CHUNK)
                tl.debug_barrier()
                unique = tl.sum(tl.zeros((CHUNK,), tl.int32), 0)
                for start in range(0, WORDS, CHUNK):
                    w = start + tl.arange(0, CHUNK)
                    word = tl.load(words + w, mask=w < WORDS, other=0, cache_modifier=".cg").to(tl.uint32,
                                                                                             bitcast=True)
                    count = _popcount32(word)
                    tl.store(ranks + w, base + unique + tl.cumsum(count, 0) - count, mask=w < WORDS)
                    unique += tl.sum(count, 0)
                tl.debug_barrier()
                for member in tl.static_range(TILE):
                    mine = tl.sum(tl.where(place == member, members.to(tl.int32), 0), 0) > 0
                    if mine:
                        reach_m = tl.sum(tl.where(place == member, reach, 0), 0)
                        row = (tile * TILE + member).to(tl.int64)
                        for start in range(0, WIDTH, CHUNK):
                            columns = start + tl.arange(0, CHUNK)
                            token = tl.load(logical_ptr + row * stride_logical + columns, mask=columns < WIDTH,
                                            other=-1)
                            held = (columns < reach_m) & (token >= 0) & (token < BOUND)
                            safe = tl.where(held, token, 0)
                            at = safe >> 5
                            word = tl.load(words + at, mask=held, other=0, cache_modifier=".cg").to(tl.uint32,
                                                                                                     bitcast=True)
                            below = word & ((tl.full((CHUNK,), 1, tl.uint32) << (safe & 31).to(tl.uint32)) - 1)
                            rank = tl.load(ranks + at, mask=held, other=0, cache_modifier=".cg") + _popcount32(below)
                            tl.store(union_row + rank, current * SPAN + safe, mask=held)
                            tl.atomic_or(bits_row + rank, tl.full((CHUNK,), 1, tl.int32) << member, mask=held)
                            taken += tl.sum(held.to(tl.int32), 0)
                base += unique
                tl.debug_barrier()
    tl.store(count_ptr + tile, base)
    tl.store(picked_ptr + tile, taken)


def tile_union_kernel(logical_indices, token_to_req, num_requests, tile, block_n, bound=None):
    rows = logical_indices.shape[0]
    width = logical_indices.shape[1] - 1
    device = logical_indices.device
    bound = TILE_SPAN if bound is None else min(int(bound), TILE_SPAN)
    chunk = TILE_UNION_CHUNK
    words = -(-bound // 32)
    words = -(-words // chunk) * chunk
    tiles = -(-rows // tile)
    union = torch.empty((tiles, tile * width), dtype=torch.int32, device=device)
    bits = torch.empty_like(union)
    count = torch.empty(tiles, dtype=torch.int32, device=device)
    picked = torch.empty(tiles, dtype=torch.int32, device=device)
    held = torch.empty((tiles, words), dtype=torch.int32, device=device)
    ranks = torch.empty((tiles, words), dtype=torch.int32, device=device)
    if tiles:
        _tile_union_kernel[(tiles,)](
            logical_indices, logical_indices.stride(0), token_to_req, union, bits, count, picked, held, ranks, rows,
            num_requests, WIDTH=width, TILE=tile, BLOCK_N=block_n, SPAN=TILE_SPAN, BOUND=bound, WORDS=words,
            CHUNK=chunk, num_warps=8, PDL=pdl_on(), launch_pdl=pdl_on())
    return union, bits, count, picked


def tile_union(logical_indices, token_to_req, num_requests, tile, block_n, bound=None):
    if tile_union_fused() and logical_indices.dtype == torch.int32 and logical_indices.stride(1) == 1:
        announce("tile_union_fused", tile=tile, rows=logical_indices.shape[0], width=logical_indices.shape[1] - 1,
                 bound=bound)
        return tile_union_kernel(logical_indices, token_to_req, num_requests, tile, block_n, bound)
    rows = logical_indices.shape[0]
    width = logical_indices.shape[1] - 1
    device = logical_indices.device
    tokens = logical_indices[:, :width]
    counts = logical_indices[:, width].clamp(0, width)
    reach = (counts + block_n - 1) // block_n * block_n
    columns = torch.arange(width, device=device, dtype=torch.int32)
    requests = token_to_req[:rows].to(torch.int32)
    valid = ((tokens >= 0) & (tokens < TILE_SPAN) & (columns[None, :] < reach[:, None])
             & (requests[:, None] >= 0) & (requests[:, None] < num_requests))
    place = torch.arange(rows, device=device, dtype=torch.int32) % tile
    keys = (requests[:, None] * TILE_SPAN + tokens) * tile + place[:, None]
    keys = torch.where(valid, keys, torch.full_like(keys, TILE_EMPTY))
    tiles = -(-rows // tile)
    if tiles * tile != rows:
        keys = torch.cat([keys, keys.new_full((tiles * tile - rows, width), TILE_EMPTY)])
    ordered = keys.reshape(tiles, tile * width).sort(dim=1).values
    held = ordered != TILE_EMPTY
    held[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    entry = torch.where(held, ordered // tile, torch.full_like(ordered, TILE_EMPTY))
    fresh = held.clone()
    last = torch.cummax(torch.where(held, entry, torch.full_like(entry, -1)), dim=1).values
    fresh[:, 1:] &= entry[:, 1:] != last[:, :-1]
    slot = (fresh.cumsum(1) - 1).clamp_(min=0)
    union = torch.full_like(ordered, TILE_EMPTY).scatter_reduce_(1, slot, entry, reduce="amin")
    union = torch.where(union == TILE_EMPTY, torch.full_like(union, -1), union)
    bits = torch.zeros_like(ordered).scatter_add_(
        1, slot, torch.where(held, torch.ones_like(ordered) << (ordered % tile), torch.zeros_like(ordered)))
    return union, bits, fresh.sum(dim=1, dtype=torch.int32), held.sum(dim=1, dtype=torch.int32)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_prefill_kv4_tile_kernel(
    q_ptr,
    signs_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    union_ptr,
    bits_ptr,
    count_ptr,
    block_table_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_union,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    SPAN: tl.constexpr,
    PDL: tl.constexpr = False,
    SKIP_EMPTY: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HALF: tl.constexpr = HEAD_DIM // 2
    NUM_CHUNKS: tl.constexpr = HEAD_DIM // CHUNK
    BLOCK_M: tl.constexpr = TILE_ROWS * GROUP_BLOCK
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    if SKIP_EMPTY:
        if tl.load(count_ptr + tile) == 0:
            return

    member_offsets = tl.arange(0, BLOCK_M)
    member = member_offsets // GROUP_BLOCK
    heads = member_offsets % GROUP_BLOCK
    rows = tile * TILE_ROWS + member
    row_mask = (heads < GROUP_SIZE) & (rows < num_rows)
    head_mask = row_mask[:, None]
    half_offsets = tl.arange(0, HALF)
    chunk_offsets = tl.arange(0, CHUNK)
    column_offsets = tl.arange(0, BLOCK_N)
    even_columns = 2 * half_offsets
    first_head = kv_head * GROUP_SIZE
    query_rows = q_ptr + rows[:, None] * stride_q_row + (first_head + heads[:, None]) * stride_q_head

    rotated_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    rotated_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        signs = tl.load(signs_ptr + dims)
        part = tl.load(query_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        part = part * signs[None, :]
        rotated_even = _bf16_dot(part, _hadamard(dims[:, None], even_columns[None, :]), rotated_even)
        rotated_odd = _bf16_dot(part, _hadamard(dims[:, None], even_columns[None, :] + 1), rotated_odd)
    query_even = _bf16(rotated_even)
    query_odd = _bf16(rotated_odd)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 / HEAD_DIM

    count = tl.load(count_ptr + tile)
    for step in range(0, tl.cdiv(count, BLOCK_N)):
        columns = step * BLOCK_N + column_offsets
        inside = columns < count
        entry = tl.load(union_ptr + tile * stride_union + columns, mask=inside, other=-1)
        bits = tl.load(bits_ptr + tile * stride_union + columns, mask=inside, other=0)
        safe_entry = tl.maximum(entry, 0)
        request = safe_entry // SPAN
        token = safe_entry % SPAN
        logical_page = token // PAGE_SIZE
        page_offset = token % PAGE_SIZE
        valid = (entry >= 0) & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH)
        safe_request = tl.minimum(request, num_requests - 1)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = (key_bits & 0xF).to(tl.float32)
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = (value_bits & 0xF).to(tl.float32)
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True)
        keep = (((bits[None, :] >> member[:, None]) & 1) != 0) & valid[None, :]
        raw = _bf16_dot(query_even, key_packed & 0xF, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, (key_packed >> 4) & 0xF, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(keep, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(keep, tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = _bf16(probabilities * value_scale[None, :])
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        accumulator_even = _bf16_dot(weights, value_packed & 0xF,
                                     accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, (value_packed >> 4) & 0xF,
                                    accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    output_even = _bf16(tl.where(has_values[:, None], accumulator_even / denominator, 0.0))
    output_odd = _bf16(tl.where(has_values[:, None], accumulator_odd / denominator, 0.0))
    output_rows = output_ptr + rows[:, None] * stride_output_row + (first_head + heads[:, None]) * stride_output_head
    gate_rows = (output_gate_ptr + rows[:, None] * stride_output_gate_row
                 + (first_head + heads[:, None]) * stride_output_gate_head)
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        restored = _bf16_dot(output_even, _hadamard(even_columns[:, None], dims[None, :]),
                             tl.zeros((BLOCK_M, CHUNK), dtype=tl.float32))
        restored = _bf16_dot(output_odd, _hadamard(even_columns[:, None] + 1, dims[None, :]), restored)
        signs = tl.load(signs_ptr + dims)
        restored = _bf16(_bf16(restored).to(tl.float32) * signs[None, :] / HEAD_DIM)
        gate = tl.load(gate_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        tl.store(output_rows + dims[None, :], _bf16(restored.to(tl.float32) * tl.sigmoid(gate)),
                 mask=head_mask)


def tile_warps(tile):
    return TILE_WARPS[tile]


def prefill_int4_tile(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                      token_to_req, out, output_gate, *, tile, block_n, num_stages):
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]
    assert q.stride(2) == 1
    if out is None:
        out = torch.zeros_like(q) if tile_skip_empty() else torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 int4 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out, None
    union, bits, count, picked = tile_union(logical_indices, token_to_req, block_table.shape[0], tile, block_n,
                                            block_table.shape[1] * key_codes.shape[1])
    signs = rotation_signs(head_dim, q.device)
    group_size = q.shape[1] // key_codes.shape[2]
    _qsa_prefill_kv4_tile_kernel[(union.shape[0], key_codes.shape[2])](
        q,
        signs,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        union,
        bits,
        count,
        block_table,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        union.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        TILE_ROWS=tile,
        GROUP_BLOCK=triton.next_power_of_2(group_size),
        BLOCK_N=TILE_BLOCK_N,
        CHUNK=min(PREFILL_ROTATE_CHUNK, head_dim),
        SPAN=TILE_SPAN,
        num_warps=tile_warps(tile),
        num_stages=num_stages,
        SKIP_EMPTY=tile_skip_empty(),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out, (count, picked)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_prefill_kv8_tile_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    union_ptr,
    bits_ptr,
    count_ptr,
    block_table_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_union,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPAN: tl.constexpr,
    SKIP_EMPTY: tl.constexpr = False,
) -> None:
    BLOCK_M: tl.constexpr = TILE_ROWS * GROUP_BLOCK
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    if SKIP_EMPTY:
        if tl.load(count_ptr + tile) == 0:
            return

    member_offsets = tl.arange(0, BLOCK_M)
    member = member_offsets // GROUP_BLOCK
    heads = member_offsets % GROUP_BLOCK
    rows = tile * TILE_ROWS + member
    row_mask = (heads < GROUP_SIZE) & (rows < num_rows)
    output_mask = row_mask[:, None]
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + rows[:, None] * stride_q_row + (first_head + heads[:, None]) * stride_q_head + dim_offsets[None, :],
        mask=output_mask,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    count = tl.load(count_ptr + tile)
    for step in range(0, tl.cdiv(count, BLOCK_N)):
        columns = step * BLOCK_N + column_offsets
        inside = columns < count
        entry = tl.load(union_ptr + tile * stride_union + columns, mask=inside, other=-1)
        bits = tl.load(bits_ptr + tile * stride_union + columns, mask=inside, other=0)
        safe_entry = tl.maximum(entry, 0)
        request = safe_entry // SPAN
        token = safe_entry % SPAN
        logical_page = token // PAGE_SIZE
        page_offset = token % PAGE_SIZE
        valid = (entry >= 0) & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH)
        safe_request = tl.minimum(request, num_requests - 1)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_codes = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_codes = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0)
        value_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
        keep = (((bits[None, :] >> member[:, None]) & 1) != 0) & valid[None, :]
        scores = tl.dot(query, key_codes.to(query.dtype))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(keep, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(keep, tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = probabilities * value_scale[None, :]
        accumulator = tl.dot(
            weights.to(query.dtype),
            value_codes.to(query.dtype),
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + rows[:, None] * stride_output_gate_row
        + (first_head + heads[:, None]) * stride_output_gate_head
        + dim_offsets[None, :],
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr
        + rows[:, None] * stride_output_row
        + (first_head + heads[:, None]) * stride_output_head
        + dim_offsets[None, :],
        normalized_output,
        mask=output_mask,
    )


def check_int8(q, key_codes, value_codes, key_scales, value_scales, out, output_gate):
    if key_codes.dtype != torch.int8 or value_codes.dtype != torch.int8 or q.shape[2] != key_codes.shape[3]:
        raise ValueError("QSA_KV8 int8 tiles need full-width int8 codes")
    if key_scales.shape != key_codes.shape[:3] or value_scales.shape != key_scales.shape:
        raise ValueError("QSA_KV8 int8 scales must be [blocks, page, kv_heads]")
    if key_scales.stride() != value_scales.stride() or key_codes.stride() != value_codes.stride():
        raise ValueError("QSA_KV8 int8 keys and values must share one layout")
    assert q.dtype == torch.bfloat16 and q.stride(2) == key_codes.stride(3) == 1
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()


def prefill_int8_rows(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                      token_to_req, out, output_gate, *, block_n, num_warps, num_stages):
    num_tiles = triton.cdiv(logical_indices.shape[1] - 1, block_n)
    return sparse_attention(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                            token_to_req, True, out, output_gate=output_gate,
                            config=(block_n, num_warps, num_tiles, 1), num_stages=num_stages)


def prefill_int8_tile(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                      token_to_req, out, output_gate, *, tile, block_n, num_stages):
    if out is None:
        out = torch.zeros_like(q) if tile_skip_empty() else torch.empty_like(q)
    check_int8(q, key_codes, value_codes, key_scales, value_scales, out, output_gate)
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out, None
    union, bits, count, picked = tile_union(logical_indices, token_to_req, block_table.shape[0], tile, block_n,
                                            block_table.shape[1] * key_codes.shape[1])
    group_size = q.shape[1] // key_codes.shape[2]
    _qsa_prefill_kv8_tile_kernel[(union.shape[0], key_codes.shape[2])](
        q,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        union,
        bits,
        count,
        block_table,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        union.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        TILE_ROWS=tile,
        GROUP_BLOCK=triton.next_power_of_2(group_size),
        BLOCK_N=TILE_BLOCK_N,
        SPAN=TILE_SPAN,
        num_warps=tile_warps(tile),
        num_stages=num_stages,
        SKIP_EMPTY=tile_skip_empty(),
    )
    return out, (count, picked)


def _tile_capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _tile_share(shares):
    count, picked = shares
    union = int(count.sum())
    selected = int(picked.sum())
    return union, selected


def _tile_tally(shares, tile):
    _TILE["calls"] += 1
    now = time.monotonic()
    if now - _TILE["at"] < TILE_TALLY_S:
        return
    _TILE["at"] = now
    union, selected = _tile_share(shares)
    print(f"QSA_TILE tally calls={_TILE['calls']} tile={tile} last_union={union} last_selected={selected} "
          f"loads_saved={selected / max(union, 1):.2f}x", flush=True)


def prefill_tiled(query, keys, values, key_scales, value_scales, logical_indices, block_table, token_to_req,
                  out, output_gate, *, block_n, num_warps, num_stages, chosen="int4"):
    tile = tile_rows()
    per_row = prefill_int4 if chosen == "int4" else prefill_int8_rows
    tiled = prefill_int4_tile if chosen == "int4" else prefill_int8_tile

    def stock(into):
        return per_row(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                       token_to_req, into, output_gate, block_n=block_n, num_warps=num_warps,
                       num_stages=num_stages)

    if _TILE["aside"]:
        return stock(out)
    if block_table.shape[0] * TILE_SPAN * tile >= TILE_EMPTY:
        tile_say("keys", f"QSA_TILE keeps the per-row kernel for {block_table.shape[0]} request slots: their "
                         f"union keys would not fit 31 bits")
        return stock(out)
    ready = _TILE["checked"] >= TILE_CHECKS
    if not ready and _tile_capturing():
        return stock(out)
    if ready:
        made, shares = tiled(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                             token_to_req, out, output_gate, tile=tile, block_n=block_n, num_stages=num_stages)
        if shares is not None and not _tile_capturing():
            _tile_tally(shares, tile)
        return made
    try:
        want = stock(None)
        made, shares = tiled(query, keys, values, key_scales, value_scales, logical_indices, block_table,
                             token_to_req, out, output_gate, tile=tile, block_n=block_n, num_stages=num_stages)
    except Exception as failure:
        _TILE["aside"] = f"{type(failure).__name__}: {failure}"[:300]
        selfcheck_fail(TILE_ENV, _TILE["aside"])
        tile_say("aside", f"QSA_TILE stands aside: {_TILE['aside']}; every QSA prefill keeps the per-row kernel")
        return stock(out)
    if shares is None:
        return made
    error = float((made.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
    worst = float((made.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
    union, selected = _tile_share(shares)
    passed = error <= TILE_TOLERANCE and bool(torch.isfinite(made).all())
    print(f"QSA_TILE check rows={query.shape[0]} tile={tile} union={union} selected={selected} "
          f"loads_saved={selected / max(union, 1):.2f}x relative={error:.2e} maxrel={worst:.2e} "
          f"tolerance={TILE_TOLERANCE:g} {'pass' if passed else 'FAIL'} layout={chosen}", flush=True)
    if not passed:
        _TILE["aside"] = f"a check at {query.shape[0]} rows left its tolerance"
        selfcheck_fail(TILE_ENV, f"{_TILE['aside']} (relative {error:.2e})")
        tile_say("aside", f"QSA_TILE stands aside: {_TILE['aside']}; every QSA prefill keeps the per-row kernel")
        made.copy_(want)
        return made
    _TILE["checked"] += 1
    if _TILE["checked"] >= TILE_CHECKS:
        tile_say("live", f"QSA_TILE live: QSA prefill attention runs {tile} rows a program over the union of their "
                         f"picks, each picked K/V row loaded once for the tile, after {TILE_CHECKS} checks against "
                         f"the per-row kernel, on the {chosen} cache")
    return made


def lean_enabled():
    if _LEAN["on"] is None:
        said = str(os.environ.get(LEAN_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{LEAN_ENV}={said!r} is not '', '0' or '1'")
        _LEAN["on"] = said == "1"
    return _LEAN["on"]


def lean_maxnreg():
    if _LEAN["maxnreg"] is None:
        said = str(os.environ.get(LEAN_MAXNREG_ENV) or "").strip()
        if said and not said.isdigit():
            raise ValueError(f"{LEAN_MAXNREG_ENV}={said!r} is not a register count")
        _LEAN["maxnreg"] = int(said or 0)
    return _LEAN["maxnreg"]


def lean_cap(maxnreg):
    cap = lean_maxnreg() if maxnreg is None else maxnreg
    return {"maxnreg": cap} if cap else {}


def lean_say(key, line):
    if key in _LEAN["said"]:
        return
    _LEAN["said"].add(key)
    print(line, flush=True)


def lean_tables(head_dim, device):
    key = (head_dim, str(device))
    if key not in _LEAN["tables"]:
        index = torch.arange(head_dim)
        bits = index[:, None] & index[None, :]
        parity = torch.zeros_like(bits)
        for shift in range(head_dim.bit_length()):
            parity ^= (bits >> shift) & 1
        full = 1.0 - 2.0 * parity.to(torch.float32)
        order = torch.cat([index[0::2], index[1::2]])
        signs = rotation_signs(head_dim, "cpu")
        forward = (signs[:, None] * full[:, order]).to(torch.bfloat16)
        backward = (full[order, :] * signs[None, :] / head_dim).to(torch.bfloat16)
        _LEAN["tables"][key] = (forward.to(device).contiguous(), backward.to(device).contiguous())
    return _LEAN["tables"][key]


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_lean_slots_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    slots_ptr,
    stride_indices_row,
    stride_slots_row,
    stride_table_req,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    work_ptr=None,
    tally_ptr=None,
    GATED: tl.constexpr = False,
    TALLY: tl.constexpr = False,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    inside = columns < TOPK
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=inside, other=-1)
    safe_token = tl.maximum(token, 0)
    logical_page = safe_token // PAGE_SIZE
    valid = (inside & (request >= 0) & (request < num_requests) & (token >= 0)
             & (logical_page < PAGE_TABLE_WIDTH))
    physical_page = tl.load(
        block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    slot = tl.where(valid, physical_page * PAGE_SIZE + safe_token % PAGE_SIZE, -1)
    tl.store(slots_ptr + row * stride_slots_row + columns, slot, mask=inside)
    count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    count = tl.where((request >= 0) & (request < num_requests), count, 0)
    tl.store(slots_ptr + row * stride_slots_row + TOPK, count)
    if TALLY:
        tl.atomic_add(tally_ptr + 1, (count > 0).to(tl.int64))
        if row == 0:
            tl.atomic_add(tally_ptr, 1)


@triton.jit
def _lean_nibbles(packed):
    wide = packed.to(tl.int16)
    low = ((wide << 3) & 0x78) | 0x3F80
    high = ((wide >> 1) & 0x78) | 0x3F80
    return low.to(tl.bfloat16, bitcast=True), high.to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["num_pairs"])
def _qsa_lean_rotate_kernel(
    q_ptr,
    table_ptr,
    rotated_ptr,
    stride_q_row,
    stride_q_head,
    num_pairs,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    pairs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    inside = pairs < num_pairs
    rows = (pairs // NUM_HEADS).to(tl.int64)
    heads = pairs % NUM_HEADS
    dims = tl.arange(0, HEAD_DIM)
    columns = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    query = tl.load(q_ptr + rows[:, None] * stride_q_row + heads[:, None] * stride_q_head + dims[None, :],
                    mask=inside[:, None], other=0.0)
    table = tl.load(table_ptr + dims[:, None] * HEAD_DIM + columns[None, :])
    rotated = _bf16_dot(query, table, tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32))
    tl.store(rotated_ptr + pairs[:, None].to(tl.int64) * HEAD_DIM + columns[None, :], _bf16(rotated),
             mask=inside[:, None])


@triton.jit(do_not_specialize=["num_pairs"])
def _qsa_lean_restore_kernel(
    attended_ptr,
    table_ptr,
    output_gate_ptr,
    output_ptr,
    stride_output_gate_row,
    stride_output_gate_head,
    stride_output_row,
    stride_output_head,
    num_pairs,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    pairs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    inside = pairs < num_pairs
    rows = (pairs // NUM_HEADS).to(tl.int64)
    heads = pairs % NUM_HEADS
    dims = tl.arange(0, HEAD_DIM)
    columns = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    attended = tl.load(attended_ptr + pairs[:, None].to(tl.int64) * HEAD_DIM + dims[None, :],
                       mask=inside[:, None], other=0.0)
    table = tl.load(table_ptr + dims[:, None] * HEAD_DIM + columns[None, :])
    restored = _bf16(_bf16_dot(attended, table, tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32))).to(tl.float32)
    gate = tl.load(output_gate_ptr + rows[:, None] * stride_output_gate_row + heads[:, None] * stride_output_gate_head
                   + columns[None, :], mask=inside[:, None], other=0.0).to(tl.float32)
    tl.store(output_ptr + rows[:, None] * stride_output_row + heads[:, None] * stride_output_head + columns[None, :],
             _bf16(restored * tl.sigmoid(gate)), mask=inside[:, None])


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_lean_kv4_kernel(
    rotated_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slots_ptr,
    attended_ptr,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_slots_row,
    num_rows,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    HALF: tl.constexpr = HEAD_DIM // 2
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    valid_count = tl.load(slots_ptr + row * stride_slots_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    head_mask = head_offsets[:, None] < GROUP_SIZE
    pairs = (row * NUM_HEADS + kv_head * GROUP_SIZE + head_offsets[:, None]).to(tl.int64) * HEAD_DIM
    query_even = tl.load(rotated_ptr + pairs + half_offsets[None, :], mask=head_mask, other=0.0)
    query_odd = tl.load(rotated_ptr + pairs + HALF + half_offsets[None, :], mask=head_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 * 16.0 / HEAD_DIM

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
    for tile in range(0, tile_end):
        columns = tile * BLOCK_N + column_offsets
        slot = tl.load(slots_ptr + row * stride_slots_row + columns, mask=columns < TOPK, other=-1)
        valid = slot >= 0
        safe_slot = tl.maximum(slot, 0)
        safe_page = (safe_slot // PAGE_SIZE).to(tl.int64)
        page_offset = safe_slot % PAGE_SIZE
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = 1.0 + (key_bits & 0xF).to(tl.float32) * 0.0625
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = 1.0 + (value_bits & 0xF).to(tl.float32) * 0.0625
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True) * 16.0
        key_low, key_high = _lean_nibbles(key_packed)
        value_low, value_high = _lean_nibbles(value_packed)
        raw = _bf16_dot(query_even, key_low, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, key_high, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = _bf16(probabilities * value_scale[None, :])
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        accumulator_even = _bf16_dot(weights, value_low, accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, value_high, accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    tl.store(attended_ptr + pairs + half_offsets[None, :],
             _bf16(tl.where(has_values[:, None], accumulator_even / denominator, 0.0)), mask=head_mask)
    tl.store(attended_ptr + pairs + HALF + half_offsets[None, :],
             _bf16(tl.where(has_values[:, None], accumulator_odd / denominator, 0.0)), mask=head_mask)


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_lean_kv8_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slots_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_slots_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    valid_count = tl.load(slots_ptr + row * stride_slots_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    output_mask = head_offsets[:, None] < GROUP_SIZE
    query = tl.load(
        q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head + dim_offsets[None, :],
        mask=output_mask,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
    for tile in range(0, tile_end):
        columns = tile * BLOCK_N + column_offsets
        slot = tl.load(slots_ptr + row * stride_slots_row + columns, mask=columns < TOPK, other=-1)
        valid = slot >= 0
        safe_slot = tl.maximum(slot, 0)
        safe_page = (safe_slot // PAGE_SIZE).to(tl.int64)
        page_offset = safe_slot % PAGE_SIZE
        key_codes = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_codes = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0)
        value_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
        scores = tl.dot(query, key_codes.to(query.dtype))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = probabilities * value_scale[None, :]
        accumulator = tl.dot(
            weights.to(query.dtype),
            value_codes.to(query.dtype),
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + (first_head + head_offsets[:, None]) * stride_output_gate_head
        + dim_offsets[None, :],
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :],
        normalized_output,
        mask=output_mask,
    )


def lean_slots(logical_indices, block_table, token_to_req, num_cache_blocks, page_size, tally=None, work=None,
               pdl=False):
    rows, columns = logical_indices.shape
    slots = torch.empty((rows, columns), dtype=torch.int32, device=logical_indices.device)
    extra = {}
    if tally is not None:
        extra.update(tally_ptr=tally, TALLY=True)
    if work is not None:
        extra.update(work_ptr=work, GATED=True)
    if pdl:
        extra.update(PDL=True, launch_pdl=True)
    _qsa_lean_slots_kernel[(rows,)](
        logical_indices,
        block_table,
        token_to_req,
        slots,
        logical_indices.stride(0),
        slots.stride(0),
        block_table.stride(0),
        num_cache_blocks,
        block_table.shape[0],
        TOPK=columns - 1,
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        BLOCK=triton.next_power_of_2(columns - 1),
        num_warps=LEAN_WARPS,
        num_stages=1,
        **extra,
    )
    return slots


def tally_part(device, first):
    held = fast_tally(device)
    return None if held is None else held[first:]


def lean_columns(head_dim):
    return min(LEAN_COLUMNS, head_dim)


def prefill_lean_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                      token_to_req, out, output_gate, *, block_n, num_warps, num_stages, maxnreg=None):
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]
    assert q.stride(2) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 int4 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out
    forward, backward = lean_tables(head_dim, q.device)
    rows, heads = q.shape[0], q.shape[1]
    pairs = rows * heads
    columns = lean_columns(head_dim)
    grid = (triton.cdiv(pairs, LEAN_PAIRS), head_dim // columns)
    slots = lean_slots(logical_indices, block_table, token_to_req, key_codes.shape[0], key_codes.shape[1],
                       tally=tally_part(q.device, 2))
    rotated = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=q.device)
    _qsa_lean_rotate_kernel[grid](q, forward, rotated, q.stride(0), q.stride(1), pairs, NUM_HEADS=heads,
                                  HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns, num_warps=LEAN_WARPS,
                                  num_stages=1)
    attended = torch.empty_like(rotated)
    group_size = heads // key_codes.shape[2]
    selection_width = logical_indices.shape[1] - 1
    _qsa_lean_kv4_kernel[(rows, key_codes.shape[2])](
        rotated,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        slots,
        attended,
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        slots.stride(0),
        rows,
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_HEADS=heads,
        NUM_TILES=triton.cdiv(selection_width, block_n),
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        **lean_cap(maxnreg),
    )
    _qsa_lean_restore_kernel[grid](attended, backward, output_gate_view, out, output_gate_view.stride(0),
                                   output_gate_view.stride(1), out.stride(0), out.stride(1), pairs,
                                   NUM_HEADS=heads, HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns,
                                   num_warps=LEAN_WARPS, num_stages=1)
    return out


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
def _qsa_lean_tile_kv4_kernel(
    rotated_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    union_ptr,
    bits_ptr,
    count_ptr,
    block_table_ptr,
    attended_ptr,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_union,
    stride_table_req,
    num_rows,
    num_requests,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPAN: tl.constexpr,
) -> None:
    HALF: tl.constexpr = HEAD_DIM // 2
    BLOCK_M: tl.constexpr = TILE_ROWS * GROUP_BLOCK
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    count = tl.load(count_ptr + tile)

    member_offsets = tl.arange(0, BLOCK_M)
    member = member_offsets // GROUP_BLOCK
    heads = member_offsets % GROUP_BLOCK
    rows = tile * TILE_ROWS + member
    row_mask = ((heads < GROUP_SIZE) & (rows < num_rows))[:, None]
    half_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    pairs = (rows[:, None] * NUM_HEADS + kv_head * GROUP_SIZE + heads[:, None]).to(tl.int64) * HEAD_DIM
    query_even = tl.load(rotated_ptr + pairs + half_offsets[None, :], mask=row_mask, other=0.0)
    query_odd = tl.load(rotated_ptr + pairs + HALF + half_offsets[None, :], mask=row_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 * 16.0 / HEAD_DIM

    for step in range(0, tl.cdiv(count, BLOCK_N)):
        columns = step * BLOCK_N + column_offsets
        inside = columns < count
        entry = tl.load(union_ptr + tile.to(tl.int64) * stride_union + columns, mask=inside, other=-1)
        bits = tl.load(bits_ptr + tile.to(tl.int64) * stride_union + columns, mask=inside, other=0)
        safe_entry = tl.maximum(entry, 0)
        request = safe_entry // SPAN
        token = safe_entry % SPAN
        logical_page = token // PAGE_SIZE
        page_offset = token % PAGE_SIZE
        valid = (entry >= 0) & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + tl.minimum(request, num_requests - 1).to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = 1.0 + (key_bits & 0xF).to(tl.float32) * 0.0625
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = 1.0 + (value_bits & 0xF).to(tl.float32) * 0.0625
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True) * 16.0
        key_low, key_high = _lean_nibbles(key_packed)
        value_low, value_high = _lean_nibbles(value_packed)
        keep = (((bits[None, :] >> member[:, None]) & 1) != 0) & valid[None, :]
        raw = _bf16_dot(query_even, key_low, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, key_high, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(keep, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(keep, tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = _bf16(probabilities * value_scale[None, :])
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        accumulator_even = _bf16_dot(weights, value_low, accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, value_high, accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    tl.store(attended_ptr + pairs + half_offsets[None, :],
             _bf16(tl.where(has_values[:, None], accumulator_even / denominator, 0.0)), mask=row_mask)
    tl.store(attended_ptr + pairs + HALF + half_offsets[None, :],
             _bf16(tl.where(has_values[:, None], accumulator_odd / denominator, 0.0)), mask=row_mask)


def prefill_lean_tile_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                           token_to_req, out, output_gate, *, tile, block_n, num_warps, num_stages, maxnreg=None):
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]
    assert q.stride(2) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 int4 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out
    forward, backward = lean_tables(head_dim, q.device)
    rows, heads = q.shape[0], q.shape[1]
    pairs = rows * heads
    columns = lean_columns(head_dim)
    grid = (triton.cdiv(pairs, LEAN_PAIRS), head_dim // columns)
    num_requests = block_table.shape[0]
    union, bits, count, _ = tile_union(logical_indices, token_to_req, num_requests, tile, block_n)
    rotated = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=q.device)
    _qsa_lean_rotate_kernel[grid](q, forward, rotated, q.stride(0), q.stride(1), pairs, NUM_HEADS=heads,
                                  HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns, num_warps=LEAN_WARPS,
                                  num_stages=1)
    attended = torch.empty_like(rotated)
    group_size = heads // key_codes.shape[2]
    announce(f"prefill_lean_tile{tile}", rows=rows, block_n=block_n, num_warps=num_warps, num_stages=num_stages)
    _qsa_lean_tile_kv4_kernel[(union.shape[0], key_codes.shape[2])](
        rotated,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        union,
        bits,
        count,
        block_table,
        attended,
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        union.stride(0),
        block_table.stride(0),
        rows,
        num_requests,
        key_codes.shape[0],
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_HEADS=heads,
        TILE_ROWS=tile,
        GROUP_BLOCK=triton.next_power_of_2(group_size),
        BLOCK_N=block_n,
        SPAN=TILE_SPAN,
        num_warps=num_warps,
        num_stages=num_stages,
        **lean_cap(maxnreg),
    )
    _qsa_lean_restore_kernel[grid](attended, backward, output_gate_view, out, output_gate_view.stride(0),
                                   output_gate_view.stride(1), out.stride(0), out.stride(1), pairs,
                                   NUM_HEADS=heads, HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns,
                                   num_warps=LEAN_WARPS, num_stages=1)
    return out


@triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
def _qsa_lean_tile_split_kv4_kernel(
    rotated_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    union_ptr,
    bits_ptr,
    count_ptr,
    block_table_ptr,
    partial_ptr,
    lse_ptr,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_union,
    stride_table_req,
    num_rows,
    num_requests,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPAN: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
) -> None:
    HALF: tl.constexpr = HEAD_DIM // 2
    BLOCK_M: tl.constexpr = TILE_ROWS * GROUP_BLOCK
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    count = tl.load(count_ptr + tile)
    steps = tl.cdiv(count, BLOCK_N)
    per_split = tl.cdiv(steps, NUM_SPLITS)
    first = split * per_split
    last = tl.minimum(first + per_split, steps)

    member_offsets = tl.arange(0, BLOCK_M)
    member = member_offsets // GROUP_BLOCK
    heads = member_offsets % GROUP_BLOCK
    rows = tile * TILE_ROWS + member
    row_valid = (heads < GROUP_SIZE) & (rows < num_rows)
    row_mask = row_valid[:, None]
    half_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    pair_index = rows * NUM_HEADS + kv_head * GROUP_SIZE + heads
    pairs = pair_index[:, None].to(tl.int64) * HEAD_DIM
    query_even = tl.load(rotated_ptr + pairs + half_offsets[None, :], mask=row_mask, other=0.0)
    query_odd = tl.load(rotated_ptr + pairs + HALF + half_offsets[None, :], mask=row_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 * 16.0 / HEAD_DIM

    for step in range(first, last):
        columns = step * BLOCK_N + column_offsets
        inside = columns < count
        entry = tl.load(union_ptr + tile.to(tl.int64) * stride_union + columns, mask=inside, other=-1)
        bits = tl.load(bits_ptr + tile.to(tl.int64) * stride_union + columns, mask=inside, other=0)
        safe_entry = tl.maximum(entry, 0)
        request = safe_entry // SPAN
        token = safe_entry % SPAN
        logical_page = token // PAGE_SIZE
        page_offset = token % PAGE_SIZE
        valid = (entry >= 0) & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + tl.minimum(request, num_requests - 1).to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = 1.0 + (key_bits & 0xF).to(tl.float32) * 0.0625
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = 1.0 + (value_bits & 0xF).to(tl.float32) * 0.0625
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True) * 16.0
        key_low, key_high = _lean_nibbles(key_packed)
        value_low, value_high = _lean_nibbles(value_packed)
        keep = (((bits[None, :] >> member[:, None]) & 1) != 0) & valid[None, :]
        raw = _bf16_dot(query_even, key_low, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, key_high, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(keep, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(keep, tl.math.exp2(scores - next_max[:, None]), 0.0)
        weights = _bf16(probabilities * value_scale[None, :])
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        accumulator_even = _bf16_dot(weights, value_low, accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, value_high, accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    slot = (split * num_rows * NUM_HEADS + pair_index).to(tl.int64)
    tl.store(partial_ptr + slot[:, None] * HEAD_DIM + half_offsets[None, :],
             tl.where(has_values[:, None], accumulator_even / denominator, 0.0), mask=row_mask)
    tl.store(partial_ptr + slot[:, None] * HEAD_DIM + HALF + half_offsets[None, :],
             tl.where(has_values[:, None], accumulator_odd / denominator, 0.0), mask=row_mask)
    tl.store(lse_ptr + slot, tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -1.0e30),
             mask=row_valid)


@triton.jit(do_not_specialize=["num_pairs"])
def _qsa_lean_tile_merge_kernel(partial_ptr, lse_ptr, attended_ptr, num_pairs, NUM_SPLITS: tl.constexpr,
                                SPLITS_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr, PAIRS: tl.constexpr):
    pair = tl.program_id(0) * PAIRS + tl.arange(0, PAIRS)
    live = pair < num_pairs
    splits = tl.arange(0, SPLITS_BLOCK)
    used = splits < NUM_SPLITS
    slot = splits[None, :].to(tl.int64) * num_pairs + pair[:, None]
    lse = tl.load(lse_ptr + slot, mask=live[:, None] & used[None, :], other=-1.0e30)
    top = tl.max(lse, axis=1)
    weight = tl.where(lse > -1.0e29, tl.math.exp2(lse - top[:, None]), 0.0)
    total = tl.sum(weight, axis=1)
    dims = tl.arange(0, HEAD_DIM)
    out = tl.zeros((PAIRS, HEAD_DIM), dtype=tl.float32)
    for index in tl.static_range(NUM_SPLITS):
        share = tl.sum(tl.where(splits[None, :] == index, weight, 0.0), axis=1)
        part = tl.load(partial_ptr + (index * num_pairs + pair[:, None]).to(tl.int64) * HEAD_DIM + dims[None, :],
                       mask=live[:, None], other=0.0)
        out += share[:, None] * part
    out = tl.where(total[:, None] > 0, out / tl.maximum(total, 1.0e-30)[:, None], 0.0)
    tl.store(attended_ptr + pair[:, None].to(tl.int64) * HEAD_DIM + dims[None, :], _bf16(out), mask=live[:, None])


def prefill_lean_tile_split_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                                 token_to_req, out, output_gate, *, tile, block_n, num_warps, num_stages, num_splits,
                                 maxnreg=None):
    check_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
               token_to_req)
    head_dim = q.shape[2]
    if out is None:
        out = torch.empty_like(q)
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out
    forward, backward = lean_tables(head_dim, q.device)
    rows, heads = q.shape[0], q.shape[1]
    pairs = rows * heads
    columns = lean_columns(head_dim)
    grid = (triton.cdiv(pairs, LEAN_PAIRS), head_dim // columns)
    num_requests = block_table.shape[0]
    union, bits, count, _ = tile_union(logical_indices, token_to_req, num_requests, tile, block_n)
    rotated = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=q.device)
    _qsa_lean_rotate_kernel[grid](q, forward, rotated, q.stride(0), q.stride(1), pairs, NUM_HEADS=heads,
                                  HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns, num_warps=LEAN_WARPS,
                                  num_stages=1)
    partial = torch.empty((num_splits, rows, heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((num_splits, rows, heads), dtype=torch.float32, device=q.device)
    group_size = heads // key_codes.shape[2]
    announce(f"prefill_lean_tile{tile}_split{num_splits}", rows=rows, block_n=block_n, num_warps=num_warps)
    _qsa_lean_tile_split_kv4_kernel[(union.shape[0], key_codes.shape[2], num_splits)](
        rotated, key_codes, value_codes, key_scales, value_scales, union, bits, count, block_table, partial, lse,
        key_codes.stride(0), key_codes.stride(1), key_codes.stride(2),
        value_codes.stride(0), value_codes.stride(1), value_codes.stride(2),
        key_scales.stride(0), key_scales.stride(1), key_scales.stride(2),
        union.stride(0), block_table.stride(0), rows, num_requests, key_codes.shape[0],
        PAGE_SIZE=key_codes.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
        HEAD_DIM=head_dim, NUM_HEADS=heads, TILE_ROWS=tile, GROUP_BLOCK=triton.next_power_of_2(group_size),
        BLOCK_N=block_n, SPAN=TILE_SPAN, NUM_SPLITS=num_splits, num_warps=num_warps, num_stages=num_stages,
        **lean_cap(maxnreg),
    )
    attended = torch.empty_like(rotated)
    merge_pairs = 16
    _qsa_lean_tile_merge_kernel[(triton.cdiv(pairs, merge_pairs),)](
        partial, lse, attended, pairs, NUM_SPLITS=num_splits, SPLITS_BLOCK=triton.next_power_of_2(num_splits),
        HEAD_DIM=head_dim, PAIRS=merge_pairs, num_warps=4)
    _qsa_lean_restore_kernel[grid](attended, backward, output_gate_view, out, output_gate_view.stride(0),
                                   output_gate_view.stride(1), out.stride(0), out.stride(1), pairs,
                                   NUM_HEADS=heads, HEAD_DIM=head_dim, BLOCK_M=LEAN_PAIRS, BLOCK_C=columns,
                                   num_warps=LEAN_WARPS, num_stages=1)
    return out


def prefill_lean_int8(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                      token_to_req, out, output_gate, *, block_n, num_warps, num_stages, maxnreg=None):
    if out is None:
        out = torch.empty_like(q)
    check_int8(q, key_codes, value_codes, key_scales, value_scales, out, output_gate)
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out
    slots = lean_slots(logical_indices, block_table, token_to_req, key_codes.shape[0], key_codes.shape[1],
                       tally=tally_part(q.device, 2))
    group_size = q.shape[1] // key_codes.shape[2]
    selection_width = logical_indices.shape[1] - 1
    _qsa_lean_kv8_kernel[(q.shape[0], key_codes.shape[2])](
        q,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        slots,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        slots.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_TILES=triton.cdiv(selection_width, block_n),
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        **lean_cap(maxnreg),
    )
    return out


def lean_ready():
    return _LEAN["checked"] >= LEAN_CHECKS or os.environ.get(LEAN_CHECKED_ENV) == "1"


def lean_announce(path, query, chosen, knobs):
    announce(f"prefill_lean_{path}", knob=LEAN_ENV, path=path, rows=query.shape[0], layout=chosen,
             block_n=knobs["block_n"], num_warps=knobs["num_warps"], num_stages=knobs["num_stages"],
             maxnreg=lean_maxnreg() or "free")


def prefill_leaned(query, keys, values, key_scales, value_scales, logical_indices, block_table, token_to_req,
                   out, output_gate, *, block_n, num_warps, num_stages, chosen="int4"):
    per_row = prefill_int4 if chosen == "int4" else prefill_int8_rows
    lean = prefill_lean_int4 if chosen == "int4" else prefill_lean_int8
    args = (query, keys, values, key_scales, value_scales, logical_indices, block_table, token_to_req)
    knobs = {"block_n": block_n, "num_warps": num_warps, "num_stages": num_stages}

    if _LEAN["aside"]:
        return per_row(*args, out, output_gate, **knobs)
    if keys.shape[0] * keys.shape[1] >= LEAN_SLOT_LIMIT:
        lean_say("slots", f"QSA_PREFILL_LEAN keeps the per-row kernel: {keys.shape[0]} cache blocks of "
                          f"{keys.shape[1]} tokens would not fit a 31-bit slot")
        return per_row(*args, out, output_gate, **knobs)
    if _tile_capturing():
        tables = chosen != "int4" or (query.shape[2], str(query.device)) in _LEAN["tables"]
        if lean_ready() and tables:
            lean_announce("captured", query, chosen, knobs)
            return lean(*args, out, output_gate, **knobs)
        lean_say("capture", "QSA_PREFILL_LEAN: a CUDA graph captured before the lean kernel was checked keeps "
                            "the per-row kernel inside that graph")
        return per_row(*args, out, output_gate, **knobs)
    if _LEAN["checked"] >= LEAN_CHECKS:
        lean_announce("eager", query, chosen, knobs)
        return lean(*args, out, output_gate, **knobs)
    lean_announce("check", query, chosen, knobs)
    try:
        want = per_row(*args, None, output_gate, **knobs)
        made = lean(*args, out, output_gate, **knobs)
    except Exception as failure:
        _LEAN["aside"] = f"{type(failure).__name__}: {failure}"[:300]
        selfcheck_fail(LEAN_ENV, _LEAN["aside"])
        lean_say("aside", f"QSA_PREFILL_LEAN stands aside: {_LEAN['aside']}; every QSA prefill keeps the per-row "
                          f"kernel")
        return per_row(*args, out, output_gate, **knobs)
    error = float((made.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
    worst = float((made.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
    passed = error <= LEAN_TOLERANCE and bool(torch.isfinite(made).all())
    print(f"QSA_PREFILL_LEAN check rows={query.shape[0]} relative={error:.2e} maxrel={worst:.2e} "
          f"tolerance={LEAN_TOLERANCE:g} {'pass' if passed else 'FAIL'} layout={chosen}", flush=True)
    if not passed:
        _LEAN["aside"] = f"a check at {query.shape[0]} rows left its tolerance"
        selfcheck_fail(LEAN_ENV, f"{_LEAN['aside']} (relative {error:.2e})")
        lean_say("aside", f"QSA_PREFILL_LEAN stands aside: {_LEAN['aside']}; every QSA prefill keeps the per-row "
                          f"kernel")
        made.copy_(want)
        return made
    _LEAN["checked"] += 1
    if _LEAN["checked"] >= LEAN_CHECKS:
        dense = ", the query rotation and its inverse with the gate as two dense passes" if chosen == "int4" else ""
        lean_say("live", f"QSA_PREFILL_LEAN live: QSA prefill attention reads each row's picks through one slot "
                         f"pass{dense}, at maxnreg {lean_maxnreg() or 'free'}, after {LEAN_CHECKS} checks against "
                         f"the per-row kernel, on the {chosen} cache")
    return made


def _lean_time(run, repeats=LEAN_TIMED):
    run()
    torch.cuda.synchronize()
    spent = []
    for _ in range(repeats):
        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        run()
        ended.record()
        torch.cuda.synchronize()
        spent.append(began.elapsed_time(ended))
    return sorted(spent)[len(spent) // 2]


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_rotate_query_kernel(
    q_ptr,
    signs_ptr,
    rotated_ptr,
    stride_q_row,
    stride_q_head,
    stride_rotated_row,
    stride_rotated_head,
    num_rows,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,
    work_ptr=None,
    GATED: tl.constexpr = False,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    NUM_CHUNKS: tl.constexpr = HEAD_DIM // CHUNK
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_offsets = tl.arange(0, BLOCK_M)
    chunk_offsets = tl.arange(0, CHUNK)
    dim_offsets = tl.arange(0, HEAD_DIM)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    query_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    rotated = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        signs = tl.load(signs_ptr + dims)
        part = tl.load(query_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        part = part * signs[None, :]
        rotated = _bf16_dot(part, _hadamard(dims[:, None], dim_offsets[None, :]), rotated)
    tl.store(rotated_ptr + row * stride_rotated_row + (first_head + head_offsets[:, None]) * stride_rotated_head
             + dim_offsets[None, :], _bf16(rotated), mask=head_mask)


@triton.jit
def _kv4_merge_restore(
    partial_output_ptr,
    partial_lse_ptr,
    signs_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    row,
    kv_head,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,
    CG: tl.constexpr,
):
    NUM_CHUNKS: tl.constexpr = HEAD_DIM // CHUNK
    head_offsets = tl.arange(0, BLOCK_M)
    chunk_offsets = tl.arange(0, CHUNK)
    dim_offsets = tl.arange(0, HEAD_DIM)
    first_head = kv_head * GROUP_SIZE
    heads = first_head + head_offsets
    head_valid = head_offsets < GROUP_SIZE
    head_mask = head_valid[:, None]
    if NUM_SPLITS == 1:
        if CG:
            merged = tl.load(partial_output_ptr + (row * NUM_QUERY_HEADS + heads[:, None]) * HEAD_DIM
                             + dim_offsets[None, :], mask=head_mask, other=0.0, cache_modifier=".cg")
        else:
            merged = tl.load(partial_output_ptr + (row * NUM_QUERY_HEADS + heads[:, None]) * HEAD_DIM
                             + dim_offsets[None, :], mask=head_mask, other=0.0)
    else:
        lse_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        for split in range(NUM_SPLITS):
            if CG:
                lse = tl.load(partial_lse_ptr + (split * num_rows + row) * NUM_QUERY_HEADS + heads,
                              mask=head_valid, other=-float("inf"), cache_modifier=".cg")
            else:
                lse = tl.load(partial_lse_ptr + (split * num_rows + row) * NUM_QUERY_HEADS + heads,
                              mask=head_valid, other=-float("inf"))
            lse_max = tl.maximum(lse_max, lse)
        has_values = lse_max > -float("inf")
        denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
        merged = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        for split in range(NUM_SPLITS):
            if CG:
                lse = tl.load(partial_lse_ptr + (split * num_rows + row) * NUM_QUERY_HEADS + heads,
                              mask=head_valid, other=-float("inf"), cache_modifier=".cg")
            else:
                lse = tl.load(partial_lse_ptr + (split * num_rows + row) * NUM_QUERY_HEADS + heads,
                              mask=head_valid, other=-float("inf"))
            weight = tl.math.exp2(tl.where(has_values, tl.where(lse > -float("inf"), lse - lse_max,
                                                                -float("inf")), -float("inf")))
            split_row = (split * num_rows + row).to(tl.int64)
            if CG:
                partial = tl.load(partial_output_ptr + (split_row * NUM_QUERY_HEADS + heads[:, None]) * HEAD_DIM
                                  + dim_offsets[None, :], mask=head_mask, other=0.0, cache_modifier=".cg")
            else:
                partial = tl.load(partial_output_ptr + (split_row * NUM_QUERY_HEADS + heads[:, None]) * HEAD_DIM
                                  + dim_offsets[None, :], mask=head_mask, other=0.0)
            merged += partial * weight[:, None]
            denominator += weight
        merged = tl.where(denominator[:, None] > 0, merged / tl.maximum(denominator[:, None], 1.0e-30), 0.0)
    rotated = _bf16(merged)
    output_rows = output_ptr + row * stride_output_row + heads[:, None] * stride_output_head
    gate_rows = output_gate_ptr + row * stride_output_gate_row + heads[:, None] * stride_output_gate_head
    for chunk in tl.static_range(NUM_CHUNKS):
        dims = chunk * CHUNK + chunk_offsets
        restored = _bf16_dot(rotated, _hadamard(dim_offsets[:, None], dims[None, :]),
                             tl.zeros((BLOCK_M, CHUNK), dtype=tl.float32))
        signs = tl.load(signs_ptr + dims)
        restored = _bf16(_bf16(restored).to(tl.float32) * signs[None, :] / HEAD_DIM)
        gate = tl.load(gate_rows + dims[None, :], mask=head_mask, other=0.0).to(tl.float32)
        tl.store(output_rows + dims[None, :], _bf16(restored.to(tl.float32) * tl.sigmoid(gate)),
                 mask=head_mask)


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_kv4_merge_restore_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    signs_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,
    work_ptr=None,
    GATED: tl.constexpr = False,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if GATED:
        if tl.load(work_ptr) == 0:
            return
    _kv4_merge_restore(
        partial_output_ptr, partial_lse_ptr, signs_ptr, output_ptr, output_gate_ptr, stride_output_row,
        stride_output_head, stride_output_gate_row, stride_output_gate_head, num_rows, tl.program_id(0),
        tl.program_id(1), GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, BLOCK_M, CHUNK, False,
    )


def decode_lean_route(key_codes, work):
    if not decode_lean_enabled() or _DLEAN["aside"]:
        return "stock"
    if key_codes.shape[0] * key_codes.shape[1] >= LEAN_SLOT_LIMIT:
        decode_lean_say("slots", f"QSA_DECODE_LEAN keeps the stock decode kernel: {key_codes.shape[0]} cache blocks of "
                                 f"{key_codes.shape[1]} tokens would not fit a 31-bit slot")
        return "stock"
    if _tile_capturing():
        if not decode_lean_ready():
            decode_lean_say("capture", "QSA_DECODE_LEAN: a CUDA graph captured before the lean decode was checked "
                                       "keeps the stock decode kernel inside that graph")
            return "stock"
        if fast_tally(key_codes.device) is None:
            decode_lean_say("capture_tally", "QSA_DECODE_LEAN: a CUDA graph reached the lean decode before its row "
                                             "tally was allocated, and the slot pass compiles against that tally, so "
                                             "this graph keeps the stock decode kernel")
            return "stock"
        return "lean"
    if _DLEAN["checked"] >= DECODE_LEAN_CHECKS or work is not None:
        return "lean" if decode_lean_ready() else "stock"
    return "check"


def decode_lean_aside(reason):
    _DLEAN["aside"] = reason[:300]
    selfcheck_fail(DECODE_LEAN_ENV, _DLEAN["aside"])
    decode_lean_say("aside", f"{DECODE_LEAN_ENV} stood aside: {_DLEAN['aside']}; every QSA decode keeps the stock "
                             f"kernel")


def decode_int4(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                token_to_req, out, output_gate_view, config, num_stages=2, work=None):
    args = (q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table, token_to_req)
    route = decode_lean_route(key_codes, work)
    if route != "check":
        return decode_int4_run(*args, out, output_gate_view, config, num_stages, work=work, lean=route == "lean")
    try:
        want = decode_int4_run(*args, torch.empty_like(out), output_gate_view, config, num_stages)
        made = decode_int4_run(*args, out, output_gate_view, config, num_stages, lean=True)
        error = float((made.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
        worst = float((made.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
        finite = bool(torch.isfinite(made).all())
    except Exception as failure:
        decode_lean_aside(f"{type(failure).__name__}: {failure}")
        return decode_int4_run(*args, out, output_gate_view, config, num_stages)
    passed = error <= DECODE_LEAN_TOLERANCE and finite
    print(f"QSA_DECODE_LEAN check rows={q.shape[0]} relative={error:.2e} maxrel={worst:.2e} "
          f"tolerance={DECODE_LEAN_TOLERANCE:g} {'pass' if passed else 'FAIL'}", flush=True)
    if not passed:
        decode_lean_aside(f"a check at {q.shape[0]} rows read relative {error:.2e} against the stock kernel")
        made.copy_(want)
        return made
    _DLEAN["checked"] += 1
    if _DLEAN["checked"] >= DECODE_LEAN_CHECKS:
        decode_lean_say("live", f"QSA_DECODE_LEAN live: the fused int4 decode resolves each row's picks to cache slots "
                                f"in one pass and its split kernel turns nibbles into bf16 by bit pattern, after "
                                f"{DECODE_LEAN_CHECKS} checks against the stock kernel")
    return made


def decode_int4_passes(q, key_codes, value_codes, key_scales, value_scales, passes, block_table, token_to_req, out,
                       output_gate_view, config, between=None, num_stages=2, lean=False):
    block_n, partial_warps, num_tiles, _ = config
    head_dim = q.shape[2]
    rows, num_query_heads = q.shape[0], q.shape[1]
    num_kv_heads = key_codes.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)
    chunk = min(PREFILL_ROTATE_CHUNK, head_dim)
    signs = rotation_signs(head_dim, q.device)
    rotated = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _qsa_rotate_query_kernel[(rows, num_kv_heads)](
        q, signs, rotated, q.stride(0), q.stride(1), rotated.stride(0), rotated.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, BLOCK_M=block_m, CHUNK=chunk,
        num_warps=DECODE_FUSED_WARPS, num_stages=1,
    )
    count = len(passes)
    partial_output = torch.empty((count, *q.shape), dtype=torch.float32, device=q.device)
    partial_lse = torch.empty((count, rows, num_query_heads), dtype=torch.float32, device=q.device)
    for slot, picks in enumerate(passes):
        if slot and between is not None:
            between()
        if lean:
            picks = lean_slots(picks, block_table, token_to_req, key_codes.shape[0], key_codes.shape[1])
        selection_width = picks.shape[1] - 1
        tiles = min(num_tiles, triton.cdiv(selection_width, block_n))
        _qsa_sparse_kv4_splitk_kernel[(rows, num_kv_heads, 1)](
            rotated, key_codes, value_codes, key_scales, value_scales, picks, block_table, token_to_req,
            partial_output, partial_lse, partial_output[slot], rotated.stride(0), rotated.stride(1),
            key_codes.stride(0), key_codes.stride(1), key_codes.stride(2), value_codes.stride(0),
            value_codes.stride(1), value_codes.stride(2), key_scales.stride(0), key_scales.stride(1),
            key_scales.stride(2), picks.stride(0), block_table.stride(0), out.stride(0), out.stride(1), rows,
            key_codes.shape[0], block_table.shape[0], TOPK=selection_width, PAGE_SIZE=key_codes.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size, HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=1, NUM_TILES=tiles, BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=partial_warps, num_stages=num_stages, PARTIAL_SLOT=slot, **({"LEAN": True} if lean else {}),
        )
    _qsa_kv4_merge_restore_kernel[(rows, num_kv_heads)](
        partial_output, partial_lse, signs, out, output_gate_view, out.stride(0), out.stride(1),
        output_gate_view.stride(0), output_gate_view.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=count,
        BLOCK_M=block_m, CHUNK=chunk, num_warps=DECODE_FUSED_WARPS, num_stages=1,
    )
    return out


def decode_int4_run(q, key_codes, value_codes, key_scales, value_scales, logical_indices, block_table,
                    token_to_req, out, output_gate_view, config, num_stages=2, work=None, lean=False):
    gated = {} if work is None else {"work_ptr": work, "GATED": True}
    block_n, partial_warps, num_tiles, num_splits = config
    num_stages = decode_stages() or num_stages
    head_dim = q.shape[2]
    rows, num_query_heads = q.shape[0], q.shape[1]
    num_kv_heads = key_codes.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)
    chunk = min(PREFILL_ROTATE_CHUNK, head_dim)
    selection_width = logical_indices.shape[1] - 1
    in_kernel = {}
    if splitk_fused():
        held = None if num_splits == 1 else splitk_counters(q.device, rows * num_kv_heads)
        if num_splits == 1 or held is not None:
            in_kernel = {"counter_ptr": held, "final_ptr": out, "gate_ptr": output_gate_view,
                         "stride_final_row": out.stride(0), "stride_final_head": out.stride(1),
                         "stride_gate_row": output_gate_view.stride(0),
                         "stride_gate_head": output_gate_view.stride(1), "CHUNK": chunk, "FUSED": True}
        else:
            splitk_aside("counters", rows=rows, kv_heads=num_kv_heads, layout="int4")
    announce("decode_fused", rows=rows, block_n=block_n, num_warps=partial_warps, num_splits=num_splits,
             kernels="rotate+split_merge_restore" if in_kernel else "rotate+split+merge_restore")
    if num_stages != 2:
        announce("decode_stages", knob=DECODE_STAGES_ENV, rows=rows, block_n=block_n, num_stages=num_stages)
    picks = logical_indices
    if lean:
        tally = tally_part(q.device, 0)
        picks = lean_slots(logical_indices, block_table, token_to_req, key_codes.shape[0], key_codes.shape[1],
                           tally=tally, work=work, pdl=pdl_on())
        announce(f"decode_lean_{block_n}x{num_splits}", knob=DECODE_LEAN_ENV, rows=rows, block_n=block_n,
                 num_warps=partial_warps,
                 num_splits=num_splits, num_stages=num_stages, gated=work is not None, tally=tally is not None,
                 path="slots+split_lean" + ("_merge_restore" if in_kernel else "+merge_restore"))
    signs = rotation_signs(head_dim, q.device)
    rotated = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _qsa_rotate_query_kernel[(rows, num_kv_heads)](
        q, signs, rotated, q.stride(0), q.stride(1), rotated.stride(0), rotated.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, BLOCK_M=block_m, CHUNK=chunk,
        num_warps=DECODE_FUSED_WARPS, num_stages=1, **gated,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
    partial_lse = (torch.empty((num_splits, rows, num_query_heads), dtype=torch.float32, device=q.device)
                   if num_splits > 1 else partial_output)
    direct = partial_output[0]
    _qsa_sparse_kv4_splitk_kernel[(rows, num_kv_heads, num_splits)](
        rotated,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        picks,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        direct,
        rotated.stride(0),
        rotated.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        picks.stride(0),
        block_table.stride(0),
        direct.stride(0),
        direct.stride(1),
        rows,
        key_codes.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=num_stages,
        **gated,
        **({"signs_ptr": signs, **in_kernel} if in_kernel else {}),
        **({"LEAN": True} if lean else {}),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    if in_kernel:
        return out
    _qsa_kv4_merge_restore_kernel[(rows, num_kv_heads)](
        partial_output, partial_lse, signs, out, output_gate_view, out.stride(0), out.stride(1),
        output_gate_view.stride(0), output_gate_view.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=num_splits,
        BLOCK_M=block_m, CHUNK=chunk, num_warps=DECODE_FUSED_WARPS, num_stages=1, **gated,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


UNION_DECODE_ENV = "TAAF_VLLM_QSA_UNION_DECODE"
UNION_DECODE_CHECKED_ENV = "TAAF_VLLM_QSA_UNION_DECODE_CHECKED"
UNION_DECODE_CHECKS = 3
UNION_BLOCK_N = 32
UNION_SPLITS = 4
UNION_WARPS = 8
UNION_MEMBER_WARPS = 4
UNION_SPEED_SLACK = 1.03
UNION_TALLY_FIELDS = ("union_launches", "union_tokens", "union_selections")
UNION_TALLY_COUNTER = "qsa_union"
_UNION = {"on": None, "checked": 0, "aside": "", "said": set(), "tally": {}, "registered": set()}


def union_decode_enabled():
    if _UNION["on"] is None:
        said = str(os.environ.get(UNION_DECODE_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{UNION_DECODE_ENV}={said!r} is not '', '0' or '1'")
        _UNION["on"] = said == "1"
    return _UNION["on"]


def union_decode_ready():
    return _UNION["checked"] >= UNION_DECODE_CHECKS or os.environ.get(UNION_DECODE_CHECKED_ENV) == "1"


def union_decode_say(key, line):
    if key in _UNION["said"]:
        return
    _UNION["said"].add(key)
    print(line, flush=True)


def union_decode_aside(reason):
    _UNION["aside"] = reason[:300]
    selfcheck_fail(UNION_DECODE_ENV, _UNION["aside"])
    union_decode_say("aside", f"{UNION_DECODE_ENV} stood aside: {_UNION['aside']}; every QSA decode keeps the per-row "
                              f"lean decode")


def union_tally(device):
    key = (device.type, device.index)
    held = _UNION["tally"].get(key)
    if held is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        held = torch.zeros(len(UNION_TALLY_FIELDS), dtype=torch.int64, device=device)
        _UNION["tally"][key] = held
    if key not in _UNION["registered"]:
        timers = sys.modules.get(TIMERS_MODULE)
        if callable(getattr(timers, "counter", None)):
            timers.counter(UNION_TALLY_COUNTER, UNION_TALLY_FIELDS, held)
            _UNION["registered"].add(key)
            print(f"QSA_KV8 tally {UNION_TALLY_COUNTER} registered with the step timers: "
                  f"{', '.join(UNION_TALLY_FIELDS)}", flush=True)
    return held


def union_tally_totals():
    columns = zip(*(held.tolist() for held in _UNION["tally"].values()))
    return dict(zip(UNION_TALLY_FIELDS, (sum(column) for column in columns)))


@triton.jit(do_not_specialize=["logical_stride", "back_stride", "resolved_stride", "resolved_width", "member_stride",
                               "num_slots", "num_tiles", "stock_n"])
def _qsa_union_member_kernel(
    logical_ptr,
    back_ptr,
    resolved_ptr,
    keep_ptr,
    member_ptr,
    count_ptr,
    logical_stride,
    back_stride,
    resolved_stride,
    resolved_width,
    member_stride,
    num_slots,
    num_tiles,
    stock_n,
    Q: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    tally_ptr=None,
    TALLY: tl.constexpr = False,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    lane = row // Q
    step = row % Q
    j = tl.arange(0, BLOCK)
    inside = j < TOPK
    pick = tl.load(logical_ptr + row.to(tl.int64) * logical_stride + j, mask=inside, other=-1)
    at = tl.load(back_ptr + lane.to(tl.int64) * back_stride + step * TOPK + j, mask=inside, other=0)
    at = tl.minimum(tl.maximum(at, 0), resolved_width - 1)
    kept = tl.load(keep_ptr + lane).to(tl.int1)
    found = tl.load(resolved_ptr + lane.to(tl.int64) * resolved_stride + at, mask=inside, other=-1)
    mapped = tl.where(inside & (pick >= 0) & kept, found, -1)
    count = tl.sum((inside & (mapped >= 0)).to(tl.int32), 0)
    limit = tl.minimum(num_tiles, tl.cdiv(tl.minimum(count, TOPK), stock_n)) * stock_n
    chosen = inside & (mapped >= 0) & (mapped < num_slots) & (j < limit)
    flag = tl.full((BLOCK,), 1, tl.int32) << step
    tl.atomic_or(member_ptr + lane.to(tl.int64) * member_stride + at, flag, mask=chosen)
    tl.atomic_max(count_ptr + lane, tl.max(tl.where(chosen, at + 1, 0), 0))
    if TALLY:
        tl.atomic_add(tally_ptr + 2, tl.sum(chosen.to(tl.int64), 0))


@triton.jit(do_not_specialize=["num_rows", "num_slots"])
def _qsa_union_kv4_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    union_ptr,
    member_ptr,
    count_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_union,
    stride_member,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_slots,
    Q: tl.constexpr,
    UNION_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MERGE_M: tl.constexpr,
    tally_ptr=None,
    TALLY: tl.constexpr = False,
    PDL: tl.constexpr = False,
    counter_ptr=None,
    signs_ptr=None,
    final_ptr=None,
    gate_ptr=None,
    stride_final_row=0,
    stride_final_head=0,
    stride_gate_row=0,
    stride_gate_head=0,
    CHUNK: tl.constexpr = 32,
    FUSED: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HALF: tl.constexpr = HEAD_DIM // 2
    lane = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    member_offsets = tl.arange(0, BLOCK_M)
    step = member_offsets // GROUP_SIZE
    head = member_offsets % GROUP_SIZE
    member_valid = member_offsets < Q * GROUP_SIZE
    row = lane * Q + tl.minimum(step, Q - 1)
    half_offsets = tl.arange(0, HALF)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = member_valid[:, None]
    query_rows = q_ptr + row[:, None].to(tl.int64) * stride_q_row + (first_head + head[:, None]) * stride_q_head
    query_even = tl.load(query_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
    query_odd = tl.load(query_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator_even = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accumulator_odd = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 / HEAD_DIM

    count = tl.load(count_ptr + lane)
    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(count, UNION_WIDTH), BLOCK_N))
    loaded = tl.sum(tl.zeros((BLOCK_N,), dtype=tl.int64), 0)

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        inside = columns < UNION_WIDTH
        slot = tl.load(union_ptr + lane.to(tl.int64) * stride_union + columns, mask=inside, other=-1)
        bits = tl.load(member_ptr + lane.to(tl.int64) * stride_member + columns, mask=inside, other=0)
        valid = (slot >= 0) & (slot < num_slots) & (bits != 0)
        safe_slot = tl.maximum(slot, 0)
        safe_page = (safe_slot // PAGE_SIZE).to(tl.int64)
        page_offset = safe_slot % PAGE_SIZE
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_packed = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + half_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = (
            safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        )
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        value_bits = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = 1.0 + (key_bits & 0xF).to(tl.float32) * 0.0625
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_zero = 1.0 + (value_bits & 0xF).to(tl.float32) * 0.0625
        value_scale = (value_bits & -16).to(tl.float32, bitcast=True) * 16.0
        key_low, key_high = _lean_nibbles(key_packed)
        raw = _bf16_dot(query_even, key_low, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        raw = _bf16_dot(query_odd, key_high, raw)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * (softmax_scale_log2 * 16.0))
        chosen = (((bits[None, :] >> step[:, None]) & 1) != 0) & valid[None, :] & member_valid[:, None]
        scores = tl.where(chosen, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            chosen, tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weights = (probabilities * value_scale[None, :]).to(query_even.dtype)
        weighted_zero = tl.sum(weights.to(tl.float32) * value_zero[None, :], axis=1)
        value_low, value_high = _lean_nibbles(value_packed)
        accumulator_even = _bf16_dot(weights, value_low, accumulator_even * alpha[:, None]) - weighted_zero[:, None]
        accumulator_odd = _bf16_dot(weights, value_high, accumulator_odd * alpha[:, None]) - weighted_zero[:, None]
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
        if TALLY:
            loaded += tl.sum(valid.to(tl.int64), 0)

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    output_even = tl.where(has_values[:, None], accumulator_even / denominator, 0.0)
    output_odd = tl.where(has_values[:, None], accumulator_odd / denominator, 0.0)
    if NUM_SPLITS == 1:
        output_rows = (output_ptr + row[:, None].to(tl.int64) * stride_output_row
                       + (first_head + head[:, None]) * stride_output_head)
        tl.store(output_rows + 2 * half_offsets[None, :], output_even, mask=head_mask)
        tl.store(output_rows + 2 * half_offsets[None, :] + 1, output_odd, mask=head_mask)
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        partial_rows = (partial_output_ptr
                        + (partial_row[:, None] * NUM_QUERY_HEADS + first_head + head[:, None]) * HEAD_DIM)
        tl.store(partial_rows + 2 * half_offsets[None, :], output_even, mask=head_mask)
        tl.store(partial_rows + 2 * half_offsets[None, :] + 1, output_odd, mask=head_mask)
        tl.store(partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head, partial_lse,
                 mask=member_valid)
    if TALLY:
        if kv_head == 0:
            tl.atomic_add(tally_ptr + 1, loaded)
            if split_id == 0:
                if lane == 0:
                    tl.atomic_add(tally_ptr, 1)
    if FUSED:
        tl.debug_barrier()
        if NUM_SPLITS == 1:
            for member in tl.static_range(Q):
                _kv4_merge_restore(
                    partial_output_ptr, partial_lse_ptr, signs_ptr, final_ptr, gate_ptr, stride_final_row,
                    stride_final_head, stride_gate_row, stride_gate_head, num_rows, lane * Q + member, kv_head,
                    GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, MERGE_M, CHUNK, True,
                )
        else:
            slot = lane * tl.num_programs(1) + kv_head
            seen = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
            if seen == NUM_SPLITS - 1:
                for member in tl.static_range(Q):
                    _kv4_merge_restore(
                        partial_output_ptr, partial_lse_ptr, signs_ptr, final_ptr, gate_ptr, stride_final_row,
                        stride_final_head, stride_gate_row, stride_gate_head, num_rows, lane * Q + member, kv_head,
                        GROUP_SIZE, HEAD_DIM, NUM_QUERY_HEADS, NUM_SPLITS, MERGE_M, CHUNK, True,
                    )
                tl.atomic_xchg(counter_ptr + slot, 0, sem="relaxed", scope="gpu")


def union_members(logical, back, resolved, keep, query_len, top_k, num_slots, stock_config, tally=None):
    lanes = resolved.shape[0]
    member = torch.zeros((lanes, resolved.shape[1]), dtype=torch.int32, device=resolved.device)
    count = torch.zeros(lanes, dtype=torch.int32, device=resolved.device)
    extra = {"tally_ptr": tally, "TALLY": True} if tally is not None else {}
    _qsa_union_member_kernel[(lanes * query_len,)](
        logical, back, resolved, keep, member, count, logical.stride(0), back.stride(0), resolved.stride(0),
        resolved.shape[1], member.stride(0), int(num_slots), int(stock_config[2]), int(stock_config[0]), Q=query_len,
        TOPK=top_k, BLOCK=triton.next_power_of_2(top_k), num_warps=UNION_MEMBER_WARPS,
        num_stages=1, **extra, PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return member, count


def union_config(width):
    block_n = UNION_BLOCK_N
    num_tiles = triton.cdiv(width, block_n)
    return block_n, UNION_WARPS, num_tiles, max(1, min(num_tiles, UNION_SPLITS))


def union_decode_run(q, key_codes, value_codes, key_scales, value_scales, logical, resolved, back, keep, query_len,
                     out, output_gate_view, num_stages=2, tally=None):
    rows, num_query_heads, head_dim = q.shape
    num_kv_heads = key_codes.shape[2]
    group_size = num_query_heads // num_kv_heads
    lanes = rows // query_len
    top_k = logical.shape[1] - 1
    stock_config = int4_config(_pinned(), rows, num_kv_heads, False, top_k)
    page = key_codes.shape[1]
    num_slots = key_codes.shape[0] * page
    member, count = union_members(logical, back, resolved, keep, query_len, top_k, num_slots, stock_config, tally)
    block_n, num_warps, num_tiles, num_splits = union_config(resolved.shape[1])
    num_stages = decode_stages() or num_stages
    block_m = max(16, triton.next_power_of_2(query_len * group_size))
    merge_m = triton.next_power_of_2(group_size)
    chunk = min(PREFILL_ROTATE_CHUNK, head_dim)
    in_kernel = {}
    if splitk_fused():
        held = None if num_splits == 1 else splitk_counters(q.device, lanes * num_kv_heads)
        if num_splits == 1 or held is not None:
            in_kernel = {"counter_ptr": held, "final_ptr": out, "gate_ptr": output_gate_view,
                         "stride_final_row": out.stride(0), "stride_final_head": out.stride(1),
                         "stride_gate_row": output_gate_view.stride(0),
                         "stride_gate_head": output_gate_view.stride(1), "CHUNK": chunk, "FUSED": True}
        else:
            splitk_aside("counters", rows=rows, kv_heads=num_kv_heads, layout="int4_union")
    union_decode_say("launch", f"QSA_UNION_DECODE launch lanes={lanes} rows={rows} query_len={query_len} "
                               f"union_width={resolved.shape[1]} block_m={block_m} block_n={block_n} "
                               f"num_warps={num_warps} num_splits={num_splits} fused={bool(in_kernel)} "
                               f"stock block_n={stock_config[0]} splits={stock_config[3]}")
    signs = rotation_signs(head_dim, q.device)
    rotated = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _qsa_rotate_query_kernel[(rows, num_kv_heads)](
        q, signs, rotated, q.stride(0), q.stride(1), rotated.stride(0), rotated.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, BLOCK_M=merge_m, CHUNK=chunk,
        num_warps=DECODE_FUSED_WARPS, num_stages=1, PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
    partial_lse = (torch.empty((num_splits, rows, num_query_heads), dtype=torch.float32, device=q.device)
                   if num_splits > 1 else partial_output)
    direct = partial_output[0]
    counted = {"tally_ptr": tally, "TALLY": True} if tally is not None else {}
    _qsa_union_kv4_kernel[(lanes, num_kv_heads, num_splits)](
        rotated, key_codes, value_codes, key_scales, value_scales, resolved, member, count, partial_output,
        partial_lse, direct, rotated.stride(0), rotated.stride(1), key_codes.stride(0), key_codes.stride(1),
        key_codes.stride(2), value_codes.stride(0), value_codes.stride(1), value_codes.stride(2), key_scales.stride(0),
        key_scales.stride(1), key_scales.stride(2), resolved.stride(0), member.stride(0), direct.stride(0),
        direct.stride(1), rows, num_slots, Q=query_len, UNION_WIDTH=resolved.shape[1], PAGE_SIZE=page,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, MERGE_M=merge_m, num_warps=num_warps,
        num_stages=num_stages, **counted, **({"signs_ptr": signs, **in_kernel} if in_kernel else {}),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    if in_kernel:
        return out
    _qsa_kv4_merge_restore_kernel[(rows, num_kv_heads)](
        partial_output, partial_lse, signs, out, output_gate_view, out.stride(0), out.stride(1),
        output_gate_view.stride(0), output_gate_view.stride(1), rows,
        GROUP_SIZE=group_size, HEAD_DIM=head_dim, NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=num_splits,
        BLOCK_M=merge_m, CHUNK=chunk, num_warps=DECODE_FUSED_WARPS, num_stages=1,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


def union_decode_route(key_codes):
    if not union_decode_enabled() or _UNION["aside"]:
        return "stock"
    if decode_lean_route(key_codes, None) != "lean":
        return "stock"
    if _tile_capturing():
        if not union_decode_ready():
            union_decode_say("capture", "QSA_UNION_DECODE: a CUDA graph captured before the union decode was checked "
                                        "keeps the per-row lean decode inside that graph")
            return "stock"
        return "union"
    if _UNION["checked"] >= UNION_DECODE_CHECKS:
        return "union"
    return "check"


def attention_passes(impl, query, kv_cache, passes, block_table, token_to_req, out, output_gate, between=None):
    chosen = layout(impl.kv_cache_dtype)
    if chosen != "int4" or not decode_fused_enabled() or query.stride(2) != 1 or out.stride(2) != 1:
        return None
    keys, values, key_scales, value_scales = views(impl, kv_cache, impl.head_size, chosen)
    width = passes[0].shape[1] - 1
    config = int4_config(_pinned(), query.shape[0], keys.shape[2], False, width)
    lean = decode_lean_route(keys, None) == "lean"
    announce("attention_passes", rows=query.shape[0], passes=len(passes), block_n=config[0], num_warps=config[1],
             lean=lean)
    return decode_int4_passes(query, keys, values, key_scales, value_scales, passes, block_table, token_to_req, out,
                              output_gate.view_as(query), config, between=between, lean=lean)


def union_attention(impl, query, kv_cache, logical, resolved, back, keep, query_len, out, output_gate, stock):
    if not union_decode_enabled():
        return None
    chosen = layout(impl.kv_cache_dtype)
    if chosen != "int4" or not decode_fused_enabled() or query.stride(2) != 1 or out.stride(2) != 1:
        union_decode_say("layout", f"QSA_UNION_DECODE serves the fused int4 decode only; this layer's {chosen} "
                                   f"cache keeps the per-row decode")
        return None
    rows = query.shape[0]
    if (os.environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1" or query_len < 1 or rows % query_len
            or resolved.shape[0] * query_len != rows or logical.stride(1) != 1
            or resolved.stride(1) != 1 or back.stride(1) != 1 or logical.dtype != torch.int32):
        return None
    keys, values, key_scales, value_scales = views(impl, kv_cache, impl.head_size, chosen)
    route = union_decode_route(keys)
    if route == "stock":
        return None
    gate_view = output_gate.view_as(query)
    args = (query, keys, values, key_scales, value_scales, logical, resolved, back, keep, query_len)
    if route == "union":
        return union_decode_run(*args, out, gate_view, tally=union_tally(query.device))
    try:
        want = stock(torch.empty_like(out))
        made = union_decode_run(*args, out, gate_view, tally=union_tally(query.device))
        error = float((made.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
        worst = float((made.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
        finite = bool(torch.isfinite(made).all())
    except Exception as failure:
        union_decode_aside(f"{type(failure).__name__}: {failure}")
        return stock(out)
    passed = error <= DECODE_LEAN_TOLERANCE and finite
    print(f"QSA_UNION_DECODE check rows={rows} query_len={query_len} relative={error:.2e} maxrel={worst:.2e} "
          f"tolerance={DECODE_LEAN_TOLERANCE:g} {'pass' if passed else 'FAIL'}", flush=True)
    if not passed:
        union_decode_aside(f"a check at {rows} rows read relative {error:.2e} against the per-row lean decode")
        out.copy_(want)
        return out
    _UNION["checked"] += 1
    if _UNION["checked"] >= UNION_DECODE_CHECKS:
        union_decode_say("live", f"QSA_UNION_DECODE live: each lane's decode reads its verification rows' union of "
                                 f"picks once for every row that selected it, after {UNION_DECODE_CHECKS} checks "
                                 f"against the per-row lean decode")
    return made


@_marked("qsa_kv8_back")
def union_tables(indices, query_len):
    rows, columns = indices.shape
    top_k = columns - 1
    lanes = rows // query_len
    device = indices.device
    picks = indices[:, :top_k].reshape(lanes, query_len * top_k).long()
    live = picks >= 0
    width = query_len * top_k
    resolved = torch.full((lanes, width), -1, dtype=torch.int32, device=device)
    back = torch.zeros((lanes, width), dtype=torch.int32, device=device)
    for lane in range(lanes):
        held = picks[lane][live[lane]]
        unique = torch.unique(held)
        resolved[lane, :unique.numel()] = unique.to(torch.int32)
        if held.numel():
            back[lane, live[lane]] = torch.searchsorted(unique, held).to(torch.int32)
    keep = torch.ones(lanes, dtype=torch.bool, device=device)
    return resolved, back, keep


def union_mapped(logical, resolved, back, keep, query_len):
    rows, columns = logical.shape
    top_k = columns - 1
    lanes = rows // query_len
    picks = logical[:, :top_k].long()
    at = back.long().view(lanes, query_len, top_k).clamp(0, resolved.shape[1] - 1).reshape(rows, top_k)
    owner = torch.arange(rows, device=logical.device) // query_len
    found = resolved.long()[owner.view(-1, 1), at]
    kept = keep[owner].view(-1, 1)
    mapped = torch.where((picks >= 0) & kept, found, torch.full_like(found, -1)).to(torch.int32)
    out = torch.empty((rows, columns), dtype=torch.int32, device=logical.device)
    out[:, :top_k] = mapped
    out[:, top_k] = (mapped >= 0).sum(dim=1).to(torch.int32)
    return out


@triton.jit
def _write_k4v8_kernel(
    key_ptr,
    value_ptr,
    key_codes_ptr,
    value_codes_ptr,
    key_scales_ptr,
    value_scales_ptr,
    slot_mapping_ptr,
    stride_key_token,
    stride_key_head,
    stride_value_token,
    stride_value_head,
    stride_kc_block,
    stride_kc_slot,
    stride_kc_head,
    stride_vc_block,
    stride_vc_slot,
    stride_vc_head,
    stride_s_block,
    stride_s_slot,
    stride_s_head,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HALF: tl.constexpr = HEAD_DIM // 2
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0:
        return
    block = slot // PAGE_SIZE
    position = slot % PAGE_SIZE
    half_offsets = tl.arange(0, HALF)
    dim_offsets = tl.arange(0, HEAD_DIM)

    key_base = key_ptr + token * stride_key_token + head * stride_key_head
    key_even = tl.load(key_base + 2 * half_offsets).to(tl.float32)
    key_odd = tl.load(key_base + 2 * half_offsets + 1).to(tl.float32)
    key_min = tl.minimum(tl.min(key_even), tl.min(key_odd))
    key_max = tl.maximum(tl.max(key_even), tl.max(key_odd))
    key_scale = tl.maximum((key_max - key_min) / 15.0, 1e-6)
    zero_raw = -key_min / key_scale
    key_zero = tl.clamp(
        tl.where(zero_raw >= 0, (zero_raw + 0.5).to(tl.int32), (zero_raw - 0.5).to(tl.int32)).to(tl.float32),
        0.0,
        15.0,
    )
    inverse = 1.0 / key_scale
    even_scaled = key_even * inverse + key_zero
    odd_scaled = key_odd * inverse + key_zero
    even_code = tl.clamp(
        tl.where(even_scaled >= 0, (even_scaled + 0.5).to(tl.int32), (even_scaled - 0.5).to(tl.int32)).to(tl.float32),
        0.0,
        15.0,
    ).to(tl.int32)
    odd_code = tl.clamp(
        tl.where(odd_scaled >= 0, (odd_scaled + 0.5).to(tl.int32), (odd_scaled - 0.5).to(tl.int32)).to(tl.float32),
        0.0,
        15.0,
    ).to(tl.int32)
    key_packed = ((even_code & 0xF) | ((odd_code & 0xF) << 4)).to(tl.uint8)
    tl.store(
        key_codes_ptr + block * stride_kc_block + position * stride_kc_slot + head * stride_kc_head + half_offsets,
        key_packed,
    )
    scale_bits = key_scale.to(tl.int32, bitcast=True)
    packed_scale = ((scale_bits & -16) | (key_zero.to(tl.int32) & 0xF)).to(tl.float32, bitcast=True)
    scale_offset = block * stride_s_block + position * stride_s_slot + head * stride_s_head
    tl.store(key_scales_ptr + scale_offset, packed_scale)

    value = tl.load(value_ptr + token * stride_value_token + head * stride_value_head + dim_offsets).to(tl.float32)
    value_scale = tl.maximum(tl.max(tl.abs(value)) / 127.0, 1e-6)
    value_scaled = value * (1.0 / value_scale)
    value_scaled = tl.where(value_scaled >= 0, value_scaled + 0.5, value_scaled - 0.5)
    value_scaled = tl.clamp(value_scaled, -128.0, 127.0)
    tl.store(
        value_codes_ptr + block * stride_vc_block + position * stride_vc_slot + head * stride_vc_head + dim_offsets,
        value_scaled,
    )
    tl.store(value_scales_ptr + scale_offset, value_scale)


def write_k4v8(key, value, key_codes, value_codes, key_scales, value_scales, slot_mapping):
    single_rht = _rht()
    if key_scales.stride() != value_scales.stride():
        raise ValueError("QSA_KV8 k4v8 key and value scales must share one layout")
    rotated = single_rht(key.float()).to(key.dtype)
    rows, heads, head_dim = key.shape
    if not rows:
        return
    _write_k4v8_kernel[(rows, heads)](
        rotated,
        value,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        slot_mapping,
        rotated.stride(0),
        rotated.stride(1),
        value.stride(0),
        value.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        PAGE_SIZE=key_codes.shape[1],
        HEAD_DIM=head_dim,
        num_warps=4,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_k4v8_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_s_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HALF: tl.constexpr = HEAD_DIM // 2
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    valid_count = tl.where((request >= 0) & (request < num_requests), valid_count, 0)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    query_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    query_even = tl.load(query_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
    query_odd = tl.load(query_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)
    query_sum = tl.sum(query_even.to(tl.float32), axis=1) + tl.sum(query_odd.to(tl.float32), axis=1)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634 / HEAD_DIM

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        key_packed = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + half_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        value_codes = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale_offsets = (
            safe_page * stride_s_block + page_offset * stride_s_token + kv_head * stride_s_head
        )
        key_bits = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.int32, bitcast=True)
        key_zero = (key_bits & 0xF).to(tl.float32)
        key_scale = (key_bits & -16).to(tl.float32, bitcast=True)
        value_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
        key_low = (key_packed & 0xF).to(query_even.dtype)
        key_high = ((key_packed >> 4) & 0xF).to(query_even.dtype)
        raw = tl.dot(query_even, key_low) + tl.dot(query_odd, key_high)
        scores = (raw - query_sum[:, None] * key_zero[None, :]) * (key_scale[None, :] * softmax_scale_log2)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weights = probabilities * value_scale[None, :]
        accumulator = tl.dot(
            weights.to(query_even.dtype),
            value_codes.to(query_even.dtype),
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


def sparse_attention_k4v8(q, key_codes, value_codes, key_scales, value_scales, logical_indices,
                          block_table, token_to_req, use_prefill_config, out=None, *, output_gate,
                          config=None, num_stages=2):
    pinned = _pinned()
    single_rht = _rht()
    if q.ndim != 3 or key_codes.ndim != 4 or value_codes.ndim != 4:
        raise ValueError("QSA_KV8 k4v8 attention received invalid Q/K/V shapes")
    if key_codes.shape[:3] != value_codes.shape[:3] or key_scales.shape != key_codes.shape[:3]:
        raise ValueError("QSA_KV8 k4v8 codes and scales must share [blocks, page, kv_heads]")
    if key_scales.stride() != value_scales.stride():
        raise ValueError("QSA_KV8 k4v8 key and value scales must share one layout")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0] or logical_indices.shape[1] < 2:
        raise ValueError("QSA_KV8 k4v8 indices must have one row per query plus the count column")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA_KV8 k4v8 metadata has invalid shapes")
    if not all(key_codes.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA_KV8 k4v8 cache and block table must be nonempty")
    head_dim = q.shape[2]
    if head_dim != 2 * key_codes.shape[3] or head_dim != value_codes.shape[3] or q.shape[1] % key_codes.shape[2]:
        raise ValueError("QSA_KV8 k4v8 needs packed keys, full-width values and grouped heads")
    assert head_dim >= 32 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert key_codes.dtype == torch.uint8 and value_codes.dtype == torch.int8
    assert key_scales.dtype == value_scales.dtype == torch.float32
    assert logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert key_codes.stride(3) == value_codes.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA_KV8 k4v8 output must match its query")
    assert out.dtype == q.dtype and out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    rotated = single_rht(q.float()).to(q.dtype).contiguous()
    group_size = q.shape[1] // key_codes.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1
    block_n, partial_warps, num_tiles, num_splits = config or pinned._select_config(
        q.shape[0], key_codes.shape[2], use_prefill_config, selection_width
    )
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32,
                                  device=q.device)

    _qsa_sparse_k4v8_splitk_kernel[(q.shape[0], key_codes.shape[2], num_splits)](
        rotated,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        rotated.stride(0),
        rotated.stride(1),
        key_codes.stride(0),
        key_codes.stride(1),
        key_codes.stride(2),
        value_codes.stride(0),
        value_codes.stride(1),
        value_codes.stride(2),
        key_scales.stride(0),
        key_scales.stride(1),
        key_scales.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        key_codes.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=key_codes.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=num_stages,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    if num_splits == 1:
        return out
    pinned._qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
        **pdl_args(pinned._qsa_merge_splitk_kernel),
    )
    return out


def warmup(holder, kv_cache, block_table, *, num_query_heads, selection_width):
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        triton_scalar_specialization_rep,
    )

    pinned = _pinned()
    chosen = layout(getattr(holder, "kv_cache_dtype", CACHE_DTYPE))
    int4 = chosen == "int4"
    head_dim = int(holder.head_size)
    key_codes, value_codes, key_scales, value_scales = views(holder, kv_cache, head_dim, chosen)
    num_kv_heads = key_codes.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)
    fast = fast_enabled() and chosen == "int8"
    follow = follows_int4(chosen)
    if follow:
        profiles = {
            int4_config(pinned, num_rows, num_kv_heads, use_prefill_config, selection_width)
            for num_rows in range(1, 8193)
            for use_prefill_config in (False, True)
        }
    else:
        profiles = {
            select_config(pinned, num_rows, num_kv_heads, use_prefill_config, selection_width, fast)
            for num_rows in range(1, 8193)
            for use_prefill_config in (False, True)
        }
    fused = follow and decode_fused_enabled()

    def like(tensor):
        return TritonWarmupTensor(tensor.dtype, aligned=tensor.data_ptr() % 16 == 0,
                                  shape=tuple(tensor.shape), strides=tuple(tensor.stride()))

    num_rows = 16
    num_requests = 16
    shape = (num_rows, num_query_heads, head_dim)
    q_ptr = TritonWarmupTensor(torch.bfloat16, shape=shape)
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(torch.float32 if int4 else torch.bfloat16, shape=shape)
    output_gate_ptr = TritonWarmupTensor(torch.bfloat16, shape=shape)
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(key_codes.shape[0])
    gate = () if int4 else (output_gate_ptr,)
    gate_strides = () if int4 else (row_stride, head_stride)
    kernel = {"int8": _qsa_sparse_kv8_splitk_kernel, "int4": _qsa_sparse_kv4_splitk_kernel,
              "k4v8": _qsa_sparse_k4v8_splitk_kernel}[chosen]
    own_merge = fused and chosen == "int8"
    merge = (_qsa_kv4_merge_kernel if int4 else
             _qsa_kv8_merge_kernel if own_merge else pinned._qsa_merge_splitk_kernel)

    warmed = []
    ungated = {"work_ptr": None, "GATED": False} if chosen in ("int4", "int8") else {}
    gated = {"work_ptr": TritonWarmupTensor(torch.int32, shape=(1,)), "GATED": True}
    spill_gate = fused and str(os.environ.get(SPILL_GATE_ENV) or "").strip().lower() in ("1", "on", "true")
    gated_profiles = sorted({int4_config(pinned, rows, num_kv_heads, False, selection_width)
                             for rows in range(1, SPILL_GATE_ROWS + 1)}) if spill_gate else []
    plan = [(profile, ungated) for profile in sorted(profiles)] + [(profile, gated) for profile in gated_profiles]
    lean_decode = fused and int4 and decode_lean_enabled()
    plan = [(profile, extra, False) for profile, extra in plan] + (
        [(profile, extra, True) for profile, extra in plan] if lean_decode else [])
    decode_stage_count = (decode_stages() or 2) if fused and int4 else 2
    if lean_decode or (follow and lean_enabled()):
        fast_tally(kv_cache.device)
    inside_on = splitk_fused() and ((int4 and fused) or chosen == "int8")
    if inside_on:
        splitk_counters(kv_cache.device, 1)
    unmerged = {"int4": {"counter_ptr": None, "signs_ptr": None, "final_ptr": None, "gate_ptr": None,
                         "stride_final_row": 0, "stride_final_head": 0, "stride_gate_row": 0,
                         "stride_gate_head": 0, "CHUNK": 32, "FUSED": False},
                "int8": {"counter_ptr": None, "BLOCK_SPLITS": 1, "FUSED": False}}.get(chosen, {})
    for (block_n, warps, num_tiles, num_splits), extra, lean_flag in plan:
        inside = dict(unmerged)
        if inside_on:
            counters = TritonWarmupTensor(torch.int32, shape=(SPLITK_COUNTERS,))
            if int4:
                inside = {"counter_ptr": counters if num_splits > 1 else None,
                          "signs_ptr": TritonWarmupTensor(torch.float32, shape=(head_dim,)),
                          "final_ptr": TritonWarmupTensor(torch.bfloat16, shape=shape),
                          "gate_ptr": output_gate_ptr, "stride_final_row": row_stride,
                          "stride_final_head": head_stride, "stride_gate_row": row_stride,
                          "stride_gate_head": head_stride, "CHUNK": min(PREFILL_ROTATE_CHUNK, head_dim),
                          "FUSED": True}
            elif num_splits > 1:
                inside = {"counter_ptr": counters, "BLOCK_SPLITS": triton.next_power_of_2(num_splits),
                          "FUSED": True}
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(torch.float32, shape=(num_splits, *shape))
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads))
        kernel.warmup(
            q_ptr,
            like(key_codes),
            like(value_codes),
            like(key_scales),
            like(value_scales),
            indices_ptr,
            like(block_table),
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            *gate,
            row_stride,
            head_stride,
            key_codes.stride(0),
            key_codes.stride(1),
            key_codes.stride(2),
            value_codes.stride(0),
            value_codes.stride(1),
            value_codes.stride(2),
            key_scales.stride(0),
            key_scales.stride(1),
            key_scales.stride(2),
            selection_width + 1,
            block_table.stride(0),
            row_stride,
            head_stride,
            *gate_strides,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_codes.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=decode_stage_count,
            grid=(num_rows, num_kv_heads, num_splits),
            **extra,
            **inside,
            **({"LEAN": lean_flag} if int4 else {}),
            **pdl_args(kernel),
        )
        if lean_flag:
            warmed.append((block_n, num_splits, warps, "decode_lean") + (("gated",) if extra.get("GATED") else ()))
            continue
        if inside.get("FUSED") and (int4 or num_splits > 1):
            warmed.append((block_n, num_splits, warps, "splitk_fused") + (("gated",) if extra.get("GATED") else ()))
            continue
        if num_splits > 1 and (own_merge or not extra.get("GATED")):
            merge.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                *gate,
                row_stride,
                head_stride,
                *gate_strides,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
                **(extra if own_merge else {}),
                **pdl_args(merge),
            )
        if fused and int4:
            _qsa_kv4_merge_restore_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                TritonWarmupTensor(torch.float32, shape=(head_dim,)),
                TritonWarmupTensor(torch.bfloat16, shape=shape),
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                GROUP_SIZE=group_size,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_M=block_m,
                CHUNK=min(PREFILL_ROTATE_CHUNK, head_dim),
                num_warps=DECODE_FUSED_WARPS,
                num_stages=1,
                grid=(num_rows, num_kv_heads),
                **extra,
                PDL=pdl_on(), launch_pdl=pdl_on(),
            )
        warmed.append((block_n, num_splits, warps) + (("gated",) if extra.get("GATED") else ()))
    for extra in ((ungated, gated) if spill_gate else (ungated,)) if fused and int4 else ():
        _qsa_rotate_query_kernel.warmup(
            q_ptr,
            TritonWarmupTensor(torch.float32, shape=(head_dim,)),
            TritonWarmupTensor(torch.bfloat16, shape=shape),
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            CHUNK=min(PREFILL_ROTATE_CHUNK, head_dim),
            num_warps=DECODE_FUSED_WARPS,
            num_stages=1,
            grid=(num_rows, num_kv_heads),
            **extra,
            PDL=pdl_on(), launch_pdl=pdl_on(),
        )
    if fused:
        warmed.append(("decode_fused", DECODE_FUSED_WARPS) if int4 else ("decode_fused", "split+merge"))
    for extra in ((ungated, gated) if spill_gate else (ungated,)) if lean_decode else ():
        _qsa_lean_slots_kernel.warmup(
            indices_ptr, like(block_table), token_to_req_ptr,
            TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1)), selection_width + 1,
            selection_width + 1, block_table.stride(0), num_cache_blocks, num_requests, TOPK=selection_width,
            PAGE_SIZE=key_codes.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
            BLOCK=triton.next_power_of_2(selection_width), num_warps=LEAN_WARPS, num_stages=1, grid=(num_rows,),
            tally_ptr=TritonWarmupTensor(torch.int64, shape=(len(FAST_TALLY_FIELDS),)), TALLY=True, **extra,
            **pdl_args(_qsa_lean_slots_kernel))
        warmed.append(("decode_lean_slots",) + (("gated",) if extra.get("GATED") else ()))
    settings = prefill_settings() if prefill_enabled() else None
    if settings and settings["mode"] == "fused":
        block_n = settings["block_n"]
        common = {"TOPK": selection_width, "PAGE_SIZE": key_codes.shape[1],
                  "PAGE_TABLE_WIDTH": block_table.shape[1], "GROUP_SIZE": group_size, "HEAD_DIM": head_dim,
                  "NUM_TILES": triton.cdiv(selection_width, block_n), "BLOCK_M": block_m, "BLOCK_N": block_n,
                  "num_warps": settings["num_warps"], "num_stages": settings["num_stages"]}
        cache_args = (like(key_codes), like(value_codes), like(key_scales), like(value_scales), indices_ptr,
                      like(block_table), token_to_req_ptr)
        cache_strides = (key_codes.stride(0), key_codes.stride(1), key_codes.stride(2), value_codes.stride(0),
                         value_codes.stride(1), value_codes.stride(2), key_scales.stride(0),
                         key_scales.stride(1), key_scales.stride(2), selection_width + 1, block_table.stride(0))
        tail = (row_stride, head_stride, row_stride, head_stride, num_rows, num_cache_blocks, num_requests)
        if int4:
            _qsa_prefill_kv4_kernel.warmup(
                q_ptr, TritonWarmupTensor(torch.float32, shape=(head_dim,)), *cache_args,
                TritonWarmupTensor(torch.bfloat16, shape=shape), output_gate_ptr, row_stride, head_stride,
                *cache_strides, *tail, CHUNK=min(PREFILL_ROTATE_CHUNK, head_dim),
                grid=(num_rows, num_kv_heads), **common, PDL=pdl_on(), launch_pdl=pdl_on())
        tile = tile_rows() if follow else 0
        if tile:
            union_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, tile * selection_width))
            signs = (TritonWarmupTensor(torch.float32, shape=(head_dim,)),) if int4 else ()
            rotation = {"CHUNK": min(PREFILL_ROTATE_CHUNK, head_dim)} if int4 else {}
            tile_kernel = _qsa_prefill_kv4_tile_kernel if int4 else _qsa_prefill_kv8_tile_kernel
            tile_kernel.warmup(
                q_ptr, *signs, *cache_args[:4], union_ptr,
                union_ptr, TritonWarmupTensor(torch.int32, shape=(num_rows,)), like(block_table),
                TritonWarmupTensor(torch.bfloat16, shape=shape), output_gate_ptr, row_stride, head_stride,
                *cache_strides[:9], tile * selection_width, block_table.stride(0), *tail,
                PAGE_SIZE=key_codes.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
                HEAD_DIM=head_dim, TILE_ROWS=tile, GROUP_BLOCK=block_m, BLOCK_N=TILE_BLOCK_N,
                SPAN=TILE_SPAN, num_warps=tile_warps(tile), num_stages=settings["num_stages"],
                grid=(num_rows, num_kv_heads), **rotation, SKIP_EMPTY=tile_skip_empty(),
                **({"PDL": pdl_on(), "launch_pdl": pdl_on()} if int4 else {}))
            if tile_union_fused():
                bound = min(block_table.shape[1] * key_codes.shape[1], TILE_SPAN)
                words = -(-(-(-bound // 32)) // TILE_UNION_CHUNK) * TILE_UNION_CHUNK
                _tile_union_kernel.warmup(
                    TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1)), selection_width + 1,
                    token_to_req_ptr, union_ptr, union_ptr, TritonWarmupTensor(torch.int32, shape=(num_rows,)),
                    TritonWarmupTensor(torch.int32, shape=(num_rows,)),
                    TritonWarmupTensor(torch.int32, shape=(num_rows, words)),
                    TritonWarmupTensor(torch.int32, shape=(num_rows, words)), num_rows, num_requests,
                    WIDTH=selection_width, TILE=tile, BLOCK_N=TILE_BLOCK_N, SPAN=TILE_SPAN, BOUND=bound, WORDS=words,
                    CHUNK=TILE_UNION_CHUNK, num_warps=8, grid=(-(-num_rows // tile),),
                    **pdl_args(_tile_union_kernel))
                warmed.append(("tile_union_fused", tile))
            warmed.append(("prefill_tile", tile, tile_warps(tile), settings["num_stages"]))
        if follow and lean_enabled():
            slots_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
            _qsa_lean_slots_kernel.warmup(
                indices_ptr, like(block_table), token_to_req_ptr, slots_ptr, selection_width + 1,
                selection_width + 1, block_table.stride(0), num_cache_blocks, num_requests, TOPK=selection_width,
                PAGE_SIZE=key_codes.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
                BLOCK=triton.next_power_of_2(selection_width), num_warps=LEAN_WARPS, num_stages=1,
                grid=(num_rows,), work_ptr=None, GATED=False, PDL=False,
                tally_ptr=TritonWarmupTensor(torch.int64, shape=(len(FAST_TALLY_FIELDS),)), TALLY=True)
            lean_shape = {"TOPK": selection_width, "PAGE_SIZE": key_codes.shape[1], "GROUP_SIZE": group_size,
                          "HEAD_DIM": head_dim, "NUM_TILES": triton.cdiv(selection_width, block_n),
                          "BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": settings["num_warps"],
                          "num_stages": settings["num_stages"], "grid": (num_rows, num_kv_heads),
                          **lean_cap(None)}
            if int4:
                if not _tile_capturing():
                    lean_tables(head_dim, kv_cache.device)
                pairs = num_rows * num_query_heads
                columns = lean_columns(head_dim)
                rotated_ptr = TritonWarmupTensor(torch.bfloat16, shape=shape)
                table_ptr = TritonWarmupTensor(torch.bfloat16, shape=(head_dim, head_dim))
                dense = {"NUM_HEADS": num_query_heads, "HEAD_DIM": head_dim, "BLOCK_M": LEAN_PAIRS,
                         "BLOCK_C": columns, "num_warps": LEAN_WARPS, "num_stages": 1,
                         "grid": (triton.cdiv(pairs, LEAN_PAIRS), head_dim // columns)}
                _qsa_lean_rotate_kernel.warmup(q_ptr, table_ptr, rotated_ptr, row_stride, head_stride, pairs,
                                               **dense)
                _qsa_lean_kv4_kernel.warmup(rotated_ptr, *cache_args[:4], slots_ptr, rotated_ptr,
                                            *cache_strides[:9], selection_width + 1, num_rows,
                                            NUM_HEADS=num_query_heads, **lean_shape)
                _qsa_lean_restore_kernel.warmup(rotated_ptr, table_ptr, output_gate_ptr,
                                                TritonWarmupTensor(torch.bfloat16, shape=shape), row_stride,
                                                head_stride, row_stride, head_stride, pairs, **dense)
            else:
                _qsa_lean_kv8_kernel.warmup(q_ptr, *cache_args[:4], slots_ptr, output_ptr, output_gate_ptr,
                                            row_stride, head_stride, *cache_strides[:9], selection_width + 1,
                                            row_stride, head_stride, row_stride, head_stride, num_rows,
                                            **lean_shape)
            warmed.append(("prefill_lean", block_n, settings["num_warps"], settings["num_stages"]))
        if not int4:
            kernel.warmup(
                q_ptr, *cache_args, output_ptr, output_ptr, output_ptr, output_gate_ptr, row_stride,
                head_stride, *cache_strides, *tail, NUM_QUERY_HEADS=num_query_heads, NUM_SPLITS=1,
                grid=(num_rows, num_kv_heads, 1), **common, **(ungated if chosen == "int8" else {}),
                **unmerged, **pdl_args(kernel))
        warmed.append(("prefill", block_n, settings["num_warps"], settings["num_stages"]))
    extra = {"prefill": settings["mode"]} if settings else {}
    if fused:
        extra["decode"] = "fused"
    if follow and decode_warps():
        extra["decode_warps"] = decode_warps()
    if chosen == "int8" and follow:
        extra["int8_fast"] = "on"
    announce("warmup", layout=chosen, profiles=len(warmed), cache=tuple(kv_cache.shape), **extra)
    return tuple(warmed)


SELF_CHECK_PAGE = 1664
SELF_CHECK_DECODE_ROWS = 8
SELF_CHECK_SAMPLED_ROWS = 16
SELF_CHECK_TOLERANCE = 3.0e-2


def dequantized(chosen, codes, scales, head_dim):
    if chosen == "int8":
        return codes.float() * scales.float().unsqueeze(-1)
    if chosen != "int4":
        raise ValueError(f"QSA_KV8 has no reference dequantization for {chosen!r}")
    bits = scales.contiguous().view(torch.int32)
    zero = (bits & 0xF).float()
    scale = (bits & -16).view(torch.float32)
    rotated = torch.stack([(codes & 0xF).float(), ((codes >> 4) & 0xF).float()], dim=-1).flatten(-2)
    rotated = (rotated - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    return _rht()(rotated, inverse=True) / head_dim


def reference_rows(rows, query, keys, values, logical_indices, block_table, token_to_req, gate):
    page, kv_heads, head_dim = keys.shape[1], keys.shape[2], keys.shape[3]
    group = query.shape[1] // kv_heads
    width = logical_indices.shape[1] - 1
    out = []
    for row in rows:
        count = int(logical_indices[row, width])
        picked = logical_indices[row, :min(count, width)].long()
        picked = picked[picked >= 0]
        pages = block_table[int(token_to_req[row])].long()[picked // page]
        offsets = picked % page
        held_keys, held_values = keys[pages, offsets], values[pages, offsets]
        grouped = query[row].float().view(kv_heads, group, head_dim)
        weights = torch.softmax(torch.einsum("hgd,nhd->hgn", grouped, held_keys) * head_dim ** -0.5, dim=-1)
        attended = torch.einsum("hgn,nhd->hgd", weights, held_values).reshape(-1, head_dim)
        out.append(attended.to(torch.bfloat16).float() * torch.sigmoid(gate[row].float()))
    return torch.stack(out)


def self_check(cache_dtype, *, head_dim, kv_heads, query_heads, width, page=SELF_CHECK_PAGE, device="cuda",
               seed=0):
    from types import SimpleNamespace

    chosen = layout(cache_dtype)
    holder = SimpleNamespace(kv_cache_dtype=cache_dtype, head_size=head_dim)
    content = content_bytes(*widths(chosen, head_dim))
    per_request = -(-width // page) + 1
    requests = 2
    blocks = requests * per_request
    generator = torch.Generator().manual_seed(seed)
    tokens = blocks * page
    key = torch.randn((tokens, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    value = torch.randn((tokens, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    storage = torch.uint8 if chosen == "int4" else torch.int8
    physical = torch.zeros((blocks, page, kv_heads, content), dtype=storage, device=device)
    cache = physical.permute(0, 2, 1, 3)
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(requests, per_request)
    results = []
    started = time.monotonic()
    write(holder, key, value, cache, torch.arange(tokens, dtype=torch.int64, device=device))
    torch.cuda.synchronize()
    key_codes, value_codes, key_scales, value_scales = views(holder, cache, head_dim, chosen)
    held_keys = dequantized(chosen, key_codes, key_scales, head_dim)
    held_values = dequantized(chosen, value_codes, value_scales, head_dim)
    written = float((held_keys.reshape(tokens, kv_heads, head_dim) - key.float()).norm() / key.float().norm())
    results.append(("write", bool(torch.isfinite(held_keys).all()) and written < 0.5,
                    f"{chosen} codes of {tokens} rows dequantize to the written keys at relative {written:.3f}",
                    time.monotonic() - started))
    prefill_rows = PREFILL_MIN_PROGRAMS // kv_heads + 8
    for name, rows, use_prefill in (("decode", SELF_CHECK_DECODE_ROWS, False), ("prefill", prefill_rows, True)):
        started = time.monotonic()
        query = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        gate = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        owner = torch.arange(rows, dtype=torch.int32) % requests
        picks = torch.stack([torch.randperm(per_request * page, generator=generator)[:width] for _ in range(rows)])
        indices = torch.cat([picks.to(torch.int32), torch.full((rows, 1), width, dtype=torch.int32)], dim=1)
        indices, owner = indices.to(device), owner.to(device)
        out = torch.zeros_like(query)
        try:
            attention(holder, query, cache, indices, table, owner, use_prefill, out, gate)
            torch.cuda.synchronize()
        except Exception as failure:
            results.append((name, False, f"{type(failure).__name__}: {failure}"[:400], time.monotonic() - started))
            continue
        sampled = sorted({int(one) for one in torch.linspace(0, rows - 1, min(rows, SELF_CHECK_SAMPLED_ROWS))})
        want = reference_rows(sampled, query, held_keys, held_values, indices, table, owner, gate)
        got = out[sampled].float()
        error = float((got - want).norm() / want.norm().clamp_min(1e-12))
        finite = bool(torch.isfinite(out).all())
        aside = _TILE["aside"] if use_prefill and tile_rows() else ""
        results.append((name, finite and error <= SELF_CHECK_TOLERANCE,
                        f"{chosen} {rows} rows x {width} picks against attention over the dequantized cache: "
                        f"relative {error:.2e} (tolerance {SELF_CHECK_TOLERANCE:g}) finite={finite}"
                        + (f" prefill={'fused' if prefill_enabled() and prefill_launch(rows, kv_heads, True) else 'stock'}"
                           f" tile={tile_rows()}{' aside: ' + aside if aside else ''}" if use_prefill else
                           f" decode_fused={int(decode_fused_enabled())} decode_warps={decode_warps()}"),
                        time.monotonic() - started))
    return results


IDENTITY_ENV = "TAAF_VLLM_KERNEL_IDENTITY"
IDENTITY_DECODE_ROWS = 440
IDENTITY_KNOBS = {"decode": (SPLITK_FUSED_ENV,), "prefill": (TILE_UNION_ENV, TILE_SKIP_ENV)}


def identity_repeats():
    said = str(os.environ.get(IDENTITY_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def identity_compare(base, other):
    differs = base != other
    count = int(differs.sum())
    gap = float((base.float() - other.float()).abs().max()) if count else 0.0
    return count, gap


def identity_check(cache_dtype, *, head_dim, kv_heads, query_heads, width, repeats, page=SELF_CHECK_PAGE,
                   device="cuda", seed=2):
    from types import SimpleNamespace

    chosen = layout(cache_dtype)
    holder = SimpleNamespace(kv_cache_dtype=cache_dtype, head_size=head_dim)
    per_request = -(-width // page) + 1
    requests = 110
    blocks = requests * per_request
    generator = torch.Generator().manual_seed(seed)
    tokens = blocks * page
    content = content_bytes(*widths(chosen, head_dim))
    storage = torch.uint8 if chosen == "int4" else torch.int8
    physical = torch.zeros((blocks, page, kv_heads, content), dtype=storage, device=device)
    cache = physical.permute(0, 2, 1, 3)
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(requests, per_request)
    chunk = 1 << 16
    for start in range(0, tokens, chunk):
        stop = min(tokens, start + chunk)
        key = torch.randn((stop - start, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        value = torch.randn((stop - start, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        write(holder, key, value, cache, torch.arange(start, stop, dtype=torch.int64, device=device))
    torch.cuda.synchronize()
    rows_out = []
    prefill_rows = PREFILL_MIN_PROGRAMS // kv_heads + 8
    for name, rows, use_prefill in (("decode", IDENTITY_DECODE_ROWS, False), ("prefill", prefill_rows, True)):
        knobs = IDENTITY_KNOBS[name]
        saved = {knob: os.environ.get(knob) for knob in knobs}
        started = time.monotonic()
        mismatched = runs = 0
        worst = 0.0
        unstable = 0
        failure = ""
        try:
            for _ in range(repeats):
                query = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
                gate = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
                owner = (torch.arange(rows, dtype=torch.int32) // 4 if not use_prefill else
                         torch.arange(rows, dtype=torch.int32)) % requests
                picks = torch.stack([torch.randperm(per_request * page, generator=generator)[:width]
                                     for _ in range(rows)])
                indices = torch.cat([picks.to(torch.int32), torch.full((rows, 1), width, dtype=torch.int32)], dim=1)
                indices, owner = indices.to(device), owner.to(device)
                outs = []
                for setting in ("0", "1", "1"):
                    for knob in knobs:
                        os.environ[knob] = setting
                    out = torch.zeros_like(query)
                    attention(holder, query, cache, indices, table, owner, use_prefill, out, gate)
                    torch.cuda.synchronize()
                    outs.append(out)
                count, gap = identity_compare(outs[0], outs[1])
                again, _ = identity_compare(outs[1], outs[2])
                mismatched += int(count > 0)
                unstable += int(again > 0)
                worst = max(worst, gap)
                runs += 1
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"[:300]
        finally:
            for knob, value in saved.items():
                if value is None:
                    os.environ.pop(knob, None)
                else:
                    os.environ[knob] = value
        verdict = "identical" if not mismatched and not unstable and not failure else "differs"
        rows_out.append((name, verdict, f"{'+'.join(knobs)} on vs off, {chosen}, {rows} rows x {width} picks, "
                                        f"{runs} launches: {mismatched} with any differing element, max abs diff "
                                        f"{worst:.3g}; on vs on again: {unstable} unstable"
                                        + (f"; stopped: {failure}" if failure else ""),
                         time.monotonic() - started))
    return rows_out


def lean_self_check(cache_dtype, *, head_dim, kv_heads, query_heads, width, page=SELF_CHECK_PAGE, device="cuda",
                    seed=1):
    from types import SimpleNamespace

    started = time.monotonic()
    chosen = layout(cache_dtype)
    if not follows_int4(chosen):
        return [("prefill_lean", False, f"QSA_PREFILL_LEAN serves the int4 cache, or the int8 one laid out as int8 "
                                        f"under {INT8_FAST_ENV}=1, not {chosen}", 0.0)]
    holder = SimpleNamespace(kv_cache_dtype=cache_dtype, head_size=head_dim)
    per_request = -(-width // page) + 1
    requests = 2
    blocks = requests * per_request
    tokens = blocks * page
    rows = PREFILL_MIN_PROGRAMS // kv_heads + 8
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn((tokens, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    value = torch.randn((tokens, kv_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    storage = torch.uint8 if chosen == "int4" else torch.int8
    physical = torch.zeros((blocks, page, kv_heads, content_bytes(*widths(chosen, head_dim))), dtype=storage,
                           device=device)
    cache = physical.permute(0, 2, 1, 3)
    table = torch.arange(blocks, dtype=torch.int32, device=device).view(requests, per_request)
    query = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    gate = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
    owner = (torch.arange(rows, dtype=torch.int32) % requests).to(device)
    picks = torch.stack([torch.randperm(per_request * page, generator=generator)[:width] for _ in range(rows)])
    indices = torch.cat([picks.to(torch.int32), torch.full((rows, 1), width, dtype=torch.int32)], dim=1).to(device)
    settings = prefill_settings()
    knobs = {"block_n": settings["block_n"], "num_warps": settings["num_warps"], "num_stages": settings["num_stages"]}
    per_row = prefill_int4 if chosen == "int4" else prefill_int8_rows
    lean = prefill_lean_int4 if chosen == "int4" else prefill_lean_int8
    try:
        write(holder, key, value, cache, torch.arange(tokens, dtype=torch.int64, device=device))
        key_codes, value_codes, key_scales, value_scales = views(holder, cache, head_dim, chosen)
        args = (query, key_codes, value_codes, key_scales, value_scales, indices, table, owner)
        want = per_row(*args, None, gate, **knobs)
        torch.cuda.synchronize()
        rows_ms = _lean_time(lambda: per_row(*args, torch.empty_like(query), gate, **knobs))
        spent = {cap: _lean_time(lambda: lean(*args, torch.empty_like(query), gate, maxnreg=cap, **knobs))
                 for cap in LEAN_CAPS}
        chosen_cap = min(spent, key=spent.get)
        got = lean(*args, None, gate, maxnreg=chosen_cap, **knobs)
        torch.cuda.synchronize()
        held_keys = dequantized(chosen, key_codes, key_scales, head_dim)
        held_values = dequantized(chosen, value_codes, value_scales, head_dim)
    except Exception as failure:
        return [("prefill_lean", False, f"{type(failure).__name__}: {failure}"[:400], time.monotonic() - started)]
    sampled = sorted({int(one) for one in torch.linspace(0, rows - 1, min(rows, SELF_CHECK_SAMPLED_ROWS))})
    reference = reference_rows(sampled, query, held_keys, held_values, indices, table, owner, gate)
    against_rows = float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
    against_reference = float((got[sampled].float() - reference).norm() / reference.norm().clamp_min(1e-12))
    finite = bool(torch.isfinite(got).all())
    passed = finite and against_rows <= LEAN_TOLERANCE and against_reference <= SELF_CHECK_TOLERANCE
    return [("prefill_lean", passed,
             f"{chosen} {rows} rows x {width} picks: against the per-row kernel relative {against_rows:.2e} "
             f"(tolerance {LEAN_TOLERANCE:g}), against attention over the dequantized cache {against_reference:.2e} "
             f"(tolerance {SELF_CHECK_TOLERANCE:g}) finite={finite}; per-row {rows_ms:.3f} ms, lean "
             + ", ".join(f"{spent[cap]:.3f} ms at maxnreg {cap or 'free'}" for cap in LEAN_CAPS)
             + f", keeping maxnreg {chosen_cap or 'free'} ({rows_ms / spent[chosen_cap]:.2f}x the per-row kernel; "
               f"block_n={knobs['block_n']} num_warps={knobs['num_warps']} num_stages={knobs['num_stages']})",
             time.monotonic() - started, {"maxnreg": chosen_cap}),
            lean_bucket_check(chosen, key_codes, value_codes, key_scales, value_scales, query_heads, width, knobs,
                              chosen_cap, device, seed)]


def device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


LEAN_BUCKET_ROWS = 2048
LEAN_BUCKET_EMPTY = 440
LEAN_BUCKET_KEEP = 0.9
LEAN_SPEED_SLACK = 1.03


def markov_picks(rows, span, width, keep, generator):
    current = torch.randperm(span, generator=generator)[:width]
    out = []
    kept_count = int(round(keep * width))
    for _ in range(rows):
        out.append(current)
        kept = current[torch.randperm(width, generator=generator)[:kept_count]]
        pool = torch.ones(span, dtype=torch.bool)
        pool[kept] = False
        fresh = pool.nonzero().flatten()
        fresh = fresh[torch.randperm(fresh.numel(), generator=generator)[:width - kept_count]]
        current = torch.cat([kept, fresh])
    return torch.stack(out)


def lean_bucket_check(chosen, key_codes, value_codes, key_scales, value_scales, query_heads, width, knobs, cap,
                      device, seed):
    started = time.monotonic()
    try:
        generator = torch.Generator().manual_seed(seed + 17)
        rows, empty = LEAN_BUCKET_ROWS, LEAN_BUCKET_EMPTY
        page, head_dim = key_codes.shape[1], key_codes.shape[3] * (2 if chosen == "int4" else 1)
        blocks = key_codes.shape[0]
        table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
        live = markov_picks(rows - empty, min(blocks * page, 16384), width, LEAN_BUCKET_KEEP, generator)
        indices = torch.full((rows, width + 1), -1, dtype=torch.int32)
        indices[empty:, :width] = live.to(torch.int32)
        indices[empty:, width] = width
        indices[:empty, :width] = live[:1].to(torch.int32).expand(empty, width)
        indices[:empty, width] = 0
        indices = indices.to(device)
        owner = torch.zeros(rows, dtype=torch.int32, device=device)
        query = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        gate = torch.randn((rows, query_heads, head_dim), generator=generator).to(torch.bfloat16).to(device)
        per_row = prefill_int4 if chosen == "int4" else prefill_int8_rows
        tiled = prefill_int4_tile if chosen == "int4" else prefill_int8_tile
        lean = prefill_lean_int4 if chosen == "int4" else prefill_lean_int8
        args = (query, key_codes, value_codes, key_scales, value_scales, indices, table, owner)
        tail = (query[empty:], key_codes, value_codes, key_scales, value_scales, indices[empty:], table, owner[empty:])
        want = per_row(*args, None, gate, **knobs)
        got = lean(*args, None, gate, maxnreg=cap, **knobs)
        device_sync()
        error = float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
        empty_zero = bool((got[:empty] == 0).all())
        finite = bool(torch.isfinite(got).all())
        tile = tile_rows() or 4
        times = {
            "per_row": _lean_time(lambda: per_row(*args, torch.empty_like(query), gate, **knobs)),
            "tile": _lean_time(lambda: tiled(*args, torch.zeros_like(query), gate, tile=tile, block_n=knobs["block_n"],
                                             num_stages=knobs["num_stages"])),
            "lean": _lean_time(lambda: lean(*args, torch.empty_like(query), gate, maxnreg=cap, **knobs)),
            "lean_live_rows_only": _lean_time(lambda: lean(*tail, torch.empty_like(query[empty:]), gate[empty:],
                                                           maxnreg=cap, **knobs)),
            "tile_live_rows_only": _lean_time(lambda: tiled(*tail, torch.zeros_like(query[empty:]), gate[empty:],
                                                            tile=tile, block_n=knobs["block_n"],
                                                            num_stages=knobs["num_stages"])),
        }
    except Exception as failure:
        return ("prefill_lean_bucket", False, f"{type(failure).__name__}: {failure}"[:400], time.monotonic() - started)
    replaced = "tile" if tile_rows() else "per_row"
    fast = times["lean"] <= LEAN_SPEED_SLACK * times[replaced]
    passed = finite and empty_zero and error <= LEAN_TOLERANCE and fast
    return ("prefill_lean_bucket", passed,
            f"{chosen} bucket of {rows} rows ({empty} empty, {rows - empty} live, consecutive rows keep "
            f"{LEAN_BUCKET_KEEP:g} of their picks): lean against the per-row kernel relative {error:.2e} "
            f"(tolerance {LEAN_TOLERANCE:g}) empty_rows_zero={empty_zero} finite={finite}; "
            + ", ".join(f"{name} {spent:.3f} ms" for name, spent in times.items())
            + f"; lean against the {replaced} path it replaces {times[replaced] / times['lean']:.2f}x"
            + ("" if fast else f", slower than {LEAN_SPEED_SLACK:g}x of it"),
            time.monotonic() - started, {"maxnreg": cap, **{name: round(spent, 4) for name, spent in times.items()}})


DECODE_LEAN_LANES = 110
DECODE_LEAN_LANE_ROWS = 4
DECODE_LEAN_KEEP = 0.77
DECODE_LEAN_MIXED_LIVE = 128
DECODE_LEAN_SAMPLED = 16
DECODE_LEAN_TIMED = 9
DECODE_LEAN_SPEED_SLACK = 1.03


def random_int4_cache(blocks, page, kv_heads, head_dim, device, generator):
    from types import SimpleNamespace

    physical = torch.empty((blocks, page, kv_heads, content_bytes(*widths("int4", head_dim))), dtype=torch.uint8,
                           device=device)
    physical.copy_(torch.randint(0, 256, physical.shape, dtype=torch.uint8, generator=generator, device=device))
    cache = physical.permute(0, 2, 1, 3)
    holder = SimpleNamespace(kv_cache_dtype=INT4_DTYPE, head_size=head_dim)
    made = views(holder, cache, head_dim, "int4")
    for scales in made[2:]:
        fresh = torch.rand(scales.shape, generator=generator, device=device) * 0.02 + 0.005
        bits = (fresh.view(torch.int32) & -16) | torch.randint(4, 12, scales.shape, generator=generator,
                                                              device=device, dtype=torch.int32)
        scales.copy_(bits.view(torch.float32))
    return made


def lane_picks(lanes, lane_rows, span, width, keep, generator, device):
    rows = []
    kept_count = int(round(keep * width))
    for _ in range(lanes):
        base = torch.randperm(span, generator=generator, device=device)[:width]
        for _ in range(lane_rows):
            kept = base[torch.randperm(width, generator=generator, device=device)[:kept_count]]
            pool = torch.ones(span, dtype=torch.bool, device=device)
            pool[kept] = False
            fresh = pool.nonzero().flatten()
            fresh = fresh[torch.randperm(fresh.numel(), generator=generator, device=device)[:width - kept_count]]
            rows.append(torch.cat([kept, fresh])[torch.randperm(width, generator=generator, device=device)])
    return torch.stack(rows)


def int4_reference(rows, query, key_codes, value_codes, key_scales, value_scales, indices, table, owners, gate):
    page, kv_heads, half = key_codes.shape[1], key_codes.shape[2], key_codes.shape[3]
    head_dim = 2 * half
    group = query.shape[1] // kv_heads
    width = indices.shape[1] - 1
    out = []
    for row in rows:
        count = int(indices[row, width])
        picked = indices[row, :min(count, width)].long()
        picked = picked[picked >= 0]
        pages = table[int(owners[row])].long()[picked // page]
        offsets = picked % page
        keys = dequantized("int4", key_codes[pages, offsets], key_scales[pages, offsets], head_dim)
        values = dequantized("int4", value_codes[pages, offsets], value_scales[pages, offsets], head_dim)
        grouped = query[row].float().view(kv_heads, group, head_dim)
        weights = torch.softmax(torch.einsum("hgd,nhd->hgn", grouped, keys) * head_dim ** -0.5, dim=-1)
        attended = torch.einsum("hgn,nhd->hgd", weights, values).reshape(-1, head_dim)
        out.append(attended.to(torch.bfloat16).float() * torch.sigmoid(gate[row].float()))
    return torch.stack(out)


def decode_lean_case(name, cache, query_heads, width, lanes, use_prefill, identity, empty, device, generator):
    key_codes, value_codes, key_scales, value_scales = cache
    page, kv_heads, half = key_codes.shape[1], key_codes.shape[2], key_codes.shape[3]
    head_dim = 2 * half
    lane_rows = DECODE_LEAN_LANE_ROWS
    per_lane = key_codes.shape[0] // max(lanes, 1)
    live = lanes * lane_rows
    rows = live + empty
    picks = lane_picks(lanes, lane_rows, per_lane * page, width, DECODE_LEAN_KEEP, generator, device)
    if identity:
        table = torch.arange(key_codes.shape[0], dtype=torch.int32, device=device).view(1, -1)
        lane_of = torch.arange(live, device=device) // lane_rows
        picks = picks + (lane_of * per_lane * page).view(-1, 1)
        owners = torch.zeros(rows, dtype=torch.int32, device=device)
    else:
        table = torch.randperm(key_codes.shape[0], generator=generator, device=device)[:lanes * per_lane]
        table = table.view(lanes, per_lane).to(torch.int32)
        owners = torch.cat([torch.zeros(empty, dtype=torch.int32, device=device),
                            (torch.arange(live, device=device) // lane_rows).to(torch.int32)])
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    indices[empty:, :width] = picks.to(torch.int32)
    indices[empty:, width] = width
    if empty:
        indices[:empty, :width] = picks[:1].to(torch.int32).expand(empty, width)
        indices[:empty, width] = 0
    query = (torch.randn((rows, query_heads, head_dim), generator=generator, device=device) * 0.5).to(torch.bfloat16)
    gate = torch.randn((rows, query_heads, head_dim), generator=generator, device=device).to(torch.bfloat16)
    config = int4_config(_pinned(), rows, kv_heads, use_prefill, width)
    args = (query, key_codes, value_codes, key_scales, value_scales, indices, table, owners)
    return name, args, gate, config, empty


def decode_lean_verdict(case, timed):
    name, args, gate, config, empty = case
    query = args[0]
    started = time.monotonic()
    want = decode_int4_run(*args, torch.empty_like(query), gate, config)
    got = decode_int4_run(*args, torch.empty_like(query), gate, config, lean=True)
    device_sync()
    error = float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
    rows = query.shape[0]
    sampled = sorted({int(one) for one in torch.linspace(empty, rows - 1, min(rows - empty, DECODE_LEAN_SAMPLED))})
    reference = int4_reference(sampled, *args, gate)
    against_reference = float((got[sampled].float() - reference).norm() / reference.norm().clamp_min(1e-12))
    stock_reference = float((want[sampled].float() - reference).norm() / reference.norm().clamp_min(1e-12))
    finite = bool(torch.isfinite(got).all())
    empty_zero = bool((got[:empty] == 0).all()) if empty else True
    passed = (finite and empty_zero and error <= DECODE_LEAN_TOLERANCE
              and against_reference <= max(SELF_CHECK_TOLERANCE, 1.5 * stock_reference))
    times = {}
    if timed:
        times = {"stock": _lean_time(lambda: decode_int4_run(*args, torch.empty_like(query), gate, config),
                                     DECODE_LEAN_TIMED),
                 "lean": _lean_time(lambda: decode_int4_run(*args, torch.empty_like(query), gate, config, lean=True),
                                    DECODE_LEAN_TIMED)}
    fast = not times or times["lean"] <= DECODE_LEAN_SPEED_SLACK * times["stock"]
    detail = (f"int4 {rows} rows ({empty} empty) x {args[5].shape[1] - 1} picks, config block_n={config[0]} "
              f"warps={config[1]} splits={config[3]} stages={decode_stages() or 2}: lean against the stock decode "
              f"relative {error:.2e} (tolerance {DECODE_LEAN_TOLERANCE:g}); against fp32 attention over the "
              f"dequantized picks lean {against_reference:.2e}, stock {stock_reference:.2e}; finite={finite} "
              f"empty_rows_zero={empty_zero}")
    if times:
        detail += (f"; stock {times['stock']:.3f} ms, lean {times['lean']:.3f} ms "
                   f"({times['stock'] / times['lean']:.2f}x)" + ("" if fast else ", lean is slower"))
    return (name, passed and fast, detail, time.monotonic() - started, {k: round(v, 4) for k, v in times.items()})


def decode_lean_graph(case):
    name, args, gate, config, _ = case
    query = args[0]
    started = time.monotonic()
    eager = decode_int4_run(*args, torch.empty_like(query), gate, config, lean=True)
    held = torch.empty_like(query)
    decode_int4_run(*args, held, gate, config, lean=True)
    device_sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_int4_run(*args, held, gate, config, lean=True)
    held.fill_(7.0)
    for _ in range(3):
        graph.replay()
    device_sync()
    replayed = torch.equal(held, eager)
    work = torch.zeros(1, dtype=torch.int32, device=query.device)
    gated = torch.full_like(query, 5.0)
    decode_int4_run(*args, gated, gate, config, work=work, lean=True)
    work.fill_(1)
    opened = torch.full_like(query, 5.0)
    decode_int4_run(*args, opened, gate, config, work=work, lean=True)
    device_sync()
    closed = bool((gated == 5.0).all())
    equal = torch.equal(opened, eager)
    return (f"{name}_graph_gate", replayed and closed and equal,
            f"lean decode captured in a CUDA graph replays equal to eager {replayed}; a zero work gate leaves the "
            f"output untouched {closed}; an open gate equals the ungated launch {equal}", time.monotonic() - started)


def decode_lean_self_check(cache_dtype, *, head_dim, kv_heads, query_heads, width, page=SELF_CHECK_PAGE, device="cuda",
                           seed=3):
    started = time.monotonic()
    if layout(cache_dtype) != "int4":
        return [("decode_lean", False, f"{DECODE_LEAN_ENV} serves the int4 cache, not {cache_dtype}", 0.0)]
    if not decode_fused_enabled():
        return [("decode_lean", False, f"{DECODE_LEAN_ENV} runs inside the fused int4 decode and needs "
                                       f"{DECODE_FUSED_ENV}=1", 0.0)]
    rows_out = []
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
        per_lane = -(-width // page) + 1
        cache = random_int4_cache(DECODE_LEAN_LANES * per_lane, page, kv_heads, head_dim, device, generator)
        lanes = DECODE_LEAN_LANES
        cases = [
            (decode_lean_case("decode_lean_host_kv", cache, query_heads, width, lanes, False, True, 0, device,
                              generator), True),
            (decode_lean_case("decode_lean_paged", cache, query_heads, width, lanes, False, False, 0, device,
                              generator), False),
            (decode_lean_case("decode_lean_small", cache, query_heads, width, 2, False, False, 0, device, generator),
             False),
            (decode_lean_case("decode_lean_mixed_bucket", cache, query_heads, width,
                              DECODE_LEAN_MIXED_LIVE // DECODE_LEAN_LANE_ROWS, True, False,
                              lanes * DECODE_LEAN_LANE_ROWS, device, generator), True),
        ]
        for case, timed in cases:
            rows_out.append(decode_lean_verdict(case, timed))
        rows_out.append(decode_lean_graph(cases[0][0]))
        tally = fast_tally_totals()
        launches, live = tally.get("decode_lean_launches", 0), tally.get("decode_lean_rows", 0)
        rows_out.append(("decode_lean_tally", launches > 0 and live > 0,
                         f"the device tally counted {launches} lean decode launches and {live} live rows", 0.0))
    except Exception as failure:
        rows_out.append(("decode_lean", False, f"{type(failure).__name__}: {failure}"[:400],
                         time.monotonic() - started))
    return rows_out


def union_decode_case(name, cache, query_heads, width, lanes, query_len, keep, device, generator, dropped=0, padded=0):
    key_codes = cache[0]
    page = key_codes.shape[1]
    per_lane = key_codes.shape[0] // max(lanes, 1)
    rows = lanes * query_len
    picks = lane_picks(lanes, query_len, per_lane * page, width, keep, generator, device)
    lane_of = torch.arange(rows, device=device) // query_len
    picks = picks + (lane_of * per_lane * page).view(-1, 1)
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32, device=device)
    indices[:, :width] = picks.to(torch.int32)
    indices[:, width] = width
    if padded:
        indices[rows - padded:, :width] = -1
        indices[rows - padded:, width] = 0
    resolved, back, kept = union_tables(indices, query_len)
    if dropped:
        kept[:dropped] = False
    mapped = union_mapped(indices, resolved, back, kept, query_len)
    table = torch.arange(key_codes.shape[0], dtype=torch.int32, device=device).view(1, -1)
    owners = torch.zeros(rows, dtype=torch.int32, device=device)
    query = (torch.randn((rows, query_heads, 2 * key_codes.shape[3]), generator=generator, device=device)
             * 0.5).to(torch.bfloat16)
    gate = torch.randn(query.shape, generator=generator, device=device).to(torch.bfloat16)
    return name, cache, query, gate, indices, resolved, back, kept, mapped, table, owners, query_len


def union_decode_pair(case, out_stock=None, out_union=None, tally=None):
    name, cache, query, gate, indices, resolved, back, kept, mapped, table, owners, query_len = case
    key_codes, value_codes, key_scales, value_scales = cache
    config = int4_config(_pinned(), query.shape[0], key_codes.shape[2], False, indices.shape[1] - 1)
    stock = decode_int4_run(query, key_codes, value_codes, key_scales, value_scales, mapped, table, owners,
                            torch.empty_like(query) if out_stock is None else out_stock, gate, config, lean=True)
    union = union_decode_run(query, key_codes, value_codes, key_scales, value_scales, indices, resolved, back, kept,
                             query_len, torch.empty_like(query) if out_union is None else out_union, gate,
                             tally=tally)
    return stock, union


def union_decode_verdict(case, timed):
    name, cache, query, gate, indices, resolved, back, kept, mapped, table, owners, query_len = case
    started = time.monotonic()
    stock, union = union_decode_pair(case, tally=union_tally(query.device))
    device_sync()
    error = float((union.float() - stock.float()).norm() / stock.float().norm().clamp_min(1e-12))
    rows = query.shape[0]
    live = [row for row in range(rows) if int(mapped[row, -1]) > 0]
    sampled = (sorted({live[int(one)] for one in torch.linspace(0, len(live) - 1,
                                                                min(len(live), DECODE_LEAN_SAMPLED))})
               if live else [])
    if sampled:
        reference = int4_reference(sampled, query, *cache, mapped, table, owners, gate)
        against = float((union[sampled].float() - reference).norm() / reference.norm().clamp_min(1e-12))
        stock_against = float((stock[sampled].float() - reference).norm() / reference.norm().clamp_min(1e-12))
    else:
        against = stock_against = 0.0
    dead = [row for row in range(rows) if int(mapped[row, -1]) == 0]
    dead_zero = bool((union[dead] == 0).all()) if dead else True
    finite = bool(torch.isfinite(union).all())
    loaded = int((resolved[kept] >= 0).sum())
    selections = int((mapped[:, :-1] >= 0).sum())
    passed = (finite and dead_zero and error <= DECODE_LEAN_TOLERANCE
              and against <= max(SELF_CHECK_TOLERANCE, 1.5 * stock_against))
    times = {}
    if timed:
        times = {"stock": _lean_time(lambda: union_decode_pair(case)[0], DECODE_LEAN_TIMED),
                 "pair": _lean_time(lambda: union_decode_pair(case), DECODE_LEAN_TIMED)}
        times["union"] = max(times["pair"] - times["stock"], 1e-6)
    fast = not times or times["union"] <= UNION_SPEED_SLACK * times["stock"]
    detail = (f"int4 {rows} rows of {query_len} a lane x {indices.shape[1] - 1} picks, {loaded} union tokens for "
              f"{selections} row picks ({loaded / max(selections, 1):.2f}): union against the per-row lean decode "
              f"relative {error:.2e} (tolerance {DECODE_LEAN_TOLERANCE:g}); against fp32 attention over the "
              f"dequantized picks union {against:.2e}, per-row {stock_against:.2e}; finite={finite} "
              f"empty_rows_zero={dead_zero}")
    if times:
        detail += (f"; per-row {times['stock']:.3f} ms, union {times['union']:.3f} ms "
                   f"({times['stock'] / times['union']:.2f}x)" + ("" if fast else ", union is slower"))
    return (name, passed and fast, detail, time.monotonic() - started, {k: round(v, 4) for k, v in times.items()})


def union_decode_graph(case):
    name, cache, query, gate, indices, resolved, back, kept, mapped, table, owners, query_len = case
    started = time.monotonic()
    key_codes, value_codes, key_scales, value_scales = cache
    tally = union_tally(query.device)
    args = (query, key_codes, value_codes, key_scales, value_scales, indices, resolved, back, kept, query_len)
    eager = union_decode_run(*args, torch.empty_like(query), gate, tally=tally)
    held = torch.empty_like(query)
    union_decode_run(*args, held, gate, tally=tally)
    device_sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        union_decode_run(*args, held, gate, tally=tally)
    held.fill_(7.0)
    for _ in range(3):
        graph.replay()
    device_sync()
    replayed = torch.equal(held, eager)
    return (f"{name}_graph", replayed, f"union decode captured in a CUDA graph replays equal to eager {replayed}",
            time.monotonic() - started)


def union_decode_self_check(cache_dtype, *, query_len, head_dim, kv_heads, query_heads, width, page=SELF_CHECK_PAGE,
                            device="cuda", seed=5):
    started = time.monotonic()
    if layout(cache_dtype) != "int4":
        return [("union_decode", False, f"{UNION_DECODE_ENV} serves the int4 cache, not {cache_dtype}", 0.0)]
    if not decode_fused_enabled():
        return [("union_decode", False, f"{UNION_DECODE_ENV} runs inside the fused int4 decode and needs "
                                        f"{DECODE_FUSED_ENV}=1", 0.0)]
    if query_len < 2:
        return [("union_decode", False, f"{UNION_DECODE_ENV} shares a lane's reads across its verification rows and "
                                        f"needs at least 2 rows a lane, found {query_len}", 0.0)]
    rows_out = []
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
        per_lane = -(-width // page) + 1
        lanes = DECODE_LEAN_LANES
        cache = random_int4_cache(lanes * per_lane, page, kv_heads, head_dim, device, generator)
        cases = [
            (union_decode_case("union_decode_served", cache, query_heads, width, lanes, query_len, DECODE_LEAN_KEEP,
                               device, generator), True),
            (union_decode_case("union_decode_disjoint", cache, query_heads, width, lanes, query_len, 0.0, device,
                               generator), False),
            (union_decode_case("union_decode_dropped", cache, query_heads, width, 4, query_len, DECODE_LEAN_KEEP,
                               device, generator, dropped=1, padded=1), False),
        ]
        for case, timed in cases:
            rows_out.append(union_decode_verdict(case, timed))
        rows_out.append(union_decode_graph(cases[0][0]))
        tally = union_tally_totals()
        launches, tokens, selections = (tally.get("union_launches", 0), tally.get("union_tokens", 0),
                                        tally.get("union_selections", 0))
        rows_out.append(("union_decode_tally", launches > 0 and 0 < tokens < selections,
                         f"the device tally counted {launches} union launches, {tokens} tokens loaded for "
                         f"{selections} row picks", 0.0))
    except Exception as failure:
        rows_out.append(("union_decode", False, f"{type(failure).__name__}: {failure}"[:400],
                         time.monotonic() - started))
    return rows_out

PICK_DUMP_ENV = "TAAF_VLLM_PICK_DUMP"
PICK_DUMP_PATH = os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "pick-dump.jsonl")
PICK_DUMP_ROWS = 4
PICK_DUMP_MIN_POSITION = 4096
PICK_DUMP_DECODE_ENV = "TAAF_VLLM_PICK_DUMP_DECODE_ROWS"
_PICK_DUMP = {"left": 0, "hooked": False, "decode_rows": 0}


def pick_dump_record(prefix, positions, selected, rows=PICK_DUMP_ROWS):
    taken = min(int(selected.shape[0]), int(positions.shape[0]), rows)
    out = []
    for row in range(taken):
        values = selected[row]
        count = int(values[-1])
        picks = sorted(int(one) for one in values[:-1][:count].tolist() if int(one) >= 0)
        out.append({"position": int(positions[row]), "count": count, "picks": picks})
    return {"prefix": prefix, "rows": out}


def pick_dump_install():
    said = str(os.environ.get(PICK_DUMP_ENV) or "").strip()
    if not said or said == "0" or _PICK_DUMP["hooked"]:
        return False
    try:
        from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
    except ImportError:
        return False
    _PICK_DUMP["left"] = int(said)
    _PICK_DUMP["decode_rows"] = int(str(os.environ.get(PICK_DUMP_DECODE_ENV) or "0").strip() or 0)
    _PICK_DUMP["hooked"] = True
    forward = QSAIndexer.forward

    def dumped(self, projected_qk, positions, out=None):
        result = forward(self, projected_qk, positions, out)
        if _PICK_DUMP["left"] <= 0 or positions.numel() == 0 or torch.cuda.is_current_stream_capturing():
            return result
        flat = positions.reshape(-1)
        if int(flat.min()) < PICK_DUMP_MIN_POSITION:
            return result
        decode_rows = _PICK_DUMP["decode_rows"]
        if decode_rows and flat.numel() > decode_rows:
            return result
        _PICK_DUMP["left"] -= 1
        record = pick_dump_record(getattr(self, "prefix", "?"), flat, result, rows=decode_rows or PICK_DUMP_ROWS)
        with open(os.environ.get("TAAF_VLLM_PICK_DUMP_PATH", PICK_DUMP_PATH), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return result

    QSAIndexer.forward = dumped
    announce("pick_dump", knob=PICK_DUMP_ENV, calls=_PICK_DUMP["left"], decode_rows=_PICK_DUMP["decode_rows"],
             path=PICK_DUMP_PATH,
             why="each indexer call past the first positions writes its first rows' picks, so two serves can be "
                 "compared on the same contexts")
    return True


pick_dump_install()


INDEX_COMPACT_ENV = "TAAF_VLLM_QSA_INDEX_COMPACT"
INDEX_OPS_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
INDEX_COMPACT_CONFIG = {"tile_r": 32, "block_n": 64, "k_tiles": 16, "stages": 2, "num_warps": 4}
INDEX_COMPACT_WIDE_CONFIG = {"tile_r": 64, "block_n": 64, "k_tiles": 16, "stages": 2, "num_warps": 4}
INDEX_COMPACT_CHECKS = 3
_INDEX = {"on": None, "stock": None, "installed": False, "checked": 0, "aside": "", "said": set()}


def index_say(key, line):
    if key in _INDEX["said"]:
        return
    _INDEX["said"].add(key)
    print(line, flush=True)


def index_compact_enabled():
    if _INDEX["on"] is None:
        said = str(os.environ.get(INDEX_COMPACT_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{INDEX_COMPACT_ENV}={said!r} is not '', '0' or '1'")
        _INDEX["on"] = said == "1"
    return _INDEX["on"]


@triton.jit(do_not_specialize=["num_rows", "query_offset", "num_requests"])
def _index_compact_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_REQ: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    owners = tl.arange(0, BLOCK_REQ)
    held = owners < num_requests
    starts = tl.maximum(tl.load(query_start_loc_ptr + owners, mask=held, other=0) - query_base, query_offset)
    ends = tl.minimum(tl.load(query_start_loc_ptr + owners + 1, mask=held, other=0) - query_base, query_end)
    tiles = tl.where(held & (ends > starts), (ends - starts + TILE_R - 1) // TILE_R, 0)
    through = tl.cumsum(tiles, 0)
    program = tl.program_id(0)
    request = tl.sum((through <= program).to(tl.int32), 0)
    if request >= num_requests:
        return
    mine = owners == request
    request_start = tl.sum(tl.where(mine, starts, 0), 0)
    request_end = tl.sum(tl.where(mine, ends, 0), 0)
    absolute_row_start = request_start + (program - tl.sum(tl.where(mine, through - tiles, 0), 0)) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(visible_blocks_ptr + absolute_rows, mask=valid_rows, other=0)
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(1) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr + q_rows[None, :] * stride_q_row + heads[None, :] * stride_q_head + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS) & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(page_table_ptr + request * stride_table_req + logical_page, mask=live, other=0)
        keys = tl.load(
            k_cache_ptr + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (valid_rows[None, :] & (columns[:, None] < visible[None, :])
                      & (columns[:, None] < num_columns))
        tl.store(logits_ptr + rows[None, :] * stride_logits_row + columns[:, None], score, mask=store_mask)


def index_compact_logits(q, k_cache, page_table, query_start_loc, visible_blocks, max_query_len, logits_width,
                         query_offset, num_queries, config=None):
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    chosen = config or (INDEX_COMPACT_CONFIG if k_cache.dtype == torch.float8_e4m3fn else INDEX_COMPACT_WIDE_CONFIG)
    logits = torch.empty((num_queries, logits_width), dtype=torch.float32, device=q.device)
    num_requests = page_table.shape[0]
    tile_r, block_n, k_tiles = chosen["tile_r"], chosen["block_n"], chosen["k_tiles"]
    grid = (triton.cdiv(num_queries, tile_r) + num_requests, triton.cdiv(logits_width, block_n * k_tiles))
    _index_compact_kernel[grid](
        q, k_cache, page_table, query_start_loc, visible_blocks, logits,
        *q.stride()[:-1], *k_cache.stride()[:2], *page_table.stride()[:-1], *logits.stride()[:-1],
        num_queries, query_offset, page_table.shape[1], num_requests,
        PAGE_SIZE=k_cache.shape[1], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2], TILE_R=tile_r, BLOCK_N=block_n,
        K_TILES=k_tiles, STAGES=chosen["stages"], BLOCK_REQ=triton.next_power_of_2(num_requests + 1),
        num_warps=chosen["num_warps"],
    )
    return logits


def index_written_equal(made, want, visible_blocks, query_offset, num_queries):
    seen = (torch.arange(made.shape[1], device=made.device)[None, :]
            < visible_blocks[query_offset:query_offset + num_queries, None])
    return bool(torch.equal(torch.where(seen, made, 0.0), torch.where(seen, want, 0.0)))


def index_compact_aside(reason):
    _INDEX["aside"] = reason[:300]
    selfcheck_fail(INDEX_COMPACT_ENV, _INDEX["aside"])
    stock = _INDEX["stock"]
    if stock is not None:
        import importlib

        importlib.import_module(INDEX_OPS_MODULE)._prefill_logits = stock
    index_say("aside", f"{INDEX_COMPACT_ENV} stood aside: {_INDEX['aside']}; the QSA indexer keeps the stock prefill "
                       f"logits launch")


def index_compact_served(*args, **kwargs):
    stock = _INDEX["stock"]
    if _INDEX["aside"]:
        return stock(*args, **kwargs)
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if capturing or _INDEX["checked"] >= INDEX_COMPACT_CHECKS:
        return index_compact_logits(*args, **kwargs)
    made = index_compact_logits(*args, **kwargs)
    want = stock(*args, **kwargs)
    query_offset = kwargs.get("query_offset", args[7] if len(args) > 7 else 0)
    num_queries = kwargs.get("num_queries", args[8] if len(args) > 8 else made.shape[0])
    visible = args[4] if len(args) > 4 else kwargs["visible_blocks"]
    if not index_written_equal(made, want, visible, query_offset, num_queries):
        index_compact_aside(f"eager call {_INDEX['checked'] + 1} wrote logits that differ from the stock launch")
        return want
    _INDEX["checked"] += 1
    print(f"QSA_INDEX_COMPACT check {_INDEX['checked']}/{INDEX_COMPACT_CHECKS}: rows={num_queries} "
          f"requests={args[2].shape[0]} logits byte-equal to the stock launch", flush=True)
    return made


def index_compact_self_check(stock, device):
    generator = torch.Generator(device="cpu").manual_seed(29)
    rows_per_request = (70, 0, 0, 33, 1, 0, 64)
    requests, page, pages, heads, dim = len(rows_per_request), 64, 6, 4, 128
    total = sum(rows_per_request)
    dtype = torch.float8_e4m3fn if device.type == "cuda" else torch.bfloat16
    host = torch.device("cpu")
    query = (torch.randn((total, heads, dim), generator=generator, device=host) * 0.5).to(device).to(dtype)
    cache = (torch.randn((requests * pages, page, 1, dim), generator=generator, device=host) * 0.5).to(device).to(dtype)
    table = torch.randperm(requests * pages, generator=generator, device=host).view(requests, pages).to(
        torch.int32).to(device)
    offsets = [5]
    for count in rows_per_request:
        offsets.append(offsets[-1] + count)
    starts = torch.tensor(offsets, dtype=torch.int32, device=device)
    visible = torch.randint(1, pages * page + 1, (total,), generator=generator, dtype=torch.int32,
                            device=host).to(device)
    width = pages * page
    for offset, count in ((0, total), (40, 100)):
        want = stock(query, cache, table, starts, visible, max(rows_per_request), width, offset, count)
        made = index_compact_logits(query, cache, table, starts, visible, max(rows_per_request), width, offset, count)
        if not index_written_equal(made, want, visible, offset, count):
            return False
    return True


def install_index_compact(device):
    if _INDEX["installed"] or not index_compact_enabled():
        return False
    import importlib

    module = importlib.import_module(INDEX_OPS_MODULE)
    _INDEX["installed"] = True
    _INDEX["stock"] = module._prefill_logits
    try:
        passed = index_compact_self_check(_INDEX["stock"], torch.device(device))
    except Exception as failure:
        index_compact_aside(f"the startup check raised {type(failure).__name__}: {failure}")
        return False
    if not passed:
        index_compact_aside("the startup check wrote logits that differ from the stock launch")
        return False
    module._prefill_logits = index_compact_served
    chosen = INDEX_COMPACT_CONFIG
    index_say("armed", f"QSA_INDEX_COMPACT armed: the QSA indexer's prefill logits launch one program per row tile "
                       f"that holds prefill rows (row tiles + requests, a fixed grid per bucket) instead of requests x "
                       f"max_query_len/{chosen['tile_r']} tiles, at tile_r={chosen['tile_r']} "
                       f"block_n={chosen['block_n']} stages={chosen['stages']} num_warps={chosen['num_warps']}; the "
                       f"startup check matched the stock launch byte for byte, and the first {INDEX_COMPACT_CHECKS} "
                       f"eager calls run the stock launch beside it and stand aside on any difference "
                       f"({INDEX_COMPACT_ENV})")
    return True


KV4_WRITE_ENV = "TAAF_VLLM_KV4_WRITE_FUSED"
KV4_WRITE_CHECKS = 3
KV4_WRITE_MAX_DIM = 1 << 15
_KV4W = {"on": None, "stock": None, "installed": False, "checked": 0, "aside": "", "said": set()}


def kv4w_say(key, line):
    if key in _KV4W["said"]:
        return
    _KV4W["said"].add(key)
    print(line, flush=True)


def kv4_write_enabled():
    if _KV4W["on"] is None:
        said = str(os.environ.get(KV4_WRITE_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{KV4_WRITE_ENV}={said!r} is not '', '0' or '1'")
        _KV4W["on"] = said == "1"
    return _KV4W["on"]


@triton.jit(do_not_specialize=["rows"])
def _kv4_signs_kernel(key_ptr, value_ptr, signs_ptr, out_ptr, stride_key_row, stride_key_head, stride_value_row,
                      stride_value_head, rows, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    signs = tl.load(signs_ptr + dims)
    key = tl.load(key_ptr + row * stride_key_row + head * stride_key_head + dims).to(tl.float32)
    value = tl.load(value_ptr + row * stride_value_row + head * stride_value_head + dims).to(tl.float32)
    place = (row * HEADS + head) * HEAD_DIM + dims
    tl.store(out_ptr + place, _bf16(key * signs))
    tl.store(out_ptr + rows.to(tl.int64) * HEADS * HEAD_DIM + place, _bf16(value * signs))


@triton.jit
def _kv4_rescale_kernel(buffer_ptr, scale, HEAD_DIM: tl.constexpr):
    place = tl.program_id(0).to(tl.int64) * HEAD_DIM + tl.arange(0, HEAD_DIM)
    tl.store(buffer_ptr + place, _bf16(tl.load(buffer_ptr + place).to(tl.float32) * scale))


def kv4_pinned():
    import importlib

    return importlib.import_module(PINNED_INT4)


def kv4_fusable(key, value):
    head_dim = key.shape[-1]
    return (key.dim() == 3 and key.shape == value.shape and key.dtype == value.dtype == torch.bfloat16
            and key.stride(2) == 1 and value.stride(2) == 1 and head_dim & (head_dim - 1) == 0
            and 16 <= head_dim <= KV4_WRITE_MAX_DIM and kv4_pinned()._hadacore_available())


def kv4_rotate(key, value):
    from vllm import _custom_ops as ops

    rows, heads, head_dim = key.shape
    buffer = torch.empty((2, rows, heads, head_dim), dtype=torch.bfloat16, device=key.device)
    if rows:
        signs = kv4_pinned()._get_rht_signs(head_dim, 0, key.device)
        _kv4_signs_kernel[(rows, heads)](key, value, signs, buffer, key.stride(0), key.stride(1), value.stride(0),
                                         value.stride(1), rows, HEADS=heads, HEAD_DIM=head_dim)
        rotated = ops.hadacore_transform(buffer.view(-1, head_dim), inplace=True)
        if rotated.data_ptr() != buffer.data_ptr():
            buffer = rotated.contiguous().view(2, rows, heads, head_dim)
        _kv4_rescale_kernel[(2 * rows * heads,)](buffer, float(head_dim ** 0.5), HEAD_DIM=head_dim)
    return buffer[0], buffer[1]


def kv4_rotate_stock(key, value):
    single_rht = _rht()
    return single_rht(key.float()).to(key.dtype), single_rht(value.float()).to(value.dtype)


def kv4_same(made, want):
    return made.shape == want.shape and bool(torch.equal(made.contiguous().view(torch.int16),
                                                         want.contiguous().view(torch.int16)))


def kv4_write_rotated(key, value, key_cache, value_cache, slot_mapping, k_scale_cache, v_scale_cache):
    pinned = kv4_pinned()
    pinned._run_reshape_kernel(pinned._reshape_cache_int4_kernel, key=key, value=value, key_cache=key_cache,
                               value_cache=value_cache, k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
                               slot_mapping=slot_mapping, packing_factor=pinned._INT4_PACKING_FACTOR)


def kv4_write_aside(reason, failed=True):
    _KV4W["aside"] = reason[:300]
    if failed:
        selfcheck_fail(KV4_WRITE_ENV, _KV4W["aside"])
    if _KV4W["stock"] is not None:
        kv4_pinned().reshape_and_cache_int4 = _KV4W["stock"]
    kv4w_say("aside", f"{KV4_WRITE_ENV} stood aside: {_KV4W['aside']}; the int4 KV write keeps the stock RHT")


def kv4_write_served(key, value, key_cache, value_cache, slot_mapping, *, k_scale_cache, v_scale_cache):
    stock = _KV4W["stock"]
    if _KV4W["aside"] or not kv4_fusable(key, value):
        return stock(key, value, key_cache, value_cache, slot_mapping, k_scale_cache=k_scale_cache,
                     v_scale_cache=v_scale_cache)
    made_key, made_value = kv4_rotate(key, value)
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if not capturing and _KV4W["checked"] < KV4_WRITE_CHECKS:
        want_key, want_value = kv4_rotate_stock(key, value)
        if kv4_same(made_key, want_key) and kv4_same(made_value, want_value):
            _KV4W["checked"] += 1
            print(f"KV4_WRITE_FUSED check {_KV4W['checked']}/{KV4_WRITE_CHECKS}: rows={key.shape[0]} rotated keys "
                  f"and values byte-equal to the stock RHT", flush=True)
        else:
            kv4_write_aside(f"eager call {_KV4W['checked'] + 1} rotated keys or values that differ from the stock RHT")
            made_key, made_value = want_key, want_value
    kv4_write_rotated(made_key, made_value, key_cache, value_cache, slot_mapping, k_scale_cache, v_scale_cache)


def kv4_check_batch(generator, rows, heads, head_dim, device):
    host = torch.randn((rows, heads + 1, 2 * head_dim), generator=generator, device="cpu")
    packed = (host * 3).to(device).to(torch.bfloat16)
    return packed[:, :heads, :head_dim], packed[:, 1:, head_dim:]


def kv4_gap(made, want):
    differ = made.contiguous().view(torch.int16) != want.contiguous().view(torch.int16)
    gap = (made.float() - want.float()).abs().max().item() if made.numel() else 0.0
    return int(differ.sum().item()), made.numel(), gap


def kv4_write_self_check(device):
    generator = torch.Generator(device="cpu").manual_seed(31)
    for rows, heads, head_dim in ((37, 2, 256), (5, 4, 128), (1, 1, 64)):
        key, value = kv4_check_batch(generator, rows, heads, head_dim, device)
        if not kv4_fusable(key, value):
            return f"rows={rows} heads={heads} dim={head_dim}: the inputs cannot take the fused path"
        made, want = kv4_rotate(key, value), kv4_rotate_stock(key, value)
        for name, one, other in (("keys", made[0], want[0]), ("values", made[1], want[1])):
            if not kv4_same(one, other):
                count, total, gap = kv4_gap(one, other)
                return (f"rows={rows} heads={heads} dim={head_dim}: {name} differ in {count} of {total} values, "
                        f"largest gap {gap:.4g}")
    return ""


def install_kv4_write(device):
    if _KV4W["installed"] or not kv4_write_enabled():
        return False
    module = kv4_pinned()
    _KV4W["installed"] = True
    _KV4W["stock"] = module.reshape_and_cache_int4
    if not module._hadacore_available():
        kv4_write_aside("hadacore is not available here", failed=False)
        return False
    try:
        differs = kv4_write_self_check(torch.device(device))
    except Exception as failure:
        kv4_write_aside(f"the startup check raised {type(failure).__name__}: {failure}")
        return False
    if differs:
        kv4_write_aside(f"the startup check rotated keys or values that differ from the stock RHT ({differs})")
        return False
    module.reshape_and_cache_int4 = kv4_write_served
    kv4w_say("armed", f"KV4_WRITE_FUSED armed: the int4 KV write rotates keys and values with one sign kernel into "
                      f"one buffer, one hadacore call over both and one rescale kernel, in place of six launches "
                      f"per tensor around two hadacore calls, then packs with the stock kernel; the startup check "
                      f"matched the stock RHT byte for byte, and the first {KV4_WRITE_CHECKS} eager calls run the "
                      f"stock RHT beside it and stand aside on any difference ({KV4_WRITE_ENV})")
    return True


INDEX_ROWS_ENV = "TAAF_VLLM_MTP_INDEX_ROWS"
INDEX_ROWS_NEEDS_ENV = "TAAF_VLLM_MTP_ATTN_ROWS"
INDEX_ROWS_CHECKS = 6
INDEX_ROWS_CHECK_ROWS = 1024
_INDEX_ROWS = {"on": None, "stock": None, "installed": False, "checked": 0, "served": 0, "aside": "",
               "said": set()}


def index_rows_say(key, line):
    if key in _INDEX_ROWS["said"]:
        return
    _INDEX_ROWS["said"].add(key)
    print(line, flush=True)


def index_rows_enabled():
    if _INDEX_ROWS["on"] is None:
        said = str(os.environ.get(INDEX_ROWS_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{INDEX_ROWS_ENV}={said!r} is not '', '0' or '1'")
        _INDEX_ROWS["on"] = said == "1"
    return _INDEX_ROWS["on"]


def index_rows_ends(query_start_loc, requests):
    return query_start_loc[1:requests + 1].to(torch.int64) - query_start_loc[0].to(torch.int64)


def index_rows_held(held, query_start_loc, ends, requests, rows):
    local = held.reshape(-1).to(torch.int64) - query_start_loc[0].to(torch.int64)
    inside = (local >= 0) & (local < rows)
    owner = torch.searchsorted(ends, local.clamp(0, rows - 1), right=True)
    slot = torch.where(inside & (owner < requests), owner, torch.full_like(owner, requests))
    kept = torch.full((requests + 1,), -1, dtype=torch.int64, device=ends.device)
    kept.scatter_reduce_(0, slot, torch.where(inside, local, torch.full_like(local, -1)), reduce="amax")
    return kept[:requests]


def index_rows_select(stock, q, k_cache, page_table, query_start_loc, visible_blocks, token_topk, compress_ratio,
                      block_indices, max_seq_len, held):
    requests, rows = page_table.shape[0], q.shape[0]
    ends = index_rows_ends(query_start_loc, requests)
    filled = ends > torch.cat((ends.new_zeros(1), ends[:-1]))
    kept = index_rows_held(held, query_start_loc, ends, requests, rows)
    last = torch.where(kept >= 0, kept, (ends - 1).clamp(0, rows - 1))
    seen = torch.where(filled, visible_blocks.index_select(0, last),
                       torch.zeros_like(last, dtype=visible_blocks.dtype))
    picked = block_indices.new_empty((requests, block_indices.shape[1]))
    starts = torch.arange(requests + 1, dtype=query_start_loc.dtype, device=query_start_loc.device)
    stock(q.index_select(0, last), k_cache, page_table, starts, seen, token_topk, compress_ratio, 1, picked,
          max_seq_len)
    owners = torch.searchsorted(ends, torch.arange(rows, dtype=torch.int64, device=ends.device), right=True)
    block_indices.copy_(picked.index_select(0, owners.clamp_(max=requests - 1)))
    return last, seen


def index_rows_differs(want, made, last, seen):
    seen = seen.to(torch.int64)
    rows = seen > 0
    count = int(rows.sum())
    if not count:
        return 0, ""
    columns = torch.arange(made.shape[1], device=made.device)[None, :] < seen[rows, None]
    one = torch.where(columns, made.index_select(0, last[rows]), 0)
    other = torch.where(columns, want.index_select(0, last[rows]), 0)
    if not torch.equal(one, other):
        return count, f"{int((one != other).any(1).sum())} of {count} gathered rows picked other blocks"
    return count, ""


def index_rows_aside(reason):
    _INDEX_ROWS["aside"] = reason[:300]
    selfcheck_fail(INDEX_ROWS_ENV, _INDEX_ROWS["aside"])
    index_rows_say("aside", f"{INDEX_ROWS_ENV} stood aside: {_INDEX_ROWS['aside']}; the draft's first pass picks "
                            f"blocks for every query row again")


def index_rows_served(q, k_cache, page_table, query_start_loc, visible_blocks, token_topk, compress_ratio,
                      max_query_len, block_indices, max_seq_len):
    stock = _INDEX_ROWS["stock"]
    held = _ROWS["last"]
    requests, rows = page_table.shape[0], q.shape[0]
    if _INDEX_ROWS["aside"] or held is None or not requests or not rows:
        return stock(q, k_cache, page_table, query_start_loc, visible_blocks, token_topk, compress_ratio,
                     max_query_len, block_indices, max_seq_len)
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    checking = not capturing and _INDEX_ROWS["checked"] < INDEX_ROWS_CHECKS and rows <= INDEX_ROWS_CHECK_ROWS
    want = None
    if checking:
        want = torch.empty_like(block_indices)
        stock(q, k_cache, page_table, query_start_loc, visible_blocks, token_topk, compress_ratio, max_query_len,
              want, max_seq_len)
    last, seen = index_rows_select(stock, q, k_cache, page_table, query_start_loc, visible_blocks, token_topk,
                                   compress_ratio, block_indices, max_seq_len, held)
    _INDEX_ROWS["served"] += 1
    width = min(max(64, triton.cdiv(triton.cdiv(int(max_seq_len), compress_ratio), 64) * 64),
                page_table.shape[1] * k_cache.shape[1])
    index_rows_say("live", f"MTP_INDEX_ROWS live: the draft's first pass scored {requests} request rows instead of "
                           f"{rows} query rows, prefill logits {requests * width * 4 / 2 ** 20:.1f} MiB instead of "
                           f"{rows * width * 4 / 2 ** 20:.1f} MiB at width {width}; every row of a request carries "
                           f"the picks of its gathered row ({INDEX_ROWS_ENV})")
    if not checking:
        return None
    compared, why = index_rows_differs(want, block_indices, last, seen)
    if why:
        block_indices.copy_(want)
        index_rows_aside(f"eager call {_INDEX_ROWS['checked'] + 1}: {why}")
        return None
    if not compared:
        return None
    _INDEX_ROWS["checked"] += 1
    print(f"MTP_INDEX_ROWS check {_INDEX_ROWS['checked']}/{INDEX_ROWS_CHECKS}: requests={requests} rows={rows} "
          f"picks of the {compared} scored rows byte-equal to the stock launch", flush=True)
    return None


def install_index_rows():
    if _INDEX_ROWS["installed"] or not index_rows_enabled():
        return False
    if str(os.environ.get(INDEX_ROWS_NEEDS_ENV) or "").strip() != "1":
        raise RuntimeError(f"{INDEX_ROWS_ENV}=1 narrows the draft's picks to the rows {INDEX_ROWS_NEEDS_ENV}=1 "
                           f"gathers, and {INDEX_ROWS_NEEDS_ENV} is off")
    import importlib

    module = importlib.import_module(INDEX_OPS_MODULE)
    _INDEX_ROWS["installed"] = True
    _INDEX_ROWS["stock"] = module.qsa_select_paged_prefill
    module.qsa_select_paged_prefill = index_rows_served
    index_rows_say("armed", f"MTP_INDEX_ROWS armed: the MTP draft's first pass scores and picks QSA blocks only for "
                            f"the row {INDEX_ROWS_NEEDS_ENV} gathers in each request (its last row when it gathers "
                            f"none there) and copies those picks to the request's other rows; the first "
                            f"{INDEX_ROWS_CHECKS} eager calls of up to "
                            f"{INDEX_ROWS_CHECK_ROWS} rows run the stock selection beside it and stand aside on any "
                            f"difference ({INDEX_ROWS_ENV})")
    return True
