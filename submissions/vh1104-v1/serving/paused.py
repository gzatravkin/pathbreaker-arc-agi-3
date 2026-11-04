import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import time
from array import array
from collections import OrderedDict

import numpy as np

if __import__("os").environ.get("TAAF_VLLM_MARKS", "").strip() == "1":
    _marked = __import__("importlib").import_module("vllm.taaf_marks").marked
else:
    def _marked(site):
        return lambda fn: fn

ENV = "TAAF_VLLM_PAUSED"
STOPS_ENV = "TAAF_VLLM_PAUSED_STOP_IDS"
SLACK_ENV = "TAAF_VLLM_PAUSED_REPLY_SLACK"
IDLE_ENV = "TAAF_VLLM_PAUSED_IDLE_S"
MAX_ENV = "TAAF_VLLM_PAUSED_MAX"
KV_HIGH_ENV = "TAAF_VLLM_PAUSED_KV_HIGH"
KV_WAIT_ENV = "TAAF_VLLM_PAUSED_KV_WAIT"
RESUME_ENV = "TAAF_VLLM_PAUSED_RESUME_UNSTOPPED"
MEMORY_ENV = "TAAF_VLLM_PAUSED_PRESSURE_MEMORY_ONLY"
MEMORY_FILE_ENV = "TAAF_VLLM_PAUSED_MEMORY_FILE"
BRANCH_ENV = "TAAF_VLLM_PAUSED_BRANCH"
VIEW_ENV = "TAAF_VLLM_PAUSED_CLIENT_VIEW"
DIFF_ENV = "TAAF_VLLM_PAUSED_DIFF_LOG_S"
SERVER_TOKENS_ENV = "TAAF_VLLM_PAUSED_CLIENT_VIEW_SERVER_TOKENS"
TEXT_GUARD_ENV = "TAAF_VLLM_PAUSED_TEXT_GUARD"
STOP_TAIL_ENV = "TAAF_VLLM_PAUSED_STOP_TAIL"
STOP_TAIL_SCAN = 64
TOOL_GUARD_ENV = "TAAF_VLLM_PAUSED_TEXT_TOOLCALL"
LEAD_GUARD_ENV = "TAAF_VLLM_PAUSED_TEXT_LEADING"
REPLY_OPENING = re.compile(r"(<\|im_start\|>assistant\n)\s+")
TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.S)
TOOL_NAME = re.compile(r"<function=([^>\s]*)>")
TRIM_HOLE_REPLY_ENV = "TAAF_VLLM_TRIM_HOLE_REPLY"
TRIM_HOLE_TEXT_ENV = "TAAF_VLLM_TRIM_HOLE_TEXT"
TRIM_HOLE_TEXT_TRIES = 8
TRIM_HOLE_TEXT_SCAN_ENV = "TAAF_VLLM_TRIM_HOLE_TEXT_SCAN"
TRIM_HOLE_TEXT_SCAN_MAX = 8
TRIM_HOLE_TEXT_TAIL = 48
TEXT_OPEN_ENV = "TAAF_VLLM_PAUSED_TEXT_OPEN"
TRIM_HOLE_IMAGE_EDGE_ENV = "TAAF_VLLM_TRIM_HOLE_IMAGE_EDGE"
TRIM_HOLE_RETRY_ENV = "TAAF_VLLM_TRIM_HOLE_RETRY"
TRIM_HOLE_RETRY_S = 0.05
TRIM_HOLE_RETRY_CODES = ("state", "host_busy")
TRIM_HOLE_REHOLE_ENV = "TAAF_VLLM_TRIM_HOLE_TEXT_REHOLE"
TRIM_HOLE_SYNC_TEXT_ENV = "TAAF_VLLM_TRIM_HOLE_SYNC_TEXT"
SYNC_MARK = "<|im_start|>"
TRIM_HOLE_POSITIONS_ENV = "TAAF_VLLM_TRIM_HOLE_POSITIONS"
CAUSE_LOG_ENV = "TAAF_VLLM_PAUSED_CAUSE_LOG"
POOL_ENV = "TAAF_VLLM_PAUSED_POOL"
POOL_FILE_ENV = "TAAF_VLLM_PAUSED_POOL_FILE"
TRIM_ENV = "TAAF_VLLM_PAUSED_TRIM"
PARK_ENV = "TAAF_VLLM_PAUSED_PARK_PROBE"
PARKING_ENV = "TAAF_VLLM_PARK"
PARKING_WAIT_ENV = "TAAF_VLLM_PARK_WAIT_S"
PARKING_MARK = "taaf_park"
PARKING_WAIT_S = 120.0
PARKING_RETRY_S = 0.25
PARK_PREFIX = "taaf-paused-"
PARK_MIN_TOKENS = 4096
PARK_KINDS = {"HiSparseSourceManager": "host", "HiSparseResidentManager": "resident", "HiSparseHotManager": "hot",
              "MambaManager": "mamba", "CircularBufferManager": "ring", "KpoolTailManager": "ring",
              "FullAttentionManager": "attention"}
POOL_FILE = "taaf_paused_pool"
LEDGER_ENV = "TAAF_VLLM_POOL_LEDGER"
LEDGER_S_ENV = "TAAF_VLLM_POOL_LEDGER_S"
LEDGER_LOG_S_ENV = "TAAF_VLLM_POOL_LEDGER_LOG_S"
LEDGER_S = 2.0
LEDGER_LOG_S = 15.0
LEDGER_HORIZONS = (1024, 2048, 4096, 8192)
LEDGER_FRESH_STEP = 2048
LEDGER_PROBE = "taaf-ledger-probe"
CENSUS_ENV = "TAAF_VLLM_POOL_CENSUS"
CENSUS_SAMPLE = 3
AGE_EDGES = (1.0, 10.0, 60.0)
LEDGER_CONSTANT = ("MambaManager", "CircularBufferManager", "HiSparseHotManager")
LEDGER_SCALARS = ("hot_missing", "reclaim", "hot_blocks", "step_blocks", "max_tokens", "watermark", "running",
                  "paused", "waiting")
POOL_WRITE_S = 1.0
POOL_FRESH_S = 10.0
RELEASE_MARK = "taaf_release"
HOT_SPANS_ENV = "TAAF_VLLM_HOT_SPANS"
HOT_SPANS_MARK = "taaf_hot_spans"
HOT_SPANS_ALIGN = 3328
HOT_SPANS_GAP = 64
HOT_SPANS_TOP = 32
HOT_SPANS_WIDEN = 256
HOT_READS_PICK_TOKENS = 8192
HOT_READS_PICK_QUANTILE = 0.9
HOT_SPANS_IMAGE = "<image>"
PAUSE_MARK = "taaf_pause_now"
ONE_SHOT_MARK = "taaf_one_shot"
PAUSE_STOP = "taaf_pause"
PAUSE_UTILITY = "taaf_pause_now"
THINK_END = "</think>"
RELEASE_S = 5.0
DIFF_SPAN = 10
BRANCH_MARK = "taaf_branch"
BRANCH_SESSION = "taaf-branch"
MEMORY_FILE = "taaf_paused_memory"
MEMORY_WRITE_S = 0.25
MEMORY_REFRESH_S = 1.0
MEMORY_FRESH_S = 5.0
SEATS_ENV = "TAAF_VLLM_PAUSED_SEAT_PRESSURE"
SEATS_SUFFIX = ".seats"
FORK_ENV = "TAAF_VLLM_FORK_BRANCH"
PROBE_TOKENS = 64
PROBE_REPLY = 1
FORK_MAX_ENV = "TAAF_VLLM_FORK_MAX"
FORK_LIVE_ENV = "TAAF_VLLM_FORK_LIVE"
FORK_MAX = 24
FORK_SEATS_ENV = "TAAF_VLLM_FORK_SEATS"
FORK_SEATS_MARK = "taaf_fork_seats"
SEAT_FREE_MARK = "taaf_seat_free"
FORK_MARK = "taaf_fork"
FORK_OF = "taaf_fork_of"
FORK_HOLE = "taaf_fork_hole"
FORK_KEY = "taaf_fork_key"
SESSION_KEY = "taaf_session_key"
FORK_HOLD_ENV = "TAAF_VLLM_FORK_HOLD_S"
FORK_HOLD_S = 20.0
FORK_HOLE_AT = "taaf_fork_hole_at"
MAMBA_HIT_ENV = "TAAF_VLLM_MAMBA_HIT_CHECK"
MAMBA_HIT_MIN_ENV = "TAAF_VLLM_MAMBA_HIT_CHECK_MIN"
MAMBA_HIT_MIN = 8192
Q8_STATE_ENV = "TAAF_VLLM_GDN_STATE_Q8"
FORK_SYNC_ENV = "TAAF_VLLM_FORK_SYNC"
FORK_SYNC_STEPS = 3
FORK_SYNC_SAY = 12
FORK_REPORT_S = 60.0
REWIND_ENV = "TAAF_VLLM_PAUSED_REWIND"
REWIND_MARK = "taaf_rewind"
REWIND_REPORT_S = 60.0
TRIM_HOLE_ENV = "TAAF_VLLM_TRIM_HOLE"
TRIM_HOLE_MAX_ENV = "TAAF_VLLM_TRIM_HOLE_MAX_POS"
TRIM_HOLE_MIN_ENV = "TAAF_VLLM_TRIM_HOLE_MIN"
TRIM_HOLE_ALIGN_ENV = "TAAF_VLLM_TRIM_HOLE_ALIGN"
TRIM_HOLE_KEYS_ENV = "TAAF_VLLM_TRIM_HOLE_KEYS"
TRIM_HOLE_POSITIONS = 262144
TRIM_HOLE_POSITIONS_TABLE = 4 * TRIM_HOLE_POSITIONS
TRIM_HOLE_HEADROOM = 16384
TRIM_HOLE_MIN = 8192
TRIM_HOLE_ALIGN = 3328
TRIM_HOLE_WINDOW = 32
TRIM_HOLE_DROP = ("FullAttentionManager", "HiSparseSourceManager", "SinkFullAttentionManager", "MambaManager",
                  "HiSparseResidentManager")
TRIM_HOLE_RELEASE = ("HiSparseHotManager",)
TRIM_HOLE_RING = ("CircularBufferManager", "KpoolTailManager")
TRIM_HOLE_MARK = "taaf_trim_hole"
TRIM_HOLE_CHECKS_ENV = "TAAF_VLLM_TRIM_HOLE_CHECKS"
TRIM_HOLE_CHECKS = 3
TRIM_HOLE_CHECK_TOKENS = 16
TRIM_HOLE_CHECK_TOP = 5
ROPE_GAP_ENV = "TAAF_VLLM_ROPE_GAP"
ROPE_GAP_XARG = "taaf_rope_gap"
ROPE_GAP_MARK = "taaf_rope_gap_at"
ROPE_GAP_HEAD = 7168
TRIM_HOLE_FAR_CHECKS_ENV = "TAAF_VLLM_TRIM_HOLE_FAR_CHECKS"
TRIM_HOLE_NO_RUN_LOG_ENV = "TAAF_VLLM_TRIM_HOLE_NO_RUN_LOG"
TRIM_HOLE_FAR_GAP_ENV = "TAAF_VLLM_TRIM_HOLE_FAR_GAP"
TRIM_HOLE_FAR_EVERY_ENV = "TAAF_VLLM_TRIM_HOLE_FAR_EVERY"
TRIM_HOLE_FAR_GAP = 300000
TRIM_HOLE_FAR_EVERY = 4
TRAINED_POSITIONS = 262144
TRIM_HOLE_CHECK_PREFIX = 4
TRIM_HOLE_CHECK_SLACK = 0.5
TRIM_HOLE_WORKER_SAY = 8
TRIM_HOLE_HOT_ENV = "TAAF_VLLM_HOST_KV_HOT_RELEASE"
MODEL_ENV = "TAAF_MODEL_PATH"
MARK = "taaf_paused"
DROP_LAST_MARK = "taaf_drop_last"
KEY = "taaf_session"
DEFAULT_SLACK = 512
DEFAULT_IDLE_S = 300.0
DEFAULT_MAX = 120
DEFAULT_KV_HIGH = 0.97
DEFAULT_KV_WAIT = 0.90
TICK_S = 1.0
REPORT_S = 60.0
CLOSED = object()

_mm = {"dropped": 0, "lost": 0}
_state = {"router": None, "stats": {"kv": 0.0, "waiting": 0, "at": 0.0}}
_memory = {"blocked": 0, "peak": 0, "written": None, "at": 0.0}
_seats = {"blocked": 0, "peak": 0, "written": None, "at": 0.0, "said": False}
_pool ={"at": float("-inf"), "failed": None, "said": False}
_park = {"on": None, "armed": None}
_ledger = {"on": None, "at": float("-inf"), "logged": float("-inf"), "fresh": None, "row": None, "failed": None,
           "armed": None, "said": False, "parts": None}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def say(text):
    print(f"PAUSED {text}", flush=True)


def resume_unstopped():
    return os.environ.get(RESUME_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def branch_on():
    return str(os.environ.get(BRANCH_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def probe(tokens, sampling_params):
    reply = getattr(sampling_params, "max_tokens", None)
    return len(tokens) <= PROBE_TOKENS and reply is not None and int(reply) <= PROBE_REPLY


def fork_on():
    return str(os.environ.get(FORK_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def fork_live_on():
    return str(os.environ.get(FORK_LIVE_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def rewind_on():
    return str(os.environ.get(REWIND_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def rewind_say(text):
    print(f"PAUSED_REWIND {text}", flush=True)


def fork_most():
    said = str(os.environ.get(FORK_MAX_ENV) or "").strip()
    return max(1, int(said)) if said else FORK_MAX


def fork_seats_start():
    said = str(os.environ.get(FORK_SEATS_ENV) or "").strip()
    return max(0, int(said)) if said else None


def fork_seats_said(request):
    said = (getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}).get(FORK_SEATS_MARK)
    try:
        return max(0, int(said)) if said is not None else None
    except (TypeError, ValueError):
        return None


def fork_fresh(request):
    return (not fork_is_branch(request) and fork_status(request) == "WAITING"
            and int(getattr(request, "num_computed_tokens", 0) or 0) == 0)


def fork_hold_s():
    said = str(os.environ.get(FORK_HOLD_ENV) or "").strip()
    return max(0.0, float(said)) if said else FORK_HOLD_S


def fork_key(kwargs, marks):
    said = str(kwargs.get("session_id") or marks.get(KEY) or "")
    if not said.startswith(BRANCH_SESSION + "-"):
        return None
    return said[len(BRANCH_SESSION) + 1:].split("#", 1)[0] or None


def client_view_on():
    return str(os.environ.get(VIEW_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def parking_on():
    return str(os.environ.get(PARKING_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def server_tokens_on():
    return str(os.environ.get(SERVER_TOKENS_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def tool_calls_agree(left, right):
    ours, theirs = TOOL_CALL.findall(left), TOOL_CALL.findall(right)
    if not ours or len(ours) != len(theirs):
        return False
    if TOOL_CALL.split(left) != TOOL_CALL.split(right):
        return False
    return all(TOOL_NAME.findall(one) == TOOL_NAME.findall(two) for one, two in zip(ours, theirs))


def text_guard_on():
    return str(os.environ.get(TEXT_GUARD_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def switch_on(env):
    return str(os.environ.get(env) or "").strip().lower() not in ("", "0", "false", "no", "off")


def switch_default_on(env):
    return str(os.environ.get(env) or "1").strip().lower() not in ("0", "false", "no", "off")


def stop_tail_on():
    return str(os.environ.get(STOP_TAIL_ENV) or "1").strip().lower() not in ("0", "false", "no", "off")


def diff_every():
    said = str(os.environ.get(DIFF_ENV) or "").strip()
    return float(said) if said else None


def branched(sampling_params, kwargs):
    extra = getattr(sampling_params, "extra_args", None) or {}
    if extra.get(BRANCH_MARK):
        return True
    return str(kwargs.get("session_id") or "").startswith(BRANCH_SESSION)


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def _eos(value):
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(item) for item in value if isinstance(item, int)}
    return set()


def stop_ids(model_path=None):
    said = str(os.environ.get(STOPS_ENV) or "").strip()
    if said:
        return frozenset(int(part) for part in said.replace(";", ",").split(",") if part.strip())
    import json
    root = model_path or os.environ.get(MODEL_ENV) or ""
    found = set()
    for name in ("generation_config.json", "config.json"):
        try:
            held = json.loads(open(os.path.join(root, name), encoding="utf-8").read())
        except (OSError, ValueError):
            continue
        found |= _eos(held.get("eos_token_id"))
        found |= _eos((held.get("text_config") or {}).get("eos_token_id"))
    return frozenset(found)


_device_stops = {}


def stops_tensor(device, dtype):
    import torch
    if not _TRIM_HOLE["worker"]:
        trim_hole_worker_install()
    if not _MAMBA_HIT["installed"]:
        mamba_hit_install()
    if not _FORK_SYNC["installed"]:
        fork_sync_install()
    key = (str(device), str(dtype))
    held = _device_stops.get(key)
    if held is None:
        values = sorted(stop_ids())
        held = torch.tensor(values, dtype=dtype, device=device) if values else None
        _device_stops[key] = held
        say(f"worker clips draft acceptance at stop ids {values}" if values
            else "worker has no stop ids; draft acceptance is not clipped")
    return held


def _number(env, default, kind=float):
    said = str(os.environ.get(env) or "").strip()
    return kind(said) if said else default


LCP_BLOCK = 2048


def lcp_blocks(left, right, count):
    for start in range(0, count, LCP_BLOCK):
        stop = min(count, start + LCP_BLOCK)
        a = np.asarray(left[start:stop], dtype=np.int64)
        b = np.asarray(right[start:stop], dtype=np.int64)
        found = np.flatnonzero(a != b)
        if found.size:
            return start + int(found[0])
    return count


def token_store(tokens):
    return array("i", tokens) if lean_on() else list(tokens)


def joined(base, appended):
    if lean_on():
        held = array("i", base)
        held.extend(appended)
        return held
    return base + appended


def lcp(left, right):
    count = min(len(left), len(right))
    if count == 0:
        return 0
    if lean_on():
        return lcp_blocks(left, right, count)
    a = np.asarray(left[:count], dtype=np.int64)
    b = np.asarray(right[:count], dtype=np.int64)
    found = np.flatnonzero(a != b)
    return int(found[0]) if found.size else count


def plan(history, reply_start, prompt, stops, slack):
    common = lcp(history, prompt)
    if common >= len(history):
        if len(prompt) > len(history):
            return "extend", list(prompt[len(history):])
        return "miss", []
    if common < reply_start or len(history) - common > slack:
        return "miss", []
    end = next((index for index in range(common, min(len(prompt), common + 4 * slack + 1))
                if prompt[index] in stops), None)
    if end is None:
        return "miss", []
    start = end + 1 if history and history[-1] == prompt[end] else end
    if start >= len(prompt):
        return "miss", []
    return "reply", list(prompt[start:])


def plan_view(view, history, reply_start, prompt, stops, slack):
    kind, delta = plan(history, reply_start, prompt, stops, slack)
    if kind == "extend" or view is None or not history:
        return kind, delta, False
    known = len(view)
    if len(prompt) <= known or lcp(view, prompt) < known:
        return kind, delta, False
    span = len(history) - reply_start
    end = next((index for index in range(known, min(len(prompt), known + span + slack + 1))
                if prompt[index] in stops), None)
    if end is None or abs(end + 1 - known - span) > slack:
        return kind, delta, False
    start = end + 1 if history[-1] == prompt[end] else end
    if start >= len(prompt):
        return kind, delta, False
    return "reply", list(prompt[start:]), kind == "miss"


def plan_exact(history, prompt):
    if len(prompt) > len(history) and lcp(history, prompt) >= len(history):
        return "extend", list(prompt[len(history):])
    return "miss", []


def trim_hole_mode():
    said = str(os.environ.get(TRIM_HOLE_ENV) or "").strip().lower()
    if said in ("", "0"):
        return ""
    if said in ("1", "dry"):
        return said
    raise ValueError(f"{TRIM_HOLE_ENV}={said!r} is not '', '0', '1' or 'dry'")


def trim_hole_keys():
    said = str(os.environ.get(TRIM_HOLE_KEYS_ENV) or "").strip()
    if not said:
        return None
    import re
    try:
        return re.compile(said)
    except re.error as failure:
        raise ValueError(f"{TRIM_HOLE_KEYS_ENV}={said!r} is not a regular expression: {failure}") from None


def trim_hole_checks():
    count = _number(TRIM_HOLE_CHECKS_ENV, TRIM_HOLE_CHECKS, int)
    if count < 0:
        raise ValueError(f"{TRIM_HOLE_CHECKS_ENV}={count} must be zero or more")
    return count


def trim_hole_check_params(params, ranked=False):
    params.temperature = 0.0
    params.max_tokens = TRIM_HOLE_CHECK_TOKENS
    params.min_tokens = 0
    if ranked:
        params.logprobs = TRIM_HOLE_CHECK_TOP
    return params


def trim_hole_value(entry):
    return float(getattr(entry, "logprob", entry))


def trim_hole_top(entry, top=TRIM_HOLE_CHECK_TOP):
    pairs = sorted(dict(entry).items(), key=lambda pair: -trim_hole_value(pair[1]))
    return [token for token, _ in pairs[:top]]


def trim_hole_verdict(picks, theirs, prefix=TRIM_HOLE_CHECK_PREFIX, slack=TRIM_HOLE_CHECK_SLACK):
    their_picks, their_ranks = theirs if theirs else ([], [])
    if not picks or not their_picks or len(their_ranks) < len(their_picks):
        return False, "a side returned no tokens or the recompute returned no logprobs"
    agree = 0
    for one, other in zip(picks, their_picks):
        if one != other:
            break
        agree += 1
    line = f"first token {picks[0]} vs {their_picks[0]}, {agree} of {len(their_picks)} greedy tokens agree"
    if agree >= min(prefix, len(their_picks)):
        return True, line
    if agree >= len(picks):
        return False, f"{line}, the holed side stopped after {len(picks)}"
    ranks = dict(their_ranks[agree])
    pick, want = picks[agree], their_picks[agree]
    near = pick in trim_hole_top(ranks)
    gap = float("inf")
    if pick in ranks and want in ranks:
        gap = trim_hole_value(ranks[want]) - trim_hole_value(ranks[pick])
    return near and gap <= slack, (f"{line}; at the first difference the holed side's token is "
                                   f"{'in' if near else 'not in'} the recompute's top-{TRIM_HOLE_CHECK_TOP}, logprob "
                                   f"gap {gap:.3f} against a slack of {slack}")


def trim_hole_native():
    native = _number(TRIM_HOLE_POSITIONS_ENV, TRIM_HOLE_POSITIONS_TABLE, int)
    if not TRIM_HOLE_POSITIONS <= native <= TRIM_HOLE_POSITIONS_TABLE:
        raise ValueError(f"{TRIM_HOLE_POSITIONS_ENV}={native} must be in [{TRIM_HOLE_POSITIONS}, "
                         f"{TRIM_HOLE_POSITIONS_TABLE}], the rows of the M-RoPE cos/sin table")
    return native


def trim_hole_retries():
    count = _number(TRIM_HOLE_RETRY_ENV, 3, int)
    if not 0 <= count <= 20:
        raise ValueError(f"{TRIM_HOLE_RETRY_ENV}={count} must be in [0, 20]")
    return count


def trim_hole_limits(native=TRIM_HOLE_POSITIONS):
    cap = _number(TRIM_HOLE_MAX_ENV, native - TRIM_HOLE_HEADROOM, int)
    least = _number(TRIM_HOLE_MIN_ENV, TRIM_HOLE_MIN, int)
    align = _number(TRIM_HOLE_ALIGN_ENV, TRIM_HOLE_ALIGN, int)
    if not 0 < cap <= native:
        raise ValueError(f"{TRIM_HOLE_MAX_ENV}={cap} must be in (0, {native}]")
    if align <= 0 or least < align:
        raise ValueError(f"{TRIM_HOLE_ALIGN_ENV}={align} must be positive and {TRIM_HOLE_MIN_ENV}={least} at least it")
    return cap, least, align


class TrimShape:
    reply_shift = 0

    def __init__(self, head, start, run, tail):
        self.head, self.start, self.run, self.tail = int(head), int(start), int(run), int(tail)


class TrimState:
    def __init__(self):
        self.head = self.pad = self.shift = self.holes = self.low = self.anchor = 0
        self.by_text = False
        self.sync = None
        self.sync_end = 0

    def mark(self):
        return [self.low, self.anchor, self.shift]

    def engine_index(self, index):
        if self.sync is not None and self.holes and self.head <= index < self.sync_end:
            return sync_lookup(self.sync, index)
        return index + self.pad if self.holes and index >= self.head else index

    def engine_map(self):
        if not self.holes:
            return [(0, None, 0)]
        ranges = [(0, self.head - 1, 0)] if self.head > 0 else []
        if self.sync is not None:
            ranges += list(self.sync)
            ranges.append((self.sync_end, None, self.pad))
        else:
            ranges.append((self.head, None, self.pad))
        return ranges


def sync_lookup(ranges, index):
    for low, high, delta in ranges:
        if low <= index and (high is None or index <= high):
            return index + delta
    return None


def sync_pairs(ours, theirs, width):
    if not len(ours) or not len(theirs):
        return None
    mine = np.fromiter((width(int(one)) for one in ours), dtype=np.int64, count=len(ours))
    other = np.fromiter((width(int(one)) for one in theirs), dtype=np.int64, count=len(theirs))
    if (mine <= 0).any() or (other <= 0).any():
        return None
    left = np.concatenate(([0], np.cumsum(mine)))
    right = np.concatenate(([0], np.cumsum(other)))
    if left[-1] != right[-1]:
        return None
    shared = np.intersect1d(left, right, assume_unique=True)
    engine = np.searchsorted(left, shared)
    client = np.searchsorted(right, shared)
    deltas = engine - client
    ranges = []
    start = 0
    for at in range(1, len(client) + 1):
        if at == len(client) or deltas[at] != deltas[start] or client[at] != client[at - 1] + 1:
            ranges.append((int(client[start]), int(client[at - 1]), int(deltas[start])))
            start = at
    return ranges


def sync_segment(mine, other, base_mine, base_other):
    ours, theirs = "".join(mine), "".join(other)
    left = np.concatenate(([0], np.cumsum([len(one) for one in mine])))
    right = np.concatenate(([0], np.cumsum([len(one) for one in other])))
    if ours == theirs:
        shared = np.intersect1d(left, right, assume_unique=True)
        return list(zip((np.searchsorted(left, shared) + base_mine).tolist(),
                        (np.searchsorted(right, shared) + base_other).tolist()))
    head = 0
    for a, b in zip(ours, theirs):
        if a != b:
            break
        head += 1
    tail = 0
    for a, b in zip(reversed(ours), reversed(theirs)):
        if a != b or tail >= min(len(ours), len(theirs)) - head:
            break
        tail += 1
    pairs = []
    front = np.intersect1d(left[left <= head], right[right <= head], assume_unique=True)
    pairs += list(zip((np.searchsorted(left, front) + base_mine).tolist(),
                      (np.searchsorted(right, front) + base_other).tolist()))
    left_back, right_back = len(ours) - left, len(theirs) - right
    back = np.intersect1d(left_back[left_back <= tail], right_back[right_back <= tail], assume_unique=True)
    for one in back.tolist():
        pairs.append((int(np.flatnonzero(left_back == one)[0]) + base_mine,
                      int(np.flatnonzero(right_back == one)[0]) + base_other))
    return pairs


def sync_pairs_text(ours, theirs, piece, mark=SYNC_MARK):
    if not len(ours) or not len(theirs):
        return None
    mine = [piece(int(one)) for one in ours]
    other = [piece(int(one)) for one in theirs]
    if not all(mine) or not all(other):
        return None
    cuts_mine = [0] + [at for at, one in enumerate(mine) if one == mark and at] + [len(mine)]
    cuts_other = [0] + [at for at, one in enumerate(other) if one == mark and at] + [len(other)]
    if len(cuts_mine) != len(cuts_other):
        cuts_mine, cuts_other = [0, len(mine)], [0, len(other)]
    pairs = set()
    for (a0, a1), (b0, b1) in zip(zip(cuts_mine, cuts_mine[1:]), zip(cuts_other, cuts_other[1:])):
        pairs.update(sync_segment(mine[a0:a1], other[b0:b1], a0, b0))
    if (0, 0) not in pairs or (len(mine), len(other)) not in pairs:
        return None
    ordered = sorted(pairs, key=lambda one: one[1])
    ranges, start = [], 0
    for at in range(1, len(ordered) + 1):
        if (at == len(ordered) or ordered[at][0] - ordered[at][1] != ordered[start][0] - ordered[start][1]
                or ordered[at][1] != ordered[at - 1][1] + 1):
            ranges.append((ordered[start][1], ordered[at - 1][1], ordered[start][0] - ordered[start][1]))
            start = at
    return ranges


def sync_compose(run_ranges, run_base, held_base, held_map, dropped):
    out = []
    for low, high, delta in run_ranges:
        held_low, held_high = held_base + low + delta, held_base + high + delta
        for mapped_low, mapped_high, mapped in held_map:
            top = held_high if mapped_high is None else min(held_high, mapped_high)
            bottom = max(held_low, mapped_low)
            if bottom > top:
                continue
            client_low = bottom - delta - held_base + run_base
            client_high = top - delta - held_base + run_base
            out.append((client_low, client_high, delta + held_base - run_base + mapped - dropped))
    out.sort()
    return out


class TrimPlan:
    def __init__(self, low, high, head, pad, shift):
        self.low, self.high, self.head, self.pad, self.shift = int(low), int(high), int(head), int(pad), int(shift)

    @property
    def dropped(self):
        return self.high - self.low


def trim_hole_find(history, prompt, window=TRIM_HOLE_WINDOW):
    head = lcp(history, prompt)
    if head >= len(history):
        return None, "extends"
    if head == 0:
        return None, "no_head"
    held = np.asarray(history, dtype=np.int64)
    rest = np.asarray(prompt[head:], dtype=np.int64)
    width = min(window, len(rest), len(held) - head - 1)
    if width <= 0:
        return None, "short"
    found = np.flatnonzero(held[head + 1:len(held) - width + 1] == rest[0]) + head + 1
    for offset in range(1, width):
        if not found.size:
            break
        found = found[held[found + offset] == rest[offset]]
    for start in found.tolist():
        run = len(held) - start
        if run <= len(rest) and np.array_equal(held[start:], rest[:run]):
            if run == len(rest):
                return None, "no_tail"
            return TrimShape(head, start, run, len(rest) - run), None
    return None, "no_run"


def lead_trimmed(text):
    return REPLY_OPENING.sub(r"\1", text.lstrip())


def trim_hole_nearest(history, prompt, window=TRIM_HOLE_WINDOW):
    head = lcp(history, prompt)
    held = np.asarray(history, dtype=np.int64)
    rest = np.asarray(prompt[head:head + window], dtype=np.int64)
    if not len(rest) or head + 1 >= len(held):
        return head, len(rest), None, 0
    found = np.flatnonzero(held[head + 1:] == rest[0]) + head + 1
    best, agreed = None, 0
    for start in found.tolist():
        size = min(len(rest), len(held) - start)
        same = held[start:start + size] == rest[:size]
        count = int(np.argmin(same)) if not same.all() else size
        if count > agreed:
            best, agreed = start, count
    return head, len(rest), best, agreed


def trim_hole_find_reply(history, prompt, reply_start, stops, same_text, slack, window=TRIM_HOLE_WINDOW):
    held = np.asarray(history, dtype=np.int64)
    head = lcp(history, prompt)
    if head == 0 or head >= reply_start:
        return None, "reply_head"
    rest = np.asarray(prompt[head:], dtype=np.int64)
    reply = held[reply_start:]
    width = min(window, len(rest), reply_start - head - 1)
    if width <= 0 or not len(reply):
        return None, "reply_short"
    found = np.flatnonzero(held[head + 1:reply_start - width + 1] == rest[0]) + head + 1
    for offset in range(1, width):
        if not found.size:
            break
        found = found[held[found + offset] == rest[offset]]
    for start in found.tolist():
        run = reply_start - start
        if run > len(rest) or not np.array_equal(held[start:reply_start], rest[:run]):
            continue
        end = next((index for index in range(run, min(len(rest), run + len(reply) + slack + 1))
                    if int(rest[index]) in stops), None)
        if end is None:
            return None, "reply_stop"
        if not same_text(reply, rest[run:end + 1]):
            return None, "reply_text"
        tail = len(rest) - (end + 1)
        if tail <= 0:
            return None, "no_tail"
        shape = TrimShape(head, start, len(held) - start, tail)
        shape.reply_shift = len(reply) - (end + 1 - run)
        return shape, None
    return None, "no_run"


def trim_hole_find_text(history, prompt, stops, same_text, slack, window=TRIM_HOLE_WINDOW,
                        tries=TRIM_HOLE_TEXT_TRIES, scan=False, decode=None, open_split=None):
    held = np.asarray(history, dtype=np.int64)
    head = lcp(history, prompt)
    if head == 0 or head >= len(history):
        return None, "text_head"
    opened = not len(held) or int(held[-1]) not in stops
    if opened and (open_split is None or not len(held)):
        return None, "text_open"
    rest = np.asarray(prompt[head:], dtype=np.int64)
    width = min(window, len(rest), len(held) - head - 1)
    if width <= 0:
        return None, "text_short"
    found = np.flatnonzero(held[head + 1:len(held) - width + 1] == rest[0]) + head + 1
    for offset in range(1, width):
        if not found.size:
            break
        found = found[held[found + offset] == rest[offset]]
    if scan or opened:
        return trim_hole_scan_text(held, rest, head, found.tolist(), stops, same_text, slack, width, tries,
                                   decode if scan else None, open_split if opened else None,
                                   TRIM_HOLE_TEXT_SCAN_MAX if scan else 1)
    for start in found.tolist():
        run = len(held) - start
        spread = max(slack, run // 10)
        ends = [index for index in range(max(width, run - spread), min(len(rest), run + spread + 1))
                if int(rest[index]) in stops]
        for end in sorted(ends, key=lambda index: abs(index + 1 - run))[:tries]:
            if not same_text(held[start:], rest[:end + 1]):
                continue
            tail = len(rest) - (end + 1)
            if tail <= 0:
                return None, "no_tail"
            shape = TrimShape(head, start, run, tail)
            shape.reply_shift = run - (end + 1)
            return shape, None
        return None, "text_differs"
    return None, "no_run"


def trim_hole_scan_text(held, rest, head, found, stops, same_text, slack, width, tries, decode, open_split,
                        most):
    def agreed(start):
        size = min(len(held) - start, len(rest))
        if size <= 0:
            return 0
        same = held[start:start + size] == rest[:size]
        return int(np.argmin(same)) if not same.all() else size

    order = sorted(found, key=lambda start: (-agreed(start), -start))[:most]
    mine_tail = None
    if decode is not None and open_split is None:
        mine_tail = decode(held[-TRIM_HOLE_TEXT_TAIL:].tolist())
    for start in order:
        run = len(held) - start
        spread = max(slack, run // 10)
        ends = [index for index in range(max(width, run - spread), min(len(rest), run + spread + 1))
                if int(rest[index]) in stops]
        checked = 0
        for end in sorted(ends, key=lambda index: abs(index + 1 - run)):
            if checked >= tries:
                break
            if mine_tail is not None:
                theirs_tail = decode(rest[max(0, end + 1 - TRIM_HOLE_TEXT_TAIL):end + 1].tolist())
                if not (mine_tail.endswith(theirs_tail) or theirs_tail.endswith(mine_tail)):
                    continue
            checked += 1
            if open_split is not None:
                at = open_split(held[start:], rest[:end + 1])
                if at is None:
                    continue
            else:
                if not same_text(held[start:], rest[:end + 1]):
                    continue
                at = end + 1
            tail = len(rest) - at
            if tail <= 0:
                return None, "no_tail"
            shape = TrimShape(head, start, run, tail)
            shape.reply_shift = run - at
            return shape, None
    return (None, "text_differs") if found else (None, "no_run")


def open_split(ours, theirs, decode):
    mine = decode(list(ours))
    whole = decode(list(theirs))
    if len(whole) <= len(mine) or not whole.startswith(mine):
        return None
    rest = whole[len(mine):]
    for at in range(len(theirs) - 1, -1, -1):
        piece = decode(list(theirs[at:]))
        if len(piece) > len(rest):
            return None
        if len(piece) == len(rest) and piece == rest and decode(list(theirs[:at])) == mine:
            return at
    return None


def trim_hole_edges(hole, images, start, limits):
    _, least, align = limits
    low, high = hole.low, hole.high
    moved = True
    while moved and low < high:
        moved = False
        for offset, length in images:
            end = offset + length
            if offset < low < end:
                low, moved = -(-end // align) * align, True
            if offset < high < end:
                high, moved = (offset // align) * align, True
    if (low, high) == (hole.low, hole.high):
        return hole, None
    if high - low < least:
        return None, "image_small"
    shift = hole.shift - hole.dropped + (high - low)
    return TrimPlan(low, high, hole.head, (low - hole.head) + (start - high), shift), None


def trim_hole_image_check(held_images, prompt_images, head, start, boundary, held_length):
    unhashed = False
    held_head = [image for image in held_images if image[0] < head]
    wanted_head = [image for image in prompt_images if image[0] < head]
    if any(image[0] + image[1] > head for image in wanted_head + held_head):
        return "image_head", unhashed
    if len(held_head) != len(wanted_head):
        return "image_head", unhashed
    for one, other in zip(held_head, wanted_head):
        if one[:3] != other[:3]:
            return "image_head", unhashed
        if one[3] is None or other[3] is None:
            unhashed = True
        elif one[3] != other[3]:
            return "image_head", unhashed
    held_run = [image for image in held_images if start <= image[0] < held_length]
    wanted_run = [image for image in prompt_images if head <= image[0] < boundary]
    if len(held_run) != len(wanted_run):
        return "image_run", unhashed
    for one, other in zip(held_run, wanted_run):
        if one[1:3] != other[1:3]:
            return "image_run", unhashed
        if one[3] is None or other[3] is None:
            unhashed = True
        elif one[3] != other[3]:
            return "image_run", unhashed
    return None, unhashed


def trim_hole_rehead(shape, history, head):
    extra = shape.head - head
    if extra <= 0 or shape.start - extra <= head:
        return shape
    if list(history[head:shape.head]) != list(history[shape.start - extra:shape.start]):
        return shape
    return TrimShape(head, shape.start - extra, shape.run + extra, shape.tail)


def trim_hole_plan(shape, state, engine_len, limits, native=TRIM_HOLE_POSITIONS):
    cap, least, align = limits
    if state.holes and shape.head != state.head:
        return None, "head_moved"
    start = state.engine_index(shape.start)
    if start is None:
        return None, "text_sync"
    low = -(-shape.head // align) * align
    high = (start // align) * align
    if high - low < least:
        return None, "small"
    shift = state.shift + high - low
    end = engine_len + state.shift
    if end + (high - low) > cap:
        return None, "positions"
    if end + shape.tail > native:
        return None, "native"
    return TrimPlan(low, high, shape.head, (low - shape.head) + (start - high), shift), None


def trim_hole_blocks(blocks, low, high, block_size):
    if low % block_size or high % block_size or not 0 <= low <= high:
        raise ValueError(f"hole [{low}, {high}) is not aligned to blocks of {block_size}")
    first, last = low // block_size, high // block_size
    return list(blocks[:first]) + list(blocks[last:]), list(blocks[first:last])


def images_of(prompt, shift=0):
    if not isinstance(prompt, dict) or prompt.get("type", "token") != "multimodal":
        return []
    places = prompt.get("mm_placeholders") or {}
    hashes = prompt.get("mm_hashes") or {}
    found = []
    for modality, ranges in places.items():
        keys = list(hashes.get(modality) or [])
        for index, place in enumerate(ranges):
            key = keys[index] if index < len(keys) else None
            found.append((place.offset + shift, place.length, modality, key))
    return sorted(found, key=lambda item: (item[0], item[2]))


def images_near(head, held, wanted, reach=4096):
    def shown(images):
        near = sorted(image for image in images if image[0] + image[1] >= head - reach and image[0] <= head + reach)
        return ", ".join(f"{offset}+{length}" + (f" {str(key)[:8]}" if key else "") for offset, length, _, key in near)
    if not held and not wanted:
        return ""
    return f"; server images near the head: [{shown(held)}]; client images near the head: [{shown(wanted)}]"


def images_agree(held, wanted, limit):
    inside = []
    for item in wanted:
        if item[0] < limit < item[0] + item[1]:
            return False
        if item[0] + item[1] <= limit:
            inside.append(item)
    if len(inside) != len(held):
        return False
    return all(one == other and one[3] is not None for one, other in zip(held, inside))


def first_key(tokens, length):
    view = np.asarray(tokens[:length], dtype=np.int64)
    return length, hashlib.blake2b(view.tobytes(), digest_size=16).hexdigest()


@_marked("paused_owner")
def clip_at_stops(num_sampled, inputs, cu_num_logits, stops):
    import torch
    if stops is None or stops.numel() == 0 or inputs.numel() == 0:
        return num_sampled
    total = inputs.shape[0]
    count = num_sampled.shape[0]
    position = torch.arange(total, device=inputs.device)
    owner = torch.searchsorted(cu_num_logits[1:count + 1].to(position.dtype), position, right=True)
    owner = owner.clamp(max=count - 1)
    offset = position - cu_num_logits[:count].to(position.dtype)[owner]
    hit = torch.isin(inputs, stops)
    far = torch.full((count,), total + 1, dtype=position.dtype, device=inputs.device)
    first = far.scatter_reduce(0, owner, torch.where(hit, offset, total + 1), reduce="amin",
                               include_self=True)
    limit = torch.where(first == 0, torch.ones_like(first), first).to(num_sampled.dtype)
    return torch.minimum(num_sampled, limit)


def clip_at_length(num_sampled, positions, cu_num_logits, max_seq_len, idx_mapping):
    import torch
    count = num_sampled.shape[0]
    if count == 0 or positions.numel() == 0:
        return num_sampled
    first = positions[cu_num_logits[:count].long().clamp(max=positions.numel() - 1)].long()
    cap = torch.as_tensor(np.asarray(max_seq_len)[np.asarray(idx_mapping)[:count]].astype(np.int64))
    cap = cap.to(first.device, non_blocking=True)
    limit = (cap - first - 1).clamp(min=1).to(num_sampled.dtype)
    return torch.minimum(num_sampled, limit)


_scheduler_stops = {}


def settle(request):
    inflight = int(getattr(request, "num_in_flight_tokens", 0) or 0)
    if inflight <= 0:
        return True
    held = _scheduler_stops.get("ids")
    if held is None:
        held = _scheduler_stops["ids"] = stop_ids()
    output = request.output_token_ids
    stopped = bool(output) and output[-1] in held
    request.num_computed_tokens = len(request._all_token_ids)
    request.num_output_placeholders = 0
    request.num_stale_output_tokens = inflight
    request.drop_stale_output = True
    request.spec_token_ids = []
    return stopped


def refeed(session, update_tokens, drop_last=False):
    computed = min(session.num_computed_tokens, len(session._all_token_ids))
    dropped = list(session._all_token_ids[computed:])
    if drop_last and dropped:
        dropped = dropped[:-1]
    session.num_computed_tokens = computed
    return dropped + list(update_tokens or ())


def cut(prompt, start):
    tokens = list(prompt["prompt_token_ids"][start:])
    if prompt.get("type", "token") != "multimodal":
        return tokens
    from vllm.multimodal.inputs import PlaceholderRange
    places = prompt.get("mm_placeholders") or {}
    kwargs = prompt.get("mm_kwargs") or {}
    hashes = prompt.get("mm_hashes") or {}
    kept_places, kept_kwargs, kept_hashes = {}, {}, {}
    for modality, ranges in places.items():
        items = list(kwargs.get(modality) or [])
        keys = list(hashes.get(modality) or [])
        for index, place in enumerate(ranges):
            if place.offset + place.length <= start:
                continue
            if place.offset < start:
                return None
            moved = PlaceholderRange(offset=place.offset - start, length=place.length,
                                     **{name: getattr(place, name) for name in ("is_embed",)
                                        if getattr(place, name, None) is not None})
            kept_places.setdefault(modality, []).append(moved)
            if index < len(items):
                kept_kwargs.setdefault(modality, []).append(items[index])
            if index < len(keys):
                kept_hashes.setdefault(modality, []).append(keys[index])
    if not kept_places:
        return tokens
    container = type(kwargs) if kwargs else dict
    chunk = {"type": "multimodal", "prompt_token_ids": tokens, "mm_kwargs": container(kept_kwargs),
             "mm_hashes": kept_hashes, "mm_placeholders": kept_places}
    if prompt.get("cache_salt") is not None:
        chunk["cache_salt"] = prompt["cache_salt"]
    return chunk


def marked(sampling_params):
    extra = getattr(sampling_params, "extra_args", None) or {}
    return bool(extra.get(MARK))


def drop_image_data(prompt):
    if not isinstance(prompt, dict) or prompt.get("type", "token") != "multimodal":
        return 0
    held = prompt.get("mm_kwargs")
    if not held:
        return 0
    count = sum(len(items or ()) for items in held.values())
    held.clear()
    return count


async def feed(inputs, dropped):
    from vllm.engine.protocol import StreamingInput
    from vllm.inputs import TokensPrompt
    while True:
        held = await inputs.get()
        if held is None:
            return
        chunk, chunk_params = held
        held = None
        if not isinstance(chunk, dict):
            chunk = TokensPrompt(prompt_token_ids=list(chunk))
        sent = StreamingInput(prompt=chunk, sampling_params=chunk_params)
        chunk = None
        yield sent
        dropped(sent.prompt)
        sent = None


def strip_heavy(data, limit=4096):
    try:
        heavy = [key for key, elem in data.items()
                 if int(getattr(getattr(elem, "data", elem), "numel", lambda: 0)()) > limit]
        if not heavy:
            return None
        kept = copy.copy(data)
        for key in heavy:
            del kept[key]
        return kept
    except Exception:
        return None


def drop_covered(session):
    computed = int(getattr(session, "num_computed_tokens", 0) or 0)
    dropped = 0
    for feature in getattr(session, "mm_features", None) or ():
        if feature.data is None or getattr(feature, "_taaf_stripped", False):
            continue
        place = feature.mm_position
        if place.offset + place.length > computed:
            continue
        stripped = strip_heavy(feature.data)
        if stripped is None:
            continue
        feature.data = stripped
        feature._taaf_stripped = True
        dropped += 1
    if dropped:
        session._taaf_mm_dropped = True
        _mm["dropped"] += dropped
    return dropped


def mm_lost(scheduler, request):
    held = scheduler.__dict__.setdefault("_taaf_mm_lost", set())
    if request.request_id not in held:
        held.add(request.request_id)
        _mm["lost"] += 1
        print(f"PAUSED mm drop: {request.request_id} needs an image whose data was let go after its tokens were "
              f"computed (a recompute from scratch outside the prefix cache); the request ends with an error so "
              f"the caller can send it again whole ({_mm['dropped']} features let go so far, {_mm['lost']} "
              f"requests lost)", flush=True)
    return True


def take_lost(scheduler):
    held = scheduler.__dict__.get("_taaf_mm_lost")
    if not held:
        return set()
    taken = set(held)
    held.clear()
    return taken


SHARED_IDS_MAX = 1 << 20
_ints = []
_shared = {"engine": False, "api": False, "ids": 0}


def shared_int(value):
    if value < 0 or value >= SHARED_IDS_MAX:
        return value
    table = _ints
    if value >= len(table):
        table.extend(range(len(table), int(value) + 1))
    return table[value]


def shared_ids(ids):
    if not isinstance(ids, list) or not ids:
        return ids
    top = max(ids)
    table = _ints
    if top >= len(table):
        table.extend(range(len(table), int(min(top, SHARED_IDS_MAX - 1)) + 1))
    limit = len(table)
    ids[:] = [table[value] if 0 <= value < limit else value for value in ids]
    _shared["ids"] += len(ids)
    return ids


def lean_engine():
    trim_hole_attach()
    pause_attach()
    hot_spans_attach()
    if _shared["engine"] or not lean_on():
        return False
    from vllm.v1.request import Request
    build = Request.from_engine_core_request.__func__
    append = Request.append_output_token_ids

    def from_engine_core_request(cls, request, block_hasher):
        shared_ids(getattr(request, "prompt_token_ids", None))
        return build(cls, request, block_hasher)

    def append_output_token_ids(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = shared_int(token_ids)
        else:
            shared_ids(token_ids)
        return append(self, token_ids)

    Request.from_engine_core_request = classmethod(from_engine_core_request)
    Request.append_output_token_ids = append_output_token_ids
    _shared["engine"] = True
    say(f"lean token ids live in the engine (pid {os.getpid()}): every request's prompt, streamed chunk and sampled "
        f"token refers to one shared int per vocabulary id, so a session's token history costs a pointer a token "
        f"in each list instead of a new int object")
    return True


def lean_api():
    if _shared["api"] or not lean_on():
        return False
    from vllm.v1.engine import output_processor
    owner = output_processor.OutputProcessor
    add = owner.add_request
    process = owner.process_outputs

    def add_request(self, request, *args, **kwargs):
        shared_ids(getattr(request, "prompt_token_ids", None))
        return add(self, request, *args, **kwargs)

    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        for output in engine_core_outputs or ():
            shared_ids(getattr(output, "new_token_ids", None))
        return process(self, engine_core_outputs, *args, **kwargs)

    owner.add_request = add_request
    owner.process_outputs = process_outputs
    _shared["api"] = True
    say(f"lean token ids live in the API server (pid {os.getpid()}): request states and detokenizers keep one "
        f"shared int per vocabulary id")
    return True


MEMTRACE_ENV = "TAAF_VLLM_MEMTRACE"
MEMTRACE_S_ENV = "TAAF_VLLM_MEMTRACE_S"
MEMTRACE_FRAMES_ENV = "TAAF_VLLM_MEMTRACE_FRAMES"
MEMTRACE_ALLOC_ENV = "TAAF_VLLM_MEMTRACE_TRACEMALLOC"
MEMTRACE_TYPES_ENV = "TAAF_VLLM_MEMTRACE_TYPES"
MEMTRACE_DEEP_ENV = "TAAF_VLLM_MEMTRACE_DEEP"
LEAN_ENV = "TAAF_VLLM_PAUSED_LEAN"
CENSUS_TOP = 20
CENSUS_NODES = 300000
CENSUS_DEPTH = 6
SMAPS_BUCKETS = ((128 << 10, "<128K"), (1 << 20, "<1M"), (4 << 20, "<4M"), (64 << 20, "<64M"), (None, ">=64M"))
MEMTRACE_DEFAULT_S = 300.0
MEMTRACE_TOP = 15
MEMTRACE_TYPE_TOP = 15
_trace = {"at": None, "begun": None, "snap": None, "types": None, "said": set(), "ticks": 0}


def _truthy(name, default=False):
    held = str(os.environ.get(name) or "").strip().lower()
    if not held:
        return default
    return held not in ("0", "false", "no", "off")


def memtrace_on():
    return _truthy(MEMTRACE_ENV)


def memtrace_deep():
    return _truthy(MEMTRACE_DEEP_ENV)


def lean_on():
    return _truthy(LEAN_ENV)


def malloc_reading():
    try:
        import ctypes

        class Info(ctypes.Structure):
            _fields_ = [(name, ctypes.c_size_t) for name in (
                "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks",
                "keepcost")]

        call = ctypes.CDLL("libc.so.6").mallinfo2
        call.restype = Info
        info = call()
    except (OSError, AttributeError):
        return None
    return {name: getattr(info, name) for name in ("arena", "uordblks", "fordblks", "hblkhd", "hblks", "ordblks",
                                                   "keepcost")}


def smaps_histogram(path="/proc/self/smaps"):
    buckets = {label: [0, 0, 0] for _, label in SMAPS_BUCKETS}
    biggest = []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    head = None
    size = anon = 0

    def close():
        if head is None or not anon:
            return
        for limit, label in SMAPS_BUCKETS:
            if limit is None or size < limit:
                row = buckets[label]
                row[0] += 1
                row[1] += size
                row[2] += anon
                break
        biggest.append((anon, size, head))

    for line in lines:
        fields = line.split()
        if len(fields) >= 5 and "-" in fields[0] and not fields[0].endswith(":"):
            close()
            lo, hi = fields[0].split("-")
            named = len(fields) > 5 and fields[5] != "[heap]"
            head = None if named else fields[0]
            size = int(hi, 16) - int(lo, 16)
            anon = 0
        elif line.startswith("Anonymous:") and head is not None:
            anon = int(fields[1]) * 1024
    close()
    biggest.sort(reverse=True)
    return buckets, biggest[:3]


def session_census(router, top=CENSUS_TOP, nodes=CENSUS_NODES, depth=CENSUS_DEPTH):
    import array
    import collections
    seen = set()
    found = []
    per_name = {}
    visited = 0
    stack = [(getattr(session, name), name, 0) for session in list(router.sessions.values())
             for name in vars(session)]
    while stack and visited < nodes:
        item, name, level = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        visited += 1
        kind = type(item).__name__
        if isinstance(item, (bytes, bytearray, str)):
            size = len(item)
        elif isinstance(item, np.ndarray):
            size = int(item.nbytes)
        elif isinstance(item, array.array):
            size = item.buffer_info()[1] * item.itemsize
        elif isinstance(item, list) and item and isinstance(item[0], int):
            size = len(item) * 36
            kind = "list[int]"
        else:
            size = 0
        if size:
            found.append((size, name, kind))
            per_name[name] = per_name.get(name, 0) + size
            continue
        if level >= depth:
            continue
        if isinstance(item, dict):
            stack.extend((value, name, level + 1) for value in list(item.values()))
        elif isinstance(item, (list, tuple, set, frozenset, collections.deque)):
            stack.extend((value, name, level + 1) for value in list(item))
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            stack.extend((value, key, level + 1) for key, value in list(vars(item).items()))
    found.sort(reverse=True)
    return {"top": found[:top], "per_name": sorted(per_name.items(), key=lambda row: -row[1])[:top],
            "visited": visited, "truncated": bool(stack)}


def deep_report(label, census_of=None):
    info = malloc_reading()
    if info is not None:
        print(f"MEMTRACE {label} mallinfo " + " ".join(f"{name}={value}" for name, value in info.items()), flush=True)
    else:
        print(f"MEMTRACE {label} mallinfo unavailable", flush=True)
    held = smaps_histogram()
    if held is not None:
        buckets, biggest = held
        print(f"MEMTRACE {label} smaps_anon " + " ".join(
            f"{name}:n={row[0]},map={row[1] / 2 ** 20:.1f}MiB,anon={row[2] / 2 ** 20:.1f}MiB"
            for name, row in buckets.items()), flush=True)
        print(f"MEMTRACE {label} smaps_biggest " + " ".join(
            f"{anon / 2 ** 20:.1f}MiB/{size / 2 ** 20:.1f}MiB@{head}" for anon, size, head in biggest), flush=True)
    if census_of is not None:
        started = time.monotonic()
        census = census_of()
        print(f"MEMTRACE {label} census visited={census['visited']} truncated={int(census['truncated'])} "
              f"took={time.monotonic() - started:.2f}s by_attr " + " ".join(
                  f"{name}={size / 2 ** 20:.1f}MiB" for name, size in census["per_name"]), flush=True)
        print(f"MEMTRACE {label} census_top " + " ".join(
            f"{size}B:{kind}@{name}" for size, name, kind in census["top"]), flush=True)


def memtrace_every():
    return max(1.0, _number(MEMTRACE_S_ENV, MEMTRACE_DEFAULT_S))


def length_of(owner, *names):
    held = owner
    for name in names:
        held = getattr(held, name, None)
        if held is None:
            return None
    try:
        return len(held)
    except TypeError:
        inner = getattr(held, "_cache", None)
        return len(inner) if inner is not None and hasattr(inner, "__len__") else None


def item_bytes(item):
    total = 0
    try:
        elements = list(item.values())
    except Exception:
        return 0
    for element in elements:
        data = getattr(element, "data", element)
        size = getattr(data, "nbytes", None)
        if size is None:
            try:
                size = data.numel() * data.element_size()
            except Exception:
                size = len(data) if isinstance(data, (bytes, bytearray, memoryview)) else 0
        total += int(size or 0)
    return total


def feature_census(requests):
    census = {"requests": 0, "features": 0, "with_data": 0, "data_mib": 0.0}
    held = 0
    for request in list(requests):
        census["requests"] += 1
        for feature in getattr(request, "mm_features", None) or ():
            census["features"] += 1
            data = getattr(feature, "data", None)
            if data is not None:
                census["with_data"] += 1
                held += item_bytes(data)
    census["data_mib"] = round(held / 2 ** 20, 1)
    return census


def engine_readings(scheduler):
    rows = {}
    for name, path in (("requests", ("requests",)), ("waiting", ("waiting",)), ("running", ("running",)),
                       ("skipped_waiting", ("skipped_waiting",)), ("finished_ids", ("finished_req_ids",)),
                       ("hash_to_block", ("kv_cache_manager", "block_pool", "cached_block_hash_to_block")),
                       ("encoder_cached", ("encoder_cache_manager", "cached")),
                       ("lost", ("_taaf_mm_lost",))):
        size = length_of(scheduler, *path)
        if size is not None:
            rows[name] = size
    try:
        rows.update(feature_census(scheduler.requests.values()))
    except Exception as failure:
        rows["census_error"] = type(failure).__name__
    managers = getattr(getattr(getattr(scheduler, "kv_cache_manager", None), "coordinator", None),
                       "single_type_managers", None) or ()
    held = [length_of(manager, "req_to_blocks") for manager in managers]
    if held:
        rows["req_to_blocks"] = sum(count or 0 for count in held)
    rows.update(mm_dropped=_mm["dropped"], mm_lost=_mm["lost"], shared_ids=_shared["ids"])
    return rows


def api_readings(router):
    engine = router.engine
    rows = {"sessions": len(router.sessions), "by_first": len(router.by_first), "by_key": len(router.by_key),
            "released": len(router.released), "parked": len(router.parked),
            "parking_now": len(router.parking_now), "turns": router.tally.get("turns", 0),
            "tokens_held": router.held()}
    for name, path in (("output_states", ("output_processor", "request_states")),
                       ("in_flight", ("engine_core", "reqs_in_flight")),
                       ("utility_results", ("engine_core", "utility_results")),
                       ("outputs_queued", ("engine_core", "outputs_queue"))):
        size = length_of(engine, *path)
        if size is not None:
            rows[name] = size
    cache = mm_cache_reading(engine)
    if cache is not None:
        rows["mm_cache_items"] = cache["items"]
        rows["mm_cache_mib"] = round(cache["bytes"] / 2 ** 20, 1)
    rows["shared_ids"] = _shared["ids"]
    return rows


def _type_census():
    import gc
    counts = {}
    for item in gc.get_objects():
        name = type(item).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _site(stat):
    frame = stat.traceback[0]
    return f"{os.path.basename(frame.filename)}:{frame.lineno}"


def memtrace_tick(label, readings, force=False, census=None):
    if not memtrace_on():
        return False
    now = time.monotonic()
    pid = os.getpid()
    if (label, pid) not in _trace["said"]:
        _trace["said"].add((label, pid))
        frames = int(_number(MEMTRACE_FRAMES_ENV, 1, int))
        print(f"MEMTRACE armed {label} pid={pid}: every {memtrace_every():.0f} s, the {MEMTRACE_TOP} largest "
              f"allocation-site growths ({frames} frame{'s' if frames != 1 else ''}, {MEMTRACE_ALLOC_ENV}="
              f"{int(_truthy(MEMTRACE_ALLOC_ENV, False))}) and the sizes of the per-request containers",
              flush=True)
    key = (label, pid)
    state = _trace.setdefault(key, {"at": None, "begun": now, "snap": None, "types": None})
    if state["at"] is not None and not force and now - state["at"] < memtrace_every():
        return False
    import tracemalloc
    allocations = _truthy(MEMTRACE_ALLOC_ENV, False)
    if allocations and not tracemalloc.is_tracing():
        tracemalloc.start(max(1, int(_number(MEMTRACE_FRAMES_ENV, 1, int))))
    first = state["at"] is None
    state["at"] = now
    memory = process_memory() or {}
    parts = [f"MEMTRACE {label} pid={pid} up={now - state['begun']:.0f}s",
             f"rss={gib(memory.get('VmRSS', 0))}GiB anon={gib(memory.get('RssAnon', 0))}GiB "
             f"hwm={gib(memory.get('VmHWM', 0))}GiB"]
    try:
        import sys
        parts.append(f"py_blocks={sys.getallocatedblocks()}")
    except Exception:
        pass
    if allocations and tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        parts.append(f"traced={current / 2 ** 20:.1f}MiB peak={peak / 2 ** 20:.1f}MiB")
    print(" ".join(parts), flush=True)
    try:
        rows = readings()
    except Exception as failure:
        rows = {"readings_error": f"{type(failure).__name__}: {failure}"}
    print(f"MEMTRACE {label} sizes " + " ".join(f"{name}={value}" for name, value in rows.items()), flush=True)
    if allocations and tracemalloc.is_tracing():
        snap = tracemalloc.take_snapshot().filter_traces((tracemalloc.Filter(False, tracemalloc.__file__),))
        before = state["snap"]
        state["snap"] = snap
        if before is None:
            print(f"MEMTRACE {label} baseline taken: {len(snap.traces)} traced blocks", flush=True)
        else:
            for index, stat in enumerate(snap.compare_to(before, "lineno")[:MEMTRACE_TOP], 1):
                print(f"MEMTRACE {label} site {index} {_site(stat)} {stat.size_diff / 2 ** 20:+.2f}MiB "
                      f"now={stat.size / 2 ** 20:.2f}MiB blocks={stat.count_diff:+d} of {stat.count}", flush=True)
    if memtrace_deep():
        try:
            deep_report(label, census)
        except Exception as failure:
            print(f"MEMTRACE {label} deep_error {type(failure).__name__}: {failure}", flush=True)
    if _truthy(MEMTRACE_TYPES_ENV, True):
        counts = _type_census()
        before = state["types"]
        state["types"] = counts
        if before is not None:
            grown = sorted(((counts[name] - before.get(name, 0), name) for name in counts), reverse=True)
            print(f"MEMTRACE {label} types " + " ".join(f"{name}={delta:+d}" for delta, name in
                                                          grown[:MEMTRACE_TYPE_TOP]), flush=True)
    _trace["ticks"] += 1
    return True


def memtrace_engine(scheduler):
    if not memtrace_on():
        return False
    return memtrace_tick("engine", lambda: engine_readings(scheduler))


def trim_on():
    return str(os.environ.get(TRIM_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def process_memory():
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            rows = dict(line.split(":", 1) for line in handle if ":" in line)
    except OSError:
        return None
    return {name: int(rows[name].split()[0]) * 1024 for name in ("VmRSS", "RssAnon", "VmHWM") if name in rows}


def trim_heap():
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return None


def mm_cache_reading(engine):
    try:
        held = getattr(getattr(engine, "renderer", None), "mm_processor_cache", None)
    except Exception:
        return None
    lru = getattr(held, "_cache", None)
    if lru is None or not hasattr(lru, "currsize"):
        return None
    row = {"bytes": int(lru.currsize), "cap": int(lru.maxsize), "items": len(lru), "kind": type(held).__name__}
    try:
        info = lru.stat()
        row.update(hits=int(info.hits), total=int(info.total))
    except Exception:
        pass
    return row


def gib(count):
    return f"{count / 2 ** 30:.2f}"


def memory_line(memory, cache, trim, dropped):
    parts = []
    if memory is None:
        parts.append("process memory unreadable here")
    else:
        parts.append(f"process rss {gib(memory.get('VmRSS', 0))} GiB (anon {gib(memory.get('RssAnon', 0))}, "
                     f"peak rss {gib(memory.get('VmHWM', 0))})")
    if cache is None:
        parts.append("no multimodal processor cache found")
    else:
        hit = f", {cache['hits']} of {cache['total']} lookups hit" if "total" in cache else ""
        parts.append(f"multimodal processor cache ({cache['kind']}) {gib(cache['bytes'])} of {gib(cache['cap'])} GiB "
                     f"in {cache['items']} items{hit}")
    if trim is not None:
        parts.append(trim)
    parts.append(f"image data let go from {dropped['prompts']} paused prompts ({dropped['items']} items) once the "
                 f"engine held them")
    return "; ".join(parts)


class Session:
    def __init__(self, key, rid, tokens, first):
        self.key = key
        self.rid = rid
        self.tokens = token_store(tokens)
        self.first = first
        self.reply_start = len(tokens)
        self.view = None
        self.images = []
        self.inputs = asyncio.Queue()
        self.outputs = asyncio.Queue()
        self.task = None
        self.busy = False
        self.closed = False
        self.used = time.monotonic()
        self.turns = 0
        self.returning = None
        self.resuming = None
        self.backup = None
        self.trim = TrimState()
        self.spec_seen = None
        self.after_pause = None


def note_capture_session(rid, key, tokens):
    if str(os.environ.get("TAAF_VLLM_DRAFT_CAPTURE") or "").strip().lower() not in ("1", "on", "true"):
        return
    folder = str(os.environ.get("TAAF_VLLM_DRAFT_CAPTURE_DIR") or "").strip() or os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "draft-capture")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "sessions.jsonl"), "a", encoding="utf-8") as sink:
        sink.write(json.dumps({"rid": rid, "key": None if key is None else str(key), "tokens": int(tokens),
                               "at": time.time()}) + chr(10))


def turn_spec_metrics(engine, session):
    processor = getattr(engine, "output_processor", None)
    states = getattr(processor, "request_states", None) or {}
    internal = [*(getattr(processor, "external_req_ids", None) or {}).get(session.rid, ()), session.rid]
    key = next((one for one in internal if getattr(states.get(one), "spec_decode_metrics", None) is not None), None)
    seen = None if key is None else states[key].spec_decode_metrics
    if seen is None or not getattr(seen, "histogram", None):
        return None
    from vllm.v1.metrics.stats import RequestSpecDecodeMetrics
    rid, before, drafted = session.spec_seen or (None, None, 0)
    if rid != key or before is None or len(before) != len(seen.histogram):
        before, drafted = [0] * len(seen.histogram), 0
    histogram = [max(0, now - was) for now, was in zip(seen.histogram, before)]
    session.spec_seen = (key, list(seen.histogram), int(seen.num_draft_tokens))
    return RequestSpecDecodeMetrics(num_spec_tokens=seen.num_spec_tokens, histogram=histogram,
                                    num_draft_tokens=max(0, int(seen.num_draft_tokens) - drafted))


class Saved:
    def __init__(self, tokens, reply_start, view, images, why=None):
        self.tokens = tokens
        self.reply_start = reply_start
        self.view = view
        self.images = images
        self.why = why


class Router:
    def __init__(self, engine, original):
        self.engine = engine
        self.original = original
        self.sessions = OrderedDict()
        self.by_first = {}
        self.by_key = {}
        self.count = 0
        self.stops = stop_ids()
        self.slack = _number(SLACK_ENV, DEFAULT_SLACK, int)
        self.idle = _number(IDLE_ENV, DEFAULT_IDLE_S)
        self.most = _number(MAX_ENV, DEFAULT_MAX, int)
        self.kv_high = _number(KV_HIGH_ENV, DEFAULT_KV_HIGH)
        self.kv_wait = _number(KV_WAIT_ENV, DEFAULT_KV_WAIT)
        self.max_len = int(getattr(getattr(engine, "model_config", None), "max_model_len", 0) or 0)
        self.resume = resume_unstopped()
        self.tally = {"turns": 0, "extend": 0, "reply": 0, "miss": 0, "fresh": 0, "bypass": 0,
                      "reused": 0, "fed": 0, "closed_idle": 0, "closed_pressure": 0,
                      "closed_count": 0, "closed_miss": 0, "closed_error": 0, "kept_length": 0}
        self.branch = branch_on()
        self.fork = fork_on()
        self.forks = {}
        if self.branch or self.fork:
            self.tally.update({"branch": 0, "branch_tokens": 0, "branch_cached": 0})
        if self.fork:
            self.tally.update({"fork_marked": 0, "fork_fallback": 0, "fork_cancelled": 0})
        self.rewind = rewind_on()
        if self.rewind:
            self.tally.update({"rewind_backup": 0, "rewind_restored": 0, "rewind_fallback": 0, "rewind_kept": 0,
                               "rewind_let_go": 0})
        self.client_view = client_view_on()
        self.exact = self.client_view and not server_tokens_on()
        if self.exact:
            self.tally.update({"rewind": 0, "rewind_kept": 0})
        elif self.client_view:
            self.tally["client_view"] = 0
        self.text_guard = self.client_view and not self.exact and text_guard_on()
        if self.text_guard:
            self.tally.update({"text_same": 0, "text_refused": 0, "text_unread": 0})
        self.tool_guard = self.text_guard and switch_default_on(TOOL_GUARD_ENV)
        if self.tool_guard:
            self.tally.update({"text_tool_call": 0, "tool_call_kept": 0, "tool_call_fed": 0})
        self.lead_guard = self.text_guard and switch_default_on(LEAD_GUARD_ENV)
        if self.lead_guard:
            self.tally.update({"text_lead_space": 0, "lead_space_kept": 0, "lead_space_fed": 0})
        self.stop_tail = self.text_guard and stop_tail_on()
        if self.stop_tail:
            self.tally.update({"stop_tail": 0, "stop_tail_tokens": 0})
        self.trim_hole = trim_hole_mode()
        self.trim_native = trim_hole_native() if self.trim_hole else TRIM_HOLE_POSITIONS
        self.trim_limits = trim_hole_limits(self.trim_native) if self.trim_hole else None
        self.trim_said = False
        self.trim_reply = bool(self.trim_hole) and str(os.environ.get(TRIM_HOLE_REPLY_ENV) or "").strip().lower()             not in ("", "0", "false", "no", "off")
        self.trim_text = bool(self.trim_hole) and str(os.environ.get(TRIM_HOLE_TEXT_ENV) or "").strip().lower() \
            not in ("", "0", "false", "no", "off")
        if self.trim_text:
            self.tally["trim_text_matched"] = 0
            say(f"TRIM_HOLE text ({TRIM_HOLE_TEXT_ENV}): when the kept run does not match the session's tokens id for "
                f"id (the client re-rendered earlier replies), the hole is cut if the whole kept run decodes to the "
                f"same text as the client's run; the engine keeps its own tokens and the length difference becomes "
                f"the position shift")
        if self.trim_reply:
            self.tally["trim_reply_matched"] = 0
            say(f"TRIM_HOLE reply ({TRIM_HOLE_REPLY_ENV}): when the kept run stops at the session's last reply, that "
                f"reply may come back re-rendered by the client; the hole is cut if the re-rendered reply decodes to "
                f"the same text as the engine's reply tokens, and the engine keeps its own reply tokens")
        self.trim_engine_said = False
        self.trim_checks = trim_hole_checks() if self.trim_hole == "1" else 0
        self.trim_far_left, self.trim_far_gap, self.trim_far_every = (
            trim_hole_far_settings() if self.trim_hole == "1" else (0, 0, 0))
        self.trim_far_seen = 0
        self.rope_gap_on = switch_on(ROPE_GAP_ENV) and self.trim_hole == "1"
        if self.trim_far_left:
            say(f"TRIM_HOLE far checks ({TRIM_HOLE_FAR_CHECKS_ENV}={self.trim_far_left}): every "
                f"{self.trim_far_every}th punched hole after the regular checks, and the first hole of a session "
                f"whose head lies past {TRAINED_POSITIONS} positions, is answered three ways - the holed session, an "
                f"honest recompute, and the same recompute with the tokens after the head {self.trim_far_gap} "
                f"positions farther - and the greedy and top-{TRIM_HOLE_CHECK_TOP} agreement of both against the "
                f"recompute is logged; that turn recomputes, and holes never stand aside on its result")
        if self.rope_gap_on:
            say(f"ROPE_GAP ({ROPE_GAP_ENV}): a session whose first request carries vllm_xargs {{{ROPE_GAP_XARG}: N}} "
                f"keeps its tokens after the first {ROPE_GAP_HEAD} (moved past an image) N rotary positions farther "
                f"for its life, as a hole that drops nothing, and later holes add to it; its prefix cache is salted "
                f"by the gap")
        self.trim_off = None
        self.trim_keys = trim_hole_keys() if self.trim_hole else None
        if self.trim_keys is not None:
            say(f"TRIM_HOLE keys: only sessions whose key matches {self.trim_keys.pattern!r} get holes; every other "
                f"session recomputes its trims as without holes")
        if self.trim_hole:
            self.tally.update({"trim_seen": 0, "trim_shape": 0, "trim_planned": 0, "trim_punched": 0,
                               "trim_check_pass": 0, "trim_check_fail": 0,
                               "trim_saved": 0, "trim_pos_max": 0})
        self.trim_scan = bool(self.trim_text) and switch_default_on(TRIM_HOLE_TEXT_SCAN_ENV)
        if self.trim_scan:
            say(f"TRIM_HOLE text scan ({TRIM_HOLE_TEXT_SCAN_ENV}): the text match tries up to "
                f"{TRIM_HOLE_TEXT_SCAN_MAX} places where the kept run's opening window recurs in the session, the "
                f"longest id-identical one first, and skips a candidate end whose last {TRIM_HOLE_TEXT_TAIL} tokens "
                f"read differently before decoding the whole run")
        self.text_open = switch_default_on(TEXT_OPEN_ENV)
        if self.text_open:
            self.tally["text_open_same"] = 0
            say(f"open replies ({TEXT_OPEN_ENV}): when the server's reply ended without a stop (a length cut) and "
                f"the client's re-render reads as that reply's text followed by more, the turn continues the "
                f"session and the engine is fed the client's tokens from the first one past the server's text; the "
                f"same rule finds a trim's kept run that ends on such a reply")
        self.trim_image_edge = bool(self.trim_hole) and switch_default_on(TRIM_HOLE_IMAGE_EDGE_ENV)
        if self.trim_image_edge:
            self.tally.update({"trim_image_moved": 0, "trim_image_unhashed": 0})
            say(f"TRIM_HOLE images ({TRIM_HOLE_IMAGE_EDGE_ENV}): a hole edge inside an image moves past that "
                f"image (the low edge up, the high edge down, both on the alignment); the images in the head must "
                f"be the session's own and the kept run's images must be the session's images there, in order, "
                f"by size and hash")
        self.trim_retry = trim_hole_retries() if self.trim_hole else 0
        if self.trim_retry:
            self.tally["trim_retry"] = 0
            say(f"TRIM_HOLE retry ({TRIM_HOLE_RETRY_ENV}={self.trim_retry}): an engine refusal coded "
                f"{' or '.join(TRIM_HOLE_RETRY_CODES)} is asked again after {TRIM_HOLE_RETRY_S:.2f} s, up to "
                f"{self.trim_retry} times")
        self.trim_rehole = bool(self.trim_hole) and switch_default_on(TRIM_HOLE_REHOLE_ENV)
        self.trim_sync_text = self.trim_rehole and switch_on(TRIM_HOLE_SYNC_TEXT_ENV)
        self._widths = {}
        if self.trim_sync_text:
            self.tally["trim_sync_text"] = 0
            say(f"TRIM_HOLE text sync ({TRIM_HOLE_SYNC_TEXT_ENV}): when a text hole's two sides differ inside a "
                f"message (a re-rendered tool call), the byte map is kept where each message's text still agrees "
                f"from its start and from its end, so the session takes later holes instead of recomputing")
        if self.trim_rehole:
            self.tally["trim_sync_lost"] = 0
            say(f"TRIM_HOLE re-hole ({TRIM_HOLE_REHOLE_ENV}): every hole records where the engine's kept tokens "
                f"and the client's tokens meet on the same byte of text, so a session holed by text takes later "
                f"holes whose kept run starts on such a point")
        if self.trim_hole and self.trim_native != TRIM_HOLE_POSITIONS:
            say(f"TRIM_HOLE positions ({TRIM_HOLE_POSITIONS_ENV}={self.trim_native}): holes may carry the rotary "
                f"positions up to {self.trim_limits[0]} (the M-RoPE cos/sin table holds {TRIM_HOLE_POSITIONS_TABLE} "
                f"rows; attention reads only the distance between positions)")
        self.cause_log = switch_default_on(CAUSE_LOG_ENV)
        self.closed_keys = {}
        self.seen_keys = set()
        if self.cause_log:
            say(f"cause log ({CAUSE_LOG_ENV}): every close, trim refusal and fresh admission prints one 'PAUSED "
                f"cause' line, and the tally counts each fresh admission by cause with its prompt tokens")
        self.diff_every = diff_every()
        self.diff_at = None
        self.diff_skipped = 0
        self.released = {}
        self.dropped = {"prompts": 0, "items": 0}
        self.trim = trim_on()
        self.reported = time.monotonic()
        self.loop_task = None
        self.parking = parking_on()
        self.parked = {}
        self.parking_now = {}
        if self.parking:
            self.park_wait = _number(PARKING_WAIT_ENV, PARKING_WAIT_S)
            self.tally.update({"park": 0, "parked": 0, "park_refused": 0, "park_tokens": 0, "resume": 0,
                               "resume_fallback": 0, "resume_tokens": 0, "resume_cached": 0, "unpark": 0})
        say(f"armed: one live request per conversation; stops {sorted(self.stops)}, reply slack "
            f"{self.slack}, idle {self.idle:.0f} s, at most {self.most} paused, kv high {self.kv_high}, "
            f"length stops {'resume' if self.resume else 'close'}")
        if self.branch:
            say(f"branches: a request marked {BRANCH_MARK} or with a session id starting {BRANCH_SESSION} runs "
                f"as an ordinary request on the prefix cache and leaves every paused session as it was")
        if self.fork:
            say(f"fork branches ({FORK_ENV}): a request whose session id is {BRANCH_SESSION}-<key>#<kind> and "
                f"whose tokens extend the paused session of <key> past its last prompt is marked {FORK_OF} so "
                f"the engine shares that session's KV and recurrent state with it; any other branch runs on the "
                f"prefix cache with one 'fork fallback' line; closing the session aborts its branches in flight")
        if self.rewind:
            say(f"rewind ({REWIND_ENV}): a turn marked {REWIND_MARK} that extends its key's paused session first has "
                f"the engine back up the session at its end (recurrent state copied, KV blocks held); the key's next "
                f"unmarked turn that extends that backup, not the side turn, rewinds the live session to it and feeds "
                f"only its new tokens; any other case takes the stale path with one 'PAUSED_REWIND fallback' line")
        if self.exact:
            say("client view: exact; a session continues only while its tokens are a prefix of the client's "
                "prompt, any other turn closes it and runs the client's tokens whole on the prefix cache")
        elif self.client_view:
            say(f"client view: {SERVER_TOKENS_ENV} is on; a turn whose prompt extends the client's previous "
                f"prompt resumes after the re-rendered reply's stop, whatever ids the client gave that reply; "
                f"the session keeps the server's own tokens")
            if self.text_guard:
                say(f"text guard ({TEXT_GUARD_ENV}): such a turn resumes only when the client's re-rendered reply "
                    f"decodes to the same text as the server's reply tokens; any other turn takes the stale path")
        if self.diff_every is not None:
            say(f"diff log: on a reply or a miss, the first history/prompt difference, at most one line "
                f"per {self.diff_every:g} s")
        say(f"release: a request marked {RELEASE_MARK} closes the paused session of its key before it runs as an "
            f"ordinary request; the key's next turn is logged as a return with its prefix-cache hit")
        if self.parking:
            say(f"park: a request marked {PARKING_MARK} parks its key's paused session (its newest prefix-cache "
                f"boundary goes to the engine's RAM store and the session closes) and answers with the outcome as "
                f"JSON text; the key's next turn copies the boundary back before it runs, waiting up to "
                f"{self.park_wait:.0f} s for room, and a {RELEASE_MARK} request for a parked key drops its park")
        if park_probe_on():
            say(f"park probe: {PARK_ENV} is on; the scheduler arms it from the pool reading"
                + ("" if pool_on() else f", and {POOL_ENV} is off, so it is never armed and prints nothing"))
        say(f"memory: a paused prompt's image data is let go once the engine holds it, and every {REPORT_S:.0f} s a "
            f"'PAUSED memory' line reads this process's rss and multimodal processor cache"
            + (f"; malloc_trim returns freed heap to the system each time ({TRIM_ENV}=0 turns it off)"
               if self.trim else f"; {TRIM_ENV} is off, so freed heap is not trimmed"))

    def eligible(self, prompt, sampling_params):
        if not isinstance(prompt, dict) or prompt.get("multi_modal_data") or prompt.get("prompt_embeds") is not None:
            return None
        if prompt.get("type", "token") not in ("token", "multimodal"):
            return None
        tokens = prompt.get("prompt_token_ids")
        if not tokens:
            return None
        if getattr(sampling_params, "n", 1) != 1 or getattr(sampling_params, "stop", None):
            return None
        if getattr(sampling_params, "prompt_logprobs", None) is not None:
            return None
        return list(tokens)

    def candidates(self, tokens, key):
        if key is not None:
            held = self.by_key.get(key)
            return [held] if held is not None else []
        found = []
        for length in sorted({session.first for session in self.sessions.values()}):
            if length > len(tokens):
                continue
            for session in self.by_first.get(first_key(tokens, length), ()):
                found.append(session)
        return found

    def match(self, tokens, key, images=None):
        best = None
        for session in self.candidates(tokens, key):
            if session.closed:
                continue
            kind, delta, viewed = self.plan_turn(session.tokens, session.reply_start, session.view, session.images,
                                                 tokens, images)
            if kind == "miss":
                if not session.busy:
                    best = best or ("stale", session, [])
                continue
            if session.busy:
                continue
            if viewed:
                self.tally["client_view"] += 1
            return kind, session, delta
        return best or ("miss", None, [])

    def plan_turn(self, history, reply_start, view, held_images, tokens, images):
        if self.exact:
            kind, delta = plan_exact(history, tokens)
            if kind == "extend" and not images_agree(held_images, images or [], len(history)):
                kind, delta = "miss", []
            return kind, delta, False
        if self.client_view:
            kind, delta, viewed = plan_view(view, history, reply_start, tokens, self.stops, self.slack)
            if kind == "reply" and self.text_guard:
                known = len(view) if viewed else reply_start
                if getattr(self, "text_open", False) and len(history) and int(history[-1]) not in self.stops:
                    at = self.open_reply(history[reply_start:], tokens[known:len(tokens) - len(delta)])
                    if at is not None:
                        self.tally["text_open_same"] = self.tally.get("text_open_same", 0) + 1
                        return kind, list(tokens[known + at:]), viewed
                    tail = self.stop_tail_delta(history[reply_start:], tokens, known, len(tokens) - len(delta),
                                                blank=True)
                    if tail is not None:
                        self.tally["text_open_tail"] = self.tally.get("text_open_tail", 0) + 1
                        return "tail", tail, viewed
                verdict = self.same_reply(history[reply_start:], tokens[known:len(tokens) - len(delta)])
                if verdict == "refused" and getattr(self, "stop_tail", False):
                    tail = self.stop_tail_delta(history[reply_start:], tokens, known, len(tokens) - len(delta))
                    if tail is not None:
                        self.tally["stop_tail"] += 1
                        return "tail", tail, viewed
                self.tally[f"text_{verdict}"] += 1
                if verdict == "tool_call":
                    self.tally["tool_call_kept"] += len(history)
                    self.tally["tool_call_fed"] += len(delta)
                elif verdict == "lead_space":
                    self.tally["lead_space_kept"] += len(history)
                    self.tally["lead_space_fed"] += len(delta)
                elif verdict != "same":
                    return "miss", [], False
            return kind, delta, viewed
        kind, delta = plan(history, reply_start, tokens, self.stops, self.slack)
        return kind, delta, False

    def decoded(self):
        decoder = self.tokenizer()
        if decoder is None:
            return None
        return lambda ids: decoder.decode([int(one) for one in ids], skip_special_tokens=False)

    def open_reply(self, ours, theirs):
        decode = self.decoded()
        if decode is None:
            return None
        try:
            return open_split(ours, theirs, decode)
        except Exception:
            return None

    def token_piece(self, token):
        pieces = self.__dict__.setdefault("_pieces", {})
        held = pieces.get(token)
        if held is not None:
            return held
        try:
            piece = self.tokenizer().convert_ids_to_tokens([token])[0]
        except Exception:
            piece = ""
        piece = piece if isinstance(piece, str) else ""
        pieces[token] = piece
        return piece

    def token_width(self, token):
        held = self._widths.get(token)
        if held is not None:
            return held
        decoder = self.tokenizer()
        width = 0
        try:
            piece = decoder.convert_ids_to_tokens([token])[0]
            width = len(piece) if isinstance(piece, str) else 0
        except Exception:
            width = 0
        if width <= 0:
            try:
                width = len(decoder.decode([token], skip_special_tokens=False).encode("utf-8"))
            except Exception:
                width = 0
        self._widths[token] = width
        return width

    def text_options(self):
        options = {}
        decode = self.decoded() if (getattr(self, "trim_scan", False) or getattr(self, "text_open", False)) else None
        if getattr(self, "trim_scan", False) and decode is not None:
            options.update(scan=True, decode=decode)
        if getattr(self, "text_open", False) and decode is not None:
            options["open_split"] = lambda ours, theirs: open_split(ours, theirs, decode)
        return options

    def trim_sync(self, session, shape, tokens, hole, boundary, engine_len):
        at = boundary - shape.head
        held = session.tokens[shape.start:]
        if getattr(shape, "by_text", False) or shape.reply_shift:
            decode = self.decoded()
            if decode is None or self.tokenizer() is None:
                return None
            try:
                ranges = sync_pairs(held, tokens[shape.head:boundary], self.token_width)
                if ranges is None and getattr(self, "trim_sync_text", False):
                    ranges = sync_pairs_text(held, tokens[shape.head:boundary], self.token_piece)
                    if ranges is not None:
                        self.bump("trim_sync_text")
                if ranges is not None:
                    total = sum(self.token_width(int(one)) for one in held)
                    if total != len(decode(held).encode("utf-8")):
                        ranges = None
            except Exception:
                ranges = None
            if ranges is None:
                return None
        else:
            if at != shape.run:
                return None
            ranges = [(0, at, 0)]
        composed = sync_compose(ranges, shape.head, shape.start, session.trim.engine_map(), hole.dropped)
        if sync_lookup(composed, boundary) != engine_len - hole.dropped:
            return None
        if sync_lookup(composed, shape.head) != shape.head + hole.pad:
            return None
        return composed

    def stop_tail_delta(self, ours, tokens, known, end, blank=False):
        if len(ours) < 2 or (int(ours[-1]) not in self.stops and not (blank and self.blank_token(ours[-1]))):
            return None
        decoder = self.tokenizer()
        if decoder is None:
            return None
        theirs = list(tokens[known:end])
        try:
            left = decoder.decode(list(ours[:-1]), skip_special_tokens=False)
            right = decoder.decode(theirs, skip_special_tokens=False)
            if len(right) <= len(left) or not right.startswith(left):
                return None
            for cut_at in range(len(theirs) - 1, max(0, len(theirs) - STOP_TAIL_SCAN) - 1, -1):
                if decoder.decode(theirs[:cut_at], skip_special_tokens=False) == left:
                    break
            else:
                return None
        except Exception:
            return None
        tail = list(tokens[known + cut_at:])
        ended = "a stop" if int(ours[-1]) in self.stops else "white space the client's render left out"
        say(f"stop tail: a reply that ended at {ended} the engine never computed is continued from the token before "
            f"it with the client's {len(tail) - (len(tokens) - end)} re-rendered tokens "
            f"{right[len(left):][:60]!r}, so the model reads the client's text without a recompute")
        return tail

    def blank_token(self, token):
        decoder = self.tokenizer()
        if decoder is None:
            return False
        try:
            piece = decoder.decode([int(token)], skip_special_tokens=False)
        except Exception:
            return False
        return bool(piece) and not piece.strip()

    def think_end(self):
        held = getattr(self, "_think_end", None)
        if held is not None:
            return held
        decoder = self.tokenizer()
        found = -1
        try:
            found = int(decoder.convert_tokens_to_ids(THINK_END))
        except Exception:
            found = -1
        self._think_end = found if found is not None and found >= 0 else -1
        return self._think_end

    async def pause_now(self, key):
        self.bump("pause")
        session = self.by_key.get(key) if key else None
        why = ("no_session" if session is None or session.closed else
               "no_turn" if not session.busy else "")
        if not why:
            processor = getattr(self.engine, "output_processor", None)
            internal = list((getattr(processor, "external_req_ids", None) or {}).get(session.rid, ()))
            answer = await self.engine_call(PAUSE_UTILITY, internal[-1] if internal else session.rid,
                                            self.think_end())
            if answer.get("paused"):
                self.bump("pause_armed")
                generated = int(answer.get("generated") or 0)
                say(f"pause {key}: {session.rid} stops at its next token after {generated} generated, its thinking "
                    f"still open; the session stays paused with them in KV")
                return {"paused": True, "generated": generated, "why": "", "key": key}
            why = str(answer.get("why") or "refused")
        self.bump("pause_refused")
        say(f"pause {key}: refused, {why}" + ("; the turn runs to its end" if session is not None and session.busy
                                               and not session.closed else ""))
        return {"paused": False, "generated": 0, "why": why, "key": key}

    def synthetic_reply(self, prompt, request_id, result):
        from vllm.outputs import CompletionOutput, RequestOutput
        text = json.dumps(result, sort_keys=True, default=str)
        ids = list(prompt.get("prompt_token_ids") or []) if isinstance(prompt, dict) else []
        return RequestOutput(request_id=request_id, prompt=None, prompt_token_ids=ids, prompt_logprobs=None,
                             outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None,
                                                       logprobs=None, finish_reason="stop", stop_reason=None)],
                             finished=True)

    async def release_reply(self, prompt, request_id, key, freed):
        if not getattr(self, "_release_said", False):
            self._release_said = True
            say(f"release reply: a {RELEASE_MARK} request is answered by the router once its key's session is let go "
                f"({freed} tokens for {key}); it no longer runs a one-token generation through the engine")
        yield self.synthetic_reply(prompt, request_id, {"released": int(freed or 0), "key": key})

    async def hot_reply(self, prompt, sampling_params, request_id, kwargs):
        marks = getattr(sampling_params, "extra_args", None) or {}
        key = kwargs.get("session_id") or marks.get(KEY)
        result = await self.hot_spans(key, marks)
        yield self.synthetic_reply(prompt, request_id, result)

    async def hot_spans(self, key, marks):
        self.bump("hot_spans")
        if not hot_spans_on():
            return {"why": f"{HOT_SPANS_ENV} is off", "key": key}
        session = self.by_key.get(key) if key else None
        if session is None or session.closed:
            self.bump("hot_spans_none")
            return {"why": "no_session", "key": key}
        processor = getattr(self.engine, "output_processor", None)
        internal = list((getattr(processor, "external_req_ids", None) or {}).get(session.rid, ()))
        align = int(marks.get("align") or HOT_SPANS_ALIGN)
        gap = int(marks.get("gap") or HOT_SPANS_GAP)
        top = int(marks.get("top") or HOT_SPANS_TOP)
        below = marks.get("below")
        limit = -1
        if below is not None:
            limit = session.trim.engine_index(int(below)) if session.trim.holes else int(below)
            limit = -1 if limit is None else int(limit)
        above = int(marks.get("above") or 0)
        if above > 0 and session.trim.holes:
            above = int(session.trim.engine_index(above) or 0)
        budget = int(marks.get("pick_tokens") or HOT_READS_PICK_TOKENS)
        began = time.monotonic()
        fresh = 1 if str(marks.get("fresh") or "").strip() in ("1", "true", "True") else 0
        answer = await self.engine_call(HOT_SPANS_MARK, internal[-1] if internal else session.rid, align, gap, top,
                                        limit, fresh, budget, above)
        answer = dict(answer, key=key, rid=session.rid, call_ms=round((time.monotonic() - began) * 1000.0, 2))
        cap = int(marks.get("text_tokens") or 0)
        if cap > 0 and answer.get("spans"):
            answer["text_tokens"] = self.hot_texts(session, answer["spans"], cap)
        if not getattr(self, "_hot_said", False) and "blocks" in answer:
            self._hot_said = True
            shown = ", ".join(f"[{one['start']}, {one['end']}) {one['rows']} rows" for one in answer["blocks"][:5])
            texts = sum(1 for one in answer.get("spans") or () if "text" in one)
            held = (f"{answer.get('rows')} reads of QSA layers {answer.get('layers')} in {answer.get('bins')} "
                    f"{answer.get('bin_tokens')}-token bins, the {answer.get('picked')} hottest at "
                    f"{answer.get('pick_floor')}+ reads making {len(answer.get('spans') or ())} span(s)"
                    if answer.get("source") == "reads"
                    else f"{answer.get('rows')} hot rows of {answer.get('region')}")
            say(f"hot spans ({HOT_SPANS_ENV}) first answer for {key}: {held} over {answer.get('computed')} computed "
                f"tokens, {len(answer['blocks'])} "
                f"{align}-token blocks; top {shown}; worker {answer.get('worker_ms')} ms, call {answer['call_ms']} ms"
                + (f"; text for {texts} span(s), {answer.get('text_tokens', 0)} of {cap} tokens" if cap > 0 else ""))
        return answer

    def hot_texts(self, session, spans, cap):
        decoder = self.tokenizer()
        if decoder is None:
            return 0
        tokens = list(session.tokens)
        images = list(getattr(session, "images", None) or ())
        used = 0
        for span in spans:
            low = hot_spans_client_index(session.trim, int(span["start"]))
            high = hot_spans_client_index(session.trim, int(span["end"]) - 1)
            if low is None or high is None or not 0 <= low <= high < len(tokens):
                continue
            low, high = hot_spans_widen(tokens, low, high + 1, decoder)
            if used + high - low > cap:
                continue
            span["text"] = hot_spans_text(tokens, images, low, high, decoder)
            span["text_start"], span["text_end"] = low, high
            used += high - low
        return used

    async def pause_reply(self, prompt, sampling_params, request_id, kwargs):
        from vllm.outputs import CompletionOutput, RequestOutput
        marks = getattr(sampling_params, "extra_args", None) or {}
        result = await self.pause_now(kwargs.get("session_id") or marks.get(KEY))
        text = json.dumps(result, sort_keys=True, default=str)
        ids = list(prompt.get("prompt_token_ids") or []) if isinstance(prompt, dict) else []
        yield RequestOutput(request_id=request_id, prompt=None, prompt_token_ids=ids, prompt_logprobs=None,
                            outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None,
                                                      logprobs=None, finish_reason="stop", stop_reason=None)],
                            finished=True)

    def same_reply(self, ours, theirs):
        decoder = self.tokenizer()
        if decoder is None:
            return "unread"
        try:
            left = decoder.decode(list(ours), skip_special_tokens=False)
            right = decoder.decode(list(theirs), skip_special_tokens=False)
        except Exception:
            return "unread"
        if left == right:
            return "same"
        lead = getattr(self, "lead_guard", False)
        trimmed = (lead_trimmed(left), lead_trimmed(right)) if lead else (left, right)
        if lead and trimmed[0] == trimmed[1]:
            if not getattr(self, "_lead_said", False):
                self._lead_said = True
                at = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                          min(len(left), len(right)))
                say(f"text guard leading space ({LEAD_GUARD_ENV}): a re-rendered reply that differs from the "
                    f"server's only by the whitespace that opens a reply continues the session on the server's "
                    f"tokens with no recompute; first at char {at}: server {left[max(0, at - 40):at + 40]!r} "
                    f"client {right[max(0, at - 40):at + 40]!r}")
            return "lead_space"
        if getattr(self, "tool_guard", False) and tool_calls_agree(*trimmed):
            if not getattr(self, "_tool_said", False):
                self._tool_said = True
                at = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b),
                          min(len(left), len(right)))
                say(f"text guard tool_call ({TOOL_GUARD_ENV}): a re-rendered reply that differs from the server's "
                    f"only inside its tool_call blocks (same calls, same text around them) continues the session on "
                    f"the server's tokens with no recompute; first at char {at}: server "
                    f"{left[max(0, at - 40):at + 40]!r} client {right[max(0, at - 40):at + 40]!r}")
            return "tool_call"
        now = time.monotonic()
        if now - getattr(self, "_text_said", 0.0) >= 30.0:
            self._text_said = now
            at = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            say(f"text guard refused: server reply {len(left)} chars, client re-render {len(right)} chars, first "
                f"difference at char {at}: server {left[max(0, at - 40):at + 40]!r} client {right[max(0, at - 40):at + 40]!r}")
        return "refused"

    async def take_backup(self, session, key):
        answer = await self.engine_call("taaf_rewind_backup", session.rid)
        if answer.get("backed"):
            self.bump("rewind_backup")
            session.backup = Saved(session.tokens, session.reply_start, session.view, session.images)
            return
        session.backup = Saved(session.tokens, session.reply_start, session.view, session.images,
                               why=f"the engine took no backup at the side turn ({answer.get('why')})")

    def rewind_fallback(self, key, why):
        self.bump("rewind_fallback")
        rewind_say(f"fallback {key}: {why}; this turn takes the stale path")

    async def rewind_turn(self, side, key, prompt, tokens, images, kind, session, delta):
        held = self.by_key.get(key)
        if held is None or held.closed or held.busy:
            return kind, session, delta
        if side:
            if session is held and kind in ("extend", "reply"):
                await self.take_backup(held, key)
            else:
                held.backup = None
            return kind, session, delta
        saved, held.backup = held.backup, None
        if saved is None:
            return kind, session, delta
        if session is held and kind in ("extend", "reply"):
            if saved.why is None:
                self.bump("rewind_let_go")
                await self.engine_call("taaf_rewind_drop", held.rid)
                rewind_say(f"let go {key}: the turn continues the side turn, so its backup is not used")
            return kind, session, delta
        why = saved.why
        found, ahead = "miss", []
        if why is None:
            found, ahead, _ = self.plan_turn(saved.tokens, saved.reply_start, saved.view, saved.images, tokens,
                                             images)
            if found not in ("extend", "reply"):
                why = (f"the turn shares {lcp(saved.tokens, tokens)} of the {len(saved.tokens)} tokens backed up "
                       f"before the side turn and does not extend them")
            elif cut(prompt, len(tokens) - len(ahead)) is None:
                why = "an image straddles the point where the turn's new tokens begin"
            if why is not None:
                await self.engine_call("taaf_rewind_drop", held.rid)
            else:
                answer = await self.engine_call("taaf_rewind_restore", held.rid)
                if not answer.get("rewound"):
                    why = f"the engine refused the rewind ({answer.get('why')})"
        if why is not None:
            self.rewind_fallback(key, why)
            return kind, session, delta
        held.tokens = saved.tokens
        held.reply_start = saved.reply_start
        held.view = saved.view
        held.images = saved.images
        self.bump("rewind_restored")
        self.bump("rewind_kept", int(answer.get("kept") or 0))
        return found, held, ahead

    def engine_len(self, session):
        return len(session.tokens) + (session.trim.pad if session.trim.holes else 0)

    def trim_same(self, ours, theirs):
        verdict = self.same_reply(ours, theirs)
        if verdict in ("tool_call", "lead_space"):
            self.bump(f"trim_text_{verdict}")
        return verdict in ("same", "tool_call", "lead_space")

    def trim_no_run_say(self, session, tokens, prompt=None):
        head, window, at, agreed = trim_hole_nearest(session.tokens, tokens)
        decode = self.decoded()
        if decode is None:
            return None
        try:
            mine = decode(list(tokens[head:head + window]))
            theirs = decode(list(session.tokens[at:at + window])) if at is not None else ""
        except Exception:
            return None
        line = (f"TRIM_HOLE no_run {session.rid} (key {getattr(session, 'key', None)}): no place in the "
                f"{len(session.tokens)}-token history after the common head of {head} tokens matches the client's "
                f"next {window} tokens; client {mine!r}; "
                + (f"nearest server place {at} agrees on {agreed} token(s): {theirs!r}" if at is not None
                   else "the client's first token there never appears after the head")
                + images_near(head, getattr(session, "images", None) or (), images_of(prompt)))
        say(line)
        return line

    def trim_refused(self, why, kind, session, delta):
        self.bump(f"trim_refused_{why.split(':')[0]}")
        if getattr(self, "cause_log", False):
            session.refused = why
            say(f"cause trim_refused {why}: key {getattr(session, 'key', None)} {getattr(session, 'rid', None)}, "
                f"history {len(session.tokens)} tokens, holes {session.trim.holes}")
        return kind, session, delta

    def trim_params(self, session, sampling_params):
        params = self.session_params(sampling_params, getattr(session, "key", None))
        if session.trim.holes:
            extra = dict(getattr(params, "extra_args", None) or {})
            extra[TRIM_HOLE_MARK] = session.trim.mark()
            params.extra_args = extra
            session.trim.anchor = session.trim.low
        return params

    async def trim_probe(self, session):
        picks = []
        while True:
            out = await session.outputs.get()
            if out is CLOSED:
                raise RuntimeError(f"paused session {session.rid} ended inside the hole check")
            if isinstance(out, BaseException):
                raise out
            piece = out.outputs[0]
            picks.extend(piece.token_ids)
            if piece.finish_reason is not None:
                return picks

    def rope_gap_start(self, session, whole, params):
        extra = dict(getattr(params, "extra_args", None) or {})
        try:
            gap = int(extra.get(ROPE_GAP_XARG) or 0)
        except (TypeError, ValueError):
            gap = 0
        if gap <= 0:
            return whole
        cap = self.trim_limits[0] if getattr(self, "trim_limits", None) else 0
        if not isinstance(whole, dict) or len(session.tokens) + gap > cap:
            self.bump("rope_gap_refused")
            if not getattr(self, "_rope_gap_refused_said", False):
                self._rope_gap_refused_said = True
                say(f"ROPE_GAP refused for {session.rid}: a gap of {gap} over {len(session.tokens)} tokens "
                    f"{'needs a prompt dict' if not isinstance(whole, dict) else f'passes the {cap} position cap'}")
            return whole
        head = rope_gap_head(ROPE_GAP_HEAD, images_of(whole))
        extra[ROPE_GAP_MARK] = [int(head), int(gap)]
        params.extra_args = extra
        session.trim.shift = gap
        session.rope_gap = gap
        self.bump("rope_gap_sessions")
        if not getattr(self, "_rope_gap_said", False):
            self._rope_gap_said = True
            say(f"ROPE_GAP first session {session.rid} (key {session.key}): tokens from {head} sit {gap} positions "
                f"farther, head distance {len(session.tokens) + gap} at the start, prefix cache salt "
                f"taaf-rope-gap-{gap}-{head}")
        return dict(whole, cache_salt=f"taaf-rope-gap-{gap}-{head}")

    def trim_far_due(self, session, holed):
        if getattr(self, "trim_far_left", 0) <= 0:
            return False
        self.trim_far_seen += 1
        late = holed > TRAINED_POSITIONS and not getattr(session, "far_checked_late", False)
        if late:
            session.far_checked_late = True
        due = late or self.trim_far_seen % max(self.trim_far_every, 1) == 0
        if due:
            self.trim_far_left -= 1
        return due

    async def trim_probe_ranked(self, session):
        picks, ranks = [], []
        while True:
            out = await session.outputs.get()
            if out is CLOSED:
                raise RuntimeError(f"paused session {session.rid} ended inside the far check")
            if isinstance(out, BaseException):
                raise out
            piece = out.outputs[0]
            picks.extend(piece.token_ids)
            ranks.extend(piece.logprobs or [])
            if piece.finish_reason is not None:
                return picks, ranks

    async def trim_far_check(self, session, prompt, sampling_params, boundary, delta, head, shift, length):
        from vllm.sampling_params import RequestOutputKind
        probe = trim_hole_check_params(self.trim_params(session, sampling_params), ranked=True)
        reference = trim_hole_check_params(sampling_params.clone(), ranked=True)
        reference.output_kind = RequestOutputKind.FINAL_ONLY
        extra = {name: value for name, value in (getattr(reference, "extra_args", None) or {}).items()
                 if name not in (KEY, MARK, TRIM_HOLE_MARK, ROPE_GAP_MARK, ROPE_GAP_XARG)}
        extra[SEAT_FREE_MARK] = 1
        reference.extra_args = extra
        moved = rope_gap_head(int(head), images_of(prompt))
        far = reference.clone()
        far.extra_args = dict(extra, **{ROPE_GAP_MARK: [int(moved), int(self.trim_far_gap)]})
        whole = dict(prompt) if isinstance(prompt, dict) else {"prompt_token_ids": list(prompt)}
        distant_prompt = dict(whole, cache_salt=f"taaf-far-{self.trim_far_gap}-{moved}")
        session.busy = True
        session.inputs.put_nowait((cut(prompt, boundary), probe))
        try:
            mine, theirs, distant = await asyncio.gather(self.trim_probe_ranked(session),
                                                         self.trim_reference(prompt, reference),
                                                         self.trim_reference(distant_prompt, far))
            near_ok, near_line = trim_hole_verdict(mine[0], theirs)
            far_ok, far_line = trim_hole_verdict(distant[0], theirs)
            overlap = (trim_hole_overlap(mine, theirs), trim_hole_overlap(distant, theirs))
        except Exception as failure:
            near_ok = far_ok = False
            near_line = far_line = f"the far check raised {type(failure).__name__}: {failure}"
            overlap = (None, None)
        session.busy = False
        await self.close(session, "trim_far_check")
        self.bump("trim_far_checks")
        self.bump("trim_far_near_pass" if near_ok else "trim_far_near_fail")
        self.bump("trim_far_far_pass" if far_ok else "trim_far_far_fail")
        say(f"TRIM_HOLE far check {session.rid} (key {session.key}): {length} prompt tokens, head {moved}; head "
            f"distance honest {length}, holed {length + int(shift)}, far {length + int(self.trim_far_gap)}; holed vs "
            f"honest {'agree' if near_ok else 'DIFFER'} ({near_line}; top-{TRIM_HOLE_CHECK_TOP} at the first token "
            f"{overlap[0]}/{TRIM_HOLE_CHECK_TOP}); far vs honest {'agree' if far_ok else 'DIFFER'} ({far_line}; "
            f"top-{TRIM_HOLE_CHECK_TOP} at the first token {overlap[1]}/{TRIM_HOLE_CHECK_TOP}); "
            f"{self.trim_far_left} far check(s) left; this turn recomputes")
        return "miss", None, delta

    async def trim_reference(self, prompt, params):
        last = None
        whole = dict(prompt) if isinstance(prompt, dict) else prompt
        async for out in self.original(self.engine, whole, params, f"{self.next_rid()}-trimcheck"):
            last = out
        if last is None or not last.outputs:
            return [], []
        piece = last.outputs[0]
        return list(piece.token_ids), list(piece.logprobs or [])

    async def trim_check(self, session, prompt, sampling_params, boundary, delta):
        from vllm.sampling_params import RequestOutputKind
        probe = trim_hole_check_params(self.trim_params(session, sampling_params))
        reference = trim_hole_check_params(sampling_params.clone(), ranked=True)
        reference.output_kind = RequestOutputKind.FINAL_ONLY
        extra = {name: value for name, value in (getattr(reference, "extra_args", None) or {}).items()
                 if name not in (KEY, MARK, TRIM_HOLE_MARK)}
        extra[SEAT_FREE_MARK] = 1
        reference.extra_args = extra
        session.busy = True
        session.inputs.put_nowait((cut(prompt, boundary), probe))
        try:
            mine, theirs = await asyncio.gather(self.trim_probe(session), self.trim_reference(prompt, reference))
            passed, line = trim_hole_verdict(mine, theirs)
        except Exception as failure:
            passed, line = False, f"the check raised {type(failure).__name__}: {failure}"
        session.busy = False
        await self.close(session, "trim_check")
        self.bump("trim_check_pass" if passed else "trim_check_fail")
        if not passed:
            selfcheck_fail(TRIM_HOLE_ENV, f"{session.rid}: {line}")
            self.trim_off = line
            self.trim_checks = 0
        say(f"TRIM_HOLE check {'pass' if passed else 'FAIL'} {session.rid}: {line}; this turn recomputes"
            + ("" if passed else "; holes stand aside for the rest of the run")
            + (f"; {self.trim_checks} check(s) left before holes serve turns" if passed else ""))
        return "miss", None, delta

    async def trim_hole_turn(self, session, tokens, prompt, sampling_params, kind, delta):
        self.bump("trim_seen")
        if self.trim_off is not None:
            return self.trim_refused("off", kind, session, delta)
        if getattr(session, "key", None) in getattr(self, "parking_now", {}):
            return self.trim_refused("parking", kind, session, delta)
        keys = getattr(self, "trim_keys", None)
        if keys is not None and not keys.search(str(getattr(session, "key", None) or "")):
            return self.trim_refused("arm", kind, session, delta)
        if session.trim.holes and getattr(session.trim, "by_text", False) and not (
                getattr(self, "trim_rehole", False) and getattr(session.trim, "sync", None) is not None):
            return self.trim_refused("text_reholed", kind, session, delta)
        shape, why = trim_hole_find(session.tokens, tokens)
        if shape is None and why == "no_run" and self.trim_reply and session.view is None:
            shape, why = trim_hole_find_reply(session.tokens, tokens, session.reply_start, self.stops,
                                              self.trim_same, self.slack)
            if shape is not None:
                self.bump("trim_reply_matched")
        if shape is None and why == "no_run" and self.trim_text:
            shape, why = trim_hole_find_text(session.tokens, tokens, self.stops, self.trim_same, self.slack,
                                             **self.text_options())
            if shape is not None:
                self.bump("trim_text_matched")
                shape.by_text = True
        if shape is None and why == "no_run" and switch_default_on(TRIM_HOLE_NO_RUN_LOG_ENV):
            self.trim_no_run_say(session, tokens, prompt)
        if shape is None:
            return self.trim_refused(why, kind, session, delta)
        self.bump("trim_shape")
        if session.trim.holes:
            moved = trim_hole_rehead(shape, session.tokens, session.trim.head)
            if moved is not shape and (getattr(self, "trim_rehole", False)
                                       or getattr(self, "trim_native", TRIM_HOLE_POSITIONS) != TRIM_HOLE_POSITIONS):
                moved.reply_shift = shape.reply_shift
                moved.by_text = getattr(shape, "by_text", False)
            shape = moved
        engine_len = self.engine_len(session)
        native = getattr(self, "trim_native", TRIM_HOLE_POSITIONS)
        hole, why = trim_hole_plan(shape, session.trim, engine_len, self.trim_limits, native)
        if hole is None:
            return self.trim_refused(why, kind, session, delta)
        boundary = len(tokens) - shape.tail
        if cut(prompt, boundary) is None:
            return self.trim_refused("image_edge", kind, session, delta)
        if getattr(self, "trim_image_edge", False):
            refused, unhashed = trim_hole_image_check(session.images, images_of(prompt), shape.head, shape.start,
                                                      boundary, len(session.tokens))
            if refused is not None:
                return self.trim_refused(refused, kind, session, delta)
            if unhashed:
                self.bump("trim_image_unhashed")
            mapped = [(session.trim.engine_index(image[0]), image[1]) for image in session.images]
            moved, why = trim_hole_edges(hole, [one for one in mapped if one[0] is not None],
                                         session.trim.engine_index(shape.start), self.trim_limits)
            if moved is None:
                return self.trim_refused(why, kind, session, delta)
            if moved is not hole:
                self.bump("trim_image_moved")
                hole = moved
        limit = int(getattr(sampling_params, "max_tokens", 0) or 0)
        reach = engine_len + session.trim.shift + shape.tail + limit
        if reach > native or (self.max_len and engine_len - hole.dropped + shape.tail + limit > self.max_len):
            return self.trim_refused("length", kind, session, delta)
        sync = None
        if getattr(self, "trim_rehole", False):
            sync = self.trim_sync(session, shape, tokens, hole, boundary, engine_len)
            if sync is None:
                self.bump("trim_sync_lost")
                if session.trim.holes:
                    return self.trim_refused("text_sync", kind, session, delta)
        self.bump("trim_planned")
        self.bump("trim_planned_saved", boundary)
        self.bump("trim_planned_hole", hole.dropped)
        if not self.trim_said:
            self.trim_said = True
            say(f"TRIM_HOLE first {self.trim_hole}: head {shape.head}, run {shape.run} from {shape.start}, tail "
                f"{shape.tail}; hole [{hole.low}, {hole.high}) of {engine_len} engine tokens, positions shift "
                f"{hole.shift}")
        if self.trim_hole == "dry":
            return kind, session, delta
        answer = await self.engine_call("taaf_trim_hole", session.rid, hole.low, hole.high, hole.shift)
        for _ in range(getattr(self, "trim_retry", 0)):
            if answer.get("punched") or answer.get("code") not in TRIM_HOLE_RETRY_CODES:
                break
            await asyncio.sleep(TRIM_HOLE_RETRY_S)
            self.bump("trim_retry")
            answer = await self.engine_call("taaf_trim_hole", session.rid, hole.low, hole.high, hole.shift)
        if not answer.get("punched"):
            code = answer.get("code") or "error"
            if not self.trim_engine_said:
                self.trim_engine_said = True
                say(f"TRIM_HOLE first engine answer: refused {code}: {answer.get('why')}; the turn recomputes")
            return self.trim_refused(f"engine_{code}", kind, session, delta)
        session.trim.head, session.trim.pad, session.trim.shift = hole.head, hole.pad + shape.reply_shift, hole.shift
        session.trim.low, session.trim.anchor = hole.low, hole.high
        session.trim.holes += 1
        session.trim.by_text = bool(getattr(shape, "by_text", False))
        if getattr(self, "trim_rehole", False):
            session.trim.sync, session.trim.sync_end = sync, boundary
            if sync is None:
                session.trim.by_text = True
            else:
                session.trim.pad = sync_lookup(sync, boundary) - boundary
        session.tokens = token_store(tokens[:boundary])
        session.reply_start = boundary
        session.images = [image for image in images_of(prompt) if image[0] + image[1] <= boundary]
        self.bump("trim_punched")
        self.tally["trim_pos_max"] = max(self.tally["trim_pos_max"], reach)
        say(f"TRIM_HOLE punched {session.rid}: hole [{hole.low}, {hole.high}) dropped {hole.dropped} tokens and "
            f"{answer.get('freed')} block(s); {boundary} tokens kept from recompute, tail {shape.tail}; positions "
            f"shift {hole.shift}; {'checked against a recompute' if self.trim_checks > 0 else 'serving the turn'}")
        if self.trim_checks > 0:
            self.trim_checks -= 1
            return await self.trim_check(session, prompt, sampling_params, boundary, delta)
        if self.trim_far_due(session, len(tokens) + hole.shift):
            return await self.trim_far_check(session, prompt, sampling_params, boundary, delta, shape.head,
                                             hole.shift, len(tokens))
        self.bump("trim_saved", boundary)
        return "extend", session, list(tokens[boundary:])

    def fresh_cause(self, key, tokens):
        held = self.closed_keys.pop(key, None) if key is not None else None
        if key is None:
            cause = "no_key"
        elif held is not None:
            why, refused, size = held
            cause = f"after_{why}" + (f"_{refused.split(':')[0]}" if why == "miss" and refused else "")
        elif key in self.seen_keys:
            cause = "unmatched"
        else:
            cause = "first"
        if key is not None:
            self.seen_keys.add(key)
        self.bump(f"fresh_{cause}")
        self.bump(f"fresh_{cause}_tokens", len(tokens))
        say(f"cause fresh {cause}: key {key}, {len(tokens)} prompt tokens"
            + (f", the closed session held {held[2]} tokens" if held is not None else ""))
        return cause

    def index(self, session):
        self.sessions[session.rid] = session
        self.by_first.setdefault(first_key(session.tokens, session.first), []).append(session)
        if session.key is not None:
            self.by_key[session.key] = session

    def forget(self, session):
        self.sessions.pop(session.rid, None)
        place = first_key(session.tokens, session.first)
        bucket = self.by_first.get(place, [])
        if session in bucket:
            bucket.remove(session)
        if not bucket:
            self.by_first.pop(place, None)
        if session.key is not None and self.by_key.get(session.key) is session:
            self.by_key.pop(session.key, None)

    async def release(self, session):
        task = session.task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), RELEASE_S)
        except (asyncio.TimeoutError, Exception):
            pass

    async def close(self, session, why):
        if session.closed:
            return
        session.closed = True
        if getattr(self, "cause_log", False):
            if session.key is not None:
                self.closed_keys[session.key] = (why, getattr(session, "refused", None), len(session.tokens))
            say(f"cause close {why}: key {session.key} {session.rid}, {len(session.tokens)} tokens, "
                f"{'busy' if session.busy else 'paused'}")
        self.forget(session)
        self.tally[f"closed_{why}"] = self.tally.get(f"closed_{why}", 0) + 1
        await self.cancel_forks(session, why)
        if session.busy:
            await self.engine.abort(session.rid)
        else:
            session.inputs.put_nowait(None)

    async def cancel_forks(self, session, why):
        held = self.forks.get(session.key) if session.key is not None else None
        if not held:
            return []
        current = self.by_key.get(session.key)
        if current is not None and current is not session and not current.closed:
            taken = [rid for rid, main in held.items() if main == session.rid]
        else:
            taken = list(held)
        for rid in taken:
            held.pop(rid, None)
        if not held:
            self.forks.pop(session.key, None)
        if not taken:
            return []
        for rid in taken:
            await self.engine.abort(rid)
        self.bump("fork_cancelled", len(taken))
        say(f"fork cancel {session.key}: main {session.rid} closed ({why}), so {len(taken)} branch request(s) in "
            f"flight were aborted and their blocks freed: {', '.join(taken)}")
        return taken

    def session_params(self, sampling_params, key=None):
        from vllm.sampling_params import RequestOutputKind
        held = sampling_params.clone()
        held.output_kind = RequestOutputKind.DELTA
        extra = dict(getattr(held, "extra_args", None) or {})
        extra[MARK] = 1
        if key is not None:
            extra[SESSION_KEY] = str(key)
        held.extra_args = extra
        held.skip_clone = True
        return held

    def let_data_go(self, prompt):
        count = drop_image_data(prompt)
        if count:
            self.dropped["prompts"] += 1
            self.dropped["items"] += count

    def next_rid(self):
        self.count += 1
        return f"taaf-paused-{self.count}-{os.getpid()}"

    async def start(self, whole, tokens, sampling_params, key, kwargs, rid=None):
        rid = rid or self.next_rid()
        session = Session(key, rid, tokens, len(tokens))
        note_capture_session(rid, key, len(tokens))
        params = self.session_params(sampling_params, key)
        if getattr(self, "rope_gap_on", False):
            whole = self.rope_gap_start(session, whole, params)
        session.inputs.put_nowait((whole, params))
        whole = None
        stream = self.original(self.engine, feed(session.inputs, self.let_data_go), params, rid, **kwargs)

        async def pump():
            try:
                async for out in stream:
                    await session.outputs.put(out)
            except Exception as failure:
                await session.outputs.put(failure)
            finally:
                session.closed = True
                self.forget(session)
                await self.cancel_forks(session, "ended")
                await session.outputs.put(CLOSED)

        session.task = asyncio.ensure_future(pump())
        self.index(session)
        self.ensure_loop()
        return session

    def ensure_loop(self):
        if self.loop_task is None or self.loop_task.done():
            self.loop_task = asyncio.ensure_future(self.watch())

    async def watch(self):
        while self.sessions:
            await asyncio.sleep(TICK_S)
            now = time.monotonic()
            paused = [session for session in self.sessions.values() if not session.busy and not session.closed]
            paused.sort(key=lambda session: session.used)
            for session in paused:
                if now - session.used > self.idle:
                    await self.close(session, "idle")
            paused = [session for session in paused if not session.closed]
            while len(paused) > self.most:
                await self.close(paused.pop(0), "count")
            stats = _state["stats"]
            pressure = under_pressure(stats, len(paused), self.kv_high, self.kv_wait,
                                      read_memory() if memory_only() else None)
            if pressure and paused and now - stats["at"] < 5 * TICK_S:
                await self.close(paused.pop(0), "pressure")
            elif paused and seats_on():
                await self.seat_pressure(paused)
            if now - self.reported >= REPORT_S:
                self.report()

    async def seat_pressure(self, paused):
        blocked = read_seats()
        if not blocked:
            return 0
        taken = paused[:blocked]
        if not _seats["said"]:
            _seats["said"] = True
            say(f"seat pressure first fired ({SEATS_ENV}): {blocked} request(s) waited for a model-runner seat while "
                f"paused sessions held the rest, so the {len(taken)} least recent paused session(s) close, first "
                f"{taken[0].key} ({len(taken[0].tokens)} tokens); each tick closes as many as requests wait for a "
                f"seat, counted as closed_seats")
        for session in taken:
            await self.close(session, "seats")
        return len(taken)

    def tokenizer(self):
        held = getattr(self, "_tokenizer", False)
        if held is not False:
            return held
        try:
            held = getattr(self.engine, "tokenizer", None)
            held = held() if callable(held) and not hasattr(held, "decode") else held
        except Exception:
            held = None
        self._tokenizer = held if hasattr(held, "decode") else None
        return self._tokenizer

    def report(self):
        self.reported = time.monotonic()
        busy = sum(1 for session in self.sessions.values() if session.busy)
        held = sum(len(session.tokens) for session in self.sessions.values())
        pool = read_pool() if pool_on() else None
        shown = "" if pool is None else f"; {pool_line(pool)}"
        if self.parking:
            shown += f"; {len(self.parked)} parked ({sum(one['tokens'] for one in self.parked.values())} tokens)"
        say(f"tally {self.tally}; live {len(self.sessions)} ({busy} generating), {held} tokens held{shown}")
        self.report_memory()
        memtrace_tick("api", lambda: api_readings(self), census=lambda: session_census(self))

    def report_memory(self):
        before = process_memory()
        trim = None
        after = before
        if self.trim and before is not None:
            began = time.perf_counter()
            done = trim_heap()
            spent = (time.perf_counter() - began) * 1000
            after = process_memory() or before
            if done is not None:
                trim = (f"malloc_trim gave back {(before.get('RssAnon', 0) - after.get('RssAnon', 0)) / 2 ** 20:.0f} "
                        f"MiB of anon in {spent:.0f} ms")
        say(f"memory: {memory_line(after, mm_cache_reading(self.engine), trim, self.dropped)}")

    def bump(self, name, count=1):
        self.tally[name] = self.tally.get(name, 0) + count

    def held(self):
        return sum(len(session.tokens) for session in self.sessions.values())

    async def let_go(self, key):
        self.bump("release")
        session = self.by_key.get(key) if key else None
        if session is None or session.closed:
            self.bump("release_none")
            say(f"release {key}: no live session under this key")
            return 0
        if session.busy:
            self.bump("release_busy")
            say(f"release {key}: {session.rid} is generating, so it is left alone")
            return 0
        tokens = len(session.tokens)
        await self.close(session, "release")
        await self.release(session)
        self.bump("release_tokens", tokens)
        self.released[key] = tokens
        say(f"release {key}: closed {session.rid} after {session.turns} turn(s), {tokens} tokens let go; "
            f"{len(self.sessions)} live session(s) hold {self.held()} tokens")
        return tokens

    def returned(self, session, prompt, cached):
        cached = int(cached or 0)
        self.bump("returns")
        self.bump("return_tokens", len(prompt))
        self.bump("return_cached", cached)
        say(f"return {session.key}: {len(prompt)} prompt tokens after a release of {session.returning}, "
            f"{cached} from the prefix cache, {len(prompt) - cached} computed")

    async def engine_call(self, method, *args):
        try:
            answer = await self.engine.engine_core.call_utility_async(method, *args)
        except Exception as failure:
            return {"why": f"engine {type(failure).__name__}: {failure}"}
        return answer if isinstance(answer, dict) else {"why": f"engine answered {answer!r}"}

    async def park_key(self, key):
        self.bump("park")
        session = self.by_key.get(key) if key else None
        if session is None or session.closed or session.busy or session.trim.holes:
            why = "busy" if session is not None and session.busy and not session.closed else (
                "holed" if session is not None and not session.closed and session.trim.holes else (
                    "already_parked" if key in self.parked else "no_session"))
            self.bump("park_refused")
            say(f"park {key}: refused, {why}")
            return {"parked": False, "why": why, "key": key}
        gate = asyncio.Event()
        self.parking_now[key] = gate
        began = time.monotonic()
        session.busy = True
        try:
            result = dict(await self.engine_call("taaf_park", session.rid, key), key=key)
            session.busy = False
            if not result.get("parked"):
                self.bump("park_refused")
                say(f"park {key}: refused by the engine, {result.get('why')}; {session.rid} stays paused")
                return result
            tokens = len(session.tokens)
            await self.close(session, "park")
            await self.release(session)
            self.parked[key] = {"tokens": tokens, "boundary": int(result.get("boundary") or 0),
                                "at": time.monotonic()}
            self.bump("parked")
            self.bump("park_tokens", tokens)
            result["ms_total"] = round((time.monotonic() - began) * 1000.0, 1)
            say(f"park {key}: closed {session.rid} after {session.turns} turn(s), {tokens} tokens, boundary "
                f"{result.get('boundary')}, {result['ms_total']} ms; {len(self.parked)} parked, "
                f"{len(self.sessions)} live session(s) hold {self.held()} tokens")
            return result
        finally:
            session.busy = False
            self.parking_now.pop(key, None)
            gate.set()

    async def unpark(self, key, why):
        if not key or key not in self.parked:
            return None
        self.parked.pop(key)
        self.bump("unpark")
        result = await self.engine_call("taaf_park_drop", key)
        say(f"park {key}: dropped on {why}, engine {'let it go' if result.get('dropped') else 'held nothing'}")
        return result

    async def bring_back(self, key, rid):
        held = self.parked.pop(key)
        began = time.monotonic()
        deadline = began + self.park_wait
        while True:
            result = await self.engine_call("taaf_park_restore", key, rid)
            if result.get("restored") or result.get("why") != "no_room" or time.monotonic() >= deadline:
                break
            await asyncio.sleep(PARKING_RETRY_S)
        waited = time.monotonic() - began
        if not result.get("restored"):
            self.bump("resume_fallback")
            if result.get("why") == "no_room":
                await self.engine_call("taaf_park_drop", key)
            say(f"park resume {key}: nothing copied back ({result.get('why')}) after {waited:.1f} s; this turn "
                f"recomputes whatever the prefix cache no longer holds")
            return dict(held, restored=False, why=result.get("why"), waited_s=round(waited, 2))
        self.bump("resume")
        say(f"park resume {key} -> {rid}: {result.get('copied')} block(s) copied back, {result.get('in_place')} "
            f"still cached, {result.get('ms')} ms after {waited:.2f} s of waiting; boundary "
            f"{result.get('boundary')} of {held['tokens']} parked tokens")
        return dict(held, restored=True, waited_s=round(waited, 2),
                    **{name: result.get(name) for name in ("copied", "in_place", "bytes", "ms")})

    def resumed(self, session, prompt, cached):
        cached = int(cached or 0)
        info = session.resuming
        boundary = int(info.get("boundary") or 0)
        self.bump("resume_tokens", len(prompt))
        self.bump("resume_cached", cached)
        short = f"; {boundary - cached} short of the parked boundary" if cached < boundary else ""
        say(f"park resumed {session.key}: {len(prompt)} prompt tokens, {cached} from the prefix cache "
            f"(parked boundary {boundary}, copy {'made' if info.get('restored') else 'not made'}), "
            f"{len(prompt) - cached} computed{short}")

    async def park_reply(self, prompt, sampling_params, request_id, kwargs):
        from vllm.outputs import CompletionOutput
        marks = getattr(sampling_params, "extra_args", None) or {}
        result = await self.park_key(kwargs.get("session_id") or marks.get(KEY))
        text = json.dumps(result, sort_keys=True, default=str)
        last = None
        async for out in self.original(self.engine, prompt, sampling_params, request_id, **kwargs):
            last = out
        if last is None:
            return
        piece = last.outputs[0]
        answer = copy.copy(last)
        answer.outputs = [CompletionOutput(index=0, text=text, token_ids=list(piece.token_ids),
                                           cumulative_logprob=None, logprobs=None,
                                           finish_reason=piece.finish_reason or "stop",
                                           stop_reason=piece.stop_reason)]
        answer.finished = True
        yield answer

    def shown(self, ids):
        decoder = self.tokenizer()
        if decoder is None:
            return repr(list(ids))
        try:
            return repr(decoder.decode(list(ids), skip_special_tokens=False))
        except Exception as failure:
            return f"{list(ids)} (decode failed: {type(failure).__name__})"

    def explain(self, kind, session, prompt):
        now = time.monotonic()
        if self.diff_at is not None and now - self.diff_at < self.diff_every:
            self.diff_skipped += 1
            return
        self.diff_at = now
        skipped, self.diff_skipped = self.diff_skipped, 0
        history = session.tokens
        at = lcp(history, prompt)
        low, high = max(0, at - DIFF_SPAN), at + DIFF_SPAN
        view = "" if session.view is None else (
            f"; client view {len(session.view)} tokens, prompt shares {lcp(session.view, prompt)} of them")
        say(f"diff {kind} {session.rid}: history {len(history)} tokens (reply from {session.reply_start}), "
            f"prompt {len(prompt)}; first difference at {at} ({at - session.reply_start:+d} from the reply "
            f"start, {len(history) - at} before the history's end){view}; history[{low}:{high}] "
            f"{list(history[low:high])} {self.shown(history[low:high])}; prompt[{low}:{high}] "
            f"{list(prompt[low:high])} {self.shown(prompt[low:high])}; {skipped} line(s) skipped since the last")

    def fork_mark(self, prompt, sampling_params, kwargs):
        marks = getattr(sampling_params, "extra_args", None) or {}
        key = fork_key(kwargs, marks)
        session = self.by_key.get(key) if key is not None else None
        tokens = prompt.get("prompt_token_ids") if isinstance(prompt, dict) else None
        why = None
        if key is None:
            why = "the session id names no main key"
        elif key in self.parked:
            why = f"main {key} is parked"
        elif session is None or session.closed:
            why = f"no live paused session for {key}"
        elif not tokens:
            why = "the prompt carries no token ids"
        else:
            mapped = self.fork_view(session, tokens)
            if mapped is not None:
                tokens = mapped
                prompt["prompt_token_ids"] = mapped
            common = lcp(session.tokens, tokens)
            if common < session.reply_start:
                why = (f"shares {common} of the session's {len(session.tokens)} tokens, short of its last prompt "
                       f"({session.reply_start})")
            elif session.trim.holes and common < len(session.tokens):
                why = (f"shares {common} of the holed session's {len(session.tokens)} tokens; a branch of a "
                       f"session with a hole must extend its whole view")
        held = sampling_params.clone()
        extra = dict(getattr(held, "extra_args", None) or {})
        extra[FORK_MARK] = 1
        if key is not None:
            extra[FORK_KEY] = str(key)
        if why is None:
            extra[FORK_OF] = session.rid
            if session.trim.holes:
                extra[FORK_HOLE] = [len(session.tokens), session.trim.pad, session.trim.head]
        held.extra_args = extra
        return held, key, session if why is None else None, why

    def fork_view(self, session, tokens):
        history = session.tokens
        if session.view is None or session.trim.holes or lcp(history, tokens) >= len(history):
            return None
        kind, delta, viewed = plan_view(session.view, history, session.reply_start, tokens, self.stops, self.slack)
        if kind != "reply" or not delta:
            return None
        self.bump("fork_view")
        return list(history) + list(delta)

    async def side(self, prompt, sampling_params, request_id, kwargs):
        self.tally["branch"] += 1
        size = len(prompt.get("prompt_token_ids") or ()) if isinstance(prompt, dict) else 0
        cached = 0
        key = main = why = None
        if self.fork:
            sampling_params, key, main, why = self.fork_mark(prompt, sampling_params, kwargs)
            if main is not None:
                self.bump("fork_marked")
                self.forks.setdefault(key, {})[request_id] = main.rid
            else:
                self.bump("fork_fallback")
                say(f"fork fallback {request_id}: {why}; it runs as an ordinary request on the prefix cache; "
                    f"{self.tally['fork_fallback']} fallback(s) here so far, which the engine's FORK_BRANCH lines "
                    f"do not count (they count only marked branches)")
        try:
            async for out in self.original(self.engine, prompt, sampling_params, request_id, **kwargs):
                cached = int(getattr(out, "num_cached_tokens", 0) or 0)
                yield out
        finally:
            if main is not None:
                held = self.forks.get(key)
                if held is not None:
                    held.pop(request_id, None)
                    if not held:
                        self.forks.pop(key, None)
        self.tally["branch_tokens"] += size
        self.tally["branch_cached"] += cached
        shown = "" if not self.fork else (f"; shares {main.rid}'s KV" if main is not None else "; fell back")
        say(f"branch {request_id}: {size} prompt tokens, {cached} from the prefix cache; "
            f"{len(self.sessions)} paused session(s) untouched{shown}")

    async def generate(self, prompt, sampling_params, request_id, kwargs):
        marks = getattr(sampling_params, "extra_args", None) or {}
        if self.parking and marks.get(PARKING_MARK):
            async for out in self.park_reply(prompt, sampling_params, request_id, kwargs):
                yield out
            return
        if marks.get(PAUSE_MARK):
            async for out in self.pause_reply(prompt, sampling_params, request_id, kwargs):
                yield out
            return
        if marks.get(RELEASE_MARK):
            if self.parking:
                await self.unpark(kwargs.get("session_id") or marks.get(KEY), "release")
            key = kwargs.get("session_id") or marks.get(KEY)
            freed = await self.let_go(key)
            async for out in self.release_reply(prompt, request_id, key, freed):
                yield out
            return
        if marks.get(HOT_SPANS_MARK):
            async for out in self.hot_reply(prompt, sampling_params, request_id, kwargs):
                yield out
            return
        if (self.branch or self.fork) and branched(sampling_params, kwargs):
            async for out in self.side(prompt, sampling_params, request_id, kwargs):
                yield out
            return
        tokens = None if marks.get(ONE_SHOT_MARK) else self.eligible(prompt, sampling_params)
        if tokens is None:
            self.tally["bypass"] += 1
            async for out in self.original(self.engine, prompt, sampling_params, request_id, **kwargs):
                yield out
            return
        extra = getattr(sampling_params, "extra_args", None) or {}
        key = kwargs.get("session_id") or extra.get(KEY)
        if key is None and probe(tokens, sampling_params):
            self.tally["probe"] = self.tally.get("probe", 0) + 1
            async for out in self.original(self.engine, prompt, sampling_params, request_id, **kwargs):
                yield out
            return
        if self.parking and key is not None and key in self.parking_now:
            await self.parking_now[key].wait()
        images = images_of(prompt)
        kind, session, delta = self.match(tokens, key, images)
        if self.rewind and key is not None:
            kind, session, delta = await self.rewind_turn(bool(marks.get(REWIND_MARK)), key, prompt, tokens, images,
                                                          kind, session, delta)
        if self.trim_hole and kind == "stale" and session is not None and not session.busy:
            kind, session, delta = await self.trim_hole_turn(session, tokens, prompt, sampling_params, kind, delta)
        if self.diff_every is not None and session is not None and kind in ("reply", "stale"):
            self.explain(kind, session, tokens)
        chunk = None
        if session is not None and kind in ("extend", "reply", "tail"):
            chunk = cut(prompt, len(tokens) - len(delta))
            if chunk is None:
                await self.close(session, "miss")
                kind, session = "miss", None
        if session is not None and session.after_pause is not None and kind in ("stale", "miss"):
            self.bump("pause_miss")
            say(f"pause continue {key}: MISS ({kind}), the client's turn does not continue the {session.after_pause} "
                f"tokens kept at the pause; {len(tokens)} prompt tokens go through the stale path")
            session.after_pause = None
        if kind == "stale" and session is not None:
            if self.exact:
                self.tally["rewind"] += 1
                self.tally["rewind_kept"] += lcp(session.tokens, tokens)
            await self.close(session, "miss")
            if self.exact:
                await self.release(session)
            kind, session = "miss", None
        limit = int(getattr(sampling_params, "max_tokens", 0) or 0)
        if session is not None and self.max_len and self.engine_len(session) + len(delta) + limit > self.max_len:
            await self.close(session, "miss")
            kind, session = "miss", None
        clean = {name: value for name, value in kwargs.items()
                 if name in ("lora_request", "trace_headers", "priority", "data_parallel_rank",
                             "tokenization_kwargs", "session_id")}
        if session is None:
            self.tally["fresh"] += 1
            if getattr(self, "cause_log", False):
                self.fresh_cause(key, tokens)
            rid = resumed = None
            if self.parking and key is not None and key in self.parked:
                rid = self.next_rid()
                resumed = await self.bring_back(key, rid)
            session = await self.start(prompt, tokens, sampling_params, key, clean, **({"rid": rid} if rid else {}))
            session.resuming = resumed
            session.images = images
            if key is not None and key in self.released:
                session.returning = self.released.pop(key)
            fed = len(tokens)
            appended = []
            base = []
        else:
            self.tally[kind if kind != "tail" else "reply"] += 1
            base = session.tokens if lean_on() else list(session.tokens)
            if kind == "tail":
                base = base[:-1]
                self.tally["stop_tail_tokens"] += len(delta)
            appended = list(delta)
            session.images = session.images + images_of(chunk, len(tokens) - len(delta))
            params = self.trim_params(session, sampling_params)
            if kind == "tail":
                extra = dict(getattr(params, "extra_args", None) or {})
                extra[DROP_LAST_MARK] = 1
                params.extra_args = extra
            session.inputs.put_nowait((chunk, params))
            chunk = None
            self.let_data_go(prompt)
            fed = min(len(tokens), len(delta) + (0 if kind == "tail" else 1))
            if session.after_pause is not None:
                self.bump("pause_continue")
                self.bump("pause_continue_computed", fed)
                self.bump("pause_continue_reused", len(tokens) - fed)
                say(f"pause continue {key}: {kind} after {session.after_pause} tokens kept at the pause; "
                    f"{len(tokens)} prompt tokens, {len(tokens) - fed} from cache, {fed} computed")
                session.after_pause = None
        self.tally["turns"] += 1
        self.tally["fed"] += fed
        self.tally["reused"] += len(tokens) - fed
        session.busy = True
        session.turns += 1
        async for out in self.turn(session, tokens, fed, joined(base, appended) if len(base) else (tokens if lean_on() else list(tokens)),
                                   sampling_params, request_id):
            yield out

    async def turn(self, session, prompt, fed, before, sampling_params, request_id):
        from vllm.outputs import CompletionOutput, RequestOutput
        from vllm.sampling_params import RequestOutputKind
        kind = getattr(sampling_params, "output_kind", RequestOutputKind.CUMULATIVE)
        want_logprobs = getattr(sampling_params, "logprobs", None) is not None
        decoder = self.tokenizer()
        token_ids, text, logprobs = [], [], []
        reason = None
        cached = None
        paused = False
        shown_prompt = prompt if lean_on() else list(prompt)
        try:
            while True:
                out = await session.outputs.get()
                if out is CLOSED:
                    raise RuntimeError(f"paused session {session.rid} ended inside a turn")
                if isinstance(out, BaseException):
                    raise out
                if cached is None:
                    cached = getattr(out, "num_cached_tokens", 0)
                piece = out.outputs[0]
                token_ids.extend(piece.token_ids)
                text.append(piece.text or "")
                if want_logprobs and piece.logprobs:
                    logprobs.extend(list(piece.logprobs))
                done = piece.finish_reason is not None
                reason = piece.finish_reason if done else reason
                paused = paused or (done and getattr(piece, "stop_reason", None) == PAUSE_STOP)
                if kind == RequestOutputKind.FINAL_ONLY and not done:
                    continue
                spec = turn_spec_metrics(self.engine, session) if done else None
                if kind == RequestOutputKind.DELTA:
                    shown = CompletionOutput(index=0, text=piece.text or "", token_ids=list(piece.token_ids),
                                             cumulative_logprob=None,
                                             logprobs=list(piece.logprobs) if want_logprobs and piece.logprobs else None,
                                             finish_reason=piece.finish_reason, stop_reason=piece.stop_reason,
                                             **({"spec_decode_metrics": spec} if spec is not None else {}))
                else:
                    whole = "".join(text)
                    if decoder is not None and done:
                        whole = decoder.decode(token_ids, skip_special_tokens=bool(
                            getattr(sampling_params, "skip_special_tokens", True)))
                    shown = CompletionOutput(index=0, text=whole, token_ids=list(token_ids),
                                             cumulative_logprob=None,
                                             logprobs=list(logprobs) if want_logprobs else None,
                                             finish_reason=piece.finish_reason, stop_reason=piece.stop_reason,
                                             **({"spec_decode_metrics": spec} if spec is not None else {}))
                result = copy.copy(out)
                result.request_id = request_id
                result.prompt_token_ids = shown_prompt
                result.prompt_logprobs = None
                result.outputs = [shown]
                result.finished = done
                result.num_cached_tokens = max(0, len(prompt) - fed)
                yield result
                if done:
                    break
        except BaseException:
            await self.close(session, "error")
            session.busy = False
            raise
        if session.returning is not None:
            self.returned(session, prompt, cached)
            session.returning = None
        if session.resuming is not None:
            self.resumed(session, prompt, cached)
            session.resuming = None
        session.tokens = joined(before, token_ids)
        session.reply_start = len(before)
        if self.client_view and not self.exact:
            session.view = np.asarray(prompt, dtype=np.int32)
        session.used = time.monotonic()
        session.busy = False
        if paused:
            self.bump("pause_kept")
            self.bump("pause_kept_tokens", len(token_ids))
            session.after_pause = len(token_ids)
            say(f"pause {session.key}: {session.rid} kept {len(token_ids)} generated tokens in KV and waits paused "
                f"for the client's next turn")
            return
        if not token_ids or token_ids[-1] not in self.stops:
            if self.resume and token_ids and reason == "length":
                self.tally["kept_length"] += 1
                return
            await self.close(session, "unstopped")


def under_pressure(stats, paused, kv_high, kv_wait, memory=None):
    queued = stats["waiting"] - paused if memory is None else memory
    return stats["kv"] >= kv_high or (stats["kv"] >= kv_wait and queued > 0)


def memory_only():
    return str(os.environ.get(MEMORY_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def memory_path():
    said = str(os.environ.get(MEMORY_FILE_ENV) or "").strip()
    if said:
        return said
    import tempfile
    folder = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    return os.path.join(folder, MEMORY_FILE)


def waiting_for_memory(scheduler):
    from vllm.v1.request import RequestStatus
    live = (RequestStatus.WAITING, RequestStatus.PREEMPTED)
    return sum(1 for queue in (scheduler.waiting, scheduler.skipped_waiting) for request in queue
               if request.status in live)


def memory_blocked(scheduler):
    _memory["blocked"] = max(_memory["blocked"], waiting_for_memory(scheduler))
    return _memory["blocked"]


def write_memory(count, path=None):
    path = path or memory_path()
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(f"{int(count)} {time.time():.3f}\n")
    os.replace(temporary, path)
    return path


def publish_memory(scheduler, now=None, path=None):
    return publish_count(_memory, path, now)


def publish_count(state, path, now=None):
    now = time.monotonic() if now is None else now
    state["peak"] = max(state["peak"], state["blocked"])
    state["blocked"] = 0
    since = now - state["at"]
    changed = state["peak"] != state["written"]
    if since < MEMORY_WRITE_S or (not changed and since < MEMORY_REFRESH_S):
        return None
    count = state["peak"]
    write_memory(count, path)
    state["written"] = count
    state["peak"] = 0
    state["at"] = now
    return count


def read_memory(path=None, now=None):
    try:
        with open(path or memory_path(), encoding="utf-8") as handle:
            count, stamp = handle.read().split()
    except FileNotFoundError:
        return 0
    now = time.time() if now is None else now
    return int(count) if now - float(stamp) <= MEMORY_FRESH_S else 0


def seats_on():
    return str(os.environ.get(SEATS_ENV) or "1").strip().lower() not in ("0", "false", "no", "off")


def seats_path():
    return memory_path() + SEATS_SUFFIX


def seats_blocked(scheduler):
    if not scheduler.num_waiting_for_streaming_input:
        return 0
    _seats["blocked"] = max(_seats["blocked"], waiting_for_memory(scheduler))
    return _seats["blocked"]


def publish_seats(scheduler, now=None, path=None):
    return publish_count(_seats, path or seats_path(), now)


def read_seats(path=None, now=None):
    return read_memory(path or seats_path(), now)


def pool_on():
    return str(os.environ.get(POOL_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def pool_path():
    said = str(os.environ.get(POOL_FILE_ENV) or "").strip()
    if said:
        return said
    import tempfile
    folder = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    return os.path.join(folder, POOL_FILE)


def pool_counts(scheduler):
    manager = scheduler.kv_cache_manager
    device = manager.block_pool
    row = {"gpu_total": int(device.num_gpu_blocks) - 1, "gpu_free": int(device.get_num_free_blocks())}
    for one in getattr(getattr(manager, "coordinator", None), "single_type_managers", None) or ():
        if type(one).__name__ == "HiSparseSourceManager":
            row.update(host_total=int(one.block_pool.num_gpu_blocks) - 1,
                       host_free=int(one.block_pool.get_num_free_blocks()), block_tokens=int(one.block_size))
            break
    return row


def pool_line(row):
    parts = [f"GPU pool {row['gpu_total'] - row['gpu_free']} of {row['gpu_total']} blocks used"]
    if "host_total" in row:
        parts.insert(0, f"host pool {row['host_total'] - row['host_free']} of {row['host_total']} blocks of "
                        f"{row['block_tokens']} tokens used")
    return ", ".join(parts)


def park_probe_on():
    return str(os.environ.get(PARK_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def group_kinds(manager):
    groups = list(getattr(getattr(manager, "kv_cache_config", None), "kv_cache_groups", None) or ())
    kinds = []
    for index, one in enumerate(manager.coordinator.single_type_managers):
        role = getattr(getattr(groups[index], "role", None), "value", None) if index < len(groups) else None
        name = type(one).__name__
        if role == "hisparse_indexer":
            kinds.append("indexer")
        else:
            kinds.append(PARK_KINDS.get(name, name))
    return kinds


def group_holdings(manager, kinds, rid):
    rows = {}
    for kind, one in zip(kinds, manager.coordinator.single_type_managers):
        row = rows.setdefault(kind, [0, 0, 0])
        row[0] += 1
        for block in one.req_to_blocks.get(rid) or ():
            if block.is_null:
                continue
            row[1] += 1
            row[2] += block.block_hash is not None
    return rows


def holdings_line(rows):
    return ", ".join(f"{kind} {row[1]} block(s) in {row[0]} group(s), {row[2]} hashed"
                     for kind, row in rows.items())


def pool_free(manager):
    row = {"gpu": int(manager.block_pool.get_num_free_blocks())}
    for one in manager.coordinator.single_type_managers:
        if type(one).__name__ == "HiSparseSourceManager":
            row["host"] = int(one.block_pool.get_num_free_blocks())
            break
    return row


def group_hits(manager, kinds, request):
    finder = getattr(manager.coordinator, "find_longest_cache_hit_per_group", None)
    if finder is None:
        return None
    _, lengths = finder(request.block_hashes, request.num_tokens - 1)
    found = {}
    for kind, length in zip(kinds, lengths):
        found.setdefault(kind, []).append(int(length))
    return found


def hits_line(found):
    if found is None:
        return "this coordinator has no per-group lookup"
    return ", ".join(f"{kind} {min(values)}" + (f"..{max(values)} over {len(values)}" if len(values) > 1 else "")
                     for kind, values in found.items())


def attach_park_probe(scheduler):
    manager = scheduler.kv_cache_manager
    kinds = group_kinds(manager)
    free, pop, lookup = manager.free, manager.pop_blocks_for_free, manager.get_computed_blocks

    def watched(request):
        return str(getattr(request, "request_id", "")).startswith(PARK_PREFIX)

    def note(request):
        try:
            return holdings_line(group_holdings(manager, kinds, request.request_id))
        except Exception as failure:
            return f"holdings unreadable ({type(failure).__name__}: {failure})"

    def park_free(request):
        if not watched(request):
            return free(request)
        held = note(request)
        before = pool_free(manager)
        free(request)
        after = pool_free(manager)
        moved = ", ".join(f"{name} pool free {before[name]} -> {after[name]}" for name in before)
        say(f"park probe free {request.request_id} ({getattr(request.status, 'name', request.status)}): "
            f"{request.num_tokens} tokens, {request.num_computed_tokens} computed; {held}; {moved}")

    def park_pop(request):
        if watched(request):
            say(f"park probe deferred free {request.request_id} "
                f"({getattr(request.status, 'name', request.status)}): {request.num_tokens} tokens, "
                f"{request.num_computed_tokens} computed; {note(request)}")
        return pop(request)

    def park_lookup(request):
        made = lookup(request)
        if watched(request) and request.num_tokens >= PARK_MIN_TOKENS:
            try:
                shown = hits_line(group_hits(manager, kinds, request))
            except Exception as failure:
                shown = f"per-group lookup failed ({type(failure).__name__}: {failure})"
            say(f"park probe lookup {request.request_id}: {request.num_tokens} tokens, engine hit {made[1]}; "
                f"each group alone: {shown}")
        return made

    manager.free = park_free
    manager.pop_blocks_for_free = park_pop
    manager.get_computed_blocks = park_lookup
    counts = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    say(f"park probe armed on the scheduler: groups {', '.join(f'{kind} {count}' for kind, count in counts.items())}; "
        f"a freed or preempted {PARK_PREFIX}* request prints the blocks each group held and how many carry a "
        f"prefix-cache hash, and one of at least {PARK_MIN_TOKENS} tokens prints, on admission, the hit each "
        f"group would give alone beside the engine's hit")
    return kinds


def arm_park_probe(scheduler):
    if _park["on"] is None:
        _park["on"] = park_probe_on()
    if not _park["on"] or _park["armed"] is not None:
        return _park["armed"]
    try:
        _park["armed"] = attach_park_probe(scheduler)
    except Exception as failure:
        _park["armed"] = False
        say(f"park probe not armed: {type(failure).__name__}: {failure}")
    return _park["armed"]


def publish_pool(scheduler, now=None, path=None):
    arm_park_probe(scheduler)
    now = time.monotonic() if now is None else now
    if _pool["failed"] is not None or now - _pool["at"] < POOL_WRITE_S:
        return None
    _pool["at"] = now
    path = path or pool_path()
    try:
        row = dict(pool_counts(scheduler), stamp=round(time.time(), 3))
        publish_ledger(scheduler, row, now)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        os.replace(temporary, path)
    except Exception as failure:
        _pool["failed"] = f"{type(failure).__name__}: {failure}"
        say(f"pool reading stops for good: {_pool['failed']}")
        return None
    if not _pool["said"]:
        _pool["said"] = True
        say(f"pool reading: {pool_line(row)}; written to {path} every {POOL_WRITE_S:g} s")
    return row


def read_pool(path=None, now=None):
    try:
        with open(path or pool_path(), encoding="utf-8") as handle:
            row = json.loads(handle.read())
    except (OSError, ValueError):
        return None
    now = time.time() if now is None else now
    if not isinstance(row, dict) or now - float(row.get("stamp") or 0.0) > POOL_FRESH_S:
        return None
    return row


def ledger_on():
    return pool_on() and str(os.environ.get(LEDGER_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def is_kind(manager, *names):
    return any(klass.__name__ in names for klass in type(manager).__mro__)


def ledger_parts(kv):
    managers = list(kv.coordinator.single_type_managers)
    device = kv.block_pool
    gpu = [one for one in managers if one.block_pool is device]
    return {"managers": managers, "kinds": group_kinds(kv), "gpu": gpu,
            "host": [one for one in managers if one.block_pool is not device],
            "grows": [one for one in gpu if not is_kind(one, *LEDGER_CONSTANT)],
            "mamba": [one for one in gpu if is_kind(one, "MambaManager")],
            "hot": [one for one in managers if is_kind(one, "HiSparseHotManager")],
            "resident": [one for one in managers if is_kind(one, "HiSparseResidentManager")],
            "hs": next((one.coordinator for one in managers
                        if hasattr(getattr(one, "coordinator", None), "request_states")), None),
            "max_len": int(getattr(kv, "max_model_len", 0) or 0)}


def predicted(manager, rid, target, computed):
    had = rid in manager.req_to_blocks
    try:
        return int(manager.get_num_blocks_to_allocate(rid, target, (), computed, computed, target,
                                                      apply_admission_cap=True))
    finally:
        if not had and not manager.req_to_blocks.get(rid, True):
            manager.req_to_blocks.pop(rid, None)


def fresh_table(parts):
    top = parts["max_len"] or LEDGER_FRESH_STEP
    rows = []
    for tokens in range(LEDGER_FRESH_STEP, top + LEDGER_FRESH_STEP, LEDGER_FRESH_STEP):
        tokens = min(tokens, top)
        rows.append([tokens, sum(predicted(one, LEDGER_PROBE, tokens, 0) for one in parts["gpu"])])
        for one in parts["managers"]:
            getattr(one, "num_cached_block", {}).pop(LEDGER_PROBE, None)
    return rows


def request_base(request):
    name = getattr(getattr(request, "status", None), "name", "")
    return int(request.num_tokens if name == "WAITING_FOR_STREAMING_REQ" else request.num_prompt_tokens)


def request_limit(request):
    params = getattr(request, "sampling_params", None)
    return int(getattr(params, "max_tokens", 0) or getattr(request, "max_tokens", 0) or 0)


def holds(parts, rid):
    return any(one.req_to_blocks.get(rid) for one in parts["managers"])


def request_growth(parts, rid, request, held, horizons):
    base, limit, computed = request_base(request), request_limit(request), int(request.num_computed_tokens)
    use = parts["grows"] if held else parts["gpu"]
    gpu, host = {}, {}
    for name, cap in [("worst", limit)] + [(str(value), min(limit, value)) for value in horizons]:
        target = base + cap
        if parts["max_len"]:
            target = min(target, parts["max_len"])
        target = max(target, int(request.num_tokens))
        gpu[name] = sum(predicted(one, rid, target, computed) for one in use)
        host[name] = sum(max(0, -(-target // one.block_size) - len(one.req_to_blocks.get(rid) or ()))
                         for one in parts["host"])
    return gpu, host


def hot_missing(parts, rid):
    return sum(one.get_num_required_blocks(rid) for one in parts["hot"] if rid in getattr(one, "hot_required", ()))


def reclaimable(parts, rid):
    hs = parts["hs"]
    state = None if hs is None else hs.request_states.get(rid)
    if state is None or not parts["resident"]:
        return 0
    pinned = len(state.pinned_clean) * len(parts["resident"])
    ready = all(one.has_hot(rid) or rid in getattr(one, "hot_required", ()) for one in parts["hot"])
    cost = 0 if ready else sum(int(one.blocks_per_request) for one in parts["hot"])
    return max(0, pinned - cost)


def step_blocks(parts, requests):
    for rid, request in requests:
        lists = [one.req_to_blocks.get(rid) for one in parts["mamba"]]
        if lists and all(lists):
            computed = int(request.num_computed_tokens)
            return sum(predicted(one, rid, (len(blocks) + 1) * one.block_size, computed)
                       for one, blocks in zip(parts["mamba"], lists))
    return len(parts["mamba"])


def census(parts, rid):
    rows = {}
    for kind, one in zip(parts["kinds"], parts["managers"]):
        rows[kind] = rows.get(kind, 0) + sum(1 for block in one.req_to_blocks.get(rid) or () if not block.is_null)
    return rows


def ledger_reading(scheduler, horizons=None):
    horizons = LEDGER_HORIZONS if horizons is None else horizons
    kv = scheduler.kv_cache_manager
    parts = _ledger.get("parts")
    if parts is None:
        parts = _ledger["parts"] = ledger_parts(kv)
    if _ledger["fresh"] is None:
        _ledger["fresh"] = fresh_table(parts)
    live = [(rid, request) for rid, request in scheduler.requests.items()
            if not (getattr(request, "is_finished", None) and request.is_finished())]
    grow = dict.fromkeys(["worst"] + [str(value) for value in horizons], 0)
    host_grow = dict(grow)
    counts = {"running": 0, "paused": 0, "waiting": 0}
    missing = reclaim = limit = 0
    rows = []
    for rid, request in live:
        held = holds(parts, rid)
        gpu, host = request_growth(parts, rid, request, held, horizons)
        for name in grow:
            grow[name] += gpu[name]
            host_grow[name] += host[name]
        status = getattr(getattr(request, "status", None), "name", "")
        counts["running" if status == "RUNNING" else "paused" if status == "WAITING_FOR_STREAMING_REQ" else "waiting"] += 1
        lack, back = hot_missing(parts, rid), reclaimable(parts, rid)
        missing += lack
        reclaim += back
        limit = max(limit, request_limit(request))
        rows.append((rid, request, status, held, gpu["worst"], lack, back))
    hot = sum(int(one.blocks_per_request) for one in parts["hot"])
    hs = parts["hs"]
    return {"grow": grow, "host_grow": host_grow, "hot_missing": missing, "reclaim": reclaim,
            "hot_blocks": hot, "step_blocks": step_blocks(parts, live), "max_tokens": limit,
            "watermark": int(getattr(hs, "transition_watermark", 0) or 0), "fresh": _ledger["fresh"],
            **counts}, rows, parts


def census_on():
    return str(os.environ.get(CENSUS_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def age_bin(seconds):
    return sum(seconds >= edge for edge in AGE_EDGES)


def block_census(managers, kinds, device, live, coordinator=None, seen_tails=None, clock=None):
    live_ids, orphan_ids, orphan_rids = {}, {}, {}
    owner_at = {}
    tails, retained, pending = set(), set(), set()
    for kind, manager in zip(kinds, managers):
        if getattr(manager, "block_pool", None) is not device:
            continue
        for rid, blocks in (getattr(manager, "req_to_blocks", None) or {}).items():
            held = [block.block_id for block in blocks or () if not block.is_null]
            if not held:
                continue
            if rid in live:
                live_ids.setdefault(kind, set()).update(held)
                if clock is not None:
                    seen = clock["sched"].get(rid)
                    if seen is not None:
                        for block_id in held:
                            owner_at[block_id] = max(owner_at.get(block_id, seen), seen)
            else:
                orphan_ids.setdefault(kind, set()).update(held)
                orphan_rids.setdefault(kind, set()).add(rid)
        tails.update(block.block_id for block in (getattr(manager, "_retained_tail_blocks", None) or {}).values()
                     if block is not None and not block.is_null)
        for rid, index in (getattr(manager, "_retained_state_block_idx", None) or {}).items():
            blocks = (getattr(manager, "req_to_blocks", None) or {}).get(rid) or ()
            if index is not None and index < len(blocks) and not blocks[index].is_null:
                retained.add(blocks[index].block_id)
        pending.update(copy.block_id for _, copy in getattr(manager, "_pending_cow_copies", None) or ()
                       if copy is not None and not copy.is_null)
    copies, spills = set(), set()
    if coordinator is not None:
        for blocks in (getattr(coordinator, "copies", None) or {}).values():
            copies.update(block.block_id for block in blocks)
        for spill in (getattr(coordinator, "pending_spills", None) or {}).values():
            spills.update(block.block_id for block in getattr(spill, "resident_blocks", ()) or ())
    hashed = set(getattr(device, "cached_block_hashes_by_block", None) or ())
    hashed.update(block.block_id for block in device.blocks if getattr(block, "block_hash", None) is not None)
    watched = set(getattr(device, "_reuse_watchers", None) or ())
    null = getattr(device, "null_block", None)
    null_id = None if null is None else null.block_id
    refd, free = {}, {}

    who = dict.fromkeys(hashed, "hashed")
    who.update(dict.fromkeys(copies, "copy"))
    who.update(dict.fromkeys(spills, "spill"))
    for kind, ids in orphan_ids.items():
        who.update(dict.fromkeys(ids, "orphan_" + kind))
    for kind, ids in live_ids.items():
        who.update(dict.fromkeys(ids, "live_" + kind))
    used = set()
    for ids in live_ids.values():
        used |= tails & ids
    who.update(dict.fromkeys(pending, "pending_cow"))
    who.update(dict.fromkeys(retained, "retained_state"))
    who.update(dict.fromkeys(tails - used, "tail_copy"))

    shared = 0
    ages, since = {}, {}
    previous = clock["since"] if clock is not None else {}

    def stamp(tag, block_id):
        if clock is None:
            return
        now = clock["now"]
        if tag.startswith("live_") and block_id in owner_at:
            idle = now - owner_at[block_id]
        else:
            held = previous.get(block_id)
            entered = held[1] if held is not None and held[0] == tag else now
            since[block_id] = (tag, entered)
            idle = now - entered
        row = ages.setdefault(tag, [0] * (len(AGE_EDGES) + 1))
        row[age_bin(idle)] += 1

    for block in device.blocks:
        if block.block_id == null_id:
            continue
        if block.ref_cnt > 0:
            name = who.get(block.block_id, "unknown")
            refd[name] = refd.get(name, 0) + 1
            shared += block.ref_cnt > 1
            stamp(name, block.block_id)
        else:
            name = who.get(block.block_id, "unknown")
            tag = "unpinned_" + name if block.block_id in watched else name
            free[tag] = free.get(tag, 0) + 1
            stamp("free_" + tag, block.block_id)
    extra = {}
    if clock is not None:
        clock["since"] = since
        extra["ages"] = dict(sorted(ages.items()))
    return {**extra, "refd": dict(sorted(refd.items())), "refd_total": sum(refd.values()), "shared": shared,
            "free": dict(sorted(free.items())), "free_total": sum(free.values()),
            "tails": {"held": len(tails), "used": len(used), "new": len(tails - seen_tails) if seen_tails is not None
                      else len(tails)},
            "tail_ids": tails,
            "orphan_rids": {kind: len(rids) for kind, rids in sorted(orphan_rids.items())},
            "orphan_sample": sorted({str(rid)[:40] for rids in orphan_rids.values() for rid in rids})[:CENSUS_SAMPLE]}


def census_log(scheduler, parts):
    kv = scheduler.kv_cache_manager
    live = {rid for rid, request in scheduler.requests.items()
            if not (getattr(request, "is_finished", None) and request.is_finished())}
    clock = _ledger.setdefault("census_clock", {"sched": {}, "since": {}})
    clock["now"] = time.monotonic()
    clock["sched"] = {rid: at for rid, at in clock["sched"].items() if rid in live}
    reading = block_census(parts["managers"], parts["kinds"], kv.block_pool, live, parts.get("hs"),
                           _ledger.get("census_tails"), clock)
    _ledger["census_tails"] = reading.pop("tail_ids")
    if not _ledger.get("census_said"):
        _ledger["census_said"] = True
        say(f"census armed ({CENSUS_ENV}=1): beside each ledger line, every GPU pool block is put in one bucket by "
            "who holds it: a live request's table by group, a table left under a request id the scheduler no longer "
            "runs, a pending spill, a recorded GPU copy, a prefix hash only, or unknown; free blocks are split the "
            "same way, with unpinned resident pages that are still read marked; 'ages' bins every block by seconds "
            f"idle at edges {list(AGE_EDGES)}: a live table's block by the last time its request was running, any "
            "other block by how long it has sat in its bucket")
    say("census " + json.dumps(reading, separators=(",", ":")))
    return reading


def ledger_log(row, rows, parts):
    held = {}
    each = []
    for rid, request, status, holding, worst, lack, back in rows:
        kinds = census(parts, rid) if holding else {}
        for kind, count in kinds.items():
            held[kind] = held.get(kind, 0) + count
        each.append([rid, status, int(request.num_tokens), int(getattr(request, "num_output_tokens", 0) or 0),
                     request_limit(request), sum(count for kind, count in kinds.items() if kind != "host"),
                     kinds.get("host", 0), back, lack, worst])
    used = row["gpu_total"] - row["gpu_free"]
    shown = {key: value for key, value in row["ledger"].items() if key != "fresh"}
    say("ledger " + json.dumps({"gpu_used": used, "gpu_total": row["gpu_total"], "host_free": row.get("host_free"),
                                "held": held, "unattributed": used - sum(count for kind, count in held.items()
                                                                          if kind != "host"),
                                **shown, "requests": each}, separators=(",", ":")))


def arm_ledger_free(scheduler):
    kv = scheduler.kv_cache_manager
    parts = _ledger.get("parts") or ledger_parts(kv)
    free = kv.free

    def ledger_free(request):
        rid = request.request_id
        kinds = census(parts, rid)
        if not any(kinds.values()):
            return free(request)
        before = pool_free(kv)
        made = free(request)
        after = pool_free(kv)
        say("ledger free " + json.dumps({"rid": rid, "status": getattr(request.status, "name", str(request.status)),
                                         "tokens": int(request.num_tokens), "held": kinds,
                                         "freed": {name: after[name] - before[name] for name in before}},
                                        separators=(",", ":")))
        return made

    kv.free = ledger_free
    return True


def publish_ledger(scheduler, row, now):
    if _ledger["on"] is None:
        _ledger["on"] = ledger_on()
        _ledger["every"] = _number(LEDGER_S_ENV, LEDGER_S)
        _ledger["log_every"] = _number(LEDGER_LOG_S_ENV, LEDGER_LOG_S)
    if not _ledger["on"] or _ledger["failed"] is not None:
        return None
    if census_on():
        clock = _ledger.setdefault("census_clock", {"sched": {}, "since": {}})
        for request in getattr(scheduler, "running", None) or ():
            clock["sched"][request.request_id] = now
    if now - _ledger["at"] < _ledger["every"] and _ledger["row"] is not None:
        row["ledger"] = _ledger["row"]
        return _ledger["row"]
    try:
        if _ledger["armed"] is None:
            _ledger["armed"] = arm_ledger_free(scheduler)
        reading, rows, parts = ledger_reading(scheduler)
        row["ledger"] = _ledger["row"] = reading
        _ledger["at"] = now
        if now - _ledger["logged"] >= _ledger["log_every"]:
            _ledger["logged"] = now
            ledger_log(row, rows, parts)
            if census_on():
                census_log(scheduler, parts)
    except Exception as failure:
        _ledger["failed"] = f"{type(failure).__name__}: {failure}"
        row.pop("ledger", None)
        say(f"ledger stops for good: {_ledger['failed']}")
        return None
    if not _ledger["said"]:
        _ledger["said"] = True
        say(f"ledger: every {_ledger['every']:g} s the pool reading carries the scheduler's own block counts: free "
            f"and reclaimable blocks, the blocks the live requests still allocate to finish their replies (at "
            f"max_tokens and at {', '.join(str(value) for value in LEDGER_HORIZONS)} reply tokens), the hot regions "
            f"paused requests lack, and a fresh request's blocks every {LEDGER_FRESH_STEP} tokens; every "
            f"{_ledger['log_every']:g} s a 'ledger' line lists each request's blocks, and each free prints a "
            f"'ledger free' line with what the request held beside what the pool got back")
    return reading


class Gauges:
    def rows(self):
        router = _state["router"]
        sessions = list(router.sessions.values()) if router is not None else []
        tally = router.tally if router is not None else {}
        rows = [("taaf_paused:sessions", "live paused-session conversations", len(sessions)),
                ("taaf_paused:generating", "paused sessions inside a turn", sum(1 for one in sessions if one.busy)),
                ("taaf_paused:held_tokens", "tokens the live paused sessions hold",
                 sum(len(one.tokens) for one in sessions)),
                ("taaf_paused:releases", "sessions closed by a release request", tally.get("closed_release", 0)),
                ("taaf_paused:returns", "first turns of a key after its release", tally.get("returns", 0))]
        if router is not None and router.parking:
            rows.append(("taaf_paused:parked", "sessions parked to the engine RAM store", len(router.parked)))
        pool = read_pool()
        for name in ("host_total", "host_free", "block_tokens", "gpu_total", "gpu_free"):
            if pool is not None and name in pool:
                rows.append((f"taaf_pool:{name}", f"KV pool reading {name}", pool[name]))
        ledger = (pool or {}).get("ledger")
        if isinstance(ledger, dict):
            for name in LEDGER_SCALARS:
                if name in ledger:
                    rows.append((f"taaf_pool:{name}", f"KV pool ledger {name}", ledger[name]))
        return rows

    def labelled(self):
        ledger = (read_pool() or {}).get("ledger")
        if not isinstance(ledger, dict):
            return []
        out = []
        for name, label in (("grow", "bound"), ("host_grow", "bound")):
            values = ledger.get(name) or {}
            out.append((f"taaf_pool:{name}", f"KV pool ledger {name} by reply bound", label,
                        [(str(key), value) for key, value in values.items()]))
        out.append(("taaf_pool:fresh", "KV pool ledger blocks a fresh request of this many tokens takes", "tokens",
                    [(str(tokens), blocks) for tokens, blocks in ledger.get("fresh") or ()]))
        return out

    def collect(self):
        from prometheus_client.core import GaugeMetricFamily
        for name, about, value in self.rows():
            yield GaugeMetricFamily(name, about, value=float(value))
        for name, about, label, values in self.labelled():
            family = GaugeMetricFamily(name, about, labels=[label])
            for key, value in values:
                family.add_metric([key], float(value))
            yield family


def register_gauges(registry=None):
    try:
        if registry is None:
            from prometheus_client import REGISTRY as registry
        registry.register(Gauges())
    except Exception as failure:
        say(f"gauges unavailable ({type(failure).__name__}: {failure}); /metrics carries no taaf_paused or "
            f"taaf_pool rows")
        return False
    say(f"gauges: /metrics carries taaf_paused:* (sessions, held tokens, releases, returns) and taaf_pool:* "
        f"(the pool reading from {pool_path()})")
    return True


FORK_SHARED = ("FullAttentionManager", "HiSparseSourceManager", "SinkFullAttentionManager")
FORK_EMPTY = ("HiSparseResidentManager", "HiSparseHotManager", "CircularBufferManager", "KpoolTailManager")
FORK_STATE = ("MambaManager",)
FORK_RING = ("CircularBufferManager", "KpoolTailManager")
FORK_KINDS = FORK_SHARED + FORK_EMPTY + FORK_STATE
FORK_TALLY = ("branched", "fallback", "shared", "computed", "hot_regions", "hot_skipped", "adopted", "lost",
              "device_copies", "host_copies", "held_mains", "held_cap", "jumped", "held_seats", "helpers_over_seats",
              "seat_waits", "seat_wait_ms", "seat_free")
HELPER_KEY_PREFIX = "slot-"


def fork_say(text):
    print(f"FORK_BRANCH {text}", flush=True)


def fork_marks(request):
    return getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}


def fork_is_branch(request):
    return bool(fork_marks(request).get(FORK_MARK))


def fork_is_helper(request):
    return str(fork_marks(request).get(SESSION_KEY) or "").startswith(HELPER_KEY_PREFIX)


def fork_main_rid(request):
    return fork_marks(request).get(FORK_OF)


def fork_status(request):
    status = getattr(request, "status", None)
    return getattr(status, "name", str(status))


def fork_tokens(request):
    held = getattr(request, "_all_token_ids", None)
    return held if held is not None else request.all_token_ids


def fork_find_main(requests, rid, key=None):
    held = requests.get(rid) if rid else None
    if held is not None:
        return held
    if rid:
        head = f"{rid}-"
        held = next((request for name, request in requests.items() if name.startswith(head)), None)
        if held is not None:
            return held
    if not key:
        return None
    live = []
    for request in requests.values():
        if fork_is_branch(request) or fork_marks(request).get(SESSION_KEY) != key:
            continue
        finished = getattr(request, "is_finished", None)
        if (callable(finished) and finished()) or fork_status(request) == "PREEMPTED":
            continue
        live.append(request)
    return max(live, key=lambda request: int(getattr(request, "num_computed_tokens", 0) or 0), default=None)


def fork_main_resuming(requests, request):
    main = fork_find_main(requests, fork_main_rid(request), fork_marks(request).get(FORK_KEY))
    return main is not None and fork_status(main) == "WAITING" and int(getattr(main, "num_computed_tokens", 0) or 0) > 0


def fork_waited(request):
    arrival = float(getattr(request, "arrival_time", 0.0) or 0.0)
    return max(0.0, time.time() - arrival) if arrival else 0.0


def fork_kind(manager):
    return next((klass.__name__ for klass in type(manager).__mro__ if klass.__name__ in FORK_KINDS), None)


def fork_cdiv(count, size):
    return -(-count // size)


def fork_page_bytes(manager):
    try:
        return int(getattr(manager.kv_cache_spec, "page_size_bytes", 0) or 0)
    except Exception:
        return 0


class ForkPlan:
    def __init__(self, rid, main_id):
        self.rid = rid
        self.main_id = main_id
        self.main_rid = main_id
        self.began = time.perf_counter()
        self.why = None
        self.code = None
        self.common = 0
        self.shared = 0
        self.blocks = None
        self.hs = None
        self.page = 0
        self.pages = 0
        self.adopt = {}
        self.cow = {}
        self.extra = {}
        self.valid = set()
        self.lost = set()
        self.hot = True
        self.fixed = False
        self.adopted = 0
        self.rings = 0
        self.rewind = None
        self.source = None
        self.hole = None
        self.source_note = None

    def refuse(self, code, why):
        self.code = code
        self.why = why
        return self


def fork_mamba_sets(coordinator, managers, kinds):
    wanted = {index for index, kind in enumerate(kinds) if kind in FORK_STATE}
    if not wanted:
        return []
    sets = []
    for group in getattr(coordinator, "attention_groups", None) or ():
        ids = [index for index in group.group_ids if index in wanted]
        if ids:
            sets.append((group.spec, ids, group.manager_cls))
            wanted -= set(ids)
    for index in sorted(wanted):
        sets.append((managers[index].kv_cache_spec, [index], type(managers[index])))
    return sets


def fork_checkpoint(request, managers, sets, limit, unit):
    shared = limit
    found = {}
    while True:
        moved = False
        for spec, ids, klass in sets:
            first = managers[ids[0]]
            blocks, hit = klass.find_longest_cache_hit(
                block_hashes=request.block_hashes, max_length=shared, kv_cache_group_ids=ids,
                block_pool=first.block_pool, kv_cache_spec=spec, drop_eagle_block=False, alignment_tokens=unit,
                dcp_world_size=getattr(first, "dcp_world_size", 1), pcp_world_size=getattr(first, "pcp_world_size", 1))
            if hit < shared:
                shared = hit
                moved = True
            for index, held in zip(ids, blocks):
                found[index] = list(held)
        if not moved or shared <= 0:
            return shared, found


def fork_live(main, managers, sets, confirmed, common, request):
    if fork_status(main) not in ("WAITING_FOR_STREAMING_REQ", "WAITING"):
        return None, f"the main is {fork_status(main)}, not paused"
    if int(getattr(main, "num_output_placeholders", 0) or 0):
        return None, "the main has tokens in flight"
    computed = int(main.num_computed_tokens)
    if confirmed != computed:
        return None, f"the main confirms {confirmed} of its {computed} computed tokens"
    if common != computed:
        return None, f"the branch shares {common} of the main's {computed} computed tokens"
    if computed > int(request.num_tokens) - 1:
        return None, f"the branch holds {request.num_tokens} tokens, nothing past the main's {computed}"
    found = {}
    for _, ids, _ in sets:
        for index in ids:
            one = managers[index]
            place = (computed - 1) // one.block_size
            table = one.req_to_blocks.get(main.request_id) or ()
            source = table[place] if place < len(table) else None
            if source is None or source.is_null:
                return None, f"Mamba group {index} of the main holds no state block at index {place}"
            found[index] = [one._null_block] * place + [source]
    return found, None


def fork_hole_rewrite(request, main, hole, limit=0):
    from dataclasses import replace
    if getattr(request, "_taaf_fork_hole", None) is not None:
        return None
    view, pad, head = (int(value) for value in hole)
    held = list(fork_tokens(main))
    mine = list(fork_tokens(request))
    whole = view + pad
    if len(held) != whole:
        return f"the main holds {len(held)} tokens, not its view of {view} and a pad of {pad}"
    if len(mine) <= view:
        return f"the branch holds {len(mine)} tokens, nothing past the main's view of {view}"
    if held[:head] != mine[:head] or held[head + pad:] != mine[head:view]:
        return "the branch's view of the main differs from the main's tokens around the hole"
    features = []
    for feature in request.mm_features:
        place = feature.mm_position
        if place.offset + place.length <= view:
            continue
        if place.offset < view:
            return f"an image at {place.offset} straddles the end of the main's view at {view}"
        features.append(replace(feature, mm_position=replace(place, offset=place.offset + pad)))
    tokens = held + mine[view:]
    if limit and len(tokens) >= limit:
        return f"the branch would hold {len(tokens)} tokens with the main's pad, not under max_model_len {limit}"
    request._taaf_fork_hole = (request.prompt_token_ids, list(request._all_token_ids), request.num_prompt_tokens,
                               request.mm_features, list(request.block_hashes), request._block_hasher,
                               getattr(request, "_taaf_hole_salted", False))
    request.prompt_token_ids = list(tokens)
    request._all_token_ids[:] = tokens
    request.num_prompt_tokens = len(tokens)
    request.mm_features = features
    request.block_hashes.clear()
    trim_hole_salted(request)
    request.update_block_hashes()
    extra = dict(getattr(request.sampling_params, "extra_args", None) or {})
    extra[FORK_HOLE_AT] = [whole, main.request_id]
    request.sampling_params.extra_args = extra
    return None


def fork_hole_undo(request):
    held = getattr(request, "_taaf_fork_hole", None)
    if held is None:
        return False
    prompt, tokens, count, features, hashes, hasher, salted = held
    request.prompt_token_ids = prompt
    request._all_token_ids[:] = tokens
    request.num_prompt_tokens = count
    request.mm_features = features
    request.block_hashes[:] = hashes
    request._block_hasher = hasher
    request._taaf_hole_salted = salted
    extra = dict(getattr(request.sampling_params, "extra_args", None) or {})
    extra.pop(FORK_HOLE_AT, None)
    request.sampling_params.extra_args = extra
    request._taaf_fork_hole = None
    return True


def fork_plan(manager, requests, request, limit=0):
    plan = ForkPlan(request.request_id, fork_main_rid(request))
    key = fork_marks(request).get(FORK_KEY)
    try:
        main = fork_find_main(requests, plan.main_id, key) if (plan.main_id or key) else None
        if main is not None and plan.main_id and main.request_id != plan.main_id                 and not main.request_id.startswith(f"{plan.main_id}-"):
            plan.source_note = f"main found by session key {key}: {plan.main_id} is now {main.request_id}"
        return fork_shape(manager, request, main, plan, limit)
    except Exception as failure:
        return plan.refuse("error", f"planning raised {type(failure).__name__}: {failure}")


def fork_shape(manager, request, main, plan, limit=0):
    if main is None:
        return plan.refuse("gone", f"main {plan.main_id} is gone from the engine")
    plan.main_id = main.request_id
    finished = getattr(main, "is_finished", None)
    if callable(finished) and finished():
        return plan.refuse("gone", f"main {main.request_id} has finished")
    if fork_status(main) == "PREEMPTED":
        return plan.refuse("preempted", f"main {main.request_id} was preempted and holds no KV")
    coordinator = manager.coordinator
    managers = list(coordinator.single_type_managers)
    kinds = [fork_kind(one) for one in managers]
    for index, kind in enumerate(kinds):
        if kind is None:
            return plan.refuse("group", f"group {index} ({type(managers[index]).__name__}) has no branch rule")
    confirmed = min(int(main.num_computed_tokens) - int(getattr(main, "num_output_placeholders", 0) or 0),
                    int(main.num_tokens))
    confirmed -= int(getattr(coordinator, "num_reprefillable_tokens", 0) or 0)
    hole = fork_marks(request).get(FORK_HOLE)
    if hole:
        if not fork_live_on():
            return plan.refuse("hole", f"a branch of a main with a hole takes only the main's live state "
                                       f"({FORK_LIVE_ENV} is off)")
        why = fork_hole_rewrite(request, main, hole, limit)
        if why is not None:
            return plan.refuse("hole", why)
        plan.hole = (int(hole[0]) + int(hole[1]), int(hole[1]))
    mine = fork_tokens(request)
    plan.common = common = lcp_blocks(mine, fork_tokens(main), max(0, min(len(mine), confirmed)))
    unit = int(getattr(coordinator, "_cache_hit_alignment_tokens", 0) or managers[0].block_pool.hash_block_size)
    margin = unit if getattr(coordinator, "eagle_group_ids", None) else 0
    limit = min(common - margin, confirmed, int(request.num_tokens) - 1)
    if limit < unit:
        return plan.refuse("short", f"shares {common} tokens with main {main.request_id} ({confirmed} confirmed), "
                                    f"under one {unit}-token match unit past the draft margin")
    shared = limit // unit * unit
    sets = fork_mamba_sets(coordinator, managers, kinds)
    found = {}
    live = None
    if sets and fork_live_on():
        live, missed = fork_live(main, managers, sets, confirmed, common, request)
        if live is not None:
            shared, found = confirmed, live
            plan.source = f"live state at {confirmed}"
        else:
            plan.source = f"checkpoint (live state not taken: {missed})"
    if plan.hole is not None and live is None:
        return plan.refuse("hole", f"a branch of a main with a hole takes only the main's live state; "
                                   f"{plan.source or 'the main has no recurrent state'}")
    if sets and live is None:
        shared, found = fork_checkpoint(request, managers, sets, shared, unit)
        if shared <= 0:
            return plan.refuse("checkpoint", f"no recurrent-state checkpoint of the shared prefix at or below "
                                             f"{limit} tokens (shares {common} with main {main.request_id})"
                                             + (f"; {plan.source}" if plan.source else ""))
        if plan.source:
            plan.source = f"{plan.source.replace('checkpoint', f'checkpoint at {shared}', 1)}"
    plan.shared = shared
    blocks = []
    for index, (one, kind) in enumerate(zip(managers, kinds)):
        if kind in FORK_STATE:
            blocks.append(found.get(index, []))
            continue
        if kind in FORK_EMPTY:
            blocks.append([])
            continue
        count = fork_cdiv(shared, one.block_size)
        held = one.req_to_blocks.get(main.request_id) or ()
        if len(held) < count:
            return plan.refuse("blocks", f"{kind} group {index} of main holds {len(held)} block(s), {count} needed "
                                         f"for {shared} tokens")
        part = list(held[:count])
        if any(block.is_null for block in part):
            return plan.refuse("blocks", f"{kind} group {index} of main has a freed block below {shared} tokens")
        blocks.append(part)
    plan.blocks = tuple(blocks)
    return fork_residency(managers, kinds, main, plan)


def fork_residency(managers, kinds, main, plan):
    residents = [(index, one) for index, (one, kind) in enumerate(zip(managers, kinds))
                 if kind == "HiSparseResidentManager"]
    if not residents:
        return plan
    state = residents[0][1].coordinator.request_states.get(main.request_id)
    valid = set(state.valid_pages) if state is not None else set()
    unpinned = set(getattr(state, "unpinned_pages", ()) or ()) if state is not None else set()
    tables = {index: one.req_to_blocks.get(main.request_id) or () for index, one in residents}
    return fork_host_pages(residents, plan, tables, valid, unpinned)


def fork_host_pages(residents, plan, tables, valid, unpinned):
    plan.hs = residents[0][1].coordinator
    plan.page = page = residents[0][1].block_size
    plan.pages = full = plan.shared // page
    last = fork_cdiv(plan.shared, page)
    ready = 0
    while ready < full and ready in valid:
        ready += 1
    for index, one in residents:
        held = tables.get(index) or ()
        plan.adopt[index] = {}
        copies = []
        for place in range(ready, last):
            block = held[place] if place < len(held) else None
            if block is None or block.is_null or place in unpinned:
                return plan.refuse("page", f"page {place} of main is neither durable on the host in an unbroken "
                                           f"prefix nor held resident by the main in group {index}")
            copies.append((place, block))
        if copies:
            plan.cow[index] = copies
    plan.lost = set(range(ready))
    plan.hot = ready > 0
    plan.valid = set(range(ready))
    plan.extra = {index: len(plan.cow.get(index, ())) for index, _ in residents}
    return plan


def fork_pages(residents, plan, tables, valid):
    plan.hs = residents[0][1].coordinator
    plan.page = page = residents[0][1].block_size
    plan.pages = full = plan.shared // page
    partial = plan.shared % page
    for index, one in residents:
        held = tables.get(index) or ()
        mine = plan.adopt.setdefault(index, {})
        for place in range(full):
            block = held[place] if place < len(held) else None
            if block is not None and not block.is_null:
                mine[place] = block
            elif place in valid:
                plan.lost.add(place)
            else:
                return plan.refuse("page", f"page {place} of main is neither resident in group {index} nor durable "
                                           f"on the host")
        if partial:
            block = held[full] if full < len(held) else None
            if block is None or block.is_null:
                return plan.refuse("page", f"the branch page {full} of main is not resident in group {index}")
            plan.cow[index] = (full, block)
    plan.hot = bool(plan.lost)
    if plan.hot:
        plan.adopt = {index: {place: block for place, block in mine.items() if block.ref_cnt > 0}
                      for index, mine in plan.adopt.items()}
    plan.valid = {place for place in range(full) if place in valid}
    plan.extra = {index: int(index in plan.cow) + sum(1 for block in plan.adopt[index].values() if block.ref_cnt == 0)
                  for index, _ in residents}
    return plan


class Forks:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.manager = scheduler.kv_cache_manager
        self.most = fork_most()
        self.plans = {}
        self.active = None
        self.held = []
        self.tally = dict.fromkeys(FORK_TALLY, 0)
        self.reasons = {}
        self.reported = time.monotonic()
        self.running = 0
        self.rewind = None
        self.seats = fork_seats_start()
        self.seats_at = 0.0
        self.seat_wait = {}
        self.over_seats = set()
        managers = list(self.manager.coordinator.single_type_managers)
        self.kinds = [fork_kind(one) for one in managers]
        self.wrap_manager()
        for index, (one, kind) in enumerate(zip(managers, self.kinds)):
            if kind == "HiSparseResidentManager":
                self.wrap_resident(index, one)
            elif kind == "HiSparseHotManager":
                self.wrap_hot(one)
            elif kind in FORK_RING:
                self.wrap_ring(one)
        counts = {}
        for kind in self.kinds:
            counts[kind or "unknown"] = counts.get(kind or "unknown", 0) + 1
        if not fork_on():
            return
        fork_say(f"armed on the scheduler: a request marked {FORK_OF} takes the named main request's KV blocks "
                 f"for the prefix they share (ref-counted, a partial last block copied), a private copy of the "
                 f"newest recurrent-state checkpoint at or below it, and the main's resident host-KV pages; "
                 f"at most {self.most} branch requests run at once ({FORK_MAX_ENV}) and none is admitted while a "
                 f"main request waits{"" if self.seats is None else f" (off: {FORK_SEATS_ENV} keeps {self.seats} runner slot(s) for branches, updated by a request's {FORK_SEATS_MARK}, and new requests wait for the rest)"}; groups {', '.join(f'{kind} {count}' for kind, count in counts.items())}")

    def wrap_manager(self):
        manager = self.manager
        lookup = manager.get_computed_blocks
        connector = getattr(manager, "get_computed_blocks_for_connector", None)
        allocate = manager.allocate_slots

        def get_computed_blocks(request):
            plan = self.lookup(request)
            if plan is None or plan.why is not None:
                return lookup(request)
            return manager.create_kv_cache_blocks(plan.blocks), plan.shared, 0

        def get_computed_blocks_for_connector(request):
            plan = self.lookup(request)
            if plan is None or plan.why is not None:
                return connector(request)
            return manager.create_kv_cache_blocks(plan.blocks), plan.shared, 0, False

        def allocate_slots(request, *args, **kwargs):
            plan = self.plans.get(request.request_id) if self.plans else None
            if plan is None or request.num_computed_tokens != 0:
                return allocate(request, *args, **kwargs)
            hit = kwargs.get("num_new_computed_tokens", args[1] if len(args) > 1 else 0)
            given = kwargs.get("new_computed_blocks", args[2] if len(args) > 2 else None)
            using = plan.why is None and given is not None and hit == plan.shared
            if plan.why is None and not using:
                plan.refuse("taken", f"the scheduler took {hit} cached tokens, not the {plan.shared} shared")
            before = self.pending()
            self.active = plan if using else None
            try:
                made = allocate(request, *args, **kwargs)
            finally:
                self.active = None
            if made is not None:
                self.plans.pop(request.request_id, None)
                self.admitted(request, plan, using, hit, before)
            return made

        manager.get_computed_blocks = get_computed_blocks
        if connector is not None:
            manager.get_computed_blocks_for_connector = get_computed_blocks_for_connector
        manager.allocate_slots = allocate_slots

    def lookup(self, request):
        if request.num_computed_tokens != 0:
            return None
        backup = self.rewind.waiting(request) if self.rewind is not None else None
        if backup is None and not fork_main_rid(request):
            return None
        enabled = getattr(self.manager, "prefix_cache_lookup_enabled", None)
        if enabled is not None and not enabled(request):
            if backup is not None:
                plan = ForkPlan(request.request_id, request.request_id).refuse(
                    "lookup", "the request does not read the prefix cache")
                plan.rewind = backup
                self.plans[request.request_id] = plan
            return None
        if backup is not None:
            plan = rewind_plan(self.manager, request, backup)
        else:
            plan = fork_plan(self.manager, self.scheduler.requests, request,
                             int(getattr(self.scheduler, "max_model_len", 0) or 0))
            if plan.why is not None:
                fork_hole_undo(request)
        self.plans[request.request_id] = plan
        return plan

    def wrap_resident(self, index, one):
        count = one.get_num_blocks_to_allocate
        local = one.add_local_computed_blocks
        fresh = one.allocate_new_blocks

        def get_num_blocks_to_allocate(request_id, *args, **kwargs):
            made = count(request_id, *args, **kwargs)
            plan = self.active
            if plan is not None and plan.rid == request_id:
                made += plan.extra.get(index, 0)
            return made

        def add_local_computed_blocks(request_id, *args, **kwargs):
            local(request_id, *args, **kwargs)
            plan = self.active
            if plan is None or plan.rid != request_id:
                return
            table = one.req_to_blocks[request_id]
            for place, block in sorted(plan.adopt.get(index, {}).items()):
                if place < len(table) and table[place].is_null:
                    one.block_pool.touch([block])
                    table[place] = block
                    plan.adopted += 1

        def allocate_new_blocks(request_id, *args, **kwargs):
            plan = self.active
            made = []
            if plan is not None and plan.rid == request_id:
                if not plan.fixed:
                    plan.fixed = True
                    self.settle(plan)
                held = plan.cow.pop(index, None)
                table = one.req_to_blocks[request_id]
                for place, source in ([held] if isinstance(held, tuple) else list(held or ())):
                    if place < len(table) and table[place].is_null:
                        one.block_pool.touch([source])
                        copy = one.block_pool.get_new_blocks(1)[0]
                        table[place] = copy
                        one._pending_cow_copies.append((source, copy))
                        copy.ref_cnt += 1
                        made.append(copy)
            return made + list(fresh(request_id, *args, **kwargs))

        one.get_num_blocks_to_allocate = get_num_blocks_to_allocate
        one.add_local_computed_blocks = add_local_computed_blocks
        one.allocate_new_blocks = allocate_new_blocks

    def wrap_ring(self, one):
        fresh = one.allocate_new_blocks
        external = getattr(one, "allocate_external_computed_blocks", None)

        def carry(request_id, made):
            plan = self.active
            if plan is None or plan.rid != request_id or plan.why is not None or not made:
                return made
            main = self.scheduler.requests.get(plan.main_id)
            if main is None or int(plan.shared) != int(main.num_computed_tokens):
                return made
            source = (one.req_to_blocks.get(plan.main_id) or (None,))[0]
            if source is None or source.is_null:
                return made
            one.block_pool.touch([source])
            one._pending_cow_copies.append((source, made[0]))
            made[0].ref_cnt += 1
            plan.rings += 1
            return made

        def allocate_new_blocks(request_id, *args, **kwargs):
            return carry(request_id, list(fresh(request_id, *args, **kwargs)))

        def allocate_external_computed_blocks(request_id, *args, **kwargs):
            before = len(one.req_to_blocks.get(request_id) or ())
            external(request_id, *args, **kwargs)
            after = list(one.req_to_blocks.get(request_id) or ())
            carry(request_id, after[before:])

        one.allocate_new_blocks = allocate_new_blocks
        if external is not None:
            one.allocate_external_computed_blocks = allocate_external_computed_blocks

    def wrap_hot(self, one):
        count = one.get_num_blocks_to_allocate
        local = one.add_local_computed_blocks

        def get_num_blocks_to_allocate(request_id, *args, **kwargs):
            plan = self.active
            if plan is not None and plan.rid == request_id and not plan.hot and request_id not in one.hot_required:
                return 0
            return count(request_id, *args, **kwargs)

        def add_local_computed_blocks(request_id, new_computed_blocks, num_local_computed_tokens,
                                      num_external_computed_tokens):
            plan = self.active
            if plan is not None and plan.rid == request_id and not plan.hot:
                num_local_computed_tokens = 0
            return local(request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens)

        one.get_num_blocks_to_allocate = get_num_blocks_to_allocate
        one.add_local_computed_blocks = add_local_computed_blocks

    def settle(self, plan):
        hs = plan.hs
        if hs is None:
            return
        state = hs._get_request_state(plan.rid)
        cut = fork_cdiv(plan.shared, plan.page)
        state.valid_pages = {place for place in state.valid_pages if place >= cut} | set(plan.valid)
        ready = 0
        while ready in state.valid_pages:
            ready += 1
        state.ready_prefix_pages = ready
        tables = [one.req_to_blocks.get(plan.rid) or () for one in hs.resident_managers]
        state.pinned_clean = {place for place in plan.valid
                              if all(place < len(table) and not table[place].is_null for table in tables)}

    def pending(self):
        return [len(getattr(one, "_pending_cow_copies", ())) for one in self.manager.coordinator.single_type_managers]

    def copies(self, before):
        managers = self.manager.coordinator.single_type_managers
        after = self.pending()
        device = host = 0
        moved = 0
        for one, kind, was, now in zip(managers, self.kinds, before, after):
            if now <= was:
                continue
            if kind == "HiSparseSourceManager":
                host += now - was
            else:
                device += now - was
                moved += (now - was) * fork_page_bytes(one)
        return device, host, moved

    def admitted(self, request, plan, using, hit, before):
        if plan.rewind is not None:
            return self.rewind.admitted(request, plan, using, hit, self.copies(before))
        managers = self.manager.coordinator.single_type_managers
        device, host, moved = self.copies(before)
        hots = [one for one, kind in zip(managers, self.kinds) if kind == "HiSparseHotManager"]
        hot_blocks = sum(len(one.req_to_blocks.get(request.request_id) or ()) for one in hots)
        rows = len(hots[0].req_to_blocks.get(request.request_id) or ()) * hots[0].block_size if hots else 0
        tail = int(request.num_tokens) - int(hit)
        took = (time.perf_counter() - plan.began) * 1000.0
        hot = (f"hot region {rows} rows ({hot_blocks} blocks)" if hot_blocks else
               "no hot region" + ("" if not hots else " (every shared page is GPU-resident)" if using
                                  else " (the stock rule asked for none)"))
        if using:
            self.tally["branched"] += 1
            self.tally["shared"] += int(hit)
            self.tally["computed"] += tail
            self.tally["adopted"] += plan.adopted
            self.tally["lost"] += len(plan.lost)
            self.tally["device_copies"] += device
            self.tally["host_copies"] += host
            self.tally["hot_regions" if hot_blocks else "hot_skipped"] += 1
            waited = fork_waited(request)
            fork_say(f"branch {request.request_id} of {plan.main_id}: {hit} prefix tokens shared ({plan.common} in "
                     f"common), {tail} tail tokens computed; state copy {device} device block(s) "
                     f"({moved / 2 ** 20:.1f} MiB) and {host} host page(s) queued for the step's copy before its "
                     f"forward, admission {took:.2f} ms; {hot}; {plan.adopted} resident page(s) shared, "
                     f"{len(plan.lost)} read from host; {plan.rings} ring block(s) carried from the main; fell back no"
                     + (f"; recurrent state from the main's {plan.source}" if plan.source else "")
                     + (f"; main with a hole, engine view {plan.hole[0]} tokens with a pad of {plan.hole[1]}"
                        if plan.hole else "")
                     + (f"; {plan.source_note}" if plan.source_note else "")
                     + f"; admitted {waited:.1f} s after arrival")
            return
        self.tally["fallback"] += 1
        self.tally["computed"] += tail
        self.tally["hot_regions" if hot_blocks else "hot_skipped"] += 1
        self.reasons[plan.code] = self.reasons.get(plan.code, 0) + 1
        fork_say(f"branch {request.request_id} of {plan.main_rid}: fell back yes ({plan.why}); {hit} tokens from the "
                 f"prefix cache, {tail} computed; {hot}; admitted {fork_waited(request):.1f} s after arrival")

    def hold(self):
        self.release()
        scheduler = self.scheduler
        requests = scheduler.requests
        if self.plans:
            for rid in [rid for rid in self.plans if rid not in requests]:
                del self.plans[rid]
        queues = [queue for queue in (scheduler.waiting, getattr(scheduler, "skipped_waiting", None))
                  if queue is not None]
        waiting = [(queue, request) for queue in queues for request in list(queue)]
        branches = [(queue, request) for queue, request in waiting if fork_is_branch(request)]
        self.running = sum(1 for request in scheduler.running if fork_is_branch(request))
        if self.seats is not None:
            self.seat_count(waiting)
            self.hold_seats(waiting)
            self.seat_waits(waiting)
        if branches:
            mains = self.seats is None and any(
                not fork_is_branch(request) and fork_status(request) in ("WAITING", "PREEMPTED")
                for _, request in waiting)
            room = max(0, self.most - self.running)
            branches.sort(key=lambda item: (float(getattr(item[1], "arrival_time", 0.0) or 0.0),
                                            item[1].request_id))
            hold_s = fork_hold_s()
            urgent = [item for item in branches if fork_main_resuming(requests, item[1])]
            if mains and hold_s > 0:
                now = time.time()
                passing = [item for item in branches if item in urgent
                           or now - float(getattr(item[1], "arrival_time", now) or now) >= hold_s][:room]
            else:
                passing = branches[:room]
            passed = {id(request) for _, request in passing}
            for queue, request in branches:
                if id(request) in passed:
                    continue
                queue.remove_request(request)
                self.held.append((queue, request))
            if (mains and hold_s > 0) or urgent or self.seats is not None:
                for queue, request in reversed(passing):
                    queue.remove_request(request)
                    queue.prepend_request(request)
                    self.tally["jumped"] += 1
            if self.held:
                self.tally["held_mains" if mains else "held_cap"] += len(self.held)
        self.report()
        return len(self.held)

    def seat_count(self, waiting):
        for request in [*self.scheduler.running, *(request for _, request in waiting)]:
            said = fork_seats_said(request)
            arrival = float(getattr(request, "arrival_time", 0.0) or 0.0)
            if said is not None and arrival >= self.seats_at:
                if said != self.seats:
                    fork_say(f"seats: {said} runner slot(s) now kept for branches, was {self.seats} "
                             f"({FORK_SEATS_MARK} of {request.request_id})")
                self.seats, self.seats_at = said, arrival

    def hold_seats(self, waiting):
        scheduler = self.scheduler
        most = int(getattr(scheduler, "max_num_running_reqs", 0) or 0)
        if not most or not self.seats:
            return
        paused = getattr(scheduler, "num_waiting_for_streaming_input", None)
        if paused is None:
            paused = sum(1 for _, request in waiting if fork_status(request) == "WAITING_FOR_STREAMING_REQ")
        taken = sum(1 for request in scheduler.running if not fork_is_branch(request)) + int(paused)
        taken += sum(1 for _, request in waiting if not fork_is_branch(request) and not fork_fresh(request)
                     and fork_status(request) != "WAITING_FOR_STREAMING_REQ")
        free = most - self.seats - taken
        live = free + int(paused)
        for queue, request in waiting:
            if not fork_fresh(request):
                continue
            if fork_marks(request).get(SEAT_FREE_MARK):
                self.tally["seat_free"] += 1
                continue
            if free > 0:
                free -= 1
                live -= 1
                continue
            if live > 0 and fork_is_helper(request):
                live -= 1
                if request.request_id not in self.over_seats:
                    self.over_seats.add(request.request_id)
                    self.tally["helpers_over_seats"] += 1
                continue
            queue.remove_request(request)
            self.held.append((queue, request))
            self.tally["held_seats"] += 1

    def seat_waits(self, waiting, now=None):
        most = int(getattr(self.scheduler, "max_num_running_reqs", 0) or 0)
        if not most:
            return
        now = time.monotonic() if now is None else now
        back = {request.request_id for _, request in waiting
                if not fork_is_branch(request) and not fork_fresh(request) and fork_status(request) == "WAITING"}
        for rid in [rid for rid in self.seat_wait if rid not in back]:
            self.tally["seat_waits"] += 1
            self.tally["seat_wait_ms"] += int((now - self.seat_wait.pop(rid)) * 1000)
        if len(self.scheduler.running) >= most:
            for rid in back:
                self.seat_wait.setdefault(rid, now)
        self.over_seats &= {request.request_id for _, request in waiting}

    def release(self):
        held, self.held = self.held, []
        requests = self.scheduler.requests
        for queue, request in held:
            finished = getattr(request, "is_finished", None)
            if requests.get(request.request_id) is request and not (callable(finished) and finished()):
                queue.add_request(request)
        return len(held)

    def report(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.reported < FORK_REPORT_S:
            return None
        span = now - self.reported
        self.reported = now
        tally, self.tally = self.tally, dict.fromkeys(FORK_TALLY, 0)
        reasons, self.reasons = self.reasons, {}
        line = (f"last {span:.0f} s: {tally['branched']} branched sharing {tally['shared']} tokens, "
                f"{tally['fallback']} fell back ({', '.join(f'{code} {count}' for code, count in sorted(reasons.items())) or 'none'}), "
                f"{tally['computed']} tail tokens computed; {tally['device_copies']} device and "
                f"{tally['host_copies']} host block copies; {tally['adopted']} resident pages shared, "
                f"{tally['lost']} read from host; hot regions {tally['hot_regions']} taken, {tally['hot_skipped']} "
                f"not needed; branch admissions held {tally['held_mains']} pass(es) for waiting mains and "
                f"{tally['held_cap']} at the cap of {self.most}, {tally['jumped']} moved to the queue head after the "
                f"{fork_hold_s():g} s hold; {self.running} running now"
                + ("" if self.seats is None else f"; {self.seats} runner slot(s) kept for branches, "
                                                 f"{tally['held_seats']} new request pass(es) held for them; "
                                                 f"{tally['helpers_over_seats']} helper(s) let in past the sessions "
                                                 f"paused between turns; {tally['seat_free']} recompute check "
                                                 f"pass(es) let past them; {tally['seat_waits']} request(s) back from "
                                                 f"a pause waited for a runner slot, "
                                                 f"{tally['seat_wait_ms'] / 1000:.1f} s in all"))
        fork_say(line)
        return line


def fork_state(scheduler):
    forks = scheduler.__dict__.get("_taaf_forks")
    if forks is None:
        try:
            forks = Forks(scheduler)
        except Exception as failure:
            forks = False
            fork_say(f"not armed: {type(failure).__name__}: {failure}; every branch request runs as an ordinary "
                     f"request")
        scheduler.__dict__["_taaf_forks"] = forks
    return forks


def fork_hold(scheduler):
    forks = fork_state(scheduler)
    if forks is False:
        return 0
    return forks.hold()


def fork_release(scheduler):
    forks = scheduler.__dict__.get("_taaf_forks")
    if not forks:
        return 0
    return forks.release()


REWIND_TALLY = ("backups", "refused", "rewinds", "kept", "refed", "side", "fell_back", "dropped",
                "device_copies", "host_copies")
_REWIND = {"installed": False}
_TRIM_HOLE = {"installed": False, "worker": False, "tally": {}}


class Backup:
    def __init__(self, rid, point, tokens, blocks, pages, valid, held):
        self.rid = rid
        self.point = point
        self.tokens = tokens
        self.blocks = blocks
        self.pages = pages
        self.valid = valid
        self.held = held
        self.side = 0
        self.began = None
        self.restore_ms = 0.0


def rewind_plan(manager, request, backup):
    plan = ForkPlan(request.request_id, request.request_id)
    plan.rewind = backup
    try:
        return rewind_shape(manager, request, backup, plan)
    except Exception as failure:
        return plan.refuse("error", f"planning raised {type(failure).__name__}: {failure}")


def rewind_shape(manager, request, backup, plan):
    managers = list(manager.coordinator.single_type_managers)
    kinds = [fork_kind(one) for one in managers]
    mine = fork_tokens(request)
    point = backup.point
    plan.common = lcp_blocks(mine, backup.tokens, min(len(mine), len(backup.tokens)))
    if plan.common < point:
        return plan.refuse("diverged", f"the request shares {plan.common} tokens with the backup, short of its "
                                       f"point {point}")
    if point > int(request.num_tokens) - 1:
        return plan.refuse("short", f"the request holds {request.num_tokens} tokens, nothing past the point {point}")
    plan.shared = point
    plan.blocks = tuple(backup.blocks)
    residents = [(index, one) for index, (one, kind) in enumerate(zip(managers, kinds))
                 if kind == "HiSparseResidentManager"]
    if not residents:
        return plan
    return fork_pages(residents, plan, backup.pages, backup.valid)


def rewind_request(request, tokens, point):
    del request._all_token_ids[:]
    request._all_token_ids.extend(tokens)
    del request.prompt_token_ids[:]
    request._output_token_ids.clear()
    request.num_prompt_tokens = 0
    request.num_computed_tokens = 0
    request.spec_token_ids = []
    del request.block_hashes[:]
    features = getattr(request, "mm_features", None)
    if features:
        features[:] = [one for one in features if one.mm_position.offset + one.mm_position.length <= point]


class Rewinds:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.manager = scheduler.kv_cache_manager
        forks = fork_state(scheduler)
        if not forks:
            raise RuntimeError("the branch admission machinery did not arm")
        forks.rewind = self
        self.forks = forks
        self.kinds = list(forks.kinds)
        self.backups = {}
        self.pending = {}
        self.tally = dict.fromkeys(REWIND_TALLY, 0)
        self.reasons = {}
        self.reported = time.monotonic()
        rewind_say(f"armed on the scheduler ({REWIND_ENV}): a backup holds a paused session's full-attention and "
                   f"host-source blocks up to its computed end and its resident host-KV pages by refcount, and "
                   f"copies its recurrent state into a spare block; a rewind frees the session's blocks, sets its "
                   f"tokens back to the backup and admits its next chunk with the backup as its prefix hit; one "
                   f"PAUSED_REWIND line per backup and per rewind and a {REWIND_REPORT_S:.0f} s tally")

    def bump(self, name, count=1):
        self.tally[name] = self.tally.get(name, 0) + count

    def refuse(self, code, why):
        self.reasons[code] = self.reasons.get(code, 0) + 1
        return {"why": why, "code": code}

    def managers(self):
        return list(self.manager.coordinator.single_type_managers)

    def find(self, rid):
        return fork_find_main(self.scheduler.requests, rid)

    def idle(self, request):
        if fork_status(request) != "WAITING_FOR_STREAMING_REQ":
            return f"the session is {fork_status(request)}, not paused"
        if int(getattr(request, "num_output_placeholders", 0) or 0):
            return "the session has tokens in flight"
        return None

    def release(self, backup):
        for pool, blocks in backup.held:
            pool.free_blocks(blocks)
        backup.held = []

    def sweep(self):
        requests = self.scheduler.requests
        for rid, backup in list(self.backups.items()):
            request = requests.get(rid)
            finished = getattr(request, "is_finished", None)
            if request is None or (callable(finished) and finished()) or fork_status(request) == "PREEMPTED":
                del self.backups[rid]
                self.release(backup)
                self.bump("dropped")
                rewind_say(f"let go {rid}: the session is {'gone' if request is None else fork_status(request)}")
        for rid, backup in list(self.pending.items()):
            if rid not in requests:
                del self.pending[rid]
                self.forks.plans.pop(rid, None)
                self.release(backup)
                self.bump("dropped")

    def backup(self, rid):
        began = time.perf_counter()
        self.sweep()
        request = self.find(rid)
        why = None if request is not None else f"no engine request {rid}"
        if why is None:
            why = self.idle(request)
        if why is not None:
            self.bump("refused")
            return self.refuse("state", why)
        held = self.backups.pop(request.request_id, None)
        if held is not None:
            self.release(held)
            self.bump("dropped")
        point = min(int(request.num_computed_tokens), int(request.num_tokens))
        if point <= 0:
            self.bump("refused")
            return self.refuse("empty", "the session has computed nothing")
        managers = self.managers()
        shared, states, residents = [], [], []
        for index, (one, kind) in enumerate(zip(managers, self.kinds)):
            table = one.req_to_blocks.get(request.request_id) or []
            if kind is None:
                self.bump("refused")
                return self.refuse("group", f"group {index} ({type(one).__name__}) has no rewind rule")
            if kind in FORK_SHARED:
                count = fork_cdiv(point, one.block_size)
                part = list(table[:count])
                if len(part) < count or any(block.is_null for block in part):
                    self.bump("refused")
                    return self.refuse("blocks", f"{kind} group {index} holds {len(part)} live block(s) of the "
                                                 f"{count} below {point} tokens")
                shared.append((index, one, part))
            elif kind in FORK_STATE:
                place = (point - 1) // one.block_size
                source = table[place] if place < len(table) else None
                if source is None or source.is_null:
                    self.bump("refused")
                    return self.refuse("state", f"Mamba group {index} holds no state block at index {place}")
                states.append((index, one, place, source))
            elif kind == "HiSparseResidentManager":
                full, partial = divmod(point, one.block_size)
                part = [table[place] if place < len(table) and not table[place].is_null else None
                        for place in range(full + (1 if partial else 0))]
                residents.append((index, one, part))
        wanted = {}
        for _, one, _, _ in states:
            wanted[id(one.block_pool)] = wanted.get(id(one.block_pool), 0) + 1
        for _, one, _, _ in states:
            if one.block_pool.get_num_free_blocks() < wanted[id(one.block_pool)]:
                self.bump("refused")
                return self.refuse("pool", f"the pool has {one.block_pool.get_num_free_blocks()} free block(s), "
                                           f"{wanted[id(one.block_pool)]} needed for the recurrent-state copies")
        blocks = [[] for _ in managers]
        pages = {}
        held = []
        for index, one, part in shared:
            one.block_pool.touch(part)
            held.append((one.block_pool, list(reversed(part))))
            blocks[index] = part
        copied = moved = 0
        for index, one, place, source in states:
            spare = one.block_pool.get_new_blocks(1)[0]
            source.ref_cnt += 1
            spare.ref_cnt += 1
            one._pending_cow_copies.append((source, spare))
            held.append((one.block_pool, [spare]))
            blocks[index] = [one._null_block] * place + [spare]
            copied += 1
            moved += fork_page_bytes(one)
        resident = 0
        valid = set()
        for index, one, part in residents:
            live = [block for block in part if block is not None]
            one.block_pool.touch(live)
            held.append((one.block_pool, live))
            pages[index] = part
            resident += len(live)
            state = one.coordinator.request_states.get(request.request_id)
            if state is not None:
                valid = set(state.valid_pages)
        tokens = list(fork_tokens(request)[:int(request.num_tokens)])
        backup = Backup(request.request_id, point, tokens, blocks, pages, valid, held)
        self.backups[request.request_id] = backup
        self.bump("backups")
        took = (time.perf_counter() - began) * 1000.0
        attention = sum(len(part) for index, _, part in shared if self.kinds[index] != "HiSparseSourceManager")
        host = sum(len(part) for index, _, part in shared if self.kinds[index] == "HiSparseSourceManager")
        rewind_say(f"backup {request.request_id}: point {point} of {len(tokens)} tokens; {attention} attention "
                   f"block(s), {host} host block(s) and {resident} resident page(s) held; {copied} recurrent-state "
                   f"block(s) copied ({moved / 2 ** 20:.1f} MiB) before the side turn's forward; {took:.2f} ms")
        self.report()
        return {"backed": True, "point": point, "tokens": len(tokens), "ms": round(took, 3)}

    def restore(self, rid):
        began = time.perf_counter()
        self.sweep()
        request = self.find(rid)
        backup = self.backups.pop(request.request_id, None) if request is not None else None
        why = None
        if request is None:
            why = f"no engine request {rid}"
        elif backup is None:
            why = "no backup is held for this session"
        else:
            why = self.idle(request)
        if why is None:
            free = getattr(self.scheduler, "_request_blocks_can_be_freed", None)
            if callable(free) and not free(request):
                why = "a step that may still write the session's blocks has not been processed"
        if why is None:
            for feature in getattr(request, "mm_features", None) or ():
                place = feature.mm_position
                if place.offset < backup.point < place.offset + place.length:
                    why = f"an image at {place.offset} straddles the point {backup.point}"
                    break
        if why is None and int(request.num_tokens) < len(backup.tokens):
            why = f"the session holds {request.num_tokens} tokens, fewer than the backup's {len(backup.tokens)}"
        if why is None and lcp_blocks(fork_tokens(request), backup.tokens, len(backup.tokens)) < len(backup.tokens):
            why = "the session's tokens no longer start with the backup's"
        if why is not None:
            if backup is not None:
                self.release(backup)
                self.bump("dropped")
            self.bump("fell_back")
            return self.refuse("restore", why)
        backup.side = int(request.num_tokens) - len(backup.tokens)
        self.manager.free(request)
        rewind_request(request, backup.tokens, backup.point)
        self.pending[request.request_id] = backup
        backup.began = began
        backup.restore_ms = (time.perf_counter() - began) * 1000.0
        return {"rewound": True, "kept": backup.point, "tokens": len(backup.tokens), "side": backup.side,
                "ms": round(backup.restore_ms, 3)}

    def drop(self, rid):
        self.sweep()
        request = self.find(rid)
        backup = self.backups.pop(request.request_id, None) if request is not None else None
        if backup is None:
            return {"dropped": False}
        self.release(backup)
        self.bump("dropped")
        return {"dropped": True}

    def waiting(self, request):
        return self.pending.get(request.request_id)

    def admitted(self, request, plan, using, hit, copies):
        backup = plan.rewind
        self.pending.pop(request.request_id, None)
        self.release(backup)
        device, host, moved = copies
        took = (time.perf_counter() - backup.began) * 1000.0 if backup.began is not None else 0.0
        refed = int(request.num_tokens) - int(hit)
        if using:
            self.bump("rewinds")
            self.bump("kept", int(hit))
            self.bump("refed", refed)
            self.bump("side", backup.side)
            self.bump("device_copies", device)
            self.bump("host_copies", host)
            rewind_say(f"rewind {request.request_id}: kept {hit} tokens, re-fed {refed} "
                       f"({len(backup.tokens) - backup.point} the session had not computed, "
                       f"{int(request.num_tokens) - len(backup.tokens)} new), let go {backup.side} side-turn tokens; "
                       f"{device} device and {host} host block copies ({moved / 2 ** 20:.1f} MiB) queued before its "
                       f"forward; restore {backup.restore_ms:.2f} ms, {took:.2f} ms to admission; fell back no")
        else:
            self.bump("fell_back")
            self.reasons[plan.code] = self.reasons.get(plan.code, 0) + 1
            rewind_say(f"rewind {request.request_id}: fell back yes ({plan.why}); {hit} tokens from the prefix "
                       f"cache, {refed} computed; {took:.2f} ms to admission")
        self.report()

    def report(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.reported < REWIND_REPORT_S:
            return None
        span = now - self.reported
        self.reported = now
        tally, self.tally = self.tally, dict.fromkeys(REWIND_TALLY, 0)
        reasons, self.reasons = self.reasons, {}
        line = (f"last {span:.0f} s: {tally['backups']} backup(s), {tally['refused']} refused; {tally['rewinds']} "
                f"rewind(s) kept {tally['kept']} tokens unprefilled and re-fed {tally['refed']}, letting go "
                f"{tally['side']} side-turn tokens; {tally['fell_back']} fell back "
                f"({', '.join(f'{code} {count}' for code, count in sorted(reasons.items())) or 'none'}); "
                f"{tally['dropped']} backup(s) let go unused; {tally['device_copies']} device and "
                f"{tally['host_copies']} host block copies; {len(self.backups)} held, {len(self.pending)} awaiting "
                f"admission now")
        rewind_say(line)
        return line


def rewind_state(scheduler):
    held = scheduler.__dict__.get("_taaf_rewinds")
    if held is None:
        try:
            held = Rewinds(scheduler)
        except Exception as failure:
            held = False
            rewind_say(f"not armed: {type(failure).__name__}: {failure}; every rewind call is refused and each "
                       f"main turn after a side turn takes the stale path")
        scheduler.__dict__["_taaf_rewinds"] = held
    return held


def rewind_call(scheduler, method, rid):
    held = rewind_state(scheduler)
    if not held:
        return {"why": "the rewind machinery is not armed"}
    try:
        return getattr(held, method)(str(rid))
    except Exception as failure:
        return {"why": f"{method} raised {type(failure).__name__}: {failure}"}


def _engine_rewind_backup(self, rid):
    return rewind_call(self.scheduler, "backup", rid)


def _engine_rewind_restore(self, rid):
    return rewind_call(self.scheduler, "restore", rid)


def _engine_rewind_drop(self, rid):
    return rewind_call(self.scheduler, "drop", rid)


def trim_hole_say(text):
    print(f"TRIM_HOLE {text}", flush=True)


def trim_hole_coordinator(managers):
    for one in managers:
        if fork_kind(one) in ("HiSparseResidentManager", "HiSparseSourceManager", "HiSparseHotManager"):
            held = getattr(one, "coordinator", None)
            if held is not None:
                return held
    return None


def trim_hole_busy(coordinator, rid):
    if coordinator is None:
        return None
    if rid in (getattr(coordinator, "_pending_imports", None) or {}):
        return "a host import is still filling the session"
    state = (getattr(coordinator, "request_states", None) or {}).get(rid)
    if state is not None and getattr(state, "pending_pages", None):
        return f"{len(state.pending_pages)} resident page(s) are still being written to host"
    if state is not None and getattr(state, "publication", None) is not None:
        return "a host publication is pending"
    for spill in (getattr(coordinator, "pending_spills", None) or {}).values():
        page = getattr(spill, "page", None)
        if page is not None and page[0] == rid:
            return "a page spill of the session is in flight"
    return None


def trim_hole_hot_release():
    return str(os.environ.get(TRIM_HOLE_HOT_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def trim_hole_pages(pages, first, last):
    gap = last - first
    return {page if page < first else page - gap for page in pages if not first <= page < last}


def trim_hole_mark(count, first, last):
    if count <= first:
        return count
    return first if count <= last else count - (last - first)


def trim_hole_probe(scheduler, rid, low, high):
    request = fork_find_main(scheduler.requests, rid)
    if request is None:
        return None, "state", f"no engine request {rid}"
    if fork_status(request) != "WAITING_FOR_STREAMING_REQ":
        return None, "state", f"the session is {fork_status(request)}, not paused"
    if int(getattr(request, "num_output_placeholders", 0) or 0) or getattr(request, "spec_token_ids", None):
        return None, "state", "the session has tokens in flight"
    free = getattr(scheduler, "_request_blocks_can_be_freed", None)
    if callable(free) and not free(request):
        return None, "state", "a step that may still write the session's blocks has not been processed"
    rewinds = getattr(scheduler, "__dict__", {}).get("_taaf_rewinds")
    if rewinds and (request.request_id in rewinds.backups or request.request_id in rewinds.pending):
        return None, "state", "a rewind backup holds the session"
    computed = int(request.num_computed_tokens)
    prompt = int(getattr(request, "num_prompt_tokens", computed))
    if not 0 < low < high <= min(computed, prompt):
        return None, "range", f"the hole {low}..{high} is not inside the {min(computed, prompt)} computed prompt tokens"
    hashed = int(getattr(scheduler, "hash_block_size", 0) or 0)
    if hashed and (low % hashed or high % hashed):
        return None, "align", f"the hole {low}..{high} is not on the {hashed}-token hash blocks"
    for feature in getattr(request, "mm_features", None) or ():
        place = feature.mm_position
        end = place.offset + place.length
        if place.offset < high and low < end and not low <= place.offset < end <= high:
            return None, "image", f"an image at {place.offset}..{end} straddles the hole"
        if end > computed:
            return None, "image", f"an image at {place.offset}..{end} is not computed"
    managers = list(scheduler.kv_cache_manager.coordinator.single_type_managers)
    span = high - low
    hosts = set()
    for index, one in enumerate(managers):
        kind = fork_kind(one)
        size = int(one.block_size)
        table = one.req_to_blocks.get(request.request_id) or []
        if request.request_id in (getattr(one, "_partial_hit_reqs", None) or {}):
            return None, "state", f"group {index} ({kind}) holds a partial prefix hit for the session"
        if kind in TRIM_HOLE_RING:
            if span % size:
                return None, "ring", f"group {index} ({kind}) rings {size} tokens and the hole is {span}"
            continue
        if kind in TRIM_HOLE_RELEASE:
            if table and not trim_hole_hot_release():
                return None, "hot", (f"group {index} holds hot blocks and {TRIM_HOLE_HOT_ENV} is off, so the worker "
                                     f"would keep their page bookkeeping across the hole")
            continue
        if kind not in TRIM_HOLE_DROP:
            return None, "group", f"group {index} ({type(one).__name__}) has no hole rule"
        if low % size or high % size:
            return None, "align", f"group {index} ({kind}) pages {size} tokens and the hole {low}..{high} is not on them"
        if kind == "MambaManager":
            for name in ("_checkpoints", "_producer_partial_tail_reqs"):
                if request.request_id in (getattr(one, name, None) or {}):
                    return None, "mamba", f"group {index} holds a pending Mamba {name.strip('_')} entry"
            last = (getattr(one, "last_state_block_idx", None) or {}).get(request.request_id)
            if last is not None and low // size <= last < high // size:
                return None, "mamba", f"group {index}'s last state block {last} lies in the hole"
        if kind == "HiSparseSourceManager":
            hosts.update(block.block_id for block in table[low // size:high // size] if not block.is_null)
    coordinator = trim_hole_coordinator(managers)
    why = trim_hole_busy(coordinator, request.request_id)
    if why is None and coordinator is not None and hosts:
        for pair in (getattr(coordinator, "_retained_copies", None) or {}).values():
            if any(getattr(block, "block_id", None) in hosts for block in pair):
                why = "a host copy of a hole page is in flight"
                break
    if why is not None:
        return None, "host_busy", why
    return {"request": request, "computed": computed, "dropped": span}, None, None


def trim_hole_resident(coordinator, one, rid, first, part):
    state = (getattr(coordinator, "request_states", None) or {}).get(rid) if coordinator is not None else None
    dead = []
    for offset, block in enumerate(part):
        if block.is_null:
            continue
        page = first + offset
        if state is not None and page in state.unpinned_pages:
            owners = coordinator._owners.get(block.block_id)
            if owners is not None:
                owners.discard((rid, page))
        elif state is not None and page in state.valid_pages and page not in state.pending_pages:
            one.block_pool.unpin_blocks([block], coordinator._on_block_reused)
        else:
            dead.append(block)
    return dead


def trim_hole_remap(coordinator, rid, first, last):
    if coordinator is None:
        return
    state = (getattr(coordinator, "request_states", None) or {}).get(rid)
    if state is not None:
        state.valid_pages = trim_hole_pages(state.valid_pages, first, last)
        state.pinned_clean = trim_hole_pages(state.pinned_clean, first, last)
        state.unpinned_pages = trim_hole_pages(state.unpinned_pages, first, last)
        state.ready_prefix_pages = trim_hole_mark(state.ready_prefix_pages, first, last)
        state.copies_recorded_blocks = trim_hole_mark(state.copies_recorded_blocks, first, last)
    for owners in (getattr(coordinator, "_owners", None) or {}).values():
        mine = [pair for pair in owners if pair[0] == rid]
        for pair in mine:
            owners.discard(pair)
            if not first <= pair[1] < last:
                owners.add((rid, pair[1] if pair[1] < first else pair[1] - (last - first)))
    updates = getattr(coordinator, "block_table_updates", None)
    if updates is not None:
        updates.add(rid)


def trim_hole_salted(request):
    if getattr(request, "_taaf_hole_salted", False):
        return
    plain = getattr(request, "_block_hasher", None)
    if plain is None:
        return
    salt = f"taaf-hole:{request.request_id}:".encode()

    def salted(held):
        made = []
        for value in plain(held):
            raw = bytes(value)
            made.append(type(value)(hashlib.sha256(salt + raw).digest()[:len(raw)]))
        return made

    request._block_hasher = salted
    request._taaf_hole_salted = True


def trim_hole_unhash(manager, blocks):
    pool = manager.block_pool
    remove = getattr(pool, "_remove_cached_block_hashes", None)
    if remove is None:
        return 0
    count = 0
    for block in blocks:
        if block.is_null:
            continue
        if getattr(block, "block_hash", None) is None and not (
                getattr(pool, "cached_block_hashes_by_block", None) or {}).get(block.block_id):
            continue
        removed = remove(block)
        emit = getattr(pool, "_emit_block_removed_events", None)
        if emit is not None:
            emit(removed)
        count += 1
    return count


def trim_hole_compact(scheduler, request, low, high):
    from dataclasses import replace
    rid = request.request_id
    span = high - low
    del request.prompt_token_ids[low:high]
    del request._all_token_ids[low:high]
    request.num_prompt_tokens = int(request.num_prompt_tokens) - span
    request.num_computed_tokens = int(request.num_computed_tokens) - span
    encoder = getattr(scheduler, "encoder_cache_manager", None)
    if encoder is not None:
        for input_id in sorted(encoder.get_cached_input_ids(request), reverse=True):
            encoder.free_encoder_input(request, input_id)
    features = getattr(request, "mm_features", None)
    if features:
        kept = []
        for one in features:
            place = one.mm_position
            if place.offset >= high:
                one.mm_position = replace(place, offset=place.offset - span)
                kept.append(one)
            elif place.offset + place.length <= low:
                kept.append(one)
        features[:] = kept
    hashed = int(getattr(scheduler, "hash_block_size", 0) or 0)
    if hashed:
        trim_hole_salted(request)
        del request.block_hashes[low // hashed:]
        request.update_block_hashes()
    managers = list(scheduler.kv_cache_manager.coordinator.single_type_managers)
    coordinator = trim_hole_coordinator(managers)
    freed = {}
    pages = None
    for one in managers:
        kind = fork_kind(one)
        table = one.req_to_blocks.get(rid)
        if kind in TRIM_HOLE_RING or not table:
            continue
        if kind in TRIM_HOLE_RELEASE:
            dead = [block for block in one.req_to_blocks.pop(rid, ()) if not block.is_null]
            one.require_hot(rid)
        else:
            size = int(one.block_size)
            first, last = low // size, high // size
            part = list(table[first:last])
            if kind == "HiSparseResidentManager":
                dead = trim_hole_resident(coordinator, one, rid, first, part)
                pages = (first, last)
            else:
                dead = [block for block in part if not block.is_null]
            del table[first:last]
            trim_hole_unhash(one, table[first:])
            cached = one.num_cached_block.get(rid)
            if cached is not None:
                one.num_cached_block[rid] = trim_hole_mark(cached, first, last)
            if kind == "MambaManager":
                states = getattr(one, "last_state_block_idx", None)
                if states is not None and rid in states and states[rid] >= last:
                    states[rid] -= last - first
                retired = getattr(one, "_num_retired_blocks", None)
                if retired is not None and rid in retired:
                    retired[rid] = trim_hole_mark(retired[rid], first, last)
        if dead:
            one.block_pool.free_blocks(list(reversed(dead)))
        freed[kind] = freed.get(kind, 0) + len(dead)
    if pages is not None:
        trim_hole_remap(coordinator, rid, *pages)
    elif coordinator is not None and getattr(coordinator, "block_table_updates", None) is not None:
        coordinator.block_table_updates.add(rid)
    return freed


def trim_hole_engine(scheduler, rid, low, high, shift):
    began = time.perf_counter()
    plan = freed = None
    try:
        plan, code, why = trim_hole_probe(scheduler, str(rid), int(low), int(high))
    except Exception as failure:
        plan, code, why = None, "error", f"the probe raised {type(failure).__name__}: {failure}"
    if plan is not None:
        try:
            freed = trim_hole_compact(scheduler, plan["request"], int(low), int(high))
            code, why = "punched", None
        except Exception as failure:
            plan, code, why = None, "broken", (f"the compaction raised {type(failure).__name__}: {failure}; the "
                                               f"session is closed and recomputed")
    took = (time.perf_counter() - began) * 1000.0
    tally = _TRIM_HOLE["tally"]
    tally[code] = tally.get(code, 0) + 1
    count = sum(freed.values()) if freed else 0
    if freed is not None:
        tally["tokens"] = tally.get("tokens", 0) + int(high) - int(low)
        tally["blocks"] = tally.get("blocks", 0) + count
        trim_hole_say(f"engine {rid}: punched hole {low}..{high} of {plan['computed']} computed tokens, positions "
                      f"shift {shift}; freed {', '.join(f'{value} {name}' for name, value in freed.items())}; "
                      f"{took:.2f} ms; tally {tally}")
        return {"punched": True, "freed": count, "ms": round(took, 3)}
    trim_hole_say(f"engine {rid}: hole {low}..{high}, positions shift {shift}; refused {code}: {why}; "
                  f"{took:.2f} ms; tally {tally}")
    return {"punched": False, "code": code, "why": why, "freed": 0}


def trim_hole_rope_at(rope, states, index, at):
    if at < int(states.prefill_len.np[index]):
        base = rope.num_dims * index
        return [int(value) for value in rope.prefill_positions.gpu[base:base + rope.num_dims, at].tolist()]
    return [at + int(rope.prefill_delta.np[index])] * rope.num_dims


def trim_hole_marked(runner, scheduler_output):
    marked = {}
    rope = getattr(runner.model_state, "rope_state", None)
    for new in scheduler_output.scheduled_new_reqs:
        extra = getattr(new.sampling_params, "extra_args", None) or {}
        mark = extra.get(TRIM_HOLE_MARK)
        if not mark:
            continue
        low, anchor, tokens = (int(value) for value in mark)
        index = runner.req_states.req_id_to_index.get(new.req_id)
        old = None
        if rope is not None and index is not None:
            old = trim_hole_rope_at(rope, runner.req_states, index, anchor)
        marked[new.req_id] = (low, anchor, tokens, old)
    return marked


def trim_hole_shift(runner, req_id, low, anchor, tokens, old):
    import torch
    tally = _TRIM_HOLE["tally"]
    rope = getattr(runner.model_state, "rope_state", None)
    index = runner.req_states.req_id_to_index.get(req_id)
    if rope is None or index is None:
        tally["worker_unshifted"] = tally.get("worker_unshifted", 0) + 1
        trim_hole_say(f"worker {req_id}: no M-RoPE state for the request, positions left compacted")
        return None
    new = trim_hole_rope_at(rope, runner.req_states, index, low)
    shift = [tokens] * rope.num_dims if old is None else [one - other for one, other in zip(old, new)]
    length = int(runner.req_states.prefill_len.np[index])
    base = rope.num_dims * index
    rows = rope.prefill_positions.gpu[base:base + rope.num_dims, low:length]
    rows += torch.tensor(shift, dtype=rows.dtype, device=rows.device).view(-1, 1)
    rope.prefill_delta.np[index] += shift[0]
    rope.prefill_delta.copy_to_uva()
    tally["worker_shifted"] = tally.get("worker_shifted", 0) + 1
    if old is None:
        tally["worker_by_tokens"] = tally.get("worker_by_tokens", 0) + 1
    if len(set(shift)) != 1:
        tally["worker_dims_differ"] = tally.get("worker_dims_differ", 0) + 1
    if tally["worker_shifted"] <= TRIM_HOLE_WORKER_SAY or old is None or len(set(shift)) != 1:
        trim_hole_say(f"worker {req_id}: rope positions from index {low} to {length} shifted by {shift} "
                      f"({'old state absent, by tokens' if old is None else f'old index {anchor} at {old}'}, new "
                      f"index {low} at {new}); tally {tally}")
    return shift


def fork_hole_marked(runner, scheduler_output):
    marked = {}
    rope = getattr(runner.model_state, "rope_state", None)
    for new in scheduler_output.scheduled_new_reqs:
        extra = getattr(new.sampling_params, "extra_args", None) or {}
        mark = extra.get(FORK_HOLE_AT)
        if not mark:
            continue
        whole, main = int(mark[0]), str(mark[1])
        index = runner.req_states.req_id_to_index.get(main)
        at = None
        if rope is not None and index is not None:
            at = trim_hole_rope_at(rope, runner.req_states, index, whole - 1)
        marked[new.req_id] = (whole, main, int(new.num_computed_tokens), at)
    return marked


def fork_hole_shift(runner, req_id, whole, main, computed, at):
    import torch
    tally = _TRIM_HOLE["tally"]
    rope = getattr(runner.model_state, "rope_state", None)
    index = runner.req_states.req_id_to_index.get(req_id)
    if rope is None or index is None or at is None:
        tally["fork_unshifted"] = tally.get("fork_unshifted", 0) + 1
        missing = ("no M-RoPE state" if rope is None else "no row for the branch" if index is None
                   else f"no row for main {main}")
        trim_hole_say(f"worker {req_id}: branch of a main with a hole left unshifted ({missing}); tally {tally}")
        return None
    mine = trim_hole_rope_at(rope, runner.req_states, index, whole - 1)
    shift = [one - other for one, other in zip(at, mine)]
    length = int(runner.req_states.prefill_len.np[index])
    base = rope.num_dims * index
    rows = rope.prefill_positions.gpu[base:base + rope.num_dims, whole:length]
    rows += torch.tensor(shift, dtype=rows.dtype, device=rows.device).view(-1, 1)
    rope.prefill_delta.np[index] += shift[0]
    rope.prefill_delta.copy_to_uva()
    tally["fork_shifted"] = tally.get("fork_shifted", 0) + 1
    if computed != whole:
        tally["fork_short"] = tally.get("fork_short", 0) + 1
    if tally["fork_shifted"] <= TRIM_HOLE_WORKER_SAY or computed != whole or len(set(shift)) != 1:
        trim_hole_say(f"worker {req_id}: branch of main {main} with a hole: rope positions from index {whole} to "
                      f"{length} shifted by {shift} (index {whole - 1} at {at} in the main, {mine} in the branch); "
                      f"{computed} of {whole} shared tokens computed at admission"
                      + ("" if computed == whole else " (fork short: the rest recompute unshifted)")
                      + f"; tally {tally}")
    return shift


_FORK_SYNC = {"installed": False, "left": 0, "step": 0, "branches": (), "said": 0, "stages": set()}


FORK_SYNC_STAGES = ("before", "add", "update", "load", "forward", "finish", "sample", "modules", "ops", "eager")
FORK_SYNC_OPT_IN = {"modules", "ops", "eager"}
FORK_SYNC_SAMPLE_CALLS = ((None, "sample", "sampling"), (None, "postprocess_sampled", "postprocess_sampled"),
                          ("model_state", "gather_mm_embeddings", "the image embeddings gathered for the drafter"),
                          ("speculator", "propose", "the drafter's propose"),
                          ("draft_tokens_handler", "set_draft_tokens", "set_draft_tokens"),
                          ("kv_connector", "post_forward", "the connector's post_forward (host-KV saves and mirrors)"))


def fork_sync_on():
    return str(os.environ.get(FORK_SYNC_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def fork_sync_stages():
    said = str(os.environ.get(FORK_SYNC_ENV) or "").strip().lower()
    if said in ("1", "true", "yes", "on", "all"):
        return set(FORK_SYNC_STAGES) - FORK_SYNC_OPT_IN
    named = {part.strip() for part in said.replace(";", ",").split(",") if part.strip()}
    wrong = named - set(FORK_SYNC_STAGES)
    if wrong:
        raise ValueError(f"{FORK_SYNC_ENV}={said!r} names {sorted(wrong)}; stages are {', '.join(FORK_SYNC_STAGES)}")
    return named


def fork_sync_say(text):
    print(f"FORK_SYNC {text}", flush=True)


def fork_sync_check(stage, short):
    if _FORK_SYNC["left"] <= 0 or short not in _FORK_SYNC["stages"]:
        return
    import torch
    try:
        torch.cuda.synchronize()
    except Exception as failure:
        fork_sync_say(f"fault after {stage}, step {_FORK_SYNC['step']} since branch(es) "
                      f"{', '.join(_FORK_SYNC['branches'])} were admitted: {type(failure).__name__}: "
                      f"{str(failure)[:300]}")
        raise


def fork_sync_open(scheduler_output):
    new = [one.req_id for one in getattr(scheduler_output, "scheduled_new_reqs", ()) or ()
           if (getattr(getattr(one, "sampling_params", None), "extra_args", None) or {}).get(FORK_OF)]
    if new:
        if _FORK_SYNC["left"] > 0:
            fork_sync_check("the previous step", "before")
        _FORK_SYNC.update(left=FORK_SYNC_STEPS, step=0, branches=tuple(new))
        fork_sync_check("the steps before the branch", "before")
    elif _FORK_SYNC["left"] > 0:
        _FORK_SYNC["step"] += 1
        fork_sync_check("the previous step", "before")


def fork_sync_names(runner):
    held = _FORK_SYNC.get("names")
    if held is None:
        model = getattr(runner, "model", None)
        try:
            held = {id(module): name for name, module in model.named_modules()} if model is not None else {}
        except Exception:
            held = {}
        _FORK_SYNC["names"] = held
    return held


def fork_sync_modules(runner):
    if _FORK_SYNC["left"] <= 0 or "modules" not in _FORK_SYNC["stages"]:
        return None
    import torch
    names = fork_sync_names(runner)
    seen = {"count": 0, "last": "the start of the forward"}

    def after(module, args, output):
        if torch.cuda.is_current_stream_capturing():
            return None
        label = f"{names.get(id(module), '?')} ({type(module).__name__})"
        try:
            torch.cuda.synchronize()
        except Exception as failure:
            fork_sync_say(f"fault after module {label}, the last clean one {seen['last']}, {seen['count']} "
                          f"modules clean in step {_FORK_SYNC['step']} since branch(es) "
                          f"{', '.join(_FORK_SYNC['branches'])}: {type(failure).__name__}: {str(failure)[:200]}")
            raise
        seen["count"] += 1
        seen["last"] = label
        return None

    handle = torch.nn.modules.module.register_module_forward_hook(after)
    return handle, seen


def fork_sync_site():
    import traceback
    here = os.path.basename(__file__)
    for frame in reversed(traceback.extract_stack()[:-1]):
        name = frame.filename.replace("\\", "/")
        if "/torch/" in name or name.rsplit("/", 1)[-1] == here:
            continue
        return f"{name.rsplit('/', 1)[-1]}:{frame.lineno} {frame.name}"
    return "?"


def fork_sync_ops(seen):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class SyncEachOp(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if torch.cuda.is_current_stream_capturing():
                return out
            site = fork_sync_site()
            try:
                torch.cuda.synchronize()
            except Exception as failure:
                fork_sync_say(f"fault after op {func} at {site}, the last clean op {seen['last']}, {seen['ops']} ops "
                              f"clean in sample_tokens, step {_FORK_SYNC['step']} since branch(es) "
                              f"{', '.join(_FORK_SYNC['branches'])}: {type(failure).__name__}: {str(failure)[:200]}")
                raise
            seen["ops"] += 1
            seen["last"] = f"{func} at {site}"
            return out

    return SyncEachOp()


def fork_sync_ended(seen):
    import threading
    import torch
    try:
        torch.cuda.synchronize()
    except Exception as failure:
        passed = ", ".join(f"{label} x{count}" for label, count in seen["passed"].items()) or "none"
        ops = (f"{seen['ops']} ops synchronized clean, the last {seen['last']}; "
               if "ops" in _FORK_SYNC["stages"] else "")
        threads = ", ".join(sorted(thread.name for thread in threading.enumerate()))
        fork_sync_say(f"fault after sample_tokens, step {_FORK_SYNC['step']} since branch(es) "
                      f"{', '.join(_FORK_SYNC['branches'])} were admitted: {type(failure).__name__}: "
                      f"{str(failure)[:300]}; inner checks passed: {passed}; {ops}threads alive: {threads}")
        raise


def fork_sync_sampled(runner, run, args, kwargs):
    stages = _FORK_SYNC["stages"]
    if _FORK_SYNC["left"] <= 0 or not {"sample", "ops"} & stages:
        return run(runner, *args, **kwargs)
    seen = {"ops": 0, "last": "the start of sample_tokens", "passed": {}}
    held = []
    for field, name, label in FORK_SYNC_SAMPLE_CALLS if "sample" in stages else ():
        owner = runner if field is None else getattr(runner, field, None)
        original = getattr(owner, name, None) if owner is not None else None
        if original is None:
            continue
        held.append((owner, name, original, name in vars(owner)))

        def checked(*inner, _original=original, _label=label, **named):
            result = _original(*inner, **named)
            fork_sync_check(_label, "sample")
            seen["passed"][_label] = seen["passed"].get(_label, 0) + 1
            return result

        setattr(owner, name, checked)
    try:
        if "ops" in stages:
            with fork_sync_ops(seen):
                result = run(runner, *args, **kwargs)
        else:
            result = run(runner, *args, **kwargs)
        fork_sync_ended(seen)
        return result
    finally:
        for owner, name, original, own in held:
            if own:
                setattr(owner, name, original)
            else:
                vars(owner).pop(name, None)


def fork_sync_close():
    if _FORK_SYNC["left"] <= 0:
        return
    fork_sync_check("finish_forward", "finish")
    if _FORK_SYNC["said"] < FORK_SYNC_SAY:
        _FORK_SYNC["said"] += 1
        fork_sync_say(f"step {_FORK_SYNC['step']} since branch(es) {', '.join(_FORK_SYNC['branches'])} clean; "
                      f"synchronized after: {', '.join(stage for stage in FORK_SYNC_STAGES if stage in _FORK_SYNC['stages'])}")
    _FORK_SYNC["left"] -= 1


def fork_sync_install():
    if _FORK_SYNC["installed"] or not fork_sync_on():
        return False
    _FORK_SYNC["stages"] = fork_sync_stages()
    runner_module = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    connector_module = sys.modules.get("vllm.v1.worker.gpu.kv_connector")
    runner = getattr(runner_module, "GPUModelRunner", None) if runner_module is not None else None
    active = getattr(connector_module, "ActiveKVConnector", None) if connector_module is not None else None
    if runner is None or active is None:
        return False
    execute, add, update = runner.execute_model, runner.add_requests, runner.update_requests
    sample_tokens = getattr(runner, "sample_tokens", None)
    pre, finish = active.pre_forward, active.finish_forward

    def execute_model(self, scheduler_output, *args, **kwargs):
        fork_sync_open(scheduler_output)
        held = fork_sync_modules(self)
        try:
            return execute(self, scheduler_output, *args, **kwargs)
        finally:
            if held is not None:
                held[0].remove()
                fork_sync_say(f"modules synchronized one by one in step {_FORK_SYNC['step']}: "
                              f"{held[1]['count']} clean, the last {held[1]['last']}")

    def add_requests(self, scheduler_output):
        result = add(self, scheduler_output)
        fork_sync_check("add_requests", "add")
        return result

    def update_requests(self, scheduler_output):
        result = update(self, scheduler_output)
        fork_sync_check("update_requests (new-block zeroing and the copy-on-write block copies)", "update")
        return result

    def pre_forward(self, *args, **kwargs):
        result = pre(self, *args, **kwargs)
        fork_sync_check("the connector's pre_forward (the host-KV load)", "load")
        return result

    def finish_forward(self, *args, **kwargs):
        fork_sync_check("the forward", "forward")
        result = finish(self, *args, **kwargs)
        fork_sync_close()
        return result

    dispatch = getattr(runner_module, "dispatch_cg_and_sync_dp", None)
    if dispatch is not None and "eager" in _FORK_SYNC["stages"]:
        def eager_dispatch(*args, **kwargs):
            if _FORK_SYNC["left"] > 0:
                kwargs["need_eager"] = True
                if _FORK_SYNC.get("eager_said", 0) < FORK_SYNC_SAY:
                    _FORK_SYNC["eager_said"] = _FORK_SYNC.get("eager_said", 0) + 1
                    fork_sync_say(f"step {_FORK_SYNC['step']} since branch(es) {', '.join(_FORK_SYNC['branches'])} "
                                  f"runs eager, so every module of its forward runs outside a graph")
            return dispatch(*args, **kwargs)

        runner_module.dispatch_cg_and_sync_dp = eager_dispatch
    runner.execute_model = execute_model
    if sample_tokens is not None:
        runner.sample_tokens = lambda self, *args, **kwargs: fork_sync_sampled(self, sample_tokens, args, kwargs)
    runner.add_requests = add_requests
    runner.update_requests = update_requests
    active.pre_forward = pre_forward
    active.finish_forward = finish_forward
    _FORK_SYNC["installed"] = True
    fork_sync_say(f"armed in pid {os.getpid()} ({FORK_SYNC_ENV}): the step that first schedules a fork branch and "
                  f"the {FORK_SYNC_STEPS - 1} after it synchronize the device after "
                  f"{', '.join(stage for stage in FORK_SYNC_STAGES if stage in _FORK_SYNC['stages'])} "
                  f"(of {', '.join(FORK_SYNC_STAGES)}), so a fault names its stage")
    return True


def trim_hole_worker_install():
    try:
        mode = trim_hole_mode()
    except ValueError:
        return False
    runner_module = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    runner = getattr(runner_module, "GPUModelRunner", None) if runner_module is not None else None
    if mode != "1" or runner is None or getattr(runner, "_taaf_trim_hole", False):
        return False
    original = runner.add_requests

    def add_requests(self, scheduler_output):
        marked = trim_hole_marked(self, scheduler_output)
        forks = fork_hole_marked(self, scheduler_output)
        gaps = rope_gap_marked(scheduler_output)
        result = original(self, scheduler_output)
        for req_id, (low, anchor, tokens, old) in marked.items():
            trim_hole_shift(self, req_id, low, anchor, tokens, old)
        for req_id, (low, gap) in gaps.items():
            rope_gap_shift(self, req_id, low, gap)
        for req_id, (whole, main, computed, at) in forks.items():
            fork_hole_shift(self, req_id, whole, main, computed, at)
        return result

    runner.add_requests = add_requests
    runner._taaf_trim_hole = True
    _TRIM_HOLE["worker"] = True
    trim_hole_say(f"worker hook installed in pid {os.getpid()}: a re-added request marked {TRIM_HOLE_MARK} gets its "
                  f"M-RoPE prefill positions and decode delta shifted from the hole's low edge by the old position of "
                  f"the same token, so the QSA rotary, the indexer's query and its compressed keys read true positions")
    return True


_MAMBA_HIT = {"installed": False, "pending": [], "groups": None, "controls": 0}
MAMBA_HIT_CONTROLS = 24
MAMBA_HIT_CONTROL_TOKENS = 1024


def mamba_hit_on():
    return str(os.environ.get(MAMBA_HIT_ENV) or "").strip() == "1"


def mamba_hit_min():
    said = str(os.environ.get(MAMBA_HIT_MIN_ENV) or "").strip()
    return int(said) if said else MAMBA_HIT_MIN


def mamba_hit_groups(runner):
    held = _MAMBA_HIT["groups"]
    if held is not None:
        return held
    layers = runner.compilation_config.static_forward_context
    held = []
    for index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        states = []
        for name in group.layer_names:
            cache = getattr(layers.get(name), "kv_cache", None)
            state = getattr(layers.get(name), "taaf_q8_full", None)
            if state is None:
                state = cache[1] if isinstance(cache, (list, tuple)) and len(cache) > 1 else None
            if state is not None and getattr(state, "dim", lambda: 0)() == 4:
                states.append(state)
        if states:
            held.append((index, int(group.kv_cache_spec.block_size), states))
    _MAMBA_HIT["groups"] = held
    return held


def mamba_state_stats(state, block):
    import torch
    blocks, heads, v_dim, k_dim = state.shape
    if str(os.environ.get(Q8_STATE_ENV) or "").strip() == "1":
        codes_bytes = heads * v_dim * k_dim
        need = codes_bytes + heads * v_dim * 4
        codes = state.view(torch.int8)
        raw = torch.as_strided(codes, (need,), (1,),
                               codes.storage_offset() + block * state.stride(0) * state.element_size())
        scales = raw[codes_bytes:].view(torch.float32)
        finite = torch.isfinite(scales)
        peak = torch.where(finite, scales.abs(), torch.zeros_like(scales)).max() * 127
        return int((~finite).sum()), float(peak), int((scales == 1.0).sum())
    values = state[block].float()
    finite = torch.isfinite(values)
    flat = torch.where(finite, values.abs(), torch.zeros_like(values))
    return int((~finite).sum()), float(flat.max()), int((values.abs().amax(-1) == 0).sum())


def mamba_blocks_stats(groups, blocks):
    bad, peaks, flat = 0, [], 0
    for (index, size, states), block in zip(groups, blocks):
        if block is None or block <= 0:
            continue
        for state in states:
            count, peak, ones = mamba_state_stats(state, block)
            bad += count
            flat += ones
            peaks.append(peak)
    peaks.sort()
    return bad, len(peaks), (peaks[-1] if peaks else 0.0), (peaks[len(peaks) // 2] if peaks else 0.0), flat


def mamba_hit_marked(runner, scheduler_output):
    floor = mamba_hit_min()
    marked = []
    groups = None
    copies = {int(dst): int(src) for src, dst in (getattr(scheduler_output, "kv_cache_block_copies", None) or ())}
    for new in scheduler_output.scheduled_new_reqs:
        computed = int(new.num_computed_tokens)
        scheduled = int(scheduler_output.num_scheduled_tokens.get(new.req_id, 0))
        control = (computed == 0 and scheduled >= MAMBA_HIT_CONTROL_TOKENS
                   and _MAMBA_HIT["controls"] < MAMBA_HIT_CONTROLS)
        if computed < floor and not control:
            continue
        groups = groups or mamba_hit_groups(runner)
        if not groups:
            return []
        if control:
            _MAMBA_HIT["controls"] += 1
        source, dest = [], []
        for index, size, _ in groups:
            row = list(new.block_ids[index]) if index < len(new.block_ids) else []
            at = (computed - 1) // size
            after = -(-(computed + scheduled) // size) - 1
            held = row[at] if computed > 0 and 0 <= at < len(row) else None
            source.append(copies.get(int(held), held) if held is not None else None)
            dest.append(row[after] if 0 <= after < len(row) else None)
        marked.append((new.req_id, computed, computed + scheduled, source, dest))
    return marked


def mamba_hit_say(rid, computed, after, before, now=None):
    bad, layers, peak, median, flat = before
    if computed == 0:
        line = f"MAMBA_HIT_CHECK control {rid}: no mamba prefix hit"
    else:
        line = (f"MAMBA_HIT_CHECK {rid}: a mamba prefix hit of {computed} tokens; the hit block of {layers} GDN "
                f"layer(s) holds {bad} non-finite value(s) or scale(s), peak |state| {peak:.4g} (layer median "
                f"{median:.4g}), {flat} row(s) at scale 1.0 or all zero")
    if now is not None:
        bad, layers, peak, median, flat = now
        line += (f"; after the step that ran it to {after} tokens: {bad} non-finite, peak {peak:.4g} "
                 f"(median {median:.4g}), {flat} row(s) at scale 1.0 or all zero over {layers} layer(s)")
    print(line, file=sys.stderr, flush=True)
    return line


def mamba_hit_checked(runner, scheduler_output, original):
    if not mamba_hit_on():
        return original(runner, scheduler_output)
    for rid, computed, after, before, dest in _MAMBA_HIT["pending"]:
        mamba_hit_say(rid, computed, after, before, mamba_blocks_stats(mamba_hit_groups(runner), dest))
    _MAMBA_HIT["pending"] = []
    for rid, computed, after, source, dest in mamba_hit_marked(runner, scheduler_output):
        _MAMBA_HIT["pending"].append((rid, computed, after,
                                      mamba_blocks_stats(mamba_hit_groups(runner), source), dest))
    return original(runner, scheduler_output)


def mamba_hit_install():
    if not mamba_hit_on():
        return False
    runner_module = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    runner = getattr(runner_module, "GPUModelRunner", None) if runner_module is not None else None
    if runner is None or getattr(runner, "_taaf_mamba_hit", False):
        return False
    original = runner.add_requests

    def add_requests(self, scheduler_output):
        return mamba_hit_checked(self, scheduler_output, original)

    runner.add_requests = add_requests
    runner._taaf_mamba_hit = True
    _MAMBA_HIT["installed"] = True
    print(f"MAMBA_HIT_CHECK armed in pid {os.getpid()}: a new request whose mamba prefix hit reaches "
          f"{mamba_hit_min()} tokens prints the hit block's non-finite count, peak |state| and rows at scale 1.0 "
          f"or all zero over its GDN layers (the copy source for a partial hit), and the same for its running "
          f"block after the step that runs it; the first {MAMBA_HIT_CONTROLS} new requests without a hit and at "
          f"least {MAMBA_HIT_CONTROL_TOKENS} scheduled tokens print the running block as a control",
          file=sys.stderr, flush=True)
    return True


_PAUSE = {"installed": False}


def pause_request(requests, rid):
    held = requests.get(rid)
    if held is not None:
        return held
    head = f"{rid}-"
    for name, request in requests.items():
        tail = name[len(head):] if name.startswith(head) else ""
        if len(tail) == 8 and all(one in "0123456789abcdef" for one in tail) and not fork_is_branch(request):
            return request
    return None


def pause_engine(scheduler, rid, think_end=-1):
    from vllm.v1.request import RequestStatus
    request = pause_request(getattr(scheduler, "requests", {}), rid)
    if request is None:
        return {"paused": False, "why": "no_request"}
    output = list(request.output_token_ids)
    if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ or request.is_finished():
        return {"paused": False, "why": "turn_ended", "generated": len(output)}
    if think_end is not None and int(think_end) >= 0 and int(think_end) in output:
        return {"paused": False, "why": "thinking_closed", "generated": len(output)}
    request.taaf_pause = True
    return {"paused": True, "generated": len(output)}


def pause_stop(request):
    if not getattr(request, "taaf_pause", False):
        return False
    from vllm.v1.request import RequestStatus
    request.taaf_pause = False
    request.status = RequestStatus.FINISHED_STOPPED
    request.stop_reason = PAUSE_STOP
    return True


def _engine_pause_now(self, rid, think_end=-1):
    return pause_engine(self.scheduler, rid, think_end)


def pause_attach():
    if _PAUSE["installed"]:
        return False
    import importlib
    try:
        scheduler = sys.modules.get("vllm.v1.core.sched.scheduler") or importlib.import_module(
            "vllm.v1.core.sched.scheduler")
        stock = scheduler.check_stop

        def check_stop(request, max_model_len, *args, **kwargs):
            if stock(request, max_model_len, *args, **kwargs):
                request.taaf_pause = False
                return True
            return pause_stop(request)

        scheduler.check_stop = check_stop
        core = importlib.import_module("vllm.v1.engine.core")
        core.EngineCore.taaf_pause_now = _engine_pause_now
    except Exception as failure:
        say(f"pause utility not attached: {type(failure).__name__}: {failure}; every {PAUSE_MARK} request is refused")
        return False
    _PAUSE["installed"] = True
    say(f"pause utility attached (pid {os.getpid()}): a {PAUSE_MARK} request stops its session's running turn at the "
        f"next sampled token as a stop ({PAUSE_STOP}) while the turn's thinking is open, and the session stays "
        f"paused with every generated token in KV")
    return True


def _engine_trim_hole(self, rid, low, high, shift):
    return trim_hole_engine(self.scheduler, rid, low, high, shift)


def trim_hole_attach():
    try:
        mode = trim_hole_mode()
    except ValueError as failure:
        trim_hole_say(f"engine utility not attached: {failure}")
        return False
    if _TRIM_HOLE["installed"] or not mode:
        return False
    import importlib
    try:
        core = importlib.import_module("vllm.v1.engine.core")
        core.EngineCore.taaf_trim_hole = _engine_trim_hole
    except Exception as failure:
        trim_hole_say(f"engine utility not attached: {type(failure).__name__}: {failure}; every hole is refused")
        return False
    _TRIM_HOLE["installed"] = True
    trim_hole_say(f"engine utility attached ({TRIM_HOLE_ENV}={mode}): a hole is checked against every KV group's "
                  f"pages, the session's state and the host tier, then cut from the request's tokens, block tables, "
                  f"host pages and recurrent-state indices; anything it cannot honor is refused and recomputed")
    return True


def rewind_attach(scheduler):
    if _REWIND["installed"]:
        return
    from vllm.v1.engine.core import EngineCore
    EngineCore.taaf_rewind_backup = _engine_rewind_backup
    EngineCore.taaf_rewind_restore = _engine_rewind_restore
    EngineCore.taaf_rewind_drop = _engine_rewind_drop
    _REWIND["installed"] = True


def _record_stats(stats):
    if stats is None:
        return
    held = _state["stats"]
    held["kv"] = float(getattr(stats, "kv_cache_usage", 0.0) or 0.0)
    held["waiting"] = int(getattr(stats, "num_waiting_reqs", 0) or 0) + int(
        getattr(stats, "num_skipped_waiting_reqs", 0) or 0)
    held["at"] = time.monotonic()


def install(engine_class):
    if not enabled() or getattr(engine_class, "_taaf_paused", False):
        return False
    original = engine_class.generate

    async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
        router = _state["router"]
        if router is None or router.engine is not self:
            router = Router(self, original)
            _state["router"] = router
        if args:
            async for out in original(self, prompt, sampling_params, request_id, *args, **kwargs):
                yield out
            return
        async for out in router.generate(prompt, sampling_params, request_id, kwargs):
            yield out

    engine_class.generate = generate
    engine_class._taaf_paused = True
    try:
        from vllm.v1.engine import output_processor
        stock = output_processor.OutputProcessor.update_scheduler_stats

        def update_scheduler_stats(self, scheduler_stats):
            _record_stats(scheduler_stats)
            return stock(self, scheduler_stats)

        output_processor.OutputProcessor.update_scheduler_stats = update_scheduler_stats
    except Exception as failure:
        say(f"pool pressure feed unavailable ({type(failure).__name__}: {failure}); idle and count limits only")
    try:
        lean_api()
    except Exception as failure:
        say(f"lean token ids unavailable in the API server ({type(failure).__name__}: {failure})")
    if pool_on():
        register_gauges()
    say("installed on AsyncLLM.generate")
    return True


trim_hole_worker_install()
fork_sync_install()
mamba_hit_install()


def hot_spans_on():
    return switch_on(HOT_SPANS_ENV)


def hot_spans_summary(rows, order, table_row, block_size, align, gap, top, computed, limit=-1):
    import numpy as np
    rows = np.asarray(rows, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    region = int(rows.shape[0])
    rank = np.zeros(region, dtype=np.float64)
    sane = (order >= 0) & (order < region)
    rank[order[sane]] = np.arange(region, dtype=np.float64)[sane]
    recency = rank / max(region - 1, 1)
    live = rows >= 0
    table_row = np.asarray(table_row, dtype=np.int64)
    known = table_row >= 0
    blocks_sorted = np.argsort(table_row[known], kind="stable")
    keys = table_row[known][blocks_sorted]
    columns = np.flatnonzero(known)[blocks_sorted]
    host = rows[live] // block_size
    at = np.searchsorted(keys, host).clip(0, max(len(keys) - 1, 0))
    found = (len(keys) > 0) & (keys[at] == host) if len(keys) else np.zeros(host.shape, dtype=bool)
    positions = columns[at][found] * block_size + rows[live][found] % block_size
    fresh = recency[live][found]
    if limit >= 0:
        kept = positions < limit
        positions, fresh = positions[kept], fresh[kept]
    result = {"rows": int(live.sum()), "region": region, "mapped": int(found.sum()),
              "unmapped": int(live.sum() - found.sum()), "computed": int(computed), "align": int(align),
              "below": int(limit), "listed": 0, "blocks": [], "spans": []}
    result["listed"] = int(positions.size)
    if positions.size == 0:
        return result
    index = positions // align
    present, inverse = np.unique(index, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=fresh) / counts
    newest = np.full(present.shape, -1.0)
    np.maximum.at(newest, inverse, fresh)
    best = np.lexsort((present, -counts))[:top]
    result["blocks"] = [{"start": int(present[at] * align), "end": int((present[at] + 1) * align),
                         "rows": int(counts[at]), "recency": round(float(means[at]), 3),
                         "newest": round(float(newest[at]), 3)} for at in best]
    ordered = np.argsort(positions, kind="stable")
    spots = positions[ordered]
    ages = fresh[ordered]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(spots) > gap) + 1])
    sizes = np.diff(np.concatenate([starts, [spots.size]]))
    mean_ages = np.add.reduceat(ages, starts) / sizes
    ends = spots[starts + sizes - 1] + 1
    best = np.lexsort((spots[starts], -sizes))[:top]
    result["spans"] = [{"start": int(spots[starts[at]]), "end": int(ends[at]), "rows": int(sizes[at]),
                        "recency": round(float(mean_ages[at]), 3)} for at in best]
    return result


def hot_spans_index(runner):
    layers = runner.compilation_config.static_forward_context
    for layer in layers.values():
        owner = getattr(layer, "taaf_hisparse_group", None)
        if owner is None:
            continue
        try:
            runtime = owner.cache(0).runtime
        except Exception:
            continue
        group = getattr(runtime, "index_group", None)
        if group is not None and getattr(group, "device_global_indices", None) is not None:
            return group
    return None


def hot_spans_worker(worker, rid, align, gap, top, limit=-1, fresh=0, budget=HOT_READS_PICK_TOKENS, above=0):
    runner = getattr(worker, "model_runner", None) or getattr(getattr(worker, "worker", None), "model_runner", None)
    if runner is None:
        return {"why": "the worker has no model runner"}
    began = time.perf_counter()
    slot = runner.req_states.req_id_to_index.get(rid)
    if slot is None:
        return {"why": f"the worker holds no request {rid}"}
    reads = hot_reads_module()
    if reads is not None and reads._READS["counts"] is not None:
        computed = int(runner.req_states.num_computed_tokens_np[slot])
        answer = hot_reads_summary(reads.hot_reads_row(slot, clear=bool(fresh)), 1 << reads.HOT_READS_SHIFT, align,
                                   top, computed, int(limit), int(budget), int(above))
        answer.update(layers=list(reads._READS["layers"]), fresh=int(bool(fresh)),
                      worker_ms=round((time.perf_counter() - began) * 1000.0, 3), slot=int(slot))
        return answer
    group = hot_spans_index(runner)
    if group is None:
        return {"why": "no HiSparse index group in the model"}
    host_ids = list(getattr(runner.kv_cache_config, "host_group_ids", None) or ())
    if len(host_ids) != 1:
        return {"why": f"the KV cache config names {len(host_ids)} host groups"}
    gid = int(host_ids[0])
    block_size = int(runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size)
    count = int(runner.block_tables.num_blocks.np[gid, slot])
    table_row = runner.block_tables.block_tables[gid].gpu[slot, :count].cpu().numpy()
    rows = group.device_global_indices[slot].cpu().numpy()
    order = group.lru_slots[slot].cpu().numpy()
    computed = int(runner.req_states.num_computed_tokens_np[slot])
    answer = hot_spans_summary(rows, order, table_row, block_size, align, gap, top, computed, int(limit))
    answer["worker_ms"] = round((time.perf_counter() - began) * 1000.0, 3)
    answer["slot"] = int(slot)
    return answer


def _engine_hot_spans(self, rid, align=HOT_SPANS_ALIGN, gap=HOT_SPANS_GAP, top=HOT_SPANS_TOP, limit=-1, fresh=0,
                      budget=HOT_READS_PICK_TOKENS, above=0):
    answer = self.model_executor.collective_rpc(hot_spans_worker, args=(rid, int(align), int(gap), int(top),
                                                                        int(limit), int(fresh), int(budget),
                                                                        int(above)))
    if isinstance(answer, list):
        answer = answer[0]
    return answer if isinstance(answer, dict) else {"why": f"worker answered {answer!r}"}


def hot_spans_attach():
    if _HOT_SPANS["installed"] or not hot_spans_on():
        return False
    import importlib
    try:
        core = importlib.import_module("vllm.v1.engine.core")
        core.EngineCore.taaf_hot_spans = _engine_hot_spans
    except Exception as failure:
        say(f"hot spans utility not attached: {type(failure).__name__}: {failure}; every {HOT_SPANS_MARK} request "
            f"answers with its reason")
        return False
    _HOT_SPANS["installed"] = True
    say(f"hot spans utility attached ({HOT_SPANS_ENV}, pid {os.getpid()}): a {HOT_SPANS_MARK} request returns, for its "
        f"key's session, the host-KV rows its QSA layers keep in the hot region (decode reads of the context's "
        f"host-only part) as token blocks of {HOT_SPANS_ALIGN} and runs with their recency, without generating")
    return True


_HOT_SPANS = {"installed": False}


def hot_spans_client_index(state, index):
    if not getattr(state, "holes", 0):
        return index
    for low, high, delta in state.engine_map():
        if low + delta <= index and (high is None or index <= high + delta):
            return index - delta
    return None


_HOT_BREAKS = {}


def hot_spans_breaks(decoder, token):
    key = (id(decoder), int(token))
    held = _HOT_BREAKS.get(key)
    if held is None:
        try:
            held = "\n" in decoder.decode([key[1]], skip_special_tokens=False)
        except Exception:
            held = False
        _HOT_BREAKS[key] = held
    return held


def hot_spans_widen(tokens, low, high, decoder, widen=HOT_SPANS_WIDEN):
    steps = 0
    while low > 0 and steps < widen and not hot_spans_breaks(decoder, tokens[low - 1]):
        low -= 1
        steps += 1
    steps = 0
    while high < len(tokens) and steps < widen and not hot_spans_breaks(decoder, tokens[high - 1]):
        high += 1
        steps += 1
    return low, high


def hot_spans_text(tokens, images, low, high, decoder):
    pieces = []
    at = low
    for offset, length, *_ in sorted(images):
        end = offset + length
        if end <= at or offset >= high:
            continue
        if offset > at:
            pieces.append(decoder.decode([int(one) for one in tokens[at:offset]], skip_special_tokens=False))
        pieces.append(HOT_SPANS_IMAGE)
        at = max(at, end)
    if at < high:
        pieces.append(decoder.decode([int(one) for one in tokens[at:high]], skip_special_tokens=False))
    return "".join(pieces)


def hot_reads_module():
    return sys.modules.get("vllm.taaf_qsa_hisparse")


def hot_reads_summary(row, bin_tokens, align, top, computed, limit=-1, budget=HOT_READS_PICK_TOKENS, above=0,
                      quantile=HOT_READS_PICK_QUANTILE):
    import numpy as np
    row = np.asarray(row, dtype=np.int64)
    bins = np.flatnonzero(row > 0)
    starts = bins * bin_tokens
    keep = starts + bin_tokens > above
    if limit >= 0:
        keep &= starts < limit
    bins, starts = bins[keep], starts[keep]
    end_cap = computed if limit < 0 else min(computed, limit)
    result = {"source": "reads", "rows": int(row[bins].sum()), "bins": int(bins.size), "bin_tokens": int(bin_tokens),
              "computed": int(computed), "align": int(align), "below": int(limit), "above": int(above),
              "listed": int(bins.size), "picked": 0, "pick_floor": 0, "blocks": [], "spans": []}
    if bins.size == 0:
        return result
    per = {}
    for at, start in zip(bins.tolist(), starts.tolist()):
        index = start // align
        per[index] = per.get(index, 0) + int(row[at])
    result["blocks"] = sorted(({"start": int(index * align), "end": int((index + 1) * align), "rows": count}
                               for index, count in per.items()), key=lambda one: (-one["rows"], one["start"]))[:top]
    counts = row[bins]
    floor = int(np.ceil(np.quantile(counts, quantile)))
    ranked = np.argsort(-counts, kind="stable")[:max(int(budget) // int(bin_tokens), 1)]
    picked = np.sort(bins[ranked[counts[ranked] >= floor]])
    result["picked"], result["pick_floor"] = int(picked.size), floor
    cuts = np.flatnonzero(np.diff(picked) > 1) + 1
    for lo, hi in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [picked.size]])):
        first, last = int(picked[lo]), int(picked[hi - 1])
        result["spans"].append({"start": max(first * bin_tokens, int(above)),
                                "end": min((last + 1) * bin_tokens, end_cap),
                                "rows": int(row[first:last + 1].sum())})
    result["spans"] = [one for one in result["spans"] if one["end"] > one["start"]]
    result["spans"].sort(key=lambda one: (-one["rows"], one["start"]))
    result["spans"] = result["spans"][:top]
    return result


def trim_hole_far_settings():
    count = _number(TRIM_HOLE_FAR_CHECKS_ENV, 0, int)
    gap = _number(TRIM_HOLE_FAR_GAP_ENV, TRIM_HOLE_FAR_GAP, int)
    every = _number(TRIM_HOLE_FAR_EVERY_ENV, TRIM_HOLE_FAR_EVERY, int)
    if count < 0 or gap <= 0 or every <= 0:
        raise ValueError(f"{TRIM_HOLE_FAR_CHECKS_ENV}={count}, {TRIM_HOLE_FAR_GAP_ENV}={gap} and "
                         f"{TRIM_HOLE_FAR_EVERY_ENV}={every} must be a count of zero or more and two positive numbers")
    return count, gap, every


def rope_gap_head(head, images):
    moved = int(head)
    for offset, length, *_ in sorted(images):
        if offset < moved < offset + length:
            moved = offset + length
    return moved


def trim_hole_overlap(side, reference):
    _, their_ranks = reference if reference else ([], [])
    _, ranks = side if side else ([], [])
    if not ranks or not their_ranks:
        return None
    return len(set(trim_hole_top(dict(ranks[0]))) & set(trim_hole_top(dict(their_ranks[0]))))


def rope_gap_marked(scheduler_output):
    marked = {}
    for new in scheduler_output.scheduled_new_reqs:
        extra = getattr(new.sampling_params, "extra_args", None) or {}
        mark = extra.get(ROPE_GAP_MARK)
        if mark:
            marked[new.req_id] = (int(mark[0]), int(mark[1]))
    return marked


def rope_gap_shift(runner, req_id, low, gap):
    shift = trim_hole_shift(runner, req_id, low, None, gap, None)
    tally = _TRIM_HOLE["tally"]
    tally["rope_gap_shifted" if shift is not None else "rope_gap_unshifted"] = tally.get(
        "rope_gap_shifted" if shift is not None else "rope_gap_unshifted", 0) + 1
    if tally.get("rope_gap_shifted", 0) == 1 and shift is not None:
        trim_hole_say(f"worker {req_id}: rope gap of {gap} from index {low} applied; tally {tally}")
    return shift
