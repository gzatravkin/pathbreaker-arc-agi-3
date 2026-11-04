import contextlib
import dataclasses
import os
import sys
import time

import torch
import triton
import triton.language as tl

if __import__("os").environ.get("TAAF_VLLM_MARKS", "").strip() == "1":
    _marked = __import__("importlib").import_module("vllm.taaf_marks").marked
else:
    def _marked(site):
        return lambda fn: fn

TALLY_S = 15.0
CHECKS = 3
SHARED_ENV = "TAAF_VLLM_MIXED_PREP_SHARED"
SHORT_ENV = "TAAF_VLLM_GDN_SHORT_PREFILL"
NOSYNC_ENV = "TAAF_VLLM_PREP_NOSYNC"
ONCE_ENV = "TAAF_VLLM_GDN_PREP_ONCE"
FI_ENV = "TAAF_VLLM_GDN_MIXED_FI"
FI_FP16_ENV = "TAAF_VLLM_GDN_MIXED_FI_FP16"
FI_FP16_FLASHINFER = "FLASHINFER_GDN_FP16_ACCUM_MMA"
FI_CHECK_REL = 0.03
FI_CHECK_LEAD = (37, 120)
FI_CHECK_SEQS = ((700, 213, 64), (512, 300, 82))
FI_CHECK_PAD = 3
FI_CHECK_TOKENS = 1100
FI_CHECK_SEED = 7
LEAN_ENV = "TAAF_VLLM_GDN_MIXED_LEAN"
LEAN_CHECKED_ENV = "TAAF_VLLM_GDN_MIXED_LEAN_CHECKED"
LEAN_CHECKS = 3
STATE_ENV = "TAAF_VLLM_GDN_STATE_INPLACE"
HALVES_ENV = "TAAF_VLLM_GDN_HALVES"
STATE_CHECKED_ENV = "TAAF_VLLM_GDN_STATE_INPLACE_CHECKED"
STATE_CHECKS = 3
STATE_STAGES = [2, 3, 4]
STATE_GRAPH = os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"
SHORT_BLOCK_V = 16
SHORT_WARPS = 1
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
_SAID = set()
_STATE = {"at": 0.0, "checked": 0, "aside": ""}
_TALLY = {"built": 0, "captured": 0, "stock": 0}
_WITNESS = {"gdn": [0, 0], "ple_conv": [0, 0]}
_REASON = {"gdn": "", "ple_conv": "", "last": ""}
_ASIDE = {}
_FI = {"state": "", "detail": ""}
_SHARED = {}
_SHARED_TALLY = {"built": 0, "reused": 0}
_LEAN = {"checked": 0, "aside": "", "tally": {"lean": 0, "checked": 0, "old": 0}}
_INPLACE = {"checked": 0, "aside": "", "tally": {"inplace": 0, "checked": 0, "old": 0}}
_HALVES = {"streams": {}, "tally": {"split": 0}}
_COMMON = {}
_ONCE = {"pass": None, "index": {}, "rows": {}, "memo": {}, "aside": "", "checked": 0, "commit_checked": 0, "tally": {"steps": 0, "index_built": 0,
                                                                       "index_reused": 0, "tail_fills": 0,
                                                                       "commits": 0}}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def _mf():
    found = sys.modules.get("vllm.taaf_mixed_full") or sys.modules.get("taaf_mixed_full")
    if found is None:
        from vllm import taaf_mixed_full as found
    return found


def _accept():
    return sys.modules.get("vllm.taaf_gdn_accept") or sys.modules.get("taaf_gdn_accept")


def on():
    return _mf().mode() == "1"


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def shared():
    return str(os.environ.get(SHARED_ENV) or "").strip().lower() in ("1", "on", "true")


def nosync():
    return str(os.environ.get(NOSYNC_ENV) or "").strip().lower() in ("1", "on", "true")


def once():
    return (not _ONCE["aside"] and shared()
            and str(os.environ.get(ONCE_ENV) or "").strip().lower() in ("1", "on", "true"))


def _switch(name):
    said = str(os.environ.get(name) or "").strip()
    if said not in ("", "0", "1"):
        raise ValueError(f"{name}={said!r} is not '', '0' or '1'")
    return said == "1"


def lean():
    return _switch(LEAN_ENV) and not _LEAN["aside"]


def lean_ready():
    return _LEAN["checked"] >= LEAN_CHECKS or _switch(LEAN_CHECKED_ENV)


def halves():
    return _switch(HALVES_ENV)


def halves_stream(device):
    if device.type != "cuda":
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    held = _HALVES["streams"].get(index)
    if held is None:
        held = torch.cuda.Stream(device=index)
        _HALVES["streams"][index] = held
    return held


def state_inplace():
    return _switch(STATE_ENV) and not _INPLACE["aside"]


def state_ready():
    return _INPLACE["checked"] >= STATE_CHECKS or _switch(STATE_CHECKED_ENV)


def _bits(tensor):
    return tensor.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()])


def _same(left, right):
    return tuple(left.shape) == tuple(right.shape) and torch.equal(_bits(left.contiguous()), _bits(right.contiguous()))


def _once_aside(why):
    _ONCE["aside"] = why
    selfcheck_fail(ONCE_ENV, why)
    print(f"GDN_PREP_ONCE stands aside for good: {why}; every GDN group builds its own prep as before", flush=True)


def _once_step(m):
    if _ONCE["pass"] is not m.query_start_loc_cpu:
        _ONCE["pass"] = m.query_start_loc_cpu
        _ONCE["index"] = {}
        _ONCE["rows"] = {}
        _ONCE["memo"] = {}
        _ONCE["tally"]["steps"] += 1
    return _ONCE


def _once_index(builder, m):
    if builder.vllm_config.cache_config.mamba_cache_mode in ("all", "none"):
        return None
    spec = builder.kv_cache_spec
    seq_lens = m.seq_lens
    key = (id(seq_lens), int(spec.block_size), int(spec.num_speculative_blocks), str(seq_lens.device))
    step = _once_step(m)
    held = step["index"].get(key)
    if held is not None and held[0] is seq_lens:
        step["tally"]["index_reused"] += 1
        return held[1]
    start = (seq_lens - 1) // spec.block_size
    start.clamp_(min=0)
    offsets = torch.arange(1 + spec.num_speculative_blocks, device=seq_lens.device, dtype=torch.int32)
    index = (start.unsqueeze(1) + offsets).to(torch.int64)
    step["index"][key] = (seq_lens, index)
    step["tally"]["index_built"] += 1
    return index


def _null_tail(st, name, buffer, full, n, null_block):
    marks = st.__dict__.setdefault("taaf_once_dirty", {})
    held = marks.get(name)
    dirty = held[1] if held is not None and held[0] == null_block else full
    if dirty > n:
        buffer[n:].fill_(null_block)
        _ONCE["tally"]["tail_fills"] += 1
        dirty = n if buffer.shape[0] >= dirty else dirty
    marks[name] = (null_block, max(dirty, n))


def _per_group_states(builder, st, m, n, rows, null_block, block_table_of):
    table = block_table_of(m.block_table_tensor, m.seq_lens, builder.kv_cache_spec,
                           builder.vllm_config.cache_config.mamba_cache_mode)
    plain = st.plain_state[:rows]
    spec_state = st.spec_state[:rows]
    width = min(st.slots, table.shape[1])
    plain[:n].copy_(table[:n, 0])
    spec_state[:n, :width].copy_(table[:n, :width])
    plain[n:].fill_(null_block)
    spec_state[n:].fill_(null_block)
    st.__dict__.pop("taaf_once_dirty", None)
    return plain, spec_state


def _once_check(builder, st, m, n, rows, null_block, block_table_of, plain, spec_state):
    if _ONCE["checked"] >= CHECKS or _capturing():
        return True
    table = block_table_of(m.block_table_tensor, m.seq_lens, builder.kv_cache_spec,
                           builder.vllm_config.cache_config.mamba_cache_mode)
    width = min(st.slots, table.shape[1])
    same = (torch.equal(spec_state[:n, :width], table[:n, :width]) and torch.equal(plain[:n], table[:n, 0])
            and bool((plain[n:] == null_block).all()) and bool((spec_state[n:] == null_block).all()))
    if not same:
        _once_aside(f"check {_ONCE['checked'] + 1}: the once-built state indices of a {n}-row mixed step differ "
                    "from the per-group block table")
        return False
    _ONCE["checked"] += 1
    print(f"GDN_PREP_ONCE check {_ONCE['checked']} of {CHECKS}: {n} rows, {width} slots, state indices and null "
          "tails equal the per-group block table", flush=True)
    return True


def _once_states(builder, st, m, n, rows, null_block, block_table_of):
    index = _once_index(builder, m)
    plain = st.plain_state[:rows]
    spec_state = st.spec_state[:rows]
    width = st.slots
    table = m.block_table_tensor
    if (index is None or index.shape[1] != width or n > table.shape[0] or n > index.shape[0]
            or table.dtype != spec_state.dtype or table.device != spec_state.device):
        return _per_group_states(builder, st, m, n, rows, null_block, block_table_of)
    torch.gather(table[:n], 1, index[:n], out=spec_state[:n])
    plain[:n].copy_(spec_state[:n, 0])
    say("once-live", f"GDN_PREP_ONCE live: a mixed step's block-table index is built once for every GDN group "
                     f"({n} rows, {width} slots); each group gathers its state rows straight into its buffer and "
                     "refills its null tail only when rows past it were written; the accept slots and computed "
                     "counts are made once a step")
    _null_tail(st, "plain", plain, st.plain_state.shape[0], n, null_block)
    _null_tail(st, "spec", spec_state, st.spec_state.shape[0], n, null_block)
    if not _once_check(builder, st, m, n, rows, null_block, block_table_of, plain, spec_state):
        return _per_group_states(builder, st, m, n, rows, null_block, block_table_of)
    return plain, spec_state


def _once_rows(m, common, lead_rows):
    step = _once_step(m)
    key = (id(common.arange_rows), lead_rows)
    held = step["rows"].get(key)
    if held is None:
        held = step["rows"][key] = common.arange_rows[:lead_rows]
    return held


def _once_commit(builder, m, commit_for, *args):
    memo = _once_step(m)["memo"]
    state = getattr(m, "__dict__", None)
    token = (id(m.seq_lens), id(m.query_start_loc))
    restore = None
    original = getattr(m, "compute_num_computed_tokens", None)
    if state is not None and original is not None:
        had = "compute_num_computed_tokens" in state
        held = memo.get("computed")
        if held is not None and held[0] == token and _ONCE["commit_checked"] < CHECKS and not _capturing():
            if not torch.equal(held[1], original()):
                _once_aside("a GDN group's computed-token counts differ from the step's shared ones")
                held = None
            else:
                _ONCE["commit_checked"] += 1
        if held is not None and held[0] == token:
            value = held[1]
            state["compute_num_computed_tokens"] = lambda: value
        else:
            def record():
                value = original()
                memo["computed"] = (token, value)
                return value

            state["compute_num_computed_tokens"] = record
        restore = (had, original)
    previous = getattr(builder, "taaf_prep_memo", None)
    builder.taaf_prep_memo = memo
    try:
        return commit_for(builder, m, *args)
    finally:
        builder.taaf_prep_memo = previous
        _ONCE["tally"]["commits"] += 1
        if restore is not None:
            if restore[0]:
                state["compute_num_computed_tokens"] = restore[1]
            else:
                state.pop("compute_num_computed_tokens", None)


def _flag_initial(has_initial, computed, n):
    if nosync():
        torch.gt(computed[:n], 0, out=has_initial[:n])
    else:
        has_initial[:n].copy_(computed[:n] > 0)


def _walk(buffers, counts, seqs, rows, count):
    arange = buffers.arange_seqs[:seqs + 1]
    if nosync():
        walk = buffers.walk[:seqs + 1]
        torch.add(counts[0].reshape(1), arange, out=walk)
        valid = buffers.valid[:seqs + 1]
        torch.lt(walk, count, out=valid)
        return walk.clamp_(max=rows - 1), valid
    walk = counts[0].reshape(1).long() + arange.long()
    valid = walk < count
    return walk.clamp_(max=rows - 1), valid


def _walk_flags(buffers, valid, has_initial, walk, out):
    if nosync():
        picked = buffers.picked[:walk.shape[0]]
        torch.gather(has_initial, 0, walk, out=picked)
        torch.logical_and(valid, picked, out=out)
    else:
        out.copy_(valid & has_initial.gather(0, walk))


def _walk_state(buffers, valid, plain, walk, null_block, out):
    if nosync():
        gathered = buffers.gathered[:walk.shape[0]]
        torch.gather(plain, 0, walk, out=gathered)
        if buffers.null_value != null_block:
            buffers.null.fill_(null_block)
            buffers.null_value = null_block
        torch.where(valid, gathered, buffers.null, out=out)
    else:
        out.copy_(torch.where(valid, plain.gather(0, walk), null_block))


def short_limit():
    said = str(os.environ.get(SHORT_ENV) or "").strip().lower()
    if said in ("", "0", "off", "false"):
        return 0
    value = int(said)
    if value < 0:
        raise RuntimeError(f"{SHORT_ENV}={said!r} is not a token count")
    return short_q8_refused(value, said)


def short_applies(lens, lead_rows):
    limit = short_limit()
    rest = [int(x) for x in lens[lead_rows:]]
    return limit > 0 and bool(rest) and max(rest) <= limit


def tally():
    now = time.monotonic()
    if now - _STATE["at"] < TALLY_S:
        return
    _STATE["at"] = now
    extra = f" shared_prep {dict(_SHARED_TALLY)}" if shared() else ""
    if once():
        extra += f" prep_once {dict(_ONCE['tally'])}"
    print(f"GDN_MIXED_FULL tally {dict(_TALLY)} aside {dict(_ASIDE)}{extra}", flush=True)


def aside(why, detail=""):
    _TALLY["stock"] += 1
    _ASIDE[why] = _ASIDE.get(why, 0) + 1
    say(f"aside-{why}", f"GDN_MIXED_FULL stands aside for a step: {why}{(' (' + detail + ')') if detail else ''}; "
                        "that step keeps the stock host-built GDN metadata")


def support(default, converted):
    if not on():
        return default
    say("support", "GDN_MIXED_FULL: the GDN metadata builder declares mixed-batch full-graph support while "
                   "TAAF_VLLM_MIXED_FULL=1; a step outside every bucket builds the stock metadata and says so")
    return converted


def short_conv_support(default, converted):
    if not on():
        return default
    say("short-conv", "GDN_MIXED_FULL: the PLE short-conv builder declares mixed-batch full-graph support while "
                      "TAAF_VLLM_MIXED_FULL=1; its layer runs a bucketed step in one conv launch whose split "
                      "comes from the device")
    return converted


def mask_fill(tensor, keep):
    return tensor.masked_fill_(~keep.view(-1, *([1] * (tensor.dim() - 1))), 0)


class Buffers:
    def __init__(self, plan, slots, device):
        self.plan = plan
        top = plan.bucket_for(plan.largest)
        self.rows = top.reqs
        self.seqs = top.seqs
        self.tokens = top.tokens
        self.slots = slots
        i32 = dict(dtype=torch.int32, device=device)
        self.qsl = torch.zeros(self.rows + 1, **i32)
        self.lead_qsl = torch.zeros(self.rows + 1, **i32)
        self.rest_qsl = torch.zeros(self.rows + 1, **i32)
        self.short_qsl = torch.zeros(self.rows + 1, **i32) if short_limit() > 0 else None
        self.cu = torch.zeros(self.seqs + 2, **i32)
        self.chunk_indices = torch.zeros(top.chunk_rows(), 2, **i32)
        self.chunk_offsets = torch.zeros(self.seqs + 2, **i32)
        self.conv_seq = torch.zeros(top.tokens // 8 + top.reqs, **i32)
        self.conv_local = torch.zeros(top.tokens // 8 + top.reqs, **i32)
        self.spec_state = torch.zeros(self.rows, slots, **i32)
        self.plain_state = torch.zeros(self.rows, **i32)
        self.accepted = torch.ones(self.rows, **i32)
        self.has_initial = torch.zeros(self.rows, dtype=torch.bool, device=device)
        self.lead_rows = torch.zeros(self.rows, dtype=torch.bool, device=device)
        self.lead_tokens = torch.zeros(self.tokens, dtype=torch.bool, device=device)
        self.prefill_state = torch.zeros(self.seqs + 1, **i32)
        self.prefill_initial = torch.zeros(self.seqs + 1, dtype=torch.bool, device=device)
        self.live = torch.zeros(1, **i32)
        self.counts = torch.zeros(5, **i32)
        self.arange_rows = torch.arange(self.rows, **i32)
        self.arange_tokens = torch.arange(self.tokens, **i32)
        self.arange_seqs = torch.arange(self.seqs + 2, **i32)
        if nosync():
            self.walk = torch.zeros(self.seqs + 2, dtype=torch.int64, device=device)
            self.valid = torch.zeros(self.seqs + 2, dtype=torch.bool, device=device)
            self.picked = torch.zeros(self.seqs + 2, dtype=torch.bool, device=device)
            self.gathered = torch.zeros(self.seqs + 2, **i32)
            self.null = torch.zeros((), **i32)
            self.null_value = 0


@dataclasses.dataclass
class Mixed:
    bucket: object
    kind: str
    lead_rows: int
    lead_tokens: int
    tokens: int
    lead_row_mask: torch.Tensor
    lead_token_mask: torch.Tensor
    lead_cu: torch.Tensor
    short_qsl: object = None


def _state(builder, thresholds):
    held = builder.__dict__.get("taaf_mixed_state")
    if held is None:
        mf = _mf()
        plans = {kind: mf.plan_from_env(qlen) for kind, qlen in thresholds.items()}
        slots = getattr(builder, "spec_state_slots", builder.num_spec + 1)
        held = builder.taaf_mixed_state = Buffers(plans["spec" if "spec" in plans else "plain"], slots,
                                                  builder.device)
        held.plans = plans
        say("armed", f"GDN_MIXED_FULL armed: {len(held.plan.sizes)} token buckets "
                     f"{held.plan.sizes[0]}..{held.plan.sizes[-1]}, up to {held.rows} rows and {held.seqs} prefill "
                     "sequences per step; chunk indices, causal-conv program lists and the decode/prefill split "
                     "are built on the device into persistent buffers")
    return held


def _host_lead(builder, m, draft_rows_cpu, rows):
    if not builder.use_spec_decode:
        return "plain", 0
    if draft_rows_cpu is None:
        return "scattered", 0
    accept = _accept()
    mask = draft_rows_cpu[:rows] >= 0
    if getattr(builder, "gdn_accept", False) and accept is not None:
        wide = int(m.query_start_loc_cpu.shape[0]) - 1
        if mask.shape[0] < wide:
            mask = torch.cat([mask, torch.zeros(wide - mask.shape[0], dtype=torch.bool, device=mask.device)])
        mask = accept.decode_rows(mask, m, builder.num_spec)[:rows]
    count = int(mask.sum())
    if not bool(mask[:count].all()):
        return "scattered", count
    return "spec", count


def decide(builder, m, draft_rows_cpu, force=None, allow_decode_only=False):
    if _STATE["aside"]:
        return None, _STATE["aside"], None
    if not _mf().graph_step():
        return None, "not_a_full_graph_step", None
    thresholds = {"plain": 1}
    if builder.use_spec_decode:
        thresholds["spec"] = builder.num_spec + 1
    st = _state(builder, thresholds)
    cpu = m.query_start_loc_cpu
    count = int(m.num_reqs)
    lens = (cpu[1:count + 1] - cpu[:count]).tolist()
    while lens and lens[-1] == 0:
        lens.pop()
    if not lens:
        return None, "empty", None
    kind, lead = force if force else _host_lead(builder, m, draft_rows_cpu, len(lens))
    if kind == "scattered":
        return None, "decode_rows_not_leading", None
    plan = st.plans.get(kind, st.plans["plain"])
    plan.qlen = builder.num_spec + 1 if kind == "spec" else 1
    bucket, why = plan.fit(lens, int(m.num_actual_tokens))
    if bucket is None and why == "decode_only" and allow_decode_only:
        bucket = plan.bucket_for(max(sum(lens), int(m.num_actual_tokens)))
        why = "" if bucket is not None else "tokens_over_largest"
    if bucket is None:
        return None, why, None
    expected = plan.split(lens)
    if kind == "spec" and expected != lead and not force:
        return None, "decode_rows_not_equal_length", None
    if kind == "plain" and any(length != 1 for length in lens[:expected]):
        return None, "plain_decode_not_single_token", None
    return (bucket, kind, expected), "", (st, lens)


def claims(builder, m, num_accepted_tokens, draft_rows_cpu):
    if not on():
        return False
    found, why, _ = decide(builder, m, draft_rows_cpu)
    return found is not None


@_marked("gdn_prepare")
def _prepare(builder, st, m, bucket, kind, lens, lead_rows, block_table_of, null_block, device_lists=True):
    mf = _mf()
    device = st.qsl.device
    rows, seqs, tokens = bucket.reqs, bucket.seqs, bucket.tokens
    count = len(lens)
    n = min(int(m.num_reqs), rows)
    qsl = st.qsl[:rows + 1]
    qsl[:n + 1].copy_(m.query_start_loc[:n + 1])
    qsl[n + 1:].copy_(qsl[n:n + 1].expand(rows - n))
    st.live.fill_(count)
    threshold = builder.num_spec + 1 if kind == "spec" else 1
    counts = st.counts
    counts.copy_(mf.split_counts(qsl, st.live, threshold))
    lead_tokens = counts[1]
    lead_qsl = st.lead_qsl[:rows + 1]
    torch.minimum(qsl, lead_tokens, out=lead_qsl)
    rest_qsl = st.rest_qsl[:rows + 1]
    torch.maximum(qsl, lead_tokens, out=rest_qsl)
    index = (counts[0].reshape(1).long() + st.arange_seqs[:seqs + 2].long()).clamp_(max=rows)
    cu = st.cu[:seqs + 2]
    cu.copy_(qsl.gather(0, index))
    short = short_applies(lens, lead_rows)
    if st.short_qsl is not None:
        short_qsl = st.short_qsl[:rows + 1]
        if short:
            cu.fill_(int(sum(lens[:lead_rows])))
            short_qsl.copy_(rest_qsl)
        else:
            short_qsl.fill_(0)
    programs = tokens // 8 + rows
    if device_lists:
        mf.chunk_meta(cu, bucket.chunk_rows(), seqs, out=st.chunk_indices[:bucket.chunk_rows()],
                      offsets_out=st.chunk_offsets[:seqs + 2])
        mf.conv_meta(rest_qsl if kind == "spec" else qsl, programs, out_seq=st.conv_seq[:programs],
                     out_local=st.conv_local[:programs])
    table = block_table_of(m.block_table_tensor, m.seq_lens, builder.kv_cache_spec,
                           builder.vllm_config.cache_config.mamba_cache_mode)
    plain = st.plain_state[:rows]
    plain[:n].copy_(table[:n, 0])
    plain[n:].fill_(null_block)
    spec_state = st.spec_state[:rows]
    width = min(st.slots, table.shape[1])
    spec_state[:n, :width].copy_(table[:n, :width])
    spec_state[n:].fill_(null_block)
    has_initial = st.has_initial[:rows]
    _flag_initial(has_initial, m.compute_num_computed_tokens(), n)
    has_initial[n:].fill_(False)
    lead_row_mask = st.lead_rows[:rows]
    torch.lt(st.arange_rows[:rows], counts[0], out=lead_row_mask)
    lead_token_mask = st.lead_tokens[:tokens]
    torch.lt(st.arange_tokens[:tokens], lead_tokens, out=lead_token_mask)
    walk, valid = _walk(st, counts, seqs, rows, count)
    prefill_state = st.prefill_state[:seqs + 1]
    _walk_state(st, valid, plain, walk, null_block, prefill_state)
    prefill_initial = st.prefill_initial[:seqs + 1]
    _walk_flags(st, valid, has_initial, walk, prefill_initial)
    if short:
        prefill_state.fill_(null_block)
        prefill_initial.fill_(False)
    accepted = None
    if kind == "spec":
        accepted = st.accepted[:rows]
        accepted.fill_(1)
    return {"n": n, "counts": counts, "qsl": qsl, "lead_qsl": lead_qsl, "rest_qsl": rest_qsl, "cu": cu,
            "plain": plain, "spec_state": spec_state, "has_initial": has_initial, "prefill_state": prefill_state,
            "prefill_initial": prefill_initial, "accepted": accepted, "lead_row_mask": lead_row_mask,
            "lead_token_mask": lead_token_mask, "programs": programs, "short": short}


def _ident(tensor):
    if tensor is None:
        return None
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, str(tensor.device))


def _common_key(builder, st, device_lists):
    return (type(builder), bool(device_lists), str(st.qsl.device), st.rows, st.seqs, st.tokens, st.slots)


def _common_buffers(ckey, st):
    held = _COMMON.get(ckey)
    if held is None:
        held = _COMMON[ckey] = Buffers(st.plan, st.slots, st.qsl.device)
        held.plans = st.plans
    return held


def _shared_key(builder, st, m, bucket, kind, lens, lead_rows, num_accepted_tokens):
    return (bucket, kind, tuple(lens), lead_rows, builder.num_spec, st.slots, int(m.num_reqs),
            int(m.num_actual_tokens), _ident(m.query_start_loc), _ident(m.seq_lens), _ident(num_accepted_tokens))


def _shared_base(builder, common, m, bucket, kind, lens, device_lists, num_accepted_tokens, lead_rows):
    mf = _mf()
    rows, seqs, tokens = bucket.reqs, bucket.seqs, bucket.tokens
    count = len(lens)
    n = min(int(m.num_reqs), rows)
    qsl = common.qsl[:rows + 1]
    qsl[:n + 1].copy_(m.query_start_loc[:n + 1])
    qsl[n + 1:].copy_(qsl[n:n + 1].expand(rows - n))
    common.live.fill_(count)
    threshold = builder.num_spec + 1 if kind == "spec" else 1
    counts = common.counts
    counts.copy_(mf.split_counts(qsl, common.live, threshold))
    lead_tokens = counts[1]
    lead_qsl = common.lead_qsl[:rows + 1]
    torch.minimum(qsl, lead_tokens, out=lead_qsl)
    rest_qsl = common.rest_qsl[:rows + 1]
    torch.maximum(qsl, lead_tokens, out=rest_qsl)
    index = (counts[0].reshape(1).long() + common.arange_seqs[:seqs + 2].long()).clamp_(max=rows)
    cu = common.cu[:seqs + 2]
    cu.copy_(qsl.gather(0, index))
    short = short_applies(lens, lead_rows)
    if common.short_qsl is not None:
        short_qsl = common.short_qsl[:rows + 1]
        if short:
            cu.fill_(int(sum(lens[:lead_rows])))
            short_qsl.copy_(rest_qsl)
        else:
            short_qsl.fill_(0)
    programs = tokens // 8 + rows
    if device_lists:
        mf.chunk_meta(cu, bucket.chunk_rows(), seqs, out=common.chunk_indices[:bucket.chunk_rows()],
                      offsets_out=common.chunk_offsets[:seqs + 2])
        mf.conv_meta(rest_qsl if kind == "spec" else qsl, programs, out_seq=common.conv_seq[:programs],
                     out_local=common.conv_local[:programs])
    has_initial = common.has_initial[:rows]
    _flag_initial(has_initial, m.compute_num_computed_tokens(), n)
    has_initial[n:].fill_(False)
    lead_row_mask = common.lead_rows[:rows]
    torch.lt(common.arange_rows[:rows], counts[0], out=lead_row_mask)
    lead_token_mask = common.lead_tokens[:tokens]
    torch.lt(common.arange_tokens[:tokens], lead_tokens, out=lead_token_mask)
    walk, valid = _walk(common, counts, seqs, rows, count)
    prefill_initial = common.prefill_initial[:seqs + 1]
    _walk_flags(common, valid, has_initial, walk, prefill_initial)
    if short:
        prefill_initial.fill_(False)
    accepted = None
    if kind == "spec":
        accepted = common.accepted[:rows]
        accepted.fill_(1)
        if num_accepted_tokens is not None:
            accepted[:min(n, num_accepted_tokens.shape[0])].copy_(num_accepted_tokens[:n].to(torch.int32))
    made = {"n": n, "counts": counts, "qsl": qsl, "lead_qsl": lead_qsl, "rest_qsl": rest_qsl, "cu": cu,
            "has_initial": has_initial, "prefill_initial": prefill_initial, "accepted": accepted,
            "lead_row_mask": lead_row_mask, "lead_token_mask": lead_token_mask, "programs": programs,
            "short": short}
    return {"made": made, "walk": walk, "valid": valid}


def _prepare_shared(builder, st, m, bucket, kind, lens, lead_rows, block_table_of, null_block, device_lists,
                    num_accepted_tokens):
    ckey = _common_key(builder, st, device_lists)
    common = _common_buffers(ckey, st)
    key = _shared_key(builder, st, m, bucket, kind, lens, lead_rows, num_accepted_tokens)
    entry = _SHARED.get(ckey)
    reuse = entry is not None and entry["pass"] is m.query_start_loc_cpu and entry["key"] == key
    if not reuse:
        held = _shared_base(builder, common, m, bucket, kind, lens, device_lists, num_accepted_tokens, lead_rows)
        _SHARED[ckey] = {"pass": m.query_start_loc_cpu, "key": key, "held": held}
        _SHARED_TALLY["built"] += 1
    else:
        held = entry["held"]
        _SHARED_TALLY["reused"] += 1
        say("shared-live", "GDN_MIXED_FULL shared prep live: a mixed step's split, query starts, chunk and conv "
                           "lists, masks and accepted counts are built once per attention pass into one set of "
                           "persistent buffers; every other GDN group gathers only its own state indices")
    rows, seqs = bucket.reqs, bucket.seqs
    n = held["made"]["n"]
    if once():
        plain, spec_state = _once_states(builder, st, m, n, rows, null_block, block_table_of)
    else:
        table = block_table_of(m.block_table_tensor, m.seq_lens, builder.kv_cache_spec,
                               builder.vllm_config.cache_config.mamba_cache_mode)
        plain = st.plain_state[:rows]
        plain[:n].copy_(table[:n, 0])
        plain[n:].fill_(null_block)
        spec_state = st.spec_state[:rows]
        width = min(st.slots, table.shape[1])
        spec_state[:n, :width].copy_(table[:n, :width])
        spec_state[n:].fill_(null_block)
    prefill_state = st.prefill_state[:seqs + 1]
    _walk_state(common, held["valid"], plain, held["walk"], null_block, prefill_state)
    if held["made"]["short"]:
        prefill_state.fill_(null_block)
    made = dict(held["made"])
    made.update({"plain": plain, "spec_state": spec_state, "prefill_state": prefill_state, "fresh": not reuse})
    return made, common


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def fi_wanted():
    return str(os.environ.get(FI_ENV) or "").strip().lower() in ("1", "on", "true")


def fi_fp16():
    return str(os.environ.get(FI_FP16_ENV) or "").strip().lower() in ("1", "on", "true")


def _fi_layout(lead, seqs, pad):
    cuts = [lead]
    for length in seqs:
        cuts.append(cuts[-1] + length)
    cuts.extend([cuts[-1]] * pad)
    return cuts


def _fi_inputs(layer, like, device, gen):
    k_heads = layer.num_k_heads // layer.tp_size
    v_heads = layer.num_v_heads // layer.tp_size
    tokens = FI_CHECK_TOKENS
    unit = torch.nn.functional.normalize
    q = unit(torch.randn(1, tokens, k_heads, layer.head_k_dim, device=device, generator=gen), dim=-1)
    k = unit(torch.randn(1, tokens, k_heads, layer.head_k_dim, device=device, generator=gen), dim=-1)
    v = torch.randn(1, tokens, v_heads, layer.head_v_dim, device=device, generator=gen) * 0.5
    g = torch.nn.functional.logsigmoid(torch.randn(1, tokens, v_heads, device=device, generator=gen) + 3.0)
    beta = torch.sigmoid(torch.randn(1, tokens, v_heads, device=device, generator=gen))
    seqs = len(FI_CHECK_SEQS[0]) + FI_CHECK_PAD
    initial = torch.randn(seqs, *like.shape[1:], device=device, generator=gen) * 0.1
    return (q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), g.float(), beta.to(torch.bfloat16),
            initial.to(like.dtype))


def _fi_gap(want, got):
    scale = float(want.float().abs().max())
    return float((want.float() - got.float()).abs().max()) / max(scale, 1e-30)


def _fi_compare(want_o, want_s, got_o, got_s, cuts, real):
    rows = slice(cuts[0], cuts[real])
    gap_o = _fi_gap(want_o[:, rows], got_o[:, rows])
    gap_s = _fi_gap(want_s[:real], got_s[:real])
    return gap_o, gap_s


def _fi_check(layer, like):
    native = getattr(layer.chunk_gated_delta_rule, "forward_native", None)
    fast = getattr(layer.chunk_gated_delta_rule, "forward_cuda", None)
    backend = getattr(layer.chunk_gated_delta_rule, "gdn_prefill_backend", "")
    if native is None or fast is None or backend != "flashinfer":
        return False, f"the serve's GDN prefill backend is {backend or 'unknown'}, not FlashInfer"
    if fi_fp16():
        os.environ[FI_FP16_FLASHINFER] = "1"
    device = like.device
    gen = torch.Generator(device=device).manual_seed(FI_CHECK_SEED)
    q, k, v, g, beta, initial = _fi_inputs(layer, like, device, gen)
    real = len(FI_CHECK_SEQS[0])
    cuts = _fi_layout(FI_CHECK_LEAD[0], FI_CHECK_SEQS[0], FI_CHECK_PAD)
    cu = torch.tensor(cuts, dtype=torch.int32, device=device)
    args = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "output_final_state": True, "cu_seqlens": cu,
            "use_qk_l2norm_in_kernel": False}
    want_o, want_s = native(initial_state=initial.clone(), **args)
    got_o, got_s = fast(initial_state=initial.clone(), **args)
    gap_o, gap_s = _fi_compare(want_o, want_s, got_o, got_s, cuts, real)
    if not (gap_o <= FI_CHECK_REL and gap_s <= FI_CHECK_REL):
        return False, f"eager FlashInfer against FLA: output gap {gap_o:.3g}, state gap {gap_s:.3g}"
    held = {"initial": initial.clone()}
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            held["o"], held["s"] = fast(initial_state=held["initial"], **args)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    q2, k2, v2, g2, beta2, initial2 = _fi_inputs(layer, like, device, gen)
    for target, source in ((q, q2), (k, k2), (v, v2), (g, g2), (beta, beta2), (held["initial"], initial2)):
        target.copy_(source)
    cuts2 = _fi_layout(FI_CHECK_LEAD[1], FI_CHECK_SEQS[1], FI_CHECK_PAD)
    cu.copy_(torch.tensor(cuts2, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize(device)
    want_o2, want_s2 = native(initial_state=initial2.clone(), **args)
    gap_o2, gap_s2 = _fi_compare(want_o2, want_s2, held["o"], held["s"], cuts2, real)
    del graph
    if not (gap_o2 <= FI_CHECK_REL and gap_s2 <= FI_CHECK_REL):
        return False, (f"a captured FlashInfer call replayed on a new layout differs from FLA: output gap "
                       f"{gap_o2:.3g}, state gap {gap_s2:.3g}")
    return True, (f"eager output gap {gap_o:.3g}, state gap {gap_s:.3g}; graph replay on a new layout output gap "
                  f"{gap_o2:.3g}, state gap {gap_s2:.3g}; limit {FI_CHECK_REL}")


def chunk_for(layer, like):
    native = getattr(layer.chunk_gated_delta_rule, "forward_native", layer.chunk_gated_delta_rule)
    if not fi_wanted() or _FI["state"] == "aside":
        return native
    if _FI["state"] == "on":
        return layer.chunk_gated_delta_rule.forward_cuda
    if _capturing():
        say("fi-capture", "GDN_MIXED_FI: first met inside a graph capture before its check; this graph keeps FLA")
        return native
    try:
        ok, detail = _fi_check(layer, like)
    except Exception as failure:
        ok, detail = False, f"{type(failure).__name__}: {str(failure)[:240]}"
    _FI["state"], _FI["detail"] = ("on" if ok else "aside"), detail
    if not ok and not detail.startswith("the serve's GDN prefill backend is"):
        selfcheck_fail(FI_ENV, detail)
    if ok:
        print(f"GDN_MIXED_FI live: mixed-step GDN prefill runs FlashInfer's chunked kernel (the serve's own "
              f"prefill backend) in place of FLA; check: {detail}"
              + ("; FLASHINFER_GDN_FP16_ACCUM_MMA=1" if fi_fp16() else ""), flush=True)
        return layer.chunk_gated_delta_rule.forward_cuda
    print(f"GDN_MIXED_FI stood aside: {detail}; mixed-step GDN prefill keeps FLA", flush=True)
    return native


def _check(made, lead_rows, lead_tokens, lens):
    if _STATE["checked"] >= CHECKS or _capturing():
        return True
    got = made["counts"].cpu().tolist()
    want = [lead_rows, lead_tokens, len(lens) - lead_rows, sum(lens) - lead_tokens, sum(lens)]
    if got != want:
        _STATE["aside"] = f"device split {got} differs from the host split {want}"
        selfcheck_fail("TAAF_VLLM_MIXED_FULL", _STATE["aside"])
        print(f"GDN_MIXED_FULL stands aside for good: the device-built split {got} differs from the host "
              f"split {want}", flush=True)
        return False
    _STATE["checked"] += 1
    print(f"GDN_MIXED_FULL check {_STATE['checked']} of {CHECKS}: {len(lens)} rows, {lead_rows} leading decode "
          f"rows, {lead_tokens} leading tokens, {len(lens) - lead_rows} prefill sequences; device split equals "
          "the host split", flush=True)
    return True


def _stage(builder, m, num_accepted_tokens, draft_rows_cpu, block_table_of, null_block, force=None,
           allow_decode_only=False, device_lists=True):
    found, why, held = decide(builder, m, draft_rows_cpu, force, allow_decode_only)
    _REASON["last"] = why
    if found is None:
        if why not in ("decode_only", "empty", "not_a_full_graph_step"):
            aside(why)
        return None
    bucket, kind, lead_rows = found
    st, lens = held
    if shared():
        made, st = _prepare_shared(builder, st, m, bucket, kind, lens, lead_rows, block_table_of, null_block,
                                   device_lists, num_accepted_tokens)
    else:
        made = _prepare(builder, st, m, bucket, kind, lens, lead_rows, block_table_of, null_block, device_lists)
    lead_tokens = sum(lens[:lead_rows])
    if not _check(made, lead_rows, lead_tokens, lens):
        return None
    n = made["n"]
    if kind == "spec" and num_accepted_tokens is not None and "fresh" not in made:
        made["accepted"][:min(n, num_accepted_tokens.shape[0])].copy_(
            num_accepted_tokens[:n].to(torch.int32))
    return st, made, bucket, kind, lead_rows, lead_tokens, lens


def build(builder, m, num_accepted_tokens, draft_rows_cpu, metadata_cls, block_table_of, null_block,
          commit_for=None, force=None):
    if not on():
        return None
    staged = _stage(builder, m, num_accepted_tokens, draft_rows_cpu, block_table_of, null_block, force)
    if staged is None:
        return None
    st, made, bucket, kind, lead_rows, lead_tokens, lens = staged
    mixed = Mixed(bucket, kind, lead_rows, lead_tokens, bucket.tokens, made["lead_row_mask"],
                  made["lead_token_mask"], made["lead_qsl"], st.short_qsl[:bucket.reqs + 1]
                  if st.short_qsl is not None else None)
    programs = made["programs"]
    nums = {8: {"nums": None, "tot": programs, "mlist": None, "mlist_len": programs, "offsetlist": None,
                "batch_ptr": st.conv_seq[:programs], "token_chunk_offset_ptr": st.conv_local[:programs]},
            "taaf_mixed": mixed}
    prefill_rows = len(lens) - lead_rows
    prefill_tokens = sum(lens) - lead_tokens
    spec = kind == "spec"
    metadata = metadata_cls(
        num_prefills=prefill_rows,
        num_prefill_tokens=prefill_tokens,
        num_decodes=0 if spec else lead_rows,
        num_decode_tokens=0 if spec else lead_tokens,
        num_spec_decodes=lead_rows if spec else 0,
        num_spec_decode_tokens=lead_tokens if spec else 0,
        num_actual_tokens=m.num_actual_tokens,
        has_initial_state=made["has_initial"],
        spec_query_start_loc=made["lead_qsl"] if spec else None,
        non_spec_query_start_loc=made["rest_qsl"] if spec else made["qsl"],
        spec_state_indices_tensor=made["spec_state"] if spec else None,
        non_spec_state_indices_tensor=made["plain"],
        spec_sequence_masks=made["lead_row_mask"] if spec else None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=made["accepted"] if spec else None,
        chunk_indices=st.chunk_indices[:bucket.chunk_rows()],
        chunk_offsets=st.chunk_offsets[:bucket.seqs + 2],
        prefill_query_start_loc=made["cu"],
        prefill_state_indices=made["prefill_state"],
        prefill_has_initial_state=made["prefill_initial"],
        nums_dict=nums,
        batch_ptr=st.conv_seq[:programs],
        token_chunk_offset_ptr=st.conv_local[:programs],
    )
    if spec and commit_for is not None and getattr(builder, "gdn_accept", False) \
            and hasattr(metadata, "accept_commit"):
        if once():
            metadata.accept_commit = _once_commit(builder, m, commit_for, made["spec_state"], made["lead_qsl"],
                                                  lead_rows, _once_rows(m, st, lead_rows))
        else:
            metadata.accept_commit = commit_for(builder, m, made["spec_state"], made["lead_qsl"], lead_rows,
                                                st.arange_rows[:lead_rows])
    _TALLY["built"] += 1
    say("live", f"GDN_MIXED_FULL live: the first mixed step ({kind} lead, {lead_rows} decode rows, "
                f"{prefill_rows} prefill sequences, {sum(lens)} tokens) takes the {bucket.tokens}-token bucket "
                f"({bucket.reqs} rows, {bucket.chunk_rows()} chunk rows, {programs} conv programs) with "
                "device-built metadata")
    tally()
    return metadata


def witness():
    return {name: tuple(pair) for name, pair in _WITNESS.items()}


def reasons():
    return {name: _REASON[name] for name in ("gdn", "ple_conv")}


def capture(builder, m, metadata_cls, block_table_of, null_block, commit_for=None):
    if not on():
        return None
    _WITNESS["gdn"][0] += 1
    _REASON["gdn"] = "no_bucket_fits_the_dummy"
    lens = (m.query_start_loc_cpu[1:int(m.num_reqs) + 1] - m.query_start_loc_cpu[:int(m.num_reqs)]).tolist()
    kind = "spec" if builder.use_spec_decode else "plain"
    thresholds = {"plain": 1}
    if builder.use_spec_decode:
        thresholds["spec"] = builder.num_spec + 1
    st = _state(builder, thresholds)
    plan = st.plans[kind]
    plan.qlen = builder.num_spec + 1 if kind == "spec" else 1
    while lens and lens[-1] == 0:
        lens.pop()
    if not lens or plan.fit(lens, int(m.num_actual_tokens))[0] is None:
        return None
    lead = plan.split(lens)
    _REASON["last"] = ""
    made = build(builder, m, None, None, metadata_cls, block_table_of, null_block, commit_for,
                 force=(kind, lead))
    _REASON["gdn"] = _REASON["last"] or "builder_returned_nothing"
    if made is not None:
        _REASON["gdn"] = ""
        _TALLY["captured"] += 1
        _WITNESS["gdn"][1] += 1
        say("capture", f"GDN_MIXED_FULL capture: a {m.num_actual_tokens}-token mixed shape was built with "
                       "persistent device metadata")
    return made


TAIL_ZERO_ENV = "TAAF_VLLM_GDN_TAIL_ZERO"
TAIL_ROWS = 16
TAIL_WIDTH = 1024


def tail_zero():
    return str(os.environ.get(TAIL_ZERO_ENV) or "1").strip().lower() in ("1", "on", "true")


def tail_owned(prefix):
    if not tail_zero():
        return False
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    held = get_forward_context().attn_metadata
    return isinstance(held, dict) and owned(held.get(prefix)) is not None


@triton.jit(do_not_specialize=["rows"])
def _tail_zero_kernel(out, real, rows, WIDTH: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
    first = tl.program_id(0) * BLOCK_R
    start = tl.load(real).to(tl.int64)
    if first + BLOCK_R <= start:
        return
    at = first + tl.arange(0, BLOCK_R).to(tl.int64)
    live = (at >= start) & (at < rows)
    zero = tl.zeros([BLOCK_R, BLOCK_W], dtype=out.dtype.element_ty)
    for col in range(0, WIDTH, BLOCK_W):
        cols = col + tl.arange(0, BLOCK_W)
        tl.store(out + at[:, None] * WIDTH + cols[None, :], zero, mask=live[:, None] & (cols[None, :] < WIDTH))


def zero_tail(core_attn_out, md):
    rows = core_attn_out.shape[0]
    flat = core_attn_out.view(rows, -1)
    width = flat.shape[1]
    _tail_zero_kernel[(triton.cdiv(rows, TAIL_ROWS),)](flat, md.non_spec_query_start_loc[-1:], rows, WIDTH=width,
                                                      BLOCK_R=TAIL_ROWS, BLOCK_W=min(TAIL_WIDTH,
                                                                                     triton.next_power_of_2(width)))
    say("tail_zero", f"GDN_TAIL_ZERO live: a mixed step's GDN output is allocated empty and only the rows past the "
                     f"step's real tokens (read on the device) are zeroed, {rows} rows of {width} at most")


def owned(metadata):
    nums = getattr(metadata, "nums_dict", None)
    if isinstance(nums, dict):
        return nums.get("taaf_mixed")
    return None


@triton.jit(do_not_specialize=["num_rows"])
def _short_prefill_kernel(
    A_log, a, b, dt_bias, q, k, v, o, state, cu_seqlens, state_indices, has_initial,
    scale, softplus_beta, softplus_threshold, num_rows,
    stride_q, stride_k, stride_v, stride_qc, stride_a, stride_b, stride_o, stride_state_block,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    length = eos - bos
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_q = q + bos * stride_q + (i_h * K + o_k) * stride_qc
    p_k = k + bos * stride_k + (i_h * K + o_k) * stride_qc
    p_v = v + bos * stride_v + (i_hv * V + o_v) * stride_qc
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    block = tl.load(state_indices + i_n).to(tl.int64)
    if block <= 0:
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    started = tl.load(has_initial + i_n)
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if started:
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    for _ in range(0, length):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        x_g = tl.load(p_a).to(tl.float32) + bias_t
        softplus_x = tl.where(
            softplus_beta * x_g <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)),
            x_g,
        )
        b_g = -tl.exp(a_log) * softplus_x
        b_beta = tl.sigmoid(b_b)
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o
    tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)


def short_prefill(layer, q, k, v, a, b, out, query_start_loc, state_indices, has_initial):
    state = layer.kv_cache[1]
    rows = state_indices.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    k_heads = layer.num_k_heads // layer.tp_size
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError("GDN_SHORT_PREFILL needs contiguous state heads")
    if q.stride(1) != k.stride(1) or q.stride(1) != v.stride(1):
        raise ValueError("GDN_SHORT_PREFILL needs one channel stride for q, k and v")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("GDN_SHORT_PREFILL needs contiguous gate heads")
    if out.stride()[1:] != (v_dim, 1):
        raise ValueError("GDN_SHORT_PREFILL needs a contiguous output row per token")
    block_v = min(triton.next_power_of_2(v_dim), SHORT_BLOCK_V)
    grid = (triton.cdiv(v_dim, block_v), rows * v_heads)
    _short_prefill_kernel[grid](
        layer.A_log, a, b, layer.dt_bias, q, k, v, out, state, query_start_loc, state_indices, has_initial,
        k_dim**-0.5, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD, rows,
        q.stride(0), k.stride(0), v.stride(0), q.stride(1), a.stride(0), b.stride(0), out.stride(0),
        state.stride(0),
        H=k_heads, HV=v_heads, K=k_dim, V=v_dim, BK=triton.next_power_of_2(k_dim), BV=block_v,
        num_warps=SHORT_WARPS,
    )
    return out


def forward(layer, mixed_qkv, b, a, core_attn_out, md):
    mixed = owned(md)
    module = sys.modules[type(layer).__module__]
    tokens = md.num_actual_tokens
    kv = layer.kv_cache
    conv_state = kv[0] if module.is_conv_state_dim_first() else kv[0].transpose(-1, -2)
    ssm_state = kv[1] if getattr(layer, "taaf_q8_full", None) is None else layer.taaf_q8_full
    mixed_qkv = mixed_qkv[:tokens]
    b = b[:tokens]
    a = a[:tokens]
    weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    spec = mixed.kind == "spec"
    conv = module.causal_conv1d_fn(
        mixed_qkv.transpose(0, 1), weights, layer.conv1d.bias, activation=layer.activation,
        conv_states=conv_state, has_initial_state=md.has_initial_state,
        cache_indices=md.non_spec_state_indices_tensor, query_start_loc=md.non_spec_query_start_loc,
        metadata=md).transpose(0, 1)
    if spec:
        module.causal_conv1d_update(
            mixed_qkv, conv_state, weights, layer.conv1d.bias, layer.activation,
            conv_state_indices=md.spec_state_indices_tensor[:, 0], num_accepted_tokens=md.num_accepted_tokens,
            query_start_loc=md.spec_query_start_loc, max_query_len=layer.num_spec + 1, validate_data=False,
            out=conv)
        lead_cu = md.spec_query_start_loc
    else:
        lead_cu = mixed.lead_cu
    query, key, value, gate, beta = module.fused_post_conv_prep(
        conv_output=conv, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads // layer.tp_size, head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
        apply_l2norm=True, output_g_exp=False)
    query, key, value, gate, beta = (x.unsqueeze(0) for x in (query, key, value, gate, beta))
    records = getattr(layer, "gdn_accept_records", None)
    if spec and records is not None and _lean_applies(core_attn_out):
        _lean_tail(layer, conv, a, b, (query, key, value, gate, beta), core_attn_out, md, mixed, tokens, ssm_state,
                   records)
        _short_tail(layer, conv, a, b, core_attn_out, md, mixed, tokens)
        if tail_zero():
            zero_tail(core_attn_out, md)
        return
    lead_q, lead_k, lead_v = layer.rearrange_mixed_qkv(conv)
    if spec and records is not None:
        rows = min(md.spec_state_indices_tensor.shape[0], records.rows)
        lead_out = _accept().verify_layer(
            layer, lead_q[0], lead_k[0], lead_v[0], a, b, md.spec_query_start_loc[:rows + 1],
            md.spec_state_indices_tensor[:rows, 0])
    elif spec:
        lead_out, _ = module.fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=lead_q, k=lead_k, v=lead_v,
            initial_state=ssm_state, inplace_final_state=True, cu_seqlens=md.spec_query_start_loc,
            ssm_state_indices=md.spec_state_indices_tensor, num_accepted_tokens=md.num_accepted_tokens,
            use_qk_l2norm_in_kernel=True)
    else:
        lead_out, _ = module.fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=lead_q, k=lead_k, v=lead_v,
            initial_state=ssm_state, inplace_final_state=True, cu_seqlens=lead_cu,
            ssm_state_indices=md.non_spec_state_indices_tensor, use_qk_l2norm_in_kernel=True)
    initial = _state_gather(ssm_state, md.prefill_state_indices)
    mask_fill(initial, md.prefill_has_initial_state)
    chunk = chunk_for(layer, ssm_state)
    rest, last = chunk(
        q=query, k=key, v=value, g=gate, beta=beta, initial_state=initial, output_final_state=True,
        cu_seqlens=md.prefill_query_start_loc, chunk_indices=md.chunk_indices, chunk_offsets=md.chunk_offsets,
        use_qk_l2norm_in_kernel=False)
    _state_scatter(ssm_state, md.prefill_state_indices, last)
    merged = torch.where(mixed.lead_token_mask.view(1, -1, 1, 1), lead_out, rest)
    core_attn_out[:tokens] = merged.squeeze(0)
    _short_tail(layer, conv, a, b, core_attn_out, md, mixed, tokens)
    if tail_zero():
        zero_tail(core_attn_out, md)


def _short_tail(layer, conv, a, b, core_attn_out, md, mixed, tokens):
    if mixed.short_qsl is not None:
        key_dim = layer.key_dim // layer.tp_size
        short_prefill(layer, conv[:, :key_dim], conv[:, key_dim:2 * key_dim], conv[:, 2 * key_dim:], a, b,
                      core_attn_out[:tokens], mixed.short_qsl, md.non_spec_state_indices_tensor,
                      md.has_initial_state)
        say("short", f"GDN_SHORT_PREFILL live: prefill sequences up to {short_limit()} tokens run through the "
                     "recurrent kernel inside the mixed graph")


@triton.autotune(configs=[triton.Config({"BV": BV}, num_warps=warps, num_stages=stages) for warps in [2, 4]
                          for stages in STATE_STAGES for BV in [32, 64]],
                 key=["H", "K", "V", "BT"], restore_value=["state"],
                 use_cuda_graph=STATE_GRAPH)
@triton.jit(do_not_specialize=["T", "slot_stride"])
def _state_h_kernel(k, v, w, v_new, g, h, state, state_idx, has_init, slot_stride, cu_seqlens, chunk_offsets, T,
                    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                    BV: tl.constexpr, Q8: tl.constexpr = False):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    slot = state + tl.load(state_idx + i_n).to(tl.int64) * slot_stride + (i_h * V * K).to(tl.int64)
    q8_base = (state + tl.load(state_idx + i_n).to(tl.int64) * slot_stride).to(tl.pointer_type(tl.int8))
    q8_rows = i_v * BV + tl.arange(0, BV)
    q8_cols = tl.arange(0, 64)
    q8_codes = q8_base + i_h * V * K + q8_rows[:, None] * K + q8_cols[None, :]
    q8_mask = (q8_rows[:, None] < V) & (q8_cols[None, :] < 64)
    q8_scales = (q8_base + H * V * K).to(tl.pointer_type(tl.float32)) + i_h * V + q8_rows
    if Q8:
        if tl.load(has_init + i_n):
            q8_row = tl.load(q8_scales, mask=q8_rows < V, other=0.0)[:, None]
            b_h1 += tl.load(q8_codes, mask=q8_mask, other=0).to(tl.float32) * q8_row
            if K > 64:
                b_h2 += tl.load(q8_codes + 64, mask=q8_mask, other=0).to(tl.float32) * q8_row
            if K > 128:
                b_h3 += tl.load(q8_codes + 128, mask=q8_mask, other=0).to(tl.float32) * q8_row
            if K > 192:
                b_h4 += tl.load(q8_codes + 192, mask=q8_mask, other=0).to(tl.float32) * q8_row
    elif tl.load(has_init + i_n):
        p_h0_1 = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64),
                                     (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64),
                                     (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64),
                                     (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        p_v = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))
    if Q8:
        q8_peak = tl.max(tl.abs(b_h1), 1)
        if K > 64:
            q8_peak = tl.maximum(q8_peak, tl.max(tl.abs(b_h2), 1))
        if K > 128:
            q8_peak = tl.maximum(q8_peak, tl.max(tl.abs(b_h3), 1))
        if K > 192:
            q8_peak = tl.maximum(q8_peak, tl.max(tl.abs(b_h4), 1))
        q8_scale = tl.where(q8_peak > 0, q8_peak / 127.0, 1.0)
        tl.store(q8_scales, q8_scale, mask=q8_rows < V)
        q8_coded = b_h1 / q8_scale[:, None]
        tl.store(q8_codes, tl.where(q8_coded >= 0, q8_coded + 0.5, q8_coded - 0.5).to(tl.int8), mask=q8_mask)
        if K > 64:
            q8_coded = b_h2 / q8_scale[:, None]
            tl.store(q8_codes + 64, tl.where(q8_coded >= 0, q8_coded + 0.5, q8_coded - 0.5).to(tl.int8),
                     mask=q8_mask)
        if K > 128:
            q8_coded = b_h3 / q8_scale[:, None]
            tl.store(q8_codes + 128, tl.where(q8_coded >= 0, q8_coded + 0.5, q8_coded - 0.5).to(tl.int8),
                     mask=q8_mask)
        if K > 192:
            q8_coded = b_h4 / q8_scale[:, None]
            tl.store(q8_codes + 192, tl.where(q8_coded >= 0, q8_coded + 0.5, q8_coded - 0.5).to(tl.int8),
                     mask=q8_mask)
        return
    p_ht = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_ht = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_ht = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
        tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_ht = tl.make_block_ptr(slot, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
        tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _q8():
    return os.environ.get("TAAF_VLLM_GDN_STATE_Q8", "").strip() == "1"


def _fused():
    return sys.modules.get("vllm.taaf_gdn_fused") or sys.modules.get("taaf_gdn_fused")


def _q8_close(left, right):
    left_codes, left_scales = _fused().q8_views(left)
    right_codes, right_scales = _fused().q8_views(right)
    if tuple(left_codes.shape) != tuple(right_codes.shape):
        return False
    finite = torch.isfinite(left_scales)
    if not torch.equal(finite, torch.isfinite(right_scales)):
        return False
    gap = (left_codes.int() - right_codes.int()).abs().masked_fill(~finite[..., None], 0)
    codes = int(gap.max()) <= 1 if gap.numel() else True
    return codes and torch.allclose(left_scales[finite], right_scales[finite], rtol=1e-4, atol=0)


def _q8_gap(left, right):
    left_codes, left_scales = _fused().q8_views(left)
    right_codes, right_scales = _fused().q8_views(right)
    codes = (left_codes.int() - right_codes.int()).abs()
    scale = (left_scales - right_scales).abs() / right_scales.abs().clamp_min(1e-30)
    values = (_fused().q8_unpack(left_codes, left_scales) - _fused().q8_unpack(right_codes, right_scales)).abs()
    peak = _fused().q8_unpack(right_codes, right_scales).abs().amax().clamp_min(1e-30)
    per_seq = [int(codes[one].max()) for one in range(codes.shape[0])]
    unfinite = [int((~torch.isfinite(side)).sum()) for side in (left_scales, right_scales)]
    return (f"codes max {int(codes.max())}, {int((codes > 1).sum())} of {codes.numel()} off by more than one, "
            f"per sequence {per_seq}; scales max rel {float(scale.max()):.3g}; values max {float(values.max() / peak):.3g} "
            f"of the peak; non-finite row scales {unfinite}")


def _state_gather(state, indices):
    if not _q8():
        return state[indices]
    return _fused().q8_gather(state, indices)


def _state_scatter(state, indices, values):
    if not _q8():
        state[indices] = values.to(state.dtype)
        return
    _fused().q8_scatter(state, indices, values)


def _fla(name):
    import importlib
    return importlib.import_module(f"vllm.third_party.flash_linear_attention.ops.{name}")


def _state_h(k, w, u, g, state, indices, has_initial, cu_seqlens, chunk_indices, chunk_offsets, size):
    index = _fla("index")
    batch, tokens, heads_k, dim_k = k.shape
    heads, dim_v = u.shape[-2], u.shape[-1]
    if chunk_indices is None:
        chunk_indices = index.prepare_chunk_indices(cu_seqlens, size)
    if chunk_offsets is None:
        chunk_offsets = index.prepare_chunk_offsets(cu_seqlens, size)
    h = k.new_empty(batch, len(chunk_indices), heads, dim_v, dim_k)
    v_new = torch.empty_like(u)
    seqs = len(cu_seqlens) - 1

    def grid(meta):
        return (triton.cdiv(dim_v, meta["BV"]), seqs * heads)

    _state_h_kernel[grid](k=k, v=u, w=w, v_new=v_new, g=g, h=h, state=state, state_idx=indices,
                          has_init=has_initial, slot_stride=state.stride(0), cu_seqlens=cu_seqlens,
                          chunk_offsets=chunk_offsets, T=tokens, H=heads, Hg=heads_k, K=dim_k, V=dim_v, BT=size,
                          Q8=_q8())
    return h, v_new


def state_chunk(q, k, v, g, beta, state, indices, has_initial, cu_seqlens, chunk_indices, chunk_offsets,
                core_attn_out):
    size = _fla("utils").FLA_CHUNK_SIZE
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    g = _fla("cumsum").chunk_local_cumsum(g, chunk_size=size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = _fla("chunk_scaled_dot_kkt").chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
                                                              chunk_indices=chunk_indices,
                                                              output_dtype=torch.float32)
    A = _fla("solve_tril").solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                                      output_dtype=k.dtype)
    w, u = _fla("wy_fast").recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu_seqlens,
                                            chunk_indices=chunk_indices)
    h, v_new = _state_h(k, w, u, g, state, indices, has_initial, cu_seqlens, chunk_indices, chunk_offsets, size)
    _fla("chunk_o").chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=k.shape[-1] ** -0.5, cu_seqlens=cu_seqlens,
                                chunk_indices=chunk_indices, core_attn_out=core_attn_out)


def _state_layout(ssm_state):
    if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
        return "FLA_USE_FAST_OPS swaps the chunk kernels' exp for a fast one this copy does not carry"
    if ssm_state.dim() != 4 or ssm_state.shape[-1] > 256:
        return f"the SSM state is {tuple(ssm_state.shape)}, not slots x heads x V x K with K up to 256"
    inner = (ssm_state.shape[2] * ssm_state.shape[3], ssm_state.shape[3], 1)
    if tuple(ssm_state.stride()[1:]) != inner:
        return f"the SSM state's per-slot strides {tuple(ssm_state.stride()[1:])} are not {inner}"
    return ""


def _state_live(ssm_state):
    if not state_inplace():
        return False
    why = _state_layout(ssm_state)
    if why:
        say("state_layout", f"GDN_STATE_INPLACE keeps the gather and scatter: {why}")
        return False
    if not state_ready():
        if _capturing():
            _INPLACE["tally"]["old"] += 1
            say("state_capture", "GDN_STATE_INPLACE: a mixed graph captured before the in-place state was checked "
                                 "keeps the gather and scatter inside that graph")
        return False
    return True


def _state_check(prepared, ssm_state, md, core_attn_out, before, last, lead, real):
    if not state_inplace() or state_ready() or _state_layout(ssm_state):
        return
    query, key, value, gate, beta = prepared
    indices = md.prefill_state_indices
    count = int(indices.shape[0])
    scratch = torch.empty_strided((count, *ssm_state.shape[1:]), ssm_state.stride(), dtype=ssm_state.dtype,
                                  device=ssm_state.device)
    scratch.copy_(before)
    out = core_attn_out.clone()
    out[lead:real] = float("nan")
    state_chunk(query, key, value, gate, beta, scratch, torch.arange(count, dtype=indices.dtype, device=indices.device),
                md.prefill_has_initial_state, md.prefill_query_start_loc, md.chunk_indices, md.chunk_offsets,
                out.view(-1))
    seqs = int(md.num_prefills)
    if _q8():
        state = _q8_close(scratch[:seqs], ssm_state[indices][:seqs])
        if not state:
            print(f"GDN_STATE_Q8 in-place check gap: {_q8_gap(scratch[:seqs], ssm_state[indices][:seqs])}", flush=True)
    else:
        state = _same(scratch[:seqs], last[:seqs].to(ssm_state.dtype))
    output = _same(out[lead:real], core_attn_out[lead:real])
    if not (state and output):
        failed = [name for name, ok in (("final state", state), ("prefill output", output)) if not ok]
        _INPLACE["aside"] = f"{', '.join(failed)} differ at {real - lead} prefill tokens in {seqs} sequences"
        selfcheck_fail(STATE_ENV, _INPLACE["aside"])
        print(f"GDN_STATE_INPLACE stands aside: the in-place chunk's {_INPLACE['aside']} from the gather, chunk and "
              "scatter; every later mixed step keeps them", flush=True)
        return
    _INPLACE["checked"] += 1
    _INPLACE["tally"]["checked"] += 1
    print(f"GDN_STATE_INPLACE check {_INPLACE['checked']} of {STATE_CHECKS}: {real - lead} prefill tokens in {seqs} "
          "sequences; the chunk that reads each initial state from its slot and writes the final state back in the "
          "state dtype gives the gather, chunk and scatter's state and output bit for bit", flush=True)


def _lean_applies(core_attn_out):
    if not lean():
        return False
    if not core_attn_out.is_contiguous():
        say("lean_layout", "GDN_MIXED_LEAN keeps the old merge: the attention output buffer is not contiguous")
        return False
    if _capturing() and not lean_ready():
        _LEAN["tally"]["old"] += 1
        say("lean_capture", "GDN_MIXED_LEAN: a mixed graph captured before the lean tail was checked keeps the old "
                            "merge inside that graph")
        return False
    return True


def _lean_width(layer, mixed, tokens):
    return min(tokens, mixed.bucket.reqs * (layer.num_spec + 1))


def _lead_views(layer, conv, width):
    key_dim = layer.key_dim // layer.tp_size
    value_dim = layer.value_dim // layer.tp_size
    head = conv[:width]
    return head[:, :key_dim], head[:, key_dim:2 * key_dim], head[:, 2 * key_dim:2 * key_dim + value_dim]


def _lean_tail(layer, conv, a, b, prepared, core_attn_out, md, mixed, tokens, ssm_state, records):
    query, key, value, gate, beta = prepared
    rows = min(md.spec_state_indices_tensor.shape[0], records.rows)
    width = _lean_width(layer, mixed, tokens)
    qsl = md.spec_query_start_loc[:rows + 1]
    states = md.spec_state_indices_tensor[:rows, 0]
    chunk = getattr(layer.chunk_gated_delta_rule, "forward_native", layer.chunk_gated_delta_rule)
    if not _capturing() and _LEAN["checked"] < LEAN_CHECKS:
        _lean_check(layer, conv, a, b, prepared, core_attn_out, md, mixed, tokens, ssm_state, width, qsl, states,
                    chunk)
        return
    side = halves_stream(conv.device) if halves() else None
    if side is not None:
        side.wait_stream(torch.cuda.current_stream(conv.device))
    with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
        lead_q, lead_k, lead_v = _lead_views(layer, conv, width)
        _accept().verify_layer(layer, lead_q, lead_k, lead_v, a[:width], b[:width], qsl, states,
                               out=core_attn_out[:width])
    if _state_live(ssm_state):
        state_chunk(query, key, value, gate, beta, ssm_state, md.prefill_state_indices,
                    md.prefill_has_initial_state, md.prefill_query_start_loc, md.chunk_indices, md.chunk_offsets,
                    core_attn_out.view(-1))
        _INPLACE["tally"]["inplace"] += 1
        say("state_live", "GDN_STATE_INPLACE live: the mixed step's prefill chunk reads each sequence's initial state "
                          "from its slot (zero where it has none) and writes the final state back into that slot in "
                          "the state dtype, so the state gather, its zero fill, the float32 copy and the scatter are gone")
    else:
        before = None if _capturing() or state_ready() else ssm_state[md.prefill_state_indices].clone()
        initial = _state_gather(ssm_state, md.prefill_state_indices)
        mask_fill(initial, md.prefill_has_initial_state)
        _, last = chunk(
            q=query, k=key, v=value, g=gate, beta=beta, initial_state=initial, output_final_state=True,
            cu_seqlens=md.prefill_query_start_loc, chunk_indices=md.chunk_indices, chunk_offsets=md.chunk_offsets,
            use_qk_l2norm_in_kernel=False, core_attn_out=core_attn_out.view(-1))
        _state_scatter(ssm_state, md.prefill_state_indices, last)
        if before is not None:
            lead = int(mixed.lead_tokens)
            _state_check(prepared, ssm_state, md, core_attn_out, before, last, lead,
                         lead + int(md.num_prefill_tokens))
    if side is not None:
        torch.cuda.current_stream(conv.device).wait_stream(side)
        _HALVES["tally"]["split"] += 1
        say("halves", "GDN_HALVES live: a mixed step's decode verification runs on a second stream beside the "
                      "prefill chunk of the same layer; the two write disjoint output rows and state slots, and the "
                      "layer waits for both before its output is read")
    _LEAN["tally"]["lean"] += 1
    say("lean_live", f"GDN_MIXED_LEAN live: a mixed step's decode rows are rearranged over the bucket's first {width} "
                     "tokens instead of all of them and verified straight into the attention output, and the prefill "
                     "chunk writes its output in place, so the full-step merge and its copy are gone")


def _lean_check(layer, conv, a, b, prepared, core_attn_out, md, mixed, tokens, ssm_state, width, qsl, states, chunk):
    query, key, value, gate, beta = prepared
    lead_q, lead_k, lead_v = layer.rearrange_mixed_qkv(conv)
    lead_out = _accept().verify_layer(layer, lead_q[0], lead_k[0], lead_v[0], a, b, qsl, states)
    short = _lead_views(layer, conv, width)
    inputs = all(_same(made, full[0, :width].reshape(width, -1)) for made, full in zip(short, (lead_q, lead_k, lead_v)))
    viewed = _accept().verify_layer(layer, *short, a[:width], b[:width], qsl, states)
    before = ssm_state[md.prefill_state_indices].clone() if state_inplace() and not state_ready() else None
    initial = _state_gather(ssm_state, md.prefill_state_indices)
    mask_fill(initial, md.prefill_has_initial_state)
    args = dict(q=query, k=key, v=value, g=gate, beta=beta, initial_state=initial, output_final_state=True,
                cu_seqlens=md.prefill_query_start_loc, chunk_indices=md.chunk_indices,
                chunk_offsets=md.chunk_offsets, use_qk_l2norm_in_kernel=False)
    rest, last = chunk(**args)
    merged = torch.where(mixed.lead_token_mask.view(1, -1, 1, 1), lead_out, rest).squeeze(0)
    _, placed = chunk(**args, core_attn_out=core_attn_out.view(-1))
    lead = int(mixed.lead_tokens)
    real = lead + int(md.num_prefill_tokens)
    core_attn_out[:lead] = lead_out[0, :lead]
    output = _same(core_attn_out[:real], merged[:real]) and _same(viewed[0, :lead], lead_out[0, :lead])
    state = _same(placed, last)
    core_attn_out[:tokens] = merged
    _state_scatter(ssm_state, md.prefill_state_indices, last)
    if not (inputs and output and state):
        failed = [name for name, ok in (("decode-row inputs", inputs), ("output", output), ("final state", state))
                  if not ok]
        _LEAN["aside"] = f"{', '.join(failed)} differ at {real} tokens ({lead} decode)"
        selfcheck_fail(LEAN_ENV, _LEAN["aside"])
        print(f"GDN_MIXED_LEAN stands aside: the lean tail's {_LEAN['aside']} from the old merge; the old merge "
              "was written and every later mixed step keeps it", flush=True)
        return
    if before is not None:
        _state_check(prepared, ssm_state, md, core_attn_out, before, last, lead, real)
    _LEAN["checked"] += 1
    _LEAN["tally"]["checked"] += 1
    print(f"GDN_MIXED_LEAN check {_LEAN['checked']} of {LEAN_CHECKS}: {real} tokens ({lead} decode, width {width}); "
          "decode-row inputs, the in-place prefill output and the final state equal the old merge bit for bit",
          flush=True)



PLE_NULL_STATE_ID = 0
PLE_BLOCK_C = 512


@dataclasses.dataclass
class PleMixed:
    bucket: object
    kind: str
    lead_rows: int
    rows: int
    qsl: torch.Tensor
    state_indices: torch.Tensor
    accepted: torch.Tensor
    has_initial: torch.Tensor
    lead: torch.Tensor


@triton.jit
def _ple_mixed_kernel(x_ptr, state_ptr, w_ptr, residual_ptr, outer_residual_ptr, state_idx_ptr, qsl_ptr,
                      num_acc_ptr, has_init_ptr, lead_ptr, num_reqs, bs_iters, state_idx_stride, state_bs,
                      state_ws, state_cs, C: tl.constexpr, BLOCK_C: tl.constexpr, STATE_LEN: tl.constexpr,
                      DILATION: tl.constexpr, KERNEL_SIZE: tl.constexpr, SPEC_QUERY_LEN: tl.constexpr,
                      LEAD_SPEC: tl.constexpr, NULL_STATE_ID: tl.constexpr):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    lo = tl.full([], 1, tl.int32)
    hi = tl.full([], num_reqs + 1, tl.int32)
    for _ in range(bs_iters):
        mid = (lo + hi) // 2
        qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
        pred = qmid <= t
        lo = tl.where(pred, mid + 1, lo)
        hi = tl.where(pred, hi, mid)
    r = tl.minimum(lo - 1, num_reqs - 1)
    row_start = tl.load(qsl_ptr + r)
    total_real = tl.load(qsl_ptr + num_reqs)
    in_range = t < total_real
    lead = r < tl.load(lead_ptr)
    never = r < 0
    if LEAD_SPEC:
        spec_row = lead
        dec_row = lead & never
    else:
        spec_row = lead & never
        dec_row = lead
    q_start = tl.where(dec_row, t, row_start)
    j = tl.where(dec_row, 0, t - row_start).to(tl.int32)
    num_acc = tl.load(num_acc_ptr + r, mask=spec_row, other=1)
    slot_off = tl.where(spec_row, tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1), 0).to(tl.int32)
    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    read_state = tl.where(spec_row, state_ok, state_ok & has_init)
    out_ok = tl.where(spec_row, in_range, in_range & state_ok)
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(base_state + (h + slot_off) * state_ws + c_offs * state_cs,
                            mask=c_mask & read_state & from_state, other=0.0)
        input_t = (q_start + h - STATE_LEN).to(tl.int64)
        input_tap = tl.load(x_ptr + input_t * C + c_offs, mask=c_mask & out_ok & (~from_state), other=0.0)
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(w_ptr + c_offs * KERNEL_SIZE + k, mask=c_mask, other=0.0).to(tl.float32)
        acc += weight * tap
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    output_t = t.to(tl.int64)
    residual = tl.load(residual_ptr + output_t * C + c_offs, mask=c_mask, other=0.0)
    ple_output = (residual + conv_output).to(residual_ptr.dtype.element_ty)
    outer_residual = tl.load(outer_residual_ptr + output_t * C + c_offs, mask=c_mask, other=0.0)
    ple_output = outer_residual.to(tl.float32) + ple_output.to(tl.float32)
    tl.store(residual_ptr + output_t * C + c_offs, ple_output, mask=c_mask)
    decode_input = tl.load(x_ptr + output_t * C + c_offs, mask=c_mask & state_ok & dec_row, other=0.0)
    for i in tl.static_range(0, STATE_LEN):
        if i < STATE_LEN - 1:
            next_state = tl.load(base_state + (i + 1) * state_ws + c_offs * state_cs,
                                 mask=c_mask & state_ok & has_init & dec_row, other=0.0)
        else:
            next_state = decode_input
        tl.store(base_state + i * state_ws + c_offs * state_cs, next_state, mask=c_mask & state_ok & dec_row)


@triton.jit
def _ple_mixed_writeback_kernel(x_ptr, state_ptr, state_idx_ptr, qsl_ptr, num_acc_ptr, has_init_ptr, lead_ptr,
                                state_idx_stride, state_bs, state_ws, state_cs, C: tl.constexpr,
                                BLOCK_C: tl.constexpr, STATE_LEN: tl.constexpr, SPEC_QUERY_LEN: tl.constexpr,
                                STATE_WIDTH: tl.constexpr, LEAD_SPEC: tl.constexpr,
                                NULL_STATE_ID: tl.constexpr):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    lead = r < tl.load(lead_ptr)
    never = r < 0
    if LEAD_SPEC:
        spec_row = lead
        dec_row = lead & never
    else:
        spec_row = lead & never
        dec_row = lead
    active = state_ok & (qlen > 0) & (~dec_row)
    num_acc = tl.load(num_acc_ptr + r, mask=spec_row, other=1)
    slot_off = tl.where(spec_row, tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1), 0).to(tl.int32)
    shift = tl.where(spec_row, 1, qlen)
    has_init = tl.load(has_init_ptr + r) != 0
    src_ok = tl.where(spec_row, state_ok, state_ok & has_init)
    write_width = tl.where(spec_row, STATE_WIDTH, STATE_LEN)
    base_state = state_ptr + tl.where(state_ok, sid, 0) * state_bs
    for i in tl.static_range(0, STATE_WIDTH):
        m = shift + i
        from_state = m <= STATE_LEN - 1
        do_write = active & tl.where(spec_row, i < STATE_LEN + qlen - 1, i < write_width)
        state_value = tl.load(base_state + (slot_off + m) * state_ws + c_offs * state_cs,
                              mask=c_mask & from_state & src_ok & active, other=0.0)
        input_t = (q_start + m - STATE_LEN).to(tl.int64)
        input_value = tl.load(x_ptr + input_t * C + c_offs, mask=c_mask & (~from_state) & do_write, other=0.0)
        value = tl.where(from_state, state_value, input_value)
        tl.store(base_state + i * state_ws + c_offs * state_cs, value, mask=c_mask & do_write)


def build_ple(builder, m, num_accepted_tokens, draft_rows_cpu, metadata_cls, block_table_of, null_block,
              force=None, allow_decode_only=False):
    if not on():
        return None
    staged = _stage(builder, m, num_accepted_tokens, draft_rows_cpu, block_table_of, null_block, force,
                    allow_decode_only, device_lists=False)
    if staged is None:
        return None
    st, made, bucket, kind, lead_rows, lead_tokens, lens = staged
    spec = kind == "spec"
    mixed = PleMixed(bucket, kind, lead_rows, bucket.reqs, made["qsl"], made["plain"],
                     made["accepted"] if spec else made["plain"], made["has_initial"], st.counts[:1])
    prefill_rows = len(lens) - lead_rows
    metadata = metadata_cls(
        num_prefills=prefill_rows,
        num_prefill_tokens=sum(lens) - lead_tokens,
        num_decodes=0 if spec else lead_rows,
        num_decode_tokens=0 if spec else lead_tokens,
        num_reqs=m.num_reqs,
        num_spec_decodes=lead_rows if spec else 0,
        num_spec_decode_tokens=lead_tokens if spec else 0,
        num_actual_tokens=m.num_actual_tokens,
        spec_query_len=builder.num_spec + 1,
        max_prefill_query_len=bucket.tokens,
        query_start_loc=made["qsl"],
        state_indices_tensor=made["plain"],
        has_initial_states_p=made["has_initial"],
        has_initial_states_d=made["has_initial"],
        non_spec_query_start_loc=made["qsl"],
        spec_query_start_loc=made["lead_qsl"] if spec else None,
        spec_state_indices_tensor=made["plain"] if spec else None,
        spec_sequence_masks=made["lead_row_mask"] if spec else None,
        num_accepted_tokens=made["accepted"] if spec else None,
        seq_lens=m.seq_lens,
        query_start_loc_p=None,
        query_start_loc_d=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
    )
    metadata.taaf_ple_mixed = mixed
    _TALLY["built"] += 1
    say("ple-live", f"GDN_MIXED_FULL PLE short-conv: a mixed step ({kind} lead, {lead_rows} decode rows, "
                    f"{prefill_rows} prefill sequences) runs one conv launch and one write-back launch over the "
                    f"{bucket.tokens}-token bucket, the outer residual added once")
    return metadata


def capture_ple(builder, m, metadata_cls, block_table_of, null_block):
    if not on():
        return None
    _WITNESS["ple_conv"][0] += 1
    _REASON["ple_conv"] = "no_bucket_fits_the_dummy"
    count = int(m.num_reqs)
    lens = (m.query_start_loc_cpu[1:count + 1] - m.query_start_loc_cpu[:count]).tolist()
    kind = "spec" if builder.use_spec_decode else "plain"
    thresholds = {"plain": 1}
    if builder.use_spec_decode:
        thresholds["spec"] = builder.num_spec + 1
    st = _state(builder, thresholds)
    plan = st.plans[kind]
    plan.qlen = builder.num_spec + 1 if kind == "spec" else 1
    while lens and lens[-1] == 0:
        lens.pop()
    if not lens or plan.fit(lens, int(m.num_actual_tokens))[0] is None:
        return None
    _REASON["last"] = ""
    made = build_ple(builder, m, None, None, metadata_cls, block_table_of, null_block,
                     force=(kind, plan.split(lens)))
    _REASON["ple_conv"] = _REASON["last"] or "builder_returned_nothing"
    if made is not None:
        _REASON["ple_conv"] = ""
        _WITNESS["ple_conv"][1] += 1
    return made


def ple_owned(metadata):
    return getattr(metadata, "taaf_ple_mixed", None)


def ple_forward(layer, inputs, residual, outer_residual, metadata, conv_state, conv_weights):
    mixed = metadata.taaf_ple_mixed
    tokens = metadata.num_actual_tokens
    inputs = inputs[:tokens]
    residual = residual[:tokens]
    outer_residual = outer_residual[:tokens]
    channels = inputs.shape[1]
    kernel = conv_weights.shape[1]
    dilation = layer.short_conv_dilation
    spec = mixed.kind == "spec"
    spec_query_len = metadata.spec_query_len if spec else 1
    state_len = (kernel - 1) * dilation
    state_width = state_len + spec_query_len - 1
    state_bs, state_cs, state_ws = conv_state.stride()
    iters = max(mixed.rows, 1).bit_length()
    common = dict(C=channels, BLOCK_C=PLE_BLOCK_C, STATE_LEN=state_len, LEAD_SPEC=spec,
                  NULL_STATE_ID=PLE_NULL_STATE_ID)
    blocks = triton.cdiv(channels, PLE_BLOCK_C)
    _ple_mixed_kernel[(tokens, blocks)](
        inputs, conv_state, conv_weights, residual, outer_residual, mixed.state_indices, mixed.qsl,
        mixed.accepted, mixed.has_initial, mixed.lead, mixed.rows, iters, mixed.state_indices.stride(0),
        state_bs, state_ws, state_cs, DILATION=dilation, KERNEL_SIZE=kernel, SPEC_QUERY_LEN=spec_query_len,
        num_warps=8 if spec else 4, **common)
    _ple_mixed_writeback_kernel[(mixed.rows, blocks)](
        inputs, conv_state, mixed.state_indices, mixed.qsl, mixed.accepted, mixed.has_initial, mixed.lead,
        mixed.state_indices.stride(0), state_bs, state_ws, state_cs, SPEC_QUERY_LEN=spec_query_len,
        STATE_WIDTH=state_width, num_warps=4, **common)


def short_q8_refused(value, said):
    if value > 0 and _q8():
        raise RuntimeError(f"{SHORT_ENV}={said!r} reads and writes the state as 16-bit values, which "
                           "TAAF_VLLM_GDN_STATE_Q8=1 stores as int8 codes and row scales; set one of them off")
    return value
