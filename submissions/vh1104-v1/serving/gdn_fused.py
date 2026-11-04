import dataclasses
import functools
import os
import sys
import time
import types

import torch
import triton
import triton.language as tl


_PDL_STATE = {"on": None}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


@triton.jit
def _q8_tile(state, block_off, i_hv, o_v, o_k, mask_h, mask_v, HV: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    base = (state + block_off).to(tl.pointer_type(tl.int8))
    codes = tl.load(base + i_hv * V * K + o_v[:, None] * K + o_k[None, :], mask=mask_h, other=0)
    scale = tl.load((base + HV * V * K).to(tl.pointer_type(tl.float32)) + i_hv * V + o_v, mask=mask_v, other=0.0)
    return codes.to(tl.float32) * scale[:, None]


@triton.jit
def _q8_put(state, block_off, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV: tl.constexpr, V: tl.constexpr,
            K: tl.constexpr):
    base = (state + block_off).to(tl.pointer_type(tl.int8))
    row_max = tl.max(tl.abs(b_h), 1)
    row_scale = tl.where(row_max > 0, row_max / 127.0, 1.0)
    coded = b_h / row_scale[:, None]
    coded = tl.where(coded >= 0, coded + 0.5, coded - 0.5).to(tl.int8)
    tl.store(base + i_hv * V * K + o_v[:, None] * K + o_k[None, :], coded, mask=mask_h)
    tl.store((base + HV * V * K).to(tl.pointer_type(tl.float32)) + i_hv * V + o_v, row_scale, mask=mask_v)
    return coded.to(tl.float32) * row_scale[:, None]


@triton.jit
def _q8_count_bad(state, block_off, i_hv, o_v, mask_v, bad, HV: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
    base = (state + block_off).to(tl.pointer_type(tl.int8))
    row_scale = tl.load((base + HV * V * K).to(tl.pointer_type(tl.float32)) + i_hv * V + o_v, mask=mask_v, other=0.0)
    broken = mask_v & ((row_scale != row_scale) | (tl.abs(row_scale) > 3.4028234e38))
    count = tl.sum(broken.to(tl.int64), 0)
    tl.atomic_add(bad, count, mask=count > 0)


def commit_cap():
    said = os.environ.get(COMMIT_CAP_ENV, "").strip()
    if not said:
        return COMMIT_CAP_STREAM_DEFAULT if os.environ.get(COMMIT_STREAM_ENV, "").strip() == "3" else 0
    try:
        cap = int(said)
    except ValueError:
        cap = -1
    if not 0 <= cap <= COMMIT_CAP_LIMIT:
        raise ValueError(f"{COMMIT_CAP_ENV} must be a whole number of rows from 0 to {COMMIT_CAP_LIMIT}, "
                         f"found {said!r}")
    return cap


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

FUSED_ENV = "TAAF_VLLM_GDN_FUSED"
MODES = {"1": "defer", "defer": "defer", "kernel": "kernel", "tiled": "tiled", "lazy": "lazy"}
DEFERRED = ("defer", "tiled", "lazy")
RECURRENT = ("tiled", "lazy")
LAZY_STEPS_ENV = "TAAF_VLLM_GDN_LAZY_STEPS"
LAZY_STEPS_DEFAULT = 3
RING_LIMIT = 32
TIMERS_MODULE = "vllm.taaf_step_timers"
LAZY_COUNTER = "gdn_lazy"
LAZY_FIELDS = ("pending", "head", "marks", "tail", "block")
LAZY_STATS = ("rows", "ring_writes", "boundary_writes", "tiled_writes", "held")
TILED_SHAPE_ENV = "TAAF_VLLM_GDN_TILED_SHAPE"
AHEAD_ENV = "TAAF_VLLM_GDN_VERIFY_AHEAD"
MAXNREG_ENV = "TAAF_VLLM_GDN_VERIFY_MAXNREG"
SLOT_FILL_ENV = "TAAF_VLLM_GDN_SLOT_FILL"
NOSYNC_ENV = "TAAF_VLLM_PREP_NOSYNC"
PREP_ENV = "TAAF_VLLM_PREP_GDN"
_SHARED_SLOTS = {}


def tiled_shape(default=(32, 4, 2)):
    said = os.environ.get(TILED_SHAPE_ENV, "").strip()
    if not said:
        return default
    try:
        shape = tuple(int(part) for part in said.split(","))
    except ValueError:
        shape = ()
    if len(shape) != 3 or min(shape) <= 0 or shape[0] & (shape[0] - 1) or shape[1] & (shape[1] - 1):
        raise ValueError(f"{TILED_SHAPE_ENV} must be block_v,warps,stages with powers of two for the first two, "
                         f"found {said!r}")
    print(f"GDN_FUSED {TILED_SHAPE_ENV}={shape[0]},{shape[1]},{shape[2]} in place of "
          f"{default[0]},{default[1]},{default[2]}", flush=True)
    return shape


TILED_BLOCK_V, TILED_WARPS, TILED_STAGES = tiled_shape()
FUSED_WARPS = 8
FUSED_STAGES = 1
COMMIT_BLOCK_V = 32
COMMIT_WARPS = 4
COMMIT_STAGES = 2
COMMIT_CAP_ENV = "TAAF_VLLM_GDN_COMMIT_CAP"
COMMIT_CAP_LIMIT = 256
COMMIT_CAP_STREAM_DEFAULT = 2
COMMIT_STREAM_ENV = "TAAF_VLLM_GDN_COMMIT_STREAM"
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
ACTIVATIONS = {"silu": 1, "swish": 1, "sigmoid": 2}
PRODUCT_TYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}
_SAID: set = set()
_PENDING: list = []
_BUILDERS: dict = {}
RESUME_SPAN = 2
_LAZY = {"stood": None, "checked": False, "counts": [], "stats": {}, "quiet": False}


def say(key, line):
    if key in _SAID or _LAZY["quiet"]:
        return
    _SAID.add(key)
    print(line, flush=True)


Q8_ENV = "TAAF_VLLM_GDN_STATE_Q8"


def q8():
    value = os.environ.get(Q8_ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{Q8_ENV}={value!r} is not one of '', '0', '1'")
    return value == "1"


def q8_refuse(path):
    if q8():
        raise RuntimeError(f"{Q8_ENV}=1 stores the GDN state as int8 codes and row scales, and {path} would read "
                           "those bytes as the state dtype")


def q8_views(state):
    blocks, heads, v_dim, k_dim = (state := q8_full(state)).shape
    codes_bytes = heads * v_dim * k_dim
    need = codes_bytes + heads * v_dim * 4
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1) or state.stride(0) * state.element_size() < need:
        raise ValueError(f"{Q8_ENV} needs contiguous state heads and at least {need} bytes a block")
    if (state.stride(0) * state.element_size()) % 4 or (state.data_ptr() + codes_bytes) % 4:
        raise ValueError(f"{Q8_ENV} needs every block's row scales on a 4-byte boundary")
    raw = torch.as_strided(state.view(torch.int8), (blocks, need), (state.stride(0) * state.element_size(), 1),
                           state.storage_offset() * state.element_size())
    codes = raw[:, :codes_bytes].view(blocks, heads, v_dim, k_dim)
    scales = raw[:, codes_bytes:need].view(torch.float32).view(blocks, heads, v_dim)
    return codes, scales


@triton.jit
def _q8_rows_kernel(state, indices, values, block_stride, rows, STORE: tl.constexpr, HV: tl.constexpr,
                    V: tl.constexpr, K: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_rh = tl.program_id(1)
    row = i_rh // HV
    i_hv = i_rh % HV
    block = tl.load(indices + row).to(tl.int64)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    mask_h = mask_v[:, None] & (o_k[None, :] < K)
    p_values = values + row.to(tl.int64) * HV * V * K + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if STORE:
        if block >= 0:
            b_h = tl.load(p_values, mask=mask_h, other=0.0).to(tl.float32)
            _q8_put(state, block * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
    else:
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        if block >= 0:
            b_h = _q8_tile(state, block * block_stride, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
        tl.store(p_values, b_h, mask=mask_h)


def _q8_rows(state, indices, values, store):
    _, heads, v_dim, k_dim = (state := q8_full(state)).shape
    flat = indices.reshape(-1)
    block_v = min(64, triton.next_power_of_2(v_dim))
    _q8_rows_kernel[(triton.cdiv(v_dim, block_v), flat.shape[0] * heads)](
        state, flat, values, state.stride(0), flat.shape[0], STORE=store, HV=heads, V=v_dim, K=k_dim, BV=block_v,
        num_warps=4)


def q8_gather(state, indices):
    if not q8():
        return state[indices]
    _, heads, v_dim, k_dim = (state := q8_full(state)).shape
    out = torch.empty((*indices.shape, heads, v_dim, k_dim), dtype=torch.float32, device=state.device)
    if indices.numel():
        _q8_rows(state, indices, out, False)
    return out


def q8_scatter(state, indices, values):
    if not q8():
        state[indices] = values.to(state.dtype)
        return
    if indices.numel():
        _q8_rows(state, indices, values.float().contiguous(), True)


def q8_inplace(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        state = kwargs.get("initial_state")
        indices = kwargs.get("ssm_state_indices")
        if not q8() or state is None or indices is None or not kwargs.get("inplace_final_state", True):
            return function(*args, **kwargs)
        temp = q8_gather(state, indices.clamp(min=-1))
        places = torch.arange(indices.numel(), device=indices.device, dtype=indices.dtype).view(indices.shape)
        places = torch.where(indices >= 0, places, indices)
        flat = temp.view(-1, *temp.shape[-3:])
        made = function(*args, **{**kwargs, "initial_state": flat, "ssm_state_indices": places})
        q8_scatter(state, indices, flat.view(temp.shape))
        return made
    return wrapped


_Q8_LIVE = {"counter": None, "at": None, "total": 0, "every_s": 60.0, "said": False}


def q8_counter(device):
    if not q8():
        return None
    held = _Q8_LIVE["counter"]
    if held is None:
        if torch.cuda.is_current_stream_capturing():
            if not _Q8_LIVE["said"]:
                _Q8_LIVE["said"] = True
                print("GDN_STATE_Q8 live tally: a verify was captured before the counter existed, so that graph "
                      "counts nothing", flush=True)
            return None
        held = _Q8_LIVE["counter"] = torch.tensor(Q8_EMPTY, dtype=torch.int64, device=device)
    return held


def q8_count_args(device):
    counter = q8_counter(device)
    return {} if counter is None else {"bad": counter, "COUNT": True, **q8_first_args(device)}


def q8_live_tally(for_capture=False):
    held = _Q8_LIVE
    if not q8() or for_capture or held["counter"] is None:
        return None
    now = time.time()
    if held["at"] is None:
        held["at"] = now
        return None
    if now - held["at"] < held["every_s"] or torch.cuda.is_current_stream_capturing():
        return None
    held["at"] = now
    read, wrote, *first = (int(value) for value in held["counter"].tolist())
    held["counter"].copy_(torch.tensor(Q8_EMPTY, dtype=torch.int64))
    held["total"] += read
    print(f"GDN_STATE_Q8 live tally: the verify read {read} non-finite row scale(s) and wrote {wrote} since the last "
          f"tally, {held['total']} read since the first; counted inside the verify kernel on the state blocks it "
          "reads, so any non-zero read count is a corrupt live state", flush=True)
    return q8_first_said(read, first)


def q8_pack(values):
    peak = values.float().abs().amax(dim=-1)
    scale = torch.where(peak > 0, peak / 127.0, torch.ones_like(peak))
    coded = values.float() / scale[..., None]
    coded = torch.where(coded >= 0, coded + 0.5, coded - 0.5).to(torch.int8)
    return coded, scale


def q8_unpack(codes, scales):
    return codes.float() * scales[..., None]


def mode():
    value = os.environ.get(FUSED_ENV, "").strip().lower()
    if value in ("", "0"):
        return None
    if value not in MODES:
        raise ValueError(f"{FUSED_ENV}={value!r} is not one of {sorted(MODES)}")
    if MODES[value] == "lazy" and _LAZY["stood"] is not None:
        return "tiled"
    return MODES[value]


def lazy_steps():
    said = os.environ.get(LAZY_STEPS_ENV, "").strip()
    if not said:
        return LAZY_STEPS_DEFAULT
    try:
        steps = int(said)
    except ValueError:
        steps = 0
    if steps < 1:
        raise ValueError(f"{LAZY_STEPS_ENV} must be a whole number of steps of at least 1, found {said!r}")
    return steps


def _attach_tiled(records):
    keys = torch.zeros((2, *records.key.shape), dtype=records.key.dtype, device=records.key.device)
    gates = torch.zeros((2, *records.gate.shape), dtype=records.gate.dtype, device=records.gate.device)
    records.key = keys[0]
    records.gate = gates[0]
    records.key_sets = keys
    records.gate_sets = gates
    records.phase = torch.zeros(records.rows, dtype=torch.int32, device=records.correction.device)
    return records


def attach_lazy(records, steps):
    rows, v_heads, window, v_dim = records.correction.shape
    k_heads, k_dim = records.key.shape[1], records.key.shape[3]
    device = records.correction.device
    cap = window * steps
    ring = cap + window
    if ring > RING_LIMIT:
        raise ValueError(f"{LAZY_STEPS_ENV}={steps} holds {cap} tokens in a {ring}-token ring; the ring's step "
                         f"marks fit {RING_LIMIT} positions")
    records.correction = torch.zeros((rows, v_heads, ring, v_dim), dtype=torch.float32, device=device)
    records.key = torch.zeros((rows, k_heads, ring, k_dim), dtype=torch.float32, device=device)
    records.gate = torch.zeros((rows, v_heads, ring), dtype=torch.float32, device=device)
    records.draft_window = window
    records.cap = cap
    records.counts = torch.zeros((len(LAZY_FIELDS), rows), dtype=torch.int32, device=device)
    records.handoff = torch.zeros((3, rows), dtype=torch.int32, device=device)
    records.pending = records.counts[0]
    return records


def _lazy_announce(records, steps):
    rows, v_heads, ring, v_dim = records.correction.shape
    k_heads, k_dim = records.key.shape[1], records.key.shape[3]
    window = records.draft_window
    tiled = rows * 4 * window * (v_heads * v_dim + 2 * k_heads * k_dim + 2 * v_heads) + rows * 8
    lazy = records.nbytes + records.counts.numel() * 4 + records.handoff.numel() * 4
    say("lazy", f"GDN_LAZY live ({FUSED_ENV}=lazy, {LAZY_STEPS_ENV}={steps}): the verify folds a per-request "
                f"ring of up to {records.cap} accepted tokens ({ring}-token ring, {window}-token draft window) into "
                f"the state it reads and writes the state only when the ring would overflow, a cache or reply "
                f"boundary is crossed, or the request leaves its slot; {rows} rows x {v_heads} v-heads x "
                f"{v_dim}x{k_dim} ({k_heads} k-heads), verify launch {TILED_BLOCK_V},{TILED_WARPS},{TILED_STAGES}; "
                f"records {lazy / 2**20:.1f} MiB a GDN layer against tiled's {tiled / 2**20:.1f} MiB")


def attach(layer, records):
    if records is None or mode() is None:
        return records
    if mode() == "lazy":
        lazy_guard(records)
    if mode() == "lazy":
        steps = lazy_steps()
        attach_lazy(records, steps)
        _LAZY["counts"].append(records.counts)
        _lazy_announce(records, steps)
        return records
    records.pending = torch.zeros(records.rows, dtype=torch.int32, device=records.correction.device)
    _PENDING.append(records.pending)
    if mode() == "tiled":
        _attach_tiled(records)
        cuda_note_shape(records)
        if cuda_variant() is not None:
            cuda_prebuild()
        say("tiled", "GDN_FUSED live (tiled): the stock convolution and gated norm stay, and the recurrence "
                     "runs in value-tiled programs that fold the previous step's accepted prefix into the "
                     "state they read, write it back once, then verify the new rows; keys and gates "
                     "alternate between two record sets so no tile overwrites what another still reads")
        return records
    say("live", f"GDN_FUSED live ({mode()}): each GDN layer's decode runs conv, gates, the "
                f"recurrence over the request's rows and the gated norm in one kernel"
                + ("; the accepted prefix is folded into the state at the next step's read, so a "
                   "decode step reads and writes each request's state once, and only a row that "
                   "crosses a cache boundary is committed right after sampling"
                   if mode() == "defer" else "; the commit after sampling is unchanged"))
    return records


def slot_fill():
    return str(os.environ.get(SLOT_FILL_ENV) or "").strip().lower() in ("1", "on", "true")


def slot_reset(slot):
    if slot_fill():
        for pending in _PENDING:
            pending[slot].fill_(0)
        for counts in _LAZY["counts"]:
            counts[:, slot].fill_(0)
        say("slot_fill", f"GDN_SLOT_FILL live: a new request clears its pending accepted count in "
                         f"{len(_PENDING) + len(_LAZY['counts'])} GDN layers with a device fill on the current "
                         f"stream, no host copy and no stream sync")
        return
    for pending in _PENDING:
        pending[slot] = 0
    for counts in _LAZY["counts"]:
        counts[:, slot] = 0


def slot_leave(slot):
    if slot is None or mode() != "lazy":
        return 0
    folded = 0
    seen = set()
    for kv_group, builder in reversed(list(_BUILDERS.values())):
        context = getattr(builder, "taaf_defer_context", None)
        if not isinstance(context, LazyContext):
            continue
        held = [state.data_ptr() for state in context.base.states]
        live = _live_states(builder)
        if live is not None and held != [state.data_ptr() for state in live]:
            continue
        if set(held) <= seen:
            continue
        seen.update(held)
        context.leave(int(slot))
        folded += len(context.base.states)
    say("leave", f"GDN_LAZY leave: a request leaving its slot folds every ring step but its last into the block "
                 f"its last step ran on, so that block holds what the tiled path holds there and the last step "
                 f"stays for a resume ({folded} GDN layers on the first one)")
    return folded


def slots_prepare(input_batch, attn_groups, for_capture):
    q8_live_tally(for_capture)
    for kv_group, groups in enumerate(attn_groups):
        for group in groups:
            builder = group.get_metadata_builder(0)
            if getattr(builder, "gdn_accept", False):
                builder.taaf_slots = None if for_capture else input_batch.idx_mapping
                _BUILDERS.pop(id(builder), None)
                _BUILDERS[id(builder)] = (kv_group, builder)


def _live_states(builder):
    names = getattr(builder, "layer_names", None)
    config = getattr(builder, "vllm_config", None)
    if not names or config is None:
        return None
    forward = config.compilation_config.static_forward_context
    return [ssm_full(forward[name]) for name in names]


def slot_resume(slot, blocks, resumed, computed, state_idx, column):
    folded = 0
    seen = set()
    repeated = 0
    stale = 0
    for kv_group, builder in reversed(list(_BUILDERS.values())):
        context = getattr(builder, "taaf_defer_context", None)
        block = blocks.get(kv_group, 0)
        if context is None or block <= 0:
            continue
        held = [state.data_ptr() for state in context.base.states]
        live = _live_states(builder)
        if live is not None and held != [state.data_ptr() for state in live]:
            stale += 1
            continue
        if set(held) <= seen:
            repeated += 1
            continue
        seen.update(held)
        context.resume(slot, block, resumed, computed, state_idx, column)
        folded += len(context.base.states)
    if repeated or stale:
        say("resume-repeated", f"GDN_FUSED resume: {stale} builder(s) held a fold context over GDN states that are no "
                               f"longer their layers' caches and {repeated} over states a more recently registered "
                               f"builder already folded; each live state is folded once")
    say("resume", f"GDN_FUSED resume: a streaming continuation folds the pending accepted tokens its scheduler "
                  f"kept into the running block before the slot is cleared ({folded} GDN layers on the first one); "
                  f"tokens the worker accepted past the scheduler's count stay out")
    return folded


@triton.jit(do_not_specialize=["num_rows"])
def _gdn_fused_kernel(
    x,
    q_in,
    k_in,
    v_in,
    a,
    b,
    z,
    o,
    conv_state,
    conv_weight,
    conv_bias,
    num_accepted,
    A_log,
    dt_bias,
    norm_weight,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    correction,
    keys,
    gates,
    scale,
    eps,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_x,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_z,
    stride_o,
    stride_conv_block,
    stride_conv_dim,
    stride_conv_tok,
    stride_w_dim,
    stride_w_width,
    stride_state_block,
    stride_index,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    CONV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PRODUCT: tl.constexpr,
    NORM_ACT: tl.constexpr,
    DEFER: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    GROUP: tl.constexpr = HV // H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    length = eos - bos
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    row = i_n.to(tl.int64)
    if DEFER:
        row = tl.load(slots + i_n).to(tl.int64)
    if (block <= 0) | (row < 0) | (row >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for j in range(0, GROUP):
            p_o = o + bos * stride_o + (i_h * GROUP + j) * V + o_v
            for _ in range(0, length):
                tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
                p_o += stride_o
        return
    count = length * 0
    if DEFER:
        count = tl.minimum(tl.maximum(tl.load(pending + row), 0), WINDOW).to(tl.int64)
    if CONV:
        offset = tl.load(num_accepted + i_n).to(tl.int64) - 1
        c_q = i_h * K + o_k
        c_k = H * K + i_h * K + o_k
        base_q = conv_state + block * stride_conv_block + c_q * stride_conv_dim
        base_k = conv_state + block * stride_conv_block + c_k * stride_conv_dim
        q0 = tl.load(base_q + offset * stride_conv_tok, mask=mask_k, other=0.0)
        q1 = tl.load(base_q + (offset + 1) * stride_conv_tok, mask=mask_k, other=0.0)
        q2 = tl.load(base_q + (offset + 2) * stride_conv_tok, mask=mask_k, other=0.0)
        k0 = tl.load(base_k + offset * stride_conv_tok, mask=mask_k, other=0.0)
        k1 = tl.load(base_k + (offset + 1) * stride_conv_tok, mask=mask_k, other=0.0)
        k2 = tl.load(base_k + (offset + 2) * stride_conv_tok, mask=mask_k, other=0.0)
        wq0 = tl.load(conv_weight + c_q * stride_w_dim, mask=mask_k, other=0.0)
        wq1 = tl.load(conv_weight + c_q * stride_w_dim + stride_w_width, mask=mask_k, other=0.0)
        wq2 = tl.load(conv_weight + c_q * stride_w_dim + 2 * stride_w_width, mask=mask_k, other=0.0)
        wq3 = tl.load(conv_weight + c_q * stride_w_dim + 3 * stride_w_width, mask=mask_k, other=0.0)
        wk0 = tl.load(conv_weight + c_k * stride_w_dim, mask=mask_k, other=0.0)
        wk1 = tl.load(conv_weight + c_k * stride_w_dim + stride_w_width, mask=mask_k, other=0.0)
        wk2 = tl.load(conv_weight + c_k * stride_w_dim + 2 * stride_w_width, mask=mask_k, other=0.0)
        wk3 = tl.load(conv_weight + c_k * stride_w_dim + 3 * stride_w_width, mask=mask_k, other=0.0)
        if HAS_BIAS:
            bq = tl.load(conv_bias + c_q, mask=mask_k, other=0.0).to(tl.float32)
            bk = tl.load(conv_bias + c_k, mask=mask_k, other=0.0).to(tl.float32)
        else:
            bq = tl.zeros([BK], dtype=tl.float32)
            bk = tl.zeros([BK], dtype=tl.float32)
        tl.debug_barrier()
        tl.store(base_q, q1, mask=mask_k)
        tl.store(base_q + stride_conv_tok, q2, mask=mask_k)
        tl.store(base_k, k1, mask=mask_k)
        tl.store(base_k + stride_conv_tok, k2, mask=mask_k)
        for t in tl.static_range(BT):
            live = mask_k & (t < length)
            xq = tl.load(x + (bos + t) * stride_x + c_q, mask=live, other=0.0)
            xk = tl.load(x + (bos + t) * stride_x + c_k, mask=live, other=0.0)
            tl.store(base_q + (2 + t) * stride_conv_tok, xq, mask=live)
            tl.store(base_k + (2 + t) * stride_conv_tok, xk, mask=live)
            acc_q = bq
            acc_q += (q0.to(tl.float32) * wq0.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_q += (q1.to(tl.float32) * wq1.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_q += (q2.to(tl.float32) * wq2.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_q += (xq.to(tl.float32) * wq3.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_k = bk
            acc_k += (k0.to(tl.float32) * wk0.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_k += (k1.to(tl.float32) * wk1.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_k += (k2.to(tl.float32) * wk2.to(tl.float32)).to(PRODUCT).to(tl.float32)
            acc_k += (xk.to(tl.float32) * wk3.to(tl.float32)).to(PRODUCT).to(tl.float32)
            q0 = q1
            q1 = q2
            q2 = xq
            k0 = k1
            k1 = k2
            k2 = xk
            acc_q = acc_q / (1 + tl.exp(-acc_q))
            acc_k = acc_k / (1 + tl.exp(-acc_k))
            tl.debug_barrier()
            tl.store(x + (bos + t) * stride_x + c_q, acc_q.to(x.dtype.element_ty), mask=live)
            tl.store(x + (bos + t) * stride_x + c_k, acc_k.to(x.dtype.element_ty), mask=live)
        tl.debug_barrier()
    Q_BASE: tl.constexpr = 0
    K_BASE: tl.constexpr = H * K if CONV else 0
    V_BASE: tl.constexpr = 2 * H * K if CONV else 0
    p_key = keys + ((row * H + i_h) * WINDOW) * K + o_k
    for j in range(0, GROUP):
        i_hv = i_h * GROUP + j
        if CONV:
            c_v = 2 * H * K + i_hv * V + o_v
            base_v = conv_state + block * stride_conv_block + c_v * stride_conv_dim
            v0 = tl.load(base_v + offset * stride_conv_tok, mask=mask_v, other=0.0)
            v1 = tl.load(base_v + (offset + 1) * stride_conv_tok, mask=mask_v, other=0.0)
            v2 = tl.load(base_v + (offset + 2) * stride_conv_tok, mask=mask_v, other=0.0)
            wv0 = tl.load(conv_weight + c_v * stride_w_dim, mask=mask_v, other=0.0)
            wv1 = tl.load(conv_weight + c_v * stride_w_dim + stride_w_width, mask=mask_v, other=0.0)
            wv2 = tl.load(conv_weight + c_v * stride_w_dim + 2 * stride_w_width, mask=mask_v,
                          other=0.0)
            wv3 = tl.load(conv_weight + c_v * stride_w_dim + 3 * stride_w_width, mask=mask_v,
                          other=0.0)
            if HAS_BIAS:
                bv = tl.load(conv_bias + c_v, mask=mask_v, other=0.0).to(tl.float32)
            else:
                bv = tl.zeros([BV], dtype=tl.float32)
            tl.debug_barrier()
            tl.store(base_v, v1, mask=mask_v)
            tl.store(base_v + stride_conv_tok, v2, mask=mask_v)
            for t in tl.static_range(BT):
                live = mask_v & (t < length)
                xv = tl.load(x + (bos + t) * stride_x + c_v, mask=live, other=0.0)
                tl.store(base_v + (2 + t) * stride_conv_tok, xv, mask=live)
                acc_v = bv
                acc_v += (v0.to(tl.float32) * wv0.to(tl.float32)).to(PRODUCT).to(tl.float32)
                acc_v += (v1.to(tl.float32) * wv1.to(tl.float32)).to(PRODUCT).to(tl.float32)
                acc_v += (v2.to(tl.float32) * wv2.to(tl.float32)).to(PRODUCT).to(tl.float32)
                acc_v += (xv.to(tl.float32) * wv3.to(tl.float32)).to(PRODUCT).to(tl.float32)
                v0 = v1
                v1 = v2
                v2 = xv
                acc_v = acc_v / (1 + tl.exp(-acc_v))
                tl.debug_barrier()
                tl.store(x + (bos + t) * stride_x + c_v, acc_v.to(x.dtype.element_ty), mask=live)
            tl.debug_barrier()
        p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        p_correction = correction + ((row * HV + i_hv) * WINDOW) * V + o_v
        p_gate = gates + (row * HV + i_hv) * WINDOW
        if DEFER:
            if count > 0:
                for i_t in range(0, count):
                    f_g = tl.load(p_gate + i_t)
                    f_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
                    f_k = tl.load(p_key + i_t * K, mask=mask_k, other=0)
                    b_h = b_h * tl.exp(f_g)
                    b_h = tl.fma(f_v[:, None], f_k[None, :], b_h)
                b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
                tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
            tl.debug_barrier()
        a_log = tl.load(A_log + i_hv).to(tl.float32)
        bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
        if NORM_ACT != 0:
            w_n = tl.load(norm_weight + o_v, mask=mask_v, other=0.0).to(tl.float32)
        p_a = a + bos * stride_a + i_hv
        p_b = b + bos * stride_b + i_hv
        p_o = o + bos * stride_o + i_hv * V + o_v
        p_z = z + bos * stride_z + i_hv * V + o_v
        p_q = q_in + bos * stride_q + Q_BASE + i_h * K + o_k
        p_k = k_in + bos * stride_k + K_BASE + i_h * K + o_k
        p_v = v_in + bos * stride_v + V_BASE + i_hv * V + o_v
        for i_t in range(0, length):
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
            b_beta = tl.sigmoid(b_b.to(tl.float32))
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
            b_q = b_q * scale
            b_h = b_h * tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_v *= b_beta
            b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
            b_o = tl.sum(b_h * b_q[None, :], 1)
            recorded = i_t < WINDOW
            tl.store(p_correction + i_t * V, b_v, mask=mask_v & recorded)
            tl.store(p_gate + i_t, b_g, mask=recorded)
            b_o = b_o.to(p_o.dtype.element_ty).to(tl.float32)
            if NORM_ACT != 0:
                x_n = tl.where(mask_v, b_o, 0.0)
                var = tl.sum(x_n * x_n, 0) / V
                rstd = tl.rsqrt(var + eps)
                y = b_o * rstd
                y = y * w_n
                b_z = tl.load(p_z, mask=mask_v, other=0.0).to(tl.float32)
                if NORM_ACT == 1:
                    y *= b_z * tl.sigmoid(b_z)
                else:
                    y *= tl.sigmoid(b_z)
                b_o = y
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            p_q += stride_q
            p_k += stride_k
            p_v += stride_v
            p_a += stride_a
            p_b += stride_b
            p_o += stride_o
            p_z += stride_z
    tl.debug_barrier()
    p_k = k_in + bos * stride_k + K_BASE + i_h * K + o_k
    for i_t in range(0, tl.minimum(length, WINDOW)):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        tl.store(p_key + i_t * K, b_k, mask=mask_k)
        p_k += stride_k


@triton.jit(do_not_specialize=["num_rows", "layer_id"])
def _verify_tiled_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    phase,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    stride_key_set,
    stride_gate_set,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
    Q8: tl.constexpr = False,
    bad=None,
    COUNT: tl.constexpr = False, layer_id=0, FIRST: tl.constexpr = False, fresh=None,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    side = tl.load(phase + slot).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(pending + slot), 0), WINDOW)
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if Q8:
        b_h += _q8_tile(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
        if COUNT:
            _q8_count_read(state, block * stride_state_block, i_hv, o_v, mask_v, bad, HV, V, K, block, slot, layer_id, FIRST, fresh)
    else:
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key_held = keys + side * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate_held = gates + side * stride_gate_set + (slot * HV + i_hv) * WINDOW
    if count > 0:
        for i_t in range(0, count):
            f_g = tl.load(p_gate_held + i_t)
            f_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
            f_k = tl.load(p_key_held + i_t * K, mask=mask_k, other=0)
            b_h = b_h * tl.exp(f_g)
            b_h = tl.fma(f_v[:, None], f_k[None, :], b_h)
        if Q8:
            b_h = _q8_put(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
            if COUNT:
                _q8_count_bad(state, block * stride_state_block, i_hv, o_v, mask_v, bad + 1, HV, V, K)
        else:
            b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
            tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    p_key = keys + (1 - side) * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate = gates + (1 - side) * stride_gate_set + (slot * HV + i_hv) * WINDOW
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, length):
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
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        recorded = i_t < WINDOW
        tl.store(p_correction + i_t * V, b_v, mask=mask_v & recorded)
        tl.store(p_gate + i_t, b_g, mask=head_writer & recorded)
        tl.store(p_key + i_t * K, b_k, mask=mask_k & key_writer & recorded)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o


@triton.jit(do_not_specialize=["num_rows", "layer_id"])
def _verify_ahead_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    phase,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    stride_key_set,
    stride_gate_set,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
    Q8: tl.constexpr = False,
    bad=None,
    COUNT: tl.constexpr = False, layer_id=0, FIRST: tl.constexpr = False, fresh=None,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    length = eos - bos
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if Q8:
        held_h = _q8_tile(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
        if COUNT:
            _q8_count_read(state, block * stride_state_block, i_hv, o_v, mask_v, bad, HV, V, K, block, slot, layer_id, FIRST, fresh)
    else:
        held_h = tl.load(p_h, mask=mask_h, other=0)
    next_q = tl.load(p_q, mask=mask_k, other=0)
    next_k = tl.load(p_k, mask=mask_k, other=0)
    next_v = tl.load(p_v, mask=mask_v, other=0)
    next_b = tl.load(p_b)
    next_a = tl.load(p_a)
    side = tl.load(phase + slot).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(pending + slot), 0), WINDOW)
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key_held = keys + side * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate_held = gates + side * stride_gate_set + (slot * HV + i_hv) * WINDOW
    some = count > 0
    f_g = tl.load(p_gate_held, mask=some, other=0)
    f_v = tl.load(p_correction, mask=mask_v & some, other=0)
    f_k = tl.load(p_key_held, mask=mask_k & some, other=0)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += held_h.to(tl.float32)
    if count > 0:
        for i_t in range(0, count):
            more = i_t + 1 < count
            g_t = f_g
            v_t = f_v
            k_t = f_k
            f_g = tl.load(p_gate_held + i_t + 1, mask=more, other=0)
            f_v = tl.load(p_correction + (i_t + 1) * V, mask=mask_v & more, other=0)
            f_k = tl.load(p_key_held + (i_t + 1) * K, mask=mask_k & more, other=0)
            b_h = b_h * tl.exp(g_t)
            b_h = tl.fma(v_t[:, None], k_t[None, :], b_h)
        if Q8:
            b_h = _q8_put(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
            if COUNT:
                _q8_count_bad(state, block * stride_state_block, i_hv, o_v, mask_v, bad + 1, HV, V, K)
        else:
            b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
            tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    p_key = keys + (1 - side) * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate = gates + (1 - side) * stride_gate_set + (slot * HV + i_hv) * WINDOW
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    for i_t in range(0, length):
        more = i_t + 1 < length
        b_q = next_q.to(tl.float32)
        b_k = next_k.to(tl.float32)
        b_v = next_v.to(tl.float32)
        b_b = next_b.to(tl.float32)
        x_g = next_a.to(tl.float32) + bias_t
        next_q = tl.load(p_q + stride_q, mask=mask_k & more, other=0)
        next_k = tl.load(p_k + stride_k, mask=mask_k & more, other=0)
        next_v = tl.load(p_v + stride_v, mask=mask_v & more, other=0)
        next_b = tl.load(p_b + stride_b, mask=more, other=0)
        next_a = tl.load(p_a + stride_a, mask=more, other=0)
        softplus_x = tl.where(
            softplus_beta * x_g <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)),
            x_g,
        )
        b_g = -tl.exp(a_log) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        recorded = i_t < WINDOW
        tl.store(p_correction + i_t * V, b_v, mask=mask_v & recorded)
        tl.store(p_gate + i_t, b_g, mask=head_writer & recorded)
        tl.store(p_key + i_t * K, b_k, mask=mask_k & key_writer & recorded)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o


@triton.jit(do_not_specialize=["num_rows", "layer_id", "total_tokens"])
def _verify_ahead_tail_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    phase,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    stride_key_set,
    stride_gate_set,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
    Q8: tl.constexpr = False,
    bad=None,
    COUNT: tl.constexpr = False, layer_id=0, FIRST: tl.constexpr = False, fresh=None,
    total_tokens=0,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    length = eos - bos
    if i_n == num_rows - 1:
        o_tail = i_v * BV + tl.arange(0, BV)
        p_tail = o + eos * stride_o + i_hv * V + o_tail
        blank = tl.zeros([BV], dtype=tl.float32)
        for _ in range(eos, total_tokens):
            tl.store(p_tail, blank.to(p_tail.dtype.element_ty), mask=o_tail < V)
            p_tail += stride_o
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    if Q8:
        held_h = _q8_tile(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
        if COUNT:
            _q8_count_read(state, block * stride_state_block, i_hv, o_v, mask_v, bad, HV, V, K, block, slot, layer_id, FIRST, fresh)
    else:
        held_h = tl.load(p_h, mask=mask_h, other=0)
    next_q = tl.load(p_q, mask=mask_k, other=0)
    next_k = tl.load(p_k, mask=mask_k, other=0)
    next_v = tl.load(p_v, mask=mask_v, other=0)
    next_b = tl.load(p_b)
    next_a = tl.load(p_a)
    side = tl.load(phase + slot).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(pending + slot), 0), WINDOW)
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key_held = keys + side * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate_held = gates + side * stride_gate_set + (slot * HV + i_hv) * WINDOW
    some = count > 0
    f_g = tl.load(p_gate_held, mask=some, other=0)
    f_v = tl.load(p_correction, mask=mask_v & some, other=0)
    f_k = tl.load(p_key_held, mask=mask_k & some, other=0)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += held_h.to(tl.float32)
    if count > 0:
        for i_t in range(0, count):
            more = i_t + 1 < count
            g_t = f_g
            v_t = f_v
            k_t = f_k
            f_g = tl.load(p_gate_held + i_t + 1, mask=more, other=0)
            f_v = tl.load(p_correction + (i_t + 1) * V, mask=mask_v & more, other=0)
            f_k = tl.load(p_key_held + (i_t + 1) * K, mask=mask_k & more, other=0)
            b_h = b_h * tl.exp(g_t)
            b_h = tl.fma(v_t[:, None], k_t[None, :], b_h)
        if Q8:
            b_h = _q8_put(state, block * stride_state_block, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
            if COUNT:
                _q8_count_bad(state, block * stride_state_block, i_hv, o_v, mask_v, bad + 1, HV, V, K)
        else:
            b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
            tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    p_key = keys + (1 - side) * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate = gates + (1 - side) * stride_gate_set + (slot * HV + i_hv) * WINDOW
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    for i_t in range(0, length):
        more = i_t + 1 < length
        b_q = next_q.to(tl.float32)
        b_k = next_k.to(tl.float32)
        b_v = next_v.to(tl.float32)
        b_b = next_b.to(tl.float32)
        x_g = next_a.to(tl.float32) + bias_t
        next_q = tl.load(p_q + stride_q, mask=mask_k & more, other=0)
        next_k = tl.load(p_k + stride_k, mask=mask_k & more, other=0)
        next_v = tl.load(p_v + stride_v, mask=mask_v & more, other=0)
        next_b = tl.load(p_b + stride_b, mask=more, other=0)
        next_a = tl.load(p_a + stride_a, mask=more, other=0)
        softplus_x = tl.where(
            softplus_beta * x_g <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)),
            x_g,
        )
        b_g = -tl.exp(a_log) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        recorded = i_t < WINDOW
        tl.store(p_correction + i_t * V, b_v, mask=mask_v & recorded)
        tl.store(p_gate + i_t, b_g, mask=head_writer & recorded)
        tl.store(p_key + i_t * K, b_k, mask=mask_k & key_writer & recorded)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o


@triton.jit(do_not_specialize=["num_rows", "num_tiles"])
def _verify_persist_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    phase,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    stride_key_set,
    stride_gate_set,
    num_tiles,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    SPLIT: tl.constexpr = (V + BV - 1) // BV
    programs = tl.num_programs(0)
    first = tl.program_id(0)
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    live = first < num_tiles
    row = (first // SPLIT) // HV
    block = tl.load(state_indices + row * stride_index, mask=live, other=0).to(tl.int64)
    bos = tl.load(cu_seqlens + row, mask=live, other=0).to(tl.int64)
    eos = tl.load(cu_seqlens + row + 1, mask=live, other=0).to(tl.int64)
    slot = tl.load(slots + row, mask=live, other=-1).to(tl.int64)
    known = live & (slot >= 0) & (slot < RECORD_ROWS)
    side = tl.load(phase + slot, mask=known, other=0).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(pending + slot, mask=known, other=0), 0), WINDOW)
    first_v = (first % SPLIT) * BV + tl.arange(0, BV)
    first_hv = (first // SPLIT) % HV
    held = tl.load(state + block * stride_state_block + first_hv * V * K + first_v[:, None] * K + o_k[None, :],
                   mask=(first_v < V)[:, None] & mask_k[None, :] & live & (block > 0) & (eos > bos), other=0)
    after = first + programs
    has_after = after < num_tiles
    row = (after // SPLIT) // HV
    next_block = tl.load(state_indices + row * stride_index, mask=has_after, other=0).to(tl.int64)
    next_bos = tl.load(cu_seqlens + row, mask=has_after, other=0).to(tl.int64)
    next_eos = tl.load(cu_seqlens + row + 1, mask=has_after, other=0).to(tl.int64)
    next_slot = tl.load(slots + row, mask=has_after, other=-1).to(tl.int64)
    for tile in range(first, num_tiles, programs):
        after = tile + programs
        has_after = after < num_tiles
        after_v = (after % SPLIT) * BV + tl.arange(0, BV)
        after_hv = (after // SPLIT) % HV
        upcoming = tl.load(
            state + next_block * stride_state_block + after_hv * V * K + after_v[:, None] * K + o_k[None, :],
            mask=(after_v < V)[:, None] & mask_k[None, :] & has_after & (next_block > 0) & (next_eos > next_bos),
            other=0)
        next_known = has_after & (next_slot >= 0) & (next_slot < RECORD_ROWS)
        next_side = tl.load(phase + next_slot, mask=next_known, other=0).to(tl.int64)
        next_count = tl.minimum(tl.maximum(tl.load(pending + next_slot, mask=next_known, other=0), 0), WINDOW)
        later = after + programs
        has_later = later < num_tiles
        row = (later // SPLIT) // HV
        later_block = tl.load(state_indices + row * stride_index, mask=has_later, other=0).to(tl.int64)
        later_bos = tl.load(cu_seqlens + row, mask=has_later, other=0).to(tl.int64)
        later_eos = tl.load(cu_seqlens + row + 1, mask=has_later, other=0).to(tl.int64)
        later_slot = tl.load(slots + row, mask=has_later, other=-1).to(tl.int64)
        i_v = tile % SPLIT
        i_hv = (tile // SPLIT) % HV
        i_h = i_hv // (HV // H)
        length = eos - bos
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]
        p_o = o + bos * stride_o + i_hv * V + o_v
        if length > 0:
            if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
                zero = tl.zeros([BV], dtype=tl.float32)
                for _ in range(0, length):
                    tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
                    p_o += stride_o
            else:
                p_q = q + bos * stride_q + i_h * K + o_k
                p_k = k + bos * stride_k + i_h * K + o_k
                p_v = v + bos * stride_v + i_hv * V + o_v
                p_a = a + bos * stride_a + i_hv
                p_b = b + bos * stride_b + i_hv
                p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                next_q = tl.load(p_q, mask=mask_k, other=0)
                next_k = tl.load(p_k, mask=mask_k, other=0)
                next_v = tl.load(p_v, mask=mask_v, other=0)
                next_b = tl.load(p_b)
                next_a = tl.load(p_a)
                a_log = tl.load(A_log + i_hv).to(tl.float32)
                bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
                p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
                p_key_held = keys + side * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
                p_gate_held = gates + side * stride_gate_set + (slot * HV + i_hv) * WINDOW
                some = count > 0
                f_g = tl.load(p_gate_held, mask=some, other=0)
                f_v = tl.load(p_correction, mask=mask_v & some, other=0)
                f_k = tl.load(p_key_held, mask=mask_k & some, other=0)
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
                b_h += held.to(tl.float32)
                if count > 0:
                    for i_t in range(0, count):
                        more = i_t + 1 < count
                        g_t = f_g
                        v_t = f_v
                        k_t = f_k
                        f_g = tl.load(p_gate_held + i_t + 1, mask=more, other=0)
                        f_v = tl.load(p_correction + (i_t + 1) * V, mask=mask_v & more, other=0)
                        f_k = tl.load(p_key_held + (i_t + 1) * K, mask=mask_k & more, other=0)
                        b_h = b_h * tl.exp(g_t)
                        b_h = tl.fma(v_t[:, None], k_t[None, :], b_h)
                    b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
                    tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
                p_key = keys + (1 - side) * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
                p_gate = gates + (1 - side) * stride_gate_set + (slot * HV + i_hv) * WINDOW
                head_writer = i_v == 0
                key_writer = head_writer & (i_hv % (HV // H) == 0)
                for i_t in range(0, length):
                    more = i_t + 1 < length
                    b_q = next_q.to(tl.float32)
                    b_k = next_k.to(tl.float32)
                    b_v = next_v.to(tl.float32)
                    b_b = next_b.to(tl.float32)
                    x_g = next_a.to(tl.float32) + bias_t
                    next_q = tl.load(p_q + stride_q, mask=mask_k & more, other=0)
                    next_k = tl.load(p_k + stride_k, mask=mask_k & more, other=0)
                    next_v = tl.load(p_v + stride_v, mask=mask_v & more, other=0)
                    next_b = tl.load(p_b + stride_b, mask=more, other=0)
                    next_a = tl.load(p_a + stride_a, mask=more, other=0)
                    softplus_x = tl.where(
                        softplus_beta * x_g <= softplus_threshold,
                        (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)),
                        x_g,
                    )
                    b_g = -tl.exp(a_log) * softplus_x
                    b_beta = tl.sigmoid(b_b.to(tl.float32))
                    b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
                    b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
                    b_q = b_q * scale
                    b_h = b_h * tl.exp(b_g)
                    b_v -= tl.sum(b_h * b_k[None, :], 1)
                    b_v *= b_beta
                    b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
                    b_o = tl.sum(b_h * b_q[None, :], 1)
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                    recorded = i_t < WINDOW
                    tl.store(p_correction + i_t * V, b_v, mask=mask_v & recorded)
                    tl.store(p_gate + i_t, b_g, mask=head_writer & recorded)
                    tl.store(p_key + i_t * K, b_k, mask=mask_k & key_writer & recorded)
                    p_q += stride_q
                    p_k += stride_k
                    p_v += stride_v
                    p_a += stride_a
                    p_b += stride_b
                    p_o += stride_o
        held = upcoming
        block = next_block
        bos = next_bos
        eos = next_eos
        slot = next_slot
        side = next_side
        count = next_count
        next_block = later_block
        next_bos = later_bos
        next_eos = later_eos
        next_slot = later_slot


@triton.jit(do_not_specialize=["num_rows"])
def _verify_chunk_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    pending,
    phase,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    stride_key_set,
    stride_gate_set,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
    BT: tl.constexpr = 8,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    length = eos - bos
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    live = o_t < length
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        tl.store(o + (bos + o_t[None, :]) * stride_o + i_hv * V + o_v[:, None],
                 tl.zeros([BV, BT], dtype=tl.float32).to(o.dtype.element_ty), mask=mask_v[:, None] & live[None, :])
        return
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    held_h = tl.load(p_h, mask=mask_h, other=0)
    rows_k = tl.load(k + (bos + o_t[:, None]) * stride_k + i_h * K + o_k[None, :], mask=live[:, None] & mask_k[None, :],
                     other=0).to(tl.float32)
    rows_v = tl.load(v + (bos + o_t[None, :]) * stride_v + i_hv * V + o_v[:, None], mask=mask_v[:, None] & live[None, :],
                     other=0).to(tl.float32)
    raw_a = tl.load(a + (bos + o_t) * stride_a + i_hv, mask=live, other=0).to(tl.float32)
    raw_b = tl.load(b + (bos + o_t) * stride_b + i_hv, mask=live, other=0).to(tl.float32)
    side = tl.load(phase + slot).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(pending + slot), 0), WINDOW)
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key_held = keys + side * stride_key_set + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate_held = gates + side * stride_gate_set + (slot * HV + i_hv) * WINDOW
    some = count > 0
    f_g = tl.load(p_gate_held, mask=some, other=0)
    f_v = tl.load(p_correction, mask=mask_v & some, other=0)
    f_k = tl.load(p_key_held, mask=mask_k & some, other=0)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += held_h.to(tl.float32)
    if count > 0:
        for i_t in range(0, count):
            more = i_t + 1 < count
            g_t = f_g
            v_t = f_v
            k_t = f_k
            f_g = tl.load(p_gate_held + i_t + 1, mask=more, other=0)
            f_v = tl.load(p_correction + (i_t + 1) * V, mask=mask_v & more, other=0)
            f_k = tl.load(p_key_held + (i_t + 1) * K, mask=mask_k & more, other=0)
            b_h = b_h * tl.exp(g_t)
            b_h = tl.fma(v_t[:, None], k_t[None, :], b_h)
        b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
        tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    x_g = raw_a + bias_t
    softplus_x = tl.where(softplus_beta * x_g <= softplus_threshold,
                          (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)), x_g)
    g_all = tl.where(live, -tl.exp(a_log) * softplus_x, 0.0)
    beta = tl.sigmoid(raw_b)
    rows_k = rows_k * tl.rsqrt(tl.sum(rows_k * rows_k, 1) + 1e-6)[:, None]
    cum = tl.cumsum(g_all, 0)
    proj_k = tl.zeros([BV, BT], dtype=tl.float32)
    proj_q = tl.zeros([BV, BT], dtype=tl.float32)
    kk = tl.zeros([BT, BT], dtype=tl.float32)
    kq = tl.zeros([BT, BT], dtype=tl.float32)
    for i_t in tl.static_range(WINDOW):
        pick = o_t == i_t
        k_t = tl.sum(tl.where(pick[:, None], rows_k, 0.0), 0)
        q_t = tl.load(q + (bos + i_t) * stride_q + i_h * K + o_k, mask=mask_k & (i_t < length), other=0).to(tl.float32)
        q_t = q_t * tl.rsqrt(tl.sum(q_t * q_t) + 1e-6) * scale
        proj_k = tl.where(pick[None, :], tl.sum(b_h * k_t[None, :], 1)[:, None], proj_k)
        proj_q = tl.where(pick[None, :], tl.sum(b_h * q_t[None, :], 1)[:, None], proj_q)
        kk = tl.where(pick[None, :], tl.sum(rows_k * k_t[None, :], 1)[:, None], kk)
        kq = tl.where(pick[None, :], tl.sum(rows_k * q_t[None, :], 1)[:, None], kq)
    lower = o_t[:, None] < o_t[None, :]
    upto = o_t[:, None] <= o_t[None, :]
    span = tl.exp(cum[None, :] - cum[:, None])
    mix = tl.where(lower, span * kk, 0.0)
    weigh = tl.where(upto & live[:, None] & live[None, :], span * kq, 0.0)
    decay = tl.exp(cum)
    fresh = tl.zeros([BV, BT], dtype=tl.float32)
    for i_t in tl.static_range(WINDOW):
        pick = o_t == i_t
        column = tl.sum(tl.where(pick[None, :], rows_v - decay[None, :] * proj_k, 0.0), 1)
        carried = tl.sum(fresh * tl.sum(tl.where(pick[None, :], mix, 0.0), 1)[None, :], 1)
        b_beta = tl.sum(tl.where(pick, beta, 0.0), 0)
        value = b_beta * (column - carried)
        fresh = tl.where(pick[None, :] & (i_t < length), value[:, None], fresh)
    out_rows = decay[None, :] * proj_q
    for i_s in tl.static_range(WINDOW):
        pick = o_t == i_s
        from_s = tl.sum(tl.where(pick[None, :], fresh, 0.0), 1)
        row_w = tl.sum(tl.where(pick[:, None], weigh, 0.0), 0)
        out_rows += from_s[:, None] * row_w[None, :]
    write = mask_v[:, None] & live[None, :]
    tl.store(o + (bos + o_t[None, :]) * stride_o + i_hv * V + o_v[:, None], out_rows.to(o.dtype.element_ty), mask=write)
    recorded = live & (o_t < WINDOW)
    tl.store(p_correction[:, None] + o_t[None, :] * V, fresh, mask=mask_v[:, None] & recorded[None, :])
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    p_gate = gates + (1 - side) * stride_gate_set + (slot * HV + i_hv) * WINDOW
    p_key = keys + (1 - side) * stride_key_set + ((slot * H + i_h) * WINDOW) * K
    tl.store(p_gate + o_t, g_all, mask=recorded & head_writer)
    tl.store(p_key + o_t[:, None] * K + o_k[None, :], rows_k, mask=recorded[:, None] & mask_k[None, :] & key_writer)


def verify_registers():
    said = os.environ.get(MAXNREG_ENV, "").strip()
    if not said:
        return {}
    if not said.isdigit() or not 32 <= int(said) <= 255 or int(said) % 8:
        raise ValueError(f"{MAXNREG_ENV} must be a multiple of 8 from 32 to 255, found {said!r}")
    say("verify-maxnreg", f"GDN_FUSED {MAXNREG_ENV}={said}: the verify kernels compile with at most {said} registers "
                          "a thread, trading spills for more programs a SM; the arithmetic is unchanged")
    return {"maxnreg": int(said)}


def verify_kernel():
    said = os.environ.get(AHEAD_ENV, "").strip()
    if said not in ("", "0", "1"):
        raise ValueError(f"{AHEAD_ENV} must be 0 or 1, found {said!r}")
    if said != "1":
        return _verify_tiled_kernel
    say("verify-ahead", f"GDN_FUSED {AHEAD_ENV}=1: the tiled verify loads the next token's inputs and the next "
                        "pending fold while it computes the current one; outputs and states match the tiled verify")
    return _verify_ahead_kernel


DECODE_TAIL_ENV = "TAAF_VLLM_GDN_DECODE_TAIL_ZERO"


def decode_tail_zero():
    value = os.environ.get(DECODE_TAIL_ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{DECODE_TAIL_ENV}={value!r} is not one of '', '0', '1'")
    return value == "1"


def decode_tail_kernel():
    return (decode_tail_zero() and mode() == "tiled" and verify_kernel() is _verify_ahead_kernel
            and cuda_variant() is None)


def decode_tail_owned(layer, num_tokens):
    if not decode_tail_kernel():
        return False
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    held = get_forward_context().attn_metadata
    metadata = held.get(layer.prefix) if isinstance(held, dict) else None
    if metadata is None or getattr(layer, "gdn_accept_records", None) is None:
        return False
    if (getattr(metadata, "spec_sequence_masks", None) is None or metadata.num_prefills != 0
            or metadata.num_decodes != 0 or metadata.num_spec_decodes <= 0
            or metadata.num_actual_tokens != num_tokens):
        return False
    say("decode-tail", f"GDN_DECODE_TAIL_ZERO live: a decode step allocates its {num_tokens}-row GDN output "
                       "empty and the tiled verify writes the rows past the last request's tokens as zeros, "
                       "in place of a zero fill of the whole output a layer")
    return True


CUDA_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA"
CUDA_TOKENS = 5
CUDA_FILES = ("taaf_gdn_verify_cuda.py", "gdn_verify_cuda.py")
_CUDA = {"module": None, "built": {}, "said": set(), "shapes": set(), "fired": False}


def _cuda_module():
    if _CUDA["module"] is None:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        found = next((os.path.join(here, name) for name in CUDA_FILES if os.path.isfile(os.path.join(here, name))), None)
        if found is None:
            raise FileNotFoundError(f"{CUDA_ENV} is set and none of {CUDA_FILES} sits beside {os.path.basename(__file__)}")
        spec = importlib.util.spec_from_file_location("taaf_gdn_verify_cuda", found)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CUDA["module"] = module
    return _CUDA["module"]


def cuda_variant():
    said = str(os.environ.get(CUDA_ENV) or "").strip()
    if not said or said == "0":
        return None
    return _cuda_module().parse(said)


def _cuda_say(key, line):
    if key not in _CUDA["said"]:
        _CUDA["said"].add(key)
        print(line, flush=True)


def _cuda_capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _cuda_name(variant):
    return ",".join(str(part) for part in variant)


def _cuda_fits(variant, shape):
    k_heads, v_heads, k_dim, v_dim, window = shape
    if window != CUDA_TOKENS or v_dim % variant[0] or k_dim % 16 or k_dim % variant[0]:
        return False
    return len(variant) != 4 or k_dim == 128


def _cuda_built(cuda, variant, shape):
    key = (tuple(variant), *shape)
    held = _CUDA["built"].get(key)
    if held is None:
        started = time.monotonic()
        held = _CUDA["built"][key] = cuda.build(variant, *shape)
        _cuda_say(("built", key), f"GDN_VERIFY_CUDA built variant={_cuda_name(variant)} k_heads={shape[0]} "
                                  f"v_heads={shape[1]} k_dim={shape[2]} v_dim={shape[3]} window={shape[4]} in "
                                  f"{time.monotonic() - started:.1f} s ({CUDA_ENV})")
    return held


def cuda_note_shape(records):
    _, v_heads, window, v_dim = records.correction.shape
    _CUDA["shapes"].add((records.key.shape[1], v_heads, records.key.shape[3], v_dim, window))


def cuda_prebuild():
    _CUDA["said"].clear()
    _CUDA["fired"] = False
    variant = cuda_variant()
    if variant is None:
        return []
    cuda = _cuda_module()
    built = []
    for shape in sorted(_CUDA["shapes"]):
        if _cuda_fits(variant, shape):
            _cuda_built(cuda, variant, shape)
            built.append(list(shape))
    return built


def _fixed_total_rows():
    return str(os.environ.get("TAAF_VLLM_DRAFT_BUDGET") or "").strip() not in ("", "0")


def _cuda_eligible(layer, decode, tokens, rows, window):
    uniform = tokens == rows * CUDA_TOKENS or (_fixed_total_rows() and 0 < tokens <= rows * CUDA_TOKENS)
    return (bool(decode) and rows > 0 and window == CUDA_TOKENS and getattr(layer, "num_spec", None) == CUDA_TOKENS - 1
            and uniform)


def _verify_cuda(layer, records, state, q2, k2, v2, a, b, o2, cu, indices, slots, rows, tokens, k_heads, v_heads,
                 k_dim, v_dim, decode):
    variant = cuda_variant()
    if variant is None or not _cuda_eligible(layer, decode, tokens, rows, records.window):
        return False
    cuda = _cuda_module()
    shape = (k_heads, v_heads, k_dim, v_dim, records.window)
    capturing = _cuda_capturing()
    refused = [] if q8() else [f"int8 state: {Q8_ENV} is not 1"]
    refused += cuda.refusals(state=state, q=q2, k=k2, v=v2, a=a, b=b, o=o2, A_log=layer.A_log, dt_bias=layer.dt_bias,
                             cu=cu, sidx=indices, slots=slots, pending=records.pending, phase=records.phase,
                             correction=records.correction, key_sets=records.key_sets, gate_sets=records.gate_sets,
                             k_dim=k_dim, v_dim=v_dim, bv=variant[0], low_register=len(variant) == 4)
    if refused:
        _cuda_say(("unsupported", tuple(refused)), f"GDN_VERIFY_CUDA stands aside: {CUDA_ENV}={_cuda_name(variant)} refused on "
                                 f"{len(refused)} condition(s), so the Triton verify runs: " + "; ".join(refused))
        return False
    if (tuple(variant), *shape) not in _CUDA["built"] and capturing:
        _cuda_say("capture", f"GDN_VERIFY_CUDA stands aside: variant {_cuda_name(variant)} was not built before "
                             "this capture, so the Triton verify runs")
        return False
    built = _cuda_built(cuda, variant, shape)
    cuda.launch(built, A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=q2, k=k2, v=v2, o=o2,
                state=state, cu=cu, sidx=indices, slots=slots, pending=records.pending, phase=records.phase,
                correction=records.correction, key_sets=records.key_sets, gate_sets=records.gate_sets, rows=rows,
                scale=k_dim**-0.5, softplus_beta=SOFTPLUS_BETA, softplus_threshold=SOFTPLUS_THRESHOLD,
                record_rows=records.rows, pdl=pdl_on(), bad=q8_counter(state.device))
    if not _CUDA["fired"]:
        _CUDA["fired"] = True
        print(f"GDN_VERIFY_CUDA fired variant={_cuda_name(variant)} tokens={CUDA_TOKENS} rows={rows} "
              f"capture={int(capturing)} ({CUDA_ENV})", flush=True)
    return True


def run_tiled(layer, *, q, k, v, a, b, out, query_start_loc, state_indices, slots, decode=False):
    records = layer.gdn_accept_records
    state = ssm_full(layer)
    rows = state_indices.shape[0]
    tokens = q.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    k_heads = records.key.shape[1]
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError("GDN_FUSED needs contiguous state heads")
    if rows > records.rows:
        raise ValueError(f"GDN_FUSED runs {rows} rows but holds records for {records.rows}")
    if query_start_loc.shape[0] < rows + 1:
        raise ValueError("GDN_FUSED query metadata is shorter than its rows")
    if v_heads % k_heads:
        raise ValueError("GDN_FUSED needs whole value-head groups per key head")
    if slots is None or slots.shape[0] < rows:
        raise ValueError("GDN_FUSED tiled needs a request slot for every row")
    if any(t.shape[0] != tokens for t in (k, v, a, b)):
        raise ValueError("GDN_FUSED inputs disagree on the token count")
    if tokens == 0 or rows == 0:
        return out
    q2, k2, v2 = _rows_of(q, tokens), _rows_of(k, tokens), _rows_of(v, tokens)
    if q2.shape[1] != k_heads * k_dim or k2.shape[1] != k_heads * k_dim or v2.shape[1] != v_heads * v_dim:
        raise ValueError("GDN_FUSED q, k and v do not match the state's heads")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("GDN_FUSED needs contiguous gate heads")
    o2 = out.reshape(tokens, v_heads * v_dim)
    if o2.data_ptr() != out.data_ptr() or o2.stride(1) != 1:
        raise ValueError("GDN_FUSED needs a contiguous output row per token")
    block_v = min(triton.next_power_of_2(v_dim), TILED_BLOCK_V)
    grid = (triton.cdiv(v_dim, block_v), rows * v_heads)
    say("verify-align", "GDN_FUSED verify alignment, 0 everywhere lets the state and inputs load 16 bytes at a time: "
                        f"pointers mod 16 state={state.data_ptr() % 16} q={q2.data_ptr() % 16} "
                        f"k={k2.data_ptr() % 16} v={v2.data_ptr() % 16} o={o2.data_ptr() % 16}, strides mod 16 "
                        f"state={state.stride(0) % 16} q={q2.stride(0) % 16} k={k2.stride(0) % 16} "
                        f"v={v2.stride(0) % 16} o={o2.stride(0) % 16}")
    if _verify_cuda(layer, records, state, q2, k2, v2, a, b, o2, query_start_loc, state_indices, slots, rows,
                    tokens, k_heads, v_heads, k_dim, v_dim, decode):
        return out
    tail = decode and decode_tail_kernel()
    (_verify_ahead_tail_kernel if tail else verify_kernel())[grid](
        layer.A_log, a, b, layer.dt_bias, q2, k2, v2, o2, state, query_start_loc, state_indices,
        slots, records.pending, records.phase, records.correction, records.key_sets, records.gate_sets,
        k_dim**-0.5, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD, rows,
        q2.stride(0), k2.stride(0), v2.stride(0), a.stride(0), b.stride(0), o2.stride(0),
        state.stride(0), state_indices.stride(0), records.key_sets.stride(0), records.gate_sets.stride(0),
        H=k_heads, HV=v_heads, K=k_dim, V=v_dim, BK=triton.next_power_of_2(k_dim), BV=block_v,
        WINDOW=records.window, RECORD_ROWS=records.rows,
        num_warps=TILED_WARPS, num_stages=TILED_STAGES, **verify_registers(),
        PDL=pdl_on(), launch_pdl=pdl_on(), Q8=q8(), layer_id=layer_index(layer), **q8_count_args(state.device),
        **({"total_tokens": tokens} if tail else {}),
    )
    return out


@triton.jit(do_not_specialize=["num_rows"])
def _verify_lazy_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    counts,
    handoff,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RING: tl.constexpr,
    CAP: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    keep = tl.minimum(tl.maximum(tl.load(counts + slot), 0), CAP)
    head = tl.load(counts + RECORD_ROWS + slot) % RING
    ends = tl.load(counts + 2 * RECORD_ROWS + slot)
    flush = (keep + length) > CAP
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    p_correction = correction + ((slot * HV + i_hv) * RING) * V + o_v
    p_key = keys + ((slot * H + i_h) * RING) * K + o_k
    p_gate = gates + (slot * HV + i_hv) * RING
    for i_t in range(0, keep):
        position = (head + i_t) % RING
        f_g = tl.load(p_gate + position)
        f_v = tl.load(p_correction + position * V, mask=mask_v, other=0)
        f_k = tl.load(p_key + position * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(f_g)
        b_h = tl.fma(f_v[:, None], f_k[None, :], b_h)
        if ((ends >> position) & 1) != 0:
            b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
    if flush & (keep > 0):
        tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    start = head + keep
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    if head_writer & (i_hv == 0):
        tl.store(handoff + slot, tl.where(flush, 0, keep))
        tl.store(handoff + RECORD_ROWS + slot, tl.where(flush, start % RING, head))
        tl.store(handoff + 2 * RECORD_ROWS + slot, tl.where(flush, 0, ends))
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, length):
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
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        recorded = i_t < (RING - CAP)
        position = (start + i_t) % RING
        tl.store(p_correction + position * V, b_v, mask=mask_v & recorded)
        tl.store(p_gate + position, b_g, mask=head_writer & recorded)
        tl.store(p_key + position * K, b_k, mask=mask_k & key_writer & recorded)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o


@triton.jit(do_not_specialize=["num_rows"])
def _verify_lazy_ahead_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    slots,
    counts,
    handoff,
    correction,
    keys,
    gates,
    scale,
    softplus_beta,
    softplus_threshold,
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_o,
    stride_state_block,
    stride_index,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RING: tl.constexpr,
    CAP: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    block = tl.load(state_indices + i_n * stride_index).to(tl.int64)
    slot = tl.load(slots + i_n).to(tl.int64)
    length = eos - bos
    if length <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_a = a + bos * stride_a + i_hv
    p_b = b + bos * stride_b + i_hv
    p_o = o + bos * stride_o + i_hv * V + o_v
    if (block <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    held_h = tl.load(p_h, mask=mask_h, other=0)
    next_q = tl.load(p_q, mask=mask_k, other=0)
    next_k = tl.load(p_k, mask=mask_k, other=0)
    next_v = tl.load(p_v, mask=mask_v, other=0)
    next_b = tl.load(p_b)
    next_a = tl.load(p_a)
    keep = tl.minimum(tl.maximum(tl.load(counts + slot), 0), CAP)
    head = tl.load(counts + RECORD_ROWS + slot) % RING
    ends = tl.load(counts + 2 * RECORD_ROWS + slot)
    flush = (keep + length) > CAP
    p_correction = correction + ((slot * HV + i_hv) * RING) * V + o_v
    p_key = keys + ((slot * H + i_h) * RING) * K + o_k
    p_gate = gates + (slot * HV + i_hv) * RING
    some = keep > 0
    f_g = tl.load(p_gate + head, mask=some, other=0)
    f_v = tl.load(p_correction + head * V, mask=mask_v & some, other=0)
    f_k = tl.load(p_key + head * K, mask=mask_k & some, other=0)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += held_h.to(tl.float32)
    for i_t in range(0, keep):
        position = (head + i_t) % RING
        following = (head + i_t + 1) % RING
        more = i_t + 1 < keep
        g_t = f_g
        v_t = f_v
        k_t = f_k
        f_g = tl.load(p_gate + following, mask=more, other=0)
        f_v = tl.load(p_correction + following * V, mask=mask_v & more, other=0)
        f_k = tl.load(p_key + following * K, mask=mask_k & more, other=0)
        b_h = b_h * tl.exp(g_t)
        b_h = tl.fma(v_t[:, None], k_t[None, :], b_h)
        if ((ends >> position) & 1) != 0:
            b_h = b_h.to(state.dtype.element_ty).to(tl.float32)
    if flush & (keep > 0):
        tl.store(p_h, b_h.to(state.dtype.element_ty), mask=mask_h)
    start = head + keep
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    if head_writer & (i_hv == 0):
        tl.store(handoff + slot, tl.where(flush, 0, keep))
        tl.store(handoff + RECORD_ROWS + slot, tl.where(flush, start % RING, head))
        tl.store(handoff + 2 * RECORD_ROWS + slot, tl.where(flush, 0, ends))
    a_log = tl.load(A_log + i_hv).to(tl.float32)
    bias_t = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, length):
        more = i_t + 1 < length
        b_q = next_q.to(tl.float32)
        b_k = next_k.to(tl.float32)
        b_v = next_v.to(tl.float32)
        b_b = next_b.to(tl.float32)
        x_g = next_a.to(tl.float32) + bias_t
        next_q = tl.load(p_q + stride_q, mask=mask_k & more, other=0)
        next_k = tl.load(p_k + stride_k, mask=mask_k & more, other=0)
        next_v = tl.load(p_v + stride_v, mask=mask_v & more, other=0)
        next_b = tl.load(p_b + stride_b, mask=more, other=0)
        next_a = tl.load(p_a + stride_a, mask=more, other=0)
        softplus_x = tl.where(
            softplus_beta * x_g <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x_g)),
            x_g,
        )
        b_g = -tl.exp(a_log) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h = b_h * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        recorded = i_t < (RING - CAP)
        position = (start + i_t) % RING
        tl.store(p_correction + position * V, b_v, mask=mask_v & recorded)
        tl.store(p_gate + position, b_g, mask=head_writer & recorded)
        tl.store(p_key + position * K, b_k, mask=mask_k & key_writer & recorded)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_a += stride_a
        p_b += stride_b
        p_o += stride_o


def run_lazy(layer, *, q, k, v, a, b, out, query_start_loc, state_indices, slots):
    q8_refuse("the lazy GDN ring")
    records = layer.gdn_accept_records
    state = ssm_full(layer)
    rows = state_indices.shape[0]
    tokens = q.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    k_heads = records.key.shape[1]
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError("GDN_FUSED needs contiguous state heads")
    if rows > records.rows:
        raise ValueError(f"GDN_FUSED runs {rows} rows but holds records for {records.rows}")
    if query_start_loc.shape[0] < rows + 1:
        raise ValueError("GDN_FUSED query metadata is shorter than its rows")
    if v_heads % k_heads:
        raise ValueError("GDN_FUSED needs whole value-head groups per key head")
    if slots is None or slots.shape[0] < rows:
        raise ValueError("GDN_FUSED lazy needs a request slot for every row")
    if any(t.shape[0] != tokens for t in (k, v, a, b)):
        raise ValueError("GDN_FUSED inputs disagree on the token count")
    if tokens == 0 or rows == 0:
        return out
    q2, k2, v2 = _rows_of(q, tokens), _rows_of(k, tokens), _rows_of(v, tokens)
    if q2.shape[1] != k_heads * k_dim or k2.shape[1] != k_heads * k_dim or v2.shape[1] != v_heads * v_dim:
        raise ValueError("GDN_FUSED q, k and v do not match the state's heads")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("GDN_FUSED needs contiguous gate heads")
    o2 = out.reshape(tokens, v_heads * v_dim)
    if o2.data_ptr() != out.data_ptr() or o2.stride(1) != 1:
        raise ValueError("GDN_FUSED needs a contiguous output row per token")
    block_v = min(triton.next_power_of_2(v_dim), TILED_BLOCK_V)
    grid = (triton.cdiv(v_dim, block_v), rows * v_heads)
    (_verify_lazy_ahead_kernel if verify_kernel() is _verify_ahead_kernel else _verify_lazy_kernel)[grid](
        layer.A_log, a, b, layer.dt_bias, q2, k2, v2, o2, state, query_start_loc, state_indices,
        slots, records.counts, records.handoff, records.correction, records.key, records.gate,
        k_dim**-0.5, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD, rows,
        q2.stride(0), k2.stride(0), v2.stride(0), a.stride(0), b.stride(0), o2.stride(0),
        state.stride(0), state_indices.stride(0),
        H=k_heads, HV=v_heads, K=k_dim, V=v_dim, BK=triton.next_power_of_2(k_dim), BV=block_v,
        RING=records.window, CAP=records.cap, RECORD_ROWS=records.rows,
        num_warps=TILED_WARPS, num_stages=TILED_STAGES, **verify_registers(),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


@dataclasses.dataclass
class Plan:
    slots: object = None
    commit: object = None

    def __call__(self, num_sampled):
        if self.commit is not None:
            self.commit(num_sampled)


def _rows_of(tensor, tokens):
    flat = tensor.reshape(tokens, -1)
    if flat.stride(1) != 1:
        raise ValueError("GDN_FUSED needs each token's channels contiguous")
    return flat


def run(layer, *, a, b, out, query_start_loc, state_indices, slots, x=None, q=None, k=None,
        v=None, output_gate=None, num_accepted=None, conv_state=None):
    q8_refuse("the fused GDN decode kernel")
    records = layer.gdn_accept_records
    state = ssm_full(layer)
    rows = state_indices.shape[0]
    tokens = a.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    k_heads = records.key.shape[1]
    defer = mode() == "defer"
    conv = x is not None
    norm = output_gate is not None
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError("GDN_FUSED needs contiguous state heads")
    if rows > records.rows:
        raise ValueError(f"GDN_FUSED runs {rows} rows but holds records for {records.rows}")
    if query_start_loc.shape[0] < rows + 1:
        raise ValueError("GDN_FUSED query metadata is shorter than its rows")
    if v_heads % k_heads:
        raise ValueError("GDN_FUSED needs whole value-head groups per key head")
    if defer and (slots is None or slots.shape[0] < rows):
        raise ValueError("GDN_FUSED defer needs a request slot for every row")
    if tokens == 0 or rows == 0:
        return out
    o2 = out.reshape(tokens, v_heads * v_dim)
    if o2.data_ptr() != out.data_ptr() or o2.stride(1) != 1:
        raise ValueError("GDN_FUSED needs a contiguous output row per token")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("GDN_FUSED needs contiguous gate heads")
    dummy = state
    if conv:
        weight = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
        if weight.shape[1] != 4:
            raise ValueError("GDN_FUSED folds a width-4 convolution only")
        if x.stride(1) != 1:
            raise ValueError("GDN_FUSED needs contiguous convolution channels")
        q2 = k2 = v2 = x
        conv_strides = conv_state.stride()
        w_strides = weight.stride()
        bias = layer.conv1d.bias
        product = PRODUCT_TYPES.get(torch.promote_types(conv_state.dtype, weight.dtype),
                                    tl.float32)
        if x.dtype != conv_state.dtype:
            raise ValueError("GDN_FUSED needs the projection in the convolution state's dtype")
    else:
        q2, k2, v2 = _rows_of(q, tokens), _rows_of(k, tokens), _rows_of(v, tokens)
        x, weight, bias, conv_state, num_accepted = q2, dummy, None, dummy, query_start_loc
        conv_strides = (0, 0, 0)
        w_strides = (0, 0)
        product = tl.float32
    if norm:
        norm_layer = layer.norm
        if (getattr(norm_layer, "bias", None) is not None or not norm_layer.norm_before_gate
                or norm_layer.group_size not in (None, v_dim)):
            raise ValueError("GDN_FUSED folds an RMS norm over one value head before its gate")
        activation = ACTIVATIONS[norm_layer.activation]
        z2 = output_gate.reshape(tokens, -1)
        if z2.stride(1) != 1:
            raise ValueError("GDN_FUSED needs contiguous gate channels")
        norm_weight = norm_layer.weight
        eps = float(norm_layer.eps)
        stride_z = output_gate.stride(0)
    else:
        activation, z2, norm_weight, eps, stride_z = 0, o2, dummy, 0.0, 0
    window = records.window
    _gdn_fused_kernel[(rows, k_heads)](
        x, q2, k2, v2, a, b, z2, o2,
        conv_state, weight, bias, num_accepted,
        layer.A_log, layer.dt_bias, norm_weight,
        state, query_start_loc, state_indices, slots if defer else state_indices,
        records.pending if defer else state_indices,
        records.correction, records.key, records.gate,
        k_dim**-0.5, eps, SOFTPLUS_BETA, SOFTPLUS_THRESHOLD, rows,
        x.stride(0), q2.stride(0), k2.stride(0), v2.stride(0), a.stride(0), b.stride(0), stride_z,
        o2.stride(0), conv_strides[0], conv_strides[1], conv_strides[2], w_strides[0], w_strides[1],
        state.stride(0), state_indices.stride(0),
        H=k_heads, HV=v_heads, K=k_dim, V=v_dim, BK=triton.next_power_of_2(k_dim),
        BV=triton.next_power_of_2(v_dim), BT=triton.next_power_of_2(window), WINDOW=window,
        RECORD_ROWS=records.rows, CONV=conv, HAS_BIAS=bias is not None,
        PRODUCT=product, NORM_ACT=activation,
        DEFER=defer, num_warps=FUSED_WARPS, num_stages=FUSED_STAGES,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


def _slots(attn_metadata, rows):
    plan = getattr(attn_metadata, "accept_commit", None)
    slots = getattr(plan, "slots", None)
    return None if slots is None else slots[:rows]


def decode(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    if mode() in RECURRENT:
        return decode_tiled(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata)
    rows = attn_metadata.num_spec_decodes
    tokens = attn_metadata.num_actual_tokens
    state_indices = attn_metadata.spec_state_indices_tensor
    query_start_loc = attn_metadata.spec_query_start_loc
    conv_state = (layer.kv_cache[0] if is_conv_state_dim_first()
                  else layer.kv_cache[0].transpose(-1, -2))
    run(layer, a=a[:tokens], b=b[:tokens], out=core_attn_out[:tokens],
        query_start_loc=query_start_loc[: rows + 1], state_indices=state_indices[:rows, 0],
        slots=_slots(attn_metadata, rows), x=mixed_qkv[:tokens], output_gate=output_gate[:tokens],
        num_accepted=attn_metadata.num_accepted_tokens[:rows], conv_state=conv_state)
    say("decode", f"GDN_FUSED decode path: {rows} row(s), {tokens} tokens through one fused "
                  f"kernel a GDN layer ({mode()})")


def decode_tiled(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    rows = attn_metadata.num_spec_decodes
    tokens = attn_metadata.num_actual_tokens
    state_indices = attn_metadata.spec_state_indices_tensor
    query_start_loc = attn_metadata.spec_query_start_loc
    conv_state = (layer.kv_cache[0] if is_conv_state_dim_first()
                  else layer.kv_cache[0].transpose(-1, -2))
    weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    mixed = causal_conv1d_update(
        mixed_qkv[:tokens],
        conv_state,
        weights,
        layer.conv1d.bias,
        layer.activation,
        conv_state_indices=state_indices[:rows, 0],
        num_accepted_tokens=attn_metadata.num_accepted_tokens[:rows],
        query_start_loc=query_start_loc[: rows + 1],
        max_query_len=layer.num_spec + 1,
        validate_data=False,
    )
    key_dim = layer.key_dim // layer.tp_size
    out = core_attn_out[:tokens]
    recurrence = run_lazy if mode() == "lazy" else run_tiled
    recurrence(layer, q=mixed[:, :key_dim], k=mixed[:, key_dim:2 * key_dim], v=mixed[:, 2 * key_dim:],
               a=a[:tokens], b=b[:tokens], out=out, query_start_loc=query_start_loc[: rows + 1],
               state_indices=state_indices[:rows, 0], slots=_slots(attn_metadata, rows),
               **({"decode": True} if recurrence is run_tiled else {}))
    layer._rms_norm_gated_cuda(out, output_gate[:tokens], out)
    say("decode_tiled", f"GDN_FUSED decode path ({mode()}): {rows} row(s), {tokens} tokens; stock convolution, "
                        f"value-tiled recurrence with the accepted prefix folded at the read, stock gated norm")


def verify_layer(layer, q, k, v, a, b, query_start_loc, state_indices, out=None):
    from vllm.forward_context import get_forward_context

    tokens = q.shape[0]
    _, v_heads, v_dim, _ = ssm_full(layer).shape
    if out is None:
        out = q.new_empty(1, tokens, v_heads, v_dim)
    metadata = get_forward_context().attn_metadata
    held = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
    rows = state_indices.shape[0]
    if mode() in RECURRENT:
        recurrence = run_lazy if mode() == "lazy" else run_tiled
        recurrence(layer, q=q, k=k, v=v, a=a, b=b, out=out, query_start_loc=query_start_loc,
                   state_indices=state_indices, slots=_slots(held, rows))
        say("mixed_tiled", f"GDN_FUSED mixed-batch path ({mode()}): {rows} speculative row(s) beside prefill "
                           f"chunks through the value-tiled recurrence")
        return out
    run(layer, a=a, b=b, out=out, query_start_loc=query_start_loc, state_indices=state_indices,
        slots=_slots(held, rows), q=q, k=k, v=v)
    say("mixed", f"GDN_FUSED mixed-batch path: {rows} speculative row(s) beside prefill chunks "
                 f"through the fused kernel ({mode()})")
    return out


@triton.heuristics({
    "HAS_REQUEST_INDICES": lambda args: args["request_indices"] is not None,
    "ALIGN": lambda args: args["block_table"] is not None,
    "UNIT": lambda args: args["unit_pairs"] is not None,
})
@triton.jit(do_not_specialize=["num_rows"])
def _gdn_defer_commit_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    pending_reference,
    pending_addrs,
    phase_reference,
    phase_addrs,
    stride_key_set,
    stride_gate_set,
    slots,
    source_blocks,
    stride_source,
    query_start_loc,
    num_sampled,
    request_indices,
    num_computed,
    block_table,
    stride_table,
    table_width,
    unit_pairs,
    unit_bounds,
    num_rows,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN: tl.constexpr,
    UNIT: tl.constexpr,
    TILED: tl.constexpr,
    PDL: tl.constexpr = False,
    Q8: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    row = tl.program_id(1)
    i_lh = tl.program_id(2)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    source = tl.load(source_blocks + row * stride_source).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if (source <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        return
    length = tl.load(query_start_loc + row + 1) - tl.load(query_start_loc + row)
    batch_row = row
    if HAS_REQUEST_INDICES:
        batch_row = tl.load(request_indices + row).to(tl.int32)
    accepted = tl.load(num_sampled + batch_row).to(tl.int32)
    accepted = tl.minimum(tl.maximum(accepted, 0), length)
    accepted = tl.minimum(accepted, WINDOW)
    pending = tl.load(pending_addrs + layer).to(tl.pointer_type(pending_reference.dtype.element_ty))
    marker = (i_v == 0) & (i_hv == 0)
    crossing = accepted * 0
    target = source
    checkpoint = accepted * 0 - 1
    if ALIGN:
        computed = tl.load(num_computed + batch_row).to(tl.int64)
        column = computed // BLOCK_SIZE
        reach = ((column + 1) * BLOCK_SIZE - computed).to(tl.int32)
        if (reach <= accepted) & (column < table_width):
            crossing = accepted * 0 + 1
            found = tl.load(block_table + batch_row * stride_table + column).to(tl.int64)
            if (found > 0) & (found != source):
                target = found
                checkpoint = reach
    unit_target = source
    unit_checkpoint = accepted * 0 - 1
    if UNIT:
        before = tl.load(num_computed + batch_row).to(tl.int64)
        bound = tl.load(unit_bounds + batch_row).to(tl.int64)
        spare = tl.load(unit_pairs + batch_row * 2 + 1).to(tl.int64)
        unit_reach = (bound - before).to(tl.int32)
        if (spare > 0) & (spare != source) & (unit_reach >= 1) & (unit_reach <= accepted):
            crossing = accepted * 0 + 1
            unit_target = spare
            unit_checkpoint = unit_reach
    if (accepted <= 0) | (crossing == 0):
        if marker:
            tl.store(pending + slot, accepted)
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
    block_stride = tl.load(state_strides + layer)
    offsets = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if Q8:
        b_h += _q8_tile(state, source * block_stride, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
    else:
        b_h += tl.load(state + source * block_stride + offsets, mask=mask_h, other=0).to(tl.float32)
    correction = tl.load(correction_addrs + layer).to(
        tl.pointer_type(correction_reference.dtype.element_ty))
    key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
    gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key = key + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate = gate + (slot * HV + i_hv) * WINDOW
    if TILED:
        phase = tl.load(phase_addrs + layer).to(tl.pointer_type(phase_reference.dtype.element_ty))
        fresh = 1 - tl.load(phase + slot).to(tl.int64)
        p_key += fresh * stride_key_set
        p_gate += fresh * stride_gate_set
    for i_t in range(0, accepted):
        b_g = tl.load(p_gate + i_t)
        b_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + i_t * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        if i_t + 1 == checkpoint:
            if Q8:
                _q8_put(state, target * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
            else:
                tl.store(state + target * block_stride + offsets,
                         b_h.to(state_reference.dtype.element_ty), mask=mask_h)
        if i_t + 1 == unit_checkpoint:
            if Q8:
                _q8_put(state, unit_target * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
            else:
                tl.store(state + unit_target * block_stride + offsets,
                         b_h.to(state_reference.dtype.element_ty), mask=mask_h)
    if Q8:
        _q8_put(state, source * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
    else:
        tl.store(state + source * block_stride + offsets, b_h.to(state_reference.dtype.element_ty),
                 mask=mask_h)
    if marker:
        tl.store(pending + slot, accepted * 0)


@triton.heuristics({"ALIGN": lambda args: args["state_idx"] is not None})
@triton.jit(do_not_specialize=["slot", "block", "resumed", "column"])
def _resume_fold_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    pending_reference,
    pending_addrs,
    phase_reference,
    phase_addrs,
    stride_key_set,
    stride_gate_set,
    num_computed,
    state_idx,
    slot,
    block,
    resumed,
    column,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    SPAN: tl.constexpr,
    ALIGN: tl.constexpr,
    TILED: tl.constexpr,
    Q8: tl.constexpr = False,
):
    i_v = tl.program_id(0)
    i_lh = tl.program_id(1)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    extra = tl.load(num_computed + slot) - resumed
    if ALIGN:
        source = tl.load(state_idx + slot) - column
        if (source < 0) | (source >= SPAN):
            return
    pending = tl.load(pending_addrs + layer).to(tl.pointer_type(pending_reference.dtype.element_ty))
    count = tl.minimum(tl.maximum(tl.load(pending + slot), 0), WINDOW) - extra
    if (extra < 0) | (count <= 0):
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
    block_stride = tl.load(state_strides + layer)
    p_h = state + block.to(tl.int64) * block_stride + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if Q8:
        b_h += _q8_tile(state, block.to(tl.int64) * block_stride, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
    else:
        b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    correction = tl.load(correction_addrs + layer).to(
        tl.pointer_type(correction_reference.dtype.element_ty))
    key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
    gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
    p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
    p_key = key + ((slot * H + i_h) * WINDOW) * K + o_k
    p_gate = gate + (slot * HV + i_hv) * WINDOW
    if TILED:
        phase = tl.load(phase_addrs + layer).to(tl.pointer_type(phase_reference.dtype.element_ty))
        side = tl.load(phase + slot).to(tl.int64)
        p_key += side * stride_key_set
        p_gate += side * stride_gate_set
    for i_t in range(0, count):
        b_g = tl.load(p_gate + i_t)
        b_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + i_t * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
    if Q8:
        _q8_put(state, block.to(tl.int64) * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
    else:
        tl.store(p_h, b_h.to(state_reference.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    "HAS_REQUEST_INDICES": lambda args: args["request_indices"] is not None,
    "ALIGN": lambda args: args["block_table"] is not None,
    "UNIT": lambda args: args["unit_pairs"] is not None,
})
@triton.jit(do_not_specialize=["num_rows"])
def _gdn_defer_gate_kernel(
    pending_reference,
    pending_addrs,
    fold_count,
    fold_records,
    slots,
    source_blocks,
    stride_source,
    query_start_loc,
    num_sampled,
    request_indices,
    num_computed,
    block_table,
    stride_table,
    table_width,
    unit_pairs,
    unit_bounds,
    num_rows,
    WINDOW: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYERS: tl.constexpr,
    LAYER_BLOCK: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN: tl.constexpr,
    UNIT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    source = tl.load(source_blocks + row * stride_source).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    length = tl.load(query_start_loc + row + 1) - tl.load(query_start_loc + row)
    batch_row = row
    if HAS_REQUEST_INDICES:
        batch_row = tl.load(request_indices + row).to(tl.int32)
    accepted = tl.load(num_sampled + batch_row).to(tl.int32)
    accepted = tl.minimum(tl.maximum(accepted, 0), length)
    accepted = tl.minimum(accepted, WINDOW)
    crossing = accepted * 0
    target = source
    checkpoint = accepted * 0 - 1
    if ALIGN:
        computed = tl.load(num_computed + batch_row).to(tl.int64)
        column = computed // BLOCK_SIZE
        reach = ((column + 1) * BLOCK_SIZE - computed).to(tl.int32)
        if (reach <= accepted) & (column < table_width):
            crossing = accepted * 0 + 1
            found = tl.load(block_table + batch_row * stride_table + column).to(tl.int64)
            if (found > 0) & (found != source):
                target = found
                checkpoint = reach
    unit_target = source
    unit_checkpoint = accepted * 0 - 1
    if UNIT:
        before = tl.load(num_computed + batch_row).to(tl.int64)
        bound = tl.load(unit_bounds + batch_row).to(tl.int64)
        spare = tl.load(unit_pairs + batch_row * 2 + 1).to(tl.int64)
        unit_reach = (bound - before).to(tl.int32)
        if (spare > 0) & (spare != source) & (unit_reach >= 1) & (unit_reach <= accepted):
            crossing = accepted * 0 + 1
            unit_target = spare
            unit_checkpoint = unit_reach
    if (source > 0) & (slot >= 0) & (slot < RECORD_ROWS):
        if (accepted <= 0) | (crossing == 0):
            o_layer = tl.arange(0, LAYER_BLOCK)
            mask_layer = o_layer < LAYERS
            pending = tl.load(pending_addrs + o_layer, mask=mask_layer, other=0).to(
                tl.pointer_type(pending_reference.dtype.element_ty))
            tl.store(pending + slot, accepted, mask=mask_layer)
        else:
            at = tl.atomic_add(fold_count, 1)
            record = fold_records + at * 8
            tl.store(record, source)
            tl.store(record + 1, slot)
            tl.store(record + 2, accepted.to(tl.int64))
            tl.store(record + 3, target)
            tl.store(record + 4, checkpoint.to(tl.int64))
            tl.store(record + 5, unit_target)
            tl.store(record + 6, unit_checkpoint.to(tl.int64))


@triton.jit
def _gdn_defer_fold_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    pending_reference,
    pending_addrs,
    phase_reference,
    phase_addrs,
    stride_key_set,
    stride_gate_set,
    fold_count,
    fold_records,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW: tl.constexpr,
    ALIGN: tl.constexpr,
    UNIT: tl.constexpr,
    TILED: tl.constexpr,
    PDL: tl.constexpr = False,
    Q8: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    i_lh = tl.program_id(2)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    marker = (i_v == 0) & (i_hv == 0)
    for at in range(tl.program_id(1), tl.load(fold_count), tl.num_programs(1)):
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]
        offsets = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        record = fold_records + at * 8
        source = tl.load(record)
        slot = tl.load(record + 1)
        accepted = tl.load(record + 2).to(tl.int32)
        state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
        block_stride = tl.load(state_strides + layer)
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        if Q8:
            b_h += _q8_tile(state, source * block_stride, i_hv, o_v, o_k, mask_h, mask_v, HV, V, K)
        else:
            b_h += tl.load(state + source * block_stride + offsets, mask=mask_h, other=0).to(tl.float32)
        correction = tl.load(correction_addrs + layer).to(tl.pointer_type(correction_reference.dtype.element_ty))
        key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
        gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
        p_correction = correction + ((slot * HV + i_hv) * WINDOW) * V + o_v
        p_key = key + ((slot * H + i_h) * WINDOW) * K + o_k
        p_gate = gate + (slot * HV + i_hv) * WINDOW
        if TILED:
            phase = tl.load(phase_addrs + layer).to(tl.pointer_type(phase_reference.dtype.element_ty))
            fresh = 1 - tl.load(phase + slot).to(tl.int64)
            p_key += fresh * stride_key_set
            p_gate += fresh * stride_gate_set
        if ALIGN:
            target = tl.load(record + 3)
            checkpoint = tl.load(record + 4).to(tl.int32)
        if UNIT:
            unit_target = tl.load(record + 5)
            unit_checkpoint = tl.load(record + 6).to(tl.int32)
        for i_t in range(0, accepted):
            b_g = tl.load(p_gate + i_t)
            b_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
            b_k = tl.load(p_key + i_t * K, mask=mask_k, other=0)
            b_h = b_h * tl.exp(b_g)
            b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
            if ALIGN:
                if i_t + 1 == checkpoint:
                    if Q8:
                        _q8_put(state, target * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
                    else:
                        tl.store(state + target * block_stride + offsets,
                                 b_h.to(state_reference.dtype.element_ty), mask=mask_h)
            if UNIT:
                if i_t + 1 == unit_checkpoint:
                    if Q8:
                        _q8_put(state, unit_target * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
                    else:
                        tl.store(state + unit_target * block_stride + offsets,
                                 b_h.to(state_reference.dtype.element_ty), mask=mask_h)
        if Q8:
            _q8_put(state, source * block_stride, i_hv, o_v, o_k, mask_h, mask_v, b_h, HV, V, K)
        else:
            tl.store(state + source * block_stride + offsets, b_h.to(state_reference.dtype.element_ty),
                     mask=mask_h)
        if marker:
            pending = tl.load(pending_addrs + layer).to(tl.pointer_type(pending_reference.dtype.element_ty))
            tl.store(pending + slot, accepted * 0)


@triton.jit
def _flip_kernel(phase_reference, phase_addrs, slots, source_blocks, stride_source, RECORD_ROWS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    layer = tl.program_id(1)
    source = tl.load(source_blocks + row * stride_source).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if (source <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        return
    phase = tl.load(phase_addrs + layer).to(tl.pointer_type(phase_reference.dtype.element_ty))
    tl.store(phase + slot, 1 - tl.load(phase + slot))


def _addresses(tensors, device):
    return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device=device)


@dataclasses.dataclass
class DeferContext:
    base: object
    pending_addrs: torch.Tensor
    pending_reference: torch.Tensor
    phase_addrs: torch.Tensor = None
    phase_reference: torch.Tensor = None
    fold_count: torch.Tensor = None
    fold_records: torch.Tensor = None

    @classmethod
    def of(cls, base):
        pendings = [record.pending for record in base.records]
        phases = [getattr(record, "phase", None) for record in base.records]
        tiled = all(phase is not None for phase in phases)
        return cls(base=base, pending_addrs=_addresses(pendings, pendings[0].device),
                   pending_reference=pendings[0],
                   phase_addrs=_addresses(phases, pendings[0].device) if tiled else None,
                   phase_reference=phases[0] if tiled else None)

    def commit(self, num_sampled, *, slots, source_blocks, query_start_loc, request_indices=None,
               num_computed=None, block_table=None, block_size=0, unit_pairs=None,
               unit_bounds=None):
        rows = source_blocks.shape[0]
        if isinstance(num_sampled, int) or rows == 0:
            return
        base = self.base
        held = base.records[0]
        if rows > held.rows:
            raise ValueError(f"GDN_FUSED commits {rows} rows but holds records for {held.rows}")
        if slots is None or slots.shape[0] < rows:
            raise ValueError("GDN_FUSED defer commit needs a request slot for every row")
        if block_table is not None and (num_computed is None or block_size <= 0
                                        or block_table.stride(1) != 1):
            raise ValueError("GDN_FUSED align commit needs computed counts, a block size and "
                             "contiguous block-table rows")
        if unit_pairs is not None and (num_computed is None or unit_bounds is None
                                       or not unit_pairs.is_contiguous()):
            raise ValueError("GDN_FUSED reply checkpoints need computed counts, boundaries and "
                             "contiguous running/spare pairs")
        first = base.states[0]
        _, v_heads, v_dim, k_dim = first.shape
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        grid = (triton.cdiv(v_dim, block_v), rows, len(base.states) * v_heads)
        tiled = self.phase_addrs is not None
        cap = commit_cap()
        arguments = (
            first, base.state_addrs, base.state_strides,
            held.correction, base.correction_addrs,
            held.key, base.key_addrs,
            held.gate, base.gate_addrs,
            self.pending_reference, self.pending_addrs,
            self.phase_reference if tiled else self.pending_reference,
            self.phase_addrs if tiled else self.pending_addrs,
            held.key_sets.stride(0) if tiled else 0, held.gate_sets.stride(0) if tiled else 0,
            slots,
            source_blocks, source_blocks.stride(0), query_start_loc, num_sampled,
            request_indices, num_computed, block_table,
            0 if block_table is None else block_table.stride(0),
            0 if block_table is None else block_table.shape[1],
            unit_pairs, unit_bounds,
            rows,
        )
        shape = dict(H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
                     BK=triton.next_power_of_2(k_dim), BV=block_v, WINDOW=held.window, TILED=tiled,
                     num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES, Q8=q8())
        if cap:
            layers = len(base.states)
            if self.fold_count is None:
                self.fold_count = torch.zeros(1, dtype=torch.int32, device=slots.device)
                self.fold_records = torch.zeros((held.rows, 8), dtype=torch.int64, device=slots.device)
            self.fold_count.zero_()
            _gdn_defer_gate_kernel[(rows,)](
                self.pending_reference, self.pending_addrs, self.fold_count, self.fold_records,
                *arguments[15:],
                WINDOW=held.window, RECORD_ROWS=held.rows, BLOCK_SIZE=max(int(block_size), 1),
                LAYERS=layers, LAYER_BLOCK=triton.next_power_of_2(layers),
                num_warps=1, PDL=pdl_on(), launch_pdl=pdl_on(),
            )
            _gdn_defer_fold_kernel[(grid[0], cap, grid[2])](
                *arguments[:15], self.fold_count, self.fold_records,
                ALIGN=block_table is not None, UNIT=unit_pairs is not None,
                PDL=pdl_on(), launch_pdl=pdl_on(), **shape)
        else:
            _gdn_defer_commit_kernel[grid](
                *arguments, RECORD_ROWS=held.rows, BLOCK_SIZE=max(int(block_size), 1),
                PDL=pdl_on(), launch_pdl=pdl_on(), **shape)
        if tiled:
            _flip_kernel[(rows, len(base.states))](
                self.phase_reference, self.phase_addrs, slots, source_blocks, source_blocks.stride(0),
                RECORD_ROWS=held.rows, PDL=pdl_on(), launch_pdl=pdl_on())
        say("commit", f"GDN_FUSED first deferred commit: {rows} request(s) x "
                      f"{len(base.states)} GDN layers; a row that crosses a cache boundary is "
                      f"folded now, every other row's accepted prefix waits for the next read")
        if cap:
            say("commit_cap", f"{COMMIT_CAP_ENV}={cap}: the deferred commit decides every row in one program "
                              f"each, writes the waiting counts of rows that fold nothing, and folds the rows "
                              f"that cross a boundary through a grid of {cap} row slot(s)")

    def resume(self, slot, block, resumed, num_computed, state_idx=None, column=0):
        base = self.base
        held = base.records[0]
        if not 0 <= slot < held.rows:
            raise ValueError(f"GDN_FUSED resumes slot {slot} but holds records for {held.rows}")
        first = base.states[0]
        _, v_heads, v_dim, k_dim = first.shape
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        tiled = self.phase_addrs is not None
        _resume_fold_kernel[(triton.cdiv(v_dim, block_v), len(base.states) * v_heads)](
            first, base.state_addrs, base.state_strides,
            held.correction, base.correction_addrs,
            held.key, base.key_addrs,
            held.gate, base.gate_addrs,
            self.pending_reference, self.pending_addrs,
            self.phase_reference if tiled else self.pending_reference,
            self.phase_addrs if tiled else self.pending_addrs,
            held.key_sets.stride(0) if tiled else 0, held.gate_sets.stride(0) if tiled else 0,
            num_computed, state_idx, int(slot), int(block), int(resumed), int(column),
            H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
            BK=triton.next_power_of_2(k_dim), BV=block_v, WINDOW=held.window, SPAN=RESUME_SPAN,
            TILED=tiled, num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES, Q8=q8(),
        )


@triton.heuristics({
    "HAS_REQUEST_INDICES": lambda args: args["request_indices"] is not None,
    "ALIGN": lambda args: args["block_table"] is not None,
    "UNIT": lambda args: args["unit_pairs"] is not None,
})
@triton.jit(do_not_specialize=["num_rows"])
def _gdn_lazy_commit_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    count_reference,
    count_addrs,
    handoff_addrs,
    stats,
    slots,
    source_blocks,
    stride_source,
    query_start_loc,
    num_sampled,
    request_indices,
    num_computed,
    block_table,
    stride_table,
    table_width,
    unit_pairs,
    unit_bounds,
    num_rows,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RING: tl.constexpr,
    CAP: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN: tl.constexpr,
    UNIT: tl.constexpr,
    COUNT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_v = tl.program_id(0)
    row = tl.program_id(1)
    i_lh = tl.program_id(2)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    source = tl.load(source_blocks + row * stride_source).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    if (source <= 0) | (slot < 0) | (slot >= RECORD_ROWS):
        return
    length = tl.load(query_start_loc + row + 1) - tl.load(query_start_loc + row)
    if length <= 0:
        return
    batch_row = row
    if HAS_REQUEST_INDICES:
        batch_row = tl.load(request_indices + row).to(tl.int32)
    accepted = tl.load(num_sampled + batch_row).to(tl.int32)
    accepted = tl.minimum(tl.maximum(accepted, 0), length)
    accepted = tl.minimum(accepted, RING - CAP)
    marker = (i_v == 0) & (i_hv == 0)
    crossing = accepted * 0
    target = source
    checkpoint = accepted * 0 - 1
    if ALIGN:
        computed = tl.load(num_computed + batch_row).to(tl.int64)
        column = computed // BLOCK_SIZE
        reach = ((column + 1) * BLOCK_SIZE - computed).to(tl.int32)
        if (reach <= accepted) & (column < table_width):
            crossing = accepted * 0 + 1
            found = tl.load(block_table + batch_row * stride_table + column).to(tl.int64)
            if (found > 0) & (found != source):
                target = found
                checkpoint = reach
    unit_target = source
    unit_checkpoint = accepted * 0 - 1
    if UNIT:
        before = tl.load(num_computed + batch_row).to(tl.int64)
        bound = tl.load(unit_bounds + batch_row).to(tl.int64)
        spare = tl.load(unit_pairs + batch_row * 2 + 1).to(tl.int64)
        unit_reach = (bound - before).to(tl.int32)
        if (spare > 0) & (spare != source) & (unit_reach >= 1) & (unit_reach <= accepted):
            crossing = accepted * 0 + 1
            unit_target = spare
            unit_checkpoint = unit_reach
    folding = (accepted > 0) & (crossing != 0)
    if (folding == 0) & (marker == 0):
        return
    count_type = count_reference.dtype.element_ty
    counts = tl.load(count_addrs + layer).to(tl.pointer_type(count_type))
    handoff = tl.load(handoff_addrs + layer).to(tl.pointer_type(count_type))
    keep = tl.minimum(tl.maximum(tl.load(handoff + slot), 0), CAP)
    head = tl.load(handoff + RECORD_ROWS + slot)
    ends = tl.load(handoff + 2 * RECORD_ROWS + slot)
    if COUNT:
        if marker & (layer == 0):
            held = tl.load(counts + slot)
            last_tail = tl.load(counts + 3 * RECORD_ROWS + slot)
            tl.atomic_add(stats, 1)
            tl.atomic_add(stats + 1, ((held > 0) & (keep == 0)).to(tl.int64))
            tl.atomic_add(stats + 2, folding.to(tl.int64))
            tl.atomic_add(stats + 3, (last_tail > 0).to(tl.int64))
            tl.atomic_add(stats + 4, tl.where(folding, 0, keep + accepted).to(tl.int64))
    if folding == 0:
        last = ((head + keep + accepted - 1) % RING).to(tl.int32)
        marks = tl.where(accepted > 0, ends | (1 << last), ends)
        tl.store(counts + slot, keep + accepted)
        tl.store(counts + RECORD_ROWS + slot, head)
        tl.store(counts + 2 * RECORD_ROWS + slot, marks)
        tl.store(counts + 3 * RECORD_ROWS + slot, accepted)
        tl.store(counts + 4 * RECORD_ROWS + slot, source.to(tl.int32))
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
    block_stride = tl.load(state_strides + layer)
    offsets = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(state + source * block_stride + offsets, mask=mask_h, other=0).to(tl.float32)
    correction = tl.load(correction_addrs + layer).to(
        tl.pointer_type(correction_reference.dtype.element_ty))
    key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
    gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
    p_correction = correction + ((slot * HV + i_hv) * RING) * V + o_v
    p_key = key + ((slot * H + i_h) * RING) * K + o_k
    p_gate = gate + (slot * HV + i_hv) * RING
    for i_t in range(0, keep):
        position = (head + i_t) % RING
        b_g = tl.load(p_gate + position)
        b_v = tl.load(p_correction + position * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + position * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        if ((ends >> position) & 1) != 0:
            b_h = b_h.to(state_reference.dtype.element_ty).to(tl.float32)
    start = head + keep
    for i_t in range(0, accepted):
        position = (start + i_t) % RING
        b_g = tl.load(p_gate + position)
        b_v = tl.load(p_correction + position * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + position * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        if i_t + 1 == checkpoint:
            tl.store(state + target * block_stride + offsets,
                     b_h.to(state_reference.dtype.element_ty), mask=mask_h)
        if i_t + 1 == unit_checkpoint:
            tl.store(state + unit_target * block_stride + offsets,
                     b_h.to(state_reference.dtype.element_ty), mask=mask_h)
    tl.store(state + source * block_stride + offsets, b_h.to(state_reference.dtype.element_ty),
             mask=mask_h)
    if marker:
        tl.store(counts + slot, keep * 0)
        tl.store(counts + RECORD_ROWS + slot, (head + keep + accepted) % RING)
        tl.store(counts + 2 * RECORD_ROWS + slot, ends * 0)
        tl.store(counts + 3 * RECORD_ROWS + slot, keep * 0)
        tl.store(counts + 4 * RECORD_ROWS + slot, source.to(tl.int32))


@triton.jit(do_not_specialize=["slot"])
def _lazy_leave_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    count_reference,
    count_addrs,
    slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RING: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_lh = tl.program_id(1)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    counts = tl.load(count_addrs + layer).to(tl.pointer_type(count_reference.dtype.element_ty))
    keep = tl.load(counts + slot)
    head = tl.load(counts + RECORD_ROWS + slot) % RING
    ends = tl.load(counts + 2 * RECORD_ROWS + slot)
    tail = tl.load(counts + 3 * RECORD_ROWS + slot)
    block = tl.load(counts + 4 * RECORD_ROWS + slot).to(tl.int64)
    prior = keep - tail
    if (prior <= 0) | (tail < 0) | (keep > RING) | (block <= 0):
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
    block_stride = tl.load(state_strides + layer)
    p_h = state + block * block_stride + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    correction = tl.load(correction_addrs + layer).to(
        tl.pointer_type(correction_reference.dtype.element_ty))
    key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
    gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
    p_correction = correction + ((slot * HV + i_hv) * RING) * V + o_v
    p_key = key + ((slot * H + i_h) * RING) * K + o_k
    p_gate = gate + (slot * HV + i_hv) * RING
    for i_t in range(0, prior):
        position = (head + i_t) % RING
        b_g = tl.load(p_gate + position)
        b_v = tl.load(p_correction + position * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + position * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
        if ((ends >> position) & 1) != 0:
            b_h = b_h.to(state_reference.dtype.element_ty).to(tl.float32)
    tl.store(p_h, b_h.to(state_reference.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["slot"])
def _lazy_leave_marks_kernel(count_reference, count_addrs, slot, RING: tl.constexpr, RECORD_ROWS: tl.constexpr):
    layer = tl.program_id(0)
    counts = tl.load(count_addrs + layer).to(tl.pointer_type(count_reference.dtype.element_ty))
    keep = tl.load(counts + slot)
    head = tl.load(counts + RECORD_ROWS + slot) % RING
    tail = tl.load(counts + 3 * RECORD_ROWS + slot)
    block = tl.load(counts + 4 * RECORD_ROWS + slot)
    prior = keep - tail
    if (prior <= 0) | (tail < 0) | (keep > RING) | (block <= 0):
        return
    last = ((head + keep - 1) % RING).to(tl.int32)
    tl.store(counts + slot, tail)
    tl.store(counts + RECORD_ROWS + slot, (head + prior) % RING)
    tl.store(counts + 2 * RECORD_ROWS + slot, tl.where(tail > 0, 1 << last, 0))


@triton.heuristics({"ALIGN": lambda args: args["state_idx"] is not None})
@triton.jit(do_not_specialize=["slot", "block", "resumed", "column"])
def _lazy_resume_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
    count_reference,
    count_addrs,
    num_computed,
    state_idx,
    slot,
    block,
    resumed,
    column,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RING: tl.constexpr,
    CAP: tl.constexpr,
    RECORD_ROWS: tl.constexpr,
    SPAN: tl.constexpr,
    ALIGN: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_lh = tl.program_id(1)
    layer = i_lh // HV
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    extra = tl.load(num_computed + slot) - resumed
    if ALIGN:
        source = tl.load(state_idx + slot) - column
        if (source < 0) | (source >= SPAN):
            return
    counts = tl.load(count_addrs + layer).to(tl.pointer_type(count_reference.dtype.element_ty))
    keep = tl.load(counts + slot)
    head = tl.load(counts + RECORD_ROWS + slot) % RING
    tail = tl.load(counts + 3 * RECORD_ROWS + slot)
    count = tl.minimum(tl.maximum(tail, 0), RING - CAP) - extra
    if (extra < 0) | (count <= 0):
        return
    start = head + tl.maximum(keep - tail, 0)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state = tl.load(state_addrs + layer).to(tl.pointer_type(state_reference.dtype.element_ty))
    block_stride = tl.load(state_strides + layer)
    p_h = state + block.to(tl.int64) * block_stride + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    correction = tl.load(correction_addrs + layer).to(
        tl.pointer_type(correction_reference.dtype.element_ty))
    key = tl.load(key_addrs + layer).to(tl.pointer_type(key_reference.dtype.element_ty))
    gate = tl.load(gate_addrs + layer).to(tl.pointer_type(gate_reference.dtype.element_ty))
    p_correction = correction + ((slot * HV + i_hv) * RING) * V + o_v
    p_key = key + ((slot * H + i_h) * RING) * K + o_k
    p_gate = gate + (slot * HV + i_hv) * RING
    for i_t in range(0, count):
        position = (start + i_t) % RING
        b_g = tl.load(p_gate + position)
        b_v = tl.load(p_correction + position * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + position * K, mask=mask_k, other=0)
        b_h = b_h * tl.exp(b_g)
        b_h = tl.fma(b_v[:, None], b_k[None, :], b_h)
    tl.store(p_h, b_h.to(state_reference.dtype.element_ty), mask=mask_h)


def _lazy_counter(device):
    key = str(device)
    held = _LAZY["stats"].get(key)
    if held is not None:
        return held
    held = torch.zeros(len(LAZY_STATS), dtype=torch.int64, device=device)
    _LAZY["stats"][key] = held
    timers = sys.modules.get(TIMERS_MODULE)
    registered = callable(getattr(timers, "counter", None))
    if registered:
        timers.counter(LAZY_COUNTER, LAZY_STATS, held)
    say("lazy_counter", f"GDN_LAZY counter: the commit of the group holding the first lazy GDN layer counts "
                        f"{', '.join(LAZY_STATS)} on the GPU; "
                        + (f"STEP_TIMERS counters lines print them per step as {LAZY_COUNTER}" if registered
                           else "no step timers here, so nothing prints them"))
    return held


@dataclasses.dataclass
class LazyContext:
    base: object
    count_addrs: torch.Tensor
    handoff_addrs: torch.Tensor
    count_reference: torch.Tensor
    stats: torch.Tensor
    count: bool = True

    @classmethod
    def of(cls, base, stats=None):
        records = base.records
        device = records[0].counts.device
        first = _LAZY["counts"][:1]
        count = stats is not None or any(record.counts is held for record in records for held in first)
        return cls(base=base,
                   count_addrs=_addresses([record.counts for record in records], device),
                   handoff_addrs=_addresses([record.handoff for record in records], device),
                   count_reference=records[0].counts,
                   stats=stats if stats is not None else _lazy_counter(device) if count else records[0].counts,
                   count=count)

    def _shape(self):
        held = self.base.records[0]
        _, v_heads, v_dim, k_dim = self.base.states[0].shape
        return held, v_heads, v_dim, k_dim

    def commit(self, num_sampled, *, slots, source_blocks, query_start_loc, request_indices=None,
               num_computed=None, block_table=None, block_size=0, unit_pairs=None,
               unit_bounds=None):
        rows = source_blocks.shape[0]
        if isinstance(num_sampled, int) or rows == 0:
            return
        held, v_heads, v_dim, k_dim = self._shape()
        if rows > held.rows:
            raise ValueError(f"GDN_FUSED commits {rows} rows but holds records for {held.rows}")
        if slots is None or slots.shape[0] < rows:
            raise ValueError("GDN_FUSED lazy commit needs a request slot for every row")
        if block_table is not None and (num_computed is None or block_size <= 0
                                        or block_table.stride(1) != 1):
            raise ValueError("GDN_FUSED align commit needs computed counts, a block size and "
                             "contiguous block-table rows")
        if unit_pairs is not None and (num_computed is None or unit_bounds is None
                                       or not unit_pairs.is_contiguous()):
            raise ValueError("GDN_FUSED reply checkpoints need computed counts, boundaries and "
                             "contiguous running/spare pairs")
        base = self.base
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        grid = (triton.cdiv(v_dim, block_v), rows, len(base.states) * v_heads)
        _gdn_lazy_commit_kernel[grid](
            base.states[0], base.state_addrs, base.state_strides,
            held.correction, base.correction_addrs,
            held.key, base.key_addrs,
            held.gate, base.gate_addrs,
            self.count_reference, self.count_addrs, self.handoff_addrs, self.stats,
            slots,
            source_blocks, source_blocks.stride(0), query_start_loc, num_sampled,
            request_indices, num_computed, block_table,
            0 if block_table is None else block_table.stride(0),
            0 if block_table is None else block_table.shape[1],
            unit_pairs, unit_bounds,
            rows,
            H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
            BK=triton.next_power_of_2(k_dim), BV=block_v, RING=held.window, CAP=held.cap,
            RECORD_ROWS=held.rows, BLOCK_SIZE=max(int(block_size), 1), COUNT=self.count,
            num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES,
            PDL=pdl_on(), launch_pdl=pdl_on(),
        )
        say("lazy_commit", f"GDN_LAZY first commit: {rows} request(s) x {len(base.states)} GDN layers; a row "
                           f"that crosses a cache or reply boundary folds its whole ring and writes its state now, "
                           f"every other row's accepted tokens join its ring")

    def leave(self, slot):
        held, v_heads, v_dim, k_dim = self._shape()
        if not 0 <= slot < held.rows:
            raise ValueError(f"GDN_FUSED leaves slot {slot} but holds records for {held.rows}")
        base = self.base
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        _lazy_leave_kernel[(triton.cdiv(v_dim, block_v), len(base.states) * v_heads)](
            base.states[0], base.state_addrs, base.state_strides,
            held.correction, base.correction_addrs,
            held.key, base.key_addrs,
            held.gate, base.gate_addrs,
            self.count_reference, self.count_addrs, int(slot),
            H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
            BK=triton.next_power_of_2(k_dim), BV=block_v, RING=held.window, RECORD_ROWS=held.rows,
            num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES,
        )
        _lazy_leave_marks_kernel[(len(base.states),)](
            self.count_reference, self.count_addrs, int(slot), RING=held.window, RECORD_ROWS=held.rows)

    def resume(self, slot, block, resumed, num_computed, state_idx=None, column=0):
        held, v_heads, v_dim, k_dim = self._shape()
        if not 0 <= slot < held.rows:
            raise ValueError(f"GDN_FUSED resumes slot {slot} but holds records for {held.rows}")
        base = self.base
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        _lazy_resume_kernel[(triton.cdiv(v_dim, block_v), len(base.states) * v_heads)](
            base.states[0], base.state_addrs, base.state_strides,
            held.correction, base.correction_addrs,
            held.key, base.key_addrs,
            held.gate, base.gate_addrs,
            self.count_reference, self.count_addrs,
            num_computed, state_idx, int(slot), int(block), int(resumed), int(column),
            H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
            BK=triton.next_power_of_2(k_dim), BV=block_v, RING=held.window, CAP=held.cap,
            RECORD_ROWS=held.rows, SPAN=RESUME_SPAN, num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES,
        )


def nosync():
    return str(os.environ.get(NOSYNC_ENV) or "").strip().lower() in ("1", "on", "true")


def prep_shared():
    return str(os.environ.get(PREP_ENV) or "").strip().lower() in ("1", "on", "true")


def _slot_buffer(builder, size, device):
    held = getattr(builder, "taaf_slot_buffer", None)
    if held is None:
        rows = max(size, int(builder.vllm_config.scheduler_config.max_num_seqs),
                   int(getattr(builder, "decode_cudagraph_max_bs", 0) or 0))
        if prep_shared():
            held = _SHARED_SLOTS.get((str(device), rows))
            if held is None:
                held = torch.full((rows,), -1, dtype=torch.int32, device=device)
                _SHARED_SLOTS[(str(device), rows)] = held
        else:
            held = torch.full((rows,), -1, dtype=torch.int32, device=device)
        builder.taaf_slot_buffer = held
    if held.shape[0] < size:
        raise ValueError(f"GDN_FUSED slot buffer holds {held.shape[0]} rows, the step has {size}")
    return held


def fill_slots(target, mapping, request_indices):
    rows = target.shape[0]
    if (nosync() and mapping.dtype == target.dtype
            and (request_indices is None or request_indices.shape == (rows,)
                 and request_indices.dtype in (torch.int32, torch.int64))):
        if request_indices is None:
            target.copy_(mapping[:rows])
        else:
            torch.index_select(mapping, 0, request_indices, out=target)
        return
    picked = (mapping[:rows] if request_indices is None
              else mapping.index_select(0, request_indices.to(torch.int64)))
    target.copy_(picked.to(torch.int32), non_blocking=True)


def commit_for(stock, builder, common, state_indices, query_start_loc, rows, request_indices):
    plan = Plan(commit=stock(builder, common, state_indices, query_start_loc, rows,
                             request_indices))
    if mode() not in DEFERRED:
        return plan
    total = state_indices.shape[0]
    buffer = _slot_buffer(builder, total, state_indices.device)
    mapping = getattr(builder, "taaf_slots", None)
    builder.taaf_slots = None
    slots = buffer[:total]
    memo = getattr(builder, "taaf_prep_memo", None)
    token = None
    if memo is not None and prep_shared():
        token = ("slots", id(buffer), None if mapping is None else id(mapping), rows, total,
                 None if request_indices is None else id(request_indices))
    if token is not None and memo.get(token):
        pass
    elif mapping is None:
        slots.fill_(-1)
        if token is not None:
            memo[token] = True
    else:
        fill_slots(slots[:rows], mapping, request_indices)
        if total > rows:
            slots[rows:].fill_(-1)
        if token is not None:
            memo[token] = True
    plan.slots = slots
    base = plan.commit.func.__self__
    context = getattr(builder, "taaf_defer_context", None)
    kind = LazyContext if mode() == "lazy" else DeferContext
    if context is None or context.base is not base or not isinstance(context, kind):
        context = kind.of(base)
        builder.taaf_defer_context = context
    plan.commit = functools.partial(context.commit, slots=slots[:rows], **plan.commit.keywords)
    return plan


class LazyCheckFailed(Exception):
    pass


def _check_group(layers):
    states = [ssm_full(layer) for layer in layers]
    records = [layer.gdn_accept_records for layer in layers]
    device = states[0].device
    return types.SimpleNamespace(
        states=tuple(states), records=tuple(records), state_addrs=_addresses(states, device),
        state_strides=torch.tensor([state.stride(0) for state in states], dtype=torch.int64, device=device),
        correction_addrs=_addresses([record.correction for record in records], device),
        key_addrs=_addresses([record.key for record in records], device),
        gate_addrs=_addresses([record.gate for record in records], device))


def _check_reset(pairs, slot):
    for tiled, lazy in pairs:
        tiled.gdn_accept_records.pending[slot] = 0
        lazy.gdn_accept_records.counts[:, slot] = 0


def _check_equal(pairs, blocks, what):
    for index, (tiled, lazy) in enumerate(pairs):
        if blocks is None:
            same = torch.equal(tiled.kv_cache[1], lazy.kv_cache[1])
        else:
            same = torch.equal(tiled.kv_cache[1][blocks], lazy.kv_cache[1][blocks])
        if not same:
            raise LazyCheckFailed(f"{what}: layer {index} holds a different recurrent state than the tiled path")


def lazy_selfcheck(record_type, device, heads_k, heads_v, k_dim, v_dim, window, record_rows, steps, rows=6,
                   layers=2, rounds=16, seed=11, block_size=8, graphs=False):
    quiet = _LAZY["quiet"]
    _LAZY["quiet"] = True
    try:
        return _lazy_selfcheck(record_type, device, heads_k, heads_v, k_dim, v_dim, window, record_rows, steps,
                               rows, layers, rounds, seed, block_size, graphs)
    finally:
        _LAZY["quiet"] = quiet


def _lazy_selfcheck(record_type, device, heads_k, heads_v, k_dim, v_dim, window, record_rows, steps, rows,
                    layers, rounds, seed, block_size, graphs):
    device = torch.device(device)
    gen = torch.Generator().manual_seed(seed)

    def drawn(scale, *shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen, device="cpu") * scale).to(dtype).to(device)

    def pick(low, high):
        return int(torch.randint(low, high, (), generator=gen, device="cpu"))

    def chance(share):
        return float(torch.rand((), generator=gen, device="cpu")) < share

    def fresh():
        return record_type(
            correction=torch.empty((record_rows, heads_v, window, v_dim), dtype=torch.float32, device=device),
            key=torch.empty((record_rows, heads_k, window, k_dim), dtype=torch.float32, device=device),
            gate=torch.empty((record_rows, heads_v, window), dtype=torch.float32, device=device))

    def ints(values, shape=None):
        made = torch.tensor(values, dtype=torch.int32, device="cpu")
        return (made if shape is None else made.reshape(shape)).contiguous().to(device)

    columns = 4
    running = [1 + r for r in range(rows)]
    spare = [1 + rows + r for r in range(rows)]
    column_of = [[1 + 2 * rows + columns * r + c for c in range(columns)] for r in range(rows)]
    pool = list(range(1 + (2 + columns) * rows, 1 + (2 + columns) * rows + 2 * rounds))
    blocks = pool[-1] + 1
    pairs = []
    for _ in range(layers):
        A_log = drawn(0.5, heads_v, dtype=torch.float32)
        dt_bias = drawn(1.0, heads_v)
        state = drawn(0.3, blocks, heads_v, v_dim, k_dim)
        tiled = fresh()
        tiled.pending = torch.zeros(record_rows, dtype=torch.int32, device=device)
        pairs.append((types.SimpleNamespace(A_log=A_log, dt_bias=dt_bias, kv_cache=(None, state.clone()),
                                            gdn_accept_records=_attach_tiled(tiled)),
                      types.SimpleNamespace(A_log=A_log, dt_bias=dt_bias, kv_cache=(None, state.clone()),
                                            gdn_accept_records=attach_lazy(fresh(), steps))))
    stats = torch.zeros(len(LAZY_STATS), dtype=torch.int64, device=device)
    tiled_context = DeferContext.of(_check_group([tiled for tiled, _ in pairs]))
    lazy_context = LazyContext.of(_check_group([lazy for _, lazy in pairs]), stats=stats)
    slot_of = [int(slot) for slot in torch.randperm(record_rows, generator=gen, device="cpu")[:rows]]
    live = list(range(rows))
    held = {"rows": 0, "resumes": 0, "joins": 0, "graph_rounds": 0}

    def verify(round_index, q, k, v, a, b, cu, indices, slots, tokens, capture):
        outs = []
        for tiled, lazy in pairs:
            for layer, runner in ((tiled, run_tiled), (lazy, run_lazy)):
                out = torch.zeros((tokens, heads_v, v_dim), dtype=torch.bfloat16, device=device)
                outs.append((layer, runner, out))
        if capture:
            for layer, runner, out in outs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    runner(layer, q=q, k=k, v=v, a=a, b=b, out=out, query_start_loc=cu, state_indices=indices,
                           slots=slots)
                graph.replay()
            torch.cuda.synchronize(device)
            held["graph_rounds"] += 1
        else:
            for layer, runner, out in outs:
                runner(layer, q=q, k=k, v=v, a=a, b=b, out=out, query_start_loc=cu, state_indices=indices,
                       slots=slots)
        for index in range(len(pairs)):
            if not torch.equal(outs[2 * index][2], outs[2 * index + 1][2]):
                raise LazyCheckFailed(f"round {round_index}: layer {index} verify outputs differ from the tiled path")
        return outs

    def resume(r, align):
        slot = slot_of[r]
        lazy_context.leave(slot)
        _check_equal(pairs, running[r], f"leave of slot {slot}")
        kept = int(pairs[0][0].gdn_accept_records.pending[slot])
        extra = pick(0, kept + 2)
        resumed = 1000 + pick(0, 100)
        computed = torch.zeros(record_rows, dtype=torch.int32, device=device)
        computed[slot] = resumed + extra
        target, state_idx, column = running[r], None, 0
        if align:
            target = pool.pop()
            for tiled, lazy in pairs:
                tiled.kv_cache[1][target] = tiled.kv_cache[1][running[r]]
                lazy.kv_cache[1][target] = lazy.kv_cache[1][running[r]]
            column = 3
            state_idx = torch.zeros(record_rows, dtype=torch.int32, device=device)
            state_idx[slot] = column + 1
        tiled_context.resume(slot, target, resumed, computed, state_idx, column)
        lazy_context.resume(slot, target, resumed, computed, state_idx, column)
        _check_equal(pairs, target, f"resume of slot {slot} with {extra} extra token(s)")
        _check_reset(pairs, slot)
        running[r] = target
        held["resumes"] += 1

    def join(r):
        old = slot_of[r]
        lazy_context.leave(old)
        _check_equal(pairs, running[r], f"leave of finished slot {old}")
        free = sorted(set(range(record_rows)) - {slot_of[other] for other in live if other != r})
        slot = free[pick(0, len(free))]
        _check_reset(pairs, slot)
        target = pool.pop()
        if chance(0.5):
            donor = column_of[pick(0, rows)][pick(0, columns)]
            for tiled, lazy in pairs:
                tiled.kv_cache[1][target] = tiled.kv_cache[1][donor]
                lazy.kv_cache[1][target] = lazy.kv_cache[1][donor]
        else:
            made = drawn(0.3, heads_v, v_dim, k_dim)
            for tiled, lazy in pairs:
                tiled.kv_cache[1][target] = made
                lazy.kv_cache[1][target] = made
        running[r] = target
        slot_of[r] = slot
        held["joins"] += 1

    for round_index in range(rounds + 1):
        capture_rows = round_index == 0
        graph_round = graphs and device.type == "cuda" and round_index == rounds
        active = [r for r in live if not chance(0.15)] or live[:1]
        order = [active[i] for i in torch.randperm(len(active), generator=gen, device="cpu").tolist()]
        lengths = [window if chance(0.7) else pick(1, window + 1) for _ in order]
        at = pick(0, len(order) + 1)
        entries = order[:at] + [None] + order[at:]
        lengths = lengths[:at] + [pick(1, window + 1)] + lengths[at:]
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        tokens = starts[-1]
        cu = ints(starts)
        indices = ints([0 if r is None else running[r] for r in entries])
        slots = ints([-1 if r is None or capture_rows else slot_of[r] for r in entries])
        q = drawn(1.0, tokens, heads_k * k_dim)
        k = drawn(1.0, tokens, heads_k * k_dim)
        v = drawn(1.0, tokens, heads_v * v_dim)
        a = drawn(1.0, tokens, heads_v)
        b = drawn(1.0, tokens, heads_v)
        if capture_rows:
            before = [tiled.kv_cache[1].clone() for tiled, _ in pairs]
        outs = verify(round_index, q, k, v, a, b, cu, indices, slots, tokens, graph_round)
        if capture_rows:
            if any(int(torch.count_nonzero(out)) for _, _, out in outs):
                raise LazyCheckFailed("capture rows wrote a non-zero output")
            for (tiled, lazy), then in zip(pairs, before):
                if not (torch.equal(tiled.kv_cache[1], then) and torch.equal(lazy.kv_cache[1], then)):
                    raise LazyCheckFailed("capture rows changed a recurrent state")
            continue
        sampled = ints([pick(0, n + 1) if chance(0.15) else pick(max(1, n - 2), n + 1) for n in lengths])
        computed = ints([pick(0, columns * block_size) for _ in entries])
        align = round_index % 2 == 1
        unit_on = round_index % 3 == 2
        table = ints([[0] * columns if r is None else column_of[r] for r in entries], (len(entries), columns))
        bounds = ints([int(c) + pick(1, 3 * window) for c in computed.tolist()])
        unit = ints([[0, 0] if r is None else [running[r], spare[r]] for r in entries], (len(entries), 2))
        kwargs = dict(source_blocks=indices, query_start_loc=cu,
                      num_computed=computed if align or unit_on else None,
                      block_table=table if align else None, block_size=block_size if align else 0,
                      unit_pairs=unit if unit_on else None, unit_bounds=bounds if unit_on else None)
        tiled_context.commit(sampled, slots=slots, **kwargs)
        lazy_context.commit(sampled, slots=slots, **kwargs)
        held["rows"] += len(order)
        moving = {running[r] for r in live}
        others = torch.tensor([block for block in range(blocks) if block not in moving], dtype=torch.long,
                              device=device)
        _check_equal(pairs, others, f"round {round_index} commit checkpoints")
        if round_index % 5 == 3:
            resume(live[pick(0, len(live))], chance(0.5))
        if round_index % 4 == 1:
            join(live[pick(0, len(live))])
    for r in live:
        lazy_context.leave(slot_of[r])
        _check_equal(pairs, running[r], f"final leave of slot {slot_of[r]}")
        computed = torch.zeros(record_rows, dtype=torch.int32, device=device)
        computed[slot_of[r]] = 1000
        tiled_context.resume(slot_of[r], running[r], 1000, computed)
        lazy_context.resume(slot_of[r], running[r], 1000, computed)
    _check_equal(pairs, None, "final fold of every request")
    totals = [int(value) for value in stats.tolist()]
    if totals[0] != held["rows"]:
        raise LazyCheckFailed(f"the lazy counters saw {totals[0]} committed rows, the check committed {held['rows']}")
    return {"ok": True, **held, **dict(zip(LAZY_STATS, totals)), "layers": layers, "steps": steps}


def lazy_guard(records):
    if _LAZY["checked"]:
        return
    _LAZY["checked"] = True
    started = time.monotonic()
    device = records.correction.device
    checking = False
    try:
        steps = lazy_steps()
        window = records.window
        if window * (steps + 1) > RING_LIMIT:
            raise ValueError(f"{LAZY_STEPS_ENV}={steps} needs a {window * (steps + 1)}-token ring and the step marks "
                             f"fit {RING_LIMIT}")
        if device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
            say("lazy_check", f"GDN_LAZY self-check skipped: the records live on {device} without the Triton "
                              f"interpreter")
            return
        checking = True
        verdict = lazy_selfcheck(type(records), device, records.key.shape[1], records.correction.shape[1],
                                 records.key.shape[3], records.correction.shape[3], window, records.rows, steps,
                                 graphs=device.type == "cuda")
    except Exception as failure:
        verdict = {"ok": False, "reason": f"{type(failure).__name__}: {failure}"}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seconds = time.monotonic() - started
    if not verdict.get("ok"):
        _LAZY["stood"] = verdict.get("reason") or "no verdict"
        if checking:
            selfcheck_fail(FUSED_ENV, _LAZY["stood"])
        print(f"GDN_LAZY stood aside: {_LAZY['stood']}; the tiled path serves every GDN layer "
              f"(self-check {seconds:.1f} s)", flush=True)
        return
    print(f"GDN_LAZY self-check passed in {seconds:.1f} s on {device}: {verdict['layers']} layers x "
          f"{verdict['rows']} committed rows, {verdict['resumes']} pause/resumes, {verdict['joins']} slot "
          f"joins, {verdict['graph_rounds']} CUDA-graph round; every verify output, checkpoint, leave and resume "
          f"bit-equal to tiled; {verdict['ring_writes']} ring flushes + {verdict['boundary_writes']} boundary "
          f"writes against tiled's {verdict['tiled_writes'] + verdict['boundary_writes']}", flush=True)


Q8_FIRST_ENV = "TAAF_VLLM_GDN_Q8_FIRST"
Q8_EMPTY = [0, 0, 0, -1, -1, -1, -1, 0, 0]
FRESH_SLOTS = 8192
Q8_KINDS = {1: "a non-finite scale", 2: "a zero scale over zero codes, a row no int8 write ever set"}


@triton.jit
def _q8_count_read(state, block_off, i_hv, o_v, mask_v, bad, HV: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                   block, slot, layer, FIRST: tl.constexpr, fresh):
    _q8_count_bad(state, block_off, i_hv, o_v, mask_v, bad, HV, V, K)
    if FIRST:
        base = (state + block_off).to(tl.pointer_type(tl.int8))
        row_scale = tl.load((base + HV * V * K).to(tl.pointer_type(tl.float32)) + i_hv * V + o_v, mask=mask_v,
                            other=1.0)
        broken = mask_v & ((row_scale != row_scale) | (tl.abs(row_scale) > 3.4028234e38))
        count = tl.sum(broken.to(tl.int64), 0)
        settled = tl.load(fresh + slot) == 0
        o_k = tl.arange(0, K)
        codes = tl.load(base + i_hv * V * K + o_v[:, None] * K + o_k[None, :], mask=mask_v[:, None], other=0)
        empty = tl.max(tl.abs(codes.to(tl.int32)), 1) == 0
        zero = mask_v & (row_scale == 0.0) & settled
        unwritten = tl.sum((zero & empty).to(tl.int64), 0)
        tiny = tl.sum((zero & ~empty).to(tl.int64), 0)
        tl.atomic_add(bad + 2, unwritten, mask=unwritten > 0)
        tl.atomic_add(bad + 8, tiny, mask=tiny > 0)
        if count * settled.to(tl.int64) + unwritten > 0:
            wide = block.to(tl.int64)
            if tl.atomic_cas(bad + 3, wide * 0 - 1, wide) == -1:
                tl.store(bad + 4, slot.to(tl.int64))
                tl.store(bad + 5, layer.to(tl.int64))
                tl.store(bad + 6, i_hv.to(tl.int64))
                tl.store(bad + 7, tl.where(count > 0, 1, 2).to(tl.int64))


def q8_first():
    value = os.environ.get(Q8_FIRST_ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{Q8_FIRST_ENV}={value!r} is not one of '', '0', '1'")
    return value == "1" and q8()


def layer_index(layer):
    import re

    found = re.search(r"layers\.(\d+)\.", str(getattr(layer, "prefix", "") or ""))
    return int(found.group(1)) if found else -1


def slot_owner(slot):
    accept = sys.modules.get("vllm.taaf_gdn_accept")
    owners = getattr(accept, "SLOT_OWNERS", None) or {}
    return owners.get(int(slot), "unknown")


def q8_fresh(device):
    held = _Q8_LIVE.get("fresh")
    if held is None:
        held = _Q8_LIVE["fresh"] = torch.zeros(FRESH_SLOTS, dtype=torch.int8, device=device)
    accept = sys.modules.get("vllm.taaf_gdn_accept")
    if accept is not None and getattr(accept, "SLOT_FRESH", None) is not held:
        accept.SLOT_FRESH = held
    return held


def q8_first_args(device):
    return {"FIRST": True, "fresh": q8_fresh(device)} if q8_first() else {}


def q8_first_said(read, first):
    if not q8_first() or len(first) < 7:
        return read
    unwritten, block, slot, layer, head, kind, tiny = first[:7]
    held = _Q8_LIVE
    held["unwritten"] = held.get("unwritten", 0) + unwritten
    named = (f"; the first since the last tally: GDN layer {layer}, value head {head}, state block {block}, "
             f"request slot {slot} last added as {slot_owner(slot)}, {Q8_KINDS.get(kind, 'nothing')}"
             if block >= 0 else "")
    print(f"GDN_STATE_Q8 first: the verify read {unwritten} unwritten row(s) (zero scale, zero codes) and "
          f"{tiny} row(s) too small for a scale (zero scale over written codes) since the last tally "
          f"({held['unwritten']} since the first){named}", flush=True)
    return read


_Q8_FULL = {}


def q8_full(state):
    if state.dim() != 2:
        return state
    full = _Q8_FULL.get((state.data_ptr(), int(state.shape[0])))
    if full is None:
        raise ValueError(f"{Q8_ENV} met a packed GDN state container with no registered full view")
    return full


def q8_codes_bytes():
    for full in _Q8_FULL.values():
        return int(full[0].numel())
    return None


def ssm_full(layer):
    full = getattr(layer, "taaf_q8_full", None)
    return layer.kv_cache[1] if full is None else full
