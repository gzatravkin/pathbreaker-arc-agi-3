import dataclasses
import functools
import os

import numpy as np
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

ARCHITECTURES = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
HOST_TIER = "SimpleCPUOffloadConnector"
SPARSE_TIER = "HiSparseConnector"
CONNECTORS = (None, HOST_TIER, SPARSE_TIER)
PARALLEL_RANKS = ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                  "decode_context_parallel_size", "prefill_context_parallel_size")
VERIFY_SHAPE_ENV = "TAAF_VLLM_GDN_VERIFY_SHAPE"
COMMIT_SHAPE_ENV = "TAAF_VLLM_GDN_COMMIT_SHAPE"


def launch_shape(name, default):
    said = os.environ.get(name, "").strip()
    if not said:
        return default
    try:
        shape = tuple(int(part) for part in said.split(","))
    except ValueError:
        shape = ()
    if len(shape) != 3 or min(shape) <= 0 or shape[0] & (shape[0] - 1) or shape[1] & (shape[1] - 1):
        raise ValueError(f"{name} must be block_v,warps,stages with powers of two for the first two, found {said!r}")
    print(f"GDN_ACCEPT {name}={shape[0]},{shape[1]},{shape[2]} in place of "
          f"{default[0]},{default[1]},{default[2]}", flush=True)
    return shape


VERIFY_BLOCK_V, VERIFY_WARPS, VERIFY_STAGES = launch_shape(VERIFY_SHAPE_ENV, (64, 4, 3))
COMMIT_BLOCK_V, COMMIT_WARPS, COMMIT_STAGES = launch_shape(COMMIT_SHAPE_ENV, (32, 4, 2))
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
_SAID: set = set()
_DECIDED: dict = {}


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def unmet(vllm_config):
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    connector = getattr(transfer, "kv_connector", None)
    mamba = getattr(vllm_config, "mamba_config", None)
    needs = {
        "Model Runner V2": bool(vllm_config.use_v2_model_runner),
        "the align or none Mamba cache mode": cache.mamba_cache_mode in ("align", "none"),
        "one rank of every parallel kind": all(
            int(getattr(parallel, name, 1) or 1) == 1 for name in PARALLEL_RANKS),
        "a bfloat16 or float32 recurrent state without stochastic rounding": (
            str(cache.mamba_ssm_cache_dtype) in ("auto", "bfloat16", "float32")
            and not getattr(mamba, "enable_stochastic_rounding", False)),
        "no other RecoverSSM or ReplaySSM path": not (
            getattr(cache, "use_kda_recoverssm", False) or getattr(cache, "use_replayssm", False)),
        f"no KV connector other than {HOST_TIER} or {SPARSE_TIER}": connector in CONNECTORS,
    }
    return [name for name, held in needs.items() if not held]


def active(vllm_config):
    if vllm_config is None:
        return False
    held = _DECIDED.get(id(vllm_config))
    if held is not None and held[0] is vllm_config:
        return held[1]
    decided = _decide(vllm_config)
    _DECIDED[id(vllm_config)] = (vllm_config, decided)
    return decided


def _decide(vllm_config):
    drafts = int(getattr(vllm_config, "num_speculative_tokens", 0) or 0)
    names = set(getattr(getattr(vllm_config, "model_config", None), "architectures", None) or ())
    if drafts <= 0 or not names & set(ARCHITECTURES):
        return False
    missing = unmet(vllm_config)
    if missing:
        raise ValueError("GDN_ACCEPT cannot serve this configuration; it needs "
                         + ", ".join(missing))
    say("live", f"GDN_ACCEPT live: the {drafts} MTP drafts are verified from each request's "
                f"running GDN state and only the accepted prefix is written back, so every Mamba "
                f"cache group keeps 1 state block per running request instead of {1 + drafts}")
    return True


def say_group(layer_names, drafts):
    say(("group", tuple(layer_names)),
        f"GDN_ACCEPT group: {len(layer_names)} GDN layers verify from 1 state block per running "
        f"request (stock {1 + drafts}), first {layer_names[0] if layer_names else '-'}")


@dataclasses.dataclass
class Records:
    correction: torch.Tensor
    key: torch.Tensor
    gate: torch.Tensor

    @property
    def rows(self):
        return self.correction.shape[0]

    @property
    def window(self):
        return self.correction.shape[2]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.correction, self.key, self.gate))


def make_records(rows, k_heads, v_heads, k_dim, v_dim, window, device):
    return Records(
        correction=torch.empty((rows, v_heads, window, v_dim), dtype=torch.float32, device=device),
        key=torch.empty((rows, k_heads, window, k_dim), dtype=torch.float32, device=device),
        gate=torch.empty((rows, v_heads, window), dtype=torch.float32, device=device),
    )


def allocate_records(layer, vllm_config):
    if not active(vllm_config):
        return None
    from vllm.platforms import current_platform

    records = make_records(vllm_config.scheduler_config.max_num_seqs,
                           layer.num_k_heads // layer.tp_size, layer.num_v_heads // layer.tp_size,
                           layer.head_k_dim, layer.head_v_dim, layer.num_spec + 1,
                           current_platform.current_device())
    say("records", f"GDN_ACCEPT records: {records.nbytes / 2**20:.2f} MiB a GDN layer "
                   f"({records.rows} rows x {records.window}-token windows) in place of "
                   f"{layer.num_spec} per-draft state blocks a request")
    return records


@triton.jit(do_not_specialize=["num_rows"])
def _verify_kernel(
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
    WINDOW: tl.constexpr,
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
    if (block <= 0) | (i_n >= RECORD_ROWS):
        zero = tl.zeros([BV], dtype=tl.float32)
        for _ in range(0, length):
            tl.store(p_o, zero.to(p_o.dtype.element_ty), mask=mask_v)
            p_o += stride_o
        return
    p_h = state + block * stride_state_block + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    p_correction = correction + ((i_n * HV + i_hv) * WINDOW) * V + o_v
    p_key = keys + ((i_n * H + i_h) * WINDOW) * K + o_k
    p_gate = gates + (i_n * HV + i_hv) * WINDOW
    head_writer = i_v == 0
    key_writer = head_writer & (i_hv % (HV // H) == 0)
    for i_t in range(0, length):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        x = tl.load(p_a).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x)),
            x,
        )
        b_g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
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


def _rows_of(tensor, tokens):
    flat = tensor.reshape(tokens, -1)
    if flat.stride(1) != 1:
        raise ValueError("GDN_ACCEPT needs each token's heads contiguous")
    return flat


def verify(A_log, dt_bias, state, records, q, k, v, a, b, query_start_loc, state_indices,
           out=None):
    tokens = q.shape[0]
    _, v_heads, v_dim, k_dim = state.shape
    k_heads = records.key.shape[1]
    rows = state_indices.shape[0]
    if state.stride()[1:] != (v_dim * k_dim, k_dim, 1):
        raise ValueError("GDN_ACCEPT needs contiguous state heads")
    if any(t.shape[0] != tokens for t in (k, v, a, b)):
        raise ValueError("GDN_ACCEPT inputs disagree on the token count")
    if query_start_loc.shape[0] < rows + 1:
        raise ValueError("GDN_ACCEPT query metadata is shorter than its rows")
    if rows > records.rows:
        raise ValueError(f"GDN_ACCEPT verifies {rows} rows but holds records for {records.rows}")
    if out is None:
        out = q.new_empty(1, tokens, v_heads, v_dim)
    if tokens == 0 or rows == 0:
        return out
    q2, k2, v2 = _rows_of(q, tokens), _rows_of(k, tokens), _rows_of(v, tokens)
    if q2.shape[1] != k_heads * k_dim or k2.shape[1] != k_heads * k_dim:
        raise ValueError("GDN_ACCEPT q and k do not match the records' key heads")
    if v2.shape[1] != v_heads * v_dim:
        raise ValueError("GDN_ACCEPT v does not match the state's value heads")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("GDN_ACCEPT needs contiguous gate heads")
    o2 = out.reshape(tokens, v_heads * v_dim)
    if o2.data_ptr() != out.data_ptr() or o2.stride(1) != 1:
        raise ValueError("GDN_ACCEPT needs a contiguous output row per token")
    block_v = min(triton.next_power_of_2(v_dim), VERIFY_BLOCK_V)
    grid = (triton.cdiv(v_dim, block_v), rows * v_heads)
    _verify_kernel[grid](
        A_log, a, b, dt_bias, q2, k2, v2, o2, state, query_start_loc, state_indices,
        records.correction, records.key, records.gate, k_dim**-0.5, SOFTPLUS_BETA,
        SOFTPLUS_THRESHOLD, rows,
        q2.stride(0), k2.stride(0), v2.stride(0), a.stride(0), b.stride(0), o2.stride(0),
        state.stride(0), state_indices.stride(0),
        H=k_heads, HV=v_heads, K=k_dim, V=v_dim, BK=triton.next_power_of_2(k_dim), BV=block_v,
        WINDOW=records.window, RECORD_ROWS=records.rows,
        num_warps=VERIFY_WARPS, num_stages=VERIFY_STAGES,
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return out


def verify_layer(layer, q, k, v, a, b, query_start_loc, state_indices, out=None):
    made = verify(layer.A_log, layer.dt_bias, ssm_full(layer), layer.gdn_accept_records,
                  q, k, v, a, b, query_start_loc, state_indices, out)
    say("mixed", f"GDN_ACCEPT mixed-batch path: {state_indices.shape[0]} speculative row(s) "
                 f"beside prefill chunks verified from their running state")
    return made


def decode(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):
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
    verify(layer.A_log, layer.dt_bias, ssm_full(layer), layer.gdn_accept_records,
           mixed[:, :key_dim], mixed[:, key_dim:2 * key_dim], mixed[:, 2 * key_dim:],
           a[:tokens], b[:tokens], query_start_loc[: rows + 1], state_indices[:rows, 0], out)
    layer._rms_norm_gated_cuda(out, output_gate[:tokens], out)
    say("decode", f"GDN_ACCEPT decode path: {rows} row(s), {tokens} tokens verified from the "
                  f"running state in place of the fused per-draft kernel")


@triton.heuristics({
    "HAS_REQUEST_INDICES": lambda args: args["request_indices"] is not None,
    "ALIGN": lambda args: args["block_table"] is not None,
    "UNIT": lambda args: args["unit_pairs"] is not None,
})
@triton.jit(do_not_specialize=["num_rows"])
def _commit_kernel(
    state_reference,
    state_addrs,
    state_strides,
    correction_reference,
    correction_addrs,
    key_reference,
    key_addrs,
    gate_reference,
    gate_addrs,
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
    if (source <= 0) | (row >= RECORD_ROWS):
        return
    length = tl.load(query_start_loc + row + 1) - tl.load(query_start_loc + row)
    batch_row = row
    if HAS_REQUEST_INDICES:
        batch_row = tl.load(request_indices + row).to(tl.int32)
    accepted = tl.load(num_sampled + batch_row).to(tl.int32)
    accepted = tl.minimum(tl.maximum(accepted, 0), length)
    accepted = tl.minimum(accepted, WINDOW)
    if accepted <= 0:
        return
    target = source
    checkpoint = accepted * 0 - 1
    if ALIGN:
        computed = tl.load(num_computed + batch_row).to(tl.int64)
        column = computed // BLOCK_SIZE
        reach = ((column + 1) * BLOCK_SIZE - computed).to(tl.int32)
        if (reach <= accepted) & (column < table_width):
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
            unit_target = spare
            unit_checkpoint = unit_reach
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
    p_correction = correction + ((row * HV + i_hv) * WINDOW) * V + o_v
    p_key = key + ((row * H + i_h) * WINDOW) * K + o_k
    p_gate = gate + (row * HV + i_hv) * WINDOW
    for i_t in range(0, accepted):
        b_g = tl.load(p_gate + i_t)
        b_v = tl.load(p_correction + i_t * V, mask=mask_v, other=0)
        b_k = tl.load(p_key + i_t * K, mask=mask_k, other=0)
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


def _addresses(tensors, device):
    return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device=device)


@dataclasses.dataclass
class CommitContext:
    states: tuple
    records: tuple
    state_addrs: torch.Tensor
    state_strides: torch.Tensor
    correction_addrs: torch.Tensor
    key_addrs: torch.Tensor
    gate_addrs: torch.Tensor

    @classmethod
    def create(cls, layers):
        if not layers:
            raise ValueError("GDN_ACCEPT commit needs at least one GDN layer")
        states = tuple(ssm_full(layer) for layer in layers)
        records = tuple(layer.gdn_accept_records for layer in layers)
        if any(record is None for record in records):
            raise ValueError("GDN_ACCEPT found a GDN layer without records")
        return cls.of(states, records)

    @classmethod
    def of(cls, states, records):
        first, held = states[0], records[0]
        if first.ndim != 4:
            raise ValueError("GDN_ACCEPT state must be [blocks, heads, value, key]")
        _, v_heads, v_dim, k_dim = first.shape
        for state in states:
            if (state.shape[1:] != first.shape[1:] or state.dtype != first.dtype
                    or state.device != first.device
                    or state.stride()[1:] != (v_dim * k_dim, k_dim, 1)):
                raise ValueError("GDN_ACCEPT layers of one group need one state layout")
        for record in records:
            for tensor, reference in ((record.correction, held.correction),
                                      (record.key, held.key), (record.gate, held.gate)):
                if (tensor.shape != reference.shape or tensor.dtype != torch.float32
                        or not tensor.is_contiguous() or tensor.device != first.device):
                    raise ValueError("GDN_ACCEPT layers of one group need one record layout")
        if held.correction.shape[1] != v_heads or held.correction.shape[3] != v_dim:
            raise ValueError("GDN_ACCEPT records do not match the state heads")
        device = first.device
        return cls(
            states=tuple(states),
            records=tuple(records),
            state_addrs=_addresses(states, device),
            state_strides=torch.tensor([state.stride(0) for state in states], dtype=torch.int64,
                                       device=device),
            correction_addrs=_addresses([record.correction for record in records], device),
            key_addrs=_addresses([record.key for record in records], device),
            gate_addrs=_addresses([record.gate for record in records], device),
        )

    def commit(self, num_sampled, *, source_blocks, query_start_loc, request_indices=None,
               num_computed=None, block_table=None, block_size=0, unit_pairs=None,
               unit_bounds=None):
        rows = source_blocks.shape[0]
        if isinstance(num_sampled, int) or rows == 0:
            return
        held = self.records[0]
        if rows > held.rows:
            raise ValueError(f"GDN_ACCEPT commits {rows} rows but holds records for {held.rows}")
        if query_start_loc.shape[0] < rows + 1:
            raise ValueError("GDN_ACCEPT commit query metadata is shorter than its rows")
        if block_table is not None and (num_computed is None or block_size <= 0
                                        or block_table.stride(1) != 1):
            raise ValueError("GDN_ACCEPT align commit needs computed counts, a block size and "
                             "contiguous block-table rows")
        if unit_pairs is not None and (num_computed is None or unit_bounds is None
                                       or not unit_pairs.is_contiguous()):
            raise ValueError("GDN_ACCEPT reply checkpoints need computed counts, boundaries and "
                             "contiguous running/spare pairs")
        first = self.states[0]
        _, v_heads, v_dim, k_dim = first.shape
        block_v = min(triton.next_power_of_2(v_dim), COMMIT_BLOCK_V)
        grid = (triton.cdiv(v_dim, block_v), rows, len(self.states) * v_heads)
        _commit_kernel[grid](
            first, self.state_addrs, self.state_strides,
            held.correction, self.correction_addrs,
            held.key, self.key_addrs,
            held.gate, self.gate_addrs,
            source_blocks, source_blocks.stride(0), query_start_loc, num_sampled,
            request_indices, num_computed, block_table,
            0 if block_table is None else block_table.stride(0),
            0 if block_table is None else block_table.shape[1],
            unit_pairs, unit_bounds,
            rows,
            H=held.key.shape[1], HV=v_heads, K=k_dim, V=v_dim,
            BK=triton.next_power_of_2(k_dim), BV=block_v, WINDOW=held.window,
            RECORD_ROWS=held.rows, BLOCK_SIZE=max(int(block_size), 1),
            num_warps=COMMIT_WARPS, num_stages=COMMIT_STAGES,
            PDL=pdl_on(), launch_pdl=pdl_on(),
        )
        say("commit", f"GDN_ACCEPT first commit: {rows} request(s) x {len(self.states)} GDN "
                      f"layers folded their accepted prefix into the running state block in one "
                      f"launch" + ("; a crossed block boundary also gets its checkpoint"
                                   if block_table is not None else ""))
        if unit_pairs is not None:
            say("unit-commit", f"CACHE_KEEP_END worker: first GDN commit with reply checkpoints: "
                               f"{rows} row(s) x {len(self.states)} layers; a row whose accepted "
                               f"tokens reach its boundary stores the state after exactly that "
                               f"many tokens into its spare block")


_MEMO = {}
MEMO_SIZE = 8


def step_memo(kind, layouts, build):
    key = (kind, *(bytes(layout.numpy().tobytes()) for layout in layouts))
    held = _MEMO.get(key)
    if held is None:
        held = build()
        if len(_MEMO) >= MEMO_SIZE:
            _MEMO.clear()
        _MEMO[key] = held
        say("memo", "GDN_ACCEPT metadata memo: the chunk and causal-conv metadata of a step's prefill "
                    "layout are built once and shared by every GDN cache group of that step")
    return held


def decode_rows(draft_rows_cpu, common, drafts):
    prefilling = common.is_prefilling
    if prefilling is None:
        return draft_rows_cpu
    lengths = common.query_start_loc_cpu.diff()
    return draft_rows_cpu | (~prefilling & (lengths > 0) & (lengths <= drafts + 1))


def commit_for(builder, common, state_indices, query_start_loc, rows, request_indices):
    context = builder.gdn_accept_context
    if context is None:
        forward = builder.vllm_config.compilation_config.static_forward_context
        context = CommitContext.create([forward[name] for name in builder.layer_names])
        builder.gdn_accept_context = context
    align = builder.vllm_config.cache_config.mamba_cache_mode == "align"
    unit = getattr(builder, "taaf_unit", None) if align else None
    return functools.partial(
        context.commit,
        source_blocks=state_indices[:rows, 0],
        query_start_loc=query_start_loc[: rows + 1],
        request_indices=request_indices,
        num_computed=common.compute_num_computed_tokens() if align else None,
        block_table=common.block_table_tensor if align else None,
        block_size=builder.kv_cache_spec.block_size if align else 0,
        unit_pairs=None if unit is None else unit[0],
        unit_bounds=None if unit is None else unit[1],
    )


UNIT_FIELD = "taaf_unit_targets"
_copy_mamba_state_block = None


@dataclasses.dataclass
class UnitTable:
    groups: tuple
    cpu: np.ndarray
    gpu: torch.Tensor
    pairs: torch.Tensor
    bounds: torch.Tensor
    pair_ptrs: torch.Tensor
    rows: int = 0
    live: int = 0

    @classmethod
    def create(cls, groups, max_reqs, device):
        pairs = torch.zeros((len(groups), max_reqs, 2), dtype=torch.int32, device=device)
        return cls(
            groups=tuple(groups),
            cpu=np.zeros((1 + len(groups), max_reqs), dtype=np.int32),
            gpu=torch.zeros((1 + len(groups), max_reqs), dtype=torch.int32, device=device),
            pairs=pairs,
            bounds=torch.zeros(max_reqs, dtype=torch.int32, device=device),
            pair_ptrs=_addresses([pairs[index] for index in range(len(groups))], device),
        )

    def upload(self):
        self.live = int(np.count_nonzero(self.cpu[0]))
        held = torch.from_numpy(self.cpu.copy())
        if self.gpu.is_cuda:
            held = held.pin_memory()
        self.gpu.copy_(held, non_blocking=self.gpu.is_cuda)


def unit_receive(runner, scheduler_output):
    sent = getattr(scheduler_output, UNIT_FIELD, None)
    if not sent:
        return
    state = runner.model_state
    table = getattr(state, "taaf_unit", None)
    if table is None:
        from vllm.v1.worker.mamba_utils import get_mamba_group_ids, get_mamba_groups

        groups = get_mamba_group_ids(get_mamba_groups(runner.kv_cache_config))
        table = UnitTable.create(groups, runner.max_num_reqs, runner.device)
        state.taaf_unit = table
        say("unit", f"CACHE_KEEP_END worker: reply checkpoints live for {len(groups)} Mamba "
                    f"cache groups; a decode step whose accepted tokens cross a request's next "
                    f"match-unit boundary writes the exact recurrent state there into that "
                    f"request's spare block")
    slots = runner.req_states.req_id_to_index
    for req_id, (bound, blocks) in sent.items():
        slot = slots.get(req_id)
        if slot is None:
            continue
        table.cpu[:, slot] = 0
        if bound > 0:
            table.cpu[0, slot] = bound
            for group, block in blocks.items():
                table.cpu[1 + table.groups.index(group), slot] = block
    table.upload()


def unit_reset(state, slot):
    table = getattr(state, "taaf_unit", None)
    if table is None or not table.cpu[:, slot].any():
        return
    table.cpu[:, slot] = 0
    table.upload()


RESUME_SPAN = 2
_resume_tiles = None


def unit_leave(state, req_id):
    owners = state.__dict__.setdefault("taaf_slot_of", {})
    state.taaf_left = (req_id, owners.pop(req_id, None))


def slot_resume(slot, blocks, resumed, computed, state_idx, column):
    return None


def unit_resume(state, slot, new_req_data):
    left = getattr(state, "taaf_left", None)
    state.taaf_left = None
    state.__dict__.setdefault("taaf_slot_of", {})[new_req_data.req_id] = slot
    if not getattr(state, "gdn_accept", False) or left != (new_req_data.req_id, slot):
        return False
    computed = getattr(state, "taaf_computed", None)
    resumed = int(new_req_data.num_computed_tokens)
    if computed is None or resumed <= 0:
        say("resume-blind", "GDN_ACCEPT resume: a continuation arrived before any step reported its computed "
                            "counts; its recurrent state and convolution windows are read as they stand")
        return False
    align = bool(getattr(state, "_align_mode", False))
    column = (resumed - 1) // state.cache_config.block_size if align else 0
    blocks = {group: int(ids[column]) for group, ids in enumerate(new_req_data.block_ids)
              if column < len(ids)}
    state_idx = state._mamba_state_idx_gpu if align else None
    ctx = getattr(state, "_mamba_ctx", None) if align else None
    if ctx is not None and ctx.is_initialized:
        _resume_windows(state, ctx, slot, resumed, column, computed, new_req_data.block_ids)
    else:
        say("resume-windows", "GDN_ACCEPT resume: no align copy context, so a continuation's convolution windows "
                              "stay where its last decode step left them")
    slot_resume(slot, blocks, resumed, computed, state_idx, column)
    say("resume", f"GDN_ACCEPT resume: a streaming continuation re-added in its own slot carries its last step's "
                  f"running state into the block of column {column} (computed {resumed}): the convolution "
                  f"windows of every Mamba state start at offset 0 there, and the temporal state moves from the "
                  f"running column when the step ran in the next block; device counts pick the shift, no host read")
    return True


def _resume_windows(state, ctx, slot, resumed, column, computed, block_ids):
    global _copy_mamba_state_block, _resume_tiles
    if _copy_mamba_state_block is None:
        from vllm.v1.worker.mamba_utils import _copy_mamba_state_block as copy

        _copy_mamba_state_block = copy
    if _resume_tiles is None:
        from vllm.v1.worker import mamba_utils

        _resume_tiles = int(getattr(mamba_utils, "_TEMPORAL_TILES", 1))
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    groups = list(ctx.mamba_group_ids)
    held = getattr(state, "taaf_resume_cands", None)
    if held is None or held.shape[0] != len(groups):
        held = torch.zeros((len(groups), RESUME_SPAN), dtype=torch.int32, device=computed.device)
        state.taaf_resume_cands = held
        state.taaf_resume_ptrs = _addresses([held[index] for index in range(len(groups))], computed.device)
    cpu = np.zeros((len(groups), RESUME_SPAN), dtype=np.int32)
    for local, group in enumerate(groups):
        ids = block_ids[group]
        span = ids[column:column + RESUME_SPAN]
        cpu[local, :len(span)] = span
    staged = torch.from_numpy(cpu)
    if held.is_cuda:
        staged = staged.pin_memory()
    held.copy_(staged, non_blocking=held.is_cuda)
    _resume_copy_kernel[(ctx.num_states, _resume_tiles)](
        slot, resumed, column, state._mamba_state_idx_gpu, state.num_accepted_tokens_gpu, computed,
        state.taaf_resume_ptrs, ctx.state_base_addrs, ctx.state_block_strides, ctx.state_elem_sizes,
        ctx.state_inner_sizes, ctx.state_conv_widths, ctx.state_group_indices, ctx.state_dim_row_count,
        ctx.state_dim_row_stride, SPAN=RESUME_SPAN, COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=is_conv_state_dim_first(), TEMPORAL_TILES=_resume_tiles,
    )


@triton.jit(do_not_specialize=["slot", "resumed", "column"])
def _resume_copy_kernel(slot, resumed, column, state_idx, num_accepted, num_computed, cand_ptrs,
                        state_base_addrs, state_block_strides, state_elem_sizes, state_inner_sizes,
                        state_conv_widths, state_group_indices, state_dim_row_count, state_dim_row_stride,
                        SPAN: tl.constexpr, COPY_BLOCK_SIZE: tl.constexpr, CONV_STATE_DIM_FIRST: tl.constexpr,
                        TEMPORAL_TILES: tl.constexpr):
    index = tl.program_id(0)
    source = tl.load(state_idx + slot) - column
    extra = tl.load(num_computed + slot) - resumed
    bias = tl.load(num_accepted + slot) - 1 - extra
    if (source < 0) | (source >= SPAN) | (extra < 0) | (bias < 0):
        return
    group = tl.load(state_group_indices + index)
    cands = tl.load(cand_ptrs + group).to(tl.pointer_type(tl.int32))
    if (tl.load(cands + source) <= 0) | (tl.load(cands) <= 0):
        return
    shift = tl.where(tl.load(state_conv_widths + index) > 0, bias, 0)
    if (source == 0) & (shift == 0):
        return
    _copy_mamba_state_block(index, 0, source, 0, shift, cand_ptrs, SPAN, state_base_addrs,
                            state_block_strides, state_elem_sizes, state_inner_sizes, state_conv_widths,
                            state_group_indices, state_dim_row_count, state_dim_row_stride, tl.program_id(1),
                            COPY_BLOCK_SIZE, CONV_STATE_DIM_FIRST, TEMPORAL_TILES, 1)


def unit_prepare(state, input_batch, attn_groups, num_reqs, for_capture):
    table = getattr(state, "taaf_unit", None)
    if table is None:
        return
    ctx = getattr(state, "_mamba_ctx", None)
    rows = input_batch.num_reqs
    active = (not for_capture and table.live > 0 and rows > 0 and ctx is not None
              and ctx.is_initialized)
    table.rows = rows if active else 0
    for local, group in enumerate(table.groups):
        for attn_group in attn_groups[group]:
            builder = attn_group.get_metadata_builder(0)
            if getattr(builder, "gdn_accept", False):
                builder.taaf_unit = (table.pairs[local], table.bounds) if active else None
    if not active:
        return
    drafts = int(state.vllm_config.num_speculative_tokens or 0)
    decoding = ((~input_batch.is_prefilling_np[:rows])
                & (input_batch.num_scheduled_tokens[:rows] <= drafts + 1)).astype(np.int32)
    held = torch.from_numpy(decoding)
    if table.gpu.is_cuda:
        held = held.pin_memory()
    mask = held.to(table.gpu.device, non_blocking=table.gpu.is_cuda)
    _unit_rows_kernel[(max(num_reqs, rows), len(table.groups))](
        input_batch.idx_mapping, mask, table.gpu, table.gpu.stride(0), state._mamba_state_idx_gpu,
        ctx.block_table_ptrs, ctx.block_table_stride_req, table.pairs, table.bounds, rows,
        table.pairs.shape[1],
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )


def unit_snapshot(state, idx_mapping, num_sampled, num_computed_tokens):
    if num_computed_tokens is not None:
        state.taaf_computed = num_computed_tokens
    table = getattr(state, "taaf_unit", None)
    if (table is None or not table.rows or isinstance(num_sampled, int)
            or num_computed_tokens is None):
        return
    global _copy_mamba_state_block
    if _copy_mamba_state_block is None:
        from vllm.v1.worker.mamba_utils import _copy_mamba_state_block as copy

        _copy_mamba_state_block = copy
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    ctx = state._mamba_ctx
    rows = min(table.rows, idx_mapping.shape[0], num_sampled.shape[0])
    if rows <= 0:
        return
    _unit_conv_kernel[(rows, ctx.num_states)](
        idx_mapping, num_sampled, num_computed_tokens, table.pair_ptrs, table.bounds,
        ctx.state_base_addrs, ctx.state_block_strides, ctx.state_elem_sizes,
        ctx.state_inner_sizes, ctx.state_conv_widths, ctx.state_group_indices,
        ctx.state_dim_row_count, ctx.state_dim_row_stride,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    say("unit-conv", f"CACHE_KEEP_END worker: first reply-checkpoint pass over {rows} row(s) x "
                     f"{ctx.num_states} Mamba states copies each crossing row's conv window, "
                     f"shifted to the boundary, into its spare block; the GDN commit writes the "
                     f"recurrent state there")


@triton.jit
def _unit_rows_kernel(idx_mapping, decoding, table, stride_table, state_idx, block_table_ptrs,
                      stride_req, pairs, bounds, rows, width, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    group = tl.program_id(1)
    live = row < rows
    slot = tl.load(idx_mapping + row, mask=live, other=-1)
    live = live & (slot >= 0)
    keep = tl.load(decoding + row, mask=live, other=0)
    live = live & (keep != 0)
    slot = tl.where(live, slot, 0)
    bound = tl.load(table + slot, mask=live, other=0)
    live = live & (bound > 0)
    spare = tl.load(table + (1 + group) * stride_table + slot, mask=live, other=0)
    paired = live & (spare > 0)
    column = tl.load(state_idx + slot, mask=paired, other=0)
    base = tl.load(block_table_ptrs + group).to(tl.pointer_type(tl.int32))
    running = tl.load(base + row.to(tl.int64) * stride_req + column, mask=paired, other=0)
    pair = pairs + (group * width + row) * 2
    tl.store(pair, tl.where(paired, running, 0))
    tl.store(pair + 1, tl.where(paired, spare, 0))
    if group == 0:
        tl.store(bounds + row, tl.where(live, bound, 0))


@triton.jit
def _unit_conv_kernel(idx_mapping, num_sampled, num_computed, pair_ptrs, bounds,
                      state_base_addrs, state_block_strides, state_elem_sizes, state_inner_sizes,
                      state_conv_widths, state_group_indices, state_dim_row_count,
                      state_dim_row_stride, COPY_BLOCK_SIZE: tl.constexpr,
                      CONV_STATE_DIM_FIRST: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    index = tl.program_id(1)
    if tl.load(state_conv_widths + index) <= 0:
        return
    bound = tl.load(bounds + row)
    if bound <= 0:
        return
    group = tl.load(state_group_indices + index)
    pair = tl.load(pair_ptrs + group).to(tl.pointer_type(tl.int32))
    if tl.load(pair + row * 2 + 1) <= 0:
        return
    slot = tl.load(idx_mapping + row)
    if slot < 0:
        return
    sampled = tl.load(num_sampled + row)
    reach = bound - (tl.load(num_computed + slot) - sampled)
    if (reach < 1) | (reach > sampled):
        return
    _copy_mamba_state_block(index, row, 0, 1, reach - 1, pair_ptrs, 2, state_base_addrs,
                            state_block_strides, state_elem_sizes, state_inner_sizes,
                            state_conv_widths, state_group_indices, state_dim_row_count,
                            state_dim_row_stride, tl.program_id(2), COPY_BLOCK_SIZE,
                            CONV_STATE_DIM_FIRST, 1, 2)


SLOT_OWNERS = {}
SLOT_FRESH = None
PRECOPY_EMPTY = [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
PRECOPY_CHECK = {"counter": None, "left": 0, "at": None, "every_s": 60.0, "failed": 0, "checked": 0}


def precopy_watch():
    return (os.environ.get("TAAF_VLLM_GDN_Q8_FIRST", "").strip() == "1"
            and os.environ.get("TAAF_VLLM_GDN_STATE_Q8", "").strip() == "1")


@triton.jit(do_not_specialize=["stride_req", "left"])
def _precopy_check_kernel(idx_mapping, src_cols, state_idx, num_accepted, gpu_computed, cpu_computed,
                          block_table_ptrs, stride_req, state_base_addrs, state_block_strides, state_inner_sizes,
                          state_conv_widths, state_group_indices, counter, left, SCALES: tl.constexpr):
    row = tl.program_id(0)
    index = tl.program_id(1)
    if tl.load(state_conv_widths + index) > 0:
        return
    slot = tl.load(idx_mapping + row).to(tl.int64)
    if slot < 0:
        return
    src = tl.load(src_cols + slot).to(tl.int64)
    dst = tl.load(state_idx + slot).to(tl.int64)
    if (src < 0) | (src == dst):
        return
    group = tl.load(state_group_indices + index).to(tl.int64)
    table = tl.load(block_table_ptrs + group).to(tl.pointer_type(tl.int32))
    block = tl.load(table + row * stride_req + dst).to(tl.int64)
    tl.atomic_add(counter + 1, 1)
    if block <= 0:
        return
    base = tl.load(state_base_addrs + index).to(tl.pointer_type(tl.int8))
    held = base + block * tl.load(state_block_strides + index) + tl.load(state_inner_sizes + index)
    scales = tl.load(held.to(tl.pointer_type(tl.float32)) + tl.arange(0, SCALES))
    plausible = (scales == scales) & (tl.abs(scales) < 1e3) & (tl.abs(scales) > 1e-12)
    if tl.sum(plausible.to(tl.int32), 0) * 2 >= SCALES:
        return
    tl.atomic_add(counter, 1)
    if tl.atomic_cas(counter + 2, block * 0 - 1, row.to(tl.int64)) == -1:
        tl.store(counter + 3, slot)
        tl.store(counter + 4, src)
        tl.store(counter + 5, dst)
        tl.store(counter + 6, block)
        tl.store(counter + 7, tl.load(gpu_computed + slot).to(tl.int64))
        tl.store(counter + 8, tl.load(cpu_computed + row).to(tl.int64))
        tl.store(counter + 9, tl.load(num_accepted + slot).to(tl.int64))
        tl.store(counter + 10, left.to(tl.int64))
        tl.store(counter + 11, index.to(tl.int64))


def precopy_check(state, input_batch, for_capture):
    if for_capture or not precopy_watch() or not getattr(state, "_align_mode", False):
        return None
    ctx = getattr(state, "_mamba_ctx", None)
    rows = int(getattr(input_batch, "num_reqs", 0) or 0)
    if ctx is None or not ctx.is_initialized or rows <= 0:
        return None
    held = PRECOPY_CHECK
    device = state._mamba_state_idx_gpu.device
    if held["counter"] is None:
        held["counter"] = torch.tensor(PRECOPY_EMPTY, dtype=torch.int64, device=device)
        held["at"] = _now()
        say("precopy-check", "GDN_PRECOPY_CHECK armed: after the align pre-copy, every request whose running "
                             "column moved has the first 64 row scales of its destination state block read; a "
                             "block where fewer than half are finite and between 1e-12 and 1e3 was not written by "
                             "the copy (zeroed or another owner's bytes)")
    cpu = torch.from_numpy(np.asarray(input_batch.num_computed_tokens_np[:rows], dtype=np.int64))
    computed = getattr(state, "taaf_computed", None)
    staged = cpu.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else cpu.to(device)
    left = held["left"]
    held["left"] = 0
    _precopy_check_kernel[(rows, ctx.num_states)](
        input_batch.idx_mapping, state._mamba_src_col_gpu, state._mamba_state_idx_gpu, state.num_accepted_tokens_gpu,
        computed if computed is not None else staged, staged, ctx.block_table_ptrs, ctx.block_table_stride_req,
        ctx.state_base_addrs, ctx.state_block_strides, scale_offsets(ctx), ctx.state_conv_widths,
        ctx.state_group_indices, held["counter"], left, SCALES=64,
    )
    return precopy_tally()


def precopy_tally(force=False):
    held = PRECOPY_CHECK
    if held["counter"] is None or (not force and _now() - held["at"] < held["every_s"]):
        return None
    held["at"] = _now()
    failed, checked, row, slot, src, dst, block, gpu, cpu, accepted, left, index = (
        int(value) for value in held["counter"].tolist())
    held["counter"].copy_(torch.tensor(PRECOPY_EMPTY, dtype=torch.int64))
    held["failed"] += failed
    held["checked"] += checked
    first = (f"; the first: batch row {row}, slot {slot} last added as {SLOT_OWNERS.get(slot, 'unknown')}, "
             f"column {src} -> {dst}, state block {block} (Mamba state {index}), computed {gpu} on the GPU and "
             f"{cpu} on the host, num_accepted {accepted}, {left} request(s) left the batch that step"
             if row >= 0 else "")
    print(f"GDN_PRECOPY_CHECK: {failed} of {checked} moved state block(s) unwritten after the pre-copy since the "
          f"last tally ({held['failed']} of {held['checked']} since the first){first}", flush=True)
    return failed


def _now():
    import time

    return time.time()


_stock_unit_leave = unit_leave
_stock_unit_resume = unit_resume
_stock_unit_snapshot = unit_snapshot
_stock_unit_prepare = unit_prepare


def _counted_unit_leave(state, req_id):
    PRECOPY_CHECK["left"] += 1
    return _stock_unit_leave(state, req_id)


def _owned_unit_resume(state, slot, new_req_data):
    resumed = _stock_unit_resume(state, slot, new_req_data)
    SLOT_OWNERS[int(slot)] = new_req_data.req_id
    if SLOT_FRESH is not None:
        SLOT_FRESH[int(slot)] = int(int(new_req_data.num_computed_tokens) == 0)
    return resumed


def _settled_unit_snapshot(state, idx_mapping, num_sampled, num_computed_tokens):
    if SLOT_FRESH is not None and torch.is_tensor(idx_mapping):
        SLOT_FRESH.index_fill_(0, idx_mapping.clamp(min=0).long(), 0)
    return _stock_unit_snapshot(state, idx_mapping, num_sampled, num_computed_tokens)


def _checked_unit_prepare(state, input_batch, attn_groups, num_reqs, for_capture):
    precopy_check(state, input_batch, for_capture)
    return _stock_unit_prepare(state, input_batch, attn_groups, num_reqs, for_capture)


unit_leave = _counted_unit_leave
unit_resume = _owned_unit_resume
unit_snapshot = _settled_unit_snapshot
unit_prepare = _checked_unit_prepare


_SCALE_OFFSETS = {}


def ssm_full(layer):
    full = getattr(layer, "taaf_q8_full", None)
    return layer.kv_cache[1] if full is None else full


def scale_offsets(ctx):
    inner = ctx.state_inner_sizes
    held = _SCALE_OFFSETS.get(id(inner))
    if held is not None and held[0] is inner:
        return held[1]
    fused = __import__("sys").modules.get("vllm.taaf_gdn_fused")
    codes = fused.q8_codes_bytes() if hasattr(fused, "q8_codes_bytes") else None
    made = inner if codes is None else torch.where(ctx.state_conv_widths > 0, inner, torch.full_like(inner, codes))
    _SCALE_OFFSETS[id(inner)] = (inner, made)
    return made
