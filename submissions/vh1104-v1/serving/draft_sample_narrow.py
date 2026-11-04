import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import _DRAFT_NOISE_SALT, tl_rand32, tl_rand64

PREV = "_taaf_draft_sample_narrow_prev"
VIRGIN = -2
UNUSED = -1
CLEAR_BLOCK = 4096


def padded(count: int) -> int:
    return 1 << max(0, (int(count) - 1).bit_length())


@triton.jit
def _narrow_kernel(
    sampled_ptr,
    values_ptr,
    values_stride,
    ids_ptr,
    ids_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col_ptr,
    prev_ptr,
    prev_stride_0,
    prev_stride_1,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    count,
    CLEAR_BLOCK: tl.constexpr,
    PADDED: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    CACHE_DTYPE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, PADDED)
    live = lane < count
    ids = tl.load(ids_ptr + token_idx * ids_stride + lane, mask=live, other=0)
    logits = tl.load(values_ptr + token_idx * values_stride + lane, mask=live, other=float("-inf")).to(tl.float32)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)
    if PER_TOKEN_COL:
        col = tl.load(col_ptr + token_idx)
    else:
        col = tl.load(col_ptr)

    if is_valid_req:
        row_base = cache_ptr + req_state_idx * cache_stride_0 + col * cache_stride_1
        prev_base = prev_ptr + req_state_idx * prev_stride_0 + col * prev_stride_1
        before = tl.load(prev_base + lane)
        virgin = tl.load(prev_base) == -2
        tl.debug_barrier()
        if virgin:
            wipe = (tl.zeros((CLEAR_BLOCK,), tl.float32) - float("inf")).to(CACHE_DTYPE)
            for start in range(0, vocab_size, CLEAR_BLOCK):
                spot = start + tl.arange(0, CLEAR_BLOCK)
                tl.store(row_base + spot, wipe, mask=spot < vocab_size)
        else:
            tl.store(
                row_base + before.to(tl.int64),
                (tl.zeros((PADDED,), tl.float32) - float("inf")).to(CACHE_DTYPE),
                mask=before >= 0,
            )
        tl.debug_barrier()
        tl.store(row_base + ids, logits.to(CACHE_DTYPE), mask=live)
        tl.store(prev_base + lane, tl.where(live, ids.to(tl.int32), -1))

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    keys = ids.to(tl.int32)
    if temp != 0.0:
        logits = logits / temp
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        pos = pos + _DRAFT_NOISE_SALT
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(live, logits + gumbel_noise, float("-inf"))
    best = tl.max(logits, axis=0)
    winner = tl.min(tl.where(live & (logits == best), ids, 0x7FFFFFFFFFFFFFFF), axis=0)
    tl.store(sampled_ptr + token_idx, winner)


def prev_table(owner, cache: torch.Tensor, count: int) -> torch.Tensor:
    width = padded(count)
    held = vars(owner).get(PREV)
    shape = (cache.shape[0], cache.shape[1], width)
    if held is None or tuple(held.shape) != shape or held.device != cache.device:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TAAF_VLLM_DRAFT_SAMPLE_NARROW: the written-ids table was first asked for inside a "
                               "CUDA graph capture; an eager draft step must build it before any capture")
        held = torch.full(shape, VIRGIN, dtype=torch.int32, device=cache.device)
        vars(owner)[PREV] = held
    return held


def narrow_sample(prev, values, ids, cache, cache_col, idx_mapping, temperature, seeds, positions, use_fp64=False):
    rows, count = ids.shape
    out = torch.empty(rows, dtype=torch.int64, device=ids.device)
    idx_mapping = idx_mapping.contiguous()
    positions = positions.contiguous()
    cache_col = cache_col.contiguous()
    dtypes = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}
    _narrow_kernel[(rows,)](
        out,
        values,
        values.stride(0),
        ids,
        ids.stride(0),
        cache,
        cache.stride(0),
        cache.stride(1),
        cache_col,
        prev,
        prev.stride(0),
        prev.stride(1),
        idx_mapping,
        seeds,
        positions,
        temperature,
        cache.shape[-1],
        count,
        CLEAR_BLOCK=CLEAR_BLOCK,
        PADDED=prev.shape[-1],
        USE_FP64=use_fp64,
        PER_TOKEN_COL=cache_col.dim() > 0,
        CACHE_DTYPE=dtypes[cache.dtype],
    )
    return out
