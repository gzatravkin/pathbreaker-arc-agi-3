import importlib
import os

import torch

from vllm.triton_utils import tl, triton

COMPACT_ENV = "TAAF_VLLM_QSA_INDEX_COMPACT"
INDEXER_OPS_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
MARK = "_taaf_index_compact"
_SAID = set()


def announce(event, line):
    if event in _SAID:
        return
    _SAID.add(event)
    print(line, flush=True)


def compact_enabled():
    said = str(os.environ.get(COMPACT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise ValueError(f"{COMPACT_ENV}={said!r} is not '', '0' or '1'")
    return said == "1"


@triton.jit(do_not_specialize=["num_rows", "query_offset", "num_requests"])
def _compact_prefill_kernel(
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
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
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
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


def prefill_logits_compact(q, k_cache, page_table, query_start_loc, visible_blocks, max_query_len, logits_width,
                           query_offset, num_queries, config=None):
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((num_queries, logits_width), dtype=torch.float32, device=q.device)
    if k_cache.dtype == torch.float8_e4m3fn:
        tile_r, stages, num_warps = 32, 2, 8
    else:
        tile_r, stages, num_warps = 64, 2, 4
    block_n, k_tiles = 64, 16
    if config:
        tile_r, block_n, k_tiles, stages, num_warps = config
    num_requests = page_table.shape[0]
    grid = (triton.cdiv(num_queries, tile_r) + num_requests, triton.cdiv(logits_width, block_n * k_tiles))
    _compact_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        page_table.shape[1],
        num_requests,
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=tile_r,
        BLOCK_N=block_n,
        K_TILES=k_tiles,
        STAGES=stages,
        BLOCK_REQ=triton.next_power_of_2(num_requests + 1),
        num_warps=num_warps,
    )
    return logits


def install():
    if not compact_enabled():
        return False
    module = importlib.import_module(INDEXER_OPS_MODULE)
    if not getattr(module, MARK, False):
        module._prefill_logits = prefill_logits_compact
        setattr(module, MARK, True)
    announce("installed", f"QSA_INDEX_COMPACT armed: the QSA indexer's prefill logits launch one program per row tile "
                          f"that holds prefill rows (row tiles + requests) instead of requests x max_query_len tiles; "
                          f"same per-tile code, byte-identical logits ({COMPACT_ENV})")
    return True
