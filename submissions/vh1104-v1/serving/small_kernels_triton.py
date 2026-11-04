import torch
import triton
import triton.language as tl

SEARCH_STEPS = 26
INVALIDATE_BLOCK = 512


@triton.jit
def _invalidate_kernel(dgi_ptr, dgi_stride, sorted_ptr, state_ptr, n_sorted, width, STEPS: tl.constexpr,
                       BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = cols < width
    row = tl.load(state_ptr + lane).to(tl.int64)
    where = dgi_ptr + row * dgi_stride + cols
    held = tl.load(where, mask=inside, other=-1)
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.zeros([BLOCK], dtype=tl.int32) + n_sorted
    for _ in range(STEPS):
        live = lo < hi
        mid = (lo + hi) >> 1
        probe = tl.load(sorted_ptr + mid, mask=live, other=0)
        below = live & (probe < held)
        lo = tl.where(below, mid + 1, lo)
        hi = tl.where(live & (probe >= held), mid, hi)
    at = tl.minimum(lo, n_sorted - 1)
    hit = tl.load(sorted_ptr + at, mask=inside, other=-2) == held
    tl.store(where, tl.full([BLOCK], -1, dtype=held.dtype), mask=inside & hit)


def invalidate_sorted(device_global_indices, sorted_slots, state_indices):
    lanes = state_indices.numel()
    if lanes == 0 or sorted_slots.numel() == 0:
        return
    width = device_global_indices.shape[1]
    _invalidate_kernel[(lanes, triton.cdiv(width, INVALIDATE_BLOCK))](
        device_global_indices, device_global_indices.stride(0), sorted_slots, state_indices, sorted_slots.numel(),
        width, STEPS=SEARCH_STEPS, BLOCK=INVALIDATE_BLOCK, num_warps=4)


@triton.jit
def _glue_kernel(columns_ptr, columns_stride, processed_ptr, processed_stride, table_ptr, table_s0, table_s1,
                 table_s2, ex_idx_ptr, ex_pos_ptr, sampled_ptr, draft_ptr, draft_s0, draft_s1, union_ptr, target_ptr,
                 local_ptr, req_at_ptr, step_at_ptr, live_ptr, req_ptr, drafts_ptr, moved_ptr, missed_ptr, rows,
                 steps, trash, K: tl.constexpr, TOP: tl.constexpr, TOP_P: tl.constexpr, K_P: tl.constexpr,
                 WIDTH: tl.constexpr, WIDTH_P: tl.constexpr):
    r = tl.program_id(0)
    req = tl.load(ex_idx_ptr + r).to(tl.int64)
    step = tl.load(ex_pos_ptr + r).to(tl.int64)
    live = (step < steps) & (req >= 0) & (req < trash)
    req_at = tl.where(live, req, trash)
    step_at = tl.where(live, step, 0)
    j = tl.arange(0, TOP_P)
    jm = j < TOP
    seen = tl.load(table_ptr + req_at * table_s0 + step_at * table_s1 + j * table_s2, mask=jm, other=-1).to(tl.int64)
    drafted = tl.where(live & (seen >= 0) & jm, seen, -1)
    kk = tl.arange(0, K_P)
    km = kk < K
    cols = tl.load(columns_ptr + r * columns_stride + kk, mask=km, other=-1).to(tl.int64)
    match = (drafted[:, None] == cols[None, :]) & km[None, :]
    shared = tl.max(match.to(tl.int32), axis=1) > 0
    first = tl.min(tl.where(match, kk[None, :], K_P), axis=1)
    local = tl.where(shared, first.to(tl.int64), K + j.to(tl.int64))
    local = tl.where(drafted >= 0, local, -1)
    tl.store(local_ptr + r * TOP + j, local, mask=jm)
    w = tl.arange(0, WIDTH_P)
    wm = w < WIDTH
    tl.store(union_ptr + r * WIDTH + kk, cols, mask=km)
    tl.store(union_ptr + r * WIDTH + K + j, tl.where(shared | (drafted < 0), -1, drafted), mask=jm)
    tl.store(union_ptr + r * WIDTH + w, tl.full([WIDTH_P], -1, dtype=tl.int64), mask=w == WIDTH - 1)
    tv = tl.load(processed_ptr + r * processed_stride + w, mask=w < K, other=float("-inf"))
    tl.store(target_ptr + r * WIDTH + w, tv, mask=wm)
    tl.store(req_at_ptr + r, req_at)
    tl.store(step_at_ptr + r, step_at)
    tl.store(live_ptr + r, live.to(tl.int8))
    tl.store(req_ptr + r, tl.where(live, req, 0))
    got = tl.load(draft_ptr + tl.where(live, req, 0) * draft_s0 + step_at * draft_s1 + tl.where(drafted >= 0, drafted, 0),
                  mask=drafted >= 0, other=0.0).to(tl.float32)
    cell = ((req_at * steps + step_at) * WIDTH) + tl.where(local >= 0, local, WIDTH - 1)
    tl.store(drafts_ptr + cell, got, mask=jm & (drafted >= 0))
    if r == 0:
        head = tl.load(sampled_ptr).to(tl.int64)
        tl.store(moved_ptr, tl.where(head < 0, head, 0).to(moved_ptr.dtype.element_ty))
    if r + 1 < rows:
        nxt = tl.load(sampled_ptr + r + 1).to(tl.int64)
        hit = drafted == nxt
        found = tl.max(hit.to(tl.int32), axis=0) > 0
        hit_at = tl.min(tl.where(hit, j, TOP_P), axis=0)
        at_found = tl.sum(tl.where(j == hit_at, local, 0), axis=0)
        in_top = (cols == nxt) & km
        any_top = tl.max(in_top.to(tl.int32), axis=0) > 0
        top_at = tl.min(tl.where(in_top, kk, K_P), axis=0)
        at = tl.where(found, at_found, tl.where(any_top, top_at.to(tl.int64), WIDTH - 1))
        next_idx = tl.load(ex_idx_ptr + r + 1)
        next_pos = tl.load(ex_pos_ptr + r + 1).to(tl.int64)
        same = (next_idx == tl.load(ex_idx_ptr + r)) & (next_pos == step + 1)
        moved = tl.where(nxt < 0, nxt, tl.where(same, at, 0))
        tl.store(moved_ptr + r + 1, moved.to(moved_ptr.dtype.element_ty))
        missed = (~found) & same & (nxt >= 0) & live
        tl.atomic_add(missed_ptr, missed.to(tl.int64))


def glue(columns, processed, ids_table, ex_idx, ex_pos, draft_logits, draft_sampled, steps, k, missed):
    rows = columns.shape[0]
    top = int(ids_table.shape[2])
    trash = int(ids_table.shape[0]) - 1
    width = k + top + 1
    device = columns.device
    union = torch.empty((rows, width), dtype=columns.dtype, device=device)
    target = torch.empty((rows, width), dtype=processed.dtype, device=device)
    local = torch.empty((rows, top), dtype=torch.int64, device=device)
    req_at = torch.empty(rows, dtype=torch.int64, device=device)
    step_at = torch.empty(rows, dtype=torch.int64, device=device)
    live = torch.empty(rows, dtype=torch.bool, device=device)
    req = torch.empty(rows, dtype=torch.int64, device=device)
    drafts = torch.full((trash + 1, steps, width), float("-inf"), dtype=torch.float32, device=device)
    moved = torch.empty(rows, dtype=draft_sampled.dtype, device=device)
    _glue_kernel[(rows,)](
        columns, columns.stride(0), processed, processed.stride(0), ids_table, ids_table.stride(0),
        ids_table.stride(1), ids_table.stride(2), ex_idx, ex_pos, draft_sampled, draft_logits,
        draft_logits.stride(0), draft_logits.stride(1), union, target, local, req_at, step_at, live.view(torch.int8),
        req, drafts, moved, missed, rows, steps, trash, K=k, TOP=top, TOP_P=triton.next_power_of_2(top),
        K_P=triton.next_power_of_2(k), WIDTH=width, WIDTH_P=triton.next_power_of_2(width), num_warps=2)
    return union, target, local, req_at, step_at, live, req, drafts, moved, width, top, trash


@triton.jit
def _gemma_norm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, new_res_ptr, x_stride, res_stride, width, eps,
                       HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    value = tl.load(x_ptr + row * x_stride + cols, mask=inside, other=0.0).to(tl.float32)
    if HAS_RES:
        value = value + tl.load(res_ptr + row * res_stride + cols, mask=inside, other=0.0).to(tl.float32)
        tl.store(new_res_ptr + row * width + cols, value.to(new_res_ptr.dtype.element_ty), mask=inside)
    mean = tl.sum(value * value, axis=0) / width
    scale = tl.math.rsqrt(mean + eps)
    weight = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32) + 1.0
    tl.store(out_ptr + row * width + cols, ((value * scale) * weight).to(out_ptr.dtype.element_ty), mask=inside)


def gemma_norm(x, residual, weight, eps):
    width = x.shape[-1]
    flat = x.view(-1, width)
    rows = flat.shape[0]
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    new_residual = torch.empty(x.shape, device=x.device, dtype=x.dtype) if residual is not None else None
    if rows == 0:
        return out, new_residual
    block = triton.next_power_of_2(width)
    res = residual.view(-1, width) if residual is not None else flat
    _gemma_norm_kernel[(rows,)](
        flat, res, weight, out, new_residual if new_residual is not None else out, flat.stride(0), res.stride(0),
        width, eps, HAS_RES=residual is not None, BLOCK=block, num_warps=8 if block >= 4096 else 4)
    return out, new_residual
