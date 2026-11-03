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

ENV = "TAAF_VLLM_MOE_TAAF_ROWS"
ADAPTIVE_ENV = "TAAF_VLLM_MOE_TAAF_ADAPTIVE"
COUNTER = "moe_taaf"
TIMERS = "vllm.taaf_step_timers"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
GROUP = 128
KT_GROUP = GROUP // 16
CHUNK = 64
TILE_K = 16
ROW_TILE = 4
SERVE_WARPS = 2
SERVE_STAGES = 3
SERVE_UNROLL = 0
WIDE_PAIRS_PER_EXPERT = 40
CHECK_ROWS = 16
CHECK_ROWS_AFTER = 4
CHECK_SEED = 11
CHECK_MARLIN_CEILING = 0.05
CHECK_GAP_CEILING = 0.05
CHECK_MEAN_TIMES = 2
CHECK_MEAN_FLOOR = 1e-4
CHECK_MAX_TIMES = 4
CHECK_MAX_FLOOR = 1e-3
OFF = ("", "0", "off", "false", "no")
ADAPTIVE_DEFAULT = ((16, SERVE_WARPS, SERVE_STAGES), (32, SERVE_WARPS, SERVE_STAGES), (64, SERVE_WARPS, SERVE_STAGES))
ADAPTIVE_SIZES = (16, 32, 64, 128)
ALIGN_BLOCK = 1024
POST_STRIDE = 4
WIDEST = 1 << 30


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def marlin_perm():
    entries = []
    for lane in range(32):
        col = lane // 4
        lane_entries = []
        for block in (0, 1):
            for row in (2 * (lane % 4), 2 * (lane % 4) + 1, 2 * (lane % 4 + 4), 2 * (lane % 4 + 4) + 1):
                lane_entries.append(16 * row + col + 8 * block)
        for tile in range(4):
            entries.extend(value + 256 * tile for value in lane_entries)
    perm = torch.tensor(entries, dtype=torch.int64).reshape(-1, 8)
    return perm[:, torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])].reshape(-1)


def marlin_pack(codes):
    k, n = codes.shape
    tiles = codes.to(torch.int64).reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n * 16)
    perm = marlin_perm()
    tiles = tiles.reshape(-1, perm.numel())[:, perm].reshape(k // 16, n * 16)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = (tiles.reshape(k // 16, n * 2, 8) << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def marlin_scales(scales):
    groups, n = scales.shape
    base = torch.arange(64).reshape(8, 8).t().reshape(-1)
    return scales.reshape(-1, 64)[:, base].reshape(groups, n).contiguous()


def dequant_reference(codes, scales):
    k, n = codes.shape
    group = k // scales.shape[0]
    return (codes.float() - 8.0) * scales.float().repeat_interleave(group, dim=0)


def permuted_position(index, width):
    group, inner = index // GROUP, index % GROUP
    tile, within = inner // 16, inner % 16
    high, low = within // 8, within % 8
    row, parity = low // 2, low % 2
    return group * GROUP + (2 * high + parity) * 32 + tile * 4 + row


def activation_permutation(width):
    return permuted_position(torch.arange(width), width)


@triton.jit
def _pos(index):
    group = index // 128
    inner = index % 128
    tile = inner // 16
    within = inner % 16
    high = within // 8
    low = within % 8
    return group * 128 + (2 * high + (low % 2)) * 32 + tile * 4 + low // 2


@triton.jit(do_not_specialize=["rows"])
def _permute_kernel(x_ptr, out_ptr, K, stride_x, tally_ptr, rows, BLOCK: tl.constexpr, TALLY: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK + tl.arange(0, BLOCK)
    live = offs < K
    value = tl.load(x_ptr + row * stride_x + offs, mask=live, other=0.0)
    tl.store(out_ptr + row * K + _pos(offs), value, mask=live)
    if TALLY:
        if row + col == 0:
            tl.atomic_add(tally_ptr, 1)
            tl.atomic_add(tally_ptr + 1, rows)


@triton.jit
def _deq(words, shift):
    code = ((words >> shift) & 15) | 0x4300
    return code.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["n_pairs"])
def _gate_up_kernel(xp_ptr, w_ptr, s_ptr, sorted_ptr, eids_ptr, npost_ptr, out_ptr,
                    n_pairs, TOPK: tl.constexpr, K: tl.constexpr, INTER: tl.constexpr,
                    BM: tl.constexpr, NC: tl.constexpr, ORDER: tl.constexpr, SU: tl.constexpr):
    first = tl.program_id(0)
    second = tl.program_id(1)
    pid_b = first * (1 - ORDER) + second * ORDER
    pid_c = second * (1 - ORDER) + first * ORDER
    if pid_b * BM >= tl.load(npost_ptr):
        return
    e = tl.load(eids_ptr + pid_b)
    if e < 0:
        return
    NOUT: tl.constexpr = 2 * INTER
    R: tl.constexpr = NOUT * 2
    GROUPS: tl.constexpr = K // 128
    UP: tl.constexpr = (INTER // 64) * 128
    rm = tl.arange(0, BM)
    rows = tl.load(sorted_ptr + pid_b * BM + rm)
    live = rows < n_pairs
    tok = (rows // TOPK).to(tl.int64)
    rk = tl.arange(0, 32)
    rq = tl.arange(0, NC * 32)
    nc = rq // 32
    cc = (rq % 32) // 4
    jj = rq % 4
    chunk0 = pid_c * NC
    woff = ((rk // 4)[:, None] * R + ((chunk0 + nc) * 128 + cc * 16 + jj)[None, :] + (rk % 4)[:, None] * 4)
    e64 = e.to(tl.int64)
    w_base = w_ptr + e64 * ((K // 16) * R)
    s_base = s_ptr + e64 * (GROUPS * NOUT)
    scol = (chunk0 + nc) * 64 + 8 * cc + 2 * jj
    acc_g0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    acc_g1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    acc_u0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    acc_u1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    if SU == 0:
        for g in range(GROUPS):
            wg = tl.load(w_base + g * 8 * R + woff)
            wu = tl.load(w_base + g * 8 * R + UP + woff)
            sg = s_base + g * NOUT + scol
            su = sg + INTER
            sg0 = tl.load(sg)[None, :]
            sg1 = tl.load(sg + 1)[None, :]
            su0 = tl.load(su)[None, :]
            su1 = tl.load(su + 1)[None, :]
            for s in range(4):
                a = tl.load(xp_ptr + tok[:, None] * K + (g * 128 + s * 32 + rk)[None, :], mask=live[:, None],
                            other=0.0)
                sh = 4 * ((s // 2) + 4 * (s % 2))
                acc_g0 = tl.dot(a, (_deq(wg, sh) - 136.0) * sg0, acc_g0)
                acc_g1 = tl.dot(a, (_deq(wg, sh + 8) - 136.0) * sg1, acc_g1)
                acc_u0 = tl.dot(a, (_deq(wu, sh) - 136.0) * su0, acc_u0)
                acc_u1 = tl.dot(a, (_deq(wu, sh + 8) - 136.0) * su1, acc_u1)
    else:
        for g in range(GROUPS):
            t_g0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            t_g1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            t_u0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            t_u1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            xsum = tl.zeros((BM,), dtype=tl.float32)
            wg = tl.load(w_base + g * 8 * R + woff)
            wu = tl.load(w_base + g * 8 * R + UP + woff)
            for r in range(4 // SU):
                for su in tl.static_range(SU):
                    s = r * SU + su
                    a = tl.load(xp_ptr + tok[:, None] * K + (g * 128 + s * 32 + rk)[None, :], mask=live[:, None],
                                other=0.0)
                    xsum += tl.sum(a.to(tl.float32), axis=1)
                    t_g0 = tl.dot(a, _deq(wg, 4 * ((s // 2) + 4 * (s % 2))), t_g0)
                    t_g1 = tl.dot(a, _deq(wg, 4 * ((s // 2) + 4 * (s % 2) + 2)), t_g1)
                    t_u0 = tl.dot(a, _deq(wu, 4 * ((s // 2) + 4 * (s % 2))), t_u0)
                    t_u1 = tl.dot(a, _deq(wu, 4 * ((s // 2) + 4 * (s % 2) + 2)), t_u1)
            bias = (xsum * 136.0)[:, None]
            sg = s_base + g * NOUT + scol
            su = sg + INTER
            acc_g0 += (t_g0 - bias) * tl.load(sg).to(tl.float32)[None, :]
            acc_g1 += (t_g1 - bias) * tl.load(sg + 1).to(tl.float32)[None, :]
            acc_u0 += (t_u0 - bias) * tl.load(su).to(tl.float32)[None, :]
            acc_u1 += (t_u1 - bias) * tl.load(su + 1).to(tl.float32)[None, :]
    base_n = (chunk0 + nc) * 64 + 16 * jj + cc
    row_ptr = out_ptr + rows.to(tl.int64)[:, None] * INTER
    g0 = acc_g0.to(tl.bfloat16).to(tl.float32)
    u0 = acc_u0.to(tl.bfloat16).to(tl.float32)
    s0 = (g0 / (1.0 + tl.exp(-g0))).to(tl.bfloat16).to(tl.float32)
    act0 = (s0 * u0).to(tl.bfloat16)
    tl.store(row_ptr + _pos(base_n)[None, :], act0, mask=live[:, None])
    g1 = acc_g1.to(tl.bfloat16).to(tl.float32)
    u1 = acc_u1.to(tl.bfloat16).to(tl.float32)
    s1 = (g1 / (1.0 + tl.exp(-g1))).to(tl.bfloat16).to(tl.float32)
    act1 = (s1 * u1).to(tl.bfloat16)
    tl.store(row_ptr + _pos(base_n + 8)[None, :], act1, mask=live[:, None])


@triton.jit(do_not_specialize=["n_pairs"])
def _down_kernel(a_ptr, w_ptr, s_ptr, sorted_ptr, eids_ptr, npost_ptr, topw_ptr, out_ptr,
                 n_pairs, K: tl.constexpr, NOUT: tl.constexpr, BM: tl.constexpr, NC: tl.constexpr,
                 ORDER: tl.constexpr, SU: tl.constexpr):
    first = tl.program_id(0)
    second = tl.program_id(1)
    pid_b = first * (1 - ORDER) + second * ORDER
    pid_c = second * (1 - ORDER) + first * ORDER
    if pid_b * BM >= tl.load(npost_ptr):
        return
    e = tl.load(eids_ptr + pid_b)
    if e < 0:
        return
    R: tl.constexpr = NOUT * 2
    GROUPS: tl.constexpr = K // 128
    rm = tl.arange(0, BM)
    rows = tl.load(sorted_ptr + pid_b * BM + rm)
    live = rows < n_pairs
    rows64 = rows.to(tl.int64)
    rk = tl.arange(0, 32)
    rq = tl.arange(0, NC * 32)
    nc = rq // 32
    cc = (rq % 32) // 4
    jj = rq % 4
    chunk0 = pid_c * NC
    woff = ((rk // 4)[:, None] * R + ((chunk0 + nc) * 128 + cc * 16 + jj)[None, :] + (rk % 4)[:, None] * 4)
    e64 = e.to(tl.int64)
    w_base = w_ptr + e64 * ((K // 16) * R)
    s_base = s_ptr + e64 * (GROUPS * NOUT)
    scol = (chunk0 + nc) * 64 + 8 * cc + 2 * jj
    acc0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    acc1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
    if SU == 0:
        for g in range(GROUPS):
            w = tl.load(w_base + g * 8 * R + woff)
            sp = s_base + g * NOUT + scol
            s0 = tl.load(sp)[None, :]
            s1 = tl.load(sp + 1)[None, :]
            for s in range(4):
                a = tl.load(a_ptr + rows64[:, None] * K + (g * 128 + s * 32 + rk)[None, :], mask=live[:, None],
                            other=0.0)
                sh = 4 * ((s // 2) + 4 * (s % 2))
                acc0 = tl.dot(a, (_deq(w, sh) - 136.0) * s0, acc0)
                acc1 = tl.dot(a, (_deq(w, sh + 8) - 136.0) * s1, acc1)
    else:
        for g in range(GROUPS):
            t0 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            t1 = tl.zeros((BM, NC * 32), dtype=tl.float32)
            xsum = tl.zeros((BM,), dtype=tl.float32)
            w = tl.load(w_base + g * 8 * R + woff)
            for r in range(4 // SU):
                for su in tl.static_range(SU):
                    s = r * SU + su
                    a = tl.load(a_ptr + rows64[:, None] * K + (g * 128 + s * 32 + rk)[None, :], mask=live[:, None],
                                other=0.0)
                    xsum += tl.sum(a.to(tl.float32), axis=1)
                    t0 = tl.dot(a, _deq(w, 4 * ((s // 2) + 4 * (s % 2))), t0)
                    t1 = tl.dot(a, _deq(w, 4 * ((s // 2) + 4 * (s % 2) + 2)), t1)
            bias = (xsum * 136.0)[:, None]
            sp = s_base + g * NOUT + scol
            acc0 += (t0 - bias) * tl.load(sp).to(tl.float32)[None, :]
            acc1 += (t1 - bias) * tl.load(sp + 1).to(tl.float32)[None, :]
    weight = tl.load(topw_ptr + rows, mask=live, other=0.0)[:, None]
    base_n = (chunk0 + nc) * 64 + 16 * jj + cc
    row_ptr = out_ptr + rows64[:, None] * NOUT
    tl.store(row_ptr + base_n[None, :], (acc0 * weight).to(tl.bfloat16), mask=live[:, None])
    tl.store(row_ptr + (base_n + 8)[None, :], (acc1 * weight).to(tl.bfloat16), mask=live[:, None])


@_marked("moe_taaf_owner")
def align_reference(topk_ids, block, experts):
    flat = topk_ids.reshape(-1).long()
    pairs = flat.numel()
    valid = (flat >= 0) & (flat < experts)
    capacity = pairs + experts * (block - 1)
    blocks = -(-capacity // block)
    counts = torch.zeros(experts, dtype=torch.long, device=flat.device).scatter_add_(
        0, flat[valid], torch.ones_like(flat[valid]))
    padded = (counts + block - 1) // block * block
    ends = torch.cumsum(padded, 0)
    order = torch.argsort(torch.where(valid, flat, experts), stable=True)[:int(valid.sum())]
    ranked = flat[order]
    rank = torch.arange(order.numel(), device=flat.device) - (torch.cumsum(counts, 0) - counts)[ranked]
    sorted_ids = torch.full((blocks * block,), pairs, dtype=torch.int32, device=flat.device)
    sorted_ids[(ends - padded)[ranked] + rank] = order.to(torch.int32)
    owner = torch.searchsorted(ends, torch.arange(blocks, device=flat.device) * block, right=True)
    expert_ids = owner.clamp(max=experts - 1).to(torch.int32)
    return sorted_ids, expert_ids, ends[-1:].to(torch.int32)


def align(topk_ids, block, experts):
    if topk_ids.device.type != "cuda":
        return align_reference(topk_ids, block, experts)
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    except ImportError:
        return align_reference(topk_ids, block, experts)
    return moe_align_block_size(topk_ids, block, experts, None, ignore_invalid_experts=True)[:3]


def permute_activations(x, out=None, tally=None):
    rows, k = x.shape
    if out is None:
        out = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    block = 1024
    _permute_kernel[(rows, triton.cdiv(k, block))](x, out, k, x.stride(0), out if tally is None else tally, rows,
                                                   BLOCK=block, TALLY=tally is not None)
    return out


def launch_gate_up(xp, w1, s1, aligned, act, topk, block, chunks, warps, stages, order=0, unroll=4):
    sorted_ids, expert_ids, post = aligned
    rows, k = xp.shape
    inter = w1.shape[2] // 4
    grid = ((inter // 64) // chunks, expert_ids.numel()) if order else (expert_ids.numel(), (inter // 64) // chunks)
    _gate_up_kernel[grid](
        xp, w1, s1, sorted_ids, expert_ids, post, act, rows * topk, TOPK=topk, K=k, INTER=inter, BM=block,
        NC=chunks, ORDER=order, SU=unroll, num_warps=warps, num_stages=stages)


def launch_down(act, w2, s2, aligned, weights, down, n_pairs, block, chunks, warps, stages, order=0, unroll=4):
    sorted_ids, expert_ids, post = aligned
    inter = act.shape[1]
    k = down.shape[1]
    grid = ((k // 64) // chunks, expert_ids.numel()) if order else (expert_ids.numel(), (k // 64) // chunks)
    _down_kernel[grid](
        act, w2, s2, sorted_ids, expert_ids, post, weights, down, n_pairs, K=inter, NOUT=k, BM=block, NC=chunks,
        ORDER=order, SU=unroll, num_warps=warps, num_stages=stages)


def moe_forward(x, w1, s1, w2, s2, topk_weights, topk_ids, block=64, chunks1=2, chunks2=2, warps=4, stages=2,
                aligned=None, workspace=None, order=0, unroll=4, tally=None):
    rows, k = x.shape
    topk = topk_ids.shape[1]
    experts = w1.shape[0]
    inter = w1.shape[2] // 4
    pairs = rows * topk
    if aligned is None:
        aligned = align(topk_ids, block, experts)
    if workspace is None:
        xp = torch.empty((rows, k), dtype=x.dtype, device=x.device)
        act = torch.empty((pairs, inter), dtype=x.dtype, device=x.device)
        down = torch.empty((pairs, k), dtype=x.dtype, device=x.device)
    else:
        xp, act, down = workspace
    permute_activations(x, xp, tally)
    launch_gate_up(xp, w1, s1, aligned, act, topk, block, chunks1, warps, stages, order, unroll)
    launch_down(act, w2, s2, aligned, topk_weights.reshape(-1), down, pairs, block, chunks2, warps, stages, order, unroll)
    return down.view(rows, topk, k).sum(dim=1)


@triton.jit(do_not_specialize=["n_pairs"])
def _count_kernel(ids_ptr, counts_ptr, n_pairs, E: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(ids_ptr + offs, mask=offs < n_pairs, other=-1)
    ok = (offs < n_pairs) & (e >= 0) & (e < E)
    tl.atomic_add(counts_ptr + e, 1, mask=ok)


@triton.jit(do_not_specialize=["k", "bm", "lo", "hi", "row_off", "blk_off"])
def _layout_kernel(counts_ptr, start_ptr, eids_ptr, post_ptr, k, bm, lo, hi, row_off, blk_off, E: tl.constexpr,
                   EB: tl.constexpr, POST: tl.constexpr):
    idx = tl.arange(0, EB)
    live = idx < E
    c = tl.load(counts_ptr + idx, mask=live, other=0)
    member = live & (c > lo) & (c <= hi)
    nblk = tl.where(member, (c + bm - 1) // bm, 0)
    first = tl.cumsum(nblk, axis=0) - nblk
    tl.store(start_ptr + idx, row_off + first * bm, mask=member)
    most = tl.max(nblk, axis=0)
    for j in range(0, most):
        tl.store(eids_ptr + blk_off + first + j, idx, mask=nblk > j)
    tl.store(post_ptr + k * POST, tl.sum(nblk, axis=0) * bm)


@triton.jit(do_not_specialize=["n_pairs"])
def _scatter_kernel(ids_ptr, start_ptr, fill_ptr, sorted_ptr, n_pairs, E: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(ids_ptr + offs, mask=offs < n_pairs, other=-1)
    ok = (offs < n_pairs) & (e >= 0) & (e < E)
    rank = tl.atomic_add(fill_ptr + e, 1, mask=ok)
    base = tl.load(start_ptr + e, mask=ok, other=0)
    tl.store(sorted_ptr + base + rank, offs, mask=ok)


def adaptive_plan(pairs, experts, sizes):
    caps, rows, blocks, lows = [], [], [], []
    low = 0
    for index, size in enumerate(sizes):
        top = index == len(sizes) - 1
        members = max(1, min(experts, pairs // (low + 1)))
        cap = members + ((pairs + size - 1) // size if top else 0)
        rows.append(sum(c * s for c, s in zip(caps, sizes)))
        blocks.append(sum(caps))
        caps.append((cap + 3) // 4 * 4)
        lows.append(low)
        low = size
    return caps, rows, blocks, lows


def align_adaptive_reference(topk_ids, sizes, experts):
    flat = topk_ids.reshape(-1).tolist()
    pairs = len(flat)
    caps, rows, blocks, lows = adaptive_plan(pairs, experts, sizes)
    counts = [0] * experts
    for e in flat:
        if 0 <= e < experts:
            counts[e] += 1
    sorted_ids = torch.full((sum(c * s for c, s in zip(caps, sizes)),), pairs, dtype=torch.int32)
    expert_ids = torch.full((sum(caps),), -1, dtype=torch.int32)
    post = torch.zeros(len(sizes) * POST_STRIDE, dtype=torch.int32)
    start = [0] * experts
    for index, size in enumerate(sizes):
        top = index == len(sizes) - 1
        used = 0
        for e, c in enumerate(counts):
            if c > lows[index] and (top or c <= size):
                start[e] = rows[index] + used * size
                for j in range((c + size - 1) // size):
                    expert_ids[blocks[index] + used + j] = e
                used += (c + size - 1) // size
        post[index * POST_STRIDE] = used * size
    fill = [0] * experts
    for i, e in enumerate(flat):
        if 0 <= e < experts:
            sorted_ids[start[e] + fill[e]] = i
            fill[e] += 1
    return sorted_ids, expert_ids, post


def align_adaptive(topk_ids, sizes, experts):
    sizes = tuple(sizes)
    if topk_ids.device.type != "cuda" and not INTERPRETED:
        return align_adaptive_reference(topk_ids, sizes, experts)
    ids = topk_ids.reshape(-1)
    pairs = ids.numel()
    caps, rows, blocks, lows = adaptive_plan(pairs, experts, sizes)
    device = ids.device
    state = torch.zeros(2 * experts, dtype=torch.int32, device=device)
    counts, fill = state[:experts], state[experts:]
    sorted_ids = torch.full((sum(c * s for c, s in zip(caps, sizes)),), pairs, dtype=torch.int32, device=device)
    expert_ids = torch.empty(sum(caps), dtype=torch.int32, device=device)
    post = torch.empty(len(sizes) * POST_STRIDE, dtype=torch.int32, device=device)
    start = torch.empty(experts, dtype=torch.int32, device=device)
    grid = (triton.cdiv(pairs, ALIGN_BLOCK),)
    _count_kernel[grid](ids, counts, pairs, E=experts, BLOCK=ALIGN_BLOCK)
    for index, size in enumerate(sizes):
        top = index == len(sizes) - 1
        _layout_kernel[(1,)](counts, start, expert_ids, post, index, size, lows[index], WIDEST if top else size,
                             rows[index], blocks[index], E=experts, EB=triton.next_power_of_2(experts),
                             POST=POST_STRIDE)
    _scatter_kernel[grid](ids, start, fill, sorted_ids, pairs, E=experts, BLOCK=ALIGN_BLOCK)
    return sorted_ids, expert_ids, post


def adaptive_views(aligned, sizes, pairs, experts):
    sorted_ids, expert_ids, post = aligned
    caps, rows, blocks, _ = adaptive_plan(pairs, experts, sizes)
    return [(sorted_ids[rows[i]:rows[i] + caps[i] * size], expert_ids[blocks[i]:blocks[i] + caps[i]],
             post[i * POST_STRIDE:i * POST_STRIDE + 1]) for i, size in enumerate(sizes)]


def launch_adaptive(xp, act, down, w1, s1, w2, s2, topk_weights, aligned, classes, topk, experts):
    pairs = xp.shape[0] * topk
    sizes = tuple(size for size, _, _ in classes)
    weights = topk_weights.reshape(-1)
    for view, size, (_, warps, stages) in reversed(list(zip(adaptive_views(aligned, sizes, pairs, experts), sizes,
                                                            classes))):
        launch_gate_up(xp, w1, s1, view, act, topk, size, 1, warps, stages, 0, SERVE_UNROLL)
        launch_down(act, w2, s2, view, weights, down, pairs, size, 1, warps, stages, 0, SERVE_UNROLL)


def adaptive_forward(x, w1, s1, w2, s2, topk_weights, topk_ids, classes, aligned=None, workspace=None, tally=None):
    rows, k = x.shape
    topk = topk_ids.shape[1]
    experts = w1.shape[0]
    inter = w1.shape[2] // 4
    pairs = rows * topk
    if aligned is None:
        aligned = align_adaptive(topk_ids, tuple(size for size, _, _ in classes), experts)
    if workspace is None:
        xp = torch.empty((rows, k), dtype=x.dtype, device=x.device)
        act = torch.empty((pairs, inter), dtype=x.dtype, device=x.device)
        down = torch.empty((pairs, k), dtype=x.dtype, device=x.device)
    else:
        xp, act, down = workspace
    permute_activations(x, xp, tally)
    launch_adaptive(xp, act, down, w1, s1, w2, s2, topk_weights, aligned, classes, topk, experts)
    return down.view(rows, topk, k).sum(dim=1)


def moe_reference(x, codes1, scales1, codes2, scales2, topk_weights, topk_ids):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    inter = codes1.shape[2] // 2
    for r in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[r, j])
            h = (x[r].float() @ dequant_reference(codes1[e], scales1[e])).to(torch.bfloat16).float()
            gate = torch.nn.functional.silu(h[:inter]).to(torch.bfloat16).float()
            act = (gate * h[inter:]).to(torch.bfloat16).float()
            y = (act @ dequant_reference(codes2[e], scales2[e])) * topk_weights[r, j]
            out[r] += y.to(torch.bfloat16).float()
    return out


_PERMS = {}


def _inverse(perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse


def _unperm(device):
    key = str(device)
    if key not in _PERMS:
        base = torch.arange(64).reshape(8, 8).t().reshape(-1)
        _PERMS[key] = (_inverse(marlin_perm()).to(device), _inverse(base).to(device))
    return _PERMS[key]


def marlin_unpack(words, k, n):
    weights, _ = _unperm(words.device)
    raw = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=words.device) * 4
    tiles = ((raw.unsqueeze(-1) >> shifts) & 15).reshape(k // 16, n * 16)
    tiles = tiles.reshape(-1, weights.numel())[:, weights].reshape(k // 16, n * 16)
    return tiles.reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)


def marlin_unscales(stored):
    _, scales = _unperm(stored.device)
    groups, n = stored.shape
    return stored.reshape(-1, 64)[:, scales].reshape(groups, n)


def marlin_dequant(words, stored, k, n):
    scales = marlin_unscales(stored).float()
    return (marlin_unpack(words, k, n).float() - 8.0) * scales.repeat_interleave(k // scales.shape[0], dim=0)


def reference_rows(x, w1, s1, w2, s2, weights, ids, picks):
    count = w1.shape[0]
    k = x.shape[1]
    inter = w2.shape[1] * 16
    chosen = ids[picks].long()
    hidden = x[picks].float()
    scaled = weights[picks].float()
    out = torch.zeros((picks.numel(), k), dtype=torch.float32, device=x.device)
    for expert in torch.unique(chosen[chosen < count]).tolist():
        at, slots = torch.nonzero(chosen == expert, as_tuple=True)
        mid = hidden[at] @ marlin_dequant(w1[expert], s1[expert], k, 2 * inter)
        act = torch.nn.functional.silu(mid[:, :inter]) * mid[:, inter:]
        down = act @ marlin_dequant(w2[expert], s2[expert], inter, k)
        out.index_add_(0, at, down * scaled[at, slots].unsqueeze(1))
    return out


def parse_window(said):
    text = str(said or "").strip().lower()
    if text in OFF:
        return None
    low, sep, high = text.partition(":")
    if not low.isdigit() or (sep and not high.isdigit()):
        raise ValueError(f"{ENV} must be <min rows> or <min rows>:<max rows>, found {said!r}")
    first, last = int(low), (int(high) if sep else WIDEST)
    if first < 1 or last < first:
        raise ValueError(f"{ENV} needs 1 <= min rows <= max rows, found {said!r}")
    return first, last


def parse_adaptive(said):
    text = str(said or "").strip().lower()
    if text in OFF:
        return None
    if text in ("1", "on", "true", "yes"):
        return ADAPTIVE_DEFAULT
    sizes = []
    for part in text.split(","):
        if not part.strip().isdigit() or int(part) not in ADAPTIVE_SIZES:
            raise ValueError(f"{ADAPTIVE_ENV} must be 1 or a comma list of ascending tile rows from "
                             f"{ADAPTIVE_SIZES}, found {said!r}")
        sizes.append(int(part))
    if sizes != sorted(set(sizes)):
        raise ValueError(f"{ADAPTIVE_ENV} tile rows must ascend without repeats, found {said!r}")
    return tuple((size, SERVE_WARPS if size < 128 else 4, SERVE_STAGES) for size in sizes)


def adaptive_label(classes):
    return "adaptive:" + ",".join(str(size) for size, _, _ in classes)


_STATE = {"window": None, "adaptive": None, "disabled": "", "checked": set(), "said": set(), "tally": None}


def say(text):
    print(f"MOE_TAAF {text}", flush=True)


def announce(key, text):
    if key not in _STATE["said"]:
        _STATE["said"].add(key)
        say(text)


def stand_aside(reason, failed=True):
    if failed:
        selfcheck_fail(ENV, reason)
    _STATE["disabled"] = reason
    print(f"{ENV} stood aside: {reason}; every forward from here on keeps vLLM's fused Marlin MoE", flush=True)


def configure(said, adaptive=None):
    _STATE.update(window=None, adaptive=None, disabled="", checked=set(), said=set(), tally=None)
    try:
        _STATE["window"] = parse_window(said)
        if _STATE["window"] is not None:
            _STATE["adaptive"] = parse_adaptive(adaptive if adaptive is not None else os.environ.get(ADAPTIVE_ENV))
    except ValueError as failure:
        stand_aside(str(failure), failed=False)
    return _STATE["window"]


def select_block(rows, experts, topk):
    return 64 if rows * topk >= WIDE_PAIRS_PER_EXPERT * experts else 32


def refusal(experts, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
            apply_router_weight_on_input):
    if hidden_states.dtype != torch.bfloat16:
        return "dtype"
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        return "layout"
    if expert_map is not None:
        return "expert-map"
    if apply_router_weight_on_input:
        return "weight-on-input"
    if getattr(activation, "name", str(activation)) != "SILU":
        return "activation"
    config = getattr(experts, "activation_config", None)
    if config is not None and getattr(config, "clamp_limit", None) is not None:
        return "clamp"
    quant = getattr(experts, "quant_config", None)
    if quant is None or not getattr(quant, "use_int4_w4a16", False):
        return "quant"
    for name in ("w1_zp", "w2_zp", "w1_bias", "w2_bias", "g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
        if getattr(experts, name, None) is not None:
            return name
    if getattr(experts, "input_dtype", None) is not None:
        return "input-dtype"
    if (topk_weights.dtype != torch.float32 or topk_ids.dtype != torch.int32 or topk_ids.dim() != 2
            or tuple(topk_weights.shape) != tuple(topk_ids.shape) or topk_ids.shape[0] != hidden_states.shape[0]
            or not topk_ids.is_contiguous() or not topk_weights.is_contiguous()):
        return "routing"
    k = hidden_states.shape[1]
    count = w1.shape[0] if w1.dim() == 3 else 0
    inter = w2.shape[1] * 16 if w2.dim() == 3 else 0
    s1, s2 = getattr(experts, "w1_scale", None), getattr(experts, "w2_scale", None)
    if (count < 1 or global_num_experts not in (count, -1) or k % GROUP or inter % GROUP
            or w1.dtype != torch.int32 or w2.dtype != torch.int32
            or tuple(w1.shape) != (count, k // 16, 4 * inter) or tuple(w2.shape) != (count, inter // 16, 2 * k)
            or s1 is None or s2 is None or s1.dtype != torch.bfloat16 or s2.dtype != torch.bfloat16
            or tuple(s1.shape) != (count, k // GROUP, 2 * inter) or tuple(s2.shape) != (count, inter // GROUP, k)
            or not all(one.is_contiguous() for one in (w1, w2, s1, s2))):
        return "shapes"
    return ""


def buffers(rows, k, inter, topk, x, workspace13=None, workspace2=None):
    pairs = rows * topk
    need13 = pairs * inter + rows * k
    need2 = pairs * k
    if all(one is not None and one.dtype == x.dtype and one.is_contiguous() for one in (workspace13, workspace2)) \
            and workspace13.numel() >= need13 and workspace2.numel() >= need2:
        flat13, flat2 = workspace13.view(-1), workspace2.view(-1)
        return (flat13[pairs * inter:need13].view(rows, k), flat13[:pairs * inter].view(pairs, inter),
                flat2[:need2].view(pairs, k))
    return (torch.empty((rows, k), dtype=x.dtype, device=x.device),
            torch.empty((pairs, inter), dtype=x.dtype, device=x.device),
            torch.empty((pairs, k), dtype=x.dtype, device=x.device))


def forward_into(experts, output, x, w1, w2, weights, ids, workspace13=None, workspace2=None, tally=None,
                 block=None):
    rows, k = x.shape
    topk = ids.shape[1]
    count = w1.shape[0]
    inter = w2.shape[1] * 16
    xp, act, down = buffers(rows, k, inter, topk, x, workspace13, workspace2)
    if isinstance(block, tuple):
        aligned = align_adaptive(ids, tuple(size for size, _, _ in block), count)
        permute_activations(x, xp, tally)
        launch_adaptive(xp, act, down, w1, experts.w1_scale, w2, experts.w2_scale, weights, aligned, block, topk,
                        count)
        experts.moe_sum(down.view(rows, topk, k), output, ids, None)
        return adaptive_label(block)
    block = block or select_block(rows, count, topk)
    aligned = align(ids, block, count)
    permute_activations(x, xp, tally)
    launch_gate_up(xp, w1, experts.w1_scale, aligned, act, topk, block, 1, SERVE_WARPS, SERVE_STAGES, 0, SERVE_UNROLL)
    launch_down(act, w2, experts.w2_scale, aligned, weights.reshape(-1), down, rows * topk, block, 1, SERVE_WARPS,
                SERVE_STAGES, 0, SERVE_UNROLL)
    experts.moe_sum(down.view(rows, topk, k), output, ids, None)
    return block


def stock_into(experts, output, x, w1, w2, weights, ids, activation, global_num_experts):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe

    fused_marlin_moe(hidden_states=x, w1=w1, w2=w2, bias1=experts.w1_bias, bias2=experts.w2_bias,
                     w1_scale=experts.w1_scale, w2_scale=experts.w2_scale, topk_weights=weights, topk_ids=ids,
                     global_scale1=experts.g1_alphas, global_scale2=experts.g2_alphas,
                     input_global_scale1=experts.a1_gscale, input_global_scale2=experts.a2_gscale,
                     w1_zeros=experts.w1_zp, w2_zeros=experts.w2_zp, quant_type_id=experts.quant_type_id,
                     apply_router_weight_on_input=False, global_num_experts=global_num_experts, activation=activation,
                     activation_func=experts.activation, activation_config=experts.activation_config,
                     moe_sum=experts.moe_sum, expert_map=None, output=output, input_dtype=experts.input_dtype)


def relative(got, want):
    if not want.numel():
        return 0.0, 0.0
    gap = (got - want).abs()
    top, mean = float(want.abs().max()), float(want.abs().mean())
    return (float(gap.max()) / top if top > 0 else float(gap.max()),
            float(gap.mean()) / mean if mean > 0 else float(gap.mean()))


def spread(x, ids, count):
    generator = torch.Generator(device=x.device).manual_seed(CHECK_SEED)
    rows, topk = ids.shape
    picked = torch.rand((rows, count), generator=generator, device=x.device).topk(topk, dim=1).indices
    weights = torch.softmax(torch.randn((rows, topk), generator=generator, device=x.device), dim=-1)
    data = torch.randn((rows, x.shape[1]), generator=generator, device=x.device).to(x.dtype)
    return data, weights.contiguous(), picked.to(torch.int32).contiguous()


def compare(experts, label, x, w1, w2, weights, ids, activation, global_num_experts, blocks, sampled):
    count = w1.shape[0]
    stock = torch.empty_like(x)
    stock_into(experts, stock, x, w1, w2, weights, ids, activation, global_num_experts)
    live = torch.nonzero((ids < count).all(dim=1)).flatten()
    want = stock[live].float()
    marlin_finite = bool(torch.isfinite(want).all())
    picks = truth = None
    if live.numel() and marlin_finite:
        spots = torch.linspace(0, live.numel() - 1, min(sampled, live.numel()), device=live.device)
        picks = torch.unique(live[spots.round().long()])
        truth = reference_rows(x, w1, experts.w1_scale, w2, experts.w2_scale, weights, ids, picks)
    out = []
    for block in blocks:
        ours = torch.empty_like(x)
        forward_into(experts, ours, x, w1, w2, weights, ids, block=block)
        got = ours[live].float()
        row = {"case": label, "rows": int(x.shape[0]), "real": int(live.numel()),
               "block": adaptive_label(block) if isinstance(block, tuple) else block,
               "sampled": 0 if picks is None else int(picks.numel()), "gap": 0.0, "taaf": 0.0, "marlin": 0.0,
               "taaf_mean": 0.0, "marlin_mean": 0.0, "finite": bool(torch.isfinite(got).all()),
               "marlin_finite": marlin_finite}
        if truth is not None and row["finite"]:
            row["gap"] = relative(got, want)[0]
            row["taaf"], row["taaf_mean"] = relative(ours[picks].float(), truth)
            row["marlin"], row["marlin_mean"] = relative(stock[picks].float(), truth)
        out.append(row)
    return out


def verdict(row):
    name = f"{row['case']} rows {row['rows']} block {row['block']}"
    if not row["marlin_finite"]:
        return f"{name}: Marlin wrote non-finite values, so the check cannot vouch"
    if not row["finite"]:
        return f"{name}: moe_taaf wrote non-finite values where Marlin did not"
    if row["marlin"] > CHECK_MARLIN_CEILING:
        return (f"{name}: the fp32 reference rebuilt from the Marlin weights reads Marlin at max rel err "
                f"{row['marlin']:.3g}, above {CHECK_MARLIN_CEILING}, so the check cannot vouch")
    if row["taaf_mean"] > max(CHECK_MEAN_TIMES * row["marlin_mean"], CHECK_MEAN_FLOOR):
        return (f"{name}: moe_taaf mean rel err {row['taaf_mean']:.3g} against the fp32 reference exceeds "
                f"{CHECK_MEAN_TIMES} times Marlin's {row['marlin_mean']:.3g}")
    if row["taaf"] > max(CHECK_MAX_TIMES * row["marlin"], CHECK_MAX_FLOOR):
        return (f"{name}: moe_taaf max rel err {row['taaf']:.3g} against the fp32 reference exceeds "
                f"{CHECK_MAX_TIMES} times Marlin's {row['marlin']:.3g}")
    if row["gap"] > CHECK_GAP_CEILING:
        return f"{name}: moe_taaf and Marlin differ by max rel {row['gap']:.3g}, above {CHECK_GAP_CEILING}"
    return ""


def check(experts, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts):
    sampled = CHECK_ROWS if not _STATE["checked"] else CHECK_ROWS_AFTER
    _STATE["checked"].add(id(experts))
    began = time.monotonic()
    try:
        count, topk = w1.shape[0], topk_ids.shape[1]
        data, weights, ids = spread(hidden_states, topk_ids, count)
        extra = [_STATE["adaptive"]] if _STATE["adaptive"] else []
        rows = (compare(experts, "live", hidden_states, w1, w2, topk_weights, topk_ids, activation,
                        global_num_experts, [select_block(hidden_states.shape[0], count, topk)] + extra, sampled)
                + compare(experts, "spread", data, w1, w2, weights, ids, activation, global_num_experts,
                          sorted({select_block(1, count, topk), select_block(WIDEST, count, topk)}) + extra,
                          sampled))
    except Exception as failure:
        stand_aside(f"the first-use check raised {type(failure).__name__}: {failure}"[:400])
        return
    failed = [reason for reason in map(verdict, rows) if reason]
    say(f"check {len(_STATE['checked'])} {'FAILED' if failed else 'passed'} in {time.monotonic() - began:.2f} s "
        f"against vLLM's fused Marlin MoE on every real row and an fp32 reference rebuilt from the Marlin weights "
        f"on sampled rows: " + "; ".join(
            f"{row['case']} rows={row['rows']} real={row['real']} sampled={row['sampled']} block={row['block']} "
            f"gap_vs_marlin={row['gap']:.3g} taaf_vs_fp32 max/mean={row['taaf']:.3g}/{row['taaf_mean']:.3g} "
            f"marlin_vs_fp32 max/mean={row['marlin']:.3g}/{row['marlin_mean']:.3g} "
            f"finite={int(row['finite'])}/{int(row['marlin_finite'])}" for row in rows))
    if failed:
        stand_aside("; ".join(failed))
        return
    if _STATE["tally"] is None:
        _STATE["tally"] = torch.zeros(2, dtype=torch.int64, device=hidden_states.device)
        timers = sys.modules.get(TIMERS)
        registered = callable(getattr(timers, "counter", None))
        if registered:
            timers.counter(COUNTER, ("fires", "rows"), _STATE["tally"])
        say(f"firing counter {'in the step timers as ' + COUNTER if registered else 'not registered (no step timers)'}"
            f": every moe_taaf forward adds one fire and its rows on the GPU, inside the captured graphs too")


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def routed(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
           expert_map, workspace13, workspace2, apply_router_weight_on_input):
    window = _STATE["window"]
    if window is None or _STATE["disabled"]:
        return False
    rows = hidden_states.shape[0]
    if not window[0] <= rows <= window[1]:
        return False
    reason = refusal(experts, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                     expert_map, apply_router_weight_on_input)
    if reason:
        announce(("stock", reason), f"stock reason={reason} rows={rows}: this forward keeps vLLM's fused Marlin MoE")
        return False
    capturing = _capturing()
    if id(experts) not in _STATE["checked"]:
        if capturing:
            announce(("stock", "unchecked"), f"stock reason=unchecked-in-capture rows={rows}: a layer met first "
                                             f"inside a graph capture keeps vLLM's fused Marlin MoE in that graph")
            return False
        check(experts, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts)
        if _STATE["disabled"]:
            return False
    block = forward_into(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, workspace13, workspace2,
                         _STATE["tally"], block=_STATE["adaptive"])
    announce(("fired", rows, block, capturing),
             f"fired {ENV}={window[0] if window[1] == WIDEST else f'{window[0]}:{window[1]}'} rows={rows} "
             f"experts={w1.shape[0]} topk={topk_ids.shape[1]} hidden={hidden_states.shape[1]} "
             f"inter={w2.shape[1] * 16} block={block} warps={SERVE_WARPS} stages={SERVE_STAGES} "
             f"capturing={int(capturing)}: the routed experts run on moe_taaf, routing and the shared expert stay "
             f"vLLM's")
    return True


configure(os.environ.get(ENV))
