import torch
import triton
import triton.language as tl

KEY_BITS = 12
DEPTH_MAX = 8
LANES_MAX = 1024


@triton.jit
def _allocate_kernel(
    conf_ptr,
    idx_ptr,
    caps_ptr,
    meta_ptr,
    edges_ptr,
    table_ptr,
    out_ptr,
    conf_stride,
    LANES: tl.constexpr,
    DEPTH: tl.constexpr,
    DEPTH_PAD: tl.constexpr,
    NEDGE: tl.constexpr,
    BITS: tl.constexpr,
    HAS_IDX: tl.constexpr,
):
    lane = tl.arange(0, LANES)
    col = tl.arange(0, DEPTH_PAD)
    num_lanes = tl.load(meta_ptr)
    budget = tl.load(meta_ptr + 1)
    live = lane < num_lanes
    cap = tl.load(caps_ptr + lane, mask=live, other=0)
    if HAS_IDX:
        row = tl.load(idx_ptr + lane, mask=live, other=0).to(tl.int64)
    else:
        row = lane.to(tl.int64)
    cols2 = col[None, :] + tl.zeros([LANES, DEPTH_PAD], dtype=tl.int32)
    keys = tl.full([LANES, DEPTH_PAD], -1, dtype=tl.int32)
    score = tl.full([LANES], 1.0, dtype=tl.float32)
    prev = tl.full([LANES], 2147483647, dtype=tl.int32)
    for d in tl.static_range(DEPTH):
        valid = live & (cap > d)
        c = tl.load(conf_ptr + row * conf_stride + d, mask=valid, other=0.0).to(tl.float32)
        c = tl.where(c > 0.0, c, 0.0)
        c = tl.where(c < 1.0, c, 1.0)
        score = score * c
        bucket = tl.zeros([LANES], dtype=tl.int32)
        for e in tl.static_range(NEDGE):
            bucket += (score >= tl.load(edges_ptr + e)).to(tl.int32)
        key = tl.load(table_ptr + d * (NEDGE + 1) + bucket)
        key = tl.minimum(key, prev)
        prev = key
        key = tl.where(valid, key, -1)
        keys = tl.where(cols2 == d, key[:, None], keys)
    total_valid = tl.sum(tl.sum((keys >= 0).to(tl.int32), axis=1), axis=0)
    want = tl.minimum(budget, total_valid)
    lo = 0
    hi = (1 << BITS) - 1
    for _ in tl.static_range(BITS):
        mid = (lo + hi + 1) >> 1
        count = tl.sum(tl.sum((keys >= mid).to(tl.int32), axis=1), axis=0)
        ok = count >= want
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    above = tl.sum(tl.sum((keys > lo).to(tl.int32), axis=1), axis=0)
    remaining = want - above
    chosen = (keys > lo).to(tl.int32)
    for d in tl.static_range(DEPTH):
        tie = tl.sum(tl.where((cols2 == d) & (keys == lo), 1, 0), axis=1)
        tied = tl.sum(tie, axis=0)
        take = tl.maximum(tl.minimum(remaining, tied), 0)
        rank = tl.cumsum(tie, axis=0) - tie
        pick = ((tie > 0) & (rank < take)).to(tl.int32)
        chosen = tl.where(cols2 == d, chosen + pick[:, None], chosen)
        remaining = remaining - take
    counts = tl.sum(chosen, axis=1)
    tl.store(out_ptr + lane, counts, mask=live)
    tl.store(out_ptr + lane, 0, mask=(~live) & (lane < LANES))


def meta_buffer(device):
    return torch.zeros(2, dtype=torch.int32, device=device)


def allocate(conf, caps, meta, edges, table, out=None, idx=None):
    lanes_capacity = caps.shape[0]
    depth = conf.shape[1]
    if not 1 <= depth <= DEPTH_MAX:
        raise ValueError(f"depth {depth} is outside 1..{DEPTH_MAX}")
    if lanes_capacity > LANES_MAX:
        raise ValueError(f"{lanes_capacity} lanes exceed the {LANES_MAX} one block holds")
    if conf.stride(1) != 1:
        raise ValueError("conf rows must be contiguous")
    if table.dim() != 2 or table.shape[0] < depth or table.shape[1] != edges.shape[0] + 1 or not table.is_contiguous():
        raise ValueError("the table needs one contiguous row per depth and one column per bucket")
    if table.dtype != torch.int32 or caps.dtype != torch.int32 or meta.dtype != torch.int32:
        raise ValueError("table, caps and meta are int32")
    lanes = max(32, triton.next_power_of_2(lanes_capacity))
    if out is None:
        out = torch.empty(lanes_capacity, dtype=torch.int32, device=conf.device)
    if out.shape[0] < lanes:
        full = torch.empty(lanes, dtype=torch.int32, device=conf.device)
    else:
        full = out
    _allocate_kernel[(1,)](
        conf, idx if idx is not None else caps, caps, meta, edges, table, full, conf.stride(0),
        LANES=lanes, DEPTH=depth, DEPTH_PAD=triton.next_power_of_2(depth), NEDGE=edges.shape[0], BITS=KEY_BITS,
        HAS_IDX=idx is not None, num_warps=min(32, lanes // 32),
    )
    if full is not out:
        out.copy_(full[:out.shape[0]])
    return out
