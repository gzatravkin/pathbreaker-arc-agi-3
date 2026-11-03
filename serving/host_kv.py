import dataclasses
import os

import torch

HOST_KV_ENV = "TAAF_VLLM_HOST_KV"
HOST_KV_SLOTS_ENV = "TAAF_VLLM_HOST_KV_SLOTS"
HOST_KV_RECENT_ENV = "TAAF_VLLM_HOST_KV_RECENT"
GROUP = 4
DEFAULT_SLOTS = 2048
DEFAULT_RECENT_TOKENS = 4096
EMPTY_KEY = -1
PINNED_KEY = torch.iinfo(torch.int64).max


def enabled():
    return str(os.environ.get(HOST_KV_ENV) or "").strip() not in ("", "0")


def settings():
    slots = int(str(os.environ.get(HOST_KV_SLOTS_ENV) or DEFAULT_SLOTS).strip())
    recent = int(str(os.environ.get(HOST_KV_RECENT_ENV) or DEFAULT_RECENT_TOKENS).strip())
    if slots < 1:
        raise ValueError(f"{HOST_KV_SLOTS_ENV} must be >= 1 group slots, found {slots}")
    if recent < 0:
        raise ValueError(f"{HOST_KV_RECENT_ENV} must be >= 0 tokens, found {recent}")
    return {"slots": slots, "recent_tokens": recent}


@dataclasses.dataclass
class HotState:
    slot_of_group: torch.Tensor
    group_of_slot: torch.Tensor
    last_step: torch.Tensor


@dataclasses.dataclass
class Plan:
    hit: torch.Tensor
    slot: torch.Tensor
    miss_rows: torch.Tensor
    miss_groups: torch.Tensor
    miss_slots: torch.Tensor
    evicted_rows: torch.Tensor
    evicted_groups: torch.Tensor
    overflow: int


def new_state(requests, max_groups, slots, device="cpu"):
    return HotState(
        slot_of_group=torch.full((requests, max_groups), -1, dtype=torch.int32, device=device),
        group_of_slot=torch.full((requests, slots), -1, dtype=torch.int64, device=device),
        last_step=torch.full((requests, slots), EMPTY_KEY, dtype=torch.int64, device=device))


def eviction_keys(state, newest, step, recent_tokens):
    occupied = state.group_of_slot >= 0
    floor = torch.div(newest - recent_tokens, GROUP, rounding_mode="floor")
    recent = occupied & (state.group_of_slot >= floor[:, None])
    touched = occupied & (state.last_step == step)
    keys = torch.where(occupied, state.last_step, torch.full_like(state.last_step, EMPTY_KEY))
    return torch.where(recent | touched, torch.full_like(keys, PINNED_KEY), keys)


def plan(state, selected, newest, step, recent_tokens=DEFAULT_RECENT_TOKENS):
    requests, slots = state.group_of_slot.shape
    device = selected.device
    valid = selected >= 0
    ordered = torch.where(valid, selected, torch.full_like(selected, -1)).sort(dim=1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise ValueError("a request selected the same group twice in one step")
    safe = selected.clamp(min=0)
    found = state.slot_of_group.gather(1, safe).to(torch.int64)
    found = torch.where(valid, found, torch.full_like(found, -1))
    hit = valid & (found >= 0)
    rows = torch.arange(requests, device=device)[:, None].expand_as(selected)
    state.last_step[rows[hit], found[hit]] = step
    miss = valid & ~hit
    keys = eviction_keys(state, newest, step, recent_tokens)
    order = keys.argsort(dim=1, stable=True)
    ranked = keys.gather(1, order)
    rank = miss.to(torch.int64).cumsum(1) - 1
    miss_rows = rows[miss]
    miss_groups = selected[miss]
    miss_rank = rank[miss]
    in_range = miss_rank < slots
    candidate = torch.full_like(miss_rank, -1)
    candidate_key = torch.full_like(miss_rank, PINNED_KEY)
    candidate[in_range] = order[miss_rows[in_range], miss_rank[in_range]]
    candidate_key[in_range] = ranked[miss_rows[in_range], miss_rank[in_range]]
    usable = candidate_key != PINNED_KEY
    overflow = int((~usable).sum())
    rows_in = miss_rows[usable]
    groups_in = miss_groups[usable]
    slots_in = candidate[usable]
    old = state.group_of_slot[rows_in, slots_in]
    had = old >= 0
    state.slot_of_group[rows_in[had], old[had]] = -1
    state.slot_of_group[rows_in, groups_in] = slots_in.to(torch.int32)
    state.group_of_slot[rows_in, slots_in] = groups_in
    state.last_step[rows_in, slots_in] = step
    slot = found.clone()
    slot[miss] = torch.where(usable, candidate, torch.full_like(candidate, -1))
    return Plan(hit=hit, slot=slot, miss_rows=rows_in, miss_groups=groups_in, miss_slots=slots_in,
                evicted_rows=rows_in[had], evicted_groups=old[had], overflow=overflow)


def token_slots(state, rows, positions):
    groups = torch.div(positions, GROUP, rounding_mode="floor")
    base = state.slot_of_group[rows, groups].to(torch.int64)
    return torch.where(base >= 0, base * GROUP + positions % GROUP, torch.full_like(base, -1))


def cdiv(numerator, denominator):
    return -(-numerator // denominator)


def hisparse_budget(*, lanes, context_tokens, layers, row_bytes, indexer_bytes_per_token, state_bytes_per_lane,
                    hot_rows, block_size):
    blocks_per_lane = cdiv(context_tokens, block_size)
    hot_blocks = cdiv(hot_rows, block_size)
    host = lanes * blocks_per_lane * block_size * layers * row_bytes
    indexer = lanes * blocks_per_lane * block_size * indexer_bytes_per_token
    hot = lanes * hot_blocks * block_size * layers * row_bytes
    states = lanes * state_bytes_per_lane
    return {"host_bytes": host, "gpu_indexer_bytes": indexer, "gpu_hot_bytes": hot, "gpu_state_bytes": states,
            "gpu_bytes": indexer + hot + states, "hot_blocks_per_request": hot_blocks,
            "context_blocks_per_lane": blocks_per_lane}


def lanes_that_fit(*, gpu_budget_bytes, host_budget_bytes, **shape):
    lanes = 0
    while True:
        need = hisparse_budget(lanes=lanes + 1, **shape)
        if need["gpu_bytes"] > gpu_budget_bytes or need["host_bytes"] > host_budget_bytes:
            return lanes
        lanes += 1


def token_major_cache(raw, blocks, heads, tokens, content):
    flat = raw.view(torch.uint8).view(-1)
    if flat.numel() < blocks * tokens * heads * content:
        raise ValueError(f"{flat.numel()} bytes cannot hold {blocks} blocks of {tokens} tokens x {heads} heads x "
                         f"{content} bytes")
    return torch.as_strided(flat, (blocks, heads, tokens, content),
                            (tokens * heads * content, content, heads * content, 1))


def token_rows(cache):
    blocks, heads, tokens, content = cache.shape
    if cache.stride() != (tokens * heads * content, content, heads * content, 1):
        raise ValueError(f"cache strides {cache.stride()} are not token-major for shape {tuple(cache.shape)}")
    return torch.as_strided(cache, (blocks * tokens, heads * content), (heads * content, 1))


def identity_block_table(num_blocks, device="cpu"):
    return torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, num_blocks)


def physical_as_logical(physical_rows, counts):
    out = torch.empty((physical_rows.shape[0], physical_rows.shape[1] + 1), dtype=torch.int32,
                      device=physical_rows.device)
    out[:, :-1] = physical_rows.to(torch.int32)
    out[:, -1] = counts.to(torch.int32)
    return out


def copy_sources(rows, groups, block_table, page_tokens):
    first = groups * GROUP
    pages = block_table[rows, torch.div(first, page_tokens, rounding_mode="floor")].to(torch.int64)
    return pages, first % page_tokens
