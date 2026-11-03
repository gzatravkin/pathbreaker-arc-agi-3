import math

import torch
import triton
import triton.language as tl

import importlib.util
import os
import sys
from pathlib import Path

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

WIDE_EMPTY = 2147483647
WIDE_SHIFT = 65536
NARROW_SPAN = 1 << 32
KEY_BYTES = {True: 4, False: 8}
ROW_BLOCK = 1024
LANE_BLOCK = 128
SPLIT_BLOCK = 512
MISS_BLOCK = 1024
NEEDED_BLOCK = 512


BITMAP_CHUNK = 2048
SOURCES_BLOCK = 1024
BITMAP_WORDS = 1024


@triton.jit
def _popc(word):
    word = word - ((word >> 1) & 0x55555555)
    word = (word & 0x33333333) + ((word >> 2) & 0x33333333)
    word = (word + (word >> 4)) & 0x0F0F0F0F
    return ((word * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _empty_row(union, union_stride, back, back_stride, r, width, OUT_W: tl.constexpr, OUT_BLOCK: tl.constexpr,
               CHUNK: tl.constexpr):
    out = union + r.to(tl.int64) * union_stride
    for start in range(0, OUT_W, OUT_BLOCK):
        at = start + tl.arange(0, OUT_BLOCK)
        tl.store(out + at, tl.full((OUT_BLOCK,), -1, tl.int32), mask=at < OUT_W)
    row = back + r.to(tl.int64) * back_stride
    for start in range(0, width, CHUNK):
        at = start + tl.arange(0, CHUNK)
        tl.store(row + at, tl.zeros((CHUNK,), tl.int32), mask=at < width)


@triton.jit(do_not_specialize=["logical", "logical_stride", "lane_mask", "union", "union_stride", "back",
                               "back_stride", "scratch", "fits", "spill", "states", "tally", "decodes"])
def _union_kernel(logical, logical_stride, lane_mask, union, union_stride, back, back_stride, scratch, fits, spill,
                  states, tally, decodes, Q: tl.constexpr, SPAN: tl.constexpr, TOPK: tl.constexpr,
                  BLOCK: tl.constexpr, OUT_W: tl.constexpr, OUT_BLOCK: tl.constexpr, CAP: tl.constexpr,
                  CAPPED: tl.constexpr, MASKED: tl.constexpr, NARROW: tl.constexpr, SHIFT: tl.constexpr,
                  EMPTY: tl.constexpr, SKIP: tl.constexpr = False, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    lane = r % decodes
    part = r // decodes
    width = SPAN * TOPK
    if SKIP and MASKED:
        if tl.load(lane_mask + lane).to(tl.int32) == 0:
            _empty_row(union, union_stride, back, back_stride, r, width, OUT_W, OUT_BLOCK, 2048)
            if CAPPED:
                tl.store(fits + lane, True)
                tl.store(spill + lane, False)
            return
    c = tl.arange(0, BLOCK)
    inside = c < width
    row = (lane * Q + part * SPAN + c // TOPK).to(tl.int64)
    pick = tl.load(logical + row * logical_stride + c % TOPK, mask=inside, other=-1)
    if MASKED:
        chosen = tl.load(lane_mask + lane).to(tl.int1)
        pick = tl.where(chosen, pick, -1)
    held = inside & (pick >= 0)
    if NARROW:
        ordered = tl.sort(tl.where(held, pick.to(tl.uint32), EMPTY) * SHIFT + c.to(tl.uint32))
    else:
        ordered = tl.sort(tl.where(held, pick.to(tl.int64), EMPTY) * SHIFT + c)
    value = (ordered // SHIFT).to(tl.int64)
    index = (ordered % SHIFT).to(tl.int32)
    valid = value != EMPTY
    base = r.to(tl.int64) * BLOCK
    tl.store(scratch + base + c, value)
    tl.debug_barrier()
    previous = tl.load(scratch + base + c - 1, mask=c > 0, other=-1)
    fresh = valid & ((c == 0) | (value != previous))
    counted = fresh.to(tl.int32)
    rank = tl.cumsum(counted, 0) - 1
    unique = tl.sum(counted, 0)
    tl.store(back + r.to(tl.int64) * back_stride + index, tl.maximum(rank, 0), mask=index < width)
    out = union + r.to(tl.int64) * union_stride
    for start in range(0, OUT_W, OUT_BLOCK):
        at = start + tl.arange(0, OUT_BLOCK)
        tl.store(out + at, tl.full((OUT_BLOCK,), -1, tl.int32), mask=at < OUT_W)
    tl.debug_barrier()
    if CAPPED:
        fit = unique <= CAP
        tl.store(out + rank, value.to(tl.int32), mask=fresh & (rank < OUT_W) & fit)
        tl.store(fits + lane, fit)
        tl.store(spill + lane, unique > CAP)
        state = tl.load(states + lane)
        live = (unique > 0) & (state >= 0)
        over = live & (unique > CAP)
        wide = unique.to(tl.int64)
        tl.atomic_add(tally, live.to(tl.int64))
        tl.atomic_add(tally + 1, over.to(tl.int64))
        tl.atomic_add(tally + 2, wide * live.to(tl.int64))
        tl.atomic_add(tally + 3, wide * over.to(tl.int64))
    else:
        tl.store(out + rank, value.to(tl.int32), mask=fresh & (rank < OUT_W))


@triton.jit(do_not_specialize=["logical", "logical_stride", "lane_mask", "union", "union_stride", "back",
                               "back_stride", "bits", "prefix", "fits", "spill", "states", "tally", "decodes"])
def _bitmap_union_kernel(logical, logical_stride, lane_mask, union, union_stride, back, back_stride, bits, prefix,
                         fits, spill, states, tally, decodes, Q: tl.constexpr, SPAN: tl.constexpr,
                         TOPK: tl.constexpr, OUT_W: tl.constexpr, OUT_BLOCK: tl.constexpr, CAP: tl.constexpr,
                         CAPPED: tl.constexpr, MASKED: tl.constexpr, BOUND: tl.constexpr, WORDS: tl.constexpr,
                         CHUNK: tl.constexpr, WORD_BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    lane = r % decodes
    part = r // decodes
    width = SPAN * TOPK
    if MASKED:
        if tl.load(lane_mask + lane).to(tl.int32) == 0:
            _empty_row(union, union_stride, back, back_stride, r, width, OUT_W, OUT_BLOCK, CHUNK)
            if CAPPED:
                tl.store(fits + lane, True)
                tl.store(spill + lane, False)
            return
    words = bits + r.to(tl.int64) * WORDS
    ranks = prefix + r.to(tl.int64) * WORDS
    for start in range(0, WORDS, WORD_BLOCK):
        w = start + tl.arange(0, WORD_BLOCK)
        tl.store(words + w, tl.zeros((WORD_BLOCK,), tl.int32), mask=w < WORDS)
    tl.debug_barrier()
    first = lane * Q + part * SPAN
    for start in range(0, width, CHUNK):
        c = start + tl.arange(0, CHUNK)
        inside = c < width
        pick = tl.load(logical + (first + c // TOPK).to(tl.int64) * logical_stride + c % TOPK, mask=inside,
                       other=-1)
        held = inside & (pick >= 0) & (pick < BOUND)
        safe = tl.where(held, pick, 0)
        flag = (tl.full((CHUNK,), 1, tl.uint32) << (safe & 31).to(tl.uint32)).to(tl.int32, bitcast=True)
        tl.atomic_or(words + (safe >> 5), flag, mask=held)
    tl.debug_barrier()
    unique = tl.sum(tl.zeros((WORD_BLOCK,), tl.int32), 0)
    for start in range(0, WORDS, WORD_BLOCK):
        w = start + tl.arange(0, WORD_BLOCK)
        word = tl.load(words + w, mask=w < WORDS, other=0, cache_modifier=".cg").to(tl.uint32, bitcast=True)
        count = _popc(word)
        tl.store(ranks + w, unique + tl.cumsum(count, 0) - count, mask=w < WORDS)
        unique += tl.sum(count, 0)
    tl.debug_barrier()
    out = union + r.to(tl.int64) * union_stride
    for start in range(0, OUT_W, OUT_BLOCK):
        at = start + tl.arange(0, OUT_BLOCK)
        tl.store(out + at, tl.full((OUT_BLOCK,), -1, tl.int32), mask=at < OUT_W)
    tl.debug_barrier()
    fit = unique <= CAP
    floor = tl.maximum(unique - 1, 0)
    row = back + r.to(tl.int64) * back_stride
    for start in range(0, width, CHUNK):
        c = start + tl.arange(0, CHUNK)
        inside = c < width
        pick = tl.load(logical + (first + c // TOPK).to(tl.int64) * logical_stride + c % TOPK, mask=inside,
                       other=-1)
        held = inside & (pick >= 0) & (pick < BOUND)
        safe = tl.where(held, pick, 0)
        at = safe >> 5
        word = tl.load(words + at, mask=held, other=0, cache_modifier=".cg").to(tl.uint32, bitcast=True)
        below = word & ((tl.full((CHUNK,), 1, tl.uint32) << (safe & 31).to(tl.uint32)) - 1)
        rank = tl.load(ranks + at, mask=held, other=0, cache_modifier=".cg") + _popc(below)
        tl.store(row + c, tl.where(held, rank, floor), mask=inside)
        if CAPPED:
            tl.store(out + rank, pick, mask=held & (rank < OUT_W) & fit)
        else:
            tl.store(out + rank, pick, mask=held & (rank < OUT_W))
    if CAPPED:
        tl.store(fits + lane, fit)
        tl.store(spill + lane, unique > CAP)
        state = tl.load(states + lane)
        live = (unique > 0) & (state >= 0)
        over = live & (unique > CAP)
        wide = unique.to(tl.int64)
        tl.atomic_add(tally, live.to(tl.int64))
        tl.atomic_add(tally + 1, over.to(tl.int64))
        tl.atomic_add(tally + 2, wide * live.to(tl.int64))
        tl.atomic_add(tally + 3, wide * over.to(tl.int64))


@triton.jit(do_not_specialize=["logical", "logical_stride", "resolved", "resolved_stride", "resolved_width", "back",
                               "back_stride", "keep", "out", "out_stride", "decodes", "part", "span", "query"])
def _map_kernel(logical, logical_stride, resolved, resolved_stride, resolved_width, back, back_stride, keep, out,
                out_stride, decodes, part, span, query, TOPK: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    o = tl.program_id(0)
    lane = o // span
    step = o % span
    j = tl.arange(0, BLOCK)
    inside = j < TOPK
    row = (lane * query + part * span + step).to(tl.int64)
    pick = tl.load(logical + row * logical_stride + j, mask=inside, other=-1)
    at = tl.load(back + (part * decodes + lane).to(tl.int64) * back_stride + step * TOPK + j, mask=inside, other=0)
    at = tl.minimum(tl.maximum(at, 0), resolved_width - 1)
    kept = tl.load(keep + lane).to(tl.int1)
    found = tl.load(resolved + lane.to(tl.int64) * resolved_stride + at, mask=inside, other=-1)
    mapped = tl.where(inside & (pick >= 0) & kept, found, -1)
    target = out + o.to(tl.int64) * out_stride
    tl.store(target + j, mapped, mask=inside)
    tl.store(target + TOPK, tl.sum((inside & (mapped >= 0)).to(tl.int32), 0))


@triton.jit
def _map_split_kernel(logical, logical_stride, resolved, resolved_stride, resolved_width, back, back_stride, keep,
                      fresh, fresh_stride, hits, misses, out_stride, decodes, part, span, query, TOPK: tl.constexpr,
                      BLOCK: tl.constexpr):
    o = tl.program_id(0)
    lane = o // span
    step = o % span
    j = tl.arange(0, BLOCK)
    inside = j < TOPK
    row = (lane * query + part * span + step).to(tl.int64)
    pick = tl.load(logical + row * logical_stride + j, mask=inside, other=-1)
    at = tl.load(back + (part * decodes + lane).to(tl.int64) * back_stride + step * TOPK + j, mask=inside, other=0)
    at = tl.minimum(tl.maximum(at, 0), resolved_width - 1)
    kept = tl.load(keep + lane).to(tl.int1)
    found = tl.load(resolved + lane.to(tl.int64) * resolved_stride + at, mask=inside, other=-1)
    mapped = tl.where(inside & (pick >= 0) & kept, found, -1)
    late = tl.load(fresh + lane.to(tl.int64) * fresh_stride + at, mask=inside, other=0) != 0
    late = late & (mapped >= 0)
    first = hits + o.to(tl.int64) * out_stride
    tl.store(first + j, tl.where(late, -1, mapped), mask=inside)
    tl.store(first + TOPK, tl.sum((inside & (mapped >= 0)).to(tl.int32), 0))
    second = misses + o.to(tl.int64) * out_stride
    rank = tl.cumsum(late.to(tl.int32), 0) - 1
    count = tl.sum(late.to(tl.int32), 0)
    tl.store(second + j, tl.full((BLOCK,), -1, tl.int32), mask=inside & (j >= count))
    tl.store(second + rank, mapped, mask=late)
    tl.store(second + TOPK, count)


@triton.jit(do_not_specialize=["source", "source_stride", "target", "target_stride", "keep", "span", "query",
                               "first"])
def _rows_kernel(source, source_stride, target, target_stride, keep, span, query, first, ROW: tl.constexpr,
                 BLOCK: tl.constexpr, GATHER: tl.constexpr, MASKED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0)
    lane = i // span
    full = lane * query + first + i % span
    if GATHER:
        read, write = full, i
    else:
        read, write = i, full
    for start in range(0, ROW, BLOCK):
        at = start + tl.arange(0, BLOCK)
        inside = at < ROW
        if MASKED:
            inside = inside & tl.load(keep + lane).to(tl.int1)
        value = tl.load(source + read.to(tl.int64) * source_stride + at, mask=inside)
        tl.store(target + write.to(tl.int64) * target_stride + at, value, mask=inside)


@triton.jit(do_not_specialize=["logical", "logical_stride", "resolved", "resolved_stride", "resolved_width", "back",
                               "back_stride", "spill", "out", "out_stride", "query", "query_stride", "picked_query",
                               "picked_query_stride", "gate", "gate_stride", "picked_gate", "picked_gate_stride",
                               "live", "decodes", "part", "span", "query_len"])
def _spill_prep_kernel(logical, logical_stride, resolved, resolved_stride, resolved_width, back, back_stride, spill,
                       out, out_stride, query, query_stride, picked_query, picked_query_stride, gate, gate_stride,
                       picked_gate, picked_gate_stride, live, decodes, part, span, query_len, TOPK: tl.constexpr,
                       BLOCK: tl.constexpr, QROW: tl.constexpr, QBLOCK: tl.constexpr, GROW: tl.constexpr,
                       GBLOCK: tl.constexpr, LANES: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    o = tl.program_id(0)
    lane = o // span
    step = o % span
    spilled = tl.sum(tl.zeros((LANES,), tl.int32), 0)
    for start in range(0, decodes, LANES):
        at = start + tl.arange(0, LANES)
        spilled += tl.sum(tl.load(spill + at, mask=at < decodes, other=0).to(tl.int32), 0)
    if o == 0:
        tl.store(live, spilled)
    target = out + o.to(tl.int64) * out_stride
    if spilled == 0:
        tl.store(target + TOPK, 0)
        return
    full = (lane * query_len + part * span + step).to(tl.int64)
    for start in range(0, QROW, QBLOCK):
        at = start + tl.arange(0, QBLOCK)
        inside = at < QROW
        tl.store(picked_query + o.to(tl.int64) * picked_query_stride + at,
                 tl.load(query + full * query_stride + at, mask=inside), mask=inside)
    for start in range(0, GROW, GBLOCK):
        at = start + tl.arange(0, GBLOCK)
        inside = at < GROW
        tl.store(picked_gate + o.to(tl.int64) * picked_gate_stride + at,
                 tl.load(gate + full * gate_stride + at, mask=inside), mask=inside)
    j = tl.arange(0, BLOCK)
    inside = j < TOPK
    pick = tl.load(logical + full * logical_stride + j, mask=inside, other=-1)
    at = tl.load(back + (part * decodes + lane).to(tl.int64) * back_stride + step * TOPK + j, mask=inside, other=0)
    at = tl.minimum(tl.maximum(at, 0), resolved_width - 1)
    kept = tl.load(spill + lane).to(tl.int1)
    found = tl.load(resolved + lane.to(tl.int64) * resolved_stride + at, mask=inside, other=-1)
    mapped = tl.where(inside & (pick >= 0) & kept, found, -1)
    tl.store(target + j, mapped, mask=inside)
    tl.store(target + TOPK, tl.sum((inside & (mapped >= 0)).to(tl.int32), 0))


@triton.jit(do_not_specialize=["counts", "idle", "pieces", "total", "factor", "width", "tally"])
def _split_kernel(counts, idle, pieces, total, factor, width, tally, HAS_IDLE: tl.constexpr, BLOCK: tl.constexpr,
                  TALLY: tl.constexpr = False, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = at < total
    lane = at // factor
    piece = at % factor
    count = tl.load(counts + lane, mask=inside, other=0).to(tl.int64)
    if HAS_IDLE:
        resting = tl.load(idle + lane, mask=inside, other=0).to(tl.int1)
        count = tl.where(resting, 0, count)
        tl.store(counts + lane, count.to(counts.dtype.element_ty), mask=inside & (piece == 0))
    left = tl.minimum(tl.maximum(count - piece * width, 0), width)
    tl.store(pieces + at, left.to(tl.int32), mask=inside)
    if TALLY:
        first = inside & (piece == 0)
        tl.atomic_add(tally, tl.sum(tl.where(first, tl.minimum(count, factor * width), 0), 0))
        tl.atomic_add(tally + 1, tl.sum((first & (count > 0)).to(tl.int64), 0))


@triton.jit(do_not_specialize=["states", "out", "count", "request_ids", "fits", "decodes", "live"])
def _spill_states_kernel(states, out, count, request_ids, fits, decodes, live, BLOCK: tl.constexpr,
                         LIVE: tl.constexpr = False, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    for start in range(0, count, BLOCK):
        at = start + tl.arange(0, BLOCK)
        inside = at < count
        tl.store(out + at, tl.load(states + at, mask=inside), mask=inside)
    tl.debug_barrier()
    for start in range(0, decodes, BLOCK):
        lane = start + tl.arange(0, BLOCK)
        inside = lane < decodes
        if LIVE:
            inside = inside & (lane < tl.load(live))
        fit = tl.load(fits + lane, mask=inside, other=0).to(tl.int1)
        request = tl.load(request_ids + lane, mask=inside, other=0).to(tl.int64)
        tl.store(out + request, tl.full((BLOCK,), -1, out.dtype.element_ty), mask=inside & fit)


@triton.jit(do_not_specialize=["counts", "states", "fresh", "misses", "decodes", "steps", "dummy"])
def _fresh_kernel(counts, states, fresh, misses, decodes, steps, dummy, LAST: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    total = tl.sum(tl.zeros((BLOCK,), tl.int64), 0)
    for start in range(0, decodes, BLOCK):
        lane = start + tl.arange(0, BLOCK)
        inside = lane < decodes
        state = tl.load(states + lane, mask=inside, other=-1).to(tl.int64)
        slot = tl.where(state >= 0, state, dummy)
        flag = tl.load(fresh + slot, mask=inside, other=0).to(tl.int64)
        summed = tl.zeros((BLOCK,), tl.int64)
        for step in range(0, steps):
            summed += tl.load(counts + step * decodes + lane, mask=inside, other=0).to(tl.int64)
        total += tl.sum(summed * flag, 0)
    tl.store(misses, tl.load(misses) + total)
    if LAST:
        tl.debug_barrier()
        for start in range(0, decodes, BLOCK):
            lane = start + tl.arange(0, BLOCK)
            inside = lane < decodes
            state = tl.load(states + lane, mask=inside, other=-1).to(tl.int64)
            slot = tl.where(state >= 0, state, dummy)
            tl.store(fresh + slot, tl.zeros((BLOCK,), fresh.dtype.element_ty), mask=inside)


@triton.jit(do_not_specialize=["logical", "logical_stride", "owners", "table", "table_stride", "needed", "requests",
                               "table_width", "staged_blocks", "block_size"])
def _needed_kernel(logical, logical_stride, owners, table, table_stride, needed, requests, table_width,
                   staged_blocks, block_size, WIDTH: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = c < WIDTH
    token = tl.load(logical + r * logical_stride + c, mask=inside, other=-1).to(tl.int64)
    count = tl.load(logical + r * logical_stride + WIDTH).to(tl.int64)
    count = tl.minimum(tl.maximum(count, 0), WIDTH)
    owner = tl.load(owners + r).to(tl.int64)
    clean = tl.maximum(token, 0)
    page = clean // block_size
    valid = inside & (c < count) & (token >= 0) & (owner >= 0) & (owner < requests) & (page < table_width)
    request = tl.minimum(tl.maximum(owner, 0), requests - 1)
    column = tl.minimum(page, table_width - 1)
    physical = tl.load(table + request * table_stride + column, mask=inside, other=-1).to(tl.int64)
    valid = valid & (physical >= 0) & (physical < staged_blocks)
    tl.store(needed + physical * block_size + clean % block_size, tl.full((BLOCK,), 1, tl.int1), mask=valid)


@triton.jit(do_not_specialize=["miss", "needed", "out", "tally", "total", "host"])
def _miss_kernel(miss, needed, out, tally, total, host, BLOCK: tl.constexpr, HOST: tl.constexpr = False, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = at < total
    want = tl.load(needed + at, mask=inside, other=0).to(tl.int1)
    held = tl.load(miss + at, mask=inside, other=0) != 0
    tl.store(out + at, (held & want).to(out.dtype.element_ty), mask=inside)
    tl.atomic_add(tally, tl.sum(want.to(tl.int64), 0))
    if HOST:
        tl.atomic_add(host, tl.sum((held & want).to(tl.int64), 0))


@triton.jit(do_not_specialize=["source", "needed", "resident", "block_stride", "token_stride", "flat", "flat_stride",
                               "size"])
def _stage_kernel(source, needed, resident, block_stride, token_stride, flat, flat_stride, size, ROW: tl.constexpr,
                  BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0).to(tl.int64)
    picked = tl.load(source + i).to(tl.int64)
    keep = (picked >= 0) & tl.load(needed + i).to(tl.int1)
    clean = tl.maximum(picked, 0)
    origin = resident + (clean // size) * block_stride + (clean % size) * token_stride
    for start in range(0, ROW, BLOCK):
        at = start + tl.arange(0, BLOCK)
        inside = (at < ROW) & keep
        tl.store(flat + i * flat_stride + at, tl.load(origin + at, mask=inside), mask=inside)


@triton.jit(do_not_specialize=["source", "needed", "resident", "block_stride", "token_stride", "flat", "flat_stride",
                               "size", "total"])
def _stage_tile_kernel(source, needed, resident, block_stride, token_stride, flat, flat_stride, size, total,
                       ROW: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    base = tl.program_id(0).to(tl.int64) * TILE
    rows = base + tl.arange(0, TILE)
    inside = rows < total
    picked = tl.load(source + rows, mask=inside, other=-1).to(tl.int64)
    want = tl.load(needed + rows, mask=inside, other=0).to(tl.int1)
    keep = inside & (picked >= 0) & want
    if tl.sum(keep.to(tl.int32), 0) > 0:
        for k in range(0, TILE):
            i = base + k
            live = i < total
            one = tl.load(source + i, mask=live, other=-1).to(tl.int64)
            wanted = tl.load(needed + i, mask=live, other=0).to(tl.int1)
            if (one >= 0) & wanted:
                origin = resident + (one // size) * block_stride + (one % size) * token_stride
                for start in range(0, ROW, BLOCK):
                    at = start + tl.arange(0, BLOCK)
                    inner = at < ROW
                    tl.store(flat + i * flat_stride + at, tl.load(origin + at, mask=inner), mask=inner)


@triton.jit(do_not_specialize=["row_ids", "reps", "table", "table_stride", "gpu", "miss", "total", "size",
                               "resident", "plan_cols", "table_cols", "table_rows", "pages"])
def _sources_kernel(row_ids, reps, table, table_stride, gpu, miss, total, size, resident, plan_cols, table_cols,
                    table_rows, pages, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = at < total
    unique = at // size
    offset = at % size
    row = tl.load(row_ids + at, mask=inside, other=-1).to(tl.int64)
    first = tl.load(row_ids + unique * size, mask=inside, other=-1).to(tl.int64)
    rep = tl.load(reps + unique, mask=inside, other=0).to(tl.int64)
    request = tl.minimum(rep // plan_cols, table_rows - 1)
    column = (rep % plan_cols) * pages + offset // resident
    within = column < table_cols
    block = tl.load(table + request * table_stride + tl.minimum(column, tl.maximum(table_cols - 1, 0)),
                    mask=inside & within, other=0).to(tl.int64)
    block = tl.where(within, block, 0)
    valid = row >= 0
    hit = (block > 0) & (first >= size) & valid
    found = tl.where(hit, block * resident + offset % resident, -1).to(tl.int32)
    tl.store(gpu + at, found, mask=inside)
    tl.store(miss + at, ((found < 0) & valid).to(tl.int32), mask=inside)


def pow2(count):
    return triton.next_power_of_2(max(int(count), 1))


def sources(row_ids, reps, table, size, resident, plan_cols):
    total = row_ids.shape[1]
    gpu = torch.empty((1, total), dtype=torch.int32, device=row_ids.device)
    miss = torch.empty((1, total), dtype=torch.int32, device=row_ids.device)
    if total:
        _sources_kernel[(triton.cdiv(total, SOURCES_BLOCK),)](
            row_ids, reps, table, table.stride(0), gpu, miss, total, size, resident, plan_cols, table.shape[1],
            table.shape[0], size // resident, BLOCK=SOURCES_BLOCK, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return gpu, miss


def union_layout(width, bound, smem):
    block = pow2(width)
    empty = NARROW_SPAN // block - 1
    if 0 < int(bound) <= empty and block * KEY_BYTES[True] <= int(smem):
        return {"NARROW": True, "SHIFT": block, "EMPTY": empty, "BLOCK": block}
    if int(width) <= WIDE_SHIFT and block * KEY_BYTES[False] <= int(smem):
        return {"NARROW": False, "SHIFT": WIDE_SHIFT, "EMPTY": WIDE_EMPTY, "BLOCK": block}
    return None


def bitmap_layout(bound):
    if int(bound) <= 0:
        return None
    return {"BITMAP": True, "BOUND": int(bound), "WORDS": -(-int(bound) // 32), "CHUNK": BITMAP_CHUNK,
            "WORD_BLOCK": BITMAP_WORDS}


def union(logical, decodes, query, span, top_k, union_out, back, layout, lane_mask=None, capped=None,
          skip_masked=False):
    passes = union_out.shape[0] // decodes
    width = span * top_k
    out_width = union_out.shape[1]
    out_block = min(pow2(out_width), ROW_BLOCK)
    if capped is None:
        fits = spill = states = tally = back
        cap = 0
    else:
        fits, spill, states, tally, cap = capped
    if layout is not None and layout.get("BITMAP"):
        rows = union_out.shape[0]
        bits = torch.empty((rows, layout["WORDS"]), dtype=torch.int32, device=logical.device)
        prefix = torch.empty((rows, layout["WORDS"]), dtype=torch.int32, device=logical.device)
        _bitmap_union_kernel[(passes * decodes,)](
            logical, logical.stride(0), back if lane_mask is None else lane_mask, union_out, union_out.stride(0),
            back, back.stride(0), bits, prefix, fits, spill, states, tally, decodes, Q=query, SPAN=span, TOPK=top_k,
            OUT_W=out_width, OUT_BLOCK=out_block, CAP=cap, CAPPED=capped is not None, MASKED=lane_mask is not None,
            BOUND=layout["BOUND"], WORDS=layout["WORDS"], CHUNK=layout["CHUNK"], WORD_BLOCK=layout["WORD_BLOCK"],
            num_warps=8, PDL=pdl_on(), launch_pdl=pdl_on())
        return
    block = pow2(width)
    if layout is None or layout["BLOCK"] != block:
        raise ValueError(f"a union of {width} picks needs a layout of block {block}, not {layout}")
    scratch = torch.empty((union_out.shape[0], block), dtype=torch.int64, device=logical.device)
    _union_kernel[(passes * decodes,)](
        logical, logical.stride(0), back if lane_mask is None else lane_mask, union_out, union_out.stride(0), back,
        back.stride(0), scratch, fits, spill, states, tally, decodes, Q=query, SPAN=span, TOPK=top_k, BLOCK=block,
        OUT_W=out_width, OUT_BLOCK=out_block, CAP=cap, CAPPED=capped is not None, MASKED=lane_mask is not None,
        NARROW=layout["NARROW"], SHIFT=layout["SHIFT"], EMPTY=layout["EMPTY"],
        SKIP=bool(skip_masked) and lane_mask is not None, num_warps=8 if block >= 4096 else 4, PDL=pdl_on(), launch_pdl=pdl_on())


def mapped(logical, resolved, resolved_width, back, keep, out, decodes, part, span, query, top_k):
    _map_kernel[(out.shape[0],)](
        logical, logical.stride(0), resolved, resolved.stride(0), resolved_width, back, back.stride(0), keep, out,
        out.stride(0), decodes, part, span, query, TOPK=top_k, BLOCK=pow2(top_k), num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())


def mapped_split(logical, resolved, resolved_width, back, keep, fresh, hits, misses, decodes, part, span, query,
                 top_k):
    _map_split_kernel[(hits.shape[0],)](
        logical, logical.stride(0), resolved, resolved.stride(0), resolved_width, back, back.stride(0), keep, fresh,
        fresh.stride(0), hits, misses, hits.stride(0), decodes, part, span, query, TOPK=top_k, BLOCK=pow2(top_k),
        num_warps=4)


WORDS = (torch.int64, torch.int32, torch.int16, torch.uint8)


def word_rows(tensor):
    if tensor.dim() < 1:
        return None
    rows = tensor.shape[0]
    inner = 1
    expected = 1
    for size, stride in reversed(list(zip(tensor.shape[1:], tensor.stride()[1:]))):
        if size != 1 and stride != expected:
            return None
        expected *= size
        inner *= size
    size = tensor.element_size()
    flat = torch.as_strided(tensor, (rows, inner), (tensor.stride(0), 1), tensor.storage_offset())
    for word in WORDS:
        bytes_ = torch.empty((), dtype=word).element_size()
        if ((inner * size) % bytes_ or (tensor.stride(0) * size) % bytes_ or (tensor.storage_offset() * size) % bytes_
                or tensor.data_ptr() % bytes_):
            continue
        if word == tensor.dtype:
            return flat
        try:
            return flat.view(word)
        except RuntimeError:
            continue
    return None


def rows_ready(*tensors):
    return all(word_rows(one) is not None for one in tensors)


def rows(source, target, span, query, first, gather, keep=None):
    read, write = word_rows(source), word_rows(target)
    if read is None or write is None or read.dtype != write.dtype or read.shape[1] != write.shape[1]:
        raise ValueError("row copies need inner-contiguous rows of one width")
    count = target.shape[0] if gather else source.shape[0]
    if count == 0:
        return target
    row = read.shape[1]
    _rows_kernel[(count,)](
        read, read.stride(0), write, write.stride(0), read if keep is None else keep, span, query, first, ROW=row,
        BLOCK=min(pow2(row), ROW_BLOCK), GATHER=gather, MASKED=keep is not None, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return target


def split(counts, idle, factor, width, tally=None):
    decodes = counts.shape[0]
    pieces = torch.empty(decodes * factor, dtype=torch.int32, device=counts.device)
    total = decodes * factor
    _split_kernel[(triton.cdiv(total, SPLIT_BLOCK),)](
        counts, counts if idle is None else idle, pieces, total, factor, width, pieces if tally is None else tally,
        HAS_IDLE=idle is not None, BLOCK=SPLIT_BLOCK, TALLY=tally is not None, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return pieces


def spill_prep(logical, resolved, resolved_width, back, spill, out, query, picked_query, gate, picked_gate, live,
               decodes, part, span, query_len, top_k):
    rows = out.shape[0]
    if rows == 0:
        return out
    read_query, write_query = word_rows(query), word_rows(picked_query)
    read_gate, write_gate = word_rows(gate), word_rows(picked_gate)
    if (read_query is None or write_query is None or read_gate is None or write_gate is None
            or read_query.dtype != write_query.dtype or read_gate.dtype != write_gate.dtype
            or read_query.shape[1] != write_query.shape[1] or read_gate.shape[1] != write_gate.shape[1]):
        raise ValueError("spill rows need inner-contiguous rows of one width")
    query_row, gate_row = read_query.shape[1], read_gate.shape[1]
    _spill_prep_kernel[(rows,)](
        logical, logical.stride(0), resolved, resolved.stride(0), resolved_width, back, back.stride(0), spill, out,
        out.stride(0), read_query, read_query.stride(0), write_query, write_query.stride(0), read_gate,
        read_gate.stride(0), write_gate, write_gate.stride(0), live, decodes, part, span, query_len, TOPK=top_k,
        BLOCK=pow2(top_k), QROW=query_row, QBLOCK=min(pow2(query_row), ROW_BLOCK), GROW=gate_row,
        GBLOCK=min(pow2(gate_row), ROW_BLOCK), LANES=LANE_BLOCK, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return out


def spill_states(states, request_ids, fits, live=None, out=None):
    out = torch.empty_like(states) if out is None else out[:states.numel()]
    _spill_states_kernel[(1,)](states, out, states.numel(), request_ids, fits, fits.shape[0],
                               states if live is None else live, BLOCK=LANE_BLOCK, LIVE=live is not None,
                               num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return out


def fresh_misses(counts, states, fresh, misses, decodes, steps, last):
    _fresh_kernel[(1,)](counts, states, fresh, misses, decodes, steps, fresh.numel() - 1, LAST=bool(last),
                        BLOCK=LANE_BLOCK, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())


def needed(logical, owners, block_table, block_size, total_rows):
    out = torch.zeros(total_rows, dtype=torch.bool, device=logical.device)
    count, width = logical.shape[0], logical.shape[1] - 1
    if count == 0 or width <= 0:
        return out
    requests, table_width = block_table.shape
    _needed_kernel[(count, triton.cdiv(width, NEEDED_BLOCK))](
        logical, logical.stride(0), owners, block_table, block_table.stride(0), out, requests, table_width,
        total_rows // block_size, block_size, WIDTH=width, BLOCK=NEEDED_BLOCK, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return out


def missed(miss_mask, needed_rows, tally, host=None):
    out = torch.empty_like(miss_mask)
    total = needed_rows.numel()
    if total:
        _miss_kernel[(triton.cdiv(total, MISS_BLOCK),)](miss_mask, needed_rows, out, tally, total,
                                                        tally if host is None else host, BLOCK=MISS_BLOCK,
                                                        HOST=host is not None, num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return out


BLOCK_STAGE_CHUNK = 4096


@triton.jit(do_not_specialize=["host", "target", "sources", "full"])
def _block_stage_kernel(host, target, sources, full, WORDS: tl.constexpr, CHUNK: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    piece = tl.program_id(1)
    if tl.load(full + block) != 0:
        source = tl.load(sources + block).to(tl.int64)
        offsets = piece * CHUNK + tl.arange(0, CHUNK)
        inside = offsets < WORDS
        values = tl.load(host + source * WORDS + offsets, mask=inside)
        tl.store(target + block * WORDS + offsets, values, mask=inside)


def block_stage(host, target, sources, full, words):
    blocks = full.numel()
    if blocks == 0 or words <= 0:
        return target
    _block_stage_kernel[(blocks, triton.cdiv(words, BLOCK_STAGE_CHUNK))](
        host, target, sources, full, WORDS=int(words), CHUNK=BLOCK_STAGE_CHUNK, num_warps=4)
    return target


AHEAD_CHUNK = 4096


@triton.jit(do_not_specialize=["blocks", "size", "host_rows"])
def _ahead_kernel(host, target, row_ids, counts, blocks, size, host_rows, WORDS: tl.constexpr, CHUNK: tl.constexpr,
                  PIECES: tl.constexpr):
    for item in range(tl.program_id(0), blocks * PIECES, tl.num_programs(0)):
        block = (item // PIECES).to(tl.int64)
        piece = item % PIECES
        count = tl.load(counts + block).to(tl.int64)
        first = tl.load(row_ids + block * size).to(tl.int64)
        if (count > 0) & (first >= 0) & (first + count <= host_rows):
            offsets = piece * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
            inside = offsets < count * WORDS
            values = tl.load(host + first * WORDS + offsets, mask=inside)
            tl.store(target + block * size * WORDS + offsets, values, mask=inside)


def ahead_counts(settled, blocks, size, out):
    torch.sum(settled[:blocks * size].view(blocks, size), 1, dtype=torch.int32, out=out[:blocks])
    return out[:blocks]


def ahead(host, target, row_ids, counts, size, words, programs):
    blocks = counts.numel()
    if blocks == 0 or words <= 0 or size <= 0:
        return target
    pieces = triton.cdiv(int(size) * int(words), AHEAD_CHUNK)
    _ahead_kernel[(max(1, min(int(programs), blocks * pieces)),)](
        host, target, row_ids, counts, blocks, int(size), host.numel() // int(words), WORDS=int(words),
        CHUNK=AHEAD_CHUNK, PIECES=pieces, num_warps=4)
    return target


@triton.jit(do_not_specialize=["ids", "miss", "kept", "full", "sources", "out", "size", "threshold"])
def _block_plan_kernel(ids, miss, kept, full, sources, out, size, threshold, HAS_KEPT: tl.constexpr,
                       SIZE: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    at = tl.arange(0, SIZE)
    inside = at < size
    row = block * size + at
    held = tl.load(ids + row, mask=inside, other=0).to(tl.int64)
    first = tl.load(ids + block * size).to(tl.int64)
    ordered = tl.sum(((held != first + at) & inside).to(tl.int32), 0) == 0
    wanted = inside & (tl.load(miss + row, mask=inside, other=0) != 0)
    whole = ordered & (first >= 0) & (tl.sum(wanted.to(tl.int32), 0) >= threshold)
    if HAS_KEPT:
        whole = whole & (tl.sum((inside & (tl.load(kept + row, mask=inside, other=0) != 0)).to(tl.int32), 0) == 0)
    tl.store(full + block, whole.to(tl.int32))
    tl.store(sources + block, (tl.maximum(first, 0) // size).to(tl.int32))
    tl.store(out + row, (wanted & ~whole).to(out.dtype.element_ty), mask=inside)


def block_plan(ids, miss, kept, size, threshold, blocks):
    full = torch.empty(blocks, dtype=torch.int32, device=ids.device)
    sources = torch.empty(blocks, dtype=torch.int32, device=ids.device)
    out = torch.empty_like(miss)
    flat_out, flat_miss = out.view(-1), miss.reshape(-1)
    tail = blocks * size
    if flat_miss.numel() > tail:
        flat_out[tail:].copy_(flat_miss[tail:])
    if blocks:
        _block_plan_kernel[(blocks,)](ids, flat_miss, ids if kept is None else kept, full, sources, flat_out,
                                      int(size), int(threshold), HAS_KEPT=kept is not None,
                                      SIZE=triton.next_power_of_2(int(size)), num_warps=4)
    return full, sources, out


def resident_rows(resident, row_bytes):
    if resident.dim() < 3:
        return None
    inner = 1
    expected = 1
    for size, stride in reversed(list(zip(resident.shape[2:], resident.stride()[2:]))):
        if size != 1 and stride != expected:
            return None
        expected *= size
        inner *= size
    if inner * resident.element_size() != row_bytes:
        return None
    return torch.as_strided(resident, (resident.shape[0], resident.shape[1], inner),
                            (resident.stride(0), resident.stride(1), 1), resident.storage_offset()).view(torch.uint8)


def stage(source, needed_rows, resident, flat, tile=0):
    target = word_rows(flat)
    row_bytes = flat.shape[1] * flat.element_size()
    view = resident_rows(resident, row_bytes)
    if target is None or view is None:
        return False
    target = target.view(torch.uint8) if target.dtype != torch.uint8 else target
    count = source.numel()
    if count and tile:
        _stage_tile_kernel[(triton.cdiv(count, tile),)](
            source, needed_rows, view, view.stride(0), view.stride(1), target, target.stride(0), resident.shape[1],
            count, ROW=row_bytes, BLOCK=min(pow2(row_bytes), ROW_BLOCK), TILE=int(tile), num_warps=4, PDL=pdl_on(),
            launch_pdl=pdl_on())
    elif count:
        _stage_kernel[(count,)](
            source, needed_rows, view, view.stride(0), view.stride(1), target, target.stride(0), resident.shape[1],
            ROW=row_bytes, BLOCK=min(pow2(row_bytes), ROW_BLOCK), num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return True


GROUP_SEGMENT = 32
GROUP_SHIFT = 16
GROUP_EMPTY = (1 << 63) - 1
GROUP_WIDTHS = (2, 4, 8)
GROUP_WORDS = (torch.int64, torch.int32)


@triton.jit(do_not_specialize=["host", "host_stride", "hot", "hot_block_stride", "hot_token_stride", "hot_size",
                               "picked_host", "picked_device", "counts", "plan_stride", "width"])
def _group_gather_kernel(host, host_stride, hot, hot_block_stride, hot_token_stride, hot_size, picked_host,
                         picked_device, counts, plan_stride, width, WORDS: tl.constexpr, COLS: tl.constexpr,
                         GROUP: tl.constexpr, SEGMENT: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    first = tl.program_id(1) * SEGMENT
    total = tl.minimum(tl.load(counts + lane), width)
    steps = tl.minimum(SEGMENT, total - first)
    base = lane * plan_stride
    slots = tl.arange(0, GROUP)
    for step in range(0, steps):
        at = first + step
        row = tl.load(picked_host + base + at).to(tl.int64)
        group = row // GROUP
        before = tl.load(picked_host + base + at - 1, mask=at > 0, other=0).to(tl.int64)
        if (at == 0) | (before // GROUP != group):
            ahead = at + slots
            inside = ahead < total
            near = tl.load(picked_host + base + ahead, mask=inside, other=-1).to(tl.int64)
            mine = inside & (near // GROUP == group) & (near >= 0)
            target = tl.load(picked_device + base + ahead, mask=mine, other=-1).to(tl.int64)
            member = mine[None, :] & ((near - group * GROUP)[None, :] == slots[:, None])
            slot = tl.max(tl.where(member, target[None, :], -1), axis=1)
            taken = slot >= 0
            safe = tl.where(taken, slot, 0)
            origin = host + (group * GROUP + slots) * host_stride
            landing = hot + (safe // hot_size) * hot_block_stride + (safe % hot_size) * hot_token_stride
            for start in range(0, WORDS, COLS):
                col = start + tl.arange(0, COLS)
                fits = col < WORDS
                value = tl.load(origin[:, None] + col[None, :], mask=fits[None, :])
                tl.store(landing[:, None] + col[None, :], value, mask=taken[:, None] & fits[None, :])


def group_word(host_cache, hot_cache):
    for word in GROUP_WORDS:
        try:
            read = host_cache.view(word)
            write = hot_cache.view(word)
        except RuntimeError:
            continue
        if read.dim() == 2 and write.dim() == 3 and read.stride(1) == 1 and write.stride(2) == 1:
            return read, write
    return None


def group_ready(host_cache, hot_cache, group):
    if group not in GROUP_WIDTHS or hot_cache.dim() != 3 or host_cache.dim() != 2:
        return False
    if hot_cache.shape[1] % group or host_cache.shape[0] % group:
        return False
    if host_cache.shape[-1] != hot_cache.shape[-1] or host_cache.dtype != hot_cache.dtype:
        return False
    return group_word(host_cache, hot_cache) is not None


def group_plan(host_rows, device_rows, counts):
    rows, width = host_rows.shape
    col = torch.arange(width, dtype=torch.int64, device=host_rows.device)
    live = (col.view(1, width) < counts.view(rows, 1).to(torch.int64)) & (host_rows >= 0)
    key = torch.where(live, (host_rows.to(torch.int64) << GROUP_SHIFT) | col.view(1, width),
                      torch.full((rows, width), GROUP_EMPTY, dtype=torch.int64, device=host_rows.device))
    key = key.sort(dim=1).values
    kept = key != GROUP_EMPTY
    origin = torch.where(kept, key & ((1 << GROUP_SHIFT) - 1), torch.zeros_like(key))
    picked_host = torch.where(kept, key >> GROUP_SHIFT, torch.full_like(key, -1)).to(torch.int32)
    picked_device = device_rows.gather(1, origin).to(torch.int32)
    return picked_host.contiguous(), picked_device.contiguous(), kept.sum(1, dtype=torch.int32)


def group_gather(host_cache, hot_cache, host_rows, device_rows, counts, group, plan=None):
    rows, width = host_rows.shape
    if rows == 0 or width == 0:
        return None
    if width >= 1 << GROUP_SHIFT:
        raise ValueError("a gather plan this wide cannot be ranked by position")
    read, write = group_word(host_cache, hot_cache)
    picked_host, picked_device, kept = group_plan(host_rows, device_rows, counts) if plan is None else plan
    words = read.shape[1]
    _group_gather_kernel[(rows, triton.cdiv(width, GROUP_SEGMENT))](
        read, read.stride(0), write, write.stride(0), write.stride(1), write.shape[1], picked_host, picked_device,
        kept, picked_host.stride(0), width, WORDS=words, COLS=min(pow2(words), ROW_BLOCK // group), GROUP=group,
        SEGMENT=GROUP_SEGMENT, num_warps=4)
    return kept


RUN_TILE = 8
RUN_SEGMENT = 32
RUN_COLS = 128
RUN_MAX_GAP = 64
VECTOR_BYTES = 16


@triton.jit
def _run_continues(picked_host, base, at, total, gap):
    live = (at > 0) & (at < total)
    row = tl.load(picked_host + base + at, mask=live, other=0).to(tl.int64)
    before = tl.load(picked_host + base + at - 1, mask=live, other=0).to(tl.int64)
    return live & (row - before <= gap + 1)


@triton.jit(do_not_specialize=["host_stride", "hot_block_stride", "hot_token_stride", "hot_size", "picked_host",
                               "picked_device", "counts", "plan_stride", "width", "gap"])
def _run_gather_kernel(host, host_stride, hot, hot_block_stride, hot_token_stride, hot_size, picked_host,
                       picked_device, counts, plan_stride, width, gap, WORDS: tl.constexpr, COLS: tl.constexpr,
                       TILE: tl.constexpr, SEGMENT: tl.constexpr, VEC: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    total = tl.minimum(tl.load(counts + lane), width)
    base = lane * plan_stride
    limit = (segment + 1) * SEGMENT
    at = segment * SEGMENT
    if segment > 0:
        while _run_continues(picked_host, base, at, total, gap):
            at += 1
    slots = tl.arange(0, TILE)
    while (at < total) & ((at < limit) | _run_continues(picked_host, base, at, total, gap)):
        first = tl.load(picked_host + base + at).to(tl.int64)
        ahead = at + slots
        inside = ahead < total
        near = tl.load(picked_host + base + ahead, mask=inside, other=-1).to(tl.int64)
        previous = tl.load(picked_host + base + ahead - 1, mask=inside & (slots > 0), other=-1).to(tl.int64)
        linked = (slots == 0) | ((near - previous) <= gap + 1)
        chained = tl.cumsum((~linked).to(tl.int32), 0) == 0
        rel = near - first
        mine = inside & chained & (rel >= 0) & (rel < TILE)
        target = tl.load(picked_device + base + ahead, mask=mine, other=-1).to(tl.int64)
        member = mine[None, :] & (rel[None, :] == slots[:, None])
        slot = tl.max(tl.where(member, target[None, :], -1), axis=1)
        last = tl.max(tl.where(mine, rel, 0), axis=0)
        used = tl.sum(mine.to(tl.int32), axis=0)
        reading = slots <= last
        taken = slot >= 0
        safe = tl.where(taken, slot, 0)
        origin = tl.multiple_of((first + slots) * host_stride, VEC)
        landing = tl.multiple_of((safe // hot_size) * hot_block_stride + (safe % hot_size) * hot_token_stride, VEC)
        for start in range(0, WORDS, COLS):
            col = tl.max_contiguous(tl.multiple_of(start + tl.arange(0, COLS), VEC), VEC)
            fits = col < WORDS
            value = tl.load(host + origin[:, None] + col[None, :], mask=reading[:, None] & fits[None, :])
            tl.store(hot + landing[:, None] + col[None, :], value, mask=taken[:, None] & fits[None, :])
        at += tl.maximum(used, 1)


def run_vector(read, write):
    unit = read.element_size()
    for vec in (4, 2):
        step = vec * unit
        if step > VECTOR_BYTES:
            continue
        if (read.shape[1] % vec or (read.stride(0) * unit) % step or read.data_ptr() % step
                or (write.stride(0) * unit) % step or (write.stride(1) * unit) % step or write.data_ptr() % step):
            continue
        return vec
    return 1


def run_ready(host_cache, hot_cache):
    return spread_ready(host_cache, hot_cache)


def run_gather(host_cache, hot_cache, host_rows, device_rows, counts, gap, tile=RUN_TILE, plan=None):
    rows, width = host_rows.shape
    if rows == 0 or width == 0:
        return None
    if width >= 1 << GROUP_SHIFT:
        raise ValueError("a gather plan this wide cannot be ranked by position")
    if not 0 <= int(gap) <= RUN_MAX_GAP:
        raise ValueError("a run merge gap is between 0 and RUN_MAX_GAP rows")
    read, write = group_word(host_cache, hot_cache)
    picked_host, picked_device, kept = group_plan(host_rows, device_rows, counts) if plan is None else plan
    words = read.shape[1]
    _run_gather_kernel[(rows, triton.cdiv(width, RUN_SEGMENT))](
        read, read.stride(0), write, write.stride(0), write.stride(1), write.shape[1], picked_host, picked_device,
        kept, picked_host.stride(0), width, int(gap), WORDS=words, COLS=min(pow2(words), RUN_COLS), TILE=int(tile),
        SEGMENT=RUN_SEGMENT, VEC=run_vector(read, write), num_warps=4)
    return kept


PCIE_FIELDS = ("calls", "rows", "group2", "group4", "group8", "run0", "run2", "run8", "span0", "span2", "span8",
               "transfers", "rows_read", "clock_ns")
PCIE_GROUPS = (2, 4, 8)
PCIE_GAPS = (0, 2, 8)
PCIE_FAR = 1 << 40


def pcie_stats(host_rows, counts, group=0, gap=-1):
    lanes, width = host_rows.shape
    device = host_rows.device
    col = torch.arange(width, dtype=torch.int64, device=device).view(1, width)
    live = (col < counts[:lanes].view(-1, 1).to(torch.int64)) & (host_rows >= 0)
    keys = torch.where(live, host_rows.to(torch.int64), torch.full((lanes, width), PCIE_FAR, dtype=torch.int64,
                                                                    device=device))
    ordered = keys.sort(dim=1).values
    valid = ordered != PCIE_FAR
    rows = valid.sum(dtype=torch.int64)
    lead = torch.full((lanes, 1), -PCIE_FAR, dtype=torch.int64, device=device)
    before = torch.cat((lead, ordered[:, :-1]), dim=1)
    out = [torch.ones((), dtype=torch.int64, device=device), rows]
    groups = {}
    for size in PCIE_GROUPS:
        cell = torch.where(valid, ordered // size, torch.full_like(ordered, -PCIE_FAR))
        left = torch.cat((lead, cell[:, :-1]), dim=1)
        groups[size] = (valid & (cell != left)).sum(dtype=torch.int64)
        out.append(groups[size])
    starts, spans = {}, {}
    for size in PCIE_GAPS:
        head = valid & ((ordered - before) > size + 1)
        starts[size] = head.sum(dtype=torch.int64)
        inner = (valid & ~head).to(torch.int64) * (ordered - before)
        spans[size] = starts[size] + inner.sum(dtype=torch.int64)
    out += [starts[size] for size in PCIE_GAPS] + [spans[size] for size in PCIE_GAPS]
    if gap >= 0:
        head = valid & ((ordered - before) > gap + 1)
        run_count = head.sum(dtype=torch.int64)
        inner = (valid & ~head).to(torch.int64) * (ordered - before)
        out += [run_count, run_count + inner.sum(dtype=torch.int64)]
    elif group > 1:
        out += [groups[group], groups[group] * group]
    else:
        out += [rows, rows]
    out.append(torch.zeros((), dtype=torch.int64, device=device))
    return torch.stack(out)


@triton.jit
def _stamp_asm(seed):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l", [seed], dtype=tl.int64, is_pure=False,
                                     pack=1)


@triton.jit
def _stamp_kernel(clock, total, END: tl.constexpr):
    now = tl.sum(_stamp_asm(tl.zeros((1,), tl.int64)), 0)
    if END:
        tl.atomic_add(total, now - tl.load(clock))
    else:
        tl.store(clock, now)


def stamp(clock, total, end):
    if os.environ.get("TRITON_INTERPRET") == "1":
        import time

        if end:
            total.add_(time.perf_counter_ns() - int(clock.item()))
        else:
            clock.fill_(time.perf_counter_ns())
        return
    _stamp_kernel[(1,)](clock, total, END=bool(end), num_warps=1)


SPREAD_ROWS = 4
SPREAD_LANES = 128
SPREAD_COLS = 1024


@triton.jit(do_not_specialize=["host", "host_stride", "hot", "hot_block_stride", "hot_token_stride", "hot_size",
                               "host_rows", "host_plan_stride", "device_rows", "device_plan_stride", "counts",
                               "lanes", "width"])
def _spread_gather_kernel(host, host_stride, hot, hot_block_stride, hot_token_stride, hot_size, host_rows,
                          host_plan_stride, device_rows, device_plan_stride, counts, lanes, width,
                          WORDS: tl.constexpr, COLS: tl.constexpr, ROWS: tl.constexpr, LANES: tl.constexpr):
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    total = tl.sum(tl.zeros((LANES,), tl.int32), 0)
    for first in range(0, lanes, LANES):
        lane = first + tl.arange(0, LANES)
        held = tl.load(counts + lane, mask=lane < lanes, other=0).to(tl.int32)
        total += tl.sum(tl.minimum(tl.maximum(held, 0), width), 0)
    share = (total + programs - 1) // programs
    start = program * share
    stop = tl.minimum(start + share, total)
    for at in range(start, stop, ROWS):
        index = at + tl.arange(0, ROWS)
        inside = index < stop
        owner = tl.zeros((ROWS,), tl.int32)
        before = tl.zeros((ROWS,), tl.int32)
        seen = total * 0
        for first in range(0, lanes, LANES):
            lane = first + tl.arange(0, LANES)
            held = tl.load(counts + lane, mask=lane < lanes, other=0).to(tl.int32)
            held = tl.minimum(tl.maximum(held, 0), width)
            ends = seen + tl.cumsum(held, 0)
            begins = ends - held
            hit = (index[:, None] >= begins[None, :]) & (index[:, None] < ends[None, :])
            owner += tl.sum(tl.where(hit, lane[None, :], 0), 1)
            before += tl.sum(tl.where(hit, begins[None, :], 0), 1)
            seen += tl.sum(held, 0)
        column = (index - before).to(tl.int64)
        lane_at = owner.to(tl.int64)
        row = tl.load(host_rows + lane_at * host_plan_stride + column, mask=inside, other=-1).to(tl.int64)
        slot = tl.load(device_rows + lane_at * device_plan_stride + column, mask=inside, other=-1).to(tl.int64)
        taken = inside & (row >= 0) & (slot >= 0)
        origin = host + tl.where(taken, row, 0) * host_stride
        safe = tl.where(taken, slot, 0)
        landing = hot + (safe // hot_size) * hot_block_stride + (safe % hot_size) * hot_token_stride
        for begin in range(0, WORDS, COLS):
            col = begin + tl.arange(0, COLS)
            keep = taken[:, None] & (col < WORDS)[None, :]
            value = tl.load(origin[:, None] + col[None, :], mask=keep)
            tl.store(landing[:, None] + col[None, :], value, mask=keep)


def spread_ready(host_cache, hot_cache):
    if hot_cache.dim() != 3 or host_cache.dim() != 2:
        return False
    if host_cache.shape[-1] != hot_cache.shape[-1] or host_cache.dtype != hot_cache.dtype:
        return False
    return group_word(host_cache, hot_cache) is not None


def spread_programs(lanes, width, programs, rows=SPREAD_ROWS):
    return max(1, min(int(programs), -(-int(lanes) * int(width) // int(rows))))


def spread_gather(host_cache, hot_cache, host_rows, device_rows, counts, programs, rows=SPREAD_ROWS):
    lanes, width = host_rows.shape
    if lanes == 0 or width == 0:
        return 0
    if (tuple(device_rows.shape) != (lanes, width) or host_rows.stride(1) != 1 or device_rows.stride(1) != 1
            or counts.dim() != 1 or counts.shape[0] < lanes or counts.stride(0) != 1):
        raise ValueError("a spread gather needs a plan of inner-contiguous rows and one count a row")
    read, write = group_word(host_cache, hot_cache)
    words = read.shape[1]
    grid = spread_programs(lanes, width, programs, rows)
    _spread_gather_kernel[(grid,)](
        read, read.stride(0), write, write.stride(0), write.stride(1), write.shape[1], host_rows, host_rows.stride(0),
        device_rows, device_rows.stride(0), counts, lanes, width, WORDS=words, COLS=min(pow2(words), SPREAD_COLS),
        ROWS=rows, LANES=SPREAD_LANES, num_warps=4)
    return grid


@triton.jit(do_not_specialize=["counts", "idle", "pieces", "host_rows", "device_rows", "host_stride", "device_stride",
                               "factor", "width", "host_total", "hot_total", "split_tally", "guard_tally"])
def _plan_gather_kernel(counts, idle, pieces, host_rows, device_rows, host_stride, device_stride, factor, width,
                        host_total, hot_total, split_tally, guard_tally, HAS_IDLE: tl.constexpr, TALLY: tl.constexpr,
                        BLOCK: tl.constexpr, PADDED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    at = tl.program_id(0).to(tl.int64)
    lane = at // factor
    piece = at % factor
    first = tl.arange(0, BLOCK)
    count = tl.load(counts + lane).to(tl.int64)
    if HAS_IDLE:
        resting = tl.load(idle + lane).to(tl.int1)
        count = tl.where(resting, 0, count)
        tl.store(counts + lane + first, tl.zeros([BLOCK], count.dtype) + count, mask=(first == 0) & (piece == 0))
    live = tl.minimum(tl.maximum(count - piece * width, 0), width)
    if TALLY:
        tl.atomic_add(split_tally, tl.where(piece == 0, tl.minimum(count, factor * width), 0))
        tl.atomic_add(split_tally + 1, ((piece == 0) & (count > 0)).to(tl.int64))
    flagged = tl.zeros([BLOCK], tl.int32)
    for start in range(0, PADDED, BLOCK):
        column = start + first
        inside = column < live
        host = tl.load(host_rows + at * host_stride + column, mask=inside, other=0).to(tl.int64)
        hot = tl.load(device_rows + at * device_stride + column, mask=inside, other=0).to(tl.int64)
        bad = inside & (host >= 0) & ((host >= host_total) | (hot < 0) | (hot >= hot_total))
        flagged = tl.maximum(flagged, bad.to(tl.int32))
    over = tl.max(flagged, 0) > 0
    kept = tl.where(over, 0, live).to(tl.int32)
    tl.store(pieces + at + first, tl.zeros([BLOCK], tl.int32) + kept, mask=first == 0)
    tl.atomic_add(guard_tally, over.to(tl.int64))


def plan_gather(counts, idle, factor, host_rows, device_rows, host_total, hot_total, split_tally, guard_tally):
    decodes = counts.shape[0]
    total = decodes * factor
    pieces = torch.empty(total, dtype=torch.int32, device=counts.device)
    lanes, width = host_rows.shape
    if total and width:
        _plan_gather_kernel[(total,)](
            counts, counts if idle is None else idle, pieces, host_rows, device_rows, host_rows.stride(0),
            device_rows.stride(0), factor, width, host_total, hot_total,
            pieces if split_tally is None else split_tally, guard_tally, HAS_IDLE=idle is not None,
            TALLY=split_tally is not None, BLOCK=GUARD_BLOCK, PADDED=triton.cdiv(width, GUARD_BLOCK) * GUARD_BLOCK,
            num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return pieces


NEAR_BANDS = 64
NEAR_RATIO = 0.985
NEAR_CHUNK = 2048


@triton.jit
def _near_band(v, cut, valid, SCALE: tl.constexpr, BANDS: tl.constexpr):
    safe = tl.where(valid, v, cut)
    band = (tl.log2(cut / safe) * SCALE).to(tl.int32)
    return tl.where(valid & (band < BANDS - 1), band, BANDS - 1)


@triton.jit(do_not_specialize=["logits", "logits_stride", "visible", "picked", "picked_stride", "near",
                               "near_stride", "columns"])
def _near_kernel(logits, logits_stride, visible, picked, picked_stride, near, near_stride, columns,
                 PICKS: tl.constexpr, PICK_BLOCK: tl.constexpr, NEAR: tl.constexpr, NEAR_BLOCK: tl.constexpr,
                 BANDS: tl.constexpr, SCALE: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    out = near + row.to(tl.int64) * near_stride
    k = tl.arange(0, NEAR_BLOCK)
    tl.store(out + k, tl.full((NEAR_BLOCK,), -1, tl.int32), mask=k < NEAR)
    seen = tl.minimum(tl.load(visible + row).to(tl.int32), columns)
    base = logits + row.to(tl.int64) * logits_stride
    j = tl.arange(0, PICK_BLOCK)
    sel = tl.load(picked + row.to(tl.int64) * picked_stride + j, mask=j < PICKS, other=-1)
    ok = (sel >= 0) & (sel < seen)
    vals = tl.load(base + tl.where(ok, sel, 0), mask=ok, other=float("inf"))
    cut = tl.min(vals, 0)
    chosen = tl.sum(ok.to(tl.int32), 0)
    if (chosen >= PICKS) & (cut > 0.0):
        b = tl.arange(0, BANDS)
        counts = tl.zeros((BANDS,), tl.int32)
        for start in range(0, seen, CHUNK):
            c = start + tl.arange(0, CHUNK)
            v = tl.load(base + c, mask=c < seen, other=0.0)
            band = _near_band(v, cut, (c < seen) & (v < cut) & (v > 0.0), SCALE, BANDS)
            counts += tl.histogram(band, BANDS)
        counts = tl.where(b < BANDS - 1, counts, 0)
        total = tl.cumsum(counts, 0)
        last = tl.max(tl.where((total <= NEAR) & (b < BANDS - 1), b, -1), 0)
        top = tl.maximum(last, 0)
        filled = tl.sum(tl.zeros((CHUNK,), tl.int32), 0)
        for start in range(0, seen, CHUNK):
            c = start + tl.arange(0, CHUNK)
            v = tl.load(base + c, mask=c < seen, other=0.0)
            band = _near_band(v, cut, (c < seen) & (v < cut) & (v > 0.0), SCALE, BANDS)
            take = band <= top
            at = filled + tl.cumsum(take.to(tl.int32), 0) - 1
            tl.store(out + at, c.to(tl.int32), mask=take & (at < NEAR))
            filled += tl.sum(take.to(tl.int32), 0)


def near(logits, visible, picked, out, ratio=NEAR_RATIO, bands=NEAR_BANDS):
    rows = out.shape[0]
    if rows == 0:
        return out
    if logits.stride(1) != 1 or picked.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("near-miss selection needs inner-contiguous logits, picks and output")
    _near_kernel[(rows,)](
        logits, logits.stride(0), visible, picked, picked.stride(0), out, out.stride(0), logits.shape[1],
        PICKS=picked.shape[1], PICK_BLOCK=pow2(picked.shape[1]), NEAR=out.shape[1], NEAR_BLOCK=pow2(out.shape[1]),
        BANDS=bands, SCALE=-1.0 / math.log2(float(ratio)), CHUNK=NEAR_CHUNK, num_warps=8)
    return out


TAKE_BLOCK = 1024


@triton.jit
def _take_rows_kernel(out, out_stride, landing, landing_stride, lead, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(lead):
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < width
    held = tl.load(landing + row.to(tl.int64) * landing_stride + cols, mask=live)
    tl.store(out + row.to(tl.int64) * out_stride + cols, held, mask=live)


def take_rows(out, landing, lead):
    rows = int(out.shape[0])
    target = out.reshape(rows, -1)
    source = landing.reshape(rows, -1)
    width = int(target.shape[1])
    if rows == 0 or width == 0:
        return out
    _take_rows_kernel[(rows, triton.cdiv(width, TAKE_BLOCK))](target, target.stride(0), source, source.stride(0),
                                                              lead, width, BLOCK=TAKE_BLOCK, num_warps=4)
    return out


@triton.jit
def _rows_below_kernel(source, source_stride, target, target_stride, width, lead, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < width
    keep = row < tl.load(lead)
    held = tl.load(source + row.to(tl.int64) * source_stride + cols, mask=live & keep, other=-1)
    tl.store(target + row.to(tl.int64) * target_stride + cols, held, mask=live)


def rows_below(source, lead):
    rows = int(source.shape[0])
    flat = source.reshape(rows, -1)
    target = torch.empty_like(flat)
    width = int(flat.shape[1])
    if rows and width:
        _rows_below_kernel[(rows, triton.cdiv(width, TAKE_BLOCK))](flat, flat.stride(0), target, target.stride(0),
                                                                   width, lead, BLOCK=TAKE_BLOCK, num_warps=4)
    return target.view(source.shape)


def near_positions(near_blocks, ratio):
    rows, width = near_blocks.shape
    span = torch.arange(int(ratio), dtype=torch.int32, device=near_blocks.device)
    held = near_blocks.unsqueeze(2)
    positions = torch.where(held >= 0, held * int(ratio) + span.view(1, 1, -1), torch.full_like(held, -1))
    return positions.reshape(rows, width * int(ratio)).contiguous()


def spread_self_check(row_bytes, per_sm, device="cuda", seed=0, hosts=4096, blocks=64, tokens=64, lanes=8,
                      width=512):
    generator = torch.Generator().manual_seed(seed)
    host = torch.randint(0, 256, (hosts, row_bytes), generator=generator, dtype=torch.uint8).pin_memory()
    raw = torch.full((blocks * (tokens + 1) * row_bytes,), 7, dtype=torch.uint8, device=device)
    hot = torch.as_strided(raw, (blocks, tokens, row_bytes), ((tokens + 1) * row_bytes, row_bytes, 1))
    if not spread_ready(host, hot):
        return [("spread_gather", False, f"rows of {row_bytes} bytes are not word-aligned for the spread gather")]
    slots = blocks * tokens
    counts = torch.tensor([(lane * 131) % (width + 1) - (1 if lane == 1 else 0) for lane in range(lanes)],
                          dtype=torch.int32)
    host_rows = torch.randint(0, hosts, (lanes, width), generator=generator, dtype=torch.int32)
    device_rows = torch.randperm(slots, generator=generator)[:lanes * width].view(lanes, width).to(torch.int32)
    host_rows, device_rows, counts = host_rows.to(device), device_rows.to(device), counts.to(device)
    want = hot.clone()
    live = (torch.arange(width, device=device).view(1, -1) < counts.clamp(0, width).view(-1, 1)) & (host_rows >= 0)
    picked = device_rows[live].long()
    want[picked // tokens, picked % tokens] = host.to(device)[host_rows[live].long()]
    programs = torch.cuda.get_device_properties(device).multi_processor_count * max(1, int(per_sm))
    grid = spread_gather(host, hot, host_rows, device_rows, counts, programs)
    torch.cuda.synchronize()
    same = bool(torch.equal(hot, want))
    return [("spread_gather", same, f"{int(live.sum())} host rows of {row_bytes} bytes over {grid} programs "
                                    f"land bit for bit: {same}")]



MIRROR_BLOCK = 512
MIRROR_EMPTY = 1 << 62


@triton.jit
def _mirror_kernel(slots, count, ref_starts, counts, dests, layer_starts, entries, src, src_s0, src_s1, src_blocks,
                   dst, dst_rows, kernel_block, tally, SEARCH: tl.constexpr, ROW: tl.constexpr,
                   BLOCK: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    live = (token < tl.load(count).to(tl.int32)) & (slot >= 0)
    lo = (token * 0).to(tl.int32)
    hi = (entries - 1).to(tl.int32)
    for _ in range(SEARCH):
        mid = (lo + hi + 1) // 2
        reach = tl.load(ref_starts + mid) <= slot
        lo = tl.where(reach, mid, lo)
        hi = tl.where(reach, hi, mid - 1)
    first = tl.load(ref_starts + lo)
    offset = slot - first
    hit = live & (first <= slot) & (offset < tl.load(counts + lo))
    row = tl.load(dests + lo) + offset
    source = tl.load(layer_starts + lo) + offset
    block = source // kernel_block
    inside = (source >= 0) & (block < src_blocks) & (row >= 0) & (row < dst_rows)
    ok = hit & inside
    tl.atomic_add(tally, ok.to(tl.int64))
    tl.atomic_add(tally + 1, (hit & ~inside).to(tl.int64))
    lane = tl.arange(0, BLOCK)
    src_at = block * src_s0 + (source % kernel_block) * src_s1
    dst_at = row * ROW
    for start in tl.static_range(0, ROW, BLOCK):
        keep = ((start + lane) < ROW) & ok
        tl.store(dst + dst_at + start + lane, tl.load(src + src_at + start + lane, mask=keep), mask=keep)


def mirror_search_steps(entries):
    steps = 1
    while (1 << steps) < max(entries, 2):
        steps += 1
    return steps + 1


def mirror_rows(slots, count, ref_starts, counts, dests, layer_starts, src, dst, kernel_block, tally, tokens):
    row_bytes = src.shape[2]
    entries = ref_starts.numel()
    if tokens:
        _mirror_kernel[(tokens,)](
            slots, count, ref_starts, counts, dests, layer_starts, entries, src, src.stride(0), src.stride(1),
            src.shape[0], dst, dst.numel() // row_bytes, kernel_block, tally,
            SEARCH=mirror_search_steps(entries), ROW=row_bytes, BLOCK=min(pow2(row_bytes), MIRROR_BLOCK), num_warps=4)


@triton.jit(do_not_specialize=["values", "count", "low", "high", "tally"])
def _guard_range_kernel(values, count, low, high, tally, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = at < count
    held = tl.load(values + at, mask=inside, other=0)
    bad = inside & ((held < low) | (held >= high))
    tl.store(values + at, tl.full((BLOCK,), -1, values.dtype.element_ty), mask=bad)
    tl.atomic_add(tally, tl.sum(bad.to(tl.int64), 0))


@triton.jit(do_not_specialize=["picks", "stride", "width", "positions", "tally", "split"])
def _guard_picks_kernel(picks, stride, width, positions, tally, split, BLOCK: tl.constexpr, PADDED: tl.constexpr,
                        LIVE: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if LIVE:
        if tl.program_id(0) >= tl.load(split + 4):
            return
    row = picks + tl.program_id(0).to(tl.int64) * stride
    first = tl.arange(0, BLOCK)
    found = tl.zeros([BLOCK], tl.int32)
    for start in range(0, PADDED, BLOCK):
        at = start + first
        inside = at < width
        held = tl.load(row + at, mask=inside, other=0)
        bad = inside & ((held < -1) | (held >= positions))
        tl.store(row + at, tl.full((BLOCK,), -1, picks.dtype.element_ty), mask=bad)
        found += bad.to(tl.int32)
    total = tl.sum(found, 0)
    kept = tl.load(row + width + first, mask=first == 0, other=0)
    tl.store(row + width + first, kept - total.to(picks.dtype.element_ty), mask=(first == 0) & (total > 0))
    tl.atomic_add(tally, total.to(tl.int64))


@triton.jit(do_not_specialize=["host_rows", "device_rows", "counts", "out", "host_stride", "device_stride", "width",
                               "host_total", "hot_total", "tally"])
def _guard_gather_kernel(host_rows, device_rows, counts, out, host_stride, device_stride, width, host_total,
                         hot_total, tally, BLOCK: tl.constexpr, PADDED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    lane = tl.program_id(0).to(tl.int64)
    live = tl.minimum(tl.maximum(tl.load(counts + lane).to(tl.int64), 0), width)
    first = tl.arange(0, BLOCK)
    flagged = tl.zeros([BLOCK], tl.int32)
    for start in range(0, PADDED, BLOCK):
        at = start + first
        inside = at < live
        host = tl.load(host_rows + lane * host_stride + at, mask=inside, other=0).to(tl.int64)
        hot = tl.load(device_rows + lane * device_stride + at, mask=inside, other=0).to(tl.int64)
        bad = inside & (host >= 0) & ((host >= host_total) | (hot < 0) | (hot >= hot_total))
        flagged = tl.maximum(flagged, bad.to(tl.int32))
    over = tl.max(flagged, 0) > 0
    held = tl.load(counts + lane + first, mask=first == 0, other=0)
    tl.store(out + lane + first, tl.where(over, tl.zeros_like(held), held), mask=first == 0)
    tl.atomic_add(tally, over.to(tl.int64))


GUARD_BLOCK = 1024


def guard_range(values, low, high, tally):
    count = values.numel()
    if count:
        _guard_range_kernel[(triton.cdiv(count, GUARD_BLOCK),)](values, count, low, high, tally, BLOCK=GUARD_BLOCK,
                                                                num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return values


def guard_picks(logical, positions, tally, split=None):
    rows, width = logical.shape[0], logical.shape[1] - 1
    if rows and width > 0:
        _guard_picks_kernel[(rows,)](logical, logical.stride(0), width, positions, tally,
                                     logical if split is None else split, BLOCK=GUARD_BLOCK,
                                     PADDED=triton.cdiv(width, GUARD_BLOCK) * GUARD_BLOCK, LIVE=split is not None,
                                     num_warps=4, PDL=pdl_on(), launch_pdl=pdl_on())
    return logical


def guard_gather(host_rows, device_rows, counts, host_total, hot_total, tally):
    out = torch.empty_like(counts)
    lanes, width = host_rows.shape
    if lanes and width:
        _guard_gather_kernel[(lanes,)](host_rows, device_rows, counts, out, host_rows.stride(0), device_rows.stride(0),
                                       width, host_total, hot_total, tally, BLOCK=GUARD_BLOCK,
                                       PADDED=triton.cdiv(width, GUARD_BLOCK) * GUARD_BLOCK, num_warps=4,
                                       PDL=pdl_on(), launch_pdl=pdl_on())
    return out


ONE_SHARE = 4
ONE_TILE = 8
ONE_WARPS = 16
ONE_TILE_ENV = "TAAF_VLLM_HOST_KV_ONE_SCAN_TILE"
ONE_WARPS_ENV = "TAAF_VLLM_HOST_KV_ONE_WARPS"
_ONE_SHAPE = {}
ONE_MAX_BLOCK = 8192
ONE_REGION_TILE = 8192
ONE_MAX_REGION = 16384
ONE_SKIP_TILE = 2048
FLAT_TILE = 16
FLAT_PER_SM = 4
FLAT_WARPS = 4
FLAT_HOST_PROGRAMS = 8
FLAT_LINES_ENV = "TAAF_VLLM_HOST_KV_FLAT_LINES"
FLAT_PROGRAMS_ENV = "TAAF_VLLM_HOST_KV_FLAT_PROGRAMS"
LINE_BYTES = 128
LINE_GROUP = 4
FLAT_LINE_WARPS = 8
ONE_PACK_FLOOR = 4
ONE_PACK_SHIFTS = (3, 2, 1)
ONE_HASH_SPREAD = 4
ONE_HASH_BUCKET = 0
ONE_HASH_SPREAD_ENV = "TAAF_VLLM_HOST_KV_ONE_HASH_SPREAD"
ONE_HASH_BUCKET_ENV = "TAAF_VLLM_HOST_KV_ONE_HASH_BUCKET"
ONE_HASH_FLOOR = 64
ONE_PHASES = ("resolve_ns", "resolve_max_ns", "sort_ns", "scan_ns", "tail_ns", "copy_ns", "wait_ns", "launches",
              "hollow_parts", "parts", "resident_parts", "lanes", "pack_parts", "span_start", "span_ns",
              "rows", "rows_max", "copy_max_ns", "resolve_span_ns", "resolve_end", "slow_lanes", "slow_rows",
              "b0_n", "b1_n", "b2_n", "b3_n", "b4_n", "b0_ns", "b1_ns", "b2_ns", "b3_ns", "b4_ns",
              "insert_rounds", "inserts")
ONE_EMPTY = 1 << 62
ONE_MARK = -(1 << 31)
ONE_PCIE = {name: PCIE_FIELDS.index(name) for name in ("calls", "rows", "transfers", "rows_read")}


@triton.jit
def _one_now():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _one_tile_sort(keys, WIDTH: tl.constexpr):
    at = tl.arange(0, WIDTH)
    tl.store(keys + at, tl.sort(tl.load(keys + at)))


@triton.jit
def _one_packed(keys, key, live, count, fired, BLOCK: tl.constexpr, SENTINEL: tl.constexpr, PACK1: tl.constexpr,
                PACK2: tl.constexpr, PACK3: tl.constexpr, COUNTED: tl.constexpr):
    at = tl.arange(0, BLOCK)
    tl.store(keys + at, tl.full((BLOCK,), SENTINEL, key.dtype), mask=at >= count)
    tl.store(keys + tl.cumsum(live.to(tl.int32), 0) - 1, key, mask=live)
    tl.debug_barrier()
    if count <= PACK1:
        _one_tile_sort(keys, PACK1)
        if COUNTED and PACK1 < BLOCK:
            tl.atomic_add(fired, 1)
    else:
        if count <= PACK2:
            _one_tile_sort(keys, PACK2)
            if COUNTED and PACK2 < BLOCK:
                tl.atomic_add(fired, 1)
        else:
            if count <= PACK3:
                _one_tile_sort(keys, PACK3)
                if COUNTED and PACK3 < BLOCK:
                    tl.atomic_add(fired, 1)
            else:
                _one_tile_sort(keys, BLOCK)


@triton.jit
def _one_home(value, HLOG: tl.constexpr):
    return ((value * 2654435761) & 0xFFFFFFFF) >> (32 - HLOG)


@triton.jit
def _one_insert(table, g, key, live, LOG: tl.constexpr, HLOG: tl.constexpr, EMPTY: tl.constexpr):
    probe = _one_home(g, HLOG)
    active = live
    reach = tl.sum(tl.zeros((1,), tl.int32), 0)
    while tl.max(active.to(tl.int32), 0) > 0:
        at = table + (probe & ((1 << HLOG) - 1))
        held = tl.load(at, mask=active, other=0)
        tl.store(at, key, mask=active & (held == EMPTY))
        tl.debug_barrier()
        now = tl.load(at, mask=active, other=0)
        tl.debug_barrier()
        twin = active & (now != key) & ((now >> LOG) == g)
        tl.atomic_min(at, key, mask=twin)
        active = active & (now != key) & ~twin
        probe = tl.where(active, probe + 1, probe)
        reach += 1
    tl.debug_barrier()
    return probe & ((1 << HLOG) - 1), reach


@triton.jit
def _one_lookup(table, cached, look, reach, LOG: tl.constexpr, KB: tl.constexpr, HLOG: tl.constexpr,
                EMPTY: tl.constexpr):
    probe = _one_home(cached, HLOG)
    column = tl.zeros(cached.shape, tl.int32)
    hit = look & (cached != cached)
    left = look
    step = tl.sum(tl.zeros((1,), tl.int32), 0)
    while (step < reach) & (tl.max(left.to(tl.int32), 0) > 0):
        entry = tl.load(table + (probe & ((1 << HLOG) - 1)), mask=left, other=EMPTY)
        match = left & (entry != EMPTY) & ((entry >> LOG) == cached)
        column = tl.where(match, (entry & (KB - 1)).to(tl.int32), column)
        hit = hit | match
        left = left & ~match & (entry != EMPTY)
        probe += 1
        step += 1
    return hit, column


@triton.jit
def _one_bucket_insert(table, g, key, live, HLOG: tl.constexpr, WIDTH: tl.constexpr):
    buckets: tl.constexpr = (1 << HLOG) // WIDTH
    counts = table + (1 << HLOG)
    bucket = _one_home(g, HLOG) // WIDTH
    slot = tl.zeros(g.shape, tl.int64)
    tries = tl.zeros(g.shape, tl.int32)
    active = live
    reach = tl.sum(tl.zeros((1,), tl.int32), 0)
    while tl.max(active.to(tl.int32), 0) > 0:
        at = (bucket + tries) & (buckets - 1)
        taken = tl.atomic_add(counts + at, tl.full(g.shape, 1, tl.int64), mask=active, sem="relaxed")
        placed = active & (taken < WIDTH)
        here = at * WIDTH + taken
        tl.store(table + here, key, mask=placed)
        slot = tl.where(placed, here, slot)
        tries = tl.where(active, tries + 1, tries)
        active = active & ~placed
        reach += 1
    tl.debug_barrier()
    return slot, reach, tries


@triton.jit
def _one_bucket_lookup(table, cached, look, reach, LOG: tl.constexpr, KB: tl.constexpr, HLOG: tl.constexpr,
                       WIDTH: tl.constexpr, EMPTY: tl.constexpr):
    buckets: tl.constexpr = (1 << HLOG) // WIDTH
    counts = table + (1 << HLOG)
    bucket = _one_home(cached, HLOG) // WIDTH
    lanes = tl.arange(0, WIDTH)
    best = tl.full(cached.shape, KB, tl.int32)
    left = look
    step = tl.sum(tl.zeros((1,), tl.int32), 0)
    while (step < reach) & (tl.max(left.to(tl.int32), 0) > 0):
        at = (bucket + step) & (buckets - 1)
        entry = tl.load(table + at[:, None] * WIDTH + lanes[None, :], mask=left[:, None], other=EMPTY)
        held = tl.load(counts + at, mask=left, other=0)
        match = left[:, None] & (entry != EMPTY) & ((entry >> LOG) == cached[:, None])
        best = tl.minimum(best, tl.min(tl.where(match, (entry & (KB - 1)).to(tl.int32), KB), 1))
        left = left & (held > WIDTH)
        step += 1
    return look & (best < KB), tl.where(best < KB, best, 0)


@triton.jit
def _one_bucket_clear(table, g, slot, tries, live, reach, HLOG: tl.constexpr, WIDTH: tl.constexpr,
                      EMPTY: tl.constexpr):
    buckets: tl.constexpr = (1 << HLOG) // WIDTH
    counts = table + (1 << HLOG)
    bucket = _one_home(g, HLOG) // WIDTH
    tl.store(table + slot, tl.full(g.shape, EMPTY, tl.int64), mask=live)
    for step in range(0, reach):
        tl.store(counts + ((bucket + step) & (buckets - 1)), tl.zeros(g.shape, tl.int64), mask=live & (step < tries))


@triton.jit
def _one_search(keys, target, BLOCK: tl.constexpr, LOG: tl.constexpr):
    at = tl.zeros(target.shape, tl.int32)
    for bit in tl.static_range(LOG):
        step = BLOCK >> (bit + 1)
        probe = tl.load(keys + at + step - 1)
        at = tl.where(probe < target, at + step, at)
    return at


@triton.jit(do_not_specialize=["union", "union_stride", "layout", "parts", "request_ids", "states", "state_count",
                               "src_table", "src_stride", "src_reqs", "src_blocks", "src_bs", "res_table",
                               "res_stride", "res_reqs", "res_blocks", "res_bs", "res_null", "hot_table",
                               "hot_table_stride", "hot_bs", "attn_block_stride", "dgi", "lru", "region",
                               "host_total", "hot_total", "attn", "attn_stride", "hot_out", "host_out", "dev_out",
                               "counts_out", "idle", "stats", "tally", "guard", "pcie", "host", "host_stride", "hot",
                               "hot_block_stride", "hot_token_stride", "keys", "marks", "slots", "devs", "plan_host",
                               "plan_dev", "plan_count", "plan_stride", "flags", "ticket", "lanes", "piece",
                               "phases", "table", "fresh", "fresh_stride"])
def _one_kernel(union, union_stride, layout, parts, request_ids, states, state_count, src_table, src_stride,
                src_reqs, src_blocks, src_bs, res_table, res_stride, res_reqs, res_blocks, res_bs, res_null,
                hot_table, hot_table_stride, hot_bs, attn_block_stride, dgi, lru, region, host_total, hot_total, attn,
                attn_stride, hot_out, host_out, dev_out, counts_out, idle, stats, tally, guard, pcie, host,
                host_stride, hot, hot_block_stride, hot_token_stride, keys, marks, slots, devs, plan_host, plan_dev,
                plan_count, plan_stride, flags, ticket, lanes, piece, phases, table, fresh, fresh_stride,
                KB: tl.constexpr,
                HB: tl.constexpr,
                LOG: tl.constexpr, SHARE: tl.constexpr, TILE: tl.constexpr, WORDS: tl.constexpr, COLS: tl.constexpr,
                HAS_RES: tl.constexpr, HAS_IDLE: tl.constexpr, TALLY: tl.constexpr, GUARD: tl.constexpr,
                PCIE: tl.constexpr, CALLS: tl.constexpr, ROWS: tl.constexpr, TRANSFERS: tl.constexpr,
                READ: tl.constexpr, EMPTY: tl.constexpr, MARK: tl.constexpr, LANES: tl.constexpr,
                RT: tl.constexpr, SCAN: tl.constexpr = 0, SKIP: tl.constexpr = False, PHASES: tl.constexpr = False,
                PACK: tl.constexpr = False, PACK1: tl.constexpr = ONE_PACK_FLOOR, PACK2: tl.constexpr = ONE_PACK_FLOOR,
                PACK3: tl.constexpr = ONE_PACK_FLOOR, HASH: tl.constexpr = False, HLOG: tl.constexpr = 1,
                MISS1: tl.constexpr = ONE_PACK_FLOOR, MISS2: tl.constexpr = ONE_PACK_FLOOR,
                MISS3: tl.constexpr = ONE_PACK_FLOOR, COPY: tl.constexpr = True, BUCKET: tl.constexpr = 0,
                HSTRIDE: tl.constexpr = 2, FRESH: tl.constexpr = False):
    taken = tl.atomic_add(ticket, 1)
    if PHASES:
        if taken == 0:
            tl.atomic_xchg(phases + 13, _one_now())
    rest = tl.maximum(taken - lanes, 0)
    lane = tl.where(taken < lanes, taken, rest % lanes)
    role = tl.where(taken < lanes, 0, 1 + rest // lanes)
    wide = lane.to(tl.int64)
    if role == 0:
        if PHASES:
            began = _one_now()
            sort_ns = tl.sum(tl.zeros((1,), tl.int64), 0)
            scan_ns = sort_ns
            tail_ns = sort_ns
        i = tl.arange(0, KB)
        tiled = tl.arange(0, RT)
        req = tl.load(request_ids + lane)
        known = (req >= 0) & (req < state_count)
        state = tl.load(states + tl.minimum(tl.maximum(req, 0), tl.maximum(state_count - 1, 0)), mask=known,
                        other=-1)
        state = tl.where(known, state, -1).to(tl.int64)
        lane_keys = keys + wide * KB
        lane_marks = marks + wide * KB
        lane_devs = devs + wide * KB
        lane_slots = slots + wide * HB
        lane_table = table + wide * HSTRIDE
        filled = tl.sum(tl.zeros((KB,), tl.int32), 0)
        for p in range(0, parts):
            tl.debug_barrier()
            first = tl.load(layout + p).to(tl.int32)
            width = tl.load(layout + p + 1).to(tl.int32) - first
            base = tl.load(layout + parts + 1 + p) * lanes
            count_at = p * lanes
            inside = i < width
            listed = base + wide * width
            attn_row = attn + wide * attn_stride + first
            fresh_row = fresh + wide * fresh_stride + first
            if state < 0:
                tl.store(hot_out + listed + i, tl.full((KB,), -1, tl.int32), mask=inside)
                tl.store(attn_row + i, tl.full((KB,), -1, tl.int32), mask=inside)
                tl.store(counts_out + count_at + lane, 0)
                if FRESH:
                    tl.store(fresh_row + i, tl.zeros((KB,), tl.int8), mask=inside)
            else:
                pick = tl.load(union + wide * union_stride + first + i, mask=inside, other=-1).to(tl.int64)
                hollow = (tl.max(tl.where(inside, pick, -1), 0) < 0) & SKIP
                if PHASES:
                    tl.atomic_add(phases + 9, 1)
                if hollow:
                    tl.store(hot_out + listed + i, tl.full((KB,), -1, tl.int32), mask=inside)
                    tl.store(attn_row + i, tl.full((KB,), -1, tl.int32), mask=inside)
                    tl.store(counts_out + count_at + lane, 0)
                    if FRESH:
                        tl.store(fresh_row + i, tl.zeros((KB,), tl.int8), mask=inside)
                    if PHASES:
                        tl.atomic_add(phases + 8, 1)
                else:
                    token = tl.maximum(pick, 0)
                    block = token // src_bs
                    found = inside & (pick >= 0) & (req >= 0) & (req < src_reqs) & (block < src_blocks)
                    physical = tl.load(src_table + req.to(tl.int64) * src_stride + block, mask=found, other=0)
                    g = tl.where(found & (physical > 0), physical.to(tl.int64) * src_bs + token % src_bs, -1)
                    if HAS_RES:
                        rblock = token // res_bs
                        rfound = inside & (pick >= 0) & (req >= 0) & (req < res_reqs) & (rblock < res_blocks)
                        rphys = tl.load(res_table + req.to(tl.int64) * res_stride + rblock, mask=rfound, other=-1)
                        resident = tl.where(rfound & (rphys != res_null) & (rphys >= 0),
                                            rphys.to(tl.int64) * res_bs + token % res_bs, -1)
                    else:
                        resident = tl.full((KB,), -1, tl.int64)
                    g = tl.where(g >= host_total, -1, g)
                    done = inside & ((resident >= 0) | (g < 0))
                    pending = inside & ~done
                    settled = tl.sum(done.to(tl.int32), 0)
                    if SKIP and settled == width:
                        here = tl.where(resident >= 0, resident, -1).to(tl.int32)
                        tl.store(hot_out + listed + i, here, mask=inside)
                        tl.store(attn_row + i, tl.where(here < 0, -1,
                                 (here // hot_bs).to(tl.int64) * attn_block_stride + here % hot_bs).to(tl.int32),
                                 mask=inside)
                        tl.store(counts_out + count_at + lane, 0)
                        if FRESH:
                            tl.store(fresh_row + i, tl.zeros((KB,), tl.int8), mask=inside)
                        tl.atomic_add(stats, settled.to(tl.int64))
                        if PHASES:
                            tl.atomic_add(phases + 10, 1)
                    else:
                        if PHASES:
                            sorted_at = _one_now()
                        if HASH:
                            if BUCKET > 0:
                                placed, reach, tries = _one_bucket_insert(lane_table, g, (g << LOG) | i.to(tl.int64),
                                                                          pending, HLOG, BUCKET)
                            else:
                                placed, reach = _one_insert(lane_table, g, (g << LOG) | i.to(tl.int64), pending, LOG,
                                                            HLOG, EMPTY)
                            if PHASES:
                                tl.atomic_add(phases + 32, reach.to(tl.int64))
                                tl.atomic_add(phases + 33, 1)
                        elif PACK:
                            _one_packed(lane_keys, (g << LOG) | i.to(tl.int64), pending, width - settled,
                                        phases + 12, KB, EMPTY, PACK1, PACK2, PACK3, PHASES)
                        else:
                            ordered = tl.sort(tl.where(pending, (g << LOG) | i.to(tl.int64), EMPTY))
                            tl.store(lane_keys + i, ordered)
                        tl.store(lane_marks + i, tl.full((KB,), MARK, tl.int32))
                        tl.debug_barrier()
                        if PHASES:
                            scanned_at = _one_now()
                            sort_ns += scanned_at - sorted_at
                        hits = tl.sum(tl.zeros((RT,), tl.int32), 0)
                        olds = tl.sum(tl.zeros((RT,), tl.int32), 0)
                        for tile in range(0, SCAN, RT):
                            pos = tile + tiled
                            held = pos < region
                            slot = tl.load(lru + state * region + pos, mask=held, other=-1).to(tl.int32)
                            sane = held & (slot >= 0) & (slot < region)
                            cached = tl.load(dgi + state * region + slot, mask=sane, other=-1).to(tl.int64)
                            if HASH:
                                if BUCKET > 0:
                                    hit, column = _one_bucket_lookup(lane_table, cached, sane & (cached >= 0), reach,
                                                                     LOG, KB, HLOG, BUCKET, EMPTY)
                                else:
                                    hit, column = _one_lookup(lane_table, cached, sane & (cached >= 0), reach, LOG,
                                                              KB, HLOG, EMPTY)
                            else:
                                at = _one_search(lane_keys, cached << LOG, KB, LOG)
                                near = tl.load(lane_keys + at)
                                hit = sane & (cached >= 0) & ((near >> LOG) == cached)
                                column = (near & (KB - 1)).to(tl.int32)
                            hot_block = tl.load(hot_table + req.to(tl.int64) * hot_table_stride + slot // hot_bs, mask=hit,
                                                other=0)
                            tl.store(lane_marks + column, (hot_block.to(tl.int64) * hot_bs + slot % hot_bs).to(tl.int32),
                                     mask=hit)
                            evictable = held & ~hit
                            rank_hit = hits + tl.cumsum(hit.to(tl.int32), 0) - 1
                            rank_old = olds + tl.cumsum(evictable.to(tl.int32), 0) - 1
                            tl.store(lane_slots + tl.where(hit, rank_hit, region - 1 - rank_old), slot, mask=held)
                            hits += tl.sum(hit.to(tl.int32), 0)
                            olds += tl.sum(evictable.to(tl.int32), 0)
                        tl.debug_barrier()
                        if HASH:
                            if BUCKET > 0:
                                _one_bucket_clear(lane_table, g, placed, tries, pending, reach, HLOG, BUCKET, EMPTY)
                            else:
                                tl.store(lane_table + placed, tl.full((KB,), EMPTY, tl.int64), mask=pending)
                        if PHASES:
                            tailed_at = _one_now()
                            scan_ns += tailed_at - scanned_at
                        mark = tl.load(lane_marks + i)
                        hit_i = pending & (mark != MARK)
                        miss = pending & ~hit_i
                        m = tl.cumsum(miss.to(tl.int32), 0) - 1
                        evict = tl.load(lane_slots + region - 1 - m, mask=miss, other=-1)
                        good = miss & (evict >= 0) & (evict < region)
                        ehot = tl.load(hot_table + req.to(tl.int64) * hot_table_stride + evict // hot_bs, mask=good, other=0)
                        landed = (ehot.to(tl.int64) * hot_bs + evict % hot_bs).to(tl.int32)
                        hot_i = tl.where(hit_i, mark, tl.where(good, landed, tl.where(resident >= 0, resident, -1).to(tl.int32)))
                        attn_i = tl.where(hot_i < 0, -1, (hot_i // hot_bs).to(tl.int64) * attn_block_stride + hot_i % hot_bs)
                        tl.store(hot_out + listed + i, hot_i, mask=inside)
                        tl.store(attn_row + i, attn_i.to(tl.int32), mask=inside)
                        if FRESH:
                            tl.store(fresh_row + i, (miss & good).to(tl.int8), mask=inside)
                        dev = tl.where(good, landed, -1)
                        tl.store(host_out + listed + m, g.to(tl.int32), mask=miss)
                        tl.store(dev_out + listed + m, dev, mask=miss)
                        tl.store(dgi + state * region + evict, g.to(tl.int32), mask=good)
                        misses = width - hits - settled
                        tl.atomic_add(stats, hits.to(tl.int64))
                        tl.atomic_add(stats + 1, misses.to(tl.int64))
                        rest = region - hits - misses
                        for tile in range(0, SCAN, RT):
                            pos = tile + tiled
                            held = pos < region
                            source = tl.where(pos < rest, region - 1 - misses - pos,
                                              tl.where(pos < rest + misses, region - 1 - (pos - rest), pos - rest - misses))
                            order = tl.load(lane_slots + source, mask=held, other=0)
                            tl.store(lru + state * region + pos, order.to(tl.int16), mask=held)
                        count = misses
                        if HAS_IDLE:
                            count = tl.where(tl.load(idle + lane).to(tl.int32) != 0, 0, count)
                        tl.store(counts_out + count_at + lane, count)
                        if TALLY:
                            tl.atomic_add(tally, count.to(tl.int64))
                            tl.atomic_add(tally + 1, (count > 0).to(tl.int64))
                        take = good & (count > 0)
                        tl.debug_barrier()
                        if GUARD:
                            tl.store(lane_marks + i, tl.zeros((KB,), tl.int32))
                            tl.debug_barrier()
                            bad = miss & (m < count) & (g >= 0) & ((g >= host_total) | (dev < 0) | (dev >= hot_total))
                            tl.store(lane_marks + m // piece, tl.full((KB,), 1, tl.int32), mask=bad)
                            tl.debug_barrier()
                            flagged = tl.load(lane_marks + m // piece, mask=miss, other=0)
                            pieces = tl.load(lane_marks + i)
                            tl.atomic_add(guard, tl.sum(tl.where(i * piece < count, pieces, 0).to(tl.int64), 0))
                            take = take & (flagged == 0)
                        tl.store(lane_devs + i, tl.where(take, dev, -1))
                        tl.debug_barrier()
                        if HASH:
                            chosen = take & (dev >= 0)
                            listed_count = tl.sum(chosen.to(tl.int32), 0)
                            if listed_count > 0:
                                if PHASES:
                                    sorting_at = _one_now()
                                _one_packed(lane_keys, (g << LOG) | i.to(tl.int64), chosen, listed_count, phases + 12,
                                            KB, EMPTY, MISS1, MISS2, MISS3, False)
                                tl.debug_barrier()
                                if PHASES:
                                    sorting = _one_now() - sorting_at
                                    sort_ns += sorting
                                    tail_ns -= sorting
                                ordered = tl.load(lane_keys + i)
                                live = i < listed_count
                                dest = tl.load(lane_devs + (ordered & (KB - 1)).to(tl.int32), mask=live, other=-1)
                                tl.store(plan_host + wide * plan_stride + filled + i, (ordered >> LOG).to(tl.int32),
                                         mask=live)
                                tl.store(plan_dev + wide * plan_stride + filled + i, dest, mask=live)
                                filled += listed_count
                        else:
                            if PACK:
                                ordered = tl.load(lane_keys + i)
                            valid = ordered != EMPTY
                            dest = tl.load(lane_devs + (ordered & (KB - 1)).to(tl.int32), mask=valid, other=-1)
                            chosen = valid & (dest >= 0)
                            rank = filled + tl.cumsum(chosen.to(tl.int32), 0) - 1
                            tl.store(plan_host + wide * plan_stride + rank, (ordered >> LOG).to(tl.int32),
                                     mask=chosen)
                            tl.store(plan_dev + wide * plan_stride + rank, dest, mask=chosen)
                            filled += tl.sum(chosen.to(tl.int32), 0)
                        if PHASES:
                            tail_ns += _one_now() - tailed_at
        tl.store(plan_count + lane, filled)
        if PHASES:
            spent = _one_now() - began
            tl.atomic_add(phases, spent)
            tl.atomic_max(phases + 1, spent)
            tl.atomic_add(phases + 2, sort_ns)
            tl.atomic_add(phases + 3, scan_ns)
            tl.atomic_add(phases + 4, tail_ns)
            tl.atomic_add(phases + 11, 1)
            if lane == 0:
                tl.atomic_add(phases + 7, 1)
            tl.atomic_add(phases + 15, filled.to(tl.int64))
            tl.atomic_max(phases + 16, filled.to(tl.int64))
            tl.atomic_max(phases + 19, _one_now())
            slow = spent > 500000
            tl.atomic_add(phases + 20, slow.to(tl.int64))
            tl.atomic_add(phases + 21, tl.where(slow, filled, 0).to(tl.int64))
            bucket = (filled > 0).to(tl.int32) + (filled > 8).to(tl.int32) + (filled > 64).to(tl.int32) + (filled > 256).to(tl.int32)
            tl.atomic_add(phases + 22 + bucket, 1)
            tl.atomic_add(phases + 27 + bucket, spent)
        if PCIE:
            tl.atomic_add(pcie + ROWS, filled.to(tl.int64))
            if lane == 0:
                tl.atomic_add(pcie + CALLS, 1)
        tl.debug_barrier()
        tl.atomic_xchg(flags + lane, 1)
    else:
        if PHASES:
            waited = _one_now()
        ready = tl.atomic_add(flags + lane, 0)
        while ready == 0:
            ready = tl.atomic_add(flags + lane, 0)
        if PHASES:
            tl.atomic_add(phases + 6, _one_now() - waited)
    if COPY:
        if PHASES:
            copied = _one_now()
        total = tl.load(plan_count + lane, cache_modifier=".cg")
        chunks = (total + TILE - 1) // TILE
        slots_t = tl.arange(0, TILE)
        for chunk in range(role, chunks, SHARE):
            at = chunk * TILE + slots_t
            live = at < total
            row = tl.load(plan_host + wide * plan_stride + at, mask=live, other=0, cache_modifier=".cg").to(tl.int64)
            target = tl.load(plan_dev + wide * plan_stride + at, mask=live, other=0, cache_modifier=".cg").to(tl.int64)
            origin = row * host_stride
            landing = (target // hot_bs) * hot_block_stride + (target % hot_bs) * hot_token_stride
            for start in range(0, WORDS, COLS):
                col = start + tl.arange(0, COLS)
                keep = live[:, None] & (col < WORDS)[None, :]
                value = tl.load(host + origin[:, None] + col[None, :], mask=keep)
                tl.store(hot + landing[:, None] + col[None, :], value, mask=keep)
            if PCIE:
                before = tl.load(plan_host + wide * plan_stride + at - 1, mask=live & (slots_t > 0), other=-2,
                                 cache_modifier=".cg").to(tl.int64)
                begins = live & ((slots_t == 0) | (row != before + 1))
                tl.atomic_add(pcie + TRANSFERS, tl.sum(begins.to(tl.int64), 0))
                tl.atomic_add(pcie + READ, tl.sum(live.to(tl.int64), 0))
        if PHASES:
            tl.atomic_add(phases + 5, _one_now() - copied)
            tl.atomic_max(phases + 17, _one_now() - copied)
    finished = tl.atomic_add(ticket + 1, 1)
    if finished == lanes * SHARE - 1:
        if PHASES:
            tl.atomic_add(phases + 18, tl.atomic_xchg(phases + 19, 0) - tl.atomic_add(phases + 13, 0))
        if PHASES and COPY:
            tl.atomic_add(phases + 14, _one_now() - tl.atomic_xchg(phases + 13, 0))
        for start in range(0, lanes, LANES):
            lane_at = start + tl.arange(0, LANES)
            tl.store(flags + lane_at, tl.zeros((LANES,), tl.int32), mask=lane_at < lanes)
        tl.atomic_xchg(ticket, 0)
        tl.atomic_xchg(ticket + 1, 0)


def one_blocks(widths, region):
    if not widths or min(widths) <= 0 or max(widths) > int(region):
        return None
    kb, hb = pow2(max(widths)), pow2(region)
    if kb > ONE_MAX_BLOCK or hb > ONE_MAX_REGION:
        return None
    return kb, hb, kb.bit_length() - 1


def one_hash_shape():
    if "hash" not in _ONE_SHAPE:
        spread = _one_env(ONE_HASH_SPREAD_ENV, {4, 8, 16})
        bucket = _one_env(ONE_HASH_BUCKET_ENV, {0, 8, 16})
        _ONE_SHAPE["hash"] = (spread if spread is not None else ONE_HASH_SPREAD,
                              bucket if bucket is not None else ONE_HASH_BUCKET)
    return _ONE_SHAPE["hash"]


def one_hash_log(kb):
    return (int(kb) * one_hash_shape()[0]).bit_length() - 1


def one_hash_stride(kb):
    slots = 1 << one_hash_log(kb)
    bucket = one_hash_shape()[1]
    return slots + (slots // bucket if bucket else 0)


def one_hash_table(lanes, kb, device):
    slots = 1 << one_hash_log(kb)
    table = torch.full((int(lanes), one_hash_stride(kb)), ONE_EMPTY, dtype=torch.int64, device=device)
    table[:, slots:] = 0
    return table.view(-1)


def one_miss_rungs(kb):
    return tuple(min(int(kb), max(ONE_HASH_FLOOR, int(kb) >> shift)) for shift in (8, 3, 1))


def one_scratch(lanes, kb, hb, columns, device, hash=False):
    return {"lanes": int(lanes), "kb": int(kb), "hb": int(hb), "columns": int(columns), "device": device,
            "hash": one_hash_table(lanes, kb, device) if hash else None,
            "keys": torch.empty(lanes * kb, dtype=torch.int64, device=device),
            "marks": torch.empty(lanes * kb, dtype=torch.int32, device=device),
            "devs": torch.empty(lanes * kb, dtype=torch.int32, device=device),
            "slots": torch.empty(lanes * hb, dtype=torch.int32, device=device),
            "plan_host": torch.empty(lanes * columns, dtype=torch.int32, device=device),
            "plan_dev": torch.empty(lanes * columns, dtype=torch.int32, device=device),
            "plan_count": torch.zeros(lanes, dtype=torch.int32, device=device),
            "attn": torch.empty(lanes * columns, dtype=torch.int32, device=device),
            "flags": torch.zeros(lanes, dtype=torch.int32, device=device),
            "flat": torch.zeros(1, dtype=torch.int32, device=device),
            "ticket": torch.zeros(2, dtype=torch.int32, device=device)}


def one_scratch_fits(scratch, lanes, kb, hb, columns, device, hash=False):
    return (scratch is not None and scratch["device"] == device and scratch["lanes"] >= lanes
            and scratch["kb"] >= kb and scratch["hb"] >= hb and scratch["columns"] >= columns
            and (not hash or (scratch.get("hash") is not None
                              and scratch["hash"].numel() == int(scratch["lanes"]) * one_hash_stride(scratch["kb"]))))


def one_rungs(kb):
    return tuple(min(int(kb), max(ONE_PACK_FLOOR, int(kb) >> shift)) for shift in ONE_PACK_SHIFTS)


def one_probe(host_cache, hot_cache, scratch, states_dtype, has_res, has_idle, tally, guard, pcie, guarded,
              share=ONE_SHARE, skip=False, phases=None, pack=False, hash=False, copy_flat=False):
    device = hot_cache.device
    region = scratch["hb"]
    union = torch.full((1, 1), -1, dtype=torch.int32, device=device)
    layout = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    flat = torch.full((4,), -1, dtype=torch.int32, device=device)
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    states = torch.full((1,), -1, dtype=states_dtype, device=device)
    stats = torch.zeros(len(PCIE_FIELDS), dtype=torch.int64, device=device)
    return one_launch(union, layout, 1, torch.zeros(1, dtype=torch.int32, device=device), states, table, 1,
                      table if has_res else None, 1, 0, table, 1,
                      torch.full((1, region), -1, dtype=torch.int32, device=device),
                      torch.zeros((1, region), dtype=torch.int16, device=device), host_cache, hot_cache,
                      torch.full((1, 1), -1, dtype=torch.int32, device=device), flat, flat, flat, counts,
                      torch.zeros(1, dtype=torch.bool, device=device) if has_idle else None, stats[:2],
                      stats[2:4] if tally else None, stats[4:5] if guard else None, stats if pcie else None, scratch,
                      1, guarded, share=share, skip=skip, phases=phases, pack=pack, hash=hash, flat=copy_flat)


def _one_env(name, allowed):
    said = str(os.environ.get(name) or "").strip()
    if not said:
        return None
    if not said.isdigit() or int(said) not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, found {said!r}")
    return int(said)


def one_scan_tile(skip):
    key = ("tile", bool(skip))
    if key not in _ONE_SHAPE:
        said = _one_env(ONE_TILE_ENV, {512, 1024, 2048, 4096, 8192, 16384})
        _ONE_SHAPE[key] = said if said is not None else (ONE_SKIP_TILE if skip else ONE_REGION_TILE)
    return _ONE_SHAPE[key]


def one_warps():
    if "warps" not in _ONE_SHAPE:
        said = _one_env(ONE_WARPS_ENV, {4, 8, 16, 32})
        _ONE_SHAPE["warps"] = said if said is not None else ONE_WARPS
    return _ONE_SHAPE["warps"]


def one_launch(union, layout, parts, request_ids, states, src_table, src_bs, res_table, res_bs, res_null, hot_table,
               attn_block_stride, dgi, lru, host_cache, hot_cache, attn, hot_out, host_out, dev_out, counts_out, idle,
               stats, tally, guard, pcie, scratch, piece, guarded, share=ONE_SHARE, tile=ONE_TILE, skip=False,
               phases=None, pack=False, hash=False, flat=False, fresh=None, copy=True, cuda=False, viol=None):
    lanes = union.shape[0]
    if lanes == 0:
        return 0
    if cuda and flat and one_cuda_run(
            dict(union=union, layout=layout, parts=parts, request_ids=request_ids, states=states,
                 src_table=src_table, src_bs=src_bs, res_table=res_table, res_bs=res_bs, res_null=res_null,
                 hot_table=hot_table, attn_block_stride=attn_block_stride, dgi=dgi, lru=lru,
                 host_cache=host_cache, hot_cache=hot_cache, attn=attn, hot_out=hot_out, host_out=host_out,
                 dev_out=dev_out, counts_out=counts_out, idle=idle, stats=stats, tally=tally, guard=guard,
                 pcie=pcie, scratch=scratch, piece=piece, guarded=guarded, skip=skip, fresh=fresh, viol=viol),
            pack, hash, phases):
        if copy:
            read, write = group_word(host_cache, hot_cache)
            flat_copy(scratch, lanes, read, write, int(hot_cache.shape[1]), pcie, phases)
        return lanes
    if hash and (scratch.get("hash") is None
                 or scratch["hash"].numel() != int(scratch["lanes"]) * one_hash_stride(scratch["kb"])):
        scratch["hash"] = one_hash_table(scratch["lanes"], scratch["kb"], hot_cache.device)
    if flat:
        share = 1
    read, write = group_word(host_cache, hot_cache)
    region = int(lru.shape[1])
    kb, hb, log = scratch["kb"], scratch["hb"], scratch["kb"].bit_length() - 1
    words = read.shape[1]
    columns = scratch["columns"]
    has_res = res_table is not None
    region_tile = min(hb, one_scan_tile(skip))
    rungs = one_rungs(kb)
    misses = one_miss_rungs(kb)
    scan = -(-region // region_tile) * region_tile if skip else hb
    _one_kernel[(lanes * share,)](
        union, union.stride(0), layout, parts, request_ids, states, states.numel(), src_table, src_table.stride(0),
        src_table.shape[0], src_table.shape[1], int(src_bs), res_table if has_res else src_table,
        res_table.stride(0) if has_res else 0, res_table.shape[0] if has_res else 0,
        res_table.shape[1] if has_res else 0, max(int(res_bs), 1), int(res_null), hot_table, hot_table.stride(0),
        int(hot_cache.shape[1]), int(attn_block_stride), dgi, lru, region, host_cache.shape[0],
        hot_cache.shape[0] * hot_cache.shape[1], attn, attn.stride(0), hot_out, host_out, dev_out, counts_out,
        idle if idle is not None else counts_out, stats, tally if tally is not None else stats,
        guard if guard is not None else stats, pcie if pcie is not None else stats, read, read.stride(0), write,
        write.stride(0), write.stride(1), scratch["keys"], scratch["marks"], scratch["slots"], scratch["devs"],
        scratch["plan_host"], scratch["plan_dev"], scratch["plan_count"], columns, scratch["flags"],
        scratch["ticket"], lanes, max(int(piece), 1), phases if phases is not None else stats,
        scratch["hash"] if hash else stats, fresh if fresh is not None else stats,
        fresh.stride(0) if fresh is not None else 0, KB=kb, HB=hb,
        LOG=log, SHARE=int(share), TILE=int(tile),
        WORDS=words, COLS=min(pow2(words), RUN_COLS), HAS_RES=has_res, HAS_IDLE=idle is not None,
        TALLY=tally is not None, GUARD=bool(guarded) and guard is not None, PCIE=pcie is not None,
        CALLS=ONE_PCIE["calls"], ROWS=ONE_PCIE["rows"], TRANSFERS=ONE_PCIE["transfers"],
        READ=ONE_PCIE["rows_read"], EMPTY=ONE_EMPTY, MARK=ONE_MARK, LANES=LANE_BLOCK, RT=region_tile, SCAN=scan,
        SKIP=bool(skip), PHASES=phases is not None, PACK=bool(pack) and not hash, PACK1=rungs[0], PACK2=rungs[1],
        PACK3=rungs[2], HASH=bool(hash), HLOG=one_hash_log(kb), MISS1=misses[0], MISS2=misses[1], MISS3=misses[2],
        COPY=not flat, BUCKET=one_hash_shape()[1] if hash else 0, HSTRIDE=one_hash_stride(kb) if hash else 2,
        FRESH=fresh is not None, num_warps=one_warps())
    if flat and copy:
        flat_copy(scratch, lanes, read, write, int(hot_cache.shape[1]), pcie, phases)
    return lanes * share


def one_copy(scratch, lanes, host_cache, hot_cache, pcie=None, phases=None, programs=None):
    read, write = group_word(host_cache, hot_cache)
    return flat_copy(scratch, lanes, read, write, int(hot_cache.shape[1]), pcie, phases, programs=programs)


def flat_programs(device):
    if device.type != "cuda":
        return FLAT_HOST_PROGRAMS
    index = torch.cuda.current_device() if device.index is None else device.index
    held = _FLAT["programs"].get(index)
    if held is None:
        said = str(os.environ.get(FLAT_PROGRAMS_ENV) or "").strip()
        held = int(said) if said else torch.cuda.get_device_properties(index).multi_processor_count * FLAT_PER_SM
        if held < 1:
            raise ValueError(f"{FLAT_PROGRAMS_ENV}={said!r} must be a positive program count")
        _FLAT["programs"][index] = held
        print(f"HOST_KV flat copy: {held} programs"
              + (f" ({FLAT_PROGRAMS_ENV})" if said else f" ({FLAT_PER_SM} a multiprocessor)"), flush=True)
    return held


def flat_lines_on():
    return str(os.environ.get(FLAT_LINES_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def flat_line_words(read):
    unit = read.element_size()
    words = read.shape[1]
    line = LINE_BYTES // unit
    if (not flat_lines_on() or LINE_BYTES % unit or read.stride(0) != words or read.data_ptr() % LINE_BYTES
            or (LINE_GROUP * words * unit) % LINE_BYTES):
        return 0
    return line


def flat_copy(scratch, lanes, read, write, hot_bs, pcie, phases, programs=None):
    programs = flat_programs(write.device) if programs is None else int(programs)
    words = read.shape[1]
    stats = scratch["flat"]
    line = flat_line_words(read)
    span = words + line if line else words
    _flat_copy_kernel[(programs,)](
        scratch["plan_host"], scratch["plan_dev"], scratch["plan_count"], scratch["columns"], lanes, read,
        read.stride(0), write, hot_bs, write.stride(0), write.stride(1), pcie if pcie is not None else stats,
        phases if phases is not None else stats, scratch["flat"], programs, LB=pow2(lanes), TILE=FLAT_TILE,
        WORDS=words, COLS=min(pow2(span), RUN_COLS), VEC=run_vector(read, write), PCIE=pcie is not None,
        TRANSFERS=ONE_PCIE["transfers"], READ=ONE_PCIE["rows_read"], PHASES=phases is not None, PDL=pdl_on(),
        LINES=line > 0, LINE=max(line, 1), GROUP=LINE_GROUP, SPAN=span,
        num_warps=FLAT_LINE_WARPS if line else FLAT_WARPS, launch_pdl=pdl_on())
    return programs


ONE_CUDA_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA"
ONE_CUDA_FILES = ("taaf_host_kv_resolve_cuda.py", "host_kv_resolve_cuda.py")
ONE_CUDA_LANES = 6
ONE_CUDA_STEPS = 4
ONE_CUDA_TENSORS = ("union", "layout", "request_ids", "states", "src_table", "res_table", "hot_table", "dgi", "lru",
                    "attn", "hot_out", "host_out", "dev_out", "counts_out", "idle", "stats", "tally", "guard",
                    "pcie", "fresh")
SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70
_ONE_CUDA = {"module": None, "aside": "", "verified": set(), "said": set(), "fired": set(), "launches": 0}


def one_cuda_wanted():
    said = str(os.environ.get(ONE_CUDA_ENV) or "").strip().lower()
    return said in ("1", "on", "true") and not _ONE_CUDA["aside"]


def one_cuda_reset():
    _ONE_CUDA["aside"] = ""
    _ONE_CUDA["verified"].clear()
    _ONE_CUDA["said"].clear()
    _ONE_CUDA["fired"].clear()


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _one_cuda_fail(detail):
    _ONE_CUDA["aside"] = detail
    print(f"SELFCHECK FAIL {ONE_CUDA_ENV}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)
    print(f"HOST_KV resolve cuda stands aside: {detail}; the Triton resolve runs", file=sys.stderr, flush=True)


def _one_cuda_decline(key, text):
    if key not in _ONE_CUDA["said"]:
        _ONE_CUDA["said"].add(key)
        print(f"HOST_KV resolve cuda stands aside for this call: {text}", file=sys.stderr, flush=True)
    return False


def _one_cuda_module():
    held = _ONE_CUDA["module"]
    if held is not None:
        return held
    here = Path(__file__).resolve().parent
    found = next((here / name for name in ONE_CUDA_FILES if (here / name).is_file()), None)
    if found is None:
        _one_cuda_fail(f"no {ONE_CUDA_FILES[1]} beside {Path(__file__).name}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("taaf_host_kv_resolve_cuda", found)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as failure:
        _one_cuda_fail(f"{found.name} did not load: {type(failure).__name__}: {str(failure)[:300]}")
        return None
    _ONE_CUDA["module"] = module
    return module


def one_cuda_run(held, pack, hash, phases):
    if _ONE_CUDA["aside"]:
        return False
    if phases is not None:
        return _one_cuda_decline("phases", "the in-kernel phase counters are on and the CUDA resolve stamps none")
    module = _one_cuda_module()
    if module is None:
        return False
    lanes = int(held["union"].shape[0])
    why = module.unsupported(held["scratch"], lanes, **{name: held[name] for name in ONE_CUDA_TENSORS})
    if why:
        return _one_cuda_decline(f"layout {why}", why)
    try:
        config = module.launch_config(held["union"], held["request_ids"], held["states"], held["src_table"],
                                      held["res_table"], held["hot_table"], held["src_bs"], held["res_bs"],
                                      held["lru"], held["hot_cache"], held["idle"], held["tally"], held["guard"],
                                      held["guarded"], held["pcie"], held["fresh"], held["scratch"], held["skip"])
    except ValueError as failure:
        return _one_cuda_decline(f"fit {failure}", str(failure))
    key = module.build_key(config, held["union"].device)
    if key not in _ONE_CUDA["verified"]:
        if _capturing():
            return _one_cuda_decline("cold", "a captured forward reached a resolve shape no eager forward had "
                                             "verified, so this graph keeps the Triton resolve")
        if not _one_cuda_verify(module, held, config, key, pack, hash):
            return False
    module.resolve_launch(
        held["union"], held["layout"], held["parts"], held["request_ids"], held["states"], held["src_table"],
        held["src_bs"], held["res_table"], held["res_bs"], held["res_null"], held["hot_table"],
        held["attn_block_stride"], held["dgi"], held["lru"], held["host_cache"], held["hot_cache"], held["attn"],
        held["hot_out"], held["host_out"], held["dev_out"], held["counts_out"], held["idle"], held["stats"],
        held["tally"], held["guard"], held["pcie"], held["scratch"], held["piece"], held["guarded"],
        skip=held["skip"], fresh=held["fresh"], pcie_calls=ONE_PCIE["calls"], pcie_rows=ONE_PCIE["rows"],
        viol=held["viol"])
    _ONE_CUDA["launches"] += 1
    kind = "captured" if _capturing() else "eager"
    if kind not in _ONE_CUDA["fired"]:
        _ONE_CUDA["fired"].add(kind)
        print(f"HOST_KV resolve cuda: first {kind} launch lanes={lanes} region={int(held['lru'].shape[1])} "
              f"threads={config['THREADS']} parts={int(held['parts'])} kb={config['KB']} hb={config['HB']} "
              f"build={module.build_path()} ({ONE_CUDA_ENV})", flush=True)
    return True


def _one_cuda_verify(module, held, config, key, pack, hash):
    try:
        why, seen = _one_cuda_compare(module, held, pack, hash)
    except Exception as failure:
        why, seen = f"{type(failure).__name__}: {str(failure)[:300]}", ""
    if why:
        _one_cuda_fail(why)
        return False
    _ONE_CUDA["verified"].add(key)
    print(f"HOST_KV resolve cuda: verified against the Triton resolve on a cloned world, {ONE_CUDA_STEPS} steps of "
          f"{ONE_CUDA_LANES} lanes ({seen}), outputs, copy plan, hot state and tallies bit-identical, built with "
          f"{module.build_path()} ({ONE_CUDA_ENV})", flush=True)
    return True


def _one_cuda_series(lows, widths, positions, generator):
    parts, reach, total = len(widths), lows[-1], sum(widths)
    picks = [torch.randperm(positions, generator=generator)[:max(total * 3 // 4, 1)] for _ in range(ONE_CUDA_LANES)]
    picks[-1] = picks[-1][:0]
    series = []
    for step in range(ONE_CUDA_STEPS):
        rows = torch.full((ONE_CUDA_LANES, reach), -1, dtype=torch.int64)
        for lane in range(ONE_CUDA_LANES):
            if step and picks[lane].numel():
                keep = torch.rand(picks[lane].numel(), generator=generator) < 0.85
                fresh = torch.randperm(positions, generator=generator)[:int((~keep).sum())]
                picks[lane] = torch.unique(torch.cat((picks[lane][keep], fresh)))[:total]
            ordered = picks[lane].sort().values
            at = 0
            for part in range(parts):
                chunk = ordered[at:at + widths[part]]
                rows[lane, lows[part]:lows[part] + chunk.numel()] = chunk
                at += widths[part]
            if lane % 3 == 1:
                rows[lane, torch.rand(reach, generator=generator) < 0.1] = -1
        series.append(rows)
    return series


def _one_cuda_world(held, lows, widths):
    device = held["union"].device
    generator = torch.Generator().manual_seed(5)
    lanes, region = ONE_CUDA_LANES, int(held["lru"].shape[1])
    src_bs, res_bs, hot_bs = int(held["src_bs"]), max(int(held["res_bs"]), 1), int(held["hot_cache"].shape[1])
    blocks = max(int(held["host_cache"].shape[0]) // src_bs - 1, 1)
    cols = max(min(blocks, 4 * -(-region // src_bs)), 1)
    positions = cols * src_bs
    per = -(-region // hot_bs)
    hot_blocks = max(int(held["hot_cache"].shape[0]), 1)
    states = torch.arange(lanes + 1, dtype=torch.int64)
    requests = torch.arange(lanes, dtype=torch.int64)
    requests[2] = -1
    if held["states"].dtype != torch.uint8:
        states[3] = -1
    world = {
        "request_ids": requests.to(held["request_ids"].dtype), "states": states.to(held["states"].dtype),
        "src_table": torch.stack([torch.randperm(blocks, generator=generator)[:cols] + 1
                                  for _ in range(lanes)]).to(held["src_table"].dtype),
        "hot_table": (torch.randperm(max(hot_blocks, lanes * per), generator=generator)[:lanes * per]
                      % hot_blocks).view(lanes, per).to(held["hot_table"].dtype),
        "dgi": torch.full((lanes + 1, region), -1, dtype=torch.int32),
        "lru": torch.arange(region, dtype=torch.int16).repeat(lanes + 1, 1).contiguous(),
        "stats": torch.zeros_like(held["stats"], device="cpu")}
    if held["res_table"] is not None:
        wide = -(-positions // res_bs)
        resident = torch.rand((lanes, wide), generator=generator) < 0.3
        world["res_table"] = torch.where(resident, torch.randint(1, 1000, (lanes, wide), generator=generator),
                                         torch.full((lanes, wide), int(held["res_null"]))).to(held["res_table"].dtype)
    if held["idle"] is not None:
        world["idle"] = torch.zeros(lanes, dtype=held["idle"].dtype)
        world["idle"][1] = 1
    for name in ("tally", "guard", "pcie"):
        if held[name] is not None:
            world[name] = torch.zeros_like(held[name], device="cpu")
    if held["fresh"] is not None:
        world["fresh"] = torch.zeros((lanes, lows[-1]), dtype=torch.int8)
    world = {name: tensor.to(device) for name, tensor in world.items()}
    return world, _one_cuda_series(lows, widths, positions, generator)


def _one_cuda_arm(held, world, lows, widths):
    device, lanes = held["union"].device, ONE_CUDA_LANES
    side = {name: tensor.clone() for name, tensor in world.items()}
    mark = dict(dtype=torch.int32, device=device)
    side["attn"] = torch.full((lanes, lows[-1]), -7, **mark)
    for name in ("hot_out", "host_out", "dev_out"):
        side[name] = torch.full((lanes * sum(widths),), -7, **mark)
    side["counts_out"] = torch.full((lanes * len(widths),), -7, **mark)
    scratch = held["scratch"]
    side["scratch"] = one_scratch(lanes, scratch["kb"], scratch["hb"], scratch["columns"], device,
                                  hash=scratch.get("hash") is not None)
    return side


def _one_cuda_view(side):
    scratch = side["scratch"]
    columns, lanes = scratch["columns"], ONE_CUDA_LANES
    count = scratch["plan_count"][:lanes]
    live = torch.arange(columns, device=count.device)[None, :] < count[:, None]
    view = {"plan_count": count}
    for name in ("plan_host", "plan_dev"):
        view[name] = torch.where(live, scratch[name][:lanes * columns].view(lanes, columns), 0)
    for name in ("attn", "counts_out", "hot_out", "host_out", "dev_out", "dgi", "lru", "stats", "tally", "guard",
                 "pcie", "fresh"):
        if name in side:
            view[name] = side[name]
    return view


def _one_cuda_compare(module, held, pack, hash):
    layout = [int(one) for one in held["layout"].tolist()]
    parts = int(held["parts"])
    lows = layout[:parts + 1]
    widths = [lows[at + 1] - lows[at] for at in range(parts)]
    world, series = _one_cuda_world(held, lows, widths)
    reference, candidate = _one_cuda_arm(held, world, lows, widths), _one_cuda_arm(held, world, lows, widths)
    for step, rows in enumerate(series):
        union = rows.to(held["union"].dtype).to(held["union"].device)
        for side, cuda in ((reference, False), (candidate, True)):
            call = (union, held["layout"], parts, side["request_ids"], side["states"], side["src_table"],
                    held["src_bs"], side.get("res_table"), held["res_bs"], held["res_null"], side["hot_table"],
                    held["attn_block_stride"], side["dgi"], side["lru"], held["host_cache"], held["hot_cache"],
                    side["attn"], side["hot_out"], side["host_out"], side["dev_out"], side["counts_out"],
                    side.get("idle"), side["stats"], side.get("tally"), side.get("guard"), side.get("pcie"),
                    side["scratch"], held["piece"], held["guarded"])
            if cuda:
                module.resolve_launch(*call, skip=held["skip"], fresh=side.get("fresh"),
                                      pcie_calls=ONE_PCIE["calls"], pcie_rows=ONE_PCIE["rows"])
            else:
                one_launch(*call, skip=held["skip"], pack=pack, hash=hash, flat=True, fresh=side.get("fresh"),
                           copy=False, cuda=False)
        torch.cuda.synchronize()
        want, got = _one_cuda_view(reference), _one_cuda_view(candidate)
        for name, tensor in want.items():
            if not torch.equal(tensor, got[name]):
                return f"the CUDA resolve changed {name} at step {step} against the Triton resolve", ""
    hits, misses = (int(one) for one in reference["stats"].view(torch.int64)[:2].tolist())
    seen = f"hits={hits} misses={misses} planned={int(reference['scratch']['plan_count'][:ONE_CUDA_LANES].sum())}"
    if hits == 0 or misses == 0:
        return f"the cloned world saw hits={hits} misses={misses}, so it proves nothing", ""
    return "", seen


_FLAT = {"programs": {}}


@triton.jit(do_not_specialize=["plan_stride", "lanes", "host_stride", "hot_bs", "hot_block_stride",
                               "hot_token_stride", "programs"])
def _flat_copy_kernel(plan_host, plan_dev, plan_count, plan_stride, lanes, host, host_stride, hot, hot_bs,
                      hot_block_stride, hot_token_stride, pcie, phases, ticket, programs, LB: tl.constexpr,
                      TILE: tl.constexpr, WORDS: tl.constexpr, COLS: tl.constexpr, VEC: tl.constexpr,
                      PCIE: tl.constexpr, TRANSFERS: tl.constexpr, READ: tl.constexpr, PHASES: tl.constexpr,
                      PDL: tl.constexpr, LINES: tl.constexpr = False, LINE: tl.constexpr = 1,
                      GROUP: tl.constexpr = 4, SPAN: tl.constexpr = 0):
    if PDL:
        tl.extra.cuda.gdc_wait()
    if PHASES:
        began = _one_now()
    me = tl.program_id(0)
    j = tl.arange(0, LB)
    counts = tl.load(plan_count + j, mask=j < lanes, other=0, cache_modifier=".cg")
    ends = tl.cumsum(counts, 0)
    total = tl.sum(counts, 0)
    per = (total + programs - 1) // programs
    per = (per + TILE - 1) // TILE * TILE
    low = me * per
    high = tl.minimum(low + per, total)
    slots = tl.arange(0, TILE)
    for start in range(low, high, TILE):
        at = start + slots
        live = at < high
        after = ends[None, :] <= at[:, None]
        lane = tl.minimum(tl.sum(after.to(tl.int32), 1), lanes - 1).to(tl.int64)
        offset = at - tl.sum(tl.where(after, counts[None, :], 0), 1)
        base = lane * plan_stride
        row = tl.load(plan_host + base + offset, mask=live, other=0, cache_modifier=".cg").to(tl.int64)
        target = tl.load(plan_dev + base + offset, mask=live, other=0, cache_modifier=".cg").to(tl.int64)
        landing = tl.multiple_of((target // hot_bs) * hot_block_stride + (target % hot_bs) * hot_token_stride, VEC)
        if LINES:
            mine = tl.sum(tl.where(j[None, :] == lane[:, None], counts[None, :], 0), 1)
            lead = row % GROUP
            head = row - lead
            first = offset - lead
            whole = live & (first >= 0) & (first + GROUP <= mine)
            for k in tl.static_range(GROUP):
                seen = tl.load(plan_host + base + first + k, mask=whole, other=-1, cache_modifier=".cg").to(tl.int64)
                whole = whole & (seen == head + k)
            rel = tl.where(whole, lead, 0)
            lo = tl.multiple_of(tl.where(whole, (rel * WORDS + LINE - 1) // LINE * LINE, 0), VEC)
            hi = tl.where(whole, ((rel + 1) * WORDS + LINE - 1) // LINE * LINE, WORDS)
            source = tl.multiple_of(tl.where(whole, head, row) * host_stride, VEC)
            spill = tl.load(plan_dev + base + offset + 1, mask=whole & (lead < GROUP - 1), other=0,
                            cache_modifier=".cg").to(tl.int64)
            spill_landing = (spill // hot_bs) * hot_block_stride + (spill % hot_bs) * hot_token_stride
            for word in range(0, SPAN, COLS):
                col = tl.max_contiguous(tl.multiple_of(word + tl.arange(0, COLS), VEC), VEC)
                at_word = lo[:, None] + col[None, :]
                keep = live[:, None] & (at_word < hi[:, None])
                value = tl.load(host + source[:, None] + at_word, mask=keep)
                own = (at_word // WORDS) == rel[:, None]
                spot = tl.where(own, landing[:, None] + at_word - rel[:, None] * WORDS,
                                spill_landing[:, None] + at_word - (rel[:, None] + 1) * WORDS)
                tl.store(hot + spot, value, mask=keep)
        else:
            origin = tl.multiple_of(row * host_stride, VEC)
            for word in range(0, WORDS, COLS):
                col = tl.max_contiguous(tl.multiple_of(word + tl.arange(0, COLS), VEC), VEC)
                keep = live[:, None] & (col < WORDS)[None, :]
                value = tl.load(host + origin[:, None] + col[None, :], mask=keep)
                tl.store(hot + landing[:, None] + col[None, :], value, mask=keep)
        if PCIE:
            before = tl.load(plan_host + base + offset - 1, mask=live & (offset > 0), other=-2,
                             cache_modifier=".cg").to(tl.int64)
            begins = live & ((offset == 0) | (row != before + 1))
            tl.atomic_add(pcie + TRANSFERS, tl.sum(begins.to(tl.int64), 0))
            tl.atomic_add(pcie + READ, tl.sum(live.to(tl.int64), 0))
    if PHASES:
        tl.atomic_add(phases + 5, _one_now() - began)
        tl.atomic_max(phases + 17, _one_now() - began)
        finished = tl.atomic_add(ticket, 1)
        if finished == programs - 1:
            tl.atomic_add(phases + 14, _one_now() - tl.atomic_xchg(phases + 13, 0))
            tl.atomic_xchg(ticket, 0)


PREV_WARPS = 16


@triton.jit(do_not_specialize=["held", "held_stride", "held_width", "fits", "table", "table_stride", "table_cols",
                               "request_ids", "states", "resident", "resident_stride", "resident_width", "region",
                               "block_size", "rows_cap", "width",
                               "listed_width", "captured", "captured_stride", "cap_counts", "cap_count_stride",
                               "cap_layout", "caps", "union",
                               "union_stride", "listed", "listed_stride", "idle", "tally", "owned", "copied"])
def _prev_plan_kernel(held, held_stride, held_width, fits, table, table_stride, table_cols, request_ids, states,
                      resident, resident_stride, resident_width, region, block_size, rows_cap, width, listed_width, captured, captured_stride,
                      cap_counts, cap_count_stride, cap_layout, caps, union, union_stride, listed, listed_stride, idle, tally, owned, copied,
                      HB: tl.constexpr, RB: tl.constexpr, RLOG: tl.constexpr, CB: tl.constexpr, CLOG: tl.constexpr,
                      SCOPED: tl.constexpr, BIG: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, HB)
    inside = c < held_width
    pos = tl.load(held + lane * held_stride + c, mask=inside, other=-1)
    req = tl.load(request_ids + lane).to(tl.int64)
    clean = tl.maximum(pos, 0).to(tl.int64)
    page = clean // block_size
    block = tl.load(table + req * table_stride + tl.minimum(page, table_cols - 1), mask=inside, other=-1).to(tl.int64)
    present = inside & (pos >= 0) & (page < table_cols) & (block >= 0)
    rows_of = block * block_size + clean % block_size
    state = tl.load(states + req).to(tl.int64)
    fit = tl.load(fits + lane).to(tl.int1)
    valid = present & fit & (state >= 0)
    r = tl.arange(0, RB)
    held_r = r < resident_width
    mine = tl.load(resident + tl.maximum(state, 0) * resident_stride + r, mask=held_r, other=-1)
    sorted_owned = tl.sort(tl.where(held_r & (mine >= 0), mine, BIG))
    lane_owned = owned + lane * RB
    tl.store(lane_owned + r, sorted_owned)
    tl.debug_barrier()
    probe = tl.where(valid, rows_of, 0)
    at = _one_search(lane_owned, probe, RB, RLOG)
    hit = valid & (tl.load(lane_owned + at).to(tl.int64) == rows_of)
    miss = valid & ~hit
    if SCOPED:
        q = tl.arange(0, CB)
        source = q < 0
        for k in range(0, caps):
            low = tl.load(cap_layout + k)
            high = tl.load(cap_layout + k + 1)
            n = tl.load(cap_counts + k * cap_count_stride + lane)
            source = source | ((q >= low) & (q < high) & (q - low < n))
        value = tl.load(captured + lane * captured_stride + q, mask=source, other=-1)
        sorted_copied = tl.sort(tl.where(source & (value >= 0), value, BIG))
        lane_copied = copied + lane * CB
        tl.store(lane_copied + q, sorted_copied)
        tl.debug_barrier()
        found = _one_search(lane_copied, probe, CB, CLOG)
        scoped = miss & (tl.load(lane_copied + found).to(tl.int64) == rows_of)
    else:
        scoped = miss
    rank = tl.cumsum(scoped.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    room = tl.minimum(tl.maximum(region - count, 0), rows_cap)
    keep = scoped & (rank < room)
    kept = tl.sum(keep.to(tl.int32), 0)
    place = tl.where(keep, rank, kept + tl.cumsum(hit.to(tl.int32), 0) - 1)
    put = (keep | hit) & (place < width)
    tl.store(union + lane * union_stride + c, tl.full((HB,), -1, tl.int32), mask=c < width)
    tl.store(listed + lane * listed_stride + c, tl.full((HB,), -1, tl.int32), mask=c < listed_width)
    tl.debug_barrier()
    tl.store(union + lane * union_stride + place, pos, mask=put)
    tl.store(listed + lane * listed_stride + rank, pos, mask=keep & (rank < listed_width))
    tl.store(idle + lane, kept <= 0)
    tl.atomic_add(tally, (kept > 0).to(tl.int64))
    tl.atomic_add(tally + 1, count.to(tl.int64))
    tl.atomic_add(tally + 2, tl.sum(hit.to(tl.int64), 0))
    tl.atomic_add(tally + 3, tl.sum(miss.to(tl.int64), 0))
    tl.atomic_add(tally + 4, tl.sum(scoped.to(tl.int64), 0))
    tl.atomic_add(tally + 5, kept.to(tl.int64))
    tl.atomic_add(tally + 8, (lane == 0).to(tl.int64))


def prev_plan(held, fits, table, request_ids, states, resident, block_size, region, rows_cap, width, listed_width,
              union, listed, idle, tally, owned, copied, captured=None):
    lanes, held_width = held.shape
    if lanes == 0:
        return 0
    hb, rb = pow2(held_width), pow2(resident.shape[1])
    if captured is None:
        cap, cap_counts, cap_layout, caps, cb = held, held, held, 0, 1
    else:
        cap, cap_counts, cap_layout, caps = captured
        cb = pow2(cap.shape[1])
    _prev_plan_kernel[(lanes,)](
        held, held.stride(0), held_width, fits, table, table.stride(0), table.shape[1], request_ids, states,
        resident, resident.stride(0), resident.shape[1], int(region), int(block_size), int(rows_cap), int(width), int(listed_width), cap, cap.stride(0),
        cap_counts, cap_counts.stride(0), cap_layout, int(caps), union, union.stride(0), listed, listed.stride(0),
        idle, tally, owned, copied, HB=hb, RB=rb, RLOG=rb.bit_length() - 1, CB=cb, CLOG=max(cb.bit_length() - 1, 0),
        SCOPED=captured is not None, BIG=(1 << 31) - 1, num_warps=PREV_WARPS)
    return lanes


READ_BINS_BLOCK = 1024


@triton.jit(do_not_specialize=["held_stride", "width"])
def _read_bins_kernel(held, held_stride, width, states, counts, BLOCK: tl.constexpr, BINS: tl.constexpr,
                      SHIFT: tl.constexpr):
    lane = tl.program_id(0)
    state = tl.load(states + lane)
    if state < 0:
        return
    base = counts + state.to(tl.int64) * BINS
    for start in range(0, width, BLOCK):
        at = start + tl.arange(0, BLOCK)
        pick = tl.load(held + lane.to(tl.int64) * held_stride + at, mask=at < width, other=-1)
        valid = (pick >= 0) & ((pick >> SHIFT) < BINS)
        tl.atomic_add(base + (pick >> SHIFT), 1, mask=valid, sem="relaxed")


def read_bins(held, decodes, states, counts, shift):
    if decodes <= 0:
        return 0
    _read_bins_kernel[(decodes,)](held, held.stride(0), held.shape[1], states, counts, BLOCK=READ_BINS_BLOCK,
                                  BINS=counts.shape[1], SHIFT=shift, num_warps=4)
    return decodes
