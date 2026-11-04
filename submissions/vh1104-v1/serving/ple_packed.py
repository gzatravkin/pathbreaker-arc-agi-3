import os

import torch
import triton
import triton.language as tl

ROWS = 2
WARPS = 2
DF11_DECODE_ENV = "TAAF_VLLM_PLE_DF11_DECODE"
DF11_STAGED_ROWS = 64
DF11_STAGED_WARPS = 2
DF11_STAGED_CHUNK = 64
DF11_STAGED_PAD = 4


@triton.jit(do_not_specialize=["count", "start", "end", "escapes"])
def _packed_gather_kernel(table, ids, out, escape_rows, escape_exps, count, start, end, escapes,
                          HALF: tl.constexpr, BLOCK: tl.constexpr, R: tl.constexpr, SEARCH: tl.constexpr):
    lookup = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    asked = lookup < count
    wanted = tl.load(ids + lookup, mask=asked, other=-1).to(tl.int64)
    inside = asked & (wanted >= start) & (wanted < end)
    row = tl.where(inside, wanted - start, 0)
    record = table + row * (1 + 3 * HALF)
    pair = tl.arange(0, BLOCK)
    live = inside[:, None] & (pair[None, :] < HALF)
    base = tl.load(record, mask=inside, other=0).to(tl.int32)
    codes = tl.load(record[:, None] + 1 + pair[None, :], mask=live, other=0).to(tl.int32)
    even_low = tl.load(record[:, None] + 1 + HALF + 2 * pair[None, :], mask=live, other=0).to(tl.int32)
    odd_low = tl.load(record[:, None] + 2 + HALF + 2 * pair[None, :], mask=live, other=0).to(tl.int32)
    even_code = codes & 15
    odd_code = codes >> 4
    even_exp = even_code + base[:, None] - 1
    odd_exp = odd_code + base[:, None] - 1
    even_out = live & (even_code == 0)
    odd_out = live & (odd_code == 0)
    pairs = even_out.to(tl.int32) + odd_out.to(tl.int32)
    needs = tl.sum(pairs, axis=1)
    if tl.max(needs, axis=0) > 0:
        low = tl.zeros([R], dtype=tl.int64)
        high = tl.where(needs > 0, low + escapes, low)
        for _ in range(SEARCH):
            middle = (low + high) // 2
            open_ = low < high
            probe = tl.load(escape_rows + middle, mask=open_, other=0).to(tl.int64)
            low = tl.where(open_ & (probe < row), middle + 1, low)
            high = tl.where(open_ & (probe >= row), middle, high)
        even_at = low[:, None] + (tl.cumsum(pairs, axis=1) - pairs).to(tl.int64)
        odd_at = even_at + even_out.to(tl.int64)
        even_exp = tl.where(even_out, tl.load(escape_exps + even_at, mask=even_out, other=0).to(tl.int32), even_exp)
        odd_exp = tl.where(odd_out, tl.load(escape_exps + odd_at, mask=odd_out, other=0).to(tl.int32), odd_exp)
    even = ((even_low & 128) << 8) | ((even_exp & 255) << 7) | (even_low & 127)
    odd = ((odd_low & 128) << 8) | ((odd_exp & 255) << 7) | (odd_low & 127)
    word = tl.where(live, even | (odd << 16), 0)
    tl.store(out + lookup[:, None] * HALF + pair[None, :], word, mask=asked[:, None] & (pair[None, :] < HALF))


def search_steps(escapes):
    return max(1, int(escapes).bit_length())


def gather(table, ids, out, escape_rows, escape_exps, escapes, start, end, rows=ROWS, warps=WARPS):
    count = ids.numel()
    half = out.shape[-1] // 2
    if count:
        _packed_gather_kernel[(triton.cdiv(count, rows),)](
            table, ids, out.view(torch.int32), escape_rows, escape_exps, count, start, end, escapes, HALF=half,
            BLOCK=triton.next_power_of_2(half), R=rows, SEARCH=search_steps(escapes), num_warps=warps)
    return out


DF11_ROWS = 1
DF11_WARPS = 1
_TOLD = set()


@triton.jit(do_not_specialize=["count", "start", "end", "escapes"])
def _df11_gather_kernel(signs, lengths, bases, stream, lut, ids, out, escape_rows, escape_exps, count, start, end,
                        escapes, WIDTH: tl.constexpr, BLOCK: tl.constexpr, GROUP: tl.constexpr, SPAN: tl.constexpr,
                        LIMIT: tl.constexpr, ESCAPE: tl.constexpr, R: tl.constexpr, SEARCH: tl.constexpr):
    lookup = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    asked = lookup < count
    wanted = tl.load(ids + lookup, mask=asked, other=-1).to(tl.int64)
    inside = asked & (wanted >= start) & (wanted < end)
    row = tl.where(inside, wanted - start, 0)
    head = row // GROUP * GROUP
    member = tl.arange(0, GROUP)
    ahead = inside[:, None] & (member[None, :] < (row - head)[:, None])
    before = tl.load(lengths + head[:, None] + member[None, :], mask=ahead, other=0).to(tl.int64)
    at = tl.load(bases + row // GROUP, mask=inside, other=0) + tl.sum(before, axis=1)
    size = tl.load(lengths + row, mask=inside, other=0).to(tl.int64)
    span = tl.arange(0, SPAN)
    near = stream + at[:, None] + span[None, :]
    held = inside[:, None] & (span[None, :] < size[:, None])
    first = tl.load(near, mask=held, other=0).to(tl.int32)
    second = tl.load(near + 1, mask=held & (span[None, :] + 1 < size[:, None]), other=0).to(tl.int32)
    third = tl.load(near + 2, mask=held & (span[None, :] + 2 < size[:, None]), other=0).to(tl.int32)
    window = first | (second << 8) | (third << 16)
    column = tl.arange(0, BLOCK)
    exps = tl.zeros([R, BLOCK], dtype=tl.int32)
    bit = tl.zeros([R], dtype=tl.int32)
    for k in range(WIDTH):
        word = tl.sum(tl.where(span[None, :] == (bit >> 3)[:, None], window, 0), axis=1)
        entry = tl.load(lut + ((word >> (bit & 7)) & ((1 << LIMIT) - 1)))
        exps = tl.where(column[None, :] == k, (entry >> 4)[:, None], exps)
        bit += entry & 15
    live = inside[:, None] & (column[None, :] < WIDTH)
    escaped = live & (exps == ESCAPE)
    marks = escaped.to(tl.int32)
    needs = tl.sum(marks, axis=1)
    if tl.max(needs, axis=0) > 0:
        low = tl.zeros([R], dtype=tl.int64)
        high = tl.where(needs > 0, low + escapes, low)
        for _ in range(SEARCH):
            middle = (low + high) // 2
            open_ = low < high
            probe = tl.load(escape_rows + middle, mask=open_, other=0).to(tl.int64)
            low = tl.where(open_ & (probe < row), middle + 1, low)
            high = tl.where(open_ & (probe >= row), middle, high)
        slot = low[:, None] + (tl.cumsum(marks, axis=1) - marks).to(tl.int64)
        exps = tl.where(escaped, tl.load(escape_exps + slot, mask=escaped, other=0).to(tl.int32), exps)
    kept = tl.load(signs + row[:, None] * WIDTH + column[None, :], mask=live, other=0).to(tl.int32)
    value = ((kept & 128) << 8) | ((exps & 255) << 7) | (kept & 127)
    tl.store(out + lookup[:, None] * WIDTH + column[None, :], tl.where(live, value, 0).to(tl.int16),
             mask=asked[:, None] & (column[None, :] < WIDTH))


def df11_span(width, limit):
    return triton.next_power_of_2(-(-int(width) * int(limit) // 8))


@triton.jit(do_not_specialize=["count", "start", "end", "escapes"])
def _df11_staged_kernel(signs, lengths, bases, stream, lut, ids, out, escape_rows, escape_exps, packed, kept_stage,
                        count, start, end, escapes, WIDTH: tl.constexpr, GROUP: tl.constexpr, SPAN: tl.constexpr,
                        STRIDE: tl.constexpr, CHUNK: tl.constexpr, LIMIT: tl.constexpr, ESCAPE: tl.constexpr,
                        R: tl.constexpr, SEARCH: tl.constexpr):
    lookup = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    asked = lookup < count
    wanted = tl.load(ids + lookup, mask=asked, other=-1).to(tl.int64)
    inside = asked & (wanted >= start) & (wanted < end)
    row = tl.where(inside, wanted - start, 0)
    head = row // GROUP * GROUP
    member = tl.arange(0, GROUP)
    ahead = inside[:, None] & (member[None, :] < (row - head)[:, None])
    before = tl.load(lengths + head[:, None] + member[None, :], mask=ahead, other=0).to(tl.int64)
    at = tl.load(bases + row // GROUP, mask=inside, other=0) + tl.sum(before, axis=1)
    size = tl.load(lengths + row, mask=inside, other=0).to(tl.int64)
    piece = tl.arange(0, CHUNK)
    for first in range(0, STRIDE, CHUNK):
        spot = first + piece
        held = inside[:, None] & (spot[None, :] < size[:, None])
        byte = tl.load(stream + at[:, None] + spot[None, :], mask=held, other=0)
        tl.store(packed + lookup[:, None] * STRIDE + spot[None, :], byte,
                 mask=asked[:, None] & (spot[None, :] < STRIDE))
    for first in range(0, WIDTH, CHUNK):
        column = first + piece
        live = inside[:, None] & (column[None, :] < WIDTH)
        sign = tl.load(signs + row[:, None] * WIDTH + column[None, :], mask=live, other=0)
        tl.store(kept_stage + lookup[:, None] * WIDTH + column[None, :], sign,
                 mask=asked[:, None] & (column[None, :] < WIDTH))
    low = tl.zeros([R], dtype=tl.int64)
    high = low + escapes
    for _ in range(SEARCH):
        middle = (low + high) // 2
        open_ = low < high
        probe = tl.load(escape_rows + middle, mask=open_, other=0).to(tl.int64)
        low = tl.where(open_ & (probe < row), middle + 1, low)
        high = tl.where(open_ & (probe >= row), middle, high)
    tl.debug_barrier()
    base = packed + lookup * STRIDE
    signs_row = kept_stage + lookup * WIDTH
    bit = tl.zeros([R], dtype=tl.int32)
    nth = tl.zeros([R], dtype=tl.int64)
    for k in range(WIDTH):
        byte_at = bit >> 3
        word = (tl.load(base + byte_at, mask=asked, other=0).to(tl.int32)
                | (tl.load(base + byte_at + 1, mask=asked, other=0).to(tl.int32) << 8)
                | (tl.load(base + byte_at + 2, mask=asked, other=0).to(tl.int32) << 16))
        entry = tl.load(lut + ((word >> (bit & 7)) & ((1 << LIMIT) - 1)))
        exp = entry >> 4
        bit += entry & 15
        escaped = inside & (exp == ESCAPE)
        exp = tl.where(escaped, tl.load(escape_exps + low + nth, mask=escaped, other=0).to(tl.int32), exp)
        nth += escaped.to(tl.int64)
        kept = tl.load(signs_row + k, mask=asked, other=0).to(tl.int32)
        value = ((kept & 128) << 8) | ((exp & 255) << 7) | (kept & 127)
        tl.store(out + lookup * WIDTH + k, tl.where(inside, value, 0).to(tl.int16), mask=asked)


def df11_staged_on():
    return str(os.environ.get(DF11_DECODE_ENV) or "").strip().lower() == "staged"


def df11_gather_staged(signs, lengths, bases, stream, lut, ids, out, escape_rows, escape_exps, escapes, start, end,
                       group, limit, rows=DF11_STAGED_ROWS, warps=DF11_STAGED_WARPS):
    count = ids.numel()
    width = out.shape[-1]
    if "staged" not in _TOLD:
        _TOLD.add("staged")
        print(f"PLE_DF11 gather launch: staged decode, {rows} rows a program, {warps} warp(s), one row a thread",
              flush=True)
    if count:
        span = df11_span(width, limit)
        stride = span + DF11_STAGED_PAD
        packed = torch.empty(count * stride + DF11_STAGED_PAD, dtype=torch.uint8, device=out.device)
        kept = torch.empty(count * width, dtype=torch.uint8, device=out.device)
        _df11_staged_kernel[(triton.cdiv(count, rows),)](
            signs, lengths, bases, stream, lut, ids, out.view(torch.int16), escape_rows, escape_exps, packed, kept,
            count, start, end, escapes, WIDTH=width, GROUP=group, SPAN=span, STRIDE=stride,
            CHUNK=DF11_STAGED_CHUNK, LIMIT=limit, ESCAPE=256, R=rows, SEARCH=search_steps(escapes),
            num_warps=warps)
    return out


def df11_gather(signs, lengths, bases, stream, lut, ids, out, escape_rows, escape_exps, escapes, start, end, group,
                limit, rows=DF11_ROWS, warps=DF11_WARPS):
    if df11_staged_on():
        return df11_gather_staged(signs, lengths, bases, stream, lut, ids, out, escape_rows, escape_exps, escapes,
                                  start, end, group, limit)
    count = ids.numel()
    width = out.shape[-1]
    if "launch" not in _TOLD:
        _TOLD.add("launch")
        print(f"PLE_DF11 gather launch: {rows} row(s) a program, {warps} warp(s), stock Huffman decode", flush=True)
    if count:
        _df11_gather_kernel[(triton.cdiv(count, rows),)](
            signs, lengths, bases, stream, lut, ids, out.view(torch.int16), escape_rows, escape_exps, count, start,
            end, escapes, WIDTH=width, BLOCK=triton.next_power_of_2(width), GROUP=group, SPAN=df11_span(width, limit),
            LIMIT=limit, ESCAPE=256, R=rows, SEARCH=search_steps(escapes), num_warps=warps)
    return out


Q_ROWS = 4
Q_WARPS = 4


@triton.jit(do_not_specialize=["count", "start", "end"])
def _q_gather_kernel(codes, peaks, scales, ids, out, count, start, end, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                     ROW_BYTES: tl.constexpr, GROUP: tl.constexpr, GROUPS: tl.constexpr, SIX: tl.constexpr,
                     R: tl.constexpr):
    lookup = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    asked = lookup < count
    wanted = tl.load(ids + lookup, mask=asked, other=-1).to(tl.int64)
    inside = asked & (wanted >= start) & (wanted < end)
    row = tl.where(inside, wanted - start, 0)
    column = tl.arange(0, BLOCK)
    live = inside[:, None] & (column[None, :] < WIDTH)
    if SIX:
        at = codes + row[:, None] * ROW_BYTES + (column[None, :] >> 2) * 3
        word = (tl.load(at, mask=live, other=0).to(tl.int32)
                | (tl.load(at + 1, mask=live, other=0).to(tl.int32) << 8)
                | (tl.load(at + 2, mask=live, other=0).to(tl.int32) << 16))
        code = (word >> ((column[None, :] & 3) * 6)) & 63
        signed_zero = code == 0
        level = (code - 32).to(tl.float32)
    else:
        code = tl.load(codes + row[:, None] * ROW_BYTES + column[None, :], mask=live, other=0).to(tl.int32)
        signed_zero = code == -128
        level = code.to(tl.float32)
    peak = tl.load(peaks + row[:, None] * GROUPS + column[None, :] // GROUP, mask=live, other=0).to(tl.int32) & 65535
    scale = tl.load(scales + peak, mask=live, other=0.0)
    wide = (level * scale).to(tl.int32, bitcast=True)
    bits = ((wide + 32767 + ((wide >> 16) & 1)) >> 16) & 65535
    bits = tl.where(signed_zero, 32768, bits)
    bits = tl.where(live, bits, 0)
    tl.store(out + lookup[:, None] * WIDTH + column[None, :], bits.to(tl.int16),
             mask=asked[:, None] & (column[None, :] < WIDTH))


def q_gather(codes, peaks, scales, ids, out, start, end, row_bytes, group, six, rows=Q_ROWS, warps=Q_WARPS):
    count = ids.numel()
    width = out.shape[-1]
    if "q" not in _TOLD:
        _TOLD.add("q")
        print(f"PLE_QSTORE gather launch: {'6' if six else '8'}-bit codes, {rows} row(s) a program, {warps} "
              f"warp(s), one scale per {group} columns", flush=True)
    if count:
        _q_gather_kernel[(triton.cdiv(count, rows),)](
            codes, peaks, scales, ids, out.view(torch.int16), count, start, end, WIDTH=width,
            BLOCK=triton.next_power_of_2(width), ROW_BYTES=row_bytes, GROUP=group, GROUPS=width // group,
            SIX=bool(six), R=rows, num_warps=warps)
    return out


QHOT_MULT = 2654435761
QHOT_PROBES = 8
QHOT_SHIFT = 13


@triton.jit(do_not_specialize=["count", "start", "end"])
def _q_gather_hot_kernel(codes, peaks, scales, ids, out, table, hot_codes, hot_peaks, count, start, end,
                         WIDTH: tl.constexpr, BLOCK: tl.constexpr, ROW_BYTES: tl.constexpr, GROUP: tl.constexpr,
                         GROUPS: tl.constexpr, SIX: tl.constexpr, R: tl.constexpr, CAP: tl.constexpr,
                         PROBES: tl.constexpr, MULT: tl.constexpr, SHIFT: tl.constexpr):
    lookup = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    asked = lookup < count
    wanted = tl.load(ids + lookup, mask=asked, other=-1).to(tl.int64)
    inside = asked & (wanted >= start) & (wanted < end)
    row = tl.where(inside, wanted - start, 0)
    home = ((row * MULT) >> SHIFT) & (CAP - 1)
    slot = tl.full([R], -1, tl.int64)
    for probe in tl.static_range(PROBES):
        entry = tl.load(table + ((home + probe) & (CAP - 1)), mask=inside, other=-1)
        found = inside & (slot < 0) & ((entry & 0xFFFFFFFF) == row)
        slot = tl.where(found, entry >> 32, slot)
    hit = slot >= 0
    slot = tl.maximum(slot, 0)
    column = tl.arange(0, BLOCK)
    live = inside[:, None] & (column[None, :] < WIDTH)
    cold = live & (~hit)[:, None]
    warm = live & hit[:, None]
    if SIX:
        step = (column[None, :] >> 2) * 3
        at = codes + row[:, None] * ROW_BYTES + step
        hat = hot_codes + slot[:, None] * ROW_BYTES + step
        word = ((tl.load(at, mask=cold, other=0).to(tl.int32) | tl.load(hat, mask=warm, other=0).to(tl.int32))
                | ((tl.load(at + 1, mask=cold, other=0).to(tl.int32)
                    | tl.load(hat + 1, mask=warm, other=0).to(tl.int32)) << 8)
                | ((tl.load(at + 2, mask=cold, other=0).to(tl.int32)
                    | tl.load(hat + 2, mask=warm, other=0).to(tl.int32)) << 16))
        code = (word >> ((column[None, :] & 3) * 6)) & 63
        signed_zero = code == 0
        level = (code - 32).to(tl.float32)
    else:
        code = (tl.load(codes + row[:, None] * ROW_BYTES + column[None, :], mask=cold, other=0).to(tl.int32)
                + tl.load(hot_codes + slot[:, None] * ROW_BYTES + column[None, :], mask=warm, other=0).to(tl.int32))
        signed_zero = code == -128
        level = code.to(tl.float32)
    peak = ((tl.load(peaks + row[:, None] * GROUPS + column[None, :] // GROUP, mask=cold, other=0).to(tl.int32)
             | tl.load(hot_peaks + slot[:, None] * GROUPS + column[None, :] // GROUP, mask=warm, other=0)
             .to(tl.int32)) & 65535)
    scale = tl.load(scales + peak, mask=live, other=0.0)
    wide = (level * scale).to(tl.int32, bitcast=True)
    bits = ((wide + 32767 + ((wide >> 16) & 1)) >> 16) & 65535
    bits = tl.where(signed_zero, 32768, bits)
    bits = tl.where(live, bits, 0)
    tl.store(out + lookup[:, None] * WIDTH + column[None, :], bits.to(tl.int16),
             mask=asked[:, None] & (column[None, :] < WIDTH))


def qhot_table(rows, capacity, probes=QHOT_PROBES):
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    table = np.full(capacity, -1, dtype=np.int64)
    slot_of = np.full(rows.shape[0], -1, dtype=np.int64)
    home = ((rows * QHOT_MULT) >> QHOT_SHIFT) & (capacity - 1)
    waiting = np.arange(rows.shape[0])
    for probe in range(probes):
        if not waiting.size:
            break
        at = (home[waiting] + probe) & (capacity - 1)
        free = table[at] == -1
        cand, cand_at = waiting[free], at[free]
        _, first = np.unique(cand_at, return_index=True)
        chosen, chosen_at = cand[first], cand_at[first]
        slot_of[chosen] = chosen
        table[chosen_at] = rows[chosen] | (chosen.astype(np.int64) << 32)
        placed = np.zeros(rows.shape[0], dtype=bool)
        placed[chosen] = True
        waiting = waiting[~placed[waiting]]
    return table, slot_of


def q_gather_hot(codes, peaks, scales, ids, out, start, end, row_bytes, group, six, hot, rows=Q_ROWS,
                 warps=Q_WARPS):
    count = ids.numel()
    width = out.shape[-1]
    table, hot_codes, hot_peaks = hot
    if "qhot" not in _TOLD:
        _TOLD.add("qhot")
        print(f"PLE_QHOT gather launch: {'6' if six else '8'}-bit codes, hot rows from the GPU copy "
              f"({hot_codes.shape[0]:,} rows, {table.numel():,}-slot table, {QHOT_PROBES} probes), misses from the "
              f"pinned table", flush=True)
    if count:
        _q_gather_hot_kernel[(triton.cdiv(count, rows),)](
            codes, peaks, scales, ids, out.view(torch.int16), table, hot_codes, hot_peaks, count, start, end,
            WIDTH=width, BLOCK=triton.next_power_of_2(width), ROW_BYTES=row_bytes, GROUP=group, GROUPS=width // group,
            SIX=bool(six), R=rows, CAP=table.numel(), PROBES=QHOT_PROBES, MULT=QHOT_MULT, SHIFT=QHOT_SHIFT,
            num_warps=warps)
    return out
