# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

BASE_ROWS = 512
ROWS_ENV = "TAAF_VLLM_HC_FUSED_ROWS"
WIDE_LIMIT = 8192
WIDE_BUCKET = 256
WIDE_TUNE_LIMIT = 32
TUNE_LIMIT = 96
REPLAYS = 20
TOLERANCE = 0.05
NORM_BLOCK = 512
DOWN_BN = 64
DOWN_BK = 64
UP_BK = 32
SPLITS = (4, 5, 8, 10, 16, 20, 32, 40)
UP_WIDTHS = (32, 64)
SPLIT_WIDTHS = (16, 32, 64)
SPLIT = "split"
UP_TILE_SMALL = 2048
UP_TILE_WIDE = 8192
WIDE_ROWS = 128
LORA_HINT = 324
FALLBACK_SMS = 188
STOCK = "stock"


MIXED = "mixed"


MIXED_F1_ENV = "TAAF_VLLM_HC_MIXED_F1"


MIXED_F2_ENV = "TAAF_VLLM_HC_MIXED_F2"


MIXED_SHAPE_ENV = "TAAF_VLLM_HC_MIXED_SHAPE"
MIXED_UP4_ENV = "TAAF_VLLM_HC_MIXED_UP4"
MIXED_UP4_DEFAULT = (128, 32, 4, 4, 32)


MIXED_DEFAULT_BAND = (2560, WIDE_LIMIT)


MIXED_MAX_SPLITS = 4


MIXED_COUNTERS = 4096


MIXED_TOLERANCE = 1e-2


MIXED_TILES = (16, 32, 64, 128)


MIXED_UP = (128, 32)


MIXED_DOWN = ((128, 128, 2), (128, 64, 1), (64, 128, 1))
DECODE = "decode"
DECODE_ENV = "TAAF_VLLM_HC_DECODE"
DECODE_SHAPE_ENV = "TAAF_VLLM_HC_DECODE_SHAPE"
DECODE_DOWN_ENV = "TAAF_VLLM_HC_DECODE_DOWN"
DECODE_UP4_ENV = "TAAF_VLLM_HC_DECODE_UP4"
DECODE_UP4_DEFAULT = (64, 32, 4, 3, 32)
DECODE_MIN_ROWS_ENV = "TAAF_VLLM_HC_DECODE_DOWN_MIN_ROWS"
DECODE_MIN_ROWS_DEFAULT = 300
DECODE_CUBLAS_TAG = "decode-cublas"
DECODE_DEFAULT_BAND = (96, 640)
DECODE_ROWS_LIMIT = 1024
DECODE_MAX_SPLITS = 40
DECODE_COUNTERS = 8192
DECODE_TOLERANCE = 1e-2
DECODE_BM_SMALL = 64
DECODE_BM_LARGE = 128
DECODE_BM_SWITCH = 256
DECODE_UP = (64, 32)
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
ENV = "TAAF_VLLM_HC_FUSED"
PICKS_ENV = "TAAF_VLLM_HC_FUSED_PICKS_FILE"
PICK_SPACE = ("BASE_ROWS", "WIDE_BUCKET", "SPLITS", "UP_WIDTHS", "SPLIT_WIDTHS", "DOWN_BN", "DOWN_BK", "UP_BK",
              "NORM_BLOCK")
ON = str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")
_SITES = {}
_PICKS = {}
_CHECKED = set()
_DISABLED = set()
_STATE = {"pdl": None, "tuned": 0, "wide_tuned": 0, "stream": None, "sms": None, "pool": None, "graph": None,
          "seeded": None}
_ANNOUNCED = set()


def rows_ceiling(said):
    said = str(said or "").strip()
    if said in ("", "0"):
        return BASE_ROWS
    if not said.isdigit() or not BASE_ROWS < int(said) <= WIDE_LIMIT or int(said) % WIDE_BUCKET:
        raise ValueError(f"{ROWS_ENV}={said!r} is not a multiple of {WIDE_BUCKET} in ({BASE_ROWS}, {WIDE_LIMIT}]")
    return int(said)


MAX_ROWS = rows_ceiling(os.environ.get(ROWS_ENV))


def mixed_band(said, name):
    said = str(said or "").strip().lower().replace(" ", "")
    if said in ("", "0", "off", "false", "no"):
        return None
    if said in ("1", "on", "true", "yes"):
        return MIXED_DEFAULT_BAND
    lo, dash, hi = said.partition("-")
    if not (dash and lo.isdigit() and hi.isdigit()) or not 1 <= int(lo) <= int(hi) <= WIDE_LIMIT:
        raise ValueError(f"{name}={said!r} is not <lo>-<hi> with 1 <= lo <= hi <= {WIDE_LIMIT}")
    return int(lo), int(hi)


def mixed_shape(said):
    said = str(said or "").strip().replace(" ", "")
    if not said:
        return None
    parts = said.split(",")
    if len(parts) != 5 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{MIXED_SHAPE_ENV}={said!r} is not bm,bn,splits,bm_up,bh")
    bm, bn, splits, bm_up, bh = (int(part) for part in parts)
    if (bm not in MIXED_TILES or bn not in MIXED_TILES or bm_up not in MIXED_TILES
            or bh not in UP_WIDTHS or not 1 <= splits <= MIXED_MAX_SPLITS):
        raise ValueError(f"{MIXED_SHAPE_ENV}={said!r} names a tile outside the supported set")
    return bm, bn, splits, bm_up, bh


def mixed_up4(said):
    said = str(said or "").strip().lower().replace(" ", "")
    if said in ("0", "off", "false", "no"):
        return None
    if said in ("", "1", "on", "true", "yes"):
        return MIXED_UP4_DEFAULT
    parts = said.split(",")
    if len(parts) != 5 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{MIXED_UP4_ENV}={said!r} is not bm,bh,warps,stages,bk")
    bm, bh, warps, stages, bk = (int(part) for part in parts)
    if bm not in MIXED_TILES or bh not in UP_WIDTHS or warps not in (2, 4, 8) or not 2 <= stages <= 5             or bk not in (16, 32, 64):
        raise ValueError(f"{MIXED_UP4_ENV}={said!r} names a tile outside the supported set")
    return bm, bh, warps, stages, bk


BAND_F1 = mixed_band(os.environ.get(MIXED_F1_ENV), MIXED_F1_ENV)


BAND_F2 = mixed_band(os.environ.get(MIXED_F2_ENV), MIXED_F2_ENV)


FORCED_MIXED = mixed_shape(os.environ.get(MIXED_SHAPE_ENV))
MIXED_UP4 = mixed_up4(os.environ.get(MIXED_UP4_ENV))


def decode_band(said):
    said = str(said or "").strip().lower().replace(" ", "")
    if said in ("", "0", "off", "false", "no"):
        return None
    if said in ("1", "on", "true", "yes"):
        return DECODE_DEFAULT_BAND
    lo, dash, hi = said.partition("-")
    if not (dash and lo.isdigit() and hi.isdigit()) or not 1 <= int(lo) <= int(hi) <= DECODE_ROWS_LIMIT:
        raise ValueError(f"{DECODE_ENV}={said!r} is not <lo>-<hi> with 1 <= lo <= hi <= {DECODE_ROWS_LIMIT}")
    return int(lo), int(hi)


def decode_shape(said):
    said = str(said or "").strip().replace(" ", "")
    if not said:
        return None
    parts = said.split(",")
    if len(parts) != 5 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{DECODE_SHAPE_ENV}={said!r} is not bm,bn,splits,bm_up,bh")
    bm, bn, splits, bm_up, bh = (int(part) for part in parts)
    if (bm not in (16, 32, 64, 128) or bn not in (16, 32, 64, 128) or bm_up not in (16, 32, 64, 128)
            or bh not in UP_WIDTHS or splits not in SPLITS):
        raise ValueError(f"{DECODE_SHAPE_ENV}={said!r} names a tile outside the supported set")
    return bm, bn, splits, bm_up, bh


def decode_down_mode(said):
    said = str(said or "").strip().lower()
    if said in ("", "split"):
        return "split"
    if said == "cublas":
        return "cublas"
    raise ValueError(f"{DECODE_DOWN_ENV}={said!r} is not 'cublas' or 'split'")


def decode_min_rows(said):
    said = str(said or "").strip()
    if not said:
        return DECODE_MIN_ROWS_DEFAULT
    if not said.isdigit() or not 1 <= int(said) <= DECODE_ROWS_LIMIT:
        raise ValueError(f"{DECODE_MIN_ROWS_ENV}={said!r} is not a row count from 1 to {DECODE_ROWS_LIMIT}")
    return int(said)


def decode_up4(said):
    said = str(said or "").strip().lower().replace(" ", "")
    if said in ("0", "off", "false", "no"):
        return None
    if said in ("", "1", "on", "true", "yes"):
        return DECODE_UP4_DEFAULT
    parts = said.split(",")
    if len(parts) != 5 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{DECODE_UP4_ENV}={said!r} is not bm,bh,warps,stages,bk")
    bm, bh, warps, stages, bk = (int(part) for part in parts)
    if bm not in (16, 32, 64, 128) or bh not in UP_WIDTHS or warps not in (2, 4, 8) or not 2 <= stages <= 5             or bk not in (16, 32, 64):
        raise ValueError(f"{DECODE_UP4_ENV}={said!r} names a tile outside the supported set")
    return bm, bh, warps, stages, bk


BAND = decode_band(os.environ.get(DECODE_ENV))
DECODE_DOWN = decode_down_mode(os.environ.get(DECODE_DOWN_ENV))
DECODE_UP4 = decode_up4(os.environ.get(DECODE_UP4_ENV))
DECODE_MIN_ROWS = decode_min_rows(os.environ.get(DECODE_MIN_ROWS_ENV))
FORCED_SHAPE = decode_shape(os.environ.get(DECODE_SHAPE_ENV))


def say_as(prefix, event, **fields):
    detail = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"{prefix} {event} {detail}".rstrip(), flush=True)


def say(event, **fields):
    say_as("HC_FUSED", event, **fields)


def announce_down(event, key, **fields):
    if ("down", event, key) not in _ANNOUNCED:
        _ANNOUNCED.add(("down", event, key))
        say_as("HC_DECODE_DOWN", event, **fields)


def decode_up_tile(hc_count):
    if DECODE_UP4 is not None and hc_count == 4:
        bm, bh, warps, stages, bk = DECODE_UP4
        return f"{bm}x{bh}/quad/w{warps}s{stages}k{bk}"
    return f"{DECODE_UP[0]}x{DECODE_UP[1]}"


def announce(event, key="", **fields):
    if (event, key) not in _ANNOUNCED:
        _ANNOUNCED.add((event, key))
        say(event, **fields)


def announce_mixed(event, key, **fields):
    if ("mixed", event, key) not in _ANNOUNCED:
        _ANNOUNCED.add(("mixed", event, key))
        say_as("HC_MIXED", event, **fields)


@triton.jit
def _hc_norm_kernel(
    res_ptr,
    blk_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    xn_ptr,
    acc_ptr,
    stride_res,
    stride_blk,
    stride_inj,
    stride_out,
    stride_xn,
    H: tl.constexpr,
    HC: tl.constexpr,
    NACC: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    HAS_INJ: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(H, BLOCK)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)
    ACC_BLOCK: tl.constexpr = triton.next_power_of_2(NACC)

    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    tiles = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tiles[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    mask = offs_inner < H
    offs = stream * H + offs_inner
    w_offs = offs_inner if W_SHARED else offs
    offs_acc = tl.arange(0, ACC_BLOCK)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    if HAS_COMBINE:
        res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)
        block = tl.load(blk_ptr + row * stride_blk + offs_inner, mask, other=0.0)
        block = block.to(tl.float32)
        if HAS_INJ:
            offs_hc = tl.arange(0, HC_PAD)
            inj = tl.load(inj_ptr + row * stride_inj + offs_hc, offs_hc < HC, other=0.0)
            inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
            block = block * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
        x = (res.to(tl.float32) + block).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + offs, x, mask=mask)
        x = x.to(tl.float32)
    else:
        x = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0).to(tl.float32)

    tl.store(
        acc_ptr + row * NACC + offs_acc,
        tl.zeros((ACC_BLOCK,), dtype=tl.float32),
        mask=(offs_acc < NACC) & (stream == 0),
    )
    rrms = tl.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / H + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    w = tl.load(w_ptr + w_offs, mask, other=0.0)
    y = x * rrms
    y += y * w.to(tl.float32)
    tl.store(xn_ptr + row * stride_xn + offs, y, mask)


@triton.jit(do_not_specialize=["rows"])
def _hc_down_kernel(
    xn_ptr,
    w_ptr,
    acc_ptr,
    rows,
    k_chunk,
    stride_xn,
    stride_w,
    NOUT: tl.constexpr,
    NACC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * k_chunk + tl.arange(0, BK)
    mask_m = offs_m < rows
    mask_n = offs_n < NOUT
    a_ptrs = xn_ptr + offs_m[:, None] * stride_xn + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * stride_w + offs_k[:, None]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, k_chunk, BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    tl.atomic_add(
        acc_ptr + offs_m[:, None] * NACC + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["rows"])
def _hc_up_mix_kernel(
    acc_ptr,
    w_ptr,
    xn_ptr,
    out_ptr,
    inj_ptr,
    rows,
    stride_acc,
    stride_w,
    stride_xn,
    stride_out,
    H: tl.constexpr,
    HC: tl.constexpr,
    R: tl.constexpr,
    HAS_INJ: tl.constexpr,
    PRESILU: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < rows
    lanes = tl.arange(0, HC * BH)
    cols = (lanes // BH) * H + pid_c * BH + lanes % BH
    offs_k = tl.arange(0, BK)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    gate = tl.zeros((BM, HC * BH), dtype=tl.float32)
    for k in range(0, R, BK):
        mask_k = (k + offs_k) < R
        lora = tl.load(
            acc_ptr + offs_m[:, None] * stride_acc + (k + offs_k)[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if PRESILU:
            lora = lora.to(w_ptr.dtype.element_ty)
        else:
            lora = lora.to(out_ptr.dtype.element_ty).to(tl.float32) / HC
            lora = (lora * tl.sigmoid(lora)).to(w_ptr.dtype.element_ty)
        w = tl.load(
            w_ptr + cols[None, :] * stride_w + (k + offs_k)[:, None],
            mask=mask_k[:, None],
            other=0.0,
        )
        gate = tl.dot(lora, w, gate)

    gate = gate.to(out_ptr.dtype.element_ty).to(tl.float32)
    x = tl.load(xn_ptr + offs_m[:, None] * stride_xn + cols[None, :], mask=mask_m[:, None], other=0.0)
    mixed = tl.sigmoid(gate) * x.to(tl.float32)
    mixed = tl.sum(tl.reshape(mixed, (BM, HC, BH)), axis=1) / HC

    if HAS_INJ:
        offs_h = tl.arange(0, HC)
        inj_mask = mask_m[:, None] & (offs_h < HC)[None, :] & (pid_c == 0)
        inj = tl.load(acc_ptr + offs_m[:, None] * stride_acc + R + offs_h[None, :], mask=inj_mask,
                      other=0.0)
        tl.store(inj_ptr + offs_m[:, None] * HC + offs_h[None, :], inj, mask=inj_mask)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    offs_c = pid_c * BH + tl.arange(0, BH)
    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_c[None, :], mixed, mask=mask_m[:, None])


@triton.jit(do_not_specialize=["rows"])
def _hc_up_mix4_kernel(
    acc_ptr,
    w_ptr,
    xn_ptr,
    out_ptr,
    inj_ptr,
    rows,
    stride_acc,
    stride_w,
    stride_xn,
    stride_out,
    H: tl.constexpr,
    HC: tl.constexpr,
    R: tl.constexpr,
    HAS_INJ: tl.constexpr,
    PRESILU: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < rows
    offs_n = pid_c * BH + tl.arange(0, BH)
    offs_k = tl.arange(0, BK)
    w_ptrs = w_ptr + offs_n[None, :] * stride_w + offs_k[:, None]
    a_ptrs = acc_ptr + offs_m[:, None] * stride_acc + offs_k[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc0 = tl.zeros((BM, BH), dtype=tl.float32)
    acc1 = tl.zeros((BM, BH), dtype=tl.float32)
    acc2 = tl.zeros((BM, BH), dtype=tl.float32)
    acc3 = tl.zeros((BM, BH), dtype=tl.float32)
    for k in range(0, R, BK):
        mask_k = (k + offs_k) < R
        lora = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        if PRESILU:
            lora = lora.to(w_ptr.dtype.element_ty)
        else:
            lora = lora.to(out_ptr.dtype.element_ty).to(tl.float32) / HC
            lora = (lora * tl.sigmoid(lora)).to(w_ptr.dtype.element_ty)
        w0 = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)
        acc0 = tl.dot(lora, w0, acc0)
        w1 = tl.load(w_ptrs + H * stride_w, mask=mask_k[:, None], other=0.0)
        acc1 = tl.dot(lora, w1, acc1)
        w2 = tl.load(w_ptrs + 2 * H * stride_w, mask=mask_k[:, None], other=0.0)
        acc2 = tl.dot(lora, w2, acc2)
        w3 = tl.load(w_ptrs + 3 * H * stride_w, mask=mask_k[:, None], other=0.0)
        acc3 = tl.dot(lora, w3, acc3)
        a_ptrs += BK
        w_ptrs += BK

    x_ptrs = xn_ptr + offs_m[:, None] * stride_xn + offs_n[None, :]
    mixed = tl.sigmoid(acc0.to(out_ptr.dtype.element_ty).to(tl.float32)) * tl.load(
        x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    mixed += tl.sigmoid(acc1.to(out_ptr.dtype.element_ty).to(tl.float32)) * tl.load(
        x_ptrs + H, mask=mask_m[:, None], other=0.0).to(tl.float32)
    mixed += tl.sigmoid(acc2.to(out_ptr.dtype.element_ty).to(tl.float32)) * tl.load(
        x_ptrs + 2 * H, mask=mask_m[:, None], other=0.0).to(tl.float32)
    mixed += tl.sigmoid(acc3.to(out_ptr.dtype.element_ty).to(tl.float32)) * tl.load(
        x_ptrs + 3 * H, mask=mask_m[:, None], other=0.0).to(tl.float32)
    mixed = mixed / HC

    if HAS_INJ:
        offs_h = tl.arange(0, HC)
        inj_mask = mask_m[:, None] & (offs_h < HC)[None, :] & (pid_c == 0)
        inj = tl.load(acc_ptr + offs_m[:, None] * stride_acc + R + offs_h[None, :], mask=inj_mask,
                      other=0.0)
        tl.store(inj_ptr + offs_m[:, None] * HC + offs_h[None, :], inj, mask=inj_mask)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_n[None, :], mixed, mask=mask_m[:, None])


@triton.jit(do_not_specialize=["rows"])
def _hc_mixed_down_kernel(
    xn_ptr,
    w_ptr,
    part_ptr,
    cnt_ptr,
    lo_ptr,
    rows,
    k_chunk,
    stride_xn,
    stride_w,
    stride_lo,
    NOUT: tl.constexpr,
    NACC: tl.constexpr,
    R: tl.constexpr,
    HC: tl.constexpr,
    SPLITS: tl.constexpr,
    SILU: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * k_chunk + tl.arange(0, BK)
    mask_m = offs_m < rows
    mask_n = offs_n < NOUT
    mask = mask_m[:, None] & mask_n[None, :]
    a_ptrs = xn_ptr + offs_m[:, None] * stride_xn + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * stride_w + offs_k[:, None]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, k_chunk, BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if SPLITS == 1:
        total = acc
    else:
        tile = pid_m * tl.num_programs(1) + pid_n
        slot = offs_m[:, None] * NACC + offs_n[None, :]
        tl.store(part_ptr + pid_k * rows * NACC + slot, acc, mask=mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        total = tl.zeros((BM, BN), dtype=tl.float32)
        if arrived == SPLITS - 1:
            tl.debug_barrier()
            for s in range(SPLITS):
                total += tl.load(part_ptr + s * rows * NACC + slot, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(cnt_ptr + tile, 0)
    lora = total.to(lo_ptr.dtype.element_ty)
    if SILU:
        scaled = lora.to(tl.float32) / HC
        silu = (scaled * tl.sigmoid(scaled)).to(lo_ptr.dtype.element_ty)
        lora = tl.where(offs_n[None, :] < R, silu, lora)
    out_mask = mask
    if SPLITS != 1:
        out_mask = mask & (arrived == SPLITS - 1)
    tl.store(lo_ptr + offs_m[:, None] * stride_lo + offs_n[None, :], lora, mask=out_mask)


@triton.jit(do_not_specialize=["rows"])
def _hc_decode_down_kernel(
    xn_ptr,
    w_ptr,
    part_ptr,
    cnt_ptr,
    lo_ptr,
    rows,
    k_chunk,
    stride_xn,
    stride_w,
    stride_lo,
    NOUT: tl.constexpr,
    NACC: tl.constexpr,
    R: tl.constexpr,
    HC: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * k_chunk + tl.arange(0, BK)
    mask_m = offs_m < rows
    mask_n = offs_n < NOUT
    mask = mask_m[:, None] & mask_n[None, :]
    a_ptrs = xn_ptr + offs_m[:, None] * stride_xn + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * stride_w + offs_k[:, None]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, k_chunk, BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    tile = pid_m * tl.num_programs(1) + pid_n
    slot = offs_m[:, None] * NACC + offs_n[None, :]
    tl.store(part_ptr + pid_k * rows * NACC + slot, acc, mask=mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
    if arrived == SPLITS - 1:
        tl.debug_barrier()
        total = tl.zeros((BM, BN), dtype=tl.float32)
        for s in range(SPLITS):
            total += tl.load(part_ptr + s * rows * NACC + slot, mask=mask, other=0.0, cache_modifier=".cg")
        lora = total.to(lo_ptr.dtype.element_ty)
        scaled = lora.to(tl.float32) / HC
        silu = (scaled * tl.sigmoid(scaled)).to(lo_ptr.dtype.element_ty)
        value = tl.where(offs_n[None, :] < R, silu, lora)
        tl.store(lo_ptr + offs_m[:, None] * stride_lo + offs_n[None, :], value, mask=mask)
        tl.store(cnt_ptr + tile, 0)


def _pdl():
    if _STATE["pdl"] is None:
        _STATE["pdl"] = bool(current_platform.is_arch_support_pdl())
    return _STATE["pdl"]


def row_block(rows):
    return min(64, max(16, triton.next_power_of_2(rows)))


def fused(hidden, block, injection, norm_weight, down_weight, up_weight, eps, hc_count,
          lora_rank, use_combine, pick):
    rows, width = hidden.shape
    hidden_size = width // hc_count
    nout = lora_rank + (hc_count if use_combine else 0)
    nacc = triton.cdiv(nout, 16) * 16
    bm, bn, splits, bm_up, bh = pick
    bk = DOWN_BK if bm * bn <= 64 * 64 else DOWN_BK // 2
    pdl = _pdl()
    has_combine = block is not None
    has_inj = injection is not None
    out = hidden.new_empty(hidden.shape) if has_combine else None
    xn = hidden.new_empty(hidden.shape)
    acc = torch.empty((rows, nacc), device=hidden.device, dtype=torch.float32)
    _hc_norm_kernel[(rows * hc_count,)](
        hidden,
        block if has_combine else hidden,
        injection if has_inj else hidden,
        norm_weight,
        out if has_combine else xn,
        xn,
        acc,
        hidden.stride(0),
        block.stride(0) if has_combine else 0,
        injection.stride(0) if has_inj else 0,
        out.stride(0) if has_combine else 0,
        xn.stride(0),
        H=hidden_size,
        HC=hc_count,
        NACC=nacc,
        HAS_COMBINE=has_combine,
        HAS_INJ=has_inj,
        W_SHARED=norm_weight.numel() == hidden_size,
        EPS=eps,
        BLOCK=NORM_BLOCK,
        launch_pdl=pdl,
    )
    if _is_packed(down_weight):
        _compressed().hc_down(xn, down_weight, acc, rows, width // splits, nout, nacc, bm, bn, splits, pdl)
    else:
        _hc_down_kernel[(triton.cdiv(rows, bm), triton.cdiv(nout, bn), splits)](
            xn,
            down_weight,
            acc,
            rows,
            width // splits,
            xn.stride(0),
            down_weight.stride(0),
            NOUT=nout,
            NACC=nacc,
            BM=bm,
            BN=bn,
            BK=bk,
            launch_pdl=pdl,
            num_warps=4 if bm * bn <= 128 * 64 else 8,
            num_stages=3 if bk == DOWN_BK else 4,
        )
    mixed, injection_out = up_mix(acc, nacc, up_weight, xn, hc_count, lora_rank, use_combine,
                                  bm_up, bh)
    return out, mixed, injection_out


def up_warps(bm_up, bh, hc_count):
    tile = bm_up * bh * hc_count
    if tile <= UP_TILE_SMALL:
        return 4
    return 16 if tile > UP_TILE_WIDE and bh >= max(UP_WIDTHS) else 8


def up_mix(source, stride_source, up_weight, xn, hc_count, lora_rank, use_combine, bm_up, bh, presilu=False,
           quad=None):
    rows = xn.shape[0]
    hidden_size = xn.shape[1] // hc_count
    mixed = xn.new_empty((rows, hidden_size))
    injection_out = xn.new_empty((rows, hc_count if use_combine else 0))
    if _is_packed(up_weight):
        _compressed().hc_up(source, stride_source, up_weight, xn, mixed, injection_out if use_combine else mixed,
                            rows, hidden_size, hc_count, lora_rank, use_combine, bm_up, bh,
                            up_warps(bm_up, bh, hc_count), _pdl())
        return mixed, injection_out
    if quad is not None and hc_count == 4:
        return _up_mix4(source, stride_source, up_weight, xn, mixed, injection_out, lora_rank, use_combine,
                        presilu, quad)
    _hc_up_mix_kernel[(triton.cdiv(rows, bm_up), hidden_size // bh)](
        source,
        up_weight,
        xn,
        mixed,
        injection_out if use_combine else mixed,
        rows,
        stride_source,
        up_weight.stride(0),
        xn.stride(0),
        mixed.stride(0),
        H=hidden_size,
        HC=hc_count,
        R=lora_rank,
        HAS_INJ=use_combine,
        PRESILU=presilu,
        BM=bm_up,
        BH=bh,
        BK=UP_BK,
        launch_pdl=_pdl(),
        num_warps=up_warps(bm_up, bh, hc_count),
        num_stages=3,
    )
    return mixed, injection_out


def _up_mix4(source, stride_source, up_weight, xn, mixed, injection_out, lora_rank, use_combine, presilu, quad):
    bm, bh, warps, stages, bk = quad
    rows = xn.shape[0]
    hidden_size = mixed.shape[1]
    _hc_up_mix4_kernel[(triton.cdiv(rows, bm), hidden_size // bh)](
        source,
        up_weight,
        xn,
        mixed,
        injection_out if use_combine else mixed,
        rows,
        stride_source,
        up_weight.stride(0),
        xn.stride(0),
        mixed.stride(0),
        H=hidden_size,
        HC=4,
        R=lora_rank,
        HAS_INJ=use_combine,
        PRESILU=presilu,
        BM=bm,
        BH=bh,
        BK=bk,
        launch_pdl=_pdl(),
        num_warps=warps,
        num_stages=stages,
    )
    return mixed, injection_out


def _down_linear(module):
    if module.use_combine:
        return module.input_mix_weight_down_block_inject
    return module.input_mix_weight_down


def split(module, hidden, block, injection, arguments, pick):
    from vllm.models.qwen4_exp.nvidia.ops.hc import grouped_gemma_rmsnorm, hc_combine_norm

    norm_weight, _, up_weight, eps, hc_count, lora_rank, use_combine = arguments
    if block is None:
        out = None
        xn = grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc_count)
    else:
        out, xn = hc_combine_norm(hidden, block, injection, norm_weight, eps, hc_count)
    down = _down_linear(module)(xn)
    mixed, injection_out = up_mix(down, down.stride(0), up_weight, xn, hc_count, lora_rank,
                                  use_combine, pick[1], pick[2])
    return out, mixed, injection_out


def mixed_f1(rows):
    return BAND_F1 is not None and BAND_F1[0] <= rows <= BAND_F1[1]


def mixed_f2(rows):
    return BAND_F2 is not None and BAND_F2[0] <= rows <= BAND_F2[1]


def mixed_active(rows):
    return mixed_f1(rows) or mixed_f2(rows)


def mixed_tiles(pick, hc_count):
    _, f1, f2, bm, bn, splits, bm_up, bh = pick
    down = f"{bm}x{bn}/splits{splits}" if f2 else "stock"
    if not f1:
        return down, "stock"
    if MIXED_UP4 is not None and hc_count == 4:
        up_bm, up_bh, warps, stages, bk = MIXED_UP4
        return down, f"{up_bm}x{up_bh}/quad/w{warps}s{stages}k{bk}"
    return down, f"{bm_up}x{bh}/w{up_warps(bm_up, bh, hc_count)}"


def mixed_pick(rows, width):
    if FORCED_MIXED is not None:
        bm, bn, splits, bm_up, bh = FORCED_MIXED
    else:
        bm_up, bh = MIXED_UP
        bm, bn, splits = MIXED_DOWN[0]
        for shape in MIXED_DOWN:
            if shape[2] == 1 and triton.cdiv(rows, shape[0]) * triton.cdiv(LORA_HINT, shape[1]) >= 0.8 * _sms():
                bm, bn, splits = shape
                break
    while splits > 1 and width % (DOWN_BK * splits):
        splits -= 1
    return (MIXED, int(mixed_f1(rows)), int(mixed_f2(rows)), bm, bn, splits, bm_up, bh)


def _mixed_nacc(lora_rank, hc_count):
    return triton.cdiv(lora_rank + hc_count, 16) * 16


def _mixed_buffers(device, nacc):
    held = _STATE.get("mixed")
    if held is None or held[0].device != device:
        if _capturing():
            raise RuntimeError("the mixed workspace must exist before capture")
        top = BAND_F2[1] if BAND_F2 else WIDE_LIMIT
        held = (torch.empty(top * nacc * MIXED_MAX_SPLITS, device=device, dtype=torch.float32),
                torch.zeros(MIXED_COUNTERS, device=device, dtype=torch.int32))
        _STATE["mixed"] = held
    return held


def mixed_down(xn, down_weight, lora_rank, hc_count, use_combine, bm, bn, splits, silu):
    rows, width = xn.shape
    nout = lora_rank + (hc_count if use_combine else 0)
    nacc = triton.cdiv(nout, 16) * 16
    m_tiles, n_tiles = triton.cdiv(rows, bm), triton.cdiv(nout, bn)
    if m_tiles * n_tiles > MIXED_COUNTERS or width % (DOWN_BK * splits) or splits > MIXED_MAX_SPLITS:
        raise ValueError("mixed shape does not fit")
    if splits > 1:
        part, counters = _mixed_buffers(xn.device, _mixed_nacc(lora_rank, hc_count))
        if rows * nacc * splits > part.numel():
            raise ValueError("mixed workspace too small")
    else:
        part, counters = xn, xn
    lo = xn.new_empty((rows, nacc))
    big = bm * bn > 64 * 64
    bk = DOWN_BK // 2 if big else DOWN_BK
    _hc_mixed_down_kernel[(m_tiles, n_tiles, splits)](
        xn,
        down_weight,
        part,
        counters,
        lo,
        rows,
        width // splits,
        xn.stride(0),
        down_weight.stride(0),
        lo.stride(0),
        NOUT=nout,
        NACC=nacc,
        R=lora_rank,
        HC=hc_count,
        SPLITS=splits,
        SILU=silu,
        BM=bm,
        BN=bn,
        BK=bk,
        launch_pdl=_pdl(),
        num_warps=8 if big else 4,
        num_stages=4 if big else 3,
    )
    return lo


def mixed_chain(module, hidden, block, injection, arguments, pick):
    from vllm.models.qwen4_exp.nvidia.ops.hc import (grouped_gemma_rmsnorm, hc_combine_norm, hc_gate_mix,
                                                      hc_silu)

    norm_weight, down_weight, up_weight, eps, hc_count, lora_rank, use_combine = arguments
    _, f1, f2, bm, bn, splits, bm_up, bh = pick
    if block is None:
        out = None
        xn = grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc_count)
    else:
        out, xn = hc_combine_norm(hidden, block, injection, norm_weight, eps, hc_count)
    if f2:
        lo = mixed_down(xn, down_weight, lora_rank, hc_count, use_combine, bm, bn, splits, bool(f1))
    else:
        lo = _down_linear(module)(xn)
    if f1:
        mixed, injection_out = up_mix(lo, lo.stride(0), up_weight, xn, hc_count, lora_rank, use_combine,
                                      bm_up, bh, presilu=bool(f2), quad=MIXED_UP4)
        return out, mixed, injection_out
    lora = hc_silu(lo[:, :lora_rank], hc_count)
    gate = module.input_mix_weight_up(lora)
    mixed = hc_gate_mix(xn, gate, hc_count)
    if use_combine:
        injection_out = lo[:, lora_rank:lora_rank + hc_count].contiguous()
    else:
        injection_out = xn.new_empty((hidden.shape[0], 0))
    return out, mixed, injection_out


def decode_active(rows):
    if BAND is None or not BAND[0] <= rows <= BAND[1]:
        return False
    return DECODE_DOWN != "cublas" or rows >= DECODE_MIN_ROWS


def decode_pick(rows, width):
    if FORCED_SHAPE is not None:
        return (DECODE, *FORCED_SHAPE)
    bm = DECODE_BM_SMALL if rows <= DECODE_BM_SWITCH else DECODE_BM_LARGE
    return (DECODE, bm, DOWN_BN, split_for(rows, width, bm, DOWN_BN), *DECODE_UP)


def _decode_buffers(device, rows, nacc):
    held = _STATE.get("decode")
    need = max(rows, BAND[1] if BAND else 0, 1) * nacc * DECODE_MAX_SPLITS
    if held is None or held[0].device != device or held[0].numel() < need:
        if _capturing():
            raise RuntimeError("the decode workspace must exist before capture")
        held = (torch.empty(need, device=device, dtype=torch.float32),
                torch.zeros(DECODE_COUNTERS, device=device, dtype=torch.int32))
        _STATE["decode"] = held
    return held


def decode_down(xn, down_weight, lora_rank, hc_count, use_combine, pick):
    _, bm, bn, splits, _, _ = pick
    rows, width = xn.shape
    nout = lora_rank + (hc_count if use_combine else 0)
    nacc = triton.cdiv(nout, 16) * 16
    m_tiles, n_tiles = triton.cdiv(rows, bm), triton.cdiv(nout, bn)
    if m_tiles * n_tiles > DECODE_COUNTERS or width % (DOWN_BK * splits):
        raise ValueError("decode shape does not fit")
    part, counters = _decode_buffers(xn.device, rows, nacc)
    lo = xn.new_empty((rows, nacc))
    bk = DOWN_BK if bm * bn <= 64 * 64 else DOWN_BK // 2
    _hc_decode_down_kernel[(m_tiles, n_tiles, splits)](
        xn,
        down_weight,
        part,
        counters,
        lo,
        rows,
        width // splits,
        xn.stride(0),
        down_weight.stride(0),
        lo.stride(0),
        NOUT=nout,
        NACC=nacc,
        R=lora_rank,
        HC=hc_count,
        SPLITS=splits,
        BM=bm,
        BN=bn,
        BK=bk,
        launch_pdl=_pdl(),
        num_warps=4 if bm * bn <= 64 * 64 else 8,
        num_stages=3 if bk == DOWN_BK else 4,
    )
    return lo


def decode_chain(hidden, block, injection, norm_weight, down_weight, up_weight, eps, hc_count,
                 lora_rank, use_combine, pick):
    from vllm.models.qwen4_exp.nvidia.ops.hc import grouped_gemma_rmsnorm, hc_combine_norm

    if block is None:
        out = None
        xn = grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc_count)
    else:
        out, xn = hc_combine_norm(hidden, block, injection, norm_weight, eps, hc_count)
    if DECODE_DOWN == "cublas":
        lo = torch.nn.functional.linear(xn, down_weight)
        presilu = False
    else:
        lo = decode_down(xn, down_weight, lora_rank, hc_count, use_combine, pick)
        presilu = True
    mixed, injection_out = up_mix(lo, lo.stride(0), up_weight, xn, hc_count, lora_rank, use_combine,
                                  pick[4], pick[5], presilu=presilu, quad=DECODE_UP4)
    return out, mixed, injection_out


def run(module, hidden, block, injection, arguments, pick):
    if pick[0] == DECODE:
        return decode_chain(hidden, block, injection, *arguments, pick)
    if pick[0] == MIXED:
        return mixed_chain(module, hidden, block, injection, arguments, pick)
    if pick[0] == SPLIT:
        return split(module, hidden, block, injection, arguments, pick)
    return fused(hidden, block, injection, *arguments, pick)


def _is_packed(weight):
    return getattr(weight, "esc_ptr", None) is not None


def _compressed():
    from vllm import taaf_bf16c

    return taaf_bf16c


def _held(linear, tensors):
    packed = getattr(linear, "taaf_bf16c", None)
    return linear.weight if packed is None or tensors else packed


def _down_weight(module, tensors=False):
    if module.use_combine:
        return _held(module.input_mix_weight_down_block_inject, tensors)
    return _held(module.input_mix_weight_down, tensors)


def _arguments(module, tensors=False):
    return (
        module.hc_norm.weight,
        _down_weight(module, tensors),
        _held(module.input_mix_weight_up, tensors),
        float(module.config.rms_norm_eps),
        int(module.hc_count),
        int(module.lora_rank),
        bool(module.use_combine),
    )


def valid_splits(width):
    return tuple(split for split in SPLITS if width % (DOWN_BK * split) == 0)


def _sms():
    if _STATE["sms"] is None:
        _STATE["sms"] = (torch.cuda.get_device_properties(torch.cuda.current_device())
                         .multi_processor_count if torch.cuda.is_available() else FALLBACK_SMS)
    return _STATE["sms"]


def split_for(rows, width, bm, bn):
    tiles = triton.cdiv(rows, bm) * triton.cdiv(LORA_HINT, bn)
    wanted = _sms() / tiles
    return min(valid_splits(width), key=lambda split: (abs(split - wanted), split))


def default_pick(rows, width):
    bm = row_block(rows)
    return (bm, DOWN_BN, split_for(rows, width, bm, DOWN_BN), bm, UP_WIDTHS[0])


def down_candidates(rows, width):
    small = row_block(rows)
    if rows <= small:
        shapes = [(small, DOWN_BN), (small, 2 * DOWN_BN)]
    else:
        shapes = [(small, DOWN_BN), (WIDE_ROWS, DOWN_BN), (small, 2 * DOWN_BN),
                  (WIDE_ROWS, 2 * DOWN_BN)]
    allowed = valid_splits(width)
    made = []
    for bm, bn in shapes:
        base = split_for(rows, width, bm, bn)
        made.append((bm, bn, base))
        if rows <= small and allowed.index(base) > 0:
            made.append((bm, bn, allowed[allowed.index(base) - 1]))
    return made


def up_candidates(rows):
    small = row_block(rows)
    made = [(small, width) for width in UP_WIDTHS]
    if rows > small:
        made.append((WIDE_ROWS, UP_WIDTHS[0]))
    return made


def split_candidates(rows):
    small = row_block(rows)
    made = [(SPLIT, small, width) for width in SPLIT_WIDTHS]
    if rows > small:
        made.append((SPLIT, WIDE_ROWS, UP_WIDTHS[0]))
    return made


def pick_key(rows):
    return rows if rows <= BASE_ROWS else ("wide", -(-rows // WIDE_BUCKET) * WIDE_BUCKET)


def check_key(name, rows):
    return name if rows <= BASE_ROWS else f"{name}@{pick_key(rows)[1]}"


def tunable(rows):
    if rows <= BASE_ROWS:
        return _STATE["tuned"] < TUNE_LIMIT
    return _STATE["wide_tuned"] < WIDE_TUNE_LIMIT


def refusal(hidden, block, injection, norm_weight, down_weight, up_weight, hc_count, lora_rank,
            use_combine):
    if not (hidden.is_cuda or INTERPRETED):
        return "device"
    if hidden.dim() != 2 or hidden.shape[0] < 1:
        return "rows"
    rows, width = hidden.shape
    if rows > MAX_ROWS and not decode_active(rows) and not mixed_active(rows):
        return f"rows>{MAX_ROWS}"
    if hc_count < 1 or hc_count & (hc_count - 1) or width % hc_count:
        return "streams"
    hidden_size = width // hc_count
    nout = lora_rank + (hc_count if use_combine else 0)
    if any(t.dtype != torch.bfloat16 for t in (hidden, down_weight, up_weight)):
        return "dtype"
    if decode_active(rows) and (_is_packed(up_weight) or _is_packed(down_weight)):
        return "decode-packed"
    if mixed_f2(rows) and _is_packed(down_weight):
        return "mixed-packed"
    if _is_packed(up_weight) and not _compressed().hc_supported(up_weight, hidden_size, min(UP_WIDTHS)):
        return "compressed-up-shape"
    if hidden.stride(1) != 1 or not (down_weight.is_contiguous() and up_weight.is_contiguous()
                                     and norm_weight.is_contiguous()):
        return "layout"
    if down_weight.shape[1] != width or down_weight.shape[0] < nout:
        return "down-shape"
    if tuple(up_weight.shape) != (width, lora_rank):
        return "up-shape"
    if norm_weight.numel() not in (hidden_size, width):
        return "norm-shape"
    if hidden_size % max(UP_WIDTHS) or not valid_splits(width) or hidden_size > 8192:
        return "width"
    if block is not None and (tuple(block.shape) != (rows, hidden_size) or block.stride(1) != 1
                              or block.dtype != hidden.dtype):
        return "block"
    if injection is not None and (tuple(injection.shape) != (rows, hc_count)
                                  or injection.stride(1) != 1):
        return "injection"
    return ""


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _error(got, want):
    if got.numel() == 0 and want.numel() == 0:
        return 0.0, 0.0
    got = got.detach().float()
    want = want.detach().float()
    if not bool(torch.isfinite(got).all()) or got.shape != want.shape:
        return float("inf"), float("inf")
    diff = (got - want).abs()
    scale = want.abs()
    worst = float(diff.max() / scale.max().clamp_min(1e-30))
    mean = float(diff.mean() / scale.mean().clamp_min(1e-30))
    return worst, mean


def _stock(module, kind, hidden, block, injection):
    if kind == "mix":
        _, mixed, inj = module._taaf_stock_mix(hidden)
        return None, mixed, inj
    return module._taaf_stock_combine_and_mix(hidden, block, injection)


def _compare(module, kind, hidden, block, injection, arguments, pick):
    want = _stock(module, kind, hidden, block, injection)
    got = run(module, hidden, block, injection, arguments, pick)
    rows = {}
    if want[0] is not None:
        rows["out"] = _error(got[0], want[0])
    rows["block"] = _error(got[1], want[1])
    if want[2] is not None:
        rows["inj"] = _error(got[2], want[2])
    return rows


def _kind_name(kind, block, injection, module):
    if kind == "mix":
        return "mix" if module.use_combine else "mix-final"
    if not module.use_combine:
        return "combine-final"
    return "combine-inj" if injection is not None else "combine-unit"


def _stood_aside(name, rows, why):
    if name.endswith("@" + MIXED):
        announce_mixed("stood-aside", name, kind=name, rows=rows, why=why)
    elif name.endswith("@" + DECODE_CUBLAS_TAG):
        announce_down("stood-aside", name, kind=name, rows=rows, why=why)


def _check(module, kind, hidden, block, injection, arguments, pick, checked=None, tolerance=None):
    name = checked or _kind_name(kind, block, injection, module)
    _CHECKED.add(name)
    try:
        errors = _compare(module, kind, hidden, block, injection, arguments, pick)
    except Exception as exc:
        _DISABLED.add(name)
        say("refuse", kind=name, rows=hidden.shape[0], error=f"{type(exc).__name__}:{exc}"[:300])
        _stood_aside(name, hidden.shape[0], f"{type(exc).__name__}:{exc}"[:300])
        return
    worst = max(value[0] for value in errors.values())
    say("check", kind=name, rows=hidden.shape[0], pick=_label(pick),
        **{f"{key}_maxrel": f"{value[0]:.2e}" for key, value in errors.items()},
        **{f"{key}_meanrel": f"{value[1]:.2e}" for key, value in errors.items()})
    limit = TOLERANCE if tolerance is None else tolerance
    if not worst <= limit:
        _DISABLED.add(name)
        say("refuse", kind=checked or name, rows=hidden.shape[0], maxrel=f"{worst:.2e}", tolerance=limit)
        _stood_aside(name, hidden.shape[0], f"maxrel {worst:.2e} above {limit:g}")


def _label(pick):
    if pick == STOCK:
        return STOCK
    if pick[0] == SPLIT:
        return f"split-m{pick[1]}h{pick[2]}"
    if pick[0] == DECODE and DECODE_DOWN == "cublas":
        tile = "old" if DECODE_UP4 is None else "q" + "-".join(map(str, DECODE_UP4))
        return f"decode-cublas-{tile}"
    if pick[0] == DECODE:
        return f"decode-m{pick[1]}n{pick[2]}k{pick[3]}-m{pick[4]}h{pick[5]}"
    if pick[0] == MIXED:
        return f"mixed-f{pick[1]}{pick[2]}-m{pick[3]}n{pick[4]}k{pick[5]}-m{pick[6]}h{pick[7]}"
    return f"m{pick[0]}n{pick[1]}k{pick[2]}-m{pick[3]}h{pick[4]}"


def _side_stream():
    if _STATE["stream"] is None:
        _STATE["stream"] = torch.cuda.Stream()
    return _STATE["stream"]


def _tuning_pool():
    if _STATE["pool"] is None:
        _STATE["pool"] = torch.cuda.graph_pool_handle()
    return _STATE["pool"]


def _graph_us(run):
    side = _side_stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        graph.capture_begin(pool=_tuning_pool(), capture_error_mode="thread_local")
        try:
            run()
        except BaseException:
            if _STATE["graph"] is None:
                _STATE["pool"] = None
            raise
        finally:
            graph.capture_end()
    _STATE["graph"] = graph
    torch.cuda.current_stream().wait_stream(side)
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    del graph
    return start.elapsed_time(end) * 1000.0 / REPLAYS


def _tune(module, kind, hidden, block, injection, arguments):
    rows, width = hidden.shape
    _STATE["tuned" if rows <= BASE_ROWS else "wide_tuned"] += 1
    started = time.monotonic()
    want = _stock(module, kind, hidden, block, injection)
    stock_us = _graph_us(lambda: _stock(module, kind, hidden, block, injection))
    timed = {}

    def attempt(pick):
        if pick in timed:
            return
        try:
            got = run(module, hidden, block, injection, arguments, pick)
            worst = max(_error(got[1], want[1])[0],
                        _error(got[2], want[2])[0] if want[2] is not None else 0.0)
            if not worst <= TOLERANCE:
                say("reject", rows=rows, pick=_label(pick), maxrel=f"{worst:.2e}")
                return
            timed[pick] = _graph_us(lambda: run(module, hidden, block, injection, arguments, pick))
        except Exception as exc:
            say("reject", rows=rows, pick=_label(pick), error=f"{type(exc).__name__}:{exc}"[:300])

    start = default_pick(rows, width)
    for down in down_candidates(rows, width):
        attempt((*down, *start[3:]))
    best = min(timed, key=timed.get) if timed else start
    for up in up_candidates(rows):
        attempt((*best[:3], *up))
    for pick in split_candidates(rows):
        attempt(pick)
    best = min(timed, key=timed.get) if timed else STOCK
    best_us = timed.get(best, float("inf"))
    chosen = best if best_us < stock_us else STOCK
    _PICKS[pick_key(rows)] = chosen
    say("tune", rows=rows, kind=_kind_name(kind, block, injection, module),
        stock_us=f"{stock_us:.1f}", fused_us=f"{best_us:.1f}", pick=_label(chosen),
        tried=",".join(f"{_label(p)}:{us:.1f}" for p, us in sorted(timed.items(), key=lambda row: row[1])),
        seconds=f"{time.monotonic() - started:.2f}")
    return chosen


def pick_space():
    held = {name: globals()[name] for name in PICK_SPACE}
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode("utf-8")).hexdigest()


def parse_label(label):
    if label == STOCK:
        return STOCK
    found = re.fullmatch(r"split-m(\d+)h(\d+)", label)
    if found:
        return (SPLIT, int(found.group(1)), int(found.group(2)))
    found = re.fullmatch(r"m(\d+)n(\d+)k(\d+)-m(\d+)h(\d+)", label)
    if found:
        return tuple(int(part) for part in found.groups())
    raise ValueError(f"unreadable pick {label!r}")


def seed_key(text):
    if text.startswith("wide:"):
        return ("wide", int(text[5:]))
    return int(text)


def _seed_picks(hidden):
    if _STATE["seeded"] is not None or not hidden.is_cuda:
        return
    _STATE["seeded"] = 0
    path = str(os.environ.get(PICKS_ENV) or "").strip()
    if not path:
        return
    try:
        held = json.loads(Path(path).read_text(encoding="utf-8"))
        device = torch.cuda.get_device_name(hidden.device)
        if held.get("space") != pick_space() or held.get("device") != device:
            say("seed-refused", path=Path(path).name, device=device,
                reason="space" if held.get("space") != pick_space() else "device")
            return
        picks = {seed_key(text): parse_label(label) for text, label in held["picks"].items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        say("seed-refused", path=Path(path).name, error=f"{type(exc).__name__}:{exc}"[:200])
        return
    for key, pick in picks.items():
        _PICKS.setdefault(key, pick)
    _STATE["seeded"] = len(picks)
    say("seeded", picks=len(picks), path=Path(path).name)


def _route_mixed(module, kind, hidden, block, injection, arguments, name):
    rows, width = hidden.shape
    checked = f"{name}@mixed"
    if checked in _DISABLED:
        return None
    capturing = _capturing()
    if capturing and checked not in _CHECKED:
        announce("stock", key=f"capture-{checked}", reason="unchecked-in-capture", kind=name, rows=rows)
        announce_mixed("stood-aside", checked, kind=checked, rows=rows, why="unchecked-in-capture")
        return None
    pick = mixed_pick(rows, width)
    if pick[2] and "mixed" not in _STATE and not capturing:
        _mixed_buffers(hidden.device, _mixed_nacc(arguments[5], arguments[4]))
    if checked not in _CHECKED and not capturing:
        _check(module, kind, hidden, block, injection, arguments, pick, checked, MIXED_TOLERANCE)
        if checked in _DISABLED:
            return None
    announce("fired", key=checked, kind=checked, rows=rows, pick=_label(pick), capturing=int(capturing),
             pdl=int(_pdl()), sites=len(_SITES))
    down, up = mixed_tiles(pick, arguments[4])
    announce_mixed("fired", (pick[1], pick[2]), f1=pick[1], f2=pick[2], kind=checked, rows=rows, down_tile=down,
                   up_tile=up, capturing=int(capturing), pdl=int(_pdl()))
    return pick


def _route_decode(module, kind, hidden, block, injection, arguments, name):
    rows, width = hidden.shape
    cublas = DECODE_DOWN == "cublas"
    checked = f"{name}@{DECODE_CUBLAS_TAG if cublas else DECODE}"
    if checked in _DISABLED:
        return None
    capturing = _capturing()
    if capturing and checked not in _CHECKED:
        announce("stock", key=f"capture-{checked}", reason="unchecked-in-capture", kind=name, rows=rows)
        if cublas:
            announce_down("stood-aside", checked, kind=checked, rows=rows, why="unchecked-in-capture")
        return None
    pick = decode_pick(rows, width)
    if checked not in _CHECKED and not capturing:
        _check(module, kind, hidden, block, injection, arguments, pick, checked, DECODE_TOLERANCE)
        if checked in _DISABLED:
            return None
    announce("fired", key=checked, kind=checked, rows=rows, pick=_label(pick), capturing=int(capturing),
             pdl=int(_pdl()), sites=len(_SITES))
    if cublas:
        announce_down("fired", (DECODE_UP4, DECODE_MIN_ROWS), kind=checked, rows=rows, down="cublas",
                      up_tile=decode_up_tile(arguments[4]), min_rows=DECODE_MIN_ROWS, capturing=int(capturing),
                      pdl=int(_pdl()))
    return pick


def _route(module, kind, hidden, block, injection, arguments):
    reason = refusal(hidden, block, injection, arguments[0], arguments[1], arguments[2],
                     arguments[4], arguments[5], arguments[6])
    if reason:
        announce("stock", key=reason, reason=reason, rows=hidden.shape[0] if hidden.dim() else 0)
        return None
    name = _kind_name(kind, block, injection, module)
    rows, width = hidden.shape
    if decode_active(rows):
        return _route_decode(module, kind, hidden, block, injection, arguments, name)
    if mixed_active(rows):
        return _route_mixed(module, kind, hidden, block, injection, arguments, name)
    checked = check_key(name, rows)
    if name in _DISABLED or checked in _DISABLED:
        return None
    capturing = _capturing()
    key = pick_key(rows)
    _seed_picks(hidden)
    pick = _PICKS.get(key)
    if capturing and (pick is None or checked not in _CHECKED):
        announce("stock", key=f"capture-{checked}", reason="unchecked-in-capture", kind=name, rows=rows)
        return None
    if pick is None:
        pick = default_pick(rows, width) if rows <= BASE_ROWS else STOCK
        if hidden.is_cuda and tunable(rows):
            try:
                pick = _tune(module, kind, hidden, block, injection, arguments)
            except Exception as exc:
                say("tune-failed", rows=rows, pick=_label(pick),
                    error=f"{type(exc).__name__}:{exc}"[:300])
        elif rows > BASE_ROWS:
            announce("stock", key=f"untuned-{checked}", reason="untuned-wide", kind=name, rows=rows)
        _PICKS[key] = pick
    if pick == STOCK:
        return None
    if checked not in _CHECKED and not capturing:
        _check(module, kind, hidden, block, injection, arguments, pick, checked)
        if checked in _DISABLED:
            return None
    announce("fired", key="" if rows <= BASE_ROWS else checked, kind=checked, rows=rows, pick=_label(pick),
             capturing=int(capturing), pdl=int(_pdl()), sites=len(_SITES))
    return pick


def _injection(injection, hidden, hc_count, use_combine):
    if use_combine and injection is not None:
        return injection.contiguous()
    return hidden.new_empty((hidden.shape[0], 0))


def _op_arguments(module, norm_weight, down_weight, up_weight, eps, hc_count, lora_rank, use_combine):
    if getattr(_down_linear(module), "taaf_bf16c", None) is None             and getattr(module.input_mix_weight_up, "taaf_bf16c", None) is None:
        return (norm_weight, down_weight, up_weight, eps, hc_count, lora_rank, use_combine)
    return (norm_weight, _down_weight(module), _held(module.input_mix_weight_up, False), eps, hc_count,
            lora_rank, use_combine)


def _mix_op(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    use_combine: bool,
    site: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _SITES[site]
    arguments = _op_arguments(module, norm_weight, down_weight, up_weight, eps, hc_count, lora_rank, use_combine)
    pick = _route(module, "mix", hidden, None, None, arguments)
    if pick is None:
        _, mixed, injection = module._taaf_stock_mix(hidden)
        return mixed, _injection(injection, hidden, hc_count, use_combine)
    _, mixed, injection = run(module, hidden, None, None, arguments, pick)
    return mixed, injection


def _mix_fake(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    use_combine: bool,
    site: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden.shape[0]
    return (hidden.new_empty((rows, hidden.shape[1] // hc_count)),
            hidden.new_empty((rows, hc_count if use_combine else 0)))


def _combine_mix_op(
    hidden: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor | None,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    use_combine: bool,
    site: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _SITES[site]
    arguments = _op_arguments(module, norm_weight, down_weight, up_weight, eps, hc_count, lora_rank, use_combine)
    pick = _route(module, "combine", hidden, block, injection, arguments)
    if pick is None:
        out, mixed, made = module._taaf_stock_combine_and_mix(hidden, block, injection)
        return out, mixed, _injection(made, hidden, hc_count, use_combine)
    return run(module, hidden, block, injection, arguments, pick)


def _combine_mix_fake(
    hidden: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor | None,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    lora_rank: int,
    use_combine: bool,
    site: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = hidden.shape[0]
    return (hidden.new_empty(hidden.shape),
            hidden.new_empty((rows, hidden.shape[1] // hc_count)),
            hidden.new_empty((rows, hc_count if use_combine else 0)))


direct_register_custom_op(
    op_name="taaf_hc_fused_mix",
    op_func=_mix_op,
    fake_impl=_mix_fake,
)


direct_register_custom_op(
    op_name="taaf_hc_fused_combine_mix",
    op_func=_combine_mix_op,
    fake_impl=_combine_mix_fake,
)


def register(module, prefix):
    base = prefix or "hyper_connection"
    name, copy = base, 1
    while name in _SITES:
        copy += 1
        name = f"{base}#{copy}"
    _SITES[name] = module
    if len(_SITES) == 1:
        say("loaded", first_site=name, max_rows=MAX_ROWS, tune_limit=TUNE_LIMIT,
            tolerance=TOLERANCE, interpreted=int(INTERPRETED),
            **({"decode_band": f"{BAND[0]}-{BAND[1]}"} if BAND else {}),
            **({"decode_down": DECODE_DOWN, "decode_min_rows": DECODE_MIN_ROWS} if BAND and DECODE_DOWN == "cublas"
               else {}),
            **({"mixed_f1": f"{BAND_F1[0]}-{BAND_F1[1]}"} if BAND_F1 else {}),
            **({"mixed_f2": f"{BAND_F2[0]}-{BAND_F2[1]}"} if BAND_F2 else {}),
            **({"wide_bucket": WIDE_BUCKET, "wide_tune_limit": WIDE_TUNE_LIMIT} if MAX_ROWS > BASE_ROWS else {}))
    return name


def _direct():
    compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return not (compiling is not None and compiling())


def _off():
    if ON:
        return False
    announce("stock", key="off", reason="off")
    return True


def mix(module, hidden_states):
    if _off():
        return module._taaf_stock_mix(hidden_states)
    if _direct():
        arguments = _arguments(module)
        pick = _route(module, "mix", hidden_states, None, None, arguments)
        if pick is None:
            return module._taaf_stock_mix(hidden_states)
        _, mixed, injection = run(module, hidden_states, None, None, arguments, pick)
        return hidden_states, mixed, injection if module.use_combine else None
    mixed, injection = torch.ops.vllm.taaf_hc_fused_mix(
        hidden_states, *_arguments(module, True), module._taaf_hc_site)
    return hidden_states, mixed, injection if module.use_combine else None


def combine_and_mix(module, hidden_states, prev_block_output, prev_injection):
    if prev_block_output is None or _off():
        return module._taaf_stock_combine_and_mix(hidden_states, prev_block_output, prev_injection)
    if _direct():
        arguments = _arguments(module)
        pick = _route(module, "combine", hidden_states, prev_block_output, prev_injection, arguments)
        if pick is None:
            return module._taaf_stock_combine_and_mix(hidden_states, prev_block_output,
                                                      prev_injection)
        out, mixed, injection = run(module, hidden_states, prev_block_output, prev_injection,
                                    arguments, pick)
        return out, mixed, injection if module.use_combine else None
    out, mixed, injection = torch.ops.vllm.taaf_hc_fused_combine_mix(
        hidden_states, prev_block_output, prev_injection, *_arguments(module, True),
        module._taaf_hc_site)
    return out, mixed, injection if module.use_combine else None
