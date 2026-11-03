import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

ENV = "TAAF_VLLM_BF16C"
GROUPS_ENV = "TAAF_VLLM_BF16C_GROUPS"
ROWS_ENV = "TAAF_VLLM_BF16C_ROWS"
FAST_ENV = "TAAF_VLLM_BF16C_FAST2"
FAST_VERDICT_ENV = "TAAF_VLLM_BF16C_FAST2_VERDICT"
PIPE_ENV = "TAAF_VLLM_BF16C_PIPE"
PIPE_ELEMENTS_MAX = 16384 * 2560
MIX_ENV = "TAAF_VLLM_BF16C_MIX"
VERDICT_WAIT_ENV = "TAAF_VLLM_BF16C_VERDICT_WAIT_S"
MIX_VERDICT_ENV = "TAAF_VLLM_BF16C_MIX_VERDICT"
MIX_DECODE_ENV = "TAAF_VLLM_BF16C_MIX_DECODE"
MIX_DECODE_BAND = (512, 400)
MIX_EB = 32
MIX_COUNTERS = 1 << 16
MIX_BANDS = ((960, 640), (1728, 1280), (1 << 30, 2176))
MIX_TILES = ((128, 128, 8, 3), (128, 128, 8, 2), (128, 128, 16, 3), (256, 64, 8, 2), (256, 64, 8, 3),
             (128, 64, 8, 4), (128, 64, 4, 3), (64, 128, 4, 3), (64, 64, 4, 4), (64, 32, 4, 4))
MIX_SPLITS = (1, 2, 3, 4, 6, 8, 12)
MIX_MARGIN = 1.03
MIX_OVERFLOW_ROWS = 3
GROUP_PATTERNS = {
    "attn": r"(?:^|\.)model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)$",
    "gdn": r"(?:^|\.)model\.layers\.\d+\.linear_attn\.(?:in_proj_qkvz|in_proj_ba|in_proj_qkv|out_proj)$",
    "shared": r"(?:^|\.)model\.layers\.\d+\.mlp\.shared_expert\.(?:gate_up_proj|down_proj)$",
    "hc": (r"(?:^|\.)model\.(?:layers\.\d+\.(?:attn|mlp)_hyper_connection|hyper_connection_mixer)"
           r"\.input_mix_weight_(?:down_block_inject|down|up)$"),
    "mtp": (r"(?:^|\.)mtp\.(?:fc_embedding|fc_hidden|layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|"
            r"mlp\.shared_expert\.(?:gate_up_proj|down_proj)))$"),
    "vision": r"(?:^|\.)visual\.",
    "head": r"^lm_head$",
}
DEFAULT_GROUPS = ("attn", "gdn", "shared", "hc", "head")
GROUP = 128
HALF = GROUP // 2
ROW_TILE = 16
WINDOW = 15
DECODE_ROWS = 64
ROWS_MAX = 2048
DECODE_ALIGN = 64
GEMM_BK = 64
GEMM_BMS = (16, 32, 64, 128, 256)
GEMM_BNS = (32, 64, 128)
GEMM_TILE_LIMIT = 256 * 64
PREFETCH_LIMIT = 1024
DECODE_BNS = (32, 64, 128)
HC_BK = 32
HC_UP_BK = 32
TUNE_WARMUP_MS = 3
TUNE_REP_MS = 12
FAST_ROWS = 256
FAST_ROWS_MAX = 4096
MIXED_ROWS = (640, 1280, 2176)
FAST_BNS = (32, 64, 128)
FAST_SPLITS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)
FAST_TILE_LIMIT = 160 * 128
SERVED_SHAPES = ((16384, 2560), (13312, 2560), (2560, 6144), (96, 2560), (1280, 2560), (2560, 640), (336, 10240),
                 (320, 10240), (10240, 320), (248320, 2560))
CHECK_ROWS = (1, 40, 152, 256, 640)
TIMED_ROWS = 152
CHECK_ESCAPES = 1.2e-4
NAMES = ("sm", "ex", "base", "esc_ptr", "esc_n", "esc_k", "esc_v")
_HALF = tl.constexpr(HALF)
_GROUP = tl.constexpr(GROUP)
_RT = tl.constexpr(ROW_TILE)
ESC_CHUNK_GROUPS = 65536
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
_ASM = tl.constexpr(not INTERPRETED)
_PTX = tl.constexpr("{\n.reg .b32 c, t, m, e, h, l;\n"
                    "shr.b32 c, $3, $5;\n"
                    "lop3.b32 t, c, 0x0F0F0F0F, 0x80808080, 0xEA;\n"
                    "sub.u32 t, t, 0x01010101;\n"
                    "prmt.b32 m, t, 0, 0xBA98;\n"
                    "lop3.b32 e, t, m, 0x7F7F7F7F, 0x80;\n"
                    "add.u32 e, e, $4;\n"
                    "and.b32 e, e, m;\n"
                    "shr.b32 h, e, 1;\n"
                    "lop3.b32 h, h, $2, 0x80808080, 0xD8;\n"
                    "shl.b32 l, e, 7;\n"
                    "lop3.b32 l, l, $2, 0x80808080, 0xE4;\n"
                    "prmt.b32 $0, l, h, 0x5140;\n"
                    "prmt.b32 $1, l, h, 0x7362;\n}")
_MIX_PTX = tl.constexpr("{\n.reg .b32 c, t, m, e, h, l, x, z;\n"
                        "shr.b32 c, $3, $8;\n"
                        "lop3.b32 t, c, 0x0F0F0F0F, 0x80808080, 0xEA;\n"
                        "sub.u32 t, t, 0x01010101;\n"
                        "prmt.b32 m, t, 0, 0xBA98;\n"
                        "lop3.b32 e, t, m, 0x7F7F7F7F, 0x80;\n"
                        "add.u32 e, e, $4;\n"
                        "and.b32 e, e, m;\n"
                        "xor.b32 x, $5, $6;\n"
                        "and.b32 z, x, 0x7F7F7F7F;\n"
                        "add.u32 z, z, 0x7F7F7F7F;\n"
                        "lop3.b32 z, z, x, 0x80808080, 0x02;\n"
                        "prmt.b32 z, z, 0, 0xBA98;\n"
                        "lop3.b32 e, e, $7, z, 0xF8;\n"
                        "shr.b32 h, e, 1;\n"
                        "lop3.b32 h, h, $2, 0x80808080, 0xD8;\n"
                        "shl.b32 l, e, 7;\n"
                        "lop3.b32 l, l, $2, 0x80808080, 0xE4;\n"
                        "prmt.b32 $0, l, h, 0x5140;\n"
                        "prmt.b32 $1, l, h, 0x7362;\n}")
_TUNED = {}
_SPLITS = {}
_STATE = {"layers": 0, "orig": 0, "packed": 0, "escapes": 0, "weights": 0, "seconds": 0.0, "told": False,
          "sms": {}, "untuned": set()}
_PATTERNS = []
_PIPE = {"slots": [None, None], "numel": 0, "side": None, "freed": [None, None], "pending": None,
         "succ": {}, "seen": set(), "last": None, "held": {}}
_FAST = {"verified": None, "counters": {}, "workspace": {}, "retired": [], "told": set()}
_MIX = {"verified": None, "index": {}, "counters": {}, "workspace": {}, "retired": [], "told": set(), "layers": 0,
        "bytes": 0}


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def groups():
    said = str(os.environ.get(GROUPS_ENV) or "").strip().lower()
    named = [part.strip() for part in said.split(",") if part.strip()] if said else list(DEFAULT_GROUPS)
    unknown = sorted(set(named) - set(GROUP_PATTERNS))
    if unknown or len(set(named)) != len(named):
        raise ValueError(f"{GROUPS_ENV} must name distinct groups from {sorted(GROUP_PATTERNS)}, found {said!r}")
    return [name for name in GROUP_PATTERNS if name in named]


def pipe_enabled():
    return str(os.environ.get(PIPE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def decode_rows():
    said = str(os.environ.get(ROWS_ENV) or "").strip()
    if not said:
        return DECODE_ROWS
    if not said.isdigit() or not 1 <= int(said) <= ROWS_MAX:
        raise ValueError(f"{ROWS_ENV} must be an integer from 1 through {ROWS_MAX}, found {said!r}")
    return int(said)


def group_of(layer):
    if not _PATTERNS:
        for name in groups():
            _PATTERNS.append((name, re.compile(GROUP_PATTERNS[name])))
    prefix = getattr(layer, "prefix", "") or ""
    if type(layer).__name__ == "ParallelLMHead":
        prefix = prefix or "lm_head"
    for name, pattern in _PATTERNS:
        if pattern.search(prefix):
            return name
    return None


class Packed:
    dtype = torch.bfloat16

    def __init__(self, shape, sm, ex, base, esc_ptr, esc_n, esc_k, esc_v):
        self.shape = tuple(shape)
        self.sm, self.ex, self.base = sm, ex, base
        self.esc_ptr, self.esc_n, self.esc_k, self.esc_v = esc_ptr, esc_n, esc_k, esc_v

    @property
    def n(self):
        return self.shape[0]

    @property
    def k(self):
        return self.shape[1]

    @property
    def kb(self):
        return -(-self.k // GROUP)

    @property
    def nt(self):
        return -(-self.n // ROW_TILE)

    @property
    def device(self):
        return self.sm.device

    def tensors(self):
        return (self.sm, self.ex, self.base, self.esc_ptr, self.esc_n, self.esc_k, self.esc_v)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def escapes(self):
        return int(self.esc_n.numel())

    def is_contiguous(self):
        return True

    def dim(self):
        return 2

    def stride(self, dim):
        return self.k if dim == 0 else 1


def _window_base(e):
    groups_, _ = e.shape
    counts = torch.zeros(groups_, 256, dtype=torch.int32, device=e.device)
    counts.scatter_add_(1, e, torch.ones_like(e, dtype=torch.int32))
    counts[:, 0] = 0
    cs = counts.cumsum(1)
    return (cs[:, WINDOW:256] - cs[:, 0:256 - WINDOW]).argmax(1) + 1


def pack(weight):
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise ValueError(f"pack takes a 2-D bfloat16 weight, found {tuple(weight.shape)} {weight.dtype}")
    n, k = weight.shape
    kb = -(-k // GROUP)
    kpad = kb * GROUP
    nt = -(-n // ROW_TILE)
    device = weight.device
    sm = torch.empty((n, k), dtype=torch.uint8, device=device)
    ex = torch.empty((n, kb * HALF), dtype=torch.uint8, device=device)
    base = torch.empty((n, kb), dtype=torch.uint8, device=device)
    rows_per = max(ROW_TILE, (ESC_CHUNK_GROUPS // kb) // ROW_TILE * ROW_TILE)
    found = []
    for r0 in range(0, n, rows_per):
        r1 = min(n, r0 + rows_per)
        rows = r1 - r0
        bits = torch.zeros((rows, kpad), dtype=torch.int32, device=device)
        bits[:, :k] = weight[r0:r1].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        e = ((bits >> 7) & 0xFF).to(torch.int64)
        chosen = _window_base(e.reshape(rows * kb, GROUP)).reshape(rows, kb)
        d = e - chosen.repeat_interleave(GROUP, dim=1) + 1
        inside = (e != 0) & (d >= 1) & (d <= WINDOW)
        code = torch.where(inside, d, torch.zeros_like(d)).reshape(rows, kb, 2, HALF)
        escaped = (e != 0) & ~inside
        sm[r0:r1] = (((bits >> 8) & 0x80) | (bits & 0x7F))[:, :k].to(torch.uint8)
        ex[r0:r1] = (code[:, :, 0] | (code[:, :, 1] << 4)).reshape(rows, kb * HALF).to(torch.uint8)
        base[r0:r1] = chosen.to(torch.uint8)
        hit = escaped.nonzero()
        if hit.numel():
            r, c = hit[:, 0], hit[:, 1]
            found.append((r + r0, c, bits[r, c].to(torch.int64)))
    if found:
        rr = torch.cat([f[0] for f in found])
        cc = torch.cat([f[1] for f in found])
        vv = torch.cat([f[2] for f in found])
    else:
        rr = cc = vv = torch.empty(0, dtype=torch.int64, device=device)
    tile = (cc // GROUP) * nt + rr // ROW_TILE
    order = torch.argsort(((tile * ROW_TILE + rr % ROW_TILE) * GROUP) + cc % GROUP)
    rr, cc, vv, tile = rr[order], cc[order], vv[order], tile[order]
    ptr = torch.zeros(kb * nt + 1, dtype=torch.int32, device=device)
    if tile.numel():
        ptr[1:] = torch.bincount(tile, minlength=kb * nt).cumsum(0).to(torch.int32)
    vv16 = torch.where(vv > 32767, vv - 65536, vv).to(torch.int16)
    return Packed((n, k), sm, ex, base, ptr, rr.to(torch.int32), (cc % GROUP).to(torch.uint8), vv16)


def unpack(packed):
    n, k = packed.shape
    kb = packed.kb
    ex = packed.ex.to(torch.int32).reshape(n, kb, HALF)
    code = torch.stack([ex & 15, ex >> 4], dim=2).reshape(n, kb * GROUP)[:, :k]
    byte = packed.sm.to(torch.int32)
    base = packed.base.to(torch.int32).repeat_interleave(GROUP, dim=1)[:, :k]
    e = torch.where(code == 0, torch.zeros_like(code), base + code - 1)
    bits = ((byte & 0x80) << 8) | (e << 7) | (byte & 0x7F)
    if packed.escapes():
        counts = packed.esc_ptr[1:].to(torch.int64) - packed.esc_ptr[:-1].to(torch.int64)
        tile = torch.repeat_interleave(torch.arange(counts.numel(), device=bits.device), counts)
        cols = (tile // packed.nt) * GROUP + packed.esc_k.to(torch.int64)
        bits[packed.esc_n.to(torch.int64), cols] = packed.esc_v.to(torch.int32) & 0xFFFF
    return torch.where(bits > 32767, bits - 65536, bits).to(torch.int16).view(torch.bfloat16)


def bits_of(weight):
    return weight.contiguous().view(torch.int16)


@triton.jit
def _bits(b, cd, base, shift):
    if _ASM:
        sh = tl.full(b.shape, 0, tl.int32) + shift
        w = tl.inline_asm_elementwise(_PTX, "=r,=r,r,r,r,r,r,r,r", [b, cd, base, sh], dtype=tl.bfloat16,
                                      is_pure=True, pack=4)
    else:
        bi = b.to(tl.int32)
        c = (cd.to(tl.int32) >> shift) & 15
        e = tl.where(c == 0, 0, base.to(tl.int32) + c - 1)
        w = (((bi & 128) << 8) | (e << 7) | (bi & 127)).to(tl.int16).to(tl.bfloat16, bitcast=True)
    return w


@triton.jit
def _raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, col0, K, NB, BK: tl.constexpr, EVEN_K: tl.constexpr):
    col0 = tl.multiple_of(col0, BK)
    g = col0 // _GROUP
    cols = col0 + tl.arange(0, BK)
    rows = offs_n.to(tl.int64)[None, :]
    if EVEN_K:
        b = tl.load(sm_ptr + rows * K + cols[:, None], mask=mask_n[None, :], other=0)
    else:
        b = tl.load(sm_ptr + rows * K + cols[:, None], mask=mask_n[None, :] & (cols < K)[:, None], other=0)
    at = tl.multiple_of(g * _HALF + col0 % _HALF, BK)
    cd = tl.load(ex_ptr + rows * tl.multiple_of(NB * _HALF, _HALF) + (at + tl.arange(0, BK))[:, None],
                 mask=mask_n[None, :], other=0)
    base = tl.load(base_ptr + offs_n.to(tl.int64) * NB + g, mask=mask_n, other=0)
    return b, cd, base


@triton.jit
def _cook(b, cd, base, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, col0, NT, t0, t1, BK: tl.constexpr):
    col0 = tl.multiple_of(col0, BK)
    g = col0 // _GROUP
    local = col0 % _GROUP
    w = _bits(b, cd, base[None, :], (local // _HALF) * 4)
    e0 = tl.load(esc_ptr + g * NT + t0)
    e1 = tl.load(esc_ptr + g * NT + t1)
    lc = local + tl.arange(0, BK)
    for e in range(e0, e1):
        en = tl.load(esc_n_ptr + e)
        ek = tl.load(esc_k_ptr + e).to(tl.int32)
        ev = tl.load(esc_v_ptr + e).to(tl.bfloat16, bitcast=True)
        w = tl.where((offs_n[None, :] == en) & (lc[:, None] == ek), ev, w)
    return w


@triton.jit
def _tile(sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, mask_n, col0, K, NB, NT,
          t0, t1, BK: tl.constexpr, EVEN_K: tl.constexpr):
    b, cd, base = _raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, col0, K, NB, BK, EVEN_K)
    return _cook(b, cd, base, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, col0, NT, t0, t1, BK)


@triton.jit
def _rows(xrow, mask_m, col0, K, BK: tl.constexpr, EVEN_K: tl.constexpr):
    cols = tl.multiple_of(col0, BK) + tl.arange(0, BK)
    if EVEN_K:
        x = tl.load(xrow + cols[None, :], mask=mask_m[:, None], other=0.0)
    else:
        x = tl.load(xrow + cols[None, :], mask=mask_m[:, None] & (cols < K)[None, :], other=0.0)
    return x


@triton.jit
def _decode_kernel(out_ptr, sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr,
                   N, K, NB, NT, row0, RB, BN: tl.constexpr, EVEN_K: tl.constexpr):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    offs_n = row0 + pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    t0 = (row0 + pid_n * BN) // _RT
    t1 = tl.minimum(t0 + BN // _RT, NT)
    lo = g * _GROUP
    b, cd, base = _raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, lo, K, NB, _HALF, EVEN_K)
    hi = lo + _HALF + tl.arange(0, _HALF)
    rows = offs_n.to(tl.int64)[None, :]
    if EVEN_K:
        bh = tl.load(sm_ptr + rows * K + hi[:, None], mask=mask_n[None, :], other=0)
    else:
        bh = tl.load(sm_ptr + rows * K + hi[:, None], mask=mask_n[None, :] & (hi < K)[:, None], other=0)
    w = _bits(b, cd, base[None, :], 0)
    wh = _bits(bh, cd, base[None, :], 4)
    e0 = tl.load(esc_ptr + g * NT + t0)
    e1 = tl.load(esc_ptr + g * NT + t1)
    lc = tl.arange(0, _HALF)
    for e in range(e0, e1):
        en = tl.load(esc_n_ptr + e)
        ek = tl.load(esc_k_ptr + e).to(tl.int32)
        ev = tl.load(esc_v_ptr + e).to(tl.bfloat16, bitcast=True)
        row = offs_n[None, :] == en
        w = tl.where(row & (lc[:, None] == ek), ev, w)
        wh = tl.where(row & ((lc + _HALF)[:, None] == ek), ev, wh)
    dst = out_ptr + (offs_n - row0).to(tl.int64)[None, :] * K
    cols = lo + lc
    if EVEN_K:
        tl.store(dst + cols[:, None], w, mask=mask_n[None, :])
        tl.store(dst + hi[:, None], wh, mask=mask_n[None, :])
    else:
        tl.store(dst + cols[:, None], w, mask=mask_n[None, :] & (cols < K)[:, None])
        tl.store(dst + hi[:, None], wh, mask=mask_n[None, :] & (hi < K)[:, None])


@triton.jit(do_not_specialize=["M", "N", "NB", "NT", "MB"])
def _gemm_kernel(x_ptr, out_ptr, ws_ptr, sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr,
                 M, N, K, stride_xm, stride_om, NB, NT, MB,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PF: tl.constexpr, SW: tl.constexpr,
                 EVEN_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    splits = tl.num_programs(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    t0 = pid_n * (BN // _RT)
    t1 = tl.minimum(t0 + BN // _RT, NT)
    chunks = tl.cdiv(K, BK)
    per = tl.cdiv(chunks, splits)
    c_lo = pid_s * per
    c_hi = tl.minimum(c_lo + per, chunks)
    xrow = x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if PF:
        at = tl.minimum(c_lo, chunks - 1) * BK
        xv = _rows(xrow, mask_m, at, K, BK, EVEN_K)
        b, cd, base = _raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, at, K, NB, BK, EVEN_K)
        for c in range(c_lo, c_hi):
            nxt = tl.minimum(c + 1, chunks - 1) * BK
            xn = _rows(xrow, mask_m, nxt, K, BK, EVEN_K)
            bn, cdn, basen = _raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, nxt, K, NB, BK, EVEN_K)
            w = _cook(b, cd, base, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, c * BK, NT, t0, t1, BK)
            acc = tl.dot(xv, w, acc)
            xv = xn
            b = bn
            cd = cdn
            base = basen
    else:
        for c in range(c_lo, c_hi):
            xv = _rows(xrow, mask_m, c * BK, K, BK, EVEN_K)
            w = _tile(sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, mask_n, c * BK, K,
                      NB, NT, t0, t1, BK, EVEN_K)
            acc = tl.dot(xv, w, acc)
    mask = mask_m[:, None] & mask_n[None, :]
    if splits > 1:
        tl.store(ws_ptr + pid_s.to(tl.int64) * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)
    else:
        tl.store(out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fast_raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, g, K, NB, EVEN_K: tl.constexpr):
    words = tl.arange(0, _HALF // 4)
    rows = offs_n.to(tl.int64)[:, None]
    sm32 = sm_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
    ex32 = ex_ptr.to(tl.pointer_type(tl.int32), bitcast=True)
    at = rows * (K // 4) + (g * (_GROUP // 4) + words)[None, :]
    if EVEN_K:
        bl = tl.load(sm32 + at, mask=mask_n[:, None], other=0)
        bh = tl.load(sm32 + at + _HALF // 4, mask=mask_n[:, None], other=0)
    else:
        cl = g * _GROUP + 4 * words
        bl = tl.load(sm32 + at, mask=mask_n[:, None] & (cl < K)[None, :], other=0)
        bh = tl.load(sm32 + at + _HALF // 4, mask=mask_n[:, None] & (cl + _HALF < K)[None, :], other=0)
    cd = tl.load(ex32 + rows * tl.multiple_of(NB * (_HALF // 4), _HALF // 4) + (g * (_HALF // 4) + words)[None, :],
                 mask=mask_n[:, None], other=0)
    base = tl.load(base_ptr + offs_n.to(tl.int64) * NB + g, mask=mask_n, other=0).to(tl.int32) * 0x01010101
    return bl, bh, cd, base


@triton.jit
def _fast_bits(b, cd, base, shift):
    sh = tl.full(b.shape, 0, tl.int32) + shift
    p, q = tl.inline_asm_elementwise(_PTX, "=r,=r,r,r,r,r", [b, cd, base[:, None] + tl.zeros_like(b), sh],
                                     dtype=(tl.int32, tl.int32), is_pure=True, pack=1)
    v0 = (p & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    v1 = ((p >> 16) & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    v2 = (q & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    v3 = ((q >> 16) & 0xFFFF).to(tl.int16).to(tl.bfloat16, bitcast=True)
    w = tl.join(tl.join(v0, v2), tl.join(v1, v3))
    return w.reshape(b.shape[0], 4 * b.shape[1])


@triton.jit
def _fast_patch(wl, wh, offs_n, en, ek, ev):
    lc = tl.arange(0, _HALF)
    row = offs_n[:, None] == en
    if ek < _HALF:
        wl = tl.where(row & (lc[None, :] == ek), ev, wl)
    else:
        wh = tl.where(row & ((lc + _HALF)[None, :] == ek), ev, wh)
    return wl, wh


@triton.jit
def _fast_cook(bl, bh, cd, base, e0, e1, en, ek, ev, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n):
    wl = _fast_bits(bl, cd, base, 0)
    wh = _fast_bits(bh, cd, base, 4)
    if e1 > e0:
        wl, wh = _fast_patch(wl, wh, offs_n, en, ek, ev)
        for e in range(e0 + 1, e1):
            wl, wh = _fast_patch(wl, wh, offs_n, tl.load(esc_n_ptr + e), tl.load(esc_k_ptr + e).to(tl.int32),
                                 tl.load(esc_v_ptr + e).to(tl.bfloat16, bitcast=True))
    return tl.trans(wl), tl.trans(wh)


@triton.jit
def _fast_first(esc_n_ptr, esc_k_ptr, esc_v_ptr, e0, e1):
    some = e0 < e1
    en = tl.load(esc_n_ptr + e0, mask=some, other=-1)
    ek = tl.load(esc_k_ptr + e0, mask=some, other=0).to(tl.int32)
    ev = tl.load(esc_v_ptr + e0, mask=some, other=0).to(tl.bfloat16, bitcast=True)
    return en, ek, ev


@triton.jit
def _fast_out(acc, ws_ptr, out_ptr, rows, mask_m, offs_n, mask_n, pid_s, M, N, stride_om, SPLIT: tl.constexpr):
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT:
        tl.store(ws_ptr + pid_s.to(tl.int64) * M * N + rows.to(tl.int64)[:, None] * N + offs_n[None, :], acc, mask=mask)
    else:
        tl.store(out_ptr + rows.to(tl.int64)[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask=mask)


@triton.jit
def _fast_sum(ws_ptr, out_ptr, rows, mask_m, offs_n, mask_n, M, N, stride_om, SPLITS, BT: tl.constexpr,
              BN: tl.constexpr):
    mask = mask_m[:, None] & mask_n[None, :]
    held = ws_ptr + rows.to(tl.int64)[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for s in range(SPLITS):
        acc += tl.load(held + s * M * N, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=mask)


@triton.jit(do_not_specialize=["M", "N", "NB", "NT", "SPLITS", "MB"])
def _fast_kernel(x_ptr, out_ptr, ws_ptr, cnt_ptr, sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr,
                 M, N, K, stride_xm, stride_om, NB, NT, SPLITS, MB,
                 BT1: tl.constexpr, BT2: tl.constexpr, BN: tl.constexpr, PF: tl.constexpr, EVEN_K: tl.constexpr):
    row0 = tl.program_id(0) * (BT1 + BT2)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    per = tl.cdiv(NB, SPLITS)
    g_lo = pid_s * per
    g_hi = tl.minimum(g_lo + per, NB)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    t0 = pid_n * (BN // _RT)
    t1 = tl.minimum(t0 + BN // _RT, NT)
    ra = row0 + tl.arange(0, BT1)
    ma = ra < M
    xa = x_ptr + ra.to(tl.int64)[:, None] * stride_xm
    acc_a = tl.zeros((BT1, BN), dtype=tl.float32)
    if BT2 > 0:
        rb = row0 + BT1 + tl.arange(0, BT2)
        mb = rb < M
        xb = x_ptr + rb.to(tl.int64)[:, None] * stride_xm
        acc_b = tl.zeros((BT2, BN), dtype=tl.float32)
    if PF:
        cap = tl.maximum(tl.minimum(g_hi, NB) - 1, 0)
        ga = tl.minimum(g_lo, cap)
        gb = tl.minimum(g_lo + 1, cap)
        bl, bh, cd, base = _fast_raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, ga, K, NB, EVEN_K)
        e0 = tl.load(esc_ptr + ga * NT + t0)
        e1 = tl.load(esc_ptr + ga * NT + t1)
        f0 = tl.load(esc_ptr + gb * NT + t0)
        f1 = tl.load(esc_ptr + gb * NT + t1)
        en, ek, ev = _fast_first(esc_n_ptr, esc_k_ptr, esc_v_ptr, e0, e1)
    for g in range(g_lo, g_hi):
        if PF:
            gn = tl.minimum(g + 1, cap)
            gnn = tl.minimum(g + 2, cap)
            nbl, nbh, ncd, nbase = _fast_raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, gn, K, NB, EVEN_K)
            nen, nek, nev = _fast_first(esc_n_ptr, esc_k_ptr, esc_v_ptr, f0, f1)
            h0 = tl.load(esc_ptr + gnn * NT + t0)
            h1 = tl.load(esc_ptr + gnn * NT + t1)
            wl, wh = _fast_cook(bl, bh, cd, base, e0, e1, en, ek, ev, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n)
            bl, bh, cd, base = nbl, nbh, ncd, nbase
            e0, e1, f0, f1 = f0, f1, h0, h1
            en, ek, ev = nen, nek, nev
        else:
            rbl, rbh, rcd, rbase = _fast_raw(sm_ptr, ex_ptr, base_ptr, offs_n, mask_n, g, K, NB, EVEN_K)
            r0 = tl.load(esc_ptr + g * NT + t0)
            r1 = tl.load(esc_ptr + g * NT + t1)
            ren, rek, rev = _fast_first(esc_n_ptr, esc_k_ptr, esc_v_ptr, r0, r1)
            wl, wh = _fast_cook(rbl, rbh, rcd, rbase, r0, r1, ren, rek, rev, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n)
        lo = g * _GROUP
        acc_a = tl.dot(_rows(xa, ma, lo, K, _HALF, EVEN_K), wl, acc_a)
        if BT2 > 0:
            acc_b = tl.dot(_rows(xb, mb, lo, K, _HALF, EVEN_K), wl, acc_b)
        if EVEN_K:
            acc_a = tl.dot(_rows(xa, ma, lo + _HALF, K, _HALF, EVEN_K), wh, acc_a)
            if BT2 > 0:
                acc_b = tl.dot(_rows(xb, mb, lo + _HALF, K, _HALF, EVEN_K), wh, acc_b)
        elif lo + _HALF < K:
            acc_a = tl.dot(_rows(xa, ma, lo + _HALF, K, _HALF, EVEN_K), wh, acc_a)
            if BT2 > 0:
                acc_b = tl.dot(_rows(xb, mb, lo + _HALF, K, _HALF, EVEN_K), wh, acc_b)
    if SPLITS == 1:
        _fast_out(acc_a, ws_ptr, out_ptr, ra, ma, offs_n, mask_n, pid_s, M, N, stride_om, False)
        if BT2 > 0:
            _fast_out(acc_b, ws_ptr, out_ptr, rb, mb, offs_n, mask_n, pid_s, M, N, stride_om, False)
    else:
        _fast_out(acc_a, ws_ptr, out_ptr, ra, ma, offs_n, mask_n, pid_s, M, N, stride_om, True)
        if BT2 > 0:
            _fast_out(acc_b, ws_ptr, out_ptr, rb, mb, offs_n, mask_n, pid_s, M, N, stride_om, True)
        tl.debug_barrier()
        seen = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if seen == SPLITS - 1:
            _fast_sum(ws_ptr, out_ptr, ra, ma, offs_n, mask_n, M, N, stride_om, SPLITS, BT1, BN)
            if BT2 > 0:
                _fast_sum(ws_ptr, out_ptr, rb, mb, offs_n, mask_n, M, N, stride_om, SPLITS, BT2, BN)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


@triton.jit
def _reduce_kernel(ws_ptr, out_ptr, total, splits, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    held = ws_ptr + offs
    for _ in range(splits):
        acc += tl.load(held, mask=mask, other=0.0)
        held += total
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["rows"])
def _hc_down_kernel(xn_ptr, sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr,
                    acc_ptr, rows, k_chunk, stride_xn, K, NB, NT,
                    NOUT: tl.constexpr, NACC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, EVEN_K: tl.constexpr, launch_pdl: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < rows
    mask_n = offs_n < NOUT
    t0 = pid_n * (BN // _RT)
    t1 = tl.minimum(t0 + BN // _RT, NT)
    p0 = pid_k * k_chunk
    xrow = xn_ptr + offs_m[:, None] * stride_xn
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for c in range(0, k_chunk // BK):
        col0 = p0 + c * BK
        xv = _rows(xrow, mask_m, col0, K, BK, EVEN_K)
        w = _tile(sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, mask_n, col0, K, NB, NT,
                  t0, t1, BK, EVEN_K)
        acc = tl.dot(xv, w, acc)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.atomic_add(acc_ptr + offs_m[:, None] * NACC + offs_n[None, :], acc,
                  mask=mask_m[:, None] & mask_n[None, :], sem="relaxed")


@triton.jit(do_not_specialize=["rows"])
def _hc_up_kernel(acc_ptr, sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr,
                  xn_ptr, out_ptr, inj_ptr, rows, stride_acc, stride_xn, stride_out, NB, NT,
                  H: tl.constexpr, HC: tl.constexpr, R: tl.constexpr, HAS_INJ: tl.constexpr,
                  BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr, launch_pdl: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < rows
    offs_j = tl.arange(0, BH)
    mask_all = offs_j >= 0
    source = acc_ptr + offs_m[:, None] * stride_acc
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    mixed = tl.zeros((BM, BH), dtype=tl.float32)
    for h in tl.static_range(HC):
        offs_n = h * H + pid_c * BH + offs_j
        t0 = (h * H + pid_c * BH) // _RT
        t1 = tl.minimum(t0 + BH // _RT, NT)
        gate = tl.zeros((BM, BH), dtype=tl.float32)
        for c in range(0, R // BK):
            lora = _rows(source, mask_m, c * BK, R, BK, True)
            lora = lora.to(out_ptr.dtype.element_ty).to(tl.float32) / HC
            lora = (lora * tl.sigmoid(lora)).to(tl.bfloat16)
            w = _tile(sm_ptr, ex_ptr, base_ptr, esc_ptr, esc_n_ptr, esc_k_ptr, esc_v_ptr, offs_n, mask_all, c * BK, R,
                      NB, NT, t0, t1, BK, True)
            gate = tl.dot(lora, w, gate)
        gate = gate.to(out_ptr.dtype.element_ty).to(tl.float32)
        x = tl.load(xn_ptr + offs_m[:, None] * stride_xn + offs_n[None, :], mask=mask_m[:, None], other=0.0)
        mixed += tl.sigmoid(gate) * x.to(tl.float32)
    mixed = mixed / HC
    if HAS_INJ:
        offs_h = tl.arange(0, HC)
        inj_mask = mask_m[:, None] & (offs_h < HC)[None, :] & (pid_c == 0)
        inj = tl.load(acc_ptr + offs_m[:, None] * stride_acc + R + offs_h[None, :], mask=inj_mask, other=0.0)
        tl.store(inj_ptr + offs_m[:, None] * HC + offs_h[None, :], inj, mask=inj_mask)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    offs_c = pid_c * BH + offs_j
    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_c[None, :], mixed, mask=mask_m[:, None])


def _sms(device):
    if INTERPRETED or device.type != "cuda":
        return 1
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _STATE["sms"]:
        _STATE["sms"][key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _STATE["sms"][key]


def bucket(rows):
    return max(16, triton.next_power_of_2(rows))


def gemm_bms(mb):
    if mb <= 16:
        return (16,)
    if mb <= 64:
        return (mb // 2, mb)
    if mb <= 256:
        return tuple(bm for bm in (64, 128, 256) if bm <= mb)
    return (64, 128, 256)


def gemm_warps(bm, bn):
    tile = bm * bn
    if tile < 128 * 64:
        return (4,)
    if tile >= GEMM_TILE_LIMIT:
        return (8,)
    return (4, 8)


def splits_for(m, n, k, bm, bn, sw, sms):
    if sw == 0:
        return 1
    chunks = -(-k // GEMM_BK)
    tiles = -(-m // bm) * -(-n // bn)
    return max(1, min(chunks // 2, -(-sw * sms // tiles)))


def prefetches(bm, bn, warps):
    return bm <= 128 and bm * bn <= PREFETCH_LIMIT * warps


def gemm_space():
    made = []
    for bm in GEMM_BMS:
        for bn in GEMM_BNS:
            if bm * bn > GEMM_TILE_LIMIT or (bn == max(GEMM_BNS) and bm > 64):
                continue
            for warps in gemm_warps(bm, bn):
                for pf in (0, 1) if prefetches(bm, bn, warps) else (0,):
                    for sw in (0, 1, 2):
                        made.append(triton.Config({"BM": bm, "BN": bn, "PF": pf, "SW": sw}, num_warps=warps,
                                                  num_stages=2))
    return made


def gemm_fits(config, m, n, k, sms):
    held = config.kwargs
    bm, bn, pf, sw = held["BM"], held["BN"], held["PF"], held["SW"]
    mb = bucket(m)
    if bm not in gemm_bms(mb) or config.num_warps not in gemm_warps(bm, bn):
        return False
    if pf and not prefetches(bm, bn, config.num_warps):
        return False
    if sw == 0:
        return True
    if mb > 256:
        return False
    made = splits_for(m, n, k, bm, bn, sw, sms)
    return made > 1 and (sw == 1 or made != splits_for(m, n, k, bm, bn, sw - 1, sms))


def _prune(configs, named, **_):
    m, n, k = named["M"], named["N"], named["K"]
    sms = _sms(named["x_ptr"].device)
    return [config for config in configs if gemm_fits(config, m, n, k, sms)]


def default_config(m):
    return triton.Config({"BM": min(64, bucket(m)), "BN": 64, "PF": 0, "SW": 1}, num_warps=4, num_stages=2)


def decode_space():
    return [triton.Config({"BN": bn}, num_warps=warps, num_stages=1) for bn in DECODE_BNS for warps in (2, 4, 8)
            if bn * HALF // (32 * warps) >= 16]


def _bench(fn, quantiles=None):
    return triton.testing.do_bench(fn, warmup=TUNE_WARMUP_MS, rep=TUNE_REP_MS, quantiles=quantiles)


def _autotune(configs, key, **extra):
    said = inspect.signature(triton.autotune).parameters
    timing = {"do_bench": _bench} if "do_bench" in said else {"warmup": TUNE_WARMUP_MS, "rep": TUNE_REP_MS}
    return triton.autotune(configs=configs, key=key, **extra, **timing)


_gemm_tuned = _autotune(gemm_space(), ["N", "K", "MB"], prune_configs_by={"early_config_prune": _prune})(_gemm_kernel)
_decode_tuned = _autotune(decode_space(), ["K", "RB"])(_decode_kernel)


def _capturing(device):
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _said(config):
    held = ",".join(f"{name}={value}" for name, value in sorted(config.kwargs.items()))
    return f"{held},warps={config.num_warps}"


def _tuned(tuner, key, what, started):
    if key in _TUNED:
        return
    _TUNED[key] = tuner.best_config
    timings = getattr(tuner, "configs_timings", None) or {}
    best = timings.get(tuner.best_config)
    best = best[0] if isinstance(best, (list, tuple)) else best
    print(f"BF16C tuned {what}: {_said(tuner.best_config)} "
          f"{'%.1f us' % (1000 * best) if best is not None else ''} of {len(timings)} configs in "
          f"{time.perf_counter() - started:.2f} s", flush=True)


def _untuned(key, what):
    if key not in _STATE["untuned"]:
        _STATE["untuned"].add(key)
        print(f"BF16C untuned {what} inside a CUDA-graph capture: the default tile runs; the eager warmup did not "
              f"reach this shape", flush=True)


def most_splits(m, n, k, sms):
    key = (m, n, k, sms)
    if key not in _SPLITS:
        _SPLITS[key] = max(splits_for(m, n, k, config.kwargs["BM"], config.kwargs["BN"], config.kwargs["SW"], sms)
                           for config in _GEMM_SPACE if gemm_fits(config, m, n, k, sms))
    return _SPLITS[key]


_GEMM_SPACE = gemm_space()


def decode(packed, out=None, row0=0, row1=None, config=None):
    n, k = packed.shape
    row1 = n if row1 is None else row1
    if row0 % DECODE_ALIGN or not 0 <= row0 < row1 <= n:
        raise ValueError(f"decode rows {row0}:{row1} must start on a {DECODE_ALIGN}-row boundary inside 0:{n}")
    if out is None:
        out = torch.empty((row1 - row0, k), dtype=torch.bfloat16, device=packed.device)
    rb = bucket(row1 - row0)
    args = (out, *packed.tensors(), row1, k, packed.kb, packed.nt, row0, rb)
    even = k % GROUP == 0
    if config is None and packed.device.type == "cuda" and not INTERPRETED:
        key = ("decode", k, rb)
        if key in _TUNED or not _capturing(packed.device):
            started = time.perf_counter()
            _decode_tuned[lambda meta: (triton.cdiv(row1 - row0, meta["BN"]), packed.kb)](*args, EVEN_K=even)
            _tuned(_decode_tuned, key, f"decode K={k} rows<={rb}", started)
            return out
        _untuned(key, f"decode K={k} rows<={rb}")
    config = config or triton.Config({"BN": 64}, num_warps=4, num_stages=1)
    bn = config.kwargs["BN"]
    _decode_kernel[(triton.cdiv(row1 - row0, bn), packed.kb)](*args, BN=bn, EVEN_K=even, num_warps=config.num_warps)
    return out


def decode_selected(packed, index, chunk=8192):
    out = torch.empty((index.numel(), packed.k), dtype=torch.bfloat16, device=packed.device)
    for row0 in range(0, packed.n, chunk):
        row1 = min(packed.n, row0 + chunk)
        at = ((index >= row0) & (index < row1)).nonzero().reshape(-1)
        if at.numel():
            out[at] = decode(packed, row0=row0, row1=row1)[index[at] - row0]
    return out


def gemm(x, packed, out_dtype=None, config=None):
    n, k = packed.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=out_dtype or x.dtype, device=x.device)
    if m == 0:
        return out.reshape(*lead, n)
    sms = _sms(x.device)
    mb = bucket(m)
    even = k % GEMM_BK == 0

    def grid(meta):
        return (triton.cdiv(n, meta["BN"]), triton.cdiv(m, meta["BM"]),
                splits_for(m, n, k, meta["BM"], meta["BN"], meta["SW"], sms))

    def launch(kernel, splits, **extra):
        ws = torch.empty((splits * m * n if splits > 1 else 1,), dtype=torch.float32, device=x.device)
        kernel[grid](x2, out, ws, *packed.tensors(), m, n, k, x2.stride(0), out.stride(0), packed.kb, packed.nt, mb,
                     BK=GEMM_BK, EVEN_K=even, **extra)
        return ws

    key = ("gemm", n, k, mb, str(x2.dtype), str(out.dtype))
    tune = config is None and x.is_cuda and not INTERPRETED
    if tune and (key in _TUNED or not _capturing(x.device)):
        started = time.perf_counter()
        ws = launch(_gemm_tuned, most_splits(m, n, k, sms))
        _tuned(_gemm_tuned, key, f"gemm {n}x{k} rows<={mb}", started)
        config = _gemm_tuned.best_config
    else:
        if tune:
            _untuned(key, f"gemm {n}x{k} rows<={mb}")
        config = config or default_config(m)
        if tune and not gemm_fits(config, m, n, k, sms):
            config = triton.Config({**config.kwargs, "SW": 0}, num_warps=config.num_warps,
                                   num_stages=config.num_stages)
        ws = launch(_gemm_kernel, grid(config.kwargs)[2], **config.all_kwargs())
    splits = grid(config.kwargs)[2]
    if splits > 1:
        total = m * n
        _reduce_kernel[(triton.cdiv(total, 1024),)](ws, out, total, splits, BLOCK=1024)
    return out.reshape(*lead, n)


def fast_enabled():
    return str(os.environ.get(FAST_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def _verdict_text(path):
    limit = float(os.environ.get(VERDICT_WAIT_ENV) or 0)
    target = Path(path)
    waited = 0.0
    while not target.is_file() and waited < limit:
        time.sleep(0.5)
        waited += 0.5
    if waited:
        print(f"BF16C waited {waited:.0f} s for the verdict {target.name}", flush=True)
    return target.read_text(encoding="utf-8")


def _write_report(path, report):
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(temporary, target)


def fast_verified():
    path = str(os.environ.get(FAST_VERDICT_ENV) or "").strip()
    if not path:
        return None
    if _FAST["verified"] is None:
        try:
            held = json.loads(_verdict_text(path))
            _FAST["verified"] = ({tuple(shape): int(held.get("rows", {}).get(f"{shape[0]}x{shape[1]}", FAST_ROWS))
                                  for shape in held.get("passed", [])} if held.get("ok") else {})
        except (OSError, ValueError) as failure:
            _FAST["verified"] = {}
            print(f"BF16C FAST2 verdict {path} unreadable ({type(failure).__name__}: {failure}): every shape keeps the "
                  f"old kernel", flush=True)
    return _FAST["verified"]


def token_tiles(m):
    if m > FAST_ROWS:
        return 128, 128
    whole = max(16, triton.next_power_of_2(m))
    if whole <= 128 and (whole <= 32 or 4 * (whole - m) < whole):
        return whole, 0
    half = min(128, whole // 2)
    return half, max(16, triton.next_power_of_2(m - half))


def fast_warps(bt, bn, pf):
    return 8 if bt * bn > 160 * 64 or bt > 160 or (pf and (bt * bn > 128 * 64 or (bn == 128 and bt >= 32))) else 4


def fast_rows(n, k):
    verified = fast_verified()
    return FAST_ROWS if verified is None else verified.get((n, k), 0)


def row_blocks(m):
    return 1 if m <= FAST_ROWS else triton.next_power_of_2(-(-m // FAST_ROWS))


def fast_fits(config, m, n, k, sms):
    held = config.kwargs
    bn, splits = held["BN"], held["SPLITS"]
    bt1, bt2 = token_tiles(m)
    if config.num_warps != fast_warps(bt1 + bt2, bn, held["PF"]) or (bt1 + bt2) * bn > FAST_TILE_LIMIT:
        return False
    if splits == 1:
        return True
    if m > FAST_ROWS:
        return False
    tiles = -(-n // bn)
    groups = -(-k // GROUP)
    per = -(-groups // splits)
    smaller = [s for s in FAST_SPLITS if s < splits]
    return (splits <= groups and tiles < 2 * sms and tiles * splits <= 4 * sms
            and all(-(-groups // s) != per for s in smaller))


def fast_space():
    return [triton.Config({"BN": bn, "PF": pf, "SPLITS": splits}, num_warps=warps, num_stages=1)
            for bn in FAST_BNS for pf in (0, 1) for warps in (4, 8) for splits in FAST_SPLITS]


def _fast_prune(configs, named, **extra):
    m, n, k = named["M"], named["N"], named["K"]
    sms = _sms(named["x_ptr"].device)
    return [config for config in configs if fast_fits(config, m, n, k, sms)]


def fast_default(m, n, k, sms):
    bt1, bt2 = token_tiles(m)
    bn = 64 if (bt1 + bt2) * 64 <= FAST_TILE_LIMIT else 32
    tiles = -(-n // bn)
    groups = -(-k // GROUP)
    splits = max(1, min(groups, -(-sms // tiles))) if tiles < sms else 1
    return triton.Config({"BN": bn, "PF": 0, "SPLITS": splits}, num_warps=fast_warps(bt1 + bt2, bn, 0), num_stages=1)


_FAST_SPACE = fast_space()
_fast_tuned = _autotune(_FAST_SPACE, ["N", "K", "MB", "BT1", "BT2"],
                        prune_configs_by={"early_config_prune": _fast_prune})(
    _fast_kernel)


def old_linear(x, packed, out_dtype=None):
    rows = x.numel() // x.shape[-1]
    if rows <= decode_rows():
        return gemm(x, packed, out_dtype=out_dtype)
    return torch.nn.functional.linear(x, decode(packed)).to(out_dtype or x.dtype)


def fast_counters(packed, make=True):
    key = packed.sm.data_ptr()
    held = _FAST["counters"].get(key)
    if held is None and make:
        held = (packed.sm, torch.zeros(-(-packed.n // min(FAST_BNS)), dtype=torch.int32, device=packed.device))
        _FAST["counters"][key] = held
    return None if held is None else held[1]


def _fast_said(key, why):
    if key not in _FAST["told"]:
        _FAST["told"].add(key)
        print(f"BF16C FAST2 {why}: the old kernel runs", flush=True)


def fast_gemm(x, packed, out_dtype=None, config=None):
    n, k = packed.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    capturing = _capturing(x.device)
    verified = fast_verified()
    if INTERPRETED or not x.is_cuda:
        return old_linear(x, packed, out_dtype)
    if m > FAST_ROWS_MAX or k % 4 or (verified is not None and (n, k) not in verified):
        _fast_said(("shape", n, k), f"not verified for {n}x{k}" if m <= FAST_ROWS_MAX and not k % 4 else
                   f"takes rows <= {FAST_ROWS_MAX} and K a multiple of 4, found {m}x{n}x{k}")
        return old_linear(x, packed, out_dtype)
    counters = fast_counters(packed, make=not capturing)
    bt1, bt2 = token_tiles(m)
    blocks = row_blocks(m)
    key = ("fast", n, k, bt1, bt2, blocks, str(x2.dtype), str(out_dtype or x.dtype))
    tune = config is None
    if counters is None or (tune and capturing and key not in _TUNED):
        _fast_said(("capture", key), f"meets {n}x{k} rows<={bt1 + bt2} x{blocks} first inside a CUDA-graph capture")
        return old_linear(x, packed, out_dtype)
    out = torch.empty((m, n), dtype=out_dtype or x.dtype, device=x.device)
    if m == 0:
        return out.reshape(*lead, n)
    sms = _sms(x.device)
    args = (x2, out, None, counters, *packed.tensors(), m, n, k, x2.stride(0), out.stride(0), packed.kb, packed.nt)
    held = dict(MB=blocks, BT1=bt1, BT2=bt2, EVEN_K=k % GROUP == 0)

    def grid(meta):
        return (triton.cdiv(m, bt1 + bt2), triton.cdiv(n, meta["BN"]), meta["SPLITS"])

    def workspace(splits):
        return torch.empty((splits * m * n if splits > 1 else 1,), dtype=torch.float32, device=x.device)

    if tune and key not in _TUNED:
        started = time.perf_counter()
        most = max(c.kwargs["SPLITS"] for c in _FAST_SPACE if fast_fits(c, m, n, k, sms))
        _fast_tuned[grid](*args[:2], workspace(most), *args[3:], **held)
        _tuned(_fast_tuned, key, f"fast {n}x{k} rows<={(bt1 + bt2) * blocks} tiles {bt1}+{bt2}", started)
        return out.reshape(*lead, n)
    config = _TUNED[key] if tune else (config or fast_default(m, n, k, sms))
    _fast_kernel[grid(config.kwargs)](*args[:2], workspace(config.kwargs["SPLITS"]), *args[3:], **held,
                                      **config.all_kwargs())
    return out.reshape(*lead, n)


@triton.jit
def _mix_bits(b, cd, base, lc, ck, ce, shift):
    if _ASM:
        sh = tl.full(b.shape, 0, tl.int32) + shift
        w = tl.inline_asm_elementwise(_MIX_PTX, "=r,=r,r,r,r,r,r,r,r,r,r,r", [b, cd, base, lc, ck, ce, sh],
                                      dtype=tl.bfloat16, is_pure=False, pack=4)
    else:
        bi = b.to(tl.int32)
        c = (cd.to(tl.int32) >> shift) & 15
        e = tl.where(c == 0, 0, base.to(tl.int32) + c - 1)
        e = tl.where(lc.to(tl.int32) == ck.to(tl.int32), e | ce.to(tl.int32), e)
        w = (((bi & 128) << 8) | (e << 7) | (bi & 127)).to(tl.int16).to(tl.bfloat16, bitcast=True)
    return w


@triton.jit
def _mix_chunk(b, cd, hd, lc, half):
    kl = (hd >> 8) & 255
    ck = tl.where((kl >> 6) == half, kl & 63, 255).to(tl.uint8)
    return _mix_bits(b, cd, (hd & 255).to(tl.uint8)[None, :], lc, ck[None, :],
                     ((hd >> 16) & 255).to(tl.uint8)[None, :], half * 4)


@triton.jit
def _mix_probe_kernel(out_ptr, sm_ptr, ex_ptr, hd_ptr, N, K, NB, BN: tl.constexpr):
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    c = tl.program_id(1)
    rn = offs_n % N
    rk = tl.arange(0, _HALF)
    wrow = rn.to(tl.int64)[None, :]
    b = tl.load(sm_ptr + wrow * K + (c * _HALF + rk)[:, None])
    cd = tl.load(ex_ptr + wrow * tl.multiple_of(NB * _HALF, _HALF) + (c // 2) * _HALF + rk[:, None])
    hd = tl.load(hd_ptr + (c // 2) * N + rn)
    w = _mix_chunk(b, cd, hd, rk.to(tl.uint8)[:, None], c % 2)
    tl.store(out_ptr + offs_n.to(tl.int64)[None, :] * K + (c * _HALF + rk)[:, None], w, mask=(offs_n < N)[None, :])


@triton.jit
def _mix_fetch(xrow, rn_ptr, rk_ptr, rv_ptr, e0, e_hi, k_lo, k_hi, EB: tl.constexpr):
    ee = e0 + tl.arange(0, EB)
    live = ee < e_hi
    en = tl.load(rn_ptr + ee, mask=live, other=-1)
    ek = tl.load(rk_ptr + ee, mask=live, other=0)
    ev = tl.load(rv_ptr + ee, mask=live, other=0).to(tl.bfloat16, bitcast=True)
    live = live & (ek >= k_lo) & (ek < k_hi)
    xg = tl.load(xrow + ek[None, :], mask=live[None, :], other=0.0)
    return xg, tl.where(live, en, -1), ev


@triton.jit
def _mix_apply(acc, xg, en, ev, offs_n, BN: tl.constexpr, EB: tl.constexpr):
    vm = tl.where(en[:, None] == offs_n[None, :], ev[:, None], tl.zeros((EB, BN), dtype=tl.bfloat16))
    return tl.dot(xg, vm, acc)


@triton.jit
def _mix_rest(acc, xrow, offs_n, rn_ptr, rk_ptr, rv_ptr, e_lo, e_hi, k_lo, k_hi, BN: tl.constexpr,
              EB: tl.constexpr):
    for e0 in tl.range(e_lo, e_hi, EB, num_stages=1):
        xg, en, ev = _mix_fetch(xrow, rn_ptr, rk_ptr, rv_ptr, e0, e_hi, k_lo, k_hi, EB)
        acc = _mix_apply(acc, xg, en, ev, offs_n, BN, EB)
    return acc


@triton.jit(do_not_specialize=["M", "SPLITS"])
def _mix_kernel(x_ptr, out_ptr, ws_ptr, cnt_ptr, sm_ptr, ex_ptr, hd_ptr, rp_ptr, rn_ptr, rk_ptr, rv_ptr,
                M, N, K, stride_xm, stride_om, NB, NT, SPLITS,
                BM: tl.constexpr, BN: tl.constexpr, GN: tl.constexpr, EB: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    width = GN * tiles_m
    first = (pid // width) * GN
    span = tl.minimum(tiles_n - first, GN)
    pid_n = first + (pid % width) % span
    pid_m = (pid % width) // span
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    rm = offs_m % M
    rn = offs_n % N
    rk = tl.arange(0, _HALF)
    lc = rk.to(tl.uint8)[:, None]
    chunks = K // _HALF
    per = tl.cdiv(chunks, SPLITS)
    c_lo = pid_s * per
    c_hi = tl.minimum(c_lo + per, chunks)
    wrow = rn.to(tl.int64)[None, :]
    xp = x_ptr + rm.to(tl.int64)[:, None] * stride_xm + (c_lo * _HALF + rk)[None, :]
    sp = sm_ptr + wrow * K + (c_lo * _HALF + rk)[:, None]
    ep = ex_ptr + wrow * tl.multiple_of(NB * _HALF, _HALF) + rk[:, None]
    hp = hd_ptr + rn
    xrow = x_ptr + rm.to(tl.int64)[:, None] * stride_xm
    t0 = (pid_n * BN) // _RT
    e_lo = tl.load(rp_ptr + t0)
    e_hi = tl.load(rp_ptr + tl.minimum(t0 + BN // _RT, NT))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    hd = tl.load(hp + (c_lo // 2) * N)
    for c in range(c_lo, c_hi):
        half = c % 2
        upcoming = tl.load(hp + (tl.minimum(c + 1, chunks - 1) // 2) * N)
        xv = tl.load(xp)
        b = tl.load(sp)
        cd = tl.load(ep + (c // 2) * _HALF)
        acc = tl.dot(xv, _mix_chunk(b, cd, hd, lc, half), acc)
        hd = upcoming
        xp += _HALF
        sp += _HALF
    acc = _mix_rest(acc, xrow, offs_n, rn_ptr, rk_ptr, rv_ptr, e_lo, e_hi, c_lo * _HALF, c_hi * _HALF, BN, EB)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if SPLIT:
        tile = pid_m * tiles_n + pid_n
        tl.store(ws_ptr + pid_s.to(tl.int64) * M * N + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :], acc,
                 mask=mask)
        tl.debug_barrier()
        seen = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if seen == SPLITS - 1:
            held = ws_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(SPLITS):
                acc += tl.load(held + s * M * N, mask=mask, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :],
                     acc.to(out_ptr.dtype.element_ty), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mask)


def mix_enabled():
    return str(os.environ.get(MIX_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def mix_decode_enabled():
    return mix_enabled() and str(os.environ.get(MIX_DECODE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def mix_bands():
    return ((MIX_DECODE_BAND,) if mix_decode_enabled() else ()) + MIX_BANDS


def mix_index(packed, make=True):
    key = packed.sm.data_ptr()
    held = _MIX["index"].get(key)
    if held is None and make:
        n, k = packed.shape
        counts = packed.esc_ptr[1:].to(torch.int64) - packed.esc_ptr[:-1].to(torch.int64)
        tile = torch.repeat_interleave(torch.arange(counts.numel(), device=packed.device), counts)
        rows = packed.esc_n.to(torch.int64)
        group = tile // packed.nt
        local = packed.esc_k.to(torch.int64)
        bits = packed.esc_v.to(torch.int64) & 0xFFFF
        head = packed.base.T.to(torch.int32) | (255 << 8)
        lead = torch.ones_like(rows, dtype=torch.bool)
        if rows.numel() > 1:
            key_ = group * n + rows
            lead[1:] = key_[1:] != key_[:-1]
        at_g, at_n = group[lead], rows[lead]
        head[at_g, at_n] = (packed.base[at_n, at_g].to(torch.int64) | local[lead] << 8
                            | ((bits[lead] >> 7) & 255) << 16).to(torch.int32)
        rest = ~lead
        rows, cols = rows[rest], group[rest] * GROUP + local[rest]
        order = torch.argsort(rows * k + cols)
        rows, cols = rows[order], cols[order]
        ptr = torch.zeros(packed.nt + 1, dtype=torch.int32, device=packed.device)
        if rows.numel():
            ptr[1:] = torch.bincount(rows // ROW_TILE, minlength=packed.nt).cumsum(0).to(torch.int32)
        held = (packed.sm, head.contiguous(), ptr, rows.to(torch.int32), cols.to(torch.int32),
                packed.esc_v[rest][order].contiguous())
        _MIX["index"][key] = held
    return None if held is None else held[1:]


def mix_counters(device, make=True):
    key = device.index if device.index is not None else torch.cuda.current_device()
    held = _MIX["counters"].get(key)
    if held is None and make:
        held = torch.zeros(MIX_COUNTERS, dtype=torch.int32, device=device)
        _MIX["counters"][key] = held
    return held


def mix_workspace(device, numel):
    key = device.index if device.index is not None else torch.cuda.current_device()
    held = _MIX["workspace"].get(key)
    if held is None or held.numel() < numel:
        if held is not None:
            _MIX["retired"].append(held)
        held = torch.empty(numel, dtype=torch.float32, device=device)
        _MIX["workspace"][key] = held
    return held


def mix_config(bm, bn, gn, splits, warps, stages):
    return triton.Config({"BM": bm, "BN": bn, "GN": gn, "SPLITS": splits}, num_warps=warps, num_stages=stages)


def mix_supported(n, k):
    return k % HALF == 0 and n * k < 1 << 31


def mix_gemm(x, packed, config, out_dtype=None):
    n, k = packed.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=out_dtype or x.dtype, device=x.device)
    if m == 0:
        return out.reshape(*lead, n)
    held = config.kwargs
    bm, bn, splits = held["BM"], held["BN"], held["SPLITS"]
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    capturing = _capturing(x.device)
    index = mix_index(packed, make=not capturing)
    if index is None:
        return None
    counters = mix_counters(x.device, make=not capturing) if splits > 1 else index[1]
    if counters is None or (splits > 1 and tiles > MIX_COUNTERS):
        return None
    ws = mix_workspace(x.device, splits * m * n if splits > 1 else 1)
    _mix_kernel[(tiles, splits)](x2, out, ws, counters, packed.sm, packed.ex, *index, m, n, k, x2.stride(0),
                                 out.stride(0), packed.kb, packed.nt, splits, BM=bm, BN=bn, GN=held["GN"], EB=MIX_EB,
                                 SPLIT=splits > 1, num_warps=config.num_warps, num_stages=config.num_stages)
    return out.reshape(*lead, n)


def mix_decode(packed):
    n, k = packed.shape
    index = mix_index(packed)
    out = torch.empty((n, k), dtype=torch.bfloat16, device=packed.device)
    _mix_probe_kernel[(triton.cdiv(n, 64), k // HALF)](out, packed.sm, packed.ex, index[0], n, k, packed.kb, BN=64,
                                                       num_warps=4)
    if index[2].numel():
        out[index[2].to(torch.int64), index[3].to(torch.int64)] = index[4].view(torch.bfloat16)
    return out


def mix_said(config):
    held = config.kwargs
    return {"BM": held["BM"], "BN": held["BN"], "GN": held["GN"], "SPLITS": held["SPLITS"], "warps": config.num_warps,
            "stages": config.num_stages}


def mix_made(said):
    return mix_config(said["BM"], said["BN"], said["GN"], said["SPLITS"], said["warps"], said["stages"])


def mix_verified():
    path = str(os.environ.get(MIX_VERDICT_ENV) or "").strip()
    if not path:
        return None
    if _MIX["verified"] is None:
        try:
            held = json.loads(_verdict_text(path))
            _MIX["verified"] = ({tuple(int(part) for part in shape.split("x")):
                                 [(int(hi), mix_made(said) if said else None) for hi, said in bands]
                                 for shape, bands in held.get("serve", {}).items()} if held.get("ok") else {})
        except (OSError, ValueError, KeyError, TypeError) as failure:
            _MIX["verified"] = {}
            print(f"BF16C MIX verdict {path} unreadable ({type(failure).__name__}: {failure}): every shape keeps "
                  f"decode plus the stock GEMM", flush=True)
    return _MIX["verified"]


def mix_pick(n, k, rows):
    for hi, config in (mix_verified() or {}).get((n, k), ()):
        if rows <= hi:
            return config
    return None


def mix_decode_pick(n, k, rows):
    if not mix_decode_enabled() or not FAST_ROWS < rows <= MIX_DECODE_BAND[0]:
        return None
    for hi, config in (mix_verified() or {}).get((n, k), ()):
        if hi == MIX_DECODE_BAND[0]:
            return config
    return None


def mix_wanted(n, k):
    return any(config is not None for _, config in (mix_verified() or {}).get((n, k), ()))


def _mix_told(key, why):
    if key not in _MIX["told"]:
        _MIX["told"].add(key)
        print(f"BF16C MIX {why}: decode plus the stock GEMM runs", flush=True)


def mix_fits(config, m, n, k, sms):
    held = config.kwargs
    bm, bn, splits = held["BM"], held["BN"], held["SPLITS"]
    if bn > max(32, triton.next_power_of_2(n)) or bm > max(64, triton.next_power_of_2(m)):
        return False
    if splits == 1:
        return True
    chunks = k // HALF
    per = -(-chunks // splits)
    tiles = -(-m // bm) * -(-n // bn)
    smaller = [s for s in MIX_SPLITS if s < splits]
    return (tiles < 2 * sms and per >= 4 and (splits - 1) * per < chunks and tiles * splits <= 4 * sms
            and tiles <= MIX_COUNTERS and all(-(-chunks // s) != per for s in smaller))


def mix_space(m, n, k, sms):
    made = [mix_config(bm, bn, 1, splits, warps, stages) for bm, bn, warps, stages in MIX_TILES for splits in MIX_SPLITS]
    return [config for config in made if mix_fits(config, m, n, k, sms)]


def mix_weight(n, k, device):
    weight, made = check_weight(n, k, device)
    bits = weight.view(torch.int16)
    step = max(1, n // MIX_OVERFLOW_ROWS)
    for index in range(min(n, MIX_OVERFLOW_ROWS)):
        for col, value in ((1, 0x4400), (2, -0x3B80), (70, 0x4500)):
            if col < k:
                bits[index * step, col] = value
    return weight, made


def _mix_bench(fn, warmup, rep):
    return 1000 * triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def _mix_errors(got, ref):
    err = (got.float() - ref).abs()
    return err.max().item(), err.mean().item()


def _mix_replays(x, packed, config, want):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mix_gemm(x, packed, config)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mix_gemm(x, packed, config)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    return out is not None and torch.equal(out.view(torch.int16), want.view(torch.int16))


def mix_check_shape(n, k, rows, device, margin=MIX_MARGIN):
    weight, made = mix_weight(n, k, device)
    packed = pack(weight)
    index = mix_index(packed)
    found = {"shape": [n, k], "escapes": packed.escapes(), "overflow": int(index[2].numel()), "bands": [],
             "exact": False}
    found["decode_exact"] = bool(torch.equal(bits_of(mix_decode(packed)), bits_of(weight)))
    if not found["decode_exact"]:
        found["failure"] = "the in-loop decode differs from the weight's bits"
        return found
    sms = _sms(device)
    for m in rows:
        x = torch.randn(m, k, generator=made, device=device).to(torch.bfloat16)
        ref = x.float() @ weight.float().T
        stock = old_linear(x, packed)
        bound, mean_bound = _mix_errors(stock, ref)
        best = None
        tried = 0
        for config in mix_space(m, n, k, sms):
            try:
                got = mix_gemm(x, packed, config)
                torch.cuda.synchronize()
            except Exception as failure:
                if "out of resource" in str(failure).lower() or type(failure).__name__ == "OutOfResources":
                    continue
                raise
            worst, mean = _mix_errors(got, ref)
            if worst > 2 * bound + 1e-6 or mean > 1.5 * mean_bound + 1e-6:
                found["failure"] = (f"rows {m} {_said(config)}: max error {worst:.4g} (stock {bound:.4g}), mean "
                                    f"{mean:.4g} (stock {mean_bound:.4g})")
                return found
            spent = _mix_bench(lambda: mix_gemm(x, packed, config), TUNE_WARMUP_MS, TUNE_REP_MS)
            tried += 1
            if best is None or spent < best[0]:
                best = (spent, config, worst)
        if best is None:
            found["failure"] = f"rows {m}: no tile fits this card"
            return found
        config = best[1]
        stock_us = min(_mix_bench(lambda: old_linear(x, packed), 5, 40) for _ in range(2))
        mix_us = min(_mix_bench(lambda: mix_gemm(x, packed, config), 5, 40) for _ in range(2))
        band = {"rows": m, "stock_us": stock_us, "mix_us": mix_us, "config": mix_said(config), "tried": tried,
                "max_err": best[2], "stock_max_err": bound, "wins": bool(mix_us * margin < stock_us)}
        if band["wins"]:
            band["captured"] = _mix_replays(x, packed, config, mix_gemm(x, packed, config))
            if not band["captured"]:
                found["failure"] = f"rows {m} {_said(config)}: a CUDA-graph replay differs from the eager result"
                return found
        found["bands"].append(band)
    found["exact"] = True
    return found


def mix_line(found):
    n, k = found["shape"]
    said = f"BF16C MIX self-check {n}x{k}: "
    if not found["decode_exact"]:
        return said + "MISMATCH: the in-loop decode differs from the weight's bits"
    said += (f"in-loop decode bit-exact ({found['escapes']} escapes, {found['overflow']} past the head)")
    for band in found["bands"]:
        said += (f"; rows {band['rows']} stock {band['stock_us']:.0f} us mix {band['mix_us']:.0f} us "
                 f"x{band['stock_us'] / band['mix_us']:.2f} ({_said(mix_made(band['config']))}, {band['tried']} tiles, "
                 f"max err {band['max_err']:.3g} vs {band['stock_max_err']:.3g}) "
                 f"{'WINS, graph replay exact' if band['wins'] else 'keeps stock'}")
    if found.get("failure"):
        said += f"; FAILED: {found['failure']}"
    return said


def mix_selfcheck(path, shapes=SERVED_SHAPES, rows=MIXED_ROWS, margin=MIX_MARGIN):
    served_bands = mix_bands()
    if mix_decode_enabled() and MIX_DECODE_BAND[1] not in rows:
        rows = (MIX_DECODE_BAND[1],) + tuple(rows)
    started = time.perf_counter()
    report = {"ok": False, "shapes": [], "serve": {}, "reason": ""}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        device = torch.device("cuda")
        report["device"] = torch.cuda.get_device_name(device)
        for n, k in shapes:
            if n >= 100_000 or not mix_supported(n, k):
                continue
            found = mix_check_shape(n, k, rows, device, margin)
            report["shapes"].append(found)
            _MIX["index"].clear()
            torch.cuda.empty_cache()
            print(mix_line(found), flush=True)
            if not found["exact"]:
                report["reason"] = f"{n}x{k}: {found.get('failure')}"
                break
            bands = {band["rows"]: band for band in found["bands"]}
            report["serve"][f"{n}x{k}"] = [[hi, bands[at]["config"] if bands.get(at, {}).get("wins") else None]
                                           for hi, at in served_bands]
        else:
            wins = sum(said is not None for bands in report["serve"].values() for _, said in bands)
            report["ok"] = wins > 0
            if not wins:
                report["reason"] = "not faster than decode plus the stock GEMM on any shape and row band"
    except Exception as failure:
        report["reason"] = f"{type(failure).__name__}: {failure}"
    report["seconds"] = round(time.perf_counter() - started, 1)
    _write_report(path, report)
    wins = sum(said is not None for bands in report["serve"].values() for _, said in bands)
    print(f"BF16C MIX self-check {'PASSED' if report['ok'] else 'FAILED'} in {report['seconds']} s: {wins} of "
          f"{len(served_bands) * len(report['serve'])} shape bands take the fused kernel"
          + (f"; {report['reason']}" if report["reason"] else ""), flush=True)
    return report


def hc_down(xn, packed, acc, rows, k_chunk, nout, nacc, bm, bn, splits, pdl):
    _hc_down_kernel[(triton.cdiv(rows, bm), triton.cdiv(nout, bn), splits)](
        xn, *packed.tensors(), acc, rows, k_chunk, xn.stride(0), packed.k, packed.kb, packed.nt, NOUT=nout,
        NACC=nacc, BM=bm, BN=bn, BK=HC_BK, EVEN_K=packed.k % HC_BK == 0, launch_pdl=pdl,
        num_warps=4 if bm * bn <= 128 * 64 else 8, num_stages=3)


def hc_up(source, stride_source, packed, xn, mixed, injection_out, rows, hidden_size, hc_count, lora_rank,
          use_combine, bm_up, bh, warps, pdl):
    _hc_up_kernel[(triton.cdiv(rows, bm_up), hidden_size // bh)](
        source, *packed.tensors(), xn, mixed, injection_out, rows, stride_source, xn.stride(0), mixed.stride(0),
        packed.kb, packed.nt, H=hidden_size, HC=hc_count, R=lora_rank, HAS_INJ=use_combine, BM=bm_up, BH=bh,
        BK=HC_UP_BK, launch_pdl=pdl, num_warps=warps, num_stages=3)


def hc_supported(packed, hidden_size, bh):
    return hidden_size % ROW_TILE == 0 and bh % ROW_TILE == 0 and packed.k % HC_UP_BK == 0


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def summary():
    done = _STATE
    saved = done["orig"] - done["packed"]
    share = 100.0 * saved / done["orig"] if done["orig"] else 0.0
    return (f"BF16C packed {done['layers']} weights ({done['weights'] / 1e9:.3f} G values): "
            f"{done['orig'] / 2**30:.3f} GiB of BF16 became {done['packed'] / 2**30:.3f} GiB "
            f"({saved / 2**30:.3f} GiB freed, {share:.2f}%), {done['escapes']} escapes "
            f"({100.0 * done['escapes'] / max(1, done['weights']):.4f}% of values) in {done['seconds']:.2f} s; "
            f"decode rows <= {decode_rows()}, larger batches decode the layer into a transient buffer for the stock GEMM"
            + (f"; MIX: {_MIX['layers']} layers of {len(mix_shapes())} verified shapes run the fused decode GEMM "
               f"above {decode_rows()} rows where the card's verdict says it wins (escape heads "
               f"{_MIX['bytes'] / 2**20:.1f} MiB)" if mix_enabled() else ""))


def mix_shapes():
    return [shape for shape in (mix_verified() or {}) if mix_wanted(*shape)]


def _verify(weight, packed):
    want = bits_of(weight)
    if not torch.equal(bits_of(unpack(packed)), want):
        raise RuntimeError(f"BF16C pack of a {tuple(weight.shape)} weight does not unpack to the same bits")
    if weight.is_cuda or INTERPRETED:
        if not torch.equal(bits_of(decode(packed)), want):
            raise RuntimeError(f"BF16C kernel decode of a {tuple(weight.shape)} weight differs from the original")


def attach(layer, packed):
    for name, tensor in zip(NAMES, packed.tensors()):
        layer.register_buffer(f"bf16c_{name}", tensor, persistent=False)
    layer.taaf_bf16c = packed
    layer.weight = torch.nn.Parameter(torch.empty((0, packed.k), dtype=torch.bfloat16, device=packed.device),
                                      requires_grad=False)


def after_loading(method, layer):
    if hasattr(layer, "taaf_bf16c"):
        return True
    name = group_of(layer)
    if name is None:
        return False
    if type(method).__name__.startswith("Qwen4ExpLowLatency"):
        print(f"BF16C skipped {getattr(layer, 'prefix', '')}: the low-latency method reads layer.weight", flush=True)
        return False
    weight = layer.weight.data
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.numel() == 0:
        print(f"BF16C skipped {getattr(layer, 'prefix', '')}: {tuple(weight.shape)} {weight.dtype}", flush=True)
        return False
    started = _synchronized(weight.device)
    packed = pack(weight)
    _verify(weight, packed)
    before = _bytes(weight)
    on_gpu = weight.is_cuda
    attach(layer, packed)
    del weight
    _STATE["seconds"] += _synchronized(packed.device) - started
    _STATE["layers"] += 1
    _STATE["orig"] += before
    _STATE["packed"] += packed.nbytes()
    _STATE["escapes"] += packed.escapes()
    _STATE["weights"] += packed.n * packed.k
    if on_gpu and fast_enabled():
        fast_counters(packed)
    if on_gpu and mix_enabled() and name != "head" and mix_wanted(packed.n, packed.k):
        held = mix_index(packed)
        mix_counters(packed.device)
        _MIX["layers"] += 1
        _MIX["bytes"] += sum(_bytes(tensor) for tensor in held)
    if on_gpu and pipe_enabled():
        pipe_reserve(packed)
    if on_gpu and _STATE["layers"] % 8 == 0:
        torch.cuda.empty_cache()
    if _STATE["layers"] == 1 or _STATE["layers"] % 40 == 0:
        print(f"BF16C packed {name} {getattr(layer, 'prefix', '') or 'lm_head'} {packed.n}x{packed.k}: "
              f"{before / 2**20:.2f} MiB -> {packed.nbytes() / 2**20:.2f} MiB, {packed.escapes()} escapes, "
              f"round trip exact (torch unpack and kernel decode)", flush=True)
    return True


def pipe_reserve(packed):
    elements = packed.n * packed.k
    if elements > PIPE_ELEMENTS_MAX or elements <= _PIPE["numel"]:
        return
    _PIPE["numel"] = elements
    _PIPE["slots"] = [torch.empty(elements, dtype=torch.bfloat16, device=packed.device) for _ in range(2)]
    _PIPE["freed"] = [None, None]
    _PIPE["side"] = None


def _pipe_join(main):
    pending = _PIPE["pending"]
    if pending is not None:
        main.wait_event(pending[2])
        _PIPE["pending"] = None


def _pipe_view(slot, packed):
    return _PIPE["slots"][slot][: packed.n * packed.k].view(packed.n, packed.k)


def _pipe_fits(packed):
    return _PIPE["slots"][0] is not None and packed.n * packed.k <= _PIPE["numel"]


def _pipe_linear(x, packed):
    device = x.device
    main = torch.cuda.current_stream(device)
    me = packed.sm.data_ptr()
    if me in _PIPE["seen"]:
        _pipe_join(main)
        _PIPE["seen"] = set()
        _PIPE["last"] = None
        _PIPE["freed"] = [None, None]
    _PIPE["seen"].add(me)
    if _PIPE["last"] is not None:
        _PIPE["succ"][_PIPE["last"]] = me
    _PIPE["last"] = me
    pending = _PIPE["pending"]
    _pipe_join(main)
    if pending is not None and pending[0] == me:
        slot = pending[1]
    else:
        slot = _PIPE["held"].get("next", 0)
        if _PIPE["freed"][slot] is not None:
            main.wait_event(_PIPE["freed"][slot])
        decode(packed, out=_pipe_view(slot, packed))
    other = 1 - slot
    ahead = _PIPE["succ"].get(me)
    target = _PIPE["held"].get(ahead) if ahead is not None else None
    if target is not None and _pipe_fits(target):
        if _PIPE["side"] is None:
            _PIPE["side"] = torch.cuda.Stream(device)
        fork = _PIPE["freed"][other]
        if fork is None:
            fork = torch.cuda.Event()
            fork.record(main)
        side = _PIPE["side"]
        side.wait_event(fork)
        with torch.cuda.stream(side):
            decode(target, out=_pipe_view(other, target))
            done = torch.cuda.Event()
            done.record(side)
        _PIPE["pending"] = (ahead, other, done)
    _PIPE["held"]["next"] = other
    out = torch.nn.functional.linear(x, _pipe_view(slot, packed))
    freed = torch.cuda.Event()
    freed.record(main)
    _PIPE["freed"][slot] = freed
    return out


def _linear(
    x: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    base: torch.Tensor,
    esc_ptr: torch.Tensor,
    esc_n: torch.Tensor,
    esc_k: torch.Tensor,
    esc_v: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    packed = Packed((n, k), sm, ex, base, esc_ptr, esc_n, esc_k, esc_v)
    rows = x.numel() // x.shape[-1]
    if x.dtype == torch.bfloat16 and x.is_cuda and fast_enabled() and rows <= fast_rows(n, k):
        return fast_gemm(x, packed)
    if x.dtype == torch.bfloat16 and x.is_cuda and not INTERPRETED and mix_decode_enabled():
        config = mix_decode_pick(n, k, rows)
        if config is not None:
            out = mix_gemm(x, packed, config)
            if out is not None:
                return out
            _mix_told(("capture", n, k), f"meets {n}x{k} without its escape head inside a CUDA-graph capture")
    if x.dtype == torch.bfloat16 and rows <= decode_rows() and (x.is_cuda or INTERPRETED):
        return gemm(x, packed)
    if x.dtype == torch.bfloat16 and x.is_cuda and mix_enabled() and not INTERPRETED:
        config = mix_pick(n, k, rows)
        if config is not None:
            out = mix_gemm(x, packed, config)
            if out is not None:
                return out
            _mix_told(("capture", n, k), f"meets {n}x{k} without its escape head inside a CUDA-graph capture")
    if x.dtype == torch.bfloat16 and x.is_cuda and pipe_enabled() and _pipe_fits(packed):
        _PIPE["held"][packed.sm.data_ptr()] = packed
        return _pipe_linear(x, packed)
    return torch.nn.functional.linear(x, decode(packed))


def _linear_fake(
    x: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    base: torch.Tensor,
    esc_ptr: torch.Tensor,
    esc_n: torch.Tensor,
    esc_k: torch.Tensor,
    esc_v: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n))


def _register():
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        return False
    direct_register_custom_op(op_name="taaf_bf16c_linear", op_func=_linear, fake_impl=_linear_fake)
    return True


REGISTERED = _register() if __name__ != "__main__" else False


def apply(method, layer, x, bias=None):
    packed = layer.taaf_bf16c
    if not _STATE["told"]:
        _STATE["told"] = True
        print(summary(), flush=True)
        if x.is_cuda:
            torch.cuda.empty_cache()
    held = (*(getattr(layer, f"bf16c_{name}") for name in NAMES), packed.n, packed.k)
    out = torch.ops.vllm.taaf_bf16c_linear(x, *held) if REGISTERED else _linear(x, *held)
    return out if bias is None else out + bias


def scale_pack(scales):
    if scales.dtype != torch.float16:
        raise ValueError(f"scale_pack takes float16 scales, found {scales.dtype}")
    flat = scales.contiguous().view(torch.int16).reshape(-1).to(torch.int32) & 0xFFFF
    count = flat.numel()
    padded = -(-count // GROUP) * GROUP
    full = torch.zeros(padded, dtype=torch.int32, device=scales.device)
    full[:count] = flat
    exp = ((full >> 10) & 31).to(torch.int64).reshape(-1, GROUP)
    counts = torch.zeros(exp.shape[0], 33, dtype=torch.int32, device=scales.device)
    counts.scatter_add_(1, exp, torch.ones_like(exp, dtype=torch.int32))
    base = (counts[:, 0:31] + counts[:, 1:32]).argmax(1)
    d = exp - base[:, None]
    inside = ((d == 0) | (d == 1)).reshape(-1)
    code = d.clamp(0, 1).reshape(-1).to(torch.int32)
    nibble = ((full >> 8) & 3) | (((full >> 15) & 1) << 2) | (code << 3)
    nibble = torch.where(inside, nibble, (full >> 8) & 3 | (((full >> 15) & 1) << 2))
    lo = (full & 0xFF).to(torch.uint8)
    hi = (nibble[0::2] | (nibble[1::2] << 4)).to(torch.uint8)
    bad = (~inside).nonzero().reshape(-1)
    bad = bad[bad < count]
    return {"shape": tuple(scales.shape), "lo": lo, "hi": hi, "base": base.to(torch.uint8),
            "esc_at": bad.to(torch.int64), "esc_v": full[bad].to(torch.int32)}


def scale_unpack(held):
    lo = held["lo"].to(torch.int32)
    hi = held["hi"].to(torch.int32)
    nibble = torch.stack([hi & 15, hi >> 4], dim=-1).reshape(-1)
    base = held["base"].to(torch.int32).repeat_interleave(GROUP)
    exp = base + ((nibble >> 3) & 1)
    bits = ((nibble & 4) << 13) | (exp << 10) | ((nibble & 3) << 8) | lo
    if held["esc_at"].numel():
        bits[held["esc_at"]] = held["esc_v"]
    count = 1
    for dim in held["shape"]:
        count *= dim
    bits = bits[:count]
    return torch.where(bits > 32767, bits - 65536, bits).to(torch.int16).view(torch.float16).reshape(held["shape"])


def scale_nbytes(held):
    return sum(held[name].numel() * held[name].element_size() for name in ("lo", "hi", "base", "esc_at", "esc_v"))


def old_reference(x, packed, splits):
    n, k = packed.shape
    m = x.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    ws = torch.empty((splits * m * n if splits > 1 else 1,), dtype=torch.float32, device=x.device)
    config = default_config(m)
    _gemm_kernel[(triton.cdiv(n, config.kwargs["BN"]), triton.cdiv(m, config.kwargs["BM"]), splits)](
        x, out, ws, *packed.tensors(), m, n, k, x.stride(0), out.stride(0), packed.kb, packed.nt, bucket(m),
        BM=config.kwargs["BM"], BN=config.kwargs["BN"], BK=GEMM_BK, PF=0, SW=0, EVEN_K=k % GEMM_BK == 0,
        num_warps=config.num_warps, num_stages=config.num_stages)
    if splits > 1:
        _reduce_kernel[(triton.cdiv(m * n, 1024),)](ws, out, m * n, splits, BLOCK=1024)
    return out


def same_splits(k, splits):
    chunks = -(-k // GEMM_BK)
    groups = -(-k // GROUP)
    old = [min(chunks, s * -(-chunks // splits)) * GEMM_BK for s in range(splits + 1)]
    new = [min(groups, s * -(-groups // splits)) * GROUP for s in range(splits + 1)]
    return [min(k, at) for at in old] == [min(k, at) for at in new]


def check_weight(n, k, device):
    made = torch.Generator(device=device).manual_seed(n * 131 + k)
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device)
    for r0 in range(0, n, 8192):
        r1 = min(n, r0 + 8192)
        weight[r0:r1] = (torch.randn(r1 - r0, k, generator=made, device=device) * 0.02).to(torch.bfloat16)
        weight[r0:r1].view(torch.int16)[torch.rand(r1 - r0, k, generator=made, device=device) < CHECK_ESCAPES] = 0x4400
    return weight, made


def _check_shape(n, k, rows, timed_rows, device, mixed=MIXED_ROWS):
    weight, made = check_weight(n, k, device)
    packed = pack(weight)
    del weight
    sms = _sms(device)
    found = {"shape": [n, k], "escapes": packed.escapes(), "exact": True, "checked": 0}
    for m in rows:
        x = torch.randn(m, k, generator=made, device=device).to(torch.bfloat16)
        bt1, bt2 = token_tiles(m)
        split = 1 if m > FAST_ROWS else max([s for s in FAST_SPLITS if s <= -(-k // GROUP) and same_splits(k, s)][:6])
        for bn in FAST_BNS:
            if (bt1 + bt2) * bn > FAST_TILE_LIMIT:
                continue
            for pf in (0, 1):
                for splits in sorted({1, split}):
                    config = triton.Config({"BN": bn, "PF": pf, "SPLITS": splits},
                                           num_warps=fast_warps(bt1 + bt2, bn, pf), num_stages=1)
                    got = fast_gemm(x, packed, config=config)
                    want = old_reference(x, packed, splits)
                    found["checked"] += 1
                    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                        found["exact"] = False
                        found["first_mismatch"] = {"rows": m, "BN": bn, "PF": pf, "SPLITS": splits}
                        return found, packed
    x = torch.randn(timed_rows, k, generator=made, device=device).to(torch.bfloat16)
    gemm(x, packed)
    fast_gemm(x, packed)
    bt1, bt2 = token_tiles(timed_rows)
    tuned = _TUNED[("fast", n, k, bt1, bt2, 1, str(x.dtype), str(x.dtype))]
    splits = tuned.kwargs["SPLITS"]
    got = fast_gemm(x, packed)
    if same_splits(k, splits):
        tuned_exact = torch.equal(got.view(torch.int16), old_reference(x, packed, splits).view(torch.int16))
    else:
        exact = (x.float() @ unpack(packed).float().T)
        bound = (old_reference(x, packed, 1).float() - exact).abs().max().item()
        tuned_exact = (got.float() - exact).abs().max().item() <= 2 * bound + 1e-6
    found["exact"] = bool(found["exact"] and tuned_exact)
    found["tuned"] = _said(tuned)
    found["old_us"] = 1000 * triton.testing.do_bench(lambda: gemm(x, packed), warmup=5, rep=40)
    found["new_us"] = 1000 * triton.testing.do_bench(lambda: fast_gemm(x, packed), warmup=5, rep=40)
    found["rows"] = timed_rows
    found["bytes"] = packed.nbytes()
    found["sms"] = sms
    found["mixed"] = []
    found["rows_max"] = FAST_ROWS
    streak = True
    for m in (mixed or ()):
        x = torch.randn(m, k, generator=made, device=device).to(torch.bfloat16)
        got = fast_gemm(x, packed)
        if not torch.equal(got.view(torch.int16), old_reference(x, packed, 1).view(torch.int16)):
            found["exact"] = False
            found["first_mismatch"] = {"rows": m, "tuned": True}
            return found, packed
        old_linear(x, packed)
        stock = 1000 * triton.testing.do_bench(lambda: old_linear(x, packed), warmup=5, rep=40)
        fast = 1000 * triton.testing.do_bench(lambda: fast_gemm(x, packed), warmup=5, rep=40)
        found["mixed"].append({"rows": m, "stock_us": stock, "new_us": fast})
        streak = streak and fast < stock
        if streak:
            found["rows_max"] = m
    return found, packed


def selfcheck(path, shapes=SERVED_SHAPES, rows=CHECK_ROWS, timed_rows=TIMED_ROWS, margin=1.02, mixed=MIXED_ROWS):
    started = time.perf_counter()
    report = {"ok": False, "passed": [], "shapes": [], "rows": {}, "reason": ""}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        device = torch.device("cuda")
        report["device"] = torch.cuda.get_device_name(device)
        for n, k in shapes:
            found, packed = _check_shape(n, k, rows, timed_rows, device, mixed if n < 100_000 else ())
            del packed
            torch.cuda.empty_cache()
            report["shapes"].append(found)
            said = (f"BF16C FAST2 self-check {n}x{k}: {found['checked']} configs bit-exact against the old kernel"
                    if found["exact"] else f"BF16C FAST2 self-check {n}x{k}: MISMATCH {found.get('first_mismatch')}")
            if "new_us" in found:
                said += (f"; rows {found['rows']} old {found['old_us']:.1f} us new {found['new_us']:.1f} us "
                         f"({found['bytes'] / found['new_us'] / 1e3:.0f} GB/s, {found['tuned']})")
            for one in found.get("mixed", []):
                said += f"; rows {one['rows']} stock {one['stock_us']:.0f} us new {one['new_us']:.0f} us"
            if found.get("mixed"):
                said += f"; the new kernel serves up to {found['rows_max']} rows"
            print(said, flush=True)
            if not found["exact"]:
                report["reason"] = f"{n}x{k} differs from the old kernel: {found.get('first_mismatch', 'tuned config')}"
                break
            if found["new_us"] <= margin * found["old_us"]:
                report["passed"].append([n, k])
                report["rows"][f"{n}x{k}"] = found["rows_max"]
        else:
            old = sum(row["old_us"] for row in report["shapes"])
            new = sum(min(row["new_us"], row["old_us"]) for row in report["shapes"])
            report["old_us"], report["new_us"] = old, new
            report["ok"] = bool(report["passed"]) and new < old
            if not report["ok"]:
                report["reason"] = f"not faster: old {old:.1f} us, new {new:.1f} us over the served shapes"
    except Exception as failure:
        report["reason"] = f"{type(failure).__name__}: {failure}"
    report["seconds"] = round(time.perf_counter() - started, 1)
    _write_report(path, report)
    print(f"BF16C FAST2 self-check {'PASSED' if report['ok'] else 'FAILED'} in {report['seconds']} s: "
          f"{len(report['passed'])} of {len(shapes)} shapes take the new kernel"
          + (f"; {report['reason']}" if report["reason"] else ""), flush=True)
    return report


def spot_shape(shapes):
    return min(shapes, key=lambda shape: shape[0] * shape[1])


def fast_spot(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    passed = [tuple(shape) for shape in report.get("passed") or ()]
    if not report.get("ok") or not passed:
        print("BF16C FAST2 spot check: the cached verdict passes no shape", flush=True)
        return False
    device = torch.device("cuda")
    n, k = spot_shape(passed)
    weight, made = check_weight(n, k, device)
    packed = pack(weight)
    del weight
    x = torch.randn(TIMED_ROWS, k, generator=made, device=device).to(torch.bfloat16)
    gemm(x, packed)
    fast_gemm(x, packed)
    bt1, bt2 = token_tiles(TIMED_ROWS)
    tuned = _TUNED[("fast", n, k, bt1, bt2, 1, str(x.dtype), str(x.dtype))]
    splits = tuned.kwargs["SPLITS"]
    got = fast_gemm(x, packed)
    if same_splits(k, splits):
        exact = torch.equal(got.view(torch.int16), old_reference(x, packed, splits).view(torch.int16))
    else:
        reference = x.float() @ unpack(packed).float().T
        bound = (old_reference(x, packed, 1).float() - reference).abs().max().item()
        exact = (got.float() - reference).abs().max().item() <= 2 * bound + 1e-6
    print(f"BF16C FAST2 spot check {n}x{k} rows {TIMED_ROWS} ({_said(tuned)}): "
          f"{'matches the old kernel' if exact else 'MISMATCH'}", flush=True)
    return bool(exact)


def mix_spot(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows_of = dict(mix_bands())
    picks = []
    for shape, bands in (report.get("serve") or {}).items():
        n, k = (int(part) for part in shape.split("x"))
        for hi, said in bands:
            if said:
                picks.append((n, k, rows_of[int(hi)], said))
    if not report.get("ok") or not picks:
        print("BF16C MIX spot check: the cached verdict wins on no shape", flush=True)
        return False
    n, k, m, said = min(picks, key=lambda pick: pick[0] * pick[1])
    device = torch.device("cuda")
    weight, made = mix_weight(n, k, device)
    packed = pack(weight)
    mix_index(packed)
    if not torch.equal(bits_of(mix_decode(packed)), bits_of(weight)):
        print(f"BF16C MIX spot check {n}x{k}: the in-loop decode differs from the weight's bits", flush=True)
        return False
    x = torch.randn(m, k, generator=made, device=device).to(torch.bfloat16)
    ref = x.float() @ weight.float().T
    bound, mean_bound = _mix_errors(old_linear(x, packed), ref)
    config = mix_made(said)
    got = mix_gemm(x, packed, config)
    torch.cuda.synchronize()
    worst, mean = _mix_errors(got, ref)
    ok = worst <= 2 * bound + 1e-6 and mean <= 1.5 * mean_bound + 1e-6 and _mix_replays(x, packed, config, got)
    print(f"BF16C MIX spot check {n}x{k} rows {m} ({_said(config)}): max error {worst:.4g} (stock {bound:.4g}), "
          f"graph replay {'exact' if ok else 'or error bound FAILED'}", flush=True)
    return bool(ok)


def spot_check(kind, path):
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        return {"fast": fast_spot, "mix": mix_spot}[kind](path)
    except Exception as failure:
        print(f"BF16C spot check {kind} failed: {type(failure).__name__}: {failure}", flush=True)
        return False


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "spotcheck":
    sys.exit(0 if spot_check(sys.argv[2], sys.argv[3]) else 1)
if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "selfcheck":
    selfcheck(sys.argv[2])
if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "mixcheck":
    mix_selfcheck(sys.argv[2])
