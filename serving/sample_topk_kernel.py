import torch
import triton
import triton.language as tl

SEG = 64
SEGS = 32
LOW = -(2 ** 62)


@triton.jit
def _ordered(x):
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    flipped = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return (flipped >> 16).to(tl.int64)


@triton.jit
def _segment_max_kernel(logits_ptr, logits_stride, maxima_ptr, maxima_stride, vocab, nseg,
                        SEG: tl.constexpr, SEGS: tl.constexpr):
    row = tl.program_id(0)
    first = tl.program_id(1) * SEGS
    segs = first + tl.arange(0, SEGS)
    cols = segs[:, None] * SEG + tl.arange(0, SEG)[None, :]
    x = tl.load(logits_ptr + row * logits_stride + cols, mask=cols < vocab, other=float("-inf"))
    tl.store(maxima_ptr + row * maxima_stride + segs, tl.max(x.to(tl.float32), axis=1), mask=segs < nseg)


@triton.jit
def _pick_kernel(logits_ptr, logits_stride, maxima_ptr, maxima_stride, values_ptr, columns_ptr, out_stride,
                 vocab, nseg, SEG: tl.constexpr, PADDED: tl.constexpr, PICK: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, PADDED)
    maxima = tl.load(maxima_ptr + row * maxima_stride + offs, mask=offs < nseg, other=float("-inf"))
    seg_key = tl.where(offs < nseg, _ordered(maxima) << 32, -(2 ** 62)) | (0xFFFFFFFF - offs.to(tl.int64))
    segs = (0xFFFFFFFF - (tl.topk(seg_key, PICK) & 0xFFFFFFFF)).to(tl.int32)
    cols = segs[:, None] * SEG + tl.arange(0, SEG)[None, :]
    x = tl.load(logits_ptr + row * logits_stride + cols, mask=cols < vocab, other=float("-inf"))
    key = tl.where(cols < vocab, _ordered(x) << 32, -(2 ** 62)) | (0xFFFFFFFF - cols.to(tl.int64))
    best = tl.topk(tl.reshape(key, [PICK * SEG]), K)
    ordered = (best >> 32).to(tl.int32) << 16
    bits = tl.where(ordered < 0, (ordered ^ 0x7FFFFFFF) & -65536, ordered)
    tl.store(values_ptr + row * out_stride + tl.arange(0, K), bits.to(tl.float32, bitcast=True))
    tl.store(columns_ptr + row * out_stride + tl.arange(0, K), 0xFFFFFFFF - (best & 0xFFFFFFFF))


def topk(logits, k):
    rows, vocab = logits.shape
    nseg = triton.cdiv(vocab, SEG)
    maxima = torch.empty((rows, nseg), dtype=torch.float32, device=logits.device)
    _segment_max_kernel[(rows, triton.cdiv(nseg, SEGS))](logits, logits.stride(0), maxima, maxima.stride(0), vocab,
                                                       nseg, SEG=SEG, SEGS=SEGS, num_warps=4)
    padded = triton.next_power_of_2(nseg)
    values = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    columns = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    _pick_kernel[(rows,)](logits, logits.stride(0), maxima, maxima.stride(0), values, columns, k, vocab, nseg,
                          SEG=SEG, PADDED=padded, PICK=min(k, padded), K=k, num_warps=8)
    return values, columns
