import torch

ENV = "TAAF_VLLM_FP8_TRITON_BAND"
ROWS_MAX = 16384
TOLERANCE = 5e-3
CONFIGS = {"large_k": (64, 128, 128, 8, 4, 4), "small_k": (64, 128, 64, 8, 4, 4)}
LARGE_K = 4096
_STATE = {"entries": [], "bands": {}, "verdict": {}, "told": set(), "announced": False, "kernel": None,
          "quant": None}


def parse(said):
    bands = {}
    for part in str(said or "").replace(" ", "").split(","):
        if not part:
            continue
        shape, colon, span = part.lower().partition(":")
        n, cross, k = shape.partition("x")
        lo, dash, hi = span.partition("-")
        if not (colon and cross and dash and n.isdigit() and k.isdigit() and lo.isdigit() and hi.isdigit()):
            raise ValueError(f"{ENV} entry {part!r} is not <N>x<K>:<lo>-<hi>")
        n, k, lo, hi = int(n), int(k), int(lo), int(hi)
        if n <= 0 or k <= 0 or k % 16 or not 1 <= lo <= hi <= ROWS_MAX:
            raise ValueError(f"{ENV} entry {part!r} needs N>0, K a positive multiple of 16, and "
                             f"1 <= lo <= hi <= {ROWS_MAX}")
        held = bands.setdefault((n, k), [])
        if any(lo <= other_hi and other_lo <= hi for other_lo, other_hi in held):
            raise ValueError(f"{ENV} entry {part!r} overlaps another band of {n}x{k}")
        held.append((lo, hi))
    if not bands:
        raise ValueError(f"{ENV} names no band")
    return {shape: sorted(spans) for shape, spans in bands.items()}


def say(line):
    print(f"FP8_BAND {line}", flush=True)


def configure(said):
    _STATE["bands"] = parse(said)
    _STATE["verdict"] = {}
    _STATE["entries"] = []


def config_for(n, k):
    return CONFIGS["large_k" if k >= LARGE_K else "small_k"]


def fits(n, k):
    _, bn, bk, _, _, _ = config_for(n, k)
    return n % bn == 0 and k % bk == 0


def in_band(shape, rows):
    return any(lo <= rows <= hi for lo, hi in _STATE["bands"].get(shape, ()))


def kernel():
    if _STATE["kernel"] is not None:
        return _STATE["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def fp8_gemm(a_ptr, as_ptr, w_ptr, ws_ptr, out_ptr, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GROUP * grid_n
        first_m = (pid // width) * GROUP
        size_m = tl.minimum(grid_m - first_m, GROUP)
        pid_m = first_m + (pid % width) % size_m
        pid_n = (pid % width) // size_m
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        live_m = offs_m < M
        a_rows = a_ptr + offs_m[:, None].to(tl.int64) * K
        w_cols = w_ptr + offs_n[None, :].to(tl.int64) * K
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_rows + (k0 + offs_k)[None, :], mask=live_m[:, None], other=0.0)
            w = tl.load(w_cols + (k0 + offs_k)[:, None])
            acc = tl.dot(a, w, acc)
        sa = tl.load(as_ptr + offs_m, mask=live_m, other=0.0)
        sw = tl.load(ws_ptr + offs_n)
        acc = acc * sa[:, None] * sw[None, :]
        tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=live_m[:, None])

    _STATE["kernel"] = (fp8_gemm, triton)
    return _STATE["kernel"]


def quantize(x):
    if _STATE["quant"] is None:
        from vllm import _custom_ops as ops
        _STATE["quant"] = lambda value: ops.scaled_fp8_quant(value, use_per_token_if_dynamic=True)
    return _STATE["quant"](x)


def gemm(q, scales, weight, weight_scale, n, k, out=None):
    fp8_gemm, triton = kernel()
    bm, bn, bk, group, warps, stages = config_for(n, k)
    m = q.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=q.device)
    fp8_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        q, scales, weight, weight_scale, out, m, n, k, BM=bm, BN=bn, BK=bk, GROUP=group, num_warps=warps,
        num_stages=stages)
    return out


def reference(q, scales, weight, weight_scale):
    return ((q.float() @ weight.float().T) * scales.float().view(-1, 1) * weight_scale.float().view(1, -1)
            ).to(torch.bfloat16)


def _views(layer, n, k):
    weight = layer.weight
    if tuple(weight.shape) == (k, n) and weight.stride() == (1, k):
        weight = weight.t()
    if tuple(weight.shape) != (n, k) or weight.stride() != (k, 1) or weight.dtype != torch.float8_e4m3fn:
        return None
    scale = getattr(layer, "weight_scale", None)
    if scale is None or scale.numel() != n or scale.dtype != torch.float32 or not scale.is_contiguous():
        return None
    return weight, scale.view(-1)


def _told(key, line):
    if key not in _STATE["told"]:
        _STATE["told"].add(key)
        say(line)


def mark(layer, stock):
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2 or not _STATE["bands"]:
        return False
    shapes = [(n, k) for (n, k) in _STATE["bands"] if tuple(weight.shape) in ((k, n), (n, k))]
    if not shapes:
        return False
    n, k = shapes[0]
    if not fits(n, k):
        _told(("fit", n, k), f"{n}x{k} does not divide the {config_for(n, k)[1]}x{config_for(n, k)[2]} tile; "
                             f"the stock kernel serves it")
        return False
    if _views(layer, n, k) is None:
        _told(("layout", n, k), f"{n}x{k} weight or scale layout is not the FP8 per-channel one; the stock "
                                f"kernel serves it")
        return False
    _STATE["entries"].append({"layer": layer, "stock": stock, "n": n, "k": k})
    layer.taaf_fp8_band = len(_STATE["entries"]) - 1
    return True


def _capturing(device):
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def check_rows(shape):
    return sorted({row for lo, hi in _STATE["bands"][shape] for row in (lo, (lo + hi) // 2, hi)})


def check(entry, device):
    n, k = entry["n"], entry["k"]
    weight, scale = _views(entry["layer"], n, k)
    gen = torch.Generator(device=device).manual_seed(n * 131 + k)
    rows = check_rows((n, k))
    worst = 0.0
    try:
        for m in rows:
            x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
            q, scales = quantize(x)
            got = gemm(q, scales.view(-1), weight, scale, n, k).float()
            want = entry["stock"](x).float()
            gap = float((got - want).abs().max() / want.abs().max().clamp(min=1e-12))
            worst = max(worst, gap)
            if not gap <= TOLERANCE:
                return False, f"rel gap {gap:.5f} at {m} rows exceeds {TOLERANCE}"
        torch.cuda.synchronize(device)
    except Exception as failure:
        return False, f"{type(failure).__name__}: {str(failure)[:200]}"
    return True, f"rel gap {worst:.5f} <= {TOLERANCE} over rows {rows}"


def verdict_of(entry, device):
    shape = (entry["n"], entry["k"])
    if shape not in _STATE["verdict"]:
        ok, detail = check(entry, device)
        _STATE["verdict"][shape] = ok
        spans = ",".join(f"{lo}-{hi}" for lo, hi in _STATE["bands"][shape])
        say(f"self-check {shape[0]}x{shape[1]} band {spans}: {detail}; "
            + ("Triton serves the band" if ok else "stood aside, the stock kernel serves every row count"))
    return _STATE["verdict"][shape]


def _linear(x: torch.Tensor, key: int) -> torch.Tensor:
    entry = _STATE["entries"][key]
    n, k = entry["n"], entry["k"]
    if (x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] != k or x.numel() == 0
            or not x.is_contiguous() or x.data_ptr() % 16):
        return entry["stock"](x)
    rows = x.numel() // k
    if (n, k) not in _STATE["verdict"]:
        if _capturing(x.device):
            _told(("capture", n, k), f"met {n}x{k} inside a CUDA-graph capture before its self-check ran "
                                     f"eagerly: the stock kernel serves it")
            return entry["stock"](x)
        verdict_of(entry, x.device)
    if not _STATE["verdict"][(n, k)] or not in_band((n, k), rows):
        return entry["stock"](x)
    weight, scale = _views(entry["layer"], n, k)
    q, scales = quantize(x.reshape(-1, k))
    out = gemm(q, scales.view(-1), weight, scale, n, k)
    if not _STATE["announced"]:
        _STATE["announced"] = True
        say(f"first use: {n}x{k} at {rows} rows took Triton {config_for(n, k)}; "
            f"{len(_STATE['entries'])} layers marked")
    return out.reshape(*x.shape[:-1], n)


def _linear_fake(x: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _STATE["entries"][key]["n"]))


def _register():
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        return False
    direct_register_custom_op(op_name="taaf_fp8_band_linear", op_func=_linear, fake_impl=_linear_fake)
    return True


REGISTERED = _register() if __name__ != "__main__" else False


def apply(layer, x):
    if REGISTERED:
        return torch.ops.vllm.taaf_fp8_band_linear(x, layer.taaf_fp8_band)
    return _linear(x, layer.taaf_fp8_band)
