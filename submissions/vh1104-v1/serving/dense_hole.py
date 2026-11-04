import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

ENV = "TAAF_VLLM_DENSE_HOLE"
VERDICT_ENV = "TAAF_VLLM_DENSE_HOLE_VERDICT"
ROWS_ENV = "TAAF_VLLM_DENSE_HOLE_ROWS"
WAIT_ENV = "TAAF_VLLM_DENSE_HOLE_WAIT_S"
SHAPES_ENV = "TAAF_VLLM_DENSE_HOLE_SHAPES"
STOCK_SHAPES = ((2560, 6144),)


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def shapes_from(said):
    found = []
    for part in str(said or "").replace(" ", "").split(","):
        if not part:
            continue
        n, _, k = part.lower().partition("x")
        if not (n.isdigit() and k.isdigit()) or int(n) <= 0 or int(k) <= 0:
            raise ValueError(f"{SHAPES_ENV} entry {part!r} is not <out>x<in>")
        if (int(n), int(k)) not in found:
            found.append((int(n), int(k)))
    return tuple(found) or STOCK_SHAPES


SHAPES = shapes_from(os.environ.get(SHAPES_ENV))
ROWS = (16, 32, 64, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 896, 1024, 1152,
        1280, 1408, 1536, 1664, 1792, 1920, 2048)
ROWS_MIN = 16
ROWS_MAX = 16384
TILES = ((128, 128, 64, 8, 3), (128, 128, 64, 4, 3), (128, 128, 32, 4, 4), (256, 128, 32, 8, 3),
         (128, 256, 32, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (64, 64, 64, 4, 4), (32, 64, 128, 4, 4),
         (16, 64, 128, 4, 4))
SPLITS = (1, 2, 3, 4, 6, 8, 12, 16)
MARGIN = 1.03
WORKSPACE_MAX = 8 << 20
SPLIT_ACC = 64
COUNTERS = 1 << 14
COPIES = 6
LAUNCHES = 12
REPLAYS = 12
KEYS = ("BM", "BN", "BK", "warps", "stages", "SPLITS")
TIMERS_MODULE = "vllm.taaf_step_timers"
COUNTER = "dense_hole"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
_STATE = {"verdict": None, "marked": 0, "announced": False, "warm": set(), "buffers": {}, "told": set(),
          "aside": None, "counted": False}


@triton.jit(do_not_specialize=["M", "SPLITS"])
def _hole_kernel(x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr, tally_ptr, M, N, K, stride_xm, stride_wn, stride_om, SPLITS,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    tiles_m = (M + BM - 1) // BM
    pid_m = pid % tiles_m
    pid_n = pid // tiles_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    rm = offs_m % M
    rn = offs_n % N
    rk = tl.arange(0, BK)
    blocks = K // BK
    per = (blocks + SPLITS - 1) // SPLITS
    lo = pid_s * per
    hi = tl.minimum(lo + per, blocks)
    xp = x_ptr + rm.to(tl.int64)[:, None] * stride_xm + (lo * BK + rk)[None, :]
    wp = w_ptr + rn.to(tl.int64)[None, :] * stride_wn + (lo * BK + rk)[:, None]
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for _ in range(lo, hi):
        acc = tl.dot(tl.load(xp), tl.load(wp), acc)
        xp += BK
        wp += BK
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    rows = offs_m.to(tl.int64)[:, None]
    if SPLIT:
        plane = M * N
        held = ws_ptr + rows * N + offs_n[None, :]
        tl.store(held + pid_s * plane, acc, mask=mask)
        tl.debug_barrier()
        seen = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if seen == SPLITS - 1:
            acc = tl.full((BM, BN), 0.0, tl.float32)
            for _ in range(SPLITS):
                acc += tl.load(held, mask=mask, other=0.0, cache_modifier=".cg")
                held += plane
            tl.store(out_ptr + rows * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask)
            tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + rows * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask)
    if pid + pid_s == 0:
        tl.atomic_add(tally_ptr, 1)
        tl.atomic_add(tally_ptr + 1, M)


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def say(line):
    print(f"DENSE_HOLE {line}", flush=True)


def config_of(said):
    return tuple(int(said[key]) for key in KEYS)


def said_of(config):
    return dict(zip(KEYS, config))


def label(config):
    bm, bn, bk, warps, stages, splits = config
    return f"{bm}x{bn}x{bk} w{warps} s{stages} k{splits}"


def served_rows():
    said = str(os.environ.get(ROWS_ENV) or "").strip()
    if not said:
        return list(ROWS)
    rows = []
    for part in said.split(","):
        part = part.strip()
        if not part.isdigit() or not ROWS_MIN <= int(part) <= ROWS_MAX:
            raise ValueError(f"{ROWS_ENV} must list row counts from {ROWS_MIN} through {ROWS_MAX}, found {part!r}")
        rows.append(int(part))
    return sorted(set(rows))


def fits(config, m, n, k, sms):
    bm, bn, bk, warps, stages, splits = config
    if k % bk or bn > max(32, triton.next_power_of_2(n)) or bm > max(16, triton.next_power_of_2(m)):
        return False
    if splits == 1:
        return True
    if bm * bn > SPLIT_ACC * 32 * warps:
        return False
    blocks = k // bk
    per = -(-blocks // splits)
    tiles = -(-m // bm) * -(-n // bn)
    smaller = [s for s in SPLITS if s < splits]
    return (tiles < 2 * sms and per >= 2 and (splits - 1) * per < blocks and tiles * splits <= 4 * sms
            and tiles <= COUNTERS and splits * m * n <= WORKSPACE_MAX
            and all(-(-blocks // s) != per for s in smaller))


def space(m, n, k, sms):
    return [(*tile, splits) for tile in TILES for splits in SPLITS if fits((*tile, splits), m, n, k, sms)]


def make_buffers(device, workspace):
    return (torch.empty(max(1, workspace), dtype=torch.float32, device=device),
            torch.zeros(COUNTERS, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int64, device=device))


def gemm(x, weight, config, buffers):
    n, k = weight.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out.reshape(*lead, n)
    bm, bn, bk, warps, stages, splits = config
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    workspace, counters, tally = buffers
    if splits > 1 and (splits * m * n > workspace.numel() or tiles > counters.numel()):
        return None
    _hole_kernel[(tiles, splits)](x2, weight, out, workspace, counters, tally, m, n, k, x2.stride(0),
                                  weight.stride(0), out.stride(0), splits, BM=bm, BN=bn, BK=bk, SPLIT=splits > 1,
                                  num_warps=warps, num_stages=stages)
    return out.reshape(*lead, n)


def _capturing(device):
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _verdict_text(path):
    limit = float(os.environ.get(WAIT_ENV) or 0)
    target = Path(path)
    waited = 0.0
    while not target.is_file() and waited < limit:
        time.sleep(0.5)
        waited += 0.5
    if waited:
        say(f"waited {waited:.0f} s for the verdict {target.name}")
    return target.read_text(encoding="utf-8")


def stand_aside(reason, failed=False):
    if failed:
        selfcheck_fail(ENV, reason)
    if _STATE["aside"] is None:
        _STATE["aside"] = reason
        say(f"stood aside: {reason}; cuBLAS serves every row count")
    _STATE["verdict"] = {}


def verified():
    if _STATE["verdict"] is not None:
        return _STATE["verdict"]
    if not enabled():
        stand_aside(f"{ENV} is off in this process")
        return _STATE["verdict"]
    path = str(os.environ.get(VERDICT_ENV) or "").strip()
    if not path:
        stand_aside(f"no verdict path in {VERDICT_ENV}")
        return _STATE["verdict"]
    try:
        held = json.loads(_verdict_text(path))
        if not held.get("ok"):
            stand_aside(f"the self-check failed: {held.get('reason') or 'no reason given'}", failed=True)
            return _STATE["verdict"]
        table = {}
        for shape, bands in (held.get("serve") or {}).items():
            n, k = (int(part) for part in shape.split("x"))
            table[(n, k)] = sorted((int(rows), config_of(said) if said else None) for rows, said in bands)
        _STATE["verdict"] = table
    except (OSError, ValueError, KeyError, TypeError) as failure:
        stand_aside(f"verdict {path} unreadable ({type(failure).__name__}: {failure})")
    return _STATE["verdict"]


def pick(n, k, rows):
    bands = verified().get((n, k)) or ()
    below = [band for band in bands if band[0] <= rows]
    above = [band for band in bands if band[0] >= rows]
    if not below or not above:
        return None
    if below[-1][1] is None or above[0][1] is None:
        return None
    return above[0][1]


def chosen():
    first = {}
    for bands in verified().values():
        for rows, config in bands:
            if config is not None:
                first[config] = min(rows, first.get(config, rows))
    return sorted(first.items())


def workspace_need():
    need = 0
    for (n, _), bands in verified().items():
        for rows, config in bands:
            if config is not None and config[-1] > 1:
                need = max(need, config[-1] * rows * n)
    return need


def _key(device):
    if device.index is not None:
        return device.index
    return torch.cuda.current_device() if device.type == "cuda" else -1


def buffers(device, make=True):
    key = _key(device)
    held = _STATE["buffers"].get(key)
    if held is None and make:
        held = make_buffers(device, workspace_need())
        _STATE["buffers"][key] = held
        timers = sys.modules.get(TIMERS_MODULE)
        if callable(getattr(timers, "counter", None)):
            timers.counter(COUNTER, ("calls", "rows"), held[2])
            _STATE["counted"] = True
    return held


def tally(device=None):
    held = _STATE["buffers"]
    if not held:
        return None
    if device is None:
        return next(iter(held.values()))[2]
    key = _key(device)
    return held[key][2] if key in held else None


def _told(key, line):
    if key not in _STATE["told"]:
        _STATE["told"].add(key)
        say(line)


def after_loading(layer):
    weight = getattr(layer, "weight", None)
    if (weight is None or hasattr(layer, "taaf_bf16c") or getattr(layer, "bias", None) is not None
            or not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or weight.dim() != 2
            or tuple(weight.shape) not in SHAPES or weight.stride() != (weight.shape[1], 1)
            or not (weight.is_cuda or INTERPRETED)):
        return False
    layer.taaf_dense_hole = True
    _STATE["marked"] += 1
    n, k = weight.shape
    if _STATE["marked"] == 1 or _STATE["marked"] % 16 == 0:
        say(f"marked {_STATE['marked']} layer(s), the latest {getattr(layer, 'prefix', '') or '?'} {n}x{k}: its GEMM "
            f"goes through the hole op, which runs the split-K Triton BF16 GEMM at the row counts the self-check "
            f"proved faster and cuBLAS at every other")
    return True


def _announce(x, n, k, rows, config):
    if _STATE["announced"]:
        return
    _STATE["announced"] = True
    bands = verified().get((n, k)) or ()
    wins = [f"{m}:{label(config_)}" for m, config_ in bands if config_ is not None]
    held = buffers(x.device, make=False)
    say(f"first use: on, {_STATE['marked']} layers marked, {n}x{k} at {rows} rows took "
        f"{'Triton ' + label(config) if config is not None else 'cuBLAS'}; "
        + (f"stood aside ({_STATE['aside']})" if _STATE["aside"] else
           f"{len(wins)} of {len(bands)} checked row counts take Triton ({', '.join(wins) or 'none'})")
        + (f"; workspace {held[0].numel() * 4 / 2**20:.1f} MiB" if held is not None else "")
        + (f"; Triton calls and rows counted in the step timers as {COUNTER}" if _STATE["counted"] else
           "; no step timers to count the Triton calls"))


def _warm(x, weight, held):
    whole = x.reshape(-1, weight.shape[1])
    for config, rows in chosen():
        if config in _STATE["warm"]:
            continue
        if gemm(whole[:max(1, min(rows, whole.shape[0]))], weight, config, held) is not None:
            _STATE["warm"].add(config)


def _hole(x, weight, n, k, rows):
    if not (x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16) and (x.is_cuda or INTERPRETED)):
        return None, None
    try:
        config = pick(n, k, rows) if _STATE["aside"] is None else None
        wanted = _STATE["aside"] is None and bool(chosen())
    except Exception as failure:
        stand_aside(f"verdict read failed ({type(failure).__name__}: {failure})")
        return None, None
    if not wanted:
        return None, None
    standard = x.is_contiguous() and x.data_ptr() % 16 == 0 and rows > 0
    try:
        capturing = _capturing(x.device)
        if standard and not capturing:
            _warm(x, weight, buffers(x.device))
        if config is None:
            return None, None
        if not standard:
            _told(("layout", rows), f"met {n}x{k} at {rows} rows with a strided or unaligned input: cuBLAS runs "
                                    f"there")
            return None, None
        held = buffers(x.device, make=not capturing)
        if held is None or (capturing and config not in _STATE["warm"]):
            _told(("capture", rows), f"met {n}x{k} at {rows} rows inside a CUDA-graph capture before an eager call "
                                     f"built its buffers and compiled its tile: cuBLAS runs there")
            return None, None
        out = gemm(x, weight, config, held)
        if out is None:
            _told(("room", rows), f"{n}x{k} at {rows} rows needs more split-K workspace than the verdict sized: "
                                  f"cuBLAS runs there")
            return None, None
        return out, config
    except Exception as failure:
        stand_aside(f"the kernel failed at {rows} rows ({type(failure).__name__}: {str(failure)[:200]})",
                    failed=True)
        return None, None


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    rows = x.numel() // k if k else 0
    out, config = _hole(x, weight, n, k, rows)
    _announce(x, n, k, rows, config)
    return out if out is not None else torch.nn.functional.linear(x, weight)


def _linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _register():
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        return False
    direct_register_custom_op(op_name="taaf_dense_hole_linear", op_func=_linear, fake_impl=_linear_fake)
    return True


REGISTERED = _register() if __name__ != "__main__" else False


def apply(layer, x):
    if REGISTERED:
        return torch.ops.vllm.taaf_dense_hole_linear(x, layer.weight)
    return _linear(x, layer.weight)


def _sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else 1


def errors(got, ref):
    err = (got.float() - ref).abs()
    return err.max().item(), err.mean().item()


def equal_share(got, want):
    return float((got.view(torch.int16) == want.view(torch.int16)).float().mean().item())


def timed(call, copies):
    if not torch.cuda.is_available():
        began = time.perf_counter()
        for i in range(LAUNCHES):
            call(i % copies)
        return (time.perf_counter() - began) * 1e6 / LAUNCHES
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(copies):
            call(i)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(LAUNCHES):
            call(i % copies)
    spans = []
    for repeat in range(REPLAYS + 2):
        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        graph.replay()
        ended.record()
        ended.synchronize()
        if repeat >= 2:
            spans.append(began.elapsed_time(ended) * 1e3 / LAUNCHES)
    del graph
    return statistics.median(spans)


def replays_exact(x, weight, config, held, want):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        gemm(x, weight, config, held)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gemm(x, weight, config, held)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    exact = out is not None and torch.equal(out.view(torch.int16), want.view(torch.int16))
    del graph
    return exact


def check_shape(n, k, rows, device, margin=MARGIN):
    gen = torch.Generator(device=device).manual_seed(n * 131 + k)
    weights = [(torch.randn(n, k, generator=gen, device=device) * 0.02).to(torch.bfloat16) for _ in range(COPIES)]
    whole = torch.randn(max(rows), k, generator=gen, device=device).to(torch.bfloat16)
    held = make_buffers(device, WORKSPACE_MAX)
    sms = _sms(device)
    found = {"shape": [n, k], "bands": []}
    for m in rows:
        x = whole[:m]
        ref = x.float() @ weights[0].float().T
        want = ref.to(torch.bfloat16)
        stock = torch.nn.functional.linear(x, weights[0])
        bound, mean_bound = errors(stock, ref)
        best = None
        tried = 0
        for config in space(m, n, k, sms):
            try:
                got = gemm(x, weights[0], config, held)
                torch.cuda.synchronize()
            except Exception as failure:
                if "out of resource" in str(failure).lower() or type(failure).__name__ == "OutOfResources":
                    continue
                raise
            worst, mean = errors(got, ref)
            if worst > 2 * bound + 1e-6 or mean > 1.5 * mean_bound + 1e-6:
                found["failure"] = (f"rows {m} {label(config)}: max error {worst:.4g} (cuBLAS {bound:.4g}), mean "
                                    f"{mean:.4g} (cuBLAS {mean_bound:.4g})")
                return found
            spent = timed(lambda i: gemm(x, weights[i], config, held), COPIES)
            tried += 1
            if best is None or spent < best[0]:
                best = (spent, config, worst, got)
        stock_us = min(timed(lambda i: torch.nn.functional.linear(x, weights[i]), COPIES) for _ in range(2))
        band = {"rows": m, "stock_us": round(stock_us, 2), "stock_max_err": bound,
                "stock_equal": round(equal_share(stock, want), 5), "tried": tried, "wins": False, "config": None}
        if best is not None:
            config = best[1]
            hole_us = min(timed(lambda i: gemm(x, weights[i], config, held), COPIES) for _ in range(2))
            band.update({"hole_us": round(hole_us, 2), "config": said_of(config), "max_err": best[2],
                         "equal": round(equal_share(best[3], want), 5), "wins": bool(hole_us * margin < stock_us)})
            if band["wins"]:
                again = gemm(x, weights[0], config, held)
                torch.cuda.synchronize()
                if not torch.equal(again.view(torch.int16), best[3].view(torch.int16)):
                    found["failure"] = f"rows {m} {label(config)}: two eager runs differ"
                    return found
                if not replays_exact(x, weights[0], config, held, best[3]):
                    found["failure"] = f"rows {m} {label(config)}: a CUDA-graph replay differs from the eager result"
                    return found
        found["bands"].append(band)
    return found


def line(found):
    n, k = found["shape"]
    parts = []
    for band in found["bands"]:
        if band["config"] is None:
            parts.append(f"{band['rows']}: cuBLAS {band['stock_us']:.1f} us, no tile fits")
            continue
        parts.append(f"{band['rows']}: cuBLAS {band['stock_us']:.1f} Triton {band['hole_us']:.1f} us "
                     f"x{band['stock_us'] / band['hole_us']:.2f} {label(config_of(band['config']))} "
                     f"{'WINS' if band['wins'] else 'keeps cuBLAS'} (equal to the rounded fp32 sum "
                     f"{100 * band['equal']:.2f}% vs cuBLAS {100 * band['stock_equal']:.2f}%)")
    said = f"self-check {n}x{k}: " + "; ".join(parts)
    if found.get("failure"):
        said += f"; FAILED: {found['failure']}"
    return said


def write_report(path, report):
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(temporary, target)


def selfcheck(path, rows=None, shapes=None, margin=MARGIN):
    shapes = shapes or SHAPES
    started = time.perf_counter()
    report = {"ok": False, "shapes": [], "serve": {}, "reason": ""}
    try:
        rows = sorted(set(rows or served_rows()))
        report["rows"] = rows
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        report["device"] = torch.cuda.get_device_name(device)
        for n, k in shapes:
            found = check_shape(n, k, rows, device, margin)
            report["shapes"].append(found)
            say(line(found))
            if found.get("failure"):
                report["reason"] = f"{n}x{k}: {found['failure']}"
                break
            report["serve"][f"{n}x{k}"] = [[band["rows"], band["config"] if band["wins"] else None]
                                           for band in found["bands"]]
        else:
            wins = sum(said is not None for bands in report["serve"].values() for _, said in bands)
            report["ok"] = wins > 0
            if not wins:
                report["reason"] = "cuBLAS is at least as fast at every checked row count"
    except Exception as failure:
        report["reason"] = f"{type(failure).__name__}: {failure}"
    report["seconds"] = round(time.perf_counter() - started, 1)
    write_report(path, report)
    wins = sum(said is not None for bands in report["serve"].values() for _, said in bands)
    total = sum(len(bands) for bands in report["serve"].values())
    say(f"self-check {'PASSED' if report['ok'] else 'FAILED'} in {report['seconds']} s: {wins} of {total} row "
        f"counts take the Triton kernel" + (f"; {report['reason']}" if report["reason"] else ""))
    return report


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "selfcheck":
    selfcheck(sys.argv[2])
