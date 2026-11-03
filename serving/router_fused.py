import os

import torch
import triton
import triton.language as tl


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

MAX_EXPERTS = 4096
MAX_TOPK = 64
TOLERANCE = 1e-5
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
ENV = "TAAF_VLLM_ROUTER_FUSED"
ON = str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")
_STATE = {"checked": False, "disabled": ""}
_ANNOUNCED = set()


def say(event, **fields):
    detail = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"ROUTER_FUSED {event} {detail}".rstrip(), flush=True)


def announce(event, key="", **fields):
    if (event, key) not in _ANNOUNCED:
        _ANNOUNCED.add((event, key))
        say(event, **fields)


@triton.jit
def _router_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    sources_ptr,
    pad_ptr,
    rows,
    stride_logits,
    stride_weights,
    stride_ids,
    stride_sources,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_PAD: tl.constexpr,
    PDL: tl.constexpr = False,
) -> None:
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    valid = offs < E
    slots = tl.arange(0, BLOCK_K)
    x = tl.load(logits_ptr + row * stride_logits + offs, mask=valid, other=float("-inf"))
    x = x.to(tl.float32)
    peak = tl.max(x, axis=0)
    e = tl.where(valid, tl.exp(x - peak), 0.0)
    total = tl.sum(e, axis=0)
    scores = tl.where(valid, e * (1.0 / total), -1.0)
    picked = tl.zeros((BLOCK_K,), dtype=tl.float32)
    index = tl.zeros((BLOCK_K,), dtype=tl.int32)
    chosen = tl.full((), 0.0, tl.float32)
    for k in tl.static_range(K):
        best = tl.max(scores, axis=0)
        at = tl.min(tl.where(scores == best, offs, BLOCK_E), axis=0)
        picked = tl.where(slots == k, best, picked)
        index = tl.where(slots == k, at, index)
        chosen += best
        scores = tl.where(offs == at, -2.0, scores)
    if RENORM:
        picked = picked / tl.where(chosen > 0.0, chosen, 1.0)
    if HAS_PAD:
        padded = tl.load(pad_ptr + row) != 0
        index = tl.where(padded, E, index)
    keep = slots < K
    tl.store(weights_ptr + row * stride_weights + slots, picked, mask=keep)
    tl.store(ids_ptr + row * stride_ids + slots, index.to(ids_ptr.dtype.element_ty), mask=keep)
    tl.store(sources_ptr + row * stride_sources + slots, slots * rows + row, mask=keep)


def warps_for(block):
    return max(1, min(4, block // 1024))


def refusal(weights, ids, sources, logits, padding):
    if not (logits.is_cuda or INTERPRETED):
        return "device"
    if logits.dim() != 2 or ids.dim() != 2 or weights.dim() != 2 or sources.dim() != 2:
        return "shape"
    rows, experts = logits.shape
    if rows < 1:
        return "rows"
    if experts > MAX_EXPERTS:
        return f"experts>{MAX_EXPERTS}"
    topk = ids.shape[1]
    if topk < 1 or topk > min(experts, MAX_TOPK):
        return "topk"
    if tuple(weights.shape) != (rows, topk) or tuple(ids.shape) != (rows, topk) \
            or tuple(sources.shape) != (rows, topk):
        return "outputs"
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return "dtype"
    if weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64) \
            or sources.dtype != torch.int32:
        return "output-dtype"
    if any(t.stride(1) != 1 for t in (logits, weights, ids, sources)):
        return "layout"
    if padding is not None and (padding.dim() != 1 or padding.shape[0] < rows
                                or padding.dtype != torch.bool):
        return "padding"
    return ""


def launch(weights, ids, sources, logits, renormalize, padding):
    rows, experts = logits.shape
    topk = ids.shape[1]
    block = triton.next_power_of_2(experts)
    pad = padding.view(torch.uint8) if padding is not None else logits
    _router_topk_kernel[(rows,)](
        logits,
        weights,
        ids,
        sources,
        pad,
        rows,
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        sources.stride(0),
        E=experts,
        K=topk,
        BLOCK_E=block,
        BLOCK_K=triton.next_power_of_2(topk),
        RENORM=bool(renormalize),
        HAS_PAD=padding is not None,
        num_warps=warps_for(block),
        PDL=pdl_on(), launch_pdl=pdl_on(),
    )
    return weights, ids


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def compare(got_weights, got_ids, want_weights, want_ids, padding):
    rows = got_ids.shape[0]
    real = torch.ones(rows, dtype=torch.bool, device=got_ids.device)
    if padding is not None:
        real = ~padding[:rows]
    same = (got_ids.long() == want_ids.long()).all(dim=1)
    scale = want_weights[real].abs().max().clamp_min(1e-30) if bool(real.any()) else None
    worst = 0.0
    if scale is not None:
        worst = float(((got_weights - want_weights).abs()[real]).max() / scale)
    return {"rows": rows, "real": int(real.sum()), "ids_differ": int((~same & real).sum()),
            "padded_ids_differ": int((~same & ~real).sum()), "weights_maxrel": worst}


def check(stock, weights, ids, sources, logits, renormalize, padding):
    _STATE["checked"] = True
    try:
        want_weights, want_ids = stock(torch.empty_like(weights), torch.empty_like(ids),
                                       torch.empty_like(sources), _dense(logits), renormalize)
        got_weights, got_ids = launch(torch.empty_like(weights), torch.empty_like(ids),
                                      torch.empty_like(sources), logits, renormalize, padding)
        row = compare(got_weights, got_ids, want_weights, want_ids, padding)
    except Exception as exc:
        _STATE["disabled"] = f"{type(exc).__name__}:{exc}"[:300]
        say("refuse", rows=logits.shape[0], error=_STATE["disabled"])
        return
    say("check", rows=row["rows"], real=row["real"], experts=logits.shape[1], topk=ids.shape[1],
        renormalize=int(bool(renormalize)), ids_differ=row["ids_differ"],
        padded_ids_differ=row["padded_ids_differ"], weights_maxrel=f"{row['weights_maxrel']:.2e}",
        tolerance=TOLERANCE)
    if row["ids_differ"] or not row["weights_maxrel"] <= TOLERANCE:
        _STATE["disabled"] = (f"ids_differ={row['ids_differ']} "
                              f"weights_maxrel={row['weights_maxrel']:.2e}")
        say("refuse", reason=_STATE["disabled"])


def _dense(logits):
    if logits.dim() == 2 and logits.stride(0) != logits.shape[1]:
        return logits.contiguous()
    return logits


def topk_softmax(stock, weights, ids, sources, logits, renormalize, padding):
    reason = ("off" if not ON else "") or _STATE["disabled"] or refusal(
        weights, ids, sources, logits, padding)
    if reason:
        announce("stock", key=reason[:40], reason=reason, rows=logits.shape[0] if logits.dim() else 0)
        return stock(weights, ids, sources, _dense(logits), renormalize)
    capturing = _capturing()
    if not _STATE["checked"]:
        if capturing:
            announce("stock", key="unchecked-in-capture", reason="unchecked-in-capture",
                     rows=logits.shape[0])
            return stock(weights, ids, sources, _dense(logits), renormalize)
        check(stock, weights, ids, sources, logits, renormalize, padding)
        if _STATE["disabled"]:
            return stock(weights, ids, sources, _dense(logits), renormalize)
    announce("fired", key=f"{logits.shape[1]}-{ids.shape[1]}-{int(capturing)}",
             experts=logits.shape[1], topk=ids.shape[1], rows=logits.shape[0],
             renormalize=int(bool(renormalize)), padding=int(padding is not None),
             capturing=int(capturing), warps=warps_for(triton.next_power_of_2(logits.shape[1])))
    return launch(weights, ids, sources, logits, renormalize, padding)
