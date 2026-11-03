import os
import sys
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "TAAF_VLLM_ROUTER_CHAIN"
LEVELS = {"gate": 1, "gate_up": 2}
VALUES = ("", "0", "1", "gate", "gate_up")
ALIGN = 8
PROBE_ROWS = 64
PROBE_SEED = 20261024
TOLERANCE = 2.0 ** -6
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70
UNSTRIDED_ENVS = ("TAAF_VLLM_MOE_ROUTE", "TAAF_VLLM_MOE_SHARE")
ORDERS = ("NO_OVERLAP", "MULTI_STREAM_OVERLAPPED")
PARALLEL_SIZES = ("tp_size", "ep_size", "dp_size", "pcp_size")
_STATE = {"level": None, "said": set(), "fired": 0, "stock": 0, "blocks": 0, "aside": 0, "checks": []}


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def say(key, line):
    if key in _STATE["said"]:
        return
    _STATE["said"].add(key)
    print(f"ROUTER_CHAIN {line}", flush=True)


def parse(value):
    said = str(value or "").strip().lower()
    if said not in VALUES:
        raise ValueError(f"{ENV}={value!r} is not one of {list(VALUES)}")
    if said in ("", "0"):
        return 0
    return LEVELS["gate_up" if said == "1" else said]


def level():
    if _STATE["level"] is None:
        _STATE["level"] = parse(os.environ.get(ENV))
    return _STATE["level"]


def on():
    return level() > 0


def tally():
    return {key: _STATE[key] for key in ("fired", "stock", "blocks", "aside")}


@triton.jit
def _round_bf16(value, EXACT: tl.constexpr):
    if EXACT:
        return value.to(tl.bfloat16).to(tl.float32)
    raw = value.to(tl.int32, bitcast=True)
    kept = (raw + 0x7FFF + ((raw >> 16) & 1)) & -65536
    return kept.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["rows"])
def _silu_mul_kernel(x_ptr, out_ptr, stride_x, stride_out, rows, N, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = (cols < N) & (row < rows)
    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    if EXACT:
        act = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
    else:
        act = gate / (1.0 + tl.exp(-gate))
    act = _round_bf16(act, EXACT)
    tl.store(out_ptr + row * stride_out + cols, _round_bf16(act * up, EXACT).to(tl.bfloat16), mask=mask)


def silu_mul(strided):
    rows, width = strided.shape
    half = width // 2
    out = torch.empty((rows, half), dtype=strided.dtype, device=strided.device)
    if rows:
        block = triton.next_power_of_2(half)
        _silu_mul_kernel[(rows,)](strided, out, strided.stride(0), out.stride(0), rows, half, BLOCK=block,
                                  EXACT=not INTERPRETED, num_warps=max(1, min(8, block // 256)))
    return out


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _compiling():
    probe = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(probe()) if callable(probe) else False


def _plain(layer):
    method = getattr(layer, "quant_method", None)
    weight = getattr(layer, "weight", None)
    return (method is not None and type(method).__name__ == "UnquantizedLinearMethod"
            and not hasattr(layer, "taaf_bf16c")
            and torch.is_tensor(weight) and weight.dtype == torch.bfloat16 and weight.dim() == 2
            and weight.numel() > 0 and weight.is_contiguous() and getattr(layer, "bias", None) is None
            and int(getattr(layer, "tp_size", 1) or 1) == 1)


def _unstrided(runner):
    if any(str(os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on") for name in UNSTRIDED_ENVS):
        return True
    if sys.modules.get("vllm.taaf_router_fused") is None:
        return True
    experts = getattr(runner, "routed_experts", None)
    method = getattr(experts, "quant_method", None)
    return bool(getattr(method, "is_monolithic", False))


class Held:
    def __init__(self, runner, want):
        self.runner = runner
        self.want = want
        self.enabled = False
        self.verified = False
        self.mode = False
        self.reason = ""
        self.note = ""
        self.stash = None
        self.parts = {}
        self.layers = {}
        self.big = None
        self.hidden = 0
        self.width = 0
        self.use_gate_up = False
        self.original_gate = None
        self.original_gate_up = None
        self.original_sgate = None
        self.original_act = None


def _modules(runner):
    gate = getattr(runner, "gate", None)
    shared = getattr(getattr(runner, "_shared_experts", None), "_layer", None)
    sgate = getattr(shared, "expert_gate", None) if shared is not None else None
    return gate, shared, sgate


def _segments(runner, want):
    gate, shared, sgate = _modules(runner)
    if gate is None or shared is None or sgate is None:
        return None, "", "the runner holds no router, shared expert or shared-expert gate"
    if not (_plain(gate) and _plain(sgate)):
        return None, "", "router or shared gate weight is not a plain bf16 matrix"
    if int(sgate.weight.shape[0]) != 1 or sgate.weight.shape[1] != gate.weight.shape[1]:
        return None, "", f"shared gate weight shape {tuple(sgate.weight.shape)}"
    segments = []
    note = ""
    if want >= LEVELS["gate_up"]:
        up = getattr(shared, "gate_up_proj", None)
        if up is None or not _plain(up) or up.weight.shape[1] != gate.weight.shape[1]:
            note = "the shared gate_up weight is not a plain bf16 matrix, so only the router and shared gate join"
        elif type(getattr(shared, "act_fn", None)).__name__ != "SiluAndMul":
            note = "the shared activation is not SiluAndMul, so only the router and shared gate join"
        else:
            segments.append(("gate_up", up))
    segments.append(("router", gate))
    segments.append(("sgate", sgate))
    return segments, note, ""


def _build(held):
    runner = held.runner
    segments, note, reason = _segments(runner, held.want)
    if segments is None:
        held.reason = reason
        return
    first = segments[0][1].weight
    hidden = int(first.shape[1])
    total = sum(int(layer.weight.shape[0]) for _, layer in segments)
    width = -(-total // ALIGN) * ALIGN
    big = torch.zeros((width, hidden), dtype=torch.bfloat16, device=first.device)
    cursor = 0
    for name, layer in segments:
        rows = int(layer.weight.shape[0])
        big[cursor:cursor + rows].copy_(layer.weight.data)
        held.parts[name] = (cursor, rows)
        held.layers[name] = layer
        layer.weight.data = big[cursor:cursor + rows]
        cursor += rows
    held.big, held.hidden, held.width, held.note = big, hidden, width, note
    held.use_gate_up = "gate_up" in held.parts
    held.enabled = True
    _STATE["blocks"] += 1
    gate, shared, sgate = _modules(runner)
    held.original_gate = gate.forward
    held.original_sgate = sgate.forward
    gate.forward = types.MethodType(lambda module, x: _router_forward(held, x), gate)
    sgate.forward = types.MethodType(lambda module, x: _sgate_forward(held, x), sgate)
    if held.use_gate_up:
        held.original_gate_up = shared.gate_up_proj.forward
        held.original_act = shared.act_fn
        shared.gate_up_proj.forward = types.MethodType(lambda module, x: _gate_up_forward(held, x),
                                                       shared.gate_up_proj)
        shared.act_fn = _Act(held)


class _Act(torch.nn.Module):
    def __init__(self, held):
        super().__init__()
        object.__setattr__(self, "held", held)

    def forward(self, x):
        held = self.held
        stash = held.stash
        if stash is not None and stash.get("gate_up_taken") and x.data_ptr() == stash["gate_up"].data_ptr() \
                and x.shape == stash["gate_up"].shape and x.stride() == stash["gate_up"].stride():
            return silu_mul(x)
        return held.original_act(x)


def _fresh(held, x, key):
    stash = held.stash
    if stash is None or stash.get(key) is None or stash.get(key + "_taken"):
        return None
    if x.data_ptr() != stash["x_ptr"] or x.shape[0] != stash["rows"] or x.shape[-1] != held.hidden:
        return None
    stash[key + "_taken"] = True
    return stash[key]


def _router_forward(held, x):
    piece = _fresh(held, x, "router")
    if piece is not None:
        return piece, None
    return held.original_gate(x)


def _gate_up_forward(held, x):
    piece = _fresh(held, x, "gate_up")
    if piece is not None:
        return piece, None
    return held.original_gate_up(x)


def _sgate_forward(held, x):
    piece = _fresh(held, x, "sgate")
    if piece is not None:
        return piece, None
    return held.original_sgate(x)


def _slice(held, out, name, base):
    start, rows = held.parts[name]
    return out[:, start - base:start - base + rows]


def _stock(held, x, names):
    return {name: F.linear(x, held.layers[name].weight) for name in names}


def _compare(want, got):
    wanted, gotten = want.float(), got.float()
    if not bool(torch.isfinite(wanted).all()):
        return None, 0
    if not bool(torch.isfinite(gotten).all()):
        return float("inf"), int((want != got).sum())
    scale = float(wanted.abs().max())
    gap = float((wanted - gotten).abs().max()) if want.numel() else 0.0
    return gap / max(scale, 1e-30), int((want != got).sum())


def _verify(held, x, base, with_up):
    generator = torch.Generator(device=x.device).manual_seed(PROBE_SEED)
    probe = torch.randn((PROBE_ROWS, held.hidden), generator=generator, device=x.device).to(torch.bfloat16)
    weight = held.big if base == 0 else held.big[base:]
    names = [name for name in held.parts if with_up or name != "gate_up"]
    runner = held.runner
    top_k = int(getattr(getattr(runner, "router", None), "top_k", 0) or 0) or 8
    worst, differ, total, ids_differ, rows, skipped = 0.0, 0, 0, 0, 0, 0
    for position, sample in enumerate((probe, x)):
        if sample.shape[0] == 0:
            continue
        want = _stock(held, sample, names)
        out = F.linear(sample, weight)
        got = {name: _slice(held, out, name, base) for name in names}
        scored = {name: _compare(want[name], got[name]) for name in names}
        if any(rel is None for rel, _ in scored.values()):
            if position == 0:
                return False, "the probe input gave a non-finite separate GEMM"
            skipped += 1
            continue
        for name in names:
            rel, count = scored[name]
            worst, differ, total = max(worst, rel), differ + count, total + want[name].numel()
        k = max(1, min(top_k, int(want["router"].shape[1])))
        one = torch.topk(want["router"].float(), k, dim=1).indices.sort(dim=1).values
        other = torch.topk(got["router"].float(), k, dim=1).indices.sort(dim=1).values
        ids_differ += int((one != other).any(dim=1).sum())
        rows += int(one.shape[0])
        if with_up:
            piece = got["gate_up"]
            if not torch.equal(silu_mul(piece), held.original_act(piece.contiguous())):
                return False, "the strided SiLU-and-mul differs from the stock activation"
    detail = (f"max rel {worst:.3g} (limit {TOLERANCE:.3g}), {differ}/{total} elements differ from the separate "
              f"GEMMs, {ids_differ}/{rows} rows pick another expert set"
              + (", the live input was not finite so only the probe was scored" if skipped else ""))
    return worst <= TOLERANCE, detail


def _held(runner, want):
    held = getattr(runner, "_taaf_router_chain", None)
    if held is not None:
        return held
    held = Held(runner, want)
    object.__setattr__(runner, "_taaf_router_chain", held)
    if _capturing():
        held.reason = "first met inside a graph capture"
    else:
        try:
            _build(held)
        except Exception as failure:
            held.enabled = False
            held.reason = f"{type(failure).__name__}: {str(failure)[:200]}"
    if not held.enabled:
        _STATE["aside"] += 1
        say(("aside", held.reason), f"stood aside for a layer ({getattr(runner, 'layer_name', '?')}): {held.reason}; "
                                    f"its stock router, shared gate and shared projections stay")
    elif held.note:
        say(("note", held.note), f"layer {getattr(runner, 'layer_name', '?')}: {held.note}")
    return held


def _order(runner, x):
    probe = getattr(getattr(runner, "_shared_experts", None), "_determine_shared_experts_order", None)
    if not callable(probe):
        return ""
    order = probe(x)
    return str(getattr(order, "name", order))


def _refusal(runner, x, shared_input):
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not (x.is_cuda or INTERPRETED) or x.stride(1) != 1:
        return "input"
    if shared_input is None or (shared_input is not x and (shared_input.data_ptr() != x.data_ptr()
                                                           or shared_input.shape != x.shape)):
        return "shared-input-differs"
    if _compiling():
        return "compiling"
    if getattr(runner, "_fse_fuse_gate", False):
        return "fused-gate"
    if getattr(runner, "enable_dbo", False):
        return "dbo"
    config = getattr(runner, "moe_config", None)
    if any(int(getattr(config, name, 1) or 1) != 1 for name in PARALLEL_SIZES):
        return "parallel"
    if getattr(config, "is_sequence_parallel", False):
        return "sequence-parallel"
    order = _order(runner, x)
    if order not in ORDERS:
        return f"shared-order-{order or 'none'}"
    return ""


def begin(runner, x, shared_input):
    held = getattr(runner, "_taaf_router_chain", None)
    if held is not None:
        held.stash = None
    want = level()
    if not want:
        return
    reason = _refusal(runner, x, shared_input)
    if reason:
        _STATE["stock"] += 1
        say(("stock", reason), f"stock for a call: {reason} (rows={int(x.shape[0]) if x.dim() else 0})")
        return
    if held is None:
        held = _held(runner, want)
    if not held.enabled or held.hidden != x.shape[1]:
        _STATE["stock"] += 1
        return
    with_up = held.use_gate_up and want >= LEVELS["gate_up"]
    if want >= LEVELS["gate_up"] and not held.use_gate_up and not held.note:
        say("rebuild", "level gate_up asked after the layers were stacked at level gate; the shared gate_up joins "
                       "only after a restart")
    base = 0 if with_up or not held.use_gate_up else held.parts["router"][0]
    if not (held.verified and held.mode == with_up):
        if _capturing():
            _STATE["stock"] += 1
            say("unverified", "a graph captured before the first eager check of a layer keeps the stock router, "
                              "shared gate and shared projections for that layer")
            return
        ok, detail = _verify(held, x, base, with_up)
        if not ok:
            held.enabled = False
            _STATE["aside"] += 1
            selfcheck_fail(ENV, detail)
            say(("failed", detail), f"stood aside: {detail}; that layer runs the stock router, shared gate and "
                                    f"shared projections")
            return
        held.verified, held.mode = True, with_up
        _STATE["checks"].append(detail)
        say(("check", with_up), f"check: one GEMM over [{' | '.join(n for n in held.parts if with_up or n != 'gate_up')}]"
                                f" x {held.hidden}, layer {getattr(held.runner, 'layer_name', '?')}: {detail}")
    weight = held.big if base == 0 else held.big[base:]
    out = F.linear(x, weight)
    router = _slice(held, out, "router", base)
    dense = _unstrided(runner)
    if dense:
        router = router.contiguous()
    held.stash = {"x_ptr": x.data_ptr(), "rows": int(x.shape[0]), "router": router,
                  "sgate": _slice(held, out, "sgate", base),
                  "gate_up": _slice(held, out, "gate_up", base) if with_up else None}
    _STATE["fired"] += 1
    capturing = _capturing()
    order = _order(runner, x)
    shapes = " | ".join(f"{name} {held.parts[name][1]}" for name in held.parts if with_up or name != "gate_up")
    padded = int(weight.shape[0]) - sum(held.parts[name][1] for name in held.parts if with_up or name != "gate_up")
    say(("live", order, int(capturing), with_up, dense),
        f"live: layer {getattr(runner, 'layer_name', '?')} rows={int(x.shape[0])} order={order} "
        f"capturing={int(capturing)}: one GEMM x[{int(x.shape[0])},{held.hidden}] @ W[{int(weight.shape[0])},"
        f"{held.hidden}]^T -> [{shapes} | pad {padded}] before the shared expert launches, router logits "
        f"{'copied dense' if dense else 'strided (row stride ' + str(int(out.stride(0))) + ')'}")


def end(runner):
    held = getattr(runner, "_taaf_router_chain", None)
    if held is not None:
        held.stash = None
