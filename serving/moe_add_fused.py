import os
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "TAAF_VLLM_MOE_ADD_FUSED"
COUNTER = "moe_add_fused"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
SWEEP_ROWS = 2048
PROGRAM_ROWS = 1
TALLY_S = 15.0
_STATE = {"aside": "", "checked": False, "tally": None, "at": 0.0}
_TALLY = {"fused": 0, "stock": 0, "deferred": 0}
_SAID = set()


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def on():
    return str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def say(line):
    print(f"MOE_ADD_FUSED {line}", flush=True)


def once(key, line):
    if key not in _SAID:
        _SAID.add(key)
        say(line)


def tally():
    now = time.monotonic()
    if now - _STATE["at"] >= TALLY_S:
        _STATE["at"] = now
        say(f"tally {dict(_TALLY)}")


@triton.jit
def _widen(bits):
    return ((bits.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _narrow(value, EXACT: tl.constexpr):
    if EXACT:
        bits = value.to(tl.bfloat16).to(tl.int16, bitcast=True)
    else:
        raw = value.to(tl.int32, bitcast=True)
        kept = ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF
        bits = tl.where(value != value, 0x7FC0, kept).to(tl.int16)
    return bits


@triton.jit(do_not_specialize=["rows"])
def _gate_add_kernel(fused_ptr, shared_ptr, gate_ptr, out_ptr, tally_ptr, rows, hidden,
                     stride_fused, stride_shared, stride_gate, stride_out,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr, EXACT: tl.constexpr, TALLY: tl.constexpr,
                     SUMS: tl.constexpr = False):
    first = tl.program_id(0) * ROWS
    part = tl.program_id(1)
    if TALLY:
        if (first == 0) & (part == 0):
            tl.atomic_add(tally_ptr, 1)
            tl.atomic_add(tally_ptr + 1, rows)
    lines = first + tl.arange(0, ROWS)
    live = lines < rows
    cols = part * BLOCK + tl.arange(0, BLOCK)
    keep = live[:, None] & (cols < hidden)[None, :]
    logit = _widen(tl.load(gate_ptr + lines * stride_gate, mask=live, other=0))
    if EXACT:
        scale = tl.div_rn(1.0, 1.0 + libdevice.exp(-logit))
    else:
        scale = tl.div_rn(1.0, 1.0 + tl.exp(-logit))
    scale = _widen(_narrow(scale, EXACT))[:, None]
    shared = _widen(tl.load(shared_ptr + lines[:, None] * stride_shared + cols[None, :], mask=keep, other=0))
    if SUMS:
        fused = _widen(_narrow(tl.load(fused_ptr + lines[:, None] * stride_fused + cols[None, :], mask=keep,
                                       other=0.0), EXACT))
    else:
        fused = _widen(tl.load(fused_ptr + lines[:, None] * stride_fused + cols[None, :], mask=keep, other=0))
    gated = _widen(_narrow(scale * shared, EXACT))
    tl.store(out_ptr + lines[:, None] * stride_out + cols[None, :], _narrow(gated + fused, EXACT), mask=keep)


def block_for(hidden):
    for block in (1024, 512, 256):
        if hidden % block == 0:
            return block
    return min(1024, triton.next_power_of_2(hidden))


def warps_for(block):
    return max(1, block // 256)


def stock(shared, gate, fused):
    return F.sigmoid(gate) * shared + fused


def refusal(shared, gate, fused):
    if not (shared.is_cuda or INTERPRETED):
        return "device"
    if shared.dim() != 2 or fused.dim() != 2 or gate.dim() != 2:
        return "rank"
    rows, hidden = shared.shape
    if tuple(fused.shape) != (rows, hidden) or tuple(gate.shape) != (rows, 1):
        return "shape"
    if not (shared.dtype == fused.dtype == gate.dtype == torch.bfloat16):
        return "dtype"
    if shared.stride(1) != 1 or fused.stride(1) != 1:
        return "layout"
    if not (shared.device == fused.device == gate.device):
        return "devices"
    return ""


def launch(shared, gate, fused, counter=None, sums=None):
    rows, hidden = shared.shape
    out = torch.empty_like(fused, memory_format=torch.contiguous_format)
    if rows == 0:
        return out
    block = block_for(hidden)
    bits = torch.int16
    read = fused.view(bits) if sums is None else sums
    _gate_add_kernel[(triton.cdiv(rows, PROGRAM_ROWS), triton.cdiv(hidden, block))](
        read, shared.view(bits), gate.view(bits), out.view(bits),
        counter if counter is not None else gate,
        rows, hidden, read.stride(0), shared.stride(0), gate.stride(0), out.stride(0),
        ROWS=PROGRAM_ROWS, BLOCK=block, EXACT=not INTERPRETED, TALLY=counter is not None, SUMS=sums is not None,
        num_warps=warps_for(block))
    return out


def _summed():
    return sys.modules.get("vllm.taaf_moe_block")


def _claim(fused):
    claim = getattr(_summed(), "claim", None)
    return claim(fused) if callable(claim) else None


def _settled(fused, sums):
    if sums is not None:
        fused.copy_(sums)
    return fused


def _sums_fit(fused, sums):
    return (sums is not None and sums.dtype == torch.float32 and tuple(sums.shape) == tuple(fused.shape)
            and sums.stride(1) == 1 and sums.device == fused.device)


def bits_of(tensor):
    return tensor.contiguous().view(torch.int16)


def compare(got, want):
    differ = bits_of(got) != bits_of(want)
    both_nan = torch.isnan(got) & torch.isnan(want)
    return int((differ & ~both_nan).sum()), int((differ & both_nan).sum())


def sweep(hidden, device):
    values = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(20261014)
    bad = nan = launches = 0
    for start in range(0, values.numel(), SWEEP_ROWS):
        gate = values[start:start + SWEEP_ROWS].reshape(-1, 1)
        rows = gate.shape[0]
        scale = torch.logspace(-3, 1, rows, device=device).reshape(-1, 1)
        shared = (torch.randn(rows, hidden, generator=generator, device=device) * scale).to(torch.bfloat16)
        fused = (torch.randn(rows, hidden, generator=generator, device=device) * scale).to(torch.bfloat16)
        apart, nans = compare(launch(shared, gate, fused), stock(shared, gate, fused))
        bad += apart
        nan += nans
        launches += 1
    return bad, nan, launches


def stand_aside(reason):
    selfcheck_fail(ENV, reason)
    _STATE["aside"] = reason
    say(f"stood aside: {reason}; every later MoE layer runs the stock sigmoid, multiply and routed+shared add")


def check(shared, gate, fused, sums=None):
    _STATE["checked"] = True
    want = stock(shared, gate, fused)
    try:
        bad, nan, launches = sweep(shared.shape[1], shared.device)
        if bad:
            stand_aside(f"the gate sweep over all 65536 bf16 gate values differs from the stock path in {bad} "
                        f"values")
            return want
        got = launch(shared, gate, fused)
        apart, nans = compare(got, want)
        if apart:
            stand_aside(f"the first live call (rows={shared.shape[0]} hidden={shared.shape[1]}) differs from the "
                        f"stock path in {apart} values")
            return want
        if sums is not None:
            apart, more = compare(launch(shared, gate, fused, sums=sums), want)
            nans += more
            if apart:
                stand_aside(f"the first live call read from the fp32 sums (rows={shared.shape[0]}) differs from the "
                            f"stock path in {apart} values")
                return want
        counter = torch.zeros(2, dtype=torch.int64, device=shared.device)
        _STATE["tally"] = counter
        timers = sys.modules.get("vllm.taaf_step_timers")
        registered = callable(getattr(timers, "counter", None))
        if registered:
            timers.counter(COUNTER, ("fires", "rows"), counter)
    except Exception as failure:
        stand_aside(f"{type(failure).__name__}: {str(failure)[:300]}")
        return want
    say(f"check: all 65536 bf16 gate values x hidden={shared.shape[1]} in {launches} launches and the first live "
        f"call (rows={shared.shape[0]}) equal the stock sigmoid, multiply and add bit for bit"
        + (f" ({nan + nans} NaN results differ only in NaN payload)" if nan + nans else "")
        + f"; firing counter {'in the step timers as ' + COUNTER if registered else 'not registered (no step timers)'}")
    return want


def _capturing(tensor):
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def _compiling():
    probe = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(probe()) if callable(probe) else False


def _shared_layer(runner):
    shared = getattr(runner, "_shared_experts", None)
    return shared, getattr(shared, "_layer", None)


def ineligible(runner, hidden_states):
    if not on():
        return "off"
    if _STATE["aside"]:
        return "aside"
    shared, layer = _shared_layer(runner)
    if layer is None or getattr(layer, "expert_gate", None) is None:
        return "no-shared-gate"
    if getattr(runner, "enable_dbo", False):
        return "dbo"
    if getattr(runner, "routed_scaling_factor", 1.0) != 1.0:
        return "routed-scale"
    if getattr(runner, "routed_output_transform", None) is not None \
            or getattr(runner, "routed_input_transform", None) is not None:
        return "routed-transform"
    config = runner.moe_config
    if any(int(getattr(config, name, 1) or 1) != 1 for name in ("tp_size", "ep_size", "dp_size", "pcp_size")):
        return "parallel"
    if getattr(config, "is_sequence_parallel", False):
        return "sequence-parallel"
    if type(getattr(runner, "router", None)).__name__ == "ZeroExpertRouter":
        return "zero-expert"
    if _compiling():
        return "compiling"
    order = shared._determine_shared_experts_order(hidden_states)
    if getattr(order, "name", str(order)) != "NO_OVERLAP":
        return "aux-stream"
    if not _STATE["checked"] and _capturing(hidden_states):
        return "unchecked-in-capture"
    return ""


def _expect_reader(ready):
    expect = getattr(_summed(), "expect_reader", None)
    if callable(expect):
        expect(ready)


def arm(runner, hidden_states):
    _, layer = _shared_layer(runner)
    if layer is not None:
        layer.taaf_gate = None
        layer.taaf_defer = False
    reason = ineligible(runner, hidden_states)
    _expect_reader(not reason)
    if reason:
        if reason != "off":
            once(f"stock-{reason}", f"stock reason={reason} rows={hidden_states.shape[0]}")
        return False
    layer.taaf_defer = True
    return True


def shared_gate(layer, x, out):
    logit = layer.expert_gate(x)[0]
    if getattr(layer, "taaf_defer", False):
        layer.taaf_defer = False
        layer.taaf_gate = logit
        _TALLY["deferred"] += 1
        return out
    return F.sigmoid(logit) * out


def add(runner, shared_output, fused_output):
    sums = _claim(fused_output)
    _, layer = _shared_layer(runner)
    gate = getattr(layer, "taaf_gate", None) if layer is not None else None
    if layer is not None:
        layer.taaf_gate = None
        layer.taaf_defer = False
    if gate is None:
        return shared_output + _settled(fused_output, sums)
    reason = _STATE["aside"] or refusal(shared_output, gate, fused_output)
    if reason:
        _TALLY["stock"] += 1
        once(f"refuse-{reason}", f"stock reason={reason} rows={shared_output.shape[0]}")
        return stock(shared_output, gate, _settled(fused_output, sums))
    if not _sums_fit(fused_output, sums):
        fused_output, sums = _settled(fused_output, sums), None
    if not _STATE["checked"]:
        if _capturing(shared_output):
            _TALLY["stock"] += 1
            return stock(shared_output, gate, _settled(fused_output, sums))
        return check(shared_output, gate, _settled(fused_output, sums), sums)
    capturing = _capturing(shared_output)
    try:
        out = launch(shared_output, gate, fused_output, _STATE["tally"], sums=sums)
    except Exception as failure:
        stand_aside(f"{type(failure).__name__}: {str(failure)[:300]}")
        _TALLY["stock"] += 1
        return stock(shared_output, gate, _settled(fused_output, sums))
    _TALLY["fused"] += 1
    rows, hidden = shared_output.shape
    block = block_for(hidden)
    once(f"live-{int(capturing)}",
         f"live: the shared expert's sigmoid gate, its scale and the routed+shared add run as one kernel "
         f"(rows={rows} hidden={hidden} block={block} warps={warps_for(block)} capturing={int(capturing)}); "
         f"the gate projection, both expert GEMMs and the routed sum are the stock kernels")
    tally()
    return out
