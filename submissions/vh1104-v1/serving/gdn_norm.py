import os
import sys
import time

import torch
import triton

NORM_ENV = "TAAF_VLLM_GDN_NORM_STRIDED"
TALLY_S = 15.0
_SAID = set()
_STATE = {"aside": "", "checked": set(), "at": 0.0}
_TALLY = {"strided": 0, "checked": 0, "stock": 0}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def on():
    return str(os.environ.get(NORM_ENV) or "").strip().lower() in ("1", "on", "true")


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def tally():
    now = time.monotonic()
    if now - _STATE["at"] < TALLY_S:
        return
    _STATE["at"] = now
    print(f"GDN_NORM_STRIDED tally {dict(_TALLY)}", flush=True)


def capturing(tensor):
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def guard():
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard

    return layernorm_guard


def eligible(norm, x, gate, out):
    if x.dim() != 3 or tuple(gate.shape) != tuple(x.shape) or tuple(out.shape) != tuple(x.shape):
        return False
    rows, heads, width = x.shape
    if rows == 0 or getattr(norm, "bias", None) is not None:
        return False
    if (norm.group_size or width) != width or not (x.dtype == gate.dtype == out.dtype):
        return False
    if any(one.stride(2) != 1 or one.stride(1) != width for one in (x, gate, out)):
        return False
    try:
        gate.view(-1, width)
    except RuntimeError:
        return True
    return False


def tiled(layer, heads, capturing):
    weight = layer.norm.weight
    held = getattr(layer, "taaf_norm_tiled", None)
    if held is not None and held.shape[0] == heads * weight.shape[0] and held.dtype == weight.dtype:
        return held
    if capturing:
        return None
    held = weight.detach().contiguous().repeat(heads)
    layer.taaf_norm_tiled = held
    return held


def launch(lib, norm, x, gate, out, weight, rows_per_block):
    rows, heads, width = x.shape
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(width))
    rstd = torch.empty((heads * rows,), dtype=torch.float32, device=x.device)
    lib.layer_norm_fwd_kernel[(triton.cdiv(rows, rows_per_block), heads)](
        x, out, weight, None, gate, None, rstd,
        x.stride(0), out.stride(0), gate.stride(0), rows, width, norm.eps,
        BLOCK_N=block_n, ROWS_PER_BLOCK=rows_per_block, HAS_BIAS=False, HAS_Z=True,
        NORM_BEFORE_GATE=norm.norm_before_gate, IS_RMS_NORM=True, ACTIVATION=norm.activation,
        num_warps=min(max(block_n // 256, 1), 8))


def stock(lib, norm, x, gate):
    width = x.shape[-1]
    held = torch.empty_like(x, memory_format=torch.contiguous_format)
    lib.layer_norm_fwd(x.reshape(-1, width), norm.weight.contiguous(), norm.bias, norm.eps,
                       z=gate.reshape(-1, width), out=held.view(-1, width), group_size=width,
                       norm_before_gate=norm.norm_before_gate, is_rms_norm=True, activation=norm.activation)
    return held


def bits(tensor):
    return tensor.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()])


def gated(layer, x, gate, out):
    if not on() or _STATE["aside"]:
        return False
    norm = layer.norm
    if not eligible(norm, x, gate, out):
        _TALLY["stock"] += 1
        return False
    lib = guard()
    rows, heads, width = x.shape
    rows_per_block = lib.calc_rows_per_block(rows * heads, x.device)
    held = capturing(x)
    weight = tiled(layer, heads, held)
    key = (rows_per_block, rows % 16 == 0, *(one.data_ptr() % 16 == 0 for one in (x, gate, out)),
           *(one.stride(0) % 16 == 0 for one in (x, gate, out)))
    if weight is None or (held and key not in _STATE["checked"]):
        _TALLY["stock"] += 1
        return False
    if key in _STATE["checked"]:
        launch(lib, norm, x, gate, out, weight, rows_per_block)
        _TALLY["strided"] += 1
        tally()
        return True
    want = stock(lib, norm, x, gate)
    launch(lib, norm, x, gate, out, weight, rows_per_block)
    if not torch.equal(bits(out), bits(want)):
        apart = int((bits(out) != bits(want)).sum())
        out.copy_(want)
        _STATE["aside"] = f"rows={rows} heads={heads} width={width} differ in {apart} values"
        selfcheck_fail(NORM_ENV, _STATE["aside"])
        print(f"GDN_NORM_STRIDED stands aside: the strided gated norm differs from the stock one "
              f"({_STATE['aside']}); the stock output was written and every later call runs the stock path",
              flush=True)
        return True
    _STATE["checked"].add(key)
    _TALLY["checked"] += 1
    print(f"GDN_NORM_STRIDED check rows={rows} heads={heads} width={width} rows_per_block={rows_per_block}: "
          f"equal to the stock gated norm in every value", flush=True)
    say("live", "GDN_NORM_STRIDED live: the gated RMS norm after a GDN layer's prefill reads the output gate "
                "where the input projection wrote it, one program per (token rows, head), so the gate is no "
                "longer copied to contiguous rows first")
    tally()
    return True
