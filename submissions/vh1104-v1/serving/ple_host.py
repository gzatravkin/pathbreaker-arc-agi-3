import mmap
import os
import sys

import numpy as np
import torch

EXACT_PIN_ENV = "TAAF_VLLM_PLE_EXACT_PIN"
_HELD = []
_SAID = set()


def _say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, file=sys.stderr, flush=True)


def exact_pin():
    return str(os.environ.get(EXACT_PIN_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def inc_method(quant_config, prefix, unquantized):
    resolved = quant_config.config_parser.resolve(None, prefix)
    if resolved.quantized:
        raise NotImplementedError(
            f"Qwen4Exp PLE embedding {prefix} resolves to {resolved.bits}-bit INC quantization, "
            f"which the PLE table cannot store")
    _say("inc", f"PLE_INC {prefix}: the INC config resolves it to {resolved.bits} bits, so the table "
                f"loads unquantized in the model dtype")
    return unquantized()


def rounded(size):
    return 1 << max(int(size) - 1, 0).bit_length()


def allocate(rows, width, dtype):
    if not exact_pin():
        return torch.empty(rows, width, dtype=dtype, device="cpu", pin_memory=True)
    size = int(rows) * int(width) * torch.empty((), dtype=dtype).element_size()
    mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
    owner = np.frombuffer(mapping, dtype=np.uint8)
    flat = torch.from_numpy(owner)
    result = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), size, 0)
    if result.value != 0:
        raise RuntimeError(f"PLE_EXACT_PIN: cudaHostRegister of {size} bytes failed: {result}")
    if not flat.is_pinned():
        raise RuntimeError("PLE_EXACT_PIN: CUDA does not report the registered PLE table as pinned")
    _HELD.append((mapping, owner))
    _say("pin", f"PLE_EXACT_PIN {int(rows)}x{int(width)} {dtype}: {size / 2**30:.2f} GiB registered in place; "
                f"the pinned host allocator would have held {rounded(size) / 2**30:.2f} GiB")
    return flat.view(dtype).view(int(rows), int(width))
