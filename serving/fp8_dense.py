import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.utils import is_weights_pre_processed

SETTINGS = Path(__file__).with_suffix(".json")
QUANTS = ("ptpc", "block128")
_CHOSEN = []
_MADE = {}
_QUANT = {}
_BAND = {"module": None}
BAND_MODULE = "taaf_fp8_band"
_DONE = {"layers": 0, "bf16": 0, "fp8": 0, "seconds": 0.0, "kernels": set(), "band": 0}


def settings():
    if not _CHOSEN:
        held = json.loads(SETTINGS.read_text(encoding="utf-8"))
        for name, pattern in held["groups"].items():
            _CHOSEN.append((name, re.compile(pattern)))
            _MADE[name] = 0
        _QUANT["name"] = held.get("quant") or QUANTS[0]
        if held.get("band"):
            _BAND["module"] = _load_band(held["band"])
    return _CHOSEN


def _load_band(said):
    module = sys.modules.get(BAND_MODULE)
    if module is None:
        spec = importlib.util.spec_from_file_location(BAND_MODULE, Path(__file__).with_name(f"{BAND_MODULE}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[BAND_MODULE] = module
        spec.loader.exec_module(module)
    module.configure(said)
    return module


def _mark_band(method, layer):
    band = _BAND["module"]
    if band is None or not isinstance(method, TaafFp8DenseMethod):
        return
    if band.mark(layer, lambda x: Fp8PtpcOnlineLinearMethod.apply(method, layer, x, None)):
        _DONE["band"] += 1


def chosen():
    return settings()


def quant():
    settings()
    return _QUANT["name"]


def group_of(prefix):
    for name, pattern in chosen():
        if pattern.search(prefix):
            return name
    return None


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _synchronized(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def summary():
    chosen()
    groups =", ".join(f"{name} {count}" for name, count in _MADE.items())
    return (f"FP8_DENSE quantized {_DONE['layers']} of {sum(_MADE.values())} linears at load "
            f"({groups}) with {quant()}: {_DONE['bf16'] / 1e9:.3f} GB of BF16 weights became "
            f"{_DONE['fp8'] / 1e9:.3f} GB of FP8 weights and scales in {_DONE['seconds']:.2f} s; "
            f"kernel {'/'.join(sorted(_DONE['kernels'])) or 'none'}"
            + (f"; {_DONE['band']} layers marked for the Triton band" if _BAND["module"] is not None else ""))


def _scale_name(layer):
    return "weight_scale" if hasattr(layer, "weight_scale") else "weight_scale_inv"


def _record(method, layer, loaded):
    if is_weights_pre_processed():
        if not hasattr(layer, "input_scale"):
            layer.input_scale = None
        layer._already_called_process_weights_after_loading = True
        _mark_band(method, layer)
        return
    if getattr(layer, "_already_called_process_weights_after_loading", False):
        return loaded()
    before = _bytes(layer.weight)
    started = _synchronized(layer.weight)
    loaded()
    _DONE["seconds"] += _synchronized(layer.weight) - started
    _DONE["layers"] += 1
    _DONE["bf16"] += before
    _DONE["fp8"] += _bytes(layer.weight) + _bytes(getattr(layer, _scale_name(layer)))
    _DONE["kernels"].add(type(method.fp8_linear).__name__)
    _mark_band(method, layer)
    if _DONE["layers"] == sum(_MADE.values()):
        print(summary(), flush=True)


class TaafFp8DenseMethod(Fp8PtpcOnlineLinearMethod):
    supports_pre_processed_weights = True

    def process_weights_after_loading(self, layer):
        _record(self, layer, lambda: super(TaafFp8DenseMethod, self).process_weights_after_loading(layer))

    def apply(self, layer, x, bias=None):
        if bias is None and getattr(layer, "taaf_fp8_band", None) is not None:
            return _BAND["module"].apply(layer, x)
        return super().apply(layer, x, bias)


class TaafFp8DenseMethodBlock128(Fp8PerBlockOnlineLinearMethod):
    supports_pre_processed_weights = True

    def process_weights_after_loading(self, layer):
        _record(self, layer,
                lambda: super(TaafFp8DenseMethodBlock128, self).process_weights_after_loading(layer))


_METHODS = {"ptpc": TaafFp8DenseMethod, "block128": TaafFp8DenseMethodBlock128}


def dense_method(prefix, method):
    name = group_of(prefix)
    if name is None:
        return method
    _MADE[name] += 1
    return _METHODS[quant()]()
