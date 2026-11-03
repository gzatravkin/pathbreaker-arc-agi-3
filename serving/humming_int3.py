import os
import re
import time

import torch

BITS_ENV = "TAAF_VLLM_EXPERT_BITS"
MAIN_LAYERS = 48
LAYER = re.compile(r"layers\.(\d+)\.")
LOW, HIGH = -4, 3
DIVISORS = (3.0, 3.25, 3.5, 3.75, 4.0, 4.5)
TINY = 2.0 ** -24
BATCH = 4
_said = {"layers": 0, "experts": 0, "error": 0.0, "signal": 0.0, "seconds": 0.0, "checked": False}


def say(text) -> None:
    print(f"HUMMING_INT3 {text}", flush=True)


def bits() -> int:
    said = str(os.environ.get(BITS_ENV) or "").strip()
    return int(said) if said else 0


def main_layer(name) -> bool:
    name = str(name)
    found = LAYER.search(name)
    return bool(found) and "mtp" not in name and int(found.group(1)) < MAIN_LAYERS


def wanted(method, layer) -> bool:
    backend = getattr(getattr(method, "wna16_moe_backend", None), "value", "")
    return bits() == 3 and str(backend).upper() == "HUMMING" and main_layer(getattr(layer, "layer_name", ""))


def unpack_gptq(qweight):
    shifts = torch.arange(8, dtype=torch.int32, device=qweight.device) * 4
    codes = (qweight.unsqueeze(-2) >> shifts.view(1, 1, 8, 1)) & 15
    batch, rows, _, n = codes.shape
    return codes.reshape(batch, rows * 8, n)


def requant(codes, scales):
    batch, k, n = codes.shape
    groups = scales.shape[1]
    width = k // groups
    weights = (codes.float() - 8.0).reshape(batch, groups, width, n) * scales.float().unsqueeze(2)
    peak = weights.abs().amax(dim=2)
    best_error = torch.full_like(peak, float("inf"))
    best_scale = torch.ones_like(peak)
    best_codes = torch.zeros_like(weights)
    for divisor in DIVISORS:
        scale = (peak / divisor).to(scales.dtype).float()
        scale = torch.where(scale > TINY, scale, torch.ones_like(scale))
        picked = torch.clamp(torch.round(weights / scale.unsqueeze(2)), LOW, HIGH)
        error = ((picked * scale.unsqueeze(2) - weights) ** 2).sum(dim=2)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_codes = torch.where(better.unsqueeze(2), picked, best_codes)
    unsigned = (best_codes - LOW).to(torch.int32).reshape(batch, k, n)
    return unsigned, best_scale.to(scales.dtype), float(best_error.sum()), float((weights ** 2).sum())


def pack_bits(values, width):
    *lead, k = values.shape
    shifts = torch.arange(width, dtype=torch.int32, device=values.device)
    stream = ((values.unsqueeze(-1) >> shifts) & 1).reshape(*lead, k // 32, 32 * width)
    stream = stream.reshape(*lead, k // 32, width, 32).to(torch.int64)
    places = torch.arange(32, dtype=torch.int64, device=values.device)
    words = (stream << places).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.reshape(*lead, k // 32 * width)


def unpack_bits(words, width, k):
    *lead, _ = words.shape
    places = torch.arange(32, dtype=torch.int64, device=words.device)
    bits_ = ((words.to(torch.int64) & 0xFFFFFFFF).reshape(*lead, k // 32, width, 1) >> places) & 1
    bits_ = bits_.reshape(*lead, k // 32, 32, width)
    shifts = torch.arange(width, dtype=torch.int64, device=words.device)
    return (bits_ << shifts).sum(-1).reshape(*lead, k).to(torch.int32)


def to_humming(qweight, scales):
    count = qweight.shape[0]
    k, n = qweight.shape[1] * 8, qweight.shape[2]
    weight = torch.empty((count, n, k * 3 // 32), dtype=torch.int32, device=qweight.device)
    weight_scale = torch.empty((count, n, scales.shape[1]), dtype=scales.dtype, device=qweight.device)
    error = signal = 0.0
    for begin in range(0, count, BATCH):
        part = slice(begin, min(count, begin + BATCH))
        unsigned, scale, err, sig = requant(unpack_gptq(qweight[part]), scales[part])
        weight[part] = pack_bits(unsigned.transpose(1, 2).contiguous(), 3)
        weight_scale[part] = scale.transpose(1, 2)
        error += err
        signal += sig
    return weight, weight_scale, error, signal


def _check(weight, weight_scale, qweight, scales):
    from humming import ops

    unsigned, scale, _, _ = requant(unpack_gptq(qweight[:1]), scales[:1])
    got = ops.unpack_weight(weight[:1].contiguous(), 3)
    want = unsigned.transpose(1, 2).contiguous()
    if not torch.equal(got.to(torch.int32), want) or not torch.equal(weight_scale[:1], scale.transpose(1, 2)):
        raise RuntimeError("HUMMING_INT3 check failed: humming's own unpack of the packed 3-bit weights does not "
                           "give back the codes")
    say(f"check: humming's unpack_weight returns the packed 3-bit codes of expert 0 bit for bit "
        f"({want.numel()} codes)")


def convert(layer, method) -> None:
    from vllm.model_executor.layers.quantization.utils.humming_utils import convert_to_humming_moe_kernel_format
    from vllm.utils.humming import HummingInputSchema, HummingWeightSchema

    began = time.time()
    for name in ("w13_bias", "w2_bias"):
        if name not in layer._loaded_expert_biases:
            layer.register_parameter(name, None)
    made = {}
    error = signal = 0.0
    with torch.no_grad():
        for sub in ("w13", "w2"):
            qweight, scales = getattr(layer, f"{sub}_qweight").data, getattr(layer, f"{sub}_scales").data
            weight, weight_scale, err, sig = to_humming(qweight, scales)
            if not _said["checked"] and qweight.is_cuda:
                _check(weight, weight_scale, qweight, scales)
                _said["checked"] = True
            made[sub] = (weight, weight_scale)
            error += err
            signal += sig
        for sub, (weight, weight_scale) in made.items():
            for stale in ("qweight", "scales", "qzeros", "g_idx", "g_idx_sort_indices"):
                if hasattr(layer, f"{sub}_{stale}"):
                    delattr(layer, f"{sub}_{stale}")
            layer.register_parameter(f"{sub}_weight", torch.nn.Parameter(weight, requires_grad=False))
            layer.register_parameter(f"{sub}_weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False))
    group = method.quant_config.group_size
    convert_to_humming_moe_kernel_format(
        layer,
        weight_schema=HummingWeightSchema(b_dtype="uint3", weight_scale_group_size=group, has_zero_point=False),
        input_schema=HummingInputSchema())
    _said["layers"] += 1
    _said["experts"] += int(made["w13"][0].shape[0])
    _said["error"] += error
    _said["signal"] += signal
    _said["seconds"] += time.time() - began
    if _said["layers"] in (1, MAIN_LAYERS):
        say(f"{_said['layers']} layers, {_said['experts']} experts served at 3 bits by humming "
            f"(uint3, group {group}, codes -4..3 around 4, scale per group by least squared error), weight error "
            f"{_said['error'] / max(_said['signal'], 1e-30):.5f} of the signal, {_said['seconds']:.1f} s")
