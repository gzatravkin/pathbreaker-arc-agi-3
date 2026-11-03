import hashlib
import os
from pathlib import Path

ENV = "TAAF_VLLM_ROUTER_PAD"
FUSED_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576)
INSTALLED = "vllm/taaf_router_pad.py"
MARKER = "vllm.taaf_router_pad"
TARGET = "vllm/model_executor/layers/fused_moe/router/fused_topk_router.py"
PINNED = "1c80f4af4d781fd66b12aa6a7a0fd30082bc521df40f1252987a8ac4c0bef51c"
EDITS = (
    ("from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter\n",
     "from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter\n"
     "from vllm.taaf_router_pad import padded as _taaf_padded\n"),
    ("        topk_func = dispatch_topk_softmax_func(\n"
     "            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()\n"
     "        )\n"
     "        topk_weights, topk_ids = topk_func(\n"
     "            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize\n"
     "        )\n",
     "        topk_func = dispatch_topk_softmax_func(\n"
     "            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()\n"
     "        )\n"
     "        topk_weights, topk_ids = topk_func(\n"
     "            topk_weights, topk_ids, token_expert_indices, _taaf_padded(gating_output),\n"
     "            renormalize,\n"
     "        )\n"),
)
ON = str(os.environ.get(ENV) or "").strip().lower() in ("1", "true", "yes", "on")
_PAD = []


def width(experts: int) -> int:
    if experts in FUSED_WIDTHS:
        return experts
    wider = [one for one in FUSED_WIDTHS if one > experts]
    return wider[0] if wider else experts


def padded(logits):
    if not ON:
        return logits
    experts = int(logits.shape[-1])
    target = width(experts)
    if target == experts:
        return logits
    if not _PAD:
        import torch.nn.functional as functional

        _PAD.append(functional.pad)
    return _PAD[0](logits, (0, target - experts), value=float("-inf"))


def _undone(text: str) -> str:
    for found, put in EDITS:
        text = text.replace(put, found, 1)
    return text


def apply(site: Path, helper: Path) -> dict:
    path = site / TARGET
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular vLLM source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    source = helper.read_bytes()
    installed = site / INSTALLED
    row = {"armed": True, "target": TARGET, "helper_sha256": hashlib.sha256(source).hexdigest()}
    if MARKER in text:
        if hashlib.sha256(_undone(text).encode("utf-8")).hexdigest() != PINNED:
            raise RuntimeError(f"{ENV} cannot apply: {TARGET} carries the edit over another file")
        made = None
    else:
        if before != PINNED:
            raise RuntimeError(f"{ENV} cannot apply: {TARGET} is {before[:16]}, not the {PINNED[:16]} "
                               f"its edits were written against.")
        for found, put in EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{ENV} cannot apply: an anchor of {TARGET} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
    if made is not None:
        temporary = path.with_name(f".{path.name}.pad.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the router padding edit of {TARGET} did not land")
    row.update({"already": made is None, "stock_sha256": PINNED,
                "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    print(f"ROUTER_PAD {'already armed' if made is None else 'armed'}: router logits whose expert "
          f"count the fused top-k softmax kernel has no case for are padded with -inf columns to the "
          f"next count it has ({', '.join(str(one) for one in FUSED_WIDTHS)}), so 400 experts route "
          f"through one fused kernel at 448 instead of a softmax kernel and a top-k kernel; the "
          f"padded columns weigh exp(-inf) = 0 and lose every tie to a lower index", flush=True)
    return row
