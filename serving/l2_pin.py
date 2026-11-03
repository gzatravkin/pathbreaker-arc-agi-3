import ctypes
import os
from typing import Any

ENV = "TAAF_VLLM_L2_PIN"
MIB = 1 << 20
ALIGN = 512
LIMIT_PERSISTING_L2 = 0x06
ATTR_MAX_WINDOW = 101
ATTR_MAX_PERSISTING = 108
ATTR_L2_SIZE = 38
STREAM_ATTR_WINDOW = 1
PROP_STREAMING = 1
PROP_PERSISTING = 2
ROUTER_SUFFIX = ".mlp.gate.weight"
DENSE_SKIP = ("experts", "embed", "lm_head", "shared_expert_gate")
GROUP_ROUTER = "router"
GROUP_DRAFT_ROUTER = "draft_router"
GROUP_DRAFT_DENSE = "draft_dense"

_STATE: dict[str, Any] = {"done": False, "held": [], "report": None, "graph_wrapped": False}


def budget_mib() -> int:
    said = str(os.environ.get(ENV) or "").strip()
    if said in ("", "0"):
        return 0
    if not said.isdigit():
        raise RuntimeError(f"{ENV} is a budget in MiB, not {said!r}")
    return int(said)


def aligned(nbytes: int) -> int:
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def is_router(name: str) -> bool:
    return name.endswith(ROUTER_SUFFIX)


def is_dense(name: str) -> bool:
    return not any(part in name for part in DENSE_SKIP) and not is_router(name)


def candidates(model_params, draft_params):
    out = []
    seen = set()
    for name, nbytes, ident in model_params:
        if is_router(name) and ident not in seen:
            seen.add(ident)
            out.append({"name": name, "nbytes": nbytes, "group": GROUP_ROUTER, "ident": ident})
    for name, nbytes, ident in draft_params:
        if is_router(name) and ident not in seen:
            seen.add(ident)
            out.append({"name": f"draft:{name}", "nbytes": nbytes, "group": GROUP_DRAFT_ROUTER, "ident": ident})
    for name, nbytes, ident in draft_params:
        if is_dense(name) and ident not in seen:
            seen.add(ident)
            out.append({"name": f"draft:{name}", "nbytes": nbytes, "group": GROUP_DRAFT_DENSE, "ident": ident})
    return out


def plan(cands, capacity: int):
    taken, skipped, offset = [], [], 0
    for one in cands:
        need = aligned(one["nbytes"])
        if one["nbytes"] <= 0 or offset + need > capacity:
            skipped.append(one)
            continue
        taken.append({**one, "offset": offset})
        offset += need
    return {"entries": taken, "skipped": skipped, "total": offset}


def hit_ratio(window_bytes: int, granted: int) -> float:
    if window_bytes <= 0:
        return 0.0
    return min(1.0, granted / window_bytes)


class WindowValue(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t), ("hitRatio", ctypes.c_float),
                ("hitProp", ctypes.c_int), ("missProp", ctypes.c_int)]


class StreamAttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", WindowValue), ("pad", ctypes.c_char * 64)]


class Cudart:
    def __init__(self, device: int, lib=None):
        self.device = device
        self.lib = lib or self.load()

    @staticmethod
    def load():
        last = None
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so", "cudart64_12.dll"):
            try:
                return ctypes.CDLL(name)
            except OSError as failure:
                last = failure
        raise RuntimeError(f"no cudart could be loaded: {last}")

    def check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} returned cudaError {code}")

    def attribute(self, attr: int) -> int:
        value = ctypes.c_int(0)
        self.check(self.lib.cudaDeviceGetAttribute(ctypes.byref(value), attr, self.device),
                   f"cudaDeviceGetAttribute({attr})")
        return int(value.value)

    def limits(self) -> dict[str, int]:
        return {"l2": self.attribute(ATTR_L2_SIZE), "max_persisting": self.attribute(ATTR_MAX_PERSISTING),
                "max_window": self.attribute(ATTR_MAX_WINDOW)}

    def set_carveout(self, nbytes: int) -> int:
        self.check(self.lib.cudaDeviceSetLimit(LIMIT_PERSISTING_L2, ctypes.c_size_t(nbytes)), "cudaDeviceSetLimit")
        got = ctypes.c_size_t(0)
        self.check(self.lib.cudaDeviceGetLimit(ctypes.byref(got), LIMIT_PERSISTING_L2), "cudaDeviceGetLimit")
        return int(got.value)

    def set_window(self, stream_handle: int, base: int, nbytes: int, ratio: float) -> None:
        value = StreamAttrValue()
        value.accessPolicyWindow = WindowValue(base, nbytes, ratio, PROP_PERSISTING, PROP_STREAMING)
        self.check(self.lib.cudaStreamSetAttribute(ctypes.c_void_p(stream_handle), STREAM_ATTR_WINDOW,
                                                   ctypes.byref(value)), "cudaStreamSetAttribute")


def ident_of(tensor) -> tuple:
    return (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))


def params_of(module):
    if module is None:
        return []
    out = []
    for name, param in module.named_parameters():
        data = param.data
        if data.numel() == 0 or not data.is_contiguous():
            continue
        out.append((name, data.numel() * data.element_size(), ident_of(data)))
    return out


def pack(module, draft, entries, total, device):
    import torch

    buffer = torch.empty(total, dtype=torch.uint8, device=device)
    lookup = {}
    for prefix, root in (("", module), ("draft:", draft)):
        if root is None:
            continue
        for name, param in root.named_parameters():
            lookup[prefix + name] = param
    moved = []
    for one in entries:
        param = lookup[one["name"]]
        data = param.data
        view = buffer[one["offset"]:one["offset"] + one["nbytes"]].view(data.dtype).view(data.shape)
        view.copy_(data)
        param.data = view
        moved.append(one["name"])
    return buffer, moved


def stream_handle(stream) -> int:
    return int(stream.cuda_stream)


def wrap_capture(rt, base, nbytes, ratio):
    import torch

    if _STATE["graph_wrapped"]:
        return
    original = torch.cuda.graph.__enter__

    def entered(self):
        rt.set_window(stream_handle(self.capture_stream), base, nbytes, ratio)
        return original(self)

    torch.cuda.graph.__enter__ = entered
    _STATE["graph_wrapped"] = True


def apply(model: Any, speculator: Any = None, rt: Any = None, device: Any = None) -> dict[str, Any]:
    if _STATE["done"] or not budget_mib():
        return _STATE["report"] or {}
    import torch

    draft = getattr(speculator, "model", None)
    device = device if device is not None else next(model.parameters()).device
    index = getattr(device, "index", None) or 0
    rt = rt or Cudart(index)
    limits = rt.limits()
    want = budget_mib() * MIB
    capacity = min(want, limits["max_persisting"], limits["max_window"])
    cands = candidates(params_of(model), params_of(draft))
    wanted_bytes = sum(aligned(one["nbytes"]) for one in cands)
    granted = rt.set_carveout(capacity)
    capacity = min(capacity, granted)
    chosen = plan(cands, capacity)
    report: dict[str, Any] = {"budget_mib": budget_mib(), "device_limits": limits, "carveout_granted": granted,
                              "candidates": len(cands), "candidate_bytes": wanted_bytes,
                              "pinned_tensors": len(chosen["entries"]), "skipped_tensors": len(chosen["skipped"]),
                              "pinned_bytes": chosen["total"], "by_group": {}}
    for one in chosen["entries"]:
        report["by_group"][one["group"]] = report["by_group"].get(one["group"], 0) + one["nbytes"]
    if not chosen["entries"]:
        print(f"L2_PIN nothing to pin: {report}", flush=True)
        _STATE.update(done=True, report=report)
        return report
    buffer, moved = pack(model, draft, chosen["entries"], chosen["total"], device)
    ratio = hit_ratio(chosen["total"], granted)
    base = buffer.data_ptr()
    rt.set_window(stream_handle(torch.cuda.current_stream(device)), base, chosen["total"], ratio)
    wrap_capture(rt, base, chosen["total"], ratio)
    _STATE["held"].append(buffer)
    report.update(hit_ratio=ratio, window_base=base, window_bytes=chosen["total"], repointed=len(moved))
    _STATE.update(done=True, report=report)
    torch.cuda.empty_cache()
    print(f"L2_PIN pinned {chosen['total']} bytes ({chosen['total'] / MIB:.1f} MiB) in {len(moved)} tensors "
          f"of {len(cands)} candidates ({wanted_bytes / MIB:.1f} MiB), hitRatio {ratio:.3f}, carve-out granted "
          f"{granted / MIB:.1f} MiB of {limits['l2'] / MIB:.1f} MiB L2 (budget {budget_mib()} MiB, device max "
          f"persisting {limits['max_persisting'] / MIB:.1f} MiB, max window {limits['max_window'] / MIB:.1f} MiB), "
          f"by group {report['by_group']}, skipped {len(chosen['skipped'])}", flush=True)
    return report
