import os

ENV = "TAAF_VLLM_MM_PROCESSOR_ONLY"
_SAID = set()


def on():
    held = str(os.environ.get(ENV) or "").strip().lower() in ("1", "on", "true")
    if held and os.getpid() not in _SAID:
        _SAID.add(os.getpid())
        print(f"MM_PROCESSOR_ONLY live (pid {os.getpid()}): the multi-modal processor cache keeps each processed "
              f"item with its data in the API process and every request carries its items' data, so the engine "
              f"keeps no receiver cache that a request could miss", flush=True)
    return held


def own(data):
    import torch
    if isinstance(data, torch.Tensor):
        if data.device.type == "cpu" and data.untyped_storage().nbytes() > data.nbytes:
            return data.clone()
        return data
    if isinstance(data, (list, tuple)):
        return type(data)(own(part) for part in data)
    return data


def compact(held):
    try:
        item = held[0]
        for elem in item.values():
            elem.data = own(elem.data)
    except Exception as failure:
        print(f"MM_PROCESSOR_ONLY compact skipped: {type(failure).__name__}: {failure}", flush=True)
    return held
