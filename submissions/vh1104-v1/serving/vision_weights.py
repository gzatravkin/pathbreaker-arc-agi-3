import hashlib
import os
from pathlib import Path

ENV = "TAAF_VLLM_VISION_WEIGHTS"
SEARCH = Path("/kaggle/input")
PREFIX = "model.visual."


def resolve(said):
    said = str(said or "").strip()
    if not said:
        return None
    path = Path(said)
    if path.is_absolute():
        if not path.is_file():
            raise RuntimeError(f"{ENV}={said} is not a file")
        return path
    found = sorted(SEARCH.rglob(said)) if SEARCH.is_dir() else []
    if len(found) != 1:
        raise RuntimeError(f"{ENV}={said} matches {len(found)} file(s) under {SEARCH}, not one")
    return found[0]


def swapped(weights):
    path = resolve(os.environ.get(ENV))
    if path is None:
        yield from weights
        return
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        stray = sorted(name for name in names if not name.startswith(PREFIX))
        if stray:
            raise RuntimeError(f"{ENV}: {path.name} carries {len(stray)} tensor(s) outside {PREFIX}: {stray[:4]}")
        used = set()
        visual = 0
        for name, weight in weights:
            if name.startswith(PREFIX):
                visual += 1
            if name in names:
                value = handle.get_tensor(name)
                if tuple(value.shape) != tuple(weight.shape) or value.dtype != weight.dtype:
                    raise RuntimeError(
                        f"{ENV}: {name} is {tuple(value.shape)} {value.dtype} in {path.name} but "
                        f"{tuple(weight.shape)} {weight.dtype} in the checkpoint")
                used.add(name)
                yield name, value.to(weight.device)
                continue
            yield name, weight
    left = sorted(names - used)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"VISION_WEIGHTS: swapped {len(used)} of the checkpoint's {visual} vision tensors for {path.name} "
          f"(sha256 {digest})", flush=True)
    if left or not used:
        raise RuntimeError(f"{ENV}: {len(left)} of the {len(names)} tensor(s) of {path.name} never reached the "
                           f"vision tower, so they were trained for another checkpoint: {left[:4]}")
