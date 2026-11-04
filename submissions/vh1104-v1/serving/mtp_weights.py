import hashlib
import os
from pathlib import Path

ENV = "TAAF_VLLM_MTP_WEIGHTS"
SEARCH = Path("/kaggle/input")


def resolve_one(said):
    path = Path(said)
    if path.is_absolute():
        if not path.is_file():
            raise RuntimeError(f"{ENV}={said} is not a file")
        return path
    found = sorted(SEARCH.rglob(said)) if SEARCH.is_dir() else []
    if len(found) != 1:
        raise RuntimeError(f"{ENV}={said} matches {len(found)} file(s) under {SEARCH}, not one")
    return found[0]


def resolve(said):
    names = [part.strip() for part in str(said or "").split(",") if part.strip()]
    return [resolve_one(name) for name in names]


def swapped(weights):
    paths = resolve(os.environ.get(ENV))
    if not paths:
        yield from weights
        return
    from safetensors import safe_open
    handles = [safe_open(str(path), framework="pt", device="cpu") for path in paths]
    owner = {}
    for path, handle in zip(paths, handles):
        for name in handle.keys():
            if name in owner:
                raise RuntimeError(f"{ENV}: {name} is in both {owner[name][0].name} and {path.name}")
            owner[name] = (path, handle)
    used = {path: 0 for path in paths}
    seen = 0
    for name, weight in weights:
        seen += 1
        hit = owner.get(name)
        if hit is None:
            yield name, weight
            continue
        path, handle = hit
        value = handle.get_tensor(name)
        if tuple(value.shape) != tuple(weight.shape) or value.dtype != weight.dtype:
            raise RuntimeError(f"{ENV}: {name} is {tuple(value.shape)} {value.dtype} in {path.name} but "
                               f"{tuple(weight.shape)} {weight.dtype} in the checkpoint")
        used[path] += 1
        owner[name] = (path, None)
        yield name, value.to(weight.device)
    print("MTP_WEIGHTS: swapped " + ", ".join(f"{used[path]}/{len(list(handle.keys()))} of {path.name}"
                                            for path, handle in zip(paths, handles))
          + f" among the draft model's {seen} tensors", flush=True)
    for path, handle in zip(paths, handles):
        total = len(list(handle.keys()))
        left = sorted(n for n, (p, h) in owner.items() if p == path and h is not None)
        hasher = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(64 << 20), b""):
                hasher.update(block)
        digest = hasher.hexdigest()[:16]
        print(f"MTP_WEIGHTS: swapped {used[path]} of the draft model's {seen} tensors for {path.name} "
              f"(sha256 {digest}); {len(left)} of its {total} tensor(s) never reached the draft model", flush=True)
        if used[path] == 0:
            raise RuntimeError(f"{ENV}: no tensor of {path.name} reached the draft model, so it was tuned for "
                               f"another checkpoint: {left[:4]}")
        if left and len(paths) == 1:
            raise RuntimeError(f"{ENV}: {len(left)} tensor(s) of {path.name} never reached the draft model, "
                               f"so it was tuned for another checkpoint: {left[:4]}")
