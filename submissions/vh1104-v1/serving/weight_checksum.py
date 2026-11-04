import hashlib


def tensor_digest(tensor):
    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = flat.numel()
    sample = flat.float() if numel else flat
    head = float(sample[0].item()) if numel else 0.0
    tail = float(sample[-1].item()) if numel else 0.0
    total = float(sample.sum().item()) if numel else 0.0
    return {"shape": list(detached.shape), "dtype": str(detached.dtype),
            "numel": numel, "sum": total, "head": head, "tail": tail}


def model_digest(model):
    rows = {}
    for name, tensor in sorted(model.named_parameters(remove_duplicate=False), key=lambda item: item[0]):
        rows[name] = tensor_digest(tensor)
    for name, tensor in sorted(model.named_buffers(remove_duplicate=False), key=lambda item: item[0]):
        if name not in rows:
            rows[name] = tensor_digest(tensor)
    packed = repr(sorted(rows.items())).encode("utf-8")
    return {"count": len(rows), "sha256": hashlib.sha256(packed).hexdigest(), "rows": rows}


def compare_digests(reference, candidate):
    left, right = reference.get("rows", {}), candidate.get("rows", {})
    missing = sorted(set(left) - set(right))
    added = sorted(set(right) - set(left))
    differing = sorted(name for name in set(left) & set(right) if left[name] != right[name])
    return {"identical": reference.get("sha256") == candidate.get("sha256"),
            "reference_count": reference.get("count"), "candidate_count": candidate.get("count"),
            "missing": missing, "added": added, "differing": differing}
