import hashlib


STOCK_SHA = "79d0540756e5049e60632f9456dee5c061d3b8671b6a4e393bd1a7913ccde915"
ANCHOR = "        assert not self.quant_method.is_monolithic"

BLOCK = '''

_TAAF_EXPERT_COUNTS: dict = {}
_TAAF_EXPERT_CALLS = [0]
_TAAF_EXPERT_PATHS: dict = {}
_TAAF_EXPERT_ON = bool(str(__import__("os").environ.get("TAAF_VLLM_COUNT_EXPERTS")
                           or "").strip() not in ("", "0", "false", "off"))


def _taaf_dump_experts() -> None:
    import json
    import os

    pid = os.getpid()
    try:
        rows = {name: held.detach().cpu().tolist()
                for name, held in _TAAF_EXPERT_COUNTS.items()}
        import base64
        import gzip

        packed = base64.b64encode(gzip.compress(
            json.dumps(rows, separators=(",", ":")).encode("utf-8"), 9)).decode()
        chunks = [packed[i:i + 7000] for i in range(0, len(packed), 7000)]
        print(f"TAAF_EXPERT_SUMMARY pid={pid} calls={_TAAF_EXPERT_CALLS[0]} "
              f"paths={json.dumps(_TAAF_EXPERT_PATHS)} layers={len(rows)} "
              f"parts={len(chunks)} chars={len(packed)}", flush=True)
        for index, chunk in enumerate(chunks):
            print(f"TAAF_EXPERT_PART pid={pid} {index}/{len(chunks)} {chunk}",
                  flush=True)
    except Exception as failure:
        print(f"TAAF_EXPERT_COUNT dump failed: {type(failure).__name__}: {failure}",
              flush=True)


def _taaf_note_logits(layer_name, router_logits, top_k, num_experts) -> None:
    import torch

    try:
        picked = torch.topk(router_logits, int(top_k), dim=-1).indices
        _taaf_note_experts(layer_name, picked, num_experts, "monolithic")
    except Exception as failure:
        print(f"TAAF_EXPERT_COUNT logits skipped: {type(failure).__name__}: {failure}",
              flush=True)


def _taaf_note_experts(layer_name, topk_ids, num_experts, path="?") -> None:
    import torch

    try:
        flat = topk_ids.detach().reshape(-1).to(torch.int64)
        width = int(num_experts) + 1
        held = _TAAF_EXPERT_COUNTS.get(layer_name)
        if held is None or held.device != flat.device:
            held = torch.zeros(width, dtype=torch.int64, device=flat.device)
            _TAAF_EXPERT_COUNTS[str(layer_name)] = held
        held.scatter_add_(0, flat.clamp(0, width - 1),
                          torch.ones_like(flat))
        _TAAF_EXPERT_CALLS[0] += 1
        _TAAF_EXPERT_PATHS[str(path)] = _TAAF_EXPERT_PATHS.get(str(path), 0) + 1
        if _TAAF_EXPERT_CALLS[0] in (1, 100, 10000):
            import os

            print(f"TAAF_EXPERT_ALIVE pid={os.getpid()} calls="
                  f"{_TAAF_EXPERT_CALLS[0]} path={path} layer={layer_name}",
                  flush=True)
        if _TAAF_EXPERT_CALLS[0] in (50000, 400000, 2000000):
            _taaf_dump_experts()
    except Exception as failure:
        print(f"TAAF_EXPERT_COUNT skipped: {type(failure).__name__}: {failure}",
              flush=True)


if _TAAF_EXPERT_ON:
    import atexit

    atexit.register(_taaf_dump_experts)
    print("TAAF_EXPERT_COUNT armed on both routing paths", flush=True)

'''

CALL = '''        assert not self.quant_method.is_monolithic
        if _TAAF_EXPERT_ON:
            _taaf_note_experts(self.layer_name, topk_ids,
                               self.global_num_experts, "modular")'''

MONO_ANCHOR = "        assert self.quant_method.is_monolithic"

MONO_CALL = '''        assert self.quant_method.is_monolithic
        if _TAAF_EXPERT_ON and router_logits is not None:
            _taaf_note_logits(self.layer_name, router_logits, self.top_k,
                              self.global_num_experts)'''


def patched(stock: bytes) -> bytes:
    if hashlib.sha256(stock).hexdigest() != STOCK_SHA:
        raise SystemExit("routed_experts.py is not the pinned stock file")
    text = stock.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"expected one anchor, found {text.count(ANCHOR)}")
    mark = '\n@PluggableLayer.register("routed_experts")\nclass RoutedExperts'
    if text.count(mark) != 1:
        raise SystemExit(f"the decorated class appears {text.count(mark)} times")
    head, _, tail = text.partition(mark)
    text = head + BLOCK + mark + tail
    text = text.replace(ANCHOR, CALL, 1)
    if text.count(MONO_ANCHOR) != 1:
        raise SystemExit(f"expected one monolithic anchor, "
                         f"found {text.count(MONO_ANCHOR)}")
    text = text.replace(MONO_ANCHOR, MONO_CALL, 1)
    return text.encode("utf-8")
