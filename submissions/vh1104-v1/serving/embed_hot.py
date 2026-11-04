import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

ENV = "TAAF_VLLM_EMBED_HOT"
FILE_ENV = "TAAF_VLLM_EMBED_HOT_FILE"
DEFAULT_FILE = Path(__file__).with_name("taaf_embed_hot_ids.json")
BLOCK = 512
_TABLES = {}
_SAID = set()


def rows():
    said = str(os.environ.get(ENV) or "").strip()
    if said in ("", "0"):
        return 0
    if not said.isdigit() or int(said) <= 0:
        raise RuntimeError(f"{ENV}={said!r} is not a positive row count")
    return int(said)


def ranked_ids(path=None):
    path = Path(path or os.environ.get(FILE_ENV) or DEFAULT_FILE)
    held = json.loads(path.read_text(encoding="utf-8"))
    ids = held["ids"] if isinstance(held, dict) else held
    return [int(i) for i in ids]


@triton.jit(do_not_specialize=["count"])
def _embed_hot_kernel(ids_ptr, slot_ptr, hot_ptr, host_ptr, out_ptr, count, DIM: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    token = tl.load(ids_ptr + row).to(tl.int64)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    live = cols < DIM
    hit = slot >= 0
    from_hot = tl.load(hot_ptr + tl.maximum(slot, 0) * DIM + cols, mask=live & hit, other=0.0)
    from_host = tl.load(host_ptr + token * DIM + cols, mask=live & (slot < 0), other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * DIM + cols, tl.where(hit, from_hot, from_host), mask=live)


def gather(ids, slot_of, hot, host):
    flat = ids.reshape(-1)
    dim = host.shape[1]
    out = torch.empty((flat.numel(), dim), dtype=host.dtype, device=ids.device)
    if flat.numel():
        _embed_hot_kernel[(flat.numel(), triton.cdiv(dim, BLOCK))](flat, slot_of, hot, host, out, flat.numel(),
                                                                  DIM=dim, BLOCK=BLOCK)
    return out.view(*ids.shape, dim)


def _embed_hot_op(ids: torch.Tensor, key: int) -> torch.Tensor:
    slot_of, hot, host = _TABLES[key]
    return gather(ids, slot_of, hot, host)


def _embed_hot_fake(ids: torch.Tensor, key: int) -> torch.Tensor:
    _, _, host = _TABLES[key]
    return ids.new_empty((*ids.shape, host.shape[1]), dtype=host.dtype)


direct_register_custom_op(op_name="taaf_embed_hot", op_func=_embed_hot_op, fake_impl=_embed_hot_fake)


class HotEmbedding:
    def __init__(self, inner, key):
        self.inner = inner
        self.key = key

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embedding(self, layer, input_):
        return torch.ops.vllm.taaf_embed_hot(input_, self.key)


def build(host_view, host_cpu, ids, limit, device):
    vocab = host_view.shape[0]
    kept = [i for i in dict.fromkeys(ids) if 0 <= i < vocab][:limit]
    index = torch.tensor(kept, dtype=torch.int64)
    hot = host_cpu.index_select(0, index).to(device) if kept else torch.empty((1, host_view.shape[1]),
                                                                             dtype=host_view.dtype, device=device)
    slot_of = torch.full((vocab,), -1, dtype=torch.int32, device=device)
    if kept:
        slot_of[index.to(device)] = torch.arange(len(kept), dtype=torch.int32, device=device)
    return slot_of, hot, len(kept)


def config_token_ids(config, seen=None):
    seen = set() if seen is None else seen
    if config is None or id(config) in seen:
        return []
    seen.add(id(config))
    found = []
    for name, value in sorted(vars(config).items()):
        if name.endswith("token_id") and isinstance(value, int) and value >= 0:
            found.append(value)
        elif name.endswith("_config") and hasattr(value, "__dict__"):
            found.extend(config_token_ids(value, seen))
    return found


def model_token_ids():
    from vllm.config import get_current_vllm_config

    model = getattr(get_current_vllm_config(), "model_config", None)
    return config_token_ids(getattr(model, "hf_config", None))


def install(module, host_cpu, device):
    limit = rows()
    if not limit:
        return None
    special = model_token_ids()
    ids = special + ranked_ids()
    slot_of, hot, kept = build(module.weight, host_cpu, ids, limit, device)
    key = len(_TABLES)
    _TABLES[key] = (slot_of, hot, module.weight)
    module.quant_method = HotEmbedding(module.quant_method, key)
    size = hot.numel() * hot.element_size() + slot_of.numel() * 4
    if key not in _SAID:
        _SAID.add(key)
        print(f"EMBED_HOT armed: {kept} of {module.weight.shape[0]} rows held on the GPU ({size / 2**20:.1f} MiB), "
              f"the model config's special token ids {sorted(set(special))} first, "
              f"the rest read from the pinned host table; lookups equal F.embedding bit for bit", flush=True)
    return {"rows": kept, "bytes": size}
