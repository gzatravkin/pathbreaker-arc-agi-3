import hashlib
import json
import os
from pathlib import Path

import torch

VOCAB = Path(__file__).with_suffix(".json")
TARGET_HEAD = "_taaf_target_vocab_head"
CACHE = "_taaf_draft_vocab"
PAD_ENV = "TAAF_VLLM_DRAFT_VOCAB_PAD"
PAD_MAX = 4096
FP8_ENV = "TAAF_VLLM_DRAFT_HEAD_FP8"
_answered = []


def load(path):
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    ids = record["ids"]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record["ids_sha256"]:
        raise RuntimeError(f"the draft vocabulary {path} does not hash to its recorded ids_sha256")
    if not ids or ids[0] < 0 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise RuntimeError(f"the draft vocabulary {path} is not a sorted list of unique token ids")
    return record, hashlib.sha256(raw).hexdigest()


def fp8_on():
    return str(os.environ.get(FP8_ENV) or "").strip() == "1"


def head_linear(hidden_states, rows):
    if fp8_on():
        from vllm.taaf_draft_head_fp8 import linear

        return linear(hidden_states, rows)
    return torch.nn.functional.linear(hidden_states.to(rows.dtype), rows)


def padding():
    said = str(os.environ.get(PAD_ENV) or "").strip()
    if not said:
        return 1
    if not (said.isascii() and said.isdigit()) or not 1 <= int(said) <= PAD_MAX:
        raise RuntimeError(f"{PAD_ENV} must be an integer from 1 through {PAD_MAX}, found {said!r}")
    return int(said)


def padded(ids, multiple):
    return list(ids) + [ids[-1]] * (-len(ids) % multiple)


def build(model):
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "the draft vocabulary was first asked for inside a CUDA graph capture; it must be "
            "built by an eager draft step before any capture")
    record, digest = load(VOCAB)
    ids = record["ids"]
    packed = getattr(model.lm_head, "taaf_bf16c", None)
    weight = None if packed is not None else getattr(model.lm_head, "weight", None)
    if packed is not None:
        valid, total, device = packed.n > ids[-1], packed.n, packed.device
        found = f"{tuple(packed.shape)} {packed.dtype}"
    else:
        valid = (isinstance(weight, torch.Tensor) and weight.dim() == 2
                 and weight.is_floating_point() and weight.shape[0] > ids[-1])
        found = (f"{tuple(weight.shape)} {weight.dtype}" if isinstance(weight, torch.Tensor)
                 else type(weight).__name__)
        total = weight.shape[0] if valid else 0
        device = weight.device if valid else None
    if not valid:
        raise RuntimeError(
            f"the draft vocabulary needs a 2-D float lm_head weight with at least {ids[-1] + 1} "
            f"rows; {type(model).__name__}.lm_head.weight is {found}")
    chosen = padded(ids, padding())
    index = torch.tensor(chosen, dtype=torch.int64, device=device)
    moved = vars(model.lm_head).get(TARGET_HEAD) if packed is None else None
    if packed is not None:
        from vllm import taaf_bf16c

        rows = taaf_bf16c.decode_selected(packed, index)
    else:
        rows = weight.detach().index_select(0, index if moved is None else moved_rows(moved, index)).contiguous()
    if fp8_on():
        from vllm.taaf_draft_head_fp8 import make

        rows = make(rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vars(model)[CACHE] = (rows, index)
    extra = (f" padded to {len(chosen)} rows by repeating id {ids[-1]}"
             if len(chosen) != len(ids) else "")
    if moved is not None:
        extra += ", read through the target vocabulary's row order (its compaction moved the shared head first)"
    print(f"DRAFT_VOCAB built: {len(ids)} of {total} lm_head rows{extra} "
          f"({record.get('name')}, ids {record['ids_sha256'][:16]}, json {digest[:16]}) "
          f"as {rows.dtype} on {rows.device}", flush=True)
    return rows, index


def moved_rows(moved, index):
    _, order, _ = moved
    at = torch.searchsorted(order, index).clamp(max=order.numel() - 1)
    kept = order[at] == index
    if bool(((~kept) & (index < order.numel())).any()):
        raise RuntimeError("the draft vocabulary holds an id outside the target vocabulary whose lm_head row the "
                           "target compaction overwrote")
    return torch.where(kept, at, index)


def top_tokens_and_logits(model, hidden_states):
    built = vars(model).get(CACHE)
    if built is None:
        built = build(model)
    rows, index = built
    logits = head_linear(hidden_states, rows)
    return index[logits.argmax(dim=-1)], logits


def top_tokens(model, hidden_states):
    built = vars(model).get(CACHE)
    if built is None:
        built = build(model)
    rows, index = built
    best = head_linear(hidden_states, rows).argmax(dim=-1)
    tokens = index[best]
    if not _answered:
        _answered.append(True)
        print(f"DRAFT_VOCAB answering: {hidden_states.shape[0]} draft position(s) chose among "
              f"{rows.shape[0]} rows", flush=True)
    return tokens
