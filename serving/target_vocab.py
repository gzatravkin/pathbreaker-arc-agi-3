import hashlib
import json
from pathlib import Path

import torch

VOCAB = Path(__file__).with_suffix(".json")
CACHE = "_taaf_target_vocab"
BUFFER = "_taaf_target_vocab_out"
HEAD = "_taaf_target_vocab_head"
CHUNK = 8192
PAD = 256
_said = []


def load(path):
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    ids = record["ids"]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record["ids_sha256"]:
        raise RuntimeError(f"the target vocabulary {path} does not hash to its recorded ids_sha256")
    if not ids or ids[0] < 0 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise RuntimeError(f"the target vocabulary {path} is not a sorted list of unique token ids")
    return record, hashlib.sha256(raw).hexdigest()


def say(line):
    print(f"TARGET_VOCAB {line}", flush=True)


def shared(model, weight):
    pointer = weight.data_ptr()
    return sum(1 for _, other in model.named_parameters(remove_duplicate=False) if other.data_ptr() == pointer) > 1


def compact(model, processor):
    record, digest = load(VOCAB)
    ids = record["ids"]
    head = model.lm_head
    weight = getattr(head, "weight", None)
    why = None
    if getattr(head, "taaf_bf16c", None) is not None:
        why = "the lm_head is packed by BF16C"
    elif getattr(head, "tp_size", 1) != 1:
        why = f"the lm_head is split over {head.tp_size} tensor-parallel ranks"
    elif not (isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.is_floating_point()):
        why = f"the lm_head weight is {type(weight).__name__}, not a 2-D float tensor"
    elif weight.shape[0] <= ids[-1] or weight.shape[0] != processor.org_vocab_size:
        why = (f"the lm_head has {weight.shape[0]} rows against id {ids[-1]} and a vocabulary of "
               f"{processor.org_vocab_size}")
    elif shared(model, weight):
        why = "the lm_head weight shares its storage with another parameter"
    if why is not None:
        vars(model)[CACHE] = False
        say(f"stands aside: {why}; logits come from the whole head")
        return False
    index = torch.tensor(ids, dtype=torch.int64, device=weight.device)
    with torch.no_grad():
        for at in range(0, len(ids), CHUNK):
            piece = index[at:at + CHUNK]
            weight.data[at:at + piece.numel()].copy_(weight.data.index_select(0, piece))
    padded = min(weight.shape[0], -(-len(ids) // PAD) * PAD)
    rows = weight.data[:padded]
    vars(model)[CACHE] = (rows, index)
    vars(head)[HEAD] = (rows, index, weight.shape[0])
    ordered = order_head(head, rows, index, weight.shape[0])
    say(f"built: {len(ids)} of {weight.shape[0]} lm_head rows moved to the front of the head in place "
        f"({record.get('name')}, ids {record['ids_sha256'][:16]}, json {digest[:16]}); the product runs over "
        f"{padded} rows so its width is a multiple of {PAD}, and the logits land in a buffer whose other "
        f"columns were set to -inf once; any other caller of the same head object (the drafter shares it) "
        f"{'gets its logits back in token order the same way' if ordered else 'is left as it was'}")
    return vars(model)[CACHE]


def order_head(head, rows, index, total):
    inner = getattr(head, "quant_method", None)
    if inner is None or not callable(getattr(inner, "apply", None)):
        return False
    if getattr(inner, "taaf_target_vocab", False):
        return True

    class Ordered(type(inner)):
        taaf_target_vocab = True

        def apply(self, layer, x, bias=None):
            kept = torch.nn.functional.linear(x.to(rows.dtype), rows)[..., :index.numel()]
            if bias is not None:
                kept = kept + bias.index_select(0, index).to(kept.dtype)
            out = kept.new_full((*kept.shape[:-1], total), float("-inf"))
            out.index_copy_(out.dim() - 1, index, kept)
            return out

    wrapped = Ordered.__new__(Ordered)
    wrapped.__dict__.update(vars(inner))
    head.quant_method = wrapped
    return True


def logits(model, processor, hidden_states):
    built = vars(model).get(CACHE)
    if built is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return processor(model.lm_head, hidden_states)
        built = compact(model, processor)
    if built is False:
        return processor(model.lm_head, hidden_states)
    rows, index = built
    kept = torch.nn.functional.linear(hidden_states.to(rows.dtype), rows)[:, :index.numel()]
    if processor.soft_cap is not None:
        kept = torch.tanh(kept / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        kept = kept * processor.scale
    out = buffer(model, kept, processor.org_vocab_size)
    out.index_copy_(1, index, kept)
    if not _said:
        _said.append(True)
        say(f"answering: {kept.shape[0]} position(s) scored over {rows.shape[0]} rows")
    return out


def buffer(model, kept, vocab):
    held = vars(model).get(BUFFER)
    if (held is None or held.shape[0] < kept.shape[0] or held.shape[1] != vocab or held.dtype != kept.dtype
            or held.device != kept.device):
        held = kept.new_full((kept.shape[0], vocab), float("-inf"))
        vars(model)[BUFFER] = held
    return held[:kept.shape[0]]
