import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

ENV = "TAAF_VLLM_TOKEN_PIN"
MARKERS_ENV = "TAAF_VLLM_TOKEN_PIN_MARKERS"
ADMIT_ENV = "TAAF_VLLM_PREP_ADMIT"
DEFAULT_MARKERS = ("<|object_ref_start|>", "<|object_ref_end|>")
COMPRESS = 4
VALUE = 3.0e38
SAY_EVERY_S = 60.0
STATE = {"slots": None, "batch": None, "width": 0, "markers": None, "token_to_req": None, "decode_rows": 0,
         "said": False, "last_said": 0.0,
         "tally": {"admitted": 0, "pinned": 0, "blocks": 0, "spans": 0, "unclosed": 0, "clipped": 0}}


def limit_tokens():
    said = str(os.environ.get(ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def on():
    return limit_tokens() > 0


def staged():
    return str(os.environ.get(ADMIT_ENV) or "").strip().lower() in ("1", "on", "true")


def width():
    blocks = max(1, limit_tokens() // COMPRESS)
    return 1 << (blocks - 1).bit_length()


def setup(max_num_reqs, device):
    if not on():
        return
    STATE["width"] = width()
    STATE["slots"] = torch.full((max_num_reqs, STATE["width"]), -1, dtype=torch.int32, device=device)
    STATE["batch"] = torch.full((max_num_reqs, STATE["width"]), -1, dtype=torch.int32, device=device)


def marker_ids(tokenizer_path):
    names = [name.strip() for name in str(os.environ.get(MARKERS_ENV) or "").split(",") if name.strip()]
    names = names or list(DEFAULT_MARKERS)
    if len(names) != 2:
        raise RuntimeError(f"{MARKERS_ENV} names {len(names)} markers, not an opening and a closing one")
    held = json.loads((Path(tokenizer_path) / "tokenizer.json").read_text(encoding="utf-8"))
    found = {token["content"]: int(token["id"]) for token in held.get("added_tokens", [])}
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"TOKEN_PIN markers {missing} are not single tokens of {tokenizer_path}")
    return found[names[0]], found[names[1]]


def spans_of(tokens, opening, closing):
    held = np.asarray(tokens, dtype=np.int64)
    opens = np.flatnonzero(held == opening)
    closes = np.flatnonzero(held == closing)
    spans, unclosed = [], 0
    for start in opens:
        if spans and start < spans[-1][1]:
            continue
        later = closes[closes > start]
        if later.size == 0:
            unclosed += 1
            break
        end = int(later[0])
        if end > start + 1:
            spans.append((int(start) + 1, end))
    return spans, unclosed


def blocks_of(spans, limit_blocks):
    blocks = []
    for start, end in spans:
        blocks.extend(range(start // COMPRESS, (end - 1) // COMPRESS + 1))
    blocks = sorted(set(blocks))
    clipped = max(0, len(blocks) - limit_blocks)
    return blocks[clipped:], clipped


def admit(runner, req_index, tokens):
    if not on() or STATE["slots"] is None:
        return
    if STATE["markers"] is None:
        STATE["markers"] = marker_ids(runner.model_config.tokenizer)
    spans, unclosed = spans_of(tokens, *STATE["markers"])
    blocks, clipped = blocks_of(spans, limit_tokens() // COMPRESS)
    row = np.full(STATE["width"], -1, dtype=np.int32)
    row[:len(blocks)] = blocks
    held = torch.from_numpy(row)
    if staged() and STATE["slots"].is_cuda:
        STATE["slots"][req_index].copy_(held.pin_memory(), non_blocking=True)
    else:
        STATE["slots"][req_index].copy_(held)
    tally = STATE["tally"]
    tally["admitted"] += 1
    tally["pinned"] += bool(blocks)
    tally["blocks"] += len(blocks)
    tally["spans"] += len(spans)
    tally["unclosed"] += unclosed
    tally["clipped"] += clipped
    say(len(tokens), spans, blocks)


def say(length, spans, blocks):
    now = time.time()
    if not STATE["said"]:
        STATE["said"] = True
        STATE["last_said"] = now
        print(f"TOKEN_PIN live (pid {os.getpid()}): markers {STATE['markers']}, up to {limit_tokens()} tokens "
              f"({STATE['width']} slots) a request; the first admitted request of {length} tokens holds "
              f"{len(spans)} marked span(s), {len(blocks)} pinned block(s)", flush=True)
    elif now - STATE["last_said"] >= SAY_EVERY_S:
        STATE["last_said"] = now
        print(f"TOKEN_PIN tally {json.dumps(STATE['tally'], sort_keys=True)}", flush=True)


def gather(idx_mapping, num_reqs):
    if STATE["batch"] is None:
        return
    STATE["batch"].fill_(-1)
    torch.index_select(STATE["slots"], 0, idx_mapping[:num_reqs].long(), out=STATE["batch"][:num_reqs])


def context(token_to_req, decode_rows):
    STATE["token_to_req"] = token_to_req
    STATE["decode_rows"] = int(decode_rows)


@triton.jit
def _pin_kernel(logits_ptr, stride_row, n_cols, row_offset, token_to_req_ptr, visible_ptr, pins_ptr, stride_pin,
                n_reqs, WIDTH: tl.constexpr, PINNED: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(token_to_req_ptr + row_offset + row)
    visible = tl.load(visible_ptr + row)
    columns = tl.arange(0, WIDTH)
    usable = (req >= 0) & (req < n_reqs)
    blocks = tl.load(pins_ptr + req * stride_pin + columns, mask=usable & (columns < WIDTH), other=-1)
    keep = usable & (blocks >= 0) & (blocks < visible) & (blocks < n_cols)
    tl.store(logits_ptr + row * stride_row + blocks, PINNED, mask=keep)


def bump(logits, visible_blocks, prefill_start=None):
    if STATE["batch"] is None or STATE["token_to_req"] is None or logits.shape[0] == 0:
        return
    offset = 0 if prefill_start is None else STATE["decode_rows"] + int(prefill_start)
    _pin_kernel[(logits.shape[0],)](
        logits, logits.stride(0), logits.shape[1], offset, STATE["token_to_req"], visible_blocks,
        STATE["batch"], STATE["batch"].stride(0), STATE["batch"].shape[0],
        WIDTH=STATE["width"], PINNED=VALUE)
