import os

import torch

ENV = "TAAF_VLLM_DRAFT_SAMPLE"
BLOCK_ENV = "TAAF_VLLM_BLOCK_VERIFY"
TOP_ENV = "TAAF_VLLM_DRAFT_SAMPLE_TOP"
TOPK_DRAFT_ENV = "TAAF_VLLM_SAMPLE_TOPK_DRAFT"
NARROW_ENV = "TAAF_VLLM_DRAFT_SAMPLE_NARROW"
FP8_ENV = "TAAF_VLLM_DRAFT_HEAD_FP8"
DEFAULT_TOP = 20
VOCAB_CACHE = "_taaf_draft_vocab"
BUFFER = "_taaf_draft_sample_buffer"
_said = []


def top() -> int:
    said = str(os.environ.get(TOP_ENV) or "").strip()
    if not said:
        return DEFAULT_TOP
    if not (said.isascii() and said.isdigit()) or not 1 <= int(said) <= 256:
        raise RuntimeError(f"{TOP_ENV} must be an integer from 1 through 256, found {said!r}")
    return int(said)


def draft_logits_and_ids(model, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    built = vars(model).get(VOCAB_CACHE)
    if built is None and hasattr(model, "get_top_tokens"):
        from vllm.taaf_draft_vocab import build

        built = build(model)
    if built is None:
        return model.compute_logits(hidden_states), None
    rows, index = built
    if str(os.environ.get(FP8_ENV) or "").strip() == "1":
        from vllm.taaf_draft_head_fp8 import linear

        return linear(hidden_states, rows), index
    return torch.nn.functional.linear(hidden_states.to(rows.dtype), rows), index


def buffer(owner, rows: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    held = vars(owner).get(BUFFER)
    if held is None or held.shape[0] < rows or held.shape[1] != width or held.dtype != dtype:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{ENV}: the sampling buffer was first asked for inside a CUDA graph capture; it "
                               f"must be built by an eager draft step before any capture")
        size = max(rows, int(getattr(owner, "max_num_reqs", rows)))
        held = torch.full((size, width), float("-inf"), dtype=dtype, device=device)
        vars(owner)[BUFFER] = held
    return held


def note(owner, draft_step, values, ids, sampled, idx_mapping, temperature) -> None:
    budget = getattr(owner, "_taaf_budget_conf", None)
    if budget is None:
        return
    from vllm.taaf_draft_budget import record

    record(budget, draft_step, values, ids, sampled, idx_mapping, temperature)


def narrow_on() -> bool:
    return str(os.environ.get(NARROW_ENV) or "").strip() == "1"


def top_columns(owner, hidden_states: torch.Tensor):
    logits, index = draft_logits_and_ids(owner.model, hidden_states)
    count = min(top(), logits.shape[-1])
    picked = None
    if str(os.environ.get("TAAF_VLLM_DRAFT_TOPK_TRITON") or "").strip() == "1":
        from vllm.taaf_small_kernels import draft_topk

        picked = draft_topk(logits, count)
    values, cols = picked if picked is not None else logits.float().topk(count, dim=-1)
    ids = cols if index is None else index[cols]
    return logits, values, ids, count


def narrow(owner, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
    from vllm.taaf_draft_sample_narrow import narrow_sample, prev_table

    logits, values, ids, count = top_columns(owner, hidden_states)
    prev = prev_table(owner, draft_logits, count)
    block = str(os.environ.get(BLOCK_ENV) or "").strip() == "1"
    if block or str(os.environ.get(TOPK_DRAFT_ENV) or "").strip() == "1":
        from vllm.taaf_block_verify import note_draft, positions

        note_draft(ids, idx_mapping, draft_step, draft_logits)
        if block:
            sample_src_positions = positions(sample_src_positions, draft_step)
    sampled = narrow_sample(prev, values.to(draft_logits.dtype), ids, draft_logits, draft_step, idx_mapping,
                            temperature, seeds, sample_src_positions,
                            use_fp64=bool(getattr(owner, "use_fp64_gumbel", False)))
    note(owner, draft_step, values, ids, sampled, idx_mapping, temperature)
    if not _said:
        _said.append(True)
        print(f"{NARROW_ENV} drafting: {logits.shape[0]} row(s) sample and cache over the drafter's top {count} "
              f"columns only, the {draft_logits.shape[-1]}-wide cache row kept -inf elsewhere", flush=True)
    summin = getattr(owner, "_taaf_summin", None)
    if summin is not None:
        from vllm.taaf_draft_summin import log_q

        log_q(summin, draft_step, logits, idx_mapping, owner.model)
    return sampled


def truncated(owner, hidden_states: torch.Tensor, width: int, dtype: torch.dtype):
    logits, values, ids, count = top_columns(owner, hidden_states)
    full = buffer(owner, logits.shape[0], width, dtype, logits.device)[: logits.shape[0]]
    full.scatter_(1, ids, values.to(dtype))
    if not _said:
        _said.append(True)
        print(f"{ENV} drafting: {logits.shape[0]} row(s) sample from the drafter's top {count} of "
              f"{logits.shape[-1]} columns, the rest -inf in a {width}-wide row", flush=True)
    return full, ids, logits


def sample(owner, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
    if getattr(owner, "draft_watermarker", None) is not None:
        raise RuntimeError(f"{ENV} drafts through its own sampler, which does not watermark; turn the draft "
                           f"watermark off")
    if narrow_on():
        return narrow(owner, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step,
                      draft_logits)
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    full, ids, logits = truncated(owner, hidden_states, draft_logits.shape[-1], draft_logits.dtype)
    block = str(os.environ.get(BLOCK_ENV) or "").strip() == "1"
    if block or str(os.environ.get(TOPK_DRAFT_ENV) or "").strip() == "1":
        from vllm.taaf_block_verify import note_draft, positions

        note_draft(ids, idx_mapping, draft_step, draft_logits)
        if block:
            sample_src_positions = positions(sample_src_positions, draft_step)
    sampled = gumbel_sample(full, idx_mapping, temperature, seeds, sample_src_positions, apply_temperature=True,
                            is_drafting=True, logits_cache=draft_logits, logits_cache_col=draft_step,
                            use_fp64=bool(getattr(owner, "use_fp64_gumbel", False)))
    if getattr(owner, "_taaf_budget_conf", None) is not None:
        note(owner, draft_step, full.gather(1, ids), ids, sampled, idx_mapping, temperature)
    full.scatter_(1, ids, float("-inf"))
    summin = getattr(owner, "_taaf_summin", None)
    if summin is not None:
        from vllm.taaf_draft_summin import log_q

        log_q(summin, draft_step, logits, idx_mapping, owner.model)
    return sampled
