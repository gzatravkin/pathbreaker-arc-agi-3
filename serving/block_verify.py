import os
import time

import torch

from vllm.triton_utils import tl, triton

ENV = "TAAF_VLLM_BLOCK_VERIFY"
EVERY_ENV = "TAAF_VLLM_BLOCK_VERIFY_LIVE_S"
DEFAULT_EVERY_S = 60.0
U_SALT = 0x5DEECE66D
DRAFT_STEP_SALT = 1 << 26
FIELDS = ("verified", "block", "token", "longer", "shorter", "block_full", "token_full")
_state = {"ids": None, "tally": None, "host": None, "pending": None, "last": None, "started": None, "fired": False,
          "seen": [0] * len(FIELDS)}


def on() -> bool:
    return str(os.environ.get(ENV) or "").strip() == "1"


def every_s() -> float:
    said = str(os.environ.get(EVERY_ENV) or "").strip()
    return max(1.0, float(said)) if said else DEFAULT_EVERY_S


def capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def positions(sample_src_positions: torch.Tensor, draft_step) -> torch.Tensor:
    return sample_src_positions + draft_step * DRAFT_STEP_SALT


def note_draft(ids: torch.Tensor, idx_mapping: torch.Tensor, draft_step, draft_logits: torch.Tensor) -> None:
    rows, steps, top = int(draft_logits.shape[0]), int(draft_logits.shape[1]), int(ids.shape[1])
    held = _state["ids"]
    if held is None or tuple(held.shape) != (rows + 1, steps, top) or held.device != ids.device:
        if capturing():
            raise RuntimeError(f"{ENV}: the drafter's top-id table was first asked for inside a CUDA graph capture; "
                               f"an eager draft step must build it before any capture")
        held = torch.full((rows + 1, steps, top), -1, dtype=torch.int32, device=ids.device)
        _state["ids"] = held
        if _state["tally"] is None or _state["tally"].device != ids.device:
            _state["tally"] = torch.zeros(len(FIELDS), dtype=torch.int64, device=ids.device)
        print(f"{ENV} drafting: each draft step stores its top {top} ids for the {rows} request slots and {steps} "
              f"draft steps, {'noise keyed by position and step' if on() else 'read by SAMPLE_TOPK_DRAFT'}", flush=True)
    where = torch.where(idx_mapping >= 0, idx_mapping, torch.full_like(idx_mapping, rows)).long()
    step = torch.as_tensor(draft_step).to(device=ids.device, dtype=torch.long).expand_as(where)
    held.index_put_((where, step), ids.to(torch.int32))


def tally(device: torch.device) -> torch.Tensor:
    held = _state["tally"]
    if held is None:
        if capturing():
            raise RuntimeError(f"{ENV}: the verification tally was first asked for inside a CUDA graph capture")
        held = _state["tally"] = torch.zeros(len(FIELDS), dtype=torch.int64, device=device)
    return held


@triton.jit
def _lse(max_ptr, sum_ptr, row, stride, blocks_n, PADDED: tl.constexpr):
    blocks = tl.arange(0, PADDED)
    mask = blocks < blocks_n
    maxes = tl.load(max_ptr + row * stride + blocks, mask=mask, other=float("-inf"))
    sums = tl.load(sum_ptr + row * stride + blocks, mask=mask, other=0.0)
    top = tl.max(maxes, axis=0)
    return top + tl.log(tl.sum(sums * tl.exp(maxes - top)))


@triton.jit
def _sparse_residual_kernel(out_ptr, out_stride, log_p_ptr, target_ptr, target_stride, t_max_ptr, t_sum_ptr, t_stride,
                            draft_ptr, draft_stride_0, draft_stride_1, d_max_ptr, d_sum_ptr, d_stride,
                            draft_sampled_ptr, ex_idx_ptr, ex_pos_ptr, temp_ptr, ids_ptr, ids_stride_0, ids_stride_1,
                            vocab_size, steps, blocks_n, PADDED: tl.constexpr, TOP: tl.constexpr,
                            TOP_PADDED: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    step = tl.load(ex_pos_ptr + row).to(tl.int64)
    if step == 0 or step >= steps:
        return
    if tl.load(draft_sampled_ptr + row + 1) < 0:
        return
    req = tl.load(ex_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    if temp == 0.0:
        return
    weight = tl.exp(tl.load(log_p_ptr + row - 1).to(tl.float32))
    target_lse = _lse(t_max_ptr, t_sum_ptr, row, t_stride, blocks_n, PADDED)
    draft_lse = _lse(d_max_ptr, d_sum_ptr, row, d_stride, blocks_n, PADDED)
    k = tl.arange(0, TOP_PADDED)
    ids = tl.load(ids_ptr + req * ids_stride_0 + step * ids_stride_1 + k, mask=k < TOP, other=-1).to(tl.int64)
    ok = (ids >= 0) & (ids < vocab_size)
    safe = tl.where(ok, ids, 0)
    p = tl.exp(tl.load(target_ptr + row * target_stride + safe, mask=ok, other=float("-inf")).to(tl.float32)
               - target_lse)
    q = tl.exp(tl.load(draft_ptr + req * draft_stride_0 + step * draft_stride_1 + safe, mask=ok,
                       other=float("-inf")).to(tl.float32) / temp - draft_lse)
    shared = tl.sum(tl.where(ok, tl.minimum(weight * p, q), 0.0), axis=0)
    tl.store(out_ptr + row * out_stride, tl.maximum(weight - shared, 0.0))


def sparse_residual(out, log_p, target_logits, t_max, t_sum, draft_logits, d_max, d_sum, draft_sampled, ex_idx, ex_pos,
                    temperature, vocab_size, steps, blocks_n, padded) -> None:
    ids = _state["ids"]
    if ids is None:
        raise RuntimeError(f"{ENV}: verification found no drafter top-id table; TAAF_VLLM_DRAFT_SAMPLE drafting "
                           f"stores it, and block verification needs it")
    top = int(ids.shape[2])
    _sparse_residual_kernel[(out.shape[0],)](
        out, out.stride(0), log_p, target_logits, target_logits.stride(0), t_max, t_sum, t_max.stride(0),
        draft_logits, draft_logits.stride(0), draft_logits.stride(1), d_max, d_sum, d_max.stride(0), draft_sampled,
        ex_idx, ex_pos, temperature, ids, ids.stride(0), ids.stride(1), vocab_size, steps, blocks_n, PADDED=padded,
        TOP=top, TOP_PADDED=triton.next_power_of_2(top), num_warps=1)


def _line(values: list[int], label: str) -> str:
    verified, block, token, longer, shorter, block_full, token_full = values
    if not verified:
        return f"{label}: no sampled request verified"
    gain = (block - token) / (verified + token) if verified + token else 0.0
    return (f"{label}: {verified} sampled verifications, tokens a step {1 + block / verified:.4f} as a block vs "
            f"{1 + token / verified:.4f} token by token on the same draws ({gain:+.2%}), the block longer in "
            f"{longer / verified:.1%} and shorter in {shorter / verified:.1%}, every draft taken in "
            f"{block_full / verified:.1%} vs {token_full / verified:.1%}")


def report(values: list[int], now: float) -> None:
    seen = _state["seen"]
    window = [one - two for one, two in zip(values, seen)]
    _state["seen"] = list(values)
    if not _state["fired"]:
        if not values[0]:
            return
        _state["fired"] = True
        print(f"{ENV} fired: the rejection sampler verified sampled drafts as a block (Sun et al. 2403.10444, "
              f"fresh draws per step); " + _line(values, "first read") + f"; a live line follows every "
              f"{every_s():.0f} s", flush=True)
        return
    minutes = (now - (_state["started"] or now)) / 60
    print(f"{ENV} live [{minutes:.0f} min]: " + _line(window, "window") + " | " + _line(values, "since start"),
          flush=True)


def tick() -> None:
    held = _state["tally"]
    if held is None or capturing():
        return
    now = time.monotonic()
    if _state["started"] is None:
        _state["started"] = now
        _state["last"] = now - every_s()
    if not held.is_cuda:
        if now - _state["last"] >= every_s() or not _state["fired"]:
            _state["last"] = now
            report([int(one) for one in held.tolist()], now)
        return
    pending = _state["pending"]
    if pending is not None and pending.query():
        _state["pending"] = None
        report([int(one) for one in _state["host"].tolist()], now)
    if _state["pending"] is None and (now - _state["last"] >= every_s() or not _state["fired"]):
        _state["last"] = now
        if _state["host"] is None:
            _state["host"] = torch.zeros(len(FIELDS), dtype=torch.int64).pin_memory()
        _state["host"].copy_(held, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        _state["pending"] = event
