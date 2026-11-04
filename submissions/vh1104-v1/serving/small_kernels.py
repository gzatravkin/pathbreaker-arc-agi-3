import os
import sys

import torch

MARLIN_ENV = "TAAF_VLLM_MARLIN_WORKSPACE_REUSE"
DRAFT_TOPK_ENV = "TAAF_VLLM_DRAFT_TOPK_TRITON"
INVALIDATE_ENV = "TAAF_VLLM_HISPARSE_INVALIDATE_FUSED"
GLUE_ENV = "TAAF_VLLM_SAMPLE_GLUE_FUSED"
_workspaces = {}
_said = set()
_tally = {"draft_topk_fired": 0, "draft_topk_stock": 0}


def flag(name):
    return str(os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def say(key, line):
    if key not in _said:
        _said.add(key)
        print(line, file=sys.stderr, flush=True)


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def marlin_workspace(device, blocks_per_sm, make):
    if not flag(MARLIN_ENV):
        return make(device, blocks_per_sm)
    key = (device.type, device.index, blocks_per_sm)
    held = _workspaces.get(key)
    if held is None:
        if _capturing():
            say("marlin-capture", f"MARLIN_WORKSPACE_REUSE stands aside inside a graph capture that precedes any "
                                  f"eager call: this layer keeps a fresh zeroed workspace")
            return make(device, blocks_per_sm)
        held = make(device, blocks_per_sm)
        _workspaces[key] = held
    say("marlin-fired", f"MARLIN_WORKSPACE_REUSE fires: every Marlin MoE call on {device} shares one "
                        f"{held.numel()}-lock workspace instead of zero-filling a new one per layer")
    return held


def draft_topk(logits, count):
    if not flag(DRAFT_TOPK_ENV):
        return None
    from vllm.taaf_draft_topk_kernel import supported, topk

    if not supported(logits, count):
        _tally["draft_topk_stock"] += 1
        say("draft-stock", f"DRAFT_TOPK_TRITON stands aside for logits {tuple(logits.shape)} {logits.dtype} "
                           f"top {count}: torch.topk runs")
        return None
    _tally["draft_topk_fired"] += 1
    say("draft-fired", f"DRAFT_TOPK_TRITON fires: the drafter's top {count} of {logits.shape[1]} columns comes "
                       f"from a two-kernel Triton top-k over the bf16 logits, not torch.topk on a float copy")
    return topk(logits, count)


def invalidate_sorted(device_global_indices, sorted_slots, state_indices):
    if not flag(INVALIDATE_ENV):
        return False
    if (not device_global_indices.is_cuda or device_global_indices.dtype != torch.int32
            or sorted_slots.dtype != torch.int32 or device_global_indices.stride(1) != 1
            or state_indices.dtype not in (torch.int32, torch.int64)):
        say("invalidate-stock", f"HISPARSE_INVALIDATE_FUSED stands aside for {device_global_indices.dtype}/"
                                f"{sorted_slots.dtype} inputs: the stock seven-op invalidation runs")
        return False
    from vllm.taaf_small_kernels_triton import invalidate_sorted as run

    say("invalidate-fired", "HISPARSE_INVALIDATE_FUSED fires: each leader runtime drops recycled hot slots with "
                            "one Triton kernel in place of index_select, searchsorted, clamp, gather, compare, "
                            "masked_fill and index_copy")
    run(device_global_indices, sorted_slots, state_indices)
    return True


def glue_on():
    return flag(GLUE_ENV)


GEMMA_NORM_ENV = "TAAF_VLLM_GEMMA_NORM_FUSED"
_gemma_state = {"stock": None}


def _gemma_fusable(x, residual, weight):
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if x.dim() < 1 or x.stride(-1) != 1 or not x.is_contiguous():
        return False
    if not weight.is_cuda or weight.numel() != x.shape[-1]:
        return False
    if residual is not None and (residual.dtype != x.dtype or residual.shape != x.shape or not residual.is_contiguous()):
        return False
    return True


def gemma_forward_cuda(self, x, residual=None):
    stock = _gemma_state["stock"]
    if not flag(GEMMA_NORM_ENV) or torch.compiler.is_compiling() or not _gemma_fusable(x, residual, self.weight):
        return stock(self, x, residual)
    from vllm.taaf_small_kernels_triton import gemma_norm

    rows = x.numel() // x.shape[-1]
    say("gemma-fired", f"GEMMA_NORM_FUSED fires: a GemmaRMSNorm over {rows} rows x {x.shape[-1]} "
                       f"{x.dtype} (residual {'yes' if residual is not None else 'no'}) runs as one Triton kernel "
                       f"in place of the eager float upcast, pow, mean, rsqrt and two multiplies")
    out, new_residual = gemma_norm(x, residual, self.weight, self.variance_epsilon)
    return out if residual is None else (out, new_residual)


def install_gemma_norm(cls):
    if getattr(cls, "_taaf_gemma_fused", False):
        return
    _gemma_state["stock"] = cls.forward_cuda
    cls.forward_cuda = gemma_forward_cuda
    cls._taaf_gemma_fused = True
    say("gemma-armed", "GEMMA_NORM_FUSED installed on GemmaRMSNorm.forward_cuda")
