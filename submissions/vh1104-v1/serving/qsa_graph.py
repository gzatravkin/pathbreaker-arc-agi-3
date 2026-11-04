import dataclasses
import importlib
import os

import torch
import triton
import triton.language as tl

MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
SHARED_ENV = "TAAF_VLLM_MIXED_PREP_SHARED"
HELPER_MODULE = "vllm.taaf_mixed_full"
ROW_BLOCK = 1024
COLUMN_BLOCK = 512
SPLIT_DECODE_TOKENS = 1
SPLIT_TOTAL = 4
SPLIT_WIDTH = 5
_SAID = set()
_PLANS = {}
_ASIDE = {}
_WITNESS = {"qsa": [0, 0]}
_REASON = {"qsa": ""}


def armed():
    return str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower() == "1"


def split_async():
    return str(os.environ.get(SHARED_ENV) or "").strip().lower() in ("1", "on", "true")


def upload_split(split, values):
    if not split_async():
        split.copy_(torch.tensor(values, dtype=torch.int32))
        return
    held = torch.tensor(values, dtype=torch.int32, device="cpu")
    if split.is_cuda:
        held = held.pin_memory()
    split.copy_(held, non_blocking=split.is_cuda)
    say("split-async", "QSA_GRAPH split upload is sync-free: the five split counts go from pinned host memory "
                       "without a stream wait, so the next step's metadata builds while the GPU still runs the last")


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def aside(reason, line):
    _REASON["qsa"] = reason
    _ASIDE[reason] = _ASIDE.get(reason, 0) + 1
    say(f"aside-{reason}", f"QSA_GRAPH stands aside for a step: {line}; that step keeps its eager QSA path")


class Bucketed(int):
    tokens = 0


@dataclasses.dataclass
class Bucket:
    tokens: int
    decode_len: int
    split: torch.Tensor
    visible_decode: torch.Tensor
    visible_prefill: torch.Tensor


def prefill_flag(bucket, stock):
    if bucket is None:
        return stock
    flag = Bucketed(1)
    flag.tokens = bucket.tokens
    return flag


def declared_support(default):
    if not armed():
        return default
    from vllm.v1.attention.backend import AttentionCGSupport

    say("support", f"QSA_GRAPH declares mixed-batch FULL graph support for the QSA backends under {MIXED_FULL_ENV}=1; "
                   "a step that does not fit a bucket builds no bucket metadata and keeps the eager QSA path")
    return AttentionCGSupport.ALWAYS


def _plan_module():
    return importlib.import_module(HELPER_MODULE)


def _plan(threshold):
    held = _PLANS.get(threshold)
    if held is None:
        held = _PLANS[threshold] = importlib.import_module(HELPER_MODULE).plan_from_env(threshold)
    return held


def unfit_reason(lens, num_tokens, num_decodes, threshold):
    plan = _plan(threshold)
    bucket, why = plan.fit(lens, num_tokens)
    if why:
        return why
    if bucket.tokens != num_tokens:
        return "padded_tokens_not_a_bucket_size"
    if plan.split(lens) != num_decodes:
        return "decode_split_differs_from_the_bucket_rule"
    if num_decodes and int(lens[0]) != threshold:
        return "decode_query_length_is_not_the_declared_one"
    return ""


def _buffers(builder):
    held = getattr(builder, "_qsa_graph_buffers", None)
    if held is None:
        rows = builder.token_to_req_buffer.shape[0]
        device = builder.token_to_req_buffer.device
        held = builder._qsa_graph_buffers = (
            torch.zeros(SPLIT_WIDTH, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
        )
    return held


@triton.jit
def _mask_visible_kernel(visible_ptr, split_ptr, decode_ptr, prefill_ptr, rows, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = row < rows
    decode_tokens = tl.load(split_ptr + 1)
    total = tl.load(split_ptr + 4)
    value = tl.load(visible_ptr + row, mask=inside & (row < total), other=0)
    is_decode = row < decode_tokens
    tl.store(decode_ptr + row, tl.where(is_decode, value, 0), mask=inside)
    tl.store(prefill_ptr + row, tl.where(is_decode, 0, value), mask=inside)


@triton.jit
def _merge_decode_rows_kernel(dst_ptr, src_ptr, split_ptr, stride_dst, stride_src, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= tl.load(split_ptr + 1):
        return
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = columns < width
    values = tl.load(src_ptr + row * stride_src + columns, mask=inside)
    tl.store(dst_ptr + row * stride_dst + columns, values, mask=inside)


def split_values(num_decodes, num_decode_tokens, lens):
    total = int(sum(int(x) for x in lens))
    return [int(num_decodes), int(num_decode_tokens), len(lens) - int(num_decodes), total - int(num_decode_tokens),
            total]


def witness():
    return {name: tuple(pair) for name, pair in _WITNESS.items()}


def reasons():
    return dict(_REASON)


def prepare(builder, common, num_decodes, num_decode_tokens, visible_blocks):
    if not armed() or not _plan_module().graph_step():
        return None
    _WITNESS["qsa"][0] += 1
    _REASON["qsa"] = ""
    num_tokens = common.num_actual_tokens
    starts = common.query_start_loc_cpu
    lens = (starts[1:] - starts[:-1]).tolist()
    while lens and lens[-1] == 0:
        lens.pop()
    threshold = builder.reorder_batch_threshold
    why = unfit_reason(lens, num_tokens, num_decodes, threshold)
    if why == "decode_only":
        _REASON["qsa"] = why
        return None
    if why:
        aside(why, f"{why} (query lengths {[int(x) for x in lens][:12]}{'...' if len(lens) > 12 else ''}, "
                   f"{num_tokens} padded tokens)")
        return None
    split, decode, prefill = _buffers(builder)
    values = split_values(num_decodes, num_decode_tokens, lens)
    upload_split(split, values)
    _mask_visible_kernel[(triton.cdiv(num_tokens, ROW_BLOCK),)](
        visible_blocks, split, decode, prefill, num_tokens, BLOCK=ROW_BLOCK)
    say("live", f"QSA_GRAPH live: a mixed step of {len(lens)} requests and {values[4]} real tokens in {num_tokens} "
                f"padded rows builds bucket metadata; indexer and attention read the {values[0]} decode requests "
                "and the padding from device tensors")
    _WITNESS["qsa"][1] += 1
    return Bucket(num_tokens, threshold, split, decode[:num_tokens], prefill[:num_tokens])


def bucket_select(decode_op, prefill_op, q, k_cache, meta, token_topk, compress_ratio, block_indices):
    bucket = meta.bucket
    rows = q.shape[0]
    table = meta.block_table
    columns = table.shape[1] * k_cache.shape[1]
    prefill_op(
        q,
        k_cache,
        table,
        meta.query_start_loc,
        bucket.visible_prefill[:rows],
        token_topk,
        compress_ratio,
        rows,
        block_indices,
        columns * compress_ratio,
    )
    length = bucket.decode_len
    requests = min(table.shape[0], rows // length) if length else 0
    if not requests:
        return
    decode_rows = requests * length
    scratch = torch.empty_like(block_indices[:decode_rows])
    decode_op(
        q[:decode_rows],
        k_cache,
        table[:requests],
        bucket.visible_decode[:decode_rows],
        token_topk,
        compress_ratio,
        length,
        scratch,
    )
    _merge_decode_rows_kernel[(decode_rows, triton.cdiv(scratch.shape[1], COLUMN_BLOCK))](
        block_indices, scratch, bucket.split, block_indices.stride(0), scratch.stride(0), scratch.shape[1],
        BLOCK=COLUMN_BLOCK)
