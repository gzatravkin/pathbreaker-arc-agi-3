import dataclasses
import importlib
import os
import sys
import types

import numpy as np
import torch
import triton
import triton.language as tl

MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
TAIL_POOL_ENV = "TAAF_VLLM_MIXED_TAIL_POOL"
PAUSED_ENV = "TAAF_VLLM_PAUSED"
PAUSED_MODULE = "vllm.taaf_paused"
NO_BLANK_ENV = "TAAF_VLLM_NO_BLANK_LINES"
NO_BLANK_MODULE = "vllm.taaf_no_blank"
SAMPLE_TOPK_ENV = "TAAF_VLLM_SAMPLE_TOPK"
SUMMIN_ENV = "TAAF_VLLM_DRAFT_SUMMIN"
SUMMIN_MODULE = "vllm.taaf_draft_summin"
SAMPLE_TOPK_MODULE = "vllm.taaf_sample_topk"
BLOCK_VERIFY_ENV = "TAAF_VLLM_BLOCK_VERIFY"
BLOCK_VERIFY_MODULE = "vllm.taaf_block_verify"
REQ_LADDER = (8, 16, 32, 48, 64, 96, 128)
PAD_SLOT = 0
STAGE_BLOCK = 128
WARMUPS = 2
CHUNK_BYTES = 2 ** 30
SYNC_ATTRIBUTES = ("item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__", "__index__")
_SAID = set()
_STATE = {"tails": {}, "aside": {}, "taken": 0, "eager": 0, "replayed": 0}
_POOL = {"handle": None, "rows": 0, "epochs": 0, "dropped": 0}

SYNC_MAP = (
    ("serving/vllm030_pinned/gpu_model_runner.py:1552", "host branch", "input_batch.num_draft_tokens == 0 picks the plain sampler or the rejection sampler from the scheduler's counts",
     "one padded layout: every request takes the rejection path, a request with no drafts is a one-row request"),
    ("serving/vllm030_pinned/gpu_model_runner.py:1998-2181", "stream order", "sample_tokens runs logits, sampling, post_update, propose and the draft scatter as separate eager launches",
     "the sampling tail is one captured body; propose stays eager until the attention metadata is bucketed"),
    ("vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:224-241", "host loop", "chunking by logits rows reads cu_num_logits_np; int() of its entries",
     "rows are the bucket constant; a bucket above the chunk limit stands aside"),
    ("vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:300", "host read", "max_num_logprobs from the numpy request table",
     "a request asking for logprobs stands the step aside"),
    ("vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:141,279", "host read", "cu_num_logits_np.tolist() when logprobs are returned",
     "unreachable once logprob steps stand aside"),
    ("vllm/v1/worker/gpu/sample/sampler.py:220", "host decision", "np.any(needs_logits_processing[idx_mapping_np]) skips the fp32 copy and every processor",
     "the fp32 copy and the in-kernel neutral-value exits always run; penalties, logit bias, bad words and a thinking budget stand aside"),
    ("vllm/v1/worker/gpu/sample/states.py:77,89", "host decision", "np.all over temperature and min_p picks whether the kernels launch",
     "both kernels always launch; each program returns early on a neutral value read from the device table"),
    ("vllm/v1/worker/gpu/sample/states.py:97-101", "host decision", "np.any over top_k and top_p picks None or a gathered tensor",
     "top_k and top_p are always gathered from the device tables by the expanded request index"),
    ("vllm/v1/worker/gpu/sample/states.py:115,118", "host decision", "any_greedy and any_explicit_seed pick the FlashInfer sampler",
     "the verification path never uses it: the Gumbel stream is keyed by seed and position on the device"),
    ("vllm/v1/worker/gpu/sample/states.py:22-26,63-68", "per-request parameters", "temperature, top_k, top_p, min_p, seeds are UVA-backed tables written at add_request and staged by apply_staged_writes",
     "their .gpu view moves to the next round-robin UVA buffer at every copy_to_uva, and so does prefill_len.gpu; the tail holds its own device copies of fixed address, refreshed from the current views by stream-ordered copies in stage before each replay"),
    ("vllm/v1/worker/gpu/sample/gumbel.py:117-127", "RNG", "tl.randint(seed, pos) then tl.rand(gumbel_seed, vocab index): counter based, no offset state, no generator",
     "nothing to advance between replays; seed and position are device inputs, so a replay draws what the eager step draws"),
    ("serving/vllm030_pinned/rejection_sampler_utils.py:570", "RNG", "u = tl_rand32(seed, pos) per drafted row",
     "same stream; position comes from the device gather of the positions buffer"),
    ("vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:314-320", "accepted length", "get_num_sampled_and_rejected on the device, num_sampled stays a device tensor",
     "folded into one finalize kernel that also packs tokens and counts, so the scheduler read is one copy"),
    ("vllm/v1/worker/gpu/async_utils.py:140-178", "scheduler read", "async D2H copies of sampled ids and counts, then tolist() on the host after the event",
     "the packed tensor is the single thing copied; the host read after the graph is unchanged"),
    ("serving/vllm030_pinned/rejection_sampler_utils.py:484-570", "zero-length request", "a padded request with no logits row reads and writes the row of its neighbour",
     "a padded request owns one bonus row on a reserved slot, so every kernel sees a valid, ordinary request"),
    ("serving/vllm030_pinned/autoregressive_speculator.py:234-239", "host decision", "max_query_len, seq_lens_cpu_upper_bound.max().item() and draft_max_seq_len are baked into attention metadata",
     "a bucket carries constants: max_query_len = steps + 1 clamp, draft_max_seq_len = max_model_len (geometry())"),
    ("serving/vllm030_pinned/autoregressive_speculator.py:261,780-871", "grid", "prepare_prefill_inputs launches one program per real request; the last program pads the tails",
     "prepare_prefill_inputs_sized: grid is the bucket, programs past the device request count return, the last real program pads"),
    ("serving/vllm030_pinned/autoregressive_speculator.py:342,874-925", "grid", "prepare_decode_inputs uses tl.num_programs - 1 as the request count",
     "prepare_decode_inputs_sized reads the request count from a device scalar"),
    ("serving/vllm030_pinned/autoregressive_speculator.py:492", "draft sampling", "sample_draft over num_reqs rows, padded rows carry idx_mapping -1",
     "already bucket-safe: gumbel_block_argmax masks index -1; sized to the bucket request count"),
    ("serving/vllm030_pinned/gpu_model_runner.py:2158", "scatter", "req_states.draft_tokens[idx_mapping] = draft_tokens over the real rows",
     "scatter_drafts masks index -1 on the device, so the padded rows can be written"),
    ("serving/vllm030_pinned/speculator_base.py:314-337", "host build", "query_start_loc_cpu and draft_seq_lens_cpu_upper_bound built per step from the numpy batch",
     "stays host-side and outside the graph; the captured kernels read the GPU copies only"),
    ("serving/dataset/draft_vocab.py:33-41,78-90", "lazy build", "the reduced lm_head is built on the first draft call and refuses a capture in progress",
     "an eager warmup step builds it before any capture; the logits and argmax are device ops"),
    ("serving/dataset/serving_setup.py:750-1070", "MTP_LAST_ROWS", "the drafter's last_token_indices select the rows the head sees",
     "device indices from prepare_prefill_inputs; bucket-sized with the padded entries pointing at row 0"),
    ("serving/dataset/qsa_hisparse.py:1616-1639", "HOST_KV_LAST_SLICE", "the last rows are sliced on the GPU, count read on the device",
     "compatible: the slice length is the bucket request count"),
    ("serving/dataset/gdn_accept.py:511-560", "GDN_ACCEPT", "commit(num_sampled, ...) launches a grid of source_blocks.shape[0] requests after sampling",
     "num_sampled is the device tensor from the packed output; the grid is the bucket request count, owned by the GDN conversion"),
    ("serving/vllm030_pinned/adaptive_verification.py:262,312-319,417", "host sync", "copy-event synchronize and host sums over scheduled drafts",
     "adaptive verification stands the step aside"),
    ("vllm/v1/worker/gpu/spec_decode/utils.py:60-67", "scheduler read", "get_draft_tokens waits on an event, then tolist() on the host",
     "unchanged: a single read after the graph, only for structured-output requests"),
)


def mode():
    value = str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower()
    return value if value in ("plan", "1") else ""


def pool_on():
    return str(os.environ.get(TAIL_POOL_ENV) or "").strip() == "1"


def tail_pool(rows, tails, fresh=None, release=None):
    if rows <= _POOL["rows"]:
        return _POOL["handle"]
    dropped = 0
    for tail in tails:
        if tail.graph is not None:
            tail.graph = None
            dropped += 1
    if release is None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        release()
    _POOL.update(handle=(fresh or torch.cuda.graph_pool_handle)(), rows=rows, epochs=_POOL["epochs"] + 1,
                 dropped=_POOL["dropped"] + dropped)
    print(f"MIXED_TAIL_POOL epoch {_POOL['epochs']}: a {rows}-row tail is the largest so far, so the tails share a "
          f"fresh graph pool sized by it; {dropped} smaller tail graph(s) dropped with the old pool and captured "
          f"again into the new one on their next step", flush=True)
    return _POOL["handle"]


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def stand_aside(why):
    _STATE["aside"][why] = _STATE["aside"].get(why, 0) + 1
    say(f"aside-{why}", f"MIXED_FULL_SPEC stands aside for a step: {why}; that step keeps the eager sampler")


def req_bucket(reqs, max_reqs):
    if reqs > max_reqs:
        return None
    for size in REQ_LADDER:
        if reqs <= size <= max_reqs:
            return size
    return max_reqs


@dataclasses.dataclass(frozen=True)
class Geometry:
    reqs: int
    rows: int
    steps: int
    packed_width: int
    max_query_len: int
    draft_max_seq_len: int


def geometry(reqs, steps, max_model_len):
    return Geometry(reqs=reqs, rows=reqs * (steps + 1), steps=steps, packed_width=2 * (steps + 1) + 2,
                    max_query_len=steps + 1, draft_max_seq_len=max_model_len)


def fits(geom, real_reqs, real_rows):
    return real_reqs <= geom.reqs and real_rows <= real_reqs * (geom.steps + 1) and real_rows + geom.reqs - real_reqs <= geom.rows


@triton.jit
def _stage_kernel(src_cu, src_idx, src_ex_idx, src_ex_pos, src_logits_idx, src_seq_lens, real_reqs_ptr, real_rows_ptr,
                  cu, idx_safe, idx_masked, ex_idx, ex_pos, logits_idx, seq_lens,
                  reqs, rows, pad_slot, bonus_pos, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real_reqs = tl.load(real_reqs_ptr)
    real_rows = tl.load(real_rows_ptr)

    in_cu = offs <= reqs
    from_src = offs <= real_reqs
    taken = tl.load(src_cu + offs, mask=in_cu & from_src, other=0).to(tl.int32)
    tl.store(cu + offs, tl.where(from_src, taken, real_rows + (offs - real_reqs)), mask=in_cu)

    is_req = offs < reqs
    real = offs < real_reqs
    slot = tl.load(src_idx + offs, mask=is_req & real, other=0).to(tl.int32)
    tl.store(idx_safe + offs, tl.where(real, slot, pad_slot), mask=is_req)
    tl.store(idx_masked + offs, tl.where(real, slot, -1), mask=is_req)
    seq = tl.load(src_seq_lens + offs, mask=is_req & real, other=0).to(tl.int32)
    tl.store(seq_lens + offs, tl.where(real, seq, 0), mask=is_req)

    is_row = offs < rows
    real_row = offs < real_rows
    expanded = tl.load(src_ex_idx + offs, mask=is_row & real_row, other=0).to(tl.int32)
    tl.store(ex_idx + offs, tl.where(real_row, expanded, pad_slot), mask=is_row)
    local = tl.load(src_ex_pos + offs, mask=is_row & real_row, other=0).to(tl.int32)
    tl.store(ex_pos + offs, tl.where(real_row, local, bonus_pos), mask=is_row)
    gather = tl.load(src_logits_idx + offs, mask=is_row & real_row, other=0).to(tl.int64)
    tl.store(logits_idx + offs, tl.where(real_row, gather, 0), mask=is_row)


@triton.jit
def _clip_kernel(num_sampled_ptr, inputs_ptr, pos_ptr, cu_ptr, stops_ptr, num_stops, caps_ptr, real_reqs_ptr,
                 STOPS: tl.constexpr, LENGTH: tl.constexpr, PADDED: tl.constexpr):
    row = tl.program_id(0)
    if row >= tl.load(real_reqs_ptr):
        return
    start = tl.load(cu_ptr + row)
    count = tl.load(cu_ptr + row + 1) - start
    num = tl.load(num_sampled_ptr + row).to(tl.int64)
    if STOPS:
        offs = tl.arange(0, PADDED)
        inside = offs < count
        tokens = tl.load(inputs_ptr + start + offs, mask=inside, other=0).to(tl.int64)
        hit = offs < 0
        for i in range(num_stops):
            hit = hit | (tokens == tl.load(stops_ptr + i).to(tl.int64))
        first = tl.min(tl.where(hit & inside, offs, PADDED), axis=0).to(tl.int64)
        num = tl.minimum(num, tl.where(first == 0, 1, first))
    if LENGTH:
        first_pos = tl.load(pos_ptr + start).to(tl.int64)
        cap = tl.load(caps_ptr + row).to(tl.int64)
        num = tl.minimum(num, tl.maximum(cap - first_pos - 1, 1))
    tl.store(num_sampled_ptr + row, num.to(num_sampled_ptr.dtype.element_ty))


@triton.jit
def _finalize_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, seq_lens_ptr, cu_ptr, idx_ptr, prefill_len_ptr,
                     packed32_ptr, packed64_ptr, real_reqs_ptr, width32, width64, STEPS1: tl.constexpr,
                     PADDED: tl.constexpr):
    row = tl.program_id(0)
    padded = row >= tl.load(real_reqs_ptr)
    slot = tl.load(idx_ptr + row)
    seq_len = tl.load(seq_lens_ptr + row)
    prefill_len = tl.load(prefill_len_ptr + slot)
    chunked = seq_len < prefill_len
    num_sampled = tl.load(num_sampled_ptr + row)
    num_sampled = tl.where(chunked | padded, 0, num_sampled)
    logits = tl.load(cu_ptr + row + 1) - tl.load(cu_ptr + row)
    num_rejected = tl.where(chunked | padded, 0, logits - num_sampled)
    offs = tl.arange(0, PADDED)
    inside = offs < STEPS1
    tokens = tl.load(sampled_ptr + row * sampled_stride + offs, mask=inside, other=0)
    tokens = tl.where(offs < num_sampled, tokens, -1)
    tl.store(packed64_ptr + row * width64 + offs, tokens.to(tl.int64), mask=inside)
    tl.store(packed32_ptr + row * width32 + 2 * STEPS1, num_sampled.to(tl.int32))
    tl.store(packed32_ptr + row * width32 + 2 * STEPS1 + 1, num_rejected.to(tl.int32))


@triton.jit
def _prepare_prefill_sized_kernel(last_token_indices_ptr, draft_current_step_ptr, draft_input_ids_ptr, draft_positions_ptr,
                                  draft_query_start_loc_ptr, draft_seq_lens_ptr, target_input_ids_ptr,
                                  target_positions_ptr, idx_mapping_ptr, last_sampled_ptr, next_prefill_tokens_ptr,
                                  num_sampled_ptr, num_rejected_ptr, query_start_loc_ptr, seq_lens_ptr,
                                  real_reqs_ptr, max_num_reqs, BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0)
    num_reqs = tl.load(real_reqs_ptr)
    if req_idx >= num_reqs:
        return
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        tl.store(draft_current_step_ptr, 0)
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


@triton.jit
def _prepare_decode_sized_kernel(draft_tokens_ptr, draft_tokens_stride, target_seq_lens_ptr, num_rejected_ptr,
                                 input_ids_ptr, positions_ptr, sample_src_positions_ptr, query_start_loc_ptr,
                                 seq_lens_ptr, real_reqs_ptr, max_model_len, max_num_reqs, BLOCK_SIZE: tl.constexpr,
                                 ADVANCE_DRAFT_POSITIONS: tl.constexpr):
    req_idx = tl.program_id(0)
    num_reqs = tl.load(real_reqs_ptr)
    if req_idx > num_reqs:
        return
    if req_idx == num_reqs:
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


@triton.jit
def _scatter_drafts_kernel(out_ptr, out_stride, drafts_ptr, drafts_stride, idx_masked_ptr, real_reqs_ptr, steps,
                           PADDED: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(idx_masked_ptr + row)
    live = (row < tl.load(real_reqs_ptr)) & (slot >= 0)
    offs = tl.arange(0, PADDED)
    mask = (offs < steps) & live
    values = tl.load(drafts_ptr + row * drafts_stride + offs, mask=mask, other=0)
    tl.store(out_ptr + slot.to(tl.int64) * out_stride + offs, values, mask=mask)


class Layout:
    def __init__(self, geom, device):
        self.geom = geom
        reqs, rows = geom.reqs, geom.rows
        self.cu = torch.zeros(reqs + 1, dtype=torch.int32, device=device)
        self.idx_safe = torch.zeros(reqs, dtype=torch.int32, device=device)
        self.idx_masked = torch.full((reqs,), -1, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(reqs, dtype=torch.int32, device=device)
        self.ex_idx = torch.zeros(rows, dtype=torch.int32, device=device)
        self.ex_pos = torch.full((rows,), geom.steps, dtype=torch.int32, device=device)
        self.logits_idx = torch.zeros(rows, dtype=torch.int64, device=device)
        self.real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        self.real_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.caps = torch.zeros(reqs, dtype=torch.int32, device=device)
        self.packed = torch.zeros(reqs, geom.packed_width, dtype=torch.int32, device=device)

    def stage(self, idx_mapping, cu_num_logits, expanded_idx_mapping, expanded_local_pos, logits_indices, seq_lens,
              real_reqs, real_rows):
        geom = self.geom
        if not fits(geom, real_reqs, real_rows):
            raise ValueError(f"{real_reqs} requests and {real_rows} logits rows do not fit {geom.reqs} requests "
                             f"and {geom.rows} rows")
        self.real_reqs.fill_(real_reqs)
        self.real_rows.fill_(real_rows)
        span = max(geom.reqs + 1, geom.rows)
        _stage_kernel[(triton.cdiv(span, STAGE_BLOCK),)](
            cu_num_logits, idx_mapping, expanded_idx_mapping, expanded_local_pos, logits_indices, seq_lens,
            self.real_reqs, self.real_rows,
            self.cu, self.idx_safe, self.idx_masked, self.ex_idx, self.ex_pos, self.logits_idx, self.seq_lens,
            geom.reqs, geom.rows, PAD_SLOT, geom.steps, BLOCK=STAGE_BLOCK)


def clip(layout, num_sampled, inputs, pos, stops, caps):
    if stops is None and caps is None:
        return num_sampled
    _clip_kernel[(layout.geom.reqs,)](
        num_sampled, inputs, pos, layout.cu, num_sampled if stops is None else stops,
        0 if stops is None else stops.shape[0], num_sampled if caps is None else caps, layout.real_reqs,
        STOPS=stops is not None, LENGTH=caps is not None, PADDED=triton.next_power_of_2(layout.geom.steps + 1))
    return num_sampled


def paused_clips(runner):
    if os.environ.get(PAUSED_ENV, "").strip() in ("", "0"):
        return None, False
    paused = importlib.import_module(PAUSED_MODULE)
    stops = paused.stops_tensor(runner.device, runner.input_buffers.input_ids.dtype)
    if stops is not None and stops.numel() == 0:
        stops = None
    length = bool(paused.resume_unstopped())
    say("paused", f"MIXED_FULL_SPEC clips acceptance as the eager PAUSED sampler does: at the stop ids "
                  f"{[] if stops is None else stops.cpu().tolist()}"
                  f"{' and at each request max length' if length else ''}, inside the captured tail")
    return stops, length


def no_blank(runner, geom):
    if str(os.environ.get(NO_BLANK_ENV) or "").strip() in ("", "0"):
        return None
    return importlib.import_module(NO_BLANK_MODULE).for_tail(runner, geom)


def finalize(layout, sampled, num_sampled, prefill_len):
    geom = layout.geom
    packed64 = layout.packed.view(torch.int64)
    _finalize_kernel[(geom.reqs,)](
        sampled, sampled.stride(0), num_sampled, layout.seq_lens, layout.cu, layout.idx_safe, prefill_len,
        layout.packed, packed64, layout.real_reqs, geom.packed_width, geom.packed_width // 2,
        STEPS1=geom.steps + 1, PADDED=triton.next_power_of_2(geom.steps + 1))
    return layout.packed


def unpack(packed, real_reqs, steps):
    width = 2 * (steps + 1)
    rows = packed[:real_reqs]
    sampled = rows[:, :width].contiguous().view(torch.int64)
    return sampled, rows[:, width].contiguous(), rows[:, width + 1].contiguous()


def prepare_prefill_inputs_sized(layout, last_token_indices, current_draft_step, input_buffers, input_batch,
                                 num_sampled, num_rejected, last_sampled, next_prefill_tokens, max_num_reqs):
    _prepare_prefill_sized_kernel[(layout.geom.reqs,)](
        last_token_indices, current_draft_step, input_buffers.input_ids, input_buffers.positions,
        input_buffers.query_start_loc, input_buffers.seq_lens, input_batch.input_ids, input_batch.positions,
        input_batch.idx_mapping, last_sampled, next_prefill_tokens, num_sampled, num_rejected,
        input_batch.query_start_loc, input_batch.seq_lens, layout.real_reqs, max_num_reqs, BLOCK_SIZE=1024)
    return last_token_indices


def prepare_decode_inputs_sized(layout, draft_tokens, target_seq_lens, num_rejected, input_buffers,
                                sample_src_positions, max_model_len, max_num_reqs, advance_draft_positions=True):
    _prepare_decode_sized_kernel[(layout.geom.reqs + 1,)](
        draft_tokens, draft_tokens.stride(0), target_seq_lens, num_rejected, input_buffers.input_ids,
        input_buffers.positions, sample_src_positions, input_buffers.query_start_loc, input_buffers.seq_lens,
        layout.real_reqs, max_model_len, max_num_reqs, BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions)


def scatter_drafts(layout, req_state_drafts, drafts):
    steps = drafts.shape[1]
    _scatter_drafts_kernel[(layout.geom.reqs,)](
        req_state_drafts, req_state_drafts.stride(0), drafts, drafts.stride(0), layout.idx_masked, layout.real_reqs,
        steps, PADDED=triton.next_power_of_2(steps))


def sampling_static(ops, tables, logits, expanded_idx_mapping):
    processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    ops.temperature(processed, expanded_idx_mapping, tables.temperature)
    ops.min_p(processed, expanded_idx_mapping, tables.min_p)
    top_k = tables.top_k[expanded_idx_mapping]
    top_p = tables.top_p[expanded_idx_mapping]
    return ops.top_k_top_p(processed, top_k, top_p)


def verify_static(ops, tables, layout, logits, draft_logits, draft_sampled, pos, steps, synthetic_rates,
                  use_fp64, use_block_verification, prefill_len, stops=None, caps=None, ban=None):
    compact = getattr(ops, "compact", None)
    if compact is not None and draft_logits is None and synthetic_rates is None and not use_block_verification:
        _, sampled, num_sampled = compact.sample(
            ops.rejection_sample, logits, draft_sampled, layout.cu, pos, layout.idx_safe, layout.ex_idx,
            layout.ex_pos, tables.temperature, tables.top_k, tables.top_p, tables.seeds, steps, use_fp64)
    elif (compact is not None and draft_logits is not None and getattr(compact, "DRAFT", False)
          and synthetic_rates is None):
        _, sampled, num_sampled = compact.sample_draft(
            ops.rejection_sample, logits, draft_logits, draft_sampled, layout.cu, pos, layout.idx_safe,
            layout.ex_idx, layout.ex_pos, tables.temperature, tables.top_k, tables.top_p, tables.seeds, steps,
            use_fp64, use_block_verification)
    else:
        processed = sampling_static(ops, tables, logits, layout.ex_idx)
        sampled, num_sampled = ops.rejection_sample(
            processed, draft_logits, draft_sampled, layout.cu, pos, layout.idx_safe, layout.ex_idx, layout.ex_pos,
            tables.temperature, tables.seeds, steps, synthetic_rates, use_fp64=use_fp64,
            use_block_verification=use_block_verification)
    if ban is not None:
        ban.settle(layout, num_sampled)
    num_sampled = clip(layout, num_sampled, draft_sampled, pos, stops, caps)
    return finalize(layout, sampled, num_sampled, prefill_len)


def summin_module():
    if os.environ.get(SUMMIN_ENV) != "1":
        return None
    return importlib.import_module(SUMMIN_MODULE)


def compact_module():
    if str(os.environ.get(SAMPLE_TOPK_ENV) or "").strip() in ("", "0"):
        return None
    module = importlib.import_module(SAMPLE_TOPK_MODULE)
    return module if module.K else None


def stock_ops():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    return types.SimpleNamespace(temperature=apply_temperature, min_p=apply_min_p, top_k_top_p=apply_top_k_top_p,
                                 rejection_sample=rejection_sample, compact=compact_module())


def tables_of(sampler):
    states = sampler.sampling_states
    return types.SimpleNamespace(temperature=states.temperature.gpu, top_k=states.top_k.gpu, top_p=states.top_p.gpu,
                                 min_p=states.min_p.gpu, seeds=states.seeds.gpu)


def hold(sampler):
    tables = types.SimpleNamespace(**{name: view.clone() for name, view in vars(tables_of(sampler)).items()})
    return tables, sampler.req_states.prefill_len.gpu.clone()


def refresh(tables, prefill_len, sampler):
    for name, view in vars(tables_of(sampler)).items():
        getattr(tables, name).copy_(view, non_blocking=True)
    prefill_len.copy_(sampler.req_states.prefill_len.gpu, non_blocking=True)


def unsupported(runner, input_batch, grammar_output):
    sampler = runner.sampler
    rejection = runner.rejection_sampler
    reasons = []
    if rejection is None or runner.speculator is None:
        return ["no rejection sampler"]
    if not torch.cuda.is_available():
        reasons.append("no cuda device")
    if grammar_output is not None or input_batch.has_structured_output_reqs:
        reasons.append("structured output")
    if runner.batch_sharder is not None:
        reasons.append("sharded sampling")
    if getattr(rejection, "enable_adaptive_verification", False):
        reasons.append("adaptive verification")
    if getattr(rejection, "watermark_key", None) is not None:
        reasons.append("watermarking")
    if getattr(sampler, "compute_nans", False):
        reasons.append("nan counting")
    if getattr(sampler, "return_sampling_mask", False):
        reasons.append("sampling mask")
    if getattr(sampler, "trace_replay_state", None) is not None:
        reasons.append("trace replay")
    if runner.pcp_manager is not None:
        reasons.append("context parallel")
    if input_batch.num_draft_tokens == 0:
        reasons.append("no drafts scheduled")
    mapping = input_batch.idx_mapping_np
    if sampler.sampling_states.max_num_logprobs(mapping) != -1:
        reasons.append("logprobs requested")
    if np.any(sampler.penalties_state.use_penalty[mapping]):
        reasons.append("penalties")
    if np.any(sampler.logit_bias_state.use_logit_bias[mapping]):
        reasons.append("logit bias")
    if np.any(sampler.bad_words_state.num_bad_words.np[mapping] > 0):
        reasons.append("bad words")
    thinking = sampler.thinking_budget_state
    if thinking.enabled and np.any(thinking.use_thinking_budget[mapping]):
        reasons.append("thinking budget")
    compact = compact_module()
    if compact is not None:
        refused = compact.refusal(sampler, runner.speculator.draft_logits, mapping)
        if refused is not None:
            reasons.append(f"sample_topk {refused}")
    return reasons


class Tail:
    def __init__(self, runner, reqs):
        self.runner = runner
        self.rejection = runner.rejection_sampler
        self.sampler = runner.sampler
        self.steps = self.rejection.num_speculative_steps
        self.geom = geometry(reqs, self.steps, runner.max_model_len)
        self.device = runner.device
        self.layout = Layout(self.geom, self.device)
        self.ops = stock_ops()
        self.tables, self.prefill_len = hold(self.sampler)
        self.stops, self.length = paused_clips(runner)
        self.ban = no_blank(runner, self.geom)
        self.hidden_rows = None
        self.graph = None
        self.warm = 0
        self.broken = ""

    def body(self):
        runner = self.runner
        layout = self.layout
        logits = None
        compact = getattr(self.ops, "compact", None)
        drafted = runner.speculator.draft_logits is not None
        if (compact is not None and self.ban is None and self.rejection.synthetic_conditional_rates is None
                and ((not drafted and not self.rejection.use_block_verification)
                     or (drafted and getattr(compact, "DRAFT", False)))):
            logits = compact.kept_logits(runner.model, self.hidden_rows)
            if logits is not None:
                compact.say("kept", "SAMPLE_TOPK samples the sampling tail over the target vocabulary's kept "
                                    "logits, so the full-width logits buffer and its scatter are skipped")
        if logits is None:
            logits = runner.model.compute_logits(self.hidden_rows)
        draft_sampled = runner.input_buffers.input_ids[layout.logits_idx]
        pos = runner.input_buffers.positions[layout.logits_idx]
        if self.ban is not None:
            self.ban.apply(logits, layout, draft_sampled, self.tables.temperature)
        summin = summin_module()
        if summin is not None and summin.state() is not None:
            live = torch.arange(logits.shape[0], device=logits.device) < layout.real_rows
            summin.observe_target(logits, layout.ex_idx, layout.ex_pos, self.tables.temperature, self.tables.top_k,
                                  self.tables.top_p, live=live)
        return verify_static(
            self.ops, self.tables, layout, logits, runner.speculator.draft_logits, draft_sampled, pos, self.steps,
            self.rejection.synthetic_conditional_rates, self.sampler.use_fp64_gumbel,
            self.rejection.use_block_verification, self.prefill_len, stops=self.stops,
            caps=layout.caps if self.length else None, ban=self.ban)

    def stage(self, hidden_states, input_batch):
        rows = input_batch.logits_indices.shape[0]
        reqs = input_batch.num_reqs
        if self.hidden_rows is None:
            self.hidden_rows = torch.zeros(self.geom.rows, hidden_states.shape[1], dtype=hidden_states.dtype,
                                           device=hidden_states.device)
        self.hidden_rows[:rows].copy_(hidden_states[input_batch.logits_indices])
        refresh(self.tables, self.prefill_len, self.sampler)
        if self.ban is not None:
            self.ban.stage()
        if self.length:
            caps = np.asarray(self.sampler.req_states.max_seq_len)[np.asarray(input_batch.idx_mapping_np)[:reqs]]
            held = torch.from_numpy(caps.astype(np.int32))
            self.layout.caps[:reqs].copy_(held.pin_memory() if self.layout.caps.is_cuda else held, non_blocking=True)
        self.layout.stage(input_batch.idx_mapping, input_batch.cu_num_logits, input_batch.expanded_idx_mapping,
                          input_batch.expanded_local_pos, input_batch.logits_indices, input_batch.seq_lens, reqs,
                          rows)

    def capture(self):
        pool = tail_pool(self.geom.rows, list(_STATE["tails"].values())) if pool_on() else None
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph, pool=pool):
                self.body()
        except Exception as error:
            self.broken = f"{type(error).__name__}: {error}"
            print(f"MIXED_FULL_SPEC capture of the {self.geom.reqs}-request tail failed and that bucket stays eager: "
                  f"{self.broken}", flush=True)
            return
        self.graph = graph
        say(f"capture-{self.geom.reqs}", f"MIXED_FULL_SPEC captured the sampling tail for {self.geom.reqs} requests, "
            f"{self.geom.rows} logits rows, {self.steps} draft steps")

    def run(self, hidden_states, input_batch):
        self.stage(hidden_states, input_batch)
        if self.graph is None and not self.broken and self.warm >= WARMUPS and torch.cuda.is_available():
            self.capture()
        if self.graph is not None:
            _STATE["replayed"] += 1
            self.graph.replay()
            packed = self.layout.packed
        else:
            self.warm += 1
            _STATE["eager"] += 1
            packed = self.body()
        return unpack(packed.clone(), input_batch.num_reqs, self.steps)


def take(runner, hidden_states, input_batch, grammar_output):
    if mode() != "1":
        return None
    reasons = unsupported(runner, input_batch, grammar_output)
    if reasons:
        stand_aside(reasons[0])
        return None
    rows = input_batch.logits_indices.shape[0]
    reqs = input_batch.num_reqs
    size = req_bucket(reqs, runner.max_num_reqs)
    if size is None:
        stand_aside("requests over the largest bucket")
        return None
    steps = runner.rejection_sampler.num_speculative_steps
    if not fits(geometry(size, steps, 0), reqs, rows):
        stand_aside("logits rows over the bucket")
        return None
    if geometry(size, steps, 0).rows * runner.vocab_size * 4 > CHUNK_BYTES:
        stand_aside("bucket logits over the chunk limit")
        return None
    tail = _STATE["tails"].get(size)
    if tail is None:
        tail = _STATE["tails"][size] = Tail(runner, size)
        say("armed", f"MIXED_FULL_SPEC armed: speculative steps are verified in a {size}-request padded layout "
                     "with device-resident acceptance counts and one packed read-back")
    sampled, num_sampled, num_rejected = tail.run(hidden_states, input_batch)
    _STATE["taken"] += 1
    if str(os.environ.get(BLOCK_VERIFY_ENV) or "").strip() == "1":
        importlib.import_module(BLOCK_VERIFY_MODULE).tick()
    if tail.ops.compact is not None:
        tail.ops.compact.say("tail", f"the {size}-request sampling tail samples over the top {tail.ops.compact.K} "
                                     "candidates of each logits row and one -inf column")
        tail.ops.compact.took("tail")
    if tail.ban is not None:
        importlib.import_module(NO_BLANK_MODULE).after_tail()
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    output = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=num_sampled,
                           num_rejected=num_rejected)
    return output, num_sampled, num_rejected


def sync_free(function, *args, **kwargs):
    held = {}
    names = [name for name in SYNC_ATTRIBUTES if hasattr(torch.Tensor, name)]

    def refuse(name):
        original = getattr(torch.Tensor, name)

        def raised(self, *a, **k):
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller.startswith("triton"):
                return original(self, *a, **k)
            raise RuntimeError(f"host synchronisation through Tensor.{name} inside a captured body")
        return raised

    for name in names:
        held[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, refuse(name))
    try:
        return function(*args, **kwargs)
    finally:
        for name, original in held.items():
            setattr(torch.Tensor, name, original)
