import dataclasses
import math
import os
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
SEQ_CAP_ENV = "TAAF_VLLM_MIXED_FULL_SEQS"
MIXED_LARGEST_ENV = "TAAF_VLLM_MIXED_FULL_LARGEST"
LARGEST_ENV = "TAAF_VLLM_PIECEWISE_LARGE"
STEP_ENV = "TAAF_VLLM_PIECEWISE_LARGE_STEP"
MAX_SEQS_ENV = "TAAF_VLLM_MAX_NUM_SEQS"
BATCHED_ENV = "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"
FLA_CHUNK = 64
CONV_BLOCK_M = 8
PAD_SLOT_ID = -1
SEQ_CAP = 16
CHECKS = 3
TALLY_S = 15.0
PROBE_BLOCK = 128
OUTSIDE_THE_GRAPH = ("mtp_draft", "sampling_tail", "gdn_accept_commit")
_SAID = set()
_VETOES = {}
_STATE = {"at": 0.0, "checked": 0, "plan": None, "aside": ""}
_TALLY = {"steps": 0, "decode_only": 0, "mixed": 0, "fit": 0, "pad_tokens": 0, "real_tokens": 0}
_ASIDE = {}
_BUCKETS = {}
_GRAPH_STEP = [False, 0]


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def mode():
    value = str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower()
    if value in ("", "0", "off"):
        return ""
    if value in ("plan", "1"):
        return value
    raise RuntimeError(f"{MIXED_FULL_ENV}={value!r} is not one of '', '0', 'plan', '1'.")


def set_graph_step(flag, tokens=0):
    _GRAPH_STEP[0] = bool(flag)
    _GRAPH_STEP[1] = int(tokens) if flag else 0


def graph_step():
    return _GRAPH_STEP[0]


def graph_tokens():
    return _GRAPH_STEP[1]


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


@dataclasses.dataclass(frozen=True)
class Bucket:
    tokens: int
    reqs: int
    seqs: int

    def chunk_rows(self):
        return self.tokens // FLA_CHUNK + self.seqs

    def conv_programs(self):
        return max(1024, self.tokens // CONV_BLOCK_M + self.seqs) * 2


def split_of(query_lens, qlen):
    lens = [int(x) for x in query_lens]
    if not lens or lens[0] > int(qlen):
        return 0
    run = 0
    while run < len(lens) and lens[run] == lens[0]:
        run += 1
    return run


def lead_reason(query_lens, drafted, prefilling, qlen):
    lens = [int(x) for x in query_lens]
    qlen = int(qlen)
    split = split_of(lens, qlen)
    if any(length != qlen for length in lens[:split]):
        return "decode_rows_not_the_decode_length"
    if qlen == 1:
        return ""
    drafts = [int(x) for x in drafted][:len(lens)]
    flags = [bool(x) for x in prefilling][:len(lens)]
    if len(drafts) != len(lens) or len(flags) != len(lens):
        return "decode_rows_unknown"
    spec = [count > 0 and length == count + 1 for length, count in zip(lens, drafts)]
    if not all(spec[:split]):
        return "undrafted_row_among_the_decode_rows"
    late = zip(drafts[split:], flags[split:], lens[split:])
    if any(count > 0 or (not flag and 0 < length <= qlen) for count, flag, length in late):
        return "decode_row_after_the_decode_rows"
    return ""


def ladder(largest, step, floor):
    first = max(floor, step)
    sizes = list(range(first, largest + 1, step))
    if not sizes or sizes[-1] != largest:
        sizes.append(largest)
    return sizes


class Plan:
    def __init__(self, largest, step=32, floor=64, max_reqs=128, qlen=4, seq_cap=SEQ_CAP):
        self.largest = int(largest)
        self.step = int(step)
        self.floor = int(floor)
        self.max_reqs = int(max_reqs)
        self.qlen = int(qlen)
        self.seq_cap = int(seq_cap)
        self.sizes = ladder(self.largest, self.step, self.floor)

    def bucket_for(self, tokens):
        for size in self.sizes:
            if size >= tokens:
                return Bucket(size, min(size, self.max_reqs), self.seq_cap)
        return None

    def split(self, query_lens):
        return split_of(query_lens, self.qlen)

    def fit(self, query_lens, padded_tokens=None):
        lens = [int(x) for x in query_lens]
        if not lens:
            return None, "empty"
        decodes = self.split(lens)
        if decodes == len(lens):
            return None, "decode_only"
        total = sum(lens)
        if len(lens) - decodes > self.seq_cap:
            return None, "prefill_seqs_over_cap"
        bucket = self.bucket_for(total if padded_tokens is None else max(total, int(padded_tokens)))
        if bucket is None:
            return None, "tokens_over_largest"
        if len(lens) > bucket.reqs:
            return None, "requests_over_bucket"
        return bucket, ""


def plan_from_env(decode_query_len):
    batched = int(os.environ.get(BATCHED_ENV) or 2048)
    largest = int(os.environ.get(LARGEST_ENV) or batched)
    cap = int(os.environ.get(MIXED_LARGEST_ENV) or 0)
    if cap:
        largest = min(largest, cap)
    return Plan(largest, step=int(os.environ.get(STEP_ENV) or 32), max_reqs=int(os.environ.get(MAX_SEQS_ENV) or 128),
                qlen=int(decode_query_len), seq_cap=int(os.environ.get(SEQ_CAP_ENV) or SEQ_CAP))


def graph_memory(sizes, kernels_per_graph, node_bytes=1536, logits_width=0, logits_itemsize=4,
                 static_bytes_per_token=64):
    graphs = len(sizes)
    nodes = graphs * kernels_per_graph * node_bytes
    largest = max(sizes) if sizes else 0
    logits = largest * logits_width * logits_itemsize
    static = largest * static_bytes_per_token
    return {"graphs": graphs, "node_bytes": nodes, "logits_scratch_bytes": logits, "static_bytes": static,
            "total_bytes": nodes + logits + static}


def tally():
    now = time.monotonic()
    if now - _STATE["at"] < TALLY_S:
        return
    _STATE["at"] = now
    top = sorted(_BUCKETS.items(), key=lambda kv: -kv[1])[:4]
    waste = (_TALLY["pad_tokens"] / _TALLY["real_tokens"]) if _TALLY["real_tokens"] else 0.0
    print(f"MIXED_FULL tally {dict(_TALLY)} aside {dict(_ASIDE)} top_buckets {top} pad_ratio {waste:.4f}",
          flush=True)


def observe(batch_state, batch_desc, decode_query_len):
    if not mode() or batch_state is None:
        return
    plan = _STATE["plan"]
    if plan is None:
        plan = _STATE["plan"] = plan_from_env(decode_query_len)
        say("armed", f"MIXED_FULL {mode()}: mixed steps are bucketed to the {len(plan.sizes)} padded token sizes "
                     f"{plan.sizes[0]}..{plan.sizes[-1]} step {plan.step}, at most {plan.seq_cap} prefill sequences "
                     f"and {plan.max_reqs} requests; the decode/prefill split is read on the device, so a bucket "
                     "is its token size")
    lens = batch_state.num_scheduled_tokens
    _TALLY["steps"] += 1
    padded = getattr(batch_desc, "num_tokens", None)
    bucket, why = plan.fit(lens, padded)
    if why == "decode_only":
        _TALLY["decode_only"] += 1
        tally()
        return
    _TALLY["mixed"] += 1
    if bucket is None:
        _ASIDE[why] = _ASIDE.get(why, 0) + 1
        say(f"aside-{why}", f"MIXED_FULL stands aside for a step: {why} (query lengths "
                            f"{[int(x) for x in lens][:12]}{'...' if len(lens) > 12 else ''}); that step keeps its "
                            "piecewise graphs")
        tally()
        return
    _TALLY["fit"] += 1
    real = int(sum(int(x) for x in lens))
    _TALLY["real_tokens"] += real
    _TALLY["pad_tokens"] += bucket.tokens - real
    _BUCKETS[bucket.tokens] = _BUCKETS.get(bucket.tokens, 0) + 1
    say("live", f"MIXED_FULL live: the first mixed step of {len(lens)} requests and {real} tokens fits the "
                f"{bucket.tokens}-token bucket ({bucket.reqs} requests, {bucket.seqs} prefill sequences)")
    if mode() == "1":
        probe(lens, bucket)
        say("flip", "MIXED_FULL graph: a fitting mixed step replays the full graph of its bucket; outside the graph "
                    "and eager by design: " + ", ".join(OUTSIDE_THE_GRAPH))
    tally()


def ensure_plan(decode_query_len):
    if _STATE["plan"] is None:
        _STATE["plan"] = plan_from_env(decode_query_len)
    return _STATE["plan"]


def register_veto(name, fn):
    _VETOES[name] = fn


def prefill_bounds(lens, computed):
    lens = [int(x) for x in lens]
    plan = _STATE["plan"]
    lead = plan.split(lens) if plan is not None else 0
    return [int(c) + int(n) for c, n in zip(computed, lens)][lead:]


def admit(lens, padded, computed=None, rows=None):
    plan = _STATE["plan"]
    if plan is None:
        return None, "no_plan"
    lens = [int(x) for x in lens]
    bucket, why = plan.fit(lens, padded)
    if bucket is None:
        return None, why
    if bucket.tokens != int(padded):
        return None, "padded_tokens_not_a_bucket_size"
    if rows is not None:
        why = lead_reason(lens, rows[0], rows[1], plan.qlen)
        if why:
            return None, why
    tail = None if computed is None else prefill_bounds(lens, computed)
    for name, veto in _VETOES.items():
        why = veto(lens, bucket, tail)
        if why:
            return None, f"{name}:{why}"
    return bucket, ""


def pad_cu(query_lens, reqs, device):
    cu = np.zeros(reqs + 1, dtype=np.int32)
    lens = np.asarray(query_lens, dtype=np.int32)[:reqs]
    cu[1:len(lens) + 1] = np.cumsum(lens)
    cu[len(lens) + 1:] = cu[len(lens)]
    return torch.from_numpy(cu).to(device)


def reference_chunk_meta(cu, rows, chunk=FLA_CHUNK, pad_seq=-1):
    lens = np.diff(np.asarray(cu, dtype=np.int64))
    counts = -(-lens // chunk)
    out = np.full((rows, 2), 0, dtype=np.int32)
    out[:, 0] = pad_seq
    at = 0
    for seq, count in enumerate(counts):
        for local in range(int(count)):
            out[at] = (seq, local)
            at += 1
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return out, offsets


def reference_conv_meta(cu, programs, block_m=CONV_BLOCK_M):
    lens = np.diff(np.asarray(cu, dtype=np.int64))
    nums = -(-lens // block_m)
    seq = np.full(programs, PAD_SLOT_ID, dtype=np.int32)
    local = np.full(programs, PAD_SLOT_ID, dtype=np.int32)
    at = 0
    for index, count in enumerate(nums):
        for step in range(int(count)):
            seq[at] = index
            local[at] = step
            at += 1
    return seq, local


@triton.jit
def _slot_meta_kernel(offsets_ptr, out_a, out_b, num_seqs, num_slots, pad_a, pad_b, stride, b_shift, SEARCH: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slot = pid * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(offsets_ptr + num_seqs)
    live = slot < total
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.zeros([BLOCK], dtype=tl.int32) + (num_seqs - 1)
    for _ in range(SEARCH):
        mid = (lo + hi + 1) // 2
        value = tl.load(offsets_ptr + mid)
        reach = value <= slot
        lo = tl.where(reach, mid, lo)
        hi = tl.where(reach, hi, mid - 1)
    first = tl.load(offsets_ptr + lo)
    inside = slot < num_slots
    tl.store(out_a + slot * stride, tl.where(live, lo, pad_a), mask=inside)
    tl.store(out_b + slot * stride + b_shift, tl.where(live, slot - first, pad_b), mask=inside)


def search_steps(num_seqs):
    return max(1, int(math.ceil(math.log2(max(num_seqs, 2)))) + 1)


def unit_offsets(cu, unit):
    lens = cu[1:] - cu[:-1]
    counts = (lens + (unit - 1)) // unit
    zero = torch.zeros(1, dtype=counts.dtype, device=cu.device)
    return torch.cat((zero, torch.cumsum(counts, 0))).to(torch.int32)


def chunk_meta(cu, rows, pad_seq, out=None, offsets_out=None):
    seqs = cu.shape[0] - 1
    offsets = unit_offsets(cu, FLA_CHUNK)
    if offsets_out is not None:
        offsets_out.copy_(offsets)
    if out is None:
        out = torch.empty((rows, 2), dtype=torch.int32, device=cu.device)
    _slot_meta_kernel[(triton.cdiv(rows, PROBE_BLOCK),)](
        offsets, out, out, seqs, rows, pad_seq, 0, 2, 1, SEARCH=search_steps(seqs), BLOCK=PROBE_BLOCK)
    return out, offsets


def conv_meta(cu, programs, out_seq=None, out_local=None):
    seqs = cu.shape[0] - 1
    offsets = unit_offsets(cu, CONV_BLOCK_M)
    if out_seq is None:
        out_seq = torch.empty(programs, dtype=torch.int32, device=cu.device)
    if out_local is None:
        out_local = torch.empty(programs, dtype=torch.int32, device=cu.device)
    _slot_meta_kernel[(triton.cdiv(programs, PROBE_BLOCK),)](
        offsets, out_seq, out_local, seqs, programs, PAD_SLOT_ID, PAD_SLOT_ID, 1, 0, SEARCH=search_steps(seqs),
        BLOCK=PROBE_BLOCK)
    return out_seq, out_local


def split_counts(query_start_loc, num_reqs, threshold):
    reqs = query_start_loc.shape[0] - 1
    lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32)
    live = torch.arange(reqs, device=query_start_loc.device) < num_reqs
    same = ((lens == lens[:1]) & live).to(torch.int32)
    lead = torch.cumprod(same, 0) * (lens[:1] <= threshold).to(torch.int32)
    dec_reqs = lead.sum().to(torch.int32)
    dec_tokens = (lens * lead).sum().to(torch.int32)
    total = query_start_loc.gather(0, num_reqs.reshape(1).long()).reshape(()).to(torch.int32)
    count = num_reqs.reshape(()).to(torch.int32)
    return torch.stack((dec_reqs, dec_tokens, count - dec_reqs, total - dec_tokens, total)).to(torch.int32)


def prefill_cu(query_start_loc, dec_reqs, seqs):
    last = query_start_loc.shape[0] - 1
    index = (dec_reqs.reshape(1).long() + torch.arange(seqs + 1, device=query_start_loc.device)).clamp(max=last)
    taken = query_start_loc.gather(0, index)
    return taken - taken[:1]


def probe(query_lens, bucket):
    if _STATE["checked"] >= CHECKS or _STATE["aside"]:
        return
    if not torch.cuda.is_available():
        _STATE["aside"] = "no cuda device"
        return
    device = torch.device("cuda")
    lens = [int(x) for x in query_lens]
    plan = _STATE["plan"]
    decodes = plan.split(lens)
    prefill = lens[decodes:]
    cu = pad_cu(prefill, bucket.seqs + 1, device)
    rows = bucket.chunk_rows()
    pad_seq = bucket.seqs
    got, got_offsets = chunk_meta(cu, rows, pad_seq)
    want, want_offsets = reference_chunk_meta(cu.cpu().numpy(), rows, pad_seq=pad_seq)
    programs = bucket.conv_programs()
    got_seq, got_local = conv_meta(cu, programs)
    want_seq, want_local = reference_conv_meta(cu.cpu().numpy(), programs)
    qsl = pad_cu(lens, bucket.reqs, device)
    split = split_counts(qsl, torch.tensor(len(lens), dtype=torch.int32, device=device), plan.qlen)
    total = sum(lens)
    want_split = [decodes, sum(lens[:decodes]), len(lens) - decodes, total - sum(lens[:decodes]), total]
    equal = (np.array_equal(got.cpu().numpy(), want) and np.array_equal(got_offsets.cpu().numpy(), want_offsets)
             and np.array_equal(got_seq.cpu().numpy(), want_seq) and np.array_equal(got_local.cpu().numpy(), want_local)
             and split.cpu().tolist() == want_split)
    if not equal:
        _STATE["aside"] = "device-built metadata differs from the host reference"
        selfcheck_fail(MIXED_FULL_ENV, _STATE["aside"])
        print("MIXED_FULL stands aside: the metadata built on the device differs from the host reference; no "
              "later step is probed", flush=True)
        return
    _STATE["checked"] += 1
    print(f"MIXED_FULL check {_STATE['checked']} of {CHECKS}: {len(prefill)} prefill sequences, {total - want_split[1]} "
          f"prefill tokens, chunk rows {rows}, conv programs {programs}, split {want_split}; device-built metadata "
          "equals the host reference", flush=True)


SPLIT_WIDTH = 5
SPLIT_DEC_REQS, SPLIT_DEC_TOKENS, SPLIT_PREFILL_REQS, SPLIT_PREFILL_TOKENS, SPLIT_TOTAL = range(SPLIT_WIDTH)


@dataclasses.dataclass
class BucketView:
    split: torch.Tensor
    tokens: int
    seqs: int
    decodes: int
    rows: int
    qlen: int


def bucket_view(source, qlen):
    held = getattr(source, "bucket", None)
    split = getattr(source, "device_split", None)
    tokens = getattr(source, "bucket_tokens", None)
    seqs = getattr(source, "bucket_seqs", None)
    if split is None:
        split = getattr(held, "split", None)
    if tokens is None:
        tokens = getattr(held, "tokens", None)
    if split is None or not tokens or int(qlen) < 1:
        return None
    if seqs is None:
        seqs = int(os.environ.get(SEQ_CAP_ENV) or SEQ_CAP)
    max_reqs = int(os.environ.get(MAX_SEQS_ENV) or 128)
    decodes = max(1, min(max_reqs, int(tokens) // int(qlen)))
    return BucketView(split=split, tokens=int(tokens), seqs=int(seqs), decodes=decodes, rows=decodes * int(qlen),
                      qlen=int(qlen))


def stage_fits(prefill_bounds, block_size, capacity):
    return sum(-(-int(bound) // int(block_size)) for bound in prefill_bounds) <= int(capacity)
