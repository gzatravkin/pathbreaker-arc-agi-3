import json
import os
import sys
import threading
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

ENV = "TAAF_VLLM_STEP_TIMERS"
PATH_ENV = "TAAF_VLLM_STEP_TIMERS_PATH"
CLOCK_ENV = "TAAF_VLLM_LINE_CLOCK_S"
CLOCK_DEFAULT_S = 900
CLOCK_TICK_S = 1.0
EVERY_ENV = "TAAF_VLLM_STEP_TIMERS_S"
DEFAULT_PATH = os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "step-timers.jsonl")
GROUPS = ("other", "moe_experts", "moe_route", "bf16c", "dense", "qsa_attn", "indexer", "hkv_plan", "hkv_gather",
          "gdn", "lm_head", "sampling", "hkv_meta", "hkv_write", "hkv_guard", "hkv_swap", "hkv_wait",
          "samp_core", "samp_post", "step_tail", "hkv_resolve", "hkv_gplan", "hkv_fused", "hkv_pwait", "dense_hole",
          "spec_index", "spec_tally", "ple", "hkv_shadow")
SHORT = ("other", "moe_x", "route", "bf16c", "dense", "qsa", "idx", "plan", "gather", "gdn", "lm", "samp", "meta",
         "write", "guard", "swap", "wait", "sampler", "post", "stail", "resolve", "gplan", "fused", "pwait", "hole",
         "spec", "stally", "ple", "shadow")
G = len(GROUPS)
(OTHER, MOE_EXPERTS, MOE_ROUTE, BF16C, DENSE, QSA_ATTN, INDEXER, HKV_PLAN, HKV_GATHER, GDN, LM_HEAD,
 SAMPLING, HKV_META, HKV_WRITE, HKV_GUARD, HKV_SWAP, HKV_WAIT, SAMP_CORE, SAMP_POST, STEP_TAIL, HKV_RESOLVE, HKV_GPLAN,
 HKV_FUSED, HKV_PWAIT, DENSE_HOLE, SPEC_INDEX, SPEC_TALLY, PLE, HKV_SHADOW) = range(G)
SIDE_SPLIT = (("queue", HKV_WAIT), ("hkv_resolve", HKV_RESOLVE), ("hkv_gplan", HKV_GPLAN), ("other", OTHER),
              ("hkv_gather", HKV_GATHER), ("hkv_fused", HKV_FUSED))
SIDE_SHORT = (("queue", "queue"), ("hkv_resolve", "resolve"), ("hkv_gplan", "gplan"), ("other", "other"),
              ("hkv_gather", "gather"), ("hkv_fused", "fused"), ("sum", "sum"))
HKV_FAMILY = (HKV_PLAN, HKV_META, HKV_WRITE, HKV_GUARD, HKV_SWAP, HKV_RESOLVE, HKV_GPLAN, HKV_FUSED)
DRAFT = G
SIDE = 2 * G
IDLE = 3 * G
COUNT = IDLE + 1
DISCARD = IDLE + 2
PREP_SEGS = ("event_wait", "host_blocks", "host_step", "connector", "gdn_prep", "inputs", "attn_meta", "states")
PREP_SEG0 = IDLE + 4
WIDTH = PREP_SEG0 + len(PREP_SEGS)
BLOCK = 128
RING = 256
READ_EVERY = 32
LANES = 8
CPU = ("sched", "gap", "prep", "launch", "post", "sample", "draft", "tail", "wall", "busy", "busy_proc", "lead", "margin", "delta")
LANE_EDGES = (1, 4, 8, 16, 32, 48, 64)
CTX_EDGES = (8192, 32768, 65536, 98304, 131072)
PREFILL_EDGES = (0, 128, 512, 2048)
PAUSE_NS = 1_000_000_000
QSA_EXACT_ENV = "TAAF_VLLM_STEP_TIMERS_QSA_EXACT"
PREP_ENV = "TAAF_VLLM_STEP_TIMERS_PREP"
PREP_EXTRA_ENV = "TAAF_VLLM_STEP_TIMERS_PREP_EXTRA"
PREP_SYNC_ENV = "TAAF_VLLM_STEP_TIMERS_PREP_SYNC_EVERY"
HC_ENV = "TAAF_VLLM_STEP_TIMERS_HC"
PREP_WAIT = IDLE + 3
BLOCK_NS = 30_000
PREP_TOP = 24
ANCHOR_MS = 0.2
LAUNCH_LATENCY_MS = 0.03
_STATE = {"timers": None, "capture": None, "experts": None}
EXPERT_ENV = "TAAF_VLLM_EXPERT_TALLY"
VISION_ENV = "TAAF_VLLM_VISION_TALLY"
VISION_MODULE = "vllm.taaf_vision_tally"
SPEC_ENV = "TAAF_VLLM_SPEC_INDEX"
SPEC_MODULE = "vllm.taaf_spec_index"
SPEC_PREFETCH_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC"
EXPERT_EVERY_ENV = "TAAF_VLLM_EXPERT_TALLY_EVERY"
EXPERT_EVERY = 500
EXPERT_KINDS = ("decode", "mixed")
EXPERT_FIELDS = ("calls", "distinct", "rows", "peak")
EXPERT_HIST_EDGES = (1, 2, 4, 8, 16, 32, 64)
EXPERT_FLOOR = 1 << 40


@triton.jit
def _now():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit(do_not_specialize=["col", "lane"])
def _stamp_kernel(acc, last, ring, col, lane, WIDTH: tl.constexpr, COUNT: tl.constexpr):
    now = _now()
    row = acc + tl.load(ring).to(tl.int64) * WIDTH
    tl.atomic_add(row + col, now - tl.load(last + lane))
    tl.atomic_add(row + COUNT, 1)
    tl.store(last + lane, now)


@triton.jit
def _begin_kernel(acc, last, ring, WIDTH: tl.constexpr, IDLE: tl.constexpr, COUNT: tl.constexpr,
                  BLOCK: tl.constexpr):
    now = _now()
    row = acc + tl.load(ring).to(tl.int64) * WIDTH
    before = tl.load(last)
    idle = tl.where(before > 0, now - before, 0)
    offs = tl.arange(0, BLOCK)
    values = tl.where(offs == IDLE, idle, tl.where(offs == COUNT, 1, 0)).to(tl.int64)
    tl.store(row + offs, values, mask=offs < WIDTH)
    tl.store(last, now)


@triton.jit(do_not_specialize=["col"])
def _end_kernel(acc, last, ring, col, WIDTH: tl.constexpr, COUNT: tl.constexpr, RING: tl.constexpr):
    now = _now()
    slot = tl.load(ring)
    row = acc + slot.to(tl.int64) * WIDTH
    tl.atomic_add(row + col, now - tl.load(last))
    tl.atomic_add(row + COUNT, 1)
    tl.store(last, now)
    tl.store(ring, (slot + 1) % RING)


def enabled():
    return str(os.environ.get(ENV) or "").strip() == "1"


def say(line):
    print(f"STEP_TIMERS {line}", flush=True)


COUNTERS = {}


def counter(name, fields, tensor):
    COUNTERS[name] = (tuple(fields), tensor)


class Counted:
    def __init__(self):
        self.host = {}
        self.seen = None
        self.base = None

    def copy(self):
        for name, (_, held) in list(COUNTERS.items()):
            host = self.host.get(name)
            if host is None or host.shape != held.shape:
                host = torch.zeros(held.shape, dtype=held.dtype).pin_memory()
                self.host[name] = host
            host.copy_(held, non_blocking=True)

    def read(self, upto):
        self.seen = (upto, {name: host.numpy().copy() for name, host in self.host.items()})
        if self.base is None:
            self.base = self.seen

    def lines(self):
        if self.seen is None or self.base is None:
            return []
        (upto, values), (start, before) = self.seen, self.base
        steps = upto - start
        self.base = self.seen
        if steps <= 0:
            return []
        out = {}
        for name, held in values.items():
            fields = COUNTERS.get(name, ((),))[0]
            delta = held - before.get(name, held * 0)
            row = {"steps": steps}
            for index, field in enumerate(fields):
                row[field] = int(delta[index])
                row[f"{field}_per_step"] = round(float(delta[index]) / steps, 2)
            out[name] = row
        if not out:
            return []
        say("counters over " + "; ".join(
            f"{name} {row['steps']} steps: " + " ".join(f"{field} {row[field + '_per_step']}/step"
                                                    for field in COUNTERS.get(name, ((),))[0])
            for name, row in out.items()))
        return [json.dumps({"t": round(time.time(), 1), "counters": out}, separators=(",", ":"))]


def _raw_stream(index):
    return torch._C._cuda_getCurrentRawStream(index)


class _Kernel:
    def __init__(self, jit, consts, tensors):
        self.jit = jit
        self.consts = consts
        self.tensors = tensors
        self.compiled = None
        self.fast = False

    def jit_launch(self, tensors, dynamic):
        return self.jit[(1,)](*tensors, *dynamic, **self.consts)

    def warm(self, dynamic):
        self.compiled = self.jit_launch(self.tensors, dynamic)

    def fast_launch(self, stream, pointers, dynamic):
        compiled = self.compiled
        compiled.run(1, 1, 1, stream, compiled.function, compiled.packed_metadata, None, None, None,
                     *pointers, *dynamic, *self.consts.values())

    def __call__(self, stream, pointers, dynamic, tensors=None):
        if self.fast:
            self.fast_launch(stream, pointers, dynamic)
        else:
            self.jit_launch(self.tensors if tensors is None else tensors, dynamic)


class Timers:
    def __init__(self, device):
        self.device = torch.device(device)
        self.index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.acc = torch.zeros(RING, WIDTH, dtype=torch.int64, device=self.device)
        self.last = torch.zeros(LANES, dtype=torch.int64, device=self.device)
        self.ring = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.host = torch.zeros(RING, WIDTH, dtype=torch.int64).pin_memory()
        self.pointers = (self.acc.data_ptr(), self.last.data_ptr(), self.ring.data_ptr())
        tensors = (self.acc, self.last, self.ring)
        self.stamp_kernel = _Kernel(_stamp_kernel, {"WIDTH": WIDTH, "COUNT": COUNT}, tensors)
        self.begin_kernel = _Kernel(_begin_kernel, {"WIDTH": WIDTH, "IDLE": IDLE, "COUNT": COUNT, "BLOCK": BLOCK},
                                    tensors)
        self.end_kernel = _Kernel(_end_kernel, {"WIDTH": WIDTH, "COUNT": COUNT, "RING": RING}, tensors)
        self.current = OTHER
        self.phase = 0
        self.timed = None
        self.lanes = {}
        self.steps = 0
        self.read_upto = 0
        self.pending = None
        self.meta = [None] * RING
        self.open = None
        self.window = []
        self.window_start = time.monotonic()
        self.every = float(os.environ.get(EVERY_ENV) or 20.0)
        self.sched_ns = 0
        self.probe = None
        self.prep_stamp = False
        self.segments = False
        self.thread = None
        self.last_end_ns = None
        self.last_busy = None
        self.chain_state = None
        self.dropped = 0
        self.told = False
        self.hooks = {}
        self.forks = {}
        self.calibration = {}
        self.counted = Counted()
        self.path = self._path()

    @staticmethod
    def _path():
        said = str(os.environ.get(PATH_ENV) or "").strip()
        if said:
            return Path(said)
        default = Path(DEFAULT_PATH)
        return default if default.parent.is_dir() else Path.cwd() / default.name

    def warm(self):
        self.stamp_kernel.warm((DISCARD, 0))
        self.begin_kernel.warm(())
        self.end_kernel.warm((DISCARD,))
        torch.cuda.synchronize(self.device)
        self.ring.zero_()
        self.acc.zero_()
        self.last.zero_()
        scratch_acc = torch.zeros(RING, WIDTH, dtype=torch.int64, device=self.device)
        scratch_last = torch.zeros(LANES, dtype=torch.int64, device=self.device)
        scratch_ring = torch.zeros(1, dtype=torch.int32, device=self.device)
        pointers = (scratch_acc.data_ptr(), scratch_last.data_ptr(), scratch_ring.data_ptr())
        stream = _raw_stream(self.index)
        fast = False
        reason = ""
        try:
            self.begin_kernel.fast_launch(stream, pointers, ())
            self.stamp_kernel.fast_launch(stream, pointers, (OTHER + 3, 0))
            self.end_kernel.fast_launch(stream, pointers, (OTHER + 5,))
            torch.cuda.synchronize(self.device)
            row = scratch_acc[0].cpu()
            fast = (int(scratch_ring.item()) == 1 and int(row[COUNT]) == 3 and int(row[OTHER + 3]) >= 0
                    and int(row[OTHER + 5]) >= 0 and int(scratch_last[0]) > 0)
            reason = "" if fast else f"scratch row {row.tolist()[:8]} ring {int(scratch_ring.item())}"
        except Exception as failure:
            reason = f"{type(failure).__name__}: {failure}"
        for kernel in (self.stamp_kernel, self.begin_kernel, self.end_kernel):
            kernel.fast = fast
        self.calibration["fast_launch"] = fast
        if not fast:
            say(f"the raw Triton launcher refused the stamp ({reason}); stamps go through the JIT launcher")
        self.calibration.update(self._measure(pointers, (scratch_acc, scratch_last, scratch_ring)))

    def _measure(self, pointers, tensors):
        out = {}
        n = 400
        stream = _raw_stream(self.index)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        for _ in range(n):
            self.stamp_kernel(stream, pointers, (DISCARD, 1), tensors)
        out["cpu_us_per_stamp"] = (time.perf_counter_ns() - started) / n / 1e3
        torch.cuda.synchronize(self.device)
        try:
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                graph.capture_begin(capture_error_mode="thread_local")
                inner = _raw_stream(self.index)
                for _ in range(256):
                    self.stamp_kernel(inner, pointers, (DISCARD, 1), tensors)
                graph.capture_end()
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph.replay()
            torch.cuda.synchronize(self.device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            out["gpu_us_per_stamp_in_graph"] = begin.elapsed_time(end) * 1e3 / 20 / 256
            del graph
        except Exception as failure:
            out["gpu_us_per_stamp_in_graph"] = None
            out["graph_failure"] = f"{type(failure).__name__}: {failure}"
        torch.cuda.synchronize(self.device)
        return out

    def stamp(self, stream, col, lane):
        self.stamp_kernel(stream, self.pointers, (col, lane))

    def switch(self, col):
        if col == self.current:
            return
        stream = _raw_stream(self.index)
        if stream != self.timed:
            return
        self.stamp_kernel(stream, self.pointers, (self.current, 0))
        self.current = col

    def lane(self, stream):
        found = self.lanes.get(stream)
        if found is None:
            found = min(len(self.lanes) + 1, LANES - 1)
            self.lanes[stream] = found
        return found

    def begin(self, scheduler_output):
        now = time.perf_counter_ns()
        stream = _raw_stream(self.index)
        self.timed = stream
        self.thread = threading.get_ident()
        self.current = PREP_WAIT if self.segments else OTHER
        self.phase = 0
        self.begin_kernel(stream, self.pointers, ())
        gap = now - self.last_end_ns if self.last_end_ns is not None else None
        self.open = {"t0": now, "shape": step_shape(scheduler_output), "mode": None, "launch0": None,
                     "launch1": None, "exec_end": None, "sample0": None, "sample1": None, "draft0": None,
                     "draft1": None, "sched": self.sched_ns, "gap": gap, "prep_on": self.prep_stamp}
        self.sched_ns = 0
        if self.probe is not None:
            self.probe.begin(self.open)

    def mark(self, key):
        held = self.open
        if held is not None and held.get(key) is None:
            held[key] = time.perf_counter_ns()

    def set_mode(self, mode):
        held = self.open
        if held is not None and held["mode"] is None and self.phase == 0:
            held["mode"] = mode
            held["launch0"] = time.perf_counter_ns()
            if self.prep_stamp:
                stream = _raw_stream(self.index)
                if stream == self.timed:
                    open_col = self.current if self.segments and self.current >= IDLE else PREP_WAIT
                    self.stamp_kernel(stream, self.pointers, (open_col, 0))
            if self.current >= IDLE:
                self.current = OTHER

    def seg_enter(self, col):
        held = self.open
        if not self.segments or held is None or held["launch0"] is not None or self.timed is None:
            return None
        if threading.get_ident() != self.thread or _raw_stream(self.index) != self.timed:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        before = self.current
        self.switch(col)
        return held, before

    def seg_exit(self, token):
        if token is None:
            return
        held, before = token
        if self.open is held and held["launch0"] is None:
            self.switch(before)

    def end(self):
        held = self.open
        stream = _raw_stream(self.index)
        if held is None or stream != self.timed:
            return
        self.end_kernel(stream, self.pointers, (self.current,))
        self.current = OTHER
        self.phase = 0
        now = time.perf_counter_ns()
        held["t1"] = now
        held["wall"] = now - self.last_end_ns if self.last_end_ns is not None else None
        busy = (time.thread_time_ns(), time.process_time_ns())
        held["busy"] = (busy[0] - self.last_busy[0], busy[1] - self.last_busy[1]) if self.last_busy else None
        self.last_busy = busy
        if self.probe is not None:
            self.probe.end(held)
        self.meta[self.steps % RING] = held
        self.steps += 1
        self.open = None
        self.last_end_ns = now
        self.poll()

    def poll(self):
        if self.pending is not None and self.pending[0].query():
            self.consume(self.pending[1])
            self.pending = None
        if self.pending is None and self.steps - self.read_upto >= READ_EVERY:
            self.host.copy_(self.acc, non_blocking=True)
            self.counted.copy()
            event = torch.cuda.Event()
            event.record()
            self.pending = (event, self.steps)
        if time.monotonic() - self.window_start >= self.every:
            self.flush()

    def consume(self, upto):
        start = max(self.read_upto, upto - RING + 1)
        self.dropped += start - self.read_upto
        rows = self.host.numpy()
        for step in range(start, upto):
            held = self.meta[step % RING]
            if held is None:
                continue
            made = record(rows[step % RING].copy(), held)
            if held.get("prep_on"):
                self.chain(made, held)
            self.window.append(made)
        self.read_upto = upto
        self.counted.read(upto)
        if not self.told and self.window:
            self.told = True
            first = self.window[-1]
            idle = "-" if first["idle"] is None else f"{first['idle']:.2f}"
            say(f"first readback: {upto} steps timed, the last a {first['kind']}/{first['mode']} step of "
                f"{first['gpu'].sum():.2f} ms on the GPU over {first['count']} stamps, idle before it {idle} ms")

    def chain(self, made, held):
        idle = made["idle"]
        total = float(made["gpu"].sum()) + prep_total(made)
        state = self.chain_state
        if state is None or idle is None or idle > ANCHOR_MS:
            lead = LAUNCH_LATENCY_MS
        else:
            lead = max(0.0, state[0] + state[1] + idle - (held["t0"] - state[2]) / 1e6)
        self.chain_state = (lead, total, held["t0"])
        prep = made["cpu"]["prep"]
        made["cpu"]["lead"] = lead
        made["cpu"]["margin"] = None if prep is None else lead - prep
        made["cpu"]["delta"] = None if state is None or lead <= LAUNCH_LATENCY_MS else state[1] - lead

    def flush(self):
        elapsed = time.monotonic() - self.window_start
        self.window_start = time.monotonic()
        steps, self.window = self.window, []
        if not steps:
            return
        say(split_line(steps, elapsed))
        by_kind = {}
        for one in steps:
            by_kind.setdefault((one["kind"], one["mode"]), []).append(one)
        lines = []
        for (kind, mode), group in sorted(by_kind.items()):
            summary = summarize(group, kind, mode, elapsed)
            if self.probe is not None:
                summary.update(summarize_prep(group))
            lines.append(json.dumps(summary, separators=(",", ":")))
            say(log_line(summary))
            if self.probe is not None:
                say(log_prep(summary))
        if self.probe is not None:
            window = self.probe.window()
            lines.append(json.dumps({"t": round(time.time(), 1), "prep_window": window}, separators=(",", ":")))
            say(f"PREP syncs: {window['sync_sampled_steps']} sampled steps; explicit blocking "
                f"{window['blocking_calls'][:4]}; implicit {window['implicit_syncs'][:4]}")
        if self.dropped:
            say(f"{self.dropped} steps dropped before their readback")
            self.dropped = 0
        lines.extend(self.counted.lines())
        self.write(lines)

    def write(self, lines):
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError as failure:
            say(f"could not append to {self.path}: {type(failure).__name__}: {failure}")


INDEXER_RATIO = 4
SCALE_NAMES = ("idx_decode_keys", "idx_prefill_pairs")


def step_shape(scheduler_output):
    scheduled = scheduler_output.num_scheduled_tokens
    drafts_by = scheduler_output.scheduled_spec_decode_tokens or {}
    computed = {}
    fresh = set()
    for request in scheduler_output.scheduled_new_reqs:
        computed[request.req_id] = request.num_computed_tokens
        fresh.add(request.req_id)
    cached = scheduler_output.scheduled_cached_reqs
    for request_id, done in zip(cached.req_ids, cached.num_computed_tokens):
        computed[request_id] = done
    decodes = drafts = prefill = 0
    contexts = []
    decode_keys = prefill_pairs = 0
    for request_id, count in scheduled.items():
        drafted = len(drafts_by.get(request_id, ()))
        context = computed.get(request_id, 0) + count
        contexts.append(context)
        if request_id not in fresh and count <= drafted + 1:
            decodes += 1
            drafts += drafted
            decode_keys += context // INDEXER_RATIO
        else:
            prefill += count
            prefill_pairs += count * (context // INDEXER_RATIO)
    lanes = len(scheduled)
    return {"lanes": lanes, "decodes": decodes, "drafts": drafts, "prefill": prefill,
            "tokens": int(scheduler_output.total_num_scheduled_tokens),
            "ctx_mean": float(sum(contexts) / lanes) if lanes else 0.0, "ctx_max": max(contexts) if contexts else 0,
            "idx_decode_keys": decode_keys, "idx_prefill_pairs": prefill_pairs}


def _span(held, first, second):
    a, b = held.get(first), held.get(second)
    return (b - a) / 1e6 if a is not None and b is not None and b >= a else None


def record(row, held):
    shape = held["shape"]
    gap = held["gap"]
    idle = row[IDLE] / 1e6 if gap is None or gap < PAUSE_NS else None
    launch0 = held["launch0"]
    cpu = {
        "sched": held["sched"] / 1e6,
        "gap": gap / 1e6 if gap is not None and gap < PAUSE_NS else None,
        "prep": _span(held, "t0", "launch0") if launch0 is not None else None,
        "launch": _span(held, "launch0", "launch1"),
        "post": _span(held, "launch1", "exec_end"),
        "sample": _span(held, "sample0", "sample1"),
        "draft": _span(held, "draft0", "draft1"),
        "tail": _span(held, "draft1" if held.get("draft1") is not None else "sample1", "t1"),
        "wall": held["wall"] / 1e6 if held.get("wall") is not None and held["wall"] < PAUSE_NS else None,
        "busy": held["busy"][0] / 1e6 if held.get("busy") and (held.get("wall") or PAUSE_NS) < PAUSE_NS else None,
        "busy_proc": held["busy"][1] / 1e6 if held.get("busy") and (held.get("wall") or PAUSE_NS) < PAUSE_NS else None,
    }
    out = {"kind": "mixed" if shape["prefill"] else "decode", "mode": held["mode"] or "eager", "shape": shape,
           "gpu": row[:SIDE] / 1e6, "side": row[SIDE:IDLE] / 1e6, "idle": idle, "count": int(row[COUNT]),
           "cpu": cpu}
    if held.get("prep_on"):
        out["prep_wait"] = row[PREP_WAIT] / 1e6 if launch0 is not None else None
        segs = row[PREP_SEG0:PREP_SEG0 + len(PREP_SEGS)]
        out["prep_segs"] = segs / 1e6 if launch0 is not None and len(segs) == len(PREP_SEGS) else None
        out["spans"] = held.get("spans")
        out["blocked"] = held.get("blocked")
    return out


def prep_total(made):
    segs = made.get("prep_segs")
    return (made.get("prep_wait") or 0.0) + (float(segs.sum()) if segs is not None else 0.0)


SPLIT_KINDS = ("decode", "mixed", "prefill")


def split_line(steps, elapsed):
    held = {kind: [0, 0.0] for kind in SPLIT_KINDS}
    for one in steps:
        shape = one["shape"]
        kind = "decode" if not shape["prefill"] else "mixed" if shape["decodes"] else "prefill"
        held[kind][0] += 1
        held[kind][1] += float(one["gpu"].sum()) + prep_total(one)
    return f"{elapsed:.0f}s split: " + ", ".join(f"{kind} {count} steps {ms:.1f} ms" for kind, (count, ms) in held.items())


def _seg_means(segs):
    held = [one for one in segs if one is not None]
    if not held:
        return None
    means = np.stack(held).mean(axis=0)
    return {name: round(float(value), 3) for name, value in zip(PREP_SEGS, means)}


def _stats(values):
    held = [v for v in values if v is not None]
    if not held:
        return None
    array = np.asarray(held, dtype=np.float64)
    return [round(float(array.mean()), 3), round(float(np.percentile(array, 50)), 3),
            round(float(np.percentile(array, 90)), 3)]


def _bucket(value, edges, unit=1, suffix=""):
    low = 0
    for edge in edges:
        if value <= edge:
            return f"{low // unit}-{edge // unit}{suffix}" if edge else "0"
        low = edge + 1
    return f">{edges[-1] // unit}{suffix}"


def summarize(steps, kind, mode, elapsed):
    gpu = np.stack([one["gpu"] for one in steps])
    side = np.stack([one["side"] for one in steps])
    target = gpu[:, :G]
    drafter = gpu[:, DRAFT:DRAFT + G]
    out = {"t": round(time.time(), 1), "window_s": round(elapsed, 1), "kind": kind, "mode": mode, "n": len(steps),
           "stamps_per_step": round(float(np.mean([one["count"] for one in steps])), 1)}
    waits = [one.get("prep_wait") for one in steps]
    segs = [one.get("prep_segs") for one in steps]
    totals = gpu.sum(axis=1) + np.asarray([prep_total(one) for one in steps], dtype=np.float64)
    gpu_ms = {"step": _stats(totals), "idle": _stats([one["idle"] for one in steps])}
    if any(wait is not None for wait in waits):
        gpu_ms["prep_wait"] = _stats(waits)
    if any(seg is not None for seg in segs):
        for index, name in enumerate(PREP_SEGS):
            gpu_ms[f"prep_{name}"] = _stats([seg[index] for seg in segs if seg is not None])
    for index, name in enumerate(GROUPS):
        gpu_ms[name] = _stats(target[:, index])
    gpu_ms["drafter"] = _stats(drafter.sum(axis=1))
    for name, index in SIDE_SPLIT:
        gpu_ms[f"side_{name}"] = _stats(side[:, index])
    gpu_ms["side_sum"] = _stats(side.sum(axis=1))
    out["gpu_ms"] = gpu_ms
    out["drafter_ms"] = {name: round(float(drafter[:, index].mean()), 4) for index, name in enumerate(GROUPS)
                         if drafter[:, index].any()}
    out["side_ms"] = {name: round(float(side[:, index].mean()), 4) for index, name in enumerate(GROUPS)
                      if side[:, index].any()}
    out["cpu_ms"] = {name: _stats([one["cpu"].get(name) for one in steps]) for name in CPU}
    margins = [one["cpu"].get("margin") for one in steps if one["cpu"].get("margin") is not None]
    if margins:
        out["starved"] = round(float(np.mean([margin < 0 for margin in margins])), 3)
    shapes = [one["shape"] for one in steps]
    for name in ("lanes", "decodes", "drafts"):
        values = [shape[name] for shape in shapes]
        out[name] = [round(float(np.mean(values)), 2), int(min(values)), int(max(values))]
    prefill = [shape["prefill"] for shape in shapes]
    out["prefill_tokens"] = [round(float(np.mean(prefill)), 1), int(np.percentile(prefill, 50)),
                             int(np.percentile(prefill, 90)), int(max(prefill))]
    for name in ("ctx_mean", "ctx_max"):
        values = [shape[name] for shape in shapes]
        out[name] = [round(float(np.mean(values))), int(min(values)), int(max(values))]
    for name in SCALE_NAMES:
        out[name] = [round(float(np.mean([shape.get(name, 0) for shape in shapes])))]
    cells = {}
    cell_shapes = {}
    for one, total, row, draft in zip(steps, totals, target, drafter.sum(axis=1)):
        shape = one["shape"]
        key = (_bucket(shape["lanes"], LANE_EDGES), _bucket(int(shape["ctx_mean"]), CTX_EDGES, 1024, "k"),
               _bucket(shape["prefill"], PREFILL_EDGES))
        cell_shapes.setdefault(key, []).append(shape)
        cells.setdefault(key, []).append((total, row, draft, one.get("prep_wait"), one["cpu"].get("margin"),
                                          one.get("prep_segs")))
    out["cells"] = []
    for (lanes, context, prefill_bucket), held in sorted(cells.items()):
        rows = np.stack([row for _, row, _, _, _, _ in held])
        waits = [wait for _, _, _, wait, _, _ in held if wait is not None]
        margins = [margin for _, _, _, _, margin, _ in held if margin is not None]
        cell = {"lanes": lanes, "ctx": context, "prefill": prefill_bucket, "n": len(held),
                "step_ms": round(float(np.mean([total for total, _, _, _, _, _ in held])), 3),
                "drafter_ms": round(float(np.mean([draft for _, _, draft, _, _, _ in held])), 3),
                "prep_wait_ms": round(float(np.mean(waits)), 3) if waits else None,
                "starved": round(float(np.mean([margin < 0 for margin in margins])), 3) if margins else None,
                "gpu_ms": [round(float(value), 3) for value in rows.mean(axis=0)]}
        for name in SCALE_NAMES:
            cell[name] = round(float(np.mean([shape.get(name, 0) for shape in cell_shapes[(lanes, context, prefill_bucket)]])))
        seg_ms = _seg_means([seg for _, _, _, _, _, seg in held])
        if seg_ms is not None:
            cell["prep_segs_ms"] = seg_ms
        out["cells"].append(cell)
    return out


def _triple(value):
    return "-" if value is None else "/".join(f"{part:.2f}" for part in value)


def log_line(summary):
    gpu = summary["gpu_ms"]
    parts = [f"{summary['window_s']:.0f}s {summary['kind']}/{summary['mode']} n={summary['n']} "
             f"stamps={summary['stamps_per_step']:.0f} ms mean/p50/p90: step {_triple(gpu['step'])} "
             f"idle {_triple(gpu['idle'])}"
             + (f" prep_wait {_triple(gpu['prep_wait'])}" if gpu.get("prep_wait") is not None else "")
             + "".join(f" prep_{name} {_triple(gpu[f'prep_{name}'])}" for name in PREP_SEGS
                       if gpu.get(f"prep_{name}") is not None and gpu[f"prep_{name}"][0] > 0.0005) + " |"]
    for name, short in zip(GROUPS, SHORT):
        if gpu[name] is not None and gpu[name][0] > 0.0005:
            parts.append(f"{short} {_triple(gpu[name])}")
    parts.append(f"draft {_triple(gpu['drafter'])}")
    for name, short in SIDE_SHORT:
        held = gpu.get(f"side_{name}")
        if held is not None and held[0] > 0.0005:
            parts.append(f"side_{short} {_triple(held)}")
    parts.append("| cpu " + " ".join(f"{name} {_triple(summary['cpu_ms'][name])}" for name in CPU
                                    if summary["cpu_ms"][name] is not None))
    parts.append(f"| lanes {summary['lanes'][0]:.1f} ({summary['lanes'][1]}-{summary['lanes'][2]}) "
                 f"dec {summary['decodes'][0]:.1f} drafts {summary['drafts'][0]:.1f} "
                 f"ctx {summary['ctx_mean'][0] / 1024:.1f}k max {summary['ctx_max'][2] / 1024:.1f}k "
                 f"prefill {summary['prefill_tokens'][0]:.0f}/{summary['prefill_tokens'][3]}")
    return " ".join(parts)


def prep_mode():
    value = str(os.environ.get(PREP_ENV) or "").strip().lower()
    return value if value in ("1", "all") else ""


def _site(depth):
    try:
        frame = sys._getframe(depth)
    except ValueError:
        return "?"
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


def _chain(start, limit):
    try:
        frame = sys._getframe(start)
    except ValueError:
        return "?"
    parts = []
    while frame is not None and len(parts) < limit:
        name = os.path.basename(frame.f_code.co_filename)
        if name != "warnings.py":
            parts.append(f"{name}:{frame.f_lineno}:{frame.f_code.co_name}")
        frame = frame.f_back
    return " < ".join(parts)


class Probe:
    def __init__(self, timers, mode):
        self.timers = timers
        self.mode = mode
        self.thread = threading.get_ident()
        self.stack = []
        self.spans = {}
        self.blocked = {}
        self.sites = {}
        self.implicit = {}
        self.sync_open_steps = 0
        self.syncing = False
        self.every = int(os.environ.get(PREP_SYNC_ENV) or 16)
        self.wrapped = {}
        self.previous_show = None
        self.overhead_ns = None
        self.idle = {"launch0": None}
        self.between = {}
        self.runner = None
        self.armed = False
        self.allocator = None

    def held(self):
        held = self.timers.open
        if threading.get_ident() != self.thread:
            return None
        if held is None:
            return self.idle if self.mode == "all" else None
        if self.mode != "all" and held["launch0"] is not None:
            return None
        return held

    def phase(self, held):
        if held is self.idle:
            return "between"
        for key, name in (("draft1", "tail"), ("draft0", "draft"), ("sample1", "mid"), ("sample0", "sample"),
                          ("exec_end", "gap"), ("launch1", "post"), ("launch0", "launch")):
            if held.get(key) is not None:
                return name
        return "prep"

    def span(self, name, function):
        probe = self
        clock = time.perf_counter_ns

        def wrapped(*args, **kwargs):
            held = probe.held()
            if held is None:
                return function(*args, **kwargs)
            frame = [0, 0]
            stack = probe.stack
            stack.append(frame)
            started = clock()
            try:
                return function(*args, **kwargs)
            finally:
                total = clock() - started
                stack.pop()
                if stack:
                    stack[-1][0] += total
                where = probe.between if held is probe.idle else probe.spans
                record = where.get(name)
                if record is None:
                    record = where[name] = [0, 0, 0]
                record[0] += total - frame[0]
                record[1] += 1
                record[2] += frame[1]

        wrapped.__wrapped__ = function
        wrapped.__name__ = getattr(function, "__name__", name)
        wrapped._taaf_probe = True
        return wrapped

    def note(self, kind, held, elapsed, depth):
        if held is not self.idle:
            phase = self.phase(held)
            self.blocked[phase] = self.blocked.get(phase, 0) + elapsed
        if self.stack:
            self.stack[-1][1] += elapsed
        key = (kind, _site(depth))
        record = self.sites.get(key)
        if record is None:
            record = self.sites[key] = [0, 0]
        record[0] += 1
        record[1] += elapsed

    def block(self, kind, function, tensor_only):
        probe = self
        clock = time.perf_counter_ns

        def wrapped(*args, **kwargs):
            held = probe.held()
            if held is None:
                return function(*args, **kwargs)
            if tensor_only is True:
                target = args[0] if args else None
                if not (isinstance(target, torch.Tensor) and target.is_cuda):
                    return function(*args, **kwargs)
            elif tensor_only == "any":
                if not any(isinstance(item, torch.Tensor) and item.is_cuda for item in args[:2]):
                    return function(*args, **kwargs)
            elif tensor_only == "device":
                if not str(kwargs.get("device") or "").startswith("cuda"):
                    return function(*args, **kwargs)
            started = clock()
            out = function(*args, **kwargs)
            elapsed = clock() - started
            if elapsed >= BLOCK_NS:
                probe.note(kind, held, elapsed, 3)
            return out

        wrapped.__wrapped__ = function
        wrapped.__name__ = getattr(function, "__name__", kind)
        wrapped._taaf_probe = True
        return wrapped

    def wrap_attr(self, owner, name, label):
        function = owner.__dict__.get(name) if hasattr(owner, "__dict__") else None
        if function is None:
            function = getattr(owner, name, None)
        if function is None or not callable(function) or getattr(function, "_taaf_probe", False):
            return False
        try:
            setattr(owner, name, self.span(label, function))
        except (AttributeError, TypeError):
            return False
        self.wrapped[label] = self.wrapped.get(label, 0) + 1
        return True

    def install_blockers(self):
        for name in ("item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__", "__index__"):
            original = getattr(torch.Tensor, name, None)
            if original is not None and not getattr(original, "_taaf_probe", False):
                setattr(torch.Tensor, name, self.block(f"Tensor.{name}", original, True))
        for name in ("copy_", "to"):
            original = getattr(torch.Tensor, name, None)
            if original is not None and not getattr(original, "_taaf_probe", False):
                setattr(torch.Tensor, name, self.block(f"Tensor.{name}", original, "any"))
        original = getattr(torch, "tensor", None)
        if original is not None and not getattr(original, "_taaf_probe", False):
            torch.tensor = self.block("torch.tensor", original, "device")
        for owner, name, label in ((torch.cuda, "synchronize", "cuda.synchronize"),
                                   (torch.cuda.Stream, "synchronize", "Stream.synchronize"),
                                   (torch.cuda.Event, "synchronize", "Event.synchronize"),
                                   (getattr(torch, "accelerator", None), "synchronize", "accelerator.synchronize")):
            original = getattr(owner, name, None) if owner is not None else None
            if original is not None and not getattr(original, "_taaf_probe", False):
                setattr(owner, name, self.block(label, original, False))

    def install_stock(self, runner):
        for name in ("update_pp_decode_requests", "finish_requests", "add_requests", "update_requests",
                     "gather_batch_req_state", "prepare_inputs", "prepare_attn"):
            self.wrap_attr(runner, name, f"runner.{name}")
        for owner_name, names in (("block_tables", ("apply_staged_writes", "gather_block_tables",
                                                    "compute_slot_mappings", "append_block_ids")),
                                  ("req_states", ("apply_staged_writes", "add_request", "remove_request")),
                                  ("model_state", ("preprocess_state", "prepare_attn", "prepare_inputs",
                                                   "add_request", "remove_request", "apply_staged_writes")),
                                  ("sampler", ("add_request", "apply_staged_writes")),
                                  ("kv_connector", ("pre_forward",)),
                                  ("eplb", ("prepare_forward",))):
            owner = getattr(runner, owner_name, None)
            if owner is not None:
                for name in names:
                    self.wrap_attr(owner, name, f"{owner_name}.{name}")
        for module_name in ("vllm.v1.worker.gpu.model_states.mamba_hybrid",
                            "vllm.v1.worker.gpu.model_states.default"):
            module = sys.modules.get(module_name)
            if module is not None:
                self.wrap_attr(module, "build_attn_metadata", "build_attn_metadata")
        seen = set()
        for groups in getattr(runner, "attn_groups", None) or ():
            for group in groups:
                try:
                    builder = group.get_metadata_builder(0)
                except Exception:
                    continue
                if id(builder) in seen:
                    continue
                seen.add(id(builder))
                self.wrap_attr(builder, "build", f"build:{type(builder).__name__}")

    def install_helpers(self):
        for module_name, module in sorted(sys.modules.items()):
            if not module_name.startswith("vllm.taaf_") or module is None or module is sys.modules.get(__name__):
                continue
            short = module_name[len("vllm.taaf_"):]
            for name, function in list(vars(module).items()):
                if (isinstance(function, types.FunctionType) and function.__module__ == module.__name__
                        and name not in ("say", "tally") and not getattr(function, "_taaf_probe", False)):
                    self.wrap_attr(module, name, f"{short}.{name}")

    def install_extra(self, runner):
        for item in str(os.environ.get(PREP_EXTRA_ENV) or "").split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            target, _, path = item.partition(":")
            owner = runner if target == "runner" else sys.modules.get(target)
            parts = path.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
            if owner is not None:
                self.wrap_attr(owner, parts[-1], item)

    def calibrate(self):
        probe = self
        clock = time.perf_counter_ns
        base = types.SimpleNamespace(open={"launch0": None})
        saved = probe.timers
        probe.timers = base
        plain = lambda: None
        wrapped = probe.span("calibration", plain)
        started = clock()
        for _ in range(2000):
            wrapped()
        spent = clock() - started
        probe.timers = saved
        probe.spans.pop("calibration", None)
        probe.stack.clear()
        self.overhead_ns = spent / 2000

    def install(self, runner):
        self.runner = runner
        return self

    def arm(self):
        self.armed = True
        self.install_blockers()
        self.install_stock(self.runner)
        self.install_helpers()
        self.install_extra(self.runner)
        self.calibrate()
        install_segments(self.timers, self.runner)
        say(f"PREP probe armed on the first real step: {len(self.wrapped)} timed call sites "
            f"({sum(self.wrapped.values())} objects), {self.overhead_ns / 1000:.2f} us of CPU per timed call, "
            f"mode {self.mode}, implicit-sync sampling every {self.every} steps")

    def begin(self, held):
        if not self.armed:
            self.arm()
        self.stack.clear()
        self.spans = {}
        self.blocked = {}
        if self.every > 0 and self.timers.steps % self.every == 0 and torch.cuda.is_available():
            self.sync_start()

    def end(self, held):
        held["spans"] = self.spans
        held["blocked"] = self.blocked
        if self.syncing:
            self.sync_stop()

    def sync_start(self):
        try:
            torch.cuda.set_sync_debug_mode("warn")
        except Exception:
            self.every = 0
            return
        self.syncing = True
        self.previous_show = warnings.showwarning
        warnings.showwarning = self.show
        warnings.filterwarnings("always", message=".*synchronizing CUDA operation.*")

    def sync_stop(self):
        self.syncing = False
        self.sync_open_steps += 1
        warnings.showwarning = self.previous_show
        try:
            torch.cuda.set_sync_debug_mode("default")
        except Exception:
            pass

    def show(self, message, category, filename, lineno, file=None, line=None):
        if "synchronizing" not in str(message):
            return self.previous_show(message, category, filename, lineno, file, line)
        if threading.get_ident() != self.thread:
            return None
        key = _chain(1, 4)
        self.implicit[key] = self.implicit.get(key, 0) + 1

    def window(self):
        out = {"sync_sampled_steps": self.sync_open_steps}
        sites = sorted(self.sites.items(), key=lambda item: -item[1][1])[:12]
        out["blocking_calls"] = [[kind, site, count, round(spent / 1e6, 3)] for (kind, site), (count, spent) in sites]
        implicit = sorted(self.implicit.items(), key=lambda item: -item[1])[:12]
        out["implicit_syncs"] = [[site, count] for site, count in implicit]
        between = sorted(self.between.items(), key=lambda item: -item[1][0])[:12]
        out["between_steps"] = [[name, calls, round(own / 1e6, 3), round(blocked / 1e6, 3)]
                                for name, (own, calls, blocked) in between]
        out["allocator"] = self.allocator_delta()
        out["gil"] = {"switch_ms": round(sys.getswitchinterval() * 1e3, 3), "threads": threading.active_count()}
        self.between = {}
        self.sites = {}
        self.implicit = {}
        self.sync_open_steps = 0
        return out

    def allocator_delta(self):
        try:
            stats = torch.cuda.memory_stats(self.timers.device)
        except Exception:
            return None
        now = {name: int(stats.get(name, 0)) for name in ("num_alloc_retries", "num_device_alloc", "num_device_free",
                                                          "num_ooms")}
        before = self.allocator
        self.allocator = now
        return None if before is None else {name: now[name] - before[name] for name in now}


def summarize_prep(steps):
    names = {}
    for one in steps:
        spans = one.get("spans") or {}
        for name, (own, calls, blocked) in spans.items():
            names.setdefault(name, []).append((own / 1e6, calls, blocked / 1e6))
    count = max(len(steps), 1)
    table = {}
    for name, rows in names.items():
        own = np.asarray([row[0] for row in rows], dtype=np.float64)
        table[name] = {"self_ms": [round(float(own.sum() / count), 3), round(float(np.percentile(own, 50)), 3),
                                   round(float(np.percentile(own, 90)), 3)],
                       "calls": round(float(sum(row[1] for row in rows) / count), 1),
                       "blocked_ms": round(float(sum(row[2] for row in rows) / count), 3),
                       "steps": len(rows)}
    ordered = sorted(table.items(), key=lambda item: -item[1]["self_ms"][0])[:PREP_TOP]
    phases = {}
    for one in steps:
        for phase, spent in (one.get("blocked") or {}).items():
            phases[phase] = phases.get(phase, 0) + spent
    return {"prep_ms": dict(ordered), "blocked_ms": {phase: round(spent / 1e6 / count, 3)
                                                      for phase, spent in sorted(phases.items())}}


def log_prep(summary):
    spans = summary.get("prep_ms") or {}
    top = " ".join(f"{name} {held['self_ms'][0]:.2f}ms*{held['calls']:.0f}"
                   + (f"(blocked {held['blocked_ms']:.2f})" if held["blocked_ms"] >= 0.01 else "")
                   for name, held in list(spans.items())[:8])
    blocked = " ".join(f"{phase} {spent:.2f}" for phase, spent in (summary.get("blocked_ms") or {}).items())
    return f"PREP {summary['kind']}/{summary['mode']} self ms/step: {top} | host blocked ms/step: {blocked or '-'}"


def _offset(timers, base):
    return base or timers.phase


def _resolve(owner, name):
    held = owner.__dict__.get(name) if hasattr(owner, "__dict__") else None
    if held is not None:
        return lambda *args, **kwargs: held(*args, **kwargs)
    kind = type(owner)
    return lambda *args, **kwargs: getattr(kind, name)(owner, *args, **kwargs)


def _hook(timers, owner, name, wrapper):
    if not hasattr(owner, name):
        return False
    original = _resolve(owner, name)
    setattr(owner, name, wrapper(original))
    timers.hooks[name] = timers.hooks.get(name, 0) + 1
    return True


def side_entry(timers, lane):
    forks = getattr(timers, "forks", None)
    held = forks.get(lane) if forks else None
    if held is None:
        return DISCARD
    if held:
        forks[lane] = False
        return SIDE + HKV_WAIT
    return SIDE + OTHER


def fork(timers):
    def wrap(original):
        def wrapped(runtime, *args, **kwargs):
            stream = _raw_stream(timers.index)
            copy = getattr(getattr(getattr(runtime, "index_group", None), "copy_stream", None), "cuda_stream", None)
            if timers.timed is None or stream != timers.timed or copy is None or kwargs.get("inline"):
                return original(runtime, *args, **kwargs)
            lane = timers.lane(copy)
            timers.stamp(stream, DISCARD, lane)
            timers.forks[lane] = True
            try:
                return original(runtime, *args, **kwargs)
            finally:
                timers.forks.pop(lane, None)
        return wrapped
    return wrap


def entry(timers, group, base):
    def wrap(original):
        def wrapped(*args, **kwargs):
            timers.switch(group + _offset(timers, base))
            return original(*args, **kwargs)
        return wrapped
    return wrap


def host_aware(timers, group):
    held = bracket(timers, group, 0)
    switched = entry(timers, group, 0)

    def wrap(original):
        inside = held(original)
        outside = switched(original)

        def wrapped(*args, **kwargs):
            current = timers.current
            if timers.timed is not None and current < IDLE and current % G == HKV_SHADOW:
                return original(*args, **kwargs)
            if timers.timed is not None and current < IDLE and current % G in HKV_FAMILY:
                return inside(*args, **kwargs)
            return outside(*args, **kwargs)
        return wrapped
    return wrap


def bracket(timers, group, base, after=None):
    def wrap(original):
        def wrapped(*args, **kwargs):
            stream = _raw_stream(timers.index)
            if timers.timed is None:
                return original(*args, **kwargs)
            if stream != timers.timed:
                lane = timers.lane(stream)
                timers.stamp(stream, side_entry(timers, lane), lane)
                out = original(*args, **kwargs)
                timers.stamp(stream, SIDE + group, lane)
                return out
            offset = _offset(timers, base)
            before = timers.current
            timers.switch(group + offset)
            out = original(*args, **kwargs)
            timers.switch(before if after is None else after + offset)
            return out
        return wrapped
    return wrap


def _linear_types():
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except Exception:
        return ()
    return (LinearBase,)


def hc_projections(model):
    if str(os.environ.get(HC_ENV) or "").strip() != "1":
        return frozenset()
    return frozenset(id(child) for module in model.modules() if type(module).__name__ == "GatedResidual"
                     for child in module.children())


def hook_model(timers, model, base):
    linear = _linear_types()
    hc_owned = hc_projections(model)
    seen = set()
    for name, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        kind = type(module).__name__
        if ".ple." in f".{name}." and kind != "Qwen4ExpPLELayer":
            continue
        if name.endswith("embed_tokens") and hasattr(module, "forward"):
            _hook(timers, module, "forward", site_marked("embed"))
        if kind == "GatedResidual":
            for method in ("mix", "combine_and_mix", "combine"):
                _hook(timers, module, method, entry(timers, BF16C, base))
        elif linear and isinstance(module, linear) and type(module).__name__ != "ParallelLMHead":
            if name.endswith(".gate") or name.endswith("shared_expert_gate"):
                continue
            group = (BF16C if hasattr(module, "taaf_bf16c") or id(module) in hc_owned else
                     DENSE_HOLE if getattr(module, "taaf_dense_hole", False) else DENSE)
            _hook(timers, module, "forward", entry(timers, group, base))
        elif kind == "QSAIndexer":
            _hook(timers, module, "forward", bracket(timers, INDEXER, base, after=QSA_ATTN))
        elif hasattr(module, "_forward_core") and hasattr(module, "in_proj_qkvz"):
            for method in ("_forward_core", "_forward_core_fused_norm_packed"):
                if hasattr(module, method):
                    _hook(timers, module, method, entry(timers, GDN, base))
        elif kind.endswith("SparseMoeBlock"):
            _hook(timers, module, "forward", entry(timers, MOE_ROUTE, base))
        elif hasattr(module, "forward_modular") and hasattr(module, "forward_monolithic"):
            for method in ("forward_modular", "forward_monolithic"):
                _hook(timers, module, method, entry(timers, MOE_EXPERTS, base))
        elif kind == "Qwen4ExpPLELayer":
            _hook(timers, module, "forward", bracket(timers, PLE, base))
            _hook(timers, module, "start_prefetch", bracket(timers, PLE, base))


def site_mark(site, phase):
    marks = sys.modules.get("vllm.taaf_marks")
    if marks is not None and getattr(marks, "ON", False):
        marks.mark(site, phase)


def site_marked(site):
    def wrap(original):
        def wrapped(*args, **kwargs):
            site_mark(site, 0)
            try:
                return original(*args, **kwargs)
            finally:
                site_mark(site, 1)
        return wrapped
    return wrap


def closing_mark(site):
    def wrap(original):
        def wrapped(*args, **kwargs):
            site_mark(site, 1)
            return original(*args, **kwargs)
        return wrapped
    return wrap


def hook_logits(timers, model, base):
    _hook(timers, model, "compute_logits", bracket(timers, LM_HEAD, base, after=SAMPLING))
    _hook(timers, model, "compute_logits", closing_mark("post_forward"))


def hook_forward(timers, model, base, mark):
    original = _resolve(model, "forward")

    def wrapped(*args, **kwargs):
        if mark:
            timers.set_mode("eager")
        timers.switch(OTHER + _offset(timers, base))
        out = original(*args, **kwargs)
        timers.switch(OTHER + _offset(timers, base))
        site_mark("post_forward", 0)
        if mark:
            timers.mark("launch1")
        return out

    model.forward = wrapped
    timers.hooks["model.forward"] = timers.hooks.get("model.forward", 0) + 1


def hook_cache_classes(timers):
    try:
        from vllm.v1.hisparse import runtime as hisparse_runtime
    except Exception as failure:
        say(f"the hisparse runtime classes are not timed: {type(failure).__name__}: {failure}")
        return
    for class_name, method, group in (("HiSparseCacheHandle", "prepare_group_for_batch", HKV_META),
                                      ("HiSparseIndexGroup", "convert_decode_logical_to_physical_topk", HKV_SWAP)):
        owner = getattr(hisparse_runtime, class_name, None)
        if owner is not None:
            _hook(timers, owner, method, bracket(timers, group, 0))


def hook_waits(timers):
    stream_class = torch.cuda.Stream
    if getattr(stream_class, "_taaf_step_timers", False):
        return
    original = stream_class.wait_event

    def wait_event(self, event=None):
        owner = _STATE["capture"]
        if owner is None or owner.timed is None or self.cuda_stream != owner.timed:
            return original(self, event)
        enter = getattr(owner, "seg_enter", None)
        token = enter(PREP_SEG0 + PREP_SEGS.index("event_wait")) if enter is not None else None
        if token is not None:
            try:
                return original(self, event)
            finally:
                owner.seg_exit(token)
        if owner.current >= IDLE or owner.current % G not in HKV_FAMILY:
            return original(self, event)
        before = owner.current
        owner.switch(before - before % G + HKV_WAIT)
        out = original(self, event)
        owner.switch(before)
        return out

    stream_class.wait_event = wait_event
    stream_class._taaf_step_timers = True
    timers.hooks["stream.wait_event"] = 1


SEG_SITES = (("runner", "prepare_inputs", "inputs"), ("model_state", "prepare_inputs", "inputs"),
             ("model_state", "prepare_inputs_embeds", "inputs"), ("runner", "prepare_attn", "attn_meta"),
             ("model_state", "prepare_attn", "attn_meta"), ("runner", "finish_requests", "states"),
             ("runner", "free_states", "states"), ("runner", "add_requests", "states"),
             ("runner", "update_requests", "states"), ("block_tables", "apply_staged_writes", "states"),
             ("model_state", "preprocess_state", "states"), ("kv_connector", "pre_forward", "connector"))
SEG_WORKER = ("vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker", "HiSparseConnectorWorker",
              (("start_step", "host_step"), ("_copy_host_blocks", "host_blocks")))
SEG_GDN = ("vllm.taaf_gdn_mixed_full", ("_once_states", "_walk_state", "_shared_base"), "gdn_prep")


def seg_wrap(timers, owner, name, seg):
    function = owner.__dict__.get(name) if hasattr(owner, "__dict__") else None
    if function is None:
        function = getattr(owner, name, None)
    if (function is None or not callable(function) or isinstance(function, (staticmethod, classmethod))
            or getattr(function, "_taaf_seg", False)):
        return False
    col = PREP_SEG0 + PREP_SEGS.index(seg)

    def wrapped(*args, **kwargs):
        token = timers.seg_enter(col)
        try:
            return function(*args, **kwargs)
        finally:
            timers.seg_exit(token)

    wrapped.__wrapped__ = function
    wrapped.__name__ = getattr(function, "__name__", name)
    wrapped._taaf_seg = True
    try:
        setattr(owner, name, wrapped)
    except (AttributeError, TypeError):
        return False
    timers.hooks[f"seg.{seg}"] = timers.hooks.get(f"seg.{seg}", 0) + 1
    return True


def hook_event_waits(timers):
    event_class = torch.cuda.Event
    if getattr(event_class, "_taaf_seg_wait", False):
        return
    original = event_class.wait
    col = PREP_SEG0 + PREP_SEGS.index("event_wait")

    def wait(self, stream=None):
        owner = _STATE["capture"]
        enter = getattr(owner, "seg_enter", None) if owner is not None else None
        if enter is None or owner.timed is None:
            return original(self, stream)
        target = stream.cuda_stream if stream is not None else _raw_stream(owner.index)
        token = enter(col) if target == owner.timed else None
        try:
            return original(self, stream)
        finally:
            owner.seg_exit(token)

    event_class.wait = wait
    event_class._taaf_seg_wait = True
    timers.hooks["event.wait"] = 1


def install_segments(timers, runner):
    if hasattr(runner, "prepare_inputs") and not getattr(runner, "_taaf_prepare_marked", False):
        _hook(timers, runner, "prepare_inputs", site_marked("prepare"))
        runner._taaf_prepare_marked = True
    for owner_name, name, seg in SEG_SITES:
        owner = runner if owner_name == "runner" else getattr(runner, owner_name, None)
        if owner is not None:
            seg_wrap(timers, owner, name, seg)
    module = sys.modules.get(SEG_WORKER[0])
    worker = getattr(module, SEG_WORKER[1], None) if module is not None else None
    if worker is not None:
        for name, seg in SEG_WORKER[2]:
            seg_wrap(timers, worker, name, seg)
    module = sys.modules.get(SEG_GDN[0])
    if module is not None:
        for name in SEG_GDN[1]:
            seg_wrap(timers, module, name, SEG_GDN[2])
    hook_event_waits(timers)
    timers.segments = True
    found = {seg: timers.hooks.get(f"seg.{seg}", 0) for seg in PREP_SEGS if seg != "event_wait"}
    say(f"PREP segments armed: the GPU time of a step's prep window is split into {', '.join(PREP_SEGS)} and the "
        f"unnamed rest in prep_wait, each by a stamp on the compute stream at the call's entry and exit "
        f"(event_wait holds every stream or event wait the compute stream takes there); hooked {found}")


def hook_samplers(timers, runner):
    for attr in ("sampler", "rejection_sampler"):
        target = getattr(runner, attr, None)
        if target is None or not callable(target):
            continue
        kind = type(target)
        call = kind.__call__
        wrapped = bracket(timers, SAMP_CORE, 0, after=SAMP_POST)(lambda self, *args, _call=call, **kwargs:
                                                                   _call(self, *args, **kwargs))
        try:
            target.__class__ = type(kind.__name__, (kind,), {"__call__": wrapped})
        except TypeError as failure:
            say(f"{attr} is not timed: {failure}")
            continue
        timers.hooks[attr] = 1


def hook_host_kv(timers):
    hisparse = sys.modules.get("vllm.taaf_qsa_hisparse")
    kv8 = sys.modules.get("vllm.taaf_qsa_kv8")
    if hisparse is None:
        try:
            from vllm import taaf_qsa_hisparse as hisparse
        except Exception:
            hisparse = None
    if kv8 is None:
        try:
            from vllm import taaf_qsa_kv8 as kv8
        except Exception:
            kv8 = None
    if hisparse is not None:
        if hasattr(hisparse, "run"):
            original = hisparse.run
            hisparse.run = bracket(timers, HKV_PLAN, 0)(original)
            timers.hooks["hisparse.run"] = 1
        if hasattr(hisparse, "gather_compact"):
            hisparse.gather_compact = bracket(timers, HKV_GATHER, 0)(hisparse.gather_compact)
            timers.hooks["hisparse.gather_compact"] = 1
        for name, group in (("step_metadata", HKV_META), ("write", HKV_WRITE), ("guard_logical", HKV_GUARD),
                            ("union_convert", HKV_SWAP), ("capped_swap_fused", HKV_SWAP),
                            ("capped_swap", HKV_SWAP), ("resolve_residency", HKV_RESOLVE),
                            ("plan_gather", HKV_GPLAN), ("guard_counts", HKV_GPLAN), ("gather_plan", HKV_GPLAN),
                            ("one_resolve", HKV_FUSED), ("prefetch_join", HKV_PWAIT),
                            ("hits_shadow", HKV_SHADOW)):
            if hasattr(hisparse, name):
                setattr(hisparse, name, bracket(timers, group, 0)(getattr(hisparse, name)))
                timers.hooks[f"hisparse.{name}"] = 1
        if hasattr(hisparse, "prefill_attention"):
            hisparse.prefill_attention = bracket(timers, QSA_ATTN, 0)(hisparse.prefill_attention)
            timers.hooks["hisparse.prefill_attention"] = 1
        if hasattr(hisparse, "union_resolve"):
            hisparse.union_resolve = fork(timers)(hisparse.union_resolve)
            timers.hooks["hisparse.union_resolve"] = 1
        fused = sys.modules.get("vllm.taaf_host_kv_fused")
        if fused is not None and hasattr(fused, "split"):
            fused.split = bracket(timers, HKV_GPLAN, 0)(fused.split)
            timers.hooks["host_kv_fused.split"] = 1
        hook_cache_classes(timers)
    if kv8 is not None:
        if hasattr(kv8, "attention"):
            exact = str(os.environ.get(QSA_EXACT_ENV) or "").strip() == "1"
            kv8.attention = (bracket(timers, QSA_ATTN, 0) if exact else host_aware(timers, QSA_ATTN))(kv8.attention)
            timers.hooks["kv8.attention"] = 1
            if exact:
                timers.hooks["kv8.attention_exact"] = 1
                say(f"{QSA_EXACT_ENV}=1: qsa_attn holds the QSA attention calls alone; the work its callers issue "
                    f"after an attention call returns goes back to the caller's group")
        if hasattr(kv8, "write"):
            kv8.write = bracket(timers, QSA_ATTN, 0)(kv8.write)
            timers.hooks["kv8.write"] = 1
        if hasattr(kv8, "attention_passes"):
            kv8.attention_passes = host_aware(timers, QSA_ATTN)(kv8.attention_passes)
            timers.hooks["kv8.attention_passes"] = 1


def hook_runner(timers, runner):
    kind = type(runner)

    def execute_model(scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
            return kind.execute_model(runner, scheduler_output, *args, **kwargs)
        timers.begin(scheduler_output)
        out = kind.execute_model(runner, scheduler_output, *args, **kwargs)
        timers.mark("exec_end")
        return out

    def sample_tokens(*args, **kwargs):
        out = kind.sample_tokens(runner, *args, **kwargs)
        timers.end()
        return out

    def sample(*args, **kwargs):
        timers.mark("sample0")
        out = kind.sample(runner, *args, **kwargs)
        timers.mark("sample1")
        return out

    runner.execute_model = execute_model
    runner.sample_tokens = sample_tokens
    runner.sample = sample
    timers.hooks["runner"] = 3


def hook_graphs(timers, runner):
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is None or getattr(manager, "_taaf_step_timers", False):
        return
    full = _resolve(manager, "run_fullgraph")
    piecewise = _resolve(manager, "run_pw_graph")

    def run_fullgraph(*args, **kwargs):
        timers.set_mode("full")
        out = full(*args, **kwargs)
        timers.mark("launch1")
        return out

    def run_pw_graph(*args, **kwargs):
        timers.set_mode("pw")
        out = piecewise(*args, **kwargs)
        timers.mark("launch1")
        return out

    manager.run_fullgraph = run_fullgraph
    manager.run_pw_graph = run_pw_graph
    manager._taaf_step_timers = True
    timers.hooks["cudagraph_manager"] = 2


def hook_speculator(timers, speculator):
    if speculator is None:
        return
    propose = _resolve(speculator, "propose")

    def wrapped_propose(*args, **kwargs):
        timers.mark("draft0")
        timers.phase = DRAFT
        timers.switch(OTHER + DRAFT)
        try:
            return propose(*args, **kwargs)
        finally:
            timers.switch(STEP_TAIL)
            timers.phase = 0
            timers.mark("draft1")

    speculator.propose = wrapped_propose
    if hasattr(speculator, "capture"):
        capture = _resolve(speculator, "capture")

        def wrapped_capture(*args, **kwargs):
            timers.phase = DRAFT
            try:
                return capture(*args, **kwargs)
            finally:
                timers.phase = 0

        speculator.capture = wrapped_capture
    timers.hooks["speculator"] = 2


def hook_capture(timers):
    _STATE["capture"] = timers
    graph = torch.cuda.CUDAGraph
    if getattr(graph, "_taaf_step_timers", False):
        return
    begin, end = graph.capture_begin, graph.capture_end
    held = []

    def capture_begin(self, *args, **kwargs):
        owner = _STATE["capture"]
        held.append(owner.timed)
        out = begin(self, *args, **kwargs)
        owner.timed = _raw_stream(owner.index)
        return out

    def capture_end(self, *args, **kwargs):
        if str(os.environ.get("TAAF_VLLM_HOST_KV_PREFETCH_SPEC") or "").strip() == "1":
            hisparse = sys.modules.get("vllm.taaf_qsa_hisparse")
            if hisparse is not None:
                hisparse.prefetch_close()
                hisparse.prefetch_join()
        out = end(self, *args, **kwargs)
        _STATE["capture"].timed = held.pop() if held else None
        return out

    graph.capture_begin = capture_begin
    graph.capture_end = capture_end
    graph._taaf_step_timers = True


def hook_scheduler(timers):
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as failure:
        say(f"the scheduler is not timed: {type(failure).__name__}: {failure}")
        return
    if getattr(Scheduler, "_taaf_step_timers", False):
        return
    schedule = Scheduler.schedule

    def timed_schedule(self, *args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return schedule(self, *args, **kwargs)
        finally:
            timers.sched_ns += time.perf_counter_ns() - started

    Scheduler.schedule = timed_schedule
    Scheduler._taaf_step_timers = True


ACCEPT_ENV = "TAAF_VLLM_REC2"
ACCEPT_EVERY_ENV = "TAAF_VLLM_REC2_EVERY"
ACCEPT_EVERY = 250
ACCEPT_EDGES = (8192, 16384, 32768, 49152, 65536, 98304, 131072)
ACCEPT_NAMES = ("0-8k", "8-16k", "16-32k", "32-48k", "48-64k", "64-96k", "96-128k", "128k+")
ACCEPT_POSITIONS = 8
ACCEPT_FIELDS = ("rows", "drafted", "accepted")


def accept_on():
    return str(os.environ.get(ACCEPT_ENV) or "").strip().lower() in ("1", "on", "true")


def accept_say(line):
    print(f"ACCEPT_TALLY {line}", flush=True)


class Accept:
    def __init__(self, device, every):
        self.device = torch.device(device)
        self.every = max(1, int(every))
        buckets = len(ACCEPT_NAMES)
        self.edges = torch.tensor(ACCEPT_EDGES, dtype=torch.int64, device=self.device)
        self.sums = torch.zeros(buckets, len(ACCEPT_FIELDS), dtype=torch.int64, device=self.device)
        self.hits = torch.zeros(buckets, ACCEPT_POSITIONS, dtype=torch.int64, device=self.device)
        self.tries = torch.zeros(buckets, ACCEPT_POSITIONS, dtype=torch.int64, device=self.device)
        self.steps = torch.arange(ACCEPT_POSITIONS, dtype=torch.int64, device=self.device)
        self.calls = 0
        self.lines = 0

    def fold(self, context, num_sampled, num_rejected, live):
        sampled = num_sampled.to(torch.int64)
        rejected = num_rejected.to(torch.int64)
        accepted = (sampled - 1).clamp_min(0)
        drafted = accepted + rejected
        spec = live & (drafted > 0)
        weight = spec.to(torch.int64)
        bucket = torch.bucketize(context.to(torch.int64), self.edges, right=True)
        self.sums.index_add_(0, bucket, torch.stack((weight, drafted * weight, accepted * weight), 1))
        steps = self.steps.view(1, -1)
        self.hits.index_add_(0, bucket, ((accepted.view(-1, 1) > steps) & spec.view(-1, 1)).to(torch.int64))
        self.tries.index_add_(0, bucket, ((drafted.view(-1, 1) > steps) & spec.view(-1, 1)).to(torch.int64))

    def lines_for(self, sums, hits, tries):
        cells = []
        for name, (rows, drafted, accepted), hit, tried in zip(ACCEPT_NAMES, sums, hits, tries):
            if not rows:
                continue
            rates = "/".join(f"{one / two:.3f}" for one, two in zip(hit, tried) if two)
            cells.append(f"{name}:{rows}:{(accepted + rows) / rows:.3f}:{rates}")
        return (f"window={self.lines} calls={self.every} cells=ctx:verify_rows:tokens_per_step:per_position "
                f"{' '.join(cells)}")

    def report(self):
        self.calls += 1
        if self.calls % self.every:
            return None
        held = [part.to("cpu", copy=True).tolist() for part in (self.sums, self.hits, self.tries)]
        for part in (self.sums, self.hits, self.tries):
            part.zero_()
        self.lines += 1
        line = self.lines_for(*held)
        accept_say(line)
        return line


def install_accept(runner):
    if not accept_on():
        return None
    states = getattr(runner, "req_states", None)
    computed = getattr(getattr(states, "num_computed_tokens", None), "gpu", None)
    original = getattr(runner, "postprocess_sampled", None)
    if computed is None or original is None:
        accept_say(f"not armed on {type(runner).__name__}: the runner has no "
                   f"{'req_states.num_computed_tokens.gpu' if computed is None else 'postprocess_sampled'}")
        return None
    tally = Accept(computed.device, os.environ.get(ACCEPT_EVERY_ENV) or ACCEPT_EVERY)

    def postprocess_sampled(idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs):
        if num_rejected is not None and num_sampled is not None:
            live = idx_mapping >= 0
            context = computed.index_select(0, idx_mapping.clamp_min(0).to(torch.int64))
            tally.fold(context, num_sampled[:idx_mapping.numel()], num_rejected[:idx_mapping.numel()], live)
            tally.report()
        return original(idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs)

    runner.postprocess_sampled = postprocess_sampled
    accept_say(f"armed on {type(runner).__name__}: each verify step adds every drafted request's drafted and accepted "
               f"tokens, and per-position acceptance, to a GPU histogram by the request's computed context "
               f"(edges {','.join(str(one) for one in ACCEPT_EDGES)}); read every {tally.every} steps")
    return tally


SNAPSHOT_ENV = "TAAF_VLLM_MEM_SNAPSHOT"
SNAPSHOT_HISTORY_ENV = "TAAF_VLLM_MEM_SNAPSHOT_HISTORY"
SNAPSHOT_DIR_ENV = "TAAF_VLLM_MEM_SNAPSHOT_DIR"
SNAPSHOT_DIR = os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working"
SNAPSHOT_ENTRIES = 20000
SNAPSHOT_TOP = 30
SNAPSHOT_SKIP = ("/torch/", "<frozen", "/contextlib.py", "/functools.py", "/threading.py")


def snapshot_times(text):
    said = str(text or "").strip()
    if not said or said.lower() in ("0", "off", "false", "no"):
        return []
    times = sorted({float(part) for part in said.split(",") if part.strip()})
    if not times or times[0] < 0:
        raise RuntimeError(f"{SNAPSHOT_ENV}={said!r} must list seconds of play, e.g. 180 or 180,330")
    return times


def snapshot_say(line):
    print(f"MEM_SNAPSHOT {line}", flush=True)


def user_frame(frames):
    for frame in frames or ():
        name = str(frame.get("filename", "")).replace(chr(92), "/")
        if name and not any(part in name for part in SNAPSHOT_SKIP):
            short = name.split("/site-packages/")[-1].split("/dist-packages/")[-1]
            return f"{short}:{frame.get('line', 0)} {frame.get('name', '')}"
    return "no history (allocated before recording began)"


def snapshot_summary(snapshot, top=SNAPSHOT_TOP):
    reserved = allocated = 0
    owners = {}
    pools = {}
    for segment in snapshot.get("segments", ()):
        reserved += int(segment.get("total_size", 0))
        pool = segment.get("segment_pool_id")
        key = "graph_pool" if pool and tuple(pool) != (0, 0) else "default"
        pools[key] = pools.get(key, 0) + int(segment.get("total_size", 0))
        for block in segment.get("blocks", ()):
            if block.get("state") != "active_allocated":
                continue
            size = int(block.get("size", 0))
            allocated += size
            owner = user_frame(block.get("frames"))
            held = owners.setdefault(owner, [0, 0])
            held[0] += size
            held[1] += 1
    ranked = sorted(owners.items(), key=lambda item: -item[1][0])[:top]
    return {"reserved": reserved, "allocated": allocated, "pools": pools,
            "top": [[owner, size, count] for owner, (size, count) in ranked]}


class Snapshots:
    def __init__(self, times, folder, history):
        self.times = list(times)
        self.folder = Path(folder)
        self.history = history
        self.first = None
        self.taken = 0

    def due(self):
        if not self.times:
            return False
        now = time.monotonic()
        if self.first is None:
            self.first = now
            return False
        return now - self.first >= self.times[0]

    def take(self):
        at = self.times.pop(0)
        began = time.monotonic()
        snapshot = torch.cuda.memory._snapshot()
        free, total = torch.cuda.mem_get_info()
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"mem-snapshot-{int(at)}s.pickle"
        import pickle
        with open(path, "wb") as handle:
            pickle.dump(snapshot, handle)
        summary = snapshot_summary(snapshot)
        outside = total - free - summary["reserved"]
        gib = 1 << 30
        snapshot_say(json.dumps({"at_s": at, "path": str(path), "total_gib": round(total / gib, 2),
                                 "free_gib": round(free / gib, 2), "reserved_gib": round(summary["reserved"] / gib, 2),
                                 "allocated_gib": round(summary["allocated"] / gib, 2),
                                 "outside_torch_gib": round(outside / gib, 2),
                                 "pools_gib": {key: round(value / gib, 2) for key, value in summary["pools"].items()},
                                 "top": [[owner, round(size / (1 << 20), 1), count]
                                         for owner, size, count in summary["top"]],
                                 "seconds": round(time.monotonic() - began, 2)}, separators=(",", ":")))
        self.taken += 1
        if not self.times and self.history:
            torch.cuda.memory._record_memory_history(enabled=None)
            snapshot_say("history recording stopped after the last snapshot")


def install_snapshots(runner):
    times = snapshot_times(os.environ.get(SNAPSHOT_ENV))
    if not times:
        return None
    history = str(os.environ.get(SNAPSHOT_HISTORY_ENV, "1")).strip().lower() not in ("0", "off", "false", "no")
    if history:
        torch.cuda.memory._record_memory_history(enabled="all", context="alloc", stacks="python",
                                                 max_entries=SNAPSHOT_ENTRIES)
    shots = Snapshots(times, os.environ.get(SNAPSHOT_DIR_ENV) or SNAPSHOT_DIR, history)
    kind = type(runner)
    inner = runner.__dict__.get("execute_model")

    def execute_model(scheduler_output, *args, **kwargs):
        if getattr(scheduler_output, "total_num_scheduled_tokens", 0) and shots.due():
            try:
                shots.take()
            except Exception as failure:
                shots.times.clear()
                snapshot_say(f"failed and stops: {type(failure).__name__}: {failure}")
        if inner is not None:
            return inner(scheduler_output, *args, **kwargs)
        return kind.execute_model(runner, scheduler_output, *args, **kwargs)

    runner.execute_model = execute_model
    snapshot_say(f"armed on {kind.__name__}: torch allocator snapshots at {times} s after the first scheduled step "
                 f"into {shots.folder}/mem-snapshot-<s>s.pickle, each with a line of reserved, allocated, memory "
                 f"outside torch and the {SNAPSHOT_TOP} largest live owners by first user frame; "
                 + ("allocation history (python stacks, alloc context, "
                    f"{SNAPSHOT_ENTRIES} entries) records from now until the last snapshot, so blocks allocated "
                    "before this point (the weights) show no frame"
                    if history else "no history: blocks carry no frames, only segments and sizes"))
    return shots


TRACE_ENV = "TAAF_VLLM_DRAFT_TRACE"
TRACE_AT_ENV = "TAAF_VLLM_DRAFT_TRACE_AT"
TRACE_DIR_ENV = "TAAF_VLLM_DRAFT_TRACE_DIR"
TRACE_AT = 60.0
TRACE_LANES = 16
TRACE_TOP = 20


def trace_steps():
    said = str(os.environ.get(TRACE_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def trace_say(line):
    print(f"DRAFT_TRACE {line}", flush=True)


def trace_rows(lens, spec):
    return np.where(lens == spec + 1, lens, 1)


def trace_record(ids, qsl, sampled, num_sampled, live, logit_rows, spec, lanes):
    lens = np.diff(qsl)
    rows = trace_rows(lens, spec)
    starts = np.concatenate(([0], np.cumsum(rows)))
    out = []
    for lane in range(len(lens)):
        if len(out) >= lanes:
            break
        if not live[lane] or lens[lane] != spec + 1:
            continue
        drafts = ids[qsl[lane] + 1:qsl[lane] + 1 + spec].tolist()
        count = int(num_sampled[lane])
        row = {"lane": lane, "drafts": drafts, "sampled": sampled[lane][:count].tolist(), "accepted": max(0, count - 1)}
        if logit_rows is not None and starts[lane + 1] <= logit_rows[0].shape[0]:
            row["argmax"] = logit_rows[0][starts[lane]:starts[lane + 1]].tolist()
            row["top1"] = [round(float(one), 4) for one in logit_rows[1][starts[lane]:starts[lane + 1]]]
            if len(logit_rows) >= 4:
                top_ids = logit_rows[2][starts[lane]:starts[lane + 1]]
                top_p = logit_rows[3][starts[lane]:starts[lane + 1]]
                row["top_ids"] = top_ids.tolist()
                row["top_p"] = [[round(float(one), 5) for one in probs] for probs in top_p]
                ranks, probs = [], []
                for at, draft in enumerate(drafts):
                    hits = np.nonzero(top_ids[at] == draft)[0]
                    ranks.append(int(hits[0]) if hits.size else -1)
                    probs.append(round(float(top_p[at][hits[0]]), 5) if hits.size else None)
                row["draft_rank"] = ranks
                row["draft_p"] = probs
        out.append(row)
    return out


def trace_summary(records, spec):
    agree = [[0, 0] for _ in range(spec)]
    cond = [[0, 0] for _ in range(spec)]
    top1 = [[0.0, 0] for _ in range(spec + 1)]
    rejected = [0, 0, 0]
    for row in records:
        argmax = row.get("argmax")
        if argmax is None:
            continue
        ranks = row.get("draft_rank")
        if ranks is not None and row["accepted"] < spec:
            rank = ranks[row["accepted"]]
            rejected[0] += 1
            rejected[1] += int(rank == 1)
            rejected[2] += int(rank < 0)
        for at in range(spec):
            hit = int(row["drafts"][at] == argmax[at])
            agree[at][0] += hit
            agree[at][1] += 1
            if row["accepted"] >= at:
                cond[at][0] += hit
                cond[at][1] += 1
        for at, value in enumerate(row["top1"][:spec + 1]):
            top1[at][0] += value
            top1[at][1] += 1
    rate = lambda pairs: "/".join(f"{one / two:.3f}" if two else "-" for one, two in pairs)
    return (f"records={len(records)} with_logits={sum('argmax' in row for row in records)} "
            f"draft_eq_target_argmax_by_row={rate(agree)} same_given_prefix_accepted={rate(cond)} "
            f"target_top1_prob_by_row={rate(top1)}"
            + (f" first_rejected={rejected[0]} of_them_target_rank2={rejected[1]} outside_top{TRACE_TOP}="
               f"{rejected[2]}" if rejected[0] else ""))


class Trace:
    def __init__(self, runner, steps):
        self.runner = runner
        self.steps = steps
        self.at = float(os.environ.get(TRACE_AT_ENV) or TRACE_AT)
        self.folder = Path(os.environ.get(TRACE_DIR_ENV) or os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working")
        self.spec = int(getattr(getattr(runner, "rejection_sampler", None), "num_speculative_steps", 0) or 0)
        self.first = None
        self.taken = 0
        self.done = False
        self.logits = None
        self.records = []

    def active(self):
        if self.done or self.first is None:
            return False
        return time.monotonic() - self.first >= self.at

    def keep_logits(self, logits):
        if not self.active() or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            return
        probs = torch.softmax(logits.float(), dim=-1)
        top, ids = probs.topk(min(TRACE_TOP, int(probs.shape[-1])), dim=-1)
        top, ids = top.to("cpu").numpy(), ids.to("cpu").numpy()
        self.logits = (ids[:, 0], top[:, 0], ids, top)

    def take(self, idx_mapping, sampled_tokens, num_sampled, query_start_loc):
        if self.done or num_sampled is None or query_start_loc is None or not self.spec:
            return
        if self.first is None:
            self.first = time.monotonic()
            return
        if not self.active():
            self.logits = None
            return
        reqs = idx_mapping.numel()
        qsl = query_start_loc[:reqs + 1].to("cpu").numpy().astype(np.int64)
        ids = self.runner.input_buffers.input_ids[:int(qsl[-1])].to("cpu").numpy()
        live = (idx_mapping >= 0).to("cpu").numpy()
        sampled = sampled_tokens[:reqs].to("cpu").numpy()
        counts = num_sampled[:reqs].to("cpu").numpy()
        rows = trace_record(ids, qsl, sampled, counts, live, self.logits, self.spec, TRACE_LANES)
        for row in rows:
            row["step"] = self.taken
        self.records.extend(rows)
        self.logits = None
        self.taken += 1
        if self.taken >= self.steps:
            self.finish()

    def finish(self):
        self.done = True
        path = self.folder / "draft-trace.jsonl"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as out:
                for row in self.records:
                    out.write(json.dumps(row) + "\n")
        except OSError as failure:
            trace_say(f"could not write {path}: {type(failure).__name__}: {failure}")
        trace_say(f"steps={self.taken} file={path} " + trace_summary(self.records, self.spec))


def install_trace(runner):
    steps = trace_steps()
    original = getattr(runner, "postprocess_sampled", None)
    model = getattr(runner, "model", None)
    compute = getattr(model, "compute_logits", None)
    if not steps or original is None or compute is None or getattr(runner, "input_buffers", None) is None:
        trace_say("not armed: the runner lacks postprocess_sampled, model.compute_logits or input_buffers")
        return None
    trace = Trace(runner, steps)

    def compute_logits(*args, **kwargs):
        logits = compute(*args, **kwargs)
        trace.keep_logits(logits)
        return logits

    def postprocess_sampled(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None, *args,
                            **kwargs):
        trace.take(idx_mapping, sampled_tokens, num_sampled, query_start_loc)
        return original(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc, *args, **kwargs)

    model.compute_logits = compute_logits
    runner.postprocess_sampled = postprocess_sampled
    trace_say(f"armed: {steps} verify steps from {trace.at:.0f} s after the first drafted step, up to {TRACE_LANES} "
              f"lanes each, eager host reads only inside that window (logits rows only on steps whose sampler runs "
              f"outside a captured graph); drafts, sampled tokens, per-row target argmax and top-1 probability to "
              f"{trace.folder}/draft-trace.jsonl")
    return trace


def clock_window():
    said = str(os.environ.get(CLOCK_ENV) or "").strip()
    if not said:
        return CLOCK_DEFAULT_S
    if not said.isdigit():
        raise ValueError(f"{CLOCK_ENV}={said!r} is not a whole number of seconds")
    return int(said)


class _Noted:
    def __init__(self, stream, seen):
        self._stream = stream
        self._seen = seen

    def write(self, text):
        if text:
            self._seen.set()
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_line_clock(window=None, tick=CLOCK_TICK_S, out=None):
    window = clock_window() if window is None else window
    if window <= 0 or _STATE.get("clock") is not None:
        return None
    began = time.monotonic()
    seen = threading.Event()
    target = out or sys.stdout
    sys.stdout = _Noted(sys.stdout, seen)
    sys.stderr = _Noted(sys.stderr, seen)

    def run():
        while time.monotonic() - began < window:
            time.sleep(tick)
            if seen.is_set():
                seen.clear()
                print(f"LINE_CLOCK +{time.monotonic() - began:.1f} s since the model runner was built; every "
                      f"line above was written before this mark", file=target, flush=True)

    thread = threading.Thread(target=run, name="taaf-line-clock", daemon=True)
    thread.start()
    _STATE["clock"] = thread
    print(f"LINE_CLOCK armed for {window} s ({CLOCK_ENV}): a mark follows any second in which the worker wrote, "
          f"so a check's cost reads as the gap between marks", file=target, flush=True)
    return thread


def install(runner):
    install_line_clock()
    timers = install_timers(runner)
    if snapshot_times(os.environ.get(SNAPSHOT_ENV)) and _STATE.get("snapshots") is None:
        _STATE["snapshots"] = install_snapshots(runner)
    if expert_tally_on() and _STATE.get("experts") is None:
        _STATE["experts"] = install_experts(runner)
    if accept_on() and _STATE.get("accept") is None:
        _STATE["accept"] = install_accept(runner)
    if trace_steps() and _STATE.get("trace") is None:
        _STATE["trace"] = install_trace(runner)
    if vision_tally_on() and _STATE.get("vision") is None:
        _STATE["vision"] = install_vision()
    if spec_index_on() and _STATE.get("spec") is None:
        _STATE["spec"] = install_spec(runner, timers)
    if capture_on() and _STATE.get("draft_capture") is None:
        _STATE["draft_capture"] = install_capture(runner)
    return timers


def install_timers(runner):
    if not enabled() or _STATE["timers"] is not None:
        return None
    timers = Timers(runner.device)
    timers.warm()
    hook_capture(timers)
    hook_scheduler(timers)
    model = runner.model
    hook_model(timers, model, 0)
    hook_logits(timers, model, 0)
    hook_forward(timers, model, 0, mark=True)
    draft = runner.get_draft_model() if hasattr(runner, "get_draft_model") else None
    if draft is not None and draft is not model:
        hook_model(timers, draft, DRAFT)
        hook_logits(timers, draft, DRAFT)
        hook_forward(timers, draft, DRAFT, mark=False)
    hook_speculator(timers, getattr(runner, "speculator", None))
    hook_host_kv(timers)
    hook_waits(timers)
    hook_samplers(timers, runner)
    hook_runner(timers, runner)
    mode = prep_mode()
    if mode:
        timers.prep_stamp = True
        timers.probe = Probe(timers, mode).install(runner)
    kind = type(runner)
    begin = timers.begin

    def begin_and_graphs(scheduler_output):
        hook_graphs(timers, runner)
        timers.begin = begin
        begin(scheduler_output)

    timers.begin = begin_and_graphs
    _STATE["timers"] = timers
    calibration = timers.calibration
    gpu = calibration.get("gpu_us_per_stamp_in_graph")
    say(f"armed on {kind.__name__}: groups {','.join(GROUPS)}; each group boundary launches a one-thread kernel "
        f"that reads %globaltimer and adds the span since the last boundary into a {RING}-step ring on the GPU, "
        f"captured into every CUDA graph; read every {READ_EVERY} steps by a non-blocking copy and logged every "
        f"{timers.every:.0f} s to {timers.path}; a stamp costs "
        f"{'unmeasured' if gpu is None else f'{gpu:.2f} us'} of GPU in a graph and "
        f"{calibration['cpu_us_per_stamp']:.1f} us of CPU per eager launch "
        f"({'raw launcher' if calibration.get('fast_launch') else 'JIT launcher'}); hooks {timers.hooks}")
    timers.write([json.dumps({"t": round(time.time(), 1), "armed": True, "groups": list(GROUPS), "width": WIDTH,
                              "ring": RING, "calibration": calibration, "hooks": timers.hooks},
                             separators=(",", ":"))])
    return timers


def expert_tally_on():
    return str(os.environ.get(EXPERT_ENV) or "").strip() == "1"


def vision_tally_on():
    return str(os.environ.get(VISION_ENV) or "").strip() == "1"


def spec_index_on():
    return any(str(os.environ.get(name) or "").strip() == "1" for name in (SPEC_ENV, SPEC_PREFETCH_ENV))


CAPTURE_ENV = "TAAF_VLLM_DRAFT_CAPTURE"
CAPTURE_MODULE = "vllm.taaf_draft_capture"


def capture_on():
    return str(os.environ.get(CAPTURE_ENV) or "").strip().lower() in ("1", "on", "true")


def install_capture(runner):
    try:
        module = __import__(CAPTURE_MODULE, fromlist=["install"])
        return module.install(runner) or False
    except Exception as failure:
        say(f"DRAFT_CAPTURE not armed: {type(failure).__name__}: {failure}")
        return False


def install_spec(runner, timers):
    def timed(name):
        if timers is None:
            return lambda function: function
        return bracket(timers, GROUPS.index(name), 0)

    try:
        module = __import__(SPEC_MODULE, fromlist=["install"])
        return module.install(runner, timed) or False
    except Exception as failure:
        say(f"SPEC_INDEX not armed: {type(failure).__name__}: {failure}")
        return False


def install_vision():
    try:
        module = __import__(VISION_MODULE, fromlist=["install"])
        return module.install() or False
    except Exception as failure:
        say(f"VISION_TALLY not armed: {type(failure).__name__}: {failure}")
        return False


def expert_say(line):
    print(f"EXPERT_TALLY {line}", flush=True)


def moe_layers(model, group):
    found = []
    seen = set()
    for name, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not (hasattr(module, "forward_modular") and hasattr(module, "forward_monolithic")):
            continue
        experts = int(getattr(module, "global_num_experts", 0) or 0)
        if experts <= 0:
            continue
        found.append((module, str(getattr(module, "layer_name", None) or name), group, experts,
                      int(getattr(module, "top_k", 0) or 0)))
    return found


class _Done:
    def query(self):
        return True


class Experts:
    def __init__(self, device, layers, every):
        self.device = torch.device(device)
        self.layers = layers
        self.count = len(layers)
        self.every = max(1, int(every))
        slots = len(EXPERT_KINDS) * self.count
        widest = max([layer[3] for layer in layers] or [1])
        self.acc = torch.zeros(slots, len(EXPERT_FIELDS), dtype=torch.int64, device=self.device)
        self.low = torch.full((slots,), EXPERT_FLOOR, dtype=torch.int64, device=self.device)
        self.high = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.flags = torch.zeros(widest + 1, dtype=torch.int64, device=self.device)
        self.edges = torch.tensor(EXPERT_HIST_EDGES, dtype=torch.int64, device=self.device)
        self.peak = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.hist = torch.zeros(slots, len(EXPERT_HIST_EDGES) + 1, dtype=torch.int64, device=self.device)
        self.one = torch.ones(1, dtype=torch.int64, device=self.device)
        self.kind = torch.zeros(1, dtype=torch.int64, device=self.device)
        pinned = self.device.type == "cuda"
        self.host = [held.new_zeros(held.shape, device="cpu") for held in (self.acc, self.low, self.high, self.peak, self.hist)]
        if pinned:
            self.host = [held.pin_memory() for held in self.host]
        self.current = 0
        self.steps = [0] * len(EXPERT_KINDS)
        self.total = 0
        self.started = False
        self.pending = None
        self.lines = 0

    def note(self, slot, ids, experts):
        flat = ids.reshape(-1).to(torch.int64).clamp(0, experts)
        flags = self.flags[:experts + 1]
        flags.zero_()
        flags.index_add_(0, flat, torch.ones_like(flat))
        per_expert = flags[:experts]
        active = per_expert > 0
        distinct = active.sum(dtype=torch.int64).view(1)
        peak = per_expert.max().view(1)
        bins = torch.bucketize(per_expert, self.edges)
        shape = torch.zeros(self.hist.shape[1], dtype=torch.int64, device=self.device)
        shape.index_add_(0, bins, active.to(torch.int64))
        rows = (ids.reshape(ids.shape[0], -1)[:, 0].to(torch.int64) < experts).sum().view(1)
        at = self.kind + slot
        self.acc.index_add_(0, at, torch.cat((self.one, distinct, rows, peak)).view(1, len(EXPERT_FIELDS)))
        self.hist.index_add_(0, at, shape.view(1, -1))
        self.peak.scatter_reduce_(0, at, peak, "amax")
        self.low.scatter_reduce_(0, at, distinct, "amin")
        self.high.scatter_reduce_(0, at, distinct, "amax")

    def step(self, scheduler_output):
        kind = 0 if step_shape(scheduler_output)["prefill"] == 0 else 1
        if not self.started:
            self.started = True
            self.reset()
        if kind != self.current:
            self.kind.fill_(kind * self.count)
            self.current = kind
        self.poll()
        self.steps[kind] += 1
        self.total += 1

    def reset(self):
        self.acc.zero_()
        self.low.fill_(EXPERT_FLOOR)
        self.high.zero_()
        self.peak.zero_()
        self.hist.zero_()

    def poll(self):
        if self.pending is not None and self.pending[0].query():
            self.report(self.pending[1])
            self.pending = None
        if self.pending is None and self.total and self.total % self.every == 0:
            for host, held in zip(self.host, (self.acc, self.low, self.high, self.peak, self.hist)):
                host.copy_(held, non_blocking=True)
            self.reset()
            event = None
            if self.device.type == "cuda":
                event = torch.cuda.Event()
                event.record()
            window, self.steps = list(self.steps), [0] * len(EXPERT_KINDS)
            self.pending = (event or _Done(), window)

    def report(self, window):
        acc, low, high, peak, hist = (held.numpy().copy() for held in self.host)
        self.lines += 1
        lines = expert_lines(self.layers, acc, low, high, window, self.lines, peak, hist)
        for line in lines:
            expert_say(line)
        return lines


def expert_lines(layers, acc, low, high, window, index, peak=None, hist=None):
    count = len(layers)
    out = []
    for kind, name in enumerate(EXPERT_KINDS):
        part = slice(kind * count, (kind + 1) * count)
        rows, lows, highs = acc[part], low[part], high[part]
        for group in ("target", "draft"):
            chosen = [at for at, layer in enumerate(layers) if layer[2] == group and rows[at, 0] > 0]
            if not chosen:
                continue
            calls = rows[chosen, 0].astype(np.float64)
            means = rows[chosen, 1] / calls
            tokens = rows[chosen, 2] / calls
            experts = layers[chosen[0]][3]
            out.append(f"{name} {group} window={index} steps={window[kind]} layers={len(chosen)} experts={experts} "
                       f"top_k={layers[chosen[0]][4]} forwards_per_layer={calls.mean():.1f} distinct mean/min/max "
                       f"{means.mean():.1f}/{int(lows[chosen].min())}/{int(highs[chosen].max())} "
                       f"layer_mean min/max {means.min():.1f}/{means.max():.1f} tokens_per_forward "
                       f"{tokens.mean():.1f} share_of_experts {means.mean() / max(experts, 1):.3f}"
                       + expert_load(rows[chosen, 3] / calls, peak, hist, part, chosen, means))
            out.append(f"{name} {group} window={index} per_layer_distinct_mean "
                       + " ".join(f"{int(round(value))}" for value in means))
    return out


def expert_load(peaks, peak, hist, part, chosen, means):
    if peak is None or hist is None:
        return ""
    edges = EXPERT_HIST_EDGES
    names = [f"{edges[0]}"] + [f"{edges[i]}" if edges[i] == edges[i - 1] + 1 else f"{edges[i - 1] + 1}-{edges[i]}"
                              for i in range(1, len(edges))] + [f">{edges[-1]}"]
    total = hist[part][chosen].sum(axis=0).astype(np.float64)
    active = max(float(total.sum()), 1.0)
    cells = " ".join(f"{name}:{share:.3f}" for name, share in zip(names, total / active))
    return (f" max_rows_per_expert mean/max {peaks.mean():.1f}/{int(peak[part][chosen].max())}"
            f" rows_per_active_expert_hist {cells}")


def expert_wrap(tally, slot, experts, top_k):
    def modular(original):
        def wrapped(*args, **kwargs):
            ids = kwargs.get("topk_ids")
            if ids is None and len(args) > 2:
                ids = args[2]
            if isinstance(ids, torch.Tensor) and ids.dim() >= 1 and ids.shape[0] > 0:
                tally.note(slot, ids, experts)
            return original(*args, **kwargs)
        return wrapped

    def monolithic(original):
        def wrapped(*args, **kwargs):
            logits = kwargs.get("router_logits")
            if logits is None and len(args) > 1:
                logits = args[1]
            if isinstance(logits, torch.Tensor) and logits.dim() == 2 and logits.shape[0] > 0 and top_k > 0:
                tally.note(slot, torch.topk(logits, top_k, dim=-1).indices, experts)
            return original(*args, **kwargs)
        return wrapped

    return modular, monolithic


def install_experts(runner):
    model = runner.model
    layers = moe_layers(model, "target")
    draft = runner.get_draft_model() if hasattr(runner, "get_draft_model") else None
    if draft is not None and draft is not model:
        layers += moe_layers(draft, "draft")
    if not layers:
        expert_say("found no MoE layer with routed experts; nothing is tallied")
        return None
    said = str(os.environ.get(EXPERT_EVERY_ENV) or "").strip()
    tally = Experts(runner.device, layers, int(said) if said.isdigit() else EXPERT_EVERY)
    for slot, (module, _, _, experts, top_k) in enumerate(layers):
        modular, monolithic = expert_wrap(tally, slot, experts, top_k)
        module.forward_modular = modular(_resolve(module, "forward_modular"))
        module.forward_monolithic = monolithic(_resolve(module, "forward_monolithic"))
    execute = _resolve(runner, "execute_model")

    def execute_model(scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if not dummy and getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            tally.step(scheduler_output)
        return execute(scheduler_output, *args, **kwargs)

    runner.execute_model = execute_model
    groups = sorted({layer[2] for layer in layers})
    expert_say(f"armed on {len(layers)} MoE layers ({', '.join(groups)}): each forward marks the experts its rows "
               f"route to in a {tally.flags.numel()}-entry device bitmap, counts the marks on the GPU and adds the "
               f"count, the row count and the per-layer min and max into accumulators split by decode and mixed "
               f"steps; captured into every CUDA graph, read by a non-blocking copy and printed every {tally.every} "
               f"steps")
    return tally
