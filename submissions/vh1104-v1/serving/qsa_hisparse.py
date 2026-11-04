import atexit
import contextlib
import copy
import dataclasses
import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if __import__("os").environ.get("TAAF_VLLM_MARKS", "").strip() == "1":
    _marked = __import__("importlib").import_module("vllm.taaf_marks").marked
else:
    def _marked(site):
        return lambda fn: fn

_ANNOUNCED = set()
_TABLES = {}
_TALLY = {"decode_rows": 0, "prefill_rows": 0, "staged_blocks": 0, "forwards": 0, "gated_writes": 0,
          "draft_mirrors": 0, "row_attend_steps": 0, "sparse_rows_needed": 0, "sparse_rows_total": 0,
          "last_rows_outside": 0, "sources_fused": 0, "sources_stock": 0, "mirror_misses": 0,
          "draft_through": 0, "reuse_rows_kept": 0, "reuse_rows_copied": 0, "block_stage_blocks": 0,
          "stage_ahead_rows": 0, "hits_shadow_calls": 0, "hits_shadow_bad": 0, "hits_shadow_err_ppm": 0,
          "hits_shadow_draft_calls": 0, "hits_shadow_draft_bad": 0}
HITS_SHADOW_BANDS = ("mixed", "pure_lt16", "pure_lt32", "pure_lt64", "pure_ge64")
_TALLY.update({f"hits_shadow_{band}_{kind}": 0 for band in HITS_SHADOW_BANDS for kind in ("calls", "bad")})
REPORT_EVERY = 2000
REPORT_STEPS = 500
GUARD_ENV = "TAAF_VLLM_HOST_KV_GUARD"
DEVICE_TALLY_ENV = "TAAF_VLLM_HOST_KV_DEVICE_TALLY"
_DEVICE_TALLY_ON = [None]
_GUARD = {"tally": {}, "seen": {}, "exit": False}
_STEP = {"open": False}
WORKER_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker"
WORKER_CLASS = "HiSparseConnectorWorker"
GATE_MARK = "_taaf_step_gate"


def varlen_decode():
    return os.environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1"


EXPAND_ENV = "TAAF_VLLM_DRAFT_BUDGET_EXPAND"
_EXPAND = {"inside": False}


def varlen_decode_only(meta):
    if _EXPAND["inside"]:
        return False
    return varlen_decode() and int(meta.num_decode_tokens) == int(getattr(meta, "num_actual_tokens", meta.num_decode_tokens))


def expand_on():
    return str(os.environ.get(EXPAND_ENV) or "").strip().lower() in ("1", "on", "true")


def expanded_layout(meta, rows, device):
    decodes = int(meta.num_decodes)
    width = int(meta.decode_max_query_len)
    lane = meta.req_id_per_token[:rows].to(torch.int64)
    start = meta.query_start_loc[:decodes].to(torch.int64)
    place = lane * width + (torch.arange(rows, dtype=torch.int64, device=device) - start.index_select(0, lane))
    slots = decodes * width
    view = SimpleNamespace(**vars(meta))
    view.num_decode_tokens = slots
    view.num_actual_tokens = slots
    view.max_query_len = width
    view.query_start_loc = torch.arange(0, slots + 1, width, dtype=meta.query_start_loc.dtype, device=device)
    view.req_id_per_token = torch.div(torch.arange(slots, device=device), width, rounding_mode="floor").to(
        meta.req_id_per_token.dtype)
    return place, slots, view


def spread_rows(source, place, slots, fill):
    landed = torch.full((slots,) + tuple(source.shape[1:]), fill, dtype=source.dtype, device=source.device)
    landed.index_copy_(0, place, source)
    return landed
SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.speculator"
SPECULATOR_CLASS = "DraftModelSpeculator"
DRAFT_MARK = "_taaf_draft_tags"
_DRAFT = set()
BATCH_FIELDS = ("num_actual_tokens", "num_decode_tokens", "max_query_len", "num_reqs", "req_id_per_token")
RUNTIME_MODULE = "vllm.v1.hisparse.runtime"
RUNTIME_CLASS = "HiSparseRuntime"
SPLIT_MARK = "_taaf_split_gather"
SPLIT_ENV = "TAAF_VLLM_HOST_KV_GATHER_SPLIT"
SPLIT_TARGET_ROWS = 576
SPLIT_MIN_COLUMNS = 64
STAGE_ENV = "TAAF_VLLM_HOST_KV_STAGE_MIB"
STAGE_LONGEST = 2
STAGE_NULL_BLOCKS = 1
STAGE_COPY_ROWS = 65536
_STAGE = {"buffer": None, "device": None}
_REFUSED = {"needed": [], "seqs": [], "last": 0.0}
REFUSED_REPORT_S = 60.0
STAGE_REUSE_ENV = "TAAF_VLLM_QSA_STAGE_REUSE"
STAGE_REUSE_MIB_ENV = "TAAF_VLLM_QSA_STAGE_REUSE_MIB"
STAGE_REUSE_MODES = ("reuse", "blocks")
STAGE_REUSE_LIVE_CHECKS = 3
STAGE_BLOCK_SHARE = 0.5
_REUSE = {"layers": {}, "aside": None, "checked": False, "bytes": 0, "live": 0}
STAGE_AHEAD_ENV = "TAAF_VLLM_QSA_STAGE_AHEAD"
STAGE_AHEAD_PROGRAMS_ENV = "TAAF_VLLM_QSA_STAGE_AHEAD_PROGRAMS"
STAGE_AHEAD_PROGRAMS = 64
STAGE_AHEAD_PROGRAMS_MAX = 4096
_AHEAD = {"stream": None, "events": {}, "armed": {}, "counts": None, "rows": None, "warm": set(), "aside": "",
          "checked": False, "hosts": False, "issued": 0, "taken": 0, "skipped": {}}
ROW_ATTEND_ENV = "TAAF_VLLM_HOST_KV_ROW_ATTEND"
PICK_LOG_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG"
FRESH_FIX_ENV = "TAAF_VLLM_HOST_KV_FRESH_FIX"
DRAFT_STOCK_ENV = "TAAF_VLLM_HOST_KV_DRAFT_STOCK"
DRAFT_DEFER_ENV = "TAAF_VLLM_HOST_KV_DRAFT_DEFER"
DRAFT_THROUGH_ENV = "TAAF_VLLM_HOST_KV_DRAFT_THROUGH"
SINGLE_FUSED_ENV = "TAAF_VLLM_HOST_KV_SINGLE_FUSED"
PICK_EVERY_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_EVERY"
PICK_DIR_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_DIR"
PICK_DIR = os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "pick-log")
PICK_EVERY = 400
PICK_REQS = 16
PICK_BURST_STEPS = 2
PICK_MAX_BYTES = 512 << 20
PICK_AT_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_AT"
PICK_STEPS_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_STEPS"
PICK_LANES_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_REQS"
PICK_MIB_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_MIB"
STREAK_STEPS = 48
STREAK_LANES = 8
STREAK_MIB = 768
STREAK_BASELINE = 64
_STREAK = {"plan": None, "clock": None, "engine": 0, "seen": None, "forced": False, "next": 0, "open": None,
           "slots": {}, "elapsed": 0.0, "bytes": 0, "ids": {}, "names": [], "baseline": [], "copies": None}
TAIL_ENV = "TAAF_VLLM_HOST_KV_TAIL_PREFETCH"
SPARSE_ENV = "TAAF_VLLM_HOST_KV_SPARSE_STAGE"
SPARSE_QUERY_CHUNK = 1024
UNION_SWAP_ENV = "TAAF_VLLM_HOST_KV_UNION_SWAP"
UNION_SMEM_FALLBACK = 48 << 10
_UNION_SWAP = {"events": {}}
HOT_ROWS_ENV = "TAAF_VLLM_HOST_KV_HOT_ROWS"
_HOT_CAP = {"tally": None, "calls": 0, "spills": 0, "starts": 0, "lines": 0, "cap": 0}
HOT_CAP_EVERY = 500
SPILL_SKIP_ENV = "TAAF_VLLM_HOST_KV_SPILL_SKIP"
SPILL_ONESORT_ENV = "TAAF_VLLM_HOST_KV_SPILL_ONESORT"
MARK_FILL_ENV = "TAAF_VLLM_HOST_KV_MARK_FILL"
GATHER_EARLY_ENV = "TAAF_VLLM_HOST_KV_GATHER_EARLY"
GATHER_EARLY_CAPTURED_ENV = "TAAF_VLLM_HOST_KV_GATHER_EARLY_CAPTURED"
QSA_HALVES_ENV = "TAAF_VLLM_QSA_HALVES"
_QSA_HALVES = {"streams": {}, "split": 0}
HOT_SPLIT_ENV = "TAAF_VLLM_HOST_KV_HOT_SPLIT"
COPY_ASYNC_ENV = "TAAF_VLLM_HISPARSE_COPY_ASYNC"
_COPY = {"aliases": {}, "steps": 0, "blocks": 0, "fallbacks": 0}
_PICKS = {"steps": 0, "first": None, "bursts": 0, "bytes": 0, "records": None, "burst_end": 0, "skipped": 0}
WIDE_ENV = "TAAF_VLLM_HOST_KV_INDEXER_WIDE"
WIDE_FACTOR = 2
WIDE_SHOWN_TOKENS = 85000
UNION_ENV = "TAAF_VLLM_HOST_KV_UNION_COUNT"
UNION_EVERY = 500
_UNION = {"hist": None, "picks": None, "starts": 0, "width": 0, "query": 0, "lines": 0}
UNION_TALLY_ENV = "TAAF_VLLM_HOST_KV_UNION_TALLY"
UNION_TALLY_EVERY = 500
HIT_BUCKETS = 10
UNION_SUMS = ("lanes", "unique", "fit_lanes", "fit_unique", "fit_misses", "calls")
_UNION_TALLY = {"hist": None, "hits": None, "sums": None, "misses": None, "armed": None, "starts": 0, "lines": 0,
                "need": 0, "bound": 0, "query": 0, "top_k": 0}
_ORDER = []
_COPY_STREAMS = {"done": False, "groups": set()}
COPY_STREAM_TRIES = 64
_ARANGE = {}
PLAN_ONCE_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE"
MASK_OWNERS_ENV = "TAAF_VLLM_HISPARSE_MASK_OWNERS"
PLAN_ONCE_CAPTURED_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE_CAPTURED"
PLAN_ONCE_CHECKS_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE_CHECKS"
PLAN_ONCE_CHECKS = 4
PLAN_PRINT_SLOTS = 256
PLAN_PRINT_PRIME = 1000003
SYNCFREE_ENV = "TAAF_VLLM_HISPARSE_SYNCFREE"
_PLAN = {"key": None, "meta": None, "built": 0, "reused": 0}
_DEVICE_TALLY = {"sparse_rows_needed": None}
_TALLY_DUE = {"layer": None, "last": None, "starts": 0}
FUSED_ENV = "TAAF_VLLM_HOST_KV_FUSED"
FUSED_MODULE = "vllm.taaf_host_kv_fused"
_FUSED = {"module": None, "owners": None, "kept": None, "warm": set(), "calls": 0, "aside": 0}
UNION_BITMAP_ENV = "TAAF_VLLM_HOST_KV_UNION_BITMAP"
SPILL_FLAG_ENV = "TAAF_VLLM_HOST_KV_SPILL_FLAG"
SPILL_TRUST_ENV = "TAAF_VLLM_HOST_KV_SPILL_TRUST"
GATHER_TALLY_ENV = "TAAF_VLLM_HOST_KV_GATHER_TALLY"
LAST_SLICE_ENV = "TAAF_VLLM_HOST_KV_LAST_SLICE"
SOURCES_FUSED_ENV = "TAAF_VLLM_HOST_KV_SOURCES_FUSED"
GATHER_EVERY = 100
PCIE_TALLY_ENV = "TAAF_VLLM_HOST_KV_PCIE_TALLY"
RUN_MERGE_ENV = "TAAF_VLLM_HOST_KV_RUN_MERGE"
PCIE_SITES = ("demand", "prefetch")
_PCIE = {"acc": None, "clock": None, "starts": 0, "steps": 0, "lines": 0, "row_bytes": 0, "gap": -1, "group": 0,
         "run_said": False, "run_refused": set(), "run_calls": 0}
GROUP_ENV = "TAAF_VLLM_HOST_KV_GROUP"
_GROUP = {"said": False, "refused": set(), "calls": 0}
GATHER_SPREAD_ENV = "TAAF_VLLM_HOST_KV_GATHER_SPREAD"
PLAN_FUSED_ENV = "TAAF_VLLM_HOST_KV_PLAN_FUSED"
PLAN_AHEAD_ENV = "TAAF_VLLM_HOST_KV_PLAN_AHEAD"
GATHER_PROBE_ENV = "TAAF_VLLM_HOST_KV_GATHER_PROBE"
SPREAD_FALLBACK_SMS = 1
_PLAN_FUSED = {"said": False, "calls": 0}
_SPREAD = {"refused": set(), "cold": set(), "calls": 0, "sms": {}}
_GATHER = {"tally": None, "starts": 0, "steps": 0, "lines": 0, "row_bytes": 0, "calls": 0}
MIXED_CAPTURE_ENV = "TAAF_VLLM_HOST_KV_MIXED_CAPTURE"
MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
MIRROR_ENTRIES_ENV = "TAAF_VLLM_HOST_KV_MIRROR_ENTRIES"
MIRROR_RING = 4
MIXED_FULL_MODULE = "vllm.taaf_mixed_full"
_FIXED = {"steps": 0, "aside": 0}
_GRAPH = {"table": None, "views": None, "pinned": [], "events": [], "next": 0, "entries": 0, "sources": 0,
          "layers": {}, "ok": None, "tally": None, "waits": 0, "staged": 0, "aside": 0, "rows": 0, "bad": 0,
          "rids": {}, "snapshot": None, "snapshot_tokens": 0, "drafted": False}
QSA_CACHE_MODULE = "vllm.models.qwen4_exp.common.qsa_cache"
QSA_BUILDER_CLASS = "QSAMetadataBuilder"
MIRROR_MARK = "_taaf_prefill_mirror"
MIRROR_FIELD = "taaf_prefill_host"
_MIRROR = {"used": 0, "absent": 0, "over": 0, "clamped": 0}
PREFETCH_ENV = "TAAF_VLLM_HOST_KV_PREFETCH"
PREFETCH_PROGRAMS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PROGRAMS"
RANKS_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_RANKS"
INDEXER_OPS_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
NEAR_MARK = "_taaf_near_select"
PREFETCH_EVERY = 100
_NEAR = {"layer": None, "pure": False, "decode": False, "buffers": {}, "ready": {}, "ranks": {}, "selects": 0}
_PREFETCH = {"stream": None, "pending": None, "open": False, "scratch": None, "tally": None, "stats": None,
             "issued": 0, "aside": 0, "starts": 0, "steps": 0, "lines": 0}
PREV_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV"
SPEC_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC"
TAKE_ROWS_ENV = "TAAF_VLLM_HOST_KV_TAKE_ROWS"
TAKE_SHADOW_ENV = "TAAF_VLLM_HOST_KV_TAKE_ROWS_SHADOW"
TAKE_SHADOW_SITES = ("output", "logical", "states", "blocks")
TAKE_SHADOW_TALLIES = tuple(f"take_shadow_{site}_{kind}" for site in TAKE_SHADOW_SITES
                            for kind in ("calls", "bad", "rows", "differ"))
TAKE_SHADOW_EVERY_S = 60.0
_TAKE_SHADOW = {"max": None, "layer": None, "told": None, "max_seen": 0.0, "layer_seen": -1}
_TALLY.update({name: 0 for name in TAKE_SHADOW_TALLIES})
PREV_ROWS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_ROWS"
PREV_SCOPE_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_SCOPE"
PREV_ROWS = 1024
PREV_SCOPES = ("missed", "picks")
PREV_STATS = ("lane_steps", "candidates", "resident", "missed", "scoped", "kept", "fetched", "used", "issues")
_SPEC = {"buffers": None}
_PREV = {"tally": None, "pending": None, "capture": None, "issued": 0, "aside": 0, "starts": 0, "steps": 0,
         "lines": 0}
PREV_ONCE_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_ONCE"
PREV_CAPS = 4
_PREV_ONCE = {"buffers": None, "layouts": {}}
ONE_LAUNCH_ENV = "TAAF_VLLM_HOST_KV_ONE_LAUNCH"
ONE_PACK_ENV = "TAAF_VLLM_HOST_KV_ONE_PACK"
ONE_HASH_ENV = "TAAF_VLLM_HOST_KV_ONE_HASH"
HITS_FIRST_ENV = "TAAF_VLLM_HOST_KV_HITS_FIRST"
HITS_FIRST_PROGRAMS_ENV = "TAAF_VLLM_HOST_KV_HITS_FIRST_PROGRAMS"
HITS_FIRST_PROGRAMS = 64
HITS_FIRST_CHECKS = 3
HITS_FIRST_TOLERANCE = 1.0e-2
_HITS = {"aside": "", "checked": 0, "events": {}, "fresh": {}, "retired": [], "taken": False}
HITS_SHADOW_ENV = "TAAF_VLLM_HOST_KV_HITS_SHADOW"
HITS_SHADOW_TALLIES = ("hits_shadow_calls", "hits_shadow_bad", "hits_shadow_err_ppm", "hits_shadow_draft_calls",
                       "hits_shadow_draft_bad") + tuple(f"hits_shadow_{band}_{kind}" for band in HITS_SHADOW_BANDS
                                                        for kind in ("calls", "bad"))
HITS_PURE_ENV = "TAAF_VLLM_HOST_KV_HITS_PURE"
HITS_PROBE_ENV = "TAAF_VLLM_HOST_KV_HITS_PROBE"
HITS_DRAFT_ENV = "TAAF_VLLM_HOST_KV_HITS_DRAFT"
HITS_SHADOW_EVERY_ENV = "TAAF_VLLM_HOST_KV_HITS_SHADOW_EVERY"
HITS_PROBE_LAYERS = 16
HITS_PROBE_BYTES = 16
HITS_PROBE_TALLIES = (("hits_probe_hit_entries", "hits_probe_moved", "hits_probe_late_bad", "hits_probe_drift_bad")
                      + tuple(f"hits_probe_{band}_{kind}" for band in ("copy", "nocopy") for kind in ("calls", "bad"))
                      + tuple(f"hits_probe_l{at}_{kind}" for at in range(HITS_PROBE_LAYERS) for kind in ("calls", "bad")))
_TALLY.update({name: 0 for name in HITS_PROBE_TALLIES})
FLAT_COPY_ENV = "TAAF_VLLM_HOST_KV_FLAT_COPY"
_FLAT_COPY = {"aside": None}
AUDIT_ENV = "TAAF_VLLM_HOST_KV_AUDIT"
DRAFT_COLD_ENV = "TAAF_VLLM_HOST_KV_DRAFT_COLD"
_DRAFT_COLD = {"checked": 0.0, "on": False, "resets": 0}
AUDIT_LANES = 16
AUDIT_MIN_LANES = 8
AUDIT_MIN_CONTEXT = 4096
AUDIT_REL = 1e-2
_AUDIT = {"left": None, "start": None, "done": 0, "draft_step": 0, "seen": None}
_ONE = {"scratch": None, "layouts": {}, "mask": None, "launches": 0, "parts": 0, "aside": 0, "probed": set(),
        "refused": set(), "skip_aside": None, "skip_checked": False, "phases": None, "phases_aside": None,
        "phase_steps": 0, "phase_lines": 0, "pack_aside": None, "hash_aside": None, "hash_checked": False}
HIT_SKIP_ENV = "TAAF_VLLM_HOST_KV_HIT_SKIP"
STAGE_TILE_ENV = "TAAF_VLLM_HOST_KV_STAGE_TILE"
STEP_TIMERS_ENV = "TAAF_VLLM_STEP_TIMERS"
_STAGE_TILE = {"aside": None, "checked": False, "said": False}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def announce(event, **fields):
    if event in _ANNOUNCED:
        return
    _ANNOUNCED.add(event)
    detail = " ".join(f"{name}={value}" for name, value in fields.items())
    print(f"QSA_HISPARSE {event} {detail}".rstrip(), file=sys.stderr, flush=True)


def armed(vllm_config):
    attention = getattr(vllm_config, "attention_config", None)
    return attention is not None and getattr(attention, "hisparse_config", None) is not None


def index_topk(budget, compress_ratio):
    return int(budget) + int(compress_ratio) - 1


def storage_dtype(kv_cache_dtype):
    if kv_cache_dtype == "int4_per_token_head":
        return torch.uint8
    if kv_cache_dtype == "int8_per_token_head":
        return torch.int8
    raise ValueError(f"host KV serves the int4 or int8 per-token-head QSA cache, not {kv_cache_dtype!r}")


def row_of(layer):
    from vllm.taaf_qsa_kv8 import content_bytes, layout, widths

    storage_dtype(layer.kv_cache_dtype)
    return int(layer.num_kv_heads) * content_bytes(*widths(layout(layer.kv_cache_dtype), int(layer.head_dim)))


def draft_stock():
    return str(os.environ.get(DRAFT_STOCK_ENV) or "").strip().lower() not in ("0", "off", "false")


def draft_defer():
    return str(os.environ.get(DRAFT_DEFER_ENV) or "").strip().lower() in ("1", "on", "true")


def draft_through():
    return knob_on(DRAFT_THROUGH_ENV)


def single_fused():
    return knob_on(SINGLE_FUSED_ENV)


def host_rows_view(host, device):
    if host is None or host.dim() != 2 or not host.is_contiguous():
        return None
    base = host if host.device == device else device_alias(host, device).view(host.dtype)
    return base.view(host.shape[0], 1, host.shape[1])


def through_write(layer, impl, key, value, cache, rows, quantized_write):
    slots = getattr(cache, "mirror_slot_mapping", None)
    if rows <= 0 or slots is None or getattr(cache, "dummy_batch", False):
        return 0
    runtime = cache.runtime
    view = host_rows_view(getattr(runtime, "host_cache", None), key.device)
    if view is None:
        announce("draft_through_aside", layer=layer.layer_name, knob=DRAFT_THROUGH_ENV,
                 why="the host source cache is not one contiguous row table, so draft rows stay GPU-only")
        return 0
    wanted = slots[:rows]
    safe = wanted.masked_fill((wanted < 0) | (wanted >= view.shape[0]), -1)
    quantized_write(impl, key[:rows], value[:rows], planes(view, heads_of(layer)), safe)
    owners = getattr(cache, "req_id_per_token", None)
    if owners is not None and runtime.is_group_leader:
        runtime.invalidate_written_slots(safe, owners[:rows])
    _TALLY["draft_through"] += 1
    announce("draft_through", layer=layer.layer_name, knob=DRAFT_THROUGH_ENV, rows=rows, host_rows=view.shape[0],
             why="a draft forward writes its rows to the host source in the same pass as the GPU page and drops "
                 "the lane's hot copies of those host rows, so a pick served through the host or the hot region "
                 "reads what the drafter wrote, in the drafter's own later passes as in the next step")
    return rows


def draft_name(name):
    return str(name).startswith("mtp.")


def main_spec(layer, vllm_config, spec):
    if not armed(vllm_config):
        return spec
    if draft_stock() and draft_name(layer.layer_name):
        announce("draft_stock", layer=layer.layer_name, knob=DRAFT_STOCK_ENV, block=spec.block_size,
                 why="the draft layer keeps the stock full-attention cache in the regular groups, so only the "
                     "target layers read their rows through host KV")
        return spec
    from vllm.taaf_qsa_kv8 import customize_spec
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

    quantized = customize_spec(spec)
    sparse = MLAAttentionSpec(
        block_size=quantized.block_size,
        num_kv_heads=1,
        head_size=row_of(layer),
        dtype=storage_dtype(layer.kv_cache_dtype),
        cache_role=SparseCacheRole.SPARSE,
        is_index_group_leader=True,
    )
    if sparse.page_size_bytes != quantized.page_size_bytes:
        raise ValueError(f"host KV row page {sparse.page_size_bytes} differs from the QSA page "
                         f"{quantized.page_size_bytes}")
    announce("main_spec", layer=layer.layer_name, row=sparse.head_size, page=sparse.page_size_bytes,
             block=sparse.block_size)
    fit_stage(vllm_config, sparse.block_size, sparse.head_size)
    reserve_reuse(layer.layer_name, sparse.block_size, sparse.head_size)
    return sparse


def indexer_wide():
    said = str(os.environ.get(WIDE_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{WIDE_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def indexer_spec(vllm_config, spec):
    if not armed(vllm_config):
        return spec
    from vllm.v1.kv_cache_interface import SparseCacheRole

    if not indexer_wide():
        return dataclasses.replace(spec, cache_role=SparseCacheRole.INDEXER)
    block = int(spec.block_size)
    wide = dataclasses.replace(spec, cache_role=SparseCacheRole.INDEXER, block_size=block * WIDE_FACTOR)
    announce("indexer_wide", knob=WIDE_ENV, block=wide.block_size, source_block=block, page=wide.page_size_bytes,
             lane_blocks_85k=f"{-(-WIDE_SHOWN_TOKENS // wide.block_size)}/{-(-WIDE_SHOWN_TOKENS // block)}",
             why="the indexer group holds its compressed keys on blocks of twice the source block, so each pool "
                 "block id it takes is filled about twice as far")
    return wide


def ring_capacity(ratio, span, block_size):
    stock = int(ratio) * -(-int(span) // int(ratio))
    capacity = stock
    while int(block_size) % capacity and capacity < int(block_size):
        capacity += int(ratio)
    if capacity != stock:
        announce("ring_fit", ratio=ratio, span=span, block=block_size, stock=stock, capacity=capacity,
                 why="the open group's ring holds the committed keys plus every speculative row, rounded up to whole "
                     "groups; the stock size does not divide the attention block, so the ring takes the next whole "
                     "number of groups that does, which still holds every row a rejected draft could write")
    return capacity


def is_indexer(spec):
    role = getattr(spec, "cache_role", None)
    return getattr(role, "name", role) == "INDEXER"


def mla_group_spec(uniform, specs):
    joined = uniform.from_specs(specs)
    if joined is not None or not indexer_wide():
        return joined
    source = {name: spec for name, spec in specs.items() if not is_indexer(spec)}
    indexer = {name: spec for name, spec in specs.items() if is_indexer(spec)}
    if not source or not indexer or uniform.from_specs(source) is None or uniform.from_specs(indexer) is None:
        return None
    return uniform(block_size=next(iter(source.values())).block_size, kv_cache_specs=specs)


def gpu_block_size(source_specs, indexer_specs):
    sizes = {spec.block_size for spec in (*source_specs.values(), *indexer_specs.values())}
    if len(sizes) == 1:
        return sizes.pop()
    if not indexer_wide():
        raise ValueError("HiSparse requires one resolved GPU block size.")
    source = {spec.block_size for spec in source_specs.values()}
    indexer = {spec.block_size for spec in indexer_specs.values()}
    if len(source) != 1 or len(indexer) != 1 or next(iter(indexer)) % next(iter(source)):
        raise ValueError(f"HiSparse with {WIDE_ENV} needs one source block size and one indexer block size that "
                         f"is a whole multiple of it, got source {sorted(source)} and indexer {sorted(indexer)}.")
    block = source.pop()
    announce("indexer_layout", gpu_block=block, indexer_block=indexer.pop(),
             rule="the hot, resident and host pages keep the source block; only the indexer group is wider")
    return block


def resolved_block_sizes(mla_specs, backend_of, select):
    sizes = {spec.block_size for spec in mla_specs.values()}
    if len(sizes) != 1 and not indexer_wide():
        raise ValueError("HiSparse requires one scheduler block size.")
    resolved = {}
    for size in sorted(sizes):
        names = [name for name, spec in mla_specs.items() if spec.block_size == size]
        backends = [backend_of(name) for name in names]
        try:
            block_size = select(size, backends)
        except ValueError as error:
            raise ValueError(
                "HiSparse requires a GPU block size supported by every sparse "
                f"attention and indexer backend: {error}"
            ) from error
        resolved.update((name, mla_specs[name].copy_with_new_block_size(block_size)) for name in names)
    return resolved


def layer_page(group, name):
    spec = group.kv_cache_spec
    per_layer = getattr(spec, "kv_cache_specs", None)
    return (spec if per_layer is None else per_layer[name]).page_size_bytes


def hot_pack_limit(groups, indexer_page):
    widest = max((sum(layer_page(group, name) for name in group.layer_names) for group in groups),
                 default=0)
    return max(int(indexer_page), int(widest))


def tag_drafts(speculator_class):
    if not isinstance(speculator_class, type) or getattr(speculator_class, DRAFT_MARK, False):
        return False
    load_model = speculator_class.load_model

    def tagged_load_model(self, *args, **kwargs):
        result = load_model(self, *args, **kwargs)
        _DRAFT.update(getattr(self, "draft_attn_layer_names", None) or ())
        return result

    speculator_class.load_model = tagged_load_model
    setattr(speculator_class, DRAFT_MARK, True)
    return True


def draft_layers(worker):
    known = getattr(worker, "_taaf_draft_layers", None)
    if known is not None:
        return known
    names = getattr(worker, "cache_layer_names", None)
    if not names or not _DRAFT:
        return ()
    known = tuple(index for index, name in enumerate(names) if name in _DRAFT)
    worker._taaf_draft_layers = known
    for index in known:
        worker.cache_handles[index].submit_layer_mirror = None
    if known:
        announce("draft_layers", layers=len(known), first=names[known[0]],
                 rule="draft-layer rows go to host at the next start_step, after the drafter wrote them, "
                      "and the step's post-forward page transfers are submitted behind them")
    return known


def fresh_fix():
    return str(os.environ.get(FRESH_FIX_ENV) or "").strip().lower() in ("1", "on", "true")


def written_drafts(worker):
    if not worker.is_host_writer:
        return ()
    handles = worker.cache_handles
    if fresh_fix() or draft_defer():
        drafts = draft_layers(worker)
        targets = [handle for index, handle in enumerate(handles) if index not in drafts]
        if not drafts or not getattr(worker, "_row_mirror_num_rows", 0):
            return ()
        if not any(handle.num_actual_tokens != 0 and handle.host_mirror_required for handle in targets):
            return ()
        announce("draft_fresh_fix", layers=len(drafts),
                 why="the drafter's own handle holds its last decode forward, which never asks for a host mirror, "
                     "so the draft layer is mirrored on the target step's row mirrors once the drafter has written")
        return drafts
    active = tuple(index for index in draft_layers(worker)
                   if handles[index].num_actual_tokens != 0 and handles[index].host_mirror_required)
    if not active:
        return ()
    lead = handles[active[0]]
    slots = lead.mirror_slot_mapping
    if slots is None or min(lead.num_actual_tokens, slots.numel()) == 0:
        return ()
    return active


def flush_drafts(worker):
    pending = getattr(worker, "_taaf_draft_pending", ())
    held = getattr(worker, "_taaf_held_transfers", ())
    worker._taaf_draft_pending = ()
    worker._taaf_held_transfers = []
    if pending and graph_drafts(worker, pending):
        pending = ()
    if pending:
        worker._enqueue_row_dma(pending)
        _TALLY["draft_mirrors"] += 1
        announce("draft_mirror", layers=len(pending),
                 why="the previous step's drafter has run, so its rows are current on the GPU")
    if held:
        worker._submit_transfers(list(held))


def gate_worker(worker_class):
    if getattr(worker_class, GATE_MARK, False):
        return False
    start_step = worker_class.start_step
    finish_forward = worker_class.finish_forward
    enqueue_host_mirror = getattr(worker_class, "_enqueue_host_mirror", None)

    def gated_start_step(self, *args, **kwargs):
        flush_drafts(self)
        report_tally()
        if _RESET["fresh"] is not None:
            flush_resets()
            report_reset()
        if _READS["counts"] is not None:
            flush_reads()
        if _UNION["hist"] is not None:
            report_union()
        if _UNION_TALLY["hist"] is not None:
            report_union_tally()
        if _REC2["age"] is not None:
            report_rec2()
        if _HOT_CAP["tally"] is not None:
            report_hot_cap()
        if gather_tally_on():
            report_gather()
        if pcie_tally_level():
            report_pcie()
        if _ONE["phases"] is not None:
            report_one_phases()
        if _PREFETCH["tally"] is not None:
            report_prefetch()
        if prev_counted():
            report_prev()
        result = start_step(self, *args, **kwargs)
        _STEP["open"] = True
        _STEP["steps"] = _STEP.get("steps", 0) + 1
        return result

    def gated_finish_forward(self, *args, **kwargs):
        held = []
        if draft_layers(self):
            held = self._post_forward_transfers
            self._post_forward_transfers = []
        try:
            return finish_forward(self, *args, **kwargs)
        finally:
            _STEP["open"] = False
            check_shared()
            if held:
                self._taaf_held_transfers = [*getattr(self, "_taaf_held_transfers", ()), *held]

    def gated_enqueue_host_mirror(self, *args, **kwargs):
        drafts = written_drafts(self)
        if drafts:
            self._submitted_mirror_layers.update(drafts)
            written = {index for index in drafts if self.cache_handles[index].num_actual_tokens != 0}
            if self._per_layer_mirrored and written:
                self._per_layer_mirrored.update(written)
            self._taaf_draft_pending = drafts
        return enqueue_host_mirror(self, *args, **kwargs)

    worker_class.start_step = gated_start_step
    worker_class.finish_forward = gated_finish_forward
    if graph_mirror():
        graph_gate(worker_class)
        if draft_defer():
            enqueue_host_mirror = getattr(worker_class, "_enqueue_host_mirror", None)
            announce("draft_defer", knob=DRAFT_DEFER_ENV,
                     why="the gate wraps the graph mirror instead of replacing it, so graph steps defer the draft "
                         "layer's rows to the next step, after the drafter has written them")
    copy_host_blocks = getattr(worker_class, "_copy_host_blocks", None)
    if copy_host_blocks is not None and copy_async():
        def async_copy_host_blocks(self, host_block_copies, previous_host_write_event):
            if (not host_block_copies or getattr(self, "shared_host_region", None) is not None or not self.host_caches
                    or capturing()):
                return copy_host_blocks(self, host_block_copies, previous_host_write_event)
            device = self.cache_handles[0].runtime.device
            stream = current()
            stream.wait_event(previous_host_write_event)
            if not copy_blocks_on_device(self.host_caches, self.host_num_blocks, host_block_copies,
                                         lambda cache: device_alias(cache, device), device):
                return copy_host_blocks(self, host_block_copies, previous_host_write_event)
            self._completed_host_copy_dst_ids.extend(copy.dst_block_id for copy in host_block_copies)
            return None

        worker_class._copy_host_blocks = async_copy_host_blocks
    if enqueue_host_mirror is not None:
        worker_class._enqueue_host_mirror = gated_enqueue_host_mirror
    prepare_forward = getattr(worker_class, "prepare_forward", None)
    if prepare_forward is not None:
        def gated_prepare_forward(self, attn_metadata, *args, **kwargs):
            return prepare_forward(self, sparse_metadata(attn_metadata), *args, **kwargs)

        worker_class.prepare_forward = gated_prepare_forward
    setattr(worker_class, GATE_MARK, True)
    return True


def graph_mirror():
    return str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower() == "1"


def graph_capture_open():
    return bool(_GRAPH["ok"]) and capturing()


def graph_aside(reason):
    _GRAPH["ok"] = False
    announce("graph_mirror_aside", reason=reason.replace(" ", "_"),
             why="the host-KV layer mirror keeps its stream-side descriptor copies for every step")
    return False


def graph_prepare(worker):
    if not worker.is_host_writer:
        _GRAPH["ok"] = False
        return False
    if getattr(worker, "shared_host_region", None) is not None or not worker.host_caches:
        return graph_aside("the host pool is a shared region or absent, so a kernel cannot address it")
    if not fused():
        return graph_aside(f"{FUSED_ENV} is off, so the mirror kernel is not installed")
    handles = worker.cache_handles
    device = handles[0].runtime.device
    kernels = fused_ops()
    layers = {}
    sources = 1
    for index, cache in enumerate(handles):
        source = worker.resident_caches[index]
        host = worker.host_caches[index]
        row_bytes = source.shape[-1] * source.element_size()
        view = kernels.resident_rows(source, row_bytes)
        if (view is None or host.dim() != 2 or host.stride(1) != 1
                or host.stride(0) * host.element_size() != row_bytes):
            return graph_aside(f"layer {index} rows are not contiguous byte rows on both sides")
        alias = device_alias(host, device)
        layers[index] = (view, alias.narrow(0, 0, host.shape[0] * row_bytes), int(worker.kernel_block_size))
        sources = max(sources, int(cache.runtime.resident_source_index) + 1)
    entries = env_int(MIRROR_ENTRIES_ENV, 4096)
    size = 1 + (3 + sources) * entries
    table = torch.zeros(size, dtype=torch.int64, device=device)
    pinned = [torch.zeros(size, dtype=torch.int64).pin_memory() for _ in range(MIRROR_RING)]
    views = {"count": table[0:1], "ref": table[1:1 + entries], "counts": table[1 + entries:1 + 2 * entries],
             "dests": table[1 + 2 * entries:1 + 3 * entries],
             "layer": [table[1 + (3 + one) * entries:1 + (4 + one) * entries] for one in range(sources)]}
    _GRAPH.update(table=table, views=views, pinned=pinned, events=[None] * MIRROR_RING, next=0, entries=entries,
                  sources=sources, layers=layers, ok=True, tally=torch.zeros(2, dtype=torch.int64, device=device))
    announce("graph_mirror_ready", layers=len(layers), entries=entries, sources=sources, ring=MIRROR_RING,
             why="a mixed step mirrors the rows its forward wrote with one kernel a layer inside the forward, "
                 "driven by a device row count and a table of the scheduler's row runs that the host fills into "
                 "pinned memory and copies before the launch, so the forward needs no host callback and no wait")
    return True


def graph_fill(flat, mirrors, num_tokens, ref, entries, sources):
    flat[:] = 0
    flat[1:1 + entries] = fused_ops().MIRROR_EMPTY
    count = len(mirrors)
    if not count:
        return
    starts = np.asarray([mirror.source_starts for mirror in mirrors], dtype=np.int64).reshape(count, -1)
    order = np.argsort(starts[:, ref])
    flat[0] = num_tokens
    flat[1:1 + count] = starts[order, ref]
    flat[1 + entries:1 + entries + count] = np.fromiter((mirror.num_rows for mirror in mirrors), dtype=np.int64,
                                                       count=count)[order]
    flat[1 + 2 * entries:1 + 2 * entries + count] = np.fromiter(
        (mirror.destination_start for mirror in mirrors), dtype=np.int64, count=count)[order]
    for one in range(sources):
        base = 1 + (3 + one) * entries
        flat[base:base + count] = starts[order, one]


def graph_fits(worker, mirrors, num_tokens):
    entries = _GRAPH["entries"]
    sources = _GRAPH["sources"]
    if len(mirrors) > entries or num_tokens > worker.cache_handles[0].slot_mapping.numel():
        return False
    return all(len(mirror.source_starts) >= sources for mirror in mirrors)


def graph_stage(worker, mirrors, num_tokens):
    fits = graph_fits(worker, mirrors, num_tokens)
    if not fits:
        _GRAPH["aside"] += 1
        announce("graph_mirror_over", runs=len(mirrors), entries=_GRAPH["entries"], tokens=num_tokens,
                 why="this step's row runs or tokens exceed the mirror table, so its forward keeps the stream-side "
                     "layer mirror and the kernel finds no rows")
    slot = _GRAPH["next"]
    _GRAPH["next"] = (slot + 1) % MIRROR_RING
    held = _GRAPH["events"][slot]
    if held is not None and not held.query():
        _GRAPH["waits"] += 1
        held.synchronize()
    buffer = _GRAPH["pinned"][slot]
    graph_fill(buffer.numpy(), mirrors if fits else (), num_tokens, int(worker.cache_handles[0].runtime.resident_source_index),
               _GRAPH["entries"], _GRAPH["sources"])
    _GRAPH["table"].copy_(buffer, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    _GRAPH["events"][slot] = event
    _GRAPH["staged"] += 1
    return fits


def graph_launch(worker, layer_index, slots=None, tokens=None):
    handle = worker.cache_handles[layer_index]
    view, alias, kernel_block = _GRAPH["layers"][layer_index]
    views = _GRAPH["views"]
    if slots is None:
        slots = worker.cache_handles[0].slot_mapping
    tokens = min(int(handle.num_actual_tokens if tokens is None else tokens), int(slots.numel()))
    fused_ops().mirror_rows(slots, views["count"], views["ref"], views["counts"], views["dests"],
                            views["layer"][int(handle.runtime.resident_source_index)], view, alias, kernel_block,
                            _GRAPH["tally"], tokens)


def graph_snapshot(worker, num_tokens):
    slots = worker.cache_handles[0].slot_mapping
    held = _GRAPH.get("snapshot")
    if held is None or held.numel() != slots.numel() or held.device != slots.device:
        held = _GRAPH["snapshot"] = torch.empty_like(slots)
    held.copy_(slots)
    _GRAPH["snapshot_tokens"] = min(int(num_tokens), int(slots.numel()))


def graph_drafts(worker, pending):
    if not _GRAPH["ok"] or not _GRAPH.get("drafted") or _GRAPH.get("snapshot") is None:
        return False
    snapshot = _GRAPH["snapshot"]
    tokens = _GRAPH.get("snapshot_tokens", 0)
    for index in pending:
        graph_launch(worker, index, snapshot, tokens)
    _GRAPH["drafted"] = False
    _TALLY["draft_mirrors"] += 1
    announce("draft_mirror_kernel", layers=len(pending),
             why="the previous step's drafter has run, so its rows are copied to the host by the mirror kernel from "
                 "that step's snapshot of the slot mapping and its row-run table, before either is rewritten, with no "
                 "descriptor copy and no wait")
    return True


def graph_after_launch(worker, replayed):
    drafts = draft_layers(worker)
    if _GRAPH["ok"] and drafts:
        worker._taaf_draft_pending = tuple(drafts)
        worker._submitted_mirror_layers.update(drafts)
        _GRAPH["drafted"] = True
    if not replayed:
        return
    for handle in worker.cache_handles:
        rows = _GRAPH["rids"].get(id(handle))
        if rows is None or not handle.runtime.is_group_leader:
            continue
        slots = handle.mirror_slot_mapping
        handle.runtime.invalidate_written_slots(slots[:_GRAPH["snapshot_tokens"]], rows)


def fold_graph_tally():
    held = _GRAPH["tally"]
    if held is None:
        return
    rows, bad = held.to("cpu", copy=True).tolist()
    held.zero_()
    _GRAPH["rows"] += int(rows)
    _GRAPH["bad"] += int(bad)


def graph_gate(worker_class):
    initialize = worker_class.initialize
    start_step = worker_class.start_step
    stage_mapping = worker_class._stage_row_mirror_mapping
    enqueue_layer = worker_class._enqueue_layer_mirror
    enqueue_host_mirror = worker_class._enqueue_host_mirror
    flatten = sys.modules[worker_class.__module__]._flatten_row_mirrors

    def graph_initialize(self, *args, **kwargs):
        result = initialize(self, *args, **kwargs)
        self._taaf_graph_step = False
        if graph_prepare(self):
            for handle in self.cache_handles:
                handle.graph_mirror = True
                handle.mirror_from_resident = True
        return result

    def graph_start_step(self, metadata, request_state_indices, request_ids=None, num_tokens=0):
        self._taaf_graph_step = False
        if _GRAPH["ok"] and not capturing():
            flush_drafts(self)
            self._taaf_graph_step = graph_stage(self, flatten(metadata.row_mirrors, request_ids), int(num_tokens))
            graph_snapshot(self, int(num_tokens))
        return start_step(self, metadata, request_state_indices, request_ids, num_tokens)

    def graph_stage_mapping(self, num_tokens):
        if getattr(self, "_taaf_graph_step", False):
            return None
        return stage_mapping(self, num_tokens)

    def graph_layer_mirror(self, layer_index):
        if not _GRAPH["ok"] or not (getattr(self, "_taaf_graph_step", False) or capturing()):
            return enqueue_layer(self, layer_index)
        handle = self.cache_handles[layer_index]
        if not handle.host_mirror_required:
            return None
        if layer_index in self._per_layer_mirrored:
            if step_open():
                raise RuntimeError(f"HiSparse layer {layer_index} mirrored twice.")
            self._per_layer_mirrored.clear()
            self._submitted_mirror_layers.clear()
        graph_launch(self, layer_index)
        if handle.runtime.is_group_leader and handle.req_id_per_token is not None:
            _GRAPH["rids"][id(handle)] = handle.req_id_per_token
        self._per_layer_mirrored.add(layer_index)
        self._submitted_mirror_layers.add(layer_index)
        return None

    def graph_enqueue_host_mirror(self, *args, **kwargs):
        if not getattr(self, "_taaf_graph_step", False):
            return enqueue_host_mirror(self, *args, **kwargs)
        replayed = not any(handle.num_actual_tokens for handle in self.cache_handles)
        self._submitted_mirror_layers.update(
            index for index, handle in enumerate(self.cache_handles)
            if (replayed or handle.num_actual_tokens != 0) and handle.submit_layer_mirror is not None)
        result = enqueue_host_mirror(self, *args, **kwargs)
        graph_after_launch(self, replayed)
        return result

    worker_class.initialize = graph_initialize
    worker_class.start_step = graph_start_step
    worker_class._stage_row_mirror_mapping = graph_stage_mapping
    worker_class._enqueue_layer_mirror = graph_layer_mirror
    worker_class._enqueue_host_mirror = graph_enqueue_host_mirror
    announce("graph_mirror_gate", worker=WORKER_CLASS, knob=MIXED_FULL_ENV)


def sparse_metadata(attn_metadata):
    if not isinstance(attn_metadata, dict):
        return attn_metadata
    kept = {name: meta for name, meta in attn_metadata.items()
            if meta is None or all(hasattr(meta, field) for field in BATCH_FIELDS)}
    left = sorted(set(attn_metadata) - set(kept))
    if left:
        announce("prepare_in_layer", layers=len(left), first=left[0],
                 why="their metadata lacks the HiSparse batch fields, so each QSA layer prepares its handle in its own forward")
    return kept


def install_step_gate():
    import importlib

    worker_class = getattr(importlib.import_module(WORKER_MODULE), WORKER_CLASS)
    if gate_worker(worker_class):
        announce("step_gate", worker=WORKER_CLASS,
                 rule="host mirrors are written only between the connector's start_step and finish_forward")
    try:
        speculator_class = getattr(importlib.import_module(SPECULATOR_MODULE), SPECULATOR_CLASS)
    except (ImportError, AttributeError) as exc:
        announce("draft_tags_missing", error=type(exc).__name__,
                 why="draft layers cannot be told apart, so they are mirrored with the target forward")
        return
    if tag_drafts(speculator_class):
        announce("draft_tags", speculator=SPECULATOR_CLASS)


def step_open():
    return _STEP["open"]


def attach(layer, vllm_config):
    layer.hisparse_cache = None
    layer.taaf_hisparse_group = None
    if not armed(vllm_config):
        return
    if draft_stock() and draft_name(layer.layer_name):
        return
    install_step_gate()
    install_hot_reset(vllm_config, layer.topk_indices_buffer.device)
    install_hot_reads(vllm_config, layer.topk_indices_buffer.device)
    install_union(vllm_config, layer.topk_indices_buffer.shape[1] - 1, layer.topk_indices_buffer.device)
    install_union_tally(vllm_config, layer.topk_indices_buffer.shape[1] - 1, layer.topk_indices_buffer.shape[0],
                        layer.topk_indices_buffer.device)
    if install_rec2(vllm_config, layer.topk_indices_buffer.device):
        from vllm.v1.hisparse.runtime import HiSparseRuntime
        install_rec2_invalidate(HiSparseRuntime)
    if _REC2["age"] is not None and not union_tally():
        announce("rec2_needs_union_tally", knob=REC2_ENV, needs=UNION_TALLY_ENV,
                 why="the recorders fold the union tally's per-lane misses, so they stay silent without it")
    if union_swap() and hot_rows() > 0:
        install_hot_cap(layer.topk_indices_buffer.device)
    from vllm.v1.attention.backends.mla.index_group import (
        SparseMLAIndexGroupBuilder,
        get_sparse_mla_index_group_max_rows,
    )

    logical = layer.topk_indices_buffer[:, :-1]
    builder = SparseMLAIndexGroupBuilder(logical, get_sparse_mla_index_group_max_rows(vllm_config))
    head = row_of(layer)
    group, index = builder.register_layer(True, vllm_config, head_size=head,
                                          kv_cache_dtype=layer.kv_cache_dtype)
    layer.hisparse_cache = group.cache(index)
    layer.taaf_hisparse_group = group
    _ORDER.append(layer)
    reserve_stage(vllm_config, head, layer.topk_indices_buffer.device)
    install_split_gather()
    from vllm.taaf_qsa_kv8 import install_index_compact, install_index_rows, install_kv4_write

    install_index_compact(layer.topk_indices_buffer.device)
    install_index_rows()
    install_kv4_write(layer.topk_indices_buffer.device)
    if fused():
        install_fused(layer.topk_indices_buffer.shape[0], layer.topk_indices_buffer.device)
    if mixed_capture():
        install_mirror()
    if near_wanted():
        install_near(layer, group)
    announce_streaks()
    announce("attach", layer=layer.layer_name, top_k=logical.shape[1], row=head,
             decode_rows=group.physical_topk_indices.shape[0] - 1)


def planes(cache, heads):
    blocks, tokens, row = cache.shape
    if row % heads:
        raise ValueError(f"a {row}-byte row does not split into {heads} heads")
    if cache.stride(1) != row or cache.stride(2) != 1:
        raise ValueError(f"cache strides {cache.stride()} are not token rows of {row}")
    return torch.as_strided(cache, (blocks, heads, tokens, row // heads),
                            (cache.stride(0), row // heads, row, 1), cache.storage_offset())


def identity_table(blocks, device):
    key = (str(device), int(blocks))
    table = _TABLES.get(key)
    if table is None:
        table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
        _TABLES[key] = table
    return table


def with_counts(physical, counts):
    out = torch.empty((physical.shape[0], physical.shape[1] + 1), dtype=torch.int32, device=physical.device)
    out[:, :-1] = physical.to(torch.int32)
    out[:, -1] = counts.to(torch.int32)
    return out


def split_factor(rows, columns, target=SPLIT_TARGET_ROWS):
    factor = 1
    while (rows * factor * 2 <= target and columns % (factor * 2) == 0
           and columns // (factor * 2) >= SPLIT_MIN_COLUMNS):
        factor *= 2
    return factor


def split_plan(host_rows, device_rows, counts, factor):
    rows, columns = host_rows.shape
    width = columns // factor
    starts = torch.arange(factor, dtype=torch.int32, device=counts.device) * width
    pieces = (counts[:rows].to(torch.int32).view(rows, 1) - starts.view(1, factor)).clamp(0, width)
    return (host_rows.reshape(rows * factor, width), device_rows.reshape(rows * factor, width),
            pieces.reshape(-1).contiguous())


def host_group():
    said = str(os.environ.get(GROUP_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def group_active(runtime):
    group = host_group()
    if group < 2 or not fused():
        return 0
    host, hot = runtime.host_cache, runtime.hot.cache
    key = (group, tuple(host.shape), tuple(hot.shape))
    if not fused_ops().group_ready(host, hot, group):
        if key not in _GROUP["refused"]:
            _GROUP["refused"].add(key)
            announce("host_group_refused", knob=GROUP_ENV, group=group, host=tuple(host.shape), hot=tuple(hot.shape),
                     why="the row layout does not admit whole-group reads, the stock gather runs")
        return 0
    if not fused_ready(("group", group, host.shape[-1], str(host.dtype))):
        return 0
    if not _GROUP["said"]:
        _GROUP["said"] = True
        announce("host_group", knob=GROUP_ENV, group=group, row_bytes=int(host.shape[-1] * host.element_size()),
                 why="consecutive tokens of a layer already sit in one host row run, so a gather reads each group "
                     "holding a missed token whole and the GPU keeps only the missed tokens")
    return group


def spread_per_sm():
    said = str(os.environ.get(GATHER_SPREAD_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def device_sms(device):
    key = str(device)
    held = _SPREAD["sms"].get(key)
    if held is None:
        held = (torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda"
                else SPREAD_FALLBACK_SMS)
        _SPREAD["sms"][key] = held
    return held


def gather_spread(runtime):
    per_sm = spread_per_sm()
    if per_sm <= 0 or not fused():
        return 0
    host, hot = runtime.host_cache, runtime.hot.cache
    if not fused_ops().spread_ready(host, hot):
        key = (tuple(host.shape), tuple(hot.shape), str(host.dtype), str(hot.dtype))
        if key not in _SPREAD["refused"]:
            _SPREAD["refused"].add(key)
            announce("gather_spread_refused", knob=GATHER_SPREAD_ENV, host=tuple(host.shape), hot=tuple(hot.shape),
                     why="the row layout does not admit word reads of whole rows, the stock gather runs")
        return 0
    programs = per_sm * device_sms(hot.device)
    announce("gather_spread", knob=GATHER_SPREAD_ENV, per_sm=per_sm, programs=programs,
             rows=fused_ops().SPREAD_ROWS, group_knob=host_group(),
             why="a decode gather's misses are dealt evenly over a fixed grid of programs sized to the SMs, each "
                 "reading several host rows at once, instead of one block per request or column piece, so host "
                 "reads stay in flight on every SM whatever each request's miss count; it takes the place of the "
                 "group gather when both are set")
    return programs


def spread_warm(runtime, host_rows, device_rows, counts):
    host = runtime.host_cache
    key = ("spread", int(host.shape[-1]), str(host.dtype), str(host_rows.dtype), str(device_rows.dtype),
           str(counts.dtype))
    if fused_ready(key):
        return True
    if key not in _SPREAD["cold"]:
        _SPREAD["cold"].add(key)
        print(f"QSA_HISPARSE gather_spread_cold knob={GATHER_SPREAD_ENV} key={key} why=a captured forward reached a "
              f"spread gather no eager forward had launched, so this graph keeps the stock gather on the whole plan",
              file=sys.stderr, flush=True)
    return False


def guard_on():
    return (fused() and str(os.environ.get(GUARD_ENV, "1")).strip().lower() not in ("0", "off", "")
            and (_FUSED["module"] is not None or FUSED_MODULE in sys.modules))


def guard_tally(site, device):
    held = _GUARD["tally"].get(site)
    if held is None or held.device != device:
        held = _GUARD["tally"][site] = torch.zeros((), dtype=torch.int64, device=device)
        if not _GUARD["exit"]:
            _GUARD["exit"] = True
            atexit.register(guard_final)
            announce("host_kv_guard", knob=GUARD_ENV, site=site,
                     why="every index tensor a host-KV gather, stage or attention reads is bounded on the device "
                         "before the read, a violation is neutralised and counted in a device counter that is read "
                         "every report interval and at shutdown, and no host read happens on the hot path")
    return held


def guard_fold():
    if capturing():
        return {}
    found = {}
    for site, held in _GUARD["tally"].items():
        count = int(held.item())
        held.zero_()
        if count:
            _GUARD["seen"][site] = _GUARD["seen"].get(site, 0) + count
            found[site] = count
    for site, count in found.items():
        print(f"HOST_KV_GUARD violations={count} at {site} (total {_GUARD['seen'][site]})", file=sys.stderr, flush=True)
    return found


def guard_final():
    try:
        guard_fold()
    except Exception as failure:
        print(f"HOST_KV_GUARD final read failed ({type(failure).__name__}: {failure})", file=sys.stderr, flush=True)
    seen = _GUARD["seen"]
    print(f"HOST_KV_GUARD final violations={sum(seen.values())} "
          + (" ".join(f"{site}={count}" for site, count in seen.items()) or "none")
          + f" checked_sites={','.join(sorted(_GUARD['tally'])) or 'none'}", file=sys.stderr, flush=True)


def guard_logical(layer, logical, meta, split=None):
    if not guard_on() or logical.dtype != torch.int32 or logical.stride(1) != 1:
        return logical
    table = getattr(layer.hisparse_cache, "source_block_table", None)
    if table is None:
        return logical
    positions = int(table.shape[1]) * int(meta.block_size)
    fused_ops().guard_picks(logical, positions, guard_tally("decode_picks", logical.device), split)
    return logical


def plan_fused():
    return knob_on(PLAN_FUSED_ENV)


def plan_ahead():
    return knob_on(PLAN_AHEAD_ENV)


def gather_probe():
    said = str(os.environ.get(GATHER_PROBE_ENV) or "").strip().lower()
    return said if said in ("demand", "all") else ""


def probe_counts(counts, site):
    mode = gather_probe()
    if not mode or (site == "prefetch" and mode != "all"):
        return counts
    announce("gather_probe", knob=GATHER_PROBE_ENV, mode=mode, site=site,
             why="DIAGNOSTIC: the host rows of this gather are not copied, so the hot region keeps stale bytes and "
                 "the attention output is wrong; the step time without the gather is the most any overlap of it "
                 "can hide")
    return torch.zeros_like(counts)


def plan_gather(runtime, counts, idle, factor, host_rows, device_rows, tally):
    if (not plan_fused() or not guard_on() or host_rows.dim() != 2 or host_rows.stride(1) != 1
            or device_rows.stride(1) != 1 or counts.dim() != 1 or host_rows.shape != device_rows.shape
            or host_rows.shape[0] != counts.shape[0] * factor or counts.stride(0) != 1):
        return None
    host, hot = runtime.host_cache, runtime.hot.cache
    if host is None or hot is None:
        return None
    if not _PLAN_FUSED["said"]:
        _PLAN_FUSED["said"] = True
        announce("plan_fused", knob=PLAN_FUSED_ENV,
                 why="a decode gather's idle mask, piece counts, tally and row-bound guard run as one kernel a "
                     "resolve instead of a split kernel and a guard kernel")
    _PLAN_FUSED["calls"] += 1
    return fused_ops().plan_gather(counts, idle, factor, host_rows, device_rows, host.numel() // host.shape[-1],
                                   hot.numel() // hot.shape[-1], tally,
                                   guard_tally("gather_rows", host_rows.device))


def guard_counts(runtime, host_rows, device_rows, counts):
    if (not guard_on() or host_rows.dim() != 2 or host_rows.stride(1) != 1 or device_rows.stride(1) != 1
            or counts.dim() != 1 or host_rows.shape != device_rows.shape):
        return counts
    host, hot = runtime.host_cache, runtime.hot.cache
    if host is None or hot is None:
        return counts
    return fused_ops().guard_gather(host_rows, device_rows, counts, host.numel() // host.shape[-1],
                                    hot.numel() // hot.shape[-1], guard_tally("gather_rows", host_rows.device))


def run_merge_gap():
    said = str(os.environ.get(RUN_MERGE_ENV) or "").strip()
    return int(said) if said.isdigit() else -1


def run_merge_active(runtime):
    gap = run_merge_gap()
    if gap < 0 or not fused():
        return -1
    host, hot = runtime.host_cache, runtime.hot.cache
    if not fused_ops().run_ready(host, hot):
        key = (tuple(host.shape), tuple(hot.shape), str(host.dtype), str(hot.dtype))
        if key not in _PCIE["run_refused"]:
            _PCIE["run_refused"].add(key)
            announce("run_merge_refused", knob=RUN_MERGE_ENV, host=tuple(host.shape), hot=tuple(hot.shape),
                     why="the row layout does not admit word reads of whole rows, the other gathers run")
        return -1
    if not fused_ready(("run", gap, host.shape[-1], str(host.dtype))):
        return -1
    if not _PCIE["run_said"]:
        _PCIE["run_said"] = True
        ops = fused_ops()
        announce("run_merge", knob=RUN_MERGE_ENV, gap=gap, tile=ops.RUN_TILE,
                 vector=ops.run_vector(*ops.group_word(host, hot)),
                 why="a gather sorts each lane's missed host rows and reads runs of rows whose neighbours lie within "
                     "the gap as contiguous tiles, keeping only the missed rows on the GPU; it takes the place of "
                     "the spread, group and split gathers")
    return gap


def pcie_tally_level():
    said = str(os.environ.get(PCIE_TALLY_ENV) or "").strip()
    return min(2, int(said)) if said.isdigit() else 0


def pcie_acc(runtime, device):
    level = pcie_tally_level()
    held = _PCIE["acc"]
    if held is None or held.device != device:
        held = _PCIE["acc"] = torch.zeros((len(PCIE_SITES), len(fused_ops().PCIE_FIELDS)), dtype=torch.int64,
                                          device=device)
        _PCIE["clock"] = torch.zeros(len(PCIE_SITES), dtype=torch.int64, device=device)
        cache = runtime.host_cache
        _PCIE["row_bytes"] = int(cache.shape[-1]) * int(cache.element_size()) if isinstance(cache, torch.Tensor) else 0
        announce("pcie_tally", knob=PCIE_TALLY_ENV, level=level, row_bytes=_PCIE["row_bytes"],
                 why="every decode demand gather and prefetch counts the host rows it reads, the transfers it "
                     "issues and the bytes it moves on the GPU"
                     + (", and stamps the GPU clock around the gather" if level > 1 else "")
                     + "; one line every few hundred steps, never a read inside a layer")
    return held


def pcie_tally_start(runtime, host_rows, counts, site, group, gap):
    level = pcie_tally_level()
    if level <= 0 or host_rows.dim() != 2 or counts.dim() != 1:
        return None
    held = pcie_acc(runtime, host_rows.device)
    index = PCIE_SITES.index(site)
    held[index].add_(fused_ops().pcie_stats(host_rows, counts, group, gap))
    if level > 1:
        fused_ops().stamp(_PCIE["clock"][index:index + 1], held[index, -1:], False)
        return index
    return -1


def pcie_tally_done(index):
    if index is not None and index >= 0:
        fused_ops().stamp(_PCIE["clock"][index:index + 1], _PCIE["acc"][index, -1:], True)


def pcie_line(steps):
    held = _PCIE["acc"].to("cpu", copy=True).tolist()
    size = _PCIE["row_bytes"]
    per = max(steps, 1)
    names = fused_ops().PCIE_FIELDS
    parts = [f"QSA_HISPARSE pcie_tally window={_PCIE['lines']} steps={steps} row_bytes={size} "
             f"gap={_PCIE['gap']} group={_PCIE['group']}"]
    for site, values in zip(PCIE_SITES, held):
        row = dict(zip(names, (int(one) for one in values)))
        rows, moved = row["rows"], row["rows_read"]
        need = rows * size
        read = moved * size
        text = (f"{site}_calls={row['calls'] / per:.1f} {site}_rows={rows / per:.1f} "
                f"{site}_need_mb={need / per / 1e6:.2f} {site}_read_mb={read / per / 1e6:.2f} "
                f"{site}_transfers={row['transfers'] / per:.1f} "
                f"{site}_mean_transfer_b={read // max(row['transfers'], 1)} "
                f"{site}_amp={moved / max(rows, 1):.3f}")
        for label in ("group2", "group4", "group8", "run0", "run2", "run8"):
            text += f" {site}_{label}_xfers={row[label] / per:.1f}"
        for label in ("span0", "span2", "span8"):
            text += f" {site}_{label}_mb={row[label] * size / per / 1e6:.2f}"
        if row["clock_ns"]:
            text += (f" {site}_gather_ms={row['clock_ns'] / per / 1e6:.3f} "
                     f"{site}_gb_s_need={need / row['clock_ns']:.2f} {site}_gb_s_read={read / row['clock_ns']:.2f}")
        parts.append(text)
    return " ".join(parts)


def report_pcie():
    _PCIE["starts"] += 1
    _PCIE["steps"] += 1
    if _PCIE["acc"] is None or _PCIE["starts"] % GATHER_EVERY or capturing():
        return None
    line = pcie_line(_PCIE["steps"])
    _PCIE["acc"].zero_()
    _PCIE["steps"] = 0
    _PCIE["lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def gather_compact(runtime, host_rows, device_rows, counts, group=0, spread=0, guarded=False, site="demand", run=-1):
    if not guarded:
        counts = guard_counts(runtime, host_rows, device_rows, counts)
    _PCIE["gap"], _PCIE["group"] = run, group
    mark = pcie_tally_start(runtime, host_rows, counts, site, group, run)
    gather_copy(runtime, host_rows, device_rows, counts, group, spread, run)
    pcie_tally_done(mark)


def gather_plan(host_rows, device_rows, counts):
    if host_rows.shape[0] == 0 or host_rows.shape[1] == 0:
        return None
    return fused_ops().group_plan(host_rows, device_rows, counts)


def gather_copy(runtime, host_rows, device_rows, counts, group, spread, run):
    if run >= 0:
        _PCIE["run_calls"] += 1
        fused_ops().run_gather(runtime.host_cache, runtime.hot.cache, host_rows, device_rows, counts, run,
                               plan=gather_plan(host_rows, device_rows, counts))
        return
    if group:
        _GROUP["calls"] += 1
        fused_ops().group_gather(runtime.host_cache, runtime.hot.cache, host_rows, device_rows, counts, group,
                                 plan=gather_plan(host_rows, device_rows, counts))
        return
    if spread and spread_warm(runtime, host_rows, device_rows, counts):
        _SPREAD["calls"] += 1
        fused_ops().spread_gather(runtime.host_cache, runtime.hot.cache, host_rows, device_rows, counts, spread)
        return
    cache_ops().hisparse_gather_compact(runtime.host_cache, runtime.hot.cache, host_rows, device_rows, counts)


def split_gather(runtime_class):
    if getattr(runtime_class, SPLIT_MARK, False):
        return False

    def swap_rows(self, shared_rows):
        shared = self.index_group.shared_topk
        host_rows = shared.swap_host_physical_rows[shared_rows]
        device_rows = shared.swap_device_physical_rows[shared_rows]
        counts = shared.swap_counts[shared_rows]
        run = run_merge_active(self)
        spread = 0 if run >= 0 else gather_spread(self)
        group = 0 if spread or run >= 0 else group_active(self)
        factor = 1 if group or spread or run >= 0 else split_factor(host_rows.shape[0], host_rows.shape[1])
        if factor > 1:
            host_rows, device_rows, counts = split_plan(host_rows, device_rows, counts, factor)
        gather_compact(self, host_rows, device_rows, counts, group, spread, run=run)

    runtime_class._swap_rows = swap_rows
    setattr(runtime_class, SPLIT_MARK, True)
    return True


def install_split_gather():
    if str(os.environ.get(SPLIT_ENV, "1")).strip() in ("0", "", "off"):
        announce("gather_split_off", knob=SPLIT_ENV)
        return
    import importlib

    runtime_class = getattr(importlib.import_module(RUNTIME_MODULE), RUNTIME_CLASS)
    if split_gather(runtime_class):
        announce("gather_split", target_rows=SPLIT_TARGET_ROWS, min_columns=SPLIT_MIN_COLUMNS,
                 why="each request's compact miss list is cut into column chunks so the host-read kernel runs "
                     "enough blocks to keep PCIe busy instead of one block per request")


def stage_bytes(max_model_len, block_size, row_bytes, mib=0):
    if mib:
        return int(mib) * 2**20
    longest = -(-int(max_model_len) // int(block_size))
    return (STAGE_LONGEST * longest + STAGE_NULL_BLOCKS) * int(block_size) * int(row_bytes)


def reserve_stage(vllm_config, row_bytes, device):
    if _STAGE["buffer"] is not None:
        return _STAGE["buffer"]
    mib = int(str(os.environ.get(STAGE_ENV) or "0").strip() or 0)
    size = stage_bytes(vllm_config.model_config.max_model_len, vllm_config.cache_config.block_size, row_bytes, mib)
    _STAGE["device"] = device
    _STAGE["buffer"] = torch.empty(size, dtype=torch.uint8, device=device)
    announce("stage_buffer", mib=round(size / 2**20, 1), knob=STAGE_ENV,
             why="prefill rows are staged in one buffer reserved before the memory profile, so a step stages "
                 "its prefills in groups that fit it instead of allocating per step")
    return _STAGE["buffer"]


def fit_stage(vllm_config, block_size, row_bytes):
    buffer = _STAGE["buffer"]
    if buffer is None:
        return None
    mib = int(str(os.environ.get(STAGE_ENV) or "0").strip() or 0)
    size = stage_bytes(vllm_config.model_config.max_model_len, block_size, row_bytes, mib)
    if buffer.numel() >= size:
        return buffer
    _STAGE["buffer"] = None
    del buffer
    _STAGE["buffer"] = torch.empty(size, dtype=torch.uint8, device=_STAGE["device"])
    announce("stage_resized", mib=round(size / 2**20, 1), block=int(block_size),
             why="the attention block size is final only after the layers attach, so the stage buffer is "
                 "re-sized to whole pages of that block plus the null page before the memory profile")
    return _STAGE["buffer"]


def stage_reuse_modes():
    said = str(os.environ.get(STAGE_REUSE_ENV) or "").strip().lower()
    if not said or said in ("0", "off", "false"):
        return ()
    if said in ("1", "on", "true"):
        return STAGE_REUSE_MODES
    modes = tuple(part.strip() for part in said.split(",") if part.strip())
    for mode in modes:
        if mode not in STAGE_REUSE_MODES:
            raise ValueError(f"{STAGE_REUSE_ENV}={said!r} is not '', '0', '1' or a comma list of {STAGE_REUSE_MODES}")
    return modes


def stage_reuse():
    return "reuse" in stage_reuse_modes() and _REUSE["aside"] is None


def stage_blocks():
    return "blocks" in stage_reuse_modes() and _REUSE["aside"] is None


def reuse_bytes():
    buffer = _STAGE["buffer"]
    mib = int(str(os.environ.get(STAGE_REUSE_MIB_ENV) or "0").strip() or 0)
    if mib:
        return mib * 2**20
    return 0 if buffer is None else int(buffer.numel())


def reserve_reuse(name, block_size, row_bytes):
    if not stage_reuse_modes():
        return None
    device = _STAGE["device"]
    if device is None:
        announce("stage_reuse_no_device", layer=name, knob=STAGE_REUSE_ENV,
                 why="the stage buffer was never reserved, so there is no device to hold the reuse buffers and every "
                     "prefill chunk stages its rows the stock way")
        return None
    if not _REUSE["checked"]:
        _REUSE["checked"] = True
        try:
            why = stage_reuse_check(torch.device(device))
        except Exception as failure:
            why = f"{type(failure).__name__}: {str(failure)[:300]}"
        if why:
            _REUSE["aside"] = why
            selfcheck_fail(STAGE_REUSE_ENV, f"the startup check failed: {why}")
            print(f"QSA_STAGE_REUSE stands aside: the self-check failed ({why}); every prefill chunk stages its "
                  "rows the stock way", file=sys.stderr, flush=True)
            return None
    if not stage_reuse():
        return None
    size = reuse_bytes()
    rows = size // max(1, int(row_bytes))
    held = _REUSE["layers"].get(name)
    if held is not None and held.buffer.numel() >= size:
        return held
    _REUSE["layers"][name] = SimpleNamespace(
        buffer=torch.empty(size, dtype=torch.uint8, device=device),
        valid=torch.zeros(rows, dtype=torch.bool, device=device),
        src=torch.full((rows,), -1, dtype=torch.int32, device=device),
        tag=torch.full((rows,), -1, dtype=torch.int32, device=device))
    _REUSE["bytes"] = sum(int(state.buffer.numel()) for state in _REUSE["layers"].values())
    announce(f"stage_reuse_buffer_{len(_REUSE['layers'])}", layer=name, mib=round(size / 2**20, 1),
             total_mib=round(_REUSE["bytes"] / 2**20, 1), knob=STAGE_REUSE_ENV,
             why="each QSA layer stages its prefill rows into a buffer of its own, reserved before the memory "
                 "profile, so rows staged for a request's last chunk are still there for its next chunk")
    return _REUSE["layers"][name]


def stable_plan(block_table, seq_lens, ctx_lens, block_size, capacity):
    from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan

    device = block_table.device
    requests, width = block_table.shape
    size = int(block_size)
    lens = seq_lens.to(torch.int64).clamp(min=0)
    used = torch.div(lens + size - 1, size, rounding_mode="floor")
    ends = torch.cumsum(used, 0)
    starts = ends - used
    offsets = torch.arange(int(capacity), device=device)
    owner = torch.searchsorted(ends, offsets, right=True)
    inside = owner < requests
    owner = owner.clamp(max=max(requests - 1, 0))
    column = offsets - starts[owner]
    inside &= (column >= 0) & (column < width)
    table = block_table.to(torch.int64)
    physical = table[owner, column.clamp(0, max(width - 1, 0))]
    inside &= physical >= 0
    columns = torch.arange(width, device=device)
    live = columns.view(1, -1) < used.view(-1, 1)
    new_bt = torch.where(live, starts.view(-1, 1) + columns.view(1, -1) + 1, torch.zeros_like(table))
    step = torch.arange(size, device=device)
    blocks = torch.where(inside.view(-1, 1), physical.view(-1, 1) * size + step.view(1, -1), torch.full_like(
        physical.view(-1, 1).expand(-1, size), -1))
    row_ids = torch.cat((step.view(1, -1), blocks)).reshape(1, -1).to(torch.int32)
    positions = column.view(-1, 1) * size + step.view(1, -1)
    ctx = ctx_lens.to(torch.int64).clamp(min=0)
    settled = inside.view(-1, 1) & (positions < ctx[owner].view(-1, 1))
    nothing = torch.zeros((1, size), dtype=torch.bool, device=device)
    first = torch.where(used > 0, table[:, 0], torch.full_like(used, -1)) if width else torch.full_like(used, -1)
    tags = torch.where(inside, first[owner], torch.full_like(owner, -1)).view(-1, 1).expand(-1, size)
    plan = HiSparsePrefillStagingPlan(
        block_table=new_bt.to(torch.int32), row_ids=row_ids,
        dst_rows=torch.arange(row_ids.shape[1], dtype=torch.int32, device=device).view(1, -1),
        miss_mask=(row_ids >= 0).to(torch.int32), block_size=size)
    plan.taaf_reuse_fixed = torch.cat((nothing, settled)).reshape(-1)
    plan.taaf_reuse_tag = torch.cat((torch.full((1, size), -1, dtype=torch.int64, device=device), tags)).reshape(
        -1).to(torch.int32)
    announce("stage_reuse_plan", knob=STAGE_REUSE_ENV,
             why="a prefill part stages its requests' blocks in table order, so a request's earlier blocks keep "
                 "their staged rows from one chunk to the next instead of moving with the physical block ids")
    return plan


def staging_plan(block_table, seq_lens, ctx_lens, block_size, capacity):
    from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan

    if stage_reuse() and block_table.shape[0] > 0 and ctx_lens is not None:
        return stable_plan(block_table, seq_lens, ctx_lens, block_size, capacity)
    return build_hisparse_prefill_staging_plan(block_table, seq_lens, block_size, capacity)


def reuse_state(layer, need, plan):
    if layer is None or not stage_reuse() or getattr(plan, "taaf_reuse_tag", None) is None:
        return None
    state = _REUSE["layers"].get(getattr(layer, "layer_name", layer))
    if state is None:
        return None
    rows = plan.row_ids.shape[1]
    if need > state.buffer.numel() or rows > state.valid.numel():
        state.valid.fill_(False)
        announce("stage_reuse_too_big", need=need, held=state.buffer.numel(), knob=STAGE_REUSE_MIB_ENV,
                 why="a prefill part needs more staged bytes than the layer's reuse buffer, so it stages through "
                     "the shared buffer and the layer's staged rows are dropped")
        return None
    return state


def reuse_split(state, plan, needed):
    rows = needed.numel()
    source = plan.row_ids[0][:rows]
    eligible = (source == state.src[:rows]) & (plan.taaf_reuse_tag[:rows] == state.tag[:rows])
    kept = state.valid[:rows] & eligible & plan.taaf_reuse_fixed[:rows]
    copy = needed & ~kept
    if device_tally_on():
        add_device_tally("reuse_rows_kept", (needed & kept).sum())
        add_device_tally("reuse_rows_copied", copy.sum())
    announce("stage_reuse", knob=STAGE_REUSE_ENV,
             why="a prefill chunk copies only the rows its picks need that the layer's buffer does not already hold "
                 "for the same request, block and position from an earlier chunk")
    return kept, copy


def reuse_commit(state, plan, kept, copy):
    rows = kept.numel()
    state.valid[:rows].copy_(kept | copy)
    state.valid[rows:].fill_(False)
    state.src[:rows].copy_(plan.row_ids[0][:rows])
    state.src[rows:].fill_(-1)
    state.tag[:rows].copy_(plan.taaf_reuse_tag[:rows])


def host_words(kv_cache, device):
    base = kv_cache if kv_cache.device == device else device_alias(kv_cache, device).view(kv_cache.dtype)
    flat = base.reshape(-1).view(torch.uint8)
    return flat.view(torch.int32) if flat.numel() % 4 == 0 and flat.data_ptr() % 4 == 0 else flat


def stock_block_plan(plan, miss_mask, kept, size, blocks):
    ids = plan.row_ids[0][:blocks * size].view(blocks, size).to(torch.int64)
    first = ids[:, 0]
    ordered = (ids == first.view(-1, 1) + torch.arange(size, device=ids.device).view(1, -1)).all(1)
    wanted = miss_mask.reshape(-1)[:blocks * size].view(blocks, size) != 0
    full = ordered & (first >= 0) & (wanted.sum(1) >= int(size * STAGE_BLOCK_SHARE))
    if kept is not None:
        full &= ~kept[:blocks * size].view(blocks, size).any(1)
    sources = torch.div(first.clamp(min=0), size, rounding_mode="floor").to(torch.int32)
    out = miss_mask.clone()
    out.view(-1)[:blocks * size] = (wanted & ~full.view(-1, 1)).reshape(-1).to(out.dtype)
    return full, sources, out


def block_stage_rows(plan, kv_cache, staged, miss_mask, kept=None):
    size = int(plan.block_size)
    rows = plan.row_ids.shape[1]
    blocks = rows // size
    if blocks == 0 or kv_cache.shape[1] != size or not kv_cache.is_contiguous() or not staged.is_contiguous():
        return miss_mask
    if (knob_on(TAKE_ROWS_ENV) and miss_mask.is_cuda and miss_mask.is_contiguous() and plan.row_ids.is_contiguous()
            and (kept is None or kept.is_contiguous())):
        host = host_words(kv_cache, staged.device)
        target = staged.reshape(-1).view(torch.uint8)
        target = target.view(torch.int32) if host.dtype == torch.int32 and target.numel() % 4 == 0 else target
        if target.dtype != host.dtype:
            return miss_mask
        words = kv_cache[0].numel() * kv_cache.element_size() // target.element_size()
        ops = fused_ops()
        full, sources, out = ops.block_plan(plan.row_ids[0], miss_mask, kept, size, int(size * STAGE_BLOCK_SHARE),
                                            blocks)
        if take_shadow_on():
            want_full, want_sources, want_out = stock_block_plan(plan, miss_mask, kept, size, blocks)
            take_shadow_note("blocks",
                             torch.cat([full.to(torch.int32), torch.where(want_full, sources, 0).to(torch.int32),
                                        out.reshape(-1).to(torch.int32)]),
                             torch.cat([want_full.to(torch.int32), torch.where(want_full, want_sources, 0),
                                        want_out.reshape(-1).to(torch.int32)]))
        ops.block_stage(host, target, sources, full, words)
        if device_tally_on():
            add_device_tally("block_stage_blocks", full.sum())
        announce("stage_blocks", knob=STAGE_REUSE_ENV, share=STAGE_BLOCK_SHARE,
                 why="a staged block whose host rows are mostly needed is copied as one contiguous run of the host "
                     "block, and only the rest of the needed rows go through the per-row gather; the block choice "
                     "and the remaining miss mask come from one kernel a block")
        return out
    ids = plan.row_ids[0][:blocks * size].view(blocks, size).to(torch.int64)
    first = ids[:, 0]
    ordered = (ids == first.view(-1, 1) + torch.arange(size, device=ids.device).view(1, -1)).all(1)
    wanted = miss_mask.reshape(-1)[:blocks * size].view(blocks, size) != 0
    full = ordered & (first >= 0) & (wanted.sum(1) >= int(size * STAGE_BLOCK_SHARE))
    if kept is not None:
        full &= ~kept[:blocks * size].view(blocks, size).any(1)
    host = host_words(kv_cache, staged.device)
    target = staged.reshape(-1).view(torch.uint8)
    target = target.view(torch.int32) if host.dtype == torch.int32 and target.numel() % 4 == 0 else target
    if target.dtype != host.dtype:
        return miss_mask
    words = kv_cache[0].numel() * kv_cache.element_size() // target.element_size()
    sources = torch.div(first.clamp(min=0), size, rounding_mode="floor").to(torch.int32)
    fused_ops().block_stage(host, target, sources, full.to(torch.int32), words)
    if device_tally_on():
        add_device_tally("block_stage_blocks", full.sum())
    announce("stage_blocks", knob=STAGE_REUSE_ENV, share=STAGE_BLOCK_SHARE,
             why="a staged block whose host rows are mostly needed is copied as one contiguous run of the host "
                 "block, and only the rest of the needed rows go through the per-row gather")
    left = (wanted & ~full.view(-1, 1)).reshape(-1)
    out = miss_mask.clone()
    out.view(-1)[:blocks * size] = left.to(out.dtype)
    return out


def stage_reuse_check(device):
    from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan

    generator = torch.Generator().manual_seed(4099)
    size, row, host_blocks = 16, 8, 12
    host = torch.randint(-2**30, 2**30, (host_blocks, size, row), generator=generator, dtype=torch.int32)
    if device.type == "cuda":
        host = host.pin_memory()
    table = torch.tensor([[7, 3, 9, 5]], dtype=torch.int32, device=device)
    capacity = 6
    state = SimpleNamespace(buffer=torch.zeros((capacity + 1) * size * row * 4, dtype=torch.uint8, device=device),
                            valid=torch.zeros((capacity + 1) * size, dtype=torch.bool, device=device),
                            src=torch.full(((capacity + 1) * size,), -1, dtype=torch.int32, device=device),
                            tag=torch.full(((capacity + 1) * size,), -1, dtype=torch.int32, device=device))
    flat_host = host.to(device).view(-1, row)
    for ctx, seq in ((0, 20), (20, 37), (37, 60)):
        plan = stable_plan(table, torch.tensor([seq], device=device), torch.tensor([ctx], device=device), size,
                           capacity)
        stock = build_hisparse_prefill_staging_plan(table, torch.tensor([seq], device=device), size, capacity)
        positions = torch.randint(0, seq, (seq - ctx, 6), generator=generator).to(device)
        staged = state.buffer.view(torch.int32).view(-1, row)
        needed = torch.zeros(plan.row_ids.shape[1], dtype=torch.bool, device=device)
        pages = positions // size
        rows = plan.block_table.to(torch.int64)[0, pages] * size + positions % size
        needed[rows.reshape(-1)] = True
        kept, copy = reuse_split(state, plan, needed)
        source = plan.row_ids[0].to(torch.int64)
        take = copy & (source >= 0)
        staged[take] = flat_host[source[take]]
        reuse_commit(state, plan, kept, copy)
        expected = flat_host[source[needed]]
        if not torch.equal(staged[needed], expected):
            return f"chunk {ctx}-{seq} read rows the host does not hold"
        stock_rows = stock.block_table.to(torch.int64)[0, pages] * size + positions % size
        if not torch.equal(flat_host[stock.row_ids[0].to(torch.int64)[stock_rows.reshape(-1)]],
                           flat_host[source[rows.reshape(-1)]]):
            return f"chunk {ctx}-{seq} maps a pick to another host row than the stock plan"
        if ctx and int(kept[needed].sum()) == 0:
            return f"chunk {ctx}-{seq} kept no row from the chunk before"
    if "blocks" in stage_reuse_modes() and device.type == "cuda":
        plan = stable_plan(table, torch.tensor([60], device=device), torch.tensor([60], device=device), size,
                           capacity)
        target = torch.zeros(plan.row_ids.shape[1], row, dtype=torch.int32, device=device)
        mask = (plan.row_ids >= 0).to(torch.int32)
        left = block_stage_rows(plan, host, target.view(-1, size, row), mask)
        source = plan.row_ids[0].to(torch.int64)
        moved = (mask.view(-1) != 0) & (left.view(-1) == 0) & (source >= 0)
        if int(moved.sum()) == 0 or not torch.equal(target[moved], flat_host[source[moved]]):
            return "the contiguous block copy moved other bytes than the per-row copy"
    for name in ("reuse_rows_kept", "reuse_rows_copied", "block_stage_blocks"):
        held = _DEVICE_TALLY.get(name)
        if held is not None:
            held.zero_()
    announce("stage_reuse_check", device=str(device),
             why="three consecutive chunks of one synthetic request kept their earlier rows, read every picked row "
                 "as the host holds it, and mapped every pick to the stock plan's host row")
    return ""


def live_check(layer, cache, kv_cache, part, meta, logical, owners, staged):
    if _REUSE["live"] >= STAGE_REUSE_LIVE_CHECKS or capturing() or logical is None or owners is None:
        return
    _REUSE["live"] += 1
    stock = stage_rows(cache, kv_cache, part, meta, logical, owners).clone()
    needed = needed_rows(logical, owners, part.plan.block_table, part.plan.block_size, part.plan.row_ids.shape[1])
    width = staged.shape[-1]
    if not torch.equal(staged.reshape(-1, width)[needed], stock.reshape(-1, width)[needed]):
        _REUSE["aside"] = f"live check {_REUSE['live']} on {getattr(layer, 'layer_name', layer)} differed"
        selfcheck_fail(STAGE_REUSE_ENV, f"{_REUSE['aside']} from the stock stage")
        print(f"QSA_STAGE_REUSE stands aside: {_REUSE['aside']} from the stock stage; every later prefill chunk "
              "stages its rows the stock way", file=sys.stderr, flush=True)
        staged.copy_(stock)
        return
    announce("stage_reuse_live", checks=STAGE_REUSE_LIVE_CHECKS,
             why="the first eager reuse stages were compared with a stock stage of the same chunk: every needed row "
                 "matched byte for byte")


def stage_capacity_blocks(block_size, row_bytes):
    buffer = _STAGE["buffer"]
    if buffer is None:
        raise RuntimeError("the host KV prefill stage buffer was never reserved")
    return max(0, buffer.numel() // (int(block_size) * int(row_bytes)) - STAGE_NULL_BLOCKS)


def pack_prefills(used, capacity):
    groups = []
    first, total = 0, 0
    for index, blocks in enumerate(used):
        if blocks > capacity:
            raise ValueError(f"a prefill needs {blocks} staged blocks and the stage buffer holds {capacity}; "
                             f"raise {STAGE_ENV}")
        if total + blocks > capacity and index > first:
            groups.append((first, index, total))
            first, total = index, 0
        total += blocks
    if used:
        groups.append((first, len(used), total))
    return groups


def mixed_capture():
    return knob_on(MIXED_CAPTURE_ENV)


def prefill_host(meta, common):
    prefills = int(getattr(meta, "num_prefills", 0) or 0)
    starts = getattr(common, "query_start_loc_cpu", None)
    bound = getattr(common, "seq_lens_cpu_upper_bound", None)
    if prefills == 0 or starts is None or bound is None:
        return None
    decodes = int(meta.num_decodes)
    reqs = decodes + prefills
    if starts.device.type != "cpu" or bound.device.type != "cpu" or starts.numel() <= reqs or bound.numel() < reqs:
        return None
    return (tuple(int(one) for one in starts[decodes:reqs + 1].tolist()),
            tuple(int(one) for one in bound[decodes:reqs].tolist()))


def mirror_builds(builder_class):
    if getattr(builder_class, MIRROR_MARK, False):
        return False
    build = builder_class.build

    def mirrored_build(self, *args, **kwargs):
        meta = build(self, *args, **kwargs)
        common = kwargs.get("common_attn_metadata", args[1] if len(args) > 1 else None)
        setattr(meta, MIRROR_FIELD, prefill_host(meta, common))
        return meta

    builder_class.build = mirrored_build
    setattr(builder_class, MIRROR_MARK, True)
    return True


def install_mirror():
    import importlib

    builder_class = getattr(importlib.import_module(QSA_CACHE_MODULE), QSA_BUILDER_CLASS)
    if mirror_builds(builder_class):
        announce("prefill_mirror_armed", knob=MIXED_CAPTURE_ENV, builder=QSA_BUILDER_CLASS,
                 why="the QSA side metadata keeps the runner's host copies of its prefill starts and sequence-length "
                     "bounds, so a mixed step packs its prefill staging without reading the GPU")


def prefill_mirror(side, prefills):
    if not mixed_capture():
        return None
    held = getattr(side, MIRROR_FIELD, None)
    if held is None or len(held[0]) != prefills + 1 or len(held[1]) != prefills:
        _MIRROR["absent"] += 1
        announce("prefill_mirror_absent", prefills=prefills, held=held is not None,
                 why="this side metadata carries no host copy of its prefill rows, so the plan reads them from the GPU")
        return None
    return held


def mirror_used(held, block_size, capacity):
    used = [-(-int(value) // block_size) for value in held[1]]
    if max(used) > capacity:
        _MIRROR["over"] += 1
        announce("prefill_mirror_over_stage", blocks=max(used), capacity=capacity,
                 why="a prefill's host sequence-length bound needs more staged blocks than the stage buffer holds, so "
                     "this step packs from the GPU values")
        return None
    _MIRROR["used"] += 1
    announce("prefill_mirror", prefills=len(used), starts=held[0][0], ends=held[0][-1], blocks=sum(used),
             capacity=capacity,
             why="the prefill parts are packed from the host copies of the starts and sequence-length bounds, and "
                 "each part's row range and staged capacity are checked against the GPU values on the GPU")
    return used


def check_mirror(side, main, decodes, first, last, start, end, total, block_size):
    if not device_tally_on():
        return
    loc = side.query_start_loc
    lens = main.seq_lens[decodes + first:decodes + last].to(torch.int64)
    used = torch.div(lens + block_size - 1, block_size, rounding_mode="floor").sum()
    bad = ((loc[decodes + first] != start).to(torch.int64) + (loc[decodes + last] != end).to(torch.int64)
           + (used > total).to(torch.int64))
    add_device_tally("mirror_misses", bad)


def prefill_parts(side, main, block_size, row_bytes):
    from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan

    decodes = int(side.num_decodes)
    num_reqs = decodes + int(side.num_prefills)
    capacity = stage_capacity_blocks(block_size, row_bytes)
    mirror = prefill_mirror(side, num_reqs - decodes)
    used = None if mirror is None else mirror_used(mirror, block_size, capacity)
    if used is None:
        mirror = None
        starts = [int(value) for value in side.query_start_loc[decodes:num_reqs + 1].tolist()]
        lens = main.seq_lens[decodes:num_reqs].to(torch.int64)
        used = [int(value) for value in ((lens + block_size - 1) // block_size).tolist()]
    else:
        starts = list(mirror[0])
    parts = []
    for first, last, total in pack_prefills(used, capacity):
        ctx = None
        if stage_reuse():
            lens_now = main.seq_lens[decodes + first:decodes + last].to(torch.int64)
            fed = torch.tensor([starts[at + 1] - starts[at] for at in range(first, last)], dtype=torch.int64,
                               device=lens_now.device)
            ctx = lens_now - fed
        plan = staging_plan(main.block_table[decodes + first:decodes + last],
                            main.seq_lens[decodes + first:decodes + last], ctx, block_size, total)
        if plan_once():
            plan.taaf_pristine = plan.miss_mask
        if mirror is not None:
            check_mirror(side, main, decodes, first, last, starts[first], starts[last], total, block_size)
        parts.append(SimpleNamespace(first=first, last=last, start=starts[first], end=starts[last], plan=plan))
    return parts


def fixed_prefill_part(block_table, seq_lens, split, req_id_per_token, seqs, capacity, block_size, tokens):
    from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan

    device = block_table.device
    lanes = torch.arange(seqs, device=device)
    lead = split[0].to(torch.int64)
    prefills = split[2].to(torch.int64)
    index = (lead + lanes).clamp(max=block_table.shape[0] - 1)
    live = lanes < prefills
    lens = torch.where(live, seq_lens.index_select(0, index).to(torch.int64), torch.zeros_like(index))
    table = block_table.index_select(0, index)
    if device_tally_on():
        add_device_tally("mirror_misses", (((lens + block_size - 1) // block_size).sum() > capacity).to(torch.int64))
    owners = req_id_per_token[:tokens].to(torch.int64) - lead
    inside = (owners >= 0) & (owners < prefills)
    fed = torch.zeros(seqs, dtype=torch.int64, device=device)
    fed.scatter_add_(0, owners.clamp(0, max(seqs - 1, 0)), inside.to(torch.int64))
    plan = staging_plan(table, lens, (lens - fed).clamp(min=0), block_size, capacity)
    if plan_once():
        plan.taaf_pristine = plan.miss_mask

    viewed = mask_owners()

    def mask(logical):
        if viewed:
            return logical[:tokens]
        counts = torch.where(inside, logical[:tokens, -1], torch.zeros_like(logical[:tokens, -1]))
        return torch.cat((logical[:tokens, :-1], counts.unsqueeze(1)), 1)

    def last_rows(held):
        picked = held.index_select(0, (lead + lanes).clamp(max=held.numel() - 1))
        return picked.clamp(0, tokens - 1)

    if viewed:
        owners = torch.where(inside, owners, -1)
        announce("mask_owners", knob=MASK_OWNERS_ENV,
                 why="a bucketed prefill part marks the rows outside it by owner -1, which every staging and "
                     "attention kernel treats as zero picks, so the part reads the layer's pick table as a view "
                     "instead of copying it with the outside rows' counts zeroed")
    return SimpleNamespace(first=0, last=seqs, start=0, end=tokens, plan=plan, table=table, index=index,
                           owners=owners.to(torch.int32), mask=mask, last_rows=last_rows)


def stock_logical(head, rows, split):
    keep = torch.arange(rows, device=head.device) < split[1]
    return torch.where(keep.unsqueeze(1), head, torch.full_like(head, -1)).contiguous()


def stock_states(states, split):
    lanes = torch.arange(states.numel(), device=states.device)
    return torch.where(lanes < split[0], states, torch.full_like(states, -1))


def fixed_logical(logical, rows, split):
    head = logical[:rows]
    if knob_on(TAKE_ROWS_ENV) and head.is_cuda and head.dim() == 2 and head.stride(1) == 1:
        made = fused_ops().rows_below(head, split[1:2])
        if take_shadow_on():
            take_shadow_note("logical", made, stock_logical(head, rows, split))
        return made
    return stock_logical(head, rows, split)


def fixed_states(states, split):
    if knob_on(TAKE_ROWS_ENV) and states.is_cuda and states.dim() == 1 and states.is_contiguous():
        made = fused_ops().rows_below(states, split[0:1])
        if take_shadow_on():
            take_shadow_note("states", made, stock_states(states, split))
        return made
    return stock_states(states, split)


def take_shadow_on():
    return knob_on(TAKE_ROWS_ENV) and knob_on(TAKE_SHADOW_ENV)


def take_shadow_ready(device):
    if not capturing():
        for name in TAKE_SHADOW_TALLIES:
            if _DEVICE_TALLY.get(name) is None or _DEVICE_TALLY[name].device != device:
                add_device_tally(name, torch.zeros((), dtype=torch.int64, device=device))
        for key, dtype, fill in (("max", torch.float32, 0.0), ("layer", torch.int32, -1)):
            held = _TAKE_SHADOW[key]
            if held is None or held.device != device:
                _TAKE_SHADOW[key] = torch.full((), fill, dtype=dtype, device=device)
        return True
    held = [_DEVICE_TALLY.get(name) for name in TAKE_SHADOW_TALLIES] + [_TAKE_SHADOW["max"], _TAKE_SHADOW["layer"]]
    return all(one is not None and one.device == device for one in held)


def take_shadow_bits(tensor):
    if not tensor.is_floating_point():
        return tensor
    return tensor.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()])


def take_shadow_note(site, made, want, layer=-1):
    device = made.device
    if made.numel() == 0 or not take_shadow_ready(device):
        return
    rows = int(made.shape[0]) if made.dim() else 1
    made, want = made.reshape(rows, -1), want.reshape(rows, -1)
    apart = take_shadow_bits(made) != take_shadow_bits(want)
    differ = apart.any(1).sum()
    gap = torch.where(apart, torch.nan_to_num((made.float() - want.float()).abs(), nan=float("inf")), 0.0).max()
    add_device_tally(f"take_shadow_{site}_calls", torch.ones((), dtype=torch.int64, device=device))
    add_device_tally(f"take_shadow_{site}_bad", (differ > 0).to(torch.int64))
    add_device_tally(f"take_shadow_{site}_rows", torch.full((), rows, dtype=torch.int64, device=device))
    add_device_tally(f"take_shadow_{site}_differ", differ)
    _TAKE_SHADOW["max"].copy_(torch.maximum(_TAKE_SHADOW["max"], gap))
    first = (differ > 0) & (_TAKE_SHADOW["layer"] < 0)
    _TAKE_SHADOW["layer"].copy_(torch.where(first, torch.full_like(_TAKE_SHADOW["layer"], int(layer)),
                                            _TAKE_SHADOW["layer"]))


def take_shadow_report(now=None):
    if not take_shadow_on() or capturing() or _TAKE_SHADOW["max"] is None:
        return None
    now = time.monotonic() if now is None else now
    if _TAKE_SHADOW["told"] is not None and now - _TAKE_SHADOW["told"] < TAKE_SHADOW_EVERY_S:
        return None
    _TAKE_SHADOW["told"] = now
    fold_device_tallies()
    _TAKE_SHADOW["max_seen"] = max(_TAKE_SHADOW["max_seen"], float(_TAKE_SHADOW["max"].item()))
    held = int(_TAKE_SHADOW["layer"].item())
    if _TAKE_SHADOW["layer_seen"] < 0 and held >= 0:
        _TAKE_SHADOW["layer_seen"] = held
    rows = sum(_TALLY[f"take_shadow_{site}_rows"] for site in TAKE_SHADOW_SITES)
    differ = sum(_TALLY[f"take_shadow_{site}_differ"] for site in TAKE_SHADOW_SITES)
    layer = _TAKE_SHADOW["layer_seen"]
    line = (f"{TAKE_ROWS_ENV} live check: max diff {_TAKE_SHADOW['max_seen']:g}, rows {differ:,} of {rows:,} differ, "
            f"layer {layer if layer >= 0 else '-'}; "
            + "; ".join(f"{site} rows {_TALLY[f'take_shadow_{site}_differ']:,} of {_TALLY[f'take_shadow_{site}_rows']:,}"
                        f" in {_TALLY[f'take_shadow_{site}_bad']:,} of {_TALLY[f'take_shadow_{site}_calls']:,} calls"
                        for site in TAKE_SHADOW_SITES)
            + " (since the start, each fused take beside the stock torch path on the same inputs)")
    print(line, file=sys.stderr, flush=True)
    return line


def mixed_full_module():
    import importlib

    return importlib.import_module(MIXED_FULL_MODULE)


def fixed_reasons(layer, view):
    why = []
    for name, on in (("fused", fused()), ("union_swap", union_swap()), ("hot_rows", hot_rows() > 0),
                     ("plan_once", plan_once()), ("syncfree", syncfree()), ("sparse_stage", sparse_stage()),
                     ("sources_fused", knob_on(SOURCES_FUSED_ENV))):
        if not on:
            why.append(f"{name}_off")
    if row_attend():
        why.append("row_attend_on")
    if view.qlen < 2:
        why.append("decode_query_len_below_two")
    return why


def stage_veto(lens, bucket, bounds):
    if _SHARED["aside"]:
        return "plan_once_captured_mismatch"
    if _FIXED.get("capture_aside"):
        return "fixed_decode_aside"
    geom = _FIXED.get("geom")
    if geom is None or bounds is None:
        return ""
    block_size, row_bytes = geom
    capacity = stage_capacity_blocks(block_size, row_bytes)
    if mixed_full_module().stage_fits(bounds, block_size, capacity):
        return ""
    refused(bounds, block_size, row_bytes, capacity)
    return "stage_over_capacity"


def refused(bounds, block_size, row_bytes, capacity):
    needed = sum(-(-int(bound) // int(block_size)) for bound in bounds)
    _REFUSED["needed"].append(needed)
    _REFUSED["seqs"].append(len(bounds))
    now = time.monotonic()
    if now - _REFUSED["last"] < REFUSED_REPORT_S:
        return
    _REFUSED["last"] = now
    held = np.asarray(_REFUSED["needed"])
    block_mib = int(block_size) * int(row_bytes) / float(1 << 20)
    have = (capacity + STAGE_NULL_BLOCKS) * block_mib
    fits = {mib: int((held + STAGE_NULL_BLOCKS <= int(mib / block_mib)).sum()) for mib in (320, 480, 640, 960)}
    print(f"QSA_HISPARSE stage_refused n={held.size} capacity_blocks={capacity} stage_mib={have:.1f} "
          f"needed_blocks p50/p90/max={int(np.percentile(held, 50))}/{int(np.percentile(held, 90))}/{int(held.max())} "
          f"prefill_seqs p50/max={int(np.percentile(_REFUSED['seqs'], 50))}/{max(_REFUSED['seqs'])} "
          f"would_fit_at_stage_mib={fits} last_bounds={[int(b) for b in bounds][:8]}", flush=True)


def register_stage_veto():
    if not _FIXED.get("veto"):
        mixed_full_module().register_veto("host_kv_stage", stage_veto)
        _FIXED["veto"] = True


def fixed_view(side):
    if not graph_mirror() or not _ORDER:
        return None
    qlen = int(getattr(side, "decode_query_len", 0) or 0)
    view = mixed_full_module().bucket_view(side, qlen)
    if view is None:
        return None
    why = fixed_reasons(_ORDER[0], view)
    if why:
        _FIXED["aside"] += 1
        if capturing():
            _FIXED["capture_aside"] = True
        announce("fixed_decode_aside", reasons=",".join(why),
                 why="the bucketed host-KV step needs the fused capped decode path and the sync-free prefill "
                     "stage, so this step keeps the host-sized path")
        return None
    return view


def fixed_metadata(side, main, block_size, row_bytes, view):
    capacity = stage_capacity_blocks(block_size, row_bytes)
    _FIXED["geom"] = (int(block_size), int(row_bytes))
    register_stage_veto()
    part = fixed_prefill_part(main.block_table, main.seq_lens, view.split, side.token_to_req, view.seqs, capacity,
                              int(block_size), view.tokens)
    announce("fixed_decode", decodes=view.decodes, rows=view.rows, tokens=view.tokens, seqs=view.seqs,
             why="the decode region is sized to the bucket maximum and its live lanes come from the device split, so "
                 "the union, resolve, gather and attention read no host count; padding lanes carry no picks and the "
                 "resolve skips them by state index")
    return SimpleNamespace(
        num_actual_tokens=view.tokens, num_decode_tokens=view.rows, num_decodes=view.decodes,
        num_prefills=view.seqs, num_reqs=view.decodes + view.seqs, max_query_len=view.tokens,
        decode_max_query_len=view.qlen, query_start_loc=side.query_start_loc, req_id_per_token=side.token_to_req,
        seq_lens=main.seq_lens, block_size=int(block_size), prefill=SimpleNamespace(parts=[part]), fixed=view)


def metadata(side, main, block_size, row_bytes):
    view = fixed_view(side)
    if view is not None:
        return fixed_metadata(side, main, block_size, row_bytes, view)
    num_reqs = int(side.num_decodes) + int(side.num_prefills)
    prefill = None
    if side.num_decode_tokens < side.num_actual_tokens:
        prefill = SimpleNamespace(parts=prefill_parts(side, main, block_size, row_bytes))
    return SimpleNamespace(
        num_actual_tokens=int(side.num_actual_tokens),
        num_decode_tokens=int(side.num_decode_tokens),
        num_decodes=int(side.num_decodes),
        num_prefills=int(side.num_prefills),
        num_reqs=num_reqs,
        max_query_len=int(side.max_query_len),
        decode_max_query_len=int(side.decode_query_len),
        query_start_loc=side.query_start_loc,
        req_id_per_token=side.token_to_req,
        seq_lens=main.seq_lens,
        block_size=int(block_size),
        prefill=prefill,
    )


def knob_on(name):
    return str(os.environ.get(name) or "").strip().lower() in ("1", "on", "true")


def mask_owners():
    return str(os.environ.get(MASK_OWNERS_ENV) or "1").strip().lower() in ("1", "on", "true")


def plan_once():
    return knob_on(PLAN_ONCE_ENV)


def syncfree():
    return knob_on(SYNCFREE_ENV)


@_marked("hkv_step_metadata")
def step_metadata(side, main, block_size, row_bytes):
    if not plan_once():
        return metadata(side, main, block_size, row_bytes)
    stamp = _STEP.get("steps", 0)
    if capturing():
        owner = capture_owner()
        if owner is None:
            return metadata(side, main, block_size, row_bytes)
        stamp = ("capture", owner)
    key = (side, main, int(block_size), int(row_bytes), stamp)
    held = _PLAN["key"]
    if held is not None and held[0] is side and held[1] is main and held[2:] == key[2:]:
        _PLAN["reused"] += 1
        if isinstance(stamp, tuple):
            _PLAN["reused_captured"] = _PLAN.get("reused_captured", 0) + 1
            announce("plan_once_captured", knob=PLAN_ONCE_CAPTURED_ENV,
                     why="inside one graph capture the first QSA layer builds the step's positions, lengths, slots "
                         "and staging plan once; every QSA layer of that capture gets its own copy of the plan whose "
                         "resident rows and miss mask it resolves against its own resident table")
            return layer_view(_PLAN["meta"], _SHARED["records"].get(stamp[1]))
        announce("plan_once", knob=PLAN_ONCE_ENV, block_size=int(block_size), row_bytes=int(row_bytes),
                 why="a step's QSA metadata and prefill staging plan are layer-independent, so the first QSA "
                     "layer builds them and the others reuse them")
        return _PLAN["meta"]
    meta = metadata(side, main, block_size, row_bytes)
    _PLAN["key"] = key
    _PLAN["meta"] = meta
    _PLAN["built"] += 1
    if isinstance(stamp, tuple):
        if not getattr(getattr(meta, "prefill", None), "parts", None):
            _PLAN["key"] = None
            return meta
        share_reps(meta)
        return layer_view(meta, keep_record(stamp[1], side, main, block_size, row_bytes, meta))
    return meta


_CAPTURES = {"count": 0, "hooked": False}
_SHARED = {"records": {}, "left": None, "aside": "", "matched": 0, "unmatched": 0}
PLAN_MEMOS = ("gpu_row_ids", "gpu_source_key", "taaf_reps", "taaf_check")
LAYER_MEMOS = ("gpu_row_ids", "gpu_source_key", "taaf_check")


def count_captures():
    if _CAPTURES["hooked"]:
        return
    graph = torch.cuda.CUDAGraph
    begin = graph.capture_begin

    def counted(self, *args, **kwargs):
        _CAPTURES["count"] += 1
        return begin(self, *args, **kwargs)

    graph.capture_begin = counted
    _CAPTURES["hooked"] = True


def plan_shared():
    if not knob_on(PLAN_ONCE_CAPTURED_ENV) or _SHARED["aside"]:
        return False
    if torch.cuda.is_available():
        count_captures()
    return True


def capture_owner():
    if not plan_shared() or not _CAPTURES["hooked"]:
        return None
    return _CAPTURES["count"]


def shared_checks():
    if _SHARED["left"] is None:
        _SHARED["left"] = max(0, int(os.environ.get(PLAN_ONCE_CHECKS_ENV) or PLAN_ONCE_CHECKS))
    return _SHARED["left"]


def keep_record(owner, side, main, block_size, row_bytes, meta):
    prefill = getattr(meta, "prefill", None)
    if not shared_checks() or prefill is None or not getattr(prefill, "parts", None):
        return None
    record = SimpleNamespace(side=side, main=main, block_size=int(block_size), row_bytes=int(row_bytes),
                             tokens=int(meta.num_actual_tokens), layers=[], shared=[], parts=len(prefill.parts),
                             prints=torch.zeros((PLAN_PRINT_SLOTS, 3), dtype=torch.int64,
                                                device=prefill.parts[0].plan.row_ids.device), used=0)
    for at, part in enumerate(prefill.parts):
        for name, value in shared_fields(part):
            record.shared.append((at, name, keep_print(record, value)))
    _SHARED["records"][owner] = record
    return record


def shared_fields(part):
    for name, value in vars(part).items():
        if name != "plan" and not callable(value):
            yield name, value
    for name, value in vars(part.plan).items():
        if name not in PLAN_MEMOS:
            yield name, value


def plan_print(value):
    flat = value.reshape(-1).to(torch.int64)
    weights = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64) % PLAN_PRINT_PRIME + 1
    return torch.stack(((flat * weights).sum(), flat.sum(), (flat != 0).sum()))


def keep_print(record, value):
    if not isinstance(value, torch.Tensor):
        return ("value", value)
    if record.used >= record.prints.shape[0]:
        return ("full", None)
    slot = record.used
    record.used += 1
    record.prints[slot].copy_(plan_print(value))
    return ("tensor", tuple(value.shape), value.dtype, slot)


def print_differs(record, held, value):
    if held[0] == "full":
        return False
    if held[0] == "value":
        return isinstance(value, torch.Tensor) or held[1] != value
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != held[1] or value.dtype != held[2]:
        return True
    return not torch.equal(record.prints[held[3]], plan_print(value))


def own_plan(plan):
    mine = copy.copy(plan)
    mine.miss_mask = getattr(plan, "taaf_pristine", None)
    if mine.miss_mask is None:
        mine.miss_mask = plan.miss_mask
    for name in LAYER_MEMOS:
        setattr(mine, name, None)
    return mine


def share_reps(meta):
    if not knob_on(SOURCES_FUSED_ENV):
        return
    for part in meta.prefill.parts:
        plan = part.plan
        size = int(plan.block_size)
        if size > 0 and plan.row_ids.dim() == 2 and plan.row_ids.is_contiguous():
            plan_reps(plan, plan.row_ids.shape[1] // size)
    announce("plan_once_reps", knob=PLAN_ONCE_CAPTURED_ENV,
             why="a prefill plan's representative request a host block is a function of the plan's own block table, "
                 "so a capture computes it once with the shared plan and every QSA layer's copy reads it")


def layer_view(meta, record=None):
    prefill = getattr(meta, "prefill", None)
    if prefill is None or not getattr(prefill, "parts", None):
        return meta
    parts = []
    for at, part in enumerate(prefill.parts):
        plan = own_plan(part.plan)
        if record is not None:
            plan.taaf_check = (record, at)
        parts.append(SimpleNamespace(**{**vars(part), "plan": plan}))
    return SimpleNamespace(**{**vars(meta), "prefill": SimpleNamespace(**{**vars(prefill), "parts": parts})})


def note_layer(plan, part, meta, source, block_size):
    held = getattr(plan, "taaf_check", None)
    if held is None or not capturing():
        return
    record, at = held
    record.layers.append(SimpleNamespace(at=at, source=source, block_size=int(block_size),
                                         first=meta.num_decodes + part.first, last=meta.num_decodes + part.last,
                                         lanes=getattr(part, "index", None) is not None,
                                         prints={name: keep_print(record, getattr(plan, name))
                                                 for name in ("gpu_row_ids", "miss_mask")}))


def layer_table(layer, part):
    if layer.lanes:
        return layer.source.index_select(0, part.index.clamp(max=layer.source.shape[0] - 1))
    return layer.source[layer.first:layer.last]


def shared_differs(record):
    fresh = metadata(record.side, record.main, record.block_size, record.row_bytes)
    fresh_parts = getattr(fresh.prefill, "parts", None) or []
    if record.parts != len(fresh_parts):
        return f"prefill parts {record.parts} captured, {len(fresh_parts)} fresh"
    names = {at: dict(shared_fields(part)) for at, part in enumerate(fresh_parts)}
    for at, name, held in record.shared:
        if print_differs(record, held, names[at].get(name)):
            return f"part {at} {name}"
    for index, layer in enumerate(record.layers):
        new = fresh_parts[layer.at]
        ref = own_plan(new.plan)
        resolve_sources(ref, layer_table(layer, new), layer.block_size)
        for name, held in layer.prints.items():
            if print_differs(record, held, getattr(ref, name)):
                return f"layer {index} of the capture, part {layer.at} {name}"
    return ""


def check_shared():
    if not _SHARED["records"] or not _SHARED["left"] or capturing():
        return
    mixed = sys.modules.get(MIXED_FULL_MODULE)
    if mixed is None or not mixed.graph_step():
        return
    tokens = mixed.graph_tokens()
    if not tokens:
        return
    found = [record for record in _SHARED["records"].values() if record.tokens == tokens]
    if len(found) != 1:
        _SHARED["unmatched"] += 1
        return
    torch.cuda.synchronize()
    why = shared_differs(found[0])
    _SHARED["left"] -= 1
    _SHARED["matched"] += 1
    layers = len(found[0].layers)
    if why:
        _SHARED["aside"] = why
        _SHARED["records"].clear()
        selfcheck_fail(PLAN_ONCE_CAPTURED_ENV, f"replay {_SHARED['matched']} at {tokens} tokens: the shared captured "
                                               f"plan differs from a fresh per-layer build ({why})")
        print(f"QSA_HISPARSE plan_once_captured MISMATCH replay={_SHARED['matched']} tokens={tokens} "
              f"layers={layers} first_difference=[{why}]: the shared captured plan differs from a fresh per-layer "
              f"build, so it stands aside; mixed full-graph replays are vetoed and every step builds its own metadata "
              f"per layer", flush=True)
        return
    print(f"QSA_HISPARSE plan_once_captured check replay={_SHARED['matched']} tokens={tokens} layers={layers} "
          f"matches a fresh per-layer build bit for bit; {_SHARED['left']} checks left", flush=True)
    if not _SHARED["left"]:
        _SHARED["records"].clear()


def resolve_sources(plan, table, block_size):
    pristine = getattr(plan, "taaf_pristine", None)
    if pristine is not None and plan.gpu_source_key != (table.data_ptr(), block_size):
        plan.miss_mask, plan.gpu_row_ids, plan.gpu_source_key = pristine, None, None
    if knob_on(SOURCES_FUSED_ENV) and fused() and sources_fused(plan, table, block_size):
        return
    if knob_on(SOURCES_FUSED_ENV):
        _TALLY["sources_stock"] += 1
    plan.ensure_gpu_sources(table, block_size)


def plan_reps(plan, unique):
    held = getattr(plan, "taaf_reps", None)
    if held is not None:
        return held
    table = plan.block_table.to(torch.int64)
    rows, cols = table.shape
    reps = torch.full((unique,), rows * cols, device=plan.row_ids.device, dtype=torch.int64)
    reps.scatter_reduce_(0, table.reshape(-1), torch.arange(rows * cols, device=plan.row_ids.device), reduce="amin",
                         include_self=True)
    plan.taaf_reps = reps
    return reps


def sources_fused(plan, table, block_size):
    key = (table.data_ptr(), block_size)
    if plan.gpu_source_key == key:
        return True
    size = int(plan.block_size)
    rows, cols = plan.block_table.shape
    if block_size <= 0 or size % block_size or rows == 0 or table.shape[0] < rows:
        return False
    if table.dim() != 2 or table.stride(1) != 1 or plan.row_ids.dim() != 2 or not plan.row_ids.is_contiguous():
        return False
    unique = plan.row_ids.shape[1] // size
    reps = plan_reps(plan, unique)
    plan.gpu_row_ids, plan.miss_mask = fused_ops().sources(plan.row_ids, reps, table, size, block_size, cols)
    plan.gpu_source_key = key
    _TALLY["sources_fused"] += 1
    announce("sources_fused", knob=SOURCES_FUSED_ENV, rows=int(plan.row_ids.shape[1]), block=size,
             why="a prefill plan's representative request a host block is computed once and shared by every "
                 "resident group; each group's resident rows and miss mask are one kernel over its own table")
    return True


def unresolved(plan):
    pristine = getattr(plan, "taaf_pristine", None)
    if pristine is not None:
        plan.miss_mask, plan.gpu_row_ids, plan.gpu_source_key = pristine, None, None


def device_tally_on():
    if _DEVICE_TALLY_ON[0] is None:
        said = str(os.environ.get(DEVICE_TALLY_ENV, "1")).strip().lower()
        _DEVICE_TALLY_ON[0] = said not in ("0", "off", "false", "no")
        if not _DEVICE_TALLY_ON[0]:
            announce("device_tally_off", knob=DEVICE_TALLY_ENV,
                     why="the host-KV path skips its device-side counters (reuse, block stage, mirror, sparse rows, "
                         "last rows): each was a reduce and an add a layer that changes no output")
    return _DEVICE_TALLY_ON[0]


def add_device_tally(name, value):
    held = _DEVICE_TALLY.get(name)
    if held is None or held.device != value.device:
        held = torch.zeros((), dtype=torch.int64, device=value.device)
        _DEVICE_TALLY[name] = held
    held.add_(value.to(torch.int64))


def fold_device_tallies():
    if capturing():
        return False
    fold_graph_tally()
    for name, held in _DEVICE_TALLY.items():
        if held is not None:
            _TALLY[name] += int(held.item())
            held.zero_()
    return True


def heads_of(layer):
    return int(layer.num_kv_heads)


def source_rows(cache):
    if cache.dim() == 3:
        return cache
    if cache.dim() == 4 and cache.shape[1] == 1:
        return cache.squeeze(1)
    raise ValueError(f"the host KV source cache {tuple(cache.shape)} is not [blocks, 1, tokens, row]")


def write(layer, impl, key, value, meta, slot_rows):
    from vllm.taaf_qsa_kv8 import write as quantized_write

    cache = layer.hisparse_cache
    target, slots, rows = cache.write_target(meta.num_actual_tokens, slot_rows)
    quantized_write(impl, key[:rows], value[:rows], planes(target, heads_of(layer)), slots)
    if is_draft(layer) and draft_through():
        through_write(layer, impl, key, value, cache, rows, quantized_write)
    if not step_open() and not graph_capture_open():
        _TALLY["gated_writes"] += 1
        announce("gated_write", layer=layer.layer_name, rows=rows,
                 why="no connector step is open (warmup, dummy or draft forward), so nothing is mirrored to host")
        return rows
    mirror = cache.mirror_write_target(rows)
    if mirror is not None:
        quantized_write(impl, key[:rows], value[:rows], planes(mirror[0], heads_of(layer)), mirror[1])
    cache.finish_kv_update()
    return rows


def row_attend():
    return str(os.environ.get(ROW_ATTEND_ENV) or "").strip().lower() in ("1", "on", "true")


def hot_floor_queries(max_query_len):
    if union_swap() and hot_rows() > 0:
        announce("hot_cap_floor", queries=max_query_len, rows=hot_rows(),
                 why="a lane whose union of verification rows overflows the hot region attends a span of rows at "
                     "a time, so the hot cache holds one row's picks at least")
        return 1
    if not row_attend():
        return max_query_len
    announce("row_attend_floor", queries=max_query_len,
             why="each verification row attends right after its own swap, so the hot cache holds one row's "
                 "picks at a time")
    return 1


def step_rows(starts, step, rows):
    tokens = starts[:-1] + step
    active = (tokens < starts[1:]) & (tokens < rows)
    return tokens, active, tokens.clamp(0, max(rows - 1, 0))


def decode_attention_rows(layer, impl, query, logical, meta, out, gate):
    from vllm.taaf_qsa_kv8 import attention

    group = layer.taaf_hisparse_group
    rows = meta.num_decode_tokens
    decodes = meta.num_decodes
    starts = meta.query_start_loc[:decodes + 1].to(torch.long)
    hot = planes(group.physical_kv_cache(0), heads_of(layer))
    table = identity_table(hot.shape[0], hot.device)
    owners = torch.zeros(decodes, dtype=torch.int32, device=query.device)
    landed = torch.empty((rows + 1,) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
    for step in range(meta.decode_max_query_len):
        tokens, active, picked = step_rows(starts, step, rows)
        group.set_logical_topk_ready(0)
        chosen = logical.index_select(0, picked)[:, :-1].masked_fill(~active.unsqueeze(1), -1).contiguous()
        physical, counts = group.convert_decode_logical_to_physical_topk(
            0, chosen, meta, return_valid_counts=True, num_decodes=decodes, decode_query_len=1)
        landing = torch.empty((decodes,) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
        attention(impl, query.index_select(0, picked), hot, with_counts(physical, counts), table, owners, False,
                  landing, gate.index_select(0, picked))
        landed.index_copy_(0, torch.where(active, tokens, torch.full_like(tokens, rows)), landing)
    out[:rows].copy_(landed[:rows])
    _TALLY["row_attend_steps"] += meta.decode_max_query_len
    announce("row_attend", layer=layer.layer_name, rows=rows, decodes=decodes, steps=meta.decode_max_query_len)


def env_int(name, default):
    said = str(os.environ.get(name) or "").strip()
    return int(said) if said else default


def capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def layer_number(name):
    numbers = [int(part) for part in str(name).split(".") if part.isdigit()]
    return numbers[0] if numbers else -1


def is_draft(layer):
    return layer.layer_name in _DRAFT


def pick_dir():
    return Path(str(os.environ.get(PICK_DIR_ENV) or PICK_DIR))


def pick_flush():
    import numpy

    records = _PICKS["records"]
    _PICKS["records"] = None
    if not records:
        return None
    arrays = {}
    for index, record in enumerate(records):
        for field, value in record.items():
            arrays[f"r{index}_{field}"] = numpy.asarray(value)
    folder = pick_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"burst{_PICKS['bursts']:03d}.npz"
    numpy.savez(path, **arrays)
    size = path.stat().st_size
    _PICKS["bytes"] += size
    announce(f"pick_log_{_PICKS['bursts']}", path=str(path), records=len(records), bytes=size,
             total_bytes=_PICKS["bytes"])
    return path


def pick_step(layer):
    bursts = env_int(PICK_LOG_ENV, 0)
    if bursts <= 0 or is_draft(layer):
        return
    if _PICKS["first"] is None:
        _PICKS["first"] = layer.layer_name
    if layer.layer_name != _PICKS["first"]:
        return
    _PICKS["steps"] += 1
    step = _PICKS["steps"]
    if _PICKS["records"] is not None and step >= _PICKS["burst_end"]:
        pick_flush()
        _PICKS["bursts"] += 1
    if (_PICKS["records"] is None and _PICKS["bursts"] < bursts and _PICKS["bytes"] < PICK_MAX_BYTES
            and step % max(1, env_int(PICK_EVERY_ENV, PICK_EVERY)) == 0):
        _PICKS["records"] = []
        _PICKS["burst_end"] = step + PICK_BURST_STEPS


def to_cpu(tensor):
    return tensor.detach().to("cpu").numpy()


def pick_before(layer, logical, meta):
    if _PICKS["records"] is None:
        return None
    if capturing():
        _PICKS["skipped"] += 1
        return None
    decodes = min(int(meta.num_decodes), PICK_REQS)
    cache = layer.hisparse_cache
    runtime = getattr(cache, "runtime", None)
    table = getattr(cache, "source_block_table", None)
    states = getattr(runtime, "request_state_indices", None)
    if decodes == 0 or table is None or states is None:
        return None
    starts = meta.query_start_loc[:decodes + 1].to(torch.long)
    rows = int(starts[-1])
    state = states[:decodes]
    resident = runtime.index_group.device_global_indices.index_select(0, state.clamp_min(0).to(torch.long))
    return {"layer": layer_number(layer.layer_name), "draft": int(is_draft(layer)), "step": _PICKS["steps"],
            "picks": to_cpu(logical[:rows, :-1].to(torch.int32)), "starts": to_cpu(starts),
            "states": to_cpu(state), "seq_lens": to_cpu(meta.seq_lens[:decodes]),
            "block_table": to_cpu(table[:decodes].to(torch.int32)), "resident": to_cpu(resident),
            "block_size": int(meta.block_size), "query_len": int(meta.decode_max_query_len),
            "decodes": int(meta.num_decodes)}


def pick_after(layer, record):
    if record is None or _PICKS["records"] is None:
        return
    runtime = layer.hisparse_cache.runtime
    shared = runtime.index_group.shared_topk
    width = record["decodes"]
    steps = max(1, int(getattr(runtime, "_swap_step", 1)))
    kept = record["starts"].shape[0] - 1
    counts = shared.swap_counts[:steps * width].view(steps, width)[:, :kept]
    hosts = shared.swap_host_physical_rows[:steps * width].view(steps, width, -1)[:, :kept]
    record["counts"] = to_cpu(counts)
    record["hosts"] = to_cpu(hosts)
    _PICKS["records"].append(record)


def streak_times(said):
    text = str(said or "").strip()
    if not text:
        return ()
    try:
        times = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise RuntimeError(f"{PICK_AT_ENV}={text!r} is not a comma list of seconds of play") from error
    if not times or any(not 0.0 <= one < 1e7 for one in times) or list(times) != sorted(set(times)):
        raise RuntimeError(f"{PICK_AT_ENV}={text!r} must list distinct seconds of play in ascending order")
    return times


def streak_plan():
    held = _STREAK["plan"]
    if held is None:
        times = streak_times(os.environ.get(PICK_AT_ENV))
        held = SimpleNamespace(at=times, steps=max(1, env_int(PICK_STEPS_ENV, STREAK_STEPS)),
                               lanes=max(1, env_int(PICK_LANES_ENV, STREAK_LANES)),
                               cap=max(1, env_int(PICK_MIB_ENV, STREAK_MIB)) << 20) if times else False
        _STREAK["plan"] = held
    return held or None


def announce_streaks():
    plan = streak_plan()
    if plan is None:
        return False
    announce("pick_streak_schedule", at=",".join(f"{one:g}" for one in plan.at), steps=plan.steps, reqs=plan.lanes,
             cap_mib=plan.cap >> 20, folder=str(pick_dir()),
             why="each listed second of play, counted from the first engine step with a decode lane, opens a streak "
                 "of consecutive decode steps that leave the full graphs, so every target QSA layer records the "
                 "picks and hot-region changes of the same requests")
    return True


def streak_lanes(runner, batch_state):
    states = np.asarray(batch_state.idx_mapping_np).reshape(-1)
    busy = np.asarray(batch_state.is_prefilling_np, dtype=bool).reshape(-1)
    held = getattr(getattr(runner, "req_states", None), "num_computed_tokens_np", None)
    done = np.asarray(held)[states] if held is not None else np.zeros(states.shape[0], dtype=np.int64)
    return [(str(name), int(state), int(count)) for name, state, count, flag
            in zip(batch_state.req_ids, states.tolist(), done.tolist(), busy.tolist()) if not flag]


def streak_spread(lanes, count):
    ordered = sorted(lanes, key=lambda lane: (lane[2], lane[0]))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[-1]]
    return [ordered[round(at * (len(ordered) - 1) / (count - 1))] for at in range(count)]


def streak_id(name):
    ids = _STREAK["ids"]
    if name not in ids:
        ids[name] = len(_STREAK["names"])
        _STREAK["names"].append(name)
    return ids[name]


def streak_track(streak, lanes, count):
    live = {name: state for name, state, _ in lanes}
    kept = [name for name in streak["tracked"] if name in live]
    streak["dropped"] += len(streak["tracked"]) - len(kept)
    if len(kept) < count:
        spare = sorted((lane for lane in lanes if lane[0] not in streak["ever"]), key=lambda lane: (-lane[2], lane[0]))
        for name, _, _ in spare[:count - len(kept)]:
            kept.append(name)
            streak["ever"].add(name)
            streak["joined"] += 1
    streak["tracked"] = kept
    return {live[name]: streak_id(name) for name in kept}


def streak_median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


def streak_interval(now):
    seen = _STREAK["seen"]
    _STREAK["seen"] = now
    if seen is None:
        return
    streak = _STREAK["open"]
    if _STREAK["forced"] and streak is not None:
        streak["intervals"].append(now - seen)
    elif not _STREAK["forced"]:
        held = _STREAK["baseline"]
        held.append(now - seen)
        del held[:-STREAK_BASELINE]


def streak_due(plan, lanes, now):
    at = _STREAK["next"]
    if at >= len(plan.at) or _STREAK["elapsed"] < plan.at[at] or not lanes:
        return None
    _STREAK["next"] = at + 1
    room = plan.cap - _STREAK["bytes"]
    if room <= 0:
        announce(f"pick_streak_{at}_skipped", at=f"{plan.at[at]:g}", elapsed=round(_STREAK["elapsed"], 1),
                 total_bytes=_STREAK["bytes"], cap_mib=plan.cap >> 20,
                 why="the pick-log byte cap is spent, so this streak records nothing and its steps keep their graphs")
        return None
    chosen = streak_spread(lanes, plan.lanes)
    base = streak_median(_STREAK["baseline"])
    streak = {"index": at, "at": plan.at[at], "opened": now, "steps": 0, "records": [], "bytes": 0, "room": room,
              "tracked": [lane[0] for lane in chosen], "ever": {lane[0] for lane in chosen}, "dropped": 0,
              "joined": 0, "idle": 0, "missing": 0, "intervals": [], "engines": [], "times": [], "last": {},
              "groups": {}, "final": {}, "baseline": base, "folder": pick_dir()}
    _STREAK["open"] = streak
    if not _STREAK.get("exit"):
        atexit.register(streak_exit)
        _STREAK["exit"] = True
    announce(f"pick_streak_{at}_start", at=f"{plan.at[at]:g}", elapsed=round(_STREAK["elapsed"], 1),
             engine=_STREAK["engine"], decode_lanes=len(lanes), reqs=len(chosen),
             contexts="/".join(str(lane[2]) for lane in chosen), steps=plan.steps,
             room_mib=round(room / (1 << 20), 1), baseline_ms=None if base is None else round(1000 * base, 1),
             why="this streak's decode steps leave the full graphs for the piecewise path, so the target QSA layers "
                 "record the chosen requests' picks every step; only these steps pay for it")
    return streak


def pick_engine_step(runner, batch_state, now=None):
    plan = streak_plan()
    if plan is None or batch_state is None:
        return False
    now = time.monotonic() if now is None else float(now)
    streak_interval(now)
    _STREAK["engine"] += 1
    _STREAK["forced"] = False
    _STREAK["slots"] = {}
    lanes = streak_lanes(runner, batch_state)
    if _STREAK["clock"] is None:
        if not lanes:
            return False
        _STREAK["clock"] = now
        announce("pick_streak_clock", engine=_STREAK["engine"], lanes=len(lanes),
                 at=",".join(f"{one:g}" for one in plan.at),
                 why="the streak schedule counts seconds from this step, the first engine step with a decode lane")
    _STREAK["elapsed"] = now - _STREAK["clock"]
    streak = _STREAK["open"]
    if streak is not None and (streak["steps"] >= plan.steps or streak["bytes"] >= streak["room"]):
        streak_close(now)
        streak = None
    if streak is None:
        streak = streak_due(plan, lanes, now)
    if streak is None or not lanes:
        if streak is not None:
            streak["idle"] += 1
        return False
    _STREAK["slots"] = streak_track(streak, lanes, plan.lanes)
    streak["steps"] += 1
    streak["engines"].append(_STREAK["engine"])
    streak["times"].append(_STREAK["elapsed"])
    streak["final"] = dict(_STREAK["slots"])
    _STREAK["forced"] = True
    return True


def streak_ends(streak):
    final = streak["final"]
    if not final or not streak["groups"] or capturing():
        return {}
    states = sorted(final)
    layers, resident, lru = [], [], []
    for number, group in sorted(streak["groups"].items()):
        index = torch.as_tensor(states, dtype=torch.long, device=group.device_global_indices.device)
        layers.append(number)
        resident.append(to_cpu(group.device_global_indices.index_select(0, index)))
        lru.append(to_cpu(group.lru_slots.index_select(0, index)))
    return {"end_layers": np.asarray(layers, dtype=np.int32), "end_states": np.asarray(states, dtype=np.int32),
            "end_ids": np.asarray([final[state] for state in states], dtype=np.int32),
            "end_resident": np.stack(resident), "end_lru": np.stack(lru)}


def streak_write(streak, tails):
    arrays = {}
    for index, record in enumerate(streak["records"]):
        for field, value in record.items():
            arrays[f"r{index}_{field}"] = np.asarray(value)
    arrays.update(tails)
    arrays["names"] = np.asarray(_STREAK["names"] or [""])
    arrays["engines"] = np.asarray(streak["engines"], dtype=np.int64)
    arrays["times"] = np.asarray(streak["times"], dtype=np.float64)
    arrays["intervals"] = np.asarray(streak["intervals"], dtype=np.float64)
    arrays["baseline"] = np.asarray(_STREAK["baseline"], dtype=np.float64)
    arrays["at"] = np.asarray(streak["at"], dtype=np.float64)
    folder = streak["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"streak{streak['index']:03d}.npz"
    np.savez(path, **arrays)
    return path, path.stat().st_size


def streak_close(now=None, ends=True, cut=None):
    streak = _STREAK["open"]
    _STREAK["open"] = None
    _STREAK["slots"] = {}
    if streak is None:
        return None
    now = time.monotonic() if now is None else now
    tails = streak_ends(streak) if ends else {}
    started = time.monotonic()
    path, size = streak_write(streak, tails)
    written = time.monotonic() - started
    _STREAK["bytes"] += size
    spent = sum(streak["intervals"])
    base = streak["baseline"]
    extra = None if base is None else spent - base * len(streak["intervals"])
    cut = cut or ("byte_cap" if streak["bytes"] >= streak["room"] else "steps")
    announce(f"pick_streak_{streak['index']}_end", path=str(path), cut=cut, steps=streak["steps"],
             engine=f"{streak['engines'][0]}..{streak['engines'][-1]}" if streak["engines"] else "none",
             wall_s=round(now - streak["opened"], 2), forced=streak["steps"], forced_s=round(spent, 2),
             step_ms=round(1000 * spent / max(len(streak["intervals"]), 1), 1),
             baseline_ms=None if base is None else round(1000 * base, 1),
             extra_s=None if extra is None else round(extra, 2), write_s=round(written, 2),
             records=len(streak["records"]), bytes=size, total_bytes=_STREAK["bytes"],
             reqs_dropped=streak["dropped"], reqs_joined=streak["joined"], lanes_missing=streak["missing"],
             idle_steps=streak["idle"])
    if not streak["records"]:
        announce(f"pick_streak_{streak['index']}_empty", steps=streak["steps"],
                 why="the runner forced the streak's steps off the full graphs but no target QSA layer saw a "
                     "chosen request among its decode lanes, so the file holds the step clock only")
    return path


def streak_exit():
    if _STREAK["open"] is not None:
        streak_close(ends=False, cut="exit")


def streak_copy(counts, idle):
    copies = _STREAK["copies"]
    if copies is None or capturing():
        return None
    rows = counts if idle is None else torch.where(idle, torch.zeros_like(counts), counts)
    copies.append(to_cpu(rows).astype(np.int32))
    return copies[-1]


def streak_changes(before, after):
    lane, slot = np.nonzero(before != after)
    return lane.astype(np.int32), slot.astype(np.int32), after[lane, slot].astype(np.int32)


def streak_before(layer, logical, meta):
    streak = _STREAK["open"]
    wanted = _STREAK["slots"]
    if streak is None or not wanted or is_draft(layer) or capturing():
        return None
    decodes = int(meta.num_decodes)
    cache = layer.hisparse_cache
    runtime = getattr(cache, "runtime", None)
    states = getattr(runtime, "request_state_indices", None)
    table = getattr(cache, "source_block_table", None)
    if decodes == 0 or states is None or table is None:
        return None
    seen = to_cpu(states[:decodes]).tolist()
    lanes = [at for at, state in enumerate(seen) if state in wanted]
    streak["missing"] += len(wanted) - len(lanes)
    if not lanes:
        return None
    starts = to_cpu(meta.query_start_loc[:decodes + 1].to(torch.long))
    rows = np.concatenate([np.arange(starts[at], starts[at + 1]) for at in lanes])
    offsets = np.concatenate([[0], np.cumsum([starts[at + 1] - starts[at] for at in lanes])])
    picks = to_cpu(logical.index_select(0, torch.as_tensor(rows, dtype=torch.long, device=logical.device))[:, :-1]
                   .to(torch.int32))
    number = layer_number(layer.layer_name)
    ids = [wanted[seen[at]] for at in lanes]
    group = runtime.index_group
    slots = torch.as_tensor([seen[at] for at in lanes], dtype=torch.long, device=group.device_global_indices.device)
    before = to_cpu(group.device_global_indices.index_select(0, slots))
    gpu = getattr(cache, "block_table", None)
    view = getattr(cache, "view", None)
    record = {"layer": number, "engine": _STREAK["engine"], "t": _STREAK["elapsed"],
              "ids": np.asarray(ids, dtype=np.int32), "states": np.asarray([seen[at] for at in lanes], dtype=np.int32),
              "seq_lens": to_cpu(meta.seq_lens[:decodes])[lanes].astype(np.int32), "starts": offsets.astype(np.int32),
              "picks": picks,
              "block_table": to_cpu(table.index_select(0, torch.as_tensor(lanes, dtype=torch.long,
                                                                           device=table.device)).to(torch.int32)),
              "gpu_table": (np.zeros((len(lanes), 0), dtype=np.int32) if gpu is None else
                            to_cpu(gpu.index_select(0, torch.as_tensor(lanes, dtype=torch.long, device=gpu.device))
                                   .to(torch.int32))),
              "block_size": int(meta.block_size), "gpu_block_size": int(getattr(view, "block_size", 0) or 0),
              "query_len": int(meta.decode_max_query_len), "decodes": decodes, "region": int(runtime.region_stride)}
    ranks_record(layer, record, rows)
    _STREAK["copies"] = []
    seeds = [at for at, ident in enumerate(ids) if (ident, number) not in streak["last"]]
    if seeds:
        record["seed_lane"] = np.asarray(seeds, dtype=np.int32)
        record["seed_resident"] = before[seeds]
        record["seed_lru"] = to_cpu(group.lru_slots.index_select(0, slots[torch.as_tensor(seeds, dtype=torch.long,
                                                                                          device=slots.device)]))
    return {"record": record, "before": before, "slots": slots, "group": group, "number": number, "lanes": lanes}


def streak_after(layer, held):
    streak = _STREAK["open"]
    copies, _STREAK["copies"] = _STREAK["copies"], None
    if held is None or streak is None:
        return
    record = held["record"]
    lanes = np.asarray(held["lanes"], dtype=np.int64)
    record["copy_counts"] = (np.stack([one[lanes] for one in copies]).astype(np.int32) if copies
                             else np.zeros((0, lanes.shape[0]), dtype=np.int32))
    before = held["before"]
    after = to_cpu(held["group"].device_global_indices.index_select(0, held["slots"]))
    record["post_lane"], record["post_slot"], record["post_value"] = streak_changes(before, after)
    lanes, slots, values = [], [], []
    for at, ident in enumerate(record["ids"].tolist()):
        key = (ident, held["number"])
        last = streak["last"].get(key)
        if last is not None:
            changed = np.nonzero(last != before[at])[0]
            lanes.append(np.full(changed.shape[0], at, dtype=np.int32))
            slots.append(changed.astype(np.int32))
            values.append(before[at, changed].astype(np.int32))
        streak["last"][key] = after[at]
    empty = np.zeros(0, dtype=np.int32)
    record["pre_lane"] = np.concatenate(lanes) if lanes else empty
    record["pre_slot"] = np.concatenate(slots) if slots else empty
    record["pre_value"] = np.concatenate(values) if values else empty
    streak["groups"][held["number"]] = held["group"]
    streak["records"].append(record)
    streak["bytes"] += sum(np.asarray(value).nbytes for value in record.values())


def tail_tokens():
    return max(0, env_int(TAIL_ENV, 0))


def next_layer(layer):
    if layer not in _ORDER:
        return None
    for later in _ORDER[_ORDER.index(layer) + 1:]:
        if not is_draft(later):
            return later
    return None


def tail_picks(seq_lens, query_len, tokens, width):
    ends = (seq_lens.to(torch.long) - query_len).clamp_min(0)
    span = torch.arange(width, dtype=torch.long, device=seq_lens.device)
    positions = (ends - tokens).unsqueeze(1) + span.unsqueeze(0)
    valid = (span.unsqueeze(0) < tokens) & (positions >= 0) & (positions < ends.unsqueeze(1))
    return torch.where(valid, positions, torch.full_like(positions, -1)).to(torch.int32).contiguous()


def tail_room(runtime, query_len, width):
    queries = 1 if row_attend() else query_len
    return int(runtime.region_stride) - queries * width


def arange_of(count, device):
    key = (str(device), int(count))
    held = _ARANGE.get(key)
    if held is None:
        held = torch.arange(count, dtype=torch.int32, device=device)
        _ARANGE[key] = held
    return held


def current():
    from vllm.utils.torch_utils import current_stream

    return current_stream()


def prefetch_next(layer, logical, meta):
    tokens = tail_tokens()
    if tokens <= 0 or capturing():
        return None
    target = next_layer(layer)
    decodes = int(meta.num_decodes)
    if target is None or decodes == 0:
        return None
    handle = target.hisparse_cache
    runtime = handle.runtime
    table = getattr(handle, "source_block_table", None)
    if table is None or getattr(runtime, "request_state_indices", None) is None:
        return None
    width = logical.shape[1] - 1
    room = tail_room(runtime, meta.decode_max_query_len, width)
    tokens = min(tokens, width, room)
    if tokens <= 0:
        announce("tail_prefetch_no_room", region=runtime.region_stride, width=width,
                 query_len=meta.decode_max_query_len,
                 why="the hot region holds no slots beyond one step's picks, so a prefetch would evict them")
        return None
    picks = tail_picks(meta.seq_lens[:decodes], meta.decode_max_query_len, tokens, width)
    group = runtime.index_group
    rows = slice(0, decodes)
    group.copy_stream.wait_stream(current())
    with group.copy_stream:
        runtime._resolve_residency(resident=handle, req_id_per_token=arange_of(decodes, picks.device),
                                   block_table=table, topk_indices=picks, block_size=int(meta.block_size),
                                   return_valid_counts=False, shared_rows=rows)
        runtime._swap_rows(rows)
    announce("tail_prefetch", layer=layer.layer_name, target=target.layer_name, tokens=tokens, decodes=decodes,
             why="the newest tokens of each request are made hot for the next index layer on its copy stream "
                 "while this layer and the layers between compute")
    return target.layer_name


def decode_attention(layer, impl, query, logical, meta, out, gate, before=None):
    if getattr(meta, "fixed", None) is not None:
        return decode_attention_once(layer, impl, query, logical, meta, out, gate, before)
    pick_step(layer)
    record = pick_before(layer, logical, meta)
    streak = streak_before(layer, logical, meta) if _STREAK["open"] is not None else None
    prefetch_next(layer, logical, meta)
    decode_attention_once(layer, impl, query, logical, meta, out, gate, before)
    if _RESET["fresh"] is not None:
        count_fresh_misses(layer, meta)
    if _UNION["hist"] is not None:
        count_union(layer, logical, meta)
    pick_after(layer, record)
    streak_after(layer, streak)


def prefetch_blocks():
    return max(0, env_int(PREFETCH_ENV, 0))


def ranks_width():
    return max(0, env_int(RANKS_ENV, 0))


def near_wanted():
    return prefetch_blocks() > 0 or ranks_width() > 0


def prefetch_programs(device):
    said = env_int(PREFETCH_PROGRAMS_ENV, 0)
    return said if said > 0 else device_sms(device)


def prefetch_aside(reason, **fields):
    _PREFETCH["aside"] += 1
    announce(f"prefetch_aside_{reason}", **fields)
    return None


def pure_decode(indexer):
    held = indexer._metadata()
    if held is None:
        return False
    compressed = held[1]
    decodes = int(getattr(compressed, "num_decode_tokens", 0) or 0)
    return (decodes > 0 and decodes == int(compressed.num_actual_tokens)
            and getattr(compressed, "bucket", None) is None)


def install_near(layer, group):
    import importlib

    module = importlib.import_module(INDEXER_OPS_MODULE)
    if not getattr(module, NEAR_MARK, False):
        topk = module._topk
        select = module.qsa_select_paged_decode

        def near_select(*args, **kwargs):
            _NEAR["decode"] = True
            try:
                return select(*args, **kwargs)
            finally:
                _NEAR["decode"] = False

        def near_topk(*args, **kwargs):
            result = topk(*args, **kwargs)
            if _NEAR["decode"]:
                _NEAR["decode"] = False
                near_after(args[0], args[1], args[4], int(args[3]))
            return result

        module._topk = near_topk
        module.qsa_select_paged_decode = near_select
        setattr(module, NEAR_MARK, True)
    indexer = layer.indexer
    kind = type(indexer)

    def near_forward(*args, **kwargs):
        _NEAR["layer"] = layer
        _NEAR["pure"] = pure_decode(indexer)
        try:
            return kind.forward(indexer, *args, **kwargs)
        finally:
            _NEAR["layer"] = None
            _NEAR["pure"] = False

    indexer.forward = near_forward
    blocks = prefetch_blocks()
    if blocks:
        rows = max(1, int(group.physical_topk_indices.shape[0]) - 1)
        _NEAR["buffers"][layer.layer_name] = torch.full((rows, blocks), -1, dtype=torch.int32,
                                                        device=layer.topk_indices_buffer.device)
    announce("near_select", knob=PREFETCH_ENV, blocks=blocks, ranks_knob=RANKS_ENV, ranks=ranks_width(),
             module=INDEXER_OPS_MODULE,
             why="the decode index top-k also yields the blocks ranked just below its cut, the rows a later step is "
                 "likely to pick, without a second full top-k")
    return True


def near_after(logits, visible, picked, ratio):
    layer = _NEAR["layer"]
    if layer is None or is_draft(layer):
        return None
    name = layer.layer_name
    if ranks_width() and _STREAK["open"] is not None and _STREAK["slots"] and not capturing():
        record_ranks(name, logits, visible, picked, ratio)
    blocks = prefetch_blocks()
    if not blocks or not _NEAR["pure"] or next_layer(layer) is None or not fused():
        return None
    held = _NEAR["buffers"].get(name)
    rows = int(picked.shape[0])
    if held is None or rows > held.shape[0] or logits.stride(1) != 1 or picked.stride(1) != 1:
        return prefetch_aside("near_shape", layer=name, rows=rows, logits=tuple(logits.stride()),
                              picked=tuple(picked.stride()))
    if not fused_ready(("near", int(picked.shape[1]), blocks, str(logits.dtype), str(visible.dtype))):
        return prefetch_aside("near_cold", layer=name, rows=rows,
                              why="a captured forward reached a near-miss selection no eager forward had launched, "
                                  "so this graph prefetches nothing")
    stream = prefetch_stream(logits.device)
    stream.wait_stream(current())
    with torch.cuda.stream(stream):
        fused_ops().near(logits, visible, picked, held[:rows])
    for tensor in (logits, visible, picked):
        tensor.record_stream(stream)
    _PREFETCH["open"] = True
    _NEAR["ready"][name] = rows
    _NEAR["selects"] += 1
    if "near_select_first" not in _ANNOUNCED and not capturing():
        stream.synchronize()
        chosen = held[:rows]
        announce("near_select_first", layer=name, rows=rows, blocks=blocks, columns=int(logits.shape[1]),
                 chosen=int((chosen >= 0).sum()), of=int(chosen.numel()),
                 visible_max=int(visible.max()) if torch.is_tensor(visible) and visible.numel() else -1,
                 why="each decode row's blocks ranked just below the index cut are chosen on the prefetch stream "
                     "while this layer's own swap-in and attention run")
    return rows


def record_ranks(name, logits, visible, picked, ratio):
    columns = int(logits.shape[1])
    width = min(columns, int(picked.shape[1]) + ranks_width())
    span = torch.arange(columns, device=logits.device).view(1, -1)
    masked = logits.masked_fill(span >= visible.view(-1, 1).to(torch.long), float("-inf"))
    values, index = torch.topk(masked, width, dim=1, sorted=True)
    ranked = torch.where(values > float("-inf"), index, torch.full_like(index, -1)).to(torch.int32)
    _NEAR["ranks"][name] = (ranked, int(picked.shape[1]), int(ratio))
    announce("pick_ranks", knob=RANKS_ENV, layer=name, rows=int(picked.shape[0]), width=width,
             cut=int(picked.shape[1]), ratio=int(ratio),
             why="a pick-log streak step also records each tracked decode row's index blocks in score order past "
                 "the cut, so the rank a later step's missed rows held at this step can be read offline")


def ranks_record(layer, record, rows):
    held = _NEAR["ranks"].pop(layer.layer_name, None)
    if held is None:
        return False
    ranked, cut, ratio = held
    if not len(rows) or int(np.max(rows)) >= ranked.shape[0]:
        announce("pick_ranks_rows", layer=layer.layer_name, rows=int(ranked.shape[0]),
                 why="the tracked rows lie past the rows the decode index ranked, so this record has no ranks")
        return False
    index = torch.as_tensor(rows, dtype=torch.long, device=ranked.device)
    record["ranks"] = to_cpu(ranked.index_select(0, index))
    record["ranks_cut"] = cut
    record["ranks_ratio"] = ratio
    return True


def prefetch_stream(device):
    held = _PREFETCH["stream"]
    if held is None or held.device != device:
        held = torch.cuda.Stream(device=device)
        _PREFETCH["stream"] = held
    return held


def prefetch_scratch(runtime, decodes, width, picks, device):
    held = _PREFETCH["scratch"]
    if (held is not None and held["device"] == device and held["lanes"] >= decodes and held["width"] >= width
            and held["picks"] >= picks):
        return held
    if capturing():
        return None
    lanes = max(int(decodes), int(getattr(runtime, "max_num_reqs", 0) or 0), held["lanes"] if held else 0)
    width = max(int(width), held["width"] if held else 0)
    picks = max(int(picks), held["picks"] if held else 0)
    flat = {name: torch.empty(lanes * width, dtype=torch.int32, device=device)
            for name in ("union", "physical", "host_rows", "device_rows")}
    held = {"device": device, "lanes": lanes, "width": width, "picks": picks,
            "topk_rows": torch.full((lanes * width,), -1, dtype=torch.int32, device=device),
            "back": torch.empty(lanes * picks, dtype=torch.int32, device=device),
            "listed": torch.full((lanes * picks,), -1, dtype=torch.int32, device=device),
            "counts": torch.zeros(lanes, dtype=torch.int32, device=device),
            "stats": torch.zeros(2, dtype=torch.uint64, device=device),
            "tally": torch.zeros(4, dtype=torch.int64, device=device), **flat}
    _PREFETCH["scratch"] = held
    _PREFETCH["tally"] = held["tally"]
    return held


def scratch_view(flat, rows, width):
    return flat[:rows * width].view(rows, width)


def prefetch_issue(layer, runtime, handle, request_ids, block_table, meta, held, fits, states, region, cap):
    name = layer.layer_name
    rows = _NEAR["ready"].pop(name, None)
    if rows is None:
        return None
    blocks = prefetch_blocks()
    decodes = int(meta.num_decodes)
    query_len = int(meta.decode_max_query_len)
    ratio = int(getattr(layer.indexer, "compress_ratio", 0) or 0)
    if ratio <= 0 or rows != decodes * query_len:
        return prefetch_aside("rows", layer=name, rows=rows, decodes=decodes, query_len=query_len, ratio=ratio)
    picks = query_len * blocks * ratio
    width = min(picks, int(cap))
    device = held.device
    ops = fused_ops()
    positions = int(block_table.shape[1]) * int(meta.block_size)
    layout = ops.bitmap_layout(positions)
    scratch = prefetch_scratch(runtime, decodes, width, picks, device)
    if layout is None or scratch is None or width <= 0:
        return prefetch_aside("scratch", layer=name, decodes=decodes, width=width, positions=positions,
                              why="the prefetch buffers are sized on an eager forward and none had run")
    if not fused_ready(("prefetch", blocks, ratio, query_len, width, positions, str(device))):
        return prefetch_aside("cold", layer=name, decodes=decodes,
                              why="a captured forward reached a prefetch shape no eager forward had launched, so "
                                  "this graph prefetches nothing")
    spread = (prefetch_programs(device) if spread_per_sm() > 0
              and ops.spread_ready(runtime.host_cache, runtime.hot.cache) else 0)
    union = scratch_view(scratch["union"], decodes, width)
    back = scratch_view(scratch["back"], decodes, picks)
    topk_rows = scratch_view(scratch["topk_rows"], decodes, width)
    physical = scratch_view(scratch["physical"], decodes, width)
    host_rows = scratch_view(scratch["host_rows"], decodes, width)
    device_rows = scratch_view(scratch["device_rows"], decodes, width)
    counts = scratch["counts"][:decodes]
    group = runtime.index_group
    stream = prefetch_stream(device)
    stream.wait_stream(current())
    with torch.cuda.stream(stream):
        ops.union(ops.near_positions(_NEAR["buffers"][name][:rows], ratio), decodes, query_len, query_len,
                  blocks * ratio, union, back, layout)
        used = (held >= 0).sum(1, dtype=torch.int32)
        budget = torch.where(fits, int(region) - used, torch.zeros_like(used)).clamp_(0, width)
        budget.masked_fill_(states.index_select(0, request_ids.to(torch.long)) < 0, 0)
        union.masked_fill_(torch.arange(width, dtype=torch.int32, device=device).view(1, -1) >= budget.view(-1, 1),
                           -1)
        idle = budget <= 0
        live = ops.spill_states(states, request_ids, idle)
        resolve_residency(
            runtime.host_cache, runtime.hot.cache, runtime.hot_block_table, union, topk_rows,
            group.device_global_indices, group.lru_slots, live, int(region), None, scratch["stats"], physical,
            runtime.hot.attention_block_stride, request_ids, block_table, int(meta.block_size), None, None, host_rows,
            device_rows, counts, handle.block_table, handle.view.block_size if handle.view is not None else 0, 0)
        counts.masked_fill_(idle, 0)
        run_gap = run_merge_active(runtime)
        gather_compact(runtime, host_rows, device_rows, probe_counts(counts, "prefetch"), 0,
                       0 if run_gap >= 0 else spread, site="prefetch", run=run_gap)
        scratch["tally"].add_(torch.stack(((budget > 0).sum(dtype=torch.int64), (union >= 0).sum(dtype=torch.int64),
                                           counts.sum(dtype=torch.int64),
                                           torch.ones((), dtype=torch.int64, device=device))))
    for tensor in (held, fits):
        tensor.record_stream(stream)
    _PREFETCH["open"] = True
    _PREFETCH["issued"] += 1
    announce("prefetch", knob=PREFETCH_ENV, layer=name, decodes=decodes, blocks=blocks, width=width,
             programs=spread or "stock",
             why="after this layer's attention has read every row it needs, the rows ranked just below each decode "
                 "row's index cut are swapped into the coldest hot slots of their lane on the prefetch stream, at "
                 "most the region minus this step's union so no row this step picked is evicted, while the layers "
                 "up to the next index layer compute; that layer waits for it before its own swap-in")
    return width


def prev_on():
    return knob_on(PREV_ENV)


def prev_once():
    return prev_on() and knob_on(PREV_ONCE_ENV)


def prev_buffers(runtime, decodes, held_width, region, states, device):
    held = _PREV_ONCE["buffers"]
    ops = fused_ops()
    rb = ops.pow2(runtime.index_group.device_global_indices.shape[1])
    if (held is not None and held["device"] == device and held["lanes"] >= decodes and held["region"] >= region
            and held["rb"] >= rb and held["states"] >= states.numel()):
        return held
    if capturing():
        return None
    lanes = max(int(decodes), int(getattr(runtime, "max_num_reqs", 0) or 0), held["lanes"] if held else 0)
    region = max(int(region), held["region"] if held else 0)
    rb = max(rb, held["rb"] if held else 0)
    count = max(int(states.numel()), held["states"] if held else 0)
    held = {"device": device, "lanes": lanes, "region": region, "rb": rb, "states": count,
            "owned": torch.empty(lanes * rb, dtype=torch.int32, device=device),
            "copied": torch.empty(lanes * ops.pow2(region), dtype=torch.int32, device=device),
            "captured": torch.empty((lanes, region), dtype=torch.int32, device=device),
            "cap_counts": torch.zeros((PREV_CAPS, lanes), dtype=torch.int32, device=device),
            "idle": torch.zeros(lanes, dtype=torch.bool, device=device),
            "live": torch.empty(count, dtype=states.dtype, device=device),
            "fetched": torch.zeros((), dtype=torch.int64, device=device)}
    _PREV_ONCE["buffers"] = held
    _PREV_ONCE["layouts"] = {}
    announce("prefetch_prev_once", knob=PREV_ONCE_ENV, lanes=lanes, region=region,
             bytes=sum(one.numel() * one.element_size() for one in held.values() if isinstance(one, torch.Tensor)),
             why="the previous index layer's prefetch plans in one Triton kernel a lane over buffers allocated once "
                 "at the most lanes before capture and shared by every layer and graph, and gathers with a kernel "
                 "that allocates nothing, instead of lane-by-region temporaries on the side stream of every graph")
    return held


def prev_layout(widths, device):
    key = (tuple(widths), str(device))
    held = _PREV_ONCE["layouts"].get(key)
    if held is None:
        if capturing():
            return None
        bounds = [0]
        for width in widths:
            bounds.append(bounds[-1] + int(width))
        held = torch.tensor(bounds, dtype=torch.int32, device=device)
        _PREV_ONCE["layouts"][key] = held
    return held


def prev_capture(host_rows, counts):
    held = _PREV["capture"]
    if held is None:
        return
    if not prev_once():
        held.append((host_rows.clone(), counts.clone()))
        return
    buffers = _PREV_ONCE["buffers"]
    decodes, width = host_rows.shape
    low = sum(one for one in held if one is not None)
    if (buffers is None or None in held or len(held) >= PREV_CAPS or low + width > buffers["region"]
            or decodes > buffers["lanes"]):
        held.append(None)
        return
    buffers["captured"][:decodes, low:low + width].copy_(host_rows)
    buffers["cap_counts"][len(held), :decodes].copy_(counts)
    held.append(int(width))


def prev_rows():
    return max(1, env_int(PREV_ROWS_ENV, PREV_ROWS))


def prev_scope():
    said = str(os.environ.get(PREV_SCOPE_ENV) or "").strip().lower()
    return said if said in PREV_SCOPES else PREV_SCOPES[0]


def prefetch_armed():
    return prefetch_blocks() > 0 or prev_on() or spec_on()


def spec_on():
    return knob_on(SPEC_ENV)


def prev_counted():
    return prev_on() or spec_on()


def layer_named(name):
    return next((layer for layer in _ORDER if getattr(layer, "layer_name", None) == name), None)


def spec_issue(name, blocks, decodes, query_len, ratio, block_size=None):
    target = layer_named(name)
    if target is None or is_draft(target):
        return prev_aside("spec_target", layer=name)
    owner = getattr(target, "taaf_hisparse_group", None)
    if owner is None or len(owner.caches) != 1:
        return prev_aside("spec_target", layer=name)
    handle = owner.cache(0)
    runtime = handle.runtime
    table = getattr(handle, "source_block_table", None)
    target_states = getattr(runtime, "request_state_indices", None)
    request_ids = getattr(owner, "request_ids", None)
    if table is None or target_states is None or request_ids is None or getattr(runtime, "index_group", None) is None:
        return prev_aside("spec_target", layer=name)
    rows = decodes * query_len
    request_ids = request_ids[:decodes]
    region = int(runtime.region_stride)
    picks = int(blocks.shape[1]) * int(ratio)
    device = blocks.device
    width = min(query_len * picks, region, capped_width(region, smem_limit(device)))
    listed_width = min(prev_rows(), width)
    ops = fused_ops()
    view = getattr(getattr(target, "hisparse_cache", None), "view", None)
    block_size = int(view.block_size) if view is not None else int(block_size or 0)
    if block_size <= 0:
        return prev_aside("spec_target", layer=name)
    positions = int(table.shape[1]) * block_size
    layout = ops.bitmap_layout(positions)
    scratch = prefetch_scratch(runtime, decodes, width, max(listed_width, query_len * picks), device)
    tally = prev_tally(device)
    buffers = prev_buffers(runtime, decodes, width, region, target_states, device)
    own = spec_buffers(runtime, decodes, width, query_len * picks, device)
    if (layout is None or scratch is None or tally is None or buffers is None or own is None or width <= 0
            or rows > blocks.shape[0]):
        return prev_aside("spec_scratch", layer=name, decodes=decodes, width=width,
                          why="the prefetch buffers are sized on an eager forward and none had run")
    if not fused_ready(("spec", query_len, picks, width, listed_width, positions, str(device))):
        return prev_aside("spec_cold", layer=name, decodes=decodes,
                          why="a captured forward reached a prefetch shape no eager forward had launched, so "
                              "this graph prefetches nothing")
    spread = (prefetch_programs(device) if spread_per_sm() > 0
              and ops.spread_ready(runtime.host_cache, runtime.hot.cache) else 0)
    held = scratch_view(own["held"], decodes, width)
    back = scratch_view(own["back"], decodes, query_len * picks)
    fits = own["fits"][:decodes]
    topk_rows = scratch_view(scratch["topk_rows"], decodes, width)
    physical = scratch_view(scratch["physical"], decodes, width)
    host_rows = scratch_view(scratch["host_rows"], decodes, width)
    device_rows = scratch_view(scratch["device_rows"], decodes, width)
    counts = scratch["counts"][:decodes]
    union = scratch_view(scratch["union"], decodes, width)
    listed = scratch_view(scratch["listed"], decodes, listed_width)
    group = runtime.index_group
    stream = prefetch_stream(device)
    stream.wait_stream(current())
    with torch.cuda.stream(stream):
        ops.union(ops.near_positions(blocks[:rows], ratio), decodes, query_len, query_len, picks, held, back, layout)
        idle = buffers["idle"][:decodes]
        ops.prev_plan(held, fits, table, request_ids, target_states, group.device_global_indices, block_size, region,
                      prev_rows(), width, listed_width, union, listed, idle, tally, buffers["owned"],
                      buffers["copied"])
        live = ops.spill_states(target_states, request_ids, idle, out=buffers["live"])
        resolve_residency(
            runtime.host_cache, runtime.hot.cache, runtime.hot_block_table, union, topk_rows,
            group.device_global_indices, group.lru_slots, live, region, None, scratch["stats"], physical,
            runtime.hot.attention_block_stride, request_ids, table, block_size, None, None, host_rows,
            device_rows, counts, handle.block_table, handle.view.block_size if handle.view is not None else 0, 0)
        counts.masked_fill_(idle, 0)
        gather_compact(runtime, host_rows, device_rows, probe_counts(counts, "prefetch"), 0, spread,
                       site="prefetch", run=-1)
        torch.sum(counts, 0, dtype=torch.int64, out=buffers["fetched"])
        tally[PREV_STATS.index("fetched")].add_(buffers["fetched"])
    for tensor in (blocks, request_ids):
        tensor.record_stream(stream)
    _PREFETCH["open"] = True
    _PREV["pending"] = {"target": name, "listed": listed}
    _PREV["issued"] += 1
    announce("prefetch_spec", knob=SPEC_ENV, layer=name, decodes=decodes, width=width, rows=prev_rows(),
             programs=spread or "stock",
             why="before the previous decoder layer's MLP, this index layer's own indexer scores the state that "
                 "enters that MLP; the rows its picks name that this layer's hot region and resident pages do not "
                 "hold are swapped into the coldest hot slots on the prefetch stream while the MLP runs, and this "
                 "layer waits for them before its own swap-in, which still loads every row the guess missed")
    prefetch_close()
    return width


def spec_buffers(runtime, decodes, width, picks, device):
    held = _SPEC["buffers"]
    if (held is not None and held["device"] == device and held["lanes"] >= decodes and held["width"] >= width
            and held["picks"] >= picks):
        return held
    if capturing():
        return None
    lanes = max(int(decodes), int(getattr(runtime, "max_num_reqs", 0) or 0), held["lanes"] if held else 0)
    width = max(int(width), held["width"] if held else 0)
    picks = max(int(picks), held["picks"] if held else 0)
    held = {"device": device, "lanes": lanes, "width": width, "picks": picks,
            "held": torch.empty(lanes * width, dtype=torch.int32, device=device),
            "back": torch.empty(lanes * picks, dtype=torch.int32, device=device),
            "fits": torch.ones(lanes, dtype=torch.bool, device=device)}
    _SPEC["buffers"] = held
    return held


def prev_aside(reason, **fields):
    _PREV["aside"] += 1
    announce(f"prefetch_prev_aside_{reason}", **fields)
    return None


def prev_tally(device):
    held = _PREV["tally"]
    if held is None or held.device != device:
        if capturing():
            return None
        held = torch.zeros(len(PREV_STATS), dtype=torch.int64, device=device)
        _PREV["tally"] = held
    return held


def prev_host_rows(positions, table, request_ids, block_size):
    pages = table.index_select(0, request_ids.to(torch.long)).to(torch.long)
    wide = positions.clamp_min(0).to(torch.long)
    page = wide // int(block_size)
    block = pages.gather(1, page.clamp_max(pages.shape[1] - 1))
    inside = (positions >= 0) & (page < pages.shape[1]) & (block >= 0)
    return block * int(block_size) + wide % int(block_size), inside


def prev_source(rows_of, captured):
    mask = torch.zeros(rows_of.shape, dtype=torch.bool, device=rows_of.device)
    wide = torch.iinfo(torch.long).max
    for host, counts in captured:
        listed = host.to(torch.long)
        live = (torch.arange(host.shape[1], device=host.device).view(1, -1) < counts.view(-1, 1)) & (listed >= 0)
        ordered = torch.where(live, listed, torch.full_like(listed, wide)).sort(1).values
        at = torch.searchsorted(ordered, rows_of).clamp_max(ordered.shape[1] - 1)
        mask |= ordered.gather(1, at) == rows_of
    return mask


def prev_plan(held, fits, resident, states, request_ids, table, block_size, region, rows, width, listed_width,
              captured=None):
    lanes = held.shape[0]
    device = held.device
    rows_of, inside = prev_host_rows(held, table, request_ids, block_size)
    lane_state = states.index_select(0, request_ids.to(torch.long))
    valid = inside & fits.view(-1, 1) & (lane_state.view(-1, 1) >= 0)
    owned = resident.index_select(0, lane_state.clamp_min(0).to(torch.long)).to(torch.long)
    ordered = torch.where(owned >= 0, owned, torch.full_like(owned, torch.iinfo(torch.long).max)).sort(1).values
    at = torch.searchsorted(ordered, rows_of).clamp_max(ordered.shape[1] - 1)
    hit = valid & (ordered.gather(1, at) == rows_of)
    miss = valid & ~hit
    scoped = miss if captured is None else miss & prev_source(rows_of, captured)
    rank = scoped.cumsum(1, dtype=torch.long) - 1
    room = (int(region) - valid.sum(1, keepdim=True, dtype=torch.long)).clamp_min(0)
    keep = scoped & (rank < torch.clamp(room, max=int(rows)))
    kept = keep.sum(1, keepdim=True, dtype=torch.long)
    place = torch.where(keep, rank, kept + hit.cumsum(1, dtype=torch.long) - 1)
    place = torch.where((keep | hit) & (place < width), place, torch.full_like(place, width))
    union = torch.full((lanes, width + 1), -1, dtype=held.dtype, device=device)
    union.scatter_(1, place, held)
    spot = torch.where(keep, rank, torch.full_like(rank, listed_width))
    listed = torch.full((lanes, listed_width + 1), -1, dtype=held.dtype, device=device)
    listed.scatter_(1, spot, held)
    return union[:, :width], listed[:, :listed_width], kept.view(-1) <= 0, valid, hit, miss, scoped, keep


def prev_issue(layer, request_ids, meta, held, fits, states, cap, captured=None):
    target = next_layer(layer)
    if target is None:
        return None
    owner = getattr(target, "taaf_hisparse_group", None)
    if owner is None or len(owner.caches) != 1:
        return prev_aside("target", layer=target.layer_name)
    handle = owner.cache(0)
    runtime = handle.runtime
    table = getattr(handle, "source_block_table", None)
    target_states = getattr(runtime, "request_state_indices", None)
    if table is None or target_states is None or getattr(runtime, "index_group", None) is None:
        return prev_aside("target", layer=target.layer_name)
    decodes = int(meta.num_decodes)
    region = int(runtime.region_stride)
    width = min(int(held.shape[1]), int(cap), region)
    listed_width = min(prev_rows(), int(held.shape[1]))
    device = held.device
    ops = fused_ops()
    scope = prev_scope()
    if scope == "missed" and not captured:
        return prev_aside("source", layer=layer.layer_name,
                          why="the rows this layer's own swap-in copied were not kept, so none is known to be "
                              "worth bringing to the next layer")
    scratch = prefetch_scratch(runtime, decodes, width, listed_width, device)
    tally = prev_tally(device)
    if scratch is None or tally is None or width <= 0:
        return prev_aside("scratch", layer=target.layer_name, decodes=decodes, width=width,
                          why="the prefetch buffers are sized on an eager forward and none had run")
    if not fused_ready(("prev", width, listed_width, str(device))):
        return prev_aside("cold", layer=target.layer_name, decodes=decodes,
                          why="a captured forward reached a prefetch shape no eager forward had launched, so "
                              "this graph prefetches nothing")
    spread = (prefetch_programs(device) if spread_per_sm() > 0
              and ops.spread_ready(runtime.host_cache, runtime.hot.cache) else 0)
    topk_rows = scratch_view(scratch["topk_rows"], decodes, width)
    physical = scratch_view(scratch["physical"], decodes, width)
    host_rows = scratch_view(scratch["host_rows"], decodes, width)
    device_rows = scratch_view(scratch["device_rows"], decodes, width)
    counts = scratch["counts"][:decodes]
    union = scratch_view(scratch["union"], decodes, width)
    listed = scratch_view(scratch["listed"], decodes, listed_width)
    group = runtime.index_group
    stream = prefetch_stream(device)
    if prev_once():
        buffers = prev_buffers(runtime, decodes, held.shape[1], region, target_states, device)
        if buffers is None:
            return prev_aside("once_buffers", layer=target.layer_name, decodes=decodes,
                              why="the prefetch-once buffers are sized on an eager forward and none had run")
        found = None
        if scope == "missed":
            if None in captured:
                return prev_aside("once_capture", layer=layer.layer_name, parts=len(captured),
                                  why="this layer's swap-in lists did not fit the prefetch-once capture buffer")
            layout = prev_layout(captured, device)
            if layout is None:
                return prev_aside("once_cold", layer=layer.layer_name, widths=tuple(captured),
                                  why="a captured forward reached capture widths no eager forward had laid out")
            found = (buffers["captured"], buffers["cap_counts"], layout, len(captured))
        if not fused_ready(("prev_once", width, listed_width, int(held.shape[1]), region, scope, str(device))):
            return prev_aside("cold", layer=target.layer_name, decodes=decodes,
                              why="a captured forward reached a prefetch shape no eager forward had launched, so "
                                  "this graph prefetches nothing")
        stream.wait_stream(current())
        with torch.cuda.stream(stream):
            idle = buffers["idle"][:decodes]
            ops.prev_plan(held, fits, table, request_ids, target_states, group.device_global_indices,
                          int(meta.block_size), region, prev_rows(), width, listed_width, union, listed, idle, tally,
                          buffers["owned"], buffers["copied"], found)
            live = ops.spill_states(target_states, request_ids, idle, out=buffers["live"])
            resolve_residency(
                runtime.host_cache, runtime.hot.cache, runtime.hot_block_table, union, topk_rows,
                group.device_global_indices, group.lru_slots, live, region, None, scratch["stats"], physical,
                runtime.hot.attention_block_stride, request_ids, table, int(meta.block_size), None, None, host_rows,
                device_rows, counts, handle.block_table, handle.view.block_size if handle.view is not None else 0, 0)
            counts.masked_fill_(idle, 0)
            gather_compact(runtime, host_rows, device_rows, probe_counts(counts, "prefetch"), 0, spread,
                           site="prefetch", run=-1)
            torch.sum(counts, 0, dtype=torch.int64, out=buffers["fetched"])
            tally[PREV_STATS.index("fetched")].add_(buffers["fetched"])
        return prev_issued(layer, target, held, fits, request_ids, stream, listed, decodes, width, scope, spread)
    stream.wait_stream(current())
    with torch.cuda.stream(stream):
        prev_fetch(runtime, handle, table, target_states, request_ids, int(meta.block_size), held, fits, region,
                   width, listed_width, scratch, tally, spread, captured if scope == "missed" else None)
    return prev_issued(layer, target, held, fits, request_ids, stream, listed, decodes, width, scope, spread)


def prev_fetch(runtime, handle, table, target_states, request_ids, block_size, held, fits, region, width,
               listed_width, scratch, tally, spread, captured=None):
    decodes = int(held.shape[0])
    device = held.device
    ops = fused_ops()
    group = runtime.index_group
    topk_rows = scratch_view(scratch["topk_rows"], decodes, width)
    physical = scratch_view(scratch["physical"], decodes, width)
    host_rows = scratch_view(scratch["host_rows"], decodes, width)
    device_rows = scratch_view(scratch["device_rows"], decodes, width)
    counts = scratch["counts"][:decodes]
    union = scratch_view(scratch["union"], decodes, width)
    listed = scratch_view(scratch["listed"], decodes, listed_width)
    chosen, missed, idle, valid, hit, miss, scoped, keep = prev_plan(
        held, fits, group.device_global_indices, target_states, request_ids, table, block_size, region,
        prev_rows(), width, listed_width, captured)
    union.copy_(chosen)
    listed.copy_(missed)
    live = ops.spill_states(target_states, request_ids, idle)
    resolve_residency(
        runtime.host_cache, runtime.hot.cache, runtime.hot_block_table, union, topk_rows,
        group.device_global_indices, group.lru_slots, live, region, None, scratch["stats"], physical,
        runtime.hot.attention_block_stride, request_ids, table, block_size, None, None, host_rows,
        device_rows, counts, handle.block_table, handle.view.block_size if handle.view is not None else 0, 0)
    counts.masked_fill_(idle, 0)
    run_gap = run_merge_active(runtime)
    gather_compact(runtime, host_rows, device_rows, probe_counts(counts, "prefetch"), 0,
                   0 if run_gap >= 0 else spread, site="prefetch", run=run_gap)
    zero = torch.zeros((), dtype=torch.int64, device=device)
    tally.add_(torch.stack(((~idle).sum(dtype=torch.int64), valid.sum(dtype=torch.int64),
                            hit.sum(dtype=torch.int64), miss.sum(dtype=torch.int64),
                            scoped.sum(dtype=torch.int64), keep.sum(dtype=torch.int64),
                            counts.sum(dtype=torch.int64), zero,
                            torch.ones((), dtype=torch.int64, device=device))))
    return listed


def prev_issued(layer, target, held, fits, request_ids, stream, listed, decodes, width, scope, spread):
    for tensor in (held, fits, request_ids):
        tensor.record_stream(stream)
    _PREFETCH["open"] = True
    _PREV["pending"] = {"target": target.layer_name, "listed": listed}
    _PREV["issued"] += 1
    announce("prefetch_prev", knob=PREV_ENV, layer=layer.layer_name, target=target.layer_name, decodes=decodes,
             width=width, rows=prev_rows(), scope=scope, programs=spread or "stock",
             why="after this index layer's attention, the rows its picks name that the next index layer's hot "
                 "region does not hold" + (" and that this layer's own swap-in had to copy" if scope == "missed"
                                           else "") + " are swapped into that region's coldest slots on the "
                 "prefetch stream, at most the knob's rows a lane and the region minus the picks, with the rows it "
                 "does hold made recent so they stay; that layer waits for it before its own swap-in")
    return width


def prev_used(layer, held, fits):
    pending = _PREV["pending"]
    if pending is None or pending["target"] != layer.layer_name:
        return None
    _PREV["pending"] = None
    tally = _PREV["tally"]
    if tally is None:
        return None
    listed = pending["listed"]
    wide = torch.iinfo(held.dtype).max
    ordered = torch.where((held >= 0) & fits.view(-1, 1), held, torch.full_like(held, wide)).sort(1).values
    at = torch.searchsorted(ordered, listed.clamp_min(0).contiguous()).clamp_max(ordered.shape[1] - 1)
    used = ((listed >= 0) & (ordered.gather(1, at) == listed)).sum(dtype=torch.int64)
    tally[PREV_STATS.index("used")].add_(used)
    return used


def report_prev():
    _PREV["starts"] += 1
    _PREV["steps"] += 1
    _PREV["pending"] = None
    if _PREV["tally"] is None or _PREV["starts"] % PREFETCH_EVERY or capturing():
        return None
    stats = dict(zip(PREV_STATS, (int(one) for one in _PREV["tally"].to("cpu", copy=True).tolist())))
    _PREV["tally"].zero_()
    steps = max(_PREV["steps"], 1)
    lanes = max(stats["lane_steps"], 1)
    line = (f"QSA_HISPARSE tally prefetch_prev window={_PREV['lines']} steps={steps} issues={stats['issues']} "
            f"lane_steps={stats['lane_steps']} candidate_rows={stats['candidates']} resident_rows={stats['resident']} "
            f"missed_rows={stats['missed']} scoped_rows={stats['scoped']} kept_rows={stats['kept']} "
            f"fetched_rows={stats['fetched']} "
            f"used_rows={stats['used']} listed_precision={stats['used'] / max(stats['kept'], 1):.3f} "
            f"fetched_rows_per_lane_step={stats['fetched'] / lanes:.1f} "
            f"fetched_rows_per_step={stats['fetched'] / steps:.1f} "
            f"capped_share={(stats['scoped'] - stats['kept']) / max(stats['scoped'], 1):.3f} aside={_PREV['aside']}")
    _PREV["steps"] = 0
    _PREV["lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def prefetch_join():
    event = _PREFETCH["pending"]
    if event is None:
        return False
    _PREFETCH["pending"] = None
    current().wait_event(event)
    return True


def prefetch_close():
    _NEAR["ready"].clear()
    if not _PREFETCH["open"]:
        return False
    _PREFETCH["open"] = False
    event = torch.cuda.Event()
    event.record(_PREFETCH["stream"])
    _PREFETCH["pending"] = event
    return True


def report_prefetch():
    _PREFETCH["starts"] += 1
    _PREFETCH["steps"] += 1
    if (_PREFETCH["tally"] is None or not prefetch_blocks() or _PREFETCH["starts"] % PREFETCH_EVERY
            or capturing()):
        return None
    lanes, requested, fetched, issues = (int(one) for one in _PREFETCH["tally"].to("cpu", copy=True).tolist())
    _PREFETCH["tally"].zero_()
    steps = max(_PREFETCH["steps"], 1)
    line = (f"QSA_HISPARSE tally prefetch window={_PREFETCH['lines']} steps={steps} issues={issues} "
            f"lanes={lanes} requested_rows={requested} fetched_rows={fetched} "
            f"fetched_rows_per_step={fetched / steps:.1f} held_share={(requested - fetched) / max(requested, 1):.3f} "
            f"aside={_PREFETCH['aside']} near_selects={_NEAR['selects']}")
    _PREFETCH["steps"] = 0
    _PREFETCH["lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def union_swap():
    return str(os.environ.get(UNION_SWAP_ENV) or "").strip().lower() in ("1", "on", "true")


def cache_ops():
    return torch.ops._C_cache_ops


def resolve_smem(width, hot_size):
    chunks = (hot_size + 31) // 32
    return 4 * (width + 2 * (chunks + 1) + 2 * width + 3) + 2 * (hot_size + 2 * width)


def smem_limit(device):
    if device.type != "cuda":
        return UNION_SMEM_FALLBACK
    properties = torch.cuda.get_device_properties(device)
    return int(getattr(properties, "shared_memory_per_block_optin", 0) or UNION_SMEM_FALLBACK)


def union_span(query_len, top_k, hot_size, limit):
    span = query_len
    while span > 1 and (query_len % span or span * top_k > hot_size or resolve_smem(span * top_k, hot_size) > limit):
        span -= 1
    return span


def union_rows(picks):
    rows, width = picks.shape
    empty = torch.iinfo(picks.dtype).max
    key = torch.where(picks >= 0, picks, torch.full_like(picks, empty))
    ordered, order = key.sort(dim=1)
    fresh = ordered != empty
    fresh[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    rank = fresh.cumsum(1, dtype=torch.int64) - 1
    union = torch.full((rows, width + 1), -1, dtype=picks.dtype, device=picks.device)
    union.scatter_(1, torch.where(fresh, rank, torch.full_like(rank, width)), ordered.masked_fill(~fresh, -1))
    back = torch.empty_like(rank)
    back.scatter_(1, order, rank.clamp_min(0))
    return union[:, :width].contiguous(), back


def hits_event(runtime):
    event = _HITS["events"].get(id(runtime))
    if event is None:
        event = torch.Event()
        _HITS["events"][id(runtime)] = event
    return event


def hits_first():
    return knob_on(HITS_FIRST_ENV) and not _HITS["aside"]


def hits_programs():
    return max(1, env_int(HITS_FIRST_PROGRAMS_ENV, HITS_FIRST_PROGRAMS))


def hits_aside(why, failed=True):
    if failed:
        selfcheck_fail(HITS_FIRST_ENV, why)
    if not _HITS["aside"]:
        _HITS["aside"] = why
        print(f"HOST_KV hits first stands aside: {why}; the attention waits for the copy as before",
              file=sys.stderr, flush=True)


def hits_fresh(runtime, decodes, width, device):
    key = (id(runtime), str(device), int(width))
    held = _HITS["fresh"].get(key)
    if held is None or held.shape[0] < decodes:
        if capturing():
            return None
        if held is not None:
            _HITS["retired"].append(held)
        held = torch.zeros((max(decodes, int(getattr(runtime, "max_num_reqs", 0) or 0)), int(width)),
                           dtype=torch.int8, device=device)
        _HITS["fresh"][key] = held
    return held[:decodes]


def union_event(runtime):
    event = _UNION_SWAP["events"].get(id(runtime))
    if event is None:
        event = torch.Event()
        _UNION_SWAP["events"][id(runtime)] = event
    return event


def resolve_residency(*args):
    cache_ops().hisparse_resolve_residency(*args)


def one_launch_on():
    return knob_on(ONE_LAUNCH_ENV)


def hit_skip():
    said = str(os.environ.get(HIT_SKIP_ENV) or "").strip()
    if said not in ("1", "2") or _ONE["skip_aside"]:
        return 0
    return int(said)


def stage_tile():
    said = str(os.environ.get(STAGE_TILE_ENV) or "").strip()
    if not said.isdigit() or _STAGE_TILE["aside"]:
        return 0
    rows = int(said)
    return rows if rows >= 2 and rows & (rows - 1) == 0 else 0


def one_pack():
    return knob_on(ONE_PACK_ENV) and not _ONE["pack_aside"]


def one_hash():
    return knob_on(ONE_HASH_ENV) and not _ONE["hash_aside"]


def flat_copy_on():
    return knob_on(FLAT_COPY_ENV) and not _FLAT_COPY["aside"]


def one_phases_on():
    return str(os.environ.get(STEP_TIMERS_ENV) or "").strip() == "1" and not _ONE["phases_aside"]


def one_cuda_on():
    return flat_copy_on() and fused_ops().one_cuda_wanted()


def one_phases(device):
    if not one_phases_on():
        return None
    if one_cuda_on():
        announce("one_phases_cuda", knob=fused_ops().ONE_CUDA_ENV,
                 why="the CUDA resolve stamps no clock inside its kernel, so the one-launch phase counters stay off "
                     "while it is armed; the step timers' groups, hkv_wait among them, run as before")
        return None
    held = _ONE["phases"]
    if held is None or held.device != device:
        if capturing():
            return None
        held = _ONE["phases"] = torch.zeros(len(fused_ops().ONE_PHASES), dtype=torch.int64, device=device)
        announce("one_phases", knob=STEP_TIMERS_ENV, fields=",".join(fused_ops().ONE_PHASES),
                 why="each one-launch resolve stamps the GPU clock inside the kernel around its sort, its scan of "
                     "the hot region and its tail, and the helper programs around their wait and their copies; one "
                     "line every hundred steps")
    return held


def one_work_fields(values, launches, lanes):
    span = max(values["span_ns"], 1)
    rows = values["rows"]
    copy_wall = max(values["span_ns"] - values["resolve_span_ns"], 0)
    buckets = " ".join(
        f"{name}_n={values[f'b{at}_n']} {name}_ms={values[f'b{at}_ns'] / max(values[f'b{at}_n'], 1) / 1e6:.4f}"
        for at, name in enumerate(("w0", "w1_8", "w9_64", "w65_256", "w257"))
    )
    return (f" resolve_span_ms_per_launch={values['resolve_span_ns'] / launches / 1e6:.4f}"
            f" resolve_span_share={values['resolve_span_ns'] / span:.3f}"
            f" copy_wall_ms_per_launch={copy_wall / launches / 1e6:.4f}"
            f" copy_max_ms={values['copy_max_ns'] / 1e6:.3f}"
            f" rows_per_launch={rows / launches:.1f}"
            f" rows_per_lane={rows / lanes:.2f}"
            f" rows_max_lane={values['rows_max']}"
            f" copy_rows_per_ms={rows / max(copy_wall / 1e6, 1e-9):.0f}"
            f" slow_lanes={values['slow_lanes']}"
            f" slow_lane_rows_mean={values['slow_rows'] / max(values['slow_lanes'], 1):.1f}"
            f" {buckets}"
            f" t={time.time():.1f}")


def report_one_phases():
    _ONE["phase_steps"] += 1
    held = _ONE["phases"]
    if held is None or _ONE["phase_steps"] % GATHER_EVERY or capturing():
        return None
    values = dict(zip(fused_ops().ONE_PHASES, (int(one) for one in held.to("cpu", copy=True).tolist())))
    held.zero_()
    launches = max(values["launches"], 1)
    lanes = max(values["lanes"], 1)
    lanes_seen = max(values["parts"] - values["hollow_parts"] - values["resident_parts"], 1)
    line = (f"QSA_HISPARSE one_phases window={_ONE['phase_lines']} steps={GATHER_EVERY} "
            f"launches_per_step={values['launches'] / GATHER_EVERY:.1f} "
            f"lanes_per_launch={values['lanes'] / launches:.1f} "
            f"lane_resolve_ms_mean={values['resolve_ns'] / lanes / 1e6:.4f} "
            f"lane_resolve_ms_max={values['resolve_max_ns'] / 1e6:.3f} "
            f"sort_ms_per_part={values['sort_ns'] / lanes_seen / 1e6:.4f} "
            f"scan_ms_per_part={values['scan_ns'] / lanes_seen / 1e6:.4f} "
            f"tail_ms_per_part={values['tail_ns'] / lanes_seen / 1e6:.4f} "
            f"copy_ms_per_launch={values['copy_ns'] / launches / 1e6:.3f} "
            f"wait_ms_per_launch={values['wait_ns'] / launches / 1e6:.3f} "
            f"parts_per_launch={values['parts'] / launches:.1f} "
            f"insert_rounds_mean={values.get('insert_rounds', 0) / max(values.get('inserts', 0), 1):.2f} "
            f"hollow_share={values['hollow_parts'] / max(values['parts'], 1):.3f} "
            f"resident_share={values['resident_parts'] / max(values['parts'], 1):.3f} "
            f"pack_share={values['pack_parts'] / lanes_seen:.3f} "
            f"span_ms_per_launch={values['span_ns'] / launches / 1e6:.4f} "
            f"span_ms_per_step={values['span_ns'] / GATHER_EVERY / 1e6:.3f}")
    line += one_work_fields(values, launches, lanes)
    _ONE["phase_lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def _check_world(device, row_bytes, hot_bs, states_dtype, seed, region=40):
    generator = torch.Generator().manual_seed(seed)
    lanes, positions, src_bs, host_rows = 4, 160, 16, 2048
    per = -(-region // hot_bs)
    blocks = lanes * per + 2
    world = SimpleNamespace(lanes=lanes, region=region, src_bs=src_bs, hot_bs=hot_bs)
    world.host = torch.randint(0, 256, (host_rows, row_bytes), generator=generator, dtype=torch.uint8).to(device)
    world.hot = torch.zeros((blocks, hot_bs, row_bytes), dtype=torch.uint8, device=device)
    world.hot_table = torch.randperm(blocks - 2, generator=generator)[:lanes * per].view(lanes, per).to(
        torch.int32).to(device)
    cols = positions // src_bs
    world.src = (torch.randperm(host_rows // src_bs - 1, generator=generator)[:lanes * cols] + 1).view(
        lanes, cols).to(torch.int32).to(device)
    world.dgi = torch.full((lanes + 1, region), -1, dtype=torch.int32, device=device)
    world.lru = torch.arange(region, dtype=torch.int16).repeat(lanes + 1, 1).contiguous().to(device)
    world.states = torch.arange(lanes).flip(0).to(states_dtype).to(device)
    world.request_ids = torch.arange(lanes, dtype=torch.int32, device=device)
    world.stats = torch.zeros(2, dtype=torch.int64, device=device)
    widths = []
    unions = []
    for step in range(3):
        rows = []
        for lane in range(lanes):
            count = 18 if lane % 2 == step % 2 else 34
            picks = torch.randperm(positions, generator=generator)[:count].sort().values.to(torch.int32)
            row = torch.full((36,), -1, dtype=torch.int32)
            row[:count] = picks
            rows.append(row)
        unions.append(torch.stack(rows).to(device))
        widths.append(36)
    world.unions = unions
    return world


def _check_copy(world):
    out = SimpleNamespace(**world.__dict__)
    for name in ("host", "hot", "hot_table", "src", "dgi", "lru", "states", "request_ids", "stats"):
        setattr(out, name, getattr(world, name).clone())
    return out


def _check_launch(world, union, bounds, skip, hash=False):
    ops = fused_ops()
    lanes = union.shape[0]
    widths = [high - low for low, high in bounds]
    total = sum(widths)
    kb, hb, _ = ops.one_blocks(widths, world.region)
    scratch = ops.one_scratch(lanes + 1, kb, hb, total, union.device, hash=hash)
    device = union.device
    flat = lanes * total
    hot_out = torch.full((flat,), -9, dtype=torch.int32, device=device)
    host_out = torch.full((flat,), -9, dtype=torch.int32, device=device)
    dev_out = torch.full((flat,), -9, dtype=torch.int32, device=device)
    counts = torch.full((lanes * len(bounds),), -9, dtype=torch.int32, device=device)
    attn = torch.full((lanes, total), -9, dtype=torch.int32, device=device)
    prefix = [sum(widths[:part]) for part in range(len(widths))]
    layout = torch.tensor([low for low, _ in bounds] + [bounds[-1][1]] + prefix, dtype=torch.int64, device=device)
    tallies = torch.zeros(2, dtype=torch.int64, device=device)
    ops.one_launch(union, layout, len(bounds), world.request_ids, world.states, world.src, world.src_bs, None,
                   world.hot_bs, 0, world.hot_table, world.hot_bs + 3, world.dgi, world.lru, world.host, world.hot,
                   attn, hot_out, host_out, dev_out, counts, None, world.stats, tallies, None, None, scratch,
                   widths[0], False, skip=skip, hash=hash)
    return SimpleNamespace(hot_out=hot_out, attn=attn, counts=counts, widths=widths, prefix=prefix, host_out=host_out,
                           dev_out=dev_out, scratch=scratch)


def _check_plans(want, got, lanes):
    columns = want.scratch["columns"]
    if not torch.equal(want.scratch["plan_count"][:lanes], got.scratch["plan_count"][:lanes]):
        return "plan_count"
    for lane, total in enumerate(want.scratch["plan_count"][:lanes].tolist()):
        for name in ("plan_host", "plan_dev"):
            if not torch.equal(want.scratch[name][lane * columns:lane * columns + total],
                               got.scratch[name][lane * columns:lane * columns + total]):
                return name
    if int((got.scratch["hash"] != fused_ops().ONE_EMPTY).sum()):
        return "the hash table left behind"
    return ""


def one_hash_selfcheck(device, row_bytes, hot_bs, states_dtype, region=40):
    world = _check_world(device, row_bytes, hot_bs, states_dtype, 4099, region)
    plain, hashed = _check_copy(world), _check_copy(world)
    for bounds in ([(0, 20), (20, 36)], [(0, 36)]):
        for union in world.unions:
            want = _check_launch(plain, union, bounds, False)
            got = _check_launch(hashed, union, bounds, False, hash=True)
            for name in ("hot_out", "attn", "counts", "host_out", "dev_out"):
                if not torch.equal(getattr(want, name), getattr(got, name)):
                    return f"the hashed resolve changed {name}"
            broken = _check_plans(want, got, union.shape[0])
            if broken:
                return f"the hashed resolve changed {broken}"
            for name in ("dgi", "lru", "stats", "hot"):
                if not torch.equal(getattr(plain, name), getattr(hashed, name)):
                    return f"the hashed resolve changed the hot state {name}"
    return ""


def one_hash_ready(runtime, union):
    if _ONE["hash_checked"]:
        return one_hash()
    if capturing() or not one_hash():
        return False
    _ONE["hash_checked"] = True
    host = runtime.host_cache
    row_bytes = int(host.shape[-1]) * int(host.element_size())
    try:
        why = one_hash_selfcheck(union.device, row_bytes, int(runtime.hot.cache.shape[1]),
                                 runtime.request_state_indices.dtype, max(40, int(runtime.region_stride)))
    except Exception as failure:
        why = f"{type(failure).__name__}: {str(failure)[:300]}"
    if why:
        _ONE["hash_aside"] = why
        selfcheck_fail(ONE_HASH_ENV, f"the self-check failed: {why}")
        print(f"HOST_KV one hash stands aside: the self-check failed ({why}); the resolve sorts as before",
              file=sys.stderr, flush=True)
        return False
    announce("one_hash", knob=ONE_HASH_ENV, spread=fused_ops().ONE_HASH_SPREAD,
             why="a resolve finds its picks in the hot region through a per-lane hash of their host rows instead of "
                 "sorting every pick and searching the sorted list, and sorts only the misses it copies, in host-row "
                 "order; the self-check matched the sorting kernel on the device")
    return True


def _check_rows(world, union, got):
    lanes = union.shape[0]
    rows = world.hot.view(-1, world.hot.shape[-1])
    for lane in range(lanes):
        state = int(world.states[lane])
        for part, (width, start) in enumerate(zip(got.widths, got.prefix)):
            listed = got.hot_out[lanes * start + lane * width: lanes * start + (lane + 1) * width]
            for column in range(width):
                pick = int(union[lane, start + column])
                slot = int(listed[column])
                if pick < 0 or state < 0:
                    if slot != -1:
                        return f"lane {lane} column {start + column} has no pick but slot {slot}"
                    continue
                physical = int(world.src[lane, pick // world.src_bs])
                source = physical * world.src_bs + pick % world.src_bs
                if slot < 0 or not torch.equal(rows[slot], world.host[source]):
                    return f"lane {lane} pick {pick} reads slot {slot}, not its host row"
    return ""


def one_selfcheck(device, row_bytes, hot_bs, states_dtype, level, region=40):
    world = _check_world(device, row_bytes, hot_bs, states_dtype, 4093, region)
    plain, skipped, wide = _check_copy(world), _check_copy(world), _check_copy(world)
    split = [(0, 20), (20, 36)]
    for union in world.unions:
        want = _check_launch(plain, union, split, False)
        got = _check_launch(skipped, union, split, True)
        for name in ("hot_out", "attn", "counts"):
            if not torch.equal(getattr(want, name), getattr(got, name)):
                return f"the hollow-part skip changed {name}"
        for name in ("dgi", "lru", "stats", "hot"):
            if not torch.equal(getattr(plain, name), getattr(skipped, name)):
                return f"the hollow-part skip changed the hot state {name}"
        if level >= 2:
            whole = _check_launch(wide, union, [(0, 32), (32, 36)], True)
            broken = _check_rows(wide, union, whole) or _check_rows(plain, union, want)
            if broken:
                return broken
    return ""


def hit_skip_ready(runtime, union):
    if _ONE["skip_checked"]:
        return hit_skip()
    if capturing():
        return 0
    _ONE["skip_checked"] = True
    level = hit_skip()
    if not level:
        return 0
    host = runtime.host_cache
    row_bytes = int(host.shape[-1]) * int(host.element_size())
    try:
        why = one_selfcheck(union.device, row_bytes, int(runtime.hot.cache.shape[1]),
                            runtime.request_state_indices.dtype, level, max(40, int(runtime.region_stride)))
    except Exception as failure:
        why = f"{type(failure).__name__}: {str(failure)[:300]}"
    if why:
        _ONE["skip_aside"] = why
        selfcheck_fail(HIT_SKIP_ENV, f"the self-check failed: {why}")
        print(f"HOST_KV hit skip stands aside: the self-check failed ({why}); the resolve runs as before",
              file=sys.stderr, flush=True)
        return 0
    announce("hit_skip", knob=HIT_SKIP_ENV, level=level, tile=fused_ops().ONE_SKIP_TILE,
             why="a lane's resolve part with no picks writes empty rows and leaves the hot state as it is, the hot "
                 "region is scanned in short tiles up to its own size"
                 + (", and the hot split's parts are as wide as the kernel takes, so one part holds every pick of "
                    "a lane whose union fits it" if level >= 2 else "")
                 + "; the self-check matched the plain kernel on the device")
    return level


def one_aside(reason, **fields):
    _ONE["aside"] += 1
    announce(f"one_launch_fallback_{reason}", knob=ONE_LAUNCH_ENV, **fields)
    return None


def one_scratch(runtime, decodes, kb, hb, columns, device, hash=False):
    ops = fused_ops()
    held = _ONE["scratch"]
    if ops.one_scratch_fits(held, decodes, kb, hb, columns, device, hash):
        return held
    if capturing():
        return None
    if hash and ops.one_scratch_fits(held, decodes, kb, hb, columns, device):
        held["hash"] = ops.one_hash_table(held["lanes"], held["kb"], device)
        return held
    lanes = max(int(decodes), int(getattr(runtime, "max_num_reqs", 0) or 0), held["lanes"] if held else 0)
    if held is not None and held["device"] == device:
        kb, hb, columns = max(kb, held["kb"]), max(hb, held["hb"]), max(columns, held["columns"])
        hash = hash or held.get("hash") is not None
    held = ops.one_scratch(lanes, kb, hb, columns, device, hash=hash)
    _ONE["scratch"] = held
    return held


def one_layout(bounds, device):
    key = (tuple(bounds), str(device))
    held = _ONE["layouts"].get(key)
    if held is None:
        if capturing():
            return None
        widths = [high - low for low, high in bounds]
        prefix = [sum(widths[:part]) for part in range(len(widths))]
        held = torch.tensor([low for low, _ in bounds] + [bounds[-1][1]] + prefix, dtype=torch.int64, device=device)
        _ONE["layouts"][key] = held
    return held


def one_plan(runtime, handle, union, rows, bounds, idle):
    if not one_launch_on() or not fused() or not bounds:
        return None
    if gather_probe():
        return one_aside("probe", why="the gather probe zeroes the copy counts between the resolve and the gather")
    host, hot = runtime.host_cache, runtime.hot.cache
    group = runtime.index_group
    dgi, lru = group.device_global_indices, group.lru_slots
    shared = group.shared_topk
    region = int(runtime.region_stride)
    ops = fused_ops()
    decodes = int(union.shape[0])
    widths = [high - low for low, high in bounds]
    if (not isinstance(host, torch.Tensor) or not isinstance(hot, torch.Tensor) or dgi is None or lru is None
            or not ops.run_ready(host, hot) or union.dtype != torch.int32 or union.stride(1) != 1
            or lru.dtype != torch.int16 or dgi.dtype != torch.int32 or not lru.is_contiguous()
            or not dgi.is_contiguous() or tuple(lru.shape[1:]) != (region,) or tuple(dgi.shape[1:]) != (region,)
            or runtime.hot_block_table is None or block_size_of(handle) is None):
        return one_aside("layout", region=region, union=tuple(union.shape),
                         why="the hot state or the row layout is not the one the one-launch kernel reads, so the "
                             "resolve and the gather run as before")
    blocks = ops.one_blocks(widths, region)
    if blocks is None:
        return one_aside("width", widths=tuple(widths), region=region, most=ops.ONE_MAX_BLOCK,
                         most_region=ops.ONE_MAX_REGION,
                         why="a resolve wider than the hot region or the kernel's block runs as before")
    if (decodes * sum(widths) > shared.device_topk_rows[rows].numel()
            or decodes * len(widths) > shared.swap_counts[rows].numel()):
        return one_aside("rows", decodes=decodes, widths=tuple(widths),
                         why="the step's swap rows cannot hold every part's lists, so the parts resolve apart")
    kb, hb, _ = blocks
    hashed = one_hash_ready(runtime, union)
    scratch = one_scratch(runtime, decodes, kb, hb, sum(widths), union.device, hashed)
    layout = one_layout(bounds, union.device)
    if scratch is None or layout is None:
        return one_aside("scratch", decodes=decodes, widths=tuple(widths),
                         why="the one-launch buffers are sized on an eager forward and none had sized them for this "
                             "shape, so this graph resolves and gathers as before")
    pcie = pcie_tally_level() > 0
    skip = hit_skip_ready(runtime, union) > 0
    phases = one_phases(union.device)
    pack = one_pack() and not hashed
    flat = flat_copy_on()
    cuda = flat and ops.one_cuda_wanted()
    while True:
        key = ("one", scratch["kb"], scratch["hb"], int(host.shape[-1]), str(host.dtype), handle.block_table is not None, idle is not None,
               gather_tally_on(), guard_on(), pcie, str(union.device), skip, phases is not None, pack, hashed, flat,
               cuda)
        if key in _ONE["refused"]:
            return one_aside("compile", key=key, why="the one-launch kernel did not build for this shape on this device")
        if key not in _ONE["probed"] and not capturing():
            try:
                ops.one_probe(host, hot, scratch, runtime.request_state_indices.dtype, handle.block_table is not None,
                              idle is not None, gather_tally_on(), guard_on(), pcie, guard_on(), skip=skip,
                              phases=phases, pack=pack, hash=hashed, copy_flat=flat)
            except Exception as failure:
                said = f"{type(failure).__name__}: {str(failure)[:300]}"
                if phases is not None or skip or pack or hashed or flat:
                    selfcheck_fail(STEP_TIMERS_ENV if phases is not None else ONE_HASH_ENV if hashed else
                                   ONE_PACK_ENV if pack else FLAT_COPY_ENV if flat else HIT_SKIP_ENV,
                                   f"the one-launch variant did not build: {said}")
                    if phases is not None:
                        _ONE["phases_aside"] = said
                        phases = None
                    elif hashed:
                        _ONE["hash_aside"] = said
                        hashed = False
                    elif pack:
                        _ONE["pack_aside"] = said
                        pack = False
                    elif flat:
                        _FLAT_COPY["aside"] = said
                        flat = False
                    else:
                        _ONE["skip_aside"] = said
                        skip = False
                    print(f"HOST_KV one-launch variant stands aside: it did not build ({said}); the plain one-launch "
                          "kernel runs", file=sys.stderr, flush=True)
                    continue
                _ONE["refused"].add(key)
                return one_aside("compile", key=key, failure=said,
                                 why="the one-launch kernel did not build for this shape on this device, so the resolve "
                                     "and the gather run as before")
            _ONE["probed"].add(key)
        break
    if not fused_ready(key):
        return one_aside("cold", key=key,
                         why="a captured forward reached a one-launch shape no eager forward had launched, so this "
                             "graph resolves and gathers as before")
    if pack:
        announce("one_pack", knob=ONE_PACK_ENV, block=scratch["kb"], rungs=ops.one_rungs(scratch["kb"]),
                 why="a part's picks that are resident or have no host row leave the resolve's key list before it is "
                     "sorted, and what remains is sorted in the smallest tile that holds it, which leaves the same "
                     "sorted list behind the search of the hot region and the same host-row order in the copy plan")
    if flat and ops.flat_lines_on():
        grouped = ops.group_word(host, hot)
        line = ops.flat_line_words(grouped[0]) if grouped is not None else 0
        announce("one_flat_lines", knob=ops.FLAT_LINES_ENV, line_words=line, group=ops.LINE_GROUP,
                 row_words=int(grouped[0].shape[1]) if grouped is not None else 0,
                 why="a missed group of four host rows that starts on a line is read as whole 128-byte lines split "
                     "among its four entries" if line else
                     "the host rows are not laid out on whole lines, so the flat copy reads row by row as before")
    if flat:
        announce("one_flat_copy", knob=FLAT_COPY_ENV, programs=ops.flat_programs(union.device), tile=ops.FLAT_TILE,
                 why="a resolve's misses of every lane are read from host memory by one kernel spread over the whole "
                     "card in equal shares of the step's list, instead of four programs a lane that start when their "
                     "own lane's resolve ends; it copies the same rows to the same hot slots")
    if cuda:
        announce("one_cuda", knob=ops.ONE_CUDA_ENV, parts=len(widths), widths=tuple(widths),
                 why="the one-launch resolve runs as a CUDA C++ kernel, a block a lane with the picks hashed in "
                     "shared memory and the misses ranked by host row, built and checked against the Triton "
                     "resolve at the first eager forward; it leaves the same lists, copy plan, hot state and tallies")
    return {"bounds": list(bounds), "widths": widths, "scratch": scratch, "layout": layout, "pcie": pcie,
            "skip": skip, "phases": phases, "pack": pack, "hash": hashed, "flat": flat, "cuda": cuda}


def block_size_of(handle):
    if handle.block_table is None:
        return 0
    return handle.view.block_size if handle.view is not None else None


def one_resolve(runtime, handle, union, request_ids, block_table, block_size, rows, states, idle, plan,
                fresh=None):
    ops = fused_ops()
    group = runtime.index_group
    shared = group.shared_topk
    decodes = int(union.shape[0])
    widths, bounds, scratch = plan["widths"], plan["bounds"], plan["scratch"]
    total = sum(widths)
    hot_flat = shared.device_topk_rows[rows].reshape(-1)
    host_flat = shared.swap_host_physical_rows[rows].reshape(-1)
    device_flat = shared.swap_device_physical_rows[rows].reshape(-1)
    counts = shared.swap_counts[rows]
    if len(widths) == 1:
        physical = shared.physical_topk_indices[rows].reshape(-1)[:decodes * total].view(decodes, total)
    else:
        physical = scratch["attn"][:decodes * total].view(decodes, total)
    device = union.device
    stats = group.swap_stats.view(torch.int64) if group.swap_stats is not None else scratch["ticket"].new_zeros(
        2, dtype=torch.int64)
    tally = gather_tally(runtime, device)
    guarded = guard_on()
    guard = guard_tally("gather_rows", device) if guarded else None
    acc = pcie_acc(runtime, device) if plan["pcie"] else None
    mark = None
    if acc is not None:
        _PCIE["gap"], _PCIE["group"] = -1, 0
        if pcie_tally_level() > 1:
            ops.stamp(_PCIE["clock"][0:1], acc[0, -1:], False)
            mark = 0
    ops.one_launch(union, plan["layout"], len(widths), request_ids, runtime.request_state_indices
                   if states is None else states, block_table, block_size, handle.block_table,
                   block_size_of(handle) or 0, 0, runtime.hot_block_table, runtime.hot.attention_block_stride,
                   group.device_global_indices, group.lru_slots, runtime.host_cache, runtime.hot.cache, physical,
                   hot_flat, host_flat, device_flat, counts, idle, stats, tally, guard,
                   acc[0] if acc is not None else None, scratch, widths[0], guarded, skip=plan.get("skip", False),
                   phases=plan.get("phases"), pack=plan.get("pack", False), hash=plan.get("hash", False),
                   flat=plan.get("flat", False), fresh=fresh, copy=fresh is None, cuda=plan.get("cuda", False),
                   viol=guard_tally("resolve_cuda", device) if guarded and plan.get("cuda", False) else None)
    if fresh is not None:
        hits_event(runtime).record(group.copy_stream)
        ops.one_copy(scratch, decodes, runtime.host_cache, runtime.hot.cache, acc[0] if acc is not None else None,
                     plan.get("phases"), programs=hits_programs())
        announce("hits_first", knob=HITS_FIRST_ENV, decodes=decodes, programs=hits_programs(),
                 why="the attention over picks already in the hot region starts once the resolve has placed them, "
                     "while the copy of the misses runs beside it on fewer programs; the misses attend after the "
                     "copy and the two passes merge")
    pcie_tally_done(mark)
    if acc is not None:
        one_pcie(acc, scratch, decodes)
    _ONE["launches"] += 1
    _ONE["parts"] += len(widths)
    announce("one_launch", knob=ONE_LAUNCH_ENV, decodes=decodes, parts=len(widths), widths=tuple(widths),
             share=ops.ONE_SHARE, tile=ops.ONE_TILE,
             why="a resolve's residency decision, its idle mask, tally and row guard, and the host reads of its "
                 "misses in host-row order run as one Triton kernel a resolve, and the hot split's consecutive "
                 "resolves run in that one launch, instead of a resolve, a plan kernel, a sort plan and a gather")
    if _STREAK["copies"] is not None:
        for part in range(len(widths)):
            streak_copy(counts[part * decodes:(part + 1) * decodes], idle)
    if _PREV["capture"] is not None:
        at = 0
        for part, width in enumerate(widths):
            prev_capture(host_flat[at:at + decodes * width].view(decodes, width),
                         counts[part * decodes:(part + 1) * decodes])
            at += decodes * width
    return physical


def one_pcie(acc, scratch, decodes):
    ops = fused_ops()
    mask = _ONE["mask"]
    if mask is None or mask.device != acc.device:
        if capturing():
            return
        mask = torch.ones(len(ops.PCIE_FIELDS), dtype=torch.int64, device=acc.device)
        for name in ops.ONE_PCIE:
            mask[ops.PCIE_FIELDS.index(name)] = 0
        mask[-1] = 0
        _ONE["mask"] = mask
    columns = scratch["columns"]
    plan = scratch["plan_host"][:decodes * columns].view(decodes, columns)
    acc[0].add_(ops.pcie_stats(plan, scratch["plan_count"][:decodes], 0, -1) * mask)


def union_resolve(runtime, handle, union, request_ids, block_table, block_size, rows, states=None, idle=None,
                  wait=True, fused_split=False, inline=False, one=None, fresh=None):
    decodes, width = union.shape
    if one is None and fused_split and width > 0:
        one = one_plan(runtime, handle, union, rows, [(0, width)], idle)
    if one is not None:
        group = runtime.index_group
        if not inline:
            group.copy_stream.wait_stream(current())
        with contextlib.nullcontext() if inline else group.copy_stream:
            physical = one_resolve(runtime, handle, union, request_ids, block_table, block_size, rows, states, idle,
                                   one, fresh=None if inline else fresh)
            union_tally_misses(runtime.index_group.shared_topk.swap_counts[rows], len(one["widths"]), decodes)
            if _REC2["armed"] is not None:
                announce("rec2_one_launch", why="the one-launch resolve writes its misses into per-part lists, "
                                                "so the refetch tally covers only the stock resolve")
            if inline:
                return physical
            event = union_event(runtime)
            event.record(group.copy_stream)
        if wait:
            current().wait_event(event)
        return physical
    shared = runtime.index_group.shared_topk

    def wide(buffer):
        return buffer[rows].reshape(-1)[:decodes * width].view(decodes, width)

    physical = wide(shared.physical_topk_indices)
    host_rows = wide(shared.swap_host_physical_rows)
    device_rows = wide(shared.swap_device_physical_rows)
    counts = shared.swap_counts[rows][:decodes]
    group = runtime.index_group
    ops = cache_ops()
    tally = gather_tally(runtime, union.device) if fused_split else None
    if not inline:
        group.copy_stream.wait_stream(current())
    with contextlib.nullcontext() if inline else group.copy_stream:
        resolve_residency(
            runtime.host_cache, runtime.hot.cache, runtime.hot_block_table, union, wide(shared.device_topk_rows),
            group.device_global_indices, group.lru_slots,
            runtime.request_state_indices if states is None else states, runtime.region_stride,
            None, group.swap_stats, physical, runtime.hot.attention_block_stride, request_ids, block_table,
            block_size, None, None, host_rows, device_rows, counts,
            handle.block_table, handle.view.block_size if handle.view is not None else 0, 0)
        union_tally_misses(counts, 1, decodes)
        rec2_misses(host_rows, counts[:decodes])
        if _STREAK["copies"] is not None:
            streak_copy(counts, idle)
        if _PREV["capture"] is not None:
            prev_capture(host_rows, counts)
        factor = 1
        run = run_merge_active(runtime)
        spread = 0 if run >= 0 else gather_spread(runtime)
        group_width = 0 if spread or run >= 0 else group_active(runtime)
        if getattr(type(runtime), SPLIT_MARK, False) and not group_width and not spread and run < 0:
            factor = split_factor(decodes, width) if idle is None else split_factor(decodes, width,
                                                                                     decodes * width)
        fused_counts = fused_split and (factor > 1 or tally is not None)
        if idle is not None and not fused_counts:
            counts.masked_fill_(idle, 0)
        gather_host, gather_device, gather_counts = host_rows, device_rows, counts
        planned = False
        if fused_counts:
            gather_host, gather_device = (host_rows.reshape(decodes * factor, width // factor),
                                          device_rows.reshape(decodes * factor, width // factor))
            gather_counts = plan_gather(runtime, counts, idle, factor, gather_host, gather_device, tally)
            planned = gather_counts is not None
            if not planned:
                gather_counts = fused_ops().split(counts, idle, factor, width // factor, tally)
        elif factor > 1:
            gather_host, gather_device, gather_counts = split_plan(host_rows, device_rows, counts, factor)
        gather_counts = probe_counts(gather_counts, "demand")
        gather_compact(runtime, gather_host, gather_device, gather_counts, group_width, spread, planned, run=run)
        if inline:
            return physical
        event = union_event(runtime)
        event.record(group.copy_stream)
    if wait:
        current().wait_event(event)
    return physical


def union_convert(group, logical, meta):
    decodes = int(meta.num_decodes)
    query_len = int(meta.decode_max_query_len)
    rows, top_k = logical.shape
    if varlen_decode_only(meta) or query_len < 2 or decodes == 0 or rows != decodes * query_len or len(group.caches) != 1:
        return None
    handle = group.cache(0)
    runtime = handle.runtime
    block_table = handle.source_block_table
    if block_table is None or runtime.request_state_indices is None:
        return None
    span = union_span(query_len, top_k, int(runtime.region_stride), smem_limit(logical.device))
    if span < 2:
        announce("union_swap_narrow", query_len=query_len, top_k=top_k, region=runtime.region_stride,
                 smem=smem_limit(logical.device),
                 why="no union of two rows fits the resolve kernel's shared memory or the hot region")
        return None
    picks = logical.reshape(decodes, query_len, top_k)
    physical = group.physical_topk_indices[:rows].view(decodes, query_len, top_k)
    counts = group.valid_topk_counts[:rows].view(decodes, query_len)
    request_ids = group.request_ids[:decodes]
    for start in range(0, query_len, span):
        chunk = picks[:, start:start + span].reshape(decodes, span * top_k)
        union, back = union_rows(chunk)
        resolved = union_resolve(runtime, handle, union, request_ids, block_table, int(meta.block_size),
                                 runtime._step_rows(decodes * span))
        mapped = resolved.gather(1, back).masked_fill(chunk < 0, -1).view(decodes, span, top_k)
        physical[:, start:start + span].copy_(mapped)
        counts[:, start:start + span].copy_((mapped >= 0).sum(2))
    announce("union_swap", decodes=decodes, query_len=query_len, span=span, top_k=top_k,
             region=runtime.region_stride,
             why="each lane's verification rows are swapped in as one deduplicated row per span, so the resolve "
                 "scans the hot LRU once per span instead of once per row")
    return physical.view(rows, top_k), counts.view(rows)


def hot_rows():
    said = str(os.environ.get(HOT_ROWS_ENV) or "").strip()
    return int(said) if said.isdigit() else 0


def capped_width(region, limit):
    fixed = 8 * ((region + 31) // 32) + 28 + 2 * region
    return max(0, min(region, (limit - fixed) // 16))


def capped_needed(meta, top_k, group):
    return union_swap() and hot_rows() > 0


def install_hot_cap(device):
    if _HOT_CAP["tally"] is None:
        _HOT_CAP["tally"] = torch.zeros(4, dtype=torch.int64, device=device)
    return _HOT_CAP["tally"]


def spill_skip():
    return knob_on(SPILL_SKIP_ENV)


def spill_onesort():
    return knob_on(SPILL_ONESORT_ENV)


def hot_split():
    return knob_on(HOT_SPLIT_ENV)


def split_bounds(region, cap):
    return [(start, min(start + cap, region)) for start in range(0, region, cap)]


def split_resolve(runtime, handle, union, fits, request_ids, block_table, block_size, scratch, region, cap,
                  before=None, fused_split=False, states=None, fresh=None):
    if fused_split and fits is None and one_launch_on():
        width = int(union.shape[1])
        part_cap = cap
        if hit_skip() >= 2:
            part_cap = max(cap, min(int(region), fused_ops().ONE_MAX_BLOCK))
        bounds = [(start, min(end, width)) for start, end in split_bounds(region, part_cap) if start < width]
        one = one_plan(runtime, handle, union, scratch, bounds, None)
        if one is not None and fresh is not None and one.get("flat", False):
            resolved = union_resolve(runtime, handle, union, request_ids, block_table, block_size, scratch,
                                     wait=False, fused_split=True, states=states, one=one, fresh=fresh)
            if before is not None:
                before()
            current().wait_event(hits_event(runtime))
            _HITS["taken"] = True
            return resolved
        if one is not None:
            resolved = union_resolve(runtime, handle, union, request_ids, block_table, block_size, scratch,
                                     wait=before is None, fused_split=True, states=states, one=one)
            if before is not None:
                before()
                current().wait_event(union_event(runtime))
            return resolved
    parts = []
    reach = int(union.shape[1])
    for index, (start, end) in enumerate((start, min(end, reach)) for start, end in split_bounds(region, cap)
                                         if start < reach):
        held = union[:, start:end]
        held = held.contiguous() if fits is None else held.masked_fill(~fits.view(-1, 1), -1).contiguous()
        if index or before is None:
            part = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch,
                                 fused_split=fused_split, states=states)
        else:
            part = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch, wait=False,
                                 fused_split=fused_split, states=states)
            before()
            current().wait_event(union_event(runtime))
        parts.append(part[:, :end - start].clone())
    return torch.cat(parts, 1)


def spill_unions(logical, spill, decodes, query_len, top_k, span):
    passes = query_len // span
    chunks = logical.reshape(decodes, passes, span * top_k).transpose(0, 1).contiguous()
    chunks.masked_fill_(~spill.view(1, decodes, 1), -1)
    chunks = chunks.view(passes * decodes, span * top_k)
    union, back = union_rows(chunks)
    lanes = torch.arange(decodes, device=logical.device).view(1, decodes, 1) * query_len
    steps = torch.arange(query_len, device=logical.device).view(passes, 1, span)
    index = (lanes + steps).reshape(passes, decodes * span)
    keep = spill.view(1, decodes, 1).expand(passes, decodes, span).reshape(passes, decodes * span)
    return [(chunks.view(passes, decodes, span * top_k)[at], union.view(passes, decodes, -1)[at],
             back.view(passes, decodes, -1)[at], index[at], keep[at]) for at in range(passes)]


def spill_states(states, request_ids, fits):
    index = request_ids.to(torch.int64)
    held = states.index_select(0, index)
    return states.clone().index_copy_(0, index, held.masked_fill(fits, -1))


def spill_columns(part, limit):
    rows, width = part.shape
    wide = min(limit, -(-width // SPLIT_MIN_COLUMNS) * SPLIT_MIN_COLUMNS)
    if wide <= width:
        return part
    return torch.cat((part, part.new_full((rows, wide - width), -1)), 1)


def capped_tally(union, fits, states):
    unique = (union >= 0).sum(1).to(torch.int64)
    live = unique > 0
    if states is not None:
        live = live & (states[:union.shape[0]] >= 0)
    over = live & ~fits
    install_hot_cap(union.device).add_(torch.stack((live.sum(), over.sum(), (unique * live).sum(),
                                                    (unique * over).sum())))


def capped_map(resolved, back, picks, keep):
    mapped = resolved.gather(1, back.clamp(0, resolved.shape[1] - 1))
    return mapped.masked_fill((picks < 0) | ~keep.view(-1, 1), -1)


def capped_swap(group, logical, meta, attend, before=None):
    decodes = int(meta.num_decodes)
    query_len = int(meta.decode_max_query_len)
    rows, top_k = logical.shape
    if varlen_decode_only(meta) or query_len < 2 or decodes == 0 or rows != decodes * query_len or len(group.caches) != 1:
        return False
    handle = group.cache(0)
    runtime = handle.runtime
    block_table = handle.source_block_table
    if block_table is None or runtime.request_state_indices is None:
        return False
    region = int(runtime.region_stride)
    cap = capped_width(region, smem_limit(logical.device))
    if cap < top_k:
        announce("hot_cap_narrow", cap=cap, top_k=top_k, region=region, smem=smem_limit(logical.device),
                 why="one row's index top-k does not fit the resolve kernel's shared memory beside this hot region")
        return False
    span = max(1, min(query_len, cap // top_k))
    picks = logical.reshape(decodes, query_len * top_k)
    request_ids = group.request_ids[:decodes]
    scratch = runtime._step_rows(decodes * query_len)
    block_size = int(meta.block_size)
    union, back = union_rows(picks)
    split = hot_split() and region > cap
    bound = region if split else cap
    fits = (union >= 0).sum(1) <= bound
    capped_tally(union, fits, runtime.request_state_indices)
    held = None if split else union[:, :cap].masked_fill(~fits.view(-1, 1), -1).contiguous()
    if split:
        resolved = split_resolve(runtime, handle, union, fits, request_ids, block_table, block_size, scratch,
                                 region, cap, before)
        announce("hot_split", knob=HOT_SPLIT_ENV, decodes=decodes, region=region, cap=cap,
                 resolves=len(split_bounds(region, cap)),
                 why="a lane whose verification union fits the hot region but not one resolve's shared memory is "
                     "swapped in by consecutive resolves of disjoint column ranges, each leaving the previous "
                     "one's rows most recent, so one attention reads them all and only a union over the region "
                     "spills")
    elif before is None:
        resolved = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch)
    else:
        resolved = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch, wait=False)
        before()
        current().wait_event(union_event(runtime))
    physical = capped_map(resolved, back, picks, fits).view(rows, top_k)
    attend(None, physical, (physical >= 0).sum(1), None)
    spill = ~fits
    lanes = torch.arange(decodes, device=logical.device).view(decodes, 1) * query_len
    skip = spill_skip()
    states = spill_states(runtime.request_state_indices, request_ids, fits) if skip else None
    passes = 0
    batched = None
    if spill_onesort() and span > 1 and query_len % span == 0 and query_len > span:
        batched = spill_unions(logical, spill, decodes, query_len, top_k, span)
        announce("spill_onesort", knob=SPILL_ONESORT_ENV, decodes=decodes, passes=len(batched), span=span,
                 why="every spill pass's per-lane union is one row of a single sort over all passes, and their "
                     "row indices and keep masks are built once, instead of one sort and index build per pass")
    for start in range(0, query_len, span):
        width = min(span, query_len - start)
        if batched is not None:
            chunk, part, part_back, index, keep = batched[passes]
            mapped = spill_pass(runtime, handle, chunk, part, part_back, spill, fits, states, skip, cap,
                                request_ids, block_table, block_size, scratch)
            mapped = mapped.reshape(decodes * width, top_k)
            attend(index, mapped, (mapped >= 0).sum(1), keep)
            passes += 1
            continue
        chunk = logical.reshape(decodes, query_len, top_k)[:, start:start + width].reshape(decodes, width * top_k)
        chunk = chunk.masked_fill(~spill.view(-1, 1), -1)
        if width > 1:
            part, part_back = union_rows(chunk)
        else:
            part = chunk.contiguous()
        if skip:
            resolved = union_resolve(runtime, handle, spill_columns(part, cap), request_ids, block_table,
                                     block_size, scratch, states=states, idle=fits)[:, :part.shape[1]]
        else:
            resolved = union_resolve(runtime, handle, part, request_ids, block_table, block_size, scratch)
        if width > 1:
            mapped = capped_map(resolved, part_back, chunk, spill)
        else:
            mapped = resolved.masked_fill(part < 0, -1)
        index = (lanes + torch.arange(start, start + width, device=logical.device).view(1, width)).reshape(-1)
        mapped = mapped.reshape(decodes * width, top_k)
        attend(index, mapped, (mapped >= 0).sum(1), spill.repeat_interleave(width))
        passes += 1
    _HOT_CAP["calls"] += 1
    _HOT_CAP["spills"] += passes
    _HOT_CAP["cap"] = bound
    if skip:
        announce("spill_skip", knob=SPILL_SKIP_ENV, decodes=decodes, cap=cap,
                 why="a spill pass resolves only the lanes whose union overflowed the hot region, the others are "
                     "skipped inside the kernel by their state index, and an overflowing lane's host reads are cut "
                     "across many blocks, so a pass with no overflowing lane launches and returns without a sync")
    announce("hot_cap", decodes=decodes, query_len=query_len, top_k=top_k, region=region, cap=cap, span=span,
             spill_passes=passes,
             why="a lane whose verification rows' distinct picks fit the hot region is swapped in once as their "
                 "union; a lane whose union does not fit attends its rows a span at a time, each span swapped in "
                 "right before its own attention")
    return True


def spill_pass(runtime, handle, chunk, part, part_back, spill, fits, states, skip, cap, request_ids, block_table,
               block_size, scratch):
    if skip:
        resolved = union_resolve(runtime, handle, spill_columns(part, cap), request_ids, block_table, block_size,
                                 scratch, states=states, idle=fits)[:, :part.shape[1]]
    else:
        resolved = union_resolve(runtime, handle, part, request_ids, block_table, block_size, scratch)
    return capped_map(resolved, part_back, chunk, spill)


def decode_attention_capped(layer, impl, query, logical, meta, out, gate, before=None):
    from vllm.taaf_qsa_kv8 import attention

    group = layer.taaf_hisparse_group
    rows = meta.num_decode_tokens
    hot = planes(group.physical_kv_cache(0), heads_of(layer))
    table = identity_table(hot.shape[0], hot.device)

    def attend(index, physical, counts, keep):
        owners = torch.zeros(physical.shape[0], dtype=torch.int32, device=query.device)
        if index is None:
            attention(impl, query[:rows], hot, with_counts(physical, counts), table, owners, False, out[:rows],
                      gate[:rows])
            return
        landing = torch.empty((physical.shape[0],) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
        attention(impl, query.index_select(0, index), hot, with_counts(physical, counts), table, owners, False,
                  landing, gate.index_select(0, index))
        held = out.index_select(0, index)
        shape = (-1,) + (1,) * (landing.dim() - 1)
        out.index_copy_(0, index, torch.where(keep.view(shape), landing, held))

    group.set_logical_topk_ready(0)
    if fused():
        done = capped_swap_fused(layer, impl, query, logical, meta, out, gate, before)
        if done is not None:
            return done
    if getattr(meta, "fixed", None) is not None:
        raise RuntimeError("a bucketed decode step needs the fused capped path and it declined this shape")
    return capped_swap(group, logical[:rows, :-1], meta, attend, before)


def decode_attention_expanded(layer, impl, query, logical, meta, out, gate, before):
    rows = int(meta.num_decode_tokens)
    place, slots, view = expanded_layout(meta, rows, query.device)
    spread_logical = spread_rows(logical[:rows], place, slots, -1)
    spread_logical[:, -1].masked_fill_(spread_logical[:, -1] < 0, 0)
    spread_out = spread_rows(out[:rows], place, slots, 0)
    spread_gate = spread_rows(gate[:rows], place, slots, 0) if gate is not None else None
    announce("draft_budget_expand", decodes=view.num_decodes, rows=rows, slots=slots, width=view.decode_max_query_len,
             why="a decode step with unequal verified rows per lane is spread to the uniform lane x width layout with "
                 "empty picks on the absent rows, so the fused host-KV paths run unchanged, and the rows are gathered "
                 "back after the attention")
    _EXPAND["inside"] = True
    try:
        decode_attention_once(layer, impl, spread_rows(query[:rows], place, slots, 0), spread_logical, view,
                              spread_out, spread_gate, before)
    finally:
        _EXPAND["inside"] = False
    out[:rows].copy_(spread_out.index_select(0, place))


def decode_attention_once(layer, impl, query, logical, meta, out, gate, before=None):
    from vllm.taaf_qsa_kv8 import attention

    if expand_on() and not _EXPAND["inside"] and varlen_decode_only(meta) and int(meta.num_decodes) > 0:
        return decode_attention_expanded(layer, impl, query, logical, meta, out, gate, before)

    if row_attend() and meta.decode_max_query_len > 1:
        return decode_attention_rows(layer, impl, query, logical, meta, out, gate)
    group = layer.taaf_hisparse_group
    rows = meta.num_decode_tokens
    if capped_needed(meta, logical.shape[1] - 1, group):
        if decode_attention_capped(layer, impl, query, logical, meta, out, gate, before):
            return None
        announce("hot_cap_rows", decodes=meta.num_decodes, rows=rows,
                 why="this batch cannot take the capped union, so each verification row attends after its own swap")
        return decode_attention_rows(layer, impl, query, logical, meta, out, gate)
    group.set_logical_topk_ready(0)
    held = union_convert(group, logical[:rows, :-1], meta) if union_swap() else None
    if held is None:
        held = group.convert_decode_logical_to_physical_topk(
            0, logical[:rows, :-1].contiguous(), meta, return_valid_counts=True,
            num_decodes=meta.num_decodes, decode_query_len=meta.decode_max_query_len)
    physical, counts = held
    hot = planes(group.physical_kv_cache(0), heads_of(layer))
    owners = torch.zeros(rows, dtype=torch.int32, device=query.device)
    attention(impl, query[:rows], hot, with_counts(physical, counts), identity_table(hot.shape[0], hot.device),
              owners, False, out[:rows], gate[:rows])


def sparse_stage():
    return str(os.environ.get(SPARSE_ENV) or "").strip().lower() in ("1", "on", "true")


def mark_fill():
    return knob_on(MARK_FILL_ENV)


def needed_rows(logical, owners, block_table, block_size, total_rows):
    free = syncfree()
    needed = torch.zeros(total_rows + 1 if free else total_rows, dtype=torch.bool, device=logical.device)
    width = logical.shape[1] - 1
    num_requests, table_width = block_table.shape
    staged_blocks = total_rows // block_size
    table = block_table.to(torch.int64)
    columns = torch.arange(width, device=logical.device)
    for at in range(0, logical.shape[0], SPARSE_QUERY_CHUNK):
        piece = logical[at:at + SPARSE_QUERY_CHUNK]
        tokens = piece[:, :width].to(torch.int64)
        counts = piece[:, width].to(torch.int64).clamp(0, width)
        owner = owners[at:at + SPARSE_QUERY_CHUNK].to(torch.int64)[:, None]
        pages = torch.div(tokens.clamp_min(0), block_size, rounding_mode="floor")
        valid = ((columns[None, :] < counts[:, None]) & (tokens >= 0) & (owner >= 0) & (owner < num_requests)
                 & (pages < table_width))
        physical = table[owner.clamp(0, num_requests - 1).expand_as(tokens), pages.clamp(max=table_width - 1)]
        valid &= (physical >= 0) & (physical < staged_blocks)
        rows = physical * block_size + tokens.clamp_min(0) % block_size
        if free and mark_fill():
            needed.index_fill_(0, torch.where(valid, rows, total_rows).reshape(-1), True)
        elif free:
            needed[torch.where(valid, rows, total_rows)] = True
        else:
            needed[rows[valid]] = True
    return needed[:total_rows] if free else needed


def stage_ahead():
    return knob_on(STAGE_AHEAD_ENV) and not _AHEAD["aside"]


def ahead_programs():
    return min(max(1, env_int(STAGE_AHEAD_PROGRAMS_ENV, STAGE_AHEAD_PROGRAMS)), STAGE_AHEAD_PROGRAMS_MAX)


def ahead_skip(reason, **fields):
    _AHEAD["skipped"][reason] = _AHEAD["skipped"].get(reason, 0) + 1
    announce(f"stage_ahead_skip_{reason}", knob=STAGE_AHEAD_ENV, **fields)
    return None


def first_main_layer():
    return next((one for one in _ORDER if not is_draft(one)), None)


def ahead_reset(layer):
    if _AHEAD["armed"] and layer is first_main_layer():
        _AHEAD["armed"].clear()


def ahead_ready(device, blocks):
    held = _AHEAD["counts"]
    if capturing():
        return (_AHEAD["stream"] is not None and held is not None and held.numel() >= blocks
                and _AHEAD["rows"] is not None)
    if _AHEAD["stream"] is None:
        least, _ = torch.cuda.Stream.priority_range()
        _AHEAD["stream"] = torch.cuda.Stream(device=device, priority=least)
    if held is None or held.numel() < blocks or held.device != device:
        _AHEAD["counts"] = torch.zeros(blocks, dtype=torch.int32, device=device)
    if _AHEAD["rows"] is None or _AHEAD["rows"].device != device:
        _AHEAD["rows"] = torch.zeros((), dtype=torch.int64, device=device)
        _DEVICE_TALLY["stage_ahead_rows"] = _AHEAD["rows"]
    return True


def ahead_event(layer):
    held = _AHEAD["events"].get(id(layer))
    if held is None and not capturing():
        held = _AHEAD["events"][id(layer)] = torch.cuda.Event()
    return held


def ahead_check(device, size, row_bytes):
    ops = fused_ops()
    words = row_bytes // 4
    generator = torch.Generator().manual_seed(6151)
    host = torch.randint(-2**31, 2**31 - 1, (6 * size, words), generator=generator, dtype=torch.int32).pin_memory()
    table = torch.tensor([[4, 1, 3], [2, 5, 0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([2 * size + 7, size + 3], dtype=torch.int64, device=device)
    ctx_lens = torch.tensor([size + 5, size], dtype=torch.int64, device=device)
    plan = stable_plan(table, seq_lens, ctx_lens, size, 6)
    blocks = plan.row_ids.shape[1] // size
    target = torch.full((blocks * size, words), 7, dtype=torch.int32, device=device)
    counts = ops.ahead_counts(plan.taaf_reuse_fixed, blocks, size,
                              torch.zeros(blocks, dtype=torch.int32, device=device))
    ops.ahead(host_words(host, device), target.view(-1), plan.row_ids, counts, size, words, 3)
    torch.cuda.synchronize(device)
    keep = plan.taaf_reuse_fixed.to(torch.bool)
    source = plan.row_ids[0].to(torch.int64).clamp_min(0)
    if int(keep.sum()) != 2 * size + 5:
        return f"the synthetic plan settled {int(keep.sum())} rows, not {2 * size + 5}"
    if not torch.equal(target[keep], host.to(device)[source][keep]):
        return "a settled row was staged with other bytes than the host holds"
    if not bool((target[~keep] == 7).all()):
        return "a row past a request's settled rows was written"
    return ""


def ahead_checked(device, size, row_bytes):
    if _AHEAD["checked"]:
        return not _AHEAD["aside"]
    if capturing():
        return False
    _AHEAD["checked"] = True
    try:
        why = ahead_check(device, size, row_bytes)
    except Exception as failure:
        why = f"{type(failure).__name__}: {str(failure)[:300]}"
    if why:
        _AHEAD["aside"] = why
        selfcheck_fail(STAGE_AHEAD_ENV, f"the self-check failed: {why}")
        print(f"QSA_STAGE_AHEAD stands aside: the self-check failed ({why}); every prefill chunk stages its rows "
              "the stock way", file=sys.stderr, flush=True)
        return False
    announce("stage_ahead_check", device=str(device), rows=size, row_bytes=row_bytes,
             why="a synthetic two-request plan staged ahead holds every settled row as the host holds it and no "
                 "row past a request's settled rows")
    return True


def ahead_issue(layer, plan, source):
    if not stage_ahead() or is_draft(layer):
        return None
    following = next_layer(layer)
    if following is None:
        return None
    if knob_on(QSA_HALVES_ENV):
        return ahead_skip("halves")
    settled = getattr(plan, "taaf_reuse_fixed", None)
    buffer = _STAGE["buffer"]
    if settled is None:
        return ahead_skip("no_settled_rows")
    if buffer is None or buffer.device.type != "cuda":
        return ahead_skip("no_stage_buffer")
    size = int(plan.block_size)
    row_bytes = int(source.shape[-1]) * int(source.element_size())
    blocks = int(plan.row_ids.shape[1]) // size
    need = blocks * size * row_bytes
    following_source = source_rows(following.kv_cache)
    if (row_bytes % 4 or need > buffer.numel() or settled.numel() != blocks * size
            or int(following_source.shape[1]) != size
            or int(following_source.shape[-1]) * int(following_source.element_size()) != row_bytes):
        return ahead_skip("shape", layer=following.layer_name, rows=blocks * size, row_bytes=row_bytes)
    device = buffer.device
    if not _AHEAD["hosts"] and not capturing():
        for one in _ORDER:
            if not is_draft(one):
                host_words(source_rows(one.kv_cache), device)
        _AHEAD["hosts"] = True
    host = host_words(following_source, device)
    if host.dtype != torch.int32:
        return ahead_skip("host_words", layer=following.layer_name)
    key = (blocks, size, row_bytes, str(device))
    if capturing() and key not in _AHEAD["warm"]:
        return ahead_skip("cold_capture", blocks=blocks)
    if not ahead_ready(device, blocks):
        return ahead_skip("cold_capture", blocks=blocks)
    event = ahead_event(following)
    if event is None:
        return ahead_skip("cold_capture", layer=following.layer_name)
    if not ahead_checked(device, size, row_bytes):
        return ahead_skip("unchecked")
    ops = fused_ops()
    stream = _AHEAD["stream"]
    target = buffer[:need].view(torch.int32)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        counts = ops.ahead_counts(settled, blocks, size, _AHEAD["counts"])
        ops.ahead(host, target, plan.row_ids, counts, size, row_bytes // 4, ahead_programs())
        event.record(stream)
    _AHEAD["warm"].add(key)
    _AHEAD["armed"][id(following)] = (event, plan.row_ids, settled, counts)
    _AHEAD["issued"] += 1
    announce("stage_ahead", knob=STAGE_AHEAD_ENV, layer=layer.layer_name, following=following.layer_name,
             blocks=blocks, programs=ahead_programs(), captured=capturing(),
             why="the next QSA layer's prefill stage needs every settled row of the step's prefills, which no later "
                 "work of the step writes, so a side stream at the device's least priority copies them from host "
                 "into the stage buffer with a small grid while the layers in between run, and that stage reads "
                 "only the rows still missing")
    return following


def ahead_take(layer, plan):
    if layer is None or not _AHEAD["armed"]:
        return None
    held = _AHEAD["armed"].pop(id(layer), None)
    if held is None:
        return None
    event, row_ids, settled, counts = held
    torch.cuda.current_stream(row_ids.device).wait_event(event)
    if plan.row_ids is not row_ids or getattr(plan, "taaf_reuse_fixed", None) is not settled:
        return ahead_skip("plan_moved", layer=layer.layer_name)
    _AHEAD["rows"].add_(counts.sum(dtype=torch.int64))
    _AHEAD["taken"] += 1
    return settled


def stage_rows(cache, kv_cache, part, meta, logical=None, owners=None, layer=None):
    plan = part.plan
    ahead = ahead_take(layer, plan)
    if kv_cache.shape[1] != plan.block_size:
        raise ValueError(f"host KV staging block size {plan.block_size} does not match the KV cache block size "
                         f"{kv_cache.shape[1]}")
    resident = None
    if cache.view is not None and cache.block_table is not None:
        decodes = meta.num_decodes
        lanes = getattr(part, "index", None)
        if lanes is None:
            table = cache.block_table[decodes + part.first:decodes + part.last]
        else:
            table = cache.block_table.index_select(0, lanes.clamp(max=cache.block_table.shape[0] - 1))
        if plan_once():
            resolve_sources(plan, table, cache.view.block_size)
            note_layer(plan, part, meta, cache.block_table, cache.view.block_size)
        else:
            plan.ensure_gpu_sources(table, cache.view.block_size)
        resident = cache.view.cache
        if guard_on() and plan.gpu_row_ids is not None and plan.gpu_row_ids.is_contiguous():
            fused_ops().guard_range(plan.gpu_row_ids, -1, resident.shape[0] * resident.shape[1],
                                    guard_tally("stage_resident_rows", plan.gpu_row_ids.device))
    elif plan_once():
        unresolved(plan)
    row_width = kv_cache.shape[-1]
    blocks = plan.row_ids.shape[1] // plan.block_size
    need = blocks * plan.block_size * row_width * kv_cache.element_size()
    state = reuse_state(layer, need, plan) if logical is not None and owners is not None and sparse_stage() else None
    if ahead is not None and state is not None:
        state.valid.fill_(False)
        state = None
    buffer = _STAGE["buffer"] if state is None else state.buffer
    if need > buffer.numel():
        raise RuntimeError(f"a prefill group needs {need} staged bytes and the stage buffer holds {buffer.numel()}")
    staged = buffer[:need].view(kv_cache.dtype).view(blocks, plan.block_size, row_width)
    flat = staged.view(-1, row_width)
    miss_mask = plan.miss_mask
    needed = None
    if (logical is not None and owners is not None and sparse_stage() and syncfree() and fused()
            and stage_rows_fused(plan, kv_cache, staged, flat, resident, logical, owners, state,
                                 ahead)):
        if state is not None:
            live_check(layer, cache, kv_cache, part, meta, logical, owners, staged)
        return staged
    kept = None
    if logical is not None and owners is not None and sparse_stage():
        needed = needed_rows(logical, owners, plan.block_table, plan.block_size, plan.row_ids.shape[1])
        if state is not None:
            kept, needed = reuse_split(state, plan, needed)
        miss_mask = (miss_mask.reshape(-1).bool() & needed).to(plan.miss_mask.dtype).view_as(plan.miss_mask)
        if syncfree() and mark_fill():
            announce("mark_fill", knob=MARK_FILL_ENV,
                     why="the rows a prefill chunk reads are marked by a fill kernel with the value inlined, not by "
                         "an index assignment whose value is a host scalar copied to the device, which waits for "
                         "the stream to drain once per layer")
        if syncfree():
            if device_tally_on():
                add_device_tally("sparse_rows_needed", needed.sum())
        else:
            _TALLY["sparse_rows_needed"] += int(needed.sum())
        _TALLY["sparse_rows_total"] += int(needed.numel())
        announce("sparse_stage", knob=SPARSE_ENV,
                 why="prefill attention reads only each query's index picks, so a chunk stages the union of its "
                     "queries' picked rows instead of every row of the prefix")
    if needed is not None and stage_blocks():
        miss_mask = block_stage_rows(plan, kv_cache, staged, miss_mask, kept)
    if plan.gpu_row_ids is not None and resident is not None:
        source = plan.gpu_row_ids[0].to(torch.long)
        size = resident.shape[1]
        if needed is not None and syncfree():
            keep = (source >= 0) & needed
            source = source.clamp_min(0)
            for at in range(0, source.numel(), STAGE_COPY_ROWS):
                piece = source[at:at + STAGE_COPY_ROWS]
                rows = resident[piece // size, piece % size]
                if rows.dtype != flat.dtype:
                    rows = rows.contiguous().view(flat.dtype)
                target = flat[at:at + piece.numel()].view(torch.uint8)
                target.copy_(torch.where(keep[at:at + piece.numel()].unsqueeze(1),
                                         rows.contiguous().view(torch.uint8), target))
        elif needed is not None:
            positions = torch.nonzero((source >= 0) & needed, as_tuple=False).view(-1)
            for at in range(0, positions.numel(), STAGE_COPY_ROWS):
                where = positions[at:at + STAGE_COPY_ROWS]
                piece = source[where]
                rows = resident[piece // size, piece % size]
                if rows.dtype != flat.dtype:
                    rows = rows.contiguous().view(flat.dtype)
                flat[where] = rows
        else:
            source = source.clamp_min(0)
            for at in range(0, source.numel(), STAGE_COPY_ROWS):
                piece = source[at:at + STAGE_COPY_ROWS]
                rows = resident[piece // size, piece % size]
                if rows.dtype != flat.dtype:
                    rows = rows.contiguous().view(flat.dtype)
                flat[at:at + piece.numel()].copy_(rows)
    torch.ops._C_cache_ops.hisparse_gather_plan(kv_cache.view(-1, row_width), staged, plan.row_ids, plan.dst_rows,
                                                miss_mask, None, None, 0)
    if state is not None:
        reuse_commit(state, plan, kept, needed)
        live_check(layer, cache, kv_cache, part, meta, logical, owners, staged)
    return staged


def last_rows_in(held, start, end):
    if held is None:
        return None
    if not mixed_capture():
        return held[(held >= start) & (held < end)] - start
    _MIRROR["clamped"] += 1
    announce("last_rows_clamped", knob=MIXED_CAPTURE_ENV, rows=held.numel(), start=start, end=end,
             why="the draft forward's last rows are clamped into each prefill part instead of filtered by a boolean "
                 "mask whose nonzero waits for the GPU; a row clamped from outside the part recomputes the part's "
                 "first or last row, which the draft layer drops right after attention")
    return (held - start).clamp(0, max(end - start - 1, 0))


def prefill_attention(layer, impl, query, logical, meta, out, gate, use_prefill_config):
    from vllm import taaf_qsa_kv8 as quantized

    held = quantized._ROWS["last"]
    source = source_rows(layer.kv_cache)
    sliced = held is not None and knob_on(LAST_SLICE_ENV)
    ahead_reset(layer)
    for part in meta.prefill.parts:
        start, end = part.start, part.end
        fixed = getattr(part, "table", None) is not None
        if fixed:
            scoped = None if held is None else part.last_rows(held)
        elif sliced:
            scoped = last_rows_of(held, int(meta.num_decodes), part, start, end)
        else:
            scoped = last_rows_in(held, start, end)
        if scoped is not None and not fixed and scoped.numel() == 0:
            continue
        if fixed:
            owners, picks = part.owners, part.mask(logical[:end])
        else:
            owners = (meta.req_id_per_token[start:end] - (meta.num_decodes + part.first)).to(torch.int32)
            picks = logical[start:end]
        staged = stage_rows(layer.hisparse_cache, source, part, meta, picks, owners, layer)
        _TALLY["staged_blocks"] += int(staged.shape[0])
        with quantized.last_rows_scope(scoped):
            quantized.attention(impl, query[start:end], planes(staged, heads_of(layer)), picks,
                                part.plan.block_table, owners, use_prefill_config, out[start:end],
                                gate[start:end])
        if fixed:
            ahead_issue(layer, part.plan, source)


def last_rows_of(held, decodes, part, start, end):
    rows = held[decodes + part.first:decodes + part.last] - start
    inside = (held >= start) & (held < end)
    if device_tally_on():
        add_device_tally("last_rows_outside", ((rows < 0) | (rows >= end - start)).sum()
                         + (inside.sum() - rows.numel()).abs())
    announce("last_slice", knob=LAST_SLICE_ENV, rows=rows.numel(),
             why="a prefill part's last rows are its requests' entries of the last-token list, read as a slice "
                 "on the GPU instead of a boolean mask whose length the host waits for; a row outside its part "
                 "is counted on the GPU and printed with the tally")
    return rows.clamp(0, max(end - start - 1, 0))


def qsa_halves_stream(device):
    if device.type != "cuda" or not knob_on(QSA_HALVES_ENV):
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    held = _QSA_HALVES["streams"].get(index)
    if held is None:
        held = torch.cuda.Stream(device=index)
        _QSA_HALVES["streams"][index] = held
    return held


def stream_scope(stream):
    return torch.cuda.stream(stream) if isinstance(stream, torch.cuda.Stream) else stream


def run_fixed(layer, impl, query, output, output_gate, use_prefill_config, meta):
    logical = guard_logical(layer, layer.topk_indices_buffer[:meta.num_actual_tokens], meta, meta.fixed.split)
    landing = torch.zeros_like(output)
    ran = []
    before = None
    side = qsa_halves_stream(output.device) if gather_early() and gather_early_captured() else None
    if gather_early() and gather_early_captured():
        def before():
            ran.append(layer.layer_name)
            if side is not None:
                side.wait_stream(current())
                with stream_scope(side):
                    prefill_attention(layer, impl, query, logical, meta, landing, output_gate, use_prefill_config)
                announce("qsa_halves", knob=QSA_HALVES_ENV, layer=layer.layer_name,
                         why="a bucketed mixed step's prefill attention runs on a second stream beside the decode "
                             "attention of the same layer; the two write different buffers, and the layer waits for "
                             "both before the prefill rows are taken into its output")
                return
            prefill_attention(layer, impl, query, logical, meta, landing, output_gate, use_prefill_config)
            announce("gather_early_fixed", knob=GATHER_EARLY_CAPTURED_ENV, layer=layer.layer_name,
                     decodes=meta.fixed.decodes, tokens=meta.fixed.tokens, captured=capturing(),
                     why="a bucketed mixed step issues its decode rows' union swap on the copy stream before the "
                         "prefill attention, which writes its own landing rows and reads none of the swapped rows, "
                         "and waits for the swap only before the decode attention")
    decode_attention(layer, impl, query, logical, meta, output, output_gate, before)
    if not ran:
        prefill_attention(layer, impl, query, logical, meta, landing, output_gate, use_prefill_config)
    elif side is not None:
        current().wait_stream(side)
        _QSA_HALVES["split"] += 1
    if knob_on(TAKE_ROWS_ENV) and output.is_cuda and output.is_contiguous() and landing.is_contiguous():
        fused_take(layer, output, landing, meta.fixed.split)
        announce("take_rows", knob=TAKE_ROWS_ENV, layer=layer.layer_name,
                 why="a bucketed mixed step copies its prefill rows from the landing buffer into the layer's output "
                     "with one kernel that reads the decode row count on the device and skips the decode rows, "
                     "instead of a row mask, a full where over both buffers and a copy back")
        return output
    take = torch.arange(output.shape[0], device=output.device) >= meta.fixed.split[1]
    output.copy_(torch.where(take.view((-1,) + (1,) * (output.dim() - 1)), landing, output))
    return output


def fused_take(layer, output, landing, split):
    held = output.clone() if take_shadow_on() else None
    fused_ops().take_rows(output, landing, split[1:2])
    if held is not None:
        take = torch.arange(output.shape[0], device=output.device) >= split[1]
        take_shadow_note("output", output, torch.where(take.view((-1,) + (1,) * (output.dim() - 1)), landing, held),
                         layer_number(layer.layer_name))
    return output


def stream_handle(stream):
    if isinstance(stream, torch.cuda.Stream):
        return int(stream.cuda_stream)
    return int(torch.cuda.Stream(stream_id=stream.stream_id, device_index=stream.device_index,
                                 device_type=stream.device_type).cuda_stream)


def unalias_copy_streams(compute):
    if _COPY_STREAMS["done"] or capturing():
        return 0
    _COPY_STREAMS["done"] = True
    seen = {stream_handle(compute)}
    moved = []
    for layer in _ORDER:
        handle = getattr(layer, "hisparse_cache", None)
        group = getattr(getattr(handle, "runtime", None), "index_group", None)
        stream = getattr(group, "copy_stream", None)
        if stream is None or id(group) in _COPY_STREAMS["groups"]:
            continue
        _COPY_STREAMS["groups"].add(id(group))
        held = stream_handle(stream)
        if held in seen:
            for _ in range(COPY_STREAM_TRIES):
                fresh = torch.cuda.Stream(device=compute.device)
                if stream_handle(fresh) not in seen:
                    group.copy_stream = fresh
                    held = stream_handle(fresh)
                    moved.append(layer.layer_name)
                    break
        seen.add(held)
    if moved:
        print(f"HOST_KV copy streams: {len(moved)} of {len(_COPY_STREAMS['groups'])} QSA layers had a copy stream "
              f"that the stream pool had already handed to the compute stream or to another layer ({moved[:4]}), "
              f"so their resolve and copy ran in line with the compute work; each now has a stream of its own",
              flush=True)
    return len(moved)


def run(layer, impl, query, key, value, main, side, output, output_gate, use_prefill_config):
    if isinstance(query, torch.Tensor) and query.is_cuda:
        unalias_copy_streams(current())
    if knob_on(HITS_SHADOW_ENV) and not capturing() and isinstance(query, torch.Tensor):
        hits_shadow(query.device, 0, None, None)
    if take_shadow_on() and not capturing() and isinstance(query, torch.Tensor):
        take_shadow_ready(query.device)
    if not prefetch_armed():
        return run_layer(layer, impl, query, key, value, main, side, output, output_gate, use_prefill_config)
    prefetch_join()
    try:
        return run_layer(layer, impl, query, key, value, main, side, output, output_gate, use_prefill_config)
    finally:
        prefetch_close()


def draft_cold():
    said = str(os.environ.get(DRAFT_COLD_ENV) or "").strip()
    if not said or said in ("0", "off"):
        return False
    if not said.startswith("file:"):
        return True
    now = time.monotonic()
    if now - _DRAFT_COLD["checked"] >= 1.0:
        _DRAFT_COLD["checked"] = now
        _DRAFT_COLD["on"] = os.path.exists(said[5:])
    return _DRAFT_COLD["on"]


def draft_cold_reset(layer):
    runtime = layer.hisparse_cache.runtime
    if not runtime.is_group_leader:
        return
    runtime.reset_hot_state()
    _DRAFT_COLD["resets"] += 1
    announce("draft_cold", layer=layer.layer_name, knob=DRAFT_COLD_ENV,
             why="the draft layer's hot region bookkeeping is dropped before each of its forwards, so every pick is "
                 "swapped in afresh into the slot of the lane that asked for it")


def run_layer(layer, impl, query, key, value, main, side, output, output_gate, use_prefill_config):
    cache = layer.hisparse_cache
    source = source_rows(layer.kv_cache)
    meta = step_metadata(side, main, cache.view.block_size, source.shape[-1] * source.element_size())
    cache.prepare_group_for_batch(meta)
    if is_draft(layer) and draft_cold():
        draft_cold_reset(layer)
    write(layer, impl, key, value, meta, main.slot_mapping.numel())
    output.zero_()
    if getattr(meta, "fixed", None) is not None:
        return run_fixed(layer, impl, query, output, output_gate, use_prefill_config, meta)
    if meta.num_actual_tokens == 0:
        return output
    logical = guard_logical(layer, layer.topk_indices_buffer[:meta.num_actual_tokens], meta)
    prefills = meta.num_decode_tokens < meta.num_actual_tokens
    ran = []
    before = None
    if meta.num_decode_tokens and prefills and gather_early() and (not capturing() or gather_early_captured()):
        def before():
            ran.append(layer.layer_name)
            prefill_attention(layer, impl, query, logical, meta, output, output_gate, use_prefill_config)
            announce("gather_early", knob=GATHER_EARLY_ENV, layer=layer.layer_name,
                     decode_tokens=meta.num_decode_tokens, tokens=meta.num_actual_tokens,
                     why="the decode rows' union swap is issued on the copy stream before this layer's prefill "
                         "attention, which reads none of the rows it writes, and the compute stream waits for it "
                         "only after the prefill attention, right before the decode attention reads them")
    if meta.num_decode_tokens:
        decode_attention(layer, impl, query, logical, meta, output, output_gate, before)
        if _AUDIT["left"] is not None or str(os.environ.get(AUDIT_ENV) or "").strip():
            _AUDIT["step_now"] = audit_step(layer)
        if audit_due():
            audit_decode(layer, impl, query, logical, meta, output, output_gate)
    if prefills and not ran:
        prefill_attention(layer, impl, query, logical, meta, output, output_gate, use_prefill_config)
    tally(layer, meta)
    return output


def audit_settings():
    said = str(os.environ.get(AUDIT_ENV) or "").strip()
    if not said:
        return 0, 0.0
    parts = [one.strip() for one in said.split(",")]
    try:
        calls = int(parts[0])
        after = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    except ValueError:
        return 0, 0.0
    return max(0, calls), max(0.0, after)


def audit_due(now=None):
    calls, after = audit_settings()
    if calls <= 0 or capturing():
        return False
    if _AUDIT["left"] is None:
        _AUDIT["left"] = calls
    if _AUDIT["left"] <= 0:
        return False
    now = time.monotonic() if now is None else now
    if _AUDIT["start"] is None:
        _AUDIT["start"] = now
    return now - _AUDIT["start"] >= after


def audit_plan(picks, lane_of_row, src_table, src_bs, res_table, res_bs, host_total):
    valid = picks >= 0
    pos = picks.clamp_min(0).to(torch.int64)
    lanes = lane_of_row.to(torch.int64).unsqueeze(1).expand_as(pos)
    sblock = pos // src_bs
    sinside = valid & (sblock < src_table.shape[1]) & (lanes < src_table.shape[0])
    sphys = src_table[lanes.clamp(max=src_table.shape[0] - 1), sblock.clamp(max=src_table.shape[1] - 1)].to(torch.int64)
    host = torch.where(sinside & (sphys > 0), sphys * src_bs + pos % src_bs, torch.full_like(pos, -1))
    host = torch.where(host >= host_total, torch.full_like(host, -1), host)
    if res_table is not None and res_bs:
        rblock = pos // res_bs
        rinside = valid & (rblock < res_table.shape[1]) & (lanes < res_table.shape[0])
        rphys = res_table[lanes.clamp(max=res_table.shape[0] - 1),
                          rblock.clamp(max=res_table.shape[1] - 1)].to(torch.int64)
        resident = torch.where(rinside & (rphys > 0), rphys, torch.full_like(pos, -1))
        offset = pos % res_bs
    else:
        resident = torch.full_like(pos, -1)
        offset = torch.zeros_like(pos)
    from_resident = valid & (resident >= 0)
    from_host = valid & ~from_resident & (host >= 0)
    missing = valid & ~from_resident & ~from_host
    return {"valid": valid, "pos": pos, "resident": resident, "offset": offset, "host": host,
            "from_resident": from_resident, "from_host": from_host, "missing": missing}


def audit_fresh(plan, lane_of_row, seq_lens, query_lens, starts, slot_mapping, res_bs):
    lanes = lane_of_row.to(torch.int64)
    first = (seq_lens.to(torch.int64) - query_lens.to(torch.int64))[lanes].unsqueeze(1)
    fresh = plan["valid"] & (plan["pos"] >= first)
    written_row = starts.to(torch.int64)[lanes].unsqueeze(1) + (plan["pos"] - first)
    written_row = written_row.clamp(0, max(slot_mapping.numel() - 1, 0))
    written = slot_mapping.to(torch.int64)[written_row]
    seen = plan["resident"] * max(res_bs, 1) + plan["offset"]
    stale = fresh & plan["from_resident"] & (seen != written)
    return {"fresh": fresh, "fresh_host": fresh & plan["from_host"], "fresh_missing": fresh & plan["missing"],
            "fresh_stale_slot": stale}


def audit_rows(plan, resident_cache, host_cache, row_shape, dtype, device):
    total = plan["pos"].numel()
    rows = torch.zeros((total,) + tuple(row_shape), dtype=dtype, device=device)
    flat_res = plan["from_resident"].reshape(-1)
    flat_host = plan["from_host"].reshape(-1)
    if bool(flat_res.any()):
        at = flat_res.nonzero().squeeze(1)
        rows[at] = resident_cache[plan["resident"].reshape(-1)[at], plan["offset"].reshape(-1)[at]].to(device)
    if bool(flat_host.any()):
        at = flat_host.nonzero().squeeze(1)
        source = plan["host"].reshape(-1)[at]
        rows[at] = host_cache.index_select(0, source.to(host_cache.device)).to(device)
    return rows


def audit_line(layer, stats):
    picks = max(stats["picks"], 1)
    return (f"HOST_KV_AUDIT layer={layer} draft={stats['draft']} lanes={stats['lanes']} rows={stats['rows']} "
            f"picks={stats['picks']} resident={stats['resident'] / picks:.4f} host={stats['host'] / picks:.4f} "
            f"missing={stats['missing']} fresh={stats['fresh']} fresh_host={stats['fresh_host']} "
            f"fresh_missing={stats['fresh_missing']} fresh_stale_slot={stats['fresh_stale_slot']} "
            f"rel_max={stats['rel_max']:.3e} rel_mean={stats['rel_mean']:.3e} rows_off={stats['rows_off']} "
            f"rows_off_fresh={stats['rows_off_fresh']} draft_step={stats.get('draft_step', -1)} "
            f"got_norm={stats.get('got_norm', 0.0):.3e} want_norm={stats.get('want_norm', 0.0):.3e} "
            f"got_zero_rows={stats.get('got_zero_rows', -1)} phys_neg={stats.get('phys_neg', -1)} "
            f"states={stats.get('states')} rows_per_lane={stats.get('rows_per_lane')} seq_lens={stats.get('seq_lens')} "
            f"(output of the decode path against attention over the "
            f"same picks read straight from the resident page or the host row)")


def audit_step(layer):
    if not is_draft(layer):
        _AUDIT["draft_step"] = 0
        _AUDIT["seen"] = None
        return 0
    if _AUDIT["seen"] != id(layer):
        _AUDIT["seen"] = id(layer)
    _AUDIT["draft_step"] += 1
    return _AUDIT["draft_step"]


def audit_decode(layer, impl, query, logical, meta, out, gate):
    from vllm.taaf_qsa_kv8 import attention

    decodes = int(meta.num_decodes)
    if decodes < AUDIT_MIN_LANES or getattr(layer.hisparse_cache, "dummy_batch", False):
        return
    if int(meta.seq_lens[:decodes].min().item()) < AUDIT_MIN_CONTEXT:
        return
    _AUDIT["left"] -= 1
    try:
        lanes = min(decodes, AUDIT_LANES)
        device = query.device
        starts = meta.query_start_loc[:lanes + 1].to(torch.int64)
        rows = int(starts[-1].item())
        if rows <= 0:
            return
        handle = layer.hisparse_cache
        runtime = handle.runtime
        picks = logical[:rows, :-1].to(torch.int64)
        counts = starts[1:] - starts[:-1]
        lane_of_row = torch.repeat_interleave(torch.arange(lanes, device=device), counts)
        res_bs = int(handle.view.block_size) if handle.view is not None and handle.block_table is not None else 0
        plan = audit_plan(picks, lane_of_row, handle.source_block_table, int(meta.block_size), handle.block_table,
                          res_bs, int(runtime.host_cache.shape[0]))
        fresh = audit_fresh(plan, lane_of_row, meta.seq_lens[:lanes], counts, meta.query_start_loc[:lanes],
                            handle.slot_mapping, res_bs)
        attention_cache = runtime.hot.attention_cache
        tokens = int(attention_cache.shape[1])
        width = picks.shape[1]
        gathered = audit_rows(plan, handle.view.cache if handle.view is not None else None, runtime.host_cache,
                              attention_cache.shape[2:], attention_cache.dtype, device)
        blocks = (gathered.shape[0] + tokens - 1) // tokens
        staging = torch.zeros((blocks * tokens,) + tuple(attention_cache.shape[2:]), dtype=attention_cache.dtype,
                              device=device)
        staging[:gathered.shape[0]] = gathered
        staging = staging.view((blocks, tokens) + tuple(attention_cache.shape[2:]))
        index = torch.arange(rows * width, dtype=torch.int64, device=device).view(rows, width)
        index = torch.where(plan["from_resident"] | plan["from_host"], index, torch.full_like(index, -1))
        full = torch.full((rows,), width, dtype=torch.int32, device=device)
        reference = torch.empty_like(out[:rows])
        attention(impl, query[:rows], planes(staging, heads_of(layer)), with_counts(index, full),
                  identity_table(blocks, device), torch.zeros(rows, dtype=torch.int32, device=device), False,
                  reference, gate[:rows])
        got = out[:rows].float().reshape(rows, -1)
        want = reference.float().reshape(rows, -1)
        rel = (got - want).abs().amax(dim=1) / want.abs().amax(dim=1).clamp_min(1e-6)
        extra = {"draft_step": _AUDIT.get("step_now", -1), "got_norm": float(got.norm()), "want_norm": float(want.norm()),
                 "got_zero_rows": int((got.abs().amax(dim=1) == 0).sum()), "phys_neg": int((picks < 0).sum()),
                 "states": runtime.request_state_indices[:lanes].tolist(),
                 "rows_per_lane": counts.tolist()[:lanes], "seq_lens": meta.seq_lens[:lanes].tolist()}
        off = rel > AUDIT_REL
        has_fresh_off = fresh["fresh"].any(dim=1)
        stats = {"draft": int(is_draft(layer)), "lanes": lanes, "rows": rows,
                 "picks": int(plan["valid"].sum()), "resident": int(plan["from_resident"].sum()),
                 "host": int(plan["from_host"].sum()), "missing": int(plan["missing"].sum()),
                 "fresh": int(fresh["fresh"].sum()), "fresh_host": int(fresh["fresh_host"].sum()),
                 "fresh_missing": int(fresh["fresh_missing"].sum()),
                 "fresh_stale_slot": int(fresh["fresh_stale_slot"].sum()), "rel_max": float(rel.max()),
                 "rel_mean": float(rel.mean()), "rows_off": int(off.sum()),
                 "rows_off_fresh": int((off & has_fresh_off).sum()), **extra}
        print(audit_line(layer.layer_name, stats), flush=True)
        _AUDIT["done"] += 1
    except Exception as error:
        print(f"HOST_KV_AUDIT stopped layer={getattr(layer, 'layer_name', '?')}: "
              f"{type(error).__name__}: {error}"[:400], flush=True)


def gather_early():
    return knob_on(GATHER_EARLY_ENV)


def gather_early_captured():
    if not knob_on(GATHER_EARLY_CAPTURED_ENV):
        return False
    announce("gather_early_captured", knob=GATHER_EARLY_CAPTURED_ENV,
             why="a captured mixed step issues the decode rows' swap on the copy stream before the prefill attention "
                 "too, so the full mixed graph holds the overlap the eager step has")
    return True


def tally(layer, meta):
    _TALLY["decode_rows"] += meta.num_decode_tokens
    _TALLY["prefill_rows"] += meta.num_actual_tokens - meta.num_decode_tokens
    _TALLY["forwards"] += 1
    announce("first_forward", layer=layer.layer_name, rows=meta.num_actual_tokens,
             decode_rows=meta.num_decode_tokens, query_len=meta.decode_max_query_len)
    _TALLY_DUE["last"] = layer
    if _TALLY["forwards"] % REPORT_EVERY == 0:
        _TALLY_DUE["layer"] = layer


def report_tally():
    if capturing():
        return None
    take_shadow_report()
    _TALLY_DUE["starts"] += 1
    if _TALLY_DUE["layer"] is None and _TALLY_DUE["starts"] % REPORT_STEPS == 0:
        _TALLY_DUE["layer"] = _TALLY_DUE["last"]
    layer = _TALLY_DUE["layer"]
    if layer is None:
        return None
    _TALLY_DUE["layer"] = None
    fold_device_tallies()
    guard_fold()
    runtime = layer.hisparse_cache.runtime
    stats = {name: getattr(runtime, name) for name in ("num_swap_rows", "num_hits", "num_misses")
             if hasattr(runtime, name)}
    line = (f"QSA_HISPARSE tally forwards={_TALLY['forwards']} decode_rows={_TALLY['decode_rows']} "
            f"prefill_rows={_TALLY['prefill_rows']} staged_blocks={_TALLY['staged_blocks']} "
            f"gated_writes={_TALLY['gated_writes']} draft_mirrors={_TALLY['draft_mirrors']} "
            + (f"draft_through={_TALLY['draft_through']} " if draft_through() else "")
            + f"sparse_rows_needed={_TALLY['sparse_rows_needed']} sparse_rows_total={_TALLY['sparse_rows_total']} "
            + (f"last_rows_outside={_TALLY['last_rows_outside']} " if knob_on(LAST_SLICE_ENV) else "")
            + (f"sources_fused={_TALLY['sources_fused']} sources_stock={_TALLY['sources_stock']} "
               if knob_on(SOURCES_FUSED_ENV) else "")
            + (f"hot_resets={_RESET['resets']} " if _RESET["fresh"] is not None else "")
            + (f"reuse_rows_kept={_TALLY['reuse_rows_kept']} reuse_rows_copied={_TALLY['reuse_rows_copied']} "
               f"block_stage_blocks={_TALLY['block_stage_blocks']} reuse_mib={round(_REUSE['bytes'] / 2**20, 1)} "
               if stage_reuse_modes() else "")
            + (f"stage_ahead_rows={_TALLY['stage_ahead_rows']} stage_ahead_issued={_AHEAD['issued']} "
               f"stage_ahead_taken={_AHEAD['taken']} "
               f"stage_ahead_skipped={','.join(f'{k}:{v}' for k, v in sorted(_AHEAD['skipped'].items())) or 0} "
               if stage_ahead() or _AHEAD["aside"] else "")
            + (f"mirror_misses={_TALLY['mirror_misses']} prefill_mirror={_MIRROR['used']} "
               f"mirror_absent={_MIRROR['absent']} mirror_over={_MIRROR['over']} "
               f"last_rows_clamped={_MIRROR['clamped']} " if mixed_capture() else "")
            + (f"hits_shadow_calls={_TALLY['hits_shadow_calls']} hits_shadow_bad={_TALLY['hits_shadow_bad']} "
               f"hits_shadow_mean_rel={_TALLY['hits_shadow_err_ppm'] / max(1, _TALLY['hits_shadow_calls']) / 1e6:.2e} "
               + f"hits_shadow_draft_calls={_TALLY['hits_shadow_draft_calls']} hits_shadow_draft_bad={_TALLY['hits_shadow_draft_bad']} "
               + "".join(f"hits_shadow_{band}={_TALLY[f'hits_shadow_{band}_bad']}/{_TALLY[f'hits_shadow_{band}_calls']} "
                         for band in HITS_SHADOW_BANDS)
               if knob_on(HITS_SHADOW_ENV) else "")
            + (" ".join(f"{name}={_TALLY[name]}" for name in HITS_PROBE_TALLIES if _TALLY[name]) + " "
               if knob_on(HITS_PROBE_ENV) else "")
            + " ".join(f"{name}={value}" for name, value in stats.items()))
    print(line, file=sys.stderr, flush=True)
    if mixed_capture() and _TALLY["mirror_misses"]:
        announce("prefill_mirror_mismatch", misses=_TALLY["mirror_misses"],
                 why="a prefill part's GPU row range or staged blocks disagreed with the host copies it was packed "
                     "from; its staging plan may have dropped rows")
    return line


def warm_cache(owner):
    cache = getattr(owner, "hisparse_cache", None)
    if cache is None:
        return owner.kv_cache
    return planes(cache.runtime.hot.attention_cache, int(owner.num_kv_heads))


ADMIT_ENV = "TAAF_VLLM_HOST_KV_ADMIT"
ADMIT_MODES = ("", "worst", "real")
ADMIT_LENGTHS = (40960, 87040, 112640)
ADMIT_CHUNK_TOKENS = 2048
ADMIT_TAIL_PAGES = 2
_ADMIT = {"early": 0}


def admit_mode():
    mode = str(os.environ.get(ADMIT_ENV) or "").strip().lower()
    if mode not in ADMIT_MODES:
        raise RuntimeError(f"{ADMIT_ENV}={mode!r} is not one of {ADMIT_MODES}")
    return mode


def pinned_pages(coordinator, request_id):
    counts = [sum(1 for block in manager.req_to_blocks.get(request_id, ()) if not block.is_null)
              for manager in coordinator.resident_managers]
    return max(counts or [0])


def hot_blocks(coordinator):
    return sum(int(manager.blocks_per_request) for manager in coordinator.hot_managers)


def early_hot(coordinator, request_id):
    if admit_mode() != "real" or not coordinator.hot_managers:
        return False
    pinned = pinned_pages(coordinator, request_id)
    if pinned <= hot_blocks(coordinator) + ADMIT_TAIL_PAGES:
        return False
    _ADMIT["early"] += 1
    announce("early_hot", request=request_id, pinned_pages=pinned, hot_blocks=hot_blocks(coordinator),
             free_blocks=coordinator.gpu_pool.get_num_free_blocks(),
             watermark=coordinator.transition_watermark)
    return True


def group_blocks(spec, tokens, host_resident, real, chunk_tokens=ADMIT_CHUNK_TOKENS):
    if host_resident:
        return 0
    kind = type(spec).__name__
    size = max(int(getattr(spec, "block_size", 0) or 1), 1)
    if kind == "HiSparseHotSpec":
        return int(spec.blocks_per_request)
    if kind == "HiSparseResidentSpec":
        if real:
            return ADMIT_TAIL_PAGES + -(-chunk_tokens // size)
        return -(-tokens // size)
    if "Mamba" in kind or "CircularBuffer" in kind:
        return 1
    return -(-tokens // size)


def admit_capacity(groups, num_blocks, lengths=ADMIT_LENGTHS, chunk_tokens=ADMIT_CHUNK_TOKENS):
    rows = []
    for tokens in lengths:
        need = {}
        for real in (False, True):
            need[real] = sum(group_blocks(group.kv_cache_spec, tokens, bool(getattr(group, "host_resident", False)),
                                          real, chunk_tokens) for group in groups)
        rows.append({"tokens": tokens, "worst_blocks": need[False], "real_blocks": need[True],
                     "worst_lanes": num_blocks // max(need[False], 1),
                     "real_lanes": num_blocks // max(need[True], 1)})
    return rows


def spec_field(spec, name, default):
    try:
        return getattr(spec, name, default)
    except NotImplementedError:
        return default


def spec_kind(spec):
    role = spec_field(spec, "cache_role", None)
    role = getattr(role, "name", role)
    return (type(spec).__name__, str(role) if role is not None else "-", int(spec_field(spec, "block_size", 0) or 0),
            int(spec_field(spec, "page_size_bytes", 0) or 0), spec_field(spec, "head_size", None),
            str(spec_field(spec, "dtype", "-")), int(spec_field(spec, "tokens_per_state", 1) or 1))


def group_census(groups, lengths=ADMIT_LENGTHS, chunk_tokens=ADMIT_CHUNK_TOKENS):
    kinds = {}
    for index, group in enumerate(groups):
        host = bool(getattr(group, "host_resident", False))
        key = (*spec_kind(group.kv_cache_spec), len(group.layer_names), host)
        row = kinds.setdefault(key, {"groups": [], "spec": group.kv_cache_spec, "host": host})
        row["groups"].append(index)
    rows = []
    for key, row in kinds.items():
        kind, role, block, page, head, dtype, per_state, layers, host = key
        lane = [len(row["groups"]) * group_blocks(row["spec"], tokens, host, True, chunk_tokens) for tokens in lengths]
        rows.append({"kind": kind, "role": role, "groups": row["groups"], "layers": layers, "page_bytes": page,
                     "block": block, "head": head, "dtype": dtype, "tokens_per_state": per_state, "host": host,
                     "lane_blocks": lane})
    return rows


def indexer_head(rows):
    found = [row for row in rows if row["role"] == "INDEXER"]
    return (found[0]["head"], found[0]["dtype"]) if found else (None, None)


def announce_groups(kv_cache_config):
    rows = group_census(kv_cache_config.kv_cache_groups)
    head, dtype = indexer_head(rows)
    print(f"HOST_KV groups: {len(kv_cache_config.kv_cache_groups)} groups in {len(rows)} kinds, pool "
          f"{kv_cache_config.num_blocks} blocks; indexer head {head} {dtype}; "
          + "; ".join(f"{len(row['groups'])}x {row['kind']} role={row['role']} groups={row['groups'][0]}"
                      f"{'..' + str(row['groups'][-1]) if len(row['groups']) > 1 else ''} layers={row['layers']} "
                      f"page={row['page_bytes']} block={row['block']} head={row['head']} dtype={row['dtype']} "
                      f"per_state={row['tokens_per_state']}{' host' if row['host'] else ''} lane_blocks="
                      + "/".join(str(value) for value in row["lane_blocks"]) for row in rows)
          + " (lane_blocks at " + "/".join(str(tokens) for tokens in ADMIT_LENGTHS) + " tokens, pages released "
          "per chunk)", file=sys.stderr, flush=True)
    return rows


LOST_REPORT_S = 60.0
_LOST = {"room": 0, "pressure": 0, "reported": 0.0, "reserve": 0, "armed": False}
HOLD_ENV = "TAAF_VLLM_HOST_KV_HOLD_CLEAN"
HOLD_RESERVE_ENV = "TAAF_VLLM_HOST_KV_HOLD_RESERVE"
HOLD_SLACK_BLOCKS = 64
HOLD_REPORT_S = 60.0
HOLD_PASS_S = 0.02
HOLD_IDLE_PASS_S = 0.25
_HOLD = {"held": 0, "released": 0, "passes": 0, "reported": 0.0, "reserve": 0, "armed": False, "skipped": 0,
         "holders": {}, "dirty": False, "pass_at": float("-inf"), "last": 0}


def free_blocks(coordinator):
    pool = coordinator.gpu_pool
    return int(pool.get_num_free_blocks()) if pool is not None else 0


def lost_line():
    return (f"QSA_HISPARSE tally page_lost room={_LOST['room']} pressure={_LOST['pressure']} "
            f"reserve={_LOST['reserve']} why=a clean resident page a lane could still read was handed out by the "
            f"pool; room counts the losses while the pool still held at least the transition watermark free")


def report_lost(now=None):
    now = time.monotonic() if now is None else now
    if now - _LOST["reported"] < LOST_REPORT_S:
        return None
    _LOST["reported"] = now
    line = lost_line()
    print(line, file=sys.stderr, flush=True)
    return line


def arm_page_census(coordinator):
    if getattr(coordinator, "_taaf_page_census", False) or not coordinator.resident_managers:
        return False
    lose = coordinator._lose_page
    _LOST["reserve"] = int(coordinator.transition_watermark)

    def counted(request_id, page_idx):
        state = coordinator.request_states.get(request_id)
        if state is not None and page_idx in state.unpinned_pages:
            _LOST["room" if free_blocks(coordinator) >= _LOST["reserve"] else "pressure"] += 1
            report_lost()
        return lose(request_id, page_idx)

    coordinator._lose_page = counted
    coordinator._taaf_page_census = True
    _LOST["armed"] = True
    announce("page_census", reserve=_LOST["reserve"],
             rule="every clean resident page the pool hands out while a lane could still read it is counted, "
                  "split by whether the pool held at least the transition watermark free at that moment")
    return True


def hold_clean():
    said = str(os.environ.get(HOLD_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{HOLD_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def hold_reserve(coordinator):
    said = str(os.environ.get(HOLD_RESERVE_ENV) or "").strip()
    if said and not said.isdigit():
        raise RuntimeError(f"{HOLD_RESERVE_ENV}={said!r} is not a block count")
    return int(said) if said else int(coordinator.transition_watermark)


def hold_line(coordinator):
    return (f"QSA_HISPARSE tally hold_clean held={_HOLD['held']} released={_HOLD['released']} "
            f"passes={_HOLD['passes']} skipped={_HOLD['skipped']} free={free_blocks(coordinator)} "
            f"reserve={_HOLD['reserve']}")


def report_hold(coordinator, now=None):
    now = time.monotonic() if now is None else now
    if now - _HOLD["reported"] < HOLD_REPORT_S:
        return None
    _HOLD["reported"] = now
    line = hold_line(coordinator)
    print(line, file=sys.stderr, flush=True)
    return line


def held_order(coordinator):
    cold = set(_HOT["released"])
    ids = list(coordinator.request_states)
    idle = [rid for rid in ids if rid in cold]
    return idle + [rid for rid in ids if rid not in cold and coordinator._can_read_from_host(rid)]


def release_held(coordinator):
    target = _HOLD["reserve"] + HOLD_SLACK_BLOCKS
    released = 0
    for request_id in held_order(coordinator):
        if free_blocks(coordinator) >= target:
            break
        state = coordinator.request_states.get(request_id)
        if state is None:
            continue
        for page_idx in sorted(state.pinned_clean):
            if free_blocks(coordinator) >= target:
                break
            if coordinator._unpin_page(request_id, state, page_idx):
                released += 1
    _HOLD["released"] += released
    _HOLD["passes"] += 1
    report_hold(coordinator)
    return released


def paced_release(coordinator, now=None):
    now = time.monotonic() if now is None else now
    wait = HOLD_PASS_S if _HOLD["last"] else HOLD_IDLE_PASS_S
    if not _HOLD["dirty"] and now - _HOLD["pass_at"] < wait:
        _HOLD["skipped"] += 1
        return None
    _HOLD["dirty"] = False
    _HOLD["pass_at"] = now
    _HOLD["last"] = release_held(coordinator)
    return _HOLD["last"]


def arm_hold_clean(coordinator):
    if not hold_clean() or not coordinator.resident_managers or not coordinator.hot_managers:
        return False
    if getattr(coordinator, "_taaf_hold_clean", False):
        return False
    _HOLD["reserve"] = hold_reserve(coordinator)

    def held_clean(request_id, state, page_idx):
        state.pinned_clean.add(page_idx)
        if free_blocks(coordinator) >= _HOLD["reserve"]:
            _HOLD["held"] += 1
            _HOLD["dirty"] = True
            report_hold(coordinator)
            return
        if coordinator._can_read_from_host(request_id) and coordinator._unpin_page(request_id, state, page_idx):
            _HOLD["released"] += 1
        paced_release(coordinator)

    def held_unpin(request_id, state):
        if free_blocks(coordinator) >= _HOLD["reserve"]:
            return
        paced_release(coordinator)

    coordinator._page_became_clean = held_clean
    coordinator._unpin_clean_pages = held_unpin
    coordinator._taaf_hold_clean = True
    _HOLD["armed"] = True
    announce("hold_clean", knob=HOLD_ENV, reserve=_HOLD["reserve"], slack=HOLD_SLACK_BLOCKS,
             rule="a clean resident page keeps its pool reference while the pool holds at least the reserve free, so "
                  "the lane reads it from the GPU; under the reserve, held pages are unpinned, idle lanes first "
                  "(hot region released) and then the oldest lanes that read from host, until the pool is back "
                  "above the reserve")
    return True


def announce_admit(coordinator, kv_cache_config):
    mode = admit_mode()
    if not coordinator.resident_managers:
        return None
    arm_page_census(coordinator)
    arm_hold_clean(coordinator)
    try:
        announce_groups(kv_cache_config)
    except Exception as failure:
        print(f"HOST_KV groups: census line failed ({type(failure).__name__}: {failure}); serving continues", file=sys.stderr, flush=True)
    rows = admit_capacity(kv_cache_config.kv_cache_groups, int(kv_cache_config.num_blocks))
    print(f"HOST_KV admit {mode or 'worst'}: pool {kv_cache_config.num_blocks} blocks, hot {hot_blocks(coordinator)} "
          f"blocks a request, transition watermark {coordinator.transition_watermark}; lanes that fit "
          + ", ".join(f"{row['tokens']} tokens {row['real_lanes']} with pages released per chunk "
                      f"({row['real_blocks']} blocks) against {row['worst_lanes']} pinned whole "
                      f"({row['worst_blocks']} blocks)" for row in rows)
          + ("; a request leaves its prompt pages pinned until the pool runs low" if mode != "real" else
             "; a request whose pinned pages pass its hot region reads from host at once, so its clean pages "
             "return to the pool as each chunk is mirrored"),
          file=sys.stderr, flush=True)
    return rows


HOT_RELEASE_ENV = "TAAF_VLLM_HOST_KV_HOT_RELEASE"
HOT_RELEASE_MARK = "_taaf_hot_release"
HOT_RESET_MARK = "_taaf_hot_reset"
HOT_REPORT_S = 60.0
SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
SCHEDULER_CLASS = "Scheduler"
BLOCK_TABLES_MODULE = "vllm.v1.worker.gpu.block_table"
BLOCK_TABLES_CLASS = "BlockTables"
STATES_MODULE = "vllm.v1.worker.gpu.states"
STATES_CLASS = "RequestState"
NEW_SLOT_MARK = "_taaf_hot_new_slot"
_HOT = {"coordinator": None, "released": set(), "releases": 0, "blocks": 0, "reallocs": 0, "deferred": 0,
        "passes": 0, "reported": 0.0}
_RESET = {"pending": set(), "resets": 0, "starts": 0, "fresh": None, "misses": None, "slots": 0}
HOT_KEEP_ENV = "TAAF_VLLM_HOT_KEEP"
HOT_KEEP_RESERVE_ENV = "TAAF_VLLM_HOT_KEEP_RESERVE"
HOT_KEEP_RESERVE_DEFAULT = 512
PAUSED_STATUS = "WAITING_FOR_STREAMING_REQ"
_KEEP = {"held": set(), "kept": 0, "pressed": 0, "continued": 0, "moved": 0}
HOT_KEEP_WARM_ENV = "TAAF_VLLM_HOT_KEEP_WARM"
HOT_KEEP_WARM_MODES = ("", "0", "1", "held", "owners")
WARM_STATUSES = ("RUNNING", "WAITING")
_WARM = {"kinds": {}, "kept_running": 0, "kept_waiting": 0, "pressed": 0, "owner_stale": 0, "owner_conflicts": 0,
         "first": None, "misses": None, "first_pending": set(), "firsts": 0}


def hot_keep_warm():
    said = str(os.environ.get(HOT_KEEP_WARM_ENV) or "").strip().lower()
    if said not in HOT_KEEP_WARM_MODES:
        raise RuntimeError(f"{HOT_KEEP_WARM_ENV}={said!r} is not one of {HOT_KEEP_WARM_MODES}")
    return "" if said == "0" else said


def warm_held():
    return hot_keep_warm() in ("1", "held")


def warm_owners():
    return hot_keep_warm() in ("1", "owners")


def status_name(request):
    status = getattr(request, "status", None)
    return getattr(status, "name", str(status))


def keep_kind(request, keep, held_too):
    if not keep:
        return None
    if paused_session(request):
        return "paused"
    if held_too and status_name(request) in WARM_STATUSES:
        return status_name(request).lower()
    return None


def note_warm(request_id, kind):
    kinds = _WARM["kinds"]
    if kinds.get(request_id) != kind:
        kinds[request_id] = kind
        _WARM[f"kept_{kind}"] += 1


def warm_line():
    return (f"QSA_HISPARSE tally hot_keep_warm mode={hot_keep_warm()} kept_running={_WARM['kept_running']} "
            f"kept_waiting={_WARM['kept_waiting']} pressed={_WARM['pressed']} held_now={len(_WARM['kinds'])}")


def warm_worker_line():
    misses = _WARM["misses"]
    return (f"QSA_HISPARSE tally hot_keep_worker mode={hot_keep_warm()} continued={_KEEP['continued']} "
            f"moved={_KEEP['moved']} new_slots={_RESET['slots']} owner_stale={_WARM['owner_stale']} "
            f"owner_conflicts={_WARM['owner_conflicts']} kept_first_decodes={_WARM['firsts']} "
            f"misses_kept_first_decode={int(misses.item()) if misses is not None else 0}")


def flush_firsts(resetting):
    first = _WARM["first"]
    waiting = _WARM["first_pending"]
    if first is None or not waiting or capturing():
        return 0
    rows = sorted(waiting - set(resetting))
    waiting.clear()
    if not rows:
        return 0
    first.index_fill_(0, torch.tensor(rows, dtype=torch.long).to(first.device, non_blocking=True), True)
    _WARM["firsts"] += len(rows)
    return len(rows)


def row_lists(row, position, block):
    return row is not None and position < len(row) and block in row[position]


def hot_keep():
    said = str(os.environ.get(HOT_KEEP_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{HOT_KEEP_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def hot_keep_reserve():
    said = str(os.environ.get(HOT_KEEP_RESERVE_ENV) or "").strip()
    return int(said) if said.isdigit() else HOT_KEEP_RESERVE_DEFAULT


def paused_session(request):
    status = getattr(request, "status", None)
    return getattr(status, "name", str(status)) == PAUSED_STATUS


def keep_line():
    return (f"QSA_HISPARSE tally hot_keep kept={_KEEP['kept']} pressed={_KEEP['pressed']} "
            f"continued={_KEEP['continued']} moved={_KEEP['moved']} held={len(_KEEP['held'])}")


def hot_release():
    said = str(os.environ.get(HOT_RELEASE_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{HOT_RELEASE_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def arm_hot_release(coordinator):
    if not hot_release() or not coordinator.hot_managers:
        return False
    import importlib

    _HOT["coordinator"] = coordinator
    scheduler_class = getattr(importlib.import_module(SCHEDULER_MODULE), SCHEDULER_CLASS)
    if release_on_build(scheduler_class):
        announce("hot_release", knob=HOT_RELEASE_ENV, hot_blocks=hot_blocks(coordinator),
                 rule="each pass frees the hot region of every request it does not schedule, behind the fence of "
                      "the last step that scheduled it, and asks for a fresh one when the request is scheduled again")
    return True


def release_on_build(scheduler_class):
    if getattr(scheduler_class, HOT_RELEASE_MARK, False):
        return False
    build = scheduler_class._build_kv_connector_meta

    def released_build(self, connector, scheduler_output):
        coordinator = _HOT["coordinator"]
        if coordinator is not None:
            release_cold(self, scheduler_output, coordinator)
        return build(self, connector, scheduler_output)

    scheduler_class._build_kv_connector_meta = released_build
    setattr(scheduler_class, HOT_RELEASE_MARK, True)
    return True


def hot_holders(managers):
    return sorted({request_id for manager in managers for request_id, blocks in manager.req_to_blocks.items()
                   if blocks})


def take_hot(managers, request_id):
    blocks = []
    for manager in managers:
        blocks.extend(block for block in manager.req_to_blocks.pop(request_id, ()) if not block.is_null)
        manager.require_hot(request_id)
    return blocks


def fenced_free(scheduler, request, blocks):
    fence = int(getattr(request, "last_sched_seq", 0) or 0)
    if not getattr(scheduler, "defer_block_free", False) or fence <= scheduler.processed_step_seq:
        scheduler.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
        return False
    scheduler.deferred_frees.append((fence, blocks))
    return True


def release_cold(scheduler, scheduler_output, coordinator):
    managers = coordinator.hot_managers
    scheduled = scheduler_output.num_scheduled_tokens
    released = _HOT["released"]
    for request_id in sorted(released):
        if request_id not in scheduler.requests:
            released.discard(request_id)
        elif request_id in scheduled and all(manager.has_hot(request_id) for manager in managers):
            released.discard(request_id)
            _HOT["reallocs"] += 1
    freed = 0
    keep = hot_keep()
    held_too = keep and warm_held()
    kinds = _WARM["kinds"]
    spare = None
    if keep:
        _KEEP["held"].intersection_update(scheduler.requests)
    if held_too:
        for request_id in [one for one in kinds if one not in scheduler.requests]:
            kinds.pop(request_id)
    for request_id in hot_holders(managers):
        if request_id in scheduled:
            _KEEP["held"].discard(request_id)
            kinds.pop(request_id, None)
            continue
        request = scheduler.requests.get(request_id)
        if request is None or request.is_finished():
            continue
        kind = keep_kind(request, keep, held_too)
        if kind is not None:
            if spare is None:
                spare = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            if spare >= hot_keep_reserve():
                if kind == "paused":
                    if request_id not in _KEEP["held"]:
                        _KEEP["held"].add(request_id)
                        _KEEP["kept"] += 1
                    if held_too:
                        kinds[request_id] = kind
                else:
                    _KEEP["held"].add(request_id)
                    note_warm(request_id, kind)
                continue
            if kind == "paused":
                _KEEP["pressed"] += 1
            else:
                _WARM["pressed"] += 1
        kinds.pop(request_id, None)
        _KEEP["held"].discard(request_id)
        blocks = take_hot(managers, request_id)
        coordinator.block_table_updates.add(request_id)
        released.add(request_id)
        _HOT["releases"] += 1
        _HOT["blocks"] += len(blocks)
        _HOT["deferred"] += int(fenced_free(scheduler, request, blocks))
        freed += len(blocks)
    _HOT["passes"] += 1
    report_release()
    return freed


def release_line():
    return (f"QSA_HISPARSE tally hot_release passes={_HOT['passes']} releases={_HOT['releases']} "
            f"released_blocks={_HOT['blocks']} deferred={_HOT['deferred']} reallocs={_HOT['reallocs']} "
            f"held_out={len(_HOT['released'])}")


def report_release(now=None):
    import time

    now = time.monotonic() if now is None else now
    if now - _HOT["reported"] < HOT_REPORT_S:
        return None
    _HOT["reported"] = now
    line = release_line()
    print(line, file=sys.stderr, flush=True)
    if hot_keep():
        print(keep_line(), file=sys.stderr, flush=True)
    if hot_keep_warm():
        print(warm_line(), file=sys.stderr, flush=True)
    return line


def install_hot_reset(vllm_config, device):
    if not hot_release():
        return False
    import importlib

    if _RESET["fresh"] is None:
        lanes = int(vllm_config.scheduler_config.max_num_seqs)
        _RESET["fresh"] = torch.zeros(lanes + 1, dtype=torch.bool, device=device)
        _RESET["misses"] = torch.zeros(1, dtype=torch.int64, device=device)
    tables = getattr(importlib.import_module(BLOCK_TABLES_MODULE), BLOCK_TABLES_CLASS)
    if track_hot_rows(tables):
        announce("hot_reset", knob=HOT_RELEASE_ENV,
                 rule="a request whose hot-region block ids change is cold: its hot bookkeeping row is cleared "
                      "before the next forward, so no slot maps to a row the new blocks do not hold")
    states = getattr(importlib.import_module(STATES_MODULE), STATES_CLASS)
    if track_new_slots(states):
        announce("hot_reset_new_slot", knob=HOT_RELEASE_ENV,
                 rule="a request that takes a state slot starts cold: the slot's hot bookkeeping row is cleared "
                      "before the next forward even when its hot blocks carry the ids the slot's last request held")
    if hot_keep():
        announce("hot_keep", knob=HOT_KEEP_ENV, reserve=hot_keep_reserve(),
                 rule="a paused session keeps its hot region while the pool holds at least the reserve of free "
                      "blocks, and a streaming continuation re-added in the slot it left keeps its hot bookkeeping "
                      "row when its hot block ids are unchanged and owned by no other lane and its host source "
                      "blocks extend the ones it left with; anything else starts cold as before")
    if hot_keep_warm():
        if not hot_keep():
            raise RuntimeError(f"{HOT_KEEP_WARM_ENV} extends {HOT_KEEP_ENV}, which is off")
        if _WARM["first"] is None:
            _WARM["first"] = torch.zeros_like(_RESET["fresh"])
            _WARM["misses"] = torch.zeros_like(_RESET["misses"])
        announce("hot_keep_warm", knob=HOT_KEEP_WARM_ENV, mode=hot_keep_warm(), reserve=hot_keep_reserve(),
                 rule="held: a running or waiting request the pass did not schedule keeps its hot region while "
                      "the pool holds at least the reserve of free blocks, as a paused session does; owners: a "
                      "hot block that reaches a lane clears another lane's bookkeeping only when that lane's "
                      "current hot row still lists the block")
    return True


def track_new_slots(states_class):
    if getattr(states_class, NEW_SLOT_MARK, False):
        return False
    add = states_class.add_request
    remove = states_class.remove_request

    def tracked_remove(self, req_id, *args, **kwargs):
        index = remove(self, req_id, *args, **kwargs)
        self.__dict__["_taaf_left"] = None if index is None else (req_id, int(index))
        return index

    def tracked_add(self, req_id, *args, **kwargs):
        left = self.__dict__.pop("_taaf_left", None)
        result = add(self, req_id, *args, **kwargs)
        index = self.req_id_to_index.get(req_id)
        if index is not None:
            if hot_keep() and left == (req_id, int(index)):
                _KEEP["continued"] += 1
                if _WARM["first"] is not None:
                    _WARM["first_pending"].add(int(index))
            else:
                _RESET["pending"].add(int(index))
                _RESET["slots"] += 1
        return result

    states_class.add_request = tracked_add
    states_class.remove_request = tracked_remove
    setattr(states_class, NEW_SLOT_MARK, True)
    return True


def hot_table_pointers():
    pointers = set()
    for layer in _ORDER:
        table = getattr(getattr(getattr(layer, "hisparse_cache", None), "runtime", None), "hot_block_table", None)
        if table is not None:
            pointers.add(table.data_ptr())
    return pointers


def source_table_pointers():
    pointers = set()
    for layer in _ORDER:
        table = getattr(getattr(layer, "hisparse_cache", None), "source_block_table", None)
        if table is not None:
            pointers.add(table.data_ptr())
    return pointers


def source_groups(tables):
    known = getattr(tables, "_taaf_source_groups", None)
    if known:
        return known
    pointers = source_table_pointers()
    found = tuple(index for index, table in enumerate(getattr(tables, "input_block_tables", ()))
                  if table.data_ptr() in pointers)
    if found:
        tables._taaf_source_groups = found
    return found


def source_moved(tables, req_index, new_block_ids, overwrite):
    groups = source_groups(tables)
    if not groups:
        return overwrite
    rows = tables.__dict__.setdefault("_taaf_source_rows", {})
    now = tuple(tuple(int(block) for block in new_block_ids[index]) for index in groups)
    before = rows.get(req_index)
    if not overwrite:
        rows[req_index] = now if before is None else tuple(held + more for held, more in zip(before, now))
        return False
    rows[req_index] = now
    return before is not None and any(fresh[:len(held)] != held for fresh, held in zip(now, before))


def hot_groups(tables):
    known = getattr(tables, "_taaf_hot_groups", None)
    if known:
        return known
    pointers = hot_table_pointers()
    found = tuple(index for index, table in enumerate(getattr(tables, "input_block_tables", ()))
                  if table.data_ptr() in pointers)
    if found:
        tables._taaf_hot_groups = found
    return found


def hot_row_after(before, new_block_ids, groups, overwrite):
    added = tuple(tuple(int(block) for block in new_block_ids[index]) for index in groups)
    if overwrite or before is None:
        return added
    return tuple(held + more for held, more in zip(before, added))


def track_hot_rows(tables_class):
    if getattr(tables_class, HOT_RESET_MARK, False):
        return False
    append = tables_class.append_block_ids

    def tracked_append(self, req_index, new_block_ids, overwrite):
        result = append(self, req_index, new_block_ids, overwrite)
        groups = hot_groups(self)
        if groups:
            rows = self.__dict__.setdefault("_taaf_hot_rows", {})
            before = rows.get(req_index)
            after = hot_row_after(before, new_block_ids, groups, overwrite)
            empty = tuple(() for _ in groups)
            unchanged = after == (empty if before is None else before)
            if not unchanged:
                _RESET["pending"].add(int(req_index))
            if hot_keep() and source_moved(self, req_index, new_block_ids, overwrite):
                _RESET["pending"].add(int(req_index))
                _KEEP["moved"] += 1
            owners = self.__dict__.setdefault("_taaf_hot_owners", {})
            exact = warm_owners()
            for position, group in enumerate(groups):
                for block in new_block_ids[group]:
                    block = int(block)
                    held = owners.get((group, block))
                    if held is not None and held != req_index:
                        if exact and not row_lists(rows.get(held), position, block):
                            _WARM["owner_stale"] += 1
                            if unchanged:
                                _RESET["pending"].add(int(req_index))
                        else:
                            _WARM["owner_conflicts"] += int(exact)
                            _RESET["pending"].add(int(req_index))
                            _RESET["pending"].add(int(held))
                    owners[(group, block)] = req_index
            rows[req_index] = after
        return result

    tables_class.append_block_ids = tracked_append
    setattr(tables_class, HOT_RESET_MARK, True)
    return True


def index_groups():
    seen = {}
    for layer in _ORDER:
        runtime = getattr(getattr(layer, "hisparse_cache", None), "runtime", None)
        group = getattr(runtime, "index_group", None)
        if group is not None:
            seen.setdefault(id(group), group)
    return list(seen.values())


def clear_rows(group, index):
    count = index.numel()
    before = rec2_before(group, index)
    group.device_global_indices.index_fill_(0, index, -1)
    if before is not None:
        rec2_cleared(group, index, before, 2)
    group.lru_slots.index_copy_(0, index, group.lru_init.unsqueeze(0).expand(count, -1).contiguous())


def flush_resets(groups=None, stream=None):
    pending = _RESET["pending"]
    flush_firsts(pending)
    if not pending or capturing():
        return 0
    groups = index_groups() if groups is None else groups
    if not groups:
        return 0
    rows = sorted(pending)
    pending.clear()
    device = groups[0].device_global_indices.device
    index = torch.tensor(rows, dtype=torch.long).to(device, non_blocking=True)
    compute = current() if stream is None else stream
    for group in groups:
        compute.wait_stream(group.copy_stream)
        clear_rows(group, index)
        group.copy_stream.wait_stream(compute)
    fresh = _RESET["fresh"]
    if fresh is not None:
        fresh.index_fill_(0, index.to(fresh.device), True)
    if _WARM["first"] is not None:
        _WARM["first"].index_fill_(0, index.to(_WARM["first"].device), False)
    rec2_reset(index)
    _RESET["resets"] += len(rows)
    announce("hot_reset_first", rows=len(rows), groups=len(groups))
    return len(rows)


def lane_index(states, dummy):
    return torch.where(states >= 0, states, torch.full_like(states, dummy)).to(torch.long)


def last_target():
    targets = [layer for layer in _ORDER if not is_draft(layer)]
    return targets[-1] if targets else None


def count_fresh_misses(layer, meta):
    fresh = _RESET["fresh"]
    if fresh is None or is_draft(layer):
        return False
    runtime = layer.hisparse_cache.runtime
    states = getattr(runtime, "request_state_indices", None)
    decodes = int(meta.num_decodes)
    if states is None or decodes == 0:
        return False
    steps = max(1, int(getattr(runtime, "_swap_step", 1) or 1))
    tally_flagged(runtime, states, fresh, _RESET["misses"], decodes, steps, layer is last_target())
    if _WARM["first"] is not None:
        tally_flagged(runtime, states, _WARM["first"], _WARM["misses"], decodes, steps, layer is last_target())
    return True


def tally_flagged(runtime, states, flagged, misses, decodes, steps, last):
    if fused() and fused_ready(("fresh", str(flagged.device))):
        fused_ops().fresh_misses(runtime.index_group.shared_topk.swap_counts, states, flagged, misses, decodes,
                                 steps, last)
        return
    counts = runtime.index_group.shared_topk.swap_counts[:steps * decodes].view(steps, decodes)
    index = lane_index(states[:decodes], flagged.numel() - 1)
    flags = flagged.index_select(0, index)
    misses.add_((counts.to(torch.int64).sum(0) * flags).sum().view(1))
    if last:
        flagged.index_fill_(0, index, False)


def reset_line():
    misses = _RESET["misses"]
    return (f"QSA_HISPARSE tally hot_reset starts={_RESET['starts']} resets={_RESET['resets']} "
            f"misses_after_reset={int(misses.item()) if misses is not None else 0} new_slots={_RESET['slots']}")


def report_reset():
    _RESET["starts"] += 1
    if _RESET["fresh"] is None or _RESET["starts"] % REPORT_EVERY or capturing():
        return None
    line = reset_line()
    print(line, file=sys.stderr, flush=True)
    if hot_keep_warm():
        print(warm_worker_line(), file=sys.stderr, flush=True)
    return line


def hot_cap_line():
    tally = _HOT_CAP["tally"].to("cpu", copy=True).tolist()
    lanes, over, unique, unique_over = (int(one) for one in tally)
    return (f"QSA_HISPARSE tally hot_cap window={_HOT_CAP['lines']} starts={_HOT_CAP['starts']} "
            f"cap={_HOT_CAP['cap']} lanes={lanes} overflow={over} overflow_share={over / max(lanes, 1):.4f} "
            f"unique_mean={unique / max(lanes, 1):.1f} overflow_unique_mean={unique_over / max(over, 1):.1f} "
            f"eager_calls={_HOT_CAP['calls']} eager_spill_passes={_HOT_CAP['spills']}")


def report_hot_cap():
    _HOT_CAP["starts"] += 1
    if _HOT_CAP["tally"] is None or _HOT_CAP["starts"] % HOT_CAP_EVERY or capturing():
        return None
    line = hot_cap_line()
    _HOT_CAP["tally"].zero_()
    _HOT_CAP["lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def copy_async():
    return str(os.environ.get(COPY_ASYNC_ENV) or "").strip().lower() in ("1", "on", "true")


def cache_extent(cache):
    return (sum((size - 1) * stride for size, stride in zip(cache.shape, cache.stride())) + 1) * cache.element_size()


def device_alias(cache, device):
    key = (cache.data_ptr(), cache_extent(cache), str(device))
    alias = _COPY["aliases"].get(key)
    if alias is None:
        holder = type("HostAlias", (), {})()
        holder.__cuda_array_interface__ = {"shape": (key[1],), "typestr": "|u1", "data": (key[0], False),
                                           "version": 2, "strides": None}
        alias = torch.as_tensor(holder, device=device)
        if alias.data_ptr() != key[0] or alias.device.type != "cuda":
            raise RuntimeError("the host pool alias does not share the registered host pointer")
        _COPY["aliases"][key] = alias
    return alias


def block_bytes_view(cache, alias, num_blocks):
    kernel, remainder = divmod(cache.shape[0], num_blocks)
    if remainder or cache.stride(-1) != 1 or alias.numel() < cache_extent(cache):
        return None
    size = cache.element_size()
    shape = (num_blocks, kernel, *cache.shape[1:-1], cache.shape[-1] * size)
    strides = (cache.stride(0) * size * kernel, *(stride * size for stride in cache.stride()[:-1]), 1)
    return torch.as_strided(alias, shape, strides, alias.storage_offset())


def copy_blocks_on_device(caches, num_blocks, copies, alias_of, device):
    pairs = torch.tensor([[int(copy.src_block_id), int(copy.dst_block_id)] for copy in copies], dtype=torch.int64)
    if device is not None and torch.device(device).type == "cuda":
        pairs = pairs.pin_memory().to(device, non_blocking=True)
    src, dst = pairs.unbind(dim=1)
    seen = set()
    views = []
    for cache in caches:
        key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()))
        if key in seen:
            continue
        seen.add(key)
        try:
            blocks = block_bytes_view(cache, alias_of(cache), num_blocks)
        except Exception as error:
            _COPY["fallbacks"] += 1
            announce("copy_async_refused", error=f"{type(error).__name__}: {error}",
                     why="the host pool could not be addressed from the GPU, so this step copies on the CPU")
            return False
        if blocks is None:
            _COPY["fallbacks"] += 1
            announce("copy_async_refused", shape=tuple(cache.shape), stride=tuple(cache.stride()),
                     why="a host cache is not whole blocks of contiguous rows, so this step copies on the CPU")
            return False
        views.append(blocks)
    for blocks in views:
        blocks[dst] = blocks[src]
    _COPY["steps"] += 1
    _COPY["blocks"] += len(copies)
    announce("copy_async", caches=len(views), blocks=len(copies),
             why="host pool block copies run as GPU copies over the registered host pool on the step's stream, "
                 "after the previous host writes, instead of CPU copies the forward waits for")
    return True


def union_count():
    said = str(os.environ.get(UNION_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{UNION_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def install_union(vllm_config, width, device):
    if not union_count() or _UNION["hist"] is not None:
        return False
    speculative = getattr(vllm_config, "speculative_config", None)
    query = 1 + int(getattr(speculative, "num_speculative_tokens", 0) or 0)
    _UNION.update(width=int(width), query=query, starts=0, lines=0)
    _UNION["hist"] = torch.zeros(query * int(width) + 1, dtype=torch.int64, device=device)
    _UNION["picks"] = torch.zeros(1, dtype=torch.int64, device=device)
    announce("union_count", knob=UNION_ENV, query=query, top_k=int(width), need=query * int(width),
             every=UNION_EVERY,
             rule="each decode step and target layer adds every lane's count of distinct rows its verification "
                  "rows pick to a GPU histogram that the connector's start_step reads and clears")
    return True


def lane_unions(picks, counts, decodes, query):
    width = picks.shape[1]
    rows = picks[:decodes * query].to(torch.int32)
    columns = torch.arange(width, dtype=torch.int32, device=picks.device)
    valid = (columns.view(1, width) < counts[:decodes * query].to(torch.int32).view(-1, 1)) & (rows >= 0)
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.sort(torch.where(valid, rows, torch.full_like(rows, sentinel)).view(decodes, query * width),
                         dim=1).values
    fresh = torch.ones_like(ordered, dtype=torch.bool)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    unique = ((ordered != sentinel) & fresh).sum(1)
    picked = valid.view(decodes, query * width).sum(1)
    return unique, picked


def count_union(layer, logical, meta):
    hist = _UNION["hist"]
    if hist is None or is_draft(layer):
        return False
    decodes = int(meta.num_decodes)
    query = int(meta.decode_max_query_len)
    if decodes == 0:
        return False
    if decodes * query != int(meta.num_decode_tokens):
        announce("union_uneven", decodes=decodes, query=query, rows=int(meta.num_decode_tokens),
                 why="decode rows do not form one query length per lane, so this step is not counted")
        return False
    width = logical.shape[1] - 1
    unique, picked = lane_unions(logical[:, :width], logical[:, width], decodes, query)
    live = picked > 0
    states = getattr(getattr(layer.hisparse_cache, "runtime", None), "request_state_indices", None)
    if states is not None:
        live = live & (states[:decodes] >= 0)
    weight = live.to(torch.int64)
    hist.index_add_(0, unique.to(torch.int64).clamp(0, hist.numel() - 1), weight)
    _UNION["picks"].add_((picked.to(torch.int64) * weight).sum().view(1))
    announce("union_first", layer=layer.layer_name, decodes=decodes, query=query, top_k=width)
    return True


def union_stats(hist, picks):
    samples = int(hist.sum())
    if samples == 0:
        return {"samples": 0}
    values = torch.arange(hist.numel(), dtype=torch.float64)
    weights = hist.to(torch.float64)
    cumulative = torch.cumsum(hist, 0)

    def quantile(share):
        return int(torch.searchsorted(cumulative, torch.tensor([max(1, -(-samples * share // 100))],
                                                               dtype=cumulative.dtype)).item())

    return {"samples": samples, "mean": float((values * weights).sum()) / samples, "p50": quantile(50),
            "p95": quantile(95), "max": int(torch.nonzero(hist).max()), "picks_mean": int(picks) / samples}


def union_line(stats):
    need = _UNION["query"] * _UNION["width"]
    head = (f"QSA_HISPARSE tally union window={_UNION['lines']} starts={_UNION['starts']} query={_UNION['query']} "
            f"top_k={_UNION['width']} need={need} samples={stats['samples']}")
    if not stats["samples"]:
        return head
    return (head + f" unique_mean={stats['mean']:.1f} unique_p50={stats['p50']} unique_p95={stats['p95']} "
            f"unique_max={stats['max']} picks_mean={stats['picks_mean']:.1f} "
            f"unique_share={stats['mean'] / max(need, 1):.3f}")


def report_union():
    _UNION["starts"] += 1
    if _UNION["hist"] is None or _UNION["starts"] % UNION_EVERY or capturing():
        return None
    hist = _UNION["hist"].to("cpu", copy=True)
    picks = int(_UNION["picks"].to("cpu", copy=True).item())
    _UNION["hist"].zero_()
    _UNION["picks"].zero_()
    _UNION["lines"] += 1
    line = union_line(union_stats(hist, picks))
    print(line, file=sys.stderr, flush=True)
    return line



def union_tally():
    return knob_on(UNION_TALLY_ENV)


def install_union_tally(vllm_config, width, rows, device):
    if not union_tally() or _UNION_TALLY["hist"] is not None:
        return False
    speculative = getattr(vllm_config, "speculative_config", None)
    query = 1 + int(getattr(speculative, "num_speculative_tokens", 0) or 0)
    need = query * int(width)
    _UNION_TALLY.update(need=need, query=query, top_k=int(width), starts=0, lines=0,
                        hist=torch.zeros(need + 1, dtype=torch.int64, device=device),
                        hits=torch.zeros(HIT_BUCKETS + 1, dtype=torch.int64, device=device),
                        sums=torch.zeros(len(UNION_SUMS), dtype=torch.int64, device=device),
                        misses=torch.zeros(max(int(rows), 1), dtype=torch.int64, device=device))
    announce("union_tally", knob=UNION_TALLY_ENV, query=query, top_k=int(width), need=need, every=UNION_TALLY_EVERY,
             why="each decode step and target layer on the fused capped path adds every live lane's count of "
                 "distinct rows its verification rows pick to a GPU histogram, and for a lane whose union fits the "
                 "hot region the share of those rows already resident before its swap-in to a histogram of tenths; "
                 "read every few hundred steps at the connector's start_step")
    return True


def union_tally_arm(layer, decodes):
    misses = _UNION_TALLY["misses"]
    if misses is None or is_draft(layer) or decodes > misses.numel():
        return False
    held = misses[:decodes]
    held.zero_()
    _UNION_TALLY["armed"] = held
    return True


def union_tally_misses(counts, parts, decodes):
    held = _UNION_TALLY["armed"]
    if held is None:
        return
    held.add_(counts.reshape(-1)[:parts * decodes].view(parts, decodes).sum(0, dtype=torch.int64))


def lane_unique(logical, back, decodes, query, top_k):
    picks = logical[:decodes * query, :top_k].reshape(decodes, query * top_k)
    present = (picks >= 0).any(1)
    return torch.where(present, back[:decodes].amax(1).to(torch.int64) + 1, torch.zeros_like(present, dtype=torch.int64))


def union_tally_fold(logical, back, fits, states, decodes, query, top_k, bound):
    _UNION_TALLY["armed"] = None
    hist, hits, sums = _UNION_TALLY["hist"], _UNION_TALLY["hits"], _UNION_TALLY["sums"]
    misses = _UNION_TALLY["misses"][:decodes]
    unique = lane_unique(logical, back, decodes, query, top_k)
    live = (unique > 0) & (states[:decodes] >= 0)
    fitted = live & fits[:decodes]
    weight, fit_weight = live.to(torch.int64), fitted.to(torch.int64)
    hist.index_add_(0, unique.clamp(0, hist.numel() - 1), weight)
    resident = (unique - misses).clamp_min(0)
    bucket = (resident * HIT_BUCKETS).div(unique.clamp_min(1), rounding_mode="floor").clamp(0, HIT_BUCKETS)
    hits.index_add_(0, bucket, fit_weight)
    sums.add_(torch.stack((weight.sum(), (unique * weight).sum(), fit_weight.sum(), (unique * fit_weight).sum(),
                           (misses.clamp_max(unique) * fit_weight).sum(), weight.new_ones(()))))
    _UNION_TALLY["bound"] = int(bound)
    return unique, misses, live


def union_tally_line(hist, hits, sums, steps):
    held = dict(zip(UNION_SUMS, (int(one) for one in sums)))
    stats = union_stats(hist, 0)
    head = (f"QSA_HISPARSE tally union_sizes window={_UNION_TALLY['lines']} steps={steps} query={_UNION_TALLY['query']} "
            f"top_k={_UNION_TALLY['top_k']} need={_UNION_TALLY['need']} bound={_UNION_TALLY['bound']} "
            f"lane_layers={held['lanes']} layer_calls={held['calls']}")
    if not stats["samples"]:
        return head
    lanes = max(held["lanes"], 1)
    fit = max(held["fit_lanes"], 1)
    tenths = [int(one) for one in hits]
    return (head + f" lane_layers_per_step={held['lanes'] / max(steps, 1):.1f} unique_mean={stats['mean']:.1f} "
            f"unique_p50={stats['p50']} unique_p95={stats['p95']} unique_max={stats['max']} "
            f"unique_share={stats['mean'] / max(_UNION_TALLY['need'], 1):.3f} "
            f"fit_share={held['fit_lanes'] / lanes:.4f} fit_unique_mean={held['fit_unique'] / fit:.1f} "
            f"fit_misses_mean={held['fit_misses'] / fit:.1f} "
            f"hot_hit_rate={1 - held['fit_misses'] / max(held['fit_unique'], 1):.4f} "
            f"lane_hit_tenths={'/'.join(str(one) for one in tenths)}")


def report_union_tally():
    _UNION_TALLY["starts"] += 1
    if _UNION_TALLY["hist"] is None or _UNION_TALLY["starts"] % UNION_TALLY_EVERY or capturing():
        return None
    parts = [_UNION_TALLY[name].to("cpu", copy=True) for name in ("hist", "hits", "sums")]
    for name in ("hist", "hits", "sums"):
        _UNION_TALLY[name].zero_()
    _UNION_TALLY["lines"] += 1
    line = union_tally_line(*parts, UNION_TALLY_EVERY)
    print(line, file=sys.stderr, flush=True)
    return line


REC2_ENV = "TAAF_VLLM_REC2"
REC2_EVERY_ENV = "TAAF_VLLM_REC2_EVERY"
REC2_EVERY = 250
REC2_AGE_EDGES = (1, 2, 4, 8, 16, 64)
REC2_AGE_NAMES = ("0", "1", "2-3", "4-7", "8-15", "16-63", "64+")
REC2_LAYERS = 64
REC2_HASH_BITS = 13
REC2_DIST = 8
REC2_RES = ("lanes", "blocks", "picks", "resident", "drops", "drop_lanes")
REC2_INV = ("recycle", "written", "reset", "other", "refetched")
REC2_INV_MARK = "_taaf_rec2_invalidate"
_REC2 = {"age": None, "armed": None, "starts": 0, "lines": 0, "ordinal": {}, "region": 0, "every": REC2_EVERY}


def rec2_on():
    return knob_on(REC2_ENV)


def install_rec2(vllm_config, device):
    if not rec2_on() or _REC2["age"] is not None:
        return False
    lanes = int(vllm_config.scheduler_config.max_num_seqs) + 1
    width = (1 << REC2_HASH_BITS) + 1
    _REC2.update(
        every=max(1, env_int(REC2_EVERY_ENV, REC2_EVERY)),
        age=torch.zeros(lanes, dtype=torch.int64, device=device),
        prev=torch.full((lanes,), -1, dtype=torch.int64, device=device),
        edges=torch.tensor(REC2_AGE_EDGES, dtype=torch.int64, device=device),
        ages=torch.zeros(len(REC2_AGE_NAMES), 3, dtype=torch.int64, device=device),
        layers=torch.zeros(REC2_LAYERS, 3, dtype=torch.int64, device=device),
        res=torch.zeros(len(REC2_RES), dtype=torch.int64, device=device),
        keys=torch.full((lanes, width), -1, dtype=torch.int32, device=device),
        stamps=torch.zeros(lanes, width, dtype=torch.int32, device=device),
        clock=torch.zeros(1, dtype=torch.int32, device=device),
        dist=torch.zeros(REC2_DIST + 1, dtype=torch.int64, device=device),
        inv=torch.zeros(len(REC2_INV), dtype=torch.int64, device=device),
        ckeys=torch.full((lanes, width), -1, dtype=torch.int32, device=device),
        cstamps=torch.zeros(lanes, width, dtype=torch.int32, device=device),
        inv_group=None)
    announce("rec2", knob=REC2_ENV, lanes=lanes - 1, age_buckets="/".join(REC2_AGE_NAMES), hash_bits=REC2_HASH_BITS,
             dist=REC2_DIST, every=_REC2["every"],
             why="each target layer on the fused capped path adds its lanes' union and miss counts by layer and by "
                 "steps since the lane's hot row was last cleared; the first target layer also counts the lanes' "
                 "GPU-resident pages, the share of picks they serve, resident pages lost between steps, and how "
                 "many of its misses fetch a row the same lane fetched one to a few steps before; read every few "
                 "hundred steps at the connector's start_step")
    return True


def rec2_ordinal(layer):
    held = _REC2["ordinal"]
    if id(layer) not in held:
        targets = [one for one in _ORDER if not is_draft(one)]
        held[id(layer)] = targets.index(layer) if layer in targets else REC2_LAYERS - 1
    return min(held[id(layer)], REC2_LAYERS - 1)


def rec2_arm(layer, base_states, decodes, region):
    if _REC2["age"] is None or is_draft(layer):
        return None
    index = lane_index(base_states[:decodes], _REC2["age"].numel() - 1)
    _REC2["region"] = int(region)
    if rec2_ordinal(layer) == 0:
        _REC2["clock"].add_(1)
        _REC2["armed"] = index
        cache = getattr(layer, "hisparse_cache", None)
        group = getattr(getattr(cache, "runtime", None), "index_group", None)
        if group is not None:
            _REC2["inv_group"] = id(group)
    return index


def rec2_age_fold(ages, age, edges, index, unique, misses, live):
    held = age.index_select(0, index)
    bucket = torch.bucketize(held, edges, right=True)
    weight = live.to(torch.int64)
    ages.index_add_(0, bucket, torch.stack((weight, unique * weight, misses.clamp_max(unique) * weight), 1))
    return held


def rec2_layer_fold(layers, ordinal, unique, misses, live):
    weight = live.to(torch.int64)
    layers[ordinal].add_(torch.stack((weight.sum(), (unique * weight).sum(),
                                      (misses.clamp_max(unique) * weight).sum())))


def rec2_resident_fold(res, prev, index, held_age, live, table, block_size, seq_lens, picks):
    lanes = index.numel()
    rows = table.index_select(0, index.clamp_max(table.shape[0] - 1))
    blocks = rows.shape[1]
    reach = (seq_lens[:lanes].to(torch.int64) + block_size - 1) // block_size
    inside = torch.arange(blocks, device=rows.device).view(1, -1) < reach.view(-1, 1)
    count = ((rows > 0) & inside).sum(1).to(torch.int64)
    valid = picks >= 0
    block = (picks.clamp_min(0) // block_size).clamp_max(blocks - 1)
    served = (rows.gather(1, block) > 0) & valid
    weight = live.to(torch.int64)
    before = prev.index_select(0, index)
    drop = (before - count).clamp_min(0) * (before >= 0).to(torch.int64) * (held_age > 0).to(torch.int64) * weight
    res.add_(torch.stack((weight.sum(), (count * weight).sum(), (valid.sum(1) * weight).sum(),
                          (served.sum(1) * weight).sum(), drop.sum(), (drop > 0).sum())))
    prev.index_copy_(0, index, torch.where(live, count, before))


def rec2_slots(keys):
    mixed = (keys.to(torch.int64) * 2654435761) & 0xFFFFFFFF
    return mixed >> (32 - REC2_HASH_BITS)


def rec2_misses_fold(store, stamps, clock, dist, index, host_rows, counts):
    lanes, width = host_rows.shape
    valid = torch.arange(width, device=host_rows.device).view(1, -1) < counts[:lanes].view(-1, 1)
    slot = torch.where(valid, rec2_slots(host_rows), torch.full_like(host_rows, store.shape[1] - 1, dtype=torch.int64))
    row = index[:lanes].view(-1, 1).expand(-1, width)
    seen = store[row, slot]
    age = clock.to(torch.int64) - stamps[row, slot].to(torch.int64)
    again = valid & (seen == host_rows) & (age >= 1) & (age <= REC2_DIST)
    bucket = torch.where(again, age, torch.zeros_like(age))
    dist.index_add_(0, bucket.reshape(-1), valid.reshape(-1).to(torch.int64))
    store.index_put_((row, slot), host_rows.to(store.dtype))
    stamps.index_put_((row, slot), clock.expand(lanes, width).to(stamps.dtype))


def rec2_misses(host_rows, counts):
    index = _REC2["armed"]
    if index is None:
        return
    rec2_misses_fold(_REC2["keys"], _REC2["stamps"], _REC2["clock"], _REC2["dist"], index, host_rows, counts)
    rec2_refetch_fold(_REC2["ckeys"], _REC2["cstamps"], _REC2["clock"], _REC2["inv"], index, host_rows, counts)


def rec2_refetch_fold(store, stamps, clock, inv, index, host_rows, counts):
    lanes, width = host_rows.shape
    valid = torch.arange(width, device=host_rows.device).view(1, -1) < counts[:lanes].view(-1, 1)
    slot = torch.where(valid, rec2_slots(host_rows), torch.full_like(host_rows, store.shape[1] - 1, dtype=torch.int64))
    row = index[:lanes].view(-1, 1).expand(-1, width)
    seen = store[row, slot]
    age = clock.to(torch.int64) - stamps[row, slot].to(torch.int64)
    again = valid & (seen == host_rows) & (age >= 0) & (age <= REC2_DIST)
    inv[len(REC2_INV) - 1:].add_(again.sum().view(1))
    store.index_put_((row, slot), torch.where(again, torch.full_like(seen, -1), seen))


def rec2_cleared_fold(store, stamps, clock, lanes, before, cleared):
    width = before.shape[1]
    values = torch.where(cleared, before, torch.full_like(before, -1))
    slot = torch.where(cleared, rec2_slots(values), torch.full_like(values, store.shape[1] - 1, dtype=torch.int64))
    row = lanes.view(-1, 1).expand(-1, width)
    store.index_put_((row, slot), torch.where(cleared, values.to(store.dtype), store[row, slot]))
    stamps.index_put_((row, slot), torch.where(cleared, clock.expand_as(cleared).to(stamps.dtype),
                                               stamps[row, slot]))


def rec2_cleared(group, rows, before, path):
    inv = _REC2.get("inv")
    if inv is None or capturing():
        return
    dgi = group.device_global_indices
    after = dgi if rows is None else dgi.index_select(0, rows)
    cleared = (before >= 0) & (after < 0)
    inv[path:path + 1].add_(cleared.sum().view(1))
    if _REC2["inv_group"] == id(group):
        lanes = torch.arange(dgi.shape[0], device=dgi.device) if rows is None else rows
        count = min(lanes.shape[0], _REC2["ckeys"].shape[0])
        rec2_cleared_fold(_REC2["ckeys"], _REC2["cstamps"], _REC2["clock"],
                          lanes[:count].clamp(max=_REC2["ckeys"].shape[0] - 1), before[:count], cleared[:count])


def rec2_before(group, rows=None):
    if _REC2.get("inv") is None or capturing():
        return None
    dgi = group.device_global_indices
    return dgi.clone() if rows is None else dgi.index_select(0, rows)


def install_rec2_invalidate(runtime_class):
    if _REC2.get("inv") is None or getattr(runtime_class, REC2_INV_MARK, False):
        return False
    stock_sorted = runtime_class.invalidate_sorted_slots
    stock_written = runtime_class.invalidate_written_slots
    stock_reset = runtime_class.reset_hot_state

    def invalidate_sorted_slots(self, sorted_slots, state_indices):
        rows = state_indices.to(torch.long)
        before = rec2_before(self.index_group, rows)
        stock_sorted(self, sorted_slots, state_indices)
        if before is not None:
            rec2_cleared(self.index_group, rows, before, 0)

    def invalidate_written_slots(self, written_slots, req_id_per_token):
        before = rec2_before(self.index_group)
        stock_written(self, written_slots, req_id_per_token)
        if before is not None:
            rec2_cleared(self.index_group, None, before, 1)

    def reset_hot_state(self):
        before = rec2_before(self.index_group)
        stock_reset(self)
        if before is not None:
            rec2_cleared(self.index_group, None, before, 3)

    runtime_class.invalidate_sorted_slots = invalidate_sorted_slots
    runtime_class.invalidate_written_slots = invalidate_written_slots
    runtime_class.reset_hot_state = reset_hot_state
    setattr(runtime_class, REC2_INV_MARK, True)
    announce("rec2_invalidate", knob=REC2_ENV, paths="/".join(REC2_INV[:-1]),
             why="every hot-index entry an invalidation clears outside the resolve is counted by path (recycled host "
                 "blocks, mirror writes, lane resets, whole-state resets), and a later miss of the first target "
                 "layer that fetches a row an invalidation cleared up to a few steps before counts as refetched")
    return True


def rec2_fold(layer, index, unique, misses, live, handle, meta, logical, decodes, query, top_k):
    _REC2["armed"] = None
    held_age = rec2_age_fold(_REC2["ages"], _REC2["age"], _REC2["edges"], index, unique, misses, live)
    ordinal = rec2_ordinal(layer)
    rec2_layer_fold(_REC2["layers"], ordinal, unique, misses, live)
    view = getattr(handle, "view", None)
    table = getattr(handle, "block_table", None)
    if ordinal == 0 and view is not None and table is not None:
        picks = logical[:decodes * query, :top_k].reshape(decodes, query * top_k).to(torch.int64)
        rec2_resident_fold(_REC2["res"], _REC2["prev"], index, held_age, live, table, int(view.block_size),
                           meta.seq_lens, picks)
    if layer is last_target():
        _REC2["age"].index_add_(0, index, torch.ones_like(index, dtype=torch.int64))


def rec2_reset(index):
    if _REC2["age"] is None:
        return
    target = index.to(_REC2["age"].device)
    _REC2["age"].index_fill_(0, target, 0)
    _REC2["prev"].index_fill_(0, target, -1)


def rec2_cells(rows):
    return [f"{name}:{lanes}:{unique / lanes:.1f}:{misses / lanes:.1f}:{1 - misses / max(unique, 1):.4f}"
            for name, (lanes, unique, misses) in rows if lanes]


def rec2_age_line(ages, steps):
    return (f"QSA_HISPARSE tally rec2_age window={_REC2['lines']} steps={steps} region={_REC2['region']} "
            f"cells=age:lane_layers:unique_mean:miss_mean:hit_rate "
            f"{' '.join(rec2_cells(zip(REC2_AGE_NAMES, ages)))}")


def rec2_layer_line(layers, steps):
    return (f"QSA_HISPARSE tally rec2_layers window={_REC2['lines']} steps={steps} region={_REC2['region']} "
            f"cells=layer:lane_layers:unique_mean:miss_mean:hit_rate "
            f"{' '.join(rec2_cells(enumerate(layers)))}")


def rec2_resident_line(res, steps):
    held = dict(zip(REC2_RES, res))
    lanes = max(held["lanes"], 1)
    return (f"QSA_HISPARSE tally rec2_resident window={_REC2['lines']} steps={steps} region={_REC2['region']} "
            f"lane_steps={held['lanes']} resident_blocks_mean={held['blocks'] / lanes:.2f} "
            f"pick_resident_share={held['resident'] / max(held['picks'], 1):.4f} "
            f"picks_mean={held['picks'] / lanes:.1f} dropped_blocks={held['drops']} "
            f"dropped_blocks_per_step={held['drops'] / max(steps, 1):.2f} "
            f"drop_lane_share={held['drop_lanes'] / lanes:.4f}")


def rec2_invalidate_line(inv, steps):
    held = dict(zip(REC2_INV, inv))
    return (f"QSA_HISPARSE tally rec2_invalidate window={_REC2['lines']} steps={steps} "
            + " ".join(f"{name}={held[name]}" for name in REC2_INV))


def rec2_dist_line(dist, steps):
    total = sum(dist)
    again = total - dist[0]
    cells = " ".join(f"d{one}={dist[one]}" for one in range(1, len(dist)))
    return (f"QSA_HISPARSE tally rec2_refetch window={_REC2['lines']} steps={steps} layer=0 misses={total} "
            f"misses_per_step={total / max(steps, 1):.1f} refetch_share={again / max(total, 1):.4f} {cells}")


def report_rec2():
    _REC2["starts"] += 1
    if _REC2["age"] is None or _REC2["starts"] % _REC2["every"] or capturing():
        return None
    names = ("ages", "layers", "res", "dist", "inv")
    parts = {name: _REC2[name].to("cpu", copy=True).tolist() for name in names}
    for name in names:
        _REC2[name].zero_()
    _REC2["lines"] += 1
    steps = _REC2["every"]
    lines = [rec2_age_line(parts["ages"], steps), rec2_layer_line(parts["layers"], steps),
             rec2_resident_line(parts["res"], steps), rec2_dist_line(parts["dist"], steps),
             rec2_invalidate_line(parts["inv"], steps)]
    for line in lines:
        print(line, file=sys.stderr, flush=True)
    return lines


def fused():
    return knob_on(FUSED_ENV)


def union_bitmap():
    return knob_on(UNION_BITMAP_ENV)


def spill_flag():
    return knob_on(SPILL_FLAG_ENV)


def spill_trust():
    return knob_on(SPILL_TRUST_ENV)


def spill_passes(query_len, top_k, bound, span):
    if query_len * top_k <= bound:
        return 0, False
    if spill_trust():
        return 0, True
    return query_len // span, False


def gather_tally_on():
    return knob_on(GATHER_TALLY_ENV)


def ensure_gather_tally(device, row_bytes):
    held = _GATHER["tally"]
    if held is None or held.device != device:
        held = torch.zeros(4, dtype=torch.int64, device=device)
        _GATHER["tally"] = held
        _GATHER["row_bytes"] = int(row_bytes)
        announce("gather_tally", knob=GATHER_TALLY_ENV, row_bytes=_GATHER["row_bytes"],
                 why="every row the decode swap-ins and the sparse prefill stage copy from host memory is counted "
                     "on the GPU and read back once every few hundred steps, never inside a layer")
    return held


def gather_tally(runtime, device):
    if not gather_tally_on():
        return None
    cache = getattr(runtime, "host_cache", None)
    size = int(cache.shape[-1]) * int(cache.element_size()) if isinstance(cache, torch.Tensor) else 0
    held = ensure_gather_tally(device, size)
    _GATHER["calls"] += 1
    return held


def prefill_host_tally(device, row_bytes):
    if not gather_tally_on():
        return None
    return ensure_gather_tally(device, row_bytes)[2:]


def gather_line(steps):
    tally = _GATHER["tally"].to("cpu", copy=True).tolist()
    rows, lanes, prefill, _ = (int(one) for one in tally)
    size = _GATHER["row_bytes"]
    per = max(steps, 1)
    return (f"QSA_HISPARSE tally gather window={_GATHER['lines']} steps={steps} decode_rows={rows} "
            f"decode_lanes={lanes} prefill_rows={prefill} row_bytes={size} decode_rows_per_step={rows / per:.1f} "
            f"prefill_rows_per_step={prefill / per:.1f} host_mb_per_step={(rows + prefill) * size / per / 1e6:.2f} "
            f"resolves={_GATHER['calls']}")


def report_gather():
    _GATHER["starts"] += 1
    _GATHER["steps"] += 1
    if _GATHER["tally"] is None or _GATHER["starts"] % GATHER_EVERY or capturing():
        return None
    line = gather_line(_GATHER["steps"])
    _GATHER["tally"].zero_()
    _GATHER["steps"] = 0
    _GATHER["calls"] = 0
    _GATHER["lines"] += 1
    print(line, file=sys.stderr, flush=True)
    return line


def fused_ops():
    module = _FUSED["module"]
    if module is None:
        import importlib

        module = importlib.import_module(FUSED_MODULE)
        _FUSED["module"] = module
    return module


def install_fused(rows, device):
    held = _FUSED["owners"]
    if held is None or held.numel() < int(rows) or held.device != device:
        _FUSED["owners"] = torch.zeros(int(rows), dtype=torch.int32, device=device)
    fused_ops()
    announce("host_kv_fused", knob=FUSED_ENV, owners=int(rows),
             why="the capped union, its spill passes, the fresh-miss count and the sparse prefill stage run as a "
                 "few Triton kernels a QSA layer instead of chains of small torch kernels, with the same outputs")
    return _FUSED["owners"]


def fused_ready(key):
    if capturing():
        return key in _FUSED["warm"]
    _FUSED["warm"].add(key)
    return True


def fused_owners(rows, device):
    held = _FUSED["owners"]
    if held is not None and held.device == device and held.numel() >= rows:
        return held[:rows]
    return torch.zeros(rows, dtype=torch.int32, device=device)


def fused_kept(lanes, device):
    held = _FUSED["kept"]
    if held is None or held.device != device or held.numel() < lanes:
        _FUSED["kept"] = held = torch.ones(max(int(lanes), 128), dtype=torch.bool, device=device)
    return held[:lanes]


def fused_aside(reason, **fields):
    _FUSED["aside"] += 1
    announce(f"host_kv_fused_aside_{reason}", **fields)
    return None


def shadow_rows(meta, decodes, query_len):
    lens = getattr(meta, "seq_lens", None)
    if lens is None:
        return None
    return (lens[:decodes] > 0).repeat_interleave(query_len)


def hits_shadow_band(lanes, pure):
    if not pure:
        return "mixed"
    return next((f"pure_lt{edge}" for edge in (16, 32, 64) if lanes < edge), "pure_ge64")


def relative_error(made, want):
    error = (made - want).norm() / want.norm().clamp_min(1e-12)
    return torch.where(torch.isfinite(made).all(), error, torch.full_like(error, 1.0e6))


def hits_shadow(device, rows, out, per_row, draft=False, band="mixed", real=None, probe=None, listed=None):
    names = HITS_SHADOW_TALLIES + (HITS_PROBE_TALLIES if knob_on(HITS_PROBE_ENV) else ())
    if not capturing():
        for name in names:
            if _DEVICE_TALLY.get(name) is None or _DEVICE_TALLY[name].device != device:
                add_device_tally(name, torch.zeros((), dtype=torch.int64, device=device))
        return
    if any(_DEVICE_TALLY.get(name) is None for name in names):
        return
    want = (per_row(torch.empty_like(out[:rows])) if listed is None
            else per_row(torch.empty_like(out[:rows]), listed)).float()
    made = out[:rows].float()
    if real is not None:
        keep = real.view((-1,) + (1,) * (made.dim() - 1)).to(made.dtype)
        want, made = want * keep, made * keep
    error = (made - want).norm() / want.norm().clamp_min(1e-12)
    error = torch.where(torch.isfinite(made).all(), error, torch.full_like(error, 1.0e6))
    add_device_tally("hits_shadow_calls", torch.ones((), dtype=torch.int64, device=device))
    add_device_tally("hits_shadow_bad", (error > HITS_FIRST_TOLERANCE).to(torch.int64))
    add_device_tally("hits_shadow_err_ppm", (error.clamp_max(1.0e6) * 1.0e6).round())
    if draft:
        add_device_tally("hits_shadow_draft_calls", torch.ones((), dtype=torch.int64, device=device))
        add_device_tally("hits_shadow_draft_bad", (error > HITS_FIRST_TOLERANCE).to(torch.int64))
    add_device_tally(f"hits_shadow_{band}_calls", torch.ones((), dtype=torch.int64, device=device))
    add_device_tally(f"hits_shadow_{band}_bad", (error > HITS_FIRST_TOLERANCE).to(torch.int64))
    if probe is not None:
        bad = (error > HITS_FIRST_TOLERANCE).to(torch.int64)
        copied = (probe["misses"] > 0).to(torch.int64)
        add_device_tally("hits_probe_copy_calls", copied)
        add_device_tally("hits_probe_copy_bad", bad * copied)
        add_device_tally("hits_probe_nocopy_calls", 1 - copied)
        add_device_tally("hits_probe_nocopy_bad", bad * (1 - copied))
        at = min(probe["layer"], HITS_PROBE_LAYERS - 1)
        add_device_tally(f"hits_probe_l{at}_calls", torch.ones((), dtype=torch.int64, device=device))
        add_device_tally(f"hits_probe_l{at}_bad", bad)
        add_device_tally("hits_probe_hit_entries", probe["entries"])
        add_device_tally("hits_probe_moved", probe["moved"])
        late = probe["late"]().float()
        if real is not None:
            late = late * keep
        add_device_tally("hits_probe_late_bad", (relative_error(late, want) > HITS_FIRST_TOLERANCE).to(torch.int64))
        add_device_tally("hits_probe_drift_bad", (relative_error(made, late) > HITS_FIRST_TOLERANCE).to(torch.int64))
    announce("hits_shadow", knob=HITS_SHADOW_ENV, rows=rows,
             why="a captured hits-first decode also runs the one-pass decode into a side buffer and tallies how "
                 "far the two outputs are apart, so a graph that attends wrongly shows in the stats line")


def hits_probe_rows(hot, picks):
    tokens = int(hot.shape[2])
    lead = torch.as_strided(hot, (int(hot.shape[0]), tokens, HITS_PROBE_BYTES), (hot.stride(0), hot.stride(2), 1),
                            hot.storage_offset())
    flat = picks.to(torch.int64).clamp(0, lead.shape[0] * tokens - 1)
    return lead[flat // tokens, flat % tokens]


def hits_probe_layer(layer_name):
    order = _HITS.setdefault("order", {})
    return order.setdefault(layer_name, len(order))


def shadow_on(layer_name, draft):
    if not knob_on(HITS_SHADOW_ENV):
        return False
    every = env_int(HITS_SHADOW_EVERY_ENV, 1)
    if every < 1:
        raise ValueError(f"{HITS_SHADOW_EVERY_ENV}={every} must be at least 1")
    if every > 1:
        announce("hits_shadow_every", knob=HITS_SHADOW_EVERY_ENV, every=every,
                 why="the hits shadow runs on the drafter layer and on every n-th target layer in call order, the "
                     "second one first, so a captured graph pays the one-pass decode beside the two passes on a "
                     "share of its layers")
    return draft or every == 1 or hits_probe_layer(layer_name) % every == 1


def hits_attend(impl, runtime, query, hot, logical, resolved, back, kept, fresh, table, rows, decodes, query_len,
                top_k, out, gate, per_row, draft=False, pure=False, real=None, layer_name=""):
    from vllm.taaf_qsa_kv8 import attention_passes

    device = query.device
    ops = fused_ops()
    hits = torch.empty((rows, top_k + 1), dtype=torch.int32, device=device)
    misses = torch.empty((rows, top_k + 1), dtype=torch.int32, device=device)
    ops.mapped_split(logical, resolved, resolved.shape[1], back, kept, fresh, hits, misses, decodes, 0, query_len,
                     query_len, top_k)
    shadowed = capturing() and shadow_on(layer_name, draft)
    listed = per_row.listing() if shadowed and hasattr(per_row, "listing") else None
    copied = union_event(runtime)
    probing = knob_on(HITS_PROBE_ENV) and shadowed
    if probing:
        picked = hits[:, :top_k]
        live = picked >= 0
        if real is not None:
            live = live & real.view(-1, 1)
        early = hits_probe_rows(hot, picked)
    done = attention_passes(impl, query[:rows], hot, [hits, misses], table, fused_owners(rows, device), out[:rows],
                            gate[:rows], between=lambda: current().wait_event(copied))
    probe = None
    if probing and done is not None:
        moved = (hits_probe_rows(hot, picked) != early).any(-1) & live
        counted = misses[:, top_k] if real is None else misses[:, top_k] * real.to(misses.dtype)
        owners = fused_owners(rows, device)
        probe = {"misses": counted.sum(), "layer": hits_probe_layer(layer_name), "entries": live.sum(),
                 "moved": moved.sum(),
                 "late": lambda: attention_passes(impl, query[:rows], hot, [hits, misses], table, owners,
                                                  torch.empty_like(out[:rows]), gate[:rows])}
        announce("hits_probe", knob=HITS_PROBE_ENV, rows=rows, layer=layer_name,
                 why="a captured hits-first decode also counts, by layer and by whether its copy had misses, how "
                     "often the two passes disagree with the one-pass decode, and how many hit rows changed bytes "
                     "between the hits pass and the end of the copy")
    if done is None:
        current().wait_event(copied)
        hits_aside("this layer's cache is not the fused int4 decode the passes run", failed=False)
        per_row(out[:rows])
        return
    if knob_on(HITS_SHADOW_ENV) and (shadowed or not capturing()):
        hits_shadow(device, rows, out, per_row, draft, hits_shadow_band(decodes, pure), real, probe, listed)
    if _HITS["checked"] >= HITS_FIRST_CHECKS or capturing():
        return
    want = per_row(torch.empty_like(out[:rows]))
    made = out[:rows]
    error = float((made.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12))
    finite = bool(torch.isfinite(made).all())
    _HITS["checked"] += 1
    passed = finite and error <= HITS_FIRST_TOLERANCE
    print(f"HOST_KV hits first check {_HITS['checked']}/{HITS_FIRST_CHECKS} rows={rows} relative={error:.2e} "
          f"tolerance={HITS_FIRST_TOLERANCE:g} {'pass' if passed else 'FAIL'}", flush=True)
    if not passed:
        out[:rows].copy_(want)
        hits_aside(f"a check at {rows} rows read relative {error:.2e} against the one-pass decode")


def capped_swap_fused(layer, impl, query, logical, meta, out, gate, before):
    from vllm.taaf_qsa_kv8 import attention

    union_attention = getattr(sys.modules.get("vllm.taaf_qsa_kv8"), "union_attention", None)

    group = layer.taaf_hisparse_group
    decodes = int(meta.num_decodes)
    query_len = int(meta.decode_max_query_len)
    rows = int(meta.num_decode_tokens)
    top_k = logical.shape[1] - 1
    fixed = getattr(meta, "fixed", None)
    shortest = 1 if single_fused() else 2
    if (varlen_decode_only(meta) and fixed is None) or query_len < shortest or decodes == 0 or rows != decodes * query_len or len(group.caches) != 1:
        return None
    handle = group.cache(0)
    runtime = handle.runtime
    block_table = handle.source_block_table
    if block_table is None or runtime.request_state_indices is None:
        return None
    region = int(runtime.region_stride)
    cap = capped_width(region, smem_limit(logical.device))
    if cap < top_k:
        return None
    span = max(one for one in range(1, min(query_len, cap // top_k) + 1) if query_len % one == 0)
    live = None
    base_states = runtime.request_state_indices
    resolve_states = None
    if fixed is not None:
        live = fixed.split
        logical = fixed_logical(logical, rows, live)
        base_states = resolve_states = fixed_states(runtime.request_state_indices, live)
        _FIXED["steps"] += 1
    ops = fused_ops()
    if logical.dtype != torch.int32 or logical.stride(1) != 1:
        return fused_aside("picks", dtype=str(logical.dtype), width=query_len * top_k)
    smem = smem_limit(logical.device)
    positions = int(block_table.shape[1]) * int(meta.block_size)
    bitmap = union_bitmap()
    if bitmap:
        whole = pieces = ops.bitmap_layout(positions)
        announce("union_bitmap", knob=UNION_BITMAP_ENV, positions=positions, words=whole and whole["WORDS"],
                 why="a lane's distinct picks are marked in a bitmap of its context positions and each pick's rank "
                     "is the count of marks below it, so the union needs no sort of every pick")
    else:
        whole = ops.union_layout(query_len * top_k, positions, smem)
        pieces = ops.union_layout(span * top_k, positions, smem)
    if whole is None or pieces is None:
        return fused_aside("smem", width=query_len * top_k, span_width=span * top_k, positions=positions, smem=smem,
                           why="the union sorts its picks in shared memory, and a sort this wide does not fit the "
                               "device's per-block limit, so the stock path serves the step")
    if not ops.rows_ready(query, gate, out):
        return fused_aside("rows", query=tuple(query.stride()), gate=tuple(gate.stride()), out=tuple(out.stride()))
    skip = spill_skip()
    flag = spill_flag()
    split = hot_split() and region > cap
    bound = region if split else cap
    key = ("capped", top_k, query_len, cap, bound, skip, str(logical.device), str(query.dtype), str(gate.dtype),
           str(out.dtype), tuple(query.shape[1:]), tuple(gate.shape[1:]), tuple(out.shape[1:]), bitmap, flag,
           gather_tally_on())
    if not fused_ready(key):
        if fixed is not None:
            raise RuntimeError("a captured bucketed forward reached a fused decode shape no eager forward had "
                               "launched yet; warm the bucket eagerly before capture")
        return fused_aside("cold_capture", top_k=top_k, query_len=query_len,
                           why="a captured forward reached a fused shape no eager forward had launched yet")
    if query_len == 1:
        announce("single_fused", knob=SINGLE_FUSED_ENV, layer=layer.layer_name, decodes=decodes, top_k=top_k,
                 why="a one-row-a-lane decode pass resolves, gathers and attends through the fused capped path the "
                     "verification rows take, instead of the stock per-row resolve and gather")
    device = logical.device
    request_ids = group.request_ids[:decodes]
    scratch = runtime._step_rows(decodes * query_len)
    block_size = int(meta.block_size)
    hot = planes(group.physical_kv_cache(0), heads_of(layer))
    table = identity_table(hot.shape[0], hot.device)
    held = torch.empty((decodes, min(bound, query_len * top_k)), dtype=torch.int32, device=device)
    back = torch.empty((decodes, query_len * top_k), dtype=torch.int32, device=device)
    fits = torch.empty(decodes, dtype=torch.bool, device=device)
    spill = torch.empty(decodes, dtype=torch.bool, device=device)
    ops.union(logical, decodes, query_len, query_len, top_k, held, back, whole,
              capped=(fits, spill, base_states, install_hot_cap(device), bound))
    reads_count(layer, held, base_states, decodes)
    tallied = union_tally_arm(layer, decodes)
    recorded = rec2_arm(layer, base_states, decodes, region) if tallied else None
    if fixed is None and prev_counted():
        prev_used(layer, held, fits)
    passes, trusted = spill_passes(query_len, top_k, bound, span)
    if trusted:
        announce("spill_trust", knob=SPILL_TRUST_ENV, decodes=decodes, query_len=query_len, top_k=top_k,
                 bound=bound, worst=query_len * top_k,
                 why="a lane's verification rows pick overlapping positions, so its distinct picks stay far below "
                     "query_len * top_k and the hot region holds them all; no spill pass is issued, the capped "
                     "union's overflow counter reports any lane that did exceed the region, and such a lane "
                     "attends the picks the region holds instead of none")
    elif not passes:
        announce("hot_cap_no_spill", decodes=decodes, query_len=query_len, top_k=top_k, bound=bound, cap=cap,
                 why="every lane's union of its verification rows' picks is at most query_len * top_k rows, which "
                     "fits the hot region, so no lane can spill and no spill pass is issued")
    width = span * top_k
    rounded = -(-width // SPLIT_MIN_COLUMNS) * SPLIT_MIN_COLUMNS
    columns = max(width, min(cap, query_len * top_k, rounded)) if skip else width
    ahead = {}
    prev_armed = fixed is None and prev_on() and int(meta.num_decode_tokens) == int(meta.num_actual_tokens)
    if prev_armed and prev_scope() == "missed":
        _PREV["capture"] = []
        if prev_once():
            prev_buffers(runtime, decodes, held.shape[1], region, base_states, device)
    if plan_ahead() and passes:
        def prepare():
            ahead["states"] = ops.spill_states(base_states, request_ids, fits, live) if skip else resolve_states
            ahead["unions"] = torch.empty((passes * decodes, columns), dtype=torch.int32, device=device)
            ahead["backs"] = torch.empty((passes * decodes, width), dtype=torch.int32, device=device)
            ops.union(logical, decodes, query_len, span, top_k, ahead["unions"], ahead["backs"], pieces,
                      lane_mask=spill, skip_masked=flag)

        if before is None:
            before = prepare
        else:
            early = before

            def before():
                prepare()
                early()
        announce("plan_ahead", knob=PLAN_AHEAD_ENV, decodes=decodes, passes=passes,
                 why="the spill passes' states and unions depend only on this layer's picks and the capped union's "
                     "flags, so they are issued on the compute stream while the copy stream resolves and gathers "
                     "the rows the first attention reads, not after that attention")
    fresh = None
    _HITS["taken"] = False
    kv8 = sys.modules.get("vllm.taaf_qsa_kv8")
    union_on = union_attention is not None and bool(getattr(kv8, "union_decode_enabled", lambda: True)())
    pure = int(meta.num_decode_tokens) == int(getattr(meta, "num_actual_tokens", meta.num_decode_tokens))
    if split and hits_first() and not union_on and (not is_draft(layer) or knob_on(HITS_DRAFT_ENV)) and (
            fixed is not None or knob_on(HITS_PURE_ENV)):
        fresh = hits_fresh(runtime, decodes, held.shape[1], device)
    if split:
        resolved = split_resolve(runtime, handle, held, None, request_ids, block_table, block_size, scratch, region,
                                 cap, before, fused_split=True, states=resolve_states, fresh=fresh)
        announce("hot_split", knob=HOT_SPLIT_ENV, decodes=decodes, region=region, cap=cap,
                 resolves=len(split_bounds(region, cap)),
                 why="a lane whose verification union fits the hot region but not one resolve's shared memory is "
                     "swapped in by consecutive resolves of disjoint column ranges, each leaving the previous "
                     "one's rows most recent, so one attention reads them all and only a union over the region "
                     "spills")
    elif before is None:
        resolved = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch,
                                 fused_split=True, states=resolve_states)
    else:
        resolved = union_resolve(runtime, handle, held, request_ids, block_table, block_size, scratch, wait=False,
                                 fused_split=True, states=resolve_states)
        before()
        current().wait_event(union_event(runtime))
    captured, _PREV["capture"] = _PREV["capture"], None
    if tallied:
        folded = union_tally_fold(logical, back, fits, base_states, decodes, query_len, top_k, bound)
        if recorded is not None:
            rec2_fold(layer, recorded, *folded, handle, meta, logical, decodes, query_len, top_k)
    _REC2["armed"] = None
    kept = fits if passes else fused_kept(decodes, device)

    def mapped_list():
        indexed = torch.empty((rows, top_k + 1), dtype=torch.int32, device=device)
        ops.mapped(logical, resolved, resolved.shape[1], back, kept, indexed, decodes, 0, query_len, query_len, top_k)
        return indexed

    def per_row(target, indexed=None):
        indexed = mapped_list() if indexed is None else indexed
        return attention(impl, query[:rows], hot, indexed, table, fused_owners(rows, device), False, target,
                         gate[:rows])

    per_row.listing = mapped_list

    taken, _HITS["taken"] = _HITS["taken"], False
    if taken:
        hits_attend(impl, runtime, query, hot, logical, resolved, back, kept, fresh, table, rows, decodes, query_len,
                    top_k, out, gate, per_row, is_draft(layer), pure and fixed is None,
                    shadow_rows(meta, decodes, query_len) if fixed is None and knob_on(HITS_SHADOW_ENV) else None,
                    layer.layer_name)
    elif union_attention is None or union_attention(impl, query[:rows], hot, logical, resolved, back, kept,
                                                    query_len, out[:rows], gate[:rows], per_row) is None:
        per_row(out[:rows])
    if ahead:
        states, unions, backs = ahead["states"], ahead["unions"], ahead["backs"]
    elif passes:
        states = ops.spill_states(base_states, request_ids, fits, live) if skip else resolve_states
        unions = torch.empty((passes * decodes, columns), dtype=torch.int32, device=device)
        backs = torch.empty((passes * decodes, width), dtype=torch.int32, device=device)
        ops.union(logical, decodes, query_len, span, top_k, unions, backs, pieces, lane_mask=spill,
                  skip_masked=flag)
    count = decodes * span
    owners = fused_owners(count, device)
    live = torch.empty(1, dtype=torch.int32, device=device) if flag else None
    for at in range(passes):
        part = unions[at * decodes:(at + 1) * decodes]
        if skip:
            resolved = union_resolve(runtime, handle, part, request_ids, block_table, block_size, scratch,
                                     states=states, idle=fits, fused_split=True, inline=flag)
        else:
            resolved = union_resolve(runtime, handle, part, request_ids, block_table, block_size, scratch,
                                     fused_split=True, inline=flag, states=resolve_states)
        indexed = torch.empty((count, top_k + 1), dtype=torch.int32, device=device)
        picked_query = query.new_empty((count,) + tuple(query.shape[1:]))
        picked_gate = gate.new_empty((count,) + tuple(gate.shape[1:]))
        if flag:
            ops.spill_prep(logical, resolved, width, backs, spill, indexed, query, picked_query, gate, picked_gate,
                           live, decodes, at, span, query_len, top_k)
        else:
            ops.rows(query, picked_query, span, query_len, at * span, True)
            ops.rows(gate, picked_gate, span, query_len, at * span, True)
            ops.mapped(logical, resolved, width, backs, spill, indexed, decodes, at, span, query_len, top_k)
        landing = out.new_empty((count,) + tuple(out.shape[1:]))
        if flag:
            attention(impl, picked_query, hot, indexed, table, owners, False, landing, picked_gate, work=live)
        else:
            attention(impl, picked_query, hot, indexed, table, owners, False, landing, picked_gate)
        ops.rows(landing, out, span, query_len, at * span, False, keep=spill)
    if prev_armed:
        prev_issue(layer, request_ids, meta, held, fits, base_states, cap, captured)
    if fixed is None and prefetch_blocks():
        prefetch_issue(layer, runtime, handle, request_ids, block_table, meta, held, fits, base_states, region, cap)
    _HOT_CAP["calls"] += 1
    _HOT_CAP["spills"] += passes
    _HOT_CAP["cap"] = bound
    _FUSED["calls"] += 1
    if skip:
        announce("spill_skip", knob=SPILL_SKIP_ENV, decodes=decodes, cap=cap,
                 why="a spill pass resolves only the lanes whose union overflowed the hot region, the others are "
                     "skipped inside the kernel by their state index, and an overflowing lane's host reads are cut "
                     "across many blocks, so a pass with no overflowing lane launches and returns without a sync")
    announce("hot_cap", decodes=decodes, query_len=query_len, top_k=top_k, region=region, cap=cap, span=span,
             spill_passes=passes,
             why="a lane whose verification rows' distinct picks fit the hot region is swapped in once as their "
                 "union; a lane whose union does not fit attends its rows a span at a time, each span swapped in "
                 "right before its own attention")
    if flag:
        announce("spill_flag", knob=SPILL_FLAG_ENV, decodes=decodes, passes=passes,
                 why="a spill pass reads its lanes' overflow on the GPU: with no overflowing lane the pass union, the "
                     "row gathers, the index map and the attention return at once, and its swap-in runs on the "
                     "compute stream with no copy-stream fork and join")
    announce("host_kv_fused_capped", decodes=decodes, query_len=query_len, top_k=top_k, cap=cap, span=span,
             passes=passes, skip=skip,
             why="the capped union with its tally, every spill pass's union, the row maps with their counts and "
                 "the spill rows' gathers and write-backs are Triton kernels")
    return True


def stage_tile_check(device, row_bytes, size, tile):
    ops = fused_ops()
    generator = torch.Generator().manual_seed(8191)
    blocks, total = 6, 3 * tile + 5
    resident = torch.randint(0, 256, (blocks, size, row_bytes), generator=generator, dtype=torch.uint8).to(device)
    source = torch.randint(-1, blocks * size, (total,), generator=generator, dtype=torch.int32).to(device)
    needed = (torch.rand(total, generator=generator) < 0.3).to(device)
    needed[tile:2 * tile] = False
    plain = torch.full((total, row_bytes), 3, dtype=torch.uint8, device=device)
    tiled = plain.clone()
    if not ops.stage(source, needed, resident, plain) or not ops.stage(source, needed, resident, tiled, tile):
        return "the stage layout was refused"
    if not torch.equal(plain, tiled):
        return "the tiled stage copied different bytes than the per-row stage"
    return ""


def stage_tile_ready(resident, flat):
    tile = stage_tile()
    if not tile:
        return 0
    if not _STAGE_TILE["checked"]:
        if capturing():
            return 0
        _STAGE_TILE["checked"] = True
        try:
            why = stage_tile_check(flat.device, int(flat.shape[1]) * int(flat.element_size()), int(resident.shape[1]),
                                   tile)
        except Exception as failure:
            why = f"{type(failure).__name__}: {str(failure)[:300]}"
        if why:
            _STAGE_TILE["aside"] = why
            selfcheck_fail(STAGE_TILE_ENV, f"the self-check failed: {why}")
            print(f"HOST_KV stage tile stands aside: the self-check failed ({why}); the per-row stage runs",
                  file=sys.stderr, flush=True)
            return 0
        announce("stage_tile", knob=STAGE_TILE_ENV, rows=tile,
                 why="a prefill stage copies its needed resident rows with one program per tile of rows, and a tile "
                     "with no needed row returns after one load, instead of one program per staged row; the self-check "
                     "matched the per-row copy on the device")
    return tile


def stage_rows_fused(plan, kv_cache, staged, flat, resident, logical, owners, state=None, ahead=None):
    ops = fused_ops()
    copy = plan.gpu_row_ids is not None and resident is not None
    if not plan.miss_mask.is_contiguous() or plan.block_table.numel() == 0:
        return fused_aside("stage_plan", table=tuple(plan.block_table.shape))
    if copy and (ops.word_rows(flat) is None
                 or ops.resident_rows(resident, flat.shape[1] * flat.element_size()) is None):
        return fused_aside("stage_rows", resident=tuple(resident.shape), stride=tuple(resident.stride()))
    key = ("stage", logical.shape[1], str(logical.device), copy, flat.shape[1] * flat.element_size(),
           gather_tally_on())
    if not fused_ready(key):
        return fused_aside("cold_capture", stage=True)
    total = plan.row_ids.shape[1]
    needed = ops.needed(logical, owners, plan.block_table, plan.block_size, total)
    kept = None
    if state is not None:
        kept, needed = reuse_split(state, plan, needed)
    held = _DEVICE_TALLY.get("sparse_rows_needed")
    if held is None or held.device != needed.device:
        held = torch.zeros((), dtype=torch.int64, device=needed.device)
        _DEVICE_TALLY["sparse_rows_needed"] = held
    miss_mask = ops.missed(plan.miss_mask, needed, held,
                           prefill_host_tally(needed.device, flat.shape[1] * flat.element_size()))
    if ahead is not None and ahead.numel() == miss_mask.numel():
        miss_mask = miss_mask.masked_fill(ahead.view_as(miss_mask), 0)
        announce("stage_ahead_take", knob=STAGE_AHEAD_ENV, rows=int(ahead.numel()),
                 why="the rows staged ahead are dropped from this chunk's host miss list, so the block copy and the "
                     "row gather read only the rows the copy stream did not bring")
    _TALLY["sparse_rows_total"] += int(needed.numel())
    if mark_fill():
        announce("mark_fill", knob=MARK_FILL_ENV,
                 why="the rows a prefill chunk reads are marked by a fill kernel with the value inlined, not by "
                     "an index assignment whose value is a host scalar copied to the device, which waits for "
                     "the stream to drain once per layer")
    announce("sparse_stage", knob=SPARSE_ENV,
             why="prefill attention reads only each query's index picks, so a chunk stages the union of its "
                 "queries' picked rows instead of every row of the prefix")
    if stage_blocks():
        miss_mask = block_stage_rows(plan, kv_cache, staged, miss_mask, kept)
    if copy:
        ops.stage(plan.gpu_row_ids[0], needed, resident, flat, stage_tile_ready(resident, flat))
    row_width = kv_cache.shape[-1]
    torch.ops._C_cache_ops.hisparse_gather_plan(kv_cache.view(-1, row_width), staged, plan.row_ids, plan.dst_rows,
                                                miss_mask, None, None, 0)
    if state is not None:
        reuse_commit(state, plan, kept, needed)
    announce("host_kv_fused_stage", rows=total, copy=copy,
             why="the sparse prefill stage marks its needed rows, masks the miss list, counts the marks and copies "
                 "the resident rows in four kernels")
    return True


HOT_READS_ENV = "TAAF_VLLM_HOT_READS"
HOT_READS_BINS = 1024
HOT_READS_SHIFT = 8
HOT_READS_MAX_LAYERS = 3
HOT_READS_MARK = "_taaf_hot_reads_slot"
_READS = {"counts": None, "layers": (), "pending": set(), "first": set(), "launches": 0, "resets": 0,
          "skipped": 0}


def hot_reads_layers():
    said = str(os.environ.get(HOT_READS_ENV) or "").strip()
    if said in ("", "0"):
        return ()
    try:
        layers = tuple(int(part) for part in said.split(",") if part.strip())
    except ValueError as error:
        raise RuntimeError(f"{HOT_READS_ENV}={said!r} is not a comma list of QSA layer numbers") from error
    if not 1 <= len(layers) <= HOT_READS_MAX_LAYERS or len(set(layers)) != len(layers) or min(layers) < 0:
        raise RuntimeError(f"{HOT_READS_ENV}={said!r} must name 1 to {HOT_READS_MAX_LAYERS} distinct QSA layer "
                           f"numbers")
    return layers


def install_hot_reads(vllm_config, device):
    layers = hot_reads_layers()
    if not layers or _READS["counts"] is not None:
        return False
    import importlib

    lanes = int(vllm_config.scheduler_config.max_num_seqs)
    _READS["counts"] = torch.zeros((lanes + 1, HOT_READS_BINS), dtype=torch.int32, device=device)
    _READS["layers"] = layers
    states = getattr(importlib.import_module(STATES_MODULE), STATES_CLASS)
    track_reads_slots(states)
    announce("hot_reads", knob=HOT_READS_ENV, layers=list(layers), slots=lanes + 1, bins=HOT_READS_BINS,
             bin_tokens=1 << HOT_READS_SHIFT, kib=(lanes + 1) * HOT_READS_BINS * 4 // 1024,
             rule="every decode step adds each lane's distinct picks of these QSA layers to its slot's count per "
                  "token bin, whether the page is GPU-resident or read from host; a slot's row starts at zero "
                  "only when another request takes the slot")
    return True


def track_reads_slots(states_class):
    if getattr(states_class, HOT_READS_MARK, False):
        return False
    add = states_class.add_request
    remove = states_class.remove_request

    def reads_remove(self, req_id, *args, **kwargs):
        index = remove(self, req_id, *args, **kwargs)
        self.__dict__["_taaf_reads_left"] = None if index is None else (req_id, int(index))
        return index

    def reads_add(self, req_id, *args, **kwargs):
        left = self.__dict__.pop("_taaf_reads_left", None)
        result = add(self, req_id, *args, **kwargs)
        index = self.req_id_to_index.get(req_id)
        if index is not None and left != (req_id, int(index)):
            _READS["pending"].add(int(index))
        return result

    states_class.add_request = reads_add
    states_class.remove_request = reads_remove
    setattr(states_class, HOT_READS_MARK, True)
    return True


def flush_reads():
    pending = _READS["pending"]
    counts = _READS["counts"]
    if counts is None or not pending or capturing():
        return 0
    rows = sorted(index for index in pending if 0 <= index < counts.shape[0])
    pending.clear()
    if rows:
        counts.index_fill_(0, torch.tensor(rows, dtype=torch.long).to(counts.device, non_blocking=True), 0)
        _READS["resets"] += len(rows)
    return len(rows)


def reads_count(layer, held, states, decodes):
    counts = _READS["counts"]
    if counts is None or is_draft(layer):
        return 0
    number = layer_number(layer.layer_name)
    if number not in _READS["layers"]:
        return 0
    if number not in _READS["first"]:
        if capturing():
            _READS["skipped"] += 1
            return 0
        _READS["first"].add(number)
        announce(f"hot_reads_first_{number}", layer=layer.layer_name, decodes=decodes, width=int(held.shape[1]))
    fused_ops().read_bins(held, decodes, states, counts, HOT_READS_SHIFT)
    _READS["launches"] += 1
    return decodes


def hot_reads_row(slot, clear=False):
    counts = _READS["counts"]
    if counts is None or not 0 <= int(slot) < counts.shape[0]:
        return None
    row = counts[int(slot)].cpu().numpy()
    if clear:
        counts[int(slot)].zero_()
    return row
