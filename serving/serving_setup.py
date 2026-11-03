"""Prepare the pinned Qwen3.8-Flash-Next vLLM server on Kaggle.

This file changes serving only. The Duck policy, game, model, and notebook stay
frozen. Every runtime and model input is offline and pinned. Fast start trusts
the immutable Kaggle payload after checking its small identity files; vLLM's
load is the payload-integrity test.
"""

from __future__ import annotations

import base64
import concurrent.futures
import glob
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any


SOURCE_DATASET = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_HF_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_HF_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_MANIFEST_SHA256 = "a09bdad3fe3240c73332c0f99f4388a547205cb488c127f9b9059c9267dd9a5b"
MODEL_CONFIG_SHA256 = "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"
MODEL_QUANT_CONFIG_SHA256 = "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8"
MODEL_FILE_COUNT = 419
MODEL_TOTAL_BYTES = 135_253_622_894
MODEL_PINNED_KAGGLE_PATH = (
    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/"
    "pytorch/radixark-modelopt-fp4/1"
)
MODEL_KAGGLE_PATH = Path(
    str(os.environ.get("TAAF_MODEL_PATH") or "").strip() or MODEL_PINNED_KAGGLE_PATH
)
MODEL_UNPINNED = str(os.environ.get("TAAF_MODEL_UNPINNED") or "").strip().lower() in (
    "1", "true", "yes", "on"
)

VLLM_IMAGE = "vllm/vllm-openai:qwen38-flash-next"
VLLM_INDEX_DIGEST = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
VLLM_AMD64_MANIFEST_DIGEST = (
    "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
)
VLLM_CONFIG_DIGEST = "sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9"
VLLM_RUNTIME_MANIFEST_SHA256 = (
    "e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad"
)
VLLM_VERSION = "0.1.dev20073+g8e685d198"
VLLM_PLE_STOCK_SHA256 = "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
VLLM_PLE_PATCHED_SHA256 = "46840e6b654c1d47f749bf2272ccb477c5acfefbc13c9accafe01491ccc2f205"
PLE_PATCH_DIRNAME = "vllm-patches"
PLE_PATCH_APPLICATOR = "apply_radixark_nvfp4_ple_fp8_patch.py"
PLE_PATCH_APPLICATOR_SHA256 = "423a4b42636dfc48a0445d926dfd605f9864a6789b59cc80c905eea1534cd802"
PLE_PATCH_APPLICATOR_BYTES = 8566
PLE_PATCH_DIFF = "radixark_nvfp4_ple_fp8.patch"
PLE_PATCH_DIFF_SHA256 = "840ba91008279dbab27877eaf826bf8dc70c0ee968524723c6299c53f0d9778c"
PLE_PATCH_DIFF_BYTES = 4287
PLE_PATCH_IDENTITY = "PATCH_IDENTITY.json"
PLE_PATCH_IDENTITY_SHA256 = "ad135e139c99953d0f4f28c7f61f1e0a87d985eb33ac9a0efec01872436c2bda"
PLE_PATCH_IDENTITY_BYTES = 1349
PLE_PATCH_README = "README.md"
PLE_PATCH_README_SHA256 = "18e73020660bfe46b66c0ec1dfc3e702e848b51bc3d4773f58f6ae06ef126f9d"
PLE_PATCH_README_BYTES = 1581
PLE_EXACT_PREFIX = (
    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
PLE_EXCLUDED_PREFIX = (
    "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
)
CUTLASS_PTH_SHA256 = "cc710b99ddb3d76361a70461461d274212a979d00c381d821fce5bb3ad9cf246"
CUTLASS_PTH_TEXT = (
    "import sys, os, nvidia_cutlass_dsl; sys.path.insert(0, "
    "os.path.join(nvidia_cutlass_dsl.__path__[0], 'dsl_packages'))"
)
CUTLASS_DSL_VERSION = "4.6.2"
TORCH_SHM_MANAGER_CMDLINE_SHA256 = (
    "fd3a41c6faf7553a76d60949dea5632c423de414c9c0e9494171ea942543f428"
)
EXPECTED_RUNTIME_LAYERS = {
    8: (
        "f4f325a97c3e4739640b0e472a818bb299e7ce8e585bdbfe5586e9d3b5a2c475",
        1_374_796_713,
    ),
    17: (
        "a5881cf32fc46f24318a6fc7015ccae863706075d4434d186b8b1b4e5bafb36e",
        4_634_843_373,
    ),
    18: (
        "692ea0d28a87e9cea6db254c830f650f29804522045c9201f8ea63f30471b33b",
        1_511_319_261,
    ),
    21: (
        "651e9c6638c77d15bf671d9264badcba80ea551f84109de486427b87d2ddbf75",
        61_571_140,
    ),
    24: (
        "10dce885652a53a2e644a75d256b90830663330988b8599c3c0e56cd6c8a4675",
        309_686_893,
    ),
    27: (
        "49a63ac9d116f92e5a8b006ae5665b940fcbc3d22d53aa7c01a3753ef341c53e",
        14_479_724,
    ),
}
FAST_RUNTIME_WHITEOUTS = {
    8: (
        "usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88",
    ),
    17: (
        "tmp/.wh.requirements-cuda.txt",
        "tmp/.wh.common.txt",
    ),
}

SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
HOST = "127.0.0.1"
PORT = 1234
BASE_URL = f"http://{HOST}:{PORT}/v1"
MAX_MODEL_LEN_ENV = "TAAF_VLLM_MAX_MODEL_LEN"
ANALYZER_CONTEXT = int(os.environ.get(MAX_MODEL_LEN_ENV) or 32_768)
# Preserve the frozen 28-lane Duck solver. The v9 OOM came from the 32K-token
# dummy warmup, so the primary profile cuts that batch without reducing lanes.
MAX_NUM_SEQS = 28
# Keep the model author's validated single-GPU setting. Raising this to 0.98
# would trade the known-safe GPU reserve for KV cache that vLLM can size itself.
# The primary profile is the v8 fit: chunked prefill with an 8K batch cap.
MAX_NUM_BATCHED_TOKENS = 8_192
# Keep the explicit no-chunk API valid for controlled diagnostics. Pinned vLLM
# rejects disabled chunked prefill when this is below max_model_len.
NO_CHUNKED_MAX_NUM_BATCHED_TOKENS = ANALYZER_CONTEXT
READY_TIMEOUT_ENV = "TAAF_VLLM_READY_TIMEOUT_S"
SETUP_TIMEOUT_ENV = "TAAF_VLLM_SETUP_TIMEOUT_S"
_READY_SAID = str(os.environ.get(READY_TIMEOUT_ENV) or "").strip()
_SETUP_SAID = str(os.environ.get(SETUP_TIMEOUT_ENV) or "").strip()
SERVER_READY_TIMEOUT = min(7_200, max(60, int(_READY_SAID))) if _READY_SAID.isdigit() else 1_500
SETUP_TOTAL_TIMEOUT = max(SERVER_READY_TIMEOUT + 300,
                          min(7_800, int(_SETUP_SAID)) if _SETUP_SAID.isdigit() else 1_800)
FAST_START_ENV = "TAAF_KAGGLE_FAST_START"
MAX_NUM_SEQS_ENV = "TAAF_VLLM_MAX_NUM_SEQS"
MAX_NUM_BATCHED_TOKENS_ENV = "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS"
OMP_THREADS_ENV = "TAAF_VLLM_OMP_THREADS"
MTP_TOKENS_ENV = "TAAF_VLLM_MTP_TOKENS"
KV_CACHE_MEMORY_BYTES_ENV = "TAAF_VLLM_KV_CACHE_MEMORY_BYTES"
GPU_MEMORY_PERMILLE_ENV = "TAAF_VLLM_GPU_MEMORY_PERMILLE"
KV_CACHE_DTYPE_ENV = "TAAF_VLLM_KV_CACHE_DTYPE"
ENABLE_PREFIX_CACHING_ENV = "TAAF_VLLM_ENABLE_PREFIX_CACHING"
KV_OFFLOADING_GIB_ENV = "TAAF_VLLM_KV_OFFLOADING_GIB"
KV_DISK_GIB_ENV = "TAAF_VLLM_KV_DISK_GIB"
KV_DISK_DIR_ENV = "TAAF_VLLM_KV_DISK_DIR"
KV_DISK_PAGE_CACHE_ENV = "TAAF_VLLM_KV_DISK_PAGE_CACHE"
KV_DISK_DIR = "/tmp/qwen38-flash-next-kv-disk"
KV_DISK_FILE = "kv-offload.bin"
KV_DISK_SPARE_GIB = 20
MAX_CUDAGRAPH_CAPTURE_SIZE_ENV = "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"
PIECEWISE_LARGE_ENV = "TAAF_VLLM_PIECEWISE_LARGE"
PIECEWISE_LARGE_STEP_ENV = "TAAF_VLLM_PIECEWISE_LARGE_STEP"
PIECEWISE_LARGE_STEP = 32
PIECEWISE_LARGE_STEP_MIN = 4
PIECEWISE_LARGE_STEP_MAX = 4096
CAPTURE_SPARSE_ENV = "TAAF_VLLM_CAPTURE_SPARSE"
MOE_BACKEND_ENV = "TAAF_VLLM_MOE_BACKEND"
RETURN_ROUTED_EXPERTS_ENV = "TAAF_VLLM_RETURN_ROUTED_EXPERTS"
IMAGE_MAX_PIXELS_ENV = "TAAF_VLLM_IMAGE_MAX_PIXELS"
MM_CACHE_GIB_ENV = "TAAF_VLLM_MM_CACHE_GIB"
NO_VIDEO_ENV = "TAAF_VLLM_NO_VIDEO"
EXPERTS_PER_TOK_ENV = "TAAF_VLLM_EXPERTS_PER_TOK"
EXPERTS_KEPT_ENV = "TAAF_VLLM_EXPERTS_KEPT"
API_MALLOC_ENV = "TAAF_VLLM_API_MALLOC"
API_MALLOC_PRESETS = {
    "arena2": {"MALLOC_ARENA_MAX": "2", "MALLOC_MMAP_THRESHOLD_": "1048576", "MALLOC_TRIM_THRESHOLD_": "4194304"},
    "mmap": {"MALLOC_MMAP_THRESHOLD_": "1048576", "MALLOC_TRIM_THRESHOLD_": "4194304"},
    "arena": {"MALLOC_ARENA_MAX": "2"},
}
API_MALLOC_ALIASES = {"1": "arena2", "true": "arena2", "on": "arena2", "yes": "arena2"}


def api_malloc_settings(said):
    name = str(said or "").strip().lower()
    if name in ("", "0", "false", "off", "no", "none"):
        return None, {}
    name = API_MALLOC_ALIASES.get(name, name)
    if name not in API_MALLOC_PRESETS:
        raise RuntimeError(f"{API_MALLOC_ENV}={said!r} names no preset; known: {sorted(API_MALLOC_PRESETS)}")
    return name, dict(API_MALLOC_PRESETS[name])


def apply_api_malloc(env):
    name, settings = api_malloc_settings(os.environ.get(API_MALLOC_ENV))
    if name is None:
        return None
    env.update(settings)
    print(f"SERVER_ENV {API_MALLOC_ENV}={name}: " + " ".join(f"{k}={v}" for k, v in settings.items())
          + " set in the launch environment; the API server reads them at start and the engine core and "
            "workers it spawns inherit them", flush=True)
    return settings

PRUNE_WORKERS_ENV = "TAAF_VLLM_PRUNE_WORKERS"
EXPERT_CUT_ENV = "TAAF_VLLM_EXPERT_CUT_AT_LOAD"
EXPERT_CUT_WORKERS_ENV = "TAAF_VLLM_EXPERT_CUT_WORKERS"
EXPERT_CUT_WINDOW_ENV = "TAAF_VLLM_EXPERT_CUT_WINDOW_MIB"
EXPERT_CUT_WORKERS_MAX = 128
EXPERT_CUT_WINDOW_MAX_MIB = 65536
EXPERT_CUT_COALESCE_ENV = "TAAF_VLLM_EXPERT_CUT_COALESCE_MIB"
EXPERT_CUT_GAP_ENV = "TAAF_VLLM_EXPERT_CUT_GAP_KIB"
EXPERT_CUT_COALESCE_MAX_MIB = 1024
EXPERT_CUT_GAP_MAX_KIB = 65536
EXPERT_CUT_HELPER = "expert_cut.py"
EXPERT_CUT_INSTALLED = "vllm/taaf_expert_cut.py"
EXPERT_HOT_ENV = "TAAF_VLLM_EXPERT_HOT"
EXPERT_HOT_HELPER = "expert_hot.py"
EXPERT_HOT_TABLE = "expert_hot_table.py"
EXPERT_HOT_INSTALLED = "vllm/taaf_expert_hot.py"
EXPERT_CUT_MARKER = "vllm.taaf_expert_cut"
EXPERT_CUT_HOOK = (
    "\n\nfrom vllm.taaf_expert_cut import install as _taaf_expert_cut_install\n\n"
    "_taaf_expert_cut_install(globals())\n")
PLE_DF11_STARTUP_ENV = "TAAF_VLLM_PLE_DF11_STARTUP"
EARLY_READ_ENV = "TAAF_VLLM_EARLY_READ"
EARLY_READ_GIB_ENV = "TAAF_VLLM_EARLY_READ_GIB"
EARLY_READ_CAP_GIB = 96
EARLY_READ_THREADS_DEFAULT = 32
LOCAL_TIER_ENV = "TAAF_VLLM_LOCAL_TIER"
LOCAL_TIER_DIR_ENV = "TAAF_VLLM_LOCAL_TIER_DIR"
LOCAL_TIER_DIR_DEFAULT = "/kaggle/tmp/taaf-stage"
TIER_PROBE_GIB_ENV = "TAAF_VLLM_TIER_PROBE_GIB"
TIER_PROBE_GIB_DEFAULT = 2.0
TIER_STAGE_MARGIN_GIB = 20
PLE_DF11_DROP_ENV = "TAAF_VLLM_PLE_DF11_DROP_BEHIND"
PRUNE_WORKERS_MAX = 32
SELFCHECK_OVERLAP_ENV = "TAAF_VLLM_SELFCHECK_OVERLAP"
VERDICT_CACHE_ENV = "TAAF_VLLM_VERDICT_CACHE"
VERDICT_CACHE_FILE = "verdict-cache.json"
VERDICT_HARVEST_FILE = "verdict-harvest.json"
VERDICT_SPOT_TIMEOUT = 180
BF16C_VERDICT_WAIT_ENV = "TAAF_VLLM_BF16C_VERDICT_WAIT_S"
BF16C_VERDICT_WAIT_S = 900
HC_FUSED_PICKS_ENV = "TAAF_VLLM_HC_FUSED_PICKS"
HC_FUSED_PICKS_FILE = "hc-picks.json"
HC_FUSED_PICKS_FILE_ENV = "TAAF_VLLM_HC_FUSED_PICKS_FILE"
COMPILE_SEED_ENV = "TAAF_VLLM_COMPILE_SEED"
COMPILE_SEED_FILE = "compile-seed.tar"
COMPILE_DELTA_FILE = "compile-seed-delta.tar"
SEED_INPUT_ROOT = Path("/kaggle/input")
COMPILE_MANIFEST = "MANIFEST.json"
COMPILE_HARVEST_ENV = "TAAF_VLLM_COMPILE_HARVEST"
INIT_TRACE_ENV = "TAAF_VLLM_INIT_TRACE"
INIT_TRACE_PATH_ENV = "TAAF_VLLM_INIT_TRACE_PATH"
INIT_TRACE_PRINT_ENV = "TAAF_VLLM_INIT_TRACE_PRINT"
INIT_TRACE_IMPORTS_ENV = "TAAF_VLLM_INIT_TRACE_IMPORTS"
INIT_TRACE_IMPORTS_DEFAULT = "worker"
INIT_TRACE_HELPER = "init_trace.py"
INIT_TRACE_DIR = Path("/tmp/qwen38-flash-next-init-trace")
PYC_SEED_ENV = "TAAF_VLLM_PYC_SEED"
PYC_HARVEST_ENV = "TAAF_VLLM_PYC_HARVEST"
HARVEST_MIN_NEW_ENV = "TAAF_VLLM_HARVEST_MIN_NEW"
HARVEST_MIN_NEW_DEFAULT = 300
SEED_HELD: dict[str, int] = {}
SEED_LIBRARIES: set[str] = set()
SEED_FILES: dict[str, int] = {}
MARLIN_BUILD_ENV = "TAAF_VLLM_MOE_SUM_EPI_DIR"
MARLIN_BUILD_DIR = "taaf_marlin"
PYC_WRITE_ENV = "TAAF_VLLM_PYC_WRITE"
PYC_SEED_FILE = "pyc-seed.tar"
PYC_MANIFEST = "PYC_MANIFEST.json"
PYC_WORKERS = 16
PYC_RESULTS: dict[str, Any] = {}
SEED_STAGE_THREADS = 32
SEED_STAGE_PIECE = 64 << 20
SEED_STAGE_DIR = Path("/tmp/qwen38-flash-next-seed-stage")
BACKGROUND: dict[str, Any] = {}
FLAT_CHUNK_STATS: list[dict[str, Any]] = []
READY_POLL_ENV = "TAAF_VLLM_READY_POLL_S"
READY_POLL_DEFAULT_S = 5.0
READY_POLL_MIN_S = 0.2
MTP_KEY = "mtp"
COUNT_EXPERTS_ENV = "TAAF_VLLM_COUNT_EXPERTS"
SCHED_SPEC_RESERVE_ENV = "TAAF_VLLM_SCHED_SPEC_RESERVE"
SCHEDULER_RELATIVE = "vllm/v1/core/sched/scheduler.py"
MTP_READS_ENV = "TAAF_VLLM_MTP_OWN_READS"
OFFLOAD_SYNC_ENV = "TAAF_VLLM_OFFLOAD_LOAD_SYNC"
OFFLOAD_WORKER_RELATIVE = "vllm/v1/simple_kv_offload/worker.py"
OFFLOAD_LOAD_FROM = """            backend.launch_copy(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                is_store=False,
                event_idx=metadata.load_event,
                events_list=self._load_events,
            )
"""
OFFLOAD_LOAD_TO = """            _taaf_load_done = torch.Event()
            _taaf_load_done.record(torch.cuda.current_stream())
            backend.launch_copy(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                is_store=False,
                event_idx=metadata.load_event,
                events_list=self._load_events,
                wait_event=_taaf_load_done,
            )
"""
QSA_KV8_ENV = "TAAF_VLLM_QSA_KV8"
QSA_KV8_DTYPE = "int8_per_token_head"
QSA_QUANT_DTYPES = ("int8_per_token_head", "int4_per_token_head")
QSA_KV8_HELPER = "qsa_kv8.py"
QSA_KV8_INSTALLED = "vllm/taaf_qsa_kv8.py"
QSA_KV8_SETTINGS = "vllm/taaf_qsa_kv8.json"
QSA_LAYOUT_ENV = "TAAF_VLLM_QSA_LAYOUT"
QSA_PREFILL_ENV = "TAAF_VLLM_QSA_PREFILL"
QSA_PREFILL_FILE_ENV = "TAAF_VLLM_QSA_PREFILL_FILE"
QSA_DECODE_FUSED_ENV = "TAAF_VLLM_QSA_DECODE_FUSED"
QSA_DECODE_WARPS_ENV = "TAAF_VLLM_QSA_DECODE_WARPS"
QSA_TILE_ENV = "TAAF_VLLM_QSA_TILE"
QSA_TILE_SIZES = ("2", "4")
QSA_LEAN_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN"
QSA_LEAN_CHECKED_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN_CHECKED"
QSA_LEAN_MAXNREG_ENV = "TAAF_VLLM_QSA_PREFILL_LEAN_MAXNREG"
QSA_DECODE_LEAN_ENV = "TAAF_VLLM_QSA_DECODE_LEAN"
QSA_DECODE_LEAN_CHECKED_ENV = "TAAF_VLLM_QSA_DECODE_LEAN_CHECKED"
QSA_UNION_DECODE_ENV = "TAAF_VLLM_QSA_UNION_DECODE"
QSA_UNION_DECODE_CHECKED_ENV = "TAAF_VLLM_QSA_UNION_DECODE_CHECKED"
QSA_DECODE_STAGES_ENV = "TAAF_VLLM_QSA_DECODE_STAGES"
QSA_DECODE_CONFIG_ENV = "TAAF_VLLM_QSA_DECODE_CONFIG"
QSA_PREFILL_CONFIG_ENV = "TAAF_VLLM_QSA_PREFILL_CONFIG"
QSA_INT8_FAST_ENV = "TAAF_VLLM_INT8_FAST"
QSA_KV8_FAST_ENV = "TAAF_VLLM_QSA_KV8_FAST"
QSA_OWNER_RELATIVE = "vllm/models/qwen4_exp/nvidia/qsa.py"
QSA_WARMUP_RELATIVE = "vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py"
QSA_KV8_TARGETS = {
    QSA_OWNER_RELATIVE: (
        "9ac72b0e5f02341704a2f5a49bf2f4a6f141d2e241e9db07ea5918592bac90ea",
        (
            ('    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n'
             '\n'
             '    @staticmethod\n'
             '    def get_name() -> str:\n'
             '        return "QWEN4_EXP_QSA_TRITON"\n',
             '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
             '        "auto",\n'
             '        "bfloat16",\n'
             '        "int8_per_token_head",\n'
             '        "int4_per_token_head",\n'
             '    ]\n'
             '\n'
             '    @classmethod\n'
             '    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:\n'
             '        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes\n'
             '\n'
             '    @classmethod\n'
             '    def customize_spec(cls, spec):\n'
             '        from vllm.taaf_qsa_kv8 import customize_spec\n'
             '\n'
             '        return customize_spec(spec)\n'
             '\n'
             '    @staticmethod\n'
             '    def get_name() -> str:\n'
             '        return "QWEN4_EXP_QSA_TRITON"\n'),
            ('    def __init__(self, *args, **kwargs) -> None:\n'
             '        super().__init__(*args, **kwargs)\n'
             '        if not is_flash_attn_varlen_func_available():\n',
             '    def __init__(self, *args, **kwargs) -> None:\n'
             '        args = list(args)\n'
             '        quantized = args[6] if len(args) > 6 and args[6] in (\n'
             '            "int8_per_token_head", "int4_per_token_head") else None\n'
             '        if quantized:\n'
             '            args[6] = "auto"\n'
             '        super().__init__(*args, **kwargs)\n'
             '        if quantized:\n'
             '            self.kv_cache_dtype = quantized\n'
             '        if not is_flash_attn_varlen_func_available():\n'),
            ('        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
             '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")\n'
             '        self.supports_quant_query_input = False\n'
             '\n'
             '    def forward_qsa(\n',
             '        if self.kv_cache_dtype not in (\n'
             '            "auto", "bfloat16", "int8_per_token_head", "int4_per_token_head"\n'
             '        ):\n'
             '            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")\n'
             '        self.supports_quant_query_input = False\n'
             '\n'
             '    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):\n'
             '        if self.kv_cache_dtype in ("int8_per_token_head", "int4_per_token_head"):\n'
             '            from vllm.taaf_qsa_kv8 import write\n'
             '\n'
             '            return write(self, key, value, kv_cache, slot_mapping)\n'
             '        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)\n'
             '\n'
             '    def forward_qsa(\n'),
            ('        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n',
             '        if self.kv_cache_dtype in ("int8_per_token_head", "int4_per_token_head"):\n'
             '            from vllm.taaf_qsa_kv8 import attention\n'
             '\n'
             '            attention(\n'
             '                self,\n'
             '                query[:num_tokens],\n'
             '                kv_cache,\n'
             '                logical_indices,\n'
             '                attn_metadata.block_table,\n'
             '                token_to_req,\n'
             '                use_prefill_config,\n'
             '                output[:num_tokens],\n'
             '                output_gate[:num_tokens],\n'
             '            )\n'
             '            return output\n'
             '        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n'),
            ('        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n',
             '        if cache_config.cache_dtype not in (\n'
             '            "auto", "bfloat16", "int8_per_token_head", "int4_per_token_head"\n'
             '        ):\n'),
            ('        if self.kv_cache_torch_dtype != torch.bfloat16:\n',
             '        if self.kv_cache_torch_dtype != torch.bfloat16 and (\n'
             '            self.kv_cache_dtype not in ("int8_per_token_head", "int4_per_token_head")\n'
             '        ):\n'),
        ),
    ),
    QSA_WARMUP_RELATIVE: (
        "ea84db483e9e6a84287639d08bd02743be8efc474c441f42fcc34198204d3ce6",
        (
            ('    attention_profiles = warmup_qsa_sparse_paged_attention(\n'
             '        kv_cache,\n'
             '        block_table_for(owner.layer_name),\n'
             '        num_query_heads=owner.num_heads,\n'
             '        selection_width=indexer.output_width,\n'
             '    )\n',
             '    if owner.kv_cache_dtype in ("int8_per_token_head", "int4_per_token_head"):\n'
             '        from vllm.taaf_qsa_kv8 import warmup\n'
             '\n'
             '        attention_profiles = warmup(\n'
             '            owner.impl,\n'
             '            kv_cache,\n'
             '            block_table_for(owner.layer_name),\n'
             '            num_query_heads=owner.num_heads,\n'
             '            selection_width=indexer.output_width,\n'
             '        )\n'
             '    else:\n'
             '        attention_profiles = warmup_qsa_sparse_paged_attention(\n'
             '            kv_cache,\n'
             '            block_table_for(owner.layer_name),\n'
             '            num_query_heads=owner.num_heads,\n'
             '            selection_width=indexer.output_width,\n'
             '        )\n'),
        ),
    ),
}
HC_FUSED_ENV = "TAAF_VLLM_HC_FUSED"
HC_FUSED_HELPER = "hc_fused.py"
HC_FUSED_INSTALLED = "vllm/taaf_hc_fused.py"
HC_FUSED_TARGETS = {
    "vllm/models/qwen4_exp/nvidia/hyperconnection.py": (
        "2a15d6b22fbe1def4d2bcd85568269d1157c5e1c298f0c6af43c1fb9797489ba",
        (
            ('from .ops.hc import (\n'
             '    grouped_gemma_rmsnorm,\n'
             '    hc_combine,\n'
             '    hc_combine_norm,\n'
             '    hc_gate_mix,\n'
             '    hc_silu,\n'
             ')\n',
             'from .ops.hc import (\n'
             '    grouped_gemma_rmsnorm,\n'
             '    hc_combine,\n'
             '    hc_combine_norm,\n'
             '    hc_gate_mix,\n'
             '    hc_silu,\n'
             ')\n'
             'from vllm import taaf_hc_fused as _taaf_hc_fused\n'),
            ('        self.use_combine = use_combine\n',
             '        self.use_combine = use_combine\n'
             '        self._taaf_hc_site = _taaf_hc_fused.register(self, prefix)\n'),
            ('    def mix(\n'
             '        self, hidden_states: torch.Tensor\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n',
             '    def mix(\n'
             '        self, hidden_states: torch.Tensor\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n'
             '        return _taaf_hc_fused.mix(self, hidden_states)\n'
             '\n'
             '    def _taaf_stock_mix(\n'
             '        self, hidden_states: torch.Tensor\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n'),
            ('    def combine_and_mix(\n'
             '        self,\n'
             '        hidden_states: torch.Tensor,\n'
             '        prev_block_output: torch.Tensor,\n'
             '        prev_injection: torch.Tensor | None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n',
             '    def combine_and_mix(\n'
             '        self,\n'
             '        hidden_states: torch.Tensor,\n'
             '        prev_block_output: torch.Tensor,\n'
             '        prev_injection: torch.Tensor | None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n'
             '        return _taaf_hc_fused.combine_and_mix(\n'
             '            self, hidden_states, prev_block_output, prev_injection\n'
             '        )\n'
             '\n'
             '    def _taaf_stock_combine_and_mix(\n'
             '        self,\n'
             '        hidden_states: torch.Tensor,\n'
             '        prev_block_output: torch.Tensor,\n'
             '        prev_injection: torch.Tensor | None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n'),
        ),
    ),
}
HC_FUSED_ROWS_ENV = "TAAF_VLLM_HC_FUSED_ROWS"
HC_DECODE_ENV = "TAAF_VLLM_HC_DECODE"
HC_DECODE_SHAPE_ENV = "TAAF_VLLM_HC_DECODE_SHAPE"
HC_DECODE_DEFAULT_BAND = (96, 640)
HC_DECODE_ROWS_LIMIT = 1024
HC_DECODE_TILES = (16, 32, 64, 128)
HC_DECODE_SPLITS = (4, 5, 8, 10, 16, 20, 32, 40)
HC_DECODE_UP_WIDTHS = (32, 64)
HC_DECODE_DOWN_ENV = "TAAF_VLLM_HC_DECODE_DOWN"
HC_DECODE_UP4_ENV = "TAAF_VLLM_HC_DECODE_UP4"
HC_DECODE_MIN_ROWS_ENV = "TAAF_VLLM_HC_DECODE_DOWN_MIN_ROWS"
HC_DECODE_MIN_ROWS_DEFAULT = 300
HC_MIXED_F1_ENV = "TAAF_VLLM_HC_MIXED_F1"
HC_MIXED_F2_ENV = "TAAF_VLLM_HC_MIXED_F2"
HC_MIXED_SHAPE_ENV = "TAAF_VLLM_HC_MIXED_SHAPE"
HC_MIXED_DEFAULT_BAND = (2560, 8192)
HC_MIXED_TILES = (16, 32, 64, 128)
HC_MIXED_UP_WIDTHS = (32, 64)
HC_MIXED_MAX_SPLITS = 4
HC_FUSED_BASE_ROWS = 512
HC_FUSED_WIDE_BUCKET = 256
HC_FUSED_WIDE_LIMIT = 8192
GDN_CP_SPLIT_ENV = "TAAF_VLLM_GDN_CP_SPLIT"
GDN_CP_SPLIT_HELPER = "gdn_cp_split.py"
GDN_CP_SPLIT_TABLE = "gdn_cp_split_table.py"
GDN_CP_SPLIT_INSTALLED = "vllm/taaf_gdn_cp_split.py"
GDN_CP_SPLIT_FINE = 64
GDN_CP_SPLIT_STOCK = 512
GDN_COMMIT_STREAM_ENV = "TAAF_VLLM_GDN_COMMIT_STREAM"
GDN_COMMIT_STREAM_HELPER = "gdn_commit_stream.py"
GDN_COMMIT_STREAM_TABLE = "gdn_commit_stream_table.py"
GDN_COMMIT_STREAM_INSTALLED = "vllm/taaf_gdn_commit_stream.py"
GDN_COMMIT_CAP_ENV = "TAAF_VLLM_GDN_COMMIT_CAP"
GDN_COMMIT_CAP_LIMIT = 256
GDN_COMMIT_CAP_STREAM_DEFAULT = 2
DENSE_HOLE_ENV = "TAAF_VLLM_DENSE_HOLE"
DENSE_HOLE_VERDICT_ENV = "TAAF_VLLM_DENSE_HOLE_VERDICT"
DENSE_HOLE_ROWS_ENV = "TAAF_VLLM_DENSE_HOLE_ROWS"
DENSE_HOLE_WAIT_ENV = "TAAF_VLLM_DENSE_HOLE_WAIT_S"
DENSE_HOLE_WAIT_S = 900
DENSE_HOLE_TIMEOUT = 600
DENSE_HOLE_ROWS_MIN = 16
DENSE_HOLE_ROWS_MAX = 16384
DENSE_HOLE_HELPER = "dense_hole.py"
DENSE_HOLE_TABLE = "dense_hole_table.py"
DENSE_HOLE_INSTALLED = "vllm/taaf_dense_hole.py"
DENSE_HOLE_VERDICT = "dense-hole-verdict.json"


FP8_DENSE_ENV = "TAAF_VLLM_FP8_DENSE"
FP8_DENSE_QUANT_ENV = "TAAF_VLLM_FP8_DENSE_QUANT"
FP8_DENSE_QUANTS = ("ptpc", "block128")
FP8_DENSE_HELPER = "fp8_dense.py"
FP8_DENSE_INSTALLED = "vllm/taaf_fp8_dense.py"
FP8_DENSE_SETTINGS = "vllm/taaf_fp8_dense.json"
W4A8_ENV = "VLLM_MARLIN_INPUT_DTYPE"
W4A8_MODES = ("int8", "fp8")
W4A8_GPTQ = "vllm/model_executor/layers/quantization/auto_gptq.py"
W4A8_GPTQ_STOCK = "3efa1ab7ee3648a1239dcb2f7e0f9528a8229e7b8e36ba1ff6ac300b58c1d226"
W4A8_GPTQ_MARKER = "_taaf_w4a8_gptq_init"
W4A8_GPTQ_ASSERTS = (('            assert self.quant_config.quant_type.size_bits == 8, (\n', '            assert self.quant_config.quant_type.size_bits == 4, (\n'), ('        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 8, (\n', '        assert not is_a_8bit or self.quant_config.quant_type.size_bits == 4, (\n'))
W4A8_GPTQ_TAIL = '\n\nimport dataclasses as _taaf_w4a8_dataclasses\n\n_taaf_w4a8_gptq_stock_init = AutoGPTQMoEMethod.__init__\n_taaf_w4a8_gptq_stock_config = AutoGPTQMoEMethod.get_fused_moe_quant_config\n_taaf_w4a8_gptq_said = []\n\n\ndef _taaf_w4a8_gptq_init(self, *args, **kwargs):\n    _taaf_w4a8_gptq_stock_init(self, *args, **kwargs)\n    if self.wna16_moe_backend in (WNA16MoEBackend.MARLIN, WNA16MoEBackend.BATCHED_MARLIN) and getattr(self.quant_config, "group_size", None) == 128:\n        self.input_dtype = get_marlin_input_dtype()\n\n\ndef _taaf_w4a8_gptq_config(self, layer):\n    made = _taaf_w4a8_gptq_stock_config(self, layer)\n    g1 = getattr(layer, \'w13_input_global_scale\', None)\n    g2 = getattr(layer, \'w2_input_global_scale\', None)\n    if made is not None and (g1 is not None or g2 is not None):\n        made = _taaf_w4a8_dataclasses.replace(\n            made, _a1=_taaf_w4a8_dataclasses.replace(made._a1, alpha_or_gscale=g1),\n            _a2=_taaf_w4a8_dataclasses.replace(made._a2, alpha_or_gscale=g2))\n    if not _taaf_w4a8_gptq_said:\n        _taaf_w4a8_gptq_said.append(1)\n        print(f\'W4A8_GPTQ live: GPTQ Marlin MoE weights repacked for {self.input_dtype} input, input global \'\n              f\'scales {"passed" if g1 is not None or g2 is not None else "none"}\', flush=True)\n    return made\n\n\n_taaf_w4a8_gptq_stock_process = AutoGPTQMoEMethod.process_weights_after_loading\n\n\ndef _taaf_w4a8_gptq_process(self, layer):\n    premixed = False\n    if (str(self.input_dtype) == \'torch.int8\'\n            and __import__(\'os\').environ.get(\'TAAF_VLLM_MOE_SILU_EPI\', \'\').strip() == \'1\'):\n        try:\n            from vllm import taaf_moe_block as _taaf_moe_block\n        except ImportError as failure:\n            print(f\'MOE_SILU_EPI int8 w1 not interleaved at load: {failure}\', flush=True)\n        else:\n            premixed = _taaf_moe_block.silu_premix(layer)\n    made = _taaf_w4a8_gptq_stock_process(self, layer)\n    if premixed:\n        _taaf_moe_block.silu_premixed(layer)\n    return made\n\n\nAutoGPTQMoEMethod.process_weights_after_loading = _taaf_w4a8_gptq_process\n\nAutoGPTQMoEMethod.__init__ = _taaf_w4a8_gptq_init\nAutoGPTQMoEMethod.get_fused_moe_quant_config = _taaf_w4a8_gptq_config\n\n\n_taaf_w4a8_gptq_stock_method = AutoGPTQConfig.get_quant_method\n\n\ndef _taaf_w4a8_gptq_method(self, layer, prefix):\n    made = _taaf_w4a8_gptq_stock_method(self, layer, prefix)\n    if isinstance(made, AutoGPTQMoEMethod) and getattr(made.quant_config, "group_size", None) != 128:\n        made.input_dtype = None\n    return made\n\n\nAutoGPTQConfig.get_quant_method = _taaf_w4a8_gptq_method\n'
W4A8_OPS = "vllm/_custom_ops.py"
W4A8_OPS_MARKER = "_taaf_w4a8_signed_gemm"
W4A8_OPS_TAIL = "\n\nimport os as _taaf_w4a8_os\n\n_taaf_w4a8_stock_gemm = moe_wna16_marlin_gemm\n_taaf_w4a8_signed_op = []\n\n\ndef _taaf_w4a8_signed_load():\n    if not _taaf_w4a8_signed_op:\n        torch.ops.load_library(_taaf_w4a8_os.path.join(_taaf_w4a8_os.path.dirname(__file__), 'taaf_w4a8fix_op.so'))\n        _taaf_w4a8_signed_op.append(torch.ops.w4a8fix.moe_wna16_marlin_gemm)\n        print('W4A8_INT8_SIGNED live: int8-input Marlin MoE runs the op that reads its int16 group scales as signed',\n              flush=True)\n    return _taaf_w4a8_signed_op[0]\n\n\ndef _taaf_w4a8_signed_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, workspace,\n                           sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k,\n                           mul_topk_weights, b_q_type, size_m, size_n, size_k, use_atomic_add, use_fp32_reduce,\n                           is_zp_float, thread_k=-1, thread_n=-1, blocks_per_sm=-1):\n    if input.dtype != torch.int8:\n        return _taaf_w4a8_stock_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros,\n                                     workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,\n                                     moe_block_size, top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k,\n                                     use_atomic_add, use_fp32_reduce, is_zp_float, thread_k, thread_n, blocks_per_sm)\n    return _taaf_w4a8_signed_load()(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros,\n                                    workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,\n                                    moe_block_size, top_k, mul_topk_weights, b_q_type.id, size_m, size_n, size_k,\n                                    use_atomic_add, use_fp32_reduce, is_zp_float, thread_k, thread_n, blocks_per_sm)\n\n\nif _taaf_w4a8_os.environ.get('VLLM_MARLIN_INPUT_DTYPE', '').strip().lower() == 'int8':\n    _taaf_w4a8_signed_load()\nmoe_wna16_marlin_gemm = _taaf_w4a8_signed_gemm\n"
W4A8_ORACLE = "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"
W4A8_ORACLE_MARKER = "_taaf_w4a8_signed_scales"
W4A8_ORACLE_TAIL = '\n\n_taaf_w4a8_stock_scales = marlin_act_int8_process_scales\n\n\ndef _taaf_w4a8_signed_scales(s):\n    peak = s.abs().max()\n    factor = 1 / 4096 * peak.float()\n    s = s / peak * 4096\n    return s.round().to(torch.int16).view(s.dtype), factor\n\n\nmarlin_act_int8_process_scales = _taaf_w4a8_signed_scales\n'
W4A8_CT_MARKER = "_taaf_w4a8_ct_group"
W4A8_CT_TAIL = "\n\n_taaf_w4a8_ct_stock_init = CompressedTensorsWNA16MoEMethod.__init__\n_taaf_w4a8_ct_said = set()\n\n\ndef _taaf_w4a8_ct_group(self, *args, **kwargs):\n    _taaf_w4a8_ct_stock_init(self, *args, **kwargs)\n    group = getattr(self, 'group_size', None)\n    if getattr(self, 'is_marlin', False) and self.input_dtype in (torch.int8, torch.float8_e4m3fn) \\\n            and group != 128:\n        if group not in _taaf_w4a8_ct_said:\n            _taaf_w4a8_ct_said.add(group)\n            print(f'W4A8_SKIP group {group}: this MoE keeps 16-bit input, the 8-bit Marlin kernels exist for '\n                  f'group 128 only', flush=True)\n        self.input_dtype = None\n\n\nCompressedTensorsWNA16MoEMethod.__init__ = _taaf_w4a8_ct_group\n"
W4A8_EXPERTS = "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
W4A8_EXPERTS_MARKER = "_taaf_w4a8_experts_group"
W4A8_EXPERTS_TAIL = "\n\n_taaf_w4a8_experts_stock_init = MarlinExpertsBase.__init__\n_taaf_w4a8_experts_said = set()\n\n\ndef _taaf_w4a8_experts_group(self, moe_config, quant_config, *args, **kwargs):\n    _taaf_w4a8_experts_stock_init(self, moe_config, quant_config, *args, **kwargs)\n    if self.input_dtype is None or self.input_dtype.itemsize != 1:\n        return\n    shape = quant_config.block_shape\n    group = shape[1] if shape else None\n    if group != 128:\n        if group not in _taaf_w4a8_experts_said:\n            _taaf_w4a8_experts_said.add(group)\n            print(f'W4A8_SKIP experts group {group}: these Marlin experts keep 16-bit input, the 8-bit kernels '\n                  f'exist for group 128 only', flush=True)\n        self.input_dtype = None\n\n\nMarlinExpertsBase.__init__ = _taaf_w4a8_experts_group\n"
W4A8_OP_LIBRARY = "w4a8fix_op.so"
W4A8_OP_INSTALLED = "vllm/taaf_w4a8fix_op.so"
W4A8_WNA16 = "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
W4A8_WNA16_MARKER = "_taaf_w4a8_gscale_config"
W4A8_WNA16_TAIL = "\n\n_taaf_w4a8_stock_config = CompressedTensorsWNA16MoEMethod.get_fused_moe_quant_config\n_taaf_w4a8_said = []\n\n\ndef _taaf_w4a8_gscale_config(self, layer):\n    made = _taaf_w4a8_stock_config(self, layer)\n    g1 = getattr(layer, 'w13_input_global_scale', None)\n    g2 = getattr(layer, 'w2_input_global_scale', None)\n    if made is None or (g1 is None and g2 is None) or self.wna16_backend == WNA16MoEBackend.HUMMING:\n        return made\n    fixed = make_wna16_moe_quant_config(\n        w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale, group_size=self.group_size,\n        num_bits=self.num_bits, w1_zp=getattr(layer, 'w13_weight_zero_point', None),\n        w2_zp=getattr(layer, 'w2_weight_zero_point', None), a1_gscale=g1, a2_gscale=g2,\n        gemm1_clamp_limit=getattr(layer, 'swiglu_limit', None), gemm1_alpha=getattr(layer, 'swiglu_alpha', None),\n        gemm1_beta=getattr(layer, 'swiglu_beta', None))\n    if not _taaf_w4a8_said:\n        _taaf_w4a8_said.append(1)\n        print('W4A8_GSCALE live: int8-input Marlin MoE gets its input global scales', flush=True)\n    return fixed\n\n\nCompressedTensorsWNA16MoEMethod.get_fused_moe_quant_config = _taaf_w4a8_gscale_config\n"
W4A8_EXPERTS_EDITS = (("", W4A8_EXPERTS_TAIL),)
W4A8_CT_EDITS = (("", W4A8_CT_TAIL),)
W4A8_WNA16_EDITS = (("", W4A8_WNA16_TAIL),)
W4A8_OPS_EDITS = (("", W4A8_OPS_TAIL),)
W4A8_ORACLE_EDITS = (("", W4A8_ORACLE_TAIL),)
FP8_DENSE_GROUPS = {
    "attn": r"(?:^|\.)model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)$",
    "gdn": r"(?:^|\.)model\.layers\.\d+\.linear_attn\.(?:in_proj_qkvz|out_proj)$",
    "shared": r"(?:^|\.)model\.layers\.\d+\.mlp\.shared_expert\.(?:gate_up_proj|down_proj)$",
    "hc": (r"(?:^|\.)model\.(?:layers\.\d+\.(?:attn|mlp)_hyper_connection|hyper_connection_mixer)"
           r"\.input_mix_weight_(?:down_block_inject|down|up)$"),
    "hc_up": (r"(?:^|\.)model\.(?:layers\.\d+\.(?:attn|mlp)_hyper_connection|hyper_connection_mixer)"
              r"\.input_mix_weight_up$"),
    "mtp": (r"(?:^|\.)mtp\.(?:fc_embedding|fc_hidden|layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|"
            r"mlp\.shared_expert\.(?:gate_up_proj|down_proj)))$"),
}
FP8_DENSE_DRAFT_GROUP = "mtp"
FP8_TRITON_BAND_ENV = "TAAF_VLLM_FP8_TRITON_BAND"
FP8_TRITON_BAND_ROWS_MAX = 16384
FP8_TRITON_BAND_HELPER = "fp8_triton_band.py"
FP8_TRITON_BAND_INSTALLED = "vllm/taaf_fp8_band.py"
FP8_DENSE_LOG_PATTERN = r"FP8_DENSE quantized .*|Selected \w+ for TaafFp8DenseMethod"
LINEAR_RELATIVE = "vllm/model_executor/layers/linear.py"
FP8_DENSE_TARGETS = {
    LINEAR_RELATIVE: (
        "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d",
        (
            ('            raise ValueError("All linear layers should support quant method.")\n'
             '        self.return_bias = return_bias\n',
             '            raise ValueError("All linear layers should support quant method.")\n'
             '        if type(self.quant_method) is UnquantizedLinearMethod:\n'
             '            from vllm.taaf_fp8_dense import dense_method\n'
             '\n'
             '            self.quant_method = dense_method(prefix, self.quant_method)\n'
             '        self.return_bias = return_bias\n'),
        ),
    ),
}
BF16C_ENV = "TAAF_VLLM_BF16C"
BF16C_GROUPS_ENV = "TAAF_VLLM_BF16C_GROUPS"
BF16C_ROWS_ENV = "TAAF_VLLM_BF16C_ROWS"
BF16C_FAST_ENV = "TAAF_VLLM_BF16C_FAST2"
BF16C_FAST_VERDICT_ENV = "TAAF_VLLM_BF16C_FAST2_VERDICT"
BF16C_FAST_TIMEOUT = 480
BF16C_MIX_ENV = "TAAF_VLLM_BF16C_MIX"
BF16C_MIX_VERDICT_ENV = "TAAF_VLLM_BF16C_MIX_VERDICT"
BF16C_MIX_DECODE_ENV = "TAAF_VLLM_BF16C_MIX_DECODE"
BF16C_MIX_TIMEOUT = 900
BF16C_GROUPS =("attn", "gdn", "shared", "hc", "mtp", "vision", "head")
BF16C_DEFAULT_GROUPS = ("attn", "gdn", "shared", "hc", "head")
BF16C_ROWS_MAX = 2048
BF16C_HELPER = "bf16c.py"
BF16C_INSTALLED = "vllm/taaf_bf16c.py"
EMBEDDING_RELATIVE = "vllm/model_executor/layers/vocab_parallel_embedding.py"
BF16C_TARGETS = {
    LINEAR_RELATIVE: (
        "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d",
        (
            ('    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
             '        if current_platform.is_cpu():\n'
             '            # MLA\'s kv_b_proj (see `skip_weight_relayout`): not perf-critical,\n',
             '    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
             '        from vllm import taaf_bf16c\n'
             '\n'
             '        if taaf_bf16c.after_loading(self, layer):\n'
             '            return\n'
             '        if current_platform.is_cpu():\n'
             '            # MLA\'s kv_b_proj (see `skip_weight_relayout`): not perf-critical,\n'),
            ('        return self._gemm_impl(layer, x, layer.weight, bias)\n',
             '        if hasattr(layer, "taaf_bf16c"):\n'
             '            from vllm import taaf_bf16c\n'
             '\n'
             '            return taaf_bf16c.apply(self, layer, x, bias)\n'
             '        return self._gemm_impl(layer, x, layer.weight, bias)\n'),
        ),
    ),
    EMBEDDING_RELATIVE: (
        "d42574ac79bd7d7847548784d2272c7c40cbd93fe6cdadb15e8c2db79d2074d7",
        (
            ('    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
             '        if current_platform.is_cpu():\n'
             '            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm\n',
             '    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
             '        from vllm import taaf_bf16c\n'
             '\n'
             '        if taaf_bf16c.after_loading(self, layer):\n'
             '            return\n'
             '        if current_platform.is_cpu():\n'
             '            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm\n'),
            ('        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)\n',
             '        if hasattr(layer, "taaf_bf16c"):\n'
             '            from vllm import taaf_bf16c\n'
             '\n'
             '            return taaf_bf16c.apply(self, layer, x, bias)\n'
             '        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)\n'),
        ),
    ),
}
WEIGHTS_PREFETCH_ENV = "TAAF_VLLM_WEIGHTS_PREFETCH"
WEIGHTS_PREFETCH_GIB_ENV = "TAAF_VLLM_WEIGHTS_PREFETCH_GIB"
WEIGHTS_PREFETCH_CHUNK_ENV = "TAAF_VLLM_WEIGHTS_PREFETCH_CHUNK_MIB"
WEIGHTS_PREFETCH_HELPER = "shard_prefetch.py"
WEIGHT_CACHE_ENV = "TAAF_VLLM_WEIGHT_CACHE"
WEIGHT_CACHE_ALLOW_BACKEND_ENV = "TAAF_VLLM_WEIGHT_CACHE_ALLOW_BACKEND"
WEIGHT_CACHE_PIDFILE = "vllm-weight-cache-daemon.pid"
WEIGHT_CACHE_DAEMON_LOG = "vllm-weight-cache-daemon.log"
WEIGHT_CACHE_SOCKET_GLOB = "vllm_weight_cache_*.sock"
WEIGHT_CACHE_READY_S = 180.0
WEIGHT_CACHE_POLL_S = 2.0
WEIGHT_CHECKSUM_HELPER = "weight_checksum.py"
WEIGHT_CHECKSUM_INSTALLED = "vllm/taaf_weight_checksum.py"
GPU_WORKER_RELATIVE = "vllm/v1/worker/gpu_worker.py"
WEIGHT_CHECKSUM_TARGETS = {
    GPU_WORKER_RELATIVE: (
        "6994436e4547c0996ab555b3e814e8ccea26b650dfc1b378b97d640745309347",
        (
            ('    def get_draft_model(self) -> nn.Module | None:\n'
             '        return self.model_runner.get_draft_model()\n',
             '    def get_draft_model(self) -> nn.Module | None:\n'
             '        return self.model_runner.get_draft_model()\n'
             '\n'
             '    def taaf_weight_checksum(self) -> dict:\n'
             '        from vllm.taaf_weight_checksum import model_digest\n'
             '\n'
             '        out = {"model": model_digest(self.get_model())}\n'
             '        draft = self.get_draft_model()\n'
             '        if draft is not None:\n'
             '            out["draft_model"] = model_digest(draft)\n'
             '        return out\n'),
        ),
    ),
}
MEM_EMBED_HOST_ENV = "TAAF_VLLM_EMBED_HOST"
MEM_VISION_HOST_ENV = "TAAF_VLLM_VISION_HOST"
MEM_LEDGER_ENV = "TAAF_VLLM_MEM_LEDGER"
MEM_LEDGER_EVERY_ENV = "TAAF_VLLM_MEM_LEDGER_S"
MEM_LEDGER_EVERY_MAX_S = 3600
MEM_OFFLOAD_HELPER = "mem_offload.py"
MEM_OFFLOAD_INSTALLED = "vllm/taaf_mem_offload.py"
EMBED_HOT_ENV = "TAAF_VLLM_EMBED_HOT"
EMBED_HOT_HELPER = "embed_hot.py"
EMBED_HOT_INSTALLED = "vllm/taaf_embed_hot.py"
EMBED_HOT_IDS = "embed_hot_ids.json"
EMBED_HOT_IDS_INSTALLED = "vllm/taaf_embed_hot_ids.json"
PLE_QHOT_ENV = "TAAF_VLLM_PLE_QHOT"
PLE_QHOT_ROWS = "ple_qhot_rows.npy"
PLE_QHOT_INSTALLED = "vllm/taaf_ple_qhot_rows.npy"
KV_AUTO_ENV = "TAAF_VLLM_KV_AUTO"
KV_AUTO_PLAN = "kv_auto_plan.json"
KV_AUTO_PLAN_INSTALLED = "vllm/taaf_kv_auto_plan.json"
MEM_OFFLOAD_TARGETS = {
    "vllm/v1/worker/gpu/model_runner.py": (
        "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
        (
            ('                    eplb_models_added = self.eplb.maybe_register_speculator(\n'
             '                        self.speculator, self.speculative_config, load_dummy_weights\n'
             '                    )\n'
             '        time_after_load = time.perf_counter()\n',
             '                    eplb_models_added = self.eplb.maybe_register_speculator(\n'
             '                        self.speculator, self.speculative_config, load_dummy_weights\n'
             '                    )\n'
             '            from vllm import taaf_mem_offload\n'
             '\n'
             '            taaf_mem_offload.apply(\n'
             '                self.model, getattr(self, "speculator", None), self.vllm_config\n'
             '            )\n'
             '        time_after_load = time.perf_counter()\n'),
        ),
    ),
}
QUANTIZATION_ENV = "TAAF_VLLM_QUANTIZATION"
QUANTIZATION_DEFAULT = "modelopt_fp4"
QUANTIZATIONS = ("modelopt_fp4", "inc")
DRAFT_MODEL_ENV = "TAAF_VLLM_DRAFT_MODEL"
PLE_EXACT_PIN_ENV = "TAAF_VLLM_PLE_EXACT_PIN"
PLE_HOST_HELPER = "ple_host.py"
PLE_HOST_INSTALLED = "vllm/taaf_ple_host.py"
PLE_HOST_TARGETS = {
    "vllm/models/qwen4_exp/nvidia/ngram_embedding.py": (
        "f3aaf29281b803dcebb4b179d5177704ed7b913170edf8107e33f729be825429",
        (
            ('        if not isinstance(quant_config, Fp8Config):\n            raise NotImplementedError(\n                "Qwen4Exp PLE embedding does not support quantization config "\n',
             '        if quant_config.get_name() == "inc":\n            from vllm import taaf_ple_host\n\n            return taaf_ple_host.inc_method(\n                quant_config, prefix, Qwen4ExpPLEUnquantizedEmbeddingMethod\n            )\n        if not isinstance(quant_config, Fp8Config):\n            raise NotImplementedError(\n                "Qwen4Exp PLE embedding does not support quantization config "\n'),
            ('        """Allocate the complete PLE weight directly in pinned CPU memory."""\n        return torch.empty(\n            num_embeddings,\n            embedding_dim,\n            dtype=dtype,\n            device="cpu",\n            pin_memory=True,\n        )\n',
             '        """Allocate the complete PLE weight directly in pinned CPU memory."""\n        from vllm import taaf_ple_host\n\n        return taaf_ple_host.allocate(num_embeddings, embedding_dim, dtype)\n'),
        ),
    ),
}
HOST_KV_PLE_GIB_ENV = "TAAF_VLLM_HOST_KV_PLE_GIB"
LOAD_DUMMY_ENV = "TAAF_VLLM_LOAD_DUMMY"
NSYS_ENV = "TAAF_VLLM_NSYS"
NSYS_DEB = "nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb"
NSYS_DEB_SHA256 = "c7309f1c9850f66a9eb95e7215883b8e8e439df6f65ea0cecd81e6b0181a4e83"
NSYS_ROOT = Path("/kaggle/tmp/nsys")
NSYS_TOOL = "opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys"
NSYS_SESSION = "taaf"
NSYS_LAUNCH = ("--trace=cuda,nvtx", "--cuda-graph-trace=graph")
PLE_HOT_ENV = "TAAF_VLLM_PLE_HOT"
PLE_HOT_GIB_ENV = "TAAF_VLLM_PLE_HOT_GIB"
PLE_HOT_DEFAULT_GIB = 16
PLE_HOT_DISK_GIB_ENV = "TAAF_VLLM_PLE_HOT_DISK_GIB"
PLE_HOT_DISK_DEFAULT_GIB = 48
PLE_HOT_TABLE_ROWS = 320_001_536
PLE_HOT_RECORD_BYTES = 241
PLE_HOT_ALIGN = 4096
PLE_HOT_TIER_SLACK_GIB = 1
PLE_HOT_HELPER = "ple_hot.py"
PLE_HOT_INSTALLED = "vllm/taaf_ple_hot.py"
PLE_PACKED_ENV = "TAAF_VLLM_PLE_PACKED"
PLE_PACKED_HELPER = "ple_packed.py"
PLE_PACKED_INSTALLED = "vllm/taaf_ple_packed.py"
PLE_DF11_ENV = "TAAF_VLLM_PLE_DF11"
PLE_DF11_RECORD_BYTES = 214
PLE_QSTORE_ENV = "TAAF_VLLM_PLE_QSTORE"
PLE_QSTORE_RECORD_BYTES = {"int8": 162, "int6": 122}
PLE_DF11_FILE_ENV = "TAAF_VLLM_PLE_DF11_FILE"
PLE_DF11_CHECKPOINT_ENV = "TAAF_VLLM_PLE_DF11_CHECKPOINT"
PLE_DF11_FILE_NAME = "ple_df11.taafdf11"
PLE_Q6_ENV = "TAAF_VLLM_PLE_QSTORE_PREBUILT"
PLE_Q6_DIR_ENV = "TAAF_VLLM_PLE_QSTORE_PREBUILT_DIR"
PLE_Q6_FILE_NAME = "ple_q6.taafq6"
PLE_Q6_PIECE_ENV = "TAAF_VLLM_PLE_Q6_PIECE_MIB"
PLE_DF11_MARK = "ple_embedding.ngram_embedding"
PREFETCH_SKIP_ENV = "TAAF_VLLM_WEIGHTS_PREFETCH_SKIP_MARK"
PLE_GRAPH_TABLE = "ple_graph_table.py"
PLE_HOT_TARGETS = {
    "vllm/models/qwen4_exp/nvidia/ngram_embedding.py": (
        "f3aaf29281b803dcebb4b179d5177704ed7b913170edf8107e33f729be825429",
        (
            ('            else Qwen4ExpPLEDeviceEmbedding\n        )\n        self.ngram_embedding = embedding_cls(\n',
             '            else Qwen4ExpPLEDeviceEmbedding\n        )\n        from vllm import taaf_ple_hot\n\n'
             '        embedding_cls = taaf_ple_hot.embedding_class(\n'
             '            embedding_cls, Qwen4ExpPLEPinnedHostEmbedding\n        )\n'
             '        self.ngram_embedding = embedding_cls(\n'),
            ('        embedding.start_prefetch(hidden_states, ngram_ids)\n',
             '        if hasattr(embedding, "taaf_source"):\n'
             '            embedding.taaf_source = (\n'
             '                self.compute_ngram_ids,\n'
             '                input_ids,\n'
             '                query_start_loc,\n'
             '                ngram_context,\n'
             '            )\n'
             '        embedding.start_prefetch(hidden_states, ngram_ids)\n'),
        ),
    ),
}
MODELOPT_RELATIVE = "vllm/model_executor/layers/quantization/modelopt.py"
WEIGHT_CACHE_BACKEND_TARGETS = {
    MODELOPT_RELATIVE: (
        "41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22",
        (
            ('            if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:\n',
             '            if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM and \\\n'
             '                    self.nvfp4_backend.value not in __import__("os").environ.get(\n'
             f'                        "{WEIGHT_CACHE_ALLOW_BACKEND_ENV}", "").split(","):\n'),
        ),
    ),
}
CACHE_KEEP_ENV = "TAAF_VLLM_CACHE_KEEP"
CACHE_KEEP_TABLE = "cache_keep.py"
CACHE_KEEP_MATCH_UNIT = 128
CACHE_KEEP_MATCH_UNIT_ENV = "TAAF_VLLM_CACHE_KEEP_MATCH_UNIT"
GDN_ACCEPT_ENV = "TAAF_VLLM_GDN_ACCEPT"
GDN_ACCEPT_HELPER = "gdn_accept.py"
GDN_ACCEPT_TABLE = "gdn_accept_table.py"
GDN_ACCEPT_INSTALLED = "vllm/taaf_gdn_accept.py"
GDN_FUSED_ENV = "TAAF_VLLM_GDN_FUSED"
GDN_ALIGN_TILES_ENV = "TAAF_VLLM_GDN_ALIGN_TILES"
GDN_ALIGN_TILES_HELPER = "gdn_align_tiles.py"
GDN_ALIGN_TILES_TABLE = "gdn_align_tiles_table.py"
GDN_ALIGN_TILES_INSTALLED = "vllm/taaf_gdn_align_tiles.py"
GDN_ALIGN_TILES_LIMIT = 64
GDN_STATE_Q8_ENV = "TAAF_VLLM_GDN_STATE_Q8"
GDN_STATE_Q8_TABLE = "gdn_q8_table.py"
GDN_Q8_FIRST_ENV = "TAAF_VLLM_GDN_Q8_FIRST"
GDN_SHORT_PREFILL_ENV = "TAAF_VLLM_GDN_SHORT_PREFILL"
GDN_STATE_Q8_PACK_ENV = "TAAF_VLLM_GDN_STATE_Q8_PACK"
GDN_STATE_Q8_PACK_HELPER = "gdn_q8_pack.py"
GDN_STATE_Q8_PACK_TABLE = "gdn_q8_pack_table.py"
GDN_STATE_Q8_PACK_INSTALLED = "vllm/taaf_gdn_q8_pack.py"
GDN_STATE_INPLACE_ENV = "TAAF_VLLM_GDN_STATE_INPLACE"
MAMBA_ZERO_NEW_ENV = "TAAF_VLLM_MAMBA_ZERO_NEW"
MAMBA_ZERO_NEW_TABLE = "mamba_zero_table.py"
GDN_FUSED_MODES = {"1": "defer", "defer": "defer", "kernel": "kernel", "tiled": "tiled", "lazy": "lazy"}
GDN_LAZY_STEPS_ENV = "TAAF_VLLM_GDN_LAZY_STEPS"
GDN_FUSED_HELPER = "gdn_fused.py"
GDN_FUSED_TABLE = "gdn_fused_table.py"
GDN_FUSED_INSTALLED = "vllm/taaf_gdn_fused.py"
GDN_VERIFY_CUDA_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA"
GDN_VERIFY_CUDA_HELPER = "gdn_verify_cuda.py"
GDN_VERIFY_CUDA_INSTALLED = "vllm/taaf_gdn_verify_cuda.py"
GDN_VERIFY_CUDA_COHERENT_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA_COHERENT"
GDN_SLOT_FILL_ENV = "TAAF_VLLM_GDN_SLOT_FILL"
PREP_NOSYNC_ENV = "TAAF_VLLM_PREP_NOSYNC"
GDN_META_ONCE_ENV = "TAAF_VLLM_GDN_META_ONCE"
GDN_META_ONCE_TABLE = "gdn_meta_once_table.py"
GDN_META_ONCE_HELPER = "gdn_meta_once.py"
GDN_META_ONCE_INSTALLED = "vllm/taaf_gdn_meta_once.py"
GDN_META_SLICE_ENV = "TAAF_VLLM_GDN_META_SLICE"
PREP_GDN_ENV = "TAAF_VLLM_PREP_GDN"
STEP_TIMERS_PREP_ENV = "TAAF_VLLM_STEP_TIMERS_PREP"
STEP_TIMERS_HC_ENV = "TAAF_VLLM_STEP_TIMERS_HC"
GDN_SPEC_PREFIX_ENV = "TAAF_VLLM_GDN_SPEC_PREFIX"
GDN_SPEC_PREFIX_TABLE = "gdn_prefix_table.py"
GDN_SPEC_PREFIX_HELPER = "gdn_prefix.py"
GDN_SPEC_PREFIX_INSTALLED = "vllm/taaf_gdn_prefix.py"
GDN_NORM_ENV = "TAAF_VLLM_GDN_NORM_STRIDED"
GDN_NORM_HELPER = "gdn_norm.py"
GDN_NORM_TABLE = "gdn_norm_table.py"
GDN_NORM_INSTALLED = "vllm/taaf_gdn_norm.py"
DRAFT_META_ENV = "TAAF_VLLM_DRAFT_FUSED_META"
DRAFT_META_HELPER = "draft_meta.py"
DRAFT_META_TABLE = "draft_meta_table.py"
DRAFT_META_INSTALLED = "vllm/taaf_draft_meta.py"
DRAFT_EARLY_STOP_ENV = "TAAF_VLLM_DRAFT_EARLY_STOP"
DRAFT_EARLY_STOP_HELPER = "draft_early_stop.py"
DRAFT_EARLY_STOP_TABLE = "draft_early_stop_table.py"
DRAFT_EARLY_STOP_INSTALLED = "vllm/taaf_draft_early_stop.py"
CONV_SYNCFREE_ENV = "TAAF_VLLM_CONV_SYNCFREE"
CONV_SYNCFREE_HELPER = "conv_syncfree.py"
CONV_SYNCFREE_TABLE = "conv_syncfree_table.py"
CONV_SYNCFREE_INSTALLED = "vllm/taaf_conv_syncfree.py"
MARKS_ENV = "TAAF_VLLM_MARKS"
MARKS_HELPER = "marks.py"
MARKS_INSTALLED = "vllm/taaf_marks.py"
GDN_TRIL_ENV = "TAAF_VLLM_GDN_TRIL_WARPS1"
GDN_TRIL_TABLE = "gdn_tril_table.py"
GDN_TRIL_ILP_ENV = "TAAF_VLLM_GDN_TRIL_ILP"
GDN_TRIL_ILP_TABLE = "gdn_tril_ilp_table.py"
GDN_BA_VIEWS_ENV = "TAAF_VLLM_GDN_BA_VIEWS"
GDN_BA_VIEWS_TABLE = "gdn_ba_views_table.py"
GDN_TAIL_ZERO_ENV = "TAAF_VLLM_GDN_TAIL_ZERO"
GDN_TAIL_ZERO_TABLE = "gdn_tail_zero_table.py"
GDN_DECODE_TAIL_ENV = "TAAF_VLLM_GDN_DECODE_TAIL_ZERO"
GDN_DECODE_TAIL_TABLE = "gdn_decode_tail_table.py"
GDN_VERIFY_AHEAD_ENV = "TAAF_VLLM_GDN_VERIFY_AHEAD"
MOE_ADD_FUSED_ENV = "TAAF_VLLM_MOE_ADD_FUSED"
MOE_ADD_FUSED_HELPER = "moe_add_fused.py"
MOE_ADD_FUSED_TABLE = "moe_add_fused_table.py"
MOE_ADD_FUSED_INSTALLED = "vllm/taaf_moe_add_fused.py"
ROUTER_CHAIN_ENV = "TAAF_VLLM_ROUTER_CHAIN"
ROUTER_CHAIN_VALUES = ("", "0", "1", "gate", "gate_up")
ROUTER_CHAIN_HELPER = "router_chain.py"
ROUTER_CHAIN_TABLE = "router_chain_table.py"
ROUTER_CHAIN_INSTALLED = "vllm/taaf_router_chain.py"
MOE_TAAF_ENV = "TAAF_VLLM_MOE_TAAF_ROWS"
MOE_TAAF_HELPER = "moe_taaf.py"
MOE_TAAF_TABLE = "moe_taaf_table.py"
MOE_TAAF_INSTALLED = "vllm/taaf_moe_taaf.py"
MOE_TAAF_WINDOW = re.compile(r"(\d+)(?::(\d+))?")
MOE_BLOCK_ENV = "TAAF_VLLM_MOE_BLOCK_M"
MOE_BLOCK_HELPER = "moe_block.py"
MOE_BLOCK_TABLE = "moe_block_table.py"
MOE_BLOCK_INSTALLED = "vllm/taaf_moe_block.py"
MOE_SUM_ENV = "TAAF_VLLM_MOE_SUM_EPI"
MOE_SILU_ENV = "TAAF_VLLM_MOE_SILU_EPI"
MOE_QUANT_ENV = "TAAF_VLLM_MOE_QUANT_FUSED"
MOE_ACT_QUANT_ENV = "TAAF_VLLM_MOE_ACT_QUANT"
MOE_SUM_HELPER = "marlin_sum.py"
MOE_SUM_INSTALLED = "vllm/taaf_marlin_sum.py"
EXPERT_BITS_ENV = "TAAF_VLLM_EXPERT_BITS"
HUMMING_INT3_HELPER = "humming_int3.py"
HUMMING_INT3_TABLE = "humming_int3_table.py"
HUMMING_INT3_INSTALLED = "vllm/taaf_humming_int3.py"
MOE_BLOCK_DEFAULT = ((1700, 2100, 32),)
MOE_BLOCK_SIZES = (8, 16, 32, 48, 64)
MOE_BLOCK_BAND = re.compile(r"(\d+)-(\d+):(\d+)")
MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
MIXED_FULL_SEQS_ENV = "TAAF_VLLM_MIXED_FULL_SEQS"
MIXED_FULL_LARGEST_ENV = "TAAF_VLLM_MIXED_FULL_LARGEST"
MIXED_FULL_HELPER = "mixed_full.py"
MIXED_FULL_TABLE = "mixed_full_table.py"
MIXED_FULL_INSTALLED = "vllm/taaf_mixed_full.py"
MIXED_FULL_SPEC_HELPER = "mixed_full_spec.py"
MIXED_FULL_SPEC_INSTALLED = "vllm/taaf_mixed_full_spec.py"
MIXED_FULL_GRAPH_HELPER = "mixed_full_graph.py"
MIXED_FULL_GRAPH_TABLE = "mixed_full_graph_table.py"
MIXED_FULL_GRAPH_INSTALLED = "vllm/taaf_mixed_full_graph.py"
QSA_GRAPH_HELPER = "qsa_graph.py"
QSA_GRAPH_TABLE = "qsa_graph_table.py"
QSA_GRAPH_INSTALLED = "vllm/taaf_qsa_graph.py"
GDN_MIXED_FULL_HELPER = "gdn_mixed_full.py"
GDN_MIXED_FULL_TABLE = "gdn_mixed_full_table.py"
GDN_MIXED_FULL_INSTALLED = "vllm/taaf_gdn_mixed_full.py"
GDN_MIXED_FI_ENV = "TAAF_VLLM_GDN_MIXED_FI"
GDN_MIXED_FI_FP16_ENV = "TAAF_VLLM_GDN_MIXED_FI_FP16"
GDN_MIXED_FI_FP16_FLASHINFER = "FLASHINFER_GDN_FP16_ACCUM_MMA"
PDL_ENV = "TAAF_VLLM_PDL"
PDL_TABLE = "pdl_table.py"
MM_PROCESSOR_ONLY_ENV = "TAAF_VLLM_MM_PROCESSOR_ONLY"
MM_PROCESSOR_ONLY_HELPER = "mm_processor_only.py"
MM_PROCESSOR_ONLY_TABLE = "mm_processor_only_table.py"
MM_PROCESSOR_ONLY_INSTALLED = "vllm/taaf_mm_processor_only.py"
NO_BLANK_ENV = "TAAF_VLLM_NO_BLANK_LINES"
NO_BLANK_HELPER = "no_blank.py"
NO_BLANK_TABLE = "no_blank_table.py"
NO_BLANK_INSTALLED = "vllm/taaf_no_blank.py"
SMALL_KERNELS_HELPER = "small_kernels.py"
SMALL_KERNELS_INSTALLED = "vllm/taaf_small_kernels.py"
SMALL_KERNELS_MARLIN_ENV = "TAAF_VLLM_MARLIN_WORKSPACE_REUSE"
SMALL_KERNELS_MARLIN_TABLE = "marlin_workspace_table.py"
SMALL_KERNELS_DRAFT_TOPK_ENV = "TAAF_VLLM_DRAFT_TOPK_TRITON"
SMALL_KERNELS_INVALIDATE_ENV = "TAAF_VLLM_HISPARSE_INVALIDATE_FUSED"
SMALL_KERNELS_INVALIDATE_TABLE = "hisparse_invalidate_table.py"
SMALL_KERNELS_GLUE_ENV = "TAAF_VLLM_SAMPLE_GLUE_FUSED"
SMALL_KERNELS_GEMMA_ENV = "TAAF_VLLM_GEMMA_NORM_FUSED"
SMALL_KERNELS_GEMMA_TABLE = "gemma_norm_table.py"
SMALL_KERNELS_TRITON = "small_kernels_triton.py"
SMALL_KERNELS_TRITON_INSTALLED = "vllm/taaf_small_kernels_triton.py"
SMALL_KERNELS_DRAFT_TOPK_KERNEL = "draft_topk_kernel.py"
SMALL_KERNELS_DRAFT_TOPK_INSTALLED = "vllm/taaf_draft_topk_kernel.py"
SAMPLE_TOPK_ENV = "TAAF_VLLM_SAMPLE_TOPK"
SAMPLE_TOPK_DRAFT_ENV = "TAAF_VLLM_SAMPLE_TOPK_DRAFT"
TAIL_KEPT_LOGITS_ENV = "TAAF_VLLM_TAIL_KEPT_LOGITS"
SAMPLE_TOPK_HELPER = "sample_topk.py"
SAMPLE_TOPK_TABLE = "sample_topk_table.py"
SAMPLE_TOPK_INSTALLED = "vllm/taaf_sample_topk.py"
SAMPLE_TOPK_KERNEL = "sample_topk_kernel.py"
SAMPLE_TOPK_KERNEL_INSTALLED = "vllm/taaf_sample_topk_kernel.py"
BLOCK_VERIFY_ENV = "TAAF_VLLM_BLOCK_VERIFY"
BLOCK_VERIFY_HELPER = "block_verify.py"
BLOCK_VERIFY_TABLE = "block_verify_table.py"
BLOCK_VERIFY_INSTALLED = "vllm/taaf_block_verify.py"
ACCEPT_LAMBDA_ENV = "TAAF_VLLM_ACCEPT_LAMBDA"
ACCEPT_LAMBDA_GUARD_ENV = "TAAF_VLLM_ACCEPT_LAMBDA_GUARD"
ACCEPT_LAMBDA_TOOL_IDS_ENV = "TAAF_VLLM_ACCEPT_LAMBDA_TOOL_IDS"
ACCEPT_LAMBDA_HELPER = "accept_lambda.py"
ACCEPT_LAMBDA_TABLE = "accept_lambda_table.py"
ACCEPT_LAMBDA_INSTALLED = "vllm/taaf_accept_lambda.py"
ACCEPT_LAMBDA_MAX = 16.0
OPEN_THINK_ENV = "TAAF_VLLM_OPEN_THINK_TEMPLATE"
OPEN_THINK_FROM = '    {%- elif message.role == "assistant" %}\n'
OPEN_THINK_TO = (
    '    {%- elif message.role == "assistant" and open_last_thinking is defined and open_last_thinking '
    'and loop.last %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + (message.reasoning_content "
    "if message.reasoning_content is string else '') }}\n" + OPEN_THINK_FROM)
STABLE_THINK_ENV = "TAAF_VLLM_STABLE_THINK_SYSTEM"
STABLE_THINK_FROM = "{%- if enable_thinking is undefined or enable_thinking is true %}\n"
STABLE_THINK_TO = "{%- if true %}\n"
STEP_TIMERS_ENV = "TAAF_VLLM_STEP_TIMERS"
QUIET_ENV = "TAAF_VLLM_QUIET"
QUIET_RERUN_ENV = "KAGGLE_IS_COMPETITION_RERUN"
EXPERT_TALLY_ENV = "TAAF_VLLM_EXPERT_TALLY"
STEP_TIMERS_HELPER = "step_timers.py"
STEP_TIMERS_TABLE = "step_timers_table.py"
STEP_TIMERS_INSTALLED = "vllm/taaf_step_timers.py"
VISION_TALLY_ENV = "TAAF_VLLM_VISION_TALLY"
VISION_TALLY_HELPER = "vision_tally.py"
VISION_TALLY_INSTALLED = "vllm/taaf_vision_tally.py"
SPEC_INDEX_ENV = "TAAF_VLLM_SPEC_INDEX"
SPEC_PREFETCH_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC"
SPEC_PREFETCH_LAYERS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC_LAYERS"
SPEC_INDEX_VARIANTS_ENV = "TAAF_VLLM_SPEC_INDEX_VARIANTS"
SPEC_INDEX_ROWS_ENV = "TAAF_VLLM_SPEC_INDEX_ROWS"
SPEC_INDEX_ROWS_MAX = 4096
SPEC_INDEX_LANES_ENV = "TAAF_VLLM_SPEC_INDEX_LANES"
SPEC_INDEX_LANES_MAX = 512
SPEC_INDEX_VARIANTS = ("self", "mlp", "attn", "prev")
SPEC_INDEX_HELPER = "spec_index.py"
SPEC_INDEX_INSTALLED = "vllm/taaf_spec_index.py"
DRAFT_CAPTURE_ENV = "TAAF_VLLM_DRAFT_CAPTURE"
DRAFT_CAPTURE_GIB_ENV = "TAAF_VLLM_DRAFT_CAPTURE_GIB"
DRAFT_CAPTURE_LANES_ENV = "TAAF_VLLM_DRAFT_CAPTURE_LANES"
DRAFT_CAPTURE_DIR_ENV = "TAAF_VLLM_DRAFT_CAPTURE_DIR"
DRAFT_CAPTURE_IDLE_ENV = "TAAF_VLLM_DRAFT_CAPTURE_IDLE_S"
DRAFT_CAPTURE_DRAFTER_ENV = "TAAF_VLLM_DRAFT_CAPTURE_DRAFTER"
DRAFT_CAPTURE_HARD_ENV = "TAAF_VLLM_DRAFT_CAPTURE_HARD"
DRAFT_CAPTURE_ROTATE_ENV = "TAAF_VLLM_DRAFT_CAPTURE_ROTATE"
DRAFT_CAPTURE_GAMES_ENV = "TAAF_VLLM_DRAFT_CAPTURE_GAMES"
DRAFT_CAPTURE_HELPER = "draft_capture.py"
DRAFT_CAPTURE_INSTALLED = "vllm/taaf_draft_capture.py"
VISION_CACHE_ENV = "TAAF_VLLM_VISION_CACHE"
VISION_CACHE_HELPER = "vision_cache.py"
VISION_CACHE_TABLE = "vision_cache_table.py"
VISION_CACHE_INSTALLED = "vllm/taaf_vision_cache.py"
VISION_CACHE_MAX_MIB = 4096
VISION_AHEAD_ENV = "TAAF_VLLM_VISION_AHEAD"
VISION_AHEAD_HELPER = "vision_ahead.py"
VISION_AHEAD_TABLE = "vision_ahead_table.py"
VISION_AHEAD_INSTALLED = "vllm/taaf_vision_ahead.py"
PREFILL_HOLD_ENV = "TAAF_VLLM_PREFILL_HOLD"
PREFILL_HOLD_TOKENS_ENV = "TAAF_VLLM_PREFILL_HOLD_TOKENS"
PREFILL_HOLD_MS_ENV = "TAAF_VLLM_PREFILL_HOLD_MS"
PREFILL_HOLD_HELPER = "prefill_hold.py"
PREFILL_HOLD_TABLE = "prefill_hold_table.py"
PREFILL_HOLD_INSTALLED = "vllm/taaf_prefill_hold.py"
PREFILL_HOLD_AUTO_ENV = "TAAF_VLLM_PREFILL_HOLD_AUTO"
PREFILL_HOLD_CAP_MS_ENV = "TAAF_VLLM_PREFILL_HOLD_CAP_MS"
PREFILL_HOLD_F_MS_ENV = "TAAF_VLLM_PREFILL_HOLD_F_MS"
PREFILL_HOLD_P_US_ENV = "TAAF_VLLM_PREFILL_HOLD_P_US"
PREFILL_HOLD_ALIGNED_ENV = "TAAF_VLLM_PREFILL_HOLD_ALIGNED"
PREFILL_HOLD_FILL_SHARE_ENV = "TAAF_VLLM_PREFILL_HOLD_FILL_SHARE"
PREFILL_HOLD_WAIT_WEIGHT_ENV = "TAAF_VLLM_PREFILL_HOLD_WAIT_WEIGHT"
PREFILL_HOLD_AUTO_TABLE = "prefill_hold_auto_table.py"
QSA_SELFCHECK_ENV = "TAAF_VLLM_QSA_SELFCHECK"
QSA_SELFCHECK_TIMEOUT_S = 240
QSA_SELFCHECK_MARK = "QSA_SELFCHECK_ROWS "
QSA_SELFCHECK_SCRIPT = """
import json, sys
shapes = json.loads(sys.argv[1])
dtype, spread = shapes.pop("dtype"), shapes.pop("spread")
import torch
from vllm import taaf_qsa_kv8 as kv8
rows = [list(row) for row in kv8.self_check(dtype, **shapes)]
if spread:
    from vllm import taaf_host_kv_fused as fused
    row_bytes = shapes["kv_heads"] * kv8.content_bytes(*kv8.widths(kv8.layout(dtype), shapes["head_dim"]))
    rows += [list(row) for row in fused.spread_self_check(row_bytes, spread)]
print("QSA_SELFCHECK_ROWS " + json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}), flush=True)
if kv8.identity_repeats():
    for row in kv8.identity_check(dtype, repeats=kv8.identity_repeats(), **shapes):
        print("QSA_IDENTITY " + json.dumps(list(row)), flush=True)
"""
MOE_IDENTITY_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[2])
try:
    import moe_identity
    for row in moe_identity.moe_identity_check(**json.loads(sys.argv[1])):
        print("QSA_IDENTITY " + json.dumps(list(row)), flush=True)
except Exception as error:
    print("QSA_IDENTITY " + json.dumps(["moe", "stopped", f"{type(error).__name__}: {error}"[:300], 0.0]), flush=True)
"""
MOE_IDENTITY_TIMEOUT_S = 180
KERNEL_IDENTITY_ENV = "TAAF_VLLM_KERNEL_IDENTITY"
MOE_IDENTITY_DEFAULTS = {"hidden": 2560, "intermediate": 640, "topk": 10, "group": 128}
QSA_SELFCHECK_LEAN_MARK = "QSA_SELFCHECK_LEAN "
QSA_SELFCHECK_LEAN_SCRIPT = """
rows = [list(row) for row in kv8.lean_self_check(dtype, **shapes)]
print("QSA_SELFCHECK_LEAN " + json.dumps(rows), flush=True)
"""
QSA_SELFCHECK_DECODE_LEAN_MARK = "QSA_SELFCHECK_DECODE_LEAN "
QSA_SELFCHECK_DECODE_LEAN_SCRIPT = """
rows = [list(row) for row in kv8.decode_lean_self_check(dtype, **shapes)]
print("QSA_SELFCHECK_DECODE_LEAN " + json.dumps(rows), flush=True)
"""
QSA_SELFCHECK_UNION_DECODE_MARK = "QSA_SELFCHECK_UNION_DECODE "
QSA_SELFCHECK_UNION_DECODE_SCRIPT = """
import os
union_rows = int(os.environ.get("TAAF_VLLM_MTP_TOKENS") or 0) + 1
rows = [list(row) for row in kv8.union_decode_self_check(dtype, query_len=union_rows, **shapes)]
print("QSA_SELFCHECK_UNION_DECODE " + json.dumps(rows), flush=True)
"""
TOKEN_PIN_ENV = "TAAF_VLLM_TOKEN_PIN"
TOKEN_PIN_HELPER = "token_pin.py"
TOKEN_PIN_TABLE = "token_pin_table.py"
TOKEN_PIN_INSTALLED = "vllm/taaf_token_pin.py"
L2_PIN_ENV = "TAAF_VLLM_L2_PIN"
L2_PIN_HELPER = "l2_pin.py"
L2_PIN_TABLE = "l2_pin_table.py"
L2_PIN_INSTALLED = "vllm/taaf_l2_pin.py"
CACHE_KEEP_END_ENV = "TAAF_VLLM_CACHE_KEEP_END"
CACHE_KEEP_END_RAM_ENV = "TAAF_VLLM_CACHE_KEEP_END_RAM"
PAUSED_REGISTER_ENV = "TAAF_VLLM_PAUSED_REGISTER"
MAMBA_JUNCTION_ENV = "TAAF_VLLM_MAMBA_JUNCTION"
HOST_KV_ENV = "TAAF_VLLM_HOST_KV"
QSA_INDEX_COMPACT_ENV = "TAAF_VLLM_QSA_INDEX_COMPACT"
MTP_INDEX_ROWS_ENV = "TAAF_VLLM_MTP_INDEX_ROWS"
KV4_WRITE_FUSED_ENV = "TAAF_VLLM_KV4_WRITE_FUSED"
HOST_KV_GIB_ENV = "TAAF_VLLM_HOST_KV_GIB"
HOST_KV_ROWS_ENV = "TAAF_VLLM_HOST_KV_ROWS"
HOST_KV_HELPER = "qsa_hisparse.py"
HOST_KV_TABLE = "host_kv_table.py"
HOST_KV_INSTALLED = "vllm/taaf_qsa_hisparse.py"
HOST_KV_DEFAULT_GIB = 78
HOST_KV_CGROUP_MAX = Path("/sys/fs/cgroup/memory.max")
HOST_KV_PLE_SHM_GIB = 65
HOST_KV_OTHER_ANON_GIB = 12
HOST_KV_HEADROOM_GIB = 20
HOST_MEMINFO = Path("/proc/meminfo")
HOST_SYSTEM_RESERVE_GIB = 8
HOST_KV_MAX_GPU_PERMILLE = 930
HOST_KV_STAGE_ENV = "TAAF_VLLM_HOST_KV_STAGE_MIB"
HOST_KV_STAGE_MAX_MIB = 4096
HOST_KV_SPLIT_ENV = "TAAF_VLLM_HOST_KV_GATHER_SPLIT"
HOST_KV_ADMIT_ENV = "TAAF_VLLM_HOST_KV_ADMIT"
HOST_KV_ADMIT_MODES = ("", "worst", "real")
HOST_KV_ROW_ATTEND_ENV = "TAAF_VLLM_HOST_KV_ROW_ATTEND"
HOST_KV_TAIL_ENV = "TAAF_VLLM_HOST_KV_TAIL_PREFETCH"
HOST_KV_SPARSE_ENV = "TAAF_VLLM_HOST_KV_SPARSE_STAGE"
HOST_KV_HOT_RELEASE_ENV = "TAAF_VLLM_HOST_KV_HOT_RELEASE"
HOST_KV_HOLD_CLEAN_ENV = "TAAF_VLLM_HOST_KV_HOLD_CLEAN"
HOST_KV_HOLD_RESERVE_ENV = "TAAF_VLLM_HOST_KV_HOLD_RESERVE"
HOT_KEEP_ENV = "TAAF_VLLM_HOT_KEEP"
HOT_KEEP_WARM_ENV = "TAAF_VLLM_HOT_KEEP_WARM"
HOT_KEEP_WARM_MODES = ("", "0", "1", "held", "owners")
QSA_STAGE_AHEAD_ENV = "TAAF_VLLM_QSA_STAGE_AHEAD"
QSA_STAGE_AHEAD_PROGRAMS_ENV = "TAAF_VLLM_QSA_STAGE_AHEAD_PROGRAMS"
QSA_STAGE_AHEAD_PROGRAMS = 64
QSA_STAGE_AHEAD_PROGRAMS_MAX = 4096
QSA_STAGE_REUSE_ENV = "TAAF_VLLM_QSA_STAGE_REUSE"
QSA_HALVES_ENV = "TAAF_VLLM_QSA_HALVES"
HOST_KV_INDEXER_WIDE_ENV = "TAAF_VLLM_HOST_KV_INDEXER_WIDE"
HOST_KV_UNION_COUNT_ENV = "TAAF_VLLM_HOST_KV_UNION_COUNT"
HOST_KV_PLAN_ONCE_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE"
HOST_KV_PLAN_ONCE_CAPTURED_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE_CAPTURED"
HOST_KV_PLAN_ONCE_CHECKS_ENV = "TAAF_VLLM_HISPARSE_PLAN_ONCE_CHECKS"
HOST_KV_TAKE_ROWS_ENV = "TAAF_VLLM_HOST_KV_TAKE_ROWS"
HOST_KV_TAKE_SHADOW_ENV = "TAAF_VLLM_HOST_KV_TAKE_ROWS_SHADOW"
SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
MAMBA_HIT_CHECK_ENV = "TAAF_VLLM_MAMBA_HIT_CHECK"
MAMBA_HIT_CHECK_MIN_ENV = "TAAF_VLLM_MAMBA_HIT_CHECK_MIN"
HOST_KV_SYNCFREE_ENV = "TAAF_VLLM_HISPARSE_SYNCFREE"
HOST_KV_UNION_ENV = "TAAF_VLLM_HOST_KV_UNION_SWAP"
HOST_KV_HOT_ROWS_ENV = "TAAF_VLLM_HOST_KV_HOT_ROWS"
HOST_KV_COPY_ASYNC_ENV = "TAAF_VLLM_HISPARSE_COPY_ASYNC"
HOST_KV_SPILL_SKIP_ENV = "TAAF_VLLM_HOST_KV_SPILL_SKIP"
HOST_KV_SPILL_ONESORT_ENV = "TAAF_VLLM_HOST_KV_SPILL_ONESORT"
HOST_KV_MARK_FILL_ENV = "TAAF_VLLM_HOST_KV_MARK_FILL"
HOST_KV_GATHER_EARLY_ENV = "TAAF_VLLM_HOST_KV_GATHER_EARLY"
HOST_KV_FUSED_ENV = "TAAF_VLLM_HOST_KV_FUSED"
HOST_KV_FUSED_HELPER = "host_kv_fused.py"
HOST_KV_FUSED_INSTALLED = "vllm/taaf_host_kv_fused.py"
HOST_KV_HOT_SPLIT_ENV = "TAAF_VLLM_HOST_KV_HOT_SPLIT"
HOST_KV_UNION_BITMAP_ENV = "TAAF_VLLM_HOST_KV_UNION_BITMAP"
HOST_KV_SPILL_FLAG_ENV = "TAAF_VLLM_HOST_KV_SPILL_FLAG"
HOST_KV_SPILL_TRUST_ENV = "TAAF_VLLM_HOST_KV_SPILL_TRUST"
HOST_KV_GATHER_TALLY_ENV = "TAAF_VLLM_HOST_KV_GATHER_TALLY"
HOST_KV_LAST_SLICE_ENV = "TAAF_VLLM_HOST_KV_LAST_SLICE"
HOST_KV_SOURCES_FUSED_ENV = "TAAF_VLLM_HOST_KV_SOURCES_FUSED"
HOST_KV_ONE_LAUNCH_ENV = "TAAF_VLLM_HOST_KV_ONE_LAUNCH"
HOST_KV_ONE_PACK_ENV = "TAAF_VLLM_HOST_KV_ONE_PACK"
HOST_KV_ONE_HASH_ENV = "TAAF_VLLM_HOST_KV_ONE_HASH"
HOST_KV_FLAT_COPY_ENV = "TAAF_VLLM_HOST_KV_FLAT_COPY"
HOST_KV_FLAT_LINES_ENV = "TAAF_VLLM_HOST_KV_FLAT_LINES"
HOST_KV_ONE_CUDA_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA"
HOST_KV_ONE_CUDA_THREADS_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_THREADS"
HOST_KV_ONE_CUDA_THREADS = (128, 256, 512, 1024)
HOST_KV_ONE_CUDA_DEFINES_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_DEFINES"
HOST_KV_ONE_CUDA_FILES = (("host_kv_resolve_cuda.py", "vllm/taaf_host_kv_resolve_cuda.py"),
                          ("host_kv_resolve.cu", "vllm/taaf_host_kv_resolve.cu"))
HOST_KV_GROUP_ENV = "TAAF_VLLM_HOST_KV_GROUP"
HOST_KV_GROUPS = (0, 1, 2, 4, 8)
HOST_KV_GATHER_SPREAD_ENV = "TAAF_VLLM_HOST_KV_GATHER_SPREAD"
HOST_KV_GATHER_SPREAD_MAX = 64
HOST_KV_PCIE_TALLY_ENV = "TAAF_VLLM_HOST_KV_PCIE_TALLY"
HOST_KV_PCIE_TALLY_LEVELS = (0, 1, 2)
HOST_KV_RUN_MERGE_ENV = "TAAF_VLLM_HOST_KV_RUN_MERGE"
HOST_KV_RUN_MERGE_MAX = 64
HOST_KV_FUSED_KNOBS = {
    HOST_KV_UNION_BITMAP_ENV: "a lane's union is ranked in a bitmap of its context positions instead of a sort",
    HOST_KV_SPILL_FLAG_ENV: "a spill pass with no overflowing lane returns on the GPU and swaps in on the compute "
                            "stream",
    HOST_KV_SPILL_TRUST_ENV: "the spill passes are not issued when a lane's distinct picks fit the hot region, "
                             "which its overflow counter reports",
    HOST_KV_GATHER_TALLY_ENV: "rows copied from host memory are counted on the GPU and read every 100 steps",
    HOST_KV_LAST_SLICE_ENV: "the drafter's prefill last rows are sliced on the GPU instead of masked, so the host "
                            "never waits for their count",
    HOST_KV_SOURCES_FUSED_ENV: "a prefill plan's resident sources are one kernel a resident group over a shared "
                               "representative per host block",
    HOST_KV_ONE_LAUNCH_ENV: "a decode resolve's residency decision, plan, guard, tally and host reads of its misses "
                            "are one Triton kernel, and the hot split's resolves share that launch",
    HOST_KV_ONE_PACK_ENV: "a resolve's key list is packed to the picks that still need a host row and sorted in the "
                          "smallest tile that holds them, which rests on the sort's readers being a search of that "
                          "list and the host-row order of the copy plan, both of which a packed tile leaves as they "
                          "were",
    HOST_KV_ONE_HASH_ENV: "a resolve finds its picks in the hot region through a per-lane hash of their host rows "
                          "instead of a sort of every pick and a binary search of it, and sorts only the misses it "
                          "copies, which leaves the hot state, the lists and the copy plan as the sort left them",
    HOST_KV_FLAT_COPY_ENV: "the one-launch resolve's host reads run as a second kernel over every lane's misses in "
                           "equal shares across the card, instead of four programs a lane waiting on their own "
                           "lane's resolve",
    HOST_KV_FLAT_LINES_ENV: "the flat copy reads a missed group of four host rows that starts on a 128-byte line as "
                            "whole lines split among its four entries, instead of each row on its own with partial "
                            "lines at both ends, and writes every word to the hot slot of the row it belongs to",
    HOST_KV_ONE_CUDA_ENV: "the flat one-launch resolve runs as a CUDA C++ kernel, a block a lane with the picks "
                          "hashed in shared memory, built with nvcc or NVRTC and checked against the Triton resolve "
                          "on a cloned world at the first eager forward, which leaves the same lists, copy plan, hot "
                          "state and tallies",
}
HOST_KV_MIXED_CAPTURE_ENV = "TAAF_VLLM_HOST_KV_MIXED_CAPTURE"
HOST_KV_FIT_ENV = "TAAF_VLLM_HOST_KV_FIT"
HOST_KV_GROWTH_ENV = "TAAF_VLLM_HOST_KV_GROWTH_GIB"
HOST_KV_GROWTH_DEFAULT_GIB = 28
HOST_KV_CGROUP_STAT = Path("/sys/fs/cgroup/memory.stat")
HOST_KV_FITTED: dict[int, int] = {}
HOST_KV_TAIL_MAX = 8192
HOST_KV_PREFETCH_ENV = "TAAF_VLLM_HOST_KV_PREFETCH"
HOST_KV_PREFETCH_MAX = 256
HOST_KV_PREFETCH_PROGRAMS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PROGRAMS"
HOST_KV_PREFETCH_PREV_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV"
HOST_KV_PREFETCH_PREV_ROWS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_ROWS"
HOST_KV_PREFETCH_PREV_ROWS_MAX = 8192
HOST_KV_PREFETCH_PREV_SCOPE_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_SCOPE"
HOST_KV_PREFETCH_PREV_SCOPES = ("missed", "picks")
HOST_KV_PREFETCH_PREV_ONCE_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_PREV_ONCE"
HOST_KV_PICK_RANKS_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_RANKS"
HOST_KV_PICK_RANKS_MAX = 4096
HOST_KV_PICK_LOG_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG"
HOST_KV_PICK_LOG_MAX = 64
HOST_KV_PICK_EVERY_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_EVERY"
HOST_KV_PICK_AT_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_AT"
HOST_KV_PICK_STEPS_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_STEPS"
HOST_KV_PICK_REQS_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_REQS"
HOST_KV_PICK_MIB_ENV = "TAAF_VLLM_HOST_KV_PICK_LOG_MIB"
HOST_KV_PICK_AT_MAX = 16
KV_REBASE_ENV = "TAAF_VLLM_KV_REBASE"
CACHE_KEEP_END_HELPER = "cache_keep_end.py"
CACHE_KEEP_END_TABLE = "cache_keep_end_table.py"
CACHE_KEEP_END_INSTALLED = "vllm/taaf_cache_keep_end.py"
PLE_RIDE_ENV = "TAAF_VLLM_PLE_RIDE"
KEEP_SOFT_ENV = "TAAF_VLLM_KEEP_SOFT"
TAIL_SOFT_ENV = "TAAF_VLLM_TAIL_SOFT"
TAIL_SOFT_RESERVE = 512
TAIL_SOFT_HARD = -1
SLIM_REPORT_ENV = "TAAF_VLLM_SLIM_REPORT"
SLIM_REPORT_S_ENV = "TAAF_VLLM_SLIM_REPORT_S"
SLIM_REPORT_S = 60.0
WAIT_KEEP_ENV = "TAAF_VLLM_WAIT_KEEP"
WAIT_KEEP_S_ENV = "TAAF_VLLM_WAIT_KEEP_S"
WAIT_KEEP_S = 5.0
WAIT_ORDER_ENV = "TAAF_VLLM_WAIT_ORDER"
WAIT_ORDER_AGE_ENV = "TAAF_VLLM_WAIT_ORDER_AGE"
WAIT_ORDER_AGE = 1000.0
SLIM_HELPER = "slim_state.py"
SLIM_TABLE = "slim_state_table.py"
SLIM_INSTALLED = "vllm/taaf_slim_state.py"
SLIM_SETTINGS = "vllm/taaf_slim_state.json"
WEIGHT_UTILS_RELATIVE = "vllm/model_executor/model_loader/weight_utils.py"
MTP_RELATIVES =("vllm/models/qwen4_exp/nvidia/mtp.py", "vllm/models/qwen4_exp/amd/mtp.py")
WEIGHT_READ_FROM = ("                    if should_skip_weight(name, local_expert_ids):\n"
                    "                        continue\n"
                    "                    param = f.get_tensor(name)\n"
                    "                    yield name, param\n")
WEIGHT_READ_TO = ("                    if should_skip_weight(name, local_expert_ids):\n"
                  "                        continue\n"
                  "                    if _taaf_wanted is not None and not _taaf_wanted(name):\n"
                  "                        continue\n"
                  "                    param = f.get_tensor(name)\n"
                  "                    yield name, param\n")
WEIGHT_ITERATOR_ANCHOR = "def safetensors_weights_iterator(\n"
WEIGHT_WANTED_LINE = "_taaf_wanted = None\n\n\n"
MTP_LOAD_FROM = "        return loader.load_weights(remap_weight_names(), mapper=mapper)\n"
MTP_LOAD_TO = (
    "        from vllm.model_executor.model_loader import weight_utils as _taaf_weight_utils\n"
    "\n"
    "        _taaf_seen = [0, 0]\n"
    "\n"
    "        def _taaf_keep(name):\n"
    "            kept = _remap_mtp_weight_name(name) is not None\n"
    "            _taaf_seen[int(kept)] += 1\n"
    "            return kept\n"
    "\n"
    "        _taaf_weight_utils._taaf_wanted = _taaf_keep\n"
    "        try:\n"
    "            return loader.load_weights(remap_weight_names(), mapper=mapper)\n"
    "        finally:\n"
    "            _taaf_weight_utils._taaf_wanted = None\n"
    "            print(f\"MTP_OWN_READS: the draft model read {_taaf_seen[1]} tensors and left \"\n"
    "                  f\"{_taaf_seen[0]} unread\", flush=True)\n")
MAMBA_SSM_DTYPE_ENV = "TAAF_VLLM_MAMBA_SSM_DTYPE"
MAMBA_SSM_DTYPES = ("float32", "float16", "bfloat16")
INDEXER_KV_DTYPE_ENV = "TAAF_VLLM_INDEXER_KV_DTYPE"
INDEXER_KV_DTYPES = ("bf16", "fp8")
ITERATION_LOG_ENV = "TAAF_VLLM_ITERATION_LOG"
PROFILER_DIR_ENV = "TAAF_VLLM_PROFILER_DIR"
PROFILER_STEPS_ENV = "TAAF_VLLM_PROFILER_STEPS"
PROFILER_STACK_ENV = "TAAF_VLLM_PROFILER_STACK"
DRAFT_VOCAB_ENV = "TAAF_VLLM_DRAFT_VOCAB"
SPEC_METRICS_ENV = "TAAF_VLLM_PER_REQUEST_SPEC_METRICS"
SPEC_METRICS_LEVELS = ("none", "summary", "detailed")
ADAPTIVE_VERIFY_ENV = "TAAF_VLLM_ADAPTIVE_VERIFY"
DRAFT_BUDGET_ENV = "TAAF_VLLM_DRAFT_BUDGET"
DRAFT_BUDGET_TABLE = "draft_budget_table.py"
DRAFT_BUDGET_MANAGER = ("draft_budget.py", "vllm/taaf_draft_budget.py")
DRAFT_BUDGET_HELPERS = (("draft_budget_policy.py", "vllm/taaf_draft_budget_policy.py"),
                        ("draft_budget_kernel.py", "vllm/taaf_draft_budget_kernel.py"),
                        ("draft_budget_tables.py", "vllm/taaf_draft_budget_tables.py"))
DRAFT_BUDGET_VARLEN: dict[str, bool] = {"on": False}
PDL_INDEXER_RELATIVE = "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py"
DRAFT_SAMPLE_ENV = "TAAF_VLLM_DRAFT_SAMPLE"
DRAFT_SAMPLE_TOP_ENV = "TAAF_VLLM_DRAFT_SAMPLE_TOP"
DRAFT_SAMPLE_NARROW_ENV = "TAAF_VLLM_DRAFT_SAMPLE_NARROW"
DRAFT_SAMPLE_NARROW_HELPER = "draft_sample_narrow.py"
DRAFT_SAMPLE_NARROW_INSTALLED = "vllm/taaf_draft_sample_narrow.py"
DRAFT_HEAD_FP8_ENV = "TAAF_VLLM_DRAFT_HEAD_FP8"
DRAFT_HEAD_FP8_HELPER = "draft_head_fp8.py"
DRAFT_HEAD_FP8_INSTALLED = "vllm/taaf_draft_head_fp8.py"
DRAFT_VOCAB_PAD_ENV = "TAAF_VLLM_DRAFT_VOCAB_PAD"
DRAFT_VOCAB_PAD_MAX = 4096
DRAFT_VOCAB_HELPER = "draft_vocab.py"
DRAFT_VOCAB_JSON = "draft_vocab.json"
DRAFT_VOCAB_FILE_ENV = "TAAF_VLLM_DRAFT_VOCAB_FILE"
DRAFT_VOCAB_SEARCH = Path("/kaggle/input")
DRAFT_VOCAB_INSTALLED = "vllm/taaf_draft_vocab.py"
DRAFT_VOCAB_INSTALLED_JSON = "vllm/taaf_draft_vocab.json"
DRAFT_VOCAB_TARGET = MTP_RELATIVES[0]
DRAFT_VOCAB_MARKER = "vllm.taaf_draft_vocab"
DRAFT_VOCAB_FROM = (
    "    def compute_logits(\n"
    "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n")
DRAFT_VOCAB_TO = (
    DRAFT_VOCAB_FROM
    + "\n"
    "    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
    "        from vllm.taaf_draft_vocab import top_tokens\n"
    "\n"
    "        return top_tokens(self, hidden_states)\n")
TARGET_VOCAB_ENV = "TAAF_VLLM_TARGET_VOCAB"
TARGET_VOCAB_HELPER = "target_vocab.py"
TARGET_VOCAB_JSON = "target_vocab.json"
TARGET_VOCAB_INSTALLED = "vllm/taaf_target_vocab.py"
TARGET_VOCAB_INSTALLED_JSON = "vllm/taaf_target_vocab.json"
TARGET_VOCAB_TARGET = "vllm/models/qwen4_exp/nvidia/model.py"
TARGET_VOCAB_MARKER = "vllm.taaf_target_vocab"
TARGET_VOCAB_FROM = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n")
TARGET_VOCAB_TO = (
    "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    "        from vllm.taaf_target_vocab import logits\n"
    "\n"
    "        return logits(self, self.logits_processor, hidden_states)\n")
TARGET_VOCAB_EDITS = ((TARGET_VOCAB_FROM, TARGET_VOCAB_TO),)
MTP_WEIGHTS_ENV = "TAAF_VLLM_MTP_WEIGHTS"
MTP_WEIGHTS_HELPER = "mtp_weights.py"
MTP_WEIGHTS_INSTALLED = "vllm/taaf_mtp_weights.py"
MTP_WEIGHTS_MARKER = "vllm.taaf_mtp_weights"
MTP_WEIGHTS_FROM = (
    "        def remap_weight_names():\n"
    "            for name, weight in weights:\n")
MTP_WEIGHTS_TO = (
    "        def remap_weight_names():\n"
    "            from vllm.taaf_mtp_weights import swapped\n"
    "\n"
    "            for name, weight in swapped(weights):\n")
VISION_WEIGHTS_ENV = "TAAF_VLLM_VISION_WEIGHTS"
VISION_WEIGHTS_HELPER = "vision_weights.py"
VISION_WEIGHTS_INSTALLED = "vllm/taaf_vision_weights.py"
VISION_WEIGHTS_MARKER = "vllm.taaf_vision_weights"
VISION_WEIGHTS_RELATIVE = "vllm/models/qwen4_exp/nvidia/model.py"
VISION_WEIGHTS_FROM = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        mapper = self.hf_to_vllm_mapper | WeightsMapper(\n"
    "            orig_to_new_substr={\"mtp.\": None},\n"
    "            orig_to_new_prefix={\"visual.\": None} if self.language_model_only else {},\n")
VISION_WEIGHTS_TO = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        from vllm.taaf_vision_weights import swapped\n"
    "\n"
    "        weights = swapped(weights)\n"
    "        mapper = self.hf_to_vllm_mapper | WeightsMapper(\n"
    "            orig_to_new_substr={\"mtp.\": None},\n"
    "            orig_to_new_prefix={\"visual.\": None} if self.language_model_only else {},\n")
VISION_WEIGHTS_EDITS = ((VISION_WEIGHTS_FROM, VISION_WEIGHTS_TO),)
PIECEWISE_DRAFT_RELATIVE = "vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py"
PIECEWISE_DRAFT_PINNED = "13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5"
PIECEWISE_DRAFT_ANCHOR = (
    '    earlier capture would execute kernels with stale buffer contents.\n'
    '    """\n'
    '\n'
    '    def capture(\n')
PIECEWISE_DRAFT_METHOD = (
    '    def _init_candidates(self) -> None:\n'
    '        sizes = self.compilation_config.cudagraph_capture_sizes\n'
    '        if not sizes or max(sizes) <= {ceiling}:\n'
    '            return super()._init_candidates()\n'
    '        self.compilation_config.cudagraph_capture_sizes = [\n'
    '            size for size in sizes if size <= {ceiling}\n'
    '        ]\n'
    '        try:\n'
    '            super()._init_candidates()\n'
    '        finally:\n'
    '            self.compilation_config.cudagraph_capture_sizes = sizes\n'
    '\n')
MTP_LAST_ROWS_ENV = "TAAF_VLLM_MTP_LAST_ROWS"
MTP_LAST_ROWS_MODEL_RELATIVE = "vllm/models/qwen4_exp/nvidia/model.py"
MTP_LAST_ROWS_SPECULATOR_RELATIVE = (
    "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
)
MTP_LAST_ROWS_MTP_RELATIVE = MTP_RELATIVES[0]
MTP_LAST_ROWS_MARKER = "supports_taaf_last_rows"
MTP_LAST_ROWS_TARGETS = {
    MTP_LAST_ROWS_MODEL_RELATIVE: (
        "2d9d98054dfa4e2c2c434526ef6fbdceede06b35c38a8ea6dccb42161a666dfb",
        (
            ('        input_ids: torch.Tensor | None,\n'
             '        query_start_loc: torch.Tensor | None,\n'
             '        ngram_context: torch.Tensor | None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n',
             '        input_ids: torch.Tensor | None,\n'
             '        query_start_loc: torch.Tensor | None,\n'
             '        ngram_context: torch.Tensor | None,\n'
             '        last_rows: torch.Tensor | None = None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n'),
            ('        if self.layer_type == "linear_attention":\n'
             '            attn_out = self.linear_attn(hidden_states=block_input)\n'
             '        elif self.layer_type == "full_attention":\n'
             '            attn_out = self.self_attn(\n'
             '                hidden_states=block_input,\n'
             '                positions=positions,\n'
             '            )\n'
             '        else:\n'
             '            raise ValueError("Invalid layer_type")\n'
             '\n'
             '        mlp_hc = self.mlp_hyper_connection\n',
             '        if self.layer_type == "linear_attention":\n'
             '            attn_out = self.linear_attn(hidden_states=block_input)\n'
             '        elif self.layer_type == "full_attention":\n'
             '            attn_out = self.self_attn(\n'
             '                hidden_states=block_input,\n'
             '                positions=positions,\n'
             '            )\n'
             '        else:\n'
             '            raise ValueError("Invalid layer_type")\n'
             '\n'
             '        if last_rows is not None:\n'
             '            hidden_states = hidden_states.index_select(0, last_rows)\n'
             '            attn_out = attn_out.index_select(0, last_rows)\n'
             '            if injection is not None:\n'
             '                injection = injection.index_select(0, last_rows)\n'
             '\n'
             '        mlp_hc = self.mlp_hyper_connection\n'),
        ),
    ),
    MTP_LAST_ROWS_SPECULATOR_RELATIVE: (
        "5b97f54e20ddf65b98e939832d050745f67ef43cfa145786da300f1a8322e3c6",
        (
            ('from typing import Any\n'
             '\n'
             'import torch\n'
             'import torch.nn as nn\n',
             'from typing import Any\n'
             '\n'
             'import os\n'
             'import torch\n'
             'import torch.nn as nn\n'),
            ('logger = init_logger(__name__)\n'
             '\n'
             '\n'
             'class AutoRegressiveSpeculator(DraftModelSpeculator):\n',
             'logger = init_logger(__name__)\n'
             '\n'
             '_TAAF_MTP_LAST_ROWS = os.environ.get("TAAF_VLLM_MTP_LAST_ROWS") == "1"\n'
             '\n'
             '\n'
             'class AutoRegressiveSpeculator(DraftModelSpeculator):\n'),
            ('    @torch.inference_mode()\n'
             '    def _run_model(\n'
             '        self,\n'
             '        num_tokens: int,\n'
             '        attn_metadata: dict[str, Any] | None,\n'
             '        slot_mappings: dict[str, torch.Tensor] | None,\n'
             '        num_tokens_across_dp: torch.Tensor | None,\n'
             '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n'
             '        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor]:\n',
             '    @torch.inference_mode()\n'
             '    def _run_model(\n'
             '        self,\n'
             '        num_tokens: int,\n'
             '        attn_metadata: dict[str, Any] | None,\n'
             '        slot_mappings: dict[str, torch.Tensor] | None,\n'
             '        num_tokens_across_dp: torch.Tensor | None,\n'
             '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n'
             '        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n'
             '        last_rows: torch.Tensor | None = None,\n'
             '    ) -> tuple[torch.Tensor, torch.Tensor]:\n'),
            ('            model_inputs = dict(\n'
             '                input_ids=input_buffers.input_ids[:num_tokens],\n'
             '                positions=input_buffers.positions[:num_tokens],\n'
             '                hidden_states=self.hidden_states[:num_tokens],\n'
             '                inputs_embeds=inputs_embeds,\n'
             '            )\n'
             '            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:\n',
             '            model_inputs = dict(\n'
             '                input_ids=input_buffers.input_ids[:num_tokens],\n'
             '                positions=input_buffers.positions[:num_tokens],\n'
             '                hidden_states=self.hidden_states[:num_tokens],\n'
             '                inputs_embeds=inputs_embeds,\n'
             '            )\n'
             '            if last_rows is not None:\n'
             '                model_inputs["last_rows"] = last_rows\n'
             '            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:\n'),
            ('        return last_hidden_states, hidden_states\n'
             '\n'
             '    def _prefill(\n',
             '        return last_hidden_states, hidden_states\n'
             '\n'
             '    def _taaf_last_rows(\n'
             '        self, last_token_indices: torch.Tensor\n'
             '    ) -> torch.Tensor | None:\n'
             '        if not _TAAF_MTP_LAST_ROWS or self.pcp_manager is not None:\n'
             '            return None\n'
             '        inner = getattr(self.model, "model", None)\n'
             '        if not getattr(inner, "supports_taaf_last_rows", False):\n'
             '            return None\n'
             '        return last_token_indices\n'
             '\n'
             '    def _prefill(\n'),
            ('    def _prefill(\n'
             '        self,\n'
             '        num_reqs: int,\n'
             '        num_tokens: int,\n'
             '        attn_metadata: dict[str, Any] | None,\n'
             '        slot_mappings: dict[str, torch.Tensor] | None,\n'
             '        num_tokens_across_dp: torch.Tensor | None,\n'
             '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n'
             '        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n'
             '    ) -> None:\n'
             '        last_token_indices = self.last_token_indices[:num_reqs]\n'
             '        positions = self.input_buffers.positions[last_token_indices]\n'
             '        # The output hidden state at position P (= positions) and the token id\n'
             '        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the\n'
             '        # position before the sampled token, so the net adjustment is +1.\n'
             '        sample_src_positions = positions + 1\n'
             '        idx_mapping = self.idx_mapping[:num_reqs]\n'
             '\n'
             '        last_hidden_states, hidden_states = self._run_model(\n'
             '            num_tokens,\n'
             '            attn_metadata,\n'
             '            slot_mappings,\n'
             '            num_tokens_across_dp=num_tokens_across_dp,\n'
             '            cudagraph_runtime_mode=cudagraph_runtime_mode,\n'
             '            mm_inputs=mm_inputs,\n'
             '        )\n'
             '        if self.pcp_manager is not None:\n'
             '            last_hidden_states, hidden_states = self.pcp_manager.restore_draft_prefill(\n'
             '                last_hidden_states, hidden_states\n'
             '            )\n'
             '\n'
             '        sample_hidden_states = last_hidden_states[last_token_indices]\n'
             '        self.draft_tokens[:num_reqs, 0] = self.sample_draft(\n'
             '            sample_hidden_states,\n'
             '            sample_src_positions,\n'
             '            idx_mapping,\n'
             '            self.temperature,\n'
             '            self.seeds,\n'
             '            self.current_draft_step,\n'
             '            self.draft_logits,\n'
             '        )\n'
             '        if last_hidden_states is hidden_states:\n'
             '            self.hidden_states[:num_reqs] = sample_hidden_states\n'
             '        else:\n'
             '            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]\n'
             '        self.input_buffers.positions[:num_reqs] = positions\n'
             '        self.sample_src_positions[:num_reqs] = sample_src_positions\n',
             '    def _prefill(\n'
             '        self,\n'
             '        num_reqs: int,\n'
             '        num_tokens: int,\n'
             '        attn_metadata: dict[str, Any] | None,\n'
             '        slot_mappings: dict[str, torch.Tensor] | None,\n'
             '        num_tokens_across_dp: torch.Tensor | None,\n'
             '        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n'
             '        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,\n'
             '    ) -> None:\n'
             '        last_token_indices = self.last_token_indices[:num_reqs]\n'
             '        positions = self.input_buffers.positions[last_token_indices]\n'
             '        # The output hidden state at position P (= positions) and the token id\n'
             '        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the\n'
             '        # position before the sampled token, so the net adjustment is +1.\n'
             '        sample_src_positions = positions + 1\n'
             '        idx_mapping = self.idx_mapping[:num_reqs]\n'
             '\n'
             '        last_rows = (\n'
             '            None if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE\n'
             '            else self._taaf_last_rows(last_token_indices)\n'
             '        )\n'
             '        last_hidden_states, hidden_states = self._run_model(\n'
             '            num_tokens,\n'
             '            attn_metadata,\n'
             '            slot_mappings,\n'
             '            num_tokens_across_dp=num_tokens_across_dp,\n'
             '            cudagraph_runtime_mode=cudagraph_runtime_mode,\n'
             '            mm_inputs=mm_inputs,\n'
             '            last_rows=last_rows,\n'
             '        )\n'
             '        if self.pcp_manager is not None:\n'
             '            last_hidden_states, hidden_states = self.pcp_manager.restore_draft_prefill(\n'
             '                last_hidden_states, hidden_states\n'
             '            )\n'
             '\n'
             '        sample_hidden_states = (\n'
             '            last_hidden_states if last_rows is not None\n'
             '            else last_hidden_states[last_token_indices]\n'
             '        )\n'
             '        self.draft_tokens[:num_reqs, 0] = self.sample_draft(\n'
             '            sample_hidden_states,\n'
             '            sample_src_positions,\n'
             '            idx_mapping,\n'
             '            self.temperature,\n'
             '            self.seeds,\n'
             '            self.current_draft_step,\n'
             '            self.draft_logits,\n'
             '        )\n'
             '        if last_hidden_states is hidden_states:\n'
             '            self.hidden_states[:num_reqs] = sample_hidden_states\n'
             '        elif last_rows is not None:\n'
             '            self.hidden_states[:num_reqs] = hidden_states\n'
             '        else:\n'
             '            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]\n'
             '        self.input_buffers.positions[:num_reqs] = positions\n'
             '        self.sample_src_positions[:num_reqs] = sample_src_positions\n'),
        ),
    ),
}
MTP_LAST_ROWS_MTP_EDITS = (
    ('class Qwen4ExpMultiTokenPredictor(nn.Module):\n'
     '    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER\n'
     '\n'
     '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:\n',
     'class Qwen4ExpMultiTokenPredictor(nn.Module):\n'
     '    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER\n'
     '    supports_taaf_last_rows = True\n'
     '\n'
     '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:\n'),
    ('    def forward(\n'
     '        self,\n'
     '        input_ids: torch.Tensor | None,\n'
     '        positions: torch.Tensor,\n'
     '        hidden_states: torch.Tensor | None = None,\n'
     '        intermediate_tensors: IntermediateTensors | None = None,\n'
     '        inputs_embeds: torch.Tensor | None = None,\n'
     '        spec_step_idx: int = 0,\n'
     '    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:\n'
     '        hc_count = self.hc_count\n',
     '    def forward(\n'
     '        self,\n'
     '        input_ids: torch.Tensor | None,\n'
     '        positions: torch.Tensor,\n'
     '        hidden_states: torch.Tensor | None = None,\n'
     '        intermediate_tensors: IntermediateTensors | None = None,\n'
     '        inputs_embeds: torch.Tensor | None = None,\n'
     '        spec_step_idx: int = 0,\n'
     '        last_rows: torch.Tensor | None = None,\n'
     '    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:\n'
     '        hc_count = self.hc_count\n'),
    ('        current_step_idx = spec_step_idx % self.num_mtp_layers\n'
     '        layer = self.layers[current_step_idx]\n'
     '        hidden_states, block_output, injection = layer(\n'
     '            hidden_states=hidden_states,\n'
     '            prev_block_output=prev_block_output,\n'
     '            prev_injection=None,\n'
     '            positions=positions,\n'
     '            input_ids=None,\n'
     '            query_start_loc=None,\n'
     '            ngram_context=None,\n'
     '        )\n',
     '        current_step_idx = spec_step_idx % self.num_mtp_layers\n'
     '        layer = self.layers[current_step_idx]\n'
     '        hidden_states, block_output, injection = layer(\n'
     '            hidden_states=hidden_states,\n'
     '            prev_block_output=prev_block_output,\n'
     '            prev_injection=None,\n'
     '            positions=positions,\n'
     '            input_ids=None,\n'
     '            query_start_loc=None,\n'
     '            ngram_context=None,\n'
     '            last_rows=last_rows,\n'
     '        )\n'),
    ('    def forward(\n'
     '        self,\n'
     '        input_ids: torch.Tensor | None,\n'
     '        positions: torch.Tensor,\n'
     '        hidden_states: torch.Tensor | None = None,\n'
     '        intermediate_tensors: IntermediateTensors | None = None,\n'
     '        inputs_embeds: torch.Tensor | None = None,\n'
     '        spec_step_idx: int = 0,\n'
     '    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:\n'
     '        return self.model(\n'
     '            input_ids,\n'
     '            positions,\n'
     '            hidden_states,\n'
     '            intermediate_tensors,\n'
     '            inputs_embeds,\n'
     '            spec_step_idx=spec_step_idx,\n'
     '        )\n',
     '    def forward(\n'
     '        self,\n'
     '        input_ids: torch.Tensor | None,\n'
     '        positions: torch.Tensor,\n'
     '        hidden_states: torch.Tensor | None = None,\n'
     '        intermediate_tensors: IntermediateTensors | None = None,\n'
     '        inputs_embeds: torch.Tensor | None = None,\n'
     '        spec_step_idx: int = 0,\n'
     '        last_rows: torch.Tensor | None = None,\n'
     '    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:\n'
     '        return self.model(\n'
     '            input_ids,\n'
     '            positions,\n'
     '            hidden_states,\n'
     '            intermediate_tensors,\n'
     '            inputs_embeds,\n'
     '            spec_step_idx=spec_step_idx,\n'
     '            last_rows=last_rows,\n'
     '        )\n'),
)
MTP_ATTN_ROWS_ENV = "TAAF_VLLM_MTP_ATTN_ROWS"
MTP_ATTN_ROWS_EDITS = (
    ('            attn_out = self.self_attn(\n'
     '                hidden_states=block_input,\n'
     '                positions=positions,\n'
     '            )\n',
     '            if last_rows is None:\n'
     '                attn_out = self.self_attn(\n'
     '                    hidden_states=block_input,\n'
     '                    positions=positions,\n'
     '                )\n'
     '            else:\n'
     '                from vllm.taaf_qsa_kv8 import last_rows_scope\n'
     '\n'
     '                with last_rows_scope(last_rows):\n'
     '                    attn_out = self.self_attn(\n'
     '                        hidden_states=block_input,\n'
     '                        positions=positions,\n'
     '                    )\n'),
)
SHARED_STREAM_ENV = "TAAF_VLLM_SHARED_STREAM_TOKENS"
SHARED_STREAM_RELATIVE = "vllm/model_executor/layers/fused_moe/runner/shared_experts.py"
SHARED_STREAM_PINNED = "d89e4c79c46f6fe61262c51f3facf102e71ce1e4eb3db67f72b6bd8470937be9"
SHARED_STREAM_MAX = 8192
SHARED_STREAM_EDITS = (
    ('from enum import IntEnum\n'
     '\n'
     'import torch\n',
     'import os\n'
     'from enum import IntEnum\n'
     '\n'
     'import torch\n'),
    ('logger = init_logger(__name__)\n'
     '\n'
     '\n'
     'class SharedExpertsOrder(IntEnum):\n',
     'logger = init_logger(__name__)\n'
     '\n'
     '_TAAF_SHARED_STREAM_TOKENS = int(os.environ.get("TAAF_VLLM_SHARED_STREAM_TOKENS") or 0)\n'
     '_TAAF_SHARED_STREAM_SEEN: set[int] = set()\n'
     '\n'
     '\n'
     'class SharedExpertsOrder(IntEnum):\n'),
    ('        should_run_shared_in_aux_stream = (\n'
     '            current_platform.is_cuda_alike()\n'
     '            and self._stream is not None\n'
     '            and hidden_states.shape[0]\n'
     '            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD\n'
     '        )\n'
     '\n'
     '        if should_run_shared_in_aux_stream:\n'
     '            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED\n',
     '        rows = hidden_states.shape[0]\n'
     '        should_run_shared_in_aux_stream = (\n'
     '            current_platform.is_cuda_alike()\n'
     '            and self._stream is not None\n'
     '            and rows\n'
     '            <= max(\n'
     '                envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,\n'
     '                _TAAF_SHARED_STREAM_TOKENS,\n'
     '            )\n'
     '        )\n'
     '\n'
     '        if should_run_shared_in_aux_stream:\n'
     '            if (\n'
     '                rows > envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD\n'
     '                and rows not in _TAAF_SHARED_STREAM_SEEN\n'
     '                and len(_TAAF_SHARED_STREAM_SEEN) < 8\n'
     '            ):\n'
     '                _TAAF_SHARED_STREAM_SEEN.add(rows)\n'
     '                print(\n'
     '                    f"SHARED_STREAM: the shared expert of {rows} rows runs on the aux "\n'
     '                    f"stream, above the stock {envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD}",\n'
     '                    flush=True,\n'
     '                )\n'
     '            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED\n'),
)
ENFORCE_EAGER_ENV = "TAAF_VLLM_ENFORCE_EAGER"
MOE_ROUTE_ENV = "TAAF_VLLM_MOE_ROUTE"
MOE_ROUTE_FILE_ENV = "TAAF_VLLM_MOE_ROUTE_FILE"
MOE_SHARE_SETTINGS_ENV = "TAAF_VLLM_MOE_SHARE_SETTINGS"
TRACE_REPLAY_ENV = "TAAF_VLLM_TRACE_REPLAY"
MOE_ROUTE_SETUP = "moe_route_setup.py"
MOE_SHARE_ENV = "TAAF_VLLM_MOE_SHARE"
MOE_SHARE_SETUP = "moe_share_setup.py"
ROUTER_PAD_ENV = "TAAF_VLLM_ROUTER_PAD"
ROUTER_PAD_HELPER = "router_pad.py"
ROUTER_FUSED_ENV = "TAAF_VLLM_ROUTER_FUSED"
ROUTER_FUSED_HELPER = "router_fused.py"
ROUTER_FUSED_INSTALLED = "vllm/taaf_router_fused.py"
ROUTER_FUSED_MARK = "vllm import taaf_router_fused"
ROUTER_PAD_MARK = "vllm.taaf_router_pad"
ROUTER_FUSED_TARGETS = {
    "vllm/model_executor/layers/fused_moe/router/fused_topk_router.py": (
        "1c80f4af4d781fd66b12aa6a7a0fd30082bc521df40f1252987a8ac4c0bef51c",
        (
            ("from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter\n",
             "from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter\n"
             "from vllm import taaf_router_fused as _taaf_router_fused\n"),
            ("        topk_func = dispatch_topk_softmax_func(\n"
             "            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()\n"
             "        )\n"
             "        topk_weights, topk_ids = topk_func(\n"
             "            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize\n"
             "        )\n",
             "        topk_func = dispatch_topk_softmax_func(\n"
             "            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()\n"
             "        )\n"
             "        topk_weights, topk_ids = _taaf_router_fused.topk_softmax(\n"
             "            topk_func, topk_weights, topk_ids, token_expert_indices, gating_output,\n"
             "            renormalize, _get_padding_mask(topk_ids.shape[0]),\n"
             "        )\n"),
        ),
    ),
}
MAMBA_COUNT_FLOOR_ENV = "TAAF_VLLM_MAMBA_COUNT_FLOOR"
MAMBA_COUNT_RELATIVE = "vllm/v1/core/single_type_kv_cache_manager.py"
MAMBA_COUNT_PINNED = "0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf"
MAMBA_COUNT_EDITS = (
    ("            num_new_blocks = (\n"
     "                num_required_blocks\n"
     "                - len(new_computed_blocks)\n"
     "                - len(self.req_to_blocks[request_id])\n"
     "            )\n",
     "            num_new_blocks = max(\n"
     "                num_required_blocks\n"
     "                - len(new_computed_blocks)\n"
     "                - len(self.req_to_blocks[request_id]),\n"
     "                0,\n"
     "            )\n"),
)
ALLOC_GUARD_ENV = "TAAF_VLLM_ALLOC_GUARD"
ALLOC_GUARD_RELATIVE = "vllm/v1/core/kv_cache_manager.py"
ALLOC_GUARD_PINNED = "58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa"
ALLOC_GUARD_EDITS = (
    ("        if (\n"
     "            new_computed_block_list is not self.empty_kv_cache_blocks.blocks\n"
     "            or num_external_computed_tokens > 0\n"
     "        ):\n"
     "            # Append the new computed blocks to the request blocks until now to\n"
     "            # avoid the case where the new blocks cannot be allocated.\n"
     "            self.coordinator.allocate_new_computed_blocks(\n"
     "                request_id=request.request_id,\n"
     "                new_computed_blocks=new_computed_block_list,\n"
     "                num_local_computed_tokens=num_local_computed_tokens,\n"
     "                num_external_computed_tokens=num_external_computed_tokens,\n"
     "            )\n"
     "\n"
     "        new_blocks = self.coordinator.allocate_new_blocks(\n"
     "            request.request_id,\n"
     "            num_tokens_need_slot,\n"
     "            num_tokens_main_model,\n"
     "            num_encoder_tokens,\n"
     "        )\n",
     "        free_before = self.block_pool.get_num_free_blocks()\n"
     "        try:\n"
     "            if (\n"
     "                new_computed_block_list is not self.empty_kv_cache_blocks.blocks\n"
     "                or num_external_computed_tokens > 0\n"
     "            ):\n"
     "                self.coordinator.allocate_new_computed_blocks(\n"
     "                    request_id=request.request_id,\n"
     "                    new_computed_blocks=new_computed_block_list,\n"
     "                    num_local_computed_tokens=num_local_computed_tokens,\n"
     "                    num_external_computed_tokens=num_external_computed_tokens,\n"
     "                )\n"
     "\n"
     "            new_blocks = self.coordinator.allocate_new_blocks(\n"
     "                request.request_id,\n"
     "                num_tokens_need_slot,\n"
     "                num_tokens_main_model,\n"
     "                num_encoder_tokens,\n"
     "            )\n"
     "        except ValueError as failure:\n"
     "            return self._refuse_short_allocation(\n"
     "                request, num_blocks_to_allocate, free_before, failure\n"
     "            )\n"),
    ("    def remove_skipped_blocks(\n",
     "    def _refuse_short_allocation(\n"
     "        self,\n"
     "        request: Request,\n"
     "        counted: int,\n"
     "        free_before: int,\n"
     "        failure: ValueError,\n"
     "    ) -> None:\n"
     "        if not str(failure).startswith(\"Cannot get\"):\n"
     "            raise failure\n"
     "        taken = free_before - self.block_pool.get_num_free_blocks()\n"
     "        held: dict[str, int] = {}\n"
     "        for manager in self.coordinator.single_type_managers:\n"
     "            blocks = manager.req_to_blocks.get(request.request_id)\n"
     "            if blocks:\n"
     "                name = type(manager).__name__\n"
     "                held[name] = held.get(name, 0) + sum(\n"
     "                    1 for block in blocks if not block.is_null\n"
     "                )\n"
     "        fresh = request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)\n"
     "        self.taaf_short_allocations = getattr(self, \"taaf_short_allocations\", 0) + 1\n"
     "        if self.taaf_short_allocations <= 20 or self.taaf_short_allocations % 100 == 0:\n"
     "            logger.warning(\n"
     "                \"ALLOC_GUARD #%d: %s (%s) counted %d block(s) against %d free and \"\n"
     "                \"took %d before the pool ran out (%s); blocks held by group kind %s; \"\n"
     "                \"%s\",\n"
     "                self.taaf_short_allocations,\n"
     "                request.request_id,\n"
     "                request.status.name,\n"
     "                counted,\n"
     "                free_before,\n"
     "                taken,\n"
     "                failure,\n"
     "                held,\n"
     "                \"the request is released and waits\"\n"
     "                if fresh\n"
     "                else \"the request keeps what it took and the scheduler preempts\",\n"
     "            )\n"
     "        if fresh:\n"
     "            self.free(request)\n"
     "        return None\n"
     "\n"
     "    def remove_skipped_blocks(\n"),
)
BLOCK_COPY_CHUNK_ENV = "TAAF_VLLM_BLOCK_COPY_CHUNK_MIB"
BLOCK_COPY_RELATIVE = "vllm/v1/worker/utils.py"
BLOCK_COPY_PINNED = "0ca3ec6bf4d20076145b7fe64e564962adebedc8f75a856b517ba0b2f2d77acf"
BLOCK_COPY_LIMIT_MARK = "__TAAF_BLOCK_COPY_LIMIT__"
BLOCK_COPY_EDITS = (
    ("    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)\n",
     "    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)\n"
     "    taaf_overlap = bool(np.intersect1d(indices_np[:, 0], indices_np[:, 1]).size)\n"),
    ("        blocks[dst] = blocks[src]\n",
     "        taaf_rows = max(1, __TAAF_BLOCK_COPY_LIMIT__ // max(1, blocks[0].numel() * blocks.element_size()))\n"
     "        if taaf_overlap or taaf_rows >= len(src):\n"
     "            blocks[dst] = blocks[src]\n"
     "        else:\n"
     "            for taaf_at in range(0, len(src), taaf_rows):\n"
     "                taaf_to = taaf_at + taaf_rows\n"
     "                blocks[dst[taaf_at:taaf_to]] = blocks[src[taaf_at:taaf_to]]\n"),
)
PLE_GROUP_ENV = "TAAF_VLLM_PLE_PREFILL_GROUPS"
PLE_LAYER_RELATIVE = "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
PLE_GROUP_CLASS_ANCHOR = "class "
PLE_GROUP_CAP_ENV = "TAAF_VLLM_PLE_GROUP_CAP"
PLE_GROUP_CAP_DEFAULT = 8192
SCHED_IMPORT_ANCHOR = 'from vllm.v1.core.encoder_cache_manager import ('
SCHED_IMPORT_LINE = 'from vllm.utils.math_utils import cdiv\n'
SCHED_RESERVE_FROM = '                    # only if it fits in (free - other in-flight reservations), to\n                    # avoid deadlock and predictable preemptions.\n                    reserved_blocks = self._inflight_prefill_reserved_blocks()\n'
SCHED_RESERVE_TO = '                    # only if it fits in (free - other in-flight reservations)\n                    # plus its own spec decode step blocks, to avoid deadlock and\n                    # predictable preemptions.\n                    reserved_blocks = (\n                        self._inflight_prefill_reserved_blocks()\n                        + self._spec_decode_step_blocks()\n                    )\n'
SCHED_REMAINING_FROM = '        """Blocks `request` still needs to allocate to hold its full sequence."""\n        full_num_tokens = min(request.num_tokens, self.max_model_len)\n        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate('
SCHED_REMAINING_TO = '        """Blocks `request` still needs to hold its full sequence and be promoted."""\n        full_num_tokens = min(request.num_tokens, self.max_model_len)\n        num_blocks = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate('
SCHED_HELPER_FROM = '            apply_admission_cap=True,\n        )\n\n    def _inflight_prefill_reserved_blocks(self) -> int:'
SCHED_HELPER_TO = '            apply_admission_cap=True,\n        )\n        return num_blocks + self._spec_decode_step_blocks()\n\n    def _spec_decode_step_blocks(self) -> int:\n        """Number of blocks for the extra KV slots a spec decode step needs.\n\n        When using async kv load, scheduler must reserve enough blocks for\n        full sequence + the spec decode step, otherwise request cannot be\n        able to run after the async_load if we are out of kv blocks.\n        """\n        if not self.num_spec_tokens:\n            return 0\n        return cdiv(\n            1 + self.num_spec_tokens + self.num_lookahead_tokens, self.block_size\n        )\n\n    def _inflight_prefill_reserved_blocks(self) -> int:'
COUNT_EXPERTS_PATH_ENV = "TAAF_VLLM_COUNT_EXPERTS_PATH"
COUNTER_NAME = "expert_counter.py"
ROUTED_EXPERTS_RELATIVE = (
    "vllm/model_executor/layers/fused_moe/routed_experts.py")
EXPERT_ORDER_NAME = "expert_saliency.json"
EXPERT_ORDER_ENV = "TAAF_VLLM_EXPERT_ORDER"
EXPERT_ORDER_FILES = {"saliency": EXPERT_ORDER_NAME, "usage": "expert_usage.json"}
MTP_INDEX_SHARE_FOR_ITERATION_ENV = (
    "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"
)
MTP_DYNAMIC_BATCH_SCHEDULE_ENV = "TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE"
DEFAULT_OMP_THREADS = 4
DEFAULT_MTP_TOKENS = 3
MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES = 96 * 1024**3
MAX_KV_OFFLOADING_GIB = 120
MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES = 64 * 1024**2
DEFAULT_GPU_MEMORY_PERMILLE = 920
MIN_GPU_MEMORY_PERMILLE = 500
MAX_GPU_MEMORY_PERMILLE = 995
VLLM_TUNING_FIELDS = {
    "schema_version",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "kv_cache_dtype",
    "kv_cache_memory_bytes",
    "gpu_memory_permille",
    "kv_offloading_gib",
    "max_cudagraph_capture_size",
    "max_num_batched_tokens",
    "max_num_batched_tokens_requested",
    "max_num_seqs",
    "moe_backend",
    "mtp_dynamic_batch_schedule",
    "mtp_index_share_for_iteration",
    "mtp_speculative_tokens",
    "omp_num_threads",
    "overridden",
}
VLLM_TUNING_OVERRIDE_FIELDS = {
    MAX_NUM_SEQS_ENV,
    MAX_NUM_BATCHED_TOKENS_ENV,
    OMP_THREADS_ENV,
    MTP_TOKENS_ENV,
    KV_CACHE_MEMORY_BYTES_ENV,
    GPU_MEMORY_PERMILLE_ENV,
    KV_CACHE_DTYPE_ENV,
    ENABLE_PREFIX_CACHING_ENV,
    KV_OFFLOADING_GIB_ENV,
    MAX_CUDAGRAPH_CAPTURE_SIZE_ENV,
    MOE_BACKEND_ENV,
    MTP_INDEX_SHARE_FOR_ITERATION_ENV,
    MTP_DYNAMIC_BATCH_SCHEDULE_ENV,
}
MIN_HOST_AVAILABLE_BYTES = 64 * 1024**3
MIN_GPU_FREE_MIB = 4_096
MIN_RUNTIME_FILESYSTEM_FREE_BYTES = 36_507_222_016
RUNTIME_EXTRACTED_BOUND_BYTES = 18_359_349_248
RUNTIME_CACHE_RESERVE_BYTES = 16 * 1024**3
RUNTIME_STORAGE_AUDIT_SHA256 = (
    "5791fbfae9b53d4b8f4000bab769f13e1b1d78c763e889663199e3ee76f5252c"
)
DRAFT_CONF_LOG_ENV = "TAAF_VLLM_DRAFT_CONF_LOG"
DRAFT_CONF_LOG_HELPER = "draft_conf_log.py"
DRAFT_CONF_LOG_INSTALLED = "vllm/taaf_draft_conf_log.py"
DRAFT_CONF_LOG_RELATIVE = "vllm/v1/worker/gpu/spec_decode/speculator.py"
DRAFT_SUMMIN_ENV = "TAAF_VLLM_DRAFT_SUMMIN"
DRAFT_SUMMIN_HELPER = "draft_summin.py"
DRAFT_SUMMIN_INSTALLED = "vllm/taaf_draft_summin.py"
DRAFT_SAMPLE_HELPER = "draft_sample.py"
DRAFT_SAMPLE_INSTALLED = "vllm/taaf_draft_sample.py"
DRAFT_CONF_LOG_PINNED = "3a1ab15bf8b20918852eb835fe9755046e99dcbac6ba6ca3d414949fc8bc695a"
DRAFT_CONF_LOG_EDITS = (
    ('from abc import ABC, abstractmethod\n'
     'from collections.abc import Mapping\n'
     'from dataclasses import replace\n'
     'from typing import TYPE_CHECKING, Any\n'
     '\n'
     'import numpy as np\n'
     'import torch\n'
     'import torch.nn as nn\n',
     'from abc import ABC, abstractmethod\n'
     'from collections.abc import Mapping\n'
     'from dataclasses import replace\n'
     'from typing import TYPE_CHECKING, Any\n'
     '\n'
     'import os\n'
     '\n'
     'import numpy as np\n'
     'import torch\n'
     'import torch.nn as nn\n'),
    ('logger = init_logger(__name__)\n'
     '\n'
     '\n'
     'def _target_feeds_hc_residual(vllm_config: VllmConfig) -> bool:\n',
     'logger = init_logger(__name__)\n'
     '\n'
     '_TAAF_DRAFT_CONF_LOG = os.environ.get("TAAF_VLLM_DRAFT_CONF_LOG") == "1"\n'
     '_TAAF_DRAFT_SUMMIN = os.environ.get("TAAF_VLLM_DRAFT_SUMMIN") == "1"\n'
     '_TAAF_DRAFT_SAMPLE = os.environ.get("TAAF_VLLM_DRAFT_SAMPLE") == "1"\n'
     '\n'
     '\n'
     'def _target_feeds_hc_residual(vllm_config: VllmConfig) -> bool:\n'),
    ('        self.use_acceptance_estimator = self.enable_adaptive_verification\n'
     '        self.acceptance_estimator: OnlineAcceptanceEstimator | None = None\n'
     '\n'
     '        self.draft_logits: torch.Tensor | None = None\n',
     '        self.use_acceptance_estimator = self.enable_adaptive_verification\n'
     '        self.acceptance_estimator: OnlineAcceptanceEstimator | None = None\n'
     '        self._taaf_budget_conf = None\n'
     '        if os.environ.get("TAAF_VLLM_DRAFT_BUDGET", "") not in ("", "0"):\n'
     '            self.use_acceptance_estimator = False\n'
     '            from vllm.taaf_draft_budget import arm_speculator\n'
     '\n'
     '            arm_speculator(self, device)\n'
     '        self._taaf_conf_log = None\n'
     '        if _TAAF_DRAFT_CONF_LOG:\n'
     '            from vllm.taaf_draft_conf_log import arm\n'
     '\n'
     '            self._taaf_conf_log = arm(self.max_num_reqs, self.num_speculative_steps, device)\n'
     '        self._taaf_summin = None\n'
     '        if _TAAF_DRAFT_SUMMIN:\n'
     '            from vllm.taaf_draft_summin import arm as _taaf_summin_arm\n'
     '\n'
     '            self._taaf_summin = _taaf_summin_arm(\n'
     '                self.max_num_reqs, self.num_speculative_steps, device\n'
     '            )\n'
     '\n'
     '        self.draft_logits: torch.Tensor | None = None\n'),
    ('        elif self.use_local_argmax_reduction:\n'
     '            return self.model.get_top_tokens(hidden_states)\n'
     '        else:\n'
     '            logits = self.model.compute_logits(hidden_states)\n'
     '            sampled = logits.argmax(dim=-1)\n'
     '        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)\n'
     '        return sampled\n',
     '        elif self.use_local_argmax_reduction:\n'
     '            if (self._taaf_conf_log is None and self.acceptance_estimator is None\n'
     '                    and self._taaf_summin is None):\n'
     '                return self.model.get_top_tokens(hidden_states)\n'
     '            from vllm.taaf_draft_vocab import top_tokens_and_logits\n'
     '\n'
     '            sampled, logits = top_tokens_and_logits(self.model, hidden_states)\n'
     '        else:\n'
     '            logits = self.model.compute_logits(hidden_states)\n'
     '            sampled = logits.argmax(dim=-1)\n'
     '        if self._taaf_summin is not None:\n'
     '            from vllm.taaf_draft_summin import log_q as _taaf_summin_log_q\n'
     '\n'
     '            _taaf_summin_log_q(\n'
     '                self._taaf_summin, draft_step, logits, idx_mapping,\n'
     '                self.model,\n'
     '            )\n'
     '        if self._taaf_conf_log is not None:\n'
     '            from vllm.taaf_draft_conf_log import log_proposed\n'
     '\n'
     '            log_proposed(\n'
     '                self._taaf_conf_log, draft_step, logits, None, idx_mapping, temperature\n'
     '            )\n'
     '        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)\n'
     '        return sampled\n'),
    ('        if self.use_acceptance_estimator:\n'
     '            if self.use_local_argmax_reduction:\n'
     '                raise ValueError(\n'
     '                    "Adaptive verification without a confidence head estimates "\n'
     '                    "per-position acceptance from the draft logits, which "\n'
     '                    "use_local_argmax_reduction never materializes. Disable one "\n'
     '                    "of them."\n'
     '                )\n',
     '        if self.use_acceptance_estimator:\n'),
    ('    def observe_verification(\n'
     '        self,\n'
     '        idx_mapping: torch.Tensor,\n'
     '        num_sampled: torch.Tensor,\n'
     '        num_rejected: torch.Tensor,\n'
     '    ) -> None:\n'
     '        """Fold the target\'s verdict on the last drafts into the estimator.\n'
     '\n'
     '        Must run before the next `propose`, which overwrites the per-slot\n'
     '        features the verdict grades.\n'
     '        """\n'
     '        if self.acceptance_estimator is not None:\n'
     '            self.acceptance_estimator.step(idx_mapping, num_sampled, num_rejected)\n',
     '    def observe_verification(\n'
     '        self,\n'
     '        idx_mapping: torch.Tensor,\n'
     '        num_sampled: torch.Tensor,\n'
     '        num_rejected: torch.Tensor,\n'
     '    ) -> None:\n'
     '        """Fold the target\'s verdict on the last drafts into the estimator.\n'
     '\n'
     '        Must run before the next `propose`, which overwrites the per-slot\n'
     '        features the verdict grades.\n'
     '        """\n'
     '        if self._taaf_conf_log is not None:\n'
     '            from vllm.taaf_draft_conf_log import log_verified\n'
     '\n'
     '            log_verified(self._taaf_conf_log, idx_mapping, num_sampled, num_rejected)\n'
     '        if self._taaf_summin is not None:\n'
     '            from vllm.taaf_draft_summin import fold as _taaf_summin_fold\n'
     '\n'
     '            _taaf_summin_fold(self._taaf_summin, idx_mapping, num_sampled, num_rejected)\n'
     '        if self.acceptance_estimator is not None:\n'
     '            self.acceptance_estimator.step(idx_mapping, num_sampled, num_rejected)\n'),
)
DRAFT_CONF_LOG_EDITS += (
    ('        if draft_logits is not None:\n'
     '            logits = self.model.compute_logits(hidden_states)\n',
     '        if draft_logits is not None and _TAAF_DRAFT_SAMPLE:\n'
     '            from vllm.taaf_draft_sample import sample as _taaf_draft_sample\n'
     '\n'
     '            return _taaf_draft_sample(\n'
     '                self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,\n'
     '                draft_step, draft_logits,\n'
     '            )\n'
     '        if draft_logits is not None:\n'
     '            logits = self.model.compute_logits(hidden_states)\n'),
)
DRAFT_CONF_LOG_TARGETS = {DRAFT_CONF_LOG_RELATIVE: (DRAFT_CONF_LOG_PINNED, DRAFT_CONF_LOG_EDITS)}
RELAX_ACCEPT_ENV = "TAAF_VLLM_RELAX_ACCEPT"
RELAX_ACCEPT_RELATIVE = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
RELAX_ACCEPT_PINNED = "214b61fa0abe88573beb779872c035b3954e5651a85bd438d99ee5905d3a62ec"
RELAX_ACCEPT_EDITS = (
    ('import torch\n'
     '\n'
     'from vllm.triton_utils import tl, tldevice, triton\n',
     'import math\n'
     'import os\n'
     '\n'
     'import torch\n'
     '\n'
     'from vllm.triton_utils import tl, tldevice, triton\n'),
    ('from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n',
     'from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n'
     '\n'
     '_TAAF_RELAX = os.environ.get("TAAF_VLLM_RELAX_ACCEPT", "").strip()\n'
     '_TAAF_RELAX_ON = tl.constexpr(_TAAF_RELAX not in ("", "0"))\n'
     '_TAAF_RELAX_LOG = tl.constexpr(math.log(float(_TAAF_RELAX)) if _TAAF_RELAX not in ("", "0") else 0.0)\n'),
    ('                    # Probability ratio test: p(x) > u * q(x)\n'
     '                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n'
     '                    accepted = target_logprob > tl.log(u) + draft_logprob\n',
     '                    # Probability ratio test: p(x) > u * q(x)\n'
     '                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n'
     '                    accepted = target_logprob > tl.log(u) + draft_logprob\n'
     '                    if _TAAF_RELAX_ON:\n'
     '                        accepted = accepted | (target_logprob >= _TAAF_RELAX_LOG)\n'),
)
RELAX_ACCEPT_TARGETS = {RELAX_ACCEPT_RELATIVE: (RELAX_ACCEPT_PINNED, RELAX_ACCEPT_EDITS)}
PAUSED_ENV = "TAAF_VLLM_PAUSED"
PAUSED_HELPER = "paused.py"
PAUSED_RESUME_ENV = "TAAF_VLLM_PAUSED_RESUME_UNSTOPPED"
PAUSED_MEMORY_ENV = "TAAF_VLLM_PAUSED_PRESSURE_MEMORY_ONLY"
PAUSED_BRANCH_ENV = "TAAF_VLLM_PAUSED_BRANCH"
PAUSED_VIEW_ENV = "TAAF_VLLM_PAUSED_CLIENT_VIEW"
PAUSED_POOL_ENV = "TAAF_VLLM_PAUSED_POOL"
PAUSED_MM_DROP_ENV = "TAAF_VLLM_PAUSED_MM_DROP"
PAUSED_MEMTRACE_ENV = "TAAF_VLLM_MEMTRACE"
PAUSED_INSTALLED = "vllm/taaf_paused.py"
PAUSED_FLAG = (
    'import os as _taaf_paused_os\n'
    '_TAAF_PAUSED_ON = _taaf_paused_os.environ.get("TAAF_VLLM_PAUSED", "").strip() not in ("", "0")\n')
PAUSED_ASYNC_EDITS = (
    ('    async def get_weight_version(self) -> str:\n'
     '        """Return the latest committed weight version."""\n'
     '        return await self.engine_core.get_weight_version_async()\n',
     '    async def get_weight_version(self) -> str:\n'
     '        """Return the latest committed weight version."""\n'
     '        return await self.engine_core.get_weight_version_async()\n'
     '\n'
     '\n'
     + PAUSED_FLAG +
     'if _TAAF_PAUSED_ON:\n'
     '    from vllm.taaf_paused import install as _taaf_paused_install\n'
     '    _taaf_paused_install(AsyncLLM)\n'),
)
PAUSED_MEMORY_FLAG = (
    '_TAAF_PAUSED_MEMORY = _TAAF_PAUSED_ON and _taaf_paused_os.environ.get(\n'
    '    "TAAF_VLLM_PAUSED_PRESSURE_MEMORY_ONLY", ""\n'
    ').strip() not in ("", "0")\n')
PAUSED_SEATS_FLAG = (
    '_TAAF_PAUSED_SEATS = _TAAF_PAUSED_ON and _taaf_paused_os.environ.get(\n'
    '    "TAAF_VLLM_PAUSED_SEAT_PRESSURE", "1"\n'
    ').strip().lower() not in ("0", "false", "no", "off")\n')
PAUSED_POOL_FLAG = (
    '_TAAF_PAUSED_POOL = _TAAF_PAUSED_ON and _taaf_paused_os.environ.get(\n'
    '    "TAAF_VLLM_PAUSED_POOL", ""\n'
    ').strip() not in ("", "0")\n')
PAUSED_MM_DROP_FLAG = (
    '_TAAF_PAUSED_MMDROP = _TAAF_PAUSED_ON and _taaf_paused_os.environ.get(\n'
    '    "TAAF_VLLM_PAUSED_MM_DROP", ""\n'
    ').strip() not in ("", "0")\n')
PAUSED_MEMTRACE_FLAG = (
    '_TAAF_PAUSED_MEMTRACE = _TAAF_PAUSED_ON and _taaf_paused_os.environ.get(\n'
    '    "TAAF_VLLM_MEMTRACE", ""\n'
    ').strip() not in ("", "0")\n')
PAUSED_LEAN_HOOK = (
    'if _TAAF_PAUSED_ON:\n'
    '    from vllm.taaf_paused import lean_engine as _taaf_lean_engine\n'
    '    _taaf_lean_engine()\n')
PAUSED_SCHEDULER_EDITS = (
    ('from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n',
     'from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n' + PAUSED_FLAG + PAUSED_MEMORY_FLAG
     + PAUSED_SEATS_FLAG + PAUSED_POOL_FLAG + PAUSED_MM_DROP_FLAG + PAUSED_MEMTRACE_FLAG + PAUSED_LEAN_HOOK),
    ('                num_running = len(self.running) + self.num_waiting_for_streaming_input\n'
     '                if num_running >= self.max_num_running_reqs:\n'
     '                    break\n',
     '                num_running = len(self.running) + self.num_waiting_for_streaming_input\n'
     '                if num_running >= self.max_num_running_reqs:\n'
     '                    if _TAAF_PAUSED_SEATS:\n'
     '                        from vllm.taaf_paused import seats_blocked as _taaf_seats_blocked\n'
     '                        _taaf_seats_blocked(self)\n'
     '                    break\n'),
    ('        error_req_ids = set(self.grammar_compile_error_reqs)\n',
     '        if _TAAF_PAUSED_MEMTRACE:\n'
     '            from vllm.taaf_paused import memtrace_engine as _taaf_memtrace_engine\n'
     '            _taaf_memtrace_engine(self)\n'
     '        error_req_ids = set(self.grammar_compile_error_reqs)\n'),
    ('        if update.mm_features:\n',
     '        if _TAAF_PAUSED_MMDROP:\n'
     '            from vllm.taaf_paused import drop_covered as _taaf_drop_covered\n'
     '            _taaf_drop_covered(session)\n'
     '        if update.mm_features:\n'),
    ('            # If no encoder input chunking is allowed, we do not want to\n',
     '            if (\n'
     '                _TAAF_PAUSED_MMDROP\n'
     '                and (mm_feature.data is None or getattr(mm_feature, "_taaf_stripped", False))\n'
     '                and getattr(request, "_taaf_mm_dropped", False)\n'
     '            ):\n'
     '                from vllm.taaf_paused import mm_lost as _taaf_mm_lost\n'
     '                _taaf_mm_lost(self, request)\n'
     '                num_new_tokens = max(\n'
     '                    0, start_pos - num_computed_tokens - shift_computed_tokens\n'
     '                )\n'
     '                break\n'
     '            # If no encoder input chunking is allowed, we do not want to\n'),
    ('        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:\n',
     '        if _TAAF_PAUSED_MMDROP:\n'
     '            from vllm.taaf_paused import take_lost as _taaf_take_lost\n'
     '            error_req_ids.update(_taaf_take_lost(self))\n'
     '        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:\n'),
    ('                    if request.has_encoder_inputs:\n'
     '                        self.encoder_cache_manager.free(request)\n'
     '                    break\n',
     '                    if request.has_encoder_inputs:\n'
     '                        self.encoder_cache_manager.free(request)\n'
     '                    if _TAAF_PAUSED_MEMORY:\n'
     '                        from vllm.taaf_paused import memory_blocked as _taaf_memory_blocked\n'
     '                        _taaf_memory_blocked(self)\n'
     '                    break\n'),
    ('            self._update_after_schedule(scheduler_output)\n'
     '        return scheduler_output\n',
     '            self._update_after_schedule(scheduler_output)\n'
     '        if _TAAF_PAUSED_MEMORY:\n'
     '            from vllm.taaf_paused import publish_memory as _taaf_publish_memory\n'
     '            _taaf_publish_memory(self)\n'
     '        if _TAAF_PAUSED_SEATS:\n'
     '            from vllm.taaf_paused import publish_seats as _taaf_publish_seats\n'
     '            _taaf_publish_seats(self)\n'
     '        if _TAAF_PAUSED_POOL:\n'
     '            from vllm.taaf_paused import publish_pool as _taaf_publish_pool\n'
     '            _taaf_publish_pool(self)\n'
     '        return scheduler_output\n'),
    ('        num_computed_tokens = session.num_computed_tokens\n'
     '        kept_output_tokens = session._all_token_ids[\n'
     '            session.num_prompt_tokens : num_computed_tokens\n'
     '        ]\n',
     '        _taaf_refeed = []\n'
     '        if _TAAF_PAUSED_ON and update.sampling_params is not None and (\n'
     '            update.sampling_params.extra_args or {}\n'
     '        ).get("taaf_paused"):\n'
     '            from vllm.taaf_paused import refeed as _taaf_refeed_fn\n'
     '            _taaf_refeed = _taaf_refeed_fn(\n'
     '                session, (), bool(update.sampling_params.extra_args.get("taaf_drop_last"))\n'
     '            )\n'
     '        num_computed_tokens = session.num_computed_tokens\n'
     '        kept_output_tokens = session._all_token_ids[\n'
     '            session.num_prompt_tokens : num_computed_tokens\n'
     '        ]\n'),
    ('            base = session.num_tokens\n',
     '            base = session.num_tokens + len(_taaf_refeed)\n'),
    ('    def _handle_stopped_request(self, request: Request) -> bool:\n'
     '        """Return True if finished (can be False for resumable requests)."""\n'
     '        if not request.resumable:\n'
     '            return True\n',
     '    def _handle_stopped_request(self, request: Request) -> bool:\n'
     '        """Return True if finished (can be False for resumable requests)."""\n'
     '        if not request.resumable:\n'
     '            return True\n'
     '        if _TAAF_PAUSED_ON and request.sampling_params is not None and (\n'
     '            request.sampling_params.extra_args or {}\n'
     '        ).get("taaf_paused"):\n'
     '            from vllm.taaf_paused import settle as _taaf_settle\n'
     '            _taaf_settle(request)\n'),
    ('        session._all_token_ids.extend(update.prompt_token_ids or ())\n'
     '        session.prompt_token_ids.extend(update.prompt_token_ids or ())\n',
     '        _taaf_update_ids = list(_taaf_refeed) + list(update.prompt_token_ids or ())\n'
     '        session._all_token_ids.extend(_taaf_update_ids)\n'
     '        session.prompt_token_ids.extend(_taaf_update_ids)\n'),
    ('        session.sampling_params = update.sampling_params\n',
     '        session.sampling_params = update.sampling_params\n'
     '        session.max_tokens = update.max_tokens\n'),
)
PAUSED_SAMPLER_EDITS = (
    ('from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (\n'
     '    rejection_sample,\n'
     ')\n',
     'from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (\n'
     '    rejection_sample,\n'
     ')\n'
     + PAUSED_FLAG),
    ('        num_sampled, num_rejected = get_num_sampled_and_rejected(\n'
     '            num_sampled,\n',
     '        if _TAAF_PAUSED_ON:\n'
     '            from vllm.taaf_paused import clip_at_stops, stops_tensor\n'
     '            num_sampled = clip_at_stops(\n'
     '                num_sampled,\n'
     '                draft_sampled,\n'
     '                input_batch.cu_num_logits,\n'
     '                stops_tensor(draft_sampled.device, draft_sampled.dtype),\n'
     '            )\n'
     '            from vllm.taaf_paused import clip_at_length, resume_unstopped\n'
     '            if resume_unstopped():\n'
     '                num_sampled = clip_at_length(\n'
     '                    num_sampled,\n'
     '                    pos,\n'
     '                    input_batch.cu_num_logits,\n'
     '                    self.sampler.req_states.max_seq_len,\n'
     '                    input_batch.idx_mapping_np,\n'
     '                )\n'
     '        num_sampled, num_rejected = get_num_sampled_and_rejected(\n'
     '            num_sampled,\n'),
)
PAUSED_TARGETS = {
    "vllm/v1/engine/async_llm.py":
        (None, PAUSED_ASYNC_EDITS),
    "vllm/v1/core/sched/scheduler.py":
        (None, PAUSED_SCHEDULER_EDITS),
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py":
        (None, PAUSED_SAMPLER_EDITS),
}
FORK_BRANCH_ENV = "TAAF_VLLM_FORK_BRANCH"
FORK_MAX_ENV = "TAAF_VLLM_FORK_MAX"
FORK_HOLD_ENV = "TAAF_VLLM_FORK_HOLD_S"
FORK_SEATS_ENV = "TAAF_VLLM_FORK_SEATS"
FORK_LIVE_ENV = "TAAF_VLLM_FORK_LIVE"
FORK_MAX_DEFAULT = 24
FORK_MAX_LIMIT = 1024
FORK_SCHEDULER_RELATIVE = "vllm/v1/core/sched/scheduler.py"
FORK_FLAG = (
    'import os as _taaf_fork_os\n'
    '_TAAF_FORK_BRANCH = _taaf_fork_os.environ.get("TAAF_VLLM_FORK_BRANCH", "").strip() not in ("", "0")\n')
FORK_SCHEDULER_EDITS = (
    ('from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n',
     FORK_FLAG + 'from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n'),
    ('        self.kv_cache_manager.new_step_starts()\n',
     '        if _TAAF_FORK_BRANCH:\n'
     '            from vllm.taaf_paused import fork_hold as _taaf_fork_hold\n'
     '            _taaf_fork_hold(self)\n'
     '        self.kv_cache_manager.new_step_starts()\n'),
    ('        with record_function_or_nullcontext("schedule: update_after_schedule"):\n',
     '        if _TAAF_FORK_BRANCH:\n'
     '            from vllm.taaf_paused import fork_release as _taaf_fork_release\n'
     '            _taaf_fork_release(self)\n'
     '        with record_function_or_nullcontext("schedule: update_after_schedule"):\n'),
)
REWIND_ENV = "TAAF_VLLM_PAUSED_REWIND"
REWIND_SCHEDULER_EDITS = (
    ('        self.num_waiting_for_streaming_input: int = 0\n',
     '        self.num_waiting_for_streaming_input: int = 0\n'
     '        import os as _taaf_rewind_os\n'
     '        if _taaf_rewind_os.environ.get("TAAF_VLLM_PAUSED_REWIND", "").strip() not in ("", "0"):\n'
     '            from vllm.taaf_paused import rewind_attach as _taaf_rewind_attach\n'
     '            _taaf_rewind_attach(self)\n'),
)
PARK_ENV = "TAAF_VLLM_PARK"
PARK_RAM_ENV = "TAAF_VLLM_PARK_RAM_GIB"
PARK_DISK_ENV = "TAAF_VLLM_PARK_DISK"
PARK_HOLD_ENV = "TAAF_VLLM_PARK_HOLD_S"
PARK_WAIT_ENV = "TAAF_VLLM_PARK_WAIT_S"
PARK_VERIFY_ENV = "TAAF_VLLM_PARK_VERIFY"
PARK_HOST_DISK_ENV = "TAAF_VLLM_PARK_HOST_DISK"
PARK_HELPER = "park.py"
PARK_INSTALLED = "vllm/taaf_park.py"
PARK_DEFAULT_RAM_GIB = 8
PARK_MAX_RAM_GIB = 256
PARK_DEFAULT_HOLD_S = 600
PARK_MAX_HOLD_S = 86400
PARK_DEFAULT_WAIT_S = 120
PARK_MAX_WAIT_S = 3600
PARK_FLAG = (
    'import os as _taaf_park_os\n'
    '_TAAF_PARK_ON = _taaf_park_os.environ.get("TAAF_VLLM_PARK", "").strip() not in ("", "0")\n'
    'if _TAAF_PARK_ON:\n'
    '    from vllm import taaf_park as _taaf_park\n')
PARK_SCHEDULER_EDITS = (
    ('logger = init_logger(__name__)\n',
     'logger = init_logger(__name__)\n' + PARK_FLAG),
    ('        self.current_step = 0\n',
     '        self.current_step = 0\n'
     '        if _TAAF_PARK_ON:\n'
     '            _taaf_park.attach(self)\n'),
    ('    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n'
     '        self.current_step += 1\n',
     '    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n'
     '        self.current_step += 1\n'
     '        if _TAAF_PARK_ON:\n'
     '            _taaf_park.begin(self)\n'),
)
PARK_RUNNER_EDITS = (
    ('logger = init_logger(__name__)\n',
     'logger = init_logger(__name__)\n' + PARK_FLAG),
    ('            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n',
     '            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n'
     '            if _TAAF_PARK_ON:\n'
     '                _taaf_park.worker_init(self)\n'),
)
PARK_TARGETS = {
    "vllm/v1/core/sched/scheduler.py":
        (None, PARK_SCHEDULER_EDITS),
    "vllm/v1/worker/gpu/model_runner.py":
        (None, PARK_RUNNER_EDITS),
}
PREFILL_WIP_ENV = "TAAF_VLLM_PREFILL_WIP"
PREFILL_WIP_HELPER = "prefill_wip.py"
PREFILL_WIP_INSTALLED = "vllm/taaf_prefill_wip.py"
PREFILL_WIP_RELATIVE = "vllm/v1/core/sched/scheduler.py"
PREFILL_WIP_FLAG = (
    'import os as _taaf_wip_os\n'
    '_TAAF_WIP_ON = _taaf_wip_os.environ.get("TAAF_VLLM_PREFILL_WIP", "").strip() not in ("", "0")\n'
    'if _TAAF_WIP_ON:\n'
    '    from vllm import taaf_prefill_wip as _taaf_wip\n')
PREFILL_WIP_EDITS = (
    ('from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n',
     PREFILL_WIP_FLAG + 'from vllm.v1.request import Request, RequestStatus, StreamingUpdate\n'),
    ('        self.kv_cache_manager.new_step_starts()\n',
     '        if _TAAF_WIP_ON:\n'
     '            _taaf_wip.begin()\n'
     '        self.kv_cache_manager.new_step_starts()\n'),
    ('                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)\n',
     '                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)\n'
     '            if _TAAF_WIP_ON:\n'
     '                max_prefill_tokens = min(max_prefill_tokens, _taaf_wip.step_tokens())\n'),
    ('            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n'
     '                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n',
     '            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n'
     '                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n'
     '            if _TAAF_WIP_ON and request.num_tokens - request.num_computed_tokens > 1:\n'
     '                num_new_tokens = _taaf_wip.grant(num_new_tokens)\n'
     '                if num_new_tokens == 0:\n'
     '                    req_index += 1\n'
     '                    continue\n'),
    ('                    num_new_tokens = min(num_new_tokens, request_token_budget)\n'
     '                    assert num_new_tokens > 0\n',
     '                    num_new_tokens = min(num_new_tokens, request_token_budget)\n'
     '                    assert num_new_tokens > 0\n'
     '                    if _TAAF_WIP_ON and not pad_spec_decode and num_new_tokens > 1:\n'
     '                        num_new_tokens = _taaf_wip.grant(num_new_tokens)\n'
     '                        if num_new_tokens == 0:\n'
     '                            break\n'),
)
ALIGN_TAIL_ENV = "TAAF_VLLM_ALIGN_TAIL"
ALIGN_TAIL_HELPER = "align_tail.py"
ALIGN_TAIL_INSTALLED = "vllm/taaf_align_tail.py"
ALIGN_TAIL_RELATIVE = "vllm/v1/core/sched/scheduler.py"
ALIGN_TAIL_EDITS = (
    ('from vllm.v1.metrics.perf import ModelMetrics, PerfStats\n',
     'from vllm.v1.metrics.perf import ModelMetrics, PerfStats\n'
     'import os as _taaf_align_os\n'
     '_TAAF_ALIGN_TAIL_ON = _taaf_align_os.environ.get("TAAF_VLLM_ALIGN_TAIL", "").strip().lower() not in (\n'
     '    "", "0", "false", "no", "off")\n'
     'if _TAAF_ALIGN_TAIL_ON:\n'
     '    from vllm import taaf_align_tail as _taaf_align\n'),
    ('        if end < prefill_end:\n'
     '            max_prefill_tokens = self.max_num_scheduled_tokens\n',
     '        loose = (\n'
     '            _TAAF_ALIGN_TAIL_ON\n'
     '            and not use_internal_checkpoint\n'
     '            and _taaf_align.loose(self)\n'
     '        )\n'
     '        if end < prefill_end and not loose:\n'
     '            max_prefill_tokens = self.max_num_scheduled_tokens\n'),
    ('            if start % block_size != 0 and not use_internal_checkpoint\n'
     '            else 0,\n',
     '            if start % block_size != 0 and not use_internal_checkpoint and not loose\n'
     '            else 0,\n'),
    ('        # Stop at the earliest mandatory position strictly inside the chunk.\n'
     '        end = min((s for s in stops if start < s < end), default=end)\n',
     '        if loose:\n'
     '            stops = (*stops, *_taaf_align.stops(self, request, block_size, start, end))\n'
     '        # Stop at the earliest mandatory position strictly inside the chunk.\n'
     '        end = min((s for s in stops if start < s < end), default=end)\n'),
)
NO_ADOPT_ENV = "TAAF_VLLM_HISPARSE_NO_ADOPT"
NO_ADOPT_RELATIVE = "vllm/v1/hisparse/coordinator.py"
HISPARSE_NO_ADOPT_EDITS = (
    ('_ACTIVE_TAIL_PAGES = 2\n',
     '_ACTIVE_TAIL_PAGES = 2\n'
     '\n'
     'import os as _taaf_adopt_os\n'
     '\n'
     '_TAAF_NO_ADOPT = _taaf_adopt_os.environ.get("TAAF_VLLM_HISPARSE_NO_ADOPT", "").strip()\n'
     '\n'
     '\n'
     'def _taaf_adopt_gate(request_id, host_blocks, copies):\n'
     '    said = _TAAF_NO_ADOPT\n'
     '    skip = said.lower() in ("1", "true", "yes", "on") or (\n'
     '        said.startswith("file:") and _taaf_adopt_os.path.exists(said[len("file:"):]))\n'
     '    pages = [index for index, host in enumerate(host_blocks)\n'
     '             if not host.is_null and host.block_hash is not None and host.block_hash in copies]\n'
     '    if pages:\n'
     '        first = [copies[host_blocks[index].block_hash][0] for index in pages]\n'
     '        print(f"HISPARSE_ADOPT {request_id} {\'skipped\' if skip else \'adopts\'} {len(pages)} of "\n'
     '              f"{len(host_blocks)} prefix page(s): pages {pages} first-group blocks "\n'
     '              f"{[block.block_id for block in first]} ref_cnt {[block.ref_cnt for block in first]}",\n'
     '              flush=True)\n'
     '    return skip\n'),
    ('        if not self.copies:\n'
     '            return\n'
     '        for host_idx, host_block in enumerate(host_blocks):\n',
     '        if not self.copies:\n'
     '            return\n'
     '        if _taaf_adopt_gate(request_id, host_blocks, self.copies):\n'
     '            return\n'
     '        for host_idx, host_block in enumerate(host_blocks):\n'),
)
TAIL_RESTORE_ENV = "TAAF_VLLM_HISPARSE_TAIL_RESTORE"
TAIL_RESTORE_RELATIVE = "vllm/v1/hisparse/coordinator.py"
HISPARSE_TAIL_RESTORE_EDITS = (
    ('    if coordinator.host_manager is None:\n'
     '        raise ValueError("No HiSparse cache group is configured.")\n'
     '    return coordinator\n',
     '    if coordinator.host_manager is None:\n'
     '        raise ValueError("No HiSparse cache group is configured.")\n'
     '    _taaf_tail_restore_arm(coordinator)\n'
     '    return coordinator\n'
     '\n'
     '\n'
     'def _taaf_tail_restore_arm(coordinator):\n'
     '    import inspect\n'
     '    import os\n'
     '\n'
     '    said = os.environ.get("TAAF_VLLM_HISPARSE_TAIL_RESTORE", "").strip()\n'
     '    if said.lower() in ("0", "false", "no", "off") or hasattr(coordinator, "_taaf_tail_restore"):\n'
     '        return\n'
     '    switch = said[len("file:"):] if said.startswith("file:") else ""\n'
     '    held = coordinator._taaf_tail_restore = {"pending": {}, "decided": {}, "restored": 0, "refused": 0}\n'
     '\n'
     '    def active(request_id):\n'
     '        if request_id not in held["decided"]:\n'
     '            held["decided"][request_id] = not (switch and os.path.exists(switch))\n'
     '        return held["decided"][request_id]\n'
     '    for manager in coordinator.resident_managers:\n'
     '        count = manager.get_num_blocks_to_allocate\n'
     '        local = manager.add_local_computed_blocks\n'
     '        shape = inspect.signature(count)\n'
     '\n'
     '        def get_num_blocks_to_allocate(*args, _count=count, _shape=shape, _manager=manager, **kwargs):\n'
     '            bound = _shape.bind(*args, **kwargs)\n'
     '            bound.apply_defaults()\n'
     '            got = bound.arguments\n'
     '            computed = got["num_local_computed_tokens"]\n'
     '            size = _manager.block_size\n'
     '            if (got["total_computed_tokens"] > computed or computed % size == 0\n'
     '                    or _manager.req_to_blocks.get(got["request_id"]) or not active(got["request_id"])):\n'
     '                return _count(*args, **kwargs)\n'
     '            required = cdiv(got["num_tokens"], size)\n'
     '            if got["apply_admission_cap"]:\n'
     '                required = min(required, _manager._max_admission_blocks_per_request)\n'
     '            return max(required - computed // size, 0)\n'
     '\n'
     '        def add_local_computed_blocks(request_id, new_computed_blocks, num_local_computed_tokens,\n'
     '                                      num_external_computed_tokens, _local=local, _manager=manager):\n'
     '            size = _manager.block_size\n'
     '            if (num_external_computed_tokens or num_local_computed_tokens % size == 0\n'
     '                    or not active(request_id)):\n'
     '                return _local(request_id, new_computed_blocks, num_local_computed_tokens,\n'
     '                              num_external_computed_tokens)\n'
     '            made = _local(request_id, new_computed_blocks, num_local_computed_tokens // size * size,\n'
     '                          num_external_computed_tokens)\n'
     '            held["pending"][request_id] = (num_local_computed_tokens // size, num_local_computed_tokens)\n'
     '            return made\n'
     '\n'
     '        manager.get_num_blocks_to_allocate = get_num_blocks_to_allocate\n'
     '        manager.add_local_computed_blocks = add_local_computed_blocks\n'
     '    plan = coordinator.plan_prefix_materialization\n'
     '    free = coordinator.free\n'
     '\n'
     '    def plan_prefix_materialization(request_id, num_computed_tokens):\n'
     '        wanted = held["pending"].pop(request_id, None)\n'
     '        if wanted is not None:\n'
     '            page, hit = wanted\n'
     '            if coordinator._plan_page_transfer(request_id, page, after_forward=False, restore=True):\n'
     '                held["restored"] += 1\n'
     '                print(f"HISPARSE_TAIL_RESTORE {request_id} page {page}: a local hit of {hit} tokens ends inside "\n'
     '                      f"it, so it is a resident page restored from the request\'s own host copy before the "\n'
     '                      f"forward ({held[\'restored\']} restored, {held[\'refused\']} refused)", flush=True)\n'
     '            else:\n'
     '                held["refused"] += 1\n'
     '                print(f"HISPARSE_TAIL_RESTORE {request_id} page {page}: REFUSED, no host or resident block to "\n'
     '                      f"restore from ({held[\'restored\']} restored, {held[\'refused\']} refused)", flush=True)\n'
     '        return plan(request_id, num_computed_tokens)\n'
     '\n'
     '    def release(request_id):\n'
     '        held["pending"].pop(request_id, None)\n'
     '        held["decided"].pop(request_id, None)\n'
     '        return free(request_id)\n'
     '\n'
     '    coordinator.plan_prefix_materialization = plan_prefix_materialization\n'
     '    coordinator.free = release\n'),
)
PREFILL_FILL_ENV = "TAAF_VLLM_PREFILL_FILL"
PREFILL_FILL_HELPER = "prefill_fill.py"
PREFILL_FILL_INSTALLED = "vllm/taaf_prefill_fill.py"
PREFILL_FILL_RELATIVE = "vllm/v1/core/sched/scheduler.py"
PREFILL_FILL_EDITS = (
    ('from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n',
     'from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n'
     'import os as _taaf_fill_os\n'
     '_TAAF_PREFILL_FILL_ON = _taaf_fill_os.environ.get("TAAF_VLLM_PREFILL_FILL", "").strip().lower() not in (\n'
     '    "", "0", "false", "no", "off")\n'
     'if _TAAF_PREFILL_FILL_ON:\n'
     '    from vllm import taaf_prefill_fill as _taaf_fill\n'),
    ('            if aligned_end > start or block_size <= max_prefill_tokens:\n'
     '                end = aligned_end\n',
     '            if aligned_end > start or (\n'
     '                block_size <= max_prefill_tokens\n'
     '                and not (_TAAF_PREFILL_FILL_ON and _taaf_fill.take(start, end))\n'
     '            ):\n'
     '                end = aligned_end\n'),
)
ADAPTIVE_GATE_RELATIVE = "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
ADAPTIVE_GATE_PINNED = "276b2fec56758a94c71b717f6786b8a6f126db2e20cf964350202feee61a0a39"
ADAPTIVE_GATE_VOUCHED = ("GDNAttentionBackend", "PleShortConvAttentionBackend")
ADAPTIVE_GATE_EDITS = (
    ('from vllm.v1.worker.gpu.attn_utils import (\n'
     '    get_attn_cg_support,\n'
     '    get_query_lens_mismatch_unsupported_backend,\n'
     ')\n',
     'from vllm.v1.worker.gpu.attn_utils import (\n'
     '    get_attn_cg_support,\n'
     '    get_query_lens_mismatch_unsupported_backend,\n'
     ')\n'
     '\n'
     f'_TAAF_VOUCHED = {ADAPTIVE_GATE_VOUCHED!r}\n'
     '\n'
     '\n'
     'def _taaf_unvouched_backend(attn_groups, checked_layer_names):\n'
     '    for groups in attn_groups:\n'
     '        for group in groups:\n'
     '            if checked_layer_names is not None and checked_layer_names.isdisjoint(\n'
     '                group.layer_names\n'
     '            ):\n'
     '                continue\n'
     '            if group.backend.__name__ in _TAAF_VOUCHED:\n'
     '                continue\n'
     '            if not group.backend.supports_device_cpu_query_lens_mismatch():\n'
     '                return group.backend.__name__\n'
     '    return None\n'),
    ('    backend = get_query_lens_mismatch_unsupported_backend(\n'
     '        attn_groups,\n'
     '        checked_layer_names=target_layer_names,\n'
     '    )\n',
     '    backend = _taaf_unvouched_backend(attn_groups, target_layer_names)\n'),
    ('    if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS:\n',
     '    if target_attn_cg_support.min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value:\n'),
    ('logger = init_logger(__name__)\n',
     'logger = init_logger(__name__)\n'
     '_TAAF_CURVES_SAID = __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_CURVES", "").strip()\n'
     '_TAAF_CURVES = __import__("json").loads(_TAAF_CURVES_SAID) if _TAAF_CURVES_SAID else None\n'),
    ('        max_num_tokens = self.req_states.max_num_batched_tokens\n'
     '        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0\n',
     '        max_num_tokens = self.req_states.max_num_batched_tokens\n'
     '        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0\n'
     '        if _TAAF_CURVES is not None:\n'
     '            return\n'),
    ('    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:\n',
     '    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:\n'
     '        if _TAAF_CURVES is not None:\n'
     '            logger.info("TAAF adaptive cost curves from TAAF_VLLM_ADAPTIVE_CURVES: %s", _TAAF_CURVES)\n'
     '            return self.set_cost_curves(\n'
     '                [tuple(point) for point in _TAAF_CURVES["draft"]],\n'
     '                [tuple(point) for point in _TAAF_CURVES["verify"]],\n'
     '            )\n'),
)
ADAPTIVE_CONFIG_RELATIVE = "vllm/config/speculative.py"
ADAPTIVE_CONFIG_PINNED = "9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19"
ADAPTIVE_CONFIG_EDITS = (
    ('        if (\n'
     '            self.method != "dspark"\n'
     '            and self.enable_adaptive_verification\n'
     '            and self.use_local_argmax_reduction\n'
     '        ):\n',
     '        if (\n'
     '            self.method != "dspark"\n'
     '            and self.enable_adaptive_verification\n'
     '            and self.use_local_argmax_reduction\n'
     '            and __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") != "1"\n'
     '        ):\n'),
)
ADAPTIVE_QSA_OPS_EDITS = (
    ('    visible_blocks_ptr,\n    logits_ptr,\n    stride_q_row,\n    stride_q_head,\n    stride_cache_block,\n    stride_cache_token,\n    stride_table_req,\n    stride_logits_row,\n    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,\n',
     '    visible_blocks_ptr,\n    query_start_loc_ptr,\n    logits_ptr,\n    stride_q_row,\n    stride_q_head,\n    stride_cache_block,\n    stride_cache_token,\n    stride_table_req,\n    stride_logits_row,\n    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,\n'),
    ('    TILES_PER_PROG: tl.constexpr,\n    STAGES: tl.constexpr,\n) -> None:\n    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE\n',
     '    TILES_PER_PROG: tl.constexpr,\n    STAGES: tl.constexpr,\n    VARLEN: tl.constexpr = False,\n) -> None:\n    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE\n'),
    ('    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)\n    valid_query_offsets = query_offsets < DECODE_QUERY_LEN\n    rows = request * DECODE_QUERY_LEN + query_offsets\n',
     '    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)\n    if VARLEN:\n        row_start = tl.load(query_start_loc_ptr + request)\n        query_count = tl.minimum(tl.load(query_start_loc_ptr + request + 1) - row_start, DECODE_QUERY_LEN)\n    else:\n        row_start = request * DECODE_QUERY_LEN\n        query_count = DECODE_QUERY_LEN\n    valid_query_offsets = query_offsets < query_count\n    rows = row_start + query_offsets\n'),
    ('    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)\n    query = tl.load(\n        q_ptr\n        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row\n',
     '    valid_query = (query_offset < query_count) & (head < NUM_HEADS)\n    query = tl.load(\n        q_ptr\n        + (row_start + query_offset)[None, :] * stride_q_row\n'),
    ('    for decode_query_len, num_requests in profiles:\n        num_rows = decode_query_len * num_requests\n',
     '    if __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1":\n        profiles = tuple(dict.fromkeys(\n            [(max_decode_query_len, num_requests) for _, num_requests in profiles]\n        ))\n    varlen = __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1"\n    for decode_query_len, num_requests in profiles:\n        num_rows = decode_query_len * num_requests\n'),
    ('            page_table_ptr,\n            visible_blocks_ptr,\n            logits_ptr,\n            num_heads * head_dim,\n',
     '            page_table_ptr,\n            visible_blocks_ptr,\n            visible_blocks_ptr,\n            logits_ptr,\n            num_heads * head_dim,\n'),
    ('            TILES_PER_PROG=tiles_per_program,\n            STAGES=2,\n            # tuned on GB300\n            num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n            grid=(\n',
     '            TILES_PER_PROG=tiles_per_program,\n            STAGES=2,\n            VARLEN=varlen,\n            # tuned on GB300\n            num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n            grid=(\n'),
    ('    decode_query_len: int,\n    block_indices: torch.Tensor,\n) -> None:\n    """Score and select compressed blocks for a request-major decode batch.\n',
     '    decode_query_len: int,\n    block_indices: torch.Tensor,\n    query_start_loc: torch.Tensor | None = None,\n) -> None:\n    """Score and select compressed blocks for a request-major decode batch.\n'),
    ('    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0\n    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"\n    num_requests = q.shape[0] // decode_query_len\n',
     '    assert decode_query_len > 0\n    assert query_start_loc is not None or q.shape[0] % decode_query_len == 0\n    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"\n    num_requests = (\n        q.shape[0] // decode_query_len if query_start_loc is None else page_table.shape[0]\n    )\n'),
    ('        page_table,\n        visible_blocks,\n        logits,\n        *q.stride()[:-1],\n',
     '        page_table,\n        visible_blocks,\n        visible_blocks if query_start_loc is None else query_start_loc,\n        logits,\n        *q.stride()[:-1],\n'),
    ('        TILES_PER_PROG=tiles_per_program,\n        STAGES=2,\n        # tuned on GB300\n        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n    )\n    _topk(\n',
     '        TILES_PER_PROG=tiles_per_program,\n        STAGES=2,\n        VARLEN=query_start_loc is not None,\n        # tuned on GB300\n        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n    )\n    _topk(\n'),
)
ADAPTIVE_QSA_INDEXER_EDITS = (
    ('            if num_decodes * decode_query_len != num_decode_tokens:\n                raise ValueError("QSA decode rows must form a uniform request batch")\n',
     '            varlen = __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1"\n            if not varlen and num_decodes * decode_query_len != num_decode_tokens:\n                raise ValueError("QSA decode rows must form a uniform request batch")\n'),
    ('                decode_query_len,\n                block_indices[decode_slice],\n            )\n',
     '                decode_query_len,\n                block_indices[decode_slice],\n                query_start_loc=(\n                    compressed_metadata.query_start_loc[: num_decodes + 1] if varlen else None\n                ),\n            )\n'),
)
ADAPTIVE_QSA_CACHE_EDITS = (
    ('                decode_threshold=decode_threshold,\n                require_uniform=True,\n',
     '                decode_threshold=decode_threshold,\n                require_uniform=__import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") != "1",\n'),
    ('            if nonzero_query_lens.numel() > 0:\n                decode_query_len = int(nonzero_query_lens[0].item())\n                assert torch.all(nonzero_query_lens == decode_query_len)\n',
     '            if nonzero_query_lens.numel() > 0 and (\n                __import__("os").environ.get("TAAF_VLLM_ADAPTIVE_VERIFY") == "1"\n            ):\n                decode_query_len = decode_threshold\n            elif nonzero_query_lens.numel() > 0:\n                decode_query_len = int(nonzero_query_lens[0].item())\n                assert torch.all(nonzero_query_lens == decode_query_len)\n'),
)
ADAPTIVE_GATE_TARGETS = {ADAPTIVE_GATE_RELATIVE: (ADAPTIVE_GATE_PINNED, ADAPTIVE_GATE_EDITS),
                         ADAPTIVE_CONFIG_RELATIVE: (ADAPTIVE_CONFIG_PINNED, ADAPTIVE_CONFIG_EDITS),
                         "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py": ("2c34883a6396bac0238f6308196854b653eadab8bbebbe7dd2d558e30d065cbb", ADAPTIVE_QSA_OPS_EDITS),
                         "vllm/models/qwen4_exp/nvidia/indexer_qsa.py": ("23a39cc4ee463bd3441e4343d9a0933ab1a81ac5a3f5a5af80f5b084f70d8fef", ADAPTIVE_QSA_INDEXER_EDITS),
                         "vllm/models/qwen4_exp/common/qsa_cache.py": ("c6651a0cf27e2ffb23b90eafe21e044df5609ce976f52ebffa7a4fbea688fc04", ADAPTIVE_QSA_CACHE_EDITS),}

BUNDLE_DIR = Path(os.environ["TAAF_KAGGLE_BUNDLE_DIR"])
WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
SETUP_ENV_PATH = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
RUNTIME_ROOT = Path("/tmp/qwen38-flash-next-vllm-runtime")
ZSTD_MANIFEST_NAME = "runtime-zstd.json"
ZSTD_MANIFEST_SHA256 = "b1d6746d84245c962497ec8c6471e5111df15590eaf07a1a5a603c5a3e97efbb"
ZSTD_CHUNK = 16 * 1024 * 1024
FLAT_MANIFEST_NAME = "runtime-flat.json"
FLAT_MANIFEST_SHA256 = ""
PREFIX_MATCH_UNIT = 0
CUDAGRAPH_CAPTURE_SIZES: tuple[int, ...] = ()
PROMPT_TOKENS_DETAILS = False
PREFIX_RETENTION_ENV = "TAAF_VLLM_PREFIX_RETENTION"
PREFIX_RETENTION = 0
TAR_EXCLUDES = ("--exclude=*/__pycache__", "--exclude=*/__pycache__/*",
                "--exclude=*.pyc", "--exclude=*.pyo")
CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-cache")
COMPILE_CACHE_ROOT = Path("/tmp/qwen38-flash-next-vllm-compile-cache")
TMP_ROOT = Path("/tmp/qwen38-flash-next-vllm-tmp")
SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
SERVER_IDENTITY = WORKING_DIR / "vllm-server-identity.json"
SETUP_PROVENANCE = WORKING_DIR / "vllm-setup-provenance.json"
PREFLIGHT_PATH = WORKING_DIR / "vllm-server-preflight.json"
GPU_PATH = WORKING_DIR / "loaded-gpu-compute.json"
METRICS_PATH = WORKING_DIR / "vllm-metrics-after-preflight.prom"
FAILURE_PATH = WORKING_DIR / "vllm-setup-failure.json"
LAUNCH_NUMBER = re.compile(r"\.launch(\d+)\.[^.]+$")


def launch_log_paths() -> list[Path]:
    return [SERVER_LOG, FAILURE_PATH, SETUP_PROVENANCE, PREFLIGHT_PATH, GPU_PATH, METRICS_PATH,
            WORKING_DIR / WEIGHT_CACHE_DAEMON_LOG]


def next_launch_number(directory: Path) -> int:
    seen = [int(found.group(1)) for path in directory.glob("*.launch*.*")
            if (found := LAUNCH_NUMBER.search(path.name))]
    return max(seen, default=0) + 1


def rotate_launch_logs(paths: list[Path] | None = None) -> list[str]:
    held = [path for path in (launch_log_paths() if paths is None else paths)
            if path.is_file() and path.stat().st_size > 0]
    if not held:
        return []
    number = next_launch_number(held[0].parent)
    moved = []
    for path in held:
        target = path.with_name(f"{path.stem}.launch{number}{path.suffix}")
        os.replace(path, target)
        moved.append(target.name)
    print("VLLM_LOGS_ROTATED", json.dumps({"launch": number, "files": moved}), flush=True)
    return moved
RUNTIME_PROFILE_ENV = "TAAF_VLLM_RUNTIME"
RUNTIME_PROFILE = os.environ.get(RUNTIME_PROFILE_ENV, "").strip() or "day0"
RUNTIME_SUBDIR = ""
RUNTIME_SELECTED_BYTES = 7_906_697_104
RUNTIME_WHITEOUTS_LISTED = False
PLE_INIT_PATTERN = (
    r"Initialized PLE embedding \S+: quantization_method=\S+,"
    r".*pinned=\S+"
)
PROFILE_IMPORT_PROBE = """
from importlib import metadata
import json
import cutlass, torch, flashinfer, vllm
import vllm.models.qwen4_exp.nvidia.ngram_embedding as ngram
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "cutlass": metadata.version("nvidia-cutlass-dsl"),
    "cutlass_file": cutlass.__file__,
    "vllm": metadata.version("vllm"),
    "vllm_file": vllm.__file__,
    "fp8_ple_loader": hasattr(ngram, "Qwen4ExpPLEFp8EmbeddingMethod"),
}, sort_keys=True))
"""
if RUNTIME_PROFILE == "v030":
    RUNTIME_DATASET = "foysalemonshanto/vllm-0-30-0-runtime"
    RUNTIME_SUBDIR = "vllm-0.30.0-runtime"
    VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"
    VLLM_INDEX_DIGEST = (
        "sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
    )
    VLLM_AMD64_MANIFEST_DIGEST = (
        "sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40"
    )
    VLLM_CONFIG_DIGEST = (
        "sha256:d076d77d8dade699041dbcb6bb989a058e51b532e6367f13f80b5a4e63ee5c53"
    )
    VLLM_RUNTIME_MANIFEST_SHA256 = (
        "20947fbc374e5599d93a0bb31f4bdd2cdba29519f3bc08ee970c3b168863712c"
    )
    VLLM_VERSION = "0.30.0"
    VLLM_PLE_PATCHED_SHA256 = (
        "b81f9372eca168157c7f3cf5646d61eb0f139b64c9420588d8288672f18a3fd2"
    )
    PLE_LAYER_RELATIVE = "vllm/models/qwen4_exp/nvidia/ple_layer.py"
    CUTLASS_DSL_VERSION = "4.7.1"
    RUNTIME_SELECTED_BYTES = 8_073_827_054
    RUNTIME_EXTRACTED_BOUND_BYTES = 19_760_729_807
    MIN_RUNTIME_FILESYSTEM_FREE_BYTES = (
        RUNTIME_EXTRACTED_BOUND_BYTES + RUNTIME_CACHE_RESERVE_BYTES
    )
    RUNTIME_WHITEOUTS_LISTED = True
    FLAT_MANIFEST_SHA256 = "86383ac9b32c0f77f4db61097829b859cd90dd89a7a009e7321622566f1358f6"
    PREFIX_MATCH_UNIT = 8
    CUDAGRAPH_CAPTURE_SIZES = (tuple(range(1, 17)) + tuple(range(20, 129, 4))
                               + tuple(range(136, 673, 8)))
    PROMPT_TOKENS_DETAILS = True
    PREFIX_RETENTION = int(os.environ.get(PREFIX_RETENTION_ENV, "").strip() or 1600)
    EXPECTED_RUNTIME_LAYERS = {
        2: (
            "0b3b5bd928240da4156c41fcfb400c1f4f8674acd2faee0413ab5a032dfb5f76",
            105_166_716,
        ),
        8: (
            "1c8ea55558c7cdc4cbc0ca8dfef8425d1d94fb52adc76105a5fbb469cf92b7b9",
            1_453_065_768,
        ),
        18: (
            "d4819562c6f5bf63bd3f6360b7e811a2e3684500666cb52fac944845fad44450",
            5_113_006_184,
        ),
        19: (
            "72a6c87f05fc584624468463a7f44162902fa52072f4cb15685bbbe29cdb53a2",
            1_016_496_432,
        ),
        22: (
            "08725c661696ad9eb516a23bf7561d40d7120c260f9e72ec38fe4af68502016c",
            61_741_903,
        ),
        25: (
            "573620a6ecbf2c45d03db06218908b2f4582fc98e63b7e25e25e326e809cbcb5",
            309_870_272,
        ),
        28: (
            "64f463222af78aeb1f4fbc6b61139d6b86d67763aba2e420b8dd8e2f59a933db",
            14_479_779,
        ),
    }
    FAST_RUNTIME_WHITEOUTS = {}
elif RUNTIME_PROFILE != "day0":
    raise RuntimeError(
        f"{RUNTIME_PROFILE_ENV}={RUNTIME_PROFILE!r} names no runtime this script pins"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fast_start_enabled() -> bool:
    raw = os.environ.get(FAST_START_ENV, "1").strip()
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{FAST_START_ENV} must be 0 or 1, found {raw!r}.")
    return raw == "1"


def _bounded_integer_environment(
    name: str, *, default: int, minimum: int, maximum: int
) -> tuple[int, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    stripped = raw.strip()
    if re.fullmatch(r"\d+", stripped) is None:
        raise RuntimeError(
            f"{name} must be an integer from {minimum} through {maximum}, "
            f"found {raw!r}."
        )
    value = int(stripped)
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} must be an integer from {minimum} through {maximum}, "
            f"found {raw!r}."
        )
    return value, True


def _binary_environment(name: str, *, default: bool = False) -> tuple[bool, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    stripped = raw.strip()
    if stripped not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, found {raw!r}.")
    return stripped == "1", True


def _validate_mtp_dynamic_batch_schedule(
    value: Any, *, max_speculative_tokens: int
) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must be a non-empty JSON list."
        )
    resolved: list[list[int]] = []
    previous_maximum = 0
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(type(item) is not int for item in row)
        ):
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] must contain exactly "
                "three integers: [minimum_batch_size, maximum_batch_size, "
                "num_speculative_tokens]."
            )
        minimum_batch_size, maximum_batch_size, speculative_tokens = row
        if index == 0 and minimum_batch_size != 1:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must start at batch size 1."
            )
        if minimum_batch_size < 1 or maximum_batch_size < minimum_batch_size:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] has an invalid batch "
                f"range: {row!r}."
            )
        if minimum_batch_size <= previous_maximum:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} batch ranges must be strictly "
                "increasing and non-overlapping."
            )
        if not 1 <= speculative_tokens <= max_speculative_tokens:
            raise RuntimeError(
                f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV}[{index}] speculative tokens "
                f"must be from 1 through {max_speculative_tokens}, found "
                f"{speculative_tokens}."
            )
        resolved.append(
            [minimum_batch_size, maximum_batch_size, speculative_tokens]
        )
        previous_maximum = maximum_batch_size
    return resolved


def _mtp_dynamic_batch_schedule_environment(
    *, max_speculative_tokens: int
) -> tuple[list[list[int]] | None, bool]:
    raw = os.environ.get(MTP_DYNAMIC_BATCH_SCHEDULE_ENV)
    if raw is None:
        return None, False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} must contain valid JSON, "
            f"found {raw!r}."
        ) from exc
    return (
        _validate_mtp_dynamic_batch_schedule(
            value, max_speculative_tokens=max_speculative_tokens
        ),
        True,
    )


def resolve_vllm_tuning(
    *, enable_chunked_prefill: bool = True
) -> dict[str, Any]:
    """Resolve and validate the small serving-only benchmark surface."""
    if not isinstance(enable_chunked_prefill, bool):
        raise TypeError("enable_chunked_prefill must be a bool.")
    max_num_seqs, max_num_seqs_overridden = _bounded_integer_environment(
        MAX_NUM_SEQS_ENV,
        default=MAX_NUM_SEQS,
        minimum=1,
        maximum=256,
    )
    requested_batch_tokens, batch_tokens_overridden = _bounded_integer_environment(
        MAX_NUM_BATCHED_TOKENS_ENV,
        default=MAX_NUM_BATCHED_TOKENS,
        minimum=1,
        maximum=1_048_576,
    )
    if enable_chunked_prefill:
        max_num_batched_tokens = requested_batch_tokens
    elif batch_tokens_overridden:
        if requested_batch_tokens < ANALYZER_CONTEXT:
            raise RuntimeError(
                f"{MAX_NUM_BATCHED_TOKENS_ENV} must be at least {ANALYZER_CONTEXT} "
                "when chunked prefill is disabled."
            )
        max_num_batched_tokens = requested_batch_tokens
    else:
        max_num_batched_tokens = NO_CHUNKED_MAX_NUM_BATCHED_TOKENS

    omp_threads, omp_threads_overridden = _bounded_integer_environment(
        OMP_THREADS_ENV,
        default=DEFAULT_OMP_THREADS,
        minimum=1,
        maximum=64,
    )
    mtp_tokens, mtp_tokens_overridden = _bounded_integer_environment(
        MTP_TOKENS_ENV,
        default=DEFAULT_MTP_TOKENS,
        minimum=0,
        maximum=12,
    )
    mtp_index_share_for_iteration, mtp_index_share_overridden = (
        _binary_environment(MTP_INDEX_SHARE_FOR_ITERATION_ENV)
    )
    mtp_dynamic_batch_schedule, mtp_dynamic_schedule_overridden = (
        _mtp_dynamic_batch_schedule_environment(
            max_speculative_tokens=mtp_tokens
        )
        if mtp_tokens > 0
        else (None, MTP_DYNAMIC_BATCH_SCHEDULE_ENV in os.environ)
    )
    if mtp_tokens == 0 and mtp_index_share_for_iteration:
        raise RuntimeError(
            f"{MTP_INDEX_SHARE_FOR_ITERATION_ENV}=1 requires "
            f"{MTP_TOKENS_ENV} to be greater than 0."
        )
    if mtp_tokens == 0 and MTP_DYNAMIC_BATCH_SCHEDULE_ENV in os.environ:
        raise RuntimeError(
            f"{MTP_DYNAMIC_BATCH_SCHEDULE_ENV} requires {MTP_TOKENS_ENV} "
            "to be greater than 0."
        )

    moe_backend_raw = os.environ.get(MOE_BACKEND_ENV)
    moe_backend = None if moe_backend_raw is None else moe_backend_raw.strip()
    if moe_backend not in {None, "flashinfer_b12x", "humming"}:
        raise RuntimeError(
            f"{MOE_BACKEND_ENV} must be 'flashinfer_b12x' or 'humming' when set, "
            f"found {moe_backend_raw!r}."
        )
    gpu_memory_permille, gpu_memory_permille_overridden = (
        _bounded_integer_environment(
            GPU_MEMORY_PERMILLE_ENV,
            default=DEFAULT_GPU_MEMORY_PERMILLE,
            minimum=MIN_GPU_MEMORY_PERMILLE,
            maximum=MAX_GPU_MEMORY_PERMILLE,
        )
    )
    kv_cache_memory_bytes, kv_cache_memory_overridden = (
        _bounded_integer_environment(
            KV_CACHE_MEMORY_BYTES_ENV,
            default=0,
            minimum=0,
            maximum=MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES,
        )
    )
    if 0 < kv_cache_memory_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        raise RuntimeError(
            f"{KV_CACHE_MEMORY_BYTES_ENV} must be 0 or at least "
            f"{MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES}, found {kv_cache_memory_bytes}."
        )

    kv_cache_dtype_raw = os.environ.get(KV_CACHE_DTYPE_ENV)
    kv_cache_dtype = (
        "auto" if kv_cache_dtype_raw is None else kv_cache_dtype_raw.strip().lower()
    )
    if kv_cache_dtype not in {"auto", "fp8", *QSA_QUANT_DTYPES}:
        raise RuntimeError(
            f"{KV_CACHE_DTYPE_ENV} must be 'auto', 'fp8' or one of {QSA_QUANT_DTYPES}, "
            f"found {kv_cache_dtype_raw!r}."
        )
    if (kv_cache_dtype in QSA_QUANT_DTYPES) != _flag_on(QSA_KV8_ENV):
        raise RuntimeError(
            f"{KV_CACHE_DTYPE_ENV}={kv_cache_dtype!r} and {QSA_KV8_ENV}="
            f"{os.environ.get(QSA_KV8_ENV)!r} disagree: the pinned QSA attention reads "
            f"only a BF16 cache, and {QSA_KV8_ENV}=1 is what teaches it {QSA_QUANT_DTYPES}, "
            f"so the two are set together or not at all."
        )

    prefix_raw = os.environ.get(ENABLE_PREFIX_CACHING_ENV)
    if prefix_raw is None:
        enable_prefix_caching = False
    else:
        stripped_prefix = prefix_raw.strip()
        if stripped_prefix not in {"0", "1"}:
            raise RuntimeError(
                f"{ENABLE_PREFIX_CACHING_ENV} must be 0 or 1, found {prefix_raw!r}."
            )
        enable_prefix_caching = stripped_prefix == "1"
    if _flag_on(CACHE_KEEP_ENV) and not enable_prefix_caching:
        raise RuntimeError(
            f"{CACHE_KEEP_ENV} keeps prefix-cache state for the next turn, but "
            f"{ENABLE_PREFIX_CACHING_ENV} is off, so nothing would look it up.")
    fp8_dense_groups()
    fp8_dense_quant()
    fp8_triton_band()

    max_cudagraph_capture_size, cudagraph_overridden = (
        _bounded_integer_environment(
            MAX_CUDAGRAPH_CAPTURE_SIZE_ENV,
            default=0,
            minimum=0,
            maximum=672,
        )
    )

    kv_offloading_gib, kv_offloading_overridden = _bounded_integer_environment(
        KV_OFFLOADING_GIB_ENV,
        default=0,
        minimum=0,
        maximum=MAX_KV_OFFLOADING_GIB,
    )
    if kv_offloading_gib > 0 and not enable_prefix_caching:
        raise RuntimeError(
            f"{KV_OFFLOADING_GIB_ENV} is {kv_offloading_gib} while "
            f"{ENABLE_PREFIX_CACHING_ENV} is off. Offloading a cache nothing "
            f"looks up spends host memory and buys nothing."
        )

    return {
        "schema_version": 1,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "kv_cache_dtype": kv_cache_dtype,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "gpu_memory_permille": gpu_memory_permille,
        "kv_offloading_gib": kv_offloading_gib,
        "max_cudagraph_capture_size": max_cudagraph_capture_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_batched_tokens_requested": requested_batch_tokens,
        "max_num_seqs": max_num_seqs,
        "moe_backend": moe_backend,
        "mtp_dynamic_batch_schedule": mtp_dynamic_batch_schedule,
        "mtp_index_share_for_iteration": mtp_index_share_for_iteration,
        "mtp_speculative_tokens": mtp_tokens,
        "omp_num_threads": omp_threads,
        "overridden": {
            MAX_NUM_SEQS_ENV: max_num_seqs_overridden,
            MAX_NUM_BATCHED_TOKENS_ENV: batch_tokens_overridden,
            OMP_THREADS_ENV: omp_threads_overridden,
            MTP_TOKENS_ENV: mtp_tokens_overridden,
            KV_CACHE_MEMORY_BYTES_ENV: kv_cache_memory_overridden,
            GPU_MEMORY_PERMILLE_ENV: gpu_memory_permille_overridden,
            KV_CACHE_DTYPE_ENV: kv_cache_dtype_raw is not None,
            ENABLE_PREFIX_CACHING_ENV: prefix_raw is not None,
            KV_OFFLOADING_GIB_ENV: kv_offloading_overridden,
            MAX_CUDAGRAPH_CAPTURE_SIZE_ENV: cudagraph_overridden,
            MOE_BACKEND_ENV: moe_backend_raw is not None,
            MTP_INDEX_SHARE_FOR_ITERATION_ENV: mtp_index_share_overridden,
            MTP_DYNAMIC_BATCH_SCHEDULE_ENV: mtp_dynamic_schedule_overridden,
        },
    }


def validate_vllm_tuning_document(value: Any) -> list[str]:
    """Validate persisted tuning without trusting it for process ownership."""
    if not isinstance(value, dict):
        return ["vllm_tuning_is_not_an_object"]
    errors: list[str] = []
    missing = sorted(VLLM_TUNING_FIELDS - set(value))
    unknown = sorted(set(value) - VLLM_TUNING_FIELDS)
    if missing:
        errors.append(f"vllm_tuning_missing_fields:{','.join(missing)}")
    if unknown:
        errors.append(f"vllm_tuning_unknown_fields:{','.join(unknown)}")
    if errors:
        return errors

    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        errors.append("vllm_tuning_wrong_schema_version")
    for field in (
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "mtp_index_share_for_iteration",
    ):
        if type(value[field]) is not bool:
            errors.append(f"vllm_tuning_invalid_{field}")
    if not isinstance(value["kv_cache_dtype"], str) or value[
        "kv_cache_dtype"
    ] not in {"auto", "fp8", *QSA_QUANT_DTYPES}:
        errors.append("vllm_tuning_invalid_kv_cache_dtype")
    if value["moe_backend"] not in {None, "flashinfer_b12x"}:
        errors.append("vllm_tuning_invalid_moe_backend")

    integer_bounds = {
        "kv_cache_memory_bytes": (0, MAX_EXPLICIT_KV_CACHE_MEMORY_BYTES),
        "gpu_memory_permille": (MIN_GPU_MEMORY_PERMILLE, MAX_GPU_MEMORY_PERMILLE),
        "kv_offloading_gib": (0, MAX_KV_OFFLOADING_GIB),
        "max_cudagraph_capture_size": (0, 672),
        "max_num_batched_tokens": (1, 1_048_576),
        "max_num_batched_tokens_requested": (1, 1_048_576),
        "max_num_seqs": (1, 256),
        "mtp_speculative_tokens": (0, 8),
        "omp_num_threads": (1, 64),
    }
    for field, (minimum, maximum) in integer_bounds.items():
        item = value[field]
        if type(item) is not int or not minimum <= item <= maximum:
            errors.append(f"vllm_tuning_invalid_{field}")
    kv_bytes = value["kv_cache_memory_bytes"]
    if type(kv_bytes) is int and 0 < kv_bytes < MIN_EXPLICIT_KV_CACHE_MEMORY_BYTES:
        errors.append("vllm_tuning_invalid_kv_cache_memory_bytes")

    dynamic_schedule = value["mtp_dynamic_batch_schedule"]
    mtp_tokens = value["mtp_speculative_tokens"]
    if dynamic_schedule is not None:
        try:
            _validate_mtp_dynamic_batch_schedule(
                dynamic_schedule,
                max_speculative_tokens=(
                    mtp_tokens if type(mtp_tokens) is int and mtp_tokens > 0 else 0
                ),
            )
        except RuntimeError:
            errors.append("vllm_tuning_invalid_mtp_dynamic_batch_schedule")
    if (
        value["mtp_index_share_for_iteration"] is True
        and (type(mtp_tokens) is not int or mtp_tokens <= 0)
    ):
        errors.append("vllm_tuning_mtp_index_share_without_mtp")

    if (
        type(value["enable_chunked_prefill"]) is bool
        and type(value["max_num_batched_tokens"]) is int
        and type(value["max_num_batched_tokens_requested"]) is int
    ):
        if (
            value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            != value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_chunked_batch_tokens_mismatch")
        if (
            not value["enable_chunked_prefill"]
            and value["max_num_batched_tokens"]
            < value["max_num_batched_tokens_requested"]
        ):
            errors.append("vllm_tuning_no_chunk_batch_tokens_mismatch")

    overridden = value["overridden"]
    if not isinstance(overridden, dict):
        errors.append("vllm_tuning_overridden_is_not_an_object")
    else:
        missing_overrides = sorted(VLLM_TUNING_OVERRIDE_FIELDS - set(overridden))
        unknown_overrides = sorted(set(overridden) - VLLM_TUNING_OVERRIDE_FIELDS)
        if missing_overrides:
            errors.append(
                "vllm_tuning_overridden_missing_fields:"
                + ",".join(missing_overrides)
            )
        if unknown_overrides:
            errors.append(
                "vllm_tuning_overridden_unknown_fields:"
                + ",".join(unknown_overrides)
            )
        if any(type(item) is not bool for item in overridden.values()):
            errors.append("vllm_tuning_overridden_values_are_not_boolean")
    return errors


def input_paths() -> dict[str, Path]:
    raw = os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.")
    return {str(key): Path(str(path)) for key, path in value.items()}


def source_identity() -> dict[str, Any]:
    path = BUNDLE_DIR / "SOURCE_IDENTITY.json"
    value = read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid source identity: {path}")
    expected_setup = str(value.get("serving_setup_sha256", ""))
    actual_setup = sha256_file(Path(__file__).resolve())
    if not expected_setup or expected_setup != actual_setup:
        raise RuntimeError(
            f"Serving setup identity mismatch: {actual_setup} != {expected_setup or '<missing>'}"
        )
    if value.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong model manifest hash.")
    runtime = value.get("runtime") or {}
    manifest_sha = value.get("runtime_manifest_sha256")
    if RUNTIME_PROFILE != "day0":
        runtime = (value.get("runtime_profiles") or {}).get(RUNTIME_PROFILE) or {}
        manifest_sha = runtime.get("manifest_sha256")
    if manifest_sha != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("Source identity has the wrong vLLM runtime manifest hash.")
    model = value.get("model") or {}
    if model.get("hf_repo") != MODEL_HF_REPO or model.get("hf_revision") != MODEL_HF_REVISION:
        raise RuntimeError("Source identity has the wrong model revision.")
    if runtime.get("image") != VLLM_IMAGE or runtime.get("kaggle_dataset") != RUNTIME_DATASET:
        raise RuntimeError("Source identity has the wrong vLLM runtime.")
    return value


def resolve_runtime_dir() -> Path:
    mapped = input_paths().get(RUNTIME_DATASET)
    candidates = [
        mapped,
        Path("/kaggle/input") / RUNTIME_DATASET.split("/", 1)[1],
        Path("/kaggle/input/datasets") / RUNTIME_DATASET,
    ]
    for candidate in candidates:
        if candidate is not None and (
            candidate / RUNTIME_SUBDIR / "runtime-manifest.json"
        ).is_file():
            return candidate / RUNTIME_SUBDIR
    matches = []
    for path in Path("/kaggle/input").rglob("runtime-manifest.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        if manifest.get("index_digest") == VLLM_INDEX_DIGEST:
            matches.append(path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve one pinned {RUNTIME_DATASET}; matches={matches}")
    return matches[0]



_EXPERT_FILE = re.compile(r"^layer-(\d+)-experts-(\d+)-(\d+)\.safetensors$")
_EXPERT_TENSOR = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(\d+)\.(.+)$")
_GATE_TENSOR = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
_MTP_TENSOR = re.compile(
    r"^mtp\.layers\.\d+\.mlp\.(?:gate\.weight|experts\.(?:down_proj|gate_up_proj))$")
_MTP_EXPERT_TENSOR = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts\.)(\d+)\.(.+)$")
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def _safetensors_header(path: Path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        head = json.loads(handle.read(size).decode("utf-8"))
    return head, 8 + size


def _write_safetensors(src: Path, dst: Path, body: int, plan, edits) -> None:
    head = {}
    offset = 0
    for name, spec in plan:
        length = (len(edits[name]) if name in edits
                  else spec["data_offsets"][1] - spec["data_offsets"][0])
        head[name] = {"dtype": spec["dtype"], "shape": spec["shape"],
                      "data_offsets": [offset, offset + length]}
        offset += length
    blob = json.dumps(head, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        writer.write(struct.pack("<Q", len(blob)))
        writer.write(blob)
        for name, spec in plan:
            if name in edits:
                writer.write(edits[name])
                continue
            start, end = spec["data_offsets"]
            reader.seek(body + start)
            left = end - start
            while left:
                chunk = reader.read(min(left, 1 << 22))
                if not chunk:
                    raise EOFError(f"{src} ended inside {name}")
                writer.write(chunk)
                left -= len(chunk)


def _expert_order_name() -> str:
    said = str(os.environ.get(EXPERT_ORDER_ENV) or "").strip().lower() or "saliency"
    if said not in EXPERT_ORDER_FILES:
        raise ValueError(f"{EXPERT_ORDER_ENV} must be one of {sorted(EXPERT_ORDER_FILES)}, "
                         f"found {said!r}")
    return said


def _expert_order() -> dict:
    path = BUNDLE_DIR / EXPERT_ORDER_FILES[_expert_order_name()]
    if not path.is_file():
        raise FileNotFoundError(
            f"{EXPERTS_KEPT_ENV} was set but {path} is not beside the script."
        )
    order = read_json(path)
    if not isinstance(order, dict) or not order:
        raise RuntimeError(f"Invalid expert order: {path}")
    lengths = {len(v) for v in order.values()}
    if lengths != {512}:
        raise RuntimeError(f"Expert order must rank all 512 per layer, saw {sorted(lengths)}")
    for layer, ranked in order.items():
        if sorted(ranked) != list(range(512)):
            raise RuntimeError(f"Layer {layer} is not a permutation of 0..511")
    if MTP_KEY not in order:
        raise RuntimeError(
            f"{path} carries no '{MTP_KEY}' ordering; the speculative head keeps "
            f"its experts as fused tensors and must be cut with the rest")
    return {(layer if layer == MTP_KEY else int(layer)): ranked
            for layer, ranked in order.items()}


EXPERT_CUT_PLAN_NAME = "expert_cut_plan.json"
EXPERT_CUT_MODULE: list = []


def _expert_cut_module() -> Any:
    if not EXPERT_CUT_MODULE:
        spec = importlib.util.spec_from_file_location("taaf_expert_cut", str(BUNDLE_DIR / EXPERT_CUT_HELPER))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EXPERT_CUT_MODULE.append(module)
    return EXPERT_CUT_MODULE[0]


def _slice_expert_shard(src: Path, dst: Path, keep: dict, layer: int) -> int:
    head, body = _safetensors_header(src)
    head.pop("__metadata__", None)
    plan = []
    for name, spec in head.items():
        found = _EXPERT_TENSOR.match(name)
        if not found:
            raise RuntimeError(f"unexpected tensor in {src.name}: {name}")
        prefix, said_layer, expert, tail = found.groups()
        if int(said_layer) != layer:
            raise RuntimeError(f"{src.name} carries layer {said_layer}")
        if int(expert) not in keep:
            continue
        plan.append((f"{prefix}{keep[int(expert)]}.{tail}", spec))
    plan.sort(key=lambda row: row[0])
    _write_safetensors(src, dst, body, plan, {})
    return len(plan)


def _take_rows(reader, spec: dict, body: int, keep) -> bytes:
    step = _DTYPE_BYTES[spec["dtype"]]
    for size in spec["shape"][1:]:
        step *= size
    start = body + spec["data_offsets"][0]
    picked = bytearray()
    for row in keep:
        reader.seek(start + row * step)
        chunk = reader.read(step)
        if len(chunk) != step:
            raise EOFError(f"row {row} ended early")
        picked += chunk
    return bytes(picked)


def _slice_router_shard(src: Path, dst: Path, rows_by_layer: dict,
                        mtp_rows) -> tuple:
    head, body = _safetensors_header(src)
    head.pop("__metadata__", None)
    edits = {}
    gates = mtp = 0
    with open(src, "rb") as reader:
        for name, spec in head.items():
            found = _GATE_TENSOR.match(name)
            if found:
                keep = rows_by_layer[int(found.group(1))]
                gates += 1
            elif _MTP_TENSOR.match(name):
                keep = mtp_rows
                mtp += 1
            else:
                continue
            if spec["shape"][0] != 512:
                raise RuntimeError(f"{name} does not lead with 512: {spec['shape']}")
            edits[name] = _take_rows(reader, spec, body, keep)
            spec["shape"] = [len(keep), *spec["shape"][1:]]
    _write_safetensors(src, dst, body, sorted(head.items(), key=lambda r: r[0]), edits)
    return gates, mtp



def _slice_any_shard(src: Path, dst: Path, keep_by_layer: dict, rows_by_layer: dict,
                     mtp_rows, mtp_keep: dict) -> tuple:
    head, body = _safetensors_header(src)
    head.pop("__metadata__", None)
    plan = []
    edits = {}
    left = []
    experts = gates = mtp = 0
    with open(src, "rb") as reader:
        for name, spec in head.items():
            found = _EXPERT_TENSOR.match(name)
            head_found = None if found else _MTP_EXPERT_TENSOR.match(name)
            if found or head_found:
                prefix, expert, tail = ((found.group(1), found.group(3), found.group(4))
                                        if found else head_found.groups())
                keep = keep_by_layer[int(found.group(2))] if found else mtp_keep
                if int(expert) not in keep:
                    continue
                plan.append((f"{prefix}{keep[int(expert)]}.{tail}", spec))
                experts += 1
                mtp += 0 if found else 1
                continue
            gate = _GATE_TENSOR.match(name)
            if gate or _MTP_TENSOR.match(name):
                rows = rows_by_layer[int(gate.group(1))] if gate else mtp_rows
                if spec["shape"][0] != 512:
                    raise RuntimeError(f"{name} does not lead with 512: {spec['shape']}")
                edits[name] = _take_rows(reader, spec, body, rows)
                spec["shape"] = [len(rows), *spec["shape"][1:]]
                gates += 1 if gate else 0
                mtp += 0 if gate else 1
            elif (spec.get("shape") and spec["shape"][0] == 512
                    and ".mlp." in name and "shared_expert" not in name
                    and ("experts" in name or "gate" in name)):
                left.append(name)
            plan.append((name, spec))
    plan.sort(key=lambda row: row[0])
    _write_safetensors(src, dst, body, plan, edits)
    return experts, gates, mtp, left


def _holds_cuttable(path: Path) -> bool:
    try:
        head, _ = _safetensors_header(path)
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False
    for name in head:
        if name == "__metadata__":
            continue
        if _EXPERT_TENSOR.match(name) or _GATE_TENSOR.match(name) or _MTP_TENSOR.match(name):
            return True
    return False


def prune_by_tensors(model_dir: Path, stamp: Path, keep_by_layer: dict,
                     rows_by_layer: dict, mtp_rows, keep_n: int) -> tuple:
    mtp_keep = {expert: index for index, expert in enumerate(mtp_rows)}
    experts = gates = mtp = linked = 0
    left = []
    workers, _ = _bounded_integer_environment(PRUNE_WORKERS_ENV, default=1, minimum=1, maximum=PRUNE_WORKERS_MAX)
    files = [path for path in sorted(model_dir.iterdir()) if path.is_file()]
    cuttable = [path for path in files if path.suffix == ".safetensors" and _holds_cuttable(path)]
    cut: dict[Path, tuple] = {}

    def slice_one(path: Path) -> tuple:
        return _slice_any_shard(path, stamp / path.name, keep_by_layer, rows_by_layer, mtp_rows, mtp_keep)

    if workers > 1 and len(cuttable) > 1:
        pool_size = min(workers, len(cuttable))
        print(f"PRUNE_WORKERS {len(cuttable)} shards cut by {pool_size} threads", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
            for path, result in zip(cuttable, pool.map(slice_one, cuttable)):
                cut[path] = result
    for path in files:
        out = stamp / path.name
        if path in cuttable:
            one, two, three, over = cut[path] if path in cut else slice_one(path)
            experts += one
            gates += two
            mtp += three
            left += over
            continue
        if path.name == "config.json":
            value = read_json(path)
            (value.get("text_config") or value)["num_experts"] = keep_n
            out.write_text(json.dumps(value, indent=2), encoding="utf-8")
            continue
        try:
            os.symlink(path, out)
        except OSError:
            shutil.copy2(path, out)
        linked += 1
    if left:
        raise RuntimeError(f"{len(left)} routing tensor(s) still lead with 512 after the "
                           f"cut, so an expert axis was missed: {sorted(left)[:6]}")
    return experts, gates, mtp, linked


SCRATCH_CANDIDATES = ("/kaggle/tmp", "/tmp", "/var/tmp", "/kaggle/working", "/root")


def _filesystems() -> list[tuple[str, str]]:
    rows = []
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                rows.append((parts[1], parts[2]))
    except OSError:
        return []
    return sorted(rows, key=lambda row: -len(row[0]))


def _fstype_of(path: Path, table: list[tuple[str, str]]) -> str:
    text = str(path)
    for mount, kind in table:
        if text == mount or text.startswith(mount.rstrip("/") + "/"):
            return kind
    return "unknown"


def _cgroup_memory() -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("memory.max", "memory.current", "memory.high"):
        for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
            item = Path(base) / name
            if not item.is_file():
                continue
            body = item.read_text(encoding="utf-8", errors="replace").strip()
            if body.isdigit():
                out[name] = int(body)
            break
    return out


def report_space(where: str) -> None:
    table = _filesystems()
    for name in SCRATCH_CANDIDATES:
        path = Path(name)
        if not path.is_dir():
            print(f"SPACE {where} {name} absent", flush=True)
            continue
        try:
            usage = shutil.disk_usage(str(path))
        except OSError as failure:
            print(f"SPACE {where} {name} {type(failure).__name__}: {failure}", flush=True)
            continue
        print(f"SPACE {where} {name} fstype={_fstype_of(path, table)} "
              f"free={usage.free / 2**30:.1f}GiB total={usage.total / 2**30:.1f}GiB",
              flush=True)
    limits = _cgroup_memory()
    print("SPACE " + where + " cgroup " + json.dumps(
        {key: round(value / 2**30, 1) for key, value in limits.items()}), flush=True)


def _scratch_dir(needed: int) -> Path:
    table = _filesystems()
    room = []
    for name in SCRATCH_CANDIDATES:
        path = Path(name)
        try:
            path.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(path))
        except OSError as failure:
            print(f"SCRATCH {name} unusable: {type(failure).__name__}: {failure}",
                  flush=True)
            continue
        kind = _fstype_of(path, table)
        print(f"SCRATCH {name} fstype={kind} free={usage.free / 2**30:.1f}GiB "
              f"needed={needed / 2**30:.1f}GiB", flush=True)
        if usage.free >= needed:
            room.append((kind != "tmpfs", usage.free, name))
    if not room:
        raise RuntimeError(
            f"No writable path holds the {needed / 2**30:.1f} GiB the cut needs; "
            f"candidates were {SCRATCH_CANDIDATES}")
    room.sort(reverse=True)
    on_disk, free, chosen = room[0]
    if not on_disk:
        limits = _cgroup_memory()
        ceiling = limits.get("memory.max", 0) - limits.get("memory.current", 0)
        print(f"SCRATCH {chosen} is tmpfs: it spends host memory, and this container "
              f"has {ceiling / 2**30:.1f}GiB of its cgroup left", flush=True)
        if ceiling and needed > ceiling * 0.6:
            raise RuntimeError(
                f"The only path with room is {chosen}, a tmpfs, and the "
                f"{needed / 2**30:.1f} GiB it needs is over 60% of the "
                f"{ceiling / 2**30:.1f} GiB this cgroup has left; a cut written there "
                f"would be paid for in host memory and the serve would be killed")
    print(f"SCRATCH chose {chosen} ({free / 2**30:.1f}GiB free)", flush=True)
    return Path(chosen) / "pruned-model"


PREBAKED_STAMP = "PRUNE_COMPLETE.json"
LINK_FARMS: dict[tuple[str, int], Path] = {}


def _prebaked_dirs():
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob(PREBAKED_STAMP):
        if path.is_file():
            found.append(path.parent)
    return sorted(found)


def _config_but_experts(value: dict) -> dict:
    held = json.loads(json.dumps(value))
    (held.get("text_config") or held).pop("num_experts", None)
    return held


def _prebaked_for(keep_n: int, model_dir: Path):
    for folder in _prebaked_dirs():
        try:
            stamp = read_json(folder / PREBAKED_STAMP)
        except Exception as exc:
            print(f"PREBAKE ignoring {folder}: unreadable stamp ({exc})", flush=True)
            continue
        if int(stamp.get("experts_kept") or 0) != keep_n:
            continue
        cut_by = str(stamp.get("expert_order") or "saliency")
        if cut_by != _expert_order_name():
            print(f"PREBAKE ignoring {folder}: cut by the {cut_by} order, the serve asks "
                  f"for the {_expert_order_name()} order", flush=True)
            continue
        shards = sum(1 for one in folder.iterdir()
                     if one.is_file() and _EXPERT_FILE.match(one.name))
        wanted = sum(1 for one in model_dir.iterdir()
                     if one.is_file() and _EXPERT_FILE.match(one.name))
        if shards != wanted:
            print(f"PREBAKE ignoring {folder}: {shards} expert shards against "
                  f"{wanted} in the mount", flush=True)
            continue
        config = folder / "config.json"
        if not config.is_file():
            print(f"PREBAKE ignoring {folder}: no config.json", flush=True)
            continue
        try:
            said = int(read_json(config)["text_config"]["num_experts"])
        except Exception as exc:
            print(f"PREBAKE ignoring {folder}: config unreadable ({exc})", flush=True)
            continue
        if said != keep_n:
            print(f"PREBAKE ignoring {folder}: config says {said} experts, "
                  f"not {keep_n}", flush=True)
            continue
        try:
            same = _config_but_experts(read_json(config)) == _config_but_experts(
                read_json(model_dir / "config.json"))
        except Exception as exc:
            print(f"PREBAKE ignoring {folder}: the mount's config is unreadable ({exc})", flush=True)
            continue
        if not same:
            print(f"PREBAKE ignoring {folder}: its config.json is another checkpoint's, not the "
                  f"mount's with fewer experts", flush=True)
            continue
        return folder
    return None


def _link_farm(baked: Path, model_dir: Path, keep_n: int):
    began = time.time()
    stamp = _scratch_dir(1 << 30) / f"k{keep_n}-linked"
    shutil.rmtree(stamp, ignore_errors=True)
    stamp.mkdir(parents=True, exist_ok=True)
    taken = carried = 0
    for path in sorted(baked.iterdir()):
        if not path.is_file() or path.name == PREBAKED_STAMP:
            continue
        os.symlink(path, stamp / path.name)
        taken += 1
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        out = stamp / path.name
        if out.exists() or out.is_symlink():
            continue
        os.symlink(path, out)
        carried += 1
    (stamp / PREBAKED_STAMP).write_text(
        json.dumps({"experts_kept": keep_n, "expert_order": _expert_order_name(),
                    "linked_from_prebake": taken,
                    "linked_from_mount": carried, "source": str(baked)}),
        encoding="utf-8")
    print(f"PREBAKE linked {stamp} in {time.time() - began:.1f}s "
          f"({taken} pruned files from {baked}, {carried} untouched from the "
          f"mount); the startup prune is skipped", flush=True)
    return stamp


def _cut_rows(keep_n: int) -> tuple[dict, list]:
    order = _expert_order()
    mtp_rows = sorted(int(one) for one in order.pop(MTP_KEY, [])[:keep_n])
    if len(mtp_rows) != keep_n:
        raise RuntimeError(
            f"the saliency file carries no '{MTP_KEY}' ordering, so the "
            f"speculative head's fused expert tensors cannot be cut with the rest")
    return {layer: sorted(ranked[:keep_n]) for layer, ranked in order.items()}, mtp_rows


def expert_cut_on() -> bool:
    return _flag_on(EXPERT_CUT_ENV)


def cut_view_active(served: Path) -> bool:
    return (Path(served) / EXPERT_CUT_PLAN_NAME).is_file()


def _cut_view(model_dir: Path, keep_n: int) -> Path:
    began = time.time()
    helper = _expert_cut_module()
    rows_by_layer, mtp_rows = _cut_rows(keep_n)
    workers, _ = _bounded_integer_environment(
        EXPERT_CUT_WORKERS_ENV, default=helper.WORKERS_DEFAULT, minimum=1, maximum=EXPERT_CUT_WORKERS_MAX)
    window, _ = _bounded_integer_environment(
        EXPERT_CUT_WINDOW_ENV, default=helper.WINDOW_MIB_DEFAULT, minimum=64, maximum=EXPERT_CUT_WINDOW_MAX_MIB)
    coalesce, _ = _bounded_integer_environment(
        EXPERT_CUT_COALESCE_ENV, default=helper.COALESCE_MIB_DEFAULT, minimum=0, maximum=EXPERT_CUT_COALESCE_MAX_MIB)
    gap, _ = _bounded_integer_environment(
        EXPERT_CUT_GAP_ENV, default=helper.GAP_KIB_DEFAULT, minimum=0, maximum=EXPERT_CUT_GAP_MAX_KIB)
    plan = helper.make_plan(rows_by_layer, mtp_rows, workers, window, coalesce, gap)
    files = [path for path in sorted(model_dir.iterdir()) if path.is_file()]
    cuttable = [path for path in files if path.suffix == ".safetensors" and _holds_cuttable(path)]
    experts, gates, mtps, left = helper.scan(cuttable, helper.Plan(plan))
    if left:
        raise RuntimeError(f"{len(left)} routing tensor(s) still lead with 512 after the "
                           f"cut, so an expert axis was missed: {sorted(left)[:6]}")
    if gates != 48:
        raise RuntimeError(f"cut {gates} routers, wanted 48")
    if mtps < 3:
        raise RuntimeError(f"cut {mtps} of the speculative head's tensors, wanted at "
                           f"least its router and its experts")
    stamp = _scratch_dir(1 << 30) / f"k{keep_n}-{'cut' if _expert_order_name() == 'saliency' else _expert_order_name() + '-cut'}"
    shutil.rmtree(stamp, ignore_errors=True)
    stamp.mkdir(parents=True, exist_ok=True)
    for path in files:
        out = stamp / path.name
        if path.name == "config.json":
            value = read_json(path)
            (value.get("text_config") or value)["num_experts"] = keep_n
            out.write_text(json.dumps(value, indent=2), encoding="utf-8")
            continue
        try:
            os.symlink(path, out)
        except OSError:
            shutil.copy2(path, out)
    (stamp / EXPERT_CUT_PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
    (stamp / PREBAKED_STAMP).write_text(
        json.dumps({"experts_kept": keep_n, "expert_order": _expert_order_name(), "cut_at_load": True,
                    "expert_tensors": experts, "routers": gates, "mtp_tensors": mtps,
                    "workers": workers, "window_mib": window, "coalesce_mib": coalesce, "gap_kib": gap}),
        encoding="utf-8")
    print(f"EXPERT_CUT {stamp} viewed in {time.time() - began:.1f}s: {len(cuttable)} of {len(files)} files hold "
          f"cuttable tensors ({experts} expert tensors kept, {gates} routers, {mtps} speculative-head tensors); "
          f"no byte is copied, the loader reads only the kept experts' byte ranges with {workers} threads and a "
          f"{window} MiB window" + (f"; each cut file is read ahead in reads of up to {coalesce} MiB that join ranges "
                                    f"at most {gap} KiB apart, its uncut tensors by the same threads"
                                    if coalesce else ""), flush=True)
    return stamp


def prune_model_dir(model_dir: Path) -> Path:
    """Build a pruned copy beside the mount; return it, or the mount unchanged."""
    raw = os.environ.get(EXPERTS_KEPT_ENV)
    if raw is None or not str(raw).strip():
        return model_dir
    keep_n = int(str(raw).strip())
    if not 8 <= keep_n <= 512:
        raise ValueError(f"{EXPERTS_KEPT_ENV} must be 8..512, found {keep_n!r}")
    if keep_n == 512:
        return model_dir
    print(f"EXPERT_ORDER {_expert_order_name()}: the cut keeps the first {keep_n} of each "
          f"layer in {EXPERT_ORDER_FILES[_expert_order_name()]}", flush=True)
    if load_dummy_enabled():
        return _dummy_model_dir(model_dir, keep_n)
    if expert_cut_on():
        return _cut_view(model_dir, keep_n)
    farm_key = (str(model_dir), keep_n)
    if LINK_FARMS.get(farm_key) is not None and LINK_FARMS[farm_key].is_dir():
        print(f"PREBAKE reusing {LINK_FARMS[farm_key]}, linked at the start of the setup", flush=True)
        return LINK_FARMS[farm_key]
    baked = _prebaked_for(keep_n, model_dir)
    if baked is not None:
        try:
            LINK_FARMS[farm_key] = _link_farm(baked, model_dir, keep_n)
            return LINK_FARMS[farm_key]
        except OSError as exc:
            print(f"PREBAKE could not link {baked} ({exc}); building instead",
                  flush=True)
    rows_by_layer, mtp_rows = _cut_rows(keep_n)
    keep_by_layer = {layer: {expert: index for index, expert in enumerate(rows)}
                     for layer, rows in rows_by_layer.items()}
    ours = [path for path in model_dir.iterdir()
            if path.is_file() and _EXPERT_FILE.match(path.name)]
    if ours:
        wanted = int((sum(path.stat().st_size for path in ours) * keep_n / 512
                      + sum(path.stat().st_size for path in model_dir.iterdir()
                            if path.is_file() and path.name.startswith("model-bf16-")
                            and path.suffix == ".safetensors")) * 1.06)
    else:
        wanted = int(sum(path.stat().st_size for path in model_dir.rglob("*")
                         if path.is_file()) * keep_n / 512 * 1.10)
    stamp = _scratch_dir(wanted) / (f"k{keep_n}" if _expert_order_name() == "saliency"
                                    else f"k{keep_n}-{_expert_order_name()}")
    done = stamp / "PRUNE_COMPLETE.json"
    if done.is_file():
        print(f"PRUNE reusing {stamp}", flush=True)
        return stamp
    if stamp.exists():
        shutil.rmtree(stamp, ignore_errors=True)
    stamp.mkdir(parents=True, exist_ok=True)
    began = time.time()
    experts = routers = mtps = linked = 0
    if not any(_EXPERT_FILE.match(path.name) for path in model_dir.iterdir()
               if path.is_file()):
        print(f"PRUNE {model_dir} carries no per-layer expert shards of ours, so the cut "
              f"goes by tensor name", flush=True)
        experts, routers, mtps, linked = prune_by_tensors(
            model_dir, stamp, keep_by_layer, rows_by_layer, mtp_rows, keep_n)
        if routers != 48:
            raise RuntimeError(f"cut {routers} routers, wanted 48")
        if mtps < 3:
            raise RuntimeError(f"cut {mtps} of the speculative head's tensors, wanted at "
                               f"least its router and its experts")
        (stamp / PREBAKED_STAMP).write_text(
            json.dumps({"experts_kept": keep_n, "expert_order": _expert_order_name(),
                        "expert_tensors": experts,
                        "routers": routers, "mtp_tensors": mtps, "linked": linked,
                        "seconds": round(time.time() - began, 1)}), encoding="utf-8")
        print(f"PRUNE built {stamp} in {time.time() - began:.0f}s by tensor name "
              f"({experts} expert tensors, {routers} routers, {mtps} speculative-head "
              f"tensors, {linked} linked)", flush=True)
        return stamp
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        out = stamp / path.name
        found = _EXPERT_FILE.match(path.name)
        if found:
            _slice_expert_shard(path, out, keep_by_layer[int(found.group(1))],
                                int(found.group(1)))
            experts += 1
        elif path.name.startswith("model-bf16-") and path.suffix == ".safetensors":
            gates, mtp = _slice_router_shard(path, out, rows_by_layer, mtp_rows)
            routers += gates
            mtps += mtp
        elif path.name == "config.json":
            value = read_json(path)
            value["text_config"]["num_experts"] = keep_n
            out.write_text(json.dumps(value, indent=2), encoding="utf-8")
        else:
            try:
                os.symlink(path, out)
            except OSError:
                shutil.copy2(path, out)
            linked += 1
    if routers != 48:
        raise RuntimeError(f"cut {routers} routers, wanted 48")
    if mtps != 3:
        raise RuntimeError(
            f"cut {mtps} of the speculative head's expert tensors, wanted 3: its "
            f"experts are two FUSED tensors plus a router, all leading with 512, "
            f"and vLLM numbers a fused tensor's experts from the tensor itself")
    done.write_text(json.dumps({"experts_kept": keep_n, "expert_order": _expert_order_name(),
                                "expert_shards": experts,
                                "routers": routers, "mtp_tensors": mtps,
                                "linked": linked,
                                "seconds": round(time.time() - began, 1)}), encoding="utf-8")
    print(f"PRUNE built {stamp} in {time.time() - began:.0f}s "
          f"({experts} expert shards, {routers} routers, {mtps} speculative-head "
          f"tensors, {linked} linked)", flush=True)
    return stamp



DUMMY_WEIGHTS_FILL_SOURCE = '''
_TAAF_TALLY = {"float": 0, "random": 0, "zero": 0, "bad": 0, "scanned": 0, "probes": 0}
_TAAF_PACKED = ("qweight", "qzeros", "weight_packed")


def _taaf_fill(name, tensor):
    import torch

    if (tensor.device.type == "meta" or torch.is_floating_point(tensor)
            or getattr(tensor, "dummy_weight_value", None) is not None):
        _TAAF_TALLY["float"] += 1
        initialize_single_dummy_weight(tensor)
        return
    with torch.no_grad():
        if name.rsplit(".", 1)[-1].endswith(_TAAF_PACKED) or tensor.dtype in (torch.uint8, torch.int8):
            tensor.view(torch.uint8).random_(0, 256)
            _TAAF_TALLY["random"] += 1
        else:
            tensor.zero_()
            _TAAF_TALLY["zero"] += 1


def _taaf_finite(tensor):
    import torch

    flat = tensor.detach().reshape(-1)
    for at in range(0, flat.numel(), 1 << 24):
        if not bool(torch.isfinite(flat[at:at + (1 << 24)].float()).all()):
            return False
    return True


def _taaf_probe(module, args, output):
    import torch

    if torch.compiler.is_compiling() or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        return
    if _TAAF_TALLY["probes"] >= 3:
        return
    first = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(first):
        return
    _TAAF_TALLY["probes"] += 1
    peak = float(first.detach().float().abs().max()) if first.numel() else 0.0
    print(f"DUMMY_WEIGHTS forward output #{_TAAF_TALLY['probes']}: shape={tuple(first.shape)} "
          f"finite={_taaf_finite(first)} abs_max={peak:.4g}", flush=True)


def _taaf_report(model):
    import torch

    for name, held in list(model.named_parameters()) + list(model.named_buffers()):
        if held.device.type == "meta" or not torch.is_floating_point(held) or not held.numel():
            continue
        _TAAF_TALLY["scanned"] += 1
        if not _taaf_finite(held):
            _TAAF_TALLY["bad"] += 1
            if _TAAF_TALLY["bad"] <= 8:
                print(f"DUMMY_WEIGHTS non-finite floating tensor after fill: {name}", flush=True)
    print(f"DUMMY_WEIGHTS filled: {_TAAF_TALLY['float']} floating tensors by vLLM, {_TAAF_TALLY['random']} packed "
          f"integer tensors with random codes, {_TAAF_TALLY['zero']} other integer tensors zeroed (a zero is a valid "
          f"index); {_TAAF_TALLY['bad']} of {_TAAF_TALLY['scanned']} scanned floating tensors non-finite", flush=True)
    model.register_forward_hook(_taaf_probe)
'''

DUMMY_WEIGHTS_TARGETS = {
    "vllm/model_executor/model_loader/dummy_loader.py": (
        "ad1c3fea9a60fdd9f31fed97bdb5731f8618e8a4f9b228b0026e21247c45e50e",
        (
            ('from vllm.model_executor.model_loader.weight_utils import (\n'
             '    initialize_single_dummy_weight,\n'
             ')\n',
             'from vllm.model_executor.model_loader.weight_utils import (\n'
             '    initialize_single_dummy_weight,\n'
             ')\n' + DUMMY_WEIGHTS_FILL_SOURCE),
            ('    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:\n'
             '        for layer in model.modules():\n',
             '    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:\n'
             '        self._taaf_load_weights(model, model_config)\n'
             '        _taaf_report(model)\n'
             '\n'
             '    def _taaf_load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:\n'
             '        for layer in model.modules():\n'),
            ('                    if name not in layer._non_persistent_buffers_set:\n'
             '                        initialize_single_dummy_weight(tensor)\n',
             '                    if name not in layer._non_persistent_buffers_set:\n'
             '                        _taaf_fill(name, tensor)\n'),
            ('        for tensor in get_layer_tensors(layer).values():\n'
             '            initialize_single_dummy_weight(tensor)\n',
             '        for name, tensor in get_layer_tensors(layer).items():\n'
             '            _taaf_fill(name, tensor)\n'),
        ),
    ),
}


def patch_dummy_weights() -> dict[str, Any]:
    if not load_dummy_enabled():
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{LOAD_DUMMY_ENV} edits vLLM 0.30.0's dummy loader; the {RUNTIME_PROFILE} runtime "
                           f"ships another one.")
    files, made = _apply_pinned_targets(LOAD_DUMMY_ENV, DUMMY_WEIGHTS_TARGETS, _runtime_site())
    row = {"armed": True, "already": not made, "files": files}
    print(f"DUMMY_WEIGHTS {'already armed' if row['already'] else 'armed'}: the dummy loader fills packed integer "
          f"weights with random codes and every other integer tensor with zeros (stock leaves them as torch.empty "
          f"garbage, which an index kernel reads as a position), scans the floating tensors for non-finite values "
          f"and prints the first uncaptured forward outputs' finiteness", flush=True)
    return row


def load_dummy_enabled() -> bool:
    return _flag_on(LOAD_DUMMY_ENV)


def _dummy_model_dir(model_dir: Path, keep_n: int) -> Path:
    stamp = _scratch_dir(1 << 20) / f"dummy-k{keep_n}"
    if stamp.exists():
        shutil.rmtree(stamp, ignore_errors=True)
    stamp.mkdir(parents=True, exist_ok=True)
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        out = stamp / path.name
        if path.name == "config.json":
            value = read_json(path)
            value["text_config"]["num_experts"] = keep_n
            out.write_text(json.dumps(value, indent=2), encoding="utf-8")
        elif path.suffix == ".safetensors":
            continue
        else:
            try:
                os.symlink(path, out)
            except OSError:
                shutil.copy2(path, out)
    print(f"LOAD_DUMMY experts kept {keep_n}: {stamp} carries the cut config and no weight shard", flush=True)
    return stamp


def _flag_on(name: str) -> bool:
    said = str(os.environ.get(name) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be a boolean, found {said!r}")


def _chosen_environment(name: str, allowed: tuple[str, ...]) -> str:
    said = str(os.environ.get(name) or "").strip().lower()
    if said and said not in allowed:
        raise RuntimeError(f"{name} must be unset or one of {list(allowed)}, found {said!r}")
    return said


def serving_flags() -> list[str]:
    flags: list[str] = []
    ssm_dtype = _chosen_environment(MAMBA_SSM_DTYPE_ENV, MAMBA_SSM_DTYPES)
    if ssm_dtype:
        flags.extend(["--mamba-ssm-cache-dtype", ssm_dtype])
    indexer_dtype = _chosen_environment(INDEXER_KV_DTYPE_ENV, INDEXER_KV_DTYPES)
    attention: dict[str, Any] = {"indexer_kv_dtype": indexer_dtype} if indexer_dtype else {}
    host_kv = host_kv_settings()
    if host_kv is not None:
        attention["hisparse_config"] = ({"device_buffer_size": host_kv["device_buffer_size"]}
                                        if host_kv["device_buffer_size"] else {})
    if attention:
        flags.extend(["--attention-config", json.dumps(attention, separators=(",", ":"))])
    if _flag_on(ITERATION_LOG_ENV):
        flags.append("--enable-logging-iteration-details")
    profile_dir = str(os.environ.get(PROFILER_DIR_ENV) or "").strip()
    if profile_dir:
        if not profile_dir.startswith("/"):
            raise RuntimeError(f"{PROFILER_DIR_ENV} must be an absolute path, found {profile_dir!r}")
        steps, _ = _bounded_integer_environment(PROFILER_STEPS_ENV, default=60, minimum=1, maximum=2000)
        flags.extend(["--profiler-config", json.dumps(
            {"profiler": "torch", "torch_profiler_dir": profile_dir,
             "torch_profiler_with_stack": _flag_on(PROFILER_STACK_ENV),
             "ignore_frontend": True, "max_iterations": steps}, separators=(",", ":"))])
    return flags


def draft_vocab_pad() -> int | None:
    said = str(os.environ.get(DRAFT_VOCAB_PAD_ENV) or "").strip()
    if not said:
        return None
    if not (said.isascii() and said.isdigit()) or not 1 <= int(said) <= DRAFT_VOCAB_PAD_MAX:
        raise RuntimeError(
            f"{DRAFT_VOCAB_PAD_ENV} must be an integer from 1 through {DRAFT_VOCAB_PAD_MAX}, "
            f"found {said!r}")
    return int(said)


def unpinned_model_dir() -> Path:
    global MODEL_HF_REPO, MODEL_HF_REVISION, MODEL_KAGGLE_PATH
    global MODEL_CONFIG_SHA256, MODEL_QUANT_CONFIG_SHA256
    if not (MODEL_KAGGLE_PATH / "config.json").is_file():
        found = [one.parent for one in sorted(MODEL_KAGGLE_PATH.rglob("config.json"))
                 if (one.parent / "model.safetensors.index.json").is_file()]
        if len(found) != 1:
            raise FileNotFoundError(
                f"TAAF_MODEL_UNPINNED is set and {MODEL_KAGGLE_PATH} holds "
                f"{len(found)} checkpoint(s), not one."
            )
        print(f"MODEL_UNPINNED found the checkpoint under {found[0]}", flush=True)
        MODEL_KAGGLE_PATH = found[0]
    config_path = MODEL_KAGGLE_PATH / "config.json"
    quant_path = MODEL_KAGGLE_PATH / "hf_quant_config.json"
    MODEL_HF_REPO = str(os.environ.get("TAAF_MODEL_REPO") or MODEL_KAGGLE_PATH.name)
    MODEL_HF_REVISION = "unpinned"
    MODEL_CONFIG_SHA256 = sha256_file(config_path)
    if quant_path.is_file():
        MODEL_QUANT_CONFIG_SHA256 = sha256_file(quant_path)
    held = [one for one in MODEL_KAGGLE_PATH.rglob("*") if one.is_file()]
    print(
        f"MODEL_UNPINNED serving {MODEL_KAGGLE_PATH} as {MODEL_HF_REPO}: "
        f"{len(held)} file(s), {sum(one.stat().st_size for one in held)} byte(s), "
        f"config sha256 {MODEL_CONFIG_SHA256}, quant config sha256 "
        f"{MODEL_QUANT_CONFIG_SHA256}",
        flush=True,
    )
    return MODEL_KAGGLE_PATH


def resolve_model_dir() -> Path:
    manifest_path = MODEL_KAGGLE_PATH / "MODEL_MANIFEST.json"
    if MODEL_UNPINNED and not manifest_path.is_file():
        return unpinned_model_dir()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected mounted {MODEL_HF_REPO}@{MODEL_HF_REVISION} at "
            f"{MODEL_KAGGLE_PATH}."
        )
    value = read_json(manifest_path)
    if (
        not isinstance(value, dict)
        or value.get("repo") != MODEL_HF_REPO
        or value.get("revision") != MODEL_HF_REVISION
    ):
        raise RuntimeError(f"Mounted model identity is wrong at {MODEL_KAGGLE_PATH}.")
    return MODEL_KAGGLE_PATH


def verify_unpinned_model(model_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    config_path = model_dir / "config.json"
    quant_path = model_dir / "hf_quant_config.json"
    config = read_json(config_path)
    quant = read_json(quant_path) if quant_path.is_file() else {}
    text_config = config.get("text_config") or {}
    quantization = config.get("quantization_config") or {}
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise RuntimeError(f"Wrong model architecture: {config.get('architectures')}")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Incomplete checkpoint; missing {name}")
    layers = quantization.get("quantized_layers") or {}
    algos = sorted({str(row.get("quant_algo")) for row in layers.values()
                    if isinstance(row, dict)} | {str(quantization.get("quant_algo"))})
    notes = []
    if config.get("model_type") != "qwen4_exp":
        notes.append(f"model type {config.get('model_type')!r}, not 'qwen4_exp'")
    if quantization.get("quant_method") != "modelopt":
        notes.append(f"quant method {quantization.get('quant_method')!r}, not 'modelopt'")
    if "NVFP4" not in algos:
        notes.append(f"quantisation {algos}, with no NVFP4 among them")
    if text_config.get("ple_embedding_dtype") != "float8_e4m3fn":
        notes.append(f"PLE embedding dtype {text_config.get('ple_embedding_dtype')!r}, "
                     f"not 'float8_e4m3fn'")
    if text_config.get("ple_layer_ids") != [2]:
        notes.append(f"PLE layer ids {text_config.get('ple_layer_ids')!r}, not [2]")
    if int(text_config.get("mtp_num_hidden_layers", -1)) != 1:
        notes.append(f"{text_config.get('mtp_num_hidden_layers')!r} MTP layer(s), not 1")
    if not quant:
        notes.append("no hf_quant_config.json beside the weights")
    held = [one for one in model_dir.rglob("*") if one.is_file()]
    total = sum(one.stat().st_size for one in held)
    print(
        f"MODEL_UNPINNED shape of {model_dir}: {config['architectures'][0]}, "
        f"quantisation {algos}, {len(held)} file(s), {total} byte(s)",
        flush=True,
    )
    for said in notes:
        print(f"MODEL_UNPINNED note: {said}; the load decides, not this check", flush=True)
    return {
        "manifest_sha256": "",
        "config_sha256": MODEL_CONFIG_SHA256,
        "quant_config_sha256": MODEL_QUANT_CONFIG_SHA256,
        "file_count": len(held),
        "total_bytes": total,
        "architecture": config["architectures"][0],
        "quant_algo": str(quantization.get("quant_algo") or ""),
        "quant_algos_seen": algos,
        "ple_embedding_dtype": text_config.get("ple_embedding_dtype"),
        "mtp_num_hidden_layers": text_config.get("mtp_num_hidden_layers"),
        "unpinned_notes": notes,
        "payload_sha256_verified": False,
        "hashed_file_count": 0,
        "hashed_bytes": 0,
        "stat_only_weight_file_count": 0,
        "stat_only_weight_bytes": 0,
        "payload_files_checked": 0,
        "payload_check_deferred_to_vllm_load": True,
        "verification_mode": "unpinned",
        "verify_seconds": time.monotonic() - started,
    }


def verify_model(model_dir: Path, *, full_file_hashes: bool = True) -> dict[str, Any]:
    manifest_path = model_dir / "MODEL_MANIFEST.json"
    if MODEL_UNPINNED and not manifest_path.is_file():
        return verify_unpinned_model(model_dir)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError(
            f"Model manifest hash mismatch: {actual_manifest_sha} != {MODEL_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("repo") != MODEL_HF_REPO or manifest.get("revision") != MODEL_HF_REVISION:
        raise RuntimeError("Mounted model provenance does not match the pinned checkpoint.")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != MODEL_FILE_COUNT:
        raise RuntimeError(f"Expected {MODEL_FILE_COUNT} model files, found {len(files or [])}.")
    try:
        total = sum(int(row["size"]) for row in files)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Model manifest file metadata is invalid.") from exc
    if total != MODEL_TOTAL_BYTES or total != int(manifest.get("total_bytes", -1)):
        raise RuntimeError(f"Model byte total mismatch: {total} != {MODEL_TOTAL_BYTES}")

    hashed_files = 0
    hashed_bytes = 0
    stat_only_weight_files = 0
    stat_only_weight_bytes = 0
    payload_files_checked = 0
    started = time.monotonic()
    if full_file_hashes:
        for index, row in enumerate(files, start=1):
            relative = str(row["path"])
            path = model_dir / relative
            expected_size = int(row["size"])
            if not path.is_file() or path.stat().st_size != expected_size:
                raise RuntimeError(f"Missing or wrong-sized model file: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != str(row["sha256"]):
                raise RuntimeError(f"Model file hash mismatch: {relative}")
            hashed_files += 1
            hashed_bytes += expected_size
            payload_files_checked += 1
            if index == len(files) or index % 25 == 0:
                print(
                    f"MODEL_VERIFY files={index}/{len(files)} bytes={hashed_bytes} "
                    f"elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    else:
        print(
            f"MODEL_IDENTITY_ONLY files={len(files)} bytes={total} "
            "payload_check=vllm_load",
            flush=True,
        )

    config_path = model_dir / "config.json"
    quant_path = model_dir / "hf_quant_config.json"
    if sha256_file(config_path) != MODEL_CONFIG_SHA256:
        raise RuntimeError("Pinned model config hash does not match.")
    if sha256_file(quant_path) != MODEL_QUANT_CONFIG_SHA256:
        raise RuntimeError("Pinned ModelOpt quantization config hash does not match.")
    config = read_json(config_path)
    quant = read_json(quant_path)
    text_config = config.get("text_config") or {}
    quantization = config.get("quantization_config") or {}
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise RuntimeError(f"Wrong model architecture: {config.get('architectures')}")
    if config.get("model_type") != "qwen4_exp":
        raise RuntimeError(f"Wrong model type: {config.get('model_type')}")
    if (
        quantization.get("quant_method") != "modelopt"
        or quantization.get("quant_algo") != "NVFP4"
        or (quantization.get("producer") or {}).get("version") != "0.46.0"
    ):
        raise RuntimeError(f"Wrong in-model quantization identity: {quantization}")
    quant_details = quant.get("quantization") or {}
    if quant_details.get("quant_algo") != "NVFP4" or int(quant_details.get("group_size", -1)) != 16:
        raise RuntimeError(f"Wrong ModelOpt NVFP4 details: {quant_details}")
    if (
        text_config.get("ple_embedding_dtype") != "float8_e4m3fn"
        or text_config.get("ple_layer_ids") != [2]
        or int(text_config.get("mtp_num_hidden_layers", -1)) != 1
        or int((text_config.get("mtp") or {}).get("num_hidden_layers", -1)) != 1
    ):
        raise RuntimeError("The pinned FP8 PLE or native MTP identity does not match.")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Incomplete checkpoint; missing {name}")
    return {
        "manifest_sha256": actual_manifest_sha,
        "config_sha256": MODEL_CONFIG_SHA256,
        "quant_config_sha256": MODEL_QUANT_CONFIG_SHA256,
        "file_count": len(files),
        "total_bytes": total,
        "architecture": config["architectures"][0],
        "quant_algo": quantization["quant_algo"],
        "ple_embedding_dtype": text_config["ple_embedding_dtype"],
        "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"],
        "payload_sha256_verified": full_file_hashes,
        "hashed_file_count": hashed_files,
        "hashed_bytes": hashed_bytes,
        "stat_only_weight_file_count": stat_only_weight_files,
        "stat_only_weight_bytes": stat_only_weight_bytes,
        "payload_files_checked": payload_files_checked,
        "payload_check_deferred_to_vllm_load": not full_file_hashes,
        "verification_mode": "full" if full_file_hashes else "identity-only",
        "verify_seconds": time.monotonic() - started,
    }


def _apply_whiteouts(expected_markers: tuple[str, ...] | None = None) -> list[str]:
    applied: list[str] = []
    if expected_markers is None:
        whiteouts = sorted(
            RUNTIME_ROOT.rglob(".wh.*"),
            key=lambda path: len(path.parts),
            reverse=True,
        )
    else:
        whiteouts = [RUNTIME_ROOT / relative for relative in expected_markers]
        missing = [str(path) for path in whiteouts if not os.path.lexists(path)]
        if missing:
            raise RuntimeError(f"Expected pinned runtime whiteouts are missing: {missing}")
    for marker in whiteouts:
        if marker.name == ".wh..wh..opq":
            raise RuntimeError(f"Unsupported opaque runtime whiteout: {marker}")
        target = marker.with_name(marker.name[4:])
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)
        applied.append(str(marker.relative_to(RUNTIME_ROOT)))
        marker.unlink(missing_ok=True)
    return applied


def _clear_listed_whiteouts(row: dict[str, Any]) -> list[str]:
    cleared: list[str] = []
    for entry in row.get("whiteouts") or ():
        kind = str(entry.get("kind") or "")
        relative = str(entry.get("path") or "")
        parts = Path(relative).parts
        if not relative or relative.startswith("/") or ".." in parts:
            raise RuntimeError(f"Unsafe runtime whiteout: {entry}")
        if kind not in {"whiteout", "opaque"}:
            raise RuntimeError(f"Unknown runtime whiteout kind: {entry}")
        target = RUNTIME_ROOT / relative
        if kind == "whiteout":
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        elif target.is_dir() and not target.is_symlink():
            for child in target.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        cleared.append(f"{kind}:{relative}")
    return cleared


def validate_runtime_storage() -> dict[str, Any]:
    roots = (RUNTIME_ROOT, CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT)
    rows: list[dict[str, Any]] = []
    devices: set[int] = set()
    for root in roots:
        if root.is_symlink():
            raise RuntimeError(f"A runtime or cache root must not be a symlink: {root}")
        probe = root.parent
        if not probe.is_dir() or probe.is_symlink():
            raise RuntimeError(f"The runtime or cache parent is not a regular directory: {probe}")
        stat = probe.stat()
        if root.exists() and root.stat().st_dev != stat.st_dev:
            raise RuntimeError(f"A runtime or cache root crossed its audited filesystem: {root}")
        filesystem = os.statvfs(probe)
        free_bytes = int(filesystem.f_bavail * filesystem.f_frsize)
        devices.add(int(stat.st_dev))
        rows.append(
            {
                "root": str(root),
                "statvfs_path": str(probe.resolve()),
                "device": int(stat.st_dev),
                "free_bytes": free_bytes,
                "threshold_bytes": MIN_RUNTIME_FILESYSTEM_FREE_BYTES,
            }
        )
        if free_bytes < MIN_RUNTIME_FILESYSTEM_FREE_BYTES:
            raise RuntimeError(
                f"The filesystem for {root} has too little free space: "
                f"{free_bytes} < {MIN_RUNTIME_FILESYSTEM_FREE_BYTES}."
            )
    if len(devices) != 1:
        raise RuntimeError(
            "The pinned runtime and JIT/cache roots do not share one audited filesystem: "
            f"{rows}"
        )
    return {
        "roots": rows,
        "shared_device": next(iter(devices)),
        "minimum_free_bytes": MIN_RUNTIME_FILESYSTEM_FREE_BYTES,
        "conservative_extracted_bound_bytes": RUNTIME_EXTRACTED_BOUND_BYTES,
        "jit_cache_reserve_bytes": RUNTIME_CACHE_RESERVE_BYTES,
        "audit_sha256": RUNTIME_STORAGE_AUDIT_SHA256,
    }


def zstd_layers(runtime_dir: Path) -> dict[int, dict[str, Any]] | None:
    path = runtime_dir / ZSTD_MANIFEST_NAME
    if not path.is_file():
        return None
    actual = sha256_file(path)
    if actual != ZSTD_MANIFEST_SHA256:
        raise RuntimeError(f"zstd runtime manifest hash mismatch: {actual} != {ZSTD_MANIFEST_SHA256}")
    held = read_json(path)
    if held.get("index_digest") != VLLM_INDEX_DIGEST or held.get("format") != "zstd":
        raise RuntimeError("zstd runtime manifest names another image or format.")
    return {int(row["index"]): row for row in held.get("layers") or ()}


def zstd_decompressor() -> tuple[str, Any]:
    try:
        import zstandard

        return "zstandard", zstandard
    except ImportError:
        pass
    for wheel in sorted((BUNDLE_DIR / "wheels").glob("zstandard-*.whl")):
        landing = Path("/tmp/zstandard-wheel")
        shutil.rmtree(landing, ignore_errors=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(landing)
        sys.path.insert(0, str(landing))
        try:
            import zstandard

            return f"zstandard from {wheel.name}", zstandard
        except ImportError as exc:
            print(f"ZSTD wheel {wheel.name} did not import: {exc}", flush=True)
            sys.path.remove(str(landing))
    if shutil.which("zstd"):
        return "zstd binary", None
    raise RuntimeError("No zstd decompressor: no zstandard module, no bundled wheel that "
                       "imports, and no zstd binary on PATH.")


def flat_chunks(runtime_dir: Path) -> dict[str, Any] | None:
    path = runtime_dir / FLAT_MANIFEST_NAME
    if not FLAT_MANIFEST_SHA256 or not path.is_file():
        return None
    actual = sha256_file(path)
    if actual != FLAT_MANIFEST_SHA256:
        raise RuntimeError(f"flat runtime manifest hash mismatch: {actual} != {FLAT_MANIFEST_SHA256}")
    held = read_json(path)
    if held.get("format") != "flat-zstd" or held.get("index_digest") != VLLM_INDEX_DIGEST:
        raise RuntimeError("flat runtime manifest names another image or format.")
    if held.get("runtime_manifest_sha256") != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("flat runtime was not built from the pinned runtime manifest.")
    built = {int(row["index"]): (row["sha256"], int(row["size"]))
             for row in held.get("source_layers") or ()}
    if built != EXPECTED_RUNTIME_LAYERS:
        raise RuntimeError("flat runtime was not built from the pinned runtime layers.")
    return held


def drop_page_cache(path: Path) -> None:
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(handle)


def extract_flat_chunk(runtime_dir: Path, row: dict[str, Any], verify: bool) -> float:
    began = time.monotonic()
    path = runtime_dir / str(row["file"])
    if not path.is_file() or path.stat().st_size != int(row["size"]):
        raise RuntimeError(f"Missing or wrong-sized flat runtime chunk: {path}")
    _, module = zstd_decompressor()
    tar = subprocess.Popen(["tar", "-xf", "-", "-C", str(RUNTIME_ROOT), "--no-same-owner"],
                           stdin=subprocess.PIPE)
    digest = hashlib.sha256() if verify else None
    total = 0
    read_s = write_s = 0.0
    helper = None
    try:
        if module is not None:
            handle = open(path, "rb")
            source = module.ZstdDecompressor().stream_reader(handle, read_size=ZSTD_CHUNK)
        else:
            helper = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
            handle, source = None, helper.stdout
        while True:
            lap = time.perf_counter()
            block = source.read(ZSTD_CHUNK)
            read_s += time.perf_counter() - lap
            if not block:
                break
            if digest is not None:
                digest.update(block)
            total += len(block)
            lap = time.perf_counter()
            tar.stdin.write(block)
            write_s += time.perf_counter() - lap
        if handle is not None:
            handle.close()
    finally:
        tar.stdin.close()
    if helper is not None and helper.wait() != 0:
        raise RuntimeError(f"zstd failed on {path.name}")
    if tar.wait() != 0:
        raise RuntimeError(f"tar failed on {path.name}")
    if total != int(row["tar_bytes"]):
        raise RuntimeError(f"flat chunk {path.name} gave {total} bytes, not {row['tar_bytes']}")
    if digest is not None and digest.hexdigest() != row["tar_sha256"]:
        raise RuntimeError(f"flat chunk {path.name} content hash mismatch")
    drop_page_cache(path)
    FLAT_CHUNK_STATS.append({"file": str(row["file"]), "stage": str(row.get("stage")), "compressed": int(row["size"]),
                             "tar_bytes": total, "seconds": round(time.monotonic() - began, 2),
                             "read_and_decompress_s": round(read_s, 2), "tar_write_s": round(write_s, 2)})
    return time.monotonic() - began


def extract_flat_runtime(runtime_dir: Path, flat: dict[str, Any], verify: bool) -> list[dict[str, Any]]:
    chunks = list(flat.get("chunks") or ())
    stages = [[row for row in chunks if row.get("stage") == stage]
              for stage in ("skeleton", "data", "links")]
    if sum(len(stage) for stage in stages) != len(chunks):
        raise RuntimeError("flat runtime manifest names a chunk stage this setup does not know")
    done: list[dict[str, Any]] = []
    for name, rows in zip(("skeleton", "data", "links"), stages):
        began = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rows))) as pool:
            spent = list(pool.map(lambda row: extract_flat_chunk(runtime_dir, row, verify), rows))
        print(f"RUNTIME_FLAT stage={name} chunks={len(rows)} "
              f"bytes={sum(int(row['tar_bytes']) for row in rows)} "
              f"seconds={time.monotonic() - began:.1f} slowest_chunk={max(spent or [0]):.1f} "
              f"pages_dropped_bytes={sum(int(row.get('size') or 0) for row in rows)}",
              flush=True)
        stats = [held for held in FLAT_CHUNK_STATS if held["stage"] == name]
        if stats and init_trace_on():
            phase_span(f"runtime unpack stage {name}", time.time() - (time.monotonic() - began), chunks=len(rows),
                       compressed_bytes=sum(held["compressed"] for held in stats),
                       tar_bytes=sum(held["tar_bytes"] for held in stats),
                       read_and_decompress_s_sum=round(sum(held["read_and_decompress_s"] for held in stats), 1),
                       tar_write_s_sum=round(sum(held["tar_write_s"] for held in stats), 1),
                       slowest_chunk_s=max(held["seconds"] for held in stats))
            print(f"RUNTIME_FLAT stage={name} per-chunk sums: read+decompress "
                  f"{sum(held['read_and_decompress_s'] for held in stats):.1f} s, tar write "
                  f"{sum(held['tar_write_s'] for held in stats):.1f} s over {len(stats)} chunks "
                  f"({sum(held['compressed'] for held in stats) / 2**30:.2f} GiB read from the mount)", flush=True)
        done += [{"file": row["file"], "stage": name, "tar_sha256": row["tar_sha256"],
                  "payload_sha256_verified": verify} for row in rows]
    return done


def extract_zstd_layer(path: Path, row: dict[str, Any], verify: bool) -> str:
    kind, module = zstd_decompressor()
    tar = subprocess.Popen(["tar", "-xf", "-", "-C", str(RUNTIME_ROOT), "--no-same-owner",
                            *TAR_EXCLUDES], stdin=subprocess.PIPE)
    digest = hashlib.sha256() if verify else None
    total = 0
    helper = None
    try:
        if module is not None:
            handle = open(path, "rb")
            source = module.ZstdDecompressor().stream_reader(handle, read_size=ZSTD_CHUNK)
        else:
            helper = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
            handle, source = None, helper.stdout
        for block in iter(lambda: source.read(ZSTD_CHUNK), b""):
            if digest is not None:
                digest.update(block)
            total += len(block)
            tar.stdin.write(block)
        if handle is not None:
            handle.close()
    finally:
        tar.stdin.close()
    if helper is not None and helper.wait() != 0:
        raise RuntimeError(f"zstd failed on {path.name}")
    if tar.wait() != 0:
        raise RuntimeError(f"tar failed on {path.name}")
    if total != int(row["tar_bytes"]):
        raise RuntimeError(f"zstd layer {path.name} gave {total} bytes, not {row['tar_bytes']}")
    if digest is not None and digest.hexdigest() != row["tar_sha256"]:
        raise RuntimeError(f"zstd layer {path.name} content hash mismatch")
    return kind


def verify_and_extract_runtime(
    runtime_dir: Path,
    *,
    full_layer_hashes: bool = True,
    scan_extracted_caches: bool = True,
) -> dict[str, Any]:
    manifest_path = runtime_dir / "runtime-manifest.json"
    actual_sha = sha256_file(manifest_path)
    if actual_sha != VLLM_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(
            f"Runtime manifest hash mismatch: {actual_sha} != {VLLM_RUNTIME_MANIFEST_SHA256}"
        )
    manifest = read_json(manifest_path)
    expected_header = {
        "image": VLLM_IMAGE,
        "index_digest": VLLM_INDEX_DIGEST,
        "amd64_manifest_digest": VLLM_AMD64_MANIFEST_DIGEST,
        "config_digest": VLLM_CONFIG_DIGEST,
        "selected_compressed_bytes": RUNTIME_SELECTED_BYTES,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Wrong vLLM runtime {key}: {manifest.get(key)!r} != {expected!r}")
    layers = manifest.get("selected_layers")
    if not isinstance(layers, list) or len(layers) != len(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError("Runtime manifest has the wrong selected layer count.")
    layer_map = {int(row["index"]): row for row in layers}
    if set(layer_map) != set(EXPECTED_RUNTIME_LAYERS):
        raise RuntimeError(f"Wrong runtime layer indexes: {sorted(layer_map)}")

    shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
    RUNTIME_ROOT.mkdir(parents=True)
    started = time.monotonic()
    verified: list[dict[str, Any]] = []
    whiteouts: list[str] = []
    flat = flat_chunks(runtime_dir)
    zstd = None if flat is not None else zstd_layers(runtime_dir)
    decompressor = "gzip"
    if flat is not None:
        verified = extract_flat_runtime(runtime_dir, flat, full_layer_hashes)
        whiteouts = list(flat.get("whiteouts_applied") or ())
        decompressor = "flat-zstd"
    for index in sorted(EXPECTED_RUNTIME_LAYERS) if flat is None else ():
        expected_hash, expected_size = EXPECTED_RUNTIME_LAYERS[index]
        row = layer_map[index]
        if (
            row.get("sha256") != expected_hash
            or row.get("digest") != f"sha256:{expected_hash}"
            or int(row.get("size", -1)) != expected_size
        ):
            raise RuntimeError(f"Runtime layer {index} identity does not match: {row}")
        if zstd is not None:
            packed = zstd.get(index) or {}
            if (packed.get("source_sha256") != expected_hash
                    or int(packed.get("source_size", -1)) != expected_size):
                raise RuntimeError(f"zstd runtime layer {index} does not come from the pinned layer")
            layer_path = runtime_dir / f"{packed['file']}.blob"
            if not layer_path.is_file():
                layer_path = runtime_dir / str(packed["file"])
            if not layer_path.is_file() or layer_path.stat().st_size != int(packed["size"]):
                raise RuntimeError(f"Missing or wrong-sized zstd runtime layer: {layer_path}")
            layer_began = time.monotonic()
            decompressor = extract_zstd_layer(layer_path, packed, full_layer_hashes)
            print(f"RUNTIME_LAYER_ZSTD index={index} bytes={packed['tar_bytes']} "
                  f"seconds={time.monotonic() - layer_began:.1f} with={decompressor}", flush=True)
            expected_whiteouts = (
                None if full_layer_hashes else FAST_RUNTIME_WHITEOUTS.get(index, ())
            )
            whiteouts.extend(_apply_whiteouts(expected_whiteouts))
            verified.append({"index": index, "file": str(row["file"]),
                             "stored_file": layer_path.name, "sha256": expected_hash,
                             "manifest_sha256": expected_hash,
                             "tar_sha256": packed["tar_sha256"],
                             "payload_sha256_verified": full_layer_hashes,
                             "size": expected_size, "compression": "zstd"})
            continue
        layer_path = runtime_dir / str(row["file"])
        if not layer_path.is_file():
            layer_path = runtime_dir / f"{row['file']}.blob"
        if not layer_path.is_file() or layer_path.stat().st_size != expected_size:
            raise RuntimeError(f"Missing or wrong-sized runtime layer: {layer_path}")
        actual_layer_sha = None
        if full_layer_hashes:
            actual_layer_sha = sha256_file(layer_path)
            if actual_layer_sha != expected_hash:
                raise RuntimeError(f"Runtime layer hash mismatch: {layer_path.name}")
            print(
                f"RUNTIME_LAYER_VERIFY index={index} bytes={expected_size} "
                f"sha256={actual_layer_sha}",
                flush=True,
            )
        else:
            print(
                f"RUNTIME_LAYER_FAST_CHECK index={index} bytes={expected_size} "
                f"manifest_sha256={expected_hash}",
                flush=True,
            )
        cleared = _clear_listed_whiteouts(row) if RUNTIME_WHITEOUTS_LISTED else []
        layer_began = time.monotonic()
        subprocess.run(
            [
                "tar",
                "-xzf",
                str(layer_path),
                "-C",
                str(RUNTIME_ROOT),
                "--no-same-owner",
                "--exclude=*/__pycache__",
                "--exclude=*/__pycache__/*",
                "--exclude=*.pyc",
                "--exclude=*.pyo",
                *(("--exclude=.wh.*", "--exclude=*/.wh.*")
                  if RUNTIME_WHITEOUTS_LISTED else ()),
            ],
            check=True,
        )
        if RUNTIME_WHITEOUTS_LISTED:
            print(f"RUNTIME_LAYER_GZIP index={index} bytes={expected_size} "
                  f"seconds={time.monotonic() - layer_began:.1f} "
                  f"whiteouts_cleared={len(cleared)}", flush=True)
            whiteouts.extend(cleared)
        else:
            expected_whiteouts = (
                None if full_layer_hashes else FAST_RUNTIME_WHITEOUTS.get(index, ())
            )
            whiteouts.extend(_apply_whiteouts(expected_whiteouts))
        verified.append(
            {
                "index": index,
                "file": str(row["file"]),
                "stored_file": layer_path.name,
                "sha256": expected_hash,
                "manifest_sha256": expected_hash,
                "actual_sha256": actual_layer_sha,
                "payload_sha256_verified": full_layer_hashes,
                "size": expected_size,
            }
        )
    if scan_extracted_caches:
        cache_artifacts = [
            path
            for path in RUNTIME_ROOT.rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        ]
        if cache_artifacts:
            raise RuntimeError(
                f"Runtime extraction produced Python cache files: {cache_artifacts[:20]}"
            )
    return {
        "manifest_sha256": actual_sha,
        "image": VLLM_IMAGE,
        "index_digest": VLLM_INDEX_DIGEST,
        "amd64_manifest_digest": VLLM_AMD64_MANIFEST_DIGEST,
        "layers": verified,
        "whiteouts_applied": whiteouts,
        "payload_sha256_verified": full_layer_hashes,
        "cache_scan_performed": scan_extracted_caches,
        "verification_mode": "full" if full_layer_hashes else "fast",
        "compression": "flat-zstd" if flat is not None else "gzip" if zstd is None else "zstd",
        "decompressor": decompressor,
        "extract_seconds": time.monotonic() - started,
    }


def patch_expert_counter() -> dict[str, Any]:
    if not _flag_on(COUNT_EXPERTS_ENV):
        return {"armed": False}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    target = site / ROUTED_EXPERTS_RELATIVE
    if not target.is_file() or target.is_symlink():
        raise RuntimeError(f"Missing regular routing source: {target}")
    source = BUNDLE_DIR / COUNTER_NAME
    if not source.is_file():
        raise RuntimeError(f"{COUNT_EXPERTS_ENV} was set but {source} is absent")
    namespace: dict[str, Any] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    stock = target.read_bytes()
    if b"_taaf_note_experts" in stock:
        print("EXPERT_COUNTER already armed", flush=True)
        return {"armed": True, "already": True,
                "target": ROUTED_EXPERTS_RELATIVE}
    before = hashlib.sha256(stock).hexdigest()
    made = namespace["patched"](stock)
    after = hashlib.sha256(made).hexdigest()
    if after == before:
        raise RuntimeError("the expert counter changed nothing")
    compile(made.decode("utf-8"), str(target), "exec")
    temporary = target.with_name(f".{target.name}.counter.tmp")
    temporary.write_bytes(made)
    os.replace(temporary, target)
    live = hashlib.sha256(target.read_bytes()).hexdigest()
    if live != after:
        raise RuntimeError(f"the counted routing source did not land: {live}")
    print(f"EXPERT_COUNTER armed on {ROUTED_EXPERTS_RELATIVE} "
          f"{before[:16]} -> {after[:16]}", flush=True)
    return {"armed": True, "stock_sha256": before, "counted_sha256": after,
            "target": ROUTED_EXPERTS_RELATIVE}


def patch_moe_route() -> dict[str, Any]:
    if not (_flag_on(MOE_ROUTE_ENV) or _flag_on(TRACE_REPLAY_ENV)):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{MOE_ROUTE_ENV} and {TRACE_REPLAY_ENV} edit vLLM 0.30.0's files; the "
            f"{RUNTIME_PROFILE} runtime ships other ones.")
    source = BUNDLE_DIR / MOE_ROUTE_SETUP
    if not source.is_file():
        raise RuntimeError(f"{MOE_ROUTE_ENV} or {TRACE_REPLAY_ENV} was set but {source} is absent")
    namespace: dict[str, Any] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    return namespace["apply"](site, BUNDLE_DIR,
                              int(resolve_vllm_tuning()["mtp_speculative_tokens"]))


def patch_moe_share() -> dict[str, Any]:
    if not _flag_on(MOE_SHARE_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{MOE_SHARE_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships "
            f"other ones.")
    source = BUNDLE_DIR / MOE_SHARE_SETUP
    if not source.is_file():
        raise RuntimeError(f"{MOE_SHARE_ENV} was set but {source} is absent")
    namespace: dict[str, Any] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    row = namespace["apply"](site, BUNDLE_DIR)
    said = str(os.environ.get(MOE_SHARE_SETTINGS_ENV) or "").strip()
    if not said:
        return row
    held = json.loads(said)
    if not isinstance(held, dict):
        raise RuntimeError(f"{MOE_SHARE_SETTINGS_ENV} must be a JSON object, found {said!r}")
    target = str(os.environ.get(MOE_ROUTE_FILE_ENV) or "").strip()
    if not target:
        raise RuntimeError(f"{MOE_SHARE_SETTINGS_ENV} needs {MOE_ROUTE_FILE_ENV} to name the control file")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(held, sort_keys=True), encoding="utf-8")
    print(f"MOE_SHARE settings at start: {json.dumps(held, sort_keys=True)} written to {path}", flush=True)
    return {**row, "start_settings": held} if isinstance(row, dict) else row


def patch_scheduler_spec_reserve() -> dict[str, Any]:
    if not _flag_on(SCHED_SPEC_RESERVE_ENV):
        return {"armed": False}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    target = site / SCHEDULER_RELATIVE
    if not target.is_file() or target.is_symlink():
        raise RuntimeError(f"Missing regular scheduler source: {target}")
    stock = target.read_text(encoding="utf-8")
    if "_spec_decode_step_blocks" in stock:
        print("SCHED_SPEC_RESERVE: this build already has it; nothing applied",
              flush=True)
        return {"armed": True, "already": True, "target": SCHEDULER_RELATIVE}
    edits = ((SCHED_IMPORT_ANCHOR, SCHED_IMPORT_LINE + SCHED_IMPORT_ANCHOR),
             (SCHED_RESERVE_FROM, SCHED_RESERVE_TO),
             (SCHED_REMAINING_FROM, SCHED_REMAINING_TO),
             (SCHED_HELPER_FROM, SCHED_HELPER_TO))
    for found, _ in edits:
        if stock.count(found) != 1:
            raise RuntimeError(
                f"{SCHED_SPEC_RESERVE_ENV} cannot apply: an anchor appears "
                f"{stock.count(found)} time(s), not once. This build differs "
                f"from the one the patch was written against; refusing rather "
                f"than editing the wrong place.")
    before = hashlib.sha256(stock.encode("utf-8")).hexdigest()
    made = stock
    for found, put in edits:
        made = made.replace(found, put, 1)
    compile(made, str(target), "exec")
    after = hashlib.sha256(made.encode("utf-8")).hexdigest()
    temporary = target.with_name(f".{target.name}.sched.tmp")
    temporary.write_text(made, encoding="utf-8")
    os.replace(temporary, target)
    live = hashlib.sha256(target.read_bytes()).hexdigest()
    if live != after:
        raise RuntimeError(f"the patched scheduler did not land: {live}")
    print(f"SCHED_SPEC_RESERVE armed on {SCHEDULER_RELATIVE} "
          f"{before[:16]} -> {after[:16]}: an async KV load now reserves the "
          f"spec-decode step's blocks at admission", flush=True)
    return {"armed": True, "stock_sha256": before, "patched_sha256": after,
            "target": SCHEDULER_RELATIVE}


def patch_ple_prefill_groups() -> dict[str, Any]:
    if not _flag_on(PLE_GROUP_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "day0":
        print(f"PLE_PREFILL_GROUPS skipped: vLLM {VLLM_VERSION} runs PLE prefill as a "
              f"variable-length kernel with no padded batch to group", flush=True)
        return {"armed": False, "skipped": True, "runtime_profile": RUNTIME_PROFILE}
    beside = Path(__file__).resolve().parent / "ple_group_patch.py"
    if not beside.is_file():
        raise RuntimeError(
            f"{PLE_GROUP_ENV} is armed but {beside.name} did not ship beside "
            f"{Path(__file__).name}; refusing rather than serving unpatched "
            f"while the provenance claims otherwise")
    blocks: dict[str, Any] = {}
    exec(compile(beside.read_text(encoding="utf-8"), str(beside), "exec"), blocks)
    DENSE_BLOCK = blocks["DENSE_BLOCK"]
    GROUPED_BLOCK = blocks["GROUPED_BLOCK"]
    try:
        cap = int(os.environ.get(PLE_GROUP_CAP_ENV, "") or PLE_GROUP_CAP_DEFAULT)
    except ValueError:
        cap = PLE_GROUP_CAP_DEFAULT
    if cap < 1:
        raise RuntimeError(f"{PLE_GROUP_CAP_ENV}={cap} would pad nothing at all")
    GROUP_CAP_LINE = f"_PLE_GROUP_CAP_SLOTS = {cap}" + chr(10) * 3
    print(f"PLE_PREFILL_GROUPS: the padded prefill tensor is capped at {cap} "
          f"slots, about {cap * 13376 / 1048576:.0f} MiB a buffer by the "
          f"geometry the last run measured", flush=True)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    target = site / PLE_LAYER_RELATIVE
    if not target.is_file() or target.is_symlink():
        raise RuntimeError(f"Missing regular PLE layer source: {target}")
    stock = target.read_text(encoding="utf-8")
    _note_applied_edits(((DENSE_BLOCK, GROUPED_BLOCK),
                         (PLE_GROUP_CLASS_ANCHOR, GROUP_CAP_LINE + PLE_GROUP_CLASS_ANCHOR)))
    if "_PLE_GROUP_CAP_SLOTS" in stock:
        print("PLE_PREFILL_GROUPS: already grouped; nothing applied", flush=True)
        return {"armed": True, "already": True, "target": PLE_LAYER_RELATIVE}
    for found in (DENSE_BLOCK, PLE_GROUP_CLASS_ANCHOR):
        if stock.count(found) < 1:
            raise RuntimeError(
                f"{PLE_GROUP_ENV} cannot apply: an anchor is absent. This "
                f"build differs from the one the patch was written against; "
                f"refusing rather than editing the wrong place.")
    if stock.count(DENSE_BLOCK) != 1:
        raise RuntimeError(
            f"{PLE_GROUP_ENV} cannot apply: the dense prefill block appears "
            f"{stock.count(DENSE_BLOCK)} times, not once.")
    before = hashlib.sha256(stock.encode("utf-8")).hexdigest()
    made = stock.replace(DENSE_BLOCK, GROUPED_BLOCK, 1)
    made = made.replace(PLE_GROUP_CLASS_ANCHOR,
                        GROUP_CAP_LINE + PLE_GROUP_CLASS_ANCHOR, 1)
    compile(made, str(target), "exec")
    after = hashlib.sha256(made.encode("utf-8")).hexdigest()
    temporary = target.with_name(f".{target.name}.group.tmp")
    temporary.write_text(made, encoding="utf-8")
    os.replace(temporary, target)
    live = hashlib.sha256(target.read_bytes()).hexdigest()
    if live != after:
        raise RuntimeError(f"the grouped PLE prefill did not land: {live}")
    print(f"PLE_PREFILL_GROUPS armed on {PLE_LAYER_RELATIVE} "
          f"{before[:16]} -> {after[:16]}: the padded prefill tensor is now "
          f"capped instead of growing with the batch's longest request",
          flush=True)
    return {"armed": True, "stock_sha256": before, "patched_sha256": after,
            "target": PLE_LAYER_RELATIVE}


def patch_mtp_own_reads() -> dict[str, Any]:
    if not _flag_on(MTP_READS_ENV):
        return {"armed": False}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    utils = site / WEIGHT_UTILS_RELATIVE
    drafts = [site / relative for relative in MTP_RELATIVES if (site / relative).is_file()]
    if not utils.is_file() or not drafts:
        print(f"MTP_OWN_READS skipped: vLLM {VLLM_VERSION} has no {WEIGHT_UTILS_RELATIVE} or "
              f"no Qwen4Exp draft model, so the draft pass reads every weight file", flush=True)
        return {"armed": False, "skipped": True}
    plans = [(utils, ((WEIGHT_READ_FROM, WEIGHT_READ_TO),
                      (WEIGHT_ITERATOR_ANCHOR, WEIGHT_WANTED_LINE + WEIGHT_ITERATOR_ANCHOR)))]
    plans += [(path, ((MTP_LOAD_FROM, MTP_LOAD_TO),)) for path in drafts]
    made_all = []
    for path, edits in plans:
        stock = path.read_text(encoding="utf-8")
        if "_taaf_wanted" in stock:
            continue
        for found, _ in edits:
            if stock.count(found) != 1:
                print(f"MTP_OWN_READS cannot apply: an anchor of {path.name} appears "
                      f"{stock.count(found)} time(s), not once; the draft pass reads every "
                      f"weight file", flush=True)
                return {"armed": False, "refused": str(path.relative_to(site))}
        made = stock
        for found, put in edits:
            made = made.replace(found, put, 1)
        compile(made, str(path), "exec")
        made_all.append((path, made))
    for path, made in made_all:
        temporary = path.with_name(f".{path.name}.mtp.tmp")
        temporary.write_text(made, encoding="utf-8")
        os.replace(temporary, path)
    print(f"MTP_OWN_READS armed on {len(made_all)} file(s): the draft model's loading pass "
          f"reads only the tensors it keeps", flush=True)
    return {"armed": True, "files": [str(path.relative_to(site)) for path, _ in made_all]}


def patch_expert_hot() -> dict[str, Any]:
    if not _flag_on(EXPERT_HOT_ENV):
        return {"armed": False}
    if not expert_cut_on():
        raise RuntimeError(f"{EXPERT_HOT_ENV} loads the cold experts beside a cut-at-load view and needs "
                           f"{EXPERT_CUT_ENV} armed")
    row = _patch_with_table(EXPERT_HOT_ENV, EXPERT_HOT_HELPER, EXPERT_HOT_TABLE, EXPERT_HOT_INSTALLED)
    print(f"EXPERT_HOT armed: the experts the cut drops are built at load, stored on disk and bound to the tail "
          f"of the KV pool; the router scores all 512 with the cold ones masked, and a switch drains that tail and "
          f"unmasks them without a restart ({EXPERT_HOT_ENV})", flush=True)
    return row


def patch_expert_cut() -> dict[str, Any]:
    if not expert_cut_on():
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{EXPERT_CUT_ENV} wraps vLLM 0.30.0's safetensors reader; the {RUNTIME_PROFILE} "
                           f"runtime ships another one")
    source = BUNDLE_DIR / EXPERT_CUT_HELPER
    if not source.is_file():
        raise RuntimeError(f"{EXPERT_CUT_ENV} was set but {source} is absent")
    helper = source.read_bytes()
    compile(helper.decode("utf-8"), str(source), "exec")
    site = _runtime_site()
    path = site / WEIGHT_UTILS_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular weight loader source: {path}")
    text = path.read_text(encoding="utf-8")
    already = EXPERT_CUT_MARKER in text
    if not already:
        text += EXPERT_CUT_HOOK
        compile(text, str(path), "exec")
    target = site / EXPERT_CUT_INSTALLED
    if not (target.is_file() and target.read_bytes() == helper):
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, target)
    if not already:
        temporary = path.with_name(f".{path.name}.cut.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    print(f"EXPERT_CUT {'already armed' if already else 'armed'}: the safetensors reader of {WEIGHT_UTILS_RELATIVE} "
          f"serves a directory that carries {EXPERT_CUT_PLAN_NAME} the cut tensors read from byte ranges; every "
          f"other directory is read as before", flush=True)
    return {"armed": True, "already": already, "helper_sha256": hashlib.sha256(helper).hexdigest()}


def patch_mtp_weights() -> dict[str, Any]:
    said = str(os.environ.get(MTP_WEIGHTS_ENV) or "").strip()
    if not said:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{MTP_WEIGHTS_ENV} edits vLLM 0.30.0's Qwen4Exp draft model; the "
                           f"{RUNTIME_PROFILE} runtime ships another one")
    source = BUNDLE_DIR / MTP_WEIGHTS_HELPER
    if not source.is_file():
        raise RuntimeError(f"{MTP_WEIGHTS_ENV} was set but {source} is absent")
    helper = source.read_bytes()
    compile(helper.decode("utf-8"), str(source), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / MTP_RELATIVES[0]
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular draft-model source: {path}")
    text = path.read_text(encoding="utf-8")
    already = MTP_WEIGHTS_MARKER in text
    if not already:
        if text.count(MTP_WEIGHTS_FROM) != 1:
            raise RuntimeError(f"{MTP_WEIGHTS_ENV} cannot apply: the remap_weight_names anchor of "
                               f"{MTP_RELATIVES[0]} appears {text.count(MTP_WEIGHTS_FROM)} time(s), not once")
        text = text.replace(MTP_WEIGHTS_FROM, MTP_WEIGHTS_TO, 1)
        compile(text, str(path), "exec")
    target = site / MTP_WEIGHTS_INSTALLED
    if not (target.is_file() and target.read_bytes() == helper):
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, target)
    if not already:
        temporary = path.with_name(f".{path.name}.weights.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    print(f"MTP_WEIGHTS armed: the draft model takes the tensors of {said} in place of the "
          f"checkpoint's", flush=True)
    return {"armed": True, "file": said, "already": already}


def patch_vision_weights() -> dict[str, Any]:
    said = str(os.environ.get(VISION_WEIGHTS_ENV) or "").strip()
    if not said:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{VISION_WEIGHTS_ENV} edits vLLM 0.30.0's Qwen4Exp model; the "
                           f"{RUNTIME_PROFILE} runtime ships another one")
    source = BUNDLE_DIR / VISION_WEIGHTS_HELPER
    if not source.is_file():
        raise RuntimeError(f"{VISION_WEIGHTS_ENV} was set but {source} is absent")
    helper = source.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(helper.decode("utf-8"), str(source), "exec"), namespace)
    found = namespace["resolve"](said)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / VISION_WEIGHTS_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular model source: {path}")
    text = path.read_text(encoding="utf-8")
    already = VISION_WEIGHTS_MARKER in text
    if not already:
        if text.count(VISION_WEIGHTS_FROM) != 1:
            raise RuntimeError(f"{VISION_WEIGHTS_ENV} cannot apply: the load_weights anchor of "
                               f"{VISION_WEIGHTS_RELATIVE} appears {text.count(VISION_WEIGHTS_FROM)} time(s), "
                               f"not once")
        text = text.replace(VISION_WEIGHTS_FROM, VISION_WEIGHTS_TO, 1)
        compile(text, str(path), "exec")
    target = site / VISION_WEIGHTS_INSTALLED
    if not (target.is_file() and target.read_bytes() == helper):
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, target)
    if not already:
        temporary = path.with_name(f".{path.name}.vision.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        _note_applied_edits(VISION_WEIGHTS_EDITS)
    print(f"VISION_WEIGHTS armed: the vision tower takes the tensors of {found} in place of the checkpoint's; "
          f"the load prints how many it swapped and the file's sha256", flush=True)
    return {"armed": True, "file": str(found), "already": already}


def draft_vocab_source() -> Path:
    said = str(os.environ.get(DRAFT_VOCAB_FILE_ENV) or "").strip()
    if not said:
        return BUNDLE_DIR / DRAFT_VOCAB_JSON
    path = Path(said)
    if path.is_absolute():
        if not path.is_file():
            raise RuntimeError(f"{DRAFT_VOCAB_FILE_ENV}={said} is not a file")
        return path
    if (BUNDLE_DIR / said).is_file():
        return BUNDLE_DIR / said
    found = sorted(DRAFT_VOCAB_SEARCH.rglob(said)) if DRAFT_VOCAB_SEARCH.is_dir() else []
    if len(found) != 1:
        raise RuntimeError(
            f"{DRAFT_VOCAB_FILE_ENV}={said} is not in {BUNDLE_DIR} and matches {len(found)} file(s) under "
            f"{DRAFT_VOCAB_SEARCH}, not one")
    return found[0]


def draft_head_fp8_check() -> bool:
    said = str(os.environ.get(DRAFT_HEAD_FP8_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{DRAFT_HEAD_FP8_ENV}={said!r} must be 0 or 1.")
    if said != "1":
        return False
    if not _flag_on(DRAFT_VOCAB_ENV):
        raise RuntimeError(f"{DRAFT_HEAD_FP8_ENV}=1 quantizes the compacted draft head that {DRAFT_VOCAB_ENV}=1 "
                           f"builds; set it.")
    return True


def patch_draft_vocab() -> dict[str, Any]:
    pad = draft_vocab_pad()
    head_fp8 = draft_head_fp8_check()
    named = str(os.environ.get(DRAFT_VOCAB_FILE_ENV) or "").strip()
    if not _flag_on(DRAFT_VOCAB_ENV):
        if named:
            raise RuntimeError(
                f"{DRAFT_VOCAB_FILE_ENV}={named} names a draft vocabulary, which {DRAFT_VOCAB_ENV} leaves off")
        if head_fp8:
            raise RuntimeError(f"{DRAFT_HEAD_FP8_ENV}=1 quantizes the draft head, which {DRAFT_VOCAB_ENV} leaves off")
        if pad is not None:
            raise RuntimeError(
                f"{DRAFT_VOCAB_PAD_ENV}={pad} pads the draft vocabulary, which "
                f"{DRAFT_VOCAB_ENV} leaves off")
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{DRAFT_VOCAB_ENV} edits vLLM 0.30.0's Qwen4Exp draft model; the {RUNTIME_PROFILE} "
            f"runtime ships another one, so use_local_argmax_reduction would find no get_top_tokens.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    payload: dict[str, bytes] = {}
    vocab_source = draft_vocab_source()
    for name, installed in ((DRAFT_VOCAB_HELPER, DRAFT_VOCAB_INSTALLED),
                            (DRAFT_VOCAB_JSON, DRAFT_VOCAB_INSTALLED_JSON)):
        source = vocab_source if name == DRAFT_VOCAB_JSON else BUNDLE_DIR / name
        if not source.is_file():
            raise RuntimeError(f"{DRAFT_VOCAB_ENV} was set but {source} is absent")
        payload[installed] = source.read_bytes()
    if head_fp8:
        fp8_source = BUNDLE_DIR / DRAFT_HEAD_FP8_HELPER
        if not fp8_source.is_file():
            raise RuntimeError(f"{DRAFT_HEAD_FP8_ENV} was set but {fp8_source} is absent")
        payload[DRAFT_HEAD_FP8_INSTALLED] = fp8_source.read_bytes()
        compile(payload[DRAFT_HEAD_FP8_INSTALLED].decode("utf-8"), str(fp8_source), "exec")
    helper = payload[DRAFT_VOCAB_INSTALLED]
    compile(helper.decode("utf-8"), str(BUNDLE_DIR / DRAFT_VOCAB_HELPER), "exec")
    record = json.loads(payload[DRAFT_VOCAB_INSTALLED_JSON])
    ids = record.get("ids") or []
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record.get("ids_sha256"):
        raise RuntimeError(f"{DRAFT_VOCAB_ENV}: {DRAFT_VOCAB_JSON} does not hash to its ids_sha256")
    if not ids or ids[0] < 0 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise RuntimeError(f"{DRAFT_VOCAB_ENV}: {DRAFT_VOCAB_JSON} is not sorted unique token ids")
    if ids[-1] >= int(record.get("vocab_rows") or 0):
        raise RuntimeError(
            f"{DRAFT_VOCAB_ENV}: {DRAFT_VOCAB_JSON} holds id {ids[-1]}, outside its "
            f"{record.get('vocab_rows')} vocabulary rows")
    path = site / DRAFT_VOCAB_TARGET
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular draft-model source: {path}")
    stock = path.read_bytes()
    before = hashlib.sha256(stock).hexdigest()
    text = stock.decode("utf-8")
    already = DRAFT_VOCAB_MARKER in text
    if not already:
        if text.count(DRAFT_VOCAB_FROM) != 1:
            raise RuntimeError(
                f"{DRAFT_VOCAB_ENV} cannot apply: the compute_logits anchor of "
                f"{DRAFT_VOCAB_TARGET} appears {text.count(DRAFT_VOCAB_FROM)} time(s), not once.")
        text = text.replace(DRAFT_VOCAB_FROM, DRAFT_VOCAB_TO, 1)
        compile(text, str(path), "exec")
    for installed, data in payload.items():
        target = site / installed
        if target.is_file() and target.read_bytes() == data:
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    made = text.encode("utf-8")
    if not already:
        temporary = path.with_name(f".{path.name}.vocab.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the draft vocabulary edit of {DRAFT_VOCAB_TARGET} did not land")
    row = {"armed": True, "already": already, "file": DRAFT_VOCAB_TARGET,
           "stock_sha256": before, "patched_sha256": hashlib.sha256(made).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "json_sha256": hashlib.sha256(payload[DRAFT_VOCAB_INSTALLED_JSON]).hexdigest(),
           "ids_sha256": record["ids_sha256"], "rows": len(ids),
           "vocab_rows": record.get("vocab_rows"), "source": str(vocab_source)}
    padding = ""
    if pad is not None:
        row["pad"] = pad
        row["padded_rows"] = len(ids) + (-len(ids) % pad)
        padding = (f"; {DRAFT_VOCAB_PAD_ENV}={pad} pads its head to {row['padded_rows']} rows "
                   f"by repeating id {ids[-1]}")
    if head_fp8:
        row["head_fp8_sha256"] = hashlib.sha256(payload[DRAFT_HEAD_FP8_INSTALLED]).hexdigest()
        padding += (f"; {DRAFT_HEAD_FP8_ENV}=1 reads its rows as e4m3 with per-row scales, helper at "
                    f"{DRAFT_HEAD_FP8_INSTALLED} {row['head_fp8_sha256'][:16]}")
    print(f"DRAFT_VOCAB {'already armed' if already else 'armed'}: the MTP head drafts from "
          f"{len(ids)} of {record.get('vocab_rows')} tokens ({vocab_source.name}, ids {record['ids_sha256'][:16]}) "
          f"through get_top_tokens in {DRAFT_VOCAB_TARGET}, helper at {DRAFT_VOCAB_INSTALLED} "
          f"{row['helper_sha256'][:16]}{padding}", flush=True)
    return row


def patch_target_vocab() -> dict[str, Any]:
    if not _flag_on(TARGET_VOCAB_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{TARGET_VOCAB_ENV} edits vLLM 0.30.0's Qwen4Exp target model; the {RUNTIME_PROFILE} runtime ships "
            f"another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    payload: dict[str, bytes] = {}
    for name, installed in ((TARGET_VOCAB_HELPER, TARGET_VOCAB_INSTALLED),
                            (TARGET_VOCAB_JSON, TARGET_VOCAB_INSTALLED_JSON)):
        source = BUNDLE_DIR / name
        if not source.is_file():
            raise RuntimeError(f"{TARGET_VOCAB_ENV} was set but {source} is absent")
        payload[installed] = source.read_bytes()
    helper = payload[TARGET_VOCAB_INSTALLED]
    compile(helper.decode("utf-8"), str(BUNDLE_DIR / TARGET_VOCAB_HELPER), "exec")
    record = json.loads(payload[TARGET_VOCAB_INSTALLED_JSON])
    ids = record.get("ids") or []
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record.get("ids_sha256"):
        raise RuntimeError(f"{TARGET_VOCAB_ENV}: {TARGET_VOCAB_JSON} does not hash to its ids_sha256")
    if not ids or ids[0] < 0 or any(a >= b for a, b in zip(ids, ids[1:])):
        raise RuntimeError(f"{TARGET_VOCAB_ENV}: {TARGET_VOCAB_JSON} is not sorted unique token ids")
    if ids[-1] >= int(record.get("vocab_rows") or 0):
        raise RuntimeError(
            f"{TARGET_VOCAB_ENV}: {TARGET_VOCAB_JSON} holds id {ids[-1]}, outside its "
            f"{record.get('vocab_rows')} vocabulary rows")
    if _flag_on(DRAFT_VOCAB_ENV):
        drafted = json.loads(draft_vocab_source().read_bytes()).get("ids") or []
        missing = sorted(set(drafted) - set(ids))
        if missing:
            raise RuntimeError(
                f"{TARGET_VOCAB_ENV}: {len(missing)} id(s) the draft head proposes are outside {TARGET_VOCAB_JSON} "
                f"(first {missing[:5]}), so the target could never accept them")
    path = site / TARGET_VOCAB_TARGET
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular target-model source: {path}")
    stock = path.read_bytes()
    before = hashlib.sha256(stock).hexdigest()
    text = stock.decode("utf-8")
    already = TARGET_VOCAB_MARKER in text
    if not already:
        if text.count(TARGET_VOCAB_FROM) != 1:
            raise RuntimeError(
                f"{TARGET_VOCAB_ENV} cannot apply: the compute_logits anchor of {TARGET_VOCAB_TARGET} appears "
                f"{text.count(TARGET_VOCAB_FROM)} time(s), not once.")
        text = text.replace(TARGET_VOCAB_FROM, TARGET_VOCAB_TO, 1)
        compile(text, str(path), "exec")
    for installed, data in payload.items():
        target = site / installed
        if target.is_file() and target.read_bytes() == data:
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    made = text.encode("utf-8")
    if not already:
        temporary = path.with_name(f".{path.name}.tvocab.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the target vocabulary edit of {TARGET_VOCAB_TARGET} did not land")
    row = {"armed": True, "already": already, "file": TARGET_VOCAB_TARGET,
           "stock_sha256": before, "patched_sha256": hashlib.sha256(made).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "json_sha256": hashlib.sha256(payload[TARGET_VOCAB_INSTALLED_JSON]).hexdigest(),
           "ids_sha256": record["ids_sha256"], "rows": len(ids), "vocab_rows": record.get("vocab_rows")}
    print(f"TARGET_VOCAB {'already armed' if already else 'armed'}: the target scores {len(ids)} of "
          f"{record.get('vocab_rows')} tokens ({record.get('name')}, ids {record['ids_sha256'][:16]}), the rest "
          f"read -inf; its lm_head rows move to the front of the head in place on the first eager logits call, "
          f"helper at {TARGET_VOCAB_INSTALLED} {row['helper_sha256'][:16]}", flush=True)
    return row


def patch_mtp_last_rows() -> dict[str, Any]:
    attn_rows = _flag_on(MTP_ATTN_ROWS_ENV)
    if attn_rows and not _flag_on(MTP_LAST_ROWS_ENV):
        raise RuntimeError(
            f"{MTP_ATTN_ROWS_ENV}=1 narrows the gather {MTP_LAST_ROWS_ENV} adds, and "
            f"{MTP_LAST_ROWS_ENV} is off.")
    if not _flag_on(MTP_LAST_ROWS_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{MTP_LAST_ROWS_ENV} edits vLLM 0.30.0's Qwen4Exp draft model and its V2 "
            f"speculator; the {RUNTIME_PROFILE} runtime ships other ones.")
    if attn_rows and not _flag_on(QSA_KV8_ENV):
        raise RuntimeError(
            f"{MTP_ATTN_ROWS_ENV}=1 runs the draft's sparse attention on the gathered rows through "
            f"the {QSA_KV8_ENV} attention kernels, and {QSA_KV8_ENV} is off.")
    targets = dict(MTP_LAST_ROWS_TARGETS)
    if attn_rows:
        pinned_model, model_edits = targets[MTP_LAST_ROWS_MODEL_RELATIVE]
        targets[MTP_LAST_ROWS_MODEL_RELATIVE] = (pinned_model, model_edits + MTP_ATTN_ROWS_EDITS)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular MTP_LAST_ROWS source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) == pinned:
                files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
                continue
            gather = MTP_LAST_ROWS_TARGETS[relative][1]
            extra = MTP_ATTN_ROWS_EDITS if relative == MTP_LAST_ROWS_MODEL_RELATIVE else ()
            if extra and attn_rows and _undone_sha256(text, gather, pinned) == pinned:
                for found, put in extra:
                    if text.count(found) != 1:
                        raise RuntimeError(
                            f"{MTP_ATTN_ROWS_ENV} cannot extend the gathered {relative}: an anchor "
                            f"appears {text.count(found)} time(s), not once.")
                    text = text.replace(found, put, 1)
                moved = "added to"
            elif extra and not attn_rows and _undone_sha256(text, gather + extra, pinned) == pinned:
                for found, put in reversed(extra):
                    text = text.replace(put, found, 1)
                moved = "taken from"
            else:
                raise RuntimeError(
                    f"{MTP_LAST_ROWS_ENV} cannot apply: {relative} is {before[:16]}, neither "
                    f"the {pinned[:16]} its edits were written against nor that file with them "
                    f"applied.")
            compile(text, str(path), "exec")
            made[relative] = text.encode("utf-8")
            files[relative] = {"stock_sha256": pinned,
                               "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                               "already": False}
            print(f"MTP_LAST_ROWS: the {MTP_ATTN_ROWS_ENV} edits were {moved} the already gathered "
                  f"{relative}, which a relaunch on a kept runtime finds as the first server left it",
                  flush=True)
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{MTP_LAST_ROWS_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}

    mtp_path = site / MTP_LAST_ROWS_MTP_RELATIVE
    if not mtp_path.is_file() or mtp_path.is_symlink():
        raise RuntimeError(f"Missing regular MTP source: {mtp_path}")
    mtp_stock = mtp_path.read_text(encoding="utf-8")
    mtp_before = hashlib.sha256(mtp_stock.encode("utf-8")).hexdigest()
    if MTP_LAST_ROWS_MARKER not in mtp_stock:
        mtp_text = mtp_stock
        for found, put in MTP_LAST_ROWS_MTP_EDITS:
            if mtp_text.count(found) != 1:
                raise RuntimeError(
                    f"{MTP_LAST_ROWS_ENV} cannot apply: an anchor of "
                    f"{MTP_LAST_ROWS_MTP_RELATIVE} appears {mtp_text.count(found)} time(s), not "
                    f"once; the draft pass still reads every row of its first forward.")
            mtp_text = mtp_text.replace(found, put, 1)
        compile(mtp_text, str(mtp_path), "exec")
        made[MTP_LAST_ROWS_MTP_RELATIVE] = mtp_text.encode("utf-8")
        files[MTP_LAST_ROWS_MTP_RELATIVE] = {
            "stock_sha256": mtp_before,
            "patched_sha256": hashlib.sha256(made[MTP_LAST_ROWS_MTP_RELATIVE]).hexdigest(),
            "already": False,
        }
    else:
        files[MTP_LAST_ROWS_MTP_RELATIVE] = {"stock_sha256": mtp_before, "already": True}

    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.lastrows.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the MTP_LAST_ROWS edit of {relative} did not land")

    row = {"armed": True, "already": not made, "files": files, "attn_rows": attn_rows}
    attention_note = (
        "its sparse attention runs only on the gathered rows (the KV and the indexer's side "
        "caches are still written for every row, the choice of kernel and tile shape is the "
        "full batch's)" if attn_rows else "attention still runs for every row")
    print(f"MTP_LAST_ROWS {'already armed' if not made else 'armed'}: {len(made)} of "
          f"{len(files)} vLLM file(s) edited; the MTP draft's first forward pass writes KV for "
          f"every row, {attention_note}, gathers each request's last accepted row "
          f"right after, and runs the MoE/dense FFN, the hyperconnection final mixer and the "
          f"draft head only on the gathered rows; "
          + "; ".join(
              f"{relative} {held['stock_sha256'][:16]}"
              + (f" -> {held['patched_sha256'][:16]}" if not held.get("already")
                 else " (already)")
              for relative, held in files.items())
          , flush=True)
    return row


def patch_shared_stream() -> dict[str, Any]:
    raw = str(os.environ.get(SHARED_STREAM_ENV) or "").strip()
    if raw in ("", "0"):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{SHARED_STREAM_ENV} edits vLLM 0.30.0's shared-expert runner; the {RUNTIME_PROFILE} "
            f"runtime ships another one.")
    try:
        tokens = int(raw)
    except ValueError:
        raise ValueError(f"{SHARED_STREAM_ENV} must be a whole number of rows, found {raw!r}") from None
    if not 0 < tokens <= SHARED_STREAM_MAX:
        raise ValueError(f"{SHARED_STREAM_ENV} must be between 1 and {SHARED_STREAM_MAX}, found {tokens}")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / SHARED_STREAM_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular SHARED_STREAM source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    already = False
    if _foreign_free_sha256(text, SHARED_STREAM_EDITS, SHARED_STREAM_PINNED) != SHARED_STREAM_PINNED:
        if _undone_sha256(text, SHARED_STREAM_EDITS, SHARED_STREAM_PINNED) != SHARED_STREAM_PINNED:
            raise RuntimeError(
                f"{SHARED_STREAM_ENV} cannot apply: {SHARED_STREAM_RELATIVE} is {before[:16]}, neither "
                f"the {SHARED_STREAM_PINNED[:16]} its edits were written against nor that file with "
                f"them applied.")
        already = True
        patched = before
    else:
        for found, put in SHARED_STREAM_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{SHARED_STREAM_ENV} cannot apply: an anchor of {SHARED_STREAM_RELATIVE} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        data = text.encode("utf-8")
        temporary = path.with_name(f".{path.name}.sharedstream.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the SHARED_STREAM edit of {SHARED_STREAM_RELATIVE} did not land")
        patched = hashlib.sha256(data).hexdigest()
    row = {"armed": True, "already": already, "tokens": tokens, "stock_sha256": SHARED_STREAM_PINNED,
           "patched_sha256": patched}
    print(f"SHARED_STREAM {'already armed' if already else 'armed'}: the shared expert of an MoE layer "
          f"runs on the aux stream beside the routed experts for steps of up to {tokens} rows (stock "
          f"256, set by VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD, which stays the floor); the math "
          f"is the stock call, only its stream moves; {SHARED_STREAM_RELATIVE} "
          f"{SHARED_STREAM_PINNED[:16]} -> {patched[:16]}", flush=True)
    return row


def piecewise_draft_edits(ceiling: int) -> tuple[tuple[str, str], ...]:
    return ((PIECEWISE_DRAFT_ANCHOR,
             PIECEWISE_DRAFT_ANCHOR.replace(
                 '    def capture(\n',
                 PIECEWISE_DRAFT_METHOD.format(ceiling=ceiling) + '    def capture(\n')),)


def patch_piecewise_large(tuning: dict[str, Any]) -> dict[str, Any]:
    largest, step = piecewise_large(tuning)
    if not largest:
        return {"armed": False}
    ceiling = int(tuning["max_cudagraph_capture_size"])
    sizes = cudagraph_capture_sizes(tuning)
    added = [size for size in sizes if size > ceiling]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / PIECEWISE_DRAFT_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular draft graph manager source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    edits = piecewise_draft_edits(ceiling)
    row: dict[str, Any] = {"armed": True, "largest": largest, "step": step, "ceiling": ceiling,
                           "added_sizes": added, "target": PIECEWISE_DRAFT_RELATIVE}
    if _foreign_free_sha256(text, edits, PIECEWISE_DRAFT_PINNED) != PIECEWISE_DRAFT_PINNED:
        if _undone_sha256(text, edits, PIECEWISE_DRAFT_PINNED) != PIECEWISE_DRAFT_PINNED:
            raise RuntimeError(
                f"{PIECEWISE_LARGE_ENV} cannot apply: {PIECEWISE_DRAFT_RELATIVE} is {before[:16]}, "
                f"neither the {PIECEWISE_DRAFT_PINNED[:16]} its edit was written against nor that "
                f"file with a {ceiling}-token draft ceiling applied.")
        row.update({"already": True, "stock_sha256": PIECEWISE_DRAFT_PINNED,
                    "patched_sha256": before})
    else:
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{PIECEWISE_LARGE_ENV} cannot apply: an anchor of {PIECEWISE_DRAFT_RELATIVE} "
                    f"appears {text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
        temporary = path.with_name(f".{path.name}.piecewise.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {PIECEWISE_LARGE_ENV} edit of {PIECEWISE_DRAFT_RELATIVE} did not land")
        row.update({"already": False, "stock_sha256": before,
                    "patched_sha256": hashlib.sha256(made).hexdigest()})
    print(f"PIECEWISE_LARGE {'already armed' if row['already'] else 'armed'}: mixed steps of "
          f"{added[0]}-{largest} tokens replay piecewise graphs at {len(added)} sizes {step} tokens "
          f"apart, padded up to the next one, attention still eager between the pieces; full "
          f"decode graphs keep their sizes, {min(ceiling, decode_graph_tokens(tuning))} tokens and "
          f"below, and the MTP "
          f"draft's first pass keeps its graphs at {ceiling} tokens and below, so a larger step "
          f"runs it eagerly; {PIECEWISE_DRAFT_RELATIVE} {row['stock_sha256'][:16]} -> "
          f"{row['patched_sha256'][:16]}", flush=True)
    return row


def patch_prefill_hold_auto() -> dict[str, Any]:
    said = {name: os.environ.get(name, "").strip()
            for name in (PREFILL_HOLD_CAP_MS_ENV, PREFILL_HOLD_F_MS_ENV, PREFILL_HOLD_P_US_ENV)}
    try:
        cap = float(said[PREFILL_HOLD_CAP_MS_ENV] or 10000)
        fixed = float(said[PREFILL_HOLD_F_MS_ENV] or 38)
        per_token = float(said[PREFILL_HOLD_P_US_ENV] or 27)
    except ValueError:
        raise RuntimeError(f"{PREFILL_HOLD_CAP_MS_ENV}, {PREFILL_HOLD_F_MS_ENV} and {PREFILL_HOLD_P_US_ENV} must be "
                           f"numbers, found {said}.") from None
    if not 100 <= cap <= 60000 or not 0 < fixed <= 1000 or not 0 <= per_token <= 1000:
        raise RuntimeError(f"{PREFILL_HOLD_CAP_MS_ENV} must be 100-60000, {PREFILL_HOLD_F_MS_ENV} above 0 and at most "
                           f"1000, {PREFILL_HOLD_P_US_ENV} 0-1000, found {cap}, {fixed} and {per_token}.")
    aligned_said = os.environ.get(PREFILL_HOLD_ALIGNED_ENV, "").strip().lower()
    if aligned_said not in ("", "0", "1", "on", "off", "true", "false", "yes", "no"):
        raise RuntimeError(f"{PREFILL_HOLD_ALIGNED_ENV} must be 0 or 1, found {aligned_said!r}.")
    aligned = aligned_said in ("1", "on", "true", "yes")
    try:
        share = float(os.environ.get(PREFILL_HOLD_FILL_SHARE_ENV, "").strip() or 0.9)
        weight = float(os.environ.get(PREFILL_HOLD_WAIT_WEIGHT_ENV, "").strip() or 1.0)
    except ValueError:
        raise RuntimeError(f"{PREFILL_HOLD_FILL_SHARE_ENV} and {PREFILL_HOLD_WAIT_WEIGHT_ENV} must be numbers.") from None
    if not 0.1 <= share <= 1.0 or not 0.01 <= weight <= 10.0:
        raise RuntimeError(f"{PREFILL_HOLD_FILL_SHARE_ENV} must be 0.1-1.0 and {PREFILL_HOLD_WAIT_WEIGHT_ENV} "
                           f"0.01-10, found {share} and {weight}.")
    row = _patch_with_table(PREFILL_HOLD_ENV, PREFILL_HOLD_HELPER, PREFILL_HOLD_TABLE, PREFILL_HOLD_INSTALLED)
    auto = _patch_with_table(PREFILL_HOLD_AUTO_ENV, PREFILL_HOLD_HELPER, PREFILL_HOLD_AUTO_TABLE,
                             PREFILL_HOLD_INSTALLED)
    ignored = [name for name in (PREFILL_HOLD_TOKENS_ENV, PREFILL_HOLD_MS_ENV) if os.environ.get(name, "").strip()]
    row = dict(row, already=row["already"] and auto["already"])
    print(_armed_said("PREFILL_HOLD", row, PREFILL_HOLD_INSTALLED,
                      f"auto: waiting prefill and the rest of a chunked one wait together until they fill the "
                      f"step budget left after its decode tokens, or until the summed wait of the held sessions "
                      f"reaches decoding lanes x F, F being the fixed cost of a mixed step over a decode step "
                      f"timed live (prior {fixed:.0f} ms and {per_token:.0f} us a prefill token until 20 mixed "
                      f"steps are timed), and never longer than {cap:.0f} ms"
                      + (f"; the fill counts what the block-aligned split would schedule, {share:.2f} of the room "
                         f"fills, a running chunk is never held" if aligned else "")
                      + (f"; the summed wait weighs {weight:.2f}" if weight != 1.0 else "")
                      + f"; auto table "
                      f"{auto['table_sha256'][:16]}" + (f"; {', '.join(ignored)} ignored" if ignored else "")),
          flush=True)
    return dict(row, auto=True, auto_table_sha256=auto["table_sha256"], cap_ms=cap, fixed_ms=fixed,
                per_token_us=per_token, aligned=aligned, fill_share=share, wait_weight=weight)


def draft_budget_wanted() -> bool:
    return str(os.environ.get(DRAFT_BUDGET_ENV) or "").strip() not in ("", "0")


def _bundle_module(name: str) -> Any:
    path = BUNDLE_DIR / name
    if not path.is_file():
        raise RuntimeError(f"{DRAFT_BUDGET_ENV} was set but {path} is absent")
    spec = importlib.util.spec_from_file_location(f"taaf_setup_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draft_budget_check(tuning: dict[str, Any]) -> dict[str, Any]:
    said = str(os.environ.get(DRAFT_BUDGET_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{DRAFT_BUDGET_ENV} edits vLLM 0.30.0's adaptive verification; the {RUNTIME_PROFILE} "
                           f"runtime ships another one.")
    if _flag_on(ADAPTIVE_VERIFY_ENV) and not DRAFT_BUDGET_VARLEN["on"]:
        raise RuntimeError(f"{DRAFT_BUDGET_ENV} and {ADAPTIVE_VERIFY_ENV} are two policies for the same rows; arm one.")
    if not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{DRAFT_BUDGET_ENV} ranks drafts by the probability the sampling drafter recorded for each "
                           f"drafted token, which only the {DRAFT_SAMPLE_ENV}=1 path records; set it.")
    depth = int(tuning["mtp_speculative_tokens"])
    if depth <= 0:
        raise RuntimeError(f"{DRAFT_BUDGET_ENV} allocates draft rows and needs speculation on; {MTP_TOKENS_ENV} is 0.")
    policy = _bundle_module("draft_budget_policy.py")
    tables = policy.load_tables()
    try:
        spec = policy.parse_knob(said, tables, boot_depth=depth)
    except ValueError as failure:
        raise RuntimeError(str(failure)) from None
    if spec["table"] not in tables["tables"]:
        raise RuntimeError(f"{DRAFT_BUDGET_ENV}={said!r} names the table {spec['table']!r}, which the bundle lacks.")
    if spec["depth"] > depth:
        raise RuntimeError(f"{DRAFT_BUDGET_ENV}={said!r} verifies up to {spec['depth']} drafts a lane but "
                           f"{MTP_TOKENS_ENV}={depth}; raise the drafter's depth or name a shallower preset.")
    DRAFT_BUDGET_VARLEN["on"] = True
    os.environ[ADAPTIVE_VERIFY_ENV] = "1"
    row = {"armed": True, "knob": said, "preset": spec["name"], "table": spec["table"], "depth": spec["depth"],
           "boot_depth": depth, "rows_per_lane_x100": spec.get("rows_per_lane_x100"),
           "rows_by_lanes": spec.get("rows_by_lanes")}
    print(f"DRAFT_BUDGET gate: {DRAFT_BUDGET_ENV}={said} verifies at most {spec['depth']} of the {depth} drafts a "
          f"lane with a fixed row total a step set by the lane count, ranked by table {spec['table']}; varlen "
          f"verification rides on {ADAPTIVE_VERIFY_ENV} set for the server; stands aside under it: the QSA "
          f"indexer's PDL launch flags, the draft-confidence acceptance estimator; composes with "
          f"{CONV_SYNCFREE_ENV} (it reads only the exact host total) and {DRAFT_SAMPLE_ENV}", flush=True)
    return row


def patch_draft_budget() -> dict[str, Any]:
    if not DRAFT_BUDGET_VARLEN["on"]:
        return {"armed": False}
    helpers = {name: _install_helper(DRAFT_BUDGET_ENV, name, installed)[:16]
               for name, installed in DRAFT_BUDGET_HELPERS}
    row = _patch_with_table(DRAFT_BUDGET_ENV, DRAFT_BUDGET_MANAGER[0], DRAFT_BUDGET_TABLE, DRAFT_BUDGET_MANAGER[1])
    row["helpers"] = helpers
    print(_armed_said("DRAFT_BUDGET", row, DRAFT_BUDGET_MANAGER[1],
                      "adaptive verification takes its row total for a step from the lane count (a fixed total "
                      "that lands on a captured graph size) and splits it across the lanes on the device with one "
                      "single-block kernel over the drafter's recorded token probabilities and a calibrated "
                      "(position, score bucket) value table; the cost-table profiling and the host confidence "
                      "copies are skipped"), flush=True)
    return row


def patch_adaptive_gate() -> dict[str, Any]:
    if not _flag_on(ADAPTIVE_VERIFY_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{ADAPTIVE_VERIFY_ENV} edits vLLM 0.30.0's adaptive verification gate; the "
            f"{RUNTIME_PROFILE} runtime ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in ADAPTIVE_GATE_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular ADAPTIVE_GATE source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{ADAPTIVE_VERIFY_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{ADAPTIVE_VERIFY_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.gate.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {ADAPTIVE_VERIFY_ENV} gate edit of {relative} did not land")
    row = {"armed": True, "already": not made, "vouched": list(ADAPTIVE_GATE_VOUCHED), "files": files}
    print(f"ADAPTIVE_GATE {'already armed' if row['already'] else 'armed'}: {ADAPTIVE_GATE_RELATIVE} "
          f"accepts {', '.join(ADAPTIVE_GATE_VOUCHED)} (their spec rows read per-request lengths from "
          f"the device) and builders that capture uniform-batch graphs for trimmed verification; "
          f"{ADAPTIVE_CONFIG_RELATIVE} lets the acceptance estimate run beside the local argmax "
          f"reduction, whose draft-vocabulary logits the speculator now hands it", flush=True)
    return row


def relax_accept_threshold() -> float:
    said = str(os.environ.get(RELAX_ACCEPT_ENV) or "").strip()
    if said in ("", "0"):
        return 0.0
    try:
        value = float(said)
    except ValueError:
        value = -1.0
    if not 0.0 < value < 1.0:
        raise RuntimeError(f"{RELAX_ACCEPT_ENV} must be a probability between 0 and 1, found {said!r}")
    return value


def patch_block_verify() -> dict[str, Any]:
    if not _knob_value(BLOCK_VERIFY_ENV):
        return {"armed": False}
    if relax_accept_threshold():
        raise RuntimeError(f"{BLOCK_VERIFY_ENV} and {RELAX_ACCEPT_ENV} both edit the rejection sampler's acceptance "
                           f"test; the relaxed test has no block form, so run one of them.")
    if not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{BLOCK_VERIFY_ENV} verifies against the drafter's top-k distribution that "
                           f"{DRAFT_SAMPLE_ENV}=1 samples from and records; set it.")
    row = _patch_with_table(BLOCK_VERIFY_ENV, BLOCK_VERIFY_HELPER, BLOCK_VERIFY_TABLE, BLOCK_VERIFY_INSTALLED)
    print(_armed_said("BLOCK_VERIFY", row, BLOCK_VERIFY_INSTALLED,
                      "sampled drafts are verified as a block (Sun et al. 2403.10444): the residual mass reads "
                      "the drafter's top-k ids instead of the vocabulary, each draft row draws its acceptance "
                      "uniform on a stream keyed by its row and the drafter keys its Gumbel noise by position "
                      "and draft step, so no draw is reused across steps; the kernel counts each verification "
                      "beside token-by-token acceptance on the same drafts and draws"), flush=True)
    return row


def accept_lambda_settings() -> tuple[float, int, int] | None:
    said = str(os.environ.get(ACCEPT_LAMBDA_ENV) or "").strip()
    if not said:
        return None
    try:
        value = float(said)
    except ValueError:
        value = 0.0
    if not 1.0 <= value <= ACCEPT_LAMBDA_MAX:
        raise RuntimeError(f"{ACCEPT_LAMBDA_ENV} must be a number in 1..{ACCEPT_LAMBDA_MAX:g}, found {said!r}")
    guard = str(os.environ.get(ACCEPT_LAMBDA_GUARD_ENV) or "").strip()
    gram = window = 0
    if guard not in ("", "0"):
        parts = [part.strip() for part in guard.split(",")]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            gram, window = int(parts[0]), int(parts[1])
        if not 2 <= gram <= 8 or not 16 <= window <= 1024 or window & (window - 1):
            raise RuntimeError(f"{ACCEPT_LAMBDA_GUARD_ENV} must read 'n,window' with n in 2..8 and the window a "
                               f"power of two in 16..1024, found {guard!r}")
    ids = str(os.environ.get(ACCEPT_LAMBDA_TOOL_IDS_ENV) or "").strip()
    if ids and not (len(ids.split(",")) == 2 and all(part.strip().isdigit() for part in ids.split(","))):
        raise RuntimeError(f"{ACCEPT_LAMBDA_TOOL_IDS_ENV} must read 'start,end' token ids, found {ids!r}")
    return value, gram, window


def patch_accept_lambda() -> dict[str, Any]:
    settings = accept_lambda_settings()
    if settings is None:
        return {"armed": False}
    if relax_accept_threshold():
        raise RuntimeError(f"{ACCEPT_LAMBDA_ENV} and {RELAX_ACCEPT_ENV} both edit the token acceptance test; run one "
                           f"of them.")
    value, gram, window = settings
    row = _patch_with_table(ACCEPT_LAMBDA_ENV, ACCEPT_LAMBDA_HELPER, ACCEPT_LAMBDA_TABLE, ACCEPT_LAMBDA_INSTALLED)
    row.update(lam=value, guard=[gram, window])
    print(_armed_said("ACCEPT_LAMBDA", row, ACCEPT_LAMBDA_INSTALLED,
                      f"a sampled draft outside a tool call is accepted at u < min(1, lambda * p / q), in the "
                      f"token test and in the block's running ratio alike, with the stock residual; global lambda "
                      f"{value:g}, a request's vllm_xargs taaf_accept_lambda overrides it; loop guard "
                      + (f"{gram}-grams over the last {window} reply tokens" if gram else "off")
                      + "; at lambda 1 the sampler is the stock one bit for bit"), flush=True)
    return row


def patch_relax_accept() -> dict[str, Any]:
    threshold = relax_accept_threshold()
    if not threshold:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{RELAX_ACCEPT_ENV} edits vLLM 0.30.0's V2 rejection sampler; the {RUNTIME_PROFILE} "
            f"runtime ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in RELAX_ACCEPT_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular RELAX_ACCEPT source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{RELAX_ACCEPT_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{RELAX_ACCEPT_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.relax.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {RELAX_ACCEPT_ENV} edit of {relative} did not land")
    row = {"armed": True, "already": not made, "threshold": threshold, "files": files}
    print(f"RELAX_ACCEPT {'already armed' if row['already'] else 'armed'}: {RELAX_ACCEPT_RELATIVE} "
          f"accepts a sampled request's draft token whenever the target gives it probability >= "
          f"{threshold} after temperature, top-k and top-p, beside the stock ratio test; this changes "
          f"the sampled distribution, greedy requests are untouched", flush=True)
    return row


def draft_sample_narrow_check() -> bool:
    said = str(os.environ.get(DRAFT_SAMPLE_NARROW_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{DRAFT_SAMPLE_NARROW_ENV}={said!r} must be 0 or 1.")
    if said != "1":
        return False
    if not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{DRAFT_SAMPLE_NARROW_ENV}=1 narrows the sampling that {DRAFT_SAMPLE_ENV}=1 does; "
                           f"set it.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{DRAFT_SAMPLE_NARROW_ENV} reads vLLM 0.30.0's gumbel kernel; the {RUNTIME_PROFILE} "
                           f"runtime ships another one.")
    return True


def patch_draft_conf_log() -> dict[str, Any]:
    draft_sample_narrow_check()
    if not (_flag_on(DRAFT_CONF_LOG_ENV) or _flag_on(ADAPTIVE_VERIFY_ENV) or _flag_on(DRAFT_SUMMIN_ENV)
            or _flag_on(DRAFT_SAMPLE_ENV) or draft_budget_wanted()):
        if str(os.environ.get(DRAFT_SAMPLE_TOP_ENV) or "").strip():
            raise RuntimeError(f"{DRAFT_SAMPLE_TOP_ENV} shapes sampled drafting and needs {DRAFT_SAMPLE_ENV}=1.")
        return {"armed": False}
    top_said = str(os.environ.get(DRAFT_SAMPLE_TOP_ENV) or "").strip()
    if top_said and not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{DRAFT_SAMPLE_TOP_ENV} shapes sampled drafting and needs {DRAFT_SAMPLE_ENV}=1.")
    if top_said and not (top_said.isascii() and top_said.isdigit() and 1 <= int(top_said) <= 256):
        raise RuntimeError(f"{DRAFT_SAMPLE_TOP_ENV}={top_said!r} is not an integer from 1 through 256.")
    if _flag_on(DRAFT_SAMPLE_ENV) and _flag_on(ADAPTIVE_VERIFY_ENV) and not DRAFT_BUDGET_VARLEN["on"]:
        raise RuntimeError(f"{DRAFT_SAMPLE_ENV} samples drafts from the drafter's distribution; vLLM 0.30.0 refuses "
                           f"{ADAPTIVE_VERIFY_ENV} beside it, so set it off.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{DRAFT_CONF_LOG_ENV} edits vLLM 0.30.0's V2 speculator base class; the "
            f"{RUNTIME_PROFILE} runtime ships another one.")
    helper_path = BUNDLE_DIR / DRAFT_CONF_LOG_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{DRAFT_CONF_LOG_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in DRAFT_CONF_LOG_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular DRAFT_CONF_LOG source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{DRAFT_CONF_LOG_ENV} cannot apply: {relative} is {before[:16]}, neither "
                    f"the {pinned[:16]} its edits were written against nor that file with them "
                    f"applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{DRAFT_CONF_LOG_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / DRAFT_CONF_LOG_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(
                f"the {DRAFT_CONF_LOG_ENV} helper {DRAFT_CONF_LOG_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.conflog.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {DRAFT_CONF_LOG_ENV} edit of {relative} did not land")
    summin_path = BUNDLE_DIR / DRAFT_SUMMIN_HELPER
    if not summin_path.is_file():
        raise RuntimeError(f"the speculator edits import {DRAFT_SUMMIN_INSTALLED}, but {summin_path} is absent")
    held_summin = summin_path.read_bytes()
    compile(held_summin.decode("utf-8"), str(summin_path), "exec")
    target = site / DRAFT_SUMMIN_INSTALLED
    if not (target.is_file() and target.read_bytes() == held_summin):
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(held_summin)
        os.replace(temporary, target)
        if target.read_bytes() != held_summin:
            raise RuntimeError(f"the helper {DRAFT_SUMMIN_INSTALLED} did not land")
    summin = held_summin if _flag_on(DRAFT_SUMMIN_ENV) else None
    sample_helper = None
    block_helper = None
    if _flag_on(DRAFT_SAMPLE_ENV):
        sample_path = BUNDLE_DIR / DRAFT_SAMPLE_HELPER
        if not sample_path.is_file():
            raise RuntimeError(f"{DRAFT_SAMPLE_ENV} was set but {sample_path} is absent")
        sample_helper = sample_path.read_bytes()
        compile(sample_helper.decode("utf-8"), str(sample_path), "exec")
        target = site / DRAFT_SAMPLE_INSTALLED
        if not (target.is_file() and target.read_bytes() == sample_helper):
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_bytes(sample_helper)
            os.replace(temporary, target)
            if target.read_bytes() != sample_helper:
                raise RuntimeError(f"the {DRAFT_SAMPLE_ENV} helper {DRAFT_SAMPLE_INSTALLED} did not land")
        block_helper = _install_helper(DRAFT_SAMPLE_ENV, BLOCK_VERIFY_HELPER, BLOCK_VERIFY_INSTALLED)
        print(f"{DRAFT_SAMPLE_ENV} helper installed: drafts sample at the request's temperature from the drafter's "
              f"top {top_said or 20} ({DRAFT_VOCAB_ENV} rows when it is on) and the rejection sampler tests p/q "
              f"against that same truncated distribution; {DRAFT_SAMPLE_INSTALLED} "
              f"{hashlib.sha256(sample_helper).hexdigest()[:16]}, with {BLOCK_VERIFY_INSTALLED} {block_helper[:16]} "
              f"beside it, which it imports only when {BLOCK_VERIFY_ENV}=1", flush=True)
    if _flag_on(DRAFT_SAMPLE_ENV) and draft_sample_narrow_check():
        narrow_sha = _install_helper(DRAFT_SAMPLE_NARROW_ENV, DRAFT_SAMPLE_NARROW_HELPER, DRAFT_SAMPLE_NARROW_INSTALLED)
        print(f"{DRAFT_SAMPLE_NARROW_ENV} armed: the drafter's sampling and its cache write touch only the kept "
              f"columns, with the stock noise keyed by vocabulary id; {DRAFT_SAMPLE_NARROW_INSTALLED} "
              f"{narrow_sha[:16]}", flush=True)
    if summin is not None:
        print(f"{DRAFT_SUMMIN_ENV} armed: each verified draft position is scored by p(greedy draft) and "
              f"by sum min(p, q_tau) over the top-20 of target and drafter on the captured verification "
              f"tail, read-only; helper at {DRAFT_SUMMIN_INSTALLED} "
              f"{hashlib.sha256(summin).hexdigest()[:16]}", flush=True)
    row = {"armed": True, "already": not made and not helper_fresh,
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files,
           "summin_sha256": hashlib.sha256(summin).hexdigest() if summin is not None else None,
           "draft_sample_sha256": hashlib.sha256(sample_helper).hexdigest() if sample_helper else None,
           "block_verify_sha256": block_helper}
    print(f"DRAFT_CONF_LOG {'already armed' if row['already'] else 'armed'}: "
          f"{DRAFT_CONF_LOG_RELATIVE} edited to record each drafted token's softmax confidence "
          f"and the verifier's accept/reject outcome into preallocated per-position histogram "
          f"buckets, read-only and gated by {DRAFT_CONF_LOG_ENV}; helper at "
          f"{DRAFT_CONF_LOG_INSTALLED} {row['helper_sha256'][:16]}", flush=True)
    return row


def patch_offload_load_sync() -> dict[str, Any]:
    if not _flag_on(OFFLOAD_SYNC_ENV):
        return {"armed": False}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / OFFLOAD_WORKER_RELATIVE
    if not path.is_file():
        print(f"OFFLOAD_LOAD_SYNC skipped: vLLM {VLLM_VERSION} has no {OFFLOAD_WORKER_RELATIVE}",
              flush=True)
        return {"armed": False, "skipped": True}
    stock = path.read_text(encoding="utf-8")
    if "_taaf_load_done" in stock:
        return {"armed": True, "already": True}
    if stock.count(OFFLOAD_LOAD_FROM) != 1:
        print(f"OFFLOAD_LOAD_SYNC cannot apply: its anchor appears {stock.count(OFFLOAD_LOAD_FROM)} "
              f"time(s) in {path.name}, not once; host-tier loads stay unordered against compute",
              flush=True)
        return {"armed": False, "refused": OFFLOAD_WORKER_RELATIVE}
    made = stock.replace(OFFLOAD_LOAD_FROM, OFFLOAD_LOAD_TO, 1)
    compile(made, str(path), "exec")
    temporary = path.with_name(f".{path.name}.sync.tmp")
    temporary.write_text(made, encoding="utf-8")
    os.replace(temporary, path)
    print("OFFLOAD_LOAD_SYNC armed: every host-tier load (CPU->GPU) now waits for the compute "
          "stream's work launched before it, as stores already do", flush=True)
    return {"armed": True, "file": OFFLOAD_WORKER_RELATIVE}


def qsa_prefill_armed() -> bool:
    said = os.environ.get(QSA_PREFILL_ENV)
    if said not in (None, "", "0", "1"):
        raise RuntimeError(f"{QSA_PREFILL_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), "
                           f"found {said!r}")
    return said == "1"


def qsa_prefill_said(layout: str) -> str:
    held = str(os.environ.get(QSA_PREFILL_FILE_ENV) or "").strip()
    rotation = ("with the int4 rotation, its inverse and the gate inside the kernel"
                if os.environ.get(KV_CACHE_DTYPE_ENV) == QSA_QUANT_DTYPES[1] else f"on the {layout} kernel")
    return (f"QSA_PREFILL armed: the eager launch the stock table singles out for prefill (rows x KV heads "
            f"over 2048) runs one pass a row at 32 columns, 4 warps and 2 stages, {rotation}; mode and "
            f"tiles are re-read at most once a second from {held or 'no file, so the defaults hold'}")


def qsa_int8_fast() -> bool:
    said = os.environ.get(QSA_INT8_FAST_ENV)
    if said not in (None, "", "0", "1"):
        raise RuntimeError(f"{QSA_INT8_FAST_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), "
                           f"found {said!r}")
    return said == "1"


def qsa_layout_said() -> str:
    return str(os.environ.get(QSA_LAYOUT_ENV) or "").strip().lower() or "int8"


def qsa_layout_shown() -> str:
    return "int4" if os.environ.get(KV_CACHE_DTYPE_ENV) == QSA_QUANT_DTYPES[1] else qsa_layout_said()


def qsa_fast_cache() -> str:
    cache = os.environ.get(KV_CACHE_DTYPE_ENV)
    if cache == QSA_QUANT_DTYPES[1]:
        return "int4"
    if cache == QSA_KV8_DTYPE and qsa_layout_said() == "int8" and qsa_int8_fast():
        return "int8"
    return ""


def qsa_tile_rows() -> int:
    said = str(os.environ.get(QSA_TILE_ENV) or "").strip()
    if said in ("", "0"):
        return 0
    if said not in QSA_TILE_SIZES:
        raise RuntimeError(f"{QSA_TILE_ENV} must be one of {QSA_TILE_SIZES}, 0 or unset, found {said!r}")
    if not (qsa_prefill_armed() and _flag_on(QSA_KV8_ENV) and qsa_fast_cache()):
        raise RuntimeError(f"{QSA_TILE_ENV} tiles the int4 QSA prefill kernel and needs {QSA_PREFILL_ENV}=1, "
                           f"{QSA_KV8_ENV}=1 and {KV_CACHE_DTYPE_ENV}={QSA_QUANT_DTYPES[1]}, or the int8 one "
                           f"with {KV_CACHE_DTYPE_ENV}={QSA_KV8_DTYPE} laid out as int8 and {QSA_INT8_FAST_ENV}=1")
    return int(said)


def qsa_lean_on() -> bool:
    said = str(os.environ.get(QSA_LEAN_ENV) or "").strip()
    if said in ("", "0"):
        return False
    if said != "1":
        raise RuntimeError(f"{QSA_LEAN_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), found {said!r}")
    if not (qsa_prefill_armed() and _flag_on(QSA_KV8_ENV) and qsa_fast_cache()):
        raise RuntimeError(f"{QSA_LEAN_ENV} replaces the QSA prefill kernel and needs {QSA_PREFILL_ENV}=1, "
                           f"{QSA_KV8_ENV}=1 and {KV_CACHE_DTYPE_ENV}={QSA_QUANT_DTYPES[1]}, or the int8 one "
                           f"with {KV_CACHE_DTYPE_ENV}={QSA_KV8_DTYPE} laid out as int8 and {QSA_INT8_FAST_ENV}=1")
    return True


def qsa_decode_lean_on() -> bool:
    said = str(os.environ.get(QSA_DECODE_LEAN_ENV) or "").strip()
    if said in ("", "0"):
        return False
    if said != "1":
        raise RuntimeError(f"{QSA_DECODE_LEAN_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), "
                           f"found {said!r}")
    if not (_flag_on(QSA_KV8_ENV) and qsa_fast_cache() == "int4" and qsa_decode_armed()["fused"]):
        raise RuntimeError(f"{QSA_DECODE_LEAN_ENV} replaces the split kernel of the fused int4 QSA decode and needs "
                           f"{QSA_KV8_ENV}=1, {KV_CACHE_DTYPE_ENV}={QSA_QUANT_DTYPES[1]} and {QSA_DECODE_FUSED_ENV}=1")
    return True


def qsa_union_decode_on() -> bool:
    said = str(os.environ.get(QSA_UNION_DECODE_ENV) or "").strip()
    if said in ("", "0"):
        return False
    if said != "1":
        raise RuntimeError(f"{QSA_UNION_DECODE_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), "
                           f"found {said!r}")
    if not (qsa_decode_lean_on() and _flag_on(HOST_KV_ENV) and _flag_on(HOST_KV_FUSED_ENV)):
        raise RuntimeError(f"{QSA_UNION_DECODE_ENV} replaces the per-row lean decode of the host-KV fused decode and "
                           f"needs {QSA_DECODE_LEAN_ENV}=1, {HOST_KV_ENV}=1 and {HOST_KV_FUSED_ENV}=1")
    return True


def qsa_union_decode_said() -> str:
    return (f"QSA_UNION_DECODE armed: a lane's decode reads each token of its verification rows' union of picks once, "
            f"for every row that selected it, instead of once per row, in place of the per-row map and lean split "
            f"kernel of the host-KV fused decode; the setup checks it against the per-row lean decode and fp32 "
            f"attention before the model load and on the card's timing, its first 3 eager calls run the per-row "
            f"decode beside it, and a failure stands it aside with '{QSA_UNION_DECODE_ENV} stood aside'")


def qsa_index_compact() -> bool:
    said = str(os.environ.get(QSA_INDEX_COMPACT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{QSA_INDEX_COMPACT_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return False
    if not (_flag_on(QSA_KV8_ENV) and _flag_on(HOST_KV_ENV)):
        raise RuntimeError(f"{QSA_INDEX_COMPACT_ENV} is installed when the host-KV helper attaches each QSA layer "
                           f"and needs {QSA_KV8_ENV}=1 and {HOST_KV_ENV}=1")
    print(f"QSA_INDEX_COMPACT armed: the QSA indexer's prefill logits run on a compact grid of row tiles after a "
          f"byte-equality check against the stock launch at layer attach ({QSA_INDEX_COMPACT_ENV})", flush=True)
    return True


def mtp_index_rows() -> bool:
    said = str(os.environ.get(MTP_INDEX_ROWS_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MTP_INDEX_ROWS_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return False
    if not (_flag_on(HOST_KV_ENV) and _flag_on(MTP_LAST_ROWS_ENV) and _flag_on(MTP_ATTN_ROWS_ENV)):
        raise RuntimeError(f"{MTP_INDEX_ROWS_ENV} is installed when the host-KV helper attaches each QSA layer and "
                           f"narrows the rows {MTP_ATTN_ROWS_ENV} gathers, so it needs {HOST_KV_ENV}=1, "
                           f"{MTP_LAST_ROWS_ENV}=1 and {MTP_ATTN_ROWS_ENV}=1")
    print(f"MTP_INDEX_ROWS armed: the MTP draft's first pass scores and picks QSA blocks only for each request's "
          f"last query row and copies them to the request's other rows; the first eager calls check against the "
          f"stock selection at layer attach and in play ({MTP_INDEX_ROWS_ENV})", flush=True)
    return True


def kv4_write_fused() -> bool:
    said = str(os.environ.get(KV4_WRITE_FUSED_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{KV4_WRITE_FUSED_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return False
    if not (_flag_on(QSA_KV8_ENV) and _flag_on(HOST_KV_ENV)):
        raise RuntimeError(f"{KV4_WRITE_FUSED_ENV} is installed when the host-KV helper attaches each QSA layer "
                           f"and needs {QSA_KV8_ENV}=1 and {HOST_KV_ENV}=1")
    if str(os.environ.get(KV_CACHE_DTYPE_ENV) or "").strip() != "int4_per_token_head":
        raise RuntimeError(f"{KV4_WRITE_FUSED_ENV} rotates the int4 KV write and needs "
                           f"{KV_CACHE_DTYPE_ENV}=int4_per_token_head")
    print(f"KV4_WRITE_FUSED armed: the int4 KV write signs keys and values in one kernel, rotates both with one "
          f"hadacore call and rescales in one kernel before the stock pack, after a byte-equality check against the "
          f"stock RHT at layer attach ({KV4_WRITE_FUSED_ENV})", flush=True)
    return True


def qsa_decode_stages() -> int:
    said = str(os.environ.get(QSA_DECODE_STAGES_ENV) or "").strip()
    if not said:
        return 0
    if not said.isdigit() or not 1 <= int(said) <= 4:
        raise RuntimeError(f"{QSA_DECODE_STAGES_ENV} must be a stage count from 1 to 4 or unset, found {said!r}")
    if not (_flag_on(QSA_KV8_ENV) and qsa_fast_cache() == "int4" and qsa_decode_armed()["fused"]):
        raise RuntimeError(f"{QSA_DECODE_STAGES_ENV} sets the stages of the fused int4 QSA decode's split kernel and "
                           f"needs {QSA_KV8_ENV}=1, {KV_CACHE_DTYPE_ENV}={QSA_QUANT_DTYPES[1]} and "
                           f"{QSA_DECODE_FUSED_ENV}=1")
    return int(said)


def qsa_decode_config() -> str:
    said = str(os.environ.get(QSA_DECODE_CONFIG_ENV) or "").strip()
    if not said:
        return ""
    parts = said.split(",")
    if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
        raise RuntimeError(f"{QSA_DECODE_CONFIG_ENV} must be 'block_n,num_warps,num_splits' or unset, found {said!r}")
    block_n, warps, splits = (int(part) for part in parts)
    if block_n < 16 or block_n & (block_n - 1) or not 1 <= warps <= 16 or warps & (warps - 1) or splits < 1:
        raise RuntimeError(f"{QSA_DECODE_CONFIG_ENV}={said!r}: block_n must be a power of 2 >= 16, num_warps a power "
                           f"of 2 from 1 to 16, num_splits at least 1")
    if not (_flag_on(QSA_KV8_ENV) and qsa_fast_cache() == "int4"):
        raise RuntimeError(f"{QSA_DECODE_CONFIG_ENV} sets the int4 QSA decode's kernel config and needs "
                           f"{QSA_KV8_ENV}=1 and {KV_CACHE_DTYPE_ENV}={QSA_QUANT_DTYPES[1]}")
    line = (f"QSA_DECODE_CONFIG armed: decode batches above 512 row-heads run the int4 QSA split kernel at "
            f"block_n={block_n} num_warps={warps} num_splits={splits} ({QSA_DECODE_CONFIG_ENV}) instead of the "
            f"pinned table's pick")
    print(line, flush=True)
    return line


def qsa_prefill_config() -> str:
    said = str(os.environ.get(QSA_PREFILL_CONFIG_ENV) or "").strip()
    if not said:
        return ""
    parts = said.split(",")
    if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
        raise RuntimeError(f"{QSA_PREFILL_CONFIG_ENV} must be 'block_n,num_warps,num_stages' or unset, found {said!r}")
    block_n, warps, stages = (int(part) for part in parts)
    if block_n < 16 or block_n & (block_n - 1) or not 1 <= warps <= 16 or warps & (warps - 1) or not 1 <= stages <= 4:
        raise RuntimeError(f"{QSA_PREFILL_CONFIG_ENV}={said!r}: block_n must be a power of 2 >= 16, num_warps a "
                           f"power of 2 from 1 to 16, num_stages from 1 to 4")
    if not (_flag_on(QSA_KV8_ENV) and qsa_prefill_armed()):
        raise RuntimeError(f"{QSA_PREFILL_CONFIG_ENV} sets the fused QSA prefill kernel config and needs "
                           f"{QSA_KV8_ENV}=1 and the QSA prefill launch armed")
    line = (f"QSA_PREFILL_CONFIG armed: the fused and lean QSA prefill kernels run at block_n={block_n} "
            f"num_warps={warps} num_stages={stages} ({QSA_PREFILL_CONFIG_ENV}) unless a prefill file overrides it")
    print(line, flush=True)
    return line


def qsa_decode_lean_said(stages: int = 0) -> str:
    staged = f"; the split kernel runs at {stages} stages ({QSA_DECODE_STAGES_ENV})" if stages else ""
    return (f"QSA_DECODE_LEAN armed: the fused int4 decode resolves each row's picks to cache slots in one pass "
            f"(block table read once a row, not once a column) and its split kernel turns the int4 nibbles into bf16 "
            f"by bit pattern instead of a float conversion, same softmax and merge{staged}; the setup checks it "
            f"against the stock decode and fp32 attention before the model load and on the card's timing, its first "
            f"3 eager calls run the stock kernel beside it, and a failure stands it aside with "
            f"'{QSA_DECODE_LEAN_ENV} stood aside'")


def qsa_lean_said(cache: str = "int4", tile: int = 0) -> str:
    dense = ("; the query rotation and the inverse rotation with the gate run as two dense passes beside it, and "
             "the int4 nibbles reach the tensor cores without a float conversion" if cache == "int4" else "")
    aside = f"; {QSA_TILE_ENV}={tile} is set and is not read under it" if tile else ""
    return (f"QSA_PREFILL_LEAN armed: the {cache} QSA prefill launch reads each row's picks through one slot pass "
            f"(block table resolved once a row, not once a column){dense}{aside}; its first 3 calls run the "
            f"per-row kernel beside it and must agree to 1e-2 (relative norm), else it stands aside")


def qsa_tile_said(tile: int, cache: str = "int4") -> str:
    return (f"QSA_TILE armed: the {cache} QSA prefill launch runs {tile} consecutive rows a program over the union of "
            f"their picks, loading each picked K/V row once for the tile, with a per-row membership mask and "
            f"online softmax; its first 3 calls run the per-row kernel beside it and must agree to 1e-2 "
            f"(relative norm), else it stands aside")


def qsa_int8_fast_said(decode: dict[str, Any], tile: int) -> str:
    moved = [f"{QSA_TILE_ENV}={tile}" if tile else "", f"{QSA_DECODE_FUSED_ENV}=1" if decode["fused"] else "",
             f"{QSA_DECODE_WARPS_ENV}={decode['warps']}" if decode["warps"] else ""]
    moved = [one for one in moved if one]
    aside = (f"; {QSA_KV8_FAST_ENV} is set and is not read under it" if _flag_on(QSA_KV8_FAST_ENV) else "")
    return (f"INT8_FAST armed: the int8 QSA cache follows {QSA_TILE_ENV}, {QSA_DECODE_FUSED_ENV}, "
            f"{QSA_DECODE_WARPS_ENV} and the host-KV spill flag's work gate as the int4 cache does, with the "
            f"int4 decode table; set: {', '.join(moved) or 'none of them'}{aside}")


def qsa_decode_armed() -> dict[str, Any]:
    said = os.environ.get(QSA_DECODE_FUSED_ENV)
    if said not in (None, "", "0", "1"):
        raise RuntimeError(f"{QSA_DECODE_FUSED_ENV} must be 1, 0 or unset (the QSA helper reads exactly '1'), "
                           f"found {said!r}")
    warps_said = str(os.environ.get(QSA_DECODE_WARPS_ENV) or "").strip()
    warps = 0
    if warps_said:
        try:
            warps = int(warps_said)
        except ValueError:
            warps = -1
        if warps < 1 or warps & (warps - 1) or warps > 16:
            raise RuntimeError(f"{QSA_DECODE_WARPS_ENV} must be a power of 2 from 1 to 16, found {warps_said!r}")
    return {"fused": said == "1", "warps": warps}


def qsa_decode_said(decode: dict[str, Any], cache: str = "int4") -> str:
    parts = []
    if decode["fused"] and cache == "int4":
        parts.append("the query rotation, the split-K merge, the inverse rotation and the gate run as two "
                     "kernels around the split kernel instead of the torch chain")
    elif decode["fused"]:
        parts.append("the split kernel and its split-K merge with the gate take the spill flag's work gate, so an "
                     "empty spill pass returns at once")
    if decode["warps"]:
        parts.append(f"decode launches take at least {decode['warps']} warps for every 32 columns of their tile, "
                     f"with the stock tiles and splits")
    return f"QSA_DECODE armed on the {cache} cache: {'; '.join(parts)}"


def patch_qsa_kv8() -> dict[str, Any]:
    prefill = qsa_prefill_armed()
    decode = qsa_decode_armed()
    int8_fast = qsa_int8_fast()
    tile = qsa_tile_rows()
    lean = qsa_lean_on()
    decode_lean = qsa_decode_lean_on()
    union_decode = qsa_union_decode_on()
    stages = qsa_decode_stages()
    qsa_decode_config()
    qsa_prefill_config()
    qsa_index_compact()
    mtp_index_rows()
    kv4_write_fused()
    if not _flag_on(QSA_KV8_ENV):
        if decode["fused"] or decode["warps"]:
            print(f"QSA_DECODE ignored: {QSA_DECODE_FUSED_ENV} or {QSA_DECODE_WARPS_ENV} is set but "
                  f"{QSA_KV8_ENV} is off, and the BF16 QSA kernel has no such path", flush=True)
        if int8_fast:
            print(f"INT8_FAST ignored: {QSA_INT8_FAST_ENV}=1 but {QSA_KV8_ENV} is off, so no int8 QSA cache is "
                  f"served", flush=True)
        if prefill:
            print(f"QSA_PREFILL ignored: {QSA_PREFILL_ENV}=1 but {QSA_KV8_ENV} is off, and the BF16 QSA "
                  f"kernel has no such path", flush=True)
            return {"armed": False, "prefill_ignored": True}
        return {"armed": False}
    fast_cache = qsa_fast_cache()
    decode_on = (decode["fused"] or bool(decode["warps"])) and bool(fast_cache)
    if (decode["fused"] or decode["warps"]) and not decode_on:
        print(f"QSA_DECODE ignored: {QSA_DECODE_FUSED_ENV} and {QSA_DECODE_WARPS_ENV} act on the int4 cache, "
              f"or on the int8 cache laid out as int8 under {QSA_INT8_FAST_ENV}=1, and {KV_CACHE_DTYPE_ENV} is "
              f"{os.environ.get(KV_CACHE_DTYPE_ENV)!r}", flush=True)
    int8_said = ""
    if int8_fast and fast_cache == "int8":
        int8_said = qsa_int8_fast_said(decode, tile)
    elif int8_fast:
        print(f"INT8_FAST ignored: {QSA_INT8_FAST_ENV} acts on the int8 cache laid out as int8, and "
              f"{KV_CACHE_DTYPE_ENV} is {os.environ.get(KV_CACHE_DTYPE_ENV)!r} laid out as {qsa_layout_shown()}",
              flush=True)
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{QSA_KV8_ENV} edits vLLM 0.30.0's QSA files; the {RUNTIME_PROFILE} "
            f"runtime ships other ones, so its QSA would refuse the int8 cache.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    helper = BUNDLE_DIR / QSA_KV8_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{QSA_KV8_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    layout = str(os.environ.get(QSA_LAYOUT_ENV) or "").strip().lower() or "int8"
    if layout not in ("int8", "k4v8"):
        raise RuntimeError(f"{QSA_LAYOUT_ENV} must be int8 or k4v8, found {layout!r}")
    if layout == "k4v8" and os.environ.get(KV_CACHE_DTYPE_ENV) != QSA_KV8_DTYPE:
        raise RuntimeError(f"{QSA_LAYOUT_ENV}=k4v8 lays out an {QSA_KV8_DTYPE} cache; "
                           f"{KV_CACHE_DTYPE_ENV} is {os.environ.get(KV_CACHE_DTYPE_ENV)!r}")
    settings = (json.dumps({"layout": layout}, sort_keys=True) + "\n").encode("utf-8")
    settings_path = site / QSA_KV8_SETTINGS
    installed = site / QSA_KV8_INSTALLED
    if (installed.is_file() and installed.read_bytes() == source and settings_path.is_file()
            and settings_path.read_bytes() == settings and all(
            b"vllm.taaf_qsa_kv8" in (site / relative).read_bytes()
            for relative in QSA_KV8_TARGETS)):
        print("QSA_KV8 already armed on this runtime; nothing applied", flush=True)
        if prefill:
            print(qsa_prefill_said(layout), flush=True)
        if int8_said:
            print(int8_said, flush=True)
        if lean:
            print(qsa_lean_said(fast_cache, tile), flush=True)
        elif tile:
            print(qsa_tile_said(tile, fast_cache), flush=True)
        if decode_on:
            print(qsa_decode_said(decode, fast_cache), flush=True)
        if decode_lean or stages:
            print(qsa_decode_lean_said(stages) if decode_lean else
                  f"QSA_DECODE_STAGES armed: the fused int4 decode's split kernel runs at {stages} stages", flush=True)
        if union_decode:
            print(qsa_union_decode_said(), flush=True)
        return {"armed": True, "already": True, "layout": layout, "prefill": prefill,
                "decode": decode if decode_on else None,
                "helper_sha256": hashlib.sha256(source).hexdigest(), **({"tile": tile} if tile else {}),
                **({"lean": True} if lean else {}),
                **({"int8_fast": True} if int8_said else {})}
    made: dict[str, bytes] = {}
    stock_hashes: dict[str, str] = {}
    for relative, (pinned, edits) in QSA_KV8_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular QSA source: {path}")
        stock = path.read_bytes()
        before = hashlib.sha256(stock).hexdigest()
        if before != pinned:
            raise RuntimeError(
                f"{QSA_KV8_ENV} cannot apply: {relative} is {before[:16]}, not the "
                f"{pinned[:16]} its edits were written against.")
        text = stock.decode("utf-8")
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{QSA_KV8_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        stock_hashes[relative] = before
    temporary = installed.with_name(f".{installed.name}.tmp")
    temporary.write_bytes(source)
    os.replace(temporary, installed)
    temporary = settings_path.with_name(f".{settings_path.name}.tmp")
    temporary.write_bytes(settings)
    os.replace(temporary, settings_path)
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.kv8.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the int8 QSA edit of {relative} did not land")
    row = {"armed": True, "layout": layout, "prefill": prefill, "decode": decode if decode_on else None,
           "helper_sha256": hashlib.sha256(source).hexdigest(),
           "files": {relative: {"stock_sha256": stock_hashes[relative],
                                "patched_sha256": hashlib.sha256(data).hexdigest()}
                     for relative, data in made.items()}}
    print(f"QSA_KV8 armed: the main QSA KV cache is {os.environ.get(KV_CACHE_DTYPE_ENV)} laid out as {qsa_layout_shown()}, "
          f"with a float32 scale "
          f"per token and head; {len(made)} vLLM file(s) edited, helper at "
          f"{QSA_KV8_INSTALLED} {row['helper_sha256'][:16]}", flush=True)
    if prefill:
        print(qsa_prefill_said(layout), flush=True)
    if int8_said:
        row["int8_fast"] = True
        print(int8_said, flush=True)
    if tile:
        row["tile"] = tile
    if lean:
        row["lean"] = True
        print(qsa_lean_said(fast_cache, tile), flush=True)
    elif tile:
        print(qsa_tile_said(tile, fast_cache), flush=True)
    if decode_on:
        print(qsa_decode_said(decode, fast_cache), flush=True)
    if decode_lean or stages:
        row.update({"decode_lean": decode_lean, "decode_stages": stages})
        print(qsa_decode_lean_said(stages) if decode_lean else
              f"QSA_DECODE_STAGES armed: the fused int4 decode's split kernel runs at {stages} stages", flush=True)
    if union_decode:
        row["union_decode"] = True
        print(qsa_union_decode_said(), flush=True)
    return row


def fp8_dense_groups() -> list[str]:
    said = str(os.environ.get(FP8_DENSE_ENV) or "").strip().lower()
    if said in ("", "0"):
        return []
    named = [part.strip() for part in said.split(",")]
    unknown = sorted(set(named) - set(FP8_DENSE_GROUPS))
    if unknown or len(set(named)) != len(named):
        raise RuntimeError(
            f"{FP8_DENSE_ENV} must be a comma-separated list of distinct groups from "
            f"{sorted(FP8_DENSE_GROUPS)}, found {os.environ.get(FP8_DENSE_ENV)!r}")
    if "hc" in named and "hc_up" in named:
        raise RuntimeError(f"{FP8_DENSE_ENV} names hc and hc_up; hc already holds the up projections.")
    return [name for name in FP8_DENSE_GROUPS if name in named]


def fp8_dense_quant() -> str:
    said = str(os.environ.get(FP8_DENSE_QUANT_ENV) or "").strip().lower() or FP8_DENSE_QUANTS[0]
    if said not in FP8_DENSE_QUANTS:
        raise RuntimeError(
            f"{FP8_DENSE_QUANT_ENV} must be one of {FP8_DENSE_QUANTS}, found "
            f"{os.environ.get(FP8_DENSE_QUANT_ENV)!r}")
    if said != FP8_DENSE_QUANTS[0] and not fp8_dense_groups():
        raise RuntimeError(
            f"{FP8_DENSE_QUANT_ENV}={said} has no effect without {FP8_DENSE_ENV} selecting a group.")
    return said


def patch_w4a8() -> dict[str, Any]:
    mode = str(os.environ.get(W4A8_ENV) or "").strip().lower()
    if not mode:
        return {"armed": False}
    if mode not in W4A8_MODES:
        raise RuntimeError(f"{W4A8_ENV} must be one of {W4A8_MODES}, found {mode!r}")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    files: dict[str, dict[str, Any]] = {}
    targets = [(W4A8_GPTQ, W4A8_GPTQ_MARKER), (W4A8_WNA16, W4A8_CT_MARKER), (W4A8_EXPERTS, W4A8_EXPERTS_MARKER)] + ([(W4A8_WNA16, W4A8_WNA16_MARKER), (W4A8_OPS, W4A8_OPS_MARKER),
                                                 (W4A8_ORACLE, W4A8_ORACLE_MARKER)] if mode == "int8" else [])
    if mode == "int8":
        library = BUNDLE_DIR / W4A8_OP_LIBRARY
        if not library.is_file():
            raise RuntimeError(f"{W4A8_ENV}=int8 needs the signed-scale Marlin op {library}")
        installed = site / W4A8_OP_INSTALLED
        if not (installed.is_file() and installed.read_bytes() == library.read_bytes()):
            shutil.copyfile(library, installed)
        files[W4A8_OP_INSTALLED] = {"sha256": hashlib.sha256(installed.read_bytes()).hexdigest()}
    for relative, marker in targets:
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source for {W4A8_ENV}={mode}: {path}")
        text = path.read_text(encoding="utf-8")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if marker not in text:
            if relative == W4A8_GPTQ:
                if before != W4A8_GPTQ_STOCK:
                    raise RuntimeError(f"{W4A8_ENV}={mode} cannot apply: {relative} is {before[:16]}, not the "
                                       f"{W4A8_GPTQ_STOCK[:16]} its edit was written against")
                for old, new in W4A8_GPTQ_ASSERTS:
                    if text.count(old) != 1:
                        raise RuntimeError(f"{W4A8_ENV}={mode} cannot apply: {relative} lacks one {old.strip()!r}")
                    text = text.replace(old, new)
                text = text + W4A8_GPTQ_TAIL
            else:
                text = text + {W4A8_WNA16_MARKER: W4A8_WNA16_TAIL, W4A8_OPS_MARKER: W4A8_OPS_TAIL,
                               W4A8_ORACLE_MARKER: W4A8_ORACLE_TAIL, W4A8_CT_MARKER: W4A8_CT_TAIL,
                               W4A8_EXPERTS_MARKER: W4A8_EXPERTS_TAIL}[marker]
            compile(text, str(path), "exec")
            temporary = path.with_name(f".{path.name}.w4a8.tmp")
            temporary.write_text(text, encoding="utf-8", newline='\n')
            os.replace(temporary, path)
        files[f"{relative}#{marker}"] = {"before_sha256": before,
                                         "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(f"W4A8 {mode}: {W4A8_GPTQ} lets int4 GPTQ MoE take {mode} input and sets the Marlin input dtype on "
          f"every AutoGPTQMoEMethod; files {sorted(files)}", flush=True)
    return {"armed": True, "mode": mode, "files": files}


def fp8_triton_band() -> str:
    said = str(os.environ.get(FP8_TRITON_BAND_ENV) or "").strip().lower().replace(" ", "")
    if said in ("", "0"):
        return ""
    taken: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for part in said.split(","):
        shape, colon, span = part.partition(":")
        n, cross, k = shape.partition("x")
        lo, dash, hi = span.partition("-")
        if not (colon and cross and dash and n.isdigit() and k.isdigit() and lo.isdigit() and hi.isdigit()):
            raise RuntimeError(f"{FP8_TRITON_BAND_ENV} entry {part!r} is not <N>x<K>:<lo>-<hi>, found "
                               f"{os.environ.get(FP8_TRITON_BAND_ENV)!r}")
        n, k, lo, hi = int(n), int(k), int(lo), int(hi)
        if n <= 0 or k <= 0 or k % 16 or not 1 <= lo <= hi <= FP8_TRITON_BAND_ROWS_MAX:
            raise RuntimeError(f"{FP8_TRITON_BAND_ENV} entry {part!r} needs N>0, K a positive multiple of 16, and "
                               f"1 <= lo <= hi <= {FP8_TRITON_BAND_ROWS_MAX}")
        held = taken.setdefault((n, k), [])
        if any(lo <= other_hi and other_lo <= hi for other_lo, other_hi in held):
            raise RuntimeError(f"{FP8_TRITON_BAND_ENV} entry {part!r} overlaps another band of {n}x{k}")
        held.append((lo, hi))
    if not fp8_dense_groups():
        raise RuntimeError(f"{FP8_TRITON_BAND_ENV} routes FP8 dense linears, and {FP8_DENSE_ENV} selects none.")
    if fp8_dense_quant() != FP8_DENSE_QUANTS[0]:
        raise RuntimeError(f"{FP8_TRITON_BAND_ENV} reads per-channel weight scales; {FP8_DENSE_QUANT_ENV} must be "
                           f"{FP8_DENSE_QUANTS[0]}.")
    return said


def patch_fp8_dense() -> dict[str, Any]:
    groups = fp8_dense_groups()
    band = fp8_triton_band()
    if not groups:
        return {"armed": False}
    quant = fp8_dense_quant()
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{FP8_DENSE_ENV} edits vLLM 0.30.0's linear layer and uses its online FP8 method; "
            f"the {RUNTIME_PROFILE} runtime ships other ones.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    helper = BUNDLE_DIR / FP8_DENSE_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{FP8_DENSE_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    patterns = {name: FP8_DENSE_GROUPS[name] for name in groups}
    held_settings: dict[str, Any] = {"groups": patterns, "quant": quant}
    band_source = b""
    if band:
        band_helper = BUNDLE_DIR / FP8_TRITON_BAND_HELPER
        if not band_helper.is_file():
            raise RuntimeError(f"{FP8_TRITON_BAND_ENV} was set but {band_helper} is absent")
        band_source = band_helper.read_bytes()
        compile(band_source.decode("utf-8"), str(band_helper), "exec")
        held_settings["band"] = band
    settings = (json.dumps(held_settings, sort_keys=True) + "\n").encode("utf-8")
    settings_path = site / FP8_DENSE_SETTINGS
    installed = site / FP8_DENSE_INSTALLED
    base = {"groups": groups, "quant": quant, "patterns": patterns, "triton_band": band or None,
            "helper_sha256": hashlib.sha256(source).hexdigest(),
            "settings_sha256": hashlib.sha256(settings).hexdigest()}
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in FP8_DENSE_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{FP8_DENSE_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{FP8_DENSE_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    placed = [(installed, source), (settings_path, settings)]
    if band:
        placed.append((site / FP8_TRITON_BAND_INSTALLED, band_source))
    already = not made and all(target.is_file() and target.read_bytes() == data for target, data in placed)
    for target, data in placed:
        if target.is_file() and target.read_bytes() == data:
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.fp8.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the FP8 dense edit of {relative} did not land")
    scheme = ("vLLM's online per-channel FP8 method (FP8 weights with a scale per output channel, "
              "FP8 activations with a dynamic scale per token)" if quant == FP8_DENSE_QUANTS[0] else
              "vLLM's online block-128 FP8 method (FP8 weights and activations with a scale per "
              "128-wide group)")
    print(f"FP8_DENSE {'already armed' if already else 'armed'} for {','.join(groups)} using {quant}: "
          "every BF16 linear whose name matches "
          + "; ".join(f"{name} {pattern}" for name, pattern in patterns.items())
          + f" is built with {scheme} and quantized as its weights load; "
          f"{len(made)} of {len(FP8_DENSE_TARGETS)} vLLM file(s) edited now, helper at "
          f"{FP8_DENSE_INSTALLED} {base['helper_sha256'][:16]}"
          + (f"; Triton band {band} installed at {FP8_TRITON_BAND_INSTALLED}, each shape self-checked at its first "
             f"use against the stock kernel (rel <= 2e-3) and left to it when the check fails" if band else "")
          + ("; the drafter's attention, input and shared-expert projections quantize too, so only draft "
             "acceptance can move: the target verifies every token" if FP8_DENSE_DRAFT_GROUP in groups else ""),
          flush=True)
    return {"armed": True, "already": already, **base, "files": files}


def _runtime_site() -> Path:
    return RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"


def _apply_pinned_targets(
    env_name: str, targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]], site: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    files: dict[str, dict[str, Any]] = {}
    made: dict[str, bytes] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if pinned is None:
            bare = _strip_edits(text, edits) if all(text.count(put) == 1 for _, put in edits) else None
            if bare is not None:
                files[relative] = {"stock_sha256": _sha256_text(bare), "patched_sha256": before, "already": True}
                continue
        elif _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{env_name} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{env_name} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.taaf.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {env_name} edit of {relative} did not land")
    return files, made


def patch_weight_checksum() -> dict[str, Any]:
    if not _flag_on(WEIGHT_CACHE_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{WEIGHT_CACHE_ENV} edits vLLM 0.30.0's GPU worker to add a weight-checksum RPC "
            f"method; the {RUNTIME_PROFILE} runtime ships another one.")
    site = _runtime_site()
    helper = BUNDLE_DIR / WEIGHT_CHECKSUM_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{WEIGHT_CACHE_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    files, made = _apply_pinned_targets(WEIGHT_CACHE_ENV, WEIGHT_CHECKSUM_TARGETS, site)
    installed = site / WEIGHT_CHECKSUM_INSTALLED
    already = not made and installed.is_file() and installed.read_bytes() == source
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
    print(f"WEIGHT_CACHE checksum RPC {'already armed' if already else 'armed'}: "
          f"{len(made)} of {len(WEIGHT_CHECKSUM_TARGETS)} vLLM file(s) edited now, helper at "
          f"{WEIGHT_CHECKSUM_INSTALLED} {hashlib.sha256(source).hexdigest()[:16]}; call it over "
          "/collective_rpc with method=taaf_weight_checksum under VLLM_SERVER_DEV_MODE=1",
          flush=True)
    return {"armed": True, "already": already, "files": files,
            "helper_sha256": hashlib.sha256(source).hexdigest()}


def weight_cache_allow_backends() -> list[str]:
    said = str(os.environ.get(WEIGHT_CACHE_ALLOW_BACKEND_ENV) or "").strip()
    return [part.strip().upper() for part in said.split(",") if part.strip()]


def patch_weight_cache_backend_guard() -> dict[str, Any]:
    allowed = weight_cache_allow_backends()
    if not allowed:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{WEIGHT_CACHE_ALLOW_BACKEND_ENV} edits vLLM 0.30.0's NVFP4 MoE pre-processed-weights "
            f"guard; the {RUNTIME_PROFILE} runtime ships another one.")
    site = _runtime_site()
    files, made = _apply_pinned_targets(WEIGHT_CACHE_ALLOW_BACKEND_ENV, WEIGHT_CACHE_BACKEND_TARGETS, site)
    already = not made
    print(f"WEIGHT_CACHE_ALLOW_BACKEND {'already armed' if already else 'armed'} for {allowed}: vLLM's "
          "own NVFP4 pre-processed-weights guard (stock: FLASHINFER_TRTLLM only) now also accepts "
          "these; UNVERIFIED for any backend but FLASHINFER_TRTLLM until the GPU equality check "
          "passes on this exact backend", flush=True)
    return {"armed": True, "already": already, "allowed": allowed, "files": files}


def _weight_cache_socket_dir() -> Path:
    getuid = getattr(os, "getuid", None)
    uid = getuid() if getuid is not None else -1
    return Path(tempfile.gettempdir()) / f"vllm_weight_cache_{uid}"


def _weight_cache_socket_present(socket_dir: Path) -> bool:
    return bool(glob.glob(str(socket_dir / WEIGHT_CACHE_SOCKET_GLOB)))


def _daemon_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def weight_cache_daemon_command(python: str, model_dir: Path, moe_backend: str | None) -> list[str]:
    command = [python, "-m", "vllm.model_executor.model_loader.weight_cache.daemon",
               "--model", str(model_dir), "--served-model-name", SERVED_MODEL_NAME,
               "--dtype", "bfloat16", "--quantization", quantization(),
               "--tensor-parallel-size", "1", "--distributed-executor-backend", "mp"]
    if moe_backend is not None:
        command.extend(["--moe-backend", str(moe_backend)])
    return command


def weight_cache_extra_config() -> str:
    return json.dumps({"mode": "zero_copy", "fallback": False}, sort_keys=True)


def weight_cache_pidfile() -> Path:
    return WORKING_DIR / WEIGHT_CACHE_PIDFILE


def _await_weight_cache_daemon(pid: int, deadline: float) -> None:
    socket_dir = _weight_cache_socket_dir()
    while time.monotonic() < deadline:
        if _weight_cache_socket_present(socket_dir):
            return
        if not _daemon_alive(pid):
            raise RuntimeError(
                f"{WEIGHT_CACHE_ENV}: the weight cache daemon (pid {pid}) died before its socket "
                f"appeared under {socket_dir}; see {WEIGHT_CACHE_DAEMON_LOG}")
        time.sleep(WEIGHT_CACHE_POLL_S)
    raise RuntimeError(
        f"{WEIGHT_CACHE_ENV}: the weight cache daemon (pid {pid}) did not bind a socket under "
        f"{socket_dir} within {WEIGHT_CACHE_READY_S:.0f}s")


def mem_offload_settings() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, name in (("embed_host", MEM_EMBED_HOST_ENV), ("vision_host", MEM_VISION_HOST_ENV),
                      ("ledger", MEM_LEDGER_ENV)):
        said = str(os.environ.get(name) or "").strip().lower()
        if said not in ("", "0", "1"):
            raise RuntimeError(f"{name}={said!r} is not one of '', '0', '1'.")
        out[key] = said == "1"
    every, given = _bounded_integer_environment(MEM_LEDGER_EVERY_ENV, default=30, minimum=1,
                                                maximum=MEM_LEDGER_EVERY_MAX_S)
    if given and not out["ledger"] and not _flag_on(KV_AUTO_ENV):
        raise RuntimeError(f"{MEM_LEDGER_EVERY_ENV} paces the memory ledger, which {MEM_LEDGER_ENV} does not arm.")
    out["ledger_s"] = every
    said = str(os.environ.get(KV_AUTO_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{KV_AUTO_ENV}={said!r} is not one of '', '0', '1'.")
    if said == "1":
        explicit = str(os.environ.get(KV_CACHE_MEMORY_BYTES_ENV) or "").strip()
        if not explicit.isdigit() or int(explicit) <= 0:
            raise RuntimeError(f"{KV_AUTO_ENV} sizes the KV pool from a measured plan and keeps "
                               f"{KV_CACHE_MEMORY_BYTES_ENV} as the size of a boot without one; set it.")
        out["kv_auto"] = True
    return out


def install_kv_auto_plan(site: Path) -> dict[str, Any]:
    plan = BUNDLE_DIR / KV_AUTO_PLAN
    installed = site / KV_AUTO_PLAN_INSTALLED
    if not plan.is_file():
        installed.unlink(missing_ok=True)
        return {"plan": None}
    source = plan.read_bytes()
    held = json.loads(source.decode("utf-8"))
    if not isinstance(held.get("fingerprint"), str) or not isinstance(held.get("boots"), list):
        raise RuntimeError(f"{plan} is not a KV_AUTO plan: it needs a fingerprint and a list of boots")
    temporary = installed.with_name(f".{installed.name}.tmp")
    temporary.write_bytes(source)
    os.replace(temporary, installed)
    return {"plan": held["fingerprint"], "boots": len(held["boots"]), "plan_sha256": hashlib.sha256(source).hexdigest()}


def patch_mem_offload() -> dict[str, Any]:
    settings = mem_offload_settings()
    auto = bool(settings.get("kv_auto"))
    if not (settings["embed_host"] or settings["vision_host"] or settings["ledger"] or auto):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{MEM_EMBED_HOST_ENV}, {MEM_VISION_HOST_ENV} and {MEM_LEDGER_ENV} edit vLLM 0.30.0's GPU model "
            f"runner; the {RUNTIME_PROFILE} runtime ships another one.")
    helper = BUNDLE_DIR / MEM_OFFLOAD_HELPER
    if not helper.is_file():
        raise RuntimeError(f"a memory offload knob was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    site = _runtime_site()
    files, made = _apply_pinned_targets(MEM_EMBED_HOST_ENV, MEM_OFFLOAD_TARGETS, site)
    installed = site / MEM_OFFLOAD_INSTALLED
    already = not made and installed.is_file() and installed.read_bytes() == source
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the memory offload helper did not land at {MEM_OFFLOAD_INSTALLED}")
    row = {"armed": True, "already": already, **settings,
           "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}
    if auto:
        row["kv_auto_plan"] = install_kv_auto_plan(site)
        held = row["kv_auto_plan"]
        print(f"KV_AUTO armed ({KV_AUTO_ENV}=1): the worker sizes the KV pool as the free memory after the profile "
              f"run minus the plan's measured post-KV growth, play peaks and boot spread, turns expandable segments "
              f"on after the KV pool and caps the process after warmup; "
              + (f"plan {held['plan']} from {held['boots']} measured boot(s)" if held["plan"] else
                 f"no {KV_AUTO_PLAN} in the bundle, so this boot keeps {KV_CACHE_MEMORY_BYTES_ENV} and measures"),
              flush=True)
    print(f"MEM_OFFLOAD {'already armed' if already else 'armed'}: embed_host={int(settings['embed_host'])} "
          f"vision_host={int(settings['vision_host'])} ledger={int(settings['ledger'])} "
          f"every={settings['ledger_s']}s; the token embedding table moves to pinned host memory read through "
          f"its device view, the vision blocks' weights stay in pinned host memory and are copied into two "
          f"block-sized device slots ahead of each block, and MEM_LEDGER lines report the device memory at "
          f"load and on the pace; helper at {MEM_OFFLOAD_INSTALLED}", flush=True)
    return row


def patch_embed_hot() -> dict[str, Any]:
    said = str(os.environ.get(EMBED_HOT_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if not said.isdigit() or int(said) <= 0:
        raise RuntimeError(f"{EMBED_HOT_ENV}={said!r} is not a positive row count.")
    if not mem_offload_settings()["embed_host"]:
        raise RuntimeError(f"{EMBED_HOT_ENV} caches rows of the host-resident embedding table, so it needs "
                           f"{MEM_EMBED_HOST_ENV}=1.")
    sha = _install_helper(EMBED_HOT_ENV, EMBED_HOT_HELPER, EMBED_HOT_INSTALLED)
    ids = BUNDLE_DIR / EMBED_HOT_IDS
    if not ids.is_file():
        raise RuntimeError(f"{EMBED_HOT_ENV} was set but {ids} is absent")
    held = json.loads(ids.read_text(encoding="utf-8"))
    count = len(held["ids"] if isinstance(held, dict) else held)
    landing = _runtime_site() / EMBED_HOT_IDS_INSTALLED
    if not (landing.is_file() and landing.read_bytes() == ids.read_bytes()):
        landing.write_bytes(ids.read_bytes())
    print(f"EMBED_HOT armed: up to {int(said)} of the {count} ranked token ids in {EMBED_HOT_IDS} are held on the "
          f"GPU beside the pinned host table, misses read the host table as before", flush=True)
    return {"armed": True, "rows": int(said), "ids": count, "helper_sha256": sha,
            "ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest()}


def patch_ple_qhot() -> dict[str, Any]:
    said = str(os.environ.get(PLE_QHOT_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if not said.isdigit() or int(said) <= 0:
        raise RuntimeError(f"{PLE_QHOT_ENV}={said!r} is not a positive row count.")
    if str(os.environ.get("TAAF_VLLM_PLE_QSTORE") or "").strip().lower() not in ("int8", "int6"):
        raise RuntimeError(f"{PLE_QHOT_ENV} caches rows of the quantized PLE store, so it needs "
                           f"TAAF_VLLM_PLE_QSTORE=int8 or int6.")
    rows = BUNDLE_DIR / PLE_QHOT_ROWS
    if not rows.is_file():
        raise RuntimeError(f"{PLE_QHOT_ENV} was set but {rows} is absent")
    landing = _runtime_site() / PLE_QHOT_INSTALLED
    held = rows.read_bytes()
    if not (landing.is_file() and landing.read_bytes() == held):
        landing.write_bytes(held)
    count = (len(held) - 128) // 4
    print(f"PLE_QHOT armed: the {int(said)} highest ranked of {count} PLE rows in {PLE_QHOT_ROWS} are held on the GPU "
          f"beside the pinned store when it seals; misses read the pinned store as before", flush=True)
    return {"armed": True, "rows": int(said), "sha256": hashlib.sha256(held).hexdigest()}


def quantization() -> str:
    said = str(os.environ.get(QUANTIZATION_ENV) or "").strip().lower() or QUANTIZATION_DEFAULT
    if said not in QUANTIZATIONS:
        raise RuntimeError(f"{QUANTIZATION_ENV} must be unset or one of {list(QUANTIZATIONS)}, found {said!r}")
    return said


def draft_model_dir() -> Path | None:
    said = str(os.environ.get(DRAFT_MODEL_ENV) or "").strip()
    if not said:
        return None
    root = Path(said)
    if (root / "config.json").is_file() and (root / "model.safetensors.index.json").is_file():
        return root
    found = [one.parent for one in sorted(root.rglob("config.json"))
             if (one.parent / "model.safetensors.index.json").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"{DRAFT_MODEL_ENV}={said} holds {len(found)} checkpoint(s), not one")
    return found[0]


def ple_hot_ram_tier_gib() -> float:
    said = str(os.environ.get(PLE_HOT_DISK_GIB_ENV) or "").strip()
    try:
        budget = float(said) if said else float(PLE_HOT_DISK_DEFAULT_GIB)
    except ValueError:
        raise RuntimeError(f"{PLE_HOT_DISK_GIB_ENV} must be a number above 0, found {said!r}")
    if not 0 < budget <= 4096:
        raise RuntimeError(f"{PLE_HOT_DISK_GIB_ENV} must be above 0 and at most 4096, found {said!r}")
    disk_rows = min(PLE_HOT_TABLE_ROWS,
                    int(budget * 1024**3) // PLE_HOT_ALIGN * PLE_HOT_ALIGN // PLE_HOT_RECORD_BYTES)
    return (PLE_HOT_TABLE_ROWS - disk_rows) * PLE_HOT_RECORD_BYTES / 1024**3


def ple_packed_gib() -> float:
    return PLE_HOT_TABLE_ROWS * PLE_HOT_RECORD_BYTES / 1024**3


def ple_df11_gib() -> float:
    return PLE_HOT_TABLE_ROWS * PLE_DF11_RECORD_BYTES / 1024**3


def ple_df11_on() -> bool:
    df11 = _flag_on(PLE_DF11_ENV)
    if df11 and not _flag_on(PLE_PACKED_ENV):
        raise RuntimeError(f"{PLE_DF11_ENV} changes how {PLE_PACKED_ENV} packs the whole PLE table; set "
                           f"{PLE_PACKED_ENV} as well")
    return df11


def ple_qstore() -> str:
    said = str(os.environ.get(PLE_QSTORE_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said not in PLE_QSTORE_RECORD_BYTES:
        raise RuntimeError(f"{PLE_QSTORE_ENV} must be unset, off, {' or '.join(PLE_QSTORE_RECORD_BYTES)}, found "
                           f"{said!r}")
    if not _flag_on(PLE_PACKED_ENV):
        raise RuntimeError(f"{PLE_QSTORE_ENV} stores the whole PLE table that {PLE_PACKED_ENV} holds in pinned host "
                           f"memory; set {PLE_PACKED_ENV} as well")
    return said


def ple_qstore_gib(kind: str) -> float:
    return PLE_HOT_TABLE_ROWS * PLE_QSTORE_RECORD_BYTES[kind] / 1024**3


def ple_packed_on() -> bool:
    whole = _flag_on(PLE_PACKED_ENV)
    if whole and ple_hot_mode():
        raise RuntimeError(f"{PLE_PACKED_ENV} holds the whole PLE table packed in pinned host memory and {PLE_HOT_ENV} "
                           f"holds a row cache in its place; set one of them")
    return whole


def host_kv_ple_gib() -> int:
    default = HOST_KV_PLE_SHM_GIB
    mode = ple_hot_mode()
    df11 = ple_df11_on()
    if ple_packed_on():
        kind = ple_qstore()
        table = ple_qstore_gib(kind) if kind else ple_df11_gib() if df11 else ple_packed_gib()
        default = math.ceil(table) + PLE_HOT_TIER_SLACK_GIB
    elif mode:
        hot, _ = _bounded_integer_environment(PLE_HOT_GIB_ENV, default=PLE_HOT_DEFAULT_GIB, minimum=1, maximum=1024)
        default = math.ceil(hot * 1.2)
        if mode == "flat":
            default += math.ceil(ple_hot_ram_tier_gib()) + PLE_HOT_TIER_SLACK_GIB
    gib, _ = _bounded_integer_environment(HOST_KV_PLE_GIB_ENV, default=default, minimum=1,
                                          maximum=512)
    return gib


def ple_hot_mode() -> str:
    said = str(os.environ.get(PLE_HOT_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said in ("1", "true", "yes", "on", "flat"):
        return "flat"
    if said == "mount":
        return "mount"
    raise RuntimeError(f"{PLE_HOT_ENV} must be unset, 0, 1, flat or mount, found {said!r}")


def patch_ple_hot() -> dict[str, Any]:
    hot = ple_hot_mode()
    whole = ple_packed_on()
    df11 = ple_df11_on()
    if not (hot or whole):
        return {"armed": False}
    knob = PLE_PACKED_ENV if whole else PLE_HOT_ENV
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{knob} edits vLLM 0.30.0's Qwen4Exp n-gram embedding; the {RUNTIME_PROFILE} "
                           f"runtime ships another one.")
    if _flag_on(WEIGHT_CACHE_ENV):
        raise RuntimeError(f"{knob} builds its {'packed table' if whole else 'miss file'} from the PLE weight loader, "
                           f"and {WEIGHT_CACHE_ENV}'s daemon load has not been checked to call it; arm one of them.")
    helpers: dict[str, bytes] = {}
    for name, relative in ((PLE_HOT_HELPER, PLE_HOT_INSTALLED), (PLE_PACKED_HELPER, PLE_PACKED_INSTALLED)):
        helper = BUNDLE_DIR / name
        if not helper.is_file():
            raise RuntimeError(f"{knob} was set but {helper} is absent")
        helpers[relative] = helper.read_bytes()
        compile(helpers[relative].decode("utf-8"), str(helper), "exec")
    site = _runtime_site()
    targets = dict(PLE_HOT_TARGETS)
    if hot:
        graph_table = BUNDLE_DIR / PLE_GRAPH_TABLE
        if not graph_table.is_file():
            raise RuntimeError(f"{PLE_HOT_ENV} was set but {graph_table} is absent")
        graph_space: dict[str, Any] = {}
        exec(compile(graph_table.read_text(encoding="utf-8"), str(graph_table), "exec"), graph_space)
        targets.update(graph_space["TARGETS"])
    files, made = _apply_pinned_targets(knob, targets, site)
    already = not made
    for relative, source in helpers.items():
        installed = site / relative
        if installed.is_file() and installed.read_bytes() == source:
            continue
        already = False
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the PLE helper did not land at {relative}")
    source = helpers[PLE_HOT_INSTALLED]
    kind = ple_qstore() if whole else ""
    if kind:
        print(f"PLE_PACKED {'already armed' if already else 'armed'} with {PLE_QSTORE_ENV}={kind}: the whole PLE table "
              f"of {PLE_HOT_TABLE_ROWS:,} rows is held as {kind} codes with one BF16 peak per group of 64 columns (one "
              f"per row when the row does not split into 64s), {PLE_QSTORE_RECORD_BYTES[kind]} B a row, one "
              f"~{ple_qstore_gib(kind):.2f} GiB pinned host table, "
              f"{'converted part by part from the DF11 file on the GPU at load' if df11 else 'quantized as the shards load'}"
              f"; a Triton kernel gathers and dequantizes each step's rows from host memory inside the forward and "
              f"inside full CUDA graphs; host-KV fit counts {host_kv_ple_gib()} GiB of PLE; helpers at "
              f"{PLE_HOT_INSTALLED} and {PLE_PACKED_INSTALLED}", flush=True)
        return {"armed": True, "already": already, "mode": "packed", "df11": df11, "qstore": kind,
                "ple_gib": host_kv_ple_gib(), "helper_sha256": hashlib.sha256(source).hexdigest(),
                "packed_sha256": hashlib.sha256(helpers[PLE_PACKED_INSTALLED]).hexdigest(), "files": files}
    if whole and df11:
        print(f"PLE_PACKED {'already armed' if already else 'armed'} with {PLE_DF11_ENV}: the whole PLE table of "
              f"{PLE_HOT_TABLE_ROWS:,} rows keeps each row's sign+mantissa bytes and Huffman-codes its exponents "
              f"with one codebook sampled from the first shard, ~{PLE_DF11_RECORD_BYTES} B a row, one "
              f"~{ple_df11_gib():.2f} GiB pinned host table; uncoded exponents go to the GPU as escapes, and a "
              f"Triton kernel gathers and decodes each step's rows from host memory inside the forward and inside "
              f"full CUDA graphs; no row cache, no host resolve, no segmented capture; host-KV fit counts "
              f"{host_kv_ple_gib()} GiB of PLE; helpers at {PLE_HOT_INSTALLED} and {PLE_PACKED_INSTALLED}",
              flush=True)
        return {"armed": True, "already": already, "mode": "packed", "df11": True, "ple_gib": host_kv_ple_gib(),
                "helper_sha256": hashlib.sha256(source).hexdigest(),
                "packed_sha256": hashlib.sha256(helpers[PLE_PACKED_INSTALLED]).hexdigest(), "files": files}
    if whole:
        print(f"PLE_PACKED {'already armed' if already else 'armed'}: the whole PLE table of {PLE_HOT_TABLE_ROWS:,} "
              f"rows is packed at {PLE_HOT_RECORD_BYTES} B a row into one {ple_packed_gib():.2f} GiB pinned host "
              f"buffer as its shards load, its escaped exponents go to the GPU, and a Triton kernel gathers and "
              f"decodes each step's rows from host memory inside the forward and inside full CUDA graphs; no row "
              f"cache, no host resolve, no segmented capture; host-KV fit counts {host_kv_ple_gib()} GiB of PLE; "
              f"helpers at {PLE_HOT_INSTALLED} and {PLE_PACKED_INSTALLED}", flush=True)
        return {"armed": True, "already": already, "mode": "packed", "ple_gib": host_kv_ple_gib(),
                "helper_sha256": hashlib.sha256(source).hexdigest(),
                "packed_sha256": hashlib.sha256(helpers[PLE_PACKED_INSTALLED]).hexdigest(), "files": files}
    disk_gib = os.environ.get(PLE_HOT_DISK_GIB_ENV) or PLE_HOT_DISK_DEFAULT_GIB
    misses = ("from the mounted checkpoint" if hot == "mount" else
              f"from a packed flat file of at most {disk_gib} GiB on local disk and a "
              f"{ple_hot_ram_tier_gib():.1f} GiB packed RAM tier, both filled as the shards load")
    print(f"PLE_HOT {'already armed' if already else 'armed'}: mode {hot}; the pinned PLE table is replaced by a "
          f"pinned row cache of {os.environ.get(PLE_HOT_GIB_ENV) or PLE_HOT_DEFAULT_GIB} GiB whose misses are read "
          f"{misses}"
          f"; host-KV fit counts {host_kv_ple_gib()} GiB of PLE; helper at {PLE_HOT_INSTALLED}; every full CUDA "
          f"graph is captured as segments around the host resolve, replayed as one call", flush=True)
    return {"armed": True, "already": already, "mode": hot, "ple_gib": host_kv_ple_gib(),
            "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}


def patch_ple_host() -> dict[str, Any]:
    inc = quantization() == "inc"
    pin = _flag_on(PLE_EXACT_PIN_ENV)
    if not (inc or pin):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{QUANTIZATION_ENV}=inc and {PLE_EXACT_PIN_ENV} edit vLLM 0.30.0's Qwen4Exp n-gram "
                           f"embedding; the {RUNTIME_PROFILE} runtime ships another one.")
    helper = BUNDLE_DIR / PLE_HOST_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{QUANTIZATION_ENV}=inc or {PLE_EXACT_PIN_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    site = _runtime_site()
    files, made = _apply_pinned_targets(QUANTIZATION_ENV if inc else PLE_EXACT_PIN_ENV, PLE_HOST_TARGETS, site)
    installed = site / PLE_HOST_INSTALLED
    already = not made and installed.is_file() and installed.read_bytes() == source
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the PLE host helper did not land at {PLE_HOST_INSTALLED}")
    print(f"PLE_HOST {'already armed' if already else 'armed'}: quantization {quantization()}"
          + ("; an INC config whose rules leave the PLE prefix at 16 bits loads the table unquantized"
             if inc else "")
          + ("; exact pin on: the PLE table is registered in place at its own size, not in the "
             "power-of-two block of the pinned host allocator" if pin else "; exact pin off")
          + f"; helper at {PLE_HOST_INSTALLED}", flush=True)
    return {"armed": True, "already": already, "inc": inc, "exact_pin": pin,
            "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}


def patch_router_pad() -> dict[str, Any]:
    if not _flag_on(ROUTER_PAD_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{ROUTER_PAD_ENV} edits vLLM 0.30.0's fused top-k router; the {RUNTIME_PROFILE} "
            f"runtime ships another one.")
    source = BUNDLE_DIR / ROUTER_PAD_HELPER
    if not source.is_file():
        raise RuntimeError(f"{ROUTER_PAD_ENV} was set but {source} is absent")
    namespace: dict[str, Any] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    return namespace["apply"](site, source)


def patch_router_fused() -> dict[str, Any]:
    if not _flag_on(ROUTER_FUSED_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{ROUTER_FUSED_ENV} edits vLLM 0.30.0's fused top-k router; the {RUNTIME_PROFILE} "
            f"runtime ships another one.")
    if _flag_on(ROUTER_PAD_ENV):
        raise RuntimeError(
            f"{ROUTER_FUSED_ENV} and {ROUTER_PAD_ENV} both replace the softmax top-k of the router; "
            f"set one of them.")
    helper = BUNDLE_DIR / ROUTER_FUSED_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{ROUTER_FUSED_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in ROUTER_FUSED_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular router source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if ROUTER_PAD_MARK in text:
            raise RuntimeError(
                f"{ROUTER_FUSED_ENV} cannot apply: {relative} carries the {ROUTER_PAD_ENV} edit, "
                f"which replaces the same softmax top-k call.")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{ROUTER_FUSED_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{ROUTER_FUSED_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / ROUTER_FUSED_INSTALLED
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the fused router helper did not land at {ROUTER_FUSED_INSTALLED}")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.rf.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the fused router edit of {relative} did not land")
    row = {"armed": True, "already": not made,
           "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}
    print(f"ROUTER_FUSED {'already armed' if not made else 'armed'}: "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; helper at {ROUTER_FUSED_INSTALLED} {row['helper_sha256'][:16]}; the softmax router's "
          f"softmax, top-k and renormalization run as one Triton kernel a row, checked against the "
          f"stock kernels on the first eager call and kept stock if they differ", flush=True)
    return row


def patch_mamba_count_floor() -> dict[str, Any]:
    if not _flag_on(MAMBA_COUNT_FLOOR_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{MAMBA_COUNT_FLOOR_ENV} edits vLLM 0.30.0's Mamba block count; the {RUNTIME_PROFILE} runtime "
            f"ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / MAMBA_COUNT_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular KV cache manager source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    if _foreign_free_sha256(text, MAMBA_COUNT_EDITS, MAMBA_COUNT_PINNED) != MAMBA_COUNT_PINNED:
        if _undone_sha256(text, MAMBA_COUNT_EDITS, MAMBA_COUNT_PINNED) != MAMBA_COUNT_PINNED:
            raise RuntimeError(
                f"{MAMBA_COUNT_FLOOR_ENV} cannot apply: {MAMBA_COUNT_RELATIVE} is {before[:16]}, neither "
                f"the {MAMBA_COUNT_PINNED[:16]} its edit was written against nor that file with it applied.")
        print(f"MAMBA_COUNT_FLOOR already armed: {MAMBA_COUNT_RELATIVE} {before[:16]}", flush=True)
        return {"armed": True, "already": True, "patched_sha256": before}
    for found, put in MAMBA_COUNT_EDITS:
        if text.count(found) != 1:
            raise RuntimeError(
                f"{MAMBA_COUNT_FLOOR_ENV} cannot apply: an anchor of {MAMBA_COUNT_RELATIVE} appears "
                f"{text.count(found)} time(s), not once.")
        text = text.replace(found, put, 1)
    compile(text, str(path), "exec")
    data = text.encode("utf-8")
    temporary = path.with_name(f".{path.name}.floor.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if path.read_bytes() != data:
        raise RuntimeError(f"the {MAMBA_COUNT_FLOOR_ENV} edit of {MAMBA_COUNT_RELATIVE} did not land")
    after = hashlib.sha256(data).hexdigest()
    print(f"MAMBA_COUNT_FLOOR armed: {MAMBA_COUNT_RELATIVE} {before[:16]} -> {after[:16]}; an align-mode "
          f"Mamba group counts its new blocks floored at zero, as its allocation does, so a request that "
          f"holds more state blocks than its step needs no longer cancels blocks the other groups take",
          flush=True)
    return {"armed": True, "already": False, "stock_sha256": before, "patched_sha256": after}


def patch_alloc_guard() -> dict[str, Any]:
    if not _flag_on(ALLOC_GUARD_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{ALLOC_GUARD_ENV} edits vLLM 0.30.0's KV cache manager; the {RUNTIME_PROFILE} runtime "
            f"ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / ALLOC_GUARD_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular KV cache manager source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    if _foreign_free_sha256(text, ALLOC_GUARD_EDITS, ALLOC_GUARD_PINNED) != ALLOC_GUARD_PINNED:
        if _undone_sha256(text, ALLOC_GUARD_EDITS, ALLOC_GUARD_PINNED) != ALLOC_GUARD_PINNED:
            raise RuntimeError(
                f"{ALLOC_GUARD_ENV} cannot apply: {ALLOC_GUARD_RELATIVE} is {before[:16]}, neither "
                f"the {ALLOC_GUARD_PINNED[:16]} its edit was written against nor that file with it applied.")
        print(f"ALLOC_GUARD already armed: {ALLOC_GUARD_RELATIVE} {before[:16]}", flush=True)
        return {"armed": True, "already": True, "patched_sha256": before}
    for found, put in ALLOC_GUARD_EDITS:
        if text.count(found) != 1:
            raise RuntimeError(
                f"{ALLOC_GUARD_ENV} cannot apply: an anchor of {ALLOC_GUARD_RELATIVE} appears "
                f"{text.count(found)} time(s), not once.")
        text = text.replace(found, put, 1)
    compile(text, str(path), "exec")
    data = text.encode("utf-8")
    temporary = path.with_name(f".{path.name}.guard.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if path.read_bytes() != data:
        raise RuntimeError(f"the {ALLOC_GUARD_ENV} edit of {ALLOC_GUARD_RELATIVE} did not land")
    after = hashlib.sha256(data).hexdigest()
    print(f"ALLOC_GUARD armed: {ALLOC_GUARD_RELATIVE} {before[:16]} -> {after[:16]}; a pool that runs out "
          f"inside an allocation the count had cleared is refused like a count that did not fit: a waiting "
          f"request is released and stays queued, a running one keeps what it took and the scheduler "
          f"preempts, and a line names the counted blocks, the blocks taken and the group kinds holding them",
          flush=True)
    return {"armed": True, "already": False, "stock_sha256": before, "patched_sha256": after}


def block_copy_chunk_mib() -> int:
    said = str(os.environ.get(BLOCK_COPY_CHUNK_ENV) or "").strip()
    value = int(said) if said else 0
    if value < 0:
        raise RuntimeError(f"{BLOCK_COPY_CHUNK_ENV} must be 0 or a positive MiB count, found {said!r}")
    return value


def block_copy_edits(limit_bytes: int) -> tuple[tuple[str, str], ...]:
    return tuple((found, put.replace(BLOCK_COPY_LIMIT_MARK, str(limit_bytes)))
                 for found, put in BLOCK_COPY_EDITS)


def patch_block_copy() -> dict[str, Any]:
    mib = block_copy_chunk_mib()
    if not mib:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{BLOCK_COPY_CHUNK_ENV} edits vLLM 0.30.0's block copy; the {RUNTIME_PROFILE} runtime "
            f"ships another one.")
    edits = block_copy_edits(mib << 20)
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / BLOCK_COPY_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular block-copy source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    if _foreign_free_sha256(text, edits, BLOCK_COPY_PINNED) != BLOCK_COPY_PINNED:
        if _undone_sha256(text, edits, BLOCK_COPY_PINNED) != BLOCK_COPY_PINNED:
            raise RuntimeError(
                f"{BLOCK_COPY_CHUNK_ENV} cannot apply: {BLOCK_COPY_RELATIVE} is {before[:16]}, neither "
                f"the {BLOCK_COPY_PINNED[:16]} its edit was written against nor that file with it applied.")
        print(f"BLOCK_COPY_CHUNK already armed: {BLOCK_COPY_RELATIVE} {before[:16]}", flush=True)
        return {"armed": True, "already": True, "mib": mib, "patched_sha256": before}
    for found, put in edits:
        if text.count(found) != 1:
            raise RuntimeError(
                f"{BLOCK_COPY_CHUNK_ENV} cannot apply: an anchor of {BLOCK_COPY_RELATIVE} appears "
                f"{text.count(found)} time(s), not once.")
        text = text.replace(found, put, 1)
    compile(text, str(path), "exec")
    data = text.encode("utf-8")
    temporary = path.with_name(f".{path.name}.copy.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if path.read_bytes() != data:
        raise RuntimeError(f"the {BLOCK_COPY_CHUNK_ENV} edit of {BLOCK_COPY_RELATIVE} did not land")
    after = hashlib.sha256(data).hexdigest()
    print(f"BLOCK_COPY_CHUNK armed: {BLOCK_COPY_RELATIVE} {before[:16]} -> {after[:16]}; a KV block "
          f"copy gathers at most {mib} MiB at a time, so copying many blocks at once never needs a "
          f"temporary the size of all of them; copies whose sources and targets overlap keep the "
          f"single gather", flush=True)
    return {"armed": True, "already": False, "mib": mib, "stock_sha256": before, "patched_sha256": after}


def cache_keep_match_unit() -> int:
    said = str(os.environ.get(CACHE_KEEP_MATCH_UNIT_ENV) or "").strip()
    if not said:
        return CACHE_KEEP_MATCH_UNIT
    if not (said.isascii() and said.isdigit()) or int(said) <= 0 or int(said) % 16:
        raise RuntimeError(f"{CACHE_KEEP_MATCH_UNIT_ENV} must be a positive multiple of 16, found {said!r}")
    return int(said)


def cache_keep_retention() -> int:
    said = str(os.environ.get(PREFIX_RETENTION_ENV) or "").strip()
    value = int(said) if said else 0
    if value < 0:
        raise RuntimeError(f"{PREFIX_RETENTION_ENV} must be 0 or positive, found {said!r}")
    return value


COMPOSE_MAX_FOREIGN = 8
_EDIT_TABLES: list[tuple[tuple[str, str], ...]] = []
_APPLIED_EDITS: list[tuple[tuple[str, str], ...]] = []


def _note_applied_edits(edits: tuple[tuple[str, str], ...]) -> None:
    held = tuple((found, put) for found, put in edits)
    if held not in _APPLIED_EDITS:
        _APPLIED_EDITS.append(held)


def _is_edits(held: Any) -> bool:
    return (isinstance(held, (tuple, list)) and bool(held)
            and all(isinstance(pair, (tuple, list)) and len(pair) == 2
                    and all(isinstance(part, str) for part in pair) for pair in held))


def _table_edits(targets: Any) -> list[tuple[tuple[str, str], ...]]:
    out = []
    if not isinstance(targets, dict):
        return out
    for value in targets.values():
        edits = value[1] if isinstance(value, (tuple, list)) and len(value) == 2 and not _is_edits(value) else value
        if _is_edits(edits):
            out.append(tuple((found, put) for found, put in edits))
    return out


def _edit_tables() -> list[tuple[tuple[str, str], ...]]:
    if _EDIT_TABLES:
        return _EDIT_TABLES
    found: list[tuple[tuple[str, str], ...]] = []
    for name, held in list(globals().items()):
        if not name.isupper():
            continue
        if name.endswith("_EDITS") and _is_edits(held):
            found.append(tuple((one, two) for one, two in held))
        elif name.endswith("_TARGETS"):
            found.extend(_table_edits(held))
    tables = sorted(BUNDLE_DIR.glob("*_table.py")) if BUNDLE_DIR.is_dir() else []
    for source in [*tables, BUNDLE_DIR / CACHE_KEEP_TABLE]:
        if not source.is_file():
            continue
        namespace: dict[str, Any] = {}
        try:
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
        except Exception as failure:
            print(f"PATCH_COMPOSE could not read the edit table {source.name}: "
                  f"{type(failure).__name__}: {failure}", flush=True)
            continue
        found.extend(_table_edits(namespace.get("TARGETS")))
    unique: list[tuple[tuple[str, str], ...]] = []
    for edits in found:
        if edits not in unique and any(one != two for one, two in edits):
            unique.append(edits)
    _EDIT_TABLES.extend(unique)
    return _EDIT_TABLES


def _composable_edits() -> list[tuple[tuple[str, str], ...]]:
    return [*_edit_tables(), *(one for one in _APPLIED_EDITS if one not in _EDIT_TABLES)]


def _strip_edits(text: str, edits: tuple[tuple[str, str], ...]) -> str | None:
    for found, put in reversed(edits):
        if text.count(put) != 1:
            return None
        text = text.replace(put, found, 1)
    return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches_pin(text: str, own: tuple[tuple[str, str], ...], pinned: str) -> bool:
    if _sha256_text(text) == pinned:
        return True
    own_pairs = {(one, two) for one, two in own}
    present = [edits for edits in _composable_edits() if not set(edits) <= own_pairs
               and all(text.count(put) == 1 for _, put in edits)]
    for size in range(1, min(len(present), COMPOSE_MAX_FOREIGN) + 1):
        for combo in itertools.combinations(present, size):
            for ordering in (combo, tuple(reversed(combo))):
                held: str | None = text
                for edits in ordering:
                    held = _strip_edits(held, edits)
                    if held is None:
                        break
                if held is not None and _sha256_text(held) == pinned:
                    return True
    return False


def _foreign_free_sha256(text: str, edits: tuple[tuple[str, str], ...], pinned: str) -> str:
    return pinned if _matches_pin(text, edits, pinned) else _sha256_text(text)


def _undone_sha256(text: str, edits: tuple[tuple[str, str], ...], pinned: str | None = None) -> str:
    held = _strip_edits(text, edits)
    if held is None:
        return ""
    if pinned is None:
        return _sha256_text(held)
    return pinned if _matches_pin(held, edits, pinned) else _sha256_text(held)


def patch_cache_keep() -> dict[str, Any]:
    if not _flag_on(CACHE_KEEP_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{CACHE_KEEP_ENV} edits vLLM 0.30.0's prefix-cache files; the {RUNTIME_PROFILE} "
            f"runtime ships other ones.")
    source = BUNDLE_DIR / CACHE_KEEP_TABLE
    if not source.is_file():
        raise RuntimeError(f"{CACHE_KEEP_ENV} was set but {source} is absent")
    table = source.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(source), "exec"), namespace)
    targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular prefix-cache source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{CACHE_KEEP_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{CACHE_KEEP_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.keep.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the cache-keep edit of {relative} did not land")
    retention = cache_keep_retention()
    row = {"armed": True, "already": not made,
           "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}
    print(f"CACHE_KEEP {'already armed' if not made else 'armed'}: {len(made)} of {len(targets)} "
          f"vLLM prefix-cache file(s) edited from table {row['table_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; the server starts with --prefix-match-unit {cache_keep_match_unit()} in place of "
          f"{PREFIX_MATCH_UNIT} and --prefix-cache-retention-interval {retention}"
          + (", rounded up by vLLM to a multiple of its hybrid block" if retention else
             ", so Mamba states are registered only at replay boundaries"), flush=True)
    row["prefix_match_unit"] = cache_keep_match_unit()
    row["prefix_cache_retention_interval"] = retention
    return row


def hc_fused_rows() -> int:
    said = os.environ.get(HC_FUSED_ROWS_ENV, "").strip()
    if said in ("", "0"):
        return 0
    if not said.isdigit() or not HC_FUSED_BASE_ROWS < int(said) <= HC_FUSED_WIDE_LIMIT \
            or int(said) % HC_FUSED_WIDE_BUCKET:
        raise RuntimeError(f"{HC_FUSED_ROWS_ENV}={said!r} is not a multiple of {HC_FUSED_WIDE_BUCKET} in "
                           f"({HC_FUSED_BASE_ROWS}, {HC_FUSED_WIDE_LIMIT}].")
    if not _flag_on(HC_FUSED_ENV):
        raise RuntimeError(f"{HC_FUSED_ROWS_ENV} widens the fused hyperconnection path and needs {HC_FUSED_ENV}.")
    return int(said)


def hc_decode_band() -> tuple[int, int] | None:
    said = str(os.environ.get(HC_DECODE_ENV) or "").strip().lower().replace(" ", "")
    shape = str(os.environ.get(HC_DECODE_SHAPE_ENV) or "").strip().replace(" ", "")
    if said in ("", "0", "off", "false", "no"):
        if shape:
            raise RuntimeError(f"{HC_DECODE_SHAPE_ENV} needs {HC_DECODE_ENV}.")
        return None
    if not _flag_on(HC_FUSED_ENV):
        raise RuntimeError(f"{HC_DECODE_ENV} is a row band of the fused hyperconnection path and needs {HC_FUSED_ENV}.")
    if said in ("1", "on", "true", "yes"):
        band = HC_DECODE_DEFAULT_BAND
    else:
        lo, dash, hi = said.partition("-")
        if not (dash and lo.isdigit() and hi.isdigit()) or not 1 <= int(lo) <= int(hi) <= HC_DECODE_ROWS_LIMIT:
            raise RuntimeError(f"{HC_DECODE_ENV}={said!r} is not <lo>-<hi> with 1 <= lo <= hi <= "
                               f"{HC_DECODE_ROWS_LIMIT}.")
        band = (int(lo), int(hi))
    if shape:
        parts = shape.split(",")
        if len(parts) != 5 or not all(part.isdigit() for part in parts):
            raise RuntimeError(f"{HC_DECODE_SHAPE_ENV}={shape!r} is not bm,bn,splits,bm_up,bh.")
        bm, bn, splits, bm_up, bh = (int(part) for part in parts)
        if (bm not in HC_DECODE_TILES or bn not in HC_DECODE_TILES or bm_up not in HC_DECODE_TILES
                or bh not in HC_DECODE_UP_WIDTHS or splits not in HC_DECODE_SPLITS):
            raise RuntimeError(f"{HC_DECODE_SHAPE_ENV}={shape!r} names a tile outside the supported set.")
    return band


def hc_decode_down(band: tuple[int, int] | None) -> dict[str, Any] | None:
    down = str(os.environ.get(HC_DECODE_DOWN_ENV) or "").strip().lower()
    up4 = str(os.environ.get(HC_DECODE_UP4_ENV) or "").strip().lower().replace(" ", "")
    floor = str(os.environ.get(HC_DECODE_MIN_ROWS_ENV) or "").strip()
    if down not in ("", "split", "cublas"):
        raise RuntimeError(f"{HC_DECODE_DOWN_ENV}={down!r} is not 'cublas' or 'split'.")
    if band is None:
        if down or up4 or floor:
            raise RuntimeError(f"{HC_DECODE_DOWN_ENV}, {HC_DECODE_UP4_ENV} and {HC_DECODE_MIN_ROWS_ENV} need "
                               f"{HC_DECODE_ENV}.")
        return None
    if down != "cublas":
        if up4 or floor:
            raise RuntimeError(f"{HC_DECODE_UP4_ENV} and {HC_DECODE_MIN_ROWS_ENV} need {HC_DECODE_DOWN_ENV}=cublas.")
        return None
    tile = "on"
    if up4 not in ("", "1", "on", "true", "yes", "0", "off", "false", "no"):
        parts = up4.split(",")
        if len(parts) != 5 or not all(part.isdigit() for part in parts):
            raise RuntimeError(f"{HC_DECODE_UP4_ENV}={up4!r} is not bm,bh,warps,stages,bk.")
        bm, bh, warps, stages, bk = (int(part) for part in parts)
        if (bm not in HC_DECODE_TILES or bh not in HC_DECODE_UP_WIDTHS or warps not in (2, 4, 8)
                or not 2 <= stages <= 5 or bk not in (16, 32, 64)):
            raise RuntimeError(f"{HC_DECODE_UP4_ENV}={up4!r} names a tile outside the supported set.")
        tile = up4
    elif up4 in ("0", "off", "false", "no"):
        tile = "off"
    if floor and (not floor.isdigit() or not 1 <= int(floor) <= HC_DECODE_ROWS_LIMIT):
        raise RuntimeError(f"{HC_DECODE_MIN_ROWS_ENV}={floor!r} is not a row count from 1 to {HC_DECODE_ROWS_LIMIT}.")
    return {"min_rows": int(floor) if floor else HC_DECODE_MIN_ROWS_DEFAULT, "up4": tile}


def hc_mixed_band(env: str) -> tuple[int, int] | None:
    said = str(os.environ.get(env) or "").strip().lower().replace(" ", "")
    if said in ("", "0", "off", "false", "no"):
        return None
    if said in ("1", "on", "true", "yes"):
        return HC_MIXED_DEFAULT_BAND
    lo, dash, hi = said.partition("-")
    if not (dash and lo.isdigit() and hi.isdigit()) or not 1 <= int(lo) <= int(hi) <= HC_FUSED_WIDE_LIMIT:
        raise RuntimeError(f"{env}={said!r} is not <lo>-<hi> with 1 <= lo <= hi <= {HC_FUSED_WIDE_LIMIT}.")
    return int(lo), int(hi)


def hc_mixed_bands() -> dict[str, tuple[int, int]]:
    bands = {name: band for name, band in (("f1", hc_mixed_band(HC_MIXED_F1_ENV)),
                                           ("f2", hc_mixed_band(HC_MIXED_F2_ENV))) if band}
    shape = str(os.environ.get(HC_MIXED_SHAPE_ENV) or "").strip().replace(" ", "")
    if not bands:
        if shape:
            raise RuntimeError(f"{HC_MIXED_SHAPE_ENV} needs {HC_MIXED_F1_ENV} or {HC_MIXED_F2_ENV}.")
        return {}
    if not _flag_on(HC_FUSED_ENV):
        raise RuntimeError(f"{HC_MIXED_F1_ENV} and {HC_MIXED_F2_ENV} are row bands of the fused hyperconnection "
                           f"path and need {HC_FUSED_ENV}.")
    if shape:
        parts = shape.split(",")
        if len(parts) != 5 or not all(part.isdigit() for part in parts):
            raise RuntimeError(f"{HC_MIXED_SHAPE_ENV}={shape!r} is not bm,bn,splits,bm_up,bh.")
        bm, bn, splits, bm_up, bh = (int(part) for part in parts)
        if (bm not in HC_MIXED_TILES or bn not in HC_MIXED_TILES or bm_up not in HC_MIXED_TILES
                or bh not in HC_MIXED_UP_WIDTHS or not 1 <= splits <= HC_MIXED_MAX_SPLITS):
            raise RuntimeError(f"{HC_MIXED_SHAPE_ENV}={shape!r} names a tile outside the supported set.")
    return bands


def patch_hc_fused() -> dict[str, Any]:
    wide = hc_fused_rows()
    decode = hc_decode_band()
    decode_down = hc_decode_down(decode)
    mixed = hc_mixed_bands()
    if not _flag_on(HC_FUSED_ENV):
        return {"armed": False}
    if {"hc", "hc_up"} & set(fp8_dense_groups()):
        raise RuntimeError(
            f"{HC_FUSED_ENV} reads the hyperconnection projections as BF16 weights, and {FP8_DENSE_ENV} "
            f"names a hyperconnection group, which stores them in FP8; drop one of them.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{HC_FUSED_ENV} edits vLLM 0.30.0's Qwen4Exp hyperconnection; the {RUNTIME_PROFILE} "
            f"runtime ships another one.")
    helper = BUNDLE_DIR / HC_FUSED_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{HC_FUSED_ENV} was set but {helper} is absent")
    if _flag_on(HC_FUSED_PICKS_ENV):
        seeds = BUNDLE_DIR / HC_FUSED_PICKS_FILE
        if seeds.is_file():
            os.environ[HC_FUSED_PICKS_FILE_ENV] = str(seeds)
            print(f"HC_FUSED_PICKS the kernel picks of {seeds.name} seed the tuner; the engine still checks every "
                  f"kind once against the stock result", flush=True)
        else:
            os.environ.pop(HC_FUSED_PICKS_FILE_ENV, None)
            print(f"HC_FUSED_PICKS {HC_FUSED_PICKS_ENV} is set but {seeds} is absent: the tuner measures every "
                  f"row count itself", flush=True)
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in HC_FUSED_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular hyperconnection source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{HC_FUSED_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{HC_FUSED_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / HC_FUSED_INSTALLED
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the fused hyperconnection helper did not land at {HC_FUSED_INSTALLED}")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.hc.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the fused hyperconnection edit of {relative} did not land")
    row = {"armed": True, "already": not made,
           "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}
    print(f"HC_FUSED {'already armed' if not made else 'armed'}: "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; helper at {HC_FUSED_INSTALLED} {row['helper_sha256'][:16]}; every hyperconnection "
          f"mix runs as three Triton kernels (combine and norm, split-K down projection, up "
          f"projection with the gated mean) for decode row counts, the stock chain above them",
          flush=True)
    if decode and decode_down:
        row["decode_band"] = list(decode)
        row["decode_down"] = decode_down
        print(f"HC_DECODE_DOWN hyperconnection mixes of {max(decode[0], decode_down['min_rows'])}-{decode[1]} rows "
              f"keep the cuBLAS down projection and fuse the silu, the up projection and the gated mean into one "
              f"four-accumulator kernel (tile {decode_down['up4']}); every reduction runs in a fixed order; each kind "
              f"is checked once against the stock chain before use, and under capture an unchecked kind keeps the "
              f"stock chain", flush=True)
    elif decode:
        row["decode_band"] = list(decode)
        print(f"HC_DECODE hyperconnection mixes of {decode[0]}-{decode[1]} rows run the combine and norm, then one "
              f"split-K down kernel that reduces its partial sums in a fixed order and applies the silu, then the "
              f"up projection with the gated mean; each kind is checked once against the stock chain before use, "
              f"and under capture an unchecked kind keeps the stock chain", flush=True)
    if mixed:
        row["mixed_bands"] = {name: list(band) for name, band in mixed.items()}
        print(f"HC_MIXED bands {row['mixed_bands']}: hyperconnection mixes inside a band run the combine and norm, "
              f"then f1 fuses the silu, the up projection and the gated mean into one kernel and f2 runs the down "
              f"projection as one Triton kernel with the silu in its epilogue; each kind is checked once against "
              f"the stock chain before use, and under capture an unchecked kind keeps the stock chain", flush=True)
    if wide:
        row["rows"] = wide
        print(f"HC_FUSED_ROWS {wide}: hyperconnection mixes of {HC_FUSED_BASE_ROWS + 1}-{wide} rows (prefill and "
              f"mixed steps) take the fused path too; each {HC_FUSED_WIDE_BUCKET}-row bucket is tuned once against "
              f"the stock chain on its first call (the stock chain stays when it is faster) and checked against "
              f"it before it is used; under capture an untuned or unchecked bucket keeps the stock chain",
              flush=True)
    return row


def bf16c_groups() -> list[str]:
    said = str(os.environ.get(BF16C_GROUPS_ENV) or "").strip().lower()
    named = [part.strip() for part in said.split(",") if part.strip()] if said else list(BF16C_DEFAULT_GROUPS)
    if set(named) - set(BF16C_GROUPS) or len(set(named)) != len(named):
        raise RuntimeError(f"{BF16C_GROUPS_ENV} must be a comma-separated list of distinct groups from "
                           f"{list(BF16C_GROUPS)}, found {os.environ.get(BF16C_GROUPS_ENV)!r}")
    return [name for name in BF16C_GROUPS if name in named]


def bf16c_rows() -> int:
    said = str(os.environ.get(BF16C_ROWS_ENV) or "").strip()
    if not said:
        return 0
    if not (said.isascii() and said.isdigit()) or not 1 <= int(said) <= BF16C_ROWS_MAX:
        raise RuntimeError(f"{BF16C_ROWS_ENV} must be an integer from 1 through {BF16C_ROWS_MAX}, found {said!r}")
    return int(said)


def patch_bf16c() -> dict[str, Any]:
    rows = bf16c_rows()
    fast = str(os.environ.get(BF16C_FAST_ENV) or "").strip()
    if fast not in ("", "0", "1"):
        raise RuntimeError(f"{BF16C_FAST_ENV} must be 0 or 1, found {fast!r}")
    mix = str(os.environ.get(BF16C_MIX_ENV) or "").strip()
    if mix not in ("", "0", "1"):
        raise RuntimeError(f"{BF16C_MIX_ENV} must be 0 or 1, found {mix!r}")
    mix_decode = str(os.environ.get(BF16C_MIX_DECODE_ENV) or "").strip()
    if mix_decode not in ("", "0", "1"):
        raise RuntimeError(f"{BF16C_MIX_DECODE_ENV} must be 0 or 1, found {mix_decode!r}")
    if mix_decode == "1" and mix != "1":
        raise RuntimeError(f"{BF16C_MIX_DECODE_ENV} widens the fused kernel's row bands and needs {BF16C_MIX_ENV}=1.")
    if not _flag_on(BF16C_ENV):
        if str(os.environ.get(BF16C_GROUPS_ENV) or "").strip() or rows:
            raise RuntimeError(f"{BF16C_GROUPS_ENV} and {BF16C_ROWS_ENV} tune the compressed weights and need {BF16C_ENV}.")
        if fast == "1":
            print(f"BF16C FAST2 has nothing to speed up: {BF16C_ENV} is off in this stage", flush=True)
        if mix == "1":
            print(f"BF16C MIX has nothing to speed up: {BF16C_ENV} is off in this stage", flush=True)
        return {"armed": False}
    groups = bf16c_groups()
    fp8_named = set(fp8_dense_groups())
    shared_groups = sorted(set(groups) & (fp8_named | ({"hc"} if "hc_up" in fp8_named else set())))
    if shared_groups:
        raise RuntimeError(f"{BF16C_ENV} and {FP8_DENSE_ENV} both name {shared_groups}; a linear layer keeps one "
                           f"method, so name each group in one of {BF16C_GROUPS_ENV} and {FP8_DENSE_ENV}.")
    if fp8_dense_groups():
        print(f"BF16C beside FP8_DENSE: FP8 builds {fp8_dense_groups()} at construction, BF16C packs the "
              f"unquantized layers of {groups}", flush=True)
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{BF16C_ENV} edits vLLM 0.30.0's unquantized methods; the {RUNTIME_PROFILE} "
                           f"runtime ships other ones.")
    helper = BUNDLE_DIR / BF16C_HELPER
    if not helper.is_file():
        raise RuntimeError(f"{BF16C_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in BF16C_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{BF16C_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{BF16C_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / BF16C_INSTALLED
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
        if installed.read_bytes() != source:
            raise RuntimeError(f"the compressed-weight helper did not land at {BF16C_INSTALLED}")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.bf16c.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the compressed-weight edit of {relative} did not land")
    row = {"armed": True, "already": not made, "groups": groups, "rows": rows, "fast2": fast == "1",
           "helper_sha256": hashlib.sha256(source).hexdigest(), "files": files}
    if mix == "1":
        row["mix"] = True
    if mix_decode == "1":
        row["mix_decode"] = True
    print(f"BF16C {'already armed' if not made else 'armed'} for {','.join(groups)}: "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; helper at {BF16C_INSTALLED} {row['helper_sha256'][:16]}; each matching BF16 weight is packed at "
          f"load to 8 bits of sign and mantissa plus a 4-bit exponent code, checked bit for bit against the "
          f"original before the BF16 tensor is freed; decode batches of up to {rows or 64} rows run a Triton GEMM "
          f"that decodes in registers, larger ones decode the layer into a transient buffer for the stock GEMM"
          + (f"; {BF16C_FAST_ENV}=1: batches of up to 256 rows take the one-pass kernel (every row in one program, "
             f"split-K summed in the kernel) once a self-check on this card passes before the server starts"
             if fast == "1" else "")
          + (f"; {BF16C_MIX_ENV}=1: batches above the decode rows run one fused GEMM that decodes each packed "
             f"weight tile in registers on its way to the tensor cores (no bf16 copy of the layer is written), for "
             f"the shapes and row bands where a self-check on this card proves it exact and faster than decode plus "
             f"the stock GEMM before the server starts"
             if mix == "1" else "")
          + (f"; {BF16C_MIX_DECODE_ENV}=1: decode batches of 257-512 rows join the fused kernel's verified bands"
             if mix_decode == "1" else ""),
          flush=True)
    return row


def _mounted_seed(env_name: str, file_name: str) -> Path | None:
    said = str(os.environ.get(env_name) or "").strip()
    if said.lower() in ("", "0", "false", "no", "off"):
        return None
    if said.lower() not in ("1", "true", "yes", "on"):
        return Path(said)
    stem = file_name.rsplit(".", 1)[0]
    names = (file_name, file_name + ".bin", stem)
    for name in names:
        beside = BUNDLE_DIR / name
        if beside.is_dir() if name == stem else beside.is_file():
            return beside
    root = Path("/kaggle/input")
    if root.is_dir():
        for depth in range(5):
            for name in names:
                for found in root.glob("/".join(["*"] * depth + [name])):
                    if found.is_dir() if name == stem else found.is_file():
                        return found
    return None


class _SeedEntry:
    def __init__(self, name: str, path: Path, size: int):
        self.name = name
        self.path = path
        self.size = size

    def isfile(self) -> bool:
        return True

    def isdir(self) -> bool:
        return False


class _SeedDirectory:
    def __init__(self, root: Path):
        entries = [_SeedEntry(path.relative_to(root).as_posix(), path, path.stat().st_size)
                   for path in sorted(root.rglob("*")) if path.is_file()]
        entries.sort(key=lambda entry: entry.name not in (COMPILE_MANIFEST, PYC_MANIFEST))
        self._entries = iter(entries)

    def __enter__(self) -> "_SeedDirectory":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> "_SeedDirectory":
        return self

    def __next__(self) -> _SeedEntry:
        return next(self._entries)

    def next(self) -> _SeedEntry | None:
        return next(self._entries, None)

    def extractfile(self, entry: _SeedEntry) -> Any:
        return entry.path.open("rb")


def _open_seed(source: Path) -> Any:
    if source.is_dir():
        return _SeedDirectory(source)
    return tarfile.open(source, "r:*")


def _compile_seed_source() -> Path | None:
    return _mounted_seed(COMPILE_SEED_ENV, COMPILE_SEED_FILE)


def _compile_seed_deltas(source: Path, root: Path | None = None) -> list[Path]:
    root = root or SEED_INPUT_ROOT
    stem = COMPILE_DELTA_FILE.rsplit(".", 1)[0]
    beside = sorted(path for path in source.parent.glob(stem + "*") if path != source)
    elsewhere: list[Path] = []
    if root.is_dir():
        for depth in range(5):
            for path in sorted(root.glob("/".join(["*"] * depth + [stem + "*"]))):
                if path.is_file() and path.parent != source.parent and path not in elsewhere:
                    elsewhere.append(path)
    return beside + elsewhere


def _seed_member_ok(member: tarfile.TarInfo) -> bool:
    name = member.name
    if not (member.isfile() or member.isdir()) or name.startswith("/") or ".." in Path(name).parts:
        return False
    return True


def compile_cache_roots() -> dict[str, Path]:
    return {"compile": COMPILE_CACHE_ROOT, "vllm": CACHE_ROOT / "vllm"}


def _gpu_card() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def stage_seed_file(source: Path, into: Path, threads: int = SEED_STAGE_THREADS,
                    piece: int = SEED_STAGE_PIECE) -> tuple[Path, int, float]:
    size = source.stat().st_size
    into.mkdir(parents=True, exist_ok=True)
    target = into / source.name
    began = time.monotonic()
    with target.open("wb") as out:
        out.truncate(size)

    def copy(offset: int) -> None:
        length = min(piece, size - offset)
        with source.open("rb") as reader, target.open("r+b") as writer:
            reader.seek(offset)
            writer.seek(offset)
            left = length
            while left:
                data = reader.read(left)
                if not data:
                    raise EOFError(f"{source} ended at {offset + length - left} of {size} bytes")
                writer.write(data)
                left -= len(data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        list(pool.map(copy, range(0, size, piece)))
    return target, size, time.monotonic() - began


def start_compile_seed() -> bool:
    source = _compile_seed_source()
    if source is None:
        return False
    box: dict[str, Any] = {}

    def run() -> None:
        began = time.monotonic()
        staged: dict[Path, Path] = {}
        try:
            if source.is_file():
                seeds = [*_compile_seed_deltas(source), source]
                read = 0
                read_began = time.monotonic()
                for seed in seeds:
                    local, size, _ = stage_seed_file(seed, SEED_STAGE_DIR)
                    staged[seed] = local
                    read += size
                box["read_seconds"] = round(time.monotonic() - read_began, 1)
                box["read_bytes"] = read
                print(f"COMPILE_SEED read {len(seeds)} seed file(s), {read / 2**30:.2f} GiB, from {source.parent} in "
                      f"{box['read_seconds']} s ({read / 1e9 / max(box['read_seconds'], 1e-3):.2f} GB/s) with "
                      f"{SEED_STAGE_THREADS} threads in {SEED_STAGE_PIECE >> 20} MiB pieces into {SEED_STAGE_DIR}",
                      flush=True)
            box["result"] = seed_compile_cache(staged=staged)
        except BaseException as failure:
            box["error"] = failure
        finally:
            shutil.rmtree(SEED_STAGE_DIR, ignore_errors=True)
            box["seconds"] = round(time.monotonic() - began, 1)

    thread = threading.Thread(target=run, name="compile-seed", daemon=True)
    thread.start()
    BACKGROUND["compile"] = {"thread": thread, "box": box}
    print(f"COMPILE_SEED unpacks in the background from {source} while the runtime is unpacked and patched; the "
          f"setup waits for it before the first kernel is built (the self-checks and the server launch)", flush=True)
    return True


def join_compile_seed(where: str) -> dict[str, Any]:
    pending = BACKGROUND.pop("compile", None)
    if pending is None:
        return seed_compile_cache()
    arrived = time.monotonic()
    pending["thread"].join()
    box = pending["box"]
    waited = time.monotonic() - arrived
    if "error" in box:
        print(f"COMPILE_SEED the background unpack failed after {box['seconds']} s "
              f"({type(box['error']).__name__}: {box['error']}); {where} unpacks it from the mount instead",
              flush=True)
        return seed_compile_cache()
    print(f"COMPILE_SEED background unpack took {box['seconds']} s (read {box.get('read_seconds', 0)} s); "
          f"{where} waited {waited:.1f} s for it", flush=True)
    result = dict(box.get("result") or {})
    result.update({"background_seconds": box["seconds"], "read_seconds": box.get("read_seconds"),
                   "waited_seconds": round(waited, 1)})
    return result


def seed_compile_cache(roots: dict[str, Path] | None = None,
                       staged: dict[Path, Path] | None = None) -> dict[str, Any]:
    source = _compile_seed_source()
    if source is None:
        if str(os.environ.get(COMPILE_SEED_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off"):
            print(f"COMPILE_SEED {COMPILE_SEED_ENV} is set but no {COMPILE_SEED_FILE} is mounted: the engine "
                  f"compiles every kernel itself", flush=True)
            return {"armed": True, "seeded": 0}
        return {"armed": False}
    landing = roots or compile_cache_roots()
    started = time.monotonic()
    files = size = skipped = 0
    sources = [*reversed(_compile_seed_deltas(source)), source]
    for seed in sources:
        try:
            with _open_seed((staged or {}).get(seed, seed)) as archive:
                for member in archive:
                    if member.name == COMPILE_MANIFEST:
                        held = json.loads(archive.extractfile(member).read().decode("utf-8"))
                        card = _gpu_card()
                        if card and held.get("card") and held["card"] != card:
                            print(f"COMPILE_SEED {seed} was exported on {held['card']!r}, this card is {card!r}: "
                                  f"nothing is unpacked from it", flush=True)
                            if seed == source:
                                return {"armed": True, "seeded": 0, "refused": "card"}
                            break
                        continue
                    parts = Path(member.name).parts
                    if not _seed_member_ok(member) or not parts or parts[0] not in landing:
                        skipped += 1
                        continue
                    target = landing[parts[0]].joinpath(*parts[1:])
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    SEED_HELD["COMPILE"] = SEED_HELD.get("COMPILE", 0) + (1 if member.isfile() else 0)
                    SEED_FILES.setdefault(member.name, int(member.size))
                    if member.isfile() and member.name.endswith(".so"):
                        SEED_LIBRARIES.add(member.name)
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    held = archive.extractfile(member)
                    if held is None:
                        skipped += 1
                        continue
                    temporary = target.with_name(f".{target.name}.seed")
                    with held, temporary.open("wb") as out:
                        shutil.copyfileobj(held, out, 1 << 22)
                    os.replace(temporary, target)
                    files += 1
                    size += member.size
        except (OSError, ValueError, tarfile.TarError) as failure:
            print(f"COMPILE_SEED {seed} could not be unpacked after {files} files ({type(failure).__name__}: "
                  f"{failure}); the engine compiles what is missing", flush=True)
            return {"armed": True, "seeded": files, "error": str(failure)[:300]}
    shown = str(source) + (f" and {len(sources) - 1} delta seed(s) ({', '.join(str(one) for one in sources[:-1])}), "
                           f"newest first, each file from the first seed that holds it" if len(sources) > 1 else "")
    print(f"COMPILE_SEED {files} cached kernels ({size / 2**20:.0f} MiB, {skipped} members refused) from {shown} "
          f"into {', '.join(str(path) for path in landing.values())} in {time.monotonic() - started:.1f} s",
          flush=True)
    return {"armed": True, "seeded": files, "bytes": size, "skipped": skipped, "source": str(source),
            "deltas": len(sources) - 1}


def harvest_min_new() -> int:
    said = str(os.environ.get(HARVEST_MIN_NEW_ENV) or "").strip()
    return int(said) if said.isdigit() else HARVEST_MIN_NEW_DEFAULT


def harvest_skip(kind: str, files: int) -> dict[str, Any] | None:
    held = SEED_HELD.get(kind)
    least = harvest_min_new()
    if held is None or files - held >= least:
        return None
    print(f"{kind}_HARVEST skipped: {files} files against {held} in the mounted seed, {files - held} new, below "
          f"{least} ({HARVEST_MIN_NEW_ENV}); the seed already covers this run", flush=True)
    return {"armed": True, "skipped": True, "files": files, "seed": held}


def harvest_compile_cache(roots: dict[str, Path] | None = None, out: Path | None = None) -> dict[str, Any]:
    if not _flag_on(COMPILE_HARVEST_ENV):
        return {"armed": False}
    sources = roots or compile_cache_roots()
    cached = [(f"{tag}/{path.relative_to(source).as_posix()}", path) for tag, source in sources.items()
              for path in sorted(source.rglob("*"))
              if path.is_file() and not path.name.endswith((".tmp", ".seed")) and ".tmp." not in path.name]
    seeded = "COMPILE" in SEED_HELD
    chosen = [(name, path) for name, path in cached
              if not seeded or SEED_FILES.get(name) != path.stat().st_size]
    if seeded and not chosen:
        print(f"COMPILE_HARVEST skipped: 0 new of {len(cached)} cached files, the mounted seed holds every one; "
              f"nothing written", flush=True)
        return {"armed": True, "skipped": True, "files": len(cached), "new": 0}
    full = out or (WORKING_DIR / COMPILE_SEED_FILE)
    target = full.with_name(COMPILE_DELTA_FILE) if seeded else full
    started = time.monotonic()
    temporary = target.with_name(f".{target.name}.tmp")
    count = 0
    try:
        with tarfile.open(temporary, "w") as archive:
            manifest = json.dumps({"card": _gpu_card(), "runtime": FLAT_MANIFEST_SHA256, "profile": RUNTIME_PROFILE,
                                   "delta": seeded}).encode("utf-8")
            info = tarfile.TarInfo(COMPILE_MANIFEST)
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            for name, path in chosen:
                archive.add(path, arcname=name, recursive=False)
                count += 1
        os.replace(temporary, target)
    except OSError as failure:
        temporary.unlink(missing_ok=True)
        print(f"COMPILE_HARVEST failed after {count} files: {type(failure).__name__}: {failure}", flush=True)
        return {"armed": True, "files": count, "error": str(failure)[:300]}
    size = target.stat().st_size
    what = (f"{count} new of {len(cached)} cached files (not in the mounted seed by name and size)" if seeded
            else f"{count} cached files, no seed was mounted")
    print(f"COMPILE_HARVEST {what}, {size / 2**20:.0f} MiB, written to {target} in "
          f"{time.monotonic() - started:.1f} s", flush=True)
    return {"armed": True, "files": count, "new": count if seeded else None, "of": len(cached), "bytes": size,
            "path": str(target), "delta": seeded}



TRACE: dict[str, Any] = {"module": None, "failed": False}
SETUP_MARKS: dict[str, float] = {}


def init_trace_on() -> bool:
    return _flag_on(INIT_TRACE_ENV)


def tracer() -> Any:
    if TRACE["module"] is not None or TRACE["failed"] or not init_trace_on():
        return TRACE["module"]
    try:
        os.environ.setdefault(INIT_TRACE_PATH_ENV, str(WORKING_DIR / "init_trace.json"))
        spec = importlib.util.spec_from_file_location("taaf_init_trace", BUNDLE_DIR / INIT_TRACE_HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Path(module.events_path()).unlink(missing_ok=True)
        module._STATE["role"] = "setup"
        module._STATE["pid"] = os.getpid()
        module.install_mount_audit()
        module.mark("setup process", argv=" ".join(sys.argv[:2])[:160])
        TRACE["module"] = module
        print(f"INIT_TRACE armed: events {module.events_path()}, json {module.trace_path()}", flush=True)
    except Exception as failure:
        TRACE["failed"] = True
        print(f"INIT_TRACE could not start, the setup goes on untraced: {type(failure).__name__}: {failure}",
              flush=True)
    return TRACE["module"]


def phase(name: str, **extra: Any) -> None:
    SETUP_MARKS[name] = time.time()
    module = tracer()
    if module is not None:
        module.mark("setup: " + name, **extra)


def phase_span(name: str, began: float, **extra: Any) -> None:
    SETUP_MARKS[name] = time.time()
    module = tracer()
    if module is not None:
        module.emit("setup: " + name, began, time.time(), **extra)


def install_init_trace(env: dict[str, str]) -> dict[str, Any]:
    module = tracer()
    if module is None:
        return {"armed": False}
    INIT_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    landed = INIT_TRACE_DIR / "sitecustomize.py"
    shutil.copyfile(BUNDLE_DIR / INIT_TRACE_HELPER, landed)
    env["PYTHONPATH"] = os.pathsep.join([str(INIT_TRACE_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env[INIT_TRACE_ENV] = "1"
    env[INIT_TRACE_PATH_ENV] = module.trace_path()
    imports = str(os.environ.get(INIT_TRACE_IMPORTS_ENV) or "").strip() or INIT_TRACE_IMPORTS_DEFAULT
    env[INIT_TRACE_IMPORTS_ENV] = imports
    print(f"INIT_TRACE the server and its children record milestones through {landed} (PYTHONPATH first entry); "
          f"INIT_T lines go to the server log, rows to {module.events_path()}, per-import times of "
          f"'{imports}' to {Path(module.trace_path()).parent}/importtime_<role>.txt", flush=True)
    return {"armed": True, "sitecustomize": str(landed), "events": module.events_path(), "imports": imports}


def finish_init_trace() -> dict[str, Any]:
    module = tracer()
    if module is None:
        return {"armed": False}
    module.mount_report("setup process")
    document = module.consolidate()
    count = len(document["events"]) if document else 0
    roles = sorted({row["role"] for row in (document["processes"].values() if document else ())})
    print(f"INIT_TRACE {count} rows from {len(document['processes']) if document else 0} processes {roles} "
          f"written to {module.trace_path()}; read it with python -m tools.startup_timeline <run>", flush=True)
    return {"armed": True, "rows": count, "path": module.trace_path()}


def launch_timing_line(popen_epoch: float) -> str:
    order = ["setup start", "runtime unpacked", "patches applied", "environment ready", "self-checks started",
             "weight cache ready", "model verified", "prune or link done"]
    parts = []
    before = SETUP_MARKS.get("setup start", popen_epoch)
    for name in order[1:]:
        at = SETUP_MARKS.get(name)
        if at is not None:
            parts.append(f"{name} +{at - before:.1f} s")
            before = at
    return (f"LAUNCH_TIMING the server process starts {popen_epoch - SETUP_MARKS.get('setup start', popen_epoch):.1f} s "
            f"into the setup: " + ", ".join(parts))


def _pyc_source(rel: str) -> str | None:
    parts = rel.split("/")
    tag = sys.implementation.cache_tag
    if len(parts) < 2 or parts[-2] != "__pycache__" or not parts[-1].endswith(f".{tag}.pyc"):
        return None
    return "/".join(parts[:-2] + [parts[-1][:-len(f".{tag}.pyc")] + ".py"])


def pyc_writes_allowed() -> bool:
    return (_flag_on(PYC_WRITE_ENV) or _flag_on(PYC_HARVEST_ENV)
            or str(os.environ.get(PYC_SEED_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off"))


def ensure_runtime_writable(site: Path) -> bool:
    probe = site / "vllm" / "__pycache__"
    for attempt in range(2):
        try:
            probe.mkdir(parents=True, exist_ok=True)
            test = probe / ".taaf-write-probe"
            test.write_bytes(b"x")
            test.unlink()
            return True
        except OSError as failure:
            if attempt == 0:
                subprocess.run(["chmod", "-R", "u+rwX", str(site)], check=False)
            else:
                print(f"PYC_WRITE {site} is not writable ({type(failure).__name__}: {failure}): imports cannot "
                      f"write bytecode there", flush=True)
    return False


def pyc_manifest_refusal(held: Any) -> str | None:
    if not isinstance(held, dict):
        return "the manifest is not an object"
    checks = (("tag", sys.implementation.cache_tag), ("magic", importlib.util.MAGIC_NUMBER.hex()),
              ("runtime_manifest_sha256", VLLM_RUNTIME_MANIFEST_SHA256), ("profile", RUNTIME_PROFILE))
    for key, want in checks:
        if held.get(key) != want:
            return f"{key} is {held.get(key)!r}, this setup needs {want!r}"
    if not isinstance(held.get("sources"), dict):
        return "the manifest names no sources"
    return None


def _land_pyc(site: Path, rel: str, data: bytes, want_sha: str) -> str:
    source = _pyc_source(rel)
    if source is None:
        return "name"
    path = site / source
    try:
        raw = path.read_bytes()
        stat = path.stat()
    except OSError:
        return "missing"
    if hashlib.sha256(raw).hexdigest() != want_sha:
        return "stale"
    if len(data) < 16 or data[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(data[4:8], "little") != 0:
        return "header"
    head = (data[:8] + (int(stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            + (stat.st_size & 0xFFFFFFFF).to_bytes(4, "little"))
    target = site / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.seed")
    temporary.write_bytes(head + data[16:])
    os.replace(temporary, target)
    return "ok"


def seed_pyc_cache(site: Path | None = None) -> dict[str, Any]:
    source = _mounted_seed(PYC_SEED_ENV, PYC_SEED_FILE)
    if source is None:
        if str(os.environ.get(PYC_SEED_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off"):
            print(f"PYC_SEED {PYC_SEED_ENV} is set but no {PYC_SEED_FILE} is mounted: every module compiles "
                  f"from source", flush=True)
            return {"armed": True, "seeded": 0}
        return {"armed": False}
    site = site or _runtime_site()
    began = time.time()
    tally: dict[str, int] = {}
    size = 0
    try:
        with _open_seed(source) as archive:
            first = archive.next()
            if first is None or first.name != PYC_MANIFEST:
                print(f"PYC_SEED {source} does not lead with {PYC_MANIFEST}: nothing is unpacked", flush=True)
                return {"armed": True, "seeded": 0, "refused": "manifest"}
            held = json.loads(archive.extractfile(first).read().decode("utf-8"))
            refusal = pyc_manifest_refusal(held)
            if refusal:
                print(f"PYC_SEED {source} was exported for another interpreter or runtime ({refusal}): nothing is "
                      f"unpacked", flush=True)
                return {"armed": True, "seeded": 0, "refused": refusal}
            sources = held["sources"]
            SEED_HELD["PYC"] = len(sources)
            with concurrent.futures.ThreadPoolExecutor(max_workers=PYC_WORKERS) as pool:
                futures = []
                for member in archive:
                    if not member.isfile():
                        continue
                    if not _seed_member_ok(member) or member.name not in sources:
                        tally["refused"] = tally.get("refused", 0) + 1
                        continue
                    handle = archive.extractfile(member)
                    futures.append((member.size, pool.submit(_land_pyc, site, member.name, handle.read(),
                                                              str(sources[member.name]))))
                for length, future in futures:
                    outcome = future.result()
                    tally[outcome] = tally.get(outcome, 0) + 1
                    size += length if outcome == "ok" else 0
    except (OSError, ValueError, tarfile.TarError) as failure:
        print(f"PYC_SEED {source} could not be unpacked ({type(failure).__name__}: {failure}); modules compile "
              f"from source", flush=True)
        return {"armed": True, "seeded": tally.get("ok", 0), "error": str(failure)[:300]}
    seconds = time.time() - began
    skipped = ", ".join(f"{key}: {value}" for key, value in sorted(tally.items()) if key != "ok")
    print(f"PYC_SEED {tally.get('ok', 0)} bytecode files ({size / 2**20:.0f} MiB) from {source} into {site} in "
          f"{seconds:.1f} s; skipped [{skipped}] (stale = the source differs from the harvest run's, missing = "
          f"no such source here)", flush=True)
    phase_span("pyc seed", began, files=tally.get("ok", 0), stale=tally.get("stale", 0))
    return {"armed": True, "seeded": tally.get("ok", 0), "bytes": size, "skipped": tally, "source": str(source),
            "seconds": round(seconds, 2)}


def start_pyc_seed(site: Path) -> None:
    if _mounted_seed(PYC_SEED_ENV, PYC_SEED_FILE) is None:
        PYC_RESULTS["seed"] = seed_pyc_cache(site)
        return
    box: dict[str, Any] = {}

    def run() -> None:
        began = time.monotonic()
        try:
            box["result"] = seed_pyc_cache(site)
        except BaseException as failure:
            box["error"] = failure
        box["seconds"] = round(time.monotonic() - began, 1)

    thread = threading.Thread(target=run, name="pyc-seed", daemon=True)
    thread.start()
    BACKGROUND["pyc"] = {"thread": thread, "box": box}
    print("PYC_SEED unpacks in a thread beside the self-checks; the server launch waits for it", flush=True)


def join_pyc_seed(where: str) -> dict[str, Any] | None:
    pending = BACKGROUND.pop("pyc", None)
    if pending is None:
        return PYC_RESULTS.get("seed")
    arrived = time.monotonic()
    pending["thread"].join()
    box = pending["box"]
    waited = time.monotonic() - arrived
    if "error" in box:
        print(f"PYC_SEED the thread failed after {box['seconds']} s ({type(box['error']).__name__}: "
              f"{box['error']}); modules compile from source", flush=True)
        PYC_RESULTS["seed"] = {"armed": True, "seeded": 0, "error": str(box["error"])[:300]}
        return PYC_RESULTS["seed"]
    print(f"PYC_SEED thread took {box['seconds']} s; {where} waited {waited:.1f} s for it", flush=True)
    PYC_RESULTS["seed"] = dict(box.get("result") or {}, waited_seconds=round(waited, 1))
    return PYC_RESULTS["seed"]


def harvest_pyc_cache(site: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    if not _flag_on(PYC_HARVEST_ENV):
        return {"armed": False}
    site = site or _runtime_site()
    target = out or (WORKING_DIR / PYC_SEED_FILE)
    began = time.time()
    tag = sys.implementation.cache_tag
    found: list[tuple[str, Path]] = []
    for root, _, names in os.walk(site):
        if os.path.basename(root) != "__pycache__":
            continue
        for name in names:
            if name.endswith(f".{tag}.pyc"):
                path = Path(root) / name
                found.append((path.relative_to(site).as_posix(), path))

    def digest(row: tuple[str, Path]) -> tuple[str, str] | None:
        source = _pyc_source(row[0])
        if source is None:
            return None
        try:
            return row[0], hashlib.sha256((site / source).read_bytes()).hexdigest()
        except OSError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=PYC_WORKERS) as pool:
        sources = dict(row for row in pool.map(digest, found) if row is not None)
    skip = harvest_skip("PYC", len(sources))
    if skip is not None:
        return skip
    temporary = target.with_name(f".{target.name}.tmp")
    count = 0
    try:
        with tarfile.open(temporary, "w") as archive:
            manifest = json.dumps({"tag": tag, "magic": importlib.util.MAGIC_NUMBER.hex(),
                                   "runtime_manifest_sha256": VLLM_RUNTIME_MANIFEST_SHA256,
                                   "profile": RUNTIME_PROFILE, "python": list(sys.version_info[:3]),
                                   "sources": sources}).encode("utf-8")
            info = tarfile.TarInfo(PYC_MANIFEST)
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            for rel, path in found:
                if rel in sources:
                    archive.add(path, arcname=rel, recursive=False)
                    count += 1
        os.replace(temporary, target)
    except OSError as failure:
        temporary.unlink(missing_ok=True)
        print(f"PYC_HARVEST failed after {count} files: {type(failure).__name__}: {failure}", flush=True)
        return {"armed": True, "files": count, "error": str(failure)[:300]}
    size = target.stat().st_size
    print(f"PYC_HARVEST {count} bytecode files, {size / 2**20:.0f} MiB, written to {target} in "
          f"{time.time() - began:.1f} s", flush=True)
    return {"armed": True, "files": count, "bytes": size, "path": str(target)}


def ready_poll_seconds() -> float:
    said = str(os.environ.get(READY_POLL_ENV) or "").strip()
    if not said:
        return READY_POLL_DEFAULT_S
    try:
        value = float(said)
    except ValueError:
        raise RuntimeError(f"{READY_POLL_ENV} must be a number of seconds, found {said!r}")
    return max(READY_POLL_MIN_S, min(READY_POLL_DEFAULT_S, value))


SELF_CHECKS: dict[str, Any] = {"pending": None, "box": None}


def selfcheck_overlap_mode() -> str:
    said = str(os.environ.get(SELFCHECK_OVERLAP_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said in ("1", "true", "yes", "on"):
        return "prune"
    if said == "2":
        return "server"
    raise ValueError(f"{SELFCHECK_OVERLAP_ENV} must be 0, 1 or 2, found {said!r}")


def release_verdict(verdict: Path, held: dict[str, Any]) -> None:
    if not verdict.is_file():
        temporary = verdict.with_name(f".{verdict.name}.tmp")
        temporary.write_text(json.dumps(held), encoding="utf-8")
        os.replace(temporary, verdict)


def predict_verdict_env(env: dict[str, str]) -> None:
    env[BF16C_VERDICT_WAIT_ENV] = str(BF16C_VERDICT_WAIT_S)
    if str(env.get(BF16C_FAST_ENV) or "").strip() == "1" and _flag_on(BF16C_ENV):
        env[BF16C_FAST_VERDICT_ENV] = str(WORKING_DIR / "bf16c-fast2-verdict.json")
    if str(env.get(BF16C_MIX_ENV) or "").strip() == "1" and _flag_on(BF16C_ENV):
        env[BF16C_MIX_VERDICT_ENV] = str(WORKING_DIR / "bf16c-mix-verdict.json")
    if str(env.get(DENSE_HOLE_ENV) or "").strip() == "1":
        env[DENSE_HOLE_VERDICT_ENV] = str(WORKING_DIR / DENSE_HOLE_VERDICT)


def qsa_gates_on() -> bool:
    return qsa_lean_on() or qsa_decode_lean_on() or qsa_union_decode_on()


def start_self_checks(env: dict[str, str], model_dir: Path) -> bool:
    if not selfcheck_overlap_mode():
        return False
    box: dict[str, Any] = {}
    first = selfcheck_overlap_mode() == "server" and qsa_gates_on()
    if first:
        began = time.monotonic()
        box["qsa"] = qsa_self_check(env, model_dir)
        box["qsa_seconds"] = round(time.monotonic() - began, 1)
        print(f"SELFCHECK_OVERLAP the QSA check ran first in {box['qsa_seconds']} s so the server launch carries its "
              f"gates; the BF16C and DENSE_HOLE checks run beside the launch and the engine waits for their verdict "
              f"files", flush=True)
    scratch = dict(env)
    base = dict(env)

    def run() -> None:
        began = time.monotonic()
        try:
            if not first:
                box["qsa"] = qsa_self_check(scratch, model_dir)
            box["fast"] = check_bf16c_fast(scratch)
            box["mix"] = check_bf16c_mix(scratch)
            box["hole"] = check_dense_hole(scratch)
        except BaseException as failure:
            box["error"] = failure
        box["seconds"] = round(time.monotonic() - began, 1)

    thread = threading.Thread(target=run, name="self-checks", daemon=True)
    thread.start()
    SELF_CHECKS["pending"] = {"thread": thread, "scratch": scratch, "base": base, "box": box,
                              "began": time.monotonic(), "qsa_first": first}
    where = ("the server launch waits for them" if selfcheck_overlap_mode() == "prune" else
             "the server launches at once and the setup waits for them before it reports ready")
    print(f"SELFCHECK_OVERLAP the QSA, BF16C FAST2 and BF16C MIX checks run in one thread beside the work that "
          f"follows; {where}", flush=True)
    return True


def join_self_checks(env: dict[str, str]) -> dict[str, Any] | None:
    pending, SELF_CHECKS["pending"] = SELF_CHECKS["pending"], None
    if pending is None:
        return None
    arrived = time.monotonic()
    pending["thread"].join()
    box = pending["box"]
    if "error" in box:
        raise box["error"]
    scratch, base = pending["scratch"], pending["base"]
    for key, value in scratch.items():
        if base.get(key) != value:
            env[key] = value
    for key in base:
        if key not in scratch:
            env.pop(key, None)
    print(f"SELFCHECK_OVERLAP joined: the three checks took {box['seconds']} s, the server launch waited "
          f"{time.monotonic() - arrived:.1f} s for them", flush=True)
    SELF_CHECKS["box"] = box
    return box


def _verdict_cache_file() -> Path | None:
    said = str(os.environ.get(VERDICT_CACHE_ENV) or "").strip()
    if said.lower() in ("", "0", "false", "no", "off"):
        return None
    return BUNDLE_DIR / VERDICT_CACHE_FILE if said.lower() in ("1", "true", "yes", "on") else Path(said)


def _verdict_key(kind: str, helper_bytes: bytes) -> str | None:
    try:
        card = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except Exception:
        return None
    material = {"version": 1, "kind": kind, "card": card, "helper": hashlib.sha256(helper_bytes).hexdigest(),
                "runtime": FLAT_MANIFEST_SHA256, "profile": RUNTIME_PROFILE}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _cached_bf16c_verdict(kind: str, helper: Path, verdict: Path, env: dict[str, str]) -> dict[str, Any] | None:
    table_path = _verdict_cache_file()
    if table_path is None:
        return None
    label = f"BF16C {kind}"
    try:
        table = json.loads(table_path.read_text(encoding="utf-8")) if table_path.is_file() else {}
        key = _verdict_key(kind, helper.read_bytes())
    except (OSError, ValueError) as failure:
        print(f"VERDICT_CACHE {label}: {table_path} unreadable ({type(failure).__name__}: {failure}); the full "
              f"self-check runs", flush=True)
        return None
    entry = (table.get("entries") or {}).get(key) if key else None
    report = (entry or {}).get("report")
    if not isinstance(report, dict) or not report.get("ok"):
        print(f"VERDICT_CACHE {label}: no entry for this card and helper "
              f"({'key ' + key[:16] if key else 'card unreadable'}) in {table_path}; the full self-check runs",
              flush=True)
        return None
    pending = verdict.with_name(f".{verdict.name}.cached")
    pending.write_text(json.dumps({**report, "cached": True}), encoding="utf-8")
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, str(helper), "spotcheck", kind, str(pending)], env=env,
                              capture_output=True, text=True, timeout=VERDICT_SPOT_TIMEOUT)
        said = done.stdout + done.stderr
        passed = done.returncode == 0
    except Exception as failure:
        said, passed = f"{type(failure).__name__}: {failure}", False
    for line in said.splitlines():
        if line.startswith("BF16C"):
            print(line, flush=True)
    seconds = round(time.monotonic() - started, 1)
    if passed:
        os.replace(pending, verdict)
    else:
        pending.unlink(missing_ok=True)
    if not passed:
        print(f"VERDICT_CACHE {label}: the spot check failed after {seconds} s ({said[-300:].strip()!r}); the full "
              f"self-check runs", flush=True)
        return None
    print(f"VERDICT_CACHE {label}: hit for key {key[:16]}, the spot check passed in {seconds} s; the cached "
          f"{report.get('seconds')} s self-check is skipped", flush=True)
    return {**report, "cached": True, "spot_seconds": seconds}


def _harvest_bf16c_verdict(kind: str, helper: Path, report: dict[str, Any]) -> None:
    if _verdict_cache_file() is None or not report.get("ok") or report.get("cached"):
        return
    try:
        key = _verdict_key(kind, helper.read_bytes())
        if key is None:
            return
        path = WORKING_DIR / VERDICT_HARVEST_FILE
        table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"entries": {}}
        table["entries"][key] = {"kind": kind, "report": report}
        path.write_text(json.dumps(table), encoding="utf-8")
        print(f"VERDICT_CACHE harvested BF16C {kind} under key {key[:16]} into {path}", flush=True)
    except Exception as failure:
        print(f"VERDICT_CACHE harvest of BF16C {kind} failed: {type(failure).__name__}: {failure}", flush=True)


def selfcheck_fatal(env: dict[str, str] | None = None) -> bool:
    said = str((env or {}).get(SELFCHECK_FATAL_ENV, os.environ.get(SELFCHECK_FATAL_ENV)) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{SELFCHECK_FATAL_ENV}={said!r} is not one of '', '0', '1'.")
    return said != "0"


def mamba_hit_check_on() -> bool:
    said = str(os.environ.get(MAMBA_HIT_CHECK_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MAMBA_HIT_CHECK_ENV}={said!r} is not one of '', '0', '1'.")
    _bounded_integer_environment(MAMBA_HIT_CHECK_MIN_ENV, default=8192, minimum=1, maximum=1 << 20)
    return said == "1"


def selfcheck_fail(env: dict[str, str], knob: str, detail: str) -> None:
    line = f"SELFCHECK FAIL {knob}: {detail}"
    print(line, flush=True)
    if selfcheck_fatal(env):
        raise RuntimeError(line)


def check_bf16c_fast(env: dict[str, str]) -> dict[str, Any]:
    if str(env.get(BF16C_FAST_ENV) or "").strip() != "1" or not _flag_on(BF16C_ENV):
        return {"armed": False}
    verdict = WORKING_DIR / "bf16c-fast2-verdict.json"
    verdict.unlink(missing_ok=True)
    helper = _runtime_site() / BF16C_INSTALLED
    started = time.monotonic()
    said = ""
    held = _cached_bf16c_verdict("fast", helper, verdict, env)
    try:
        if held is None:
            done = subprocess.run([sys.executable, str(helper), "selfcheck", str(verdict)], env=env,
                                  capture_output=True, text=True, timeout=BF16C_FAST_TIMEOUT)
            said = (done.stdout + done.stderr)[-6000:]
            held = json.loads(verdict.read_text(encoding="utf-8"))
            if done.returncode != 0 and held.get("ok"):
                held = {**held, "ok": False, "reason": f"the self-check exited {done.returncode}"}
            _harvest_bf16c_verdict("fast", helper, held)
    except Exception as failure:
        held = {"ok": False, "reason": f"{type(failure).__name__}: {failure}"[:500], "passed": []}
    for line in said.splitlines():
        if line.startswith("BF16C FAST2"):
            print(line, flush=True)
    seconds = round(time.monotonic() - started, 1)
    if not held.get("ok"):
        selfcheck_fail(env, BF16C_FAST_ENV, f"the {seconds} s self-check found {held.get('reason') or 'no verdict'}")
        release_verdict(verdict, held)
        env.pop(BF16C_FAST_ENV, None)
        os.environ.pop(BF16C_FAST_ENV, None)
        print(f"BF16C FAST2 stands aside for the whole run after a {seconds} s self-check: "
              f"{held.get('reason') or 'no verdict'}; the old kernel serves every shape", flush=True)
        if said and not any(line.startswith("BF16C FAST2 self-check") for line in said.splitlines()):
            print(said[-1500:], flush=True)
        return {"armed": False, "stood_aside": held.get("reason") or "no verdict", "seconds": seconds}
    env[BF16C_FAST_VERDICT_ENV] = str(verdict)
    print(f"BF16C FAST2 armed after a {seconds} s {'spot check of a cached verdict' if held.get('cached') else 'self-check'}: {len(held.get('passed', []))} of "
          f"{len(held.get('shapes', []))} shapes take the new kernel ({held.get('passed')}), the rest keep the old "
          f"one; verdict at {verdict}", flush=True)
    return {"armed": True, "passed": held.get("passed"), "seconds": seconds, "verdict": str(verdict)}


def check_bf16c_mix(env: dict[str, str]) -> dict[str, Any]:
    if str(env.get(BF16C_MIX_ENV) or "").strip() != "1" or not _flag_on(BF16C_ENV):
        return {"armed": False}
    verdict = WORKING_DIR / "bf16c-mix-verdict.json"
    verdict.unlink(missing_ok=True)
    helper = _runtime_site() / BF16C_INSTALLED
    started = time.monotonic()
    said = ""
    held = _cached_bf16c_verdict("mix", helper, verdict, env)
    try:
        if held is None:
            done = subprocess.run([sys.executable, str(helper), "mixcheck", str(verdict)], env=env,
                                  capture_output=True, text=True, timeout=BF16C_MIX_TIMEOUT)
            said = done.stdout + done.stderr
            held = json.loads(verdict.read_text(encoding="utf-8"))
            if done.returncode != 0 and held.get("ok"):
                held = {**held, "ok": False, "reason": f"the self-check exited {done.returncode}"}
            _harvest_bf16c_verdict("mix", helper, held)
    except Exception as failure:
        held = {"ok": False, "reason": f"{type(failure).__name__}: {failure}"[:500], "serve": {}}
    for line in said.splitlines():
        if line.startswith("BF16C MIX"):
            print(line, flush=True)
    seconds = round(time.monotonic() - started, 1)
    serve = held.get("serve") or {}
    bands = sorted(f"{shape}<={hi}" for shape, rows in serve.items() for hi, config in rows if config)
    if not held.get("ok"):
        selfcheck_fail(env, BF16C_MIX_ENV, f"the {seconds} s self-check found {held.get('reason') or 'no verdict'}")
        release_verdict(verdict, held)
        env.pop(BF16C_MIX_ENV, None)
        os.environ.pop(BF16C_MIX_ENV, None)
        print(f"BF16C MIX stands aside for the whole run after a {seconds} s self-check: "
              f"{held.get('reason') or 'no verdict'}; decode plus the stock GEMM serves every batch above the decode "
              f"rows", flush=True)
        if said and not any(line.startswith("BF16C MIX self-check") for line in said.splitlines()):
            print(said[-1500:], flush=True)
        return {"armed": False, "stood_aside": held.get("reason") or "no verdict", "seconds": seconds}
    env[BF16C_MIX_VERDICT_ENV] = str(verdict)
    print(f"BF16C MIX armed after a {seconds} s {'spot check of a cached verdict' if held.get('cached') else 'self-check'}: {len(bands)} of "
          f"{sum(len(rows) for rows in serve.values())} shape row bands take "
          f"the fused kernel ({', '.join(bands)}), the rest keep decode plus the stock GEMM; verdict at {verdict}",
          flush=True)
    return {"armed": True, "bands": bands, "seconds": seconds, "verdict": str(verdict)}


def check_dense_hole(env: dict[str, str]) -> dict[str, Any]:
    if str(env.get(DENSE_HOLE_ENV) or "").strip() != "1":
        return {"armed": False}
    verdict = WORKING_DIR / DENSE_HOLE_VERDICT
    verdict.unlink(missing_ok=True)
    helper = _runtime_site() / DENSE_HOLE_INSTALLED
    started = time.monotonic()
    said = ""
    try:
        done = subprocess.run([sys.executable, str(helper), "selfcheck", str(verdict)], env=env,
                              capture_output=True, text=True, timeout=DENSE_HOLE_TIMEOUT)
        said = done.stdout + done.stderr
        held = json.loads(verdict.read_text(encoding="utf-8"))
        if done.returncode != 0 and held.get("ok"):
            held = {**held, "ok": False, "reason": f"the self-check exited {done.returncode}"}
    except Exception as failure:
        held = {"ok": False, "reason": f"{type(failure).__name__}: {failure}"[:500], "serve": {}}
    for line in said.splitlines():
        if line.startswith("DENSE_HOLE"):
            print(line, flush=True)
    seconds = round(time.monotonic() - started, 1)
    if not held.get("ok"):
        selfcheck_fail(env, DENSE_HOLE_ENV, f"the {seconds} s self-check found {held.get('reason') or 'no verdict'}")
        verdict.unlink(missing_ok=True)
        release_verdict(verdict, held)
        env.pop(DENSE_HOLE_ENV, None)
        os.environ.pop(DENSE_HOLE_ENV, None)
        reason = held.get("reason") or "no verdict"
        print(f"DENSE_HOLE stood aside: {reason} (after a {seconds} s self-check); cuBLAS serves every row count",
              flush=True)
        if said and not any(line.startswith("DENSE_HOLE self-check") for line in said.splitlines()):
            print(said[-1500:], flush=True)
        return {"armed": False, "stood_aside": reason, "seconds": seconds}
    env[DENSE_HOLE_VERDICT_ENV] = str(verdict)
    serve = held.get("serve") or {}
    wins = sorted(f"{shape}@{rows}" for shape, bands in serve.items() for rows, config in bands if config)
    print(f"DENSE_HOLE armed after a {seconds} s self-check: {len(wins)} of "
          f"{sum(len(bands) for bands in serve.values())} shape row counts take the Triton kernel "
          f"({', '.join(wins)}), the rest keep cuBLAS; verdict at {verdict}", flush=True)
    return {"armed": True, "wins": wins, "seconds": seconds, "verdict": str(verdict)}


def _bundle_table_patch(env_name: str, helper_name: str, table_name: str, installed_name: str) -> dict[str, Any]:
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{env_name} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    helper_path = BUNDLE_DIR / helper_name
    table_path = BUNDLE_DIR / table_name
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{env_name} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    site = _runtime_site()
    files, made = _apply_pinned_targets(env_name, namespace["TARGETS"], site)
    installed = site / installed_name
    fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {env_name} helper {installed_name} did not land")
    return {"armed": True, "already": not made and not fresh, "table_sha256": hashlib.sha256(table).hexdigest(),
            "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}


def gdn_align_tiles() -> int:
    said = str(os.environ.get(GDN_ALIGN_TILES_ENV) or "").strip()
    if said in ("", "0"):
        return 0
    if not said.isdigit() or not 1 <= int(said) <= GDN_ALIGN_TILES_LIMIT:
        raise RuntimeError(f"{GDN_ALIGN_TILES_ENV}={said!r} is not '', '0' or a whole number of tiles from 1 to "
                           f"{GDN_ALIGN_TILES_LIMIT}.")
    return int(said)


def patch_gdn_align_tiles() -> dict[str, Any]:
    tiles = gdn_align_tiles()
    if not tiles:
        return {"armed": False}
    row = _bundle_table_patch(GDN_ALIGN_TILES_ENV, GDN_ALIGN_TILES_HELPER, GDN_ALIGN_TILES_TABLE,
                              GDN_ALIGN_TILES_INSTALLED)
    row["tiles"] = tiles
    print(f"GDN_ALIGN_TILES {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}, "
          f"helper at {GDN_ALIGN_TILES_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in row["files"].items())
          + f"; the Mamba align pre-copy and post-copy launch {tiles} temporal tile(s) per request and state "
          "instead of the stock count, read on every eager call", flush=True)
    return row


def gdn_cp_split_granularity() -> int:
    said = os.environ.get(GDN_CP_SPLIT_ENV, "").strip().lower()
    if said in ("", "0"):
        return 0
    if said == "1":
        return GDN_CP_SPLIT_FINE
    if said.isdigit() and int(said) % GDN_CP_SPLIT_FINE == 0 and GDN_CP_SPLIT_FINE <= int(said) < GDN_CP_SPLIT_STOCK:
        return int(said)
    raise RuntimeError(f"{GDN_CP_SPLIT_ENV}={said!r} is not '', '0', '1' or a multiple of {GDN_CP_SPLIT_FINE} "
                       f"below {GDN_CP_SPLIT_STOCK}.")


def patch_gdn_cp_split() -> dict[str, Any]:
    granularity = gdn_cp_split_granularity()
    if not granularity:
        return {"armed": False}
    row = _bundle_table_patch(GDN_CP_SPLIT_ENV, GDN_CP_SPLIT_HELPER, GDN_CP_SPLIT_TABLE, GDN_CP_SPLIT_INSTALLED)
    row["granularity"] = granularity
    print(f"GDN_CP_SPLIT {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}, "
          f"helper at {GDN_CP_SPLIT_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in row["files"].items())
          + f"; a single-sequence GDN chunked prefill picks its context-parallel chunk length on a "
          f"{granularity}-token grid instead of {GDN_CP_SPLIT_STOCK}, still at most one wave of chunks, after "
          f"its first splits match the sequential kernel as closely as the stock split does", flush=True)
    return row


def gdn_commit_cap(stream_said: str = "") -> int:
    said = str(os.environ.get(GDN_COMMIT_CAP_ENV) or "").strip()
    if said:
        try:
            cap = int(said)
        except ValueError:
            cap = -1
        if not 0 <= cap <= GDN_COMMIT_CAP_LIMIT:
            raise RuntimeError(f"{GDN_COMMIT_CAP_ENV}={said!r} is not a whole number from 0 to "
                               f"{GDN_COMMIT_CAP_LIMIT}.")
    else:
        cap = GDN_COMMIT_CAP_STREAM_DEFAULT if stream_said == "3" else 0
    fused = os.environ.get(GDN_FUSED_ENV, "").strip().lower()
    if cap and fused not in ("1", "defer", "tiled"):
        raise RuntimeError(f"The capped GDN commit ({GDN_COMMIT_CAP_ENV}, or {GDN_COMMIT_STREAM_ENV}=3) caps the "
                           f"grid of the deferred commit and needs {GDN_FUSED_ENV}=defer or tiled, found "
                           f"{fused!r}.")
    return cap


def patch_gdn_commit_stream() -> dict[str, Any]:
    said = str(os.environ.get(GDN_COMMIT_STREAM_ENV) or "").strip().lower()
    effective = gdn_commit_cap(said)
    if said in ("", "0", "off", "false", "no"):
        return {"armed": False}
    if said not in ("1", "2", "3", "on", "true", "yes"):
        raise RuntimeError(f"{GDN_COMMIT_STREAM_ENV}={said!r} is not '', '0', '1', '2' or '3'.")
    row = _bundle_table_patch(GDN_COMMIT_STREAM_ENV, GDN_COMMIT_STREAM_HELPER, GDN_COMMIT_STREAM_TABLE,
                              GDN_COMMIT_STREAM_INSTALLED)
    print(f"GDN_COMMIT_STREAM {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}, "
          f"helper at {GDN_COMMIT_STREAM_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in row["files"].items())
          + "; the GDN state commit and the align postprocess run on a side stream forked after the sampler "
          "and joined at the top of the next execute_model, so the drafter passes overlap them"
          + ("; the drafter runs on a higher-priority stream so its kernels outrank the commit's"
             if said == "2" else "")
          + (f"; the commit decides every row in one program and folds only the rows that cross a boundary, "
             f"through {effective} row slot(s), so the stream holds few CTAs" if effective else ""),
          flush=True)
    return row


def dense_hole_on() -> bool:
    said = str(os.environ.get(DENSE_HOLE_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{DENSE_HOLE_ENV} must be 0 or 1, found {said!r}")
    return said == "1"


def dense_hole_rows(tuning: dict[str, Any] | None) -> list[int]:
    if not tuning:
        return []
    try:
        sizes = cudagraph_capture_sizes(tuning)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return []
    budget = int(tuning.get("max_num_batched_tokens") or 0)
    return sorted({size for size in [*sizes, budget] if DENSE_HOLE_ROWS_MIN <= size <= DENSE_HOLE_ROWS_MAX})


def patch_dense_hole(tuning: dict[str, Any] | None = None) -> dict[str, Any]:
    if not dense_hole_on():
        return {"armed": False}
    row = _bundle_table_patch(DENSE_HOLE_ENV, DENSE_HOLE_HELPER, DENSE_HOLE_TABLE, DENSE_HOLE_INSTALLED)
    rows = dense_hole_rows(tuning)
    if rows:
        os.environ[DENSE_HOLE_ROWS_ENV] = ",".join(str(size) for size in rows)
    else:
        os.environ.pop(DENSE_HOLE_ROWS_ENV, None)
    os.environ[DENSE_HOLE_WAIT_ENV] = str(DENSE_HOLE_WAIT_S)
    row["rows"] = rows
    print(f"DENSE_HOLE {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}, "
          f"helper at {DENSE_HOLE_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in row["files"].items())
          + f"; every unquantized BF16 2560x6144 linear without a bias (o_proj, GDN out_proj) goes through one op "
          f"that runs a split-K Triton GEMM (BF16 operands, fp32 accumulation, partials summed in fp32 in split "
          f"order, one rounding to BF16) at the row counts where a self-check on this card before the server "
          f"starts finds it 3% faster than cuBLAS, bit-stable and exact under a CUDA-graph replay, and cuBLAS at "
          f"every other; the check times "
          + (f"the {len(rows)} served row counts {rows[0]}-{rows[-1]}" if rows else "its default row counts"),
          flush=True)
    return row


def patch_gdn_accept(tuning: dict[str, Any]) -> dict[str, Any]:
    if not _flag_on(GDN_ACCEPT_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{GDN_ACCEPT_ENV} edits vLLM 0.30.0's GDN, Mamba-state and model-runner files; the "
            f"{RUNTIME_PROFILE} runtime ships other ones.")
    drafts = int(tuning.get("mtp_speculative_tokens") or 0)
    if drafts <= 0:
        raise RuntimeError(
            f"{GDN_ACCEPT_ENV} keeps only the accepted MTP state, but MTP is off "
            f"({MTP_TOKENS_ENV} resolves to {drafts}); there is no draft state to drop.")
    helper_path = BUNDLE_DIR / GDN_ACCEPT_HELPER
    table_path = BUNDLE_DIR / GDN_ACCEPT_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{GDN_ACCEPT_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular GDN source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{GDN_ACCEPT_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{GDN_ACCEPT_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / GDN_ACCEPT_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the GDN_ACCEPT helper {GDN_ACCEPT_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.accept.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the GDN_ACCEPT edit of {relative} did not land")
    keep = _flag_on(CACHE_KEEP_ENV)
    row = {"armed": True, "already": not made and not helper_fresh,
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "mtp_speculative_tokens": drafts,
           "state_blocks_per_request_per_group": {"stock": 1 + drafts, "accept": 1},
           "files": files}
    print(f"GDN_ACCEPT {'already armed' if row['already'] else 'armed'}: {len(made)} of "
          f"{len(targets)} vLLM file(s) edited from table {row['table_sha256'][:16]}, helper at "
          f"{GDN_ACCEPT_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; the {drafts} MTP drafts are verified from the running GDN state and only the "
          f"accepted prefix is written, so each Mamba cache group holds 1 state block per running "
          f"request instead of {1 + drafts}"
          + (", plus at most one that CACHE_KEEP holds as the newest cached state or prompt-tail "
             "copy" if keep else ""), flush=True)
    return row


def patch_gdn_state_q8() -> dict[str, Any]:
    if _knob_value(GDN_Q8_FIRST_ENV) and not _knob_value(GDN_STATE_Q8_ENV):
        raise RuntimeError(f"{GDN_Q8_FIRST_ENV} names the first corrupt int8 state row the verify reads and "
                           f"needs {GDN_STATE_Q8_ENV}=1.")
    if not _knob_value(GDN_STATE_Q8_ENV):
        return {"armed": False}
    short = str(os.environ.get(GDN_SHORT_PREFILL_ENV) or "").strip().lower()
    if short not in ("", "0", "off", "false"):
        raise RuntimeError(f"{GDN_SHORT_PREFILL_ENV}={short!r} runs short prefills through a kernel that reads and "
                           f"writes the state as 16-bit values, while {GDN_STATE_Q8_ENV}=1 keeps int8 codes and row "
                           "scales in those bytes; set one of them off.")
    fused = os.environ.get(GDN_FUSED_ENV, "").strip().lower()
    if GDN_FUSED_MODES.get(fused) != "tiled":
        raise RuntimeError(f"{GDN_STATE_Q8_ENV} quantizes the state in the value-tiled verify and needs "
                           f"{GDN_FUSED_ENV}=tiled, found {fused!r}.")
    if os.environ.get(MAMBA_SSM_DTYPE_ENV, "").strip() not in ("", "bfloat16", "float16", "float32"):
        raise RuntimeError(f"{GDN_STATE_Q8_ENV} keeps int8 codes and row scales inside each state block and needs "
                           f"a 16- or 32-bit {MAMBA_SSM_DTYPE_ENV}.")
    table_path = BUNDLE_DIR / GDN_STATE_Q8_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_STATE_Q8_ENV} was set but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_STATE_Q8_ENV, namespace["TARGETS"], _runtime_site())
    row = {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}
    print(f"GDN_STATE_Q8 {'already armed' if row['already'] else 'armed'}: each GDN state block holds int8 codes and "
          "one fp32 scale per state row in its own bytes; the tiled verify, the deferred commit, the resume fold "
          "and the in-place prefill chunk quantize on write and dequantize on read, and the stock recurrent and "
          "chunk paths run on dequantized copies of their slots", flush=True)
    return row


def patch_gdn_state_q8_pack() -> dict[str, Any]:
    said = str(os.environ.get(GDN_STATE_Q8_PACK_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_STATE_Q8_PACK_ENV}={said!r} is not one of '', '0', '1'.")
    if said != "1":
        return {"armed": False}
    if not _knob_value(GDN_STATE_Q8_ENV):
        raise RuntimeError(f"{GDN_STATE_Q8_PACK_ENV} packs the int8 GDN states two to a pool block and needs "
                           f"{GDN_STATE_Q8_ENV}=1.")
    if str(os.environ.get(GDN_STATE_INPLACE_ENV) or "").strip() == "1":
        raise RuntimeError(f"{GDN_STATE_INPLACE_ENV}=1 compares whole 16-bit states and cannot run on packed int8 "
                           f"states; set it off with {GDN_STATE_Q8_PACK_ENV}=1.")
    if not _flag_on(HOST_KV_ENV):
        raise RuntimeError(f"{GDN_STATE_Q8_PACK_ENV} pairs the Mamba groups inside the HiSparse pool layout and "
                           f"needs {HOST_KV_ENV}=1.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_STATE_Q8_PACK_ENV} edits vLLM 0.30.0's GDN layer and HiSparse layout; the "
                           f"{RUNTIME_PROFILE} runtime ships other ones.")
    helper_path = BUNDLE_DIR / GDN_STATE_Q8_PACK_HELPER
    table_path = BUNDLE_DIR / GDN_STATE_Q8_PACK_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{GDN_STATE_Q8_PACK_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = _runtime_site()
    installed = site / GDN_STATE_Q8_PACK_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {GDN_STATE_Q8_PACK_ENV} helper {GDN_STATE_Q8_PACK_INSTALLED} did not land")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_STATE_Q8_PACK_ENV, namespace["TARGETS"], site)
    row = {"armed": True, "already": not made and not helper_fresh,
           "table_sha256": hashlib.sha256(table).hexdigest(), "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "files": files}
    print(f"GDN_STATE_Q8_PACK {'already armed' if row['already'] else 'armed'}: each GDN layer declares only its "
          "convolution window and int8 codes and row scales as its state, so two GDN layers of a lane share one "
          "pool block and a lane holds about half as many Mamba blocks; the HiSparse layout pairs the Mamba "
          "groups, and the Q8 kernels read each state through a full-shape view", flush=True)
    return row


def patch_mamba_zero_new() -> dict[str, Any]:
    said = os.environ.get(MAMBA_ZERO_NEW_ENV, "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MAMBA_ZERO_NEW_ENV}={said!r} is not one of '', '0', '1'.")
    if said == "0":
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{MAMBA_ZERO_NEW_ENV} edits vLLM 0.30.0's Mamba manager and KV block zeroer; the "
                           f"{RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / MAMBA_ZERO_NEW_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{MAMBA_ZERO_NEW_ENV} was set but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(MAMBA_ZERO_NEW_ENV, namespace["TARGETS"], _runtime_site())
    row = {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}
    print(f"MAMBA_ZERO_NEW {'already armed' if row['already'] else 'armed'}: blocks the scheduler hands a Mamba "
          "group join the new-block zeroing and the worker's zeroer covers every Mamba state's bytes in a block, "
          "so a fresh state block reads as zeros instead of the bytes its previous owner left", flush=True)
    return row


def gdn_verify_cuda_variant() -> str:
    said = os.environ.get(GDN_VERIFY_CUDA_ENV, "").strip()
    if said in ("", "0"):
        return ""
    try:
        parts = [int(part) for part in said.split(",")]
    except ValueError:
        parts = []
    low = (len(parts) == 4 and parts[3] == 1 and parts[0] in (64, 128) and parts[1] in (0, 4, 6, 8, 12, 16)
           and parts[2] in (0, 1, 2))
    legacy = (len(parts) == 4 and parts[3] == 0 and parts[0] in (32, 64, 128) and parts[1] in (2, 4, 8)
              and parts[2] in (0, 1))
    if not (low or legacy):
        raise RuntimeError(f"{GDN_VERIFY_CUDA_ENV}={said!r} is not '', '0' or four whole numbers: block_v 64 or 128, "
                           "min_blocks 0/4/6/8/12/16, load 0/1/2 and 1 for the low-register kernel, or block_v "
                           "32/64/128, accumulators 2/4/8, prefetch 0/1 and 0 for the earlier one.")
    return said


def gdn_verify_cuda_coherent(variant: str) -> bool:
    said = str(os.environ.get(GDN_VERIFY_CUDA_COHERENT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_VERIFY_CUDA_COHERENT_ENV}={said!r} is not one of '', '0', '1'.")
    if said == "1" and not variant.endswith(",1"):
        raise RuntimeError(f"{GDN_VERIFY_CUDA_COHERENT_ENV} shapes the low-register CUDA verify and needs "
                           f"{GDN_VERIFY_CUDA_ENV} armed with a bv,min_blocks,load,1 variant.")
    return said == "1"


def stage_gdn_verify_cuda(site: Path, mode: str) -> dict[str, Any]:
    variant = gdn_verify_cuda_variant()
    coherent = gdn_verify_cuda_coherent(variant)
    if variant and mode != "tiled":
        raise RuntimeError(f"{GDN_VERIFY_CUDA_ENV} replaces the tiled verify's kernel and needs {GDN_FUSED_ENV}=tiled.")
    if variant and os.environ.get(GDN_STATE_Q8_ENV, "").strip() != "1":
        raise RuntimeError(f"{GDN_VERIFY_CUDA_ENV} runs on the int8 recurrent state and needs {GDN_STATE_Q8_ENV}=1.")
    helper_path = BUNDLE_DIR / GDN_VERIFY_CUDA_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{GDN_FUSED_ENV} stages {GDN_VERIFY_CUDA_HELPER} for {GDN_VERIFY_CUDA_ENV} and "
                           f"{helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    installed = site / GDN_VERIFY_CUDA_INSTALLED
    fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {GDN_VERIFY_CUDA_ENV} helper {GDN_VERIFY_CUDA_INSTALLED} did not land")
    digest = hashlib.sha256(helper).hexdigest()
    print(f"GDN_VERIFY_CUDA {'armed at boot' if variant else 'staged, off at boot'}: helper at "
          f"{GDN_VERIFY_CUDA_INSTALLED} {digest[:16]}; variant {variant or 'none'}; the uniform 5-token decode verify "
          "runs the CUDA kernel when the knob names a variant, built before the graphs are captured, and every "
          "other verify call keeps the Triton kernel"
          + (f"; {GDN_VERIFY_CUDA_COHERENT_ENV}=1: its inputs and indices load through coherent global loads"
             if coherent else ""), flush=True)
    return {"staged": True, "variant": variant, "helper_sha256": digest, "fresh": fresh, "coherent": coherent}


def patch_gdn_fused() -> dict[str, Any]:
    value = os.environ.get(GDN_FUSED_ENV, "").strip().lower()
    fill = os.environ.get(GDN_SLOT_FILL_ENV, "").strip().lower()
    if fill not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_SLOT_FILL_ENV}={fill!r} is not one of '', '0', '1'.")
    if fill == "1" and value in ("", "0"):
        raise RuntimeError(f"{GDN_SLOT_FILL_ENV} clears GDN_FUSED's pending counts and needs {GDN_FUSED_ENV} "
                           f"armed.")
    if value in ("", "0"):
        return {"armed": False}
    if value not in GDN_FUSED_MODES:
        raise RuntimeError(f"{GDN_FUSED_ENV}={value!r} is not one of {sorted(GDN_FUSED_MODES)}")
    if not _flag_on(GDN_ACCEPT_ENV):
        raise RuntimeError(f"{GDN_FUSED_ENV} fuses the GDN_ACCEPT decode path and needs "
                           f"{GDN_ACCEPT_ENV} armed.")
    steps = str(os.environ.get(GDN_LAZY_STEPS_ENV) or "").strip()
    if steps and (GDN_FUSED_MODES[value] != "lazy" or not steps.isdigit() or not 1 <= int(steps) <= 7):
        raise RuntimeError(f"{GDN_LAZY_STEPS_ENV}={steps!r} sizes the lazy GDN ring and needs {GDN_FUSED_ENV}=lazy "
                           f"and a whole number of steps from 1 to 7.")
    helper_path = BUNDLE_DIR / GDN_FUSED_HELPER
    table_path = BUNDLE_DIR / GDN_FUSED_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{GDN_FUSED_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{GDN_FUSED_ENV} needs {path} installed first")
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{GDN_FUSED_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{GDN_FUSED_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / GDN_FUSED_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the GDN_FUSED helper {GDN_FUSED_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.fused.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the GDN_FUSED edit of {relative} did not land")
    mode = GDN_FUSED_MODES[value]
    verify_cuda = stage_gdn_verify_cuda(site, mode)
    row = {"armed": True, "mode": mode, "already": not made and not helper_fresh and not verify_cuda["fresh"],
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files, "verify_cuda": verify_cuda}
    print(f"GDN_FUSED {'already armed' if row['already'] else 'armed'} ({mode}): {len(made)} of "
          f"{len(targets)} file(s) edited from table {row['table_sha256'][:16]}, helper at "
          f"{GDN_FUSED_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + ("; the stock convolution and gated norm stay and the recurrence runs in value-tiled "
             "programs that fold the previous step's accepted prefix into the state they read and "
             "write it back once" if mode == "tiled" else
             "; the tiled verify folds a per-request ring of up to "
             f"{steps or 3} steps of accepted tokens into the state it reads and writes the state only when the "
             "ring would overflow, a cache or reply boundary is crossed or the request leaves its slot; a GPU "
             "self-check at the first GDN layer compares it bit for bit with tiled and makes it stand aside to "
             "tiled on any difference" if mode == "lazy" else
             "; each GDN layer's decode runs conv, gates, the recurrence over the request's rows "
             "and the gated norm in one kernel")
          + ("; the accepted prefix is folded at the next step's state read, so a decode step "
             "reads and writes each request's recurrent state once" if mode == "defer" else ""),
          flush=True)
    if fill == "1":
        row["slot_fill"] = True
        print("GDN_SLOT_FILL armed: a new request clears its pending accepted count in every GDN layer with a "
              "device fill on the current stream in place of a host scalar copy and its stream sync",
              flush=True)
    return row


def patch_gdn_meta_once() -> dict[str, Any]:
    value = os.environ.get(GDN_META_ONCE_ENV, "").strip().lower()
    sliced = os.environ.get(GDN_META_SLICE_ENV, "").strip().lower()
    if sliced not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_META_SLICE_ENV}={sliced!r} is not one of '', '0', '1'.")
    if sliced == "1" and value != "1":
        raise RuntimeError(f"{GDN_META_SLICE_ENV} slices the state indices GDN_META_ONCE derives per group and "
                           f"needs {GDN_META_ONCE_ENV}=1.")
    prep = os.environ.get(PREP_GDN_ENV, "").strip().lower()
    if prep not in ("", "0", "1"):
        raise RuntimeError(f"{PREP_GDN_ENV}={prep!r} is not one of '', '0', '1'.")
    if prep == "1" and (value != "1" or sliced != "1" or not _flag_on(GDN_ACCEPT_ENV)):
        raise RuntimeError(f"{PREP_GDN_ENV} derives every GDN group's state indices in one kernel from the "
                           f"metadata GDN_META_ONCE shares and needs {GDN_META_ONCE_ENV}=1, "
                           f"{GDN_META_SLICE_ENV}=1 and {GDN_ACCEPT_ENV} armed.")
    if value in ("", "0"):
        return {"armed": False}
    if value != "1":
        raise RuntimeError(f"{GDN_META_ONCE_ENV}={value!r} is not one of '', '0', '1'.")
    if not _flag_on(GDN_ACCEPT_ENV):
        raise RuntimeError(f"{GDN_META_ONCE_ENV} shares the GDN metadata through the GDN_ACCEPT helper and "
                           f"needs {GDN_ACCEPT_ENV} armed.")
    helper_path = BUNDLE_DIR / GDN_META_ONCE_HELPER
    table_path = BUNDLE_DIR / GDN_META_ONCE_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{GDN_META_ONCE_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    if not (site / GDN_ACCEPT_INSTALLED).is_file():
        raise RuntimeError(f"{GDN_META_ONCE_ENV} needs the GDN_ACCEPT helper {GDN_ACCEPT_INSTALLED} installed first")
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{GDN_META_ONCE_ENV} needs {path} installed first")
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{GDN_META_ONCE_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{GDN_META_ONCE_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / GDN_META_ONCE_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the GDN_META_ONCE helper {GDN_META_ONCE_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.metaonce.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the GDN_META_ONCE edit of {relative} did not land")
    row = {"armed": True, "already": not made and not helper_fresh, "prep_gdn": prep == "1",
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}
    if prep == "1":
        print("PREP_GDN armed: a pure speculative decode step writes every GDN cache group's state indices with "
              "one kernel, and the groups share one slot buffer, one accepted-count tensor and one metadata "
              "copy per step", flush=True)
    print(f"GDN_META_ONCE {'already armed' if row['already'] else 'armed'}: {len(made)} of {len(targets)} "
          f"file(s) edited from table {row['table_sha256'][:16]}, helper at {GDN_META_ONCE_INSTALLED} "
          f"{row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + "; each attention-metadata pass builds the GDN metadata for its first GDN cache group and every "
          "other group of the same shape reuses it with its own state indices gathered, the persistent "
          "graph buffers shared across the GDN builders", flush=True)
    return row


def patch_gdn_spec_prefix() -> dict[str, Any]:
    value = os.environ.get(GDN_SPEC_PREFIX_ENV, "").strip().lower()
    if value in ("", "0"):
        return {"armed": False}
    if value != "1":
        raise RuntimeError(f"{GDN_SPEC_PREFIX_ENV}={value!r} is not one of '', '0', '1'.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_SPEC_PREFIX_ENV} edits vLLM 0.30.0's GDN files; the {RUNTIME_PROFILE} "
                           f"runtime ships other ones.")
    helper_path = BUNDLE_DIR / GDN_SPEC_PREFIX_HELPER
    table_path = BUNDLE_DIR / GDN_SPEC_PREFIX_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{GDN_SPEC_PREFIX_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{GDN_SPEC_PREFIX_ENV} needs {path} installed first")
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        before = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{GDN_SPEC_PREFIX_ENV} cannot apply: {relative} is {before[:16]}, neither the "
                    f"{pinned[:16]} its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{GDN_SPEC_PREFIX_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / GDN_SPEC_PREFIX_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the GDN_SPEC_PREFIX helper {GDN_SPEC_PREFIX_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.prefix.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the GDN_SPEC_PREFIX edit of {relative} did not land")
    row = {"armed": True, "already": not made and not helper_fresh,
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}
    print(f"GDN_SPEC_PREFIX {'already armed' if row['already'] else 'armed'}: {len(made)} of {len(targets)} "
          f"file(s) edited from table {row['table_sha256'][:16]}, helper at {GDN_SPEC_PREFIX_INSTALLED} "
          f"{row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + "; a mixed step whose speculative rows lead the batch slices each GDN layer's tokens in place "
          "of index_select and index_copy", flush=True)
    return row


def _knob_value(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{name}={value!r} is not one of '', '0', '1'.")
    return value == "1"


def _patch_with_table(env_name: str, helper_name: str, table_name: str, installed_name: str) -> dict[str, Any]:
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{env_name} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    helper_path = BUNDLE_DIR / helper_name
    table_path = BUNDLE_DIR / table_name
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{env_name} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    site = _runtime_site()
    files, made = _apply_pinned_targets(env_name, namespace["TARGETS"], site)
    installed = site / installed_name
    fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {env_name} helper {installed_name} did not land")
    return {"armed": True, "already": not made and not fresh, "table_sha256": hashlib.sha256(table).hexdigest(),
            "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}


def _install_helper(env_name: str, helper_name: str, installed_name: str) -> str:
    helper_path = BUNDLE_DIR / helper_name
    if not helper_path.is_file():
        raise RuntimeError(f"{env_name} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    installed = _runtime_site() / installed_name
    if not (installed.is_file() and installed.read_bytes() == helper):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {env_name} helper {installed_name} did not land")
    return hashlib.sha256(helper).hexdigest()


def _armed_said(name: str, row: dict[str, Any], installed: str, what: str) -> str:
    return (f"{name} {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}, helper at "
            f"{installed} {row['helper_sha256'][:16]}; "
            + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                        for relative, held in row["files"].items())
            + f"; {what}")


def patch_draft_meta() -> dict[str, Any]:
    if not _knob_value(DRAFT_META_ENV):
        return {"armed": False}
    row = _patch_with_table(DRAFT_META_ENV, DRAFT_META_HELPER, DRAFT_META_TABLE, DRAFT_META_INSTALLED)
    print(_armed_said("DRAFT_FUSED_META", row, DRAFT_META_INSTALLED,
                      "the QSA side-cache builder rebuilds its draft decode metadata in place on the device, so "
                      "the drafter runs steps 1+ as one fused loop without a host metadata rebuild between "
                      "steps"), flush=True)
    return row


def draft_early_stop_check(tuning: dict[str, Any]) -> dict[str, Any]:
    said = str(os.environ.get(DRAFT_EARLY_STOP_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{DRAFT_EARLY_STOP_ENV} edits vLLM 0.30.0's speculator; the {RUNTIME_PROFILE} runtime "
                           f"ships another one.")
    if not DRAFT_BUDGET_VARLEN["on"]:
        raise RuntimeError(f"{DRAFT_EARLY_STOP_ENV} stops lanes by the probabilities the sampling drafter recorded "
                           f"and caps the verification rows through the draft budget; arm {DRAFT_BUDGET_ENV} with "
                           f"{DRAFT_SAMPLE_ENV}=1 first.")
    depth = int(tuning["mtp_speculative_tokens"])
    helper = _bundle_module(DRAFT_EARLY_STOP_HELPER)
    try:
        plan = helper.parse(said, depth)
    except ValueError as failure:
        raise RuntimeError(str(failure)) from None
    if plan is None or not plan.stages:
        raise RuntimeError(f"{DRAFT_EARLY_STOP_ENV}={said!r} names no stage; give 'base,depth:fraction,...' or one "
                           f"of {sorted(helper.PRESETS)}.")
    return {"armed": True, "base": plan.base, "stages": [list(stage) for stage in plan.stages], "depth": plan.depth}


def patch_draft_early_stop() -> dict[str, Any]:
    if str(os.environ.get(DRAFT_EARLY_STOP_ENV) or "").strip() in ("", "0"):
        return {"armed": False}
    row = _patch_with_table(DRAFT_EARLY_STOP_ENV, DRAFT_EARLY_STOP_HELPER, DRAFT_EARLY_STOP_TABLE,
                            DRAFT_EARLY_STOP_INSTALLED)
    print(_armed_said("DRAFT_EARLY_STOP", row, DRAFT_EARLY_STOP_INSTALLED,
                      "the drafter runs its first depths on every lane, then each later depth range on a compacted "
                      "batch of the highest-value lanes at a captured graph size, so lanes the calibrated value "
                      "table ranks low stop drafting; the fused multi-step draft decode is replaced by per-step "
                      "decode; a DRAFT_EARLY_STOP window line follows the first speculative step"), flush=True)
    return row


def patch_gdn_norm() -> dict[str, Any]:
    if not _knob_value(GDN_NORM_ENV):
        return {"armed": False}
    row = _patch_with_table(GDN_NORM_ENV, GDN_NORM_HELPER, GDN_NORM_TABLE, GDN_NORM_INSTALLED)
    print(_armed_said("GDN_NORM_STRIDED", row, GDN_NORM_INSTALLED,
                      "the gated RMS norm after a GDN layer's prefill reads the output gate in place, one program "
                      "per token rows and head, after its first call of each launch shape matched the stock norm "
                      "bit for bit"), flush=True)
    return row


def patch_conv_syncfree() -> dict[str, Any]:
    if not _knob_value(CONV_SYNCFREE_ENV):
        return {"armed": False}
    if _flag_on(ADAPTIVE_VERIFY_ENV) and not DRAFT_BUDGET_VARLEN["on"]:
        raise RuntimeError(f"{CONV_SYNCFREE_ENV} sizes token groups from the host query lengths, which "
                           f"{ADAPTIVE_VERIFY_ENV} rewrites on the device only.")
    row = _patch_with_table(CONV_SYNCFREE_ENV, CONV_SYNCFREE_HELPER, CONV_SYNCFREE_TABLE, CONV_SYNCFREE_INSTALLED)
    print(_armed_said("CONV_SYNCFREE", row, CONV_SYNCFREE_INSTALLED,
                      "the PLE short-conv metadata of a mixed speculative step passes repeat_interleave the token "
                      "count the host already holds, so building it no longer waits for the GPU"), flush=True)
    return row


def patch_gdn_tril() -> dict[str, Any]:
    value = os.environ.get(GDN_TRIL_ENV, "").strip().lower()
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_TRIL_ENV}={value!r} is not one of '', '0', '1'.")
    if value == "0":
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_TRIL_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / GDN_TRIL_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_TRIL_ENV} is on but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_TRIL_ENV, namespace["TARGETS"], _runtime_site())
    print(f"GDN_TRIL_WARPS1 armed: the GDN prefill's 64x64 triangular merge autotunes over 1-8 warps and 1-5 "
          f"stages ({'edited' if made else 'already edited'})", flush=True)
    return {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}


def patch_gdn_tril_ilp() -> dict[str, Any]:
    value = os.environ.get(GDN_TRIL_ILP_ENV, "").strip().lower()
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_TRIL_ILP_ENV}={value!r} is not one of '', '0', '1'.")
    if value == "0":
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_TRIL_ILP_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / GDN_TRIL_ILP_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_TRIL_ILP_ENV} is on but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_TRIL_ILP_ENV, namespace["TARGETS"], _runtime_site())
    print(f"GDN_TRIL_ILP armed: the GDN prefill's 64x64 triangular merge runs its four 16x16 forward substitutions "
          f"in one loop and stores the upper blocks as zeros itself, so the output is allocated without a fill "
          f"({'edited' if made else 'already edited'}); every element equals the stock merge", flush=True)
    return {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}


def patch_gdn_ba_views() -> dict[str, Any]:
    value = os.environ.get(GDN_BA_VIEWS_ENV, "").strip().lower()
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_BA_VIEWS_ENV}={value!r} is not one of '', '0', '1'.")
    if value == "0":
        return {"armed": False}
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1":
        if value == "1":
            raise RuntimeError(f"{GDN_BA_VIEWS_ENV}=1 hands the GDN gates to the mixed-step forward as views, so it "
                               f"needs {MIXED_FULL_ENV}=1.")
        return {"armed": False, "why": f"{MIXED_FULL_ENV} is off"}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_BA_VIEWS_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / GDN_BA_VIEWS_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_BA_VIEWS_ENV} is on but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_BA_VIEWS_ENV, namespace["TARGETS"], _runtime_site())
    print(f"GDN_BA_VIEWS armed: a GDN layer hands its b and a gates to the attention op as views of the ba "
          f"projection; the mixed-step forward reads them by row stride and every other path copies them "
          f"contiguous as before ({'edited' if made else 'already edited'})", flush=True)
    return {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}


def patch_gdn_tail_zero() -> dict[str, Any]:
    value = os.environ.get(GDN_TAIL_ZERO_ENV, "").strip().lower()
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{GDN_TAIL_ZERO_ENV}={value!r} is not one of '', '0', '1'.")
    if value == "0":
        return {"armed": False}
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1":
        if value == "1":
            raise RuntimeError(f"{GDN_TAIL_ZERO_ENV}=1 zeroes the padding rows inside the mixed-step forward, so it "
                               f"needs {MIXED_FULL_ENV}=1.")
        return {"armed": False, "why": f"{MIXED_FULL_ENV} is off"}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_TAIL_ZERO_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / GDN_TAIL_ZERO_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_TAIL_ZERO_ENV} is on but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_TAIL_ZERO_ENV, namespace["TARGETS"], _runtime_site())
    print(f"GDN_TAIL_ZERO armed: a GDN layer whose step the mixed forward owns allocates its attention output empty, "
          f"and that forward zeroes only the rows past the step's real tokens; every other step allocates zeros as "
          f"before ({'edited' if made else 'already edited'})", flush=True)
    return {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}


def patch_gdn_decode_tail() -> dict[str, Any]:
    if not _knob_value(GDN_DECODE_TAIL_ENV):
        return {"armed": False}
    fused = os.environ.get(GDN_FUSED_ENV, "").strip().lower()
    if GDN_FUSED_MODES.get(fused) != "tiled":
        raise RuntimeError(f"{GDN_DECODE_TAIL_ENV}=1 zeroes the decode tail inside the tiled verify, so it needs "
                           f"{GDN_FUSED_ENV}=tiled; found {fused!r}.")
    if os.environ.get(GDN_VERIFY_AHEAD_ENV, "").strip() != "1":
        raise RuntimeError(f"{GDN_DECODE_TAIL_ENV}=1 extends the verify that {GDN_VERIFY_AHEAD_ENV}=1 selects; "
                           f"set it.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{GDN_DECODE_TAIL_ENV} edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships "
                           f"other ones.")
    table_path = BUNDLE_DIR / GDN_DECODE_TAIL_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{GDN_DECODE_TAIL_ENV} is on but {table_path} is absent")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    files, made = _apply_pinned_targets(GDN_DECODE_TAIL_ENV, namespace["TARGETS"], _runtime_site())
    print(f"GDN_DECODE_TAIL_ZERO armed: a decode step whose GDN layers run the tiled verify allocates their "
          f"attention output empty, and the verify writes zeros to the rows past the last request's tokens; "
          f"every other step allocates zeros as before ({'edited' if made else 'already edited'})", flush=True)
    return {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}


def patch_marks() -> dict[str, Any]:
    if not _knob_value(MARKS_ENV):
        return {"armed": False}
    sha = _install_helper(MARKS_ENV, MARKS_HELPER, MARKS_INSTALLED)
    print(f"MARKS armed: {MARKS_INSTALLED} brackets the step-metadata candidates with named marker kernels for the "
          f"profiler (each mark is one launch; profiling runs only)", flush=True)
    return {"armed": True, "helper_sha256": sha}


def vision_cache_mib() -> int:
    said = str(os.environ.get(VISION_CACHE_ENV) or "").strip()
    if said in ("", "0"):
        return 0
    try:
        mib = int(said)
    except ValueError:
        raise RuntimeError(f"{VISION_CACHE_ENV}={said!r} is not a whole number of MiB.") from None
    if not 0 < mib <= VISION_CACHE_MAX_MIB:
        raise RuntimeError(f"{VISION_CACHE_ENV}={mib} is outside 1..{VISION_CACHE_MAX_MIB} MiB.")
    return mib


def patch_vision_cache() -> dict[str, Any]:
    mib = vision_cache_mib()
    ahead = _knob_value(VISION_AHEAD_ENV)
    if not mib and not ahead:
        return {"armed": False}
    row = _patch_with_table(VISION_CACHE_ENV, VISION_CACHE_HELPER, VISION_CACHE_TABLE, VISION_CACHE_INSTALLED)
    row["mib"] = mib
    print(_armed_said("VISION_CACHE", row, VISION_CACHE_INSTALLED,
                      f"an image embedding the scheduler frees stays on the GPU by mm hash, up to {mib} MiB outside "
                      "the KV pool, and an image scheduled again takes it instead of running the encoder"
                      if mib else f"no MiB is set, so a freed embedding is dropped as in stock; {VISION_AHEAD_ENV} "
                      "routes the worker's frees through it to settle an encode still running on its side stream"),
          flush=True)
    return row


def patch_vision_ahead(tuning: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _knob_value(VISION_AHEAD_ENV):
        return {"armed": False}
    if tuning is not None and not tuning.get("enable_chunked_prefill", True):
        raise RuntimeError(f"{VISION_AHEAD_ENV} stops a prefill chunk just before an image, which needs chunked "
                           "prefill on.")
    row = _patch_with_table(VISION_AHEAD_ENV, VISION_AHEAD_HELPER, VISION_AHEAD_TABLE, VISION_AHEAD_INSTALLED)
    print(_armed_said("VISION_AHEAD", row, VISION_AHEAD_INSTALLED,
                      "a prefill chunk that reaches an image neither encoder cache holds stops just before it while "
                      "the image is encoded on a least-priority side stream beside the step's forward; the next "
                      "step prefills it from the cache and waits on the encode's event, not on the host"),
          flush=True)
    return row


def patch_prefill_hold() -> dict[str, Any]:
    if not _knob_value(PREFILL_HOLD_ENV):
        if _knob_value(PREFILL_HOLD_AUTO_ENV):
            raise RuntimeError(f"{PREFILL_HOLD_AUTO_ENV}=1 decides the holds of {PREFILL_HOLD_ENV}, which is off.")
        return {"armed": False}
    if _knob_value(PREFILL_HOLD_AUTO_ENV):
        return patch_prefill_hold_auto()
    tokens_said = os.environ.get(PREFILL_HOLD_TOKENS_ENV, "").strip() or "1024"
    ms_said = os.environ.get(PREFILL_HOLD_MS_ENV, "").strip() or "300"
    try:
        tokens, ms = int(tokens_said), float(ms_said)
    except ValueError:
        raise RuntimeError(f"{PREFILL_HOLD_TOKENS_ENV} must be an integer and {PREFILL_HOLD_MS_ENV} a number, "
                           f"found {tokens_said!r} and {ms_said!r}.") from None
    if not 1 <= tokens <= 8192 or not 0 < ms <= 2000:
        raise RuntimeError(f"{PREFILL_HOLD_TOKENS_ENV} must be 1-8192 and {PREFILL_HOLD_MS_ENV} above 0 and at "
                           f"most 2000, found {tokens} and {ms}.")
    row = _patch_with_table(PREFILL_HOLD_ENV, PREFILL_HOLD_HELPER, PREFILL_HOLD_TABLE, PREFILL_HOLD_INSTALLED)
    print(_armed_said("PREFILL_HOLD", row, PREFILL_HOLD_INSTALLED,
                      f"a step whose running requests decode while fewer than {tokens} prefill tokens wait, none "
                      f"of them older than {ms:.0f} ms, schedules no waiting request, so it runs decode only and "
                      "the waiting prefill goes later in one larger mixed step"),
          flush=True)
    return dict(row, tokens=tokens, ms=ms)


def patch_humming_int3() -> dict[str, Any]:
    said = os.environ.get(EXPERT_BITS_ENV, "").strip()
    if said in ("", "4"):
        return {"armed": False}
    if said != "3":
        raise RuntimeError(f"{EXPERT_BITS_ENV} must be 3 or 4 when set, found {said!r}")
    backend = os.environ.get(MOE_BACKEND_ENV, "").strip()
    if backend != "humming":
        raise RuntimeError(
            f"{EXPERT_BITS_ENV}=3 serves the experts through humming's 3-bit MoE kernel and needs "
            f"{MOE_BACKEND_ENV}=humming, found {backend!r}; Marlin has no 3-bit kernel.")
    row = _patch_with_table(EXPERT_BITS_ENV, HUMMING_INT3_HELPER, HUMMING_INT3_TABLE, HUMMING_INT3_INSTALLED)
    print(_armed_said("EXPERT_BITS=3", row, HUMMING_INT3_INSTALLED,
                      "each main MoE layer's int4 experts are re-quantized at load to 3 bits (codes -4..3, "
                      "scale per group of 128 by least squared error) and handed to humming as uint3; the "
                      "draft layer keeps int4; humming's own unpack checks the first layer's packing"),
          flush=True)
    return row


def patch_moe_add_fused() -> dict[str, Any]:
    if not _knob_value(MOE_ADD_FUSED_ENV):
        return {"armed": False}
    clash = [name for name in (MOE_SHARE_ENV, MOE_ROUTE_ENV) if _flag_on(name)]
    try:
        if clash:
            raise RuntimeError(f"{' and '.join(clash)} edit the same MoE runner file without composing")
        if RUNTIME_PROFILE != "v030":
            raise RuntimeError(f"it edits vLLM 0.30.0's files; the {RUNTIME_PROFILE} runtime ships other ones")
        table_path = BUNDLE_DIR / MOE_ADD_FUSED_TABLE
        if not table_path.is_file():
            raise RuntimeError(f"{table_path} is absent")
        table = table_path.read_bytes()
        namespace: dict[str, Any] = {}
        exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
        helper_sha256 = _install_helper(MOE_ADD_FUSED_ENV, MOE_ADD_FUSED_HELPER, MOE_ADD_FUSED_INSTALLED)
        files, made = _apply_pinned_targets(MOE_ADD_FUSED_ENV, namespace["TARGETS"], _runtime_site())
    except (RuntimeError, OSError, SyntaxError) as failure:
        print(f"MOE_ADD_FUSED stood aside at setup: {failure}; the MoE layers keep the stock sigmoid, "
              f"multiply and routed+shared add", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    row = {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": helper_sha256, "files": files}
    print(_armed_said("MOE_ADD_FUSED", row, MOE_ADD_FUSED_INSTALLED,
                      "when the shared expert runs inline, its sigmoid gate, the scaling and the routed+shared add "
                      "run as one kernel, checked bit for bit against the stock three over every bf16 gate value "
                      "and the first live call before any graph capture, and stock if they differ"), flush=True)
    return row


def patch_router_chain() -> dict[str, Any]:
    value = os.environ.get(ROUTER_CHAIN_ENV, "").strip().lower()
    if value not in ROUTER_CHAIN_VALUES:
        raise RuntimeError(f"{ROUTER_CHAIN_ENV}={value!r} is not one of {list(ROUTER_CHAIN_VALUES)}.")
    if value in ("", "0"):
        return {"armed": False}
    try:
        row = _patch_with_table(ROUTER_CHAIN_ENV, ROUTER_CHAIN_HELPER, ROUTER_CHAIN_TABLE, ROUTER_CHAIN_INSTALLED)
    except (RuntimeError, OSError, SyntaxError) as failure:
        print(f"ROUTER_CHAIN stood aside at setup: {failure}; the MoE layers keep the separate router, shared "
              f"gate_up and shared gate GEMMs", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    print(_armed_said("ROUTER_CHAIN", row, ROUTER_CHAIN_INSTALLED,
                      f"level {'gate_up' if value == '1' else value}: at the top of the MoE runner's _forward_impl, "
                      f"before the shared expert launches, each layer's router"
                      f"{', shared gate_up' if value in ('1', 'gate_up') else ''} and shared-gate projections run as "
                      "one GEMM over weights stacked in place at the first eager call, checked against the separate "
                      "GEMMs before any graph capture, and stock if the check fails"), flush=True)
    return row


def moe_taaf_window() -> tuple[int, int] | None:
    said = str(os.environ.get(MOE_TAAF_ENV) or "").strip().lower()
    if said in ("", "0", "off", "false", "no"):
        return None
    found = MOE_TAAF_WINDOW.fullmatch(said)
    first = int(found.group(1)) if found else 0
    last = int(found.group(2)) if found and found.group(2) else 0
    if first < 1 or (found.group(2) and last < first):
        raise RuntimeError(f"{MOE_TAAF_ENV} must be <min rows> or <min rows>:<max rows> with 1 <= min <= max, "
                           f"found {said!r}")
    return first, last


def patch_moe_taaf() -> dict[str, Any]:
    try:
        window = moe_taaf_window()
    except RuntimeError as failure:
        print(f"{MOE_TAAF_ENV} stood aside at setup: {failure}; every MoE forward keeps vLLM's fused Marlin "
              f"MoE", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    if window is None:
        return {"armed": False}
    try:
        _install_helper(MOE_TAAF_ENV, MOE_TAAF_HELPER, MOE_TAAF_INSTALLED)
        row = _patch_with_table(MOE_TAAF_ENV, MOE_TAAF_HELPER, MOE_TAAF_TABLE, MOE_TAAF_INSTALLED)
    except (RuntimeError, OSError, SyntaxError) as failure:
        print(f"{MOE_TAAF_ENV} stood aside at setup: {failure}; every MoE forward keeps vLLM's fused Marlin "
              f"MoE", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    row["window"] = list(window)
    rows = f"{window[0]} to {window[1]}" if window[1] else f"{window[0]} or more"
    print(_armed_said("MOE_TAAF", row, MOE_TAAF_INSTALLED,
                      f"MoE forwards of {rows} rows run the routed experts on the moe_taaf Triton kernel, which "
                      f"reads Marlin's repacked int4 weights; routing, the shared expert and every other forward "
                      f"stay vLLM's fused Marlin MoE; each layer's first eager forward in the window is checked "
                      f"against Marlin and an fp32 reference and the knob stands aside on any failure"), flush=True)
    return row


def moe_block_bands() -> tuple[tuple[int, int, int], ...]:
    said = str(os.environ.get(MOE_BLOCK_ENV) or "").strip().lower()
    if said in ("", "0", "off", "false", "no"):
        return ()
    if said in ("1", "on", "true", "yes"):
        return MOE_BLOCK_DEFAULT
    bands = []
    for part in said.split(","):
        found = MOE_BLOCK_BAND.fullmatch(part.strip())
        if not found:
            raise RuntimeError(f"{MOE_BLOCK_ENV} must be 1 or <first rows>-<last rows>:<block>[,...], "
                               f"found {said!r}")
        first, last, block = (int(found.group(index)) for index in (1, 2, 3))
        if first < 1 or last < first or block not in MOE_BLOCK_SIZES:
            raise RuntimeError(f"{MOE_BLOCK_ENV} band {part!r} needs 1 <= first <= last and a block in "
                               f"{list(MOE_BLOCK_SIZES)}")
        bands.append((first, last, block))
    ordered = sorted(bands)
    for (_, last, _), (first, _, _) in zip(ordered, ordered[1:]):
        if first <= last:
            raise RuntimeError(f"{MOE_BLOCK_ENV} bands overlap: {said!r}")
    return tuple(ordered)


def moe_sum_epi() -> bool:
    said = str(os.environ.get(MOE_SUM_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MOE_SUM_ENV} must be 1, 0 or unset, found {said!r}")
    return said == "1"


def patch_moe_silu() -> dict[str, Any]:
    said = str(os.environ.get(MOE_SILU_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MOE_SILU_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return {"armed": False}
    if not moe_block_bands():
        raise RuntimeError(f"{MOE_SILU_ENV} rides on the Marlin MoE edits of {MOE_BLOCK_ENV} and needs it armed")
    helper = _install_helper(MOE_SILU_ENV, MOE_SUM_HELPER, MOE_SUM_INSTALLED)
    print(f"MOE_SILU_EPI armed: helper at {MOE_SUM_INSTALLED} {helper[:16]}; the Marlin MoE w1 writes silu(gate)*up "
          f"from its epilogue into the half-width activation buffer, its gate and up columns interleaved in "
          f"64-column chunks in place at the first eager forward of each layer, built with tiles at least 128 "
          f"wide; the first 3 eager forwards are checked against the stock w1 and silu_and_mul and a failure puts "
          f"the columns back ({MOE_SILU_ENV})", flush=True)
    return {"armed": True, "helper_sha256": helper}


def patch_moe_quant() -> dict[str, Any]:
    said = str(os.environ.get(MOE_QUANT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MOE_QUANT_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return {"armed": False}
    if not moe_block_bands():
        raise RuntimeError(f"{MOE_QUANT_ENV} rides on the Marlin MoE edits of {MOE_BLOCK_ENV} and needs it armed")
    print(f"MOE_QUANT_FUSED armed: int8-input Marlin MoE quantizes each token and multiplies its scale by the "
          f"input global scale in one Triton kernel, in place of the stock per-token quant and a separate multiply, "
          f"before w1 and before w2; the first 3 eager calls run both and stand aside unless codes and scales are "
          f"bit-exact ({MOE_QUANT_ENV})", flush=True)
    return {"armed": True}


def patch_moe_act_quant() -> dict[str, Any]:
    said = str(os.environ.get(MOE_ACT_QUANT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{MOE_ACT_QUANT_ENV} must be 1, 0 or unset, found {said!r}")
    if said != "1":
        return {"armed": False}
    if not moe_block_bands():
        raise RuntimeError(f"{MOE_ACT_QUANT_ENV} rides on the Marlin MoE edits of {MOE_BLOCK_ENV} and needs it armed")
    print(f"MOE_ACT_QUANT armed: int8-input Marlin MoE computes the w2 input as SiLU(gate) x up of w1's output, "
          f"quantized to int8 with its per-token scale times the input global scale, in one Triton kernel in place "
          f"of silu_and_mul and the separate quant; the first 3 eager calls run both and stand aside unless codes "
          f"and scales are bit-exact ({MOE_ACT_QUANT_ENV})", flush=True)
    return {"armed": True}


def patch_moe_sum() -> dict[str, Any]:
    if not moe_sum_epi():
        return {"armed": False}
    if not moe_block_bands():
        raise RuntimeError(f"{MOE_SUM_ENV} rides on the Marlin MoE edits of {MOE_BLOCK_ENV} and needs it armed")
    helper = _install_helper(MOE_SUM_ENV, MOE_SUM_HELPER, MOE_SUM_INSTALLED)
    print(f"MOE_SUM_EPI armed: helper at {MOE_SUM_INSTALLED} {helper[:16]}; the Marlin MoE w2 adds each token's top-k "
          f"weighted rows in its epilogue with fp32 vector atomics into an fp32 view of its output and the top-k sum "
          f"becomes one cast; the Marlin build compiles at the first eager forward, the first {3} eager forwards run "
          f"the stock w2 and sum beside it and stand aside on a relative gap over 1e-2 ({MOE_SUM_ENV})", flush=True)
    return {"armed": True, "helper_sha256": helper}


def patch_moe_block() -> dict[str, Any]:
    try:
        bands = moe_block_bands()
    except RuntimeError as failure:
        print(f"{MOE_BLOCK_ENV} stood aside at setup: {failure}; Marlin MoE keeps vLLM's block_size_m", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    if not bands:
        return {"armed": False}
    try:
        row = _patch_with_table(MOE_BLOCK_ENV, MOE_BLOCK_HELPER, MOE_BLOCK_TABLE, MOE_BLOCK_INSTALLED)
    except (RuntimeError, OSError, SyntaxError) as failure:
        print(f"{MOE_BLOCK_ENV} stood aside at setup: {failure}; Marlin MoE keeps vLLM's block_size_m", flush=True)
        return {"armed": False, "aside": str(failure)[:300]}
    row["bands"] = [list(one) for one in bands]
    row["sum_helper_sha256"] = _install_helper(MOE_BLOCK_ENV, MOE_SUM_HELPER, MOE_SUM_INSTALLED)
    said = ", ".join(f"{first}-{last} rows at block {block}" for first, last, block in bands)
    print(_armed_said("MOE_BLOCK_M", row, MOE_BLOCK_INSTALLED,
                      f"Marlin MoE forwards of {said} replace vLLM's block_size_m choice; each expert shape's "
                      f"first eager forward in a band runs both blocks and keeps the override only if the "
                      f"outputs agree, and a band first met inside a graph capture keeps vLLM's block"), flush=True)
    return row


def patch_mixed_full() -> dict[str, Any]:
    value = os.environ.get(MIXED_FULL_ENV, "").strip().lower()
    if value in ("", "0", "off"):
        for sizing in (MIXED_FULL_SEQS_ENV, MIXED_FULL_LARGEST_ENV):
            if str(os.environ.get(sizing) or "").strip():
                raise RuntimeError(f"{sizing} sizes the mixed buckets and needs {MIXED_FULL_ENV} armed.")
        return {"armed": False}
    if value not in ("plan", "1"):
        raise RuntimeError(f"{MIXED_FULL_ENV}={value!r} is not one of '', '0', 'plan', '1'.")
    _bounded_integer_environment(MIXED_FULL_SEQS_ENV, default=16, minimum=1, maximum=64)
    _bounded_integer_environment(MIXED_FULL_LARGEST_ENV, default=2048, minimum=64, maximum=16384)
    row = _patch_with_table(MIXED_FULL_ENV, MIXED_FULL_HELPER, MIXED_FULL_TABLE, MIXED_FULL_INSTALLED)
    row["mode"] = value
    row["spec_helper_sha256"] = _install_helper(MIXED_FULL_ENV, MIXED_FULL_SPEC_HELPER, MIXED_FULL_SPEC_INSTALLED)
    if value == "1":
        runner = _patch_with_table(MIXED_FULL_ENV, MIXED_FULL_GRAPH_HELPER, MIXED_FULL_GRAPH_TABLE,
                                   MIXED_FULL_GRAPH_INSTALLED)
        print(_armed_said("MIXED_FULL_GRAPH", runner, MIXED_FULL_GRAPH_INSTALLED,
                          "the graph manager adds a full-graph descriptor for every mixed bucket, captures it on a "
                          "mixed dummy batch, and a step the planner or any backend refuses is dispatched to its "
                          "piecewise graph again"), flush=True)
        row["runner"] = runner
        graph = _patch_with_table(MIXED_FULL_ENV, QSA_GRAPH_HELPER, QSA_GRAPH_TABLE, QSA_GRAPH_INSTALLED)
        print(_armed_said("QSA_GRAPH", graph, QSA_GRAPH_INSTALLED,
                          "the QSA indexer and attention of a mixed bucket read the decode/prefill split and the "
                          "padding from device tensors and declare mixed-batch full-graph support"), flush=True)
        row["qsa_graph"] = graph
    print(_armed_said("MIXED_FULL", row, MIXED_FULL_INSTALLED,
                      "each mixed step is fitted to a padded token bucket and counted, and the decisions are "
                      "printed as tallies; " + ("the metadata of three steps is also built on the GPU and "
                                                "compared with the host reference, and the full-graph flip stands "
                                                "aside until the attention backends are converted"
                                                if value == "1" else "nothing else changes")), flush=True)
    return row


def check_prep_nosync() -> bool:
    said = os.environ.get(PREP_NOSYNC_ENV, "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{PREP_NOSYNC_ENV}={said!r} is not one of '', '0', '1'.")
    if said != "1":
        return False
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1" and os.environ.get(GDN_FUSED_ENV, "").strip().lower() in ("", "0"):
        raise RuntimeError(f"{PREP_NOSYNC_ENV} rewrites host-side prep of the GDN builders and needs "
                           f"{MIXED_FULL_ENV}=1 or {GDN_FUSED_ENV} armed.")
    print(f"PREP_NOSYNC armed: the GDN builders' per-step prep writes its comparisons, index walks and slot gathers "
          f"into persistent buffers (no temporary tensor per op) and the slot fill is one gather into the slot "
          f"buffer; the metadata is the same bits", flush=True)
    return True


def gdn_mixed_fi() -> dict[str, Any]:
    wanted = _flag_on(GDN_MIXED_FI_ENV)
    fp16 = _flag_on(GDN_MIXED_FI_FP16_ENV)
    if not wanted:
        if fp16:
            print(f"{GDN_MIXED_FI_FP16_ENV} stood aside at setup: it needs {GDN_MIXED_FI_ENV}=1", flush=True)
        return {"armed": False}
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1":
        print(f"{GDN_MIXED_FI_ENV} stood aside at setup: it needs {MIXED_FULL_ENV}=1, the mixed graph it changes",
              flush=True)
        return {"armed": False, "aside": "needs MIXED_FULL"}
    if fp16:
        os.environ[GDN_MIXED_FI_FP16_FLASHINFER] = "1"
        print(f"SERVER_ENV {GDN_MIXED_FI_FP16_FLASHINFER}=1 (from {GDN_MIXED_FI_FP16_ENV})", flush=True)
    print(f"GDN_MIXED_FI armed: a mixed step's GDN prefill calls the serve's selected FlashInfer prefill kernel "
          f"in place of FLA once its first eager warmup has matched FLA eagerly and after a graph replay on a "
          f"new layout; FLA otherwise" + (" (FlashInfer fp16 accumulate on)" if fp16 else ""), flush=True)
    return {"armed": True, "fp16": fp16}


def patch_gdn_mixed_full() -> dict[str, Any]:
    check_prep_nosync()
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1":
        gdn_mixed_fi()
        return {"armed": False}
    row = _patch_with_table(MIXED_FULL_ENV, GDN_MIXED_FULL_HELPER, GDN_MIXED_FULL_TABLE, GDN_MIXED_FULL_INSTALLED)
    row["fi"] = gdn_mixed_fi()
    print(_armed_said("GDN_MIXED_FULL", row, GDN_MIXED_FULL_INSTALLED,
                      "the GDN builders build chunk, causal-conv and split metadata on the device into persistent "
                      "buffers for a mixed step that fits a bucket, and declare mixed-batch full-graph support"),
          flush=True)
    return row


def patch_pdl() -> dict[str, Any]:
    said = str(os.environ.get(PDL_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return {"armed": False}
    if said not in ("1", "true", "yes", "on", "interpret"):
        raise RuntimeError(f"{PDL_ENV}={said!r} is not a boolean.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{PDL_ENV} edits vLLM 0.30.0's kernels; the {RUNTIME_PROFILE} runtime ships other ones.")
    table_path = BUNDLE_DIR / PDL_TABLE
    if not table_path.is_file():
        raise RuntimeError(f"{PDL_ENV} was set but {table_path} is absent")
    if str(os.environ.get(ADAPTIVE_VERIFY_ENV) or "").strip() == "1" and not DRAFT_BUDGET_VARLEN["on"]:
        raise RuntimeError(f"{PDL_ENV} and {ADAPTIVE_VERIFY_ENV} both edit the signature of the QSA indexer's "
                           f"uniform decode kernel and do not compose; arm one.")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets = dict(namespace["TARGETS"])
    if DRAFT_BUDGET_VARLEN["on"] and targets.pop(PDL_INDEXER_RELATIVE, None) is not None:
        print(f"PDL stands aside on {PDL_INDEXER_RELATIVE} under {DRAFT_BUDGET_ENV}: the varlen launch of the QSA "
              f"indexer's decode kernel edits the same signature lines; every other PDL launch stays armed",
              flush=True)
    files, made = _apply_pinned_targets(PDL_ENV, targets, _runtime_site())
    row = {"armed": True, "already": not made, "table_sha256": hashlib.sha256(table).hexdigest(), "files": files}
    print(f"PDL {'already armed' if row['already'] else 'armed'}: table {row['table_sha256'][:16]}; "
          + "; ".join(f"{relative} {held['stock_sha256'][:16]} -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + "; the stock kernels the owned launches overlap wait for their predecessor and release their successor "
            "first thing; the flags reach a launch only on a device of compute capability 9 or newer", flush=True)
    return row


def patch_token_pin() -> dict[str, Any]:
    said = str(os.environ.get(TOKEN_PIN_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if not said.isdigit() or not 4 <= int(said) <= 2048:
        raise ValueError(f"{TOKEN_PIN_ENV} is the most tokens pinned a request, 4..2048, not {said!r}")
    row = _patch_with_table(TOKEN_PIN_ENV, TOKEN_PIN_HELPER, TOKEN_PIN_TABLE, TOKEN_PIN_INSTALLED)
    print(_armed_said("TOKEN_PIN", row, TOKEN_PIN_INSTALLED,
                      f"tokens between the two marker tokens of a request (up to {said} a request, its latest marked "
                      f"spans first) are always among the compressed blocks the QSA indexer selects, in every "
                      f"attention layer and step; a request with no markers selects exactly as before"),
          flush=True)
    return row


def patch_l2_pin() -> dict[str, Any]:
    said = str(os.environ.get(L2_PIN_ENV) or "").strip()
    if said in ("", "0"):
        return {"armed": False}
    if not said.isdigit() or not 1 <= int(said) <= 4096:
        raise ValueError(f"{L2_PIN_ENV} is the persisting L2 budget in MiB, 1..4096, not {said!r}")
    row = _patch_with_table(L2_PIN_ENV, L2_PIN_HELPER, L2_PIN_TABLE, L2_PIN_INSTALLED)
    print(_armed_said("L2_PIN", row, L2_PIN_INSTALLED,
                      f"before CUDA graph capture the MoE router weights of every layer and the draft head's "
                      f"router and dense weights are packed into one device buffer of at most {said} MiB (capped by "
                      f"the device's persisting-L2 and access-window limits), a persisting L2 carve-out is set and "
                      f"an access-policy window over the buffer is set on the capture and current streams; the "
                      f"granted bytes and hit ratio are printed as an L2_PIN line"),
          flush=True)
    return row


def patch_mm_processor_only() -> dict[str, Any]:
    if not _knob_value(MM_PROCESSOR_ONLY_ENV):
        return {"armed": False}
    row = _patch_with_table(MM_PROCESSOR_ONLY_ENV, MM_PROCESSOR_ONLY_HELPER, MM_PROCESSOR_ONLY_TABLE,
                            MM_PROCESSOR_ONLY_INSTALLED)
    print(_armed_said("MM_PROCESSOR_ONLY", row, MM_PROCESSOR_ONLY_INSTALLED,
                      "the multi-modal processor cache runs processor-only: the API process keeps each processed "
                      "item with its data and sends it with every request, the engine keeps no receiver cache, so "
                      "a request whose items the engine evicted no longer fails as a P0/P1 cache miss"),
          flush=True)
    return row


def patch_no_blank() -> dict[str, Any]:
    if not _knob_value(NO_BLANK_ENV):
        return {"armed": False}
    row = _patch_with_table(NO_BLANK_ENV, NO_BLANK_HELPER, NO_BLANK_TABLE, NO_BLANK_INSTALLED)
    print(_armed_said("NO_BLANK_LINES", row, NO_BLANK_INSTALLED,
                      "the target logits of every sampled and verified position are masked so the reasoning and "
                      "content bodies hold no blank or whitespace-only line, no line break but \\n, no leading "
                      "whitespace, </think> only at the start of a line and the content's blank line only before "
                      "<tool_call>; tool-call bodies and the drafter are not masked; the removed mass, the drafts "
                      "rejected at a banned token and the positions by state print every 15 s"), flush=True)
    return row


def sample_topk_count() -> int:
    raw = os.environ.get(SAMPLE_TOPK_ENV, "").strip()
    if raw in ("", "0"):
        return 0
    if not raw.isdigit() or not 2 <= int(raw) <= 256 or int(raw) & (int(raw) - 1):
        raise RuntimeError(f"{SAMPLE_TOPK_ENV}={raw!r} is not a power of two in 2..256.")
    return int(raw)


def sample_topk_draft_on() -> bool:
    said = str(os.environ.get(SAMPLE_TOPK_DRAFT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{SAMPLE_TOPK_DRAFT_ENV}={said!r} is not one of '', '0', '1'.")
    if said != "1":
        return False
    if not sample_topk_count():
        raise RuntimeError(f"{SAMPLE_TOPK_DRAFT_ENV}=1 verifies sampled drafts over the {SAMPLE_TOPK_ENV} "
                           f"candidates; set {SAMPLE_TOPK_ENV}.")
    if not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{SAMPLE_TOPK_DRAFT_ENV}=1 reads the drafter's top ids that {DRAFT_SAMPLE_ENV}=1 "
                           f"records; set it.")
    return True


def tail_kept_logits_on() -> bool:
    if not _knob_value(TAIL_KEPT_LOGITS_ENV):
        return False
    if not sample_topk_count():
        raise RuntimeError(f"{TAIL_KEPT_LOGITS_ENV}=1 hands the sampling tail's {SAMPLE_TOPK_ENV} top-k the kept "
                           f"logits; set {SAMPLE_TOPK_ENV}.")
    if not _flag_on(TARGET_VOCAB_ENV):
        raise RuntimeError(f"{TAIL_KEPT_LOGITS_ENV}=1 reads the kept head that {TARGET_VOCAB_ENV}=1 builds; "
                           f"set it.")
    if os.environ.get(MIXED_FULL_ENV, "").strip().lower() != "1":
        raise RuntimeError(f"{TAIL_KEPT_LOGITS_ENV}=1 changes the sampling tail that {MIXED_FULL_ENV}=1 "
                           f"captures; set it.")
    if os.environ.get(DRAFT_SUMMIN_ENV, "").strip() == "1":
        raise RuntimeError(f"{TAIL_KEPT_LOGITS_ENV}=1 hands the tail a kept-column pair that "
                           f"{DRAFT_SUMMIN_ENV}=1 cannot observe; unset one of them.")
    if _knob_value(NO_BLANK_ENV):
        print(f"TAIL_KEPT_LOGITS stands aside: {NO_BLANK_ENV}=1 masks the full-width logits, so the tail "
              f"keeps them", flush=True)
        return False
    return True


def patch_small_kernels() -> dict[str, Any]:
    marlin = _knob_value(SMALL_KERNELS_MARLIN_ENV)
    draft = _knob_value(SMALL_KERNELS_DRAFT_TOPK_ENV)
    invalidate = _knob_value(SMALL_KERNELS_INVALIDATE_ENV)
    glue = _knob_value(SMALL_KERNELS_GLUE_ENV)
    gemma = _knob_value(SMALL_KERNELS_GEMMA_ENV)
    if not marlin and not draft and not invalidate and not glue and not gemma:
        return {"armed": False}
    if glue and not (sample_topk_count() and sample_topk_draft_on()):
        raise RuntimeError(f"{SMALL_KERNELS_GLUE_ENV}=1 fuses the sampled-draft verification that "
                           f"{SAMPLE_TOPK_ENV} and {SAMPLE_TOPK_DRAFT_ENV}=1 run; set them.")
    if draft and not _flag_on(DRAFT_SAMPLE_ENV):
        raise RuntimeError(f"{SMALL_KERNELS_DRAFT_TOPK_ENV}=1 replaces the top-k of the sampling drafter that "
                           f"{DRAFT_SAMPLE_ENV}=1 runs; set it.")
    row: dict[str, Any] = {"armed": True, "marlin": marlin, "draft_topk": draft, "invalidate": invalidate,
                           "glue": glue, "gemma_norm": gemma}
    if invalidate or glue or gemma:
        row["triton_sha256"] = _install_helper("TAAF_VLLM_SMALL_KERNELS", SMALL_KERNELS_TRITON,
                                               SMALL_KERNELS_TRITON_INSTALLED)
    row["helper_sha256"] = _install_helper("TAAF_VLLM_SMALL_KERNELS", SMALL_KERNELS_HELPER, SMALL_KERNELS_INSTALLED)
    if marlin:
        held = _patch_with_table(SMALL_KERNELS_MARLIN_ENV, SMALL_KERNELS_HELPER, SMALL_KERNELS_MARLIN_TABLE,
                                 SMALL_KERNELS_INSTALLED)
        row["marlin_table_sha256"] = held["table_sha256"]
        row["files"] = held["files"]
        print(_armed_said("MARLIN_WORKSPACE_REUSE", held, SMALL_KERNELS_INSTALLED,
                          "the Marlin MoE path takes one persistent zeroed lock workspace per device instead of "
                          "allocating and zero-filling one in every layer's call"), flush=True)
    if invalidate:
        held = _patch_with_table(SMALL_KERNELS_INVALIDATE_ENV, SMALL_KERNELS_HELPER, SMALL_KERNELS_INVALIDATE_TABLE,
                                 SMALL_KERNELS_INSTALLED)
        row["invalidate_table_sha256"] = held["table_sha256"]
        print(_armed_said("HISPARSE_INVALIDATE_FUSED", held, SMALL_KERNELS_INSTALLED,
                          "each HiSparse leader runtime drops recycled hot slots with one in-place Triton kernel "
                          "instead of seven eager ops"), flush=True)
    if gemma:
        held = _patch_with_table(SMALL_KERNELS_GEMMA_ENV, SMALL_KERNELS_HELPER, SMALL_KERNELS_GEMMA_TABLE,
                                 SMALL_KERNELS_INSTALLED)
        row["gemma_table_sha256"] = held["table_sha256"]
        row["gemma_files"] = held["files"]
        print(_armed_said("GEMMA_NORM_FUSED", held, SMALL_KERNELS_INSTALLED,
                          "the MTP drafter's GemmaRMSNorm calls run as one Triton kernel each instead of eager "
                          "torch ops; compiled callers keep the stock path"), flush=True)
    if glue:
        print(f"SAMPLE_GLUE_FUSED armed: the sampled-draft verification builds its union layout, compact draft "
              f"rows and local draft ids in one Triton kernel ({SMALL_KERNELS_TRITON_INSTALLED} "
              f"{row['triton_sha256'][:16]})", flush=True)
    if draft:
        row["draft_topk_kernel_sha256"] = _install_helper(SMALL_KERNELS_DRAFT_TOPK_ENV,
                                                          SMALL_KERNELS_DRAFT_TOPK_KERNEL,
                                                          SMALL_KERNELS_DRAFT_TOPK_INSTALLED)
        print(f"DRAFT_TOPK_TRITON armed: the sampling drafter's top-k over bf16 logits runs as two Triton kernels "
              f"({SMALL_KERNELS_DRAFT_TOPK_INSTALLED} {row['draft_topk_kernel_sha256'][:16]}); any other dtype "
              f"or shape keeps torch.topk", flush=True)
    return row


def patch_sample_topk() -> dict[str, Any]:
    kept = tail_kept_logits_on()
    count = sample_topk_count()
    drafted = sample_topk_draft_on()
    if not count:
        return {"armed": False}
    row = _patch_with_table(SAMPLE_TOPK_ENV, SAMPLE_TOPK_HELPER, SAMPLE_TOPK_TABLE, SAMPLE_TOPK_INSTALLED)
    row["kernel_sha256"] = _install_helper(SAMPLE_TOPK_ENV, SAMPLE_TOPK_KERNEL, SAMPLE_TOPK_KERNEL_INSTALLED)
    row["candidates"] = count
    row["drafts"] = drafted
    row["tail_kept_logits"] = kept
    if kept:
        print(f"TAIL_KEPT_LOGITS armed: the captured sampling tail takes its top {count} from the target "
              f"vocabulary's kept logits of the language model under the multimodal wrapper and maps the "
              f"columns to token ids, so the -inf full-vocabulary buffer, its scatter and the dropped "
              f"columns' segment maxima leave the tail", flush=True)
    if drafted:
        print(f"{SAMPLE_TOPK_DRAFT_ENV} armed: a step with sampled drafts verifies each request whose top_k is at "
              f"most {count} (or whose temperature is 0) over one compact row, the target's top {count} plus the "
              f"drafter's own top ids, through the stock rejection kernels, block verification included; a request "
              f"with a wider or no top_k is verified the stock way in the same step", flush=True)
    print(_armed_said("SAMPLE_TOPK", row, SAMPLE_TOPK_INSTALLED,
                      f"verification samples over the top {count} logits of each row: a two-pass Triton top-k "
                      f"reads the bf16 logits once, temperature, top-k, top-p and the rejection sampler run on "
                      f"{count} candidates and one -inf column, and the sampled columns map back to token ids; "
                      f"a step whose requests ask for min_p, logprobs, penalties or a top_k above {count} at a "
                      f"nonzero temperature samples the stock way"), flush=True)
    return row


def patch_step_timers() -> dict[str, Any]:
    prep = os.environ.get(STEP_TIMERS_PREP_ENV, "").strip().lower()
    if prep not in ("", "0", "1", "all"):
        raise RuntimeError(f"{STEP_TIMERS_PREP_ENV}={prep!r} is not one of '', '0', '1', 'all'.")
    if prep in ("1", "all") and not _knob_value(STEP_TIMERS_ENV):
        raise RuntimeError(f"{STEP_TIMERS_PREP_ENV} adds host spans to the step timers and needs {STEP_TIMERS_ENV}=1.")
    if _knob_value(STEP_TIMERS_HC_ENV) and not _knob_value(STEP_TIMERS_ENV):
        raise RuntimeError(f"{STEP_TIMERS_HC_ENV} files the hyperconnection projections under the bf16c group of the "
                           f"step timers and needs {STEP_TIMERS_ENV}=1.")
    if _knob_value(VISION_TALLY_ENV) and not _knob_value(STEP_TIMERS_ENV):
        raise RuntimeError(f"{VISION_TALLY_ENV} is armed by the step timers after loading and needs {STEP_TIMERS_ENV}=1.")
    spec_prefetch = str(os.environ.get(SPEC_PREFETCH_ENV) or "").strip()
    if spec_prefetch not in ("", "0", "1"):
        raise RuntimeError(f"{SPEC_PREFETCH_ENV}={spec_prefetch!r} is not one of '', '0', '1'.")
    if spec_prefetch == "1":
        if _knob_value(SPEC_INDEX_ENV):
            raise RuntimeError(f"{SPEC_PREFETCH_ENV} and {SPEC_INDEX_ENV} wrap the same indexer and mixer; arm one.")
        if not _knob_value(STEP_TIMERS_ENV) or not _flag_on(HOST_KV_ENV) or not _flag_on(HOST_KV_FUSED_ENV):
            raise RuntimeError(f"{SPEC_PREFETCH_ENV} is armed by the step timers and copies through the fused "
                               f"host-KV path, so it needs {STEP_TIMERS_ENV}=1, {HOST_KV_ENV}=1 and "
                               f"{HOST_KV_FUSED_ENV}=1.")
    layers_said = str(os.environ.get(SPEC_PREFETCH_LAYERS_ENV) or "").strip()
    if layers_said:
        if spec_prefetch != "1":
            raise RuntimeError(f"{SPEC_PREFETCH_LAYERS_ENV} chooses the prefetched layers and needs {SPEC_PREFETCH_ENV}=1.")
        if not re.fullmatch(r"\d+(,\d+)*", layers_said):
            raise RuntimeError(f"{SPEC_PREFETCH_LAYERS_ENV}={layers_said!r} is not a comma list of layer numbers.")
    if _knob_value(SPEC_INDEX_ENV) and not _knob_value(STEP_TIMERS_ENV):
        raise RuntimeError(f"{SPEC_INDEX_ENV} is armed by the step timers after loading and needs {STEP_TIMERS_ENV}=1.")
    if _knob_value(SPEC_INDEX_ENV) and not _flag_on(HOST_KV_ENV):
        raise RuntimeError(f"{SPEC_INDEX_ENV} tests the picks against the host-KV hot region and needs {HOST_KV_ENV}=1.")
    said = str(os.environ.get(SPEC_INDEX_VARIANTS_ENV) or "").strip()
    if said and not _knob_value(SPEC_INDEX_ENV):
        raise RuntimeError(f"{SPEC_INDEX_VARIANTS_ENV} chooses the spec-index variants and needs {SPEC_INDEX_ENV}=1.")
    chosen = [one.strip() for one in said.split(",") if one.strip()]
    if said and (not chosen or any(one not in SPEC_INDEX_VARIANTS for one in chosen)):
        raise RuntimeError(f"{SPEC_INDEX_VARIANTS_ENV}={said!r} is not a comma list of {SPEC_INDEX_VARIANTS}.")
    rows = str(os.environ.get(SPEC_INDEX_ROWS_ENV) or "").strip()
    if rows and not _knob_value(SPEC_INDEX_ENV):
        raise RuntimeError(f"{SPEC_INDEX_ROWS_ENV} sizes the spec-index state buffers and needs {SPEC_INDEX_ENV}=1.")
    if rows and not (rows.isdigit() and 0 < int(rows) <= SPEC_INDEX_ROWS_MAX):
        raise RuntimeError(f"{SPEC_INDEX_ROWS_ENV}={rows!r} is not a row count in 1..{SPEC_INDEX_ROWS_MAX}.")
    lanes = str(os.environ.get(SPEC_INDEX_LANES_ENV) or "").strip()
    if lanes and not _knob_value(SPEC_INDEX_ENV):
        raise RuntimeError(f"{SPEC_INDEX_LANES_ENV} picks how many lanes the spec index measures and needs {SPEC_INDEX_ENV}=1.")
    if lanes and not (lanes.isdigit() and 0 < int(lanes) <= SPEC_INDEX_LANES_MAX):
        raise RuntimeError(f"{SPEC_INDEX_LANES_ENV}={lanes!r} is not a lane count in 1..{SPEC_INDEX_LANES_MAX}.")
    capture = _knob_value(DRAFT_CAPTURE_ENV)
    for name in (DRAFT_CAPTURE_GIB_ENV, DRAFT_CAPTURE_LANES_ENV, DRAFT_CAPTURE_DIR_ENV, DRAFT_CAPTURE_IDLE_ENV,
                 DRAFT_CAPTURE_DRAFTER_ENV, DRAFT_CAPTURE_HARD_ENV, DRAFT_CAPTURE_ROTATE_ENV,
                 DRAFT_CAPTURE_GAMES_ENV):
        if str(os.environ.get(name) or "").strip() and not capture:
            raise RuntimeError(f"{name} shapes the decode-path capture and needs {DRAFT_CAPTURE_ENV}=1.")
    if capture and not _knob_value(STEP_TIMERS_ENV):
        raise RuntimeError(f"{DRAFT_CAPTURE_ENV} is armed by the step timers after loading and needs {STEP_TIMERS_ENV}=1.")
    if capture and not str(os.environ.get(MTP_TOKENS_ENV) or "").strip().isdigit():
        raise RuntimeError(f"{DRAFT_CAPTURE_ENV} records what the MTP drafter reads and needs {MTP_TOKENS_ENV} set.")
    gib = str(os.environ.get(DRAFT_CAPTURE_GIB_ENV) or "").strip()
    if gib:
        try:
            ok = 0 < float(gib) <= 64
        except ValueError:
            ok = False
        if not ok:
            raise RuntimeError(f"{DRAFT_CAPTURE_GIB_ENV}={gib!r} is not a size in GiB in (0, 64].")
    lanes_said = str(os.environ.get(DRAFT_CAPTURE_LANES_ENV) or "").strip()
    if lanes_said and not (lanes_said.isdigit() and 0 < int(lanes_said) <= 64):
        raise RuntimeError(f"{DRAFT_CAPTURE_LANES_ENV}={lanes_said!r} is not a session count in 1..64.")
    idle = str(os.environ.get(DRAFT_CAPTURE_IDLE_ENV) or "").strip()
    if idle:
        try:
            ok = float(idle) > 0
        except ValueError:
            ok = False
        if not ok:
            raise RuntimeError(f"{DRAFT_CAPTURE_IDLE_ENV}={idle!r} is not a positive number of seconds.")
    drafter = _knob_value(DRAFT_CAPTURE_DRAFTER_ENV)
    if drafter and not _flag_on(DRAFT_CONF_LOG_ENV):
        raise RuntimeError(f"{DRAFT_CAPTURE_DRAFTER_ENV} reads the drafter's confidence from the draft confidence log and "
                           f"needs {DRAFT_CONF_LOG_ENV}=1.")
    hard = str(os.environ.get(DRAFT_CAPTURE_HARD_ENV) or "").strip()
    if hard:
        try:
            ok = 0 < float(hard) <= 1
        except ValueError:
            ok = False
        if not ok:
            raise RuntimeError(f"{DRAFT_CAPTURE_HARD_ENV}={hard!r} is not a confidence in (0, 1].")
        if not drafter:
            raise RuntimeError(f"{DRAFT_CAPTURE_HARD_ENV} judges the drafter's confidence and needs {DRAFT_CAPTURE_DRAFTER_ENV}=1.")
    rotate = str(os.environ.get(DRAFT_CAPTURE_ROTATE_ENV) or "").strip()
    if rotate and not (rotate.isdigit() and int(rotate) > 0):
        raise RuntimeError(f"{DRAFT_CAPTURE_ROTATE_ENV}={rotate!r} is not a positive row count.")
    experts = _knob_value(EXPERT_TALLY_ENV)
    if not _knob_value(STEP_TIMERS_ENV):
        if not experts:
            return {"armed": False}
        row = _patch_with_table(EXPERT_TALLY_ENV, STEP_TIMERS_HELPER, STEP_TIMERS_TABLE, STEP_TIMERS_INSTALLED)
        print(_armed_said("EXPERT_TALLY", row, STEP_TIMERS_INSTALLED,
                          "the model runner arms the expert tally after loading: every MoE layer marks the experts "
                          "its tokens route to in a device bitmap each forward, counts them on the GPU and adds the "
                          "count into a per-layer accumulator, read without a sync and printed as an EXPERT_TALLY "
                          "line every few hundred steps; the step timers stay off"), flush=True)
        return row
    row = _patch_with_table(STEP_TIMERS_ENV, STEP_TIMERS_HELPER, STEP_TIMERS_TABLE, STEP_TIMERS_INSTALLED)
    if _knob_value(VISION_TALLY_ENV):
        row["vision_tally_sha256"] = _install_helper(VISION_TALLY_ENV, VISION_TALLY_HELPER, VISION_TALLY_INSTALLED)
        print(f"VISION_TALLY installed at {VISION_TALLY_INSTALLED}: the step timers wrap the encoder runner after "
              "loading and print one vision_tally line every 100 steps", flush=True)
    if capture:
        row["draft_capture_sha256"] = _install_helper(DRAFT_CAPTURE_ENV, DRAFT_CAPTURE_HELPER, DRAFT_CAPTURE_INSTALLED)
        print(f"DRAFT_CAPTURE installed at {DRAFT_CAPTURE_INSTALLED}: the step timers arm it after loading; it records "
              "the MTP head's input rows and the target's verify distribution for a few sessions to the working "
              "directory, within its byte budget", flush=True)
    if _knob_value(SPEC_INDEX_ENV) or str(os.environ.get(SPEC_PREFETCH_ENV) or "").strip() == "1":
        row["spec_index_sha256"] = _install_helper(SPEC_INDEX_ENV, SPEC_INDEX_HELPER, SPEC_INDEX_INSTALLED)
        print(f"SPEC_INDEX installed at {SPEC_INDEX_INSTALLED}: the step timers arm it after loading; each QSA layer's "
              "indexer also selects from speculative inputs for the first lanes of every forward with decode rows "
              "outside a bucketed graph, and one SPEC_INDEX line per variant is printed every 200 steps", flush=True)
    print(_armed_said("STEP_TIMERS", row, STEP_TIMERS_INSTALLED,
                      "the model runner arms GPU step timers after loading: each op-group boundary launches a "
                      "one-thread %globaltimer kernel, captured into every CUDA graph, that adds the span since the "
                      "last boundary into a per-step ring on the GPU; the ring is read by a non-blocking copy every "
                      "32 steps and each step kind prints one STEP_TIMERS line every 20 s and appends a window "
                      "record to step-timers.jsonl"), flush=True)
    return row


def qsa_self_check_shapes(model_dir: Path) -> dict[str, int]:
    held = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = held.get("text_config", held)
    return {"head_dim": int(text["head_dim"]), "kv_heads": int(text["num_key_value_heads"]),
            "query_heads": int(text["num_attention_heads"]),
            "width": int(text["indexer_budget"]) + int(text["indexer_compress_ratio"]) - 1}


def qsa_self_check(env: dict[str, str], model_dir: Path) -> dict[str, Any]:
    env.pop(QSA_LEAN_CHECKED_ENV, None)
    env.pop(QSA_DECODE_LEAN_CHECKED_ENV, None)
    env.pop(QSA_UNION_DECODE_CHECKED_ENV, None)
    if not _flag_on(QSA_SELFCHECK_ENV):
        return {"armed": False}
    dtype = str(os.environ.get(KV_CACHE_DTYPE_ENV) or "").strip().lower()
    if not _flag_on(QSA_KV8_ENV) or dtype not in QSA_QUANT_DTYPES:
        raise RuntimeError(f"{QSA_SELFCHECK_ENV} checks the quantized QSA kernels before the model load and needs "
                           f"{QSA_KV8_ENV}=1 with {KV_CACHE_DTYPE_ENV} one of {QSA_QUANT_DTYPES}, found "
                           f"{QSA_KV8_ENV}={os.environ.get(QSA_KV8_ENV)!r} {KV_CACHE_DTYPE_ENV}={dtype!r}.")
    spread = 0
    if _flag_on(HOST_KV_ENV) and _flag_on(HOST_KV_FUSED_ENV):
        spread, _ = _bounded_integer_environment(HOST_KV_GATHER_SPREAD_ENV, default=0, minimum=0,
                                                 maximum=HOST_KV_GATHER_SPREAD_MAX)
    shapes = {**qsa_self_check_shapes(model_dir), "dtype": dtype, "spread": spread}
    lean = qsa_lean_on()
    decode_lean = qsa_decode_lean_on()
    union_decode = qsa_union_decode_on()
    optional = lean or decode_lean or union_decode
    script = (QSA_SELFCHECK_SCRIPT + (QSA_SELFCHECK_LEAN_SCRIPT if lean else "")
              + (QSA_SELFCHECK_DECODE_LEAN_SCRIPT if decode_lean else "")
              + (QSA_SELFCHECK_UNION_DECODE_SCRIPT if union_decode else ""))
    dropped = {name for name, armed in ((QSA_LEAN_ENV, lean), (QSA_DECODE_LEAN_ENV, decode_lean),
                                        (QSA_UNION_DECODE_ENV, union_decode)) if armed}
    child = {key: value for key, value in env.items() if key not in dropped} if dropped else env
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-c", script, json.dumps(shapes)], env=child,
                              capture_output=True, text=True, timeout=QSA_SELFCHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired as failure:
        partial = failure.stdout.decode("utf-8", "replace") if isinstance(failure.stdout, bytes) else ""
        if not (optional and QSA_SELFCHECK_MARK in partial):
            raise RuntimeError(f"QSA_SELFCHECK: the {dtype} kernel check did not finish in {QSA_SELFCHECK_TIMEOUT_S} s; "
                               f"the setup stops before the model load") from failure
        done = subprocess.CompletedProcess(failure.cmd, -9, stdout=partial, stderr="the lean check timed out")
    found = [line for line in done.stdout.splitlines() if line.startswith(QSA_SELFCHECK_MARK)]
    lean_found = [line for line in done.stdout.splitlines() if line.startswith(QSA_SELFCHECK_LEAN_MARK)]
    decode_found = [line for line in done.stdout.splitlines() if line.startswith(QSA_SELFCHECK_DECODE_LEAN_MARK)]
    union_found = [line for line in done.stdout.splitlines() if line.startswith(QSA_SELFCHECK_UNION_DECODE_MARK)]
    if not found or (done.returncode and not optional):
        raise RuntimeError(f"QSA_SELFCHECK: the {dtype} kernel check exited {done.returncode} with no verdict; the "
                           f"setup stops before the model load: {(done.stderr or done.stdout)[-1500:]}")
    held = json.loads(found[-1][len(QSA_SELFCHECK_MARK):])
    seconds = time.monotonic() - started
    for line in done.stdout.splitlines():
        if line.startswith("QSA_IDENTITY "):
            name, verdict, detail, spent = json.loads(line[len("QSA_IDENTITY "):])
            print(f"KERNEL_IDENTITY {verdict} {name}: {detail} ({spent:.1f} s)", flush=True)
    for name, ok, detail, *spent in held["rows"]:
        print(f"QSA_SELFCHECK {'pass' if ok else 'FAIL'} {name}: {detail}"
              + (f" ({spent[0]:.1f} s)" if spent else ""), flush=True)
    moe_identity(env, model_dir)
    failed = [row[0] for row in held["rows"] if not row[1]]
    print(f"QSA_SELFCHECK {dtype} on {held['device']}: {len(held['rows']) - len(failed)} of {len(held['rows'])} "
          f"pass in {seconds:.1f} s, before the model load", flush=True)
    if failed:
        raise RuntimeError(f"QSA_SELFCHECK: {failed} failed on {held['device']} for the {dtype} cache; the setup "
                           f"stops before the model load")
    row = {"armed": True, "dtype": dtype, "shapes": shapes, "rows": held["rows"], "seconds": round(seconds, 1),
           "device": held["device"]}
    if lean:
        later = decode_lean and bool(lean_found) and not decode_found
        row["lean"] = qsa_lean_verdict(env, held["device"], lean_found,
                                       subprocess.CompletedProcess(done.args, 0, done.stdout, done.stderr)
                                       if later else done)
    if decode_lean:
        row["decode_lean"] = qsa_decode_lean_verdict(env, held["device"], decode_found, done)
    if union_decode:
        row["union_decode"] = qsa_union_decode_verdict(env, held["device"], union_found, done)
    return row


def qsa_union_decode_verdict(env: dict[str, str], device: str, found: list[str], done: Any) -> dict[str, Any]:
    rows = json.loads(found[-1][len(QSA_SELFCHECK_UNION_DECODE_MARK):]) if found else []
    for name, ok, detail, *spent in rows:
        print(f"QSA_SELFCHECK {'pass' if ok else 'FAIL'} {name}: {detail}"
              + (f" ({spent[0]:.1f} s)" if spent else ""), flush=True)
    lean_live = str(env.get(QSA_DECODE_LEAN_ENV) or "").strip() == "1"
    if lean_live and rows and all(one[1] for one in rows):
        env[QSA_UNION_DECODE_CHECKED_ENV] = "1"
        print(f"QSA_UNION_DECODE checked on {device} before the model load; CUDA graphs captured before its first "
              f"in-engine checks carry it", flush=True)
        return {"checked": True, "rows": rows}
    if not lean_live:
        why = f"{QSA_DECODE_LEAN_ENV} stood aside and the union decode runs inside it"
    elif rows:
        why = [one[0] for one in rows if not one[1]]
    else:
        why = f"its check exited {done.returncode}: {(done.stderr or '')[-400:]}"
    if lean_live:
        selfcheck_fail(env, QSA_UNION_DECODE_ENV, f"the check before the model load failed on {device}: {why}")
    env[QSA_UNION_DECODE_ENV] = "0"
    env.pop(QSA_UNION_DECODE_CHECKED_ENV, None)
    print(f"{QSA_UNION_DECODE_ENV} stood aside: {why} on {device}; the engine serves the per-row lean decode "
          f"({QSA_UNION_DECODE_ENV}=0 in its environment)", flush=True)
    return {"checked": False, "rows": rows, "aside": str(why)}


def qsa_decode_lean_verdict(env: dict[str, str], device: str, found: list[str], done: Any) -> dict[str, Any]:
    rows = json.loads(found[-1][len(QSA_SELFCHECK_DECODE_LEAN_MARK):]) if found else []
    for name, ok, detail, *spent in rows:
        print(f"QSA_SELFCHECK {'pass' if ok else 'FAIL'} {name}: {detail}"
              + (f" ({spent[0]:.1f} s)" if spent else ""), flush=True)
    if rows and all(one[1] for one in rows):
        env[QSA_DECODE_LEAN_CHECKED_ENV] = "1"
        print(f"QSA_DECODE_LEAN checked on {device} before the model load; CUDA graphs captured before its first "
              f"in-engine checks carry it", flush=True)
        return {"checked": True, "rows": rows}
    why = ([one[0] for one in rows if not one[1]] if rows else
           f"its check exited {done.returncode}: {(done.stderr or '')[-400:]}")
    selfcheck_fail(env, QSA_DECODE_LEAN_ENV, f"the check before the model load failed on {device}: {why}")
    env[QSA_DECODE_LEAN_ENV] = "0"
    env.pop(QSA_DECODE_LEAN_CHECKED_ENV, None)
    print(f"{QSA_DECODE_LEAN_ENV} stood aside: {why} on {device}; the engine serves the stock QSA decode "
          f"({QSA_DECODE_LEAN_ENV}=0 in its environment)", flush=True)
    return {"checked": False, "rows": rows, "aside": str(why)}


def moe_identity_shapes(model_dir: Path) -> dict[str, int]:
    held = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = held.get("text_config", held)
    quant = held.get("quantization_config") or text.get("quantization_config") or {}
    found = {"hidden": text.get("hidden_size"), "intermediate": text.get("moe_intermediate_size"),
             "topk": text.get("num_experts_per_tok"), "group": quant.get("group_size")}
    return {key: int(found[key]) if isinstance(found[key], int) and found[key] > 0 else value
            for key, value in MOE_IDENTITY_DEFAULTS.items()}


def moe_identity(env: dict[str, str], model_dir: Path) -> list[list[Any]]:
    said = str(os.environ.get(KERNEL_IDENTITY_ENV) or "").strip()
    if not said.isdigit() or not int(said):
        return []
    try:
        shapes = moe_identity_shapes(model_dir)
        done = subprocess.run([sys.executable, "-c", MOE_IDENTITY_SCRIPT, json.dumps(shapes), str(BUNDLE_DIR)], env=env,
                              capture_output=True, text=True, timeout=MOE_IDENTITY_TIMEOUT_S)
        out, tail = done.stdout, f"exit {done.returncode}: {(done.stderr or '')[-300:]}"
    except subprocess.TimeoutExpired as failure:
        out = failure.stdout.decode("utf-8", "replace") if isinstance(failure.stdout, bytes) else ""
        tail = f"timed out at {MOE_IDENTITY_TIMEOUT_S} s"
    except Exception as error:
        out, tail = "", f"{type(error).__name__}: {error}"[:300]
    rows = [json.loads(line[len("QSA_IDENTITY "):]) for line in out.splitlines() if line.startswith("QSA_IDENTITY ")]
    for name, verdict, detail, spent in rows:
        print(f"KERNEL_IDENTITY {verdict} {name}: {detail} ({spent:.1f} s)", flush=True)
    if not rows:
        print(f"KERNEL_IDENTITY stopped moe: no verdict, {tail}", flush=True)
    return rows


def qsa_lean_verdict(env: dict[str, str], device: str, found: list[str], done: Any) -> dict[str, Any]:
    rows = json.loads(found[-1][len(QSA_SELFCHECK_LEAN_MARK):]) if found else []
    for name, ok, detail, *spent in rows:
        print(f"QSA_SELFCHECK {'pass' if ok else 'FAIL'} {name}: {detail}"
              + (f" ({spent[0]:.1f} s)" if spent else ""), flush=True)
    if rows and all(one[1] for one in rows) and not done.returncode:
        env[QSA_LEAN_CHECKED_ENV] = "1"
        caps = [one[4]["maxnreg"] for one in rows if len(one) > 4 and isinstance(one[4], dict)]
        if caps:
            env[QSA_LEAN_MAXNREG_ENV] = str(int(caps[-1]))
        print(f"QSA_PREFILL_LEAN checked on {device} before the model load; CUDA graphs captured before its first "
              f"in-engine checks carry it" + (f"; the engine compiles it at maxnreg {caps[-1] or 'free'}, the "
                                              f"fastest on this card" if caps else ""), flush=True)
        return {"checked": True, "rows": rows, **({"maxnreg": int(caps[-1])} if caps else {})}
    why = ([one[0] for one in rows if not one[1]] if rows and not done.returncode else
           f"its check exited {done.returncode}: {(done.stderr or '')[-400:]}")
    selfcheck_fail(env, QSA_LEAN_ENV, f"the check before the model load failed on {device}: {why}")
    env[QSA_LEAN_ENV] = "0"
    env.pop(QSA_LEAN_CHECKED_ENV, None)
    print(f"QSA_PREFILL_LEAN stands aside: {why} on {device}; the engine serves QSA prefill without it "
          f"({QSA_LEAN_ENV}=0 in its environment)", flush=True)
    return {"checked": False, "rows": rows, "aside": str(why)}


def patch_cache_keep_end() -> dict[str, Any]:
    if not _flag_on(CACHE_KEEP_END_ENV):
        if str(os.environ.get(CACHE_KEEP_END_RAM_ENV) or "").strip() not in ("", "0"):
            raise RuntimeError(f"{CACHE_KEEP_END_RAM_ENV} queues CACHE_KEEP_END's reply snapshots to "
                               f"the RAM tier and needs {CACHE_KEEP_END_ENV} armed.")
        if _flag_on(PAUSED_REGISTER_ENV):
            raise RuntimeError(f"{PAUSED_REGISTER_ENV} keys a paused session's attention blocks beside "
                               f"CACHE_KEEP_END's reply snapshots and needs {CACHE_KEEP_END_ENV} armed.")
        return {"armed": False}
    if _flag_on(PAUSED_REGISTER_ENV) and not _flag_on("TAAF_VLLM_PAUSED"):
        raise RuntimeError(f"{PAUSED_REGISTER_ENV} keys the blocks of paused sessions and needs TAAF_VLLM_PAUSED.")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{CACHE_KEEP_END_ENV} edits vLLM 0.30.0's KV cache manager, scheduler and model "
            f"runner; the {RUNTIME_PROFILE} runtime ships other ones.")
    for needed, why in ((CACHE_KEEP_ENV, "registers its snapshots at the fine prefix-match-unit "
                                         "and pins the newest one through the retained-copy hook "
                                         "it adds"),
                        (GDN_ACCEPT_ENV, "has the GDN commit write the exact state at a reply "
                                         "boundary while it folds the accepted tokens")):
        if not _flag_on(needed):
            raise RuntimeError(f"{CACHE_KEEP_END_ENV} needs {needed} armed: {needed} {why}.")
    helper_path = BUNDLE_DIR / CACHE_KEEP_END_HELPER
    table_path = BUNDLE_DIR / CACHE_KEEP_END_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{CACHE_KEEP_END_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str | None, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    if not (site / GDN_ACCEPT_INSTALLED).is_file():
        raise RuntimeError(
            f"{CACHE_KEEP_END_ENV} needs the GDN_ACCEPT helper {GDN_ACCEPT_INSTALLED} installed "
            f"first; it is absent from {site}")
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if pinned is None:
            if all(text.count(put) == 1 for _, put in edits):
                files[relative] = {"anchored": True, "patched_sha256": before, "already": True}
                continue
        elif _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{CACHE_KEEP_END_ENV} cannot apply: {relative} is {before[:16]}, neither "
                    f"the {pinned[:16]} its edits were written against nor that file with them "
                    f"applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{CACHE_KEEP_END_ENV} cannot apply: an anchor of {relative} appears "
                    f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before if pinned is not None else None,
                           "anchored": pinned is None,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / CACHE_KEEP_END_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(
                f"the {CACHE_KEEP_END_ENV} helper {CACHE_KEEP_END_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.keepend.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {CACHE_KEEP_END_ENV} edit of {relative} did not land")
    row = {"armed": True, "already": not made and not helper_fresh,
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "files": files}
    print(f"CACHE_KEEP_END {'already armed' if row['already'] else 'armed'}: {len(made)} of "
          f"{len(targets)} vLLM file(s) edited from table {row['table_sha256'][:16]}, helper at "
          f"{CACHE_KEEP_END_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} "
                      + (f"{held['stock_sha256'][:16]}" if held.get("stock_sha256") else "anchored")
                      + f" -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + f"; a decoding request gets one spare state block per Mamba group while it is a few "
          f"steps from its next {cache_keep_match_unit()}-token boundary, the GDN commit and the "
          f"conv copy write the exact state at that boundary into it, and the scheduler "
          f"registers it there once the step's output is processed; the newest snapshot is "
          f"pinned in place of the prompt-tail copy and the one before it stays cached, so a "
          f"request that extends this one resumes one to two match units before its end",
          flush=True)
    return row


def host_kv_pick_streaks(picks: int) -> dict[str, Any] | None:
    said = str(os.environ.get(HOST_KV_PICK_AT_ENV) or "").strip()
    if not said:
        for name in (HOST_KV_PICK_STEPS_ENV, HOST_KV_PICK_REQS_ENV, HOST_KV_PICK_MIB_ENV):
            if str(os.environ.get(name) or "").strip():
                raise RuntimeError(f"{name} shapes the pick-log streaks, which need {HOST_KV_PICK_AT_ENV}.")
        return None
    try:
        times = [float(part) for part in said.split(",") if part.strip()]
    except ValueError as error:
        raise RuntimeError(f"{HOST_KV_PICK_AT_ENV}={said!r} is not a comma list of seconds of play.") from error
    if (not times or len(times) > HOST_KV_PICK_AT_MAX or any(not 0.0 <= one < 1e7 for one in times)
            or times != sorted(set(times))):
        raise RuntimeError(f"{HOST_KV_PICK_AT_ENV}={said!r} must list 1 to {HOST_KV_PICK_AT_MAX} distinct seconds of "
                           "play in ascending order.")
    steps, _ = _bounded_integer_environment(HOST_KV_PICK_STEPS_ENV, default=48, minimum=1, maximum=4096)
    reqs, _ = _bounded_integer_environment(HOST_KV_PICK_REQS_ENV, default=8, minimum=1, maximum=256)
    mib, _ = _bounded_integer_environment(HOST_KV_PICK_MIB_ENV, default=768, minimum=1, maximum=16384)
    if not (_flag_on(ENFORCE_EAGER_ENV) or str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower() == "1"):
        print(f"HOST_KV pick streaks stand aside: {HOST_KV_PICK_AT_ENV} is set, but the runner hook that takes a "
              f"streak's steps off the full graphs comes with {MIXED_FULL_ENV}=1, which this launch does not arm, "
              "so no streak opens and no streak file is written", flush=True)
        return None
    if picks:
        print(f"HOST_KV pick streaks beside the pick log: the {picks} two-step bursts stay as they are and write "
              "burst*.npz; the streaks write streak*.npz into the same folder under their own byte cap", flush=True)
    return {"at": times, "steps": steps, "reqs": reqs, "mib": mib}


def host_kv_settings() -> dict[str, Any] | None:
    if not _flag_on(HOST_KV_ENV):
        for name in (HOST_KV_GIB_ENV, HOST_KV_ROWS_ENV, HOST_KV_STAGE_ENV, HOST_KV_SPLIT_ENV, HOST_KV_ADMIT_ENV,
                     HOST_KV_ROW_ATTEND_ENV, HOST_KV_TAIL_ENV, HOST_KV_PICK_LOG_ENV, HOST_KV_PICK_EVERY_ENV,
                     HOST_KV_SPARSE_ENV, HOST_KV_HOT_RELEASE_ENV, HOST_KV_INDEXER_WIDE_ENV, HOST_KV_UNION_COUNT_ENV,
                     HOST_KV_HOLD_CLEAN_ENV, HOST_KV_HOLD_RESERVE_ENV, HOT_KEEP_WARM_ENV,
                     QSA_STAGE_AHEAD_ENV, QSA_STAGE_AHEAD_PROGRAMS_ENV,
                     HOST_KV_PLAN_ONCE_ENV, HOST_KV_PLAN_ONCE_CAPTURED_ENV, HOST_KV_SYNCFREE_ENV, HOST_KV_UNION_ENV,
                     HOST_KV_HOT_ROWS_ENV, HOST_KV_PLAN_ONCE_CHECKS_ENV, HOST_KV_TAKE_ROWS_ENV,
                     HOST_KV_TAKE_SHADOW_ENV,
                     HOST_KV_COPY_ASYNC_ENV, HOST_KV_FIT_ENV, HOST_KV_GROWTH_ENV, HOST_KV_SPILL_SKIP_ENV,
                     HOST_KV_SPILL_ONESORT_ENV, HOST_KV_MARK_FILL_ENV, HOST_KV_GATHER_EARLY_ENV,
                     HOST_KV_FUSED_ENV, HOST_KV_HOT_SPLIT_ENV, HOST_KV_GROUP_ENV, HOST_KV_GATHER_SPREAD_ENV,
                     HOST_KV_PCIE_TALLY_ENV, HOST_KV_RUN_MERGE_ENV,
                     *HOST_KV_FUSED_KNOBS,
                     HOST_KV_MIXED_CAPTURE_ENV, HOST_KV_PICK_AT_ENV, HOST_KV_PICK_STEPS_ENV,
                     HOST_KV_PICK_REQS_ENV, HOST_KV_PICK_MIB_ENV, HOST_KV_PREFETCH_ENV,
                     HOST_KV_PREFETCH_PROGRAMS_ENV, HOST_KV_PICK_RANKS_ENV, HOST_KV_PREFETCH_PREV_ENV,
                     HOST_KV_PREFETCH_PREV_ROWS_ENV, HOST_KV_PREFETCH_PREV_SCOPE_ENV,
                     HOST_KV_PREFETCH_PREV_ONCE_ENV):
            if str(os.environ.get(name) or "").strip():
                raise RuntimeError(f"{name} shapes the host KV path and needs {HOST_KV_ENV} armed.")
        return None
    gib, _ = _bounded_integer_environment(HOST_KV_GIB_ENV, default=HOST_KV_DEFAULT_GIB, minimum=1, maximum=2048)
    gib = host_kv_fit(gib)
    rows, _ = _bounded_integer_environment(HOST_KV_ROWS_ENV, default=0, minimum=0, maximum=32768)
    stage, _ = _bounded_integer_environment(HOST_KV_STAGE_ENV, default=0, minimum=0, maximum=HOST_KV_STAGE_MAX_MIB)
    admit = str(os.environ.get(HOST_KV_ADMIT_ENV) or "").strip().lower()
    if admit not in HOST_KV_ADMIT_MODES:
        raise RuntimeError(f"{HOST_KV_ADMIT_ENV}={admit!r} is not one of {HOST_KV_ADMIT_MODES}.")
    row_attend = str(os.environ.get(HOST_KV_ROW_ATTEND_ENV) or "").strip().lower()
    if row_attend not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_ROW_ATTEND_ENV}={row_attend!r} is not one of '', '0', '1'.")
    union_swap = str(os.environ.get(HOST_KV_UNION_ENV) or "").strip().lower()
    if union_swap not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_UNION_ENV}={union_swap!r} is not one of '', '0', '1'.")
    if union_swap == "1" and row_attend == "1":
        raise RuntimeError(f"{HOST_KV_UNION_ENV} swaps a lane's verification rows together before attention and "
                           f"{HOST_KV_ROW_ATTEND_ENV} attends each row right after its own swap; arm one of them.")
    hot_rows, _ = _bounded_integer_environment(HOST_KV_HOT_ROWS_ENV, default=0, minimum=0, maximum=32768)
    if hot_rows and union_swap != "1":
        raise RuntimeError(f"{HOST_KV_HOT_ROWS_ENV} sizes the hot region below the union of a lane's verification "
                           f"rows, which only {HOST_KV_UNION_ENV}=1 can spill; arm it.")
    if hot_rows:
        rows = hot_rows
    copy_async = str(os.environ.get(HOST_KV_COPY_ASYNC_ENV) or "").strip().lower()
    if copy_async not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_COPY_ASYNC_ENV}={copy_async!r} is not one of '', '0', '1'.")
    spill_skip = str(os.environ.get(HOST_KV_SPILL_SKIP_ENV) or "").strip().lower()
    if spill_skip not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_SPILL_SKIP_ENV}={spill_skip!r} is not one of '', '0', '1'.")
    if spill_skip == "1" and not hot_rows:
        raise RuntimeError(f"{HOST_KV_SPILL_SKIP_ENV} trims the spill passes of a hot region capped by "
                           f"{HOST_KV_HOT_ROWS_ENV}, which is not set; without it there is nothing to trim.")
    onesort = str(os.environ.get(HOST_KV_SPILL_ONESORT_ENV) or "").strip().lower()
    if onesort not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_SPILL_ONESORT_ENV}={onesort!r} is not one of '', '0', '1'.")
    if onesort == "1" and not hot_rows:
        raise RuntimeError(f"{HOST_KV_SPILL_ONESORT_ENV} sorts the spill passes of a hot region capped by "
                           f"{HOST_KV_HOT_ROWS_ENV}, which is not set; without it there are no spill passes.")
    early = str(os.environ.get(HOST_KV_GATHER_EARLY_ENV) or "").strip().lower()
    if early not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_GATHER_EARLY_ENV}={early!r} is not one of '', '0', '1'.")
    if early == "1" and not hot_rows:
        raise RuntimeError(f"{HOST_KV_GATHER_EARLY_ENV} moves the union swap of a hot region capped by "
                           f"{HOST_KV_HOT_ROWS_ENV}, which is not set; without it there is no union swap to move.")
    fused = str(os.environ.get(HOST_KV_FUSED_ENV) or "").strip().lower()
    if fused not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_FUSED_ENV}={fused!r} is not one of '', '0', '1'.")
    if fused == "1" and not hot_rows:
        raise RuntimeError(f"{HOST_KV_FUSED_ENV} fuses the capped union swap of a hot region capped by "
                           f"{HOST_KV_HOT_ROWS_ENV}, which is not set; without it there is nothing to fuse.")
    hot_split = str(os.environ.get(HOST_KV_HOT_SPLIT_ENV) or "").strip().lower()
    if hot_split not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_HOT_SPLIT_ENV}={hot_split!r} is not one of '', '0', '1'.")
    if hot_split == "1" and not hot_rows:
        raise RuntimeError(f"{HOST_KV_HOT_SPLIT_ENV} resolves a hot region capped by {HOST_KV_HOT_ROWS_ENV} in "
                           f"several passes, which is not set; without it the region already holds every union.")
    fused_knobs = {}
    for name, said in HOST_KV_FUSED_KNOBS.items():
        value = str(os.environ.get(name) or "").strip().lower()
        if value not in ("", "0", "1"):
            raise RuntimeError(f"{name}={value!r} is not one of '', '0', '1'.")
        if value == "1" and fused != "1":
            raise RuntimeError(f"{name} changes a kernel of the fused host KV path, which needs {HOST_KV_FUSED_ENV}=1.")
        fused_knobs[name] = value == "1"
        if value == "1":
            print(f"HOST_KV {name.removeprefix('TAAF_VLLM_HOST_KV_').lower()}: {said}", flush=True)
    if fused_knobs[HOST_KV_ONE_HASH_ENV] and not fused_knobs[HOST_KV_ONE_LAUNCH_ENV]:
        raise RuntimeError(f"{HOST_KV_ONE_HASH_ENV} replaces the sort of the one-launch resolve, which needs "
                           f"{HOST_KV_ONE_LAUNCH_ENV}=1.")
    if fused_knobs[HOST_KV_ONE_PACK_ENV] and not fused_knobs[HOST_KV_ONE_LAUNCH_ENV]:
        raise RuntimeError(f"{HOST_KV_ONE_PACK_ENV} packs the key list of the one-launch resolve, which needs "
                           f"{HOST_KV_ONE_LAUNCH_ENV}=1.")
    if fused_knobs[HOST_KV_FLAT_COPY_ENV] and not fused_knobs[HOST_KV_ONE_LAUNCH_ENV]:
        raise RuntimeError(f"{HOST_KV_FLAT_COPY_ENV} copies the misses of the one-launch resolve, which needs "
                           f"{HOST_KV_ONE_LAUNCH_ENV}=1.")
    if fused_knobs[HOST_KV_FLAT_LINES_ENV] and not fused_knobs[HOST_KV_FLAT_COPY_ENV]:
        raise RuntimeError(f"{HOST_KV_FLAT_LINES_ENV} changes how the flat copy reads host rows, which needs "
                           f"{HOST_KV_FLAT_COPY_ENV}=1.")
    if fused_knobs[HOST_KV_ONE_CUDA_ENV] and not (fused_knobs[HOST_KV_ONE_LAUNCH_ENV]
                                                   and fused_knobs[HOST_KV_FLAT_COPY_ENV]):
        raise RuntimeError(f"{HOST_KV_ONE_CUDA_ENV} replaces the resolve kernel of the flat one-launch path, which "
                           f"needs {HOST_KV_ONE_LAUNCH_ENV}=1 and {HOST_KV_FLAT_COPY_ENV}=1.")
    cuda_threads = str(os.environ.get(HOST_KV_ONE_CUDA_THREADS_ENV) or "").strip()
    if cuda_threads and not (cuda_threads.isdigit() and int(cuda_threads) in HOST_KV_ONE_CUDA_THREADS):
        raise RuntimeError(f"{HOST_KV_ONE_CUDA_THREADS_ENV}={cuda_threads!r} is not one of "
                           f"{HOST_KV_ONE_CUDA_THREADS}.")
    cuda_defines = str(os.environ.get(HOST_KV_ONE_CUDA_DEFINES_ENV) or "").strip()
    if cuda_defines and not re.fullmatch(r"[A-Z][A-Z0-9_]*=[A-Za-z0-9_.-]+(,[A-Z][A-Z0-9_]*=[A-Za-z0-9_.-]+)*",
                                         cuda_defines):
        raise RuntimeError(f"{HOST_KV_ONE_CUDA_DEFINES_ENV}={cuda_defines!r} is not a comma list of NAME=VALUE.")
    if (cuda_threads or cuda_defines) and not fused_knobs[HOST_KV_ONE_CUDA_ENV]:
        raise RuntimeError(f"{HOST_KV_ONE_CUDA_THREADS_ENV} and {HOST_KV_ONE_CUDA_DEFINES_ENV} shape the CUDA "
                           f"resolve, which needs {HOST_KV_ONE_CUDA_ENV}=1.")
    mixed_capture = str(os.environ.get(HOST_KV_MIXED_CAPTURE_ENV) or "").strip().lower()
    if mixed_capture not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_MIXED_CAPTURE_ENV}={mixed_capture!r} is not one of '', '0', '1'.")
    group, _ = _bounded_integer_environment(HOST_KV_GROUP_ENV, default=0, minimum=0, maximum=max(HOST_KV_GROUPS))
    if group not in HOST_KV_GROUPS:
        raise RuntimeError(f"{HOST_KV_GROUP_ENV}={group} is not one of {HOST_KV_GROUPS}.")
    if group > 1 and fused != "1":
        raise RuntimeError(f"{HOST_KV_GROUP_ENV} reads host KV in whole groups with a Triton kernel of the fused "
                           f"host KV path, which needs {HOST_KV_FUSED_ENV}=1.")
    if group > 1:
        print(f"HOST_KV group: a gather reads every group of {group} consecutive tokens that holds a missed token as "
              f"one contiguous row and keeps only the missed tokens on the GPU", flush=True)
    spread, _ = _bounded_integer_environment(HOST_KV_GATHER_SPREAD_ENV, default=0, minimum=0,
                                             maximum=HOST_KV_GATHER_SPREAD_MAX)
    if spread and fused != "1":
        raise RuntimeError(f"{HOST_KV_GATHER_SPREAD_ENV} deals decode host reads over a grid of Triton programs "
                           f"of the fused host KV path, which needs {HOST_KV_FUSED_ENV}=1.")
    if spread:
        print(f"HOST_KV gather spread: a decode gather deals its missed rows evenly over {spread} programs an SM, "
              f"each reading several host rows at once, in place of the split and group gathers", flush=True)
    pcie_tally, _ = _bounded_integer_environment(HOST_KV_PCIE_TALLY_ENV, default=0, minimum=0,
                                                 maximum=max(HOST_KV_PCIE_TALLY_LEVELS))
    if pcie_tally and fused != "1":
        raise RuntimeError(f"{HOST_KV_PCIE_TALLY_ENV} counts the gathers of the fused host KV path, which needs "
                           f"{HOST_KV_FUSED_ENV}=1.")
    if pcie_tally:
        print(f"HOST_KV pcie tally level {pcie_tally}: decode demand gathers and prefetches count host rows, "
              f"transfers and bytes on the GPU" + (" and stamp the GPU clock around each gather" if pcie_tally > 1
                                                   else "") + ", one line every 100 steps", flush=True)
    run_merge_said = str(os.environ.get(HOST_KV_RUN_MERGE_ENV) or "").strip()
    run_merge = -1
    if run_merge_said:
        run_merge, _ = _bounded_integer_environment(HOST_KV_RUN_MERGE_ENV, default=0, minimum=0,
                                                    maximum=HOST_KV_RUN_MERGE_MAX)
        if fused != "1":
            raise RuntimeError(f"{HOST_KV_RUN_MERGE_ENV} reads host KV in merged runs with a Triton kernel of the "
                               f"fused host KV path, which needs {HOST_KV_FUSED_ENV}=1.")
        print(f"HOST_KV run merge: a gather sorts each lane's missed rows and reads runs whose neighbours lie within "
              f"{run_merge} rows as contiguous tiles, in place of the spread, group and split gathers", flush=True)
    tail, _ = _bounded_integer_environment(HOST_KV_TAIL_ENV, default=0, minimum=0, maximum=HOST_KV_TAIL_MAX)
    prefetch, _ = _bounded_integer_environment(HOST_KV_PREFETCH_ENV, default=0, minimum=0,
                                               maximum=HOST_KV_PREFETCH_MAX)
    prefetch_programs, _ = _bounded_integer_environment(HOST_KV_PREFETCH_PROGRAMS_ENV, default=0, minimum=0,
                                                        maximum=65536)
    if prefetch_programs and not prefetch:
        raise RuntimeError(f"{HOST_KV_PREFETCH_PROGRAMS_ENV} sizes the near-miss prefetch gather, which needs "
                           f"{HOST_KV_PREFETCH_ENV}.")
    if prefetch and not (fused == "1" and union_swap == "1" and hot_rows):
        raise RuntimeError(f"{HOST_KV_PREFETCH_ENV} prefetches into the capped hot region of the fused union swap, "
                           f"which needs {HOST_KV_FUSED_ENV}=1, {HOST_KV_UNION_ENV}=1 and {HOST_KV_HOT_ROWS_ENV}.")
    prev = str(os.environ.get(HOST_KV_PREFETCH_PREV_ENV) or "").strip().lower()
    if prev not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_PREFETCH_PREV_ENV}={prev!r} is not one of '', '0', '1'.")
    if prev == "1" and not (fused == "1" and union_swap == "1" and hot_rows):
        raise RuntimeError(f"{HOST_KV_PREFETCH_PREV_ENV} prefetches into the capped hot region of the fused union "
                           f"swap, which needs {HOST_KV_FUSED_ENV}=1, {HOST_KV_UNION_ENV}=1 and "
                           f"{HOST_KV_HOT_ROWS_ENV}.")
    prev_rows, _ = _bounded_integer_environment(HOST_KV_PREFETCH_PREV_ROWS_ENV, default=1024, minimum=1,
                                                maximum=HOST_KV_PREFETCH_PREV_ROWS_MAX)
    prev_scope = str(os.environ.get(HOST_KV_PREFETCH_PREV_SCOPE_ENV) or "").strip().lower()
    if prev_scope and prev_scope not in HOST_KV_PREFETCH_PREV_SCOPES:
        raise RuntimeError(f"{HOST_KV_PREFETCH_PREV_SCOPE_ENV}={prev_scope!r} is not one of "
                           f"{HOST_KV_PREFETCH_PREV_SCOPES}.")
    prev_once = str(os.environ.get(HOST_KV_PREFETCH_PREV_ONCE_ENV) or "").strip().lower()
    if prev_once not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_PREFETCH_PREV_ONCE_ENV}={prev_once!r} is not one of '', '0', '1'.")
    for name in (HOST_KV_PREFETCH_PREV_ROWS_ENV, HOST_KV_PREFETCH_PREV_SCOPE_ENV, HOST_KV_PREFETCH_PREV_ONCE_ENV):
        if str(os.environ.get(name) or "").strip() and prev != "1":
            raise RuntimeError(f"{name} shapes the previous index layer's prefetch, which needs "
                               f"{HOST_KV_PREFETCH_PREV_ENV}=1.")
    prev_scope = prev_scope or HOST_KV_PREFETCH_PREV_SCOPES[0]
    ranks, _ = _bounded_integer_environment(HOST_KV_PICK_RANKS_ENV, default=0, minimum=0,
                                            maximum=HOST_KV_PICK_RANKS_MAX)
    sparse = str(os.environ.get(HOST_KV_SPARSE_ENV) or "").strip().lower()
    if sparse not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_SPARSE_ENV}={sparse!r} is not one of '', '0', '1'.")
    hot_release = str(os.environ.get(HOST_KV_HOT_RELEASE_ENV) or "").strip().lower()
    if hot_release not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_HOT_RELEASE_ENV}={hot_release!r} is not one of '', '0', '1'.")
    hold = str(os.environ.get(HOST_KV_HOLD_CLEAN_ENV) or "").strip().lower()
    if hold not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_HOLD_CLEAN_ENV}={hold!r} is not one of '', '0', '1'.")
    hold_reserve = str(os.environ.get(HOST_KV_HOLD_RESERVE_ENV) or "").strip()
    if hold_reserve and not hold_reserve.isdigit():
        raise RuntimeError(f"{HOST_KV_HOLD_RESERVE_ENV}={hold_reserve!r} is not a block count.")
    if hold_reserve and hold != "1":
        raise RuntimeError(f"{HOST_KV_HOLD_RESERVE_ENV} sizes the reserve of {HOST_KV_HOLD_CLEAN_ENV}, which is off.")
    warm = str(os.environ.get(HOT_KEEP_WARM_ENV) or "").strip().lower()
    if warm not in HOT_KEEP_WARM_MODES:
        raise RuntimeError(f"{HOT_KEEP_WARM_ENV}={warm!r} is not one of {HOT_KEEP_WARM_MODES}.")
    if warm not in ("", "0"):
        keep = str(os.environ.get(HOT_KEEP_ENV) or "").strip().lower()
        if hot_release != "1" or keep != "1":
            raise RuntimeError(f"{HOT_KEEP_WARM_ENV}={warm} extends {HOT_KEEP_ENV}=1 and {HOST_KV_HOT_RELEASE_ENV}=1; "
                               f"this launch has {HOT_KEEP_ENV}={keep!r} and {HOST_KV_HOT_RELEASE_ENV}={hot_release!r}.")
        print(f"HOST_KV hot keep warm ({warm}): "
              + ("unscheduled running and waiting requests keep their hot region above the hot-keep reserve; "
                 if warm in ("1", "held") else "")
              + ("a hot block clears another lane only while that lane's hot row still lists it"
                 if warm in ("1", "owners") else ""), flush=True)
    fresh = str(os.environ.get("TAAF_VLLM_HOST_KV_FRESH_FIX") or "").strip().lower()
    stock = str(os.environ.get("TAAF_VLLM_HOST_KV_DRAFT_STOCK") or "").strip().lower()
    if fresh in ("1", "on", "true") and stock in ("0", "off", "false"):
        raise RuntimeError("TAAF_VLLM_HOST_KV_FRESH_FIX=1 mirrors the draft layer on the target's row mirrors and runs "
                           "only with the draft layer on its stock GPU cache; TAAF_VLLM_HOST_KV_DRAFT_STOCK=0 puts it on "
                           "host KV, where the draft-through write and the single fused resolve already keep it fresh. "
                           "Set TAAF_VLLM_HOST_KV_FRESH_FIX=0.")
    wide = str(os.environ.get(HOST_KV_INDEXER_WIDE_ENV) or "").strip().lower()
    if wide not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_INDEXER_WIDE_ENV}={wide!r} is not one of '', '0', '1'.")
    union = str(os.environ.get(HOST_KV_UNION_COUNT_ENV) or "").strip().lower()
    if union not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_UNION_COUNT_ENV}={union!r} is not one of '', '0', '1'.")
    plan_once = str(os.environ.get(HOST_KV_PLAN_ONCE_ENV) or "").strip().lower()
    if plan_once not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_PLAN_ONCE_ENV}={plan_once!r} is not one of '', '0', '1'.")
    captured = str(os.environ.get(HOST_KV_PLAN_ONCE_CAPTURED_ENV) or "").strip().lower()
    if captured not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_PLAN_ONCE_CAPTURED_ENV}={captured!r} is not one of '', '0', '1'.")
    if captured == "1" and plan_once != "1":
        raise RuntimeError(f"{HOST_KV_PLAN_ONCE_CAPTURED_ENV}=1 shares the plan inside a graph capture, which needs "
                           f"{HOST_KV_PLAN_ONCE_ENV}=1.")
    _bounded_integer_environment(HOST_KV_PLAN_ONCE_CHECKS_ENV, default=4, minimum=0, maximum=1000)
    take = str(os.environ.get(HOST_KV_TAKE_ROWS_ENV) or "").strip().lower()
    if take not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_TAKE_ROWS_ENV}={take!r} is not one of '', '0', '1'.")
    take_shadow = str(os.environ.get(HOST_KV_TAKE_SHADOW_ENV) or "").strip().lower()
    if take_shadow not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_TAKE_SHADOW_ENV}={take_shadow!r} is not one of '', '0', '1'.")
    if take_shadow == "1" and take != "1":
        raise RuntimeError(f"{HOST_KV_TAKE_SHADOW_ENV}=1 checks the fused takes of {HOST_KV_TAKE_ROWS_ENV}, which "
                           f"is not armed.")
    syncfree = str(os.environ.get(HOST_KV_SYNCFREE_ENV) or "").strip().lower()
    if syncfree not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_SYNCFREE_ENV}={syncfree!r} is not one of '', '0', '1'.")
    mark_fill = str(os.environ.get(HOST_KV_MARK_FILL_ENV) or "").strip().lower()
    if mark_fill not in ("", "0", "1"):
        raise RuntimeError(f"{HOST_KV_MARK_FILL_ENV}={mark_fill!r} is not one of '', '0', '1'.")
    if mark_fill == "1" and not (sparse == "1" and syncfree == "1"):
        raise RuntimeError(f"{HOST_KV_MARK_FILL_ENV} marks the rows of the sync-free sparse prefill stage, which "
                           f"needs {HOST_KV_SPARSE_ENV}=1 and {HOST_KV_SYNCFREE_ENV}=1.")
    stage_ahead = str(os.environ.get(QSA_STAGE_AHEAD_ENV) or "").strip().lower()
    if stage_ahead not in ("", "0", "1"):
        raise RuntimeError(f"{QSA_STAGE_AHEAD_ENV}={stage_ahead!r} is not one of '', '0', '1'.")
    ahead_programs = QSA_STAGE_AHEAD_PROGRAMS
    if str(os.environ.get(QSA_STAGE_AHEAD_PROGRAMS_ENV) or "").strip():
        if stage_ahead != "1":
            raise RuntimeError(f"{QSA_STAGE_AHEAD_PROGRAMS_ENV} sizes the copy grid of {QSA_STAGE_AHEAD_ENV}, "
                               f"which is off.")
        ahead_programs, _ = _bounded_integer_environment(QSA_STAGE_AHEAD_PROGRAMS_ENV, default=QSA_STAGE_AHEAD_PROGRAMS,
                                                         minimum=1, maximum=QSA_STAGE_AHEAD_PROGRAMS_MAX)
    if stage_ahead == "1":
        reuse = str(os.environ.get(QSA_STAGE_REUSE_ENV) or "").strip().lower()
        reuse_modes = (("reuse", "blocks") if reuse in ("1", "on", "true")
                       else tuple(part.strip() for part in reuse.split(",") if part.strip()))
        wanted = ((MIXED_FULL_ENV, str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower() == "1"),
                  (HOST_KV_PLAN_ONCE_ENV, plan_once == "1"), (HOST_KV_SPARSE_ENV, sparse == "1"),
                  (HOST_KV_SYNCFREE_ENV, syncfree == "1"), (HOST_KV_FUSED_ENV, fused == "1"),
                  (QSA_STAGE_REUSE_ENV, "reuse" in reuse_modes))
        missing = [name for name, held in wanted if not held]
        if missing:
            raise RuntimeError(f"{QSA_STAGE_AHEAD_ENV}=1 copies the settled rows of a bucketed mixed step's "
                               f"table-order stage plan, shared by every QSA layer and staged by the fused sparse "
                               f"stage; it needs {', '.join(f'{name}=1' for name in missing)}.")
        if str(os.environ.get(QSA_HALVES_ENV) or "").strip().lower() in ("1", "on", "true"):
            raise RuntimeError(f"{QSA_STAGE_AHEAD_ENV}=1 writes the stage buffer once the compute stream has run "
                               f"a layer's prefill attention, and {QSA_HALVES_ENV}=1 runs that attention on another "
                               f"stream; arm one of them.")
        print(f"QSA stage ahead: after a QSA layer's prefill attention in a bucketed mixed step, a side stream at "
              f"the device's least priority copies the next QSA layer's settled prefill rows from host into the stage buffer with "
              f"{ahead_programs} programs while the layers in between run; that layer's stage waits for the copy "
              f"and reads only the rows still missing", flush=True)
    picks, _ = _bounded_integer_environment(HOST_KV_PICK_LOG_ENV, default=0, minimum=0,
                                            maximum=HOST_KV_PICK_LOG_MAX)
    every, _ = _bounded_integer_environment(HOST_KV_PICK_EVERY_ENV, default=400, minimum=1, maximum=1000000)
    streaks = host_kv_pick_streaks(picks)
    if ranks and not streaks:
        raise RuntimeError(f"{HOST_KV_PICK_RANKS_ENV} adds index ranks to the pick-log streaks, which need "
                           f"{HOST_KV_PICK_AT_ENV}.")
    host_kv_memory_check(gib)
    if prefetch:
        print(f"HOST_KV prefetch: after each index layer's decode attention, the {prefetch} blocks ranked just below "
              f"every decode row's index cut are swapped into the coldest hot slots of their lane on a side stream, "
              f"never more than the region minus the step's union, and the next index layer waits for it; pure "
              f"decode steps only, the last index layer excluded; "
              f"{prefetch_programs or 'one per SM'} gather programs", flush=True)
    if prev == "1":
        print(f"HOST_KV prefetch prev: after each index layer's decode attention, the rows its picks name that the "
              f"next index layer's hot region does not hold"
              + (" and that this layer's own swap-in had to copy" if prev_scope == "missed" else "")
              + f" are swapped into that region on a side stream, at most {prev_rows} rows a lane, and the next index layer waits for it; pure decode steps only", flush=True)
    if prev == "1" and prev_once == "1":
        print("HOST_KV prefetch prev once: the prefetch plans in one Triton kernel a lane over buffers allocated "
              "once before capture and gathers without allocating, so no graph keeps its own plan temporaries",
              flush=True)
    if ranks:
        print(f"HOST_KV pick ranks: every streak record also holds each tracked decode row's index blocks in score "
              f"order to {ranks} past the cut, for tools/miss_origin.py", flush=True)
    if tail:
        print(f"HOST_KV tail prefetch: the newest {tail} tokens of each decode request are made hot for the next "
              "index layer on its copy stream, capped so the previous step's picks stay resident", flush=True)
    if picks:
        print(f"HOST_KV pick log: {picks} bursts of two decode steps, one every {every} steps, written as .npz "
              "under the pick-log folder for tools/prefetch_eval.py", flush=True)
    if streaks:
        print(f"HOST_KV pick streaks: at {','.join(f'{one:g}' for one in streaks['at'])} s of play, counted from the "
              f"first engine step with a decode lane, {streaks['steps']} consecutive decode steps leave the full "
              f"graphs and every target QSA layer records the picks and hot-region changes of {streaks['reqs']} "
              f"requests; at most {streaks['mib']} MiB, written as streak*.npz under the pick-log folder for "
              "tools/hot_policy_eval.py", flush=True)
    print("HOST_KV prefill staging: " + ("each prefill chunk stages only the rows its queries' index picks read"
          if sparse == "1" else "each prefill chunk stages every row of its requests' prefixes"), flush=True)
    print("HOST_KV decode: " + ("each verification row attends right after its own swap, so the hot cache floor "
          "is one row's index top-k" if row_attend == "1" else "every verification row is swapped in before "
          "attention reads them, so the hot cache floor is the decode query length times the index top-k"),
          flush=True)
    if hot_rows:
        print(f"HOST_KV hot rows: {hot_rows} hot rows a request in place of {HOST_KV_ROWS_ENV}; a lane whose "
              "verification rows' distinct picks overflow them attends a span of rows at a time, each swapped in "
              "right before its own attention, and the overflow is tallied on the GPU", flush=True)
    if copy_async == "1":
        print("HOST_KV copy async: host pool block copies run as GPU copies over the registered host pool on the "
              "step's stream instead of CPU copies before the forward", flush=True)
    if spill_skip == "1":
        print("HOST_KV spill skip: a spill pass resolves only the lanes whose verification union overflowed the hot "
              "region and spreads their host reads across many gather blocks; the other lanes are skipped inside "
              "the kernel, so a pass with no overflowing lane does no work and needs no sync", flush=True)
    if hot_split == "1":
        print("HOST_KV hot split: a lane whose verification union fits the hot region but not one resolve's shared "
              "memory is swapped in by consecutive resolves over disjoint column ranges and attended once; only a "
              "union larger than the region spills", flush=True)
    if mixed_capture == "1":
        print("HOST_KV mixed capture: a mixed step packs its prefill staging from the host copies of the prefill "
              "starts and sequence-length bounds the runner already holds, checked on the GPU, and the draft "
              "forward's last rows are clamped into each prefill part instead of masked, so the QSA layers of a "
              "mixed step read nothing back from the GPU", flush=True)
    if union_swap == "1":
        print("HOST_KV union swap: a lane's verification rows are deduplicated into one row per span that fits the "
              "resolve kernel's shared memory and swapped in together", flush=True)
    print("HOST_KV admission: " + ("pinned prompt pages go back to the pool chunk by chunk once a request's pinned "
          "pages pass its hot region" if admit == "real" else "vLLM keeps prompt pages pinned until the pool runs "
          "low"), flush=True)
    print("HOST_KV hot region: " + ("freed behind its step fence whenever the scheduler leaves a request out of a "
          "pass, and allocated cold when the request is scheduled again" if hot_release == "1" else "held by "
          "its request from the transition to host reads until the request is freed"), flush=True)
    if wide == "1":
        print("HOST_KV indexer group: compressed keys on blocks of twice the attention block, so each pool block "
              "it takes is filled twice as far", flush=True)
    if union == "1":
        print("HOST_KV union count: each decode step adds every lane's distinct picked rows across its verification "
              "rows to a GPU histogram, read and cleared every few hundred steps", flush=True)
    if plan_once == "1":
        print("HOST_KV plan once: a step's QSA metadata and prefill staging plan are built by its first QSA layer "
              "and reused by the others", flush=True)
    if syncfree == "1":
        print("HOST_KV sync-free staging: needed rows, resident copies and the sparse tallies stay on the GPU, "
              "the tallies read every few thousand forwards", flush=True)
    print(f"HOST_KV prefill stage: "
          + (f"{stage} MiB" if stage else "two longest requests at the model length")
          + " reserved once on the GPU before the memory profile; prefills stage in groups that fit it", flush=True)
    return {"host_pool_gib": gib, "device_buffer_size": rows or None, "stage_mib": stage or None,
            "admit": admit or "worst", "row_attend": row_attend == "1", "tail_prefetch": tail,
            "pick_log": picks, "pick_every": every, "pick_streaks": streaks, "sparse_stage": sparse == "1",
            "hot_release": hot_release == "1", "indexer_wide": wide == "1", "union_count": union == "1",
            "plan_once": plan_once == "1", "syncfree": syncfree == "1", "union_swap": union_swap == "1",
            "hot_rows": hot_rows, "copy_async": copy_async == "1", "spill_skip": spill_skip == "1",
            "spill_onesort": onesort == "1", "mark_fill": mark_fill == "1", "gather_early": early == "1",
            "fused": fused == "1", "hot_split": hot_split == "1", "host_group": group, "gather_spread": spread,
            "pcie_tally": pcie_tally, "run_merge": run_merge,
            "union_bitmap": fused_knobs[HOST_KV_UNION_BITMAP_ENV], "spill_flag": fused_knobs[HOST_KV_SPILL_FLAG_ENV],
            "spill_trust": fused_knobs[HOST_KV_SPILL_TRUST_ENV],
            "gather_tally": fused_knobs[HOST_KV_GATHER_TALLY_ENV], "last_slice": fused_knobs[HOST_KV_LAST_SLICE_ENV],
            "sources_fused": fused_knobs[HOST_KV_SOURCES_FUSED_ENV], "one_launch": fused_knobs[HOST_KV_ONE_LAUNCH_ENV],
            "one_pack": fused_knobs[HOST_KV_ONE_PACK_ENV], "one_hash": fused_knobs[HOST_KV_ONE_HASH_ENV],
            "flat_copy": fused_knobs[HOST_KV_FLAT_COPY_ENV], "flat_lines": fused_knobs[HOST_KV_FLAT_LINES_ENV],
            "one_cuda": fused_knobs[HOST_KV_ONE_CUDA_ENV],
            "mixed_capture": mixed_capture == "1", "prefetch_prev": prev_rows if prev == "1" else 0,
            "prefetch_prev_once": prev == "1" and prev_once == "1"}


def _cgroup_pinned_bytes(stat_path: Path) -> int | None:
    try:
        lines = stat_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    held = {}
    for line in lines:
        name, _, number = line.partition(" ")
        if name in ("anon", "shmem") and number.strip().isdigit():
            held[name] = int(number)
    return held["anon"] + held.get("shmem", 0) if "anon" in held else None


def host_kv_fit(asked: int, limit_path: Path = HOST_KV_CGROUP_MAX,
                stat_path: Path = HOST_KV_CGROUP_STAT) -> int:
    if not _flag_on(HOST_KV_FIT_ENV):
        return asked
    if asked in HOST_KV_FITTED:
        return HOST_KV_FITTED[asked]
    growth, _ = _bounded_integer_environment(HOST_KV_GROWTH_ENV, default=HOST_KV_GROWTH_DEFAULT_GIB,
                                             minimum=0, maximum=1024)
    try:
        said = limit_path.read_text(encoding="utf-8").strip()
    except OSError:
        said = ""
    pinned = _cgroup_pinned_bytes(stat_path)
    if not said.isdigit() or pinned is None:
        print(f"HOST_KV fit: the cgroup limit or its memory.stat is unreadable, so the pool stays at the asked "
              f"{asked} GiB", flush=True)
        HOST_KV_FITTED[asked] = asked
        return asked
    room = (int(said) - pinned) / 1024**3
    park = park_ram_gib()
    fit = int(room - host_kv_ple_gib() - HOST_KV_OTHER_ANON_GIB - growth - park)
    if fit < 1:
        raise RuntimeError(
            f"{HOST_KV_FIT_ENV}: {room:.1f} GiB of the cgroup is free of anon and shmem now, which leaves no host KV "
            f"pool beside ~{host_kv_ple_gib()} GiB of PLE, ~{HOST_KV_OTHER_ANON_GIB} GiB of process memory and "
            f"{growth} GiB of growth" + (f" and the {park} GiB park store" if park else "")
            + f"; lower {HOST_KV_GROWTH_ENV}.")
    fitted = min(asked, fit)
    print(f"HOST_KV fit: {room:.1f} GiB free of anon and shmem - {host_kv_ple_gib()} PLE - "
          f"{HOST_KV_OTHER_ANON_GIB} other - {growth} growth" + (f" - {park} park store" if park else "")
          + f" = {fit} GiB; pool {fitted} GiB (asked {asked})",
          flush=True)
    HOST_KV_FITTED[asked] = fitted
    return fitted


def host_memtotal_gib(meminfo: Path = HOST_MEMINFO) -> float | None:
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_kv_memory_check(gib: int, limit_path: Path = HOST_KV_CGROUP_MAX,
                         meminfo: Path = HOST_MEMINFO) -> dict[str, Any]:
    permille, _ = _bounded_integer_environment(GPU_MEMORY_PERMILLE_ENV, default=DEFAULT_GPU_MEMORY_PERMILLE,
                                               minimum=MIN_GPU_MEMORY_PERMILLE,
                                               maximum=MAX_GPU_MEMORY_PERMILLE)
    if permille > HOST_KV_MAX_GPU_PERMILLE:
        raise RuntimeError(
            f"{HOST_KV_ENV} copies resident prefill rows and builds staging plans on the GPU outside the memory "
            f"profile, so {GPU_MEMORY_PERMILLE_ENV}={permille} leaves no room for them; use at most "
            f"{HOST_KV_MAX_GPU_PERMILLE}.")
    park = park_ram_gib()
    need = gib + host_kv_ple_gib() + HOST_KV_OTHER_ANON_GIB + HOST_KV_HEADROOM_GIB + park
    try:
        said = limit_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {"limit_gib": None, "need_gib": need}
    if not said.isdigit():
        return {"limit_gib": None, "need_gib": need}
    cgroup = int(said) / 1024**3
    machine = host_memtotal_gib(meminfo)
    limit = min(cgroup, machine - HOST_SYSTEM_RESERVE_GIB) if machine else cgroup
    print(f"HOST_KV memory limit: cgroup {cgroup:.1f} GiB, machine MemTotal "
          f"{f'{machine:.1f} GiB less {HOST_SYSTEM_RESERVE_GIB} for the system' if machine else 'unreadable'}; the budget uses {limit:.1f} GiB", flush=True)
    if need > limit:
        raise RuntimeError(
            f"{HOST_KV_ENV} would pin {gib} GiB of host KV beside ~{host_kv_ple_gib()} GiB of PLE shared "
            f"memory and ~{HOST_KV_OTHER_ANON_GIB} GiB of process memory with {HOST_KV_HEADROOM_GIB} GiB "
            f"headroom" + (f" and the {park} GiB park store" if park else "")
            + f": {need} GiB against the cgroup's {limit:.1f} GiB; lower {HOST_KV_GIB_ENV}"
            + (f" or {PARK_RAM_ENV}" if park else "") + ".")
    print(f"HOST_KV memory: {gib} GiB pinned + {host_kv_ple_gib()} PLE + {HOST_KV_OTHER_ANON_GIB} other "
        f"+ {HOST_KV_HEADROOM_GIB} headroom" + (f" + {park} park store" if park else "")
        + f" = {need} of {limit:.1f} GiB; GPU permille {permille}", flush=True)
    return {"limit_gib": limit, "need_gib": need}


def patch_host_kv() -> dict[str, Any]:
    if not _flag_on(HOST_KV_ENV):
        host_kv_settings()
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{HOST_KV_ENV} edits vLLM 0.30.0's QSA owner and HiSparse layout; the "
                           f"{RUNTIME_PROFILE} runtime ships other ones.")
    if not _flag_on(QSA_KV8_ENV) or os.environ.get(KV_CACHE_DTYPE_ENV) not in QSA_QUANT_DTYPES:
        raise RuntimeError(f"{HOST_KV_ENV} keeps the quantized QSA rows in host memory and needs {QSA_KV8_ENV} "
                           f"armed with {KV_CACHE_DTYPE_ENV} one of {QSA_QUANT_DTYPES}.")
    settings = host_kv_settings()
    helper_path = BUNDLE_DIR / HOST_KV_HELPER
    table_path = BUNDLE_DIR / HOST_KV_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{HOST_KV_ENV} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str | None, tuple[tuple[str, str], ...]]] = namespace["TARGETS"]
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    if not (site / QSA_KV8_INSTALLED).is_file():
        raise RuntimeError(f"{HOST_KV_ENV} needs the QSA_KV8 helper {QSA_KV8_INSTALLED} installed first; it is "
                           f"absent from {site}")
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if pinned is None:
            if all(text.count(put) == 1 for _, put in edits):
                files[relative] = {"anchored": True, "patched_sha256": before, "already": True}
                continue
        elif _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{HOST_KV_ENV} cannot apply: {relative} is {before[:16]}, neither the {pinned[:16]} its "
                    f"edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{HOST_KV_ENV} cannot apply: an anchor of {relative} appears {text.count(found)} "
                    f"time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before if pinned is not None else None,
                           "anchored": pinned is None,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    installed = site / HOST_KV_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {HOST_KV_ENV} helper {HOST_KV_INSTALLED} did not land")
    fused_sha256 = None
    if settings["fused"]:
        fused_path = BUNDLE_DIR / HOST_KV_FUSED_HELPER
        if not fused_path.is_file():
            raise RuntimeError(f"{HOST_KV_FUSED_ENV} was set but {fused_path} is absent")
        kernels = fused_path.read_bytes()
        compile(kernels.decode("utf-8"), str(fused_path), "exec")
        fused_installed = site / HOST_KV_FUSED_INSTALLED
        if not (fused_installed.is_file() and fused_installed.read_bytes() == kernels):
            helper_fresh = True
            temporary = fused_installed.with_name(f".{fused_installed.name}.tmp")
            temporary.write_bytes(kernels)
            os.replace(temporary, fused_installed)
            if fused_installed.read_bytes() != kernels:
                raise RuntimeError(f"the {HOST_KV_FUSED_ENV} kernels {HOST_KV_FUSED_INSTALLED} did not land")
        for helper_name, installed_name in HOST_KV_ONE_CUDA_FILES:
            cuda_path = BUNDLE_DIR / helper_name
            if not cuda_path.is_file():
                if settings["one_cuda"]:
                    raise RuntimeError(f"{HOST_KV_ONE_CUDA_ENV} was set but {cuda_path} is absent")
                continue
            cuda_bytes = cuda_path.read_bytes()
            if helper_name.endswith(".py"):
                compile(cuda_bytes.decode("utf-8"), str(cuda_path), "exec")
            cuda_installed = site / installed_name
            if not (cuda_installed.is_file() and cuda_installed.read_bytes() == cuda_bytes):
                helper_fresh = True
                temporary = cuda_installed.with_name(f".{cuda_installed.name}.tmp")
                temporary.write_bytes(cuda_bytes)
                os.replace(temporary, cuda_installed)
                if cuda_installed.read_bytes() != cuda_bytes:
                    raise RuntimeError(f"the {HOST_KV_ONE_CUDA_ENV} file {installed_name} did not land")
        fused_sha256 = hashlib.sha256(kernels).hexdigest()
        print(f"HOST_KV fused: kernels at {HOST_KV_FUSED_INSTALLED} {fused_sha256[:16]}; the capped union swap, its "
              "spill passes, the fresh-miss count and the sparse prefill stage launch as Triton kernels", flush=True)
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.hostkv.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {HOST_KV_ENV} edit of {relative} did not land")
    row = {"armed": True, "already": not made and not helper_fresh, **settings,
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(),
           "fused_sha256": fused_sha256,
           "files": files}
    print(f"HOST_KV {'already armed' if row['already'] else 'armed'}: {len(made)} of {len(targets)} vLLM file(s) "
          f"edited from table {row['table_sha256'][:16]}, helper at {HOST_KV_INSTALLED} "
          f"{row['helper_sha256'][:16]}; the QSA main KV is a host-resident sparse source in a "
          f"{settings['host_pool_gib']} GiB pinned pool behind vLLM's HiSparse, attention reads a GPU hot cache of "
          f"{settings['device_buffer_size'] or 'the default'} rows a request, and the CPU offload tier is "
          f"replaced by that pool", flush=True)
    return row


def tail_soft_reserve() -> int:
    said = str(os.environ.get(TAIL_SOFT_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return 0
    if said in ("1", "true", "yes", "on"):
        return TAIL_SOFT_RESERVE
    if said == "hard":
        return TAIL_SOFT_HARD
    if not said.isdigit():
        raise RuntimeError(f"{TAIL_SOFT_ENV} must be 1, hard or a reserve of free blocks, found {said!r}")
    return int(said)


def patch_slim_state() -> dict[str, Any]:
    ride = _flag_on(PLE_RIDE_ENV)
    soft = _flag_on(KEEP_SOFT_ENV)
    tail = tail_soft_reserve()
    report = _flag_on(SLIM_REPORT_ENV)
    wait = _flag_on(WAIT_KEEP_ENV)
    order = _flag_on(WAIT_ORDER_ENV)
    if not ride and not soft and not tail and not report and not wait and not order:
        return {"armed": False}
    named = " and ".join(name for name, on in ((PLE_RIDE_ENV, ride), (KEEP_SOFT_ENV, soft), (TAIL_SOFT_ENV, tail),
                                               (SLIM_REPORT_ENV, report), (WAIT_KEEP_ENV, wait),
                                               (WAIT_ORDER_ENV, order)) if on)
    said_age = str(os.environ.get(WAIT_ORDER_AGE_ENV) or "").strip()
    try:
        wait_age = float(said_age) if said_age else WAIT_ORDER_AGE
    except ValueError:
        wait_age = -1.0
    if not wait_age >= 0:
        raise RuntimeError(f"{WAIT_ORDER_AGE_ENV} must be zero or a positive number of tokens a second, "
                           f"found {said_age!r}")
    said_wait = str(os.environ.get(WAIT_KEEP_S_ENV) or "").strip()
    try:
        wait_every = float(said_wait) if said_wait else WAIT_KEEP_S
    except ValueError:
        wait_every = -1.0
    if not wait_every > 0:
        raise RuntimeError(f"{WAIT_KEEP_S_ENV} must be a positive number of seconds, found {said_wait!r}")
    said = str(os.environ.get(SLIM_REPORT_S_ENV) or "").strip()
    try:
        every = float(said) if said else SLIM_REPORT_S
    except ValueError:
        every = -1.0
    if not every > 0:
        raise RuntimeError(f"{SLIM_REPORT_S_ENV} must be a positive number of seconds, found {said!r}")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{named} edits vLLM 0.30.0's KV cache spec and scheduler; the {RUNTIME_PROFILE} "
            f"runtime ships other ones.")
    if tail and not _flag_on(CACHE_KEEP_ENV):
        raise RuntimeError(
            f"{TAIL_SOFT_ENV} releases CACHE_KEEP's retained Mamba snapshots ahead of reused resident pages; "
            f"{CACHE_KEEP_ENV} is off, so no snapshot is retained.")
    if soft and not _flag_on(CACHE_KEEP_ENV):
        raise RuntimeError(
            f"{KEEP_SOFT_ENV} hands CACHE_KEEP's pinned Mamba snapshots to the evictable cache when "
            f"the pool runs short; {CACHE_KEEP_ENV} is off, so no snapshot is pinned.")
    helper_path = BUNDLE_DIR / SLIM_HELPER
    table_path = BUNDLE_DIR / SLIM_TABLE
    for path in (helper_path, table_path):
        if not path.is_file():
            raise RuntimeError(f"{named} was set but {path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    table = table_path.read_bytes()
    namespace: dict[str, Any] = {}
    exec(compile(table.decode("utf-8"), str(table_path), "exec"), namespace)
    targets: dict[str, tuple[str | None, tuple[tuple[str, str], ...]]] = {
        relative: held for relative, held in namespace["TARGETS"].items()
        if ride or relative not in namespace["RIDE"]}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if pinned is None:
            if all(text.count(put) == 1 for _, put in edits):
                files[relative] = {"anchored": True, "patched_sha256": before, "already": True}
                continue
        elif _foreign_free_sha256(text, edits, pinned) != pinned:
            if _undone_sha256(text, edits, pinned) != pinned:
                raise RuntimeError(
                    f"{named} cannot apply: {relative} is {before[:16]}, neither the {pinned[:16]} "
                    f"its edits were written against nor that file with them applied.")
            files[relative] = {"stock_sha256": pinned, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if pinned is None and text.count(put) == 1:
                continue
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{named} cannot apply: an anchor of {relative} appears {text.count(found)} "
                    f"time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before if pinned is not None else None,
                           "anchored": pinned is None,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    chosen = json.dumps({"keep_soft": soft, "tail_soft": tail, "ple_ride": ride, "report_s": every, "wait_keep": wait,
                         "wait_keep_s": wait_every, "wait_order": order, "wait_order_age": wait_age},
                        sort_keys=True).encode("utf-8")
    fresh = []
    for relative, data in ((SLIM_INSTALLED, helper), (SLIM_SETTINGS, chosen)):
        path = site / relative
        if path.is_file() and path.read_bytes() == data:
            continue
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"{relative} did not land")
        fresh.append(relative)
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.slim.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {named} edit of {relative} did not land")
    row = {"armed": True, "already": not made and not fresh, "ple_ride": ride, "keep_soft": soft, "tail_soft": tail,
           "report_s": every, "wait_keep": wait, "wait_keep_s": wait_every, "wait_order": order,
           "wait_order_age": wait_age,
           "table_sha256": hashlib.sha256(table).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}
    print(f"SLIM_STATE {'already armed' if row['already'] else 'armed'} ({named}): {len(made)} of "
          f"{len(targets)} vLLM file(s) edited from table {row['table_sha256'][:16]}, helper at "
          f"{SLIM_INSTALLED} {row['helper_sha256'][:16]}; "
          + "; ".join(f"{relative} "
                      + (f"{held['stock_sha256'][:16]}" if held.get("stock_sha256") else "anchored")
                      + f" -> {held['patched_sha256'][:16]}"
                      for relative, held in files.items())
          + ("; the PLE short-conv state joins a GDN cache group's block instead of holding a "
             "group of its own (MambaSpec stops splitting groups on tp_replicated, which only the "
             "NIXL connector reads, and this serve runs one rank), so a running request holds one "
             "Mamba state block fewer and, under CACHE_KEEP, one pinned snapshot fewer"
             if ride else "")
          + ("; CACHE_KEEP's pinned snapshots stay pinned while the pool has room and go to the "
             "evictable cache, newest-admitted request first, once fewer than a sixteenth of the "
             "pool plus one block a Mamba group (plus what the oldest waiting request may need) "
             "are free" if soft else "")
          + (f"; CACHE_KEEP's retained snapshots go to the front of the free queue, still cached, whenever "
             f"fewer than {tail} blocks are free besides unpinned resident pages, so the pool reuses them "
             f"before any resident page" if tail else "")
          + (f"; a waiting request's cached prefix goes to the back of the eviction queue on arrival "
             f"and every {wait_every:g} s while it waits, so a turn queued behind other prefills "
             f"still finds the blocks it resumes from" if wait else "")
          + (f"; the waiting queue is served fewest uncached prompt tokens first, the count falling "
             f"by {wait_age:g} tokens a second of waiting" if order else "")
          + f"; the scheduler prints a SLIM_STATE line with the state blocks its running requests "
          f"hold every {every:g} s", flush=True)
    return row


def patch_ple_layer() -> dict[str, Any]:
    if RUNTIME_PROFILE != "day0":
        print(f"PLE_PATCH skipped: vLLM {VLLM_VERSION} picks its FP8 n-gram loader from "
              f"the checkpoint's ple_embedding_dtype", flush=True)
        return {"skipped": True, "runtime_profile": RUNTIME_PROFILE}
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / "vllm" / "models" / "qwen3_8_flash_next" / "nvidia" / "ple_layer.py"
    patch_dir = BUNDLE_DIR / PLE_PATCH_DIRNAME
    expected_files = {
        PLE_PATCH_APPLICATOR: (PLE_PATCH_APPLICATOR_BYTES, PLE_PATCH_APPLICATOR_SHA256),
        PLE_PATCH_IDENTITY: (PLE_PATCH_IDENTITY_BYTES, PLE_PATCH_IDENTITY_SHA256),
        PLE_PATCH_DIFF: (PLE_PATCH_DIFF_BYTES, PLE_PATCH_DIFF_SHA256),
        PLE_PATCH_README: (PLE_PATCH_README_BYTES, PLE_PATCH_README_SHA256),
    }
    if not patch_dir.is_dir() or patch_dir.is_symlink():
        raise RuntimeError(f"Missing regular PLE patch directory: {patch_dir}")
    actual_names = {entry.name for entry in patch_dir.iterdir()}
    if actual_names != set(expected_files):
        raise RuntimeError(
            "PLE patch bundle allowlist mismatch: "
            f"actual={sorted(actual_names)} expected={sorted(expected_files)}"
        )
    artifact_rows: dict[str, dict[str, Any]] = {}
    for name, (expected_size, expected_sha) in expected_files.items():
        artifact = patch_dir / name
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"PLE patch payload is not a regular file: {artifact}")
        actual_size = artifact.stat().st_size
        actual_sha = sha256_file(artifact)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(
                f"PLE patch payload mismatch for {name}: "
                f"size={actual_size}/{expected_size} sha256={actual_sha}/{expected_sha}"
            )
        artifact_rows[name] = {"bytes": actual_size, "sha256": actual_sha}

    patch_identity = read_json(patch_dir / PLE_PATCH_IDENTITY)
    required_identity = {
        "artifact": PLE_PATCH_DIFF,
        "artifact_bytes": PLE_PATCH_DIFF_BYTES,
        "artifact_sha256": PLE_PATCH_DIFF_SHA256,
        "exact_model_config": "model/RadixArk-Qwen3.8-Flash-Next-NVFP4/config.json",
        "exact_model_config_bytes": 4_983,
        "exact_model_config_producer": {"name": "modelopt", "version": "0.46.0"},
        "exact_model_config_sha256": MODEL_CONFIG_SHA256,
        "modelopt_internal_gate": {
            "checkpoint_excluded_prefix": PLE_EXCLUDED_PREFIX,
            "group_size": 16,
            "is_checkpoint_nvfp4_serialized": True,
            "is_layer_excluded_checkpoint_prefix": True,
            "quant_method": "NVFP4",
            "runtime_prefix": PLE_EXACT_PREFIX,
        },
        "patched_target_sha256": VLLM_PLE_PATCHED_SHA256,
        "stock_target_sha256": VLLM_PLE_STOCK_SHA256,
        "target": "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py",
        "vllm_image_amd64_manifest": VLLM_AMD64_MANIFEST_DIGEST,
        "vllm_image_index_digest": VLLM_INDEX_DIGEST,
        "vllm_version": VLLM_VERSION,
    }
    if patch_identity != required_identity:
        raise RuntimeError("PATCH_IDENTITY.json did not match the exact approved patch identity.")

    before_sha = sha256_file(path)
    if before_sha != VLLM_PLE_STOCK_SHA256:
        raise RuntimeError(f"Stock vLLM PLE source hash mismatch: {before_sha}")
    completed = subprocess.run(
        [sys.executable, "-B", str(patch_dir / PLE_PATCH_APPLICATOR), str(site)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "patched":
        raise RuntimeError(
            "Pinned PLE patch applicator failed: "
            f"returncode={completed.returncode} stdout={completed.stdout!r} "
            f"stderr={completed.stderr!r}"
        )
    after_sha = sha256_file(path)
    if after_sha != VLLM_PLE_PATCHED_SHA256:
        raise RuntimeError(f"Written vLLM PLE patch hash mismatch: {after_sha}")
    cache_artifacts = [
        child
        for root in (patch_dir, site / "vllm" / "models" / "qwen3_8_flash_next")
        for child in root.rglob("*")
        if child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}
    ]
    if cache_artifacts:
        raise RuntimeError(f"PLE patching produced forbidden Python cache files: {cache_artifacts}")
    return {
        "path": str(path),
        "stock_sha256": before_sha,
        "patched_sha256": after_sha,
        "artifact_directory": str(patch_dir),
        "artifacts": artifact_rows,
        "patch_identity": patch_identity,
        "applicator_result": completed.stdout.strip(),
        "gate_environment": "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1",
        "gate_model_config_sha256": MODEL_CONFIG_SHA256,
        "gate_runtime_prefix": PLE_EXACT_PREFIX,
        "gate_checkpoint_excluded_prefix": PLE_EXCLUDED_PREFIX,
    }


WEIGHT_CACHE_IDENTITY_FILE = "vllm-weight-cache-daemon.identity.json"
WEIGHT_CACHE_STOP_S = 30.0
WEIGHT_CACHE_TAAF_PREFIX = "TAAF_VLLM_"
WEIGHT_CACHE_SAFE_ENV_NAMES = frozenset({
    KV_CACHE_DTYPE_ENV, PREFIX_RETENTION_ENV, CACHE_KEEP_ENV, CACHE_KEEP_END_ENV, CACHE_KEEP_END_RAM_ENV,
    CACHE_KEEP_MATCH_UNIT_ENV, MAMBA_JUNCTION_ENV, KEEP_SOFT_ENV, TAIL_SOFT_ENV, MAX_NUM_SEQS_ENV, MAX_CUDAGRAPH_CAPTURE_SIZE_ENV,
    KV_DISK_GIB_ENV, KV_DISK_DIR_ENV, KV_DISK_PAGE_CACHE_ENV,
    PIECEWISE_LARGE_ENV, PIECEWISE_LARGE_STEP_ENV, CAPTURE_SPARSE_ENV, MIXED_FULL_ENV, MIXED_FULL_SEQS_ENV, MIXED_FULL_LARGEST_ENV, PDL_ENV,
    MAX_NUM_BATCHED_TOKENS_ENV, GPU_MEMORY_PERMILLE_ENV, MTP_TOKENS_ENV,
    MTP_DYNAMIC_BATCH_SCHEDULE_ENV, ADAPTIVE_VERIFY_ENV, DRAFT_BUDGET_ENV, MOE_ROUTE_ENV, MOE_ROUTE_FILE_ENV,
    MOE_SHARE_ENV, MOE_SHARE_SETTINGS_ENV, ITERATION_LOG_ENV, PROFILER_DIR_ENV,
    PROFILER_STEPS_ENV, SLIM_REPORT_ENV, SLIM_REPORT_S_ENV, WAIT_KEEP_ENV, WAIT_KEEP_S_ENV,
    WAIT_ORDER_ENV, WAIT_ORDER_AGE_ENV,
    PREFILL_HOLD_ALIGNED_ENV, PREFILL_HOLD_FILL_SHARE_ENV, PREFILL_HOLD_WAIT_WEIGHT_ENV,
    "TAAF_VLLM_TRIM_HOLE_SYNC_TEXT", STABLE_THINK_ENV,
    DRAFT_CONF_LOG_ENV, RELAX_ACCEPT_ENV, BLOCK_VERIFY_ENV, ACCEPT_LAMBDA_ENV, ACCEPT_LAMBDA_GUARD_ENV,
    ACCEPT_LAMBDA_TOOL_IDS_ENV,
    INDEXER_KV_DTYPE_ENV, MAMBA_SSM_DTYPE_ENV, "TAAF_VLLM_QSA_KV8_TUNE_FILE",
    "ARC_SERVE_STATS_S", "VLLM_SERVER_DEV_MODE",
})
WEIGHT_CACHE_OTHER_TRACKED_ENV_NAMES = frozenset({
    "VLLM_GDN_DECODE_KERNEL", "VLLM_USE_BREAKABLE_CUDAGRAPH", "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_TUNING", "PYTORCH_TUNABLEOP_FILENAME",
})
WEIGHT_CACHE_HASHED_SITE_DIRS = ("vllm/model_executor", "vllm/models", "vllm/v1/worker")


def weight_cache_tracked_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in env.items():
        if name in WEIGHT_CACHE_SAFE_ENV_NAMES:
            continue
        if name.startswith(WEIGHT_CACHE_TAAF_PREFIX) or name in WEIGHT_CACHE_OTHER_TRACKED_ENV_NAMES:
            out[name] = value
    return out


def installed_files_digest(site: Path) -> str:
    paths: set[Path] = set()
    for relative_root in WEIGHT_CACHE_HASHED_SITE_DIRS:
        root = site / relative_root
        if root.is_dir():
            paths.update(path for path in root.rglob("*")
                        if path.suffix in (".py", ".json") and "__pycache__" not in path.parts
                        and path.is_file())
    vllm_root = site / "vllm"
    if vllm_root.is_dir():
        paths.update(path for path in vllm_root.glob("taaf_*.py") if path.is_file())
        paths.update(path for path in vllm_root.glob("taaf_*.json") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda one: one.relative_to(site).as_posix()):
        digest.update(path.relative_to(site).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def weight_cache_identity(env: dict[str, str], site: Path) -> dict[str, Any]:
    tracked = weight_cache_tracked_env(env)
    files_digest = installed_files_digest(site)
    payload = json.dumps({"env": tracked, "files": files_digest}, sort_keys=True)
    return {"hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "env": tracked,
            "files_digest": files_digest}


def weight_cache_identity_diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    moves = []
    old_env, new_env = old.get("env") or {}, new.get("env") or {}
    for name in sorted(set(old_env) | set(new_env)):
        if old_env.get(name) != new_env.get(name):
            moves.append(f"{name}: {old_env.get(name)!r} -> {new_env.get(name)!r}")
    if old.get("files_digest") != new.get("files_digest"):
        moves.append(f"installed files: {str(old.get('files_digest') or '')[:16]} -> "
                     f"{str(new.get('files_digest') or '')[:16]}")
    return moves


def weight_cache_identity_file() -> Path:
    return WORKING_DIR / WEIGHT_CACHE_IDENTITY_FILE


def _read_weight_cache_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return held if isinstance(held, dict) else {}


def _stop_weight_cache_daemon(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.monotonic() + WEIGHT_CACHE_STOP_S
    while time.monotonic() < deadline and _daemon_alive(pid):
        time.sleep(WEIGHT_CACHE_POLL_S)
    if _daemon_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass



def patch_paused() -> dict[str, Any]:
    resume = _flag_on(PAUSED_RESUME_ENV)
    if resume and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_RESUME_ENV} resumes a paused session after a length stop, so it needs "
                           f"{PAUSED_ENV}")
    memory = _flag_on(PAUSED_MEMORY_ENV)
    if memory and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_MEMORY_ENV} narrows the PAUSED pressure rule, so it needs {PAUSED_ENV}")
    if _flag_on(PAUSED_BRANCH_ENV) and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_BRANCH_ENV} routes branch requests around the paused sessions, so it needs "
                           f"{PAUSED_ENV}")
    if _flag_on(PAUSED_VIEW_ENV) and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_VIEW_ENV} matches a turn against the paused session's client view, so it "
                           f"needs {PAUSED_ENV}")
    if _flag_on(PAUSED_MM_DROP_ENV) and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_MM_DROP_ENV} lets go the image data of a paused session's computed tokens in "
                           f"the PAUSED scheduler edit, so it needs {PAUSED_ENV}")
    if _flag_on(PAUSED_MEMTRACE_ENV) and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_MEMTRACE_ENV} logs from the PAUSED helper in the API server and the engine "
                           f"core, so it needs {PAUSED_ENV}")
    pool = _flag_on(PAUSED_POOL_ENV)
    if pool and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PAUSED_POOL_ENV} publishes the KV pool reading from the PAUSED scheduler edit, so it "
                           f"needs {PAUSED_ENV}")
    fork = _flag_on(FORK_BRANCH_ENV)
    if fork and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{FORK_BRANCH_ENV} branches a request off a paused session's KV, so it needs "
                           f"{PAUSED_ENV}")
    fork_most = _bounded_integer_environment(FORK_MAX_ENV, default=FORK_MAX_DEFAULT, minimum=1,
                                             maximum=FORK_MAX_LIMIT)[0] if fork else 0
    hold_said = str(os.environ.get(FORK_HOLD_ENV) or "").strip()
    try:
        fork_hold = float(hold_said) if hold_said else 20.0
    except ValueError:
        raise RuntimeError(f"{FORK_HOLD_ENV}={hold_said!r} is not a number of seconds.") from None
    if not 0.0 <= fork_hold <= 3600.0:
        raise RuntimeError(f"{FORK_HOLD_ENV}={hold_said!r} is outside 0..3600 seconds.")
    if os.environ.get(FORK_SEATS_ENV) is not None:
        _bounded_integer_environment(FORK_SEATS_ENV, default=0, minimum=0, maximum=FORK_MAX_LIMIT)
        if not fork:
            raise RuntimeError(f"{FORK_SEATS_ENV} keeps runner slots for fork branches, so it needs {FORK_BRANCH_ENV}")
    if _flag_on(FORK_LIVE_ENV) and not fork:
        raise RuntimeError(f"{FORK_LIVE_ENV} hands a branch the paused main's live recurrent state, so it needs "
                           f"{FORK_BRANCH_ENV}")
    rewind = _flag_on(REWIND_ENV)
    if rewind and not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{REWIND_ENV} rewinds a paused session to its backup, so it needs {PAUSED_ENV}")
    if not _flag_on(PAUSED_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{PAUSED_ENV} edits vLLM 0.30.0's AsyncLLM, scheduler and V2 rejection sampler; the "
                           f"{RUNTIME_PROFILE} runtime ships other ones.")
    helper_path = BUNDLE_DIR / PAUSED_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{PAUSED_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in PAUSED_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular {PAUSED_ENV} source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if all(text.count(put) == 1 for _, put in edits):
            files[relative] = {"anchored": True, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{PAUSED_ENV} cannot apply: an anchor of {relative} appears {text.count(found)} time(s), "
                    f"not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"anchored": True, "before_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(),
                           "already": False}
    if fork:
        path = site / FORK_SCHEDULER_RELATIVE
        text = (made.get(FORK_SCHEDULER_RELATIVE) or path.read_bytes()).decode("utf-8")
        if not all(text.count(put) == 1 for _, put in FORK_SCHEDULER_EDITS):
            for found, put in FORK_SCHEDULER_EDITS:
                if text.count(found) != 1:
                    raise RuntimeError(
                        f"{FORK_BRANCH_ENV} cannot apply: an anchor of {FORK_SCHEDULER_RELATIVE} appears "
                        f"{text.count(found)} time(s), not once.")
                text = text.replace(found, put, 1)
            compile(text, str(path), "exec")
            made[FORK_SCHEDULER_RELATIVE] = text.encode("utf-8")
            files[FORK_SCHEDULER_RELATIVE] = dict(files.get(FORK_SCHEDULER_RELATIVE) or {}, already=False,
                                                  patched_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    if rewind:
        path = site / FORK_SCHEDULER_RELATIVE
        text = (made.get(FORK_SCHEDULER_RELATIVE) or path.read_bytes()).decode("utf-8")
        if not all(text.count(put) == 1 for _, put in REWIND_SCHEDULER_EDITS):
            for found, put in REWIND_SCHEDULER_EDITS:
                if text.count(found) != 1:
                    raise RuntimeError(
                        f"{REWIND_ENV} cannot apply: an anchor of {FORK_SCHEDULER_RELATIVE} appears "
                        f"{text.count(found)} time(s), not once.")
                text = text.replace(found, put, 1)
            compile(text, str(path), "exec")
            made[FORK_SCHEDULER_RELATIVE] = text.encode("utf-8")
            files[FORK_SCHEDULER_RELATIVE] = dict(files.get(FORK_SCHEDULER_RELATIVE) or {}, already=False,
                                                  patched_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    installed = site / PAUSED_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {PAUSED_ENV} helper {PAUSED_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.paused.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {PAUSED_ENV} edit of {relative} did not land")
    row = {"armed": True, "already": not made and not helper_fresh,
           "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files,
           "fork": fork, "fork_max": fork_most}
    if rewind:
        row["rewind"] = True
    print(f"PAUSED {'already armed' if row['already'] else 'armed'}: each conversation is one resumable "
          f"request; a turn that extends the paused token sequence feeds only its new tokens, the "
          f"scheduler re-feeds the uncomputed last sampled token, and draft acceptance stops at a stop "
          f"token; helper at {PAUSED_INSTALLED} {row['helper_sha256'][:16]}", flush=True)
    print(f"PAUSED length stops: {'resume (acceptance clipped at each request own max_tokens)' if resume else 'close'}",
          flush=True)
    print("PAUSED pressure queue: " + ("requests the scheduler could not allocate memory for in the last interval"
          if memory else "every waiting request that is not a paused session"), flush=True)
    print("PAUSED memory trace: " + ("MEMTRACE lines every TAAF_VLLM_MEMTRACE_S seconds in the API server and the "
          "engine core: rss, the largest allocation-site growths and the per-request container sizes"
          if _flag_on(PAUSED_MEMTRACE_ENV) else "off"), flush=True)
    print("PAUSED image data: " + ("a session's processed image data is let go once its tokens are computed; a "
          "recompute that needs it again ends that request with an error" if _flag_on(PAUSED_MM_DROP_ENV)
          else "kept for the session's life (a recompute from scratch can re-encode)"), flush=True)
    print("PAUSED pool reading: " + ("the scheduler writes the host and GPU KV pools' used and total blocks once a "
          "second, and /metrics carries them as taaf_pool:* beside taaf_paused:*" if pool else "off"), flush=True)
    print("PAUSED fork branches: " + (f"a taaf-branch-<key>#<kind> request whose tokens extend a live paused session shares "
          f"its KV blocks, its recurrent-state checkpoint and its resident host-KV pages and computes only its own "
          f"tail; branches wait behind waiting main requests for at most {fork_hold:g} s ({FORK_HOLD_ENV}), at most "
          f"{fork_most} at once ({FORK_MAX_ENV})"
          if fork else "off"), flush=True)
    if fork and _flag_on(FORK_LIVE_ENV):
        print(f"PAUSED fork live state ({FORK_LIVE_ENV}): a branch that shares every token its paused main has "
              f"computed takes a private copy of the main's live recurrent state, with no checkpoint needed",
              flush=True)
    if rewind:
        print(f"PAUSED rewind ({REWIND_ENV}): a turn marked taaf_rewind backs up its key's paused session before it "
              f"runs, and the key's next turn that extends the backup rewinds the live session to it in place of a "
              f"stale restart", flush=True)
    return row


def park_ram_gib() -> int:
    if not _flag_on(PARK_ENV):
        return 0
    if str(os.environ.get(PARK_DISK_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off"):
        return 0
    gib, _ = _bounded_integer_environment(PARK_RAM_ENV, default=PARK_DEFAULT_RAM_GIB, minimum=1,
                                          maximum=PARK_MAX_RAM_GIB)
    return gib


def park_memory_check(limit_path: Path | None = None) -> dict[str, Any]:
    limit_path = limit_path or HOST_KV_CGROUP_MAX
    park = park_ram_gib()
    if _flag_on(HOST_KV_ENV):
        return {"park_gib": park, "counted_by": "host_kv_memory_check"}
    offload, _ = _bounded_integer_environment(KV_OFFLOADING_GIB_ENV, default=0, minimum=0,
                                              maximum=MAX_KV_OFFLOADING_GIB)
    need = park + offload + host_kv_ple_gib() + HOST_KV_OTHER_ANON_GIB + HOST_KV_HEADROOM_GIB
    try:
        said = limit_path.read_text(encoding="utf-8").strip()
    except OSError:
        said = ""
    if not said.isdigit():
        print(f"PARK memory: {park} GiB store beside a {offload} GiB offload tier; the cgroup limit is unreadable, "
              f"so nothing was checked", flush=True)
        return {"park_gib": park, "offload_gib": offload, "need_gib": need, "limit_gib": None}
    limit = int(said) / 1024**3
    if need > limit:
        raise RuntimeError(
            f"{PARK_ENV} would pin a {park} GiB store beside the {offload} GiB offload tier, ~{host_kv_ple_gib()} GiB "
            f"of PLE, ~{HOST_KV_OTHER_ANON_GIB} GiB of process memory and {HOST_KV_HEADROOM_GIB} GiB headroom: "
            f"{need} GiB against the cgroup's {limit:.1f} GiB; lower {PARK_RAM_ENV} or {KV_OFFLOADING_GIB_ENV}.")
    print(f"PARK memory: {park} GiB store + {offload} offload tier + {host_kv_ple_gib()} PLE + "
          f"{HOST_KV_OTHER_ANON_GIB} other + {HOST_KV_HEADROOM_GIB} headroom = {need} of {limit:.1f} GiB", flush=True)
    return {"park_gib": park, "offload_gib": offload, "need_gib": need, "limit_gib": limit}


def patch_park() -> dict[str, Any]:
    if not _flag_on(PARK_ENV):
        for name in (PARK_RAM_ENV, PARK_HOLD_ENV, PARK_WAIT_ENV, PARK_VERIFY_ENV, PARK_HOST_DISK_ENV):
            if str(os.environ.get(name) or "").strip():
                raise RuntimeError(f"{name} shapes parking and needs {PARK_ENV} armed.")
        return {"armed": False}
    if not _flag_on(PAUSED_ENV):
        raise RuntimeError(f"{PARK_ENV} parks PAUSED sessions and resumes them by their key, so it needs {PAUSED_ENV}")
    if not _flag_on(ENABLE_PREFIX_CACHING_ENV):
        raise RuntimeError(f"{PARK_ENV} brings a parked session back through the prefix cache, so it needs "
                           f"{ENABLE_PREFIX_CACHING_ENV}")
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{PARK_ENV} edits vLLM 0.30.0's scheduler and V2 model runner; the {RUNTIME_PROFILE} "
                           f"runtime ships other ones.")
    gib = park_ram_gib()
    hold, _ = _bounded_integer_environment(PARK_HOLD_ENV, default=PARK_DEFAULT_HOLD_S, minimum=1,
                                           maximum=PARK_MAX_HOLD_S)
    wait, _ = _bounded_integer_environment(PARK_WAIT_ENV, default=PARK_DEFAULT_WAIT_S, minimum=0,
                                           maximum=PARK_MAX_WAIT_S)
    memory = park_memory_check()
    helper_path = BUNDLE_DIR / PARK_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{PARK_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = _runtime_site()
    made: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative, (pinned, edits) in PARK_TARGETS.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular {PARK_ENV} source: {path}")
        held = path.read_bytes()
        before = hashlib.sha256(held).hexdigest()
        text = held.decode("utf-8")
        if all(text.count(put) == 1 for _, put in edits):
            files[relative] = {"anchored": True, "patched_sha256": before, "already": True}
            continue
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(f"{PARK_ENV} cannot apply: an anchor of {relative} appears {text.count(found)} "
                                   f"time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"anchored": True, "before_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest(), "already": False}
    installed = site / PARK_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {PARK_ENV} helper {PARK_INSTALLED} did not land")
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.park.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {PARK_ENV} edit of {relative} did not land")
    verify = _flag_on(PARK_VERIFY_ENV)
    row = {"armed": True, "already": not made and not helper_fresh, "ram_gib": gib, "hold_s": hold, "wait_s": wait,
           "verify": verify, "memory": memory, "helper_sha256": hashlib.sha256(helper).hexdigest(), "files": files}
    if _flag_on(PARK_HOST_DISK_ENV):
        row["host_disk"] = True
    print(f"PARK {'already armed' if row['already'] else 'armed'}: a request marked taaf_park copies its key's "
          f"paused session's prefix-cached GPU blocks at the newest registered boundary into a {gib} GiB pinned RAM "
          f"store, pins its host pages and closes it; the key's next turn copies back what the pool evicted, "
          f"re-registers it under the same hashes and holds it until admitted (at most {hold} s, waiting up to "
          f"{wait} s for room)" + ("; every copied block is checksummed on the card and compared after it comes back"
                                  if verify else "")
          + ("; the session's host-pool pages go to a file on local disk and back to the host pool"
             if row.get("host_disk") else "") + f"; helper at {PARK_INSTALLED} {row['helper_sha256'][:16]}",
          flush=True)
    return row


def prefill_wip_settings() -> tuple[int, int] | None:
    said = str(os.environ.get(PREFILL_WIP_ENV) or "").strip()
    if said in ("", "0"):
        return None
    parts = said.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts) or min(int(part) for part in parts) < 1:
        raise RuntimeError(f"{PREFILL_WIP_ENV}={said!r} is not 'requests:tokens' with both at least 1.")
    return int(parts[0]), int(parts[1])


def patch_prefill_wip() -> dict[str, Any]:
    settings = prefill_wip_settings()
    if settings is None:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{PREFILL_WIP_ENV} edits vLLM 0.30.0's scheduler; the {RUNTIME_PROFILE} runtime ships "
                           f"another one.")
    helper_path = BUNDLE_DIR / PREFILL_WIP_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{PREFILL_WIP_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / PREFILL_WIP_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular {PREFILL_WIP_ENV} source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    made = None
    if not all(text.count(put) == 1 for _, put in PREFILL_WIP_EDITS):
        for found, put in PREFILL_WIP_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{PREFILL_WIP_ENV} cannot apply: an anchor of {PREFILL_WIP_RELATIVE} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
    installed = site / PREFILL_WIP_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {PREFILL_WIP_ENV} helper {PREFILL_WIP_INSTALLED} did not land")
    if made is not None:
        temporary = path.with_name(f".{path.name}.wip.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {PREFILL_WIP_ENV} edit of {PREFILL_WIP_RELATIVE} did not land")
    requests, tokens = settings
    row = {"armed": True, "already": made is None and not helper_fresh, "requests": requests, "tokens": tokens,
           "before_sha256": before, "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest()}
    print(f"PREFILL_WIP {'already armed' if row['already'] else 'armed'}: each scheduler step lets at most "
          f"{requests} request(s) prefill and at most {tokens} prefill tokens in all, so decode rows keep the rest "
          f"of the token budget; a chunk under the Mamba block advances sub-block", flush=True)
    return row


def patch_align_tail() -> dict[str, Any]:
    if not _flag_on(ALIGN_TAIL_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{ALIGN_TAIL_ENV} edits vLLM 0.30.0's scheduler; the {RUNTIME_PROFILE} runtime ships "
                           f"another one.")
    helper_path = BUNDLE_DIR / ALIGN_TAIL_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{ALIGN_TAIL_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / ALIGN_TAIL_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular {ALIGN_TAIL_ENV} source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    made = None
    if not all(text.count(put) == 1 for _, put in ALIGN_TAIL_EDITS):
        for found, put in ALIGN_TAIL_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{ALIGN_TAIL_ENV} cannot apply: an anchor of {ALIGN_TAIL_RELATIVE} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
    installed = site / ALIGN_TAIL_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {ALIGN_TAIL_ENV} helper {ALIGN_TAIL_INSTALLED} did not land")
    if made is not None:
        temporary = path.with_name(f".{path.name}.align.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {ALIGN_TAIL_ENV} edit of {ALIGN_TAIL_RELATIVE} did not land")
    row = {"armed": True, "already": made is None and not helper_fresh, "before_sha256": before,
           "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest()}
    print(f"ALIGN_TAIL {'already armed' if row['already'] else 'armed'}: with sparse Mamba retention a prefill "
          f"chunk stops only where a Mamba state is registered and fills the token budget across the other "
          f"block boundaries; the scheduler says 'ALIGN_TAIL live' or 'stands aside' at its first prefill",
          flush=True)
    return row


def no_adopt_setting() -> str:
    said = str(os.environ.get(NO_ADOPT_ENV) or "").strip()
    if said.lower() in ("", "0", "false", "no", "off"):
        return ""
    if said.lower() in ("1", "true", "yes", "on"):
        return "always"
    if said.startswith("file:") and said[len("file:"):].startswith("/"):
        return said
    raise ValueError(f"{NO_ADOPT_ENV} must be a boolean or file:<absolute path>, found {said!r}")


def patch_hisparse_no_adopt() -> dict[str, Any]:
    setting = no_adopt_setting()
    if not setting:
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{NO_ADOPT_ENV} edits vLLM 0.30.0's HiSparse coordinator; the {RUNTIME_PROFILE} "
                           f"runtime ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / NO_ADOPT_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular {NO_ADOPT_ENV} source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    made = None
    if not all(text.count(put) == 1 for _, put in HISPARSE_NO_ADOPT_EDITS):
        for found, put in HISPARSE_NO_ADOPT_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{NO_ADOPT_ENV} cannot apply: an anchor of {NO_ADOPT_RELATIVE} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
        temporary = path.with_name(f".{path.name}.noadopt.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {NO_ADOPT_ENV} edit of {NO_ADOPT_RELATIVE} did not land")
    _note_applied_edits(HISPARSE_NO_ADOPT_EDITS)
    row = {"armed": True, "already": made is None, "setting": setting, "before_sha256": before,
           "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    when = ("always" if setting == "always" else f"while {setting[len('file:'):]} exists")
    print(f"HISPARSE_NO_ADOPT {'already armed' if row['already'] else 'armed'}: a host prefix hit skips pointing "
          f"its null pages at recorded GPU copies {when}, and reads those pages from the host instead; every "
          f"admission that finds copies prints a HISPARSE_ADOPT line with its pages, blocks and ref counts",
          flush=True)
    return row


def tail_restore_setting() -> str:
    said = str(os.environ.get(TAIL_RESTORE_ENV) or "").strip()
    if said.lower() in ("0", "false", "no", "off"):
        return ""
    if said.lower() in ("", "1", "true", "yes", "on"):
        return "always"
    if said.startswith("file:") and said[len("file:"):].startswith("/"):
        return said
    raise ValueError(f"{TAIL_RESTORE_ENV} must be a boolean or file:<absolute path>, found {said!r}")


def patch_hisparse_tail_restore() -> dict[str, Any]:
    setting = tail_restore_setting()
    if not setting:
        return {"armed": False}
    if not str(os.environ.get(TAIL_RESTORE_ENV) or "").strip() and (
            RUNTIME_PROFILE != "v030" or not _flag_on(HOST_KV_ENV)):
        return {"armed": False, "default": True}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{TAIL_RESTORE_ENV} edits vLLM 0.30.0's HiSparse coordinator; the {RUNTIME_PROFILE} "
                           f"runtime ships another one.")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / TAIL_RESTORE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular {TAIL_RESTORE_ENV} source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    made = None
    if not all(text.count(put) == 1 for _, put in HISPARSE_TAIL_RESTORE_EDITS):
        for found, put in HISPARSE_TAIL_RESTORE_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{TAIL_RESTORE_ENV} cannot apply: an anchor of {TAIL_RESTORE_RELATIVE} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
        temporary = path.with_name(f".{path.name}.tailrestore.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {TAIL_RESTORE_ENV} edit of {TAIL_RESTORE_RELATIVE} did not land")
    _note_applied_edits(HISPARSE_TAIL_RESTORE_EDITS)
    row = {"armed": True, "already": made is None, "setting": setting, "before_sha256": before,
           "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    when = "always" if setting == "always" else f"unless {setting[len('file:'):]} exists"
    print(f"HISPARSE_TAIL_RESTORE {'already armed' if row['already'] else 'armed'} ({when}): a request whose local "
          f"prefix hit ends inside a host page gets a resident block for that page, restored from its own host copy "
          f"before the forward, instead of a null page whose rows it would write nowhere and read from the "
          f"producer's host page; every restore prints a HISPARSE_TAIL_RESTORE line", flush=True)
    return row


def patch_prefill_fill() -> dict[str, Any]:
    if not _flag_on(PREFILL_FILL_ENV):
        return {"armed": False}
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(f"{PREFILL_FILL_ENV} edits vLLM 0.30.0's scheduler; the {RUNTIME_PROFILE} runtime ships "
                           f"another one.")
    helper_path = BUNDLE_DIR / PREFILL_FILL_HELPER
    if not helper_path.is_file():
        raise RuntimeError(f"{PREFILL_FILL_ENV} was set but {helper_path} is absent")
    helper = helper_path.read_bytes()
    compile(helper.decode("utf-8"), str(helper_path), "exec")
    site = RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    path = site / PREFILL_FILL_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular {PREFILL_FILL_ENV} source: {path}")
    held = path.read_bytes()
    before = hashlib.sha256(held).hexdigest()
    text = held.decode("utf-8")
    made = None
    if not all(text.count(put) == 1 for _, put in PREFILL_FILL_EDITS):
        for found, put in PREFILL_FILL_EDITS:
            if text.count(found) != 1:
                raise RuntimeError(f"{PREFILL_FILL_ENV} cannot apply: an anchor of {PREFILL_FILL_RELATIVE} appears "
                                   f"{text.count(found)} time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
    installed = site / PREFILL_FILL_INSTALLED
    helper_fresh = not (installed.is_file() and installed.read_bytes() == helper)
    if helper_fresh:
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(helper)
        os.replace(temporary, installed)
        if installed.read_bytes() != helper:
            raise RuntimeError(f"the {PREFILL_FILL_ENV} helper {PREFILL_FILL_INSTALLED} did not land")
    if made is not None:
        temporary = path.with_name(f".{path.name}.fill.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {PREFILL_FILL_ENV} edit of {PREFILL_FILL_RELATIVE} did not land")
    row = {"armed": True, "already": made is None and not helper_fresh, "before_sha256": before,
           "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "helper_sha256": hashlib.sha256(helper).hexdigest()}
    print(f"PREFILL_FILL {'already armed' if row['already'] else 'armed'}: a prefill chunk that cannot reach the "
          f"next Mamba block boundary within the step's token budget takes a sub-block chunk instead of nothing; "
          f"the scheduler says 'PREFILL_FILL live' at the first one", flush=True)
    return row


def patch_weight_cache_daemon(tuning: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    if not _flag_on(WEIGHT_CACHE_ENV):
        return {"armed": False}
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    site = _runtime_site()
    identity = weight_cache_identity(dict(env) if env is not None else dict(os.environ), site)
    pidfile = weight_cache_pidfile()
    identity_path = weight_cache_identity_file()
    running_pid = None
    if pidfile.is_file():
        try:
            running_pid = int(pidfile.read_text(encoding="utf-8").strip())
        except ValueError:
            running_pid = None
    if running_pid is not None and _daemon_alive(running_pid):
        stored = _read_weight_cache_identity(identity_path)
        if stored.get("hash") == identity["hash"]:
            print(f"WEIGHT_CACHE daemon reused at pid {running_pid}, identity "
                  f"{identity['hash'][:16]}", flush=True)
            return {"armed": True, "already": True, "reused": True, "pid": running_pid,
                    "identity": identity["hash"]}
        moves = weight_cache_identity_diff(stored, identity)
        print(f"WEIGHT_CACHE daemon identity changed ("
              f"{str(stored.get('hash') or '(none)')[:16]} -> {identity['hash'][:16]}): "
              f"{moves or ['no tracked env or hashed file differs; the old identity record is missing']}; "
              f"stopping pid {running_pid} and starting fresh", flush=True)
        _stop_weight_cache_daemon(running_pid)
    model_dir = resolve_model_dir()
    command = weight_cache_daemon_command(sys.executable, model_dir, tuning.get("moe_backend"))
    log_path = WORKING_DIR / WEIGHT_CACHE_DAEMON_LOG
    rotate_launch_logs([log_path])
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    finally:
        handle.close()
    pidfile.write_text(str(process.pid), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    print(f"WEIGHT_CACHE daemon started pid {process.pid}, identity {identity['hash'][:16]}, "
          f"waiting for its socket (up to {WEIGHT_CACHE_READY_S:.0f}s), log at {log_path}: "
          f"{' '.join(command)}", flush=True)
    _await_weight_cache_daemon(process.pid, time.monotonic() + WEIGHT_CACHE_READY_S)
    print(f"WEIGHT_CACHE daemon pid {process.pid} socket is up", flush=True)
    return {"armed": True, "already": False, "reused": False, "pid": process.pid, "command": command,
            "log": str(log_path), "identity": identity["hash"]}


def _find_real_cuda_driver() -> Path:
    candidates: list[Path] = []
    completed = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True)
    for line in completed.stdout.splitlines():
        if "libcuda.so.1" not in line or "=>" not in line:
            continue
        candidate = Path(line.split("=>", 1)[1].strip())
        if candidate.is_file() and "stubs" not in str(candidate).lower():
            candidates.append(candidate.resolve())
    search_dirs = [
        Path(entry)
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    search_dirs.extend([Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu")])
    for directory in search_dirs:
        candidate = directory / "libcuda.so.1"
        if candidate.is_file() and "stubs" not in str(candidate.resolve()).lower():
            candidates.append(candidate.resolve())
    if not candidates:
        raise RuntimeError("No real non-stub libcuda.so.1 was found.")
    return max(set(candidates), key=lambda path: path.stat().st_size)


def validate_jit_toolchain(
    env: dict[str, str], cuda_home: Path, cuda_lib: Path, nvcc: Path
) -> dict[str, Any]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The pinned vLLM runtime requires CPython 3.12, found {sys.implementation.name} "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    glibc_raw = os.confstr("CS_GNU_LIBC_VERSION")
    glibc_match = re.fullmatch(r"glibc\s+(\d+)\.(\d+)", glibc_raw or "")
    if glibc_match is None or tuple(map(int, glibc_match.groups())) < (2, 28):
        raise RuntimeError(f"The vLLM runtime requires glibc >=2.28, found {glibc_raw!r}.")
    resolved: dict[str, Path] = {}
    for name in ("gcc", "g++", "ld", "ninja"):
        value = shutil.which(name, path=env["PATH"])
        if not value:
            raise RuntimeError(f"Required vLLM JIT tool was absent from final PATH: {name}")
        resolved[name] = Path(value).resolve()
    if nvcc.resolve() != Path(shutil.which("nvcc", path=env["PATH"]) or "").resolve():
        raise RuntimeError("Final PATH did not resolve nvcc to the pinned CUDA 13 toolkit.")
    if any(str(resolved[name]).startswith(str(RUNTIME_ROOT)) for name in ("gcc", "g++", "ld")):
        raise RuntimeError("Relocated image GCC was selected without validated absolute spec paths.")
    python_h = Path(sysconfig.get_path("include")) / "Python.h"
    if not python_h.is_file():
        raise RuntimeError(f"The Kaggle runtime Python development header is absent: {python_h}")
    if not (cuda_lib / "libcudart.so").is_file():
        raise RuntimeError(f"The pinned CUDA runtime linker name is absent: {cuda_lib}")

    version_commands = {
        "gcc": [str(resolved["gcc"]), "--version"],
        "g++": [str(resolved["g++"]), "--version"],
        "ld": [str(resolved["ld"]), "--version"],
        "ninja": [str(resolved["ninja"]), "--version"],
        "nvcc": [str(nvcc), "--version"],
        "python": [sys.executable, "--version"],
    }
    versions: dict[str, str] = {}
    for name, command in version_commands.items():
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=60
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"JIT tool version probe failed for {name}: "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        versions[name] = (completed.stdout + completed.stderr).strip()
    if "release 13.0" not in versions["nvcc"]:
        raise RuntimeError(f"The pinned nvcc was not CUDA 13.0: {versions['nvcc']}")
    if not versions["python"].startswith("Python 3.12"):
        raise RuntimeError(f"The vLLM runtime requires Python 3.12: {versions['python']}")

    libstdcpp_query = subprocess.run(
        [str(resolved["g++"]), "-print-file-name=libstdc++.so.6"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    libstdcpp_raw = libstdcpp_query.stdout.strip()
    libstdcpp = Path(libstdcpp_raw).resolve() if libstdcpp_raw else Path()
    if (
        libstdcpp_query.returncode != 0
        or not libstdcpp_raw
        or libstdcpp_raw == "libstdc++.so.6"
        or not libstdcpp.is_file()
    ):
        raise RuntimeError(
            "The host g++ did not resolve a real libstdc++.so.6: "
            f"returncode={libstdcpp_query.returncode} stdout={libstdcpp_query.stdout!r} "
            f"stderr={libstdcpp_query.stderr!r}"
        )
    libstdcpp_bytes = libstdcpp.read_bytes()
    required_abi_symbols = (b"GLIBCXX_3.4.22", b"CXXABI_1.3.11")
    missing_abi_symbols = [
        symbol.decode("ascii") for symbol in required_abi_symbols if symbol not in libstdcpp_bytes
    ]
    if missing_abi_symbols:
        raise RuntimeError(
            f"The host libstdc++ is missing required ABI symbols: {missing_abi_symbols}"
        )

    probe_root = TMP_ROOT / "jit-toolchain-probe"
    shutil.rmtree(probe_root, ignore_errors=True)
    probe_root.mkdir(parents=True)
    source = (
        "#include <Python.h>\n"
        "#include <cuda_runtime_api.h>\n"
        "__global__ void duck_probe_kernel(int* value) { *value = 13000; }\n"
        'extern "C" int duck_vllm_jit_probe() {\n'
        "  int version = 0, host_value = 0;\n"
        "  int* device_value = nullptr;\n"
        "  if (cudaRuntimeGetVersion(&version) != cudaSuccess || version != 13000) return -1;\n"
        "  if (cudaMalloc(&device_value, sizeof(int)) != cudaSuccess) return -2;\n"
        "  duck_probe_kernel<<<1, 1>>>(device_value);\n"
        "  if (cudaGetLastError() != cudaSuccess) { cudaFree(device_value); return -3; }\n"
        "  if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(device_value); return -4; }\n"
        "  if (cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost) "
        "!= cudaSuccess) { cudaFree(device_value); return -5; }\n"
        "  if (cudaFree(device_value) != cudaSuccess) return -6;\n"
        "  return host_value;\n"
        "}\n"
    )
    source_path = probe_root / "probe.cu"
    object_path = probe_root / "probe.o"
    library_path = probe_root / "probe.so"
    source_path.write_text(source, encoding="utf-8", newline="\n")
    compile_command = [
        str(nvcc),
        "-std=c++17",
        "-arch=sm_120",
        "-Xcompiler",
        "-fPIC",
        "-ccbin",
        str(resolved["g++"]),
        f"-I{python_h.parent}",
        "-c",
        str(source_path),
        "-o",
        str(object_path),
    ]
    link_command = [
        str(resolved["g++"]),
        "-shared",
        str(object_path),
        f"-L{cuda_lib}",
        f"-Wl,-rpath,{cuda_lib}",
        "-lcudart",
        "-o",
        str(library_path),
    ]

    def run_checked(command: list[str], timeout: int) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            env=env,
            cwd=probe_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "JIT-equivalent toolchain probe failed: "
                f"command={command} returncode={completed.returncode} "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        return {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    try:
        compiled = run_checked(compile_command, 180)
        if not object_path.is_file() or object_path.stat().st_size <= 0:
            raise RuntimeError("nvcc did not create the JIT-equivalent CUDA object.")
        linked = run_checked(link_command, 120)
        if not library_path.is_file() or library_path.stat().st_size <= 0:
            raise RuntimeError("g++ did not create the JIT-equivalent shared library.")
        library_sha256 = sha256_file(library_path)
        load_script = """
import ctypes, json, sys
library = ctypes.CDLL(sys.argv[1])
function = library.duck_vllm_jit_probe
function.argtypes = []
function.restype = ctypes.c_int
print(json.dumps({"cuda_runtime_version": function()}))
"""
        loaded = subprocess.run(
            [sys.executable, "-c", load_script, str(library_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if loaded.returncode != 0:
            raise RuntimeError(
                "ctypes could not load the JIT-equivalent shared library: "
                f"stdout={loaded.stdout!r} stderr={loaded.stderr!r}"
            )
        load_result = json.loads(loaded.stdout.strip().splitlines()[-1])
        if int(load_result.get("cuda_runtime_version", -1)) != 13_000:
            raise RuntimeError(f"Wrong CUDA runtime linked into JIT probe: {load_result}")
        return {
            "compiler_source": "Kaggle host compilers with pinned CUDA 13.0 toolkit",
            "python_runtime": {
                "implementation": sys.implementation.name,
                "version": sys.version,
                "glibc": glibc_raw,
            },
            "resolved_paths": {
                **{name: str(path) for name, path in resolved.items()},
                "nvcc": str(nvcc.resolve()),
                "python": str(Path(sys.executable).resolve()),
                "Python.h": str(python_h.resolve()),
                "libcudart.so": str((cuda_lib / "libcudart.so").resolve()),
                "cuda_home": str(cuda_home.resolve()),
            },
            "versions": versions,
            "host_cxx_abi": {
                "libstdcpp": str(libstdcpp),
                "libstdcpp_sha256": sha256_file(libstdcpp),
                "required_symbols": [
                    symbol.decode("ascii") for symbol in required_abi_symbols
                ],
            },
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "compile": compiled,
            "link": linked,
            "shared_library_sha256": library_sha256,
            "ctypes_load": load_result,
        }
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)


def check_runtime_sources(site: Path, cuda_home: Path) -> str:
    cutlass_pth = site / "nvidia_cutlass_dsl_packages.pth"
    cutlass_nested = site / "nvidia_cutlass_dsl" / "dsl_packages"
    cuda_lib = cuda_home / "targets" / "x86_64-linux" / "lib"
    ple_path = site / PLE_LAYER_RELATIVE
    required = [
        site / "torch" / "__init__.py",
        site / "vllm" / "__init__.py",
        site / "flashinfer" / "__init__.py",
        ple_path,
        cuda_home / "bin" / "nvcc",
        cuda_lib / "libcudart.so",
        cutlass_pth,
        cutlass_nested / "cutlass" / "__init__.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete pinned vLLM runtime: {missing}")
    held = ple_path.read_bytes()
    if not _matches_pin(held.decode("utf-8"), (), VLLM_PLE_PATCHED_SHA256):
        raise RuntimeError(
            "The PLE source is not the pinned patched file with only this build's own edits on it; "
            f"the file is {hashlib.sha256(held).hexdigest()[:16]} and the pinned file is "
            f"{VLLM_PLE_PATCHED_SHA256[:16]}")
    if sha256_file(cutlass_pth) != CUTLASS_PTH_SHA256:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth hash changed.")
    if cutlass_pth.read_text(encoding="utf-8").strip() != CUTLASS_PTH_TEXT:
        raise RuntimeError("The pinned nvidia-cutlass-dsl .pth instruction changed.")
    return hashlib.sha256(held).hexdigest()


def runtime_environment(
    *,
    deep_preload_validation: bool = True,
    tuning: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved_tuning = tuning or resolve_vllm_tuning()
    allocator_input = {
        key: os.environ.get(key)
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    }
    expandable_true = re.compile(
        r"\bexpandable_segments\s*:\s*(?:true|1|yes)\b",
        re.IGNORECASE,
    )
    bad_allocator = {
        key: value
        for key, value in allocator_input.items()
        if value is not None and expandable_true.search(value)
    }
    if bad_allocator:
        raise RuntimeError(
            "Expandable CUDA allocator segments are forbidden for PLE CPU-offload IPC: "
            f"{bad_allocator}"
        )
    site = _runtime_site()
    cutlass_pth = site / "nvidia_cutlass_dsl_packages.pth"
    cutlass_nested = site / "nvidia_cutlass_dsl" / "dsl_packages"
    cuda_home = RUNTIME_ROOT / "usr" / "local" / "cuda-13.0"
    cuda_lib = cuda_home / "targets" / "x86_64-linux" / "lib"
    nvcc = cuda_home / "bin" / "nvcc"
    ple_path = site / PLE_LAYER_RELATIVE
    check_runtime_sources(site, cuda_home)
    real_driver = _find_real_cuda_driver()

    pinned_nvidia_libraries = sorted(
        path for path in (site / "nvidia").glob("*/lib") if path.is_dir()
    )
    library_dirs = [
        cuda_lib,
        cuda_home / "lib64",
        site / "torch" / "lib",
        *pinned_nvidia_libraries,
        real_driver.parent,
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    for root in (CACHE_ROOT, COMPILE_CACHE_ROOT, TMP_ROOT):
        root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "XDG_CACHE_HOME": CACHE_ROOT,
        "TORCH_HOME": CACHE_ROOT / "torch",
        "TORCHINDUCTOR_CACHE_DIR": COMPILE_CACHE_ROOT / "torchinductor",
        "TRITON_CACHE_DIR": COMPILE_CACHE_ROOT / "triton",
        "CUDA_CACHE_PATH": COMPILE_CACHE_ROOT / "cuda",
        "FLASHINFER_WORKSPACE_BASE": COMPILE_CACHE_ROOT / "flashinfer",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    marlin_builds = Path(os.environ.get(MARLIN_BUILD_ENV) or COMPILE_CACHE_ROOT / MARLIN_BUILD_DIR)
    marlin_builds.mkdir(parents=True, exist_ok=True)
    PYC_RESULTS["compile"] = join_compile_seed("the runtime environment, before the self-checks")
    if pyc_writes_allowed():
        ensure_runtime_writable(site)
    start_pyc_seed(site)

    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cutlass_nested), str(site)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PATH"] = os.pathsep.join(
        [str(cuda_home / "bin"), str(RUNTIME_ROOT / "usr" / "local" / "bin"), env.get("PATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in library_dirs if path.is_dir()]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    for key, path in cache_paths.items():
        env[key] = str(path)
    env[MARLIN_BUILD_ENV] = str(marlin_builds)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_HOME": str(cuda_home),
            "CUDACXX": str(nvcc),
            "TMPDIR": str(TMP_ROOT),
            "VLLM_PLE_CPU_OFFLOAD": "1",
            "VLLM_PLE_OFFLOAD_READY_TIMEOUT": str(SERVER_READY_TIMEOUT),
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
            "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(resolved_tuning["omp_num_threads"]),
            "MKL_NUM_THREADS": str(resolved_tuning["omp_num_threads"]),
        }
    )
    for name in (EXPERTS_KEPT_ENV, COUNT_EXPERTS_ENV, COUNT_EXPERTS_PATH_ENV,
                 MOE_ROUTE_ENV, MOE_ROUTE_FILE_ENV, ENFORCE_EAGER_ENV, TRACE_REPLAY_ENV,
                 "VLLM_LOGGING_LEVEL"):
        said = str(os.environ.get(name) or "").strip()
        if said:
            env[name] = said
            print(f"SERVER_ENV {name}={said}", flush=True)
    env.update(
        {}
    )
    apply_api_malloc(env)
    if pyc_writes_allowed():
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env.pop("PYTHONPYCACHEPREFIX", None)
        print(f"PYC_WRITE imports write bytecode beside their sources under {site} "
              f"({PYC_SEED_ENV}/{PYC_HARVEST_ENV}/{PYC_WRITE_ENV}); PYTHONDONTWRITEBYTECODE and "
              f"PYTHONPYCACHEPREFIX are not passed to the server", flush=True)
    PYC_RESULTS["trace"] = install_init_trace(env)
    if not deep_preload_validation:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                "The pinned vLLM runtime requires CPython 3.12, found "
                f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}."
            )
        glibc_raw = os.confstr("CS_GNU_LIBC_VERSION")
        glibc_match = re.fullmatch(r"glibc\s+(\d+)\.(\d+)", glibc_raw or "")
        if glibc_match is None or tuple(map(int, glibc_match.groups())) < (2, 28):
            raise RuntimeError(f"The vLLM runtime requires glibc >=2.28, found {glibc_raw!r}.")
        tools = {
            name: shutil.which(name, path=env["PATH"])
            for name in ("gcc", "g++", "ld", "ninja", "nvcc")
        }
        missing_tools = [name for name, path in tools.items() if not path]
        if missing_tools:
            raise RuntimeError(f"Required vLLM tools are absent: {missing_tools}")
        return env, {
            "validation_mode": "fast",
            "deep_preload_validation": False,
            "skipped_duplicate_probes": [
                "CUDA compile/link/device probe",
                "torch/flashinfer/vllm import subprocess",
                "NCCL metadata and runtime probe",
            ],
            "allocator_input": allocator_input,
            "allocator_effective": "expandable_segments:False",
            "allocator_alias_removed": True,
            "cuda_compatibility": "0",
            "torch_cuda_arch_list": "12.0",
            "python": sys.version,
            "glibc": glibc_raw,
            "site_packages": str(site),
            "cutlass_pth": str(cutlass_pth),
            "cutlass_pth_sha256": CUTLASS_PTH_SHA256,
            "cutlass_nested_path": str(cutlass_nested),
            "cuda_home": str(cuda_home),
            "cuda_library_dir": str(cuda_lib),
            "real_cuda_driver": str(real_driver),
            "required_tools": tools,
            "ple_patch_sha256": VLLM_PLE_PATCHED_SHA256,
            "ple_source_sha256": sha256_file(ple_path),
        }
    jit_toolchain = validate_jit_toolchain(env, cuda_home, cuda_lib, nvcc)
    nvcc_result = subprocess.run(
        [str(nvcc), "--version"], env=env, capture_output=True, text=True, check=True, timeout=60
    )
    if RUNTIME_PROFILE != "day0":
        probed = subprocess.run(
            [sys.executable, "-c", PROFILE_IMPORT_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        import_info = json.loads(probed.stdout.strip().splitlines()[-1])
        wanted = {"vllm": VLLM_VERSION, "cutlass": CUTLASS_DSL_VERSION,
                  "torch_cuda": "13.0", "fp8_ple_loader": True}
        wrong = {key: import_info.get(key) for key, expected in wanted.items()
                 if import_info.get(key) != expected}
        if wrong or not str(import_info.get("vllm_file", "")).startswith(str(site)):
            raise RuntimeError(
                f"The {RUNTIME_PROFILE} runtime did not import as pinned: {import_info}")
        return env, {
            "validation_mode": "full",
            "deep_preload_validation": True,
            "runtime_profile": RUNTIME_PROFILE,
            "allocator_effective": "expandable_segments:False",
            "site_packages": str(site),
            "cuda_home": str(cuda_home),
            "cuda_library_dir": str(cuda_lib),
            "real_cuda_driver": str(real_driver),
            "nvcc": nvcc_result.stdout.strip(),
            "imports": import_info,
            "jit_toolchain": jit_toolchain,
            "ple_source_sha256": sha256_file(ple_path),
        }
    import_script = """
from importlib import metadata
import inspect, json, os
from types import SimpleNamespace
import cutlass, torch, flashinfer, vllm
import vllm.models.qwen3_8_flash_next.nvidia.ple_layer as ple_module
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import (
    Qwen3_8FlashNextPLEFp8EmbeddingMethod,
    _get_ple_embedding_quant_method,
    _is_exact_radixark_nvfp4_ple,
)
runtime_prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
excluded_prefix = "model.language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
config = ModelOptNvFp4Config(
    quant_method="NVFP4",
    is_checkpoint_nvfp4_serialized=True,
    exclude_modules=["*.ple.*"],
    group_size=16,
)
text_config = SimpleNamespace(
    model_type="qwen4_exp_text",
    hidden_size=2560,
    num_hidden_layers=48,
    num_experts=512,
    ple_embed_dim=2560,
    ple_embedding_dtype="float8_e4m3fn",
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    split_ngram_parts=128,
    ple_layer_ids=[2],
    layer_types=[
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(48)
    ],
)
method = _get_ple_embedding_quant_method(
    config,
    runtime_prefix,
    text_config,
)
checkpoint_method = _get_ple_embedding_quant_method(
    config,
    excluded_prefix,
    text_config,
)
print(json.dumps({
    "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "allocator_alias": os.environ.get("PYTORCH_ALLOC_CONF"),
    "cuda_compatibility": os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY"),
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_nccl": list(torch.cuda.nccl.version()),
    "torch_nccl_requirements": [
        item for item in (metadata.requires("torch") or [])
        if item.lower().startswith("nvidia-nccl-cu13")
    ],
    "installed_nccl_dist": metadata.version("nvidia-nccl-cu13"),
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "cutlass": metadata.version("nvidia-cutlass-dsl"),
    "cutlass_file": cutlass.__file__,
    "vllm": metadata.version("vllm"),
    "vllm_file": vllm.__file__,
    "ple_method": type(method).__name__,
    "ple_gate_source": inspect.getsource(_get_ple_embedding_quant_method),
    "ple_exact_gate_source": inspect.getsource(_is_exact_radixark_nvfp4_ple),
    "quant_config_name": config.get_name(),
    "quant_method": config.quant_method,
    "quant_serialized": config.is_checkpoint_nvfp4_serialized,
    "quant_group_size": config.group_size,
    "checkpoint_prefix_excluded": config.is_layer_excluded(excluded_prefix),
    "checkpoint_prefix_selected": checkpoint_method is not None,
    "patch_env_name": ple_module._RADIXARK_NVFP4_PLE_FP8_ENV,
    "patch_config_env_name": ple_module._RADIXARK_NVFP4_CONFIG_SHA256_ENV,
    "patch_config_sha256": ple_module._RADIXARK_NVFP4_CONFIG_SHA256,
    "patch_runtime_prefix": ple_module._RADIXARK_NVFP4_PLE_PREFIX,
    "patch_excluded_prefix": ple_module._RADIXARK_NVFP4_PLE_EXCLUDED_PREFIX,
}, sort_keys=True))
"""
    imported = subprocess.run(
        [sys.executable, "-c", import_script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    import_info = json.loads(imported.stdout.strip().splitlines()[-1])
    if import_info.get("allocator_conf") != "expandable_segments:False":
        raise RuntimeError(f"Pinned allocator mode was not active before torch import: {import_info}")
    if import_info.get("allocator_alias") is not None:
        raise RuntimeError(f"Conflicting allocator alias reached torch import: {import_info}")
    if import_info.get("cuda_compatibility") != "0":
        raise RuntimeError(f"CUDA compatibility mode was not disabled before import: {import_info}")
    if import_info.get("torch_cuda_arch_list") != "12.0":
        raise RuntimeError(f"Wrong CUDA architecture list reached torch import: {import_info}")
    if import_info.get("vllm") != VLLM_VERSION:
        raise RuntimeError(f"Wrong vLLM version: {import_info}")
    if import_info.get("torch") != "2.13.0+cu130" or import_info.get("torch_cuda") != "13.0":
        raise RuntimeError(f"Wrong pinned PyTorch/CUDA build: {import_info}")
    if import_info.get("installed_nccl_dist") != "2.30.7":
        raise RuntimeError(f"Wrong installed nvidia-nccl-cu13 distribution: {import_info}")
    if import_info.get("torch_nccl") != [2, 29, 7]:
        raise RuntimeError(f"PyTorch's pinned NCCL build version changed: {import_info}")
    nccl_requirements = import_info.get("torch_nccl_requirements") or []
    if len(nccl_requirements) != 1 or not str(nccl_requirements[0]).startswith(
        "nvidia-nccl-cu13==2.29.7;"
    ):
        raise RuntimeError(f"PyTorch's pinned NCCL dependency record changed: {import_info}")
    if import_info.get("cutlass") != CUTLASS_DSL_VERSION:
        raise RuntimeError(f"Wrong nvidia-cutlass-dsl version: {import_info}")
    if not str(import_info.get("cutlass_file", "")).startswith(str(cutlass_nested)):
        raise RuntimeError(f"cutlass did not import from its pinned nested path: {import_info}")
    if not str(import_info.get("vllm_file", "")).startswith(str(site)):
        raise RuntimeError(f"vLLM did not import from the pinned runtime: {import_info}")
    if import_info.get("ple_method") != "Qwen3_8FlashNextPLEFp8EmbeddingMethod":
        raise RuntimeError(f"The exact NVFP4-to-FP8 PLE gate did not select its loader: {import_info}")
    gate_source = str(import_info.pop("ple_gate_source", ""))
    exact_gate_source = str(import_info.pop("ple_exact_gate_source", ""))
    expected_quant_gate = {
        "quant_config_name": "modelopt_fp4",
        "quant_method": "NVFP4",
        "quant_serialized": True,
        "quant_group_size": 16,
        "checkpoint_prefix_excluded": True,
        "checkpoint_prefix_selected": False,
    }
    if any(import_info.get(key) != value for key, value in expected_quant_gate.items()):
        raise RuntimeError(f"ModelOpt NVFP4 internals did not pass the exact PLE gate: {import_info}")
    expected_patch_constants = {
        "patch_env_name": "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8",
        "patch_config_env_name": "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256",
        "patch_config_sha256": MODEL_CONFIG_SHA256,
        "patch_runtime_prefix": PLE_EXACT_PREFIX,
        "patch_excluded_prefix": PLE_EXCLUDED_PREFIX,
    }
    if any(import_info.get(key) != value for key, value in expected_patch_constants.items()):
        raise RuntimeError(f"Patched PLE constants did not match their pins: {import_info}")
    for marker in ("get_name", "group_size", "is_layer_excluded", "prefix !="):
        if marker not in exact_gate_source:
            raise RuntimeError(f"Patched PLE function is missing gate marker {marker!r}.")
    return env, {
        "validation_mode": "full",
        "deep_preload_validation": True,
        "allocator_input": allocator_input,
        "allocator_effective": "expandable_segments:False",
        "allocator_alias_removed": True,
        "cuda_compatibility": "0",
        "torch_cuda_arch_list": "12.0",
        "site_packages": str(site),
        "cutlass_pth": str(cutlass_pth),
        "cutlass_pth_sha256": sha256_file(cutlass_pth),
        "cutlass_nested_path": str(cutlass_nested),
        "cuda_home": str(cuda_home),
        "cuda_library_dir": str(cuda_lib),
        "real_cuda_driver": str(real_driver),
        "real_cuda_driver_sha256": sha256_file(real_driver),
        "nvcc": nvcc_result.stdout.strip(),
        "imports": import_info,
        "nccl_dependency_mismatch": {
            "torch_metadata_requires": nccl_requirements[0],
            "installed_distribution": import_info["installed_nccl_dist"],
            "loaded_runtime": import_info["torch_nccl"],
            "tp1_no_nccl_expected": True,
        },
        "jit_toolchain": jit_toolchain,
        "ple_patch_sha256": sha256_file(ple_path),
        "ple_gate_source_sha256": hashlib.sha256(gate_source.encode("utf-8")).hexdigest(),
        "ple_exact_gate_source_sha256": hashlib.sha256(
            exact_gate_source.encode("utf-8")
        ).hexdigest(),
    }


def _pidfd_probe_once(env: dict[str, str]) -> dict[str, Any]:
    holder_script = """
import json, os, sys
fd = os.open('/dev/null', os.O_RDONLY)
print(json.dumps({'pid': os.getpid(), 'fd': fd}), flush=True)
sys.stdin.buffer.read(1)
os.close(fd)
"""
    reader_script = """
import ctypes, json, os, sys
pid, target_fd = map(int, sys.argv[1:3])
result = {'pid': pid, 'target_fd': target_fd}
pidfd = os.pidfd_open(pid, 0)
try:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    copied = int(libc.syscall(438, pidfd, target_fd, 0))
    if copied < 0:
        error = ctypes.get_errno()
        result.update({'ok': False, 'errno': error, 'error': os.strerror(error)})
    else:
        os.close(copied)
        result.update({'ok': True, 'copied_fd': copied})
finally:
    os.close(pidfd)
print(json.dumps(result, sort_keys=True))
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        readable, _, _ = select.select([holder.stdout], [], [], 10)
        if not readable:
            raise TimeoutError("pidfd diagnostic holder produced no identity within 10s")
        line = holder.stdout.readline()
        if not line:
            stderr = holder.stderr.read() if holder.stderr is not None else ""
            raise RuntimeError(f"pidfd holder failed before identity: {stderr}")
        target = json.loads(line)
        reader = subprocess.run(
            [sys.executable, "-c", reader_script, str(target["pid"]), str(target["fd"])],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if reader.returncode != 0:
            return {
                "ok": False,
                "reader_returncode": reader.returncode,
                "reader_stdout": reader.stdout,
                "reader_stderr": reader.stderr,
            }
        value = json.loads(reader.stdout.strip().splitlines()[-1])
        value["holder_pid"] = int(target["pid"])
        return value
    finally:
        if holder.stdin is not None:
            try:
                holder.stdin.write("x")
                holder.stdin.flush()
                holder.stdin.close()
            except OSError:
                pass
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=10)


def diagnose_pidfd_ipc(env: dict[str, str]) -> dict[str, Any]:
    """Record sibling pidfd_getfd behavior without changing process permissions.

    The pinned PyTorch allocator uses classic CUDA IPC because expandable
    segments are disabled. This CPU probe is diagnostic only.
    """
    if not hasattr(os, "pidfd_open"):
        return {"diagnostic_only": True, "supported": False, "result": None}
    try:
        result = _pidfd_probe_once(env)
    except Exception as exc:
        result = {
            "ok": False,
            "diagnostic_exception_type": type(exc).__name__,
            "diagnostic_exception": str(exc),
        }
    return {
        "diagnostic_only": True,
        "supported": True,
        "allocator": "classic_cuda_ipc",
        "permissions_changed": False,
        "result": result,
    }


def validate_cli_contract(env: dict[str, str]) -> dict[str, Any]:
    full_help_option = "--help=all"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            full_help_option,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    help_text = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"Pinned vLLM CLI help failed: {help_text[-4000:]}")
    required_flags = [
        "--distributed-executor-backend",
        "--gpu-memory-utilization",
        "--kv-cache-memory-bytes",
        "--kv-cache-dtype",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
        "--speculative-config",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "--reasoning-parser",
        "--max-cudagraph-capture-size",
    ]
    required_flags.extend(flag for flag in serving_flags() if flag.startswith("--"))
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(f"Pinned vLLM CLI is missing required flags: {missing}")
    return {
        "module": "vllm.entrypoints.cli.main",
        "subcommand": "serve",
        "help_option": full_help_option,
        "required_flags": required_flags,
        "help_sha256": hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
    }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        match = re.search(r"(\d+)", raw)
        if match:
            values[key] = int(match.group(1)) * 1024
    return values


def gpu_inventory() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or "rtx pro 6000" not in rows[0].lower():
        raise RuntimeError(f"Expected one RTX PRO 6000, found: {rows}")
    meminfo = _meminfo()
    available = int(meminfo.get("MemAvailable", 0))
    if available < MIN_HOST_AVAILABLE_BYTES:
        raise RuntimeError(
            f"Host memory is too small for FP8 PLE CPU offload: {available} < {MIN_HOST_AVAILABLE_BYTES}"
        )
    shm = os.statvfs("/dev/shm")
    shm_free = int(shm.f_bavail * shm.f_frsize)
    probe = Path("/dev/shm") / f"duck-vllm-ipc-{os.getpid()}"
    try:
        probe.write_bytes(b"duck-vllm-ipc")
        if probe.read_bytes() != b"duck-vllm-ipc":
            raise RuntimeError("/dev/shm readback did not match.")
    finally:
        probe.unlink(missing_ok=True)
    disk = subprocess.run(
        ["df", "-B1", str(WORKING_DIR), "/tmp", "/dev/shm"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return {
        "gpu_rows": rows,
        "meminfo_bytes": meminfo,
        "host_mem_available_bytes": available,
        "shm_free_bytes": shm_free,
        "disk": disk.stdout,
    }


def proc_record(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        cmdline_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [
            part.decode("utf-8", errors="replace")
            for part in cmdline_bytes.split(b"\x00")
            if part
        ]
        cmdline = cmdline_bytes.replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
            "cmdline": cmdline,
            "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
            "argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def process_table() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            record = proc_record(int(item.name))
            if record is not None:
                records[int(item.name)] = record
    return records


def descendant_ids(root_pid: int, records: dict[int, dict[str, Any]]) -> set[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if record["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def allowed_detached_worker(
    record: dict[str, Any],
    workers_by_pid: dict[int, dict[str, Any]],
    root_pgid: int,
    root_sid: int,
) -> bool:
    pid = int(record.get("pid", -1))
    if pid > 1 and _flag_on(NSYS_ENV):
        return True
    parent = workers_by_pid.get(int(record.get("ppid", -1)))
    return (
        pid > 1
        and parent is not None
        and int(parent.get("pgid", -1)) == root_pgid
        and int(parent.get("sid", -1)) == root_sid
        and int(record.get("pgid", -1)) == pid
        and int(record.get("sid", -1)) == pid
        and record.get("cmdline_sha256") == TORCH_SHM_MANAGER_CMDLINE_SHA256
    )


def start_ticks(pid: int) -> int:
    record = proc_record(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return int(record["start_ticks"])


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def tail_log(lines: int = 200) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(
        SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def alive(pid: int, ticks: int | None = None) -> bool:
    record = proc_record(pid)
    return (
        record is not None
        and record["state"] != "Z"
        and (ticks is None or int(record["start_ticks"]) == ticks)
    )


def stop_owned_server() -> dict[str, Any]:
    stop_weights_prefetch(keep_early=True)
    result: dict[str, Any] = {"identity_exists": SERVER_IDENTITY.exists(), "stopped": True}
    if not SERVER_IDENTITY.exists():
        return result
    identity = read_json(SERVER_IDENTITY)
    if not isinstance(identity, dict):
        result.update(
            {
                "identity_matches": False,
                "identity_rejection_reasons": ["identity is not a JSON object"],
                "survivors": None,
                "port_open": port_open(),
                "stopped": False,
            }
        )
        return result
    identity_keys = {
        "schema_version",
        "backend",
        "boot_id",
        "host",
        "port",
        "pid",
        "start_ticks",
        "pgid",
        "sid",
        "argv",
        "argv_sha256",
        "vllm_tuning",
        "started_epoch",
        "phase",
        "snapshot_epoch",
        "ready_epoch",
        "workers",
    }
    worker_keys = {"pid", "start_ticks", "ppid", "pgid", "sid", "cmdline_sha256"}
    reasons: list[str] = []
    if set(identity) != identity_keys:
        reasons.append("identity key set mismatch")
    reasons.extend(validate_vllm_tuning_document(identity.get("vllm_tuning")))
    try:
        pid = int(identity["pid"])
        ticks = int(identity["start_ticks"])
        pgid = int(identity["pgid"])
        sid = int(identity["sid"])
    except (KeyError, TypeError, ValueError):
        pid = ticks = pgid = sid = -1
        reasons.append("invalid root numeric identity")
    argv = identity.get("argv")
    expected_argv_sha = (
        hashlib.sha256(
            json.dumps(argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv)
        else None
    )
    if expected_argv_sha is None or identity.get("argv_sha256") != expected_argv_sha:
        reasons.append("canonical argv hash mismatch")
    if (
        identity.get("schema_version") != 1
        or identity.get("backend") != "vllm"
        or identity.get("boot_id") != boot_id()
        or identity.get("host") != HOST
        or identity.get("port") != PORT
        or pid <= 1
        or pgid != pid
        or sid != pid
    ):
        reasons.append("root ownership header mismatch")
    phase = identity.get("phase")
    workers = identity.get("workers")
    started_epoch = identity.get("started_epoch")
    snapshot_epoch = identity.get("snapshot_epoch")
    ready_epoch = identity.get("ready_epoch")
    if (
        phase not in {"starting", "ready"}
        or not isinstance(started_epoch, (int, float))
        or not isinstance(snapshot_epoch, (int, float))
        or snapshot_epoch < started_epoch
        or not isinstance(workers, list)
        or not workers
    ):
        reasons.append("invalid phase or snapshot semantics")
    elif phase == "starting" and (ready_epoch is not None or len(workers) != 1):
        reasons.append("invalid starting-phase snapshot")
    elif phase == "ready" and (
        not isinstance(ready_epoch, (int, float))
        or ready_epoch < started_epoch
        or ready_epoch > snapshot_epoch
    ):
        reasons.append("invalid ready-phase snapshot")
    if isinstance(workers, list):
        workers_by_pid = {
            int(worker["pid"]): worker
            for worker in workers
            if isinstance(worker, dict) and type(worker.get("pid")) is int
        }
        for worker in workers:
            if not isinstance(worker, dict) or set(worker) != worker_keys:
                reasons.append("worker key set mismatch")
                break
            if int(worker.get("sid", -1)) != sid and not allowed_detached_worker(
                worker, workers_by_pid, pgid, sid
            ):
                reasons.append(
                    f"unrecognized detached worker for pid {worker.get('pid')}"
                )
                break
        root_workers = [
            worker
            for worker in workers
            if isinstance(worker, dict) and worker.get("pid") == pid
        ]
        if (
            len(root_workers) != 1
            or int(root_workers[0].get("start_ticks", -1)) != ticks
            or int(root_workers[0].get("pgid", -1)) != pgid
            or int(root_workers[0].get("sid", -1)) != sid
        ):
            reasons.append("root worker snapshot mismatch")

    current_root = proc_record(pid) if pid > 1 else None
    if current_root is not None and (
        int(current_root["start_ticks"]) != ticks
        or int(current_root["pgid"]) != pgid
        or int(current_root["sid"]) != sid
        or current_root["argv_sha256"] != identity.get("argv_sha256")
    ):
        reasons.append("live root fingerprint mismatch")
    if current_root is not None and isinstance(workers, list):
        root_worker = next(
            (worker for worker in workers if isinstance(worker, dict) and worker.get("pid") == pid),
            None,
        )
        if root_worker is None or current_root["cmdline_sha256"] != root_worker.get(
            "cmdline_sha256"
        ):
            reasons.append("live root cmdline fingerprint mismatch")
    table = process_table()
    group_records = [record for record in table.values() if int(record["pgid"]) == pgid]
    if current_root is not None and (
        not group_records or any(int(record["sid"]) != sid for record in group_records)
    ):
        reasons.append("live process group is not wholly inside the owned session")
    if isinstance(workers, list):
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            live = table.get(int(worker.get("pid", -1)))
            if live is None:
                continue
            if (
                int(live["start_ticks"]) != int(worker.get("start_ticks", -1))
                or int(live["pgid"]) != int(worker.get("pgid", -1))
                or int(live["sid"]) != int(worker.get("sid", -1))
                or live["cmdline_sha256"] != worker.get("cmdline_sha256")
            ):
                reasons.append(f"live worker fingerprint mismatch for pid {worker.get('pid')}")
                break

    matches = current_root is not None and not reasons
    result.update(
        {
            "pid": pid,
            "start_ticks": ticks,
            "pgid": pgid,
            "sid": sid,
            "identity_matches": matches,
            "identity_rejection_reasons": reasons,
        }
    )

    def live_owned_records() -> list[dict[str, Any]]:
        if sid <= 1:
            return []
        table = process_table()
        owned = {
            int(record["pid"]): record
            for record in table.values()
            if int(record["sid"]) == sid and record["state"] != "Z"
        }
        if isinstance(workers, list):
            workers_by_pid = {
                int(worker["pid"]): worker
                for worker in workers
                if isinstance(worker, dict) and type(worker.get("pid")) is int
            }
            for worker in workers:
                if (
                    not isinstance(worker, dict)
                    or int(worker.get("sid", -1)) == sid
                    or not allowed_detached_worker(
                        worker, workers_by_pid, pgid, sid
                    )
                ):
                    continue
                live = table.get(int(worker["pid"]))
                if live is not None and (
                    int(live["start_ticks"]) == int(worker["start_ticks"])
                    and int(live["pgid"]) == int(worker["pgid"])
                    and int(live["sid"]) == int(worker["sid"])
                    and live["cmdline_sha256"] == worker["cmdline_sha256"]
                ):
                    owned[int(live["pid"])] = live
        return list(owned.values())

    if matches:
        try:
            os.killpg(pgid, signal.SIGTERM)
            result["sigterm_sent"] = True
        except ProcessLookupError:
            pass
        separately_signaled: list[int] = []
        for record in live_owned_records():
            if int(record["pgid"]) == pgid:
                continue
            try:
                os.kill(int(record["pid"]), signal.SIGTERM)
                separately_signaled.append(int(record["pid"]))
            except ProcessLookupError:
                pass
        if separately_signaled:
            result["separate_sigterm_pids"] = separately_signaled
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and live_owned_records():
            time.sleep(1)
        survivors = live_owned_records()
        if survivors:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            separately_killed: list[int] = []
            for record in survivors:
                if int(record["pgid"]) == pgid:
                    continue
                current = proc_record(int(record["pid"]))
                if (
                    current is None
                    or int(current["start_ticks"]) != int(record["start_ticks"])
                    or int(current["pgid"]) != int(record["pgid"])
                    or int(current["sid"]) != int(record["sid"])
                    or current["cmdline_sha256"] != record["cmdline_sha256"]
                ):
                    continue
                try:
                    os.kill(int(record["pid"]), signal.SIGKILL)
                    separately_killed.append(int(record["pid"]))
                except ProcessLookupError:
                    pass
            result["sigkill_sent"] = True
            if separately_killed:
                result["separate_sigkill_pids"] = separately_killed
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and live_owned_records():
                time.sleep(0.5)
    survivors = live_owned_records()
    result["survivors"] = [
        {
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "ppid": int(record["ppid"]),
            "pgid": int(record["pgid"]),
            "sid": int(record["sid"]),
            "cmdline_sha256": str(record["cmdline_sha256"]),
        }
        for record in sorted(survivors, key=lambda item: int(item["pid"]))
    ]
    result["port_open"] = port_open()
    result["stopped"] = not survivors and not result["port_open"]
    return result


def port_open(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def request_bytes(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> bytes:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body[:4000]}") from exc


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    value = json.loads(request_bytes(url, payload=payload, timeout=timeout).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}, found {type(value).__name__}.")
    return value


def decode_graph_tokens(tuning: dict[str, Any]) -> int:
    rows = int(tuning["mtp_speculative_tokens"]) + 1
    return int(tuning["max_num_seqs"]) * (min(rows, 5) if draft_budget_wanted() else rows)


def piecewise_large(tuning: dict[str, Any]) -> tuple[int, int]:
    largest, _ = _bounded_integer_environment(
        PIECEWISE_LARGE_ENV, default=0, minimum=0, maximum=1_048_576)
    if largest == 0:
        return 0, 0
    step, _ = _bounded_integer_environment(
        PIECEWISE_LARGE_STEP_ENV, default=PIECEWISE_LARGE_STEP,
        minimum=PIECEWISE_LARGE_STEP_MIN, maximum=PIECEWISE_LARGE_STEP_MAX)
    base = int(tuning["max_cudagraph_capture_size"])
    budget = int(tuning["max_num_batched_tokens"])
    floor = max(base, decode_graph_tokens(tuning))
    if RUNTIME_PROFILE != "v030":
        raise RuntimeError(
            f"{PIECEWISE_LARGE_ENV} extends vLLM 0.30.0's V2 capture list and edits its draft "
            f"graph manager; the {RUNTIME_PROFILE} runtime ships other ones.")
    if _flag_on(ENFORCE_EAGER_ENV):
        raise RuntimeError(
            f"{PIECEWISE_LARGE_ENV}={largest} asks for captured graphs while "
            f"{ENFORCE_EAGER_ENV} turns every graph off.")
    if base <= 0:
        raise RuntimeError(
            f"{PIECEWISE_LARGE_ENV}={largest} extends the capture list "
            f"{MAX_CUDAGRAPH_CAPTURE_SIZE_ENV} opens, which is unset or 0, so no graph is "
            f"captured at all.")
    if largest <= floor:
        raise RuntimeError(
            f"{PIECEWISE_LARGE_ENV}={largest} adds no size: the capture list already reaches "
            f"{base} and the full decode graphs {decode_graph_tokens(tuning)} tokens.")
    if largest > budget:
        raise RuntimeError(
            f"{PIECEWISE_LARGE_ENV}={largest} is above the {budget}-token step budget: vLLM drops "
            f"every capture size above the budget and then refuses a maximum that names one, so "
            f"the server would not start.")
    return largest, step


def capture_sparse(tuning: dict[str, Any]) -> tuple[int, int]:
    raw = str(os.environ.get(CAPTURE_SPARSE_ENV) or "").strip()
    if not raw:
        return 0, 0
    found = re.fullmatch(r"(\d+):(\d+)", raw)
    if found is None:
        raise RuntimeError(f"{CAPTURE_SPARSE_ENV} must be <from>:<stride> in tokens, found {raw!r}.")
    start, stride = int(found[1]), int(found[2])
    rows = int(tuning["mtp_speculative_tokens"]) + 1
    if stride < 1 or (stride % rows and not draft_budget_wanted()):
        raise RuntimeError(f"{CAPTURE_SPARSE_ENV}={raw}: the stride must be a positive multiple of the "
                           f"{rows} rows a decode request takes, or the full decode graphs above {start} "
                           f"tokens would all be dropped.")
    return start, stride


def sparse_sizes(sizes: list[int], start: int, stride: int, keep: int) -> list[int]:
    if not stride or not sizes:
        return sizes
    return [size for size in sizes
            if size <= start or size % stride == 0 or size % keep == 0 or size == sizes[-1]]


def cudagraph_capture_sizes(tuning: dict[str, Any]) -> list[int]:
    base = int(tuning["max_cudagraph_capture_size"])
    sizes = [size for size in CUDAGRAPH_CAPTURE_SIZES if size <= base]
    start, stride = capture_sparse(tuning)
    if stride:
        keep = _bounded_integer_environment(
            PIECEWISE_LARGE_STEP_ENV, default=PIECEWISE_LARGE_STEP,
            minimum=PIECEWISE_LARGE_STEP_MIN, maximum=PIECEWISE_LARGE_STEP_MAX)[0]
        sizes = sparse_sizes(sizes, start, stride, keep)
    largest, step = piecewise_large(tuning)
    widest = decode_graph_tokens(tuning)
    if not largest:
        return sorted(set(sizes) | ({widest} if widest <= base else set()))
    floor = max(base, widest)
    first = (floor // step + 1) * step
    return sorted(set(sizes) | set(range(first, largest + 1, step)) | {largest} | {widest})


def _kv_rebase_refusal() -> str:
    spec = importlib.util.spec_from_file_location("taaf_kv_rebase", Path(__file__).with_name("kv_rebase.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.refusal()


WEIGHTS_PREFETCH: Any = None
PREFETCH_MODULE: Any = None
TIER: dict[str, Any] = {"root": None, "use": False, "launches": 0}


def prefetch_module() -> Any:
    global PREFETCH_MODULE
    if PREFETCH_MODULE is None:
        spec = importlib.util.spec_from_file_location("taaf_shard_prefetch", str(BUNDLE_DIR / WEIGHTS_PREFETCH_HELPER))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PREFETCH_MODULE = module
    return PREFETCH_MODULE


def stop_weights_prefetch(keep_early: bool = False) -> None:
    global WEIGHTS_PREFETCH
    if keep_early and WEIGHTS_PREFETCH is not None and getattr(WEIGHTS_PREFETCH, "early_pending", False):
        return
    running, WEIGHTS_PREFETCH = WEIGHTS_PREFETCH, None
    if running is None:
        return
    try:
        running.stop()
    except Exception as exc:
        print(f"SHARD_PREFETCH error while stopping: {exc!r}", flush=True)


def early_read_threads() -> int:
    said = str(os.environ.get(EARLY_READ_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return 0
    if said in ("1", "true", "yes", "on"):
        base = str(os.environ.get(WEIGHTS_PREFETCH_ENV) or "").strip()
        return int(base) if base.isdigit() and int(base) > 1 else EARLY_READ_THREADS_DEFAULT
    if said.isdigit() and 2 <= int(said) <= 256:
        return int(said)
    raise RuntimeError(f"{EARLY_READ_ENV} must be 0, 1 (threads from {WEIGHTS_PREFETCH_ENV}) or a thread count "
                       f"from 2 through 256, found {said!r}")


def local_tier_mode() -> str:
    said = str(os.environ.get(LOCAL_TIER_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said in ("1", "true", "yes", "on", "relaunch"):
        return "relaunch"
    raise RuntimeError(f"{LOCAL_TIER_ENV} must be 0, 1 or relaunch, found {said!r}")


def cache_room_gib(limit_path: Path | None = None, meminfo: Path | None = None) -> dict[str, float] | None:
    limit_path = limit_path or HOST_KV_CGROUP_MAX
    meminfo = meminfo or HOST_MEMINFO
    try:
        said = limit_path.read_text(encoding="utf-8").strip()
    except OSError:
        said = ""
    machine = host_memtotal_gib(meminfo)
    caps = []
    if said.isdigit():
        caps.append(int(said) / 1024**3)
    if machine:
        caps.append(machine - HOST_SYSTEM_RESERVE_GIB)
    if not caps:
        return None
    limit = min(caps)
    anon = host_kv_ple_gib() + HOST_KV_OTHER_ANON_GIB + HOST_KV_HEADROOM_GIB + park_ram_gib()
    if _flag_on(HOST_KV_ENV):
        pinned, _ = _bounded_integer_environment(HOST_KV_GIB_ENV, default=HOST_KV_DEFAULT_GIB, minimum=1,
                                                 maximum=2048)
    else:
        pinned, _ = _bounded_integer_environment(KV_OFFLOADING_GIB_ENV, default=0, minimum=0,
                                                 maximum=MAX_KV_OFFLOADING_GIB)
    return {"limit_gib": limit, "anon_gib": float(anon), "pinned_gib": float(pinned),
            "load_room_gib": limit - anon, "pin_room_gib": limit - anon - pinned}


def early_read_window_bytes(room: dict[str, float] | None, df11_bytes: int) -> tuple[int, str]:
    said = str(os.environ.get(EARLY_READ_GIB_ENV) or "").strip()
    cap = float(said) if said else float(EARLY_READ_CAP_GIB)
    if not 1 <= cap <= 1024:
        raise RuntimeError(f"{EARLY_READ_GIB_ENV} must be from 1 through 1024, found {said!r}")
    if room is None:
        gib = min(cap, 24.0)
        why = "the cgroup limit is unreadable, so the window is the smaller of the cap and 24 GiB"
    else:
        gib = min(cap, max(room["load_room_gib"], 0.0))
        why = (f"cgroup {room['limit_gib']:.1f} GiB less {room['anon_gib']:.0f} GiB of PLE table, process memory, "
               f"headroom and park store leaves {room['load_room_gib']:.1f} GiB for page cache while the weights "
               f"load; the {room['pinned_gib']:.0f} GiB host pool is pinned after them, when the window is behind "
               f"the loader and dropped, leaving {room['pin_room_gib']:.1f} GiB beside it")
    return int(gib * 1024**3) + int(df11_bytes), f"{gib:.1f} GiB + {df11_bytes / 1024**3:.1f} GiB DF11 table, {why}"


def early_served_dir(model_dir: Path) -> Path | None:
    raw = os.environ.get(EXPERTS_KEPT_ENV)
    if raw is None or not str(raw).strip():
        return model_dir
    keep_n = int(str(raw).strip())
    if not 8 <= keep_n <= 512:
        raise ValueError(f"{EXPERTS_KEPT_ENV} must be 8..512, found {keep_n!r}")
    if keep_n == 512:
        return model_dir
    key = (str(model_dir), keep_n)
    if LINK_FARMS.get(key) is not None and LINK_FARMS[key].is_dir():
        return LINK_FARMS[key]
    if expert_cut_on():
        return None
    baked = _prebaked_for(keep_n, model_dir)
    if baked is None:
        return None
    LINK_FARMS[key] = _link_farm(baked, model_dir, keep_n)
    return LINK_FARMS[key]


def local_tier_root(mode: str) -> Path | None:
    if not mode:
        return None
    return Path(str(os.environ.get(LOCAL_TIER_DIR_ENV) or "").strip() or LOCAL_TIER_DIR_DEFAULT)


def tier_served(served: Path) -> Path:
    root = TIER["root"]
    if root is None or not (TIER["use"] or TIER["launches"] > 0):
        return served
    try:
        module = prefetch_module()
        index = module.read_stage_index(root)
        view = Path(root) / "view" / hashlib.sha1(str(served).encode("utf-8")).hexdigest()[:10]
        shutil.rmtree(view, ignore_errors=True)
        view.mkdir(parents=True)
        local = kept = 0
        for path in sorted(served.iterdir()):
            if not path.is_file():
                continue
            real = os.path.realpath(path)
            copy = module.staged_copy(root, real, index) if path.suffix == ".safetensors" else None
            os.symlink(copy or real, view / path.name)
            local += copy is not None
            kept += copy is None
        if not local:
            print(f"LOCAL_TIER no complete staged copy under {root}: the relaunch reads {served} as before", flush=True)
            return served
        print(f"LOCAL_TIER the relaunch loads {local} weight file(s) from {root} and {kept} other file(s) from "
              f"their original paths: {view}", flush=True)
        return view
    except Exception as failure:
        print(f"LOCAL_TIER error, the relaunch reads {served} as before: {type(failure).__name__}: {failure}",
              flush=True)
        return served


def tier_df11(env: dict[str, str]) -> None:
    root = TIER["root"]
    if root is None or not (TIER["use"] or TIER["launches"] > 0):
        return
    try:
        module = prefetch_module()
        _, rows = module.df11_rows(os.environ)
        if not rows:
            return
        index = module.read_stage_index(root)
        copies = [module.staged_copy(root, path, index) for path, _ in rows]
        if any(one is None for one in copies):
            print(f"LOCAL_TIER the DF11 table is not staged whole ({sum(one is not None for one in copies)} of "
                  f"{len(copies)} parts): the relaunch reads it from the mount", flush=True)
            return
        env[PLE_DF11_FILE_ENV] = re.sub(r"\.p[0-9]{3}$", "", copies[0])
        print(f"LOCAL_TIER the relaunch reads the DF11 table from {env[PLE_DF11_FILE_ENV]}", flush=True)
    except Exception as failure:
        print(f"LOCAL_TIER error, the DF11 table is read from the mount: {type(failure).__name__}: {failure}",
              flush=True)


def start_early_read(model_dir: Path, setup_started: float) -> None:
    global WEIGHTS_PREFETCH
    mode = local_tier_mode()
    threads = early_read_threads()
    if not threads:
        if mode:
            print(f"LOCAL_TIER {LOCAL_TIER_ENV} stages what the early read reads and {EARLY_READ_ENV} is off: "
                  f"nothing is staged", flush=True)
        return
    try:
        if load_dummy_enabled():
            print(f"EARLY_READ skipped: {LOAD_DUMMY_ENV} reads no shard", flush=True)
            return
        module = prefetch_module()
        served = early_served_dir(model_dir)
        if served is None:
            print("EARLY_READ the cut checkpoint is built by the startup prune, no prebake is mounted for this cut: "
                  "nothing to read ahead of the loader", flush=True)
            return
        df11_base, df11 = module.df11_rows(os.environ) if ple_df11_on() else (None, [])
        startup = ple_df11_startup_on()
        skip = PLE_DF11_MARK if df11 or startup else None
        if df11 or startup:
            os.environ[PREFETCH_SKIP_ENV] = PLE_DF11_MARK
            os.environ.setdefault(PLE_DF11_DROP_ENV, "1")
        main_rows = module.checkpoint_files(served, skip)
        draft_rows: list[Any] = []
        try:
            draft = draft_model_dir()
            if draft is not None:
                draft_rows = module.checkpoint_files(draft)
        except Exception as failure:
            print(f"EARLY_READ the drafter is not read ahead: {type(failure).__name__}: {failure}", flush=True)
        rows = [*df11, *main_rows, *draft_rows]
        if not rows:
            print("EARLY_READ no safetensors file found, nothing to read ahead", flush=True)
            return
        total = sum(size for _, size in rows)
        root = local_tier_root(mode)
        TIER["root"] = root
        stager = None
        reading = rows
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            probe_gib = float(str(os.environ.get(TIER_PROBE_GIB_ENV) or "").strip() or TIER_PROBE_GIB_DEFAULT)
            module.probe_tiers(max(rows, key=lambda row: row[1])[0], root, probe_gib, min(threads, 16),
                               report=lambda line: print(f"TIER_PROBE {line}", flush=True))
            have = module.Stager(root, rows).have
            if have:
                TIER["use"] = True
                view = tier_served(served)
                if view != served:
                    served = view
                    main_rows = module.checkpoint_files(served, skip)
                    local_df11 = []
                    if df11:
                        copies = [module.staged_copy(root, path) for path, _ in df11]
                        local_df11 = [(copy, size) for copy, (_, size) in zip(copies, df11)] \
                            if all(copies) else df11
                    reading = [*local_df11, *main_rows, *draft_rows]
                print(f"LOCAL_TIER {len(have)} of {len(rows)} files are staged complete under {root} by an earlier "
                      f"launch: this launch reads and loads those from the local disk", flush=True)
            else:
                free = shutil.disk_usage(str(root)).free / 1024**3
                if free < total / 1024**3 + TIER_STAGE_MARGIN_GIB:
                    print(f"LOCAL_TIER {root} has {free:.0f} GiB free for {total / 1024**3:.0f} GiB: nothing is staged",
                          flush=True)
                else:
                    stager = module.Stager(root, rows, report=lambda line: print(f"LOCAL_TIER {line}", flush=True))
        room = cache_room_gib()
        window, why = early_read_window_bytes(room, sum(size for _, size in df11))
        chunk_said = str(os.environ.get(WEIGHTS_PREFETCH_CHUNK_ENV) or "").strip()
        chunk = (int(chunk_said) if chunk_said.isdigit() and int(chunk_said) >= 1 else 16) << 20
        WEIGHTS_PREFETCH = module.launch_rows(reading, threads, window, chunk, stager=stager, served=str(served),
                                              early=True)
        print(f"EARLY_READ started {time.monotonic() - setup_started:.1f} s into the setup: {len(reading)} files, "
              f"{total / 1024**3:.1f} GiB ({len(df11)} DF11 table parts, {len(main_rows)} checkpoint files from "
              f"{served}, {len(draft_rows)} drafter files), {threads} threads, window {why}", flush=True)
    except Exception as failure:
        print(f"EARLY_READ error before the start, nothing is read ahead and the serve goes on: "
              f"{type(failure).__name__}: {failure}", flush=True)


def ple_df11_file_present(root: Path = Path("/kaggle/input")) -> bool:
    said = str(os.environ.get(PLE_DF11_FILE_ENV) or "").strip()
    if said.lower() in ("0", "false", "no", "off"):
        return False
    if said:
        base = Path(said)
        return base.is_file() or any(base.parent.glob(base.name + ".p[0-9][0-9][0-9]"))
    if not root.is_dir():
        return False
    for depth in range(5):
        pattern = "/".join(["*"] * depth + [PLE_DF11_FILE_NAME + "*"])
        if next(iter(root.glob(pattern)), None) is not None:
            return True
    return False


def ple_q6_present(root: Path = Path("/kaggle/input")) -> bool:
    said = str(os.environ.get(PLE_Q6_ENV) or "").strip().lower()
    if said in ("0", "off", "false", "no"):
        return False
    if said not in ("", "auto", "1", "on", "true"):
        raise RuntimeError(f"{PLE_Q6_ENV} must be unset, auto, 1 or 0, found {said!r}")
    if ple_qstore() != "int6":
        return False
    where = str(os.environ.get(PLE_Q6_DIR_ENV) or "").strip()
    root = Path(where) if where else root
    if not root.is_dir():
        return False
    for depth in range(6):
        pattern = "/".join(["*"] * depth + [PLE_Q6_FILE_NAME + ".part*of*"])
        if next(iter(root.glob(pattern)), None) is not None:
            return True
    return False


def ple_df11_startup_on() -> bool:
    return (ple_df11_on() and _flag_on(PLE_DF11_STARTUP_ENV) and not ple_df11_file_present()
            and not ple_q6_present())


def arm_ple_df11_reads(model_dir: Path, env: dict[str, str]) -> bool:
    if not ple_df11_on():
        return False
    env[PLE_DF11_CHECKPOINT_ENV] = str(model_dir)
    prebuilt = ple_q6_present()
    present = ple_df11_file_present() or prebuilt
    startup = ple_df11_startup_on()
    piece, _ = _bounded_integer_environment(PLE_Q6_PIECE_ENV, default=64, minimum=1, maximum=1024)
    if prebuilt:
        print(f"PLE_QSTORE prebuilt int6 parts ({PLE_Q6_FILE_NAME}.part*) are mounted, read in {piece} MiB pieces "
              f"({PLE_Q6_PIECE_ENV}): the loader reads them straight "
              f"into the pinned table and decodes no DF11", flush=True)
    if present or startup:
        os.environ[PREFETCH_SKIP_ENV] = PLE_DF11_MARK
    if startup:
        print(f"PLE_DF11 no precomputed table is mounted and {PLE_DF11_STARTUP_ENV} is on: the weight prefetch "
              f"leaves the PLE files unread and the table is encoded from {model_dir} by a background thread "
              f"beside the weight load", flush=True)
        return True
    print(f"PLE_DF11 {'a precomputed table is mounted' if present else 'no precomputed table is mounted'}: "
          f"{'the weight prefetch leaves the PLE files unread and the loader is asked for a table check' if present else 'the PLE table is encoded from the checkpoint as it loads'}; "
          f"checkpoint {model_dir}", flush=True)
    return present


def start_draft_prefetch(process: subprocess.Popen) -> None:
    global WEIGHTS_PREFETCH
    if str(os.environ.get(WEIGHTS_PREFETCH_ENV) or "").strip().lower() in ("", "0", "false", "no", "off"):
        return
    if load_dummy_enabled():
        return
    try:
        draft = draft_model_dir()
        if draft is None:
            return
        module = prefetch_module()
        files = module.checkpoint_files(draft)
        WEIGHTS_PREFETCH = module.launch([draft], os.environ, alive=lambda: process.poll() is None)
        print(f"DRAFT_PREFETCH the drafter's {len(files)} file(s), {sum(size for _, size in files) / 2**30:.2f} GiB "
              f"under {draft}, are read into the page cache beside the target's load, so the drafter's own load "
              f"finds them there; the server log's drafter 'Prefetching checkpoint files ... finished in' line is "
              f"what the drafter still waited", flush=True)
    except Exception as exc:
        print(f"DRAFT_PREFETCH error before the start, the drafter reads its files itself: {exc!r}", flush=True)


def start_weights_prefetch(served: Path, process: subprocess.Popen) -> None:
    global WEIGHTS_PREFETCH
    if cut_view_active(served):
        stop_weights_prefetch()
        print(f"EXPERT_CUT {served} is a cut view: the loader reads the kept experts' byte ranges itself and a "
              f"whole-file prefetch would read the experts it drops; no prefetch is started", flush=True)
        start_draft_prefetch(process)
        return
    early = WEIGHTS_PREFETCH
    if early is not None and getattr(early, "early_pending", False):
        if str(early.served) == str(served):
            early.alive = lambda: process.poll() is None
            early.early_pending = False
            print(f"EARLY_READ adopted by the server launch: {early.read / 1024**3:.1f} of {early.total / 1024**3:.1f} "
                  f"GiB already read, {early.ahead_bytes() / 1024**3:.1f} GiB ahead of the loader, "
                  f"{early.clock() - early.began:.0f} s after the first read; the usual prefetch is not started "
                  f"beside it", flush=True)
            return
        print(f"EARLY_READ covered {early.served} and the server loads {served}: the early read stops and the usual "
              f"prefetch, if armed, takes over", flush=True)
    stop_weights_prefetch()
    if str(os.environ.get(WEIGHTS_PREFETCH_ENV) or "").strip().lower() in ("", "0", "false", "no", "off"):
        return
    if load_dummy_enabled():
        print(f"LOAD_DUMMY {WEIGHTS_PREFETCH_ENV} skipped: no shard is read", flush=True)
        return
    try:
        module = prefetch_module()
        directories = [served]
        draft = draft_model_dir()
        if draft is not None:
            directories.append(draft)
        WEIGHTS_PREFETCH = module.launch(directories, os.environ, alive=lambda: process.poll() is None)
    except Exception as exc:
        print(f"SHARD_PREFETCH error before the start, no prefetch and the serve goes on: {exc!r}", flush=True)


def nsys_install(inputs: Path = Path("/kaggle/input"), root: Path = NSYS_ROOT) -> Path:
    import io
    import tarfile

    tool = root / NSYS_TOOL
    if tool.is_file():
        return tool
    found = sorted(inputs.rglob(NSYS_DEB)) if inputs.is_dir() else []
    if not found:
        raise FileNotFoundError(f"{NSYS_ENV} is set but no {NSYS_DEB} is mounted under {inputs}")
    data = found[0].read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != NSYS_DEB_SHA256:
        raise RuntimeError(f"{found[0]} is sha256 {digest[:16]}, not the pinned {NSYS_DEB_SHA256[:16]}")
    if data[:8] != b"!<arch>\n":
        raise RuntimeError(f"{found[0]} is not a Debian archive")
    offset = 8
    member = None
    while offset + 60 <= len(data):
        name = data[offset:offset + 16].decode("ascii", "replace").strip()
        size = int(data[offset + 48:offset + 58].decode("ascii").strip())
        if name.startswith("data.tar"):
            member = data[offset + 60:offset + 60 + size]
            break
        offset += 60 + size + (size % 2)
    if member is None:
        raise RuntimeError(f"{found[0]} holds no data.tar member")
    began = time.time()
    with tarfile.open(fileobj=io.BytesIO(member), mode="r:*") as archive:
        archive.extractall(root, members=[one for one in archive.getmembers()
                                          if one.name.lstrip("./").startswith("opt/nvidia/nsight-systems/")],
                           filter="tar")
    if not tool.is_file():
        raise RuntimeError(f"the Nsight Systems archive did not yield {tool}")
    tool.chmod(0o755)
    print(f"NSYS installed {tool} from {found[0]} in {time.time() - began:.0f}s", flush=True)
    return tool


def nsys_wrap(command: list[str]) -> list[str]:
    if not _flag_on(NSYS_ENV):
        return command
    tool = nsys_install()
    try:
        status = subprocess.run([str(tool), "status", "--environment"], capture_output=True, text=True, timeout=120)
        for line in (status.stdout + status.stderr).splitlines():
            print(f"NSYS environment: {line}", flush=True)
    except (OSError, subprocess.SubprocessError) as failure:
        print(f"NSYS environment check failed: {type(failure).__name__}: {failure}", flush=True)
    try:
        ended = subprocess.run([str(tool), "shutdown"], capture_output=True, text=True, timeout=120)
        said = (ended.stdout + ended.stderr).strip().replace("\n", " ")[-200:]
        print(f"NSYS shutdown of any earlier default session exited {ended.returncode}: {said}", flush=True)
    except (OSError, subprocess.SubprocessError) as failure:
        print(f"NSYS shutdown of any earlier default session failed: {type(failure).__name__}: {failure}",
              flush=True)
    (WORKING_DIR / "nsys_bin.txt").write_text(str(tool), encoding="utf-8")
    (WORKING_DIR / "nsys_session.txt").write_text(NSYS_SESSION, encoding="utf-8")
    wrapped = [str(tool), "launch", f"--session-new={NSYS_SESSION}", *NSYS_LAUNCH, *command]
    print(f"NSYS the server runs under Nsight Systems session {NSYS_SESSION} with {' '.join(NSYS_LAUNCH)}; "
          f"collection starts only on nsys start --session={NSYS_SESSION}", flush=True)
    return wrapped


def experts_per_tok_override(model_dir: Path) -> dict[str, Any]:
    wanted, _ = _bounded_integer_environment(EXPERTS_PER_TOK_ENV, default=0, minimum=0, maximum=64)
    if not wanted:
        return {}
    config_path = Path(model_dir) / "config.json"
    held = read_json(config_path) if config_path.is_file() else {}
    nested = isinstance(held.get("text_config"), dict)
    stock = (held.get("text_config") if nested else held).get("num_experts_per_tok")
    if isinstance(stock, int) and wanted > stock:
        raise RuntimeError(f"{EXPERTS_PER_TOK_ENV}={wanted} asks for more experts a token than the checkpoint's "
                           f"{stock}; it only narrows the routing")
    override = {"text_config": {"num_experts_per_tok": wanted}} if nested else {"num_experts_per_tok": wanted}
    print(f"EXPERTS_PER_TOK: the target routes each token to its top {wanted} experts instead of the "
          f"checkpoint's {stock}, renormalized as the router does; the draft model keeps its own "
          f"config (--hf-overrides {json.dumps(override)})", flush=True)
    return override


def chat_template_path(model_dir: Path) -> Path:
    stock = Path(model_dir) / "chat_template.jinja"
    open_think, stable = _flag_on(OPEN_THINK_ENV), _flag_on(STABLE_THINK_ENV)
    if not open_think and not stable:
        return stock
    text = stock.read_text(encoding="utf-8")
    if open_think:
        found = text.count(OPEN_THINK_FROM)
        if found != 1:
            raise RuntimeError(f"{OPEN_THINK_ENV} cannot apply: {stock} has {found} assistant branches, not 1")
        text = text.replace(OPEN_THINK_FROM, OPEN_THINK_TO, 1)
    if stable:
        found = text.count(STABLE_THINK_FROM)
        if found != 1:
            raise RuntimeError(f"{STABLE_THINK_ENV} cannot apply: {stock} has {found} reasoning-instruction "
                               f"conditions, not 1")
        text = text.replace(STABLE_THINK_FROM, STABLE_THINK_TO, 1)
    out = Path(tempfile.gettempdir()) / "taaf_chat_template_open_think.jinja"
    out.write_text(text, encoding="utf-8")
    if open_think:
        print(f"OPEN THINK template: {out}; a request whose last message is the assistant's and whose "
              f"chat_template_kwargs carry open_last_thinking renders that message as an unclosed think block, "
              f"every other request renders as {stock.name} does", flush=True)
    if stable:
        print(f"STABLE THINK SYSTEM template: {out}; the reasoning-effort line opens the system message whether or "
              f"not a request turns thinking off, so a request with enable_thinking false keeps the prefix of its "
              f"session and differs only in the empty think block it ends with", flush=True)
    return out


def server_command(
    model_dir: Path,
    *,
    enable_chunked_prefill: bool = True,
    tuning: dict[str, Any] | None = None,
) -> list[str]:
    selfcheck_fatal()
    mamba_hit_check_on()
    if not isinstance(enable_chunked_prefill, bool):
        raise TypeError("enable_chunked_prefill must be a bool.")
    if _flag_on(KV_REBASE_ENV):
        raise RuntimeError(_kv_rebase_refusal())
    resolved_tuning = tuning or resolve_vllm_tuning(
        enable_chunked_prefill=enable_chunked_prefill
    )
    if resolved_tuning.get("enable_chunked_prefill") is not enable_chunked_prefill:
        raise RuntimeError("Resolved vLLM tuning has the wrong chunked-prefill mode.")
    chunked_prefill_flag = (
        "--enable-chunked-prefill"
        if enable_chunked_prefill
        else "--no-enable-chunked-prefill"
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_dir),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--load-format",
        "dummy" if load_dummy_enabled() else "safetensors",
        "--dtype",
        "bfloat16",
        "--quantization",
        quantization(),
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
    ]
    if resolved_tuning["moe_backend"] is not None:
        command.extend(["--moe-backend", str(resolved_tuning["moe_backend"])])
    if _flag_on(RETURN_ROUTED_EXPERTS_ENV):
        command.append("--enable-return-routed-experts")
    if int(resolved_tuning["kv_cache_memory_bytes"]) > 0:
        command.extend(
            [
                "--kv-cache-memory-bytes",
                str(resolved_tuning["kv_cache_memory_bytes"]),
            ]
        )
    else:
        command.extend(
            ["--gpu-memory-utilization",
             f"{int(resolved_tuning['gpu_memory_permille']) / 1000:.3f}"]
        )
    command.extend(
        [
            "--max-model-len",
            str(ANALYZER_CONTEXT),
            "--max-num-seqs",
            str(resolved_tuning["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(resolved_tuning["max_num_batched_tokens"]),
            "--async-scheduling",
            chunked_prefill_flag,
        ]
    )
    if resolved_tuning["kv_cache_dtype"] != "auto":
        command.extend(
            ["--kv-cache-dtype", str(resolved_tuning["kv_cache_dtype"])]
        )
    command.extend(serving_flags())
    image_max_pixels, _ = _bounded_integer_environment(
        IMAGE_MAX_PIXELS_ENV, default=0, minimum=0, maximum=64 * 1024 * 1024)
    if image_max_pixels > 0:
        command.extend(
            ["--mm-processor-kwargs", json.dumps({"max_pixels": image_max_pixels})]
        )
    if _flag_on(NO_VIDEO_ENV):
        command.extend(["--limit-mm-per-prompt", json.dumps({"video": 0})])
    mm_cache_gib, mm_cache_set = _bounded_integer_environment(
        MM_CACHE_GIB_ENV, default=4, minimum=0, maximum=16)
    if mm_cache_set:
        command.extend(["--mm-processor-cache-gb", str(mm_cache_gib)])
    per_tok = experts_per_tok_override(model_dir)
    if per_tok:
        command.extend(["--hf-overrides", json.dumps(per_tok)])
    if _flag_on(TRACE_REPLAY_ENV):
        command.append("--enable-trace-replay")
    largest, _ = piecewise_large(resolved_tuning)
    if _flag_on(ENFORCE_EAGER_ENV):
        command.append("--enforce-eager")
    elif int(resolved_tuning["max_cudagraph_capture_size"]) > 0:
        command.extend(
            [
                "--max-cudagraph-capture-size",
                str(largest or resolved_tuning["max_cudagraph_capture_size"]),
            ]
        )
        sizes = cudagraph_capture_sizes(resolved_tuning)
        start, stride = capture_sparse(resolved_tuning)
        if stride:
            whole = sorted(set(size for size in CUDAGRAPH_CAPTURE_SIZES
                               if size <= int(resolved_tuning["max_cudagraph_capture_size"]))
                           | set(size for size in sizes
                                 if size > int(resolved_tuning["max_cudagraph_capture_size"])))
            dropped = sorted(set(whole) - set(sizes))
            print(f"CAPTURE_SPARSE ({CAPTURE_SPARSE_ENV}={start}:{stride}): {len(sizes)} capture sizes in "
                  f"place of {len(whole)}; above {start} tokens only multiples of {stride} (and of the "
                  f"mixed-bucket step) are captured, so {len(dropped)} sizes are dropped from every graph "
                  f"manager and a step there pads to the next kept size; dropped "
                  f"{','.join(str(size) for size in dropped) or 'none'}", flush=True)
        if sizes:
            command.extend(["--cudagraph-capture-sizes", *(str(size) for size in sizes)])
    keep = _flag_on(CACHE_KEEP_ENV) and bool(resolved_tuning["enable_prefix_caching"])
    junction = _flag_on(MAMBA_JUNCTION_ENV)
    host_kv = host_kv_settings()
    if host_kv is not None:
        resolved_tuning = {**resolved_tuning, "kv_offloading_gib": 0}
        command.extend(["--kv-transfer-config", json.dumps(
            {"kv_connector": "HiSparseConnector", "kv_role": "kv_both",
             "kv_connector_extra_config": {"host_pool_gib": host_kv["host_pool_gib"]}}, separators=(",", ":"))])
    disk = kv_disk_config(int(resolved_tuning.get("kv_offloading_gib") or 0))
    if junction and not keep:
        raise RuntimeError(
            f"{MAMBA_JUNCTION_ENV} asks vLLM to keep a Mamba state where a request's attention hit "
            f"outran its Mamba hit, which needs {CACHE_KEEP_ENV} and prefix caching on for the "
            f"{cache_keep_match_unit()}-token match unit and the retained copies it is built on.")
    command.extend(
        [
            (
                "--enable-prefix-caching"
                if resolved_tuning["enable_prefix_caching"]
                else "--no-enable-prefix-caching"
            ),
            *(
                [
                    "--kv-offloading-size",
                    str(int(resolved_tuning["kv_offloading_gib"])),
                    "--kv-offloading-backend",
                    "native",
                    *(["--prefix-match-unit", str(PREFIX_MATCH_UNIT)]
                      if PREFIX_MATCH_UNIT > 0 and not keep else []),
                    *(["--kv-transfer-config", json.dumps({"kv_connector_extra_config": disk})]
                      if disk else []),
                ]
                if int(resolved_tuning.get("kv_offloading_gib") or 0) > 0
                else []
            ),
            *(["--prefix-match-unit", str(cache_keep_match_unit()),
               "--prefix-cache-retention-interval", str(cache_keep_retention())] if keep else []),
            *(["--enable-mamba-fine-grained-prefix-cache"] if junction else []),
            *(["--enable-prompt-tokens-details"] if PROMPT_TOKENS_DETAILS else []),
            *(["--prefix-cache-retention-interval", str(PREFIX_RETENTION)]
              if PREFIX_RETENTION > 0 and resolved_tuning["enable_prefix_caching"]
              and not keep else []),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--chat-template",
            str(chat_template_path(model_dir)),
        ]
    )
    if int(resolved_tuning["mtp_speculative_tokens"]) > 0:
        speculative_config: dict[str, Any] = {
            "method": "mtp",
            "num_speculative_tokens": resolved_tuning[
                "mtp_speculative_tokens"
            ],
        }
        if resolved_tuning["mtp_index_share_for_iteration"]:
            speculative_config["index_share_for_mtp_iteration"] = True
        if resolved_tuning["mtp_dynamic_batch_schedule"] is not None:
            speculative_config["num_speculative_tokens_per_batch_size"] = (
                resolved_tuning["mtp_dynamic_batch_schedule"]
            )
        draft = draft_model_dir()
        if draft is not None:
            speculative_config["model"] = str(draft)
            print(f"DRAFT_MODEL: the MTP drafter loads from {draft} with the quantization its own config "
                  f"names, not the target's own MTP layer", flush=True)
        if _flag_on(DRAFT_SAMPLE_ENV):
            speculative_config["draft_sample_method"] = "probabilistic"
            print(f"{DRAFT_SAMPLE_ENV} armed: the MTP drafter samples its drafts at the request's temperature from "
                  f"its top-k and the rejection sampler tests p/q against that distribution"
                  + (f"; {DRAFT_VOCAB_ENV} rows feed the drafter without local argmax"
                     if _flag_on(DRAFT_VOCAB_ENV) else "")
                  + ((f"; {SAMPLE_TOPK_DRAFT_ENV} verifies these steps over the compact rows"
                      if _flag_on(SAMPLE_TOPK_DRAFT_ENV) else
                      f"; the compact verification of {SAMPLE_TOPK_ENV} stands aside for these steps")
                     if sample_topk_count() else ""), flush=True)
        if _knob_value(BLOCK_VERIFY_ENV):
            speculative_config["rejection_sample_method"] = "block"
            print(f"{BLOCK_VERIFY_ENV} armed: vLLM verifies each request's sampled drafts as one block "
                  f"(rejection_sample_method block); greedy requests keep the argmax test", flush=True)
        if _flag_on(DRAFT_VOCAB_ENV) and not _flag_on(DRAFT_SAMPLE_ENV):
            speculative_config["use_local_argmax_reduction"] = True
        if _flag_on(ADAPTIVE_VERIFY_ENV):
            speculative_config["enable_adaptive_verification"] = True
            if DRAFT_BUDGET_VARLEN["on"]:
                print(f"{DRAFT_BUDGET_ENV} armed: vLLM verifies a fixed row total a step and splits it over the "
                      f"lanes by the drafter's recorded token probabilities", flush=True)
            else:
                print(f"{ADAPTIVE_VERIFY_ENV} armed: vLLM trims each request's MTP drafts by its "
                      f"calibrated acceptance and the step-time cost table"
                      + (f"; the acceptance estimate reads the logits of the {DRAFT_VOCAB_ENV} rows"
                         if _flag_on(DRAFT_VOCAB_ENV) else ""), flush=True)
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    speculative_config,
                    separators=(",", ":"),
                ),
            ]
        )
    spec_metrics = _chosen_environment(SPEC_METRICS_ENV, SPEC_METRICS_LEVELS)
    if spec_metrics:
        if int(resolved_tuning["mtp_speculative_tokens"]) <= 0:
            raise RuntimeError(f"{SPEC_METRICS_ENV}={spec_metrics} needs speculation on; MTP tokens is 0")
        if spec_metrics != "none":
            command.extend(["--per-request-spec-decode-metrics", spec_metrics])
            print(f"{SPEC_METRICS_ENV} armed: each response carries metrics.speculative_decoding ({spec_metrics})",
                  flush=True)
    command.extend(
        [
            "--no-enable-log-requests",
            "--disable-uvicorn-access-log",
            "--uvicorn-log-level",
            "info",
        ]
    )
    if _flag_on(WEIGHT_CACHE_ENV):
        if load_dummy_enabled():
            raise RuntimeError(f"{LOAD_DUMMY_ENV} loads no checkpoint and {WEIGHT_CACHE_ENV} serves one from a cache daemon")
        command[command.index("--load-format") + 1] = "ipc_cache"
        command.extend(["--model-loader-extra-config", weight_cache_extra_config()])
    return command


def start_server(
    model_dir: Path,
    env: dict[str, str],
    *,
    enable_chunked_prefill: bool = True,
    tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleanup = stop_owned_server()
    if not cleanup.get("stopped", False):
        raise RuntimeError(f"A prior owned vLLM server did not stop: {cleanup}")
    if port_open():
        raise RuntimeError(f"Port {HOST}:{PORT} is already in use by an unowned process.")
    rotate_launch_logs([SERVER_LOG])
    SERVER_LOG.unlink(missing_ok=True)
    resolved_tuning = tuning or resolve_vllm_tuning(
        enable_chunked_prefill=enable_chunked_prefill
    )
    served = tier_served(prune_model_dir(model_dir))
    phase("prune or link done")
    pending = SELF_CHECKS["pending"]
    if (selfcheck_overlap_mode() == "server" and pending is not None
            and (pending.get("qsa_first") or not qsa_gates_on())):
        predict_verdict_env(env)
        print("SELFCHECK_OVERLAP the server launches before the checks end; a failed check stops it", flush=True)
    else:
        join_self_checks(env)
    arm_ple_df11_reads(model_dir, env)
    tier_df11(env)
    command = nsys_wrap(server_command(
        served,
        enable_chunked_prefill=enable_chunked_prefill,
        tuning=resolved_tuning,
    ))
    print("VLLM_START_COMMAND", json.dumps(command), flush=True)
    report_space("before-launch")
    join_pyc_seed("the server launch")
    if init_trace_on():
        env[INIT_TRACE_PRINT_ENV] = "1"
    log_handle = SERVER_LOG.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    phase("server process started", pid=process.pid)
    if init_trace_on():
        print(launch_timing_line(SETUP_MARKS["server process started"]), flush=True)
    start_weights_prefetch(served, process)
    TIER["launches"] += 1
    deadline = time.monotonic() + 60
    root_record = None
    while time.monotonic() < deadline:
        root_record = proc_record(process.pid)
        if process.poll() is not None:
            break
        if (
            root_record is not None
            and root_record["pgid"] == process.pid
            and root_record["sid"] == process.pid
            and root_record["cmdline"]
        ):
            break
        time.sleep(0.05)
    if process.poll() is not None:
        tail = ""
        try:
            tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            pass
        report_space("after-death")
        raise RuntimeError(
            f"The vLLM root exited before it could be identified, code "
            f"{process.poll()}, after writing {len(tail)} characters: {tail[-1500:]}"
        )
    if (root_record is None or root_record["pgid"] != process.pid
            or root_record["sid"] != process.pid or not root_record["cmdline"]):
        process.terminate()
        raise RuntimeError(
            f"vLLM root did not enter its isolated process session with an argv "
            f"of its own within 60s: {root_record}"
        )
    argv_sha256 = hashlib.sha256(
        json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if root_record["argv_sha256"] != argv_sha256:
        report_space("after-death")
        tail = ""
        try:
            tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            pass
        print("VLLM_ROOT_RECORD", json.dumps(root_record), flush=True)
        print("VLLM_ROOT_EXIT", json.dumps(
            {"poll": process.poll(), "state": root_record.get("state"),
             "log_chars": len(tail)}), flush=True)
        print("VLLM_ROOT_LOG_TAIL", tail, flush=True)
        for source in ("/dev/kmsg", "/var/log/kern.log"):
            item = Path(source)
            if not item.is_file() and source != "/dev/kmsg":
                continue
            try:
                lines = [line for line in
                         item.read_text(encoding="utf-8", errors="replace").splitlines()
                         if "Kill" in line or "oom" in line.lower()]
                if lines:
                    print(f"VLLM_ROOT_KERNEL {source} " + " | ".join(lines[-6:]),
                          flush=True)
            except OSError:
                pass
        process.terminate()
        raise RuntimeError(
            "The live vLLM root argv did not match the exact launched command: "
            f"{root_record['argv_sha256']} != {argv_sha256}"
        )
    started_epoch = time.time()
    root_worker = {
        "pid": int(root_record["pid"]),
        "start_ticks": int(root_record["start_ticks"]),
        "ppid": int(root_record["ppid"]),
        "pgid": int(root_record["pgid"]),
        "sid": int(root_record["sid"]),
        "cmdline_sha256": str(root_record["cmdline_sha256"]),
    }
    identity = {
        "schema_version": 1,
        "backend": "vllm",
        "boot_id": boot_id(),
        "host": HOST,
        "port": PORT,
        "pid": process.pid,
        "start_ticks": int(root_record["start_ticks"]),
        "pgid": int(root_record["pgid"]),
        "sid": int(root_record["sid"]),
        "argv": command,
        "argv_sha256": argv_sha256,
        "vllm_tuning": resolved_tuning,
        "started_epoch": started_epoch,
        "phase": "starting",
        "snapshot_epoch": started_epoch,
        "ready_epoch": None,
        "workers": [root_worker],
    }
    write_json(SERVER_IDENTITY, identity)
    return identity


def update_ready_identity(identity: dict[str, Any]) -> dict[str, Any]:
    if identity.get("boot_id") != boot_id():
        raise RuntimeError("vLLM identity boot ID changed before worker snapshot.")
    root = proc_record(int(identity["pid"]))
    if (
        root is None
        or int(root["start_ticks"]) != int(identity["start_ticks"])
        or int(root["pgid"]) != int(identity["pgid"])
        or int(root["sid"]) != int(identity["sid"])
    ):
        raise RuntimeError(f"vLLM root identity changed before worker snapshot: {root}")
    table = process_table()
    descendants = descendant_ids(int(identity["pid"]), table)
    detached = [
        table[pid]
        for pid in sorted(descendants)
        if pid in table
        and int(table[pid]["sid"]) != int(identity["sid"])
        and allowed_detached_worker(
            table[pid], table, int(identity["pgid"]), int(identity["sid"])
        )
    ]
    escaped = [
        table[pid]
        for pid in sorted(descendants)
        if pid in table
        and int(table[pid]["sid"]) != int(identity["sid"])
        and not allowed_detached_worker(
            table[pid], table, int(identity["pgid"]), int(identity["sid"])
        )
    ]
    if escaped:
        raise RuntimeError(f"Owned vLLM descendants escaped the isolated session: {escaped}")
    owned_by_pid = {
        int(record["pid"]): record
        for record in table.values()
        if int(record["sid"]) == int(identity["sid"])
    }
    owned_by_pid.update({int(record["pid"]): record for record in detached})
    owned = list(owned_by_pid.values())
    if not any(int(record["pid"]) == int(identity["pid"]) for record in owned):
        raise RuntimeError("The vLLM root was absent from its post-readiness session snapshot.")
    workers = [
        {
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "ppid": int(record["ppid"]),
            "pgid": int(record["pgid"]),
            "sid": int(record["sid"]),
            "cmdline_sha256": str(record["cmdline_sha256"]),
        }
        for record in sorted(owned, key=lambda item: int(item["pid"]))
    ]
    updated = dict(identity)
    now = time.time()
    updated["phase"] = "ready"
    updated["snapshot_epoch"] = now
    updated["ready_epoch"] = identity.get("ready_epoch") or now
    updated["workers"] = workers
    write_json(SERVER_IDENTITY, updated)
    return updated


def _bounded_timeout(setup_deadline: float, requested: int) -> int:
    remaining = setup_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"The {SETUP_TOTAL_TIMEOUT}s setup/preflight ceiling expired.")
    return max(1, min(requested, math.ceil(remaining)))


FELL_BACK = (
    "VLLM_OFFLOAD_FALLBACK the serve refused {gib} GiB of host KV offloading "
    "({kind}: {detail}); {left:.0f}s of setup clock remain, so the same "
    "command is launched again with offloading off. The run that follows is "
    "NOT the configuration that was asked for."
)

NO_ROOM = (
    "VLLM_OFFLOAD_FALLBACK the serve refused {gib} GiB of host KV offloading "
    "and only {left:.0f}s of setup clock remain, which is under the "
    "{needed:.0f}s a second launch needs. Nothing is retried."
)

FALLBACK_FLOOR_S = 420.0


def kv_disk_config(offloading_gib: int) -> dict[str, Any] | None:
    said = str(os.environ.get(KV_DISK_GIB_ENV) or "").strip()
    if not said or said == "0":
        return None
    try:
        gib = int(said)
    except ValueError:
        gib = -1
    if gib <= 0:
        raise RuntimeError(f"{KV_DISK_GIB_ENV} must be a whole number of GiB, found {said!r}")
    if offloading_gib <= 0:
        raise RuntimeError(
            f"{KV_DISK_GIB_ENV}={gib} moves vLLM's host KV tier onto a disk file, which needs the "
            f"tier itself on: {KV_OFFLOADING_GIB_ENV} is {offloading_gib}.")
    folder = Path(str(os.environ.get(KV_DISK_DIR_ENV) or "").strip() or KV_DISK_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < (gib + KV_DISK_SPARE_GIB) * (1 << 30):
        raise RuntimeError(
            f"{KV_DISK_GIB_ENV}={gib} needs {gib + KV_DISK_SPARE_GIB} GiB free under {folder}, "
            f"found {free / (1 << 30):.0f} GiB")
    return {"kv_offload_backend": "disk", "disk_path": str(folder / KV_DISK_FILE),
            "disk_capacity_bytes": gib * (1 << 30),
            "use_page_cache": _flag_on(KV_DISK_PAGE_CACHE_ENV)}


def serve_or_fall_back(
    model_dir: Path,
    env: dict[str, str],
    tuning: dict[str, Any],
    setup_deadline: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Launch the server; if host KV offloading is what it refuses, drop it."""
    try:
        identity = start_server(model_dir, env, tuning=tuning)
        return identity, wait_for_server(identity, setup_deadline), tuning
    except Exception as exc:
        asked = int(tuning.get("kv_offloading_gib") or 0)
        if asked <= 0:
            raise
        left = setup_deadline - time.monotonic()
        if left < FALLBACK_FLOOR_S:
            print(
                NO_ROOM.format(gib=asked, left=left, needed=FALLBACK_FLOOR_S),
                flush=True,
            )
            raise
        print(
            FELL_BACK.format(
                gib=asked,
                kind=type(exc).__name__,
                detail=str(exc)[:300],
                left=left,
            ),
            flush=True,
        )
        stop_owned_server()
        dropped = os.environ.pop(KV_DISK_GIB_ENV, None)
        if dropped:
            print(f"{KV_DISK_GIB_ENV}={dropped} goes with the host tier: the fallback serve keeps no "
                  f"KV file on disk either", flush=True)
        plain = dict(tuning)
        plain["kv_offloading_gib"] = 0
        plain["overridden"] = dict(plain.get("overridden") or {})
        plain["overridden"][KV_OFFLOADING_GIB_ENV] = False
        identity = start_server(model_dir, env, tuning=plain)
        return identity, wait_for_server(identity, setup_deadline), plain


def wait_for_server(identity: dict[str, Any], setup_deadline: float) -> dict[str, Any]:
    deadline = min(time.monotonic() + SERVER_READY_TIMEOUT, setup_deadline)
    started = time.monotonic()
    last_notice = -1
    while time.monotonic() < deadline:
        if not alive(int(identity["pid"]), int(identity["start_ticks"])):
            raise RuntimeError(f"vLLM exited before readiness.\n{tail_log(300)}")
        try:
            models = request_json(
                f"{BASE_URL}/models", timeout=_bounded_timeout(setup_deadline, 10)
            )
            rows = models.get("data") or []
            ids = [row.get("id") for row in rows if isinstance(row, dict)]
            if ids != [SERVED_MODEL_NAME]:
                raise RuntimeError(f"vLLM served the wrong model identity: {ids}")
            stop_weights_prefetch()
            phase("server answered /models")
            return {"models": models, "ready_seconds": time.monotonic() - started}
        except Exception:
            elapsed = int(time.monotonic() - started)
            if elapsed // 60 != last_notice:
                last_notice = elapsed // 60
                print(f"VLLM_WAIT elapsed_s={elapsed}\n{tail_log(30)}", flush=True)
            time.sleep(ready_poll_seconds())
    stop_weights_prefetch()
    raise TimeoutError(f"Timed out waiting for vLLM after {SERVER_READY_TIMEOUT}s.\n{tail_log(300)}")


def message_text(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    return "\n".join(
        str(value or "")
        for value in (
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("content"),
        )
    ).strip()


def completion_tokens(response: dict[str, Any]) -> int:
    return int((response.get("usage") or {}).get("completion_tokens") or 0)


def chat(
    payload: dict[str, Any], setup_deadline: float, timeout: int = 900
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = request_json(
        f"{BASE_URL}/chat/completions",
        payload=payload,
        timeout=_bounded_timeout(setup_deadline, timeout),
    )
    return response, time.monotonic() - started


def base_payload(prompt: Any, max_tokens: int, enable_thinking: bool) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }


def red_blue_png_data_url() -> str:
    width = height = 8
    rows = []
    for _ in range(height):
        rows.append(b"\x00" + b"\xff\x00\x00" * 4 + b"\x00\x00\xff" * 4)
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _parse_metric_sum(metrics: str, metric: str) -> float:
    total = 0.0
    found = False
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([^\s]+)\s*$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if match:
            value = float(match.group(1))
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric value for {metric}: {value}")
            total += value
            found = True
    if not found:
        raise RuntimeError(f"Required vLLM metric was absent: {metric}")
    return total


def run_preflights(setup_deadline: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = (
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    baseline_metrics = request_bytes(
        f"http://{HOST}:{PORT}/metrics",
        timeout=_bounded_timeout(setup_deadline, 30),
    ).decode("utf-8", errors="replace")
    baseline_counters = {
        metric: _parse_metric_sum(baseline_metrics, metric) for metric in metric_names
    }
    text_response, text_seconds = chat(
        base_payload("Answer with only: 4", 32, False), setup_deadline
    )
    text_content = str(text_response["choices"][0]["message"].get("content") or "").strip()
    if completion_tokens(text_response) <= 0 or text_content != "4":
        raise RuntimeError(f"Text preflight was not exact: {text_response}")
    result["text"] = {
        "seconds": text_seconds,
        "completion_tokens": completion_tokens(text_response),
        "text": text_content,
    }

    tool_payload = base_payload("Call submit_number with value 4 now. Do not answer in prose.", 128, False)
    tool_payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "submit_number",
                "description": "Submit one integer.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        }
    ]
    tool_payload["tool_choice"] = "required"
    tool_response, tool_seconds = chat(tool_payload, setup_deadline)
    tool_message = tool_response["choices"][0]["message"]
    calls = tool_message.get("tool_calls") or []
    if not calls:
        raise RuntimeError(f"Tool preflight produced no native structured call: {tool_response}")
    function = calls[0].get("function") or {}
    raw_arguments = function.get("arguments") or {}
    arguments = raw_arguments if isinstance(raw_arguments, dict) else json.loads(str(raw_arguments))
    if function.get("name") != "submit_number" or int(arguments.get("value", -1)) != 4:
        raise RuntimeError(f"Wrong native tool preflight call: {calls}")
    if tool_response["choices"][0].get("finish_reason") != "tool_calls":
        raise RuntimeError(f"Tool preflight had the wrong finish reason: {tool_response}")
    result["tool"] = {
        "seconds": tool_seconds,
        "completion_tokens": completion_tokens(tool_response),
        "structured": True,
        "calls": calls,
        "finish_reason": "tool_calls",
    }

    vision_prompt = [
        {"type": "text", "text": "Name one color visible in this image. Answer with one word."},
        {"type": "image_url", "image_url": {"url": red_blue_png_data_url()}},
    ]
    vision_response, vision_seconds = chat(
        base_payload(vision_prompt, 32, False), setup_deadline
    )
    vision_text = str(vision_response["choices"][0]["message"].get("content") or "").strip()
    if completion_tokens(vision_response) <= 0 or not any(
        color in vision_text.lower() for color in ("red", "blue")
    ):
        raise RuntimeError(f"Vision preflight did not identify the known image: {vision_response}")
    result["vision"] = {
        "seconds": vision_seconds,
        "completion_tokens": completion_tokens(vision_response),
        "text": vision_text[:500],
        "image": "generated 8x8 half-red half-blue PNG data URL",
    }

    warm_response, warm_seconds = chat(
        base_payload("Print the word velocity followed by a space repeatedly until stopped.", 512, False),
        setup_deadline,
    )
    warm_tokens = completion_tokens(warm_response)
    if warm_tokens < 128:
        raise RuntimeError(f"C1 throughput preflight ended too early: {warm_tokens} tokens")
    c1_rate = warm_tokens / max(warm_seconds, 1e-6)
    result["c1_throughput"] = {
        "seconds": warm_seconds,
        "completion_tokens": warm_tokens,
        "completion_tokens_per_second": c1_rate,
        "finish_reason": warm_response["choices"][0].get("finish_reason"),
    }

    def one_concurrent(index: int) -> dict[str, Any]:
        response, seconds = chat(
            base_payload(
                f"Stream {index}: print the word arc followed by a space repeatedly until stopped.",
                256,
                False,
            ),
            setup_deadline,
        )
        tokens = completion_tokens(response)
        if tokens < 64:
            raise RuntimeError(f"C4 preflight {index} ended too early: {tokens}")
        return {
            "index": index,
            "seconds": seconds,
            "completion_tokens": tokens,
            "completion_tokens_per_second": tokens / max(seconds, 1e-6),
        }

    c4_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        c4_rows = list(executor.map(one_concurrent, range(4)))
    c4_seconds = time.monotonic() - c4_started
    c4_tokens = sum(row["completion_tokens"] for row in c4_rows)
    c4_rate = c4_tokens / max(c4_seconds, 1e-6)
    if c4_rate < 1.5 * c1_rate:
        raise RuntimeError(f"C4 aggregate rate did not reach 1.5x C1: {c4_rate} < {1.5*c1_rate}")
    weak = [row for row in c4_rows if row["completion_tokens_per_second"] < 0.35 * c1_rate]
    if weak:
        raise RuntimeError(f"One or more C4 lanes fell below 0.35x C1: {weak}")
    result["c4_throughput"] = {
        "seconds": c4_seconds,
        "completion_tokens": c4_tokens,
        "aggregate_completion_tokens_per_second": c4_rate,
        "aggregate_over_c1": c4_rate / c1_rate,
        "requests": c4_rows,
    }

    metrics = request_bytes(
        f"http://{HOST}:{PORT}/metrics",
        timeout=_bounded_timeout(setup_deadline, 30),
    ).decode("utf-8", errors="replace")
    METRICS_PATH.write_text(metrics, encoding="utf-8")
    after_counters = {metric: _parse_metric_sum(metrics, metric) for metric in metric_names}
    deltas = {
        metric: after_counters[metric] - baseline_counters[metric] for metric in metric_names
    }
    drafts = deltas["vllm:spec_decode_num_drafts_total"]
    draft_tokens = deltas["vllm:spec_decode_num_draft_tokens_total"]
    accepted_tokens = deltas["vllm:spec_decode_num_accepted_tokens_total"]
    if drafts <= 0 or draft_tokens <= 0 or accepted_tokens <= 0:
        raise RuntimeError(
            "Native MTP telemetry deltas were not positive: "
            f"drafts={drafts} draft_tokens={draft_tokens} accepted={accepted_tokens}"
        )
    result["native_mtp"] = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens,
        "baseline_counters": baseline_counters,
        "after_counters": after_counters,
        "counter_deltas": deltas,
        "baseline_metrics_sha256": hashlib.sha256(
            baseline_metrics.encode("utf-8")
        ).hexdigest(),
        "metrics_sha256": sha256_file(METRICS_PATH),
    }
    return result


def capture_process_and_gpu(identity: dict[str, Any]) -> dict[str, Any]:
    table = process_table()
    owned_ids = {
        pid for pid, record in table.items() if int(record["sid"]) == int(identity["sid"])
    }
    records = [table[pid] for pid in sorted(owned_ids)]
    if not records:
        raise RuntimeError("The owned vLLM process session disappeared after preflight.")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    gpu_rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0].isdigit():
            gpu_rows.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": int(parts[2].split()[0]),
                    "gpu_uuid": parts[3],
                }
            )
    if len(gpu_rows) != 1 or gpu_rows[0]["pid"] not in owned_ids:
        raise RuntimeError(f"Expected one owned vLLM GPU process, found: {gpu_rows}")
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    memory_parts = [int(part.strip().split()[0]) for part in memory.stdout.strip().split(",")]
    if len(memory_parts) != 3 or memory_parts[2] < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"vLLM left too little free GPU memory: {memory.stdout.strip()}")
    log_text = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    ple_log_patterns = {
        "found": r"Found\s+\d+\s+PleOffloadLayer\(s\):",
        "gpu_registration": (
            r"GPU worker\s+0\s+registered\s+\(dp_rank=0,\s*tp_rank=0,\s*layers="
        ),
        "ready": r"Worker ready\s+-\s+\d+\s+PleOffloadLayer\(s\):",
        "busy_loop": r"Busy-loop started\.",
    }
    if RUNTIME_PROFILE != "day0":
        ple_log_patterns = {"initialized_fp8_pinned": PLE_INIT_PATTERN}
    ple_log_matches = {
        name: (match.group(0) if (match := re.search(pattern, log_text)) else None)
        for name, pattern in ple_log_patterns.items()
    }
    if any(value is None for value in ple_log_matches.values()):
        raise RuntimeError(
            f"vLLM did not prove native PLE CPU-offload readiness: {ple_log_matches}"
        )
    value = {
        "server_identity": identity,
        "process_tree": records,
        "gpu_rows": gpu_rows,
        "gpu_memory_mib": {
            "total": memory_parts[0],
            "used": memory_parts[1],
            "free": memory_parts[2],
        },
        "ple_offload_log_matches": ple_log_matches,
        "captured_epoch": time.time(),
    }
    write_json(GPU_PATH, value)
    return value


def persist_analyzer_environment(env: dict[str, str]) -> dict[str, str]:
    setup_env = read_json(SETUP_ENV_PATH) if SETUP_ENV_PATH.exists() else {}
    if not isinstance(setup_env, dict):
        raise RuntimeError("TAAF_KAGGLE_SETUP_ENV must contain a JSON object.")
    runtime_keys = {
        "PYTHONPATH": os.pathsep.join(
            part for part in env["PYTHONPATH"].split(os.pathsep) if part != str(INIT_TRACE_DIR)),
        "PATH": env["PATH"],
        "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
        "CUDA_HOME": env["CUDA_HOME"],
        "CUDACXX": env["CUDACXX"],
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8": "1",
        "VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256": MODEL_CONFIG_SHA256,
    }
    analyzer = {
        "LOCAL_ANALYZER_BASE_URL": BASE_URL,
        "OPENAI_BASE_URL": BASE_URL,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
        "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
        "OPENAI_API_KEY": "offline-kaggle-local-server",
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(ANALYZER_CONTEXT),
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
    }
    setup_env.update(runtime_keys)
    setup_env.update(analyzer)
    write_json(SETUP_ENV_PATH, setup_env)
    return {**runtime_keys, **analyzer}



def main() -> None:
    apply_quiet()
    setup_started = time.monotonic()
    setup_started_epoch = time.time()
    setup_deadline = setup_started + SETUP_TOTAL_TIMEOUT
    fast_start = fast_start_enabled()
    tuning = resolve_vllm_tuning()
    draft_budget_row = draft_budget_check(tuning)
    draft_early_stop_row = draft_early_stop_check(tuning)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    rotate_launch_logs()
    FAILURE_PATH.unlink(missing_ok=True)
    phase("setup start", epoch=setup_started_epoch)
    source_identity()
    model_dir = resolve_model_dir()
    start_early_read(model_dir, setup_started)
    start_compile_seed()
    print(
        f"VLLM_SETUP_MODE {'fast' if fast_start else 'full'} "
        f"{FAST_START_ENV}={int(fast_start)}",
        flush=True,
    )
    print(f"VLLM_RUNTIME_PROFILE {RUNTIME_PROFILE} image={VLLM_IMAGE} "
          f"dataset={RUNTIME_DATASET} version={VLLM_VERSION}", flush=True)
    prior = stop_owned_server()
    if prior.get("sigterm_sent"):
        print("PRIOR_SERVER stopped before the host memory check: " + json.dumps(prior, sort_keys=True), flush=True)
    if not prior.get("stopped", False):
        raise RuntimeError(f"A prior owned vLLM server did not stop: {prior}")
    inventory = gpu_inventory()
    runtime_dir = resolve_runtime_dir()
    print(f"PINNED_VLLM_RUNTIME_PATH {runtime_dir}", flush=True)
    print(f"PINNED_MODEL_PATH {model_dir}", flush=True)
    storage_check = validate_runtime_storage()
    runtime_check = verify_and_extract_runtime(
        runtime_dir,
        full_layer_hashes=not fast_start,
        scan_extracted_caches=not fast_start,
    )
    phase("runtime unpacked")
    ple_patch = patch_ple_layer()
    ple_host_patch = patch_ple_host()
    ple_hot_patch = patch_ple_hot()
    counter_patch = patch_expert_counter()
    moe_route_patch = patch_moe_route()
    moe_share_patch = patch_moe_share()
    sched_patch = patch_scheduler_spec_reserve()
    group_patch = patch_ple_prefill_groups()
    mtp_patch = patch_mtp_own_reads()
    draft_vocab_patch = patch_draft_vocab()
    target_vocab_patch = patch_target_vocab()
    mtp_weights_patch = patch_mtp_weights()
    vision_weights_patch = patch_vision_weights()
    expert_cut_patch = patch_expert_cut()
    expert_hot_patch = patch_expert_hot()
    last_rows_patch = patch_mtp_last_rows()
    shared_stream_patch = patch_shared_stream()
    piecewise_patch = patch_piecewise_large(tuning)
    draft_conf_log_patch = patch_draft_conf_log()
    relax_accept_patch = patch_relax_accept()
    block_verify_patch = patch_block_verify()
    accept_lambda_patch = patch_accept_lambda()
    adaptive_gate_patch = patch_adaptive_gate()
    draft_budget_patch = patch_draft_budget()
    offload_sync_patch = patch_offload_load_sync()
    kv8_patch = patch_qsa_kv8()
    keep_patch = patch_cache_keep()
    fp8_patch = patch_fp8_dense()
    pad_patch = patch_router_pad()
    router_fused_patch = patch_router_fused()
    moe_taaf_patch = patch_moe_taaf()
    moe_block_patch = patch_moe_block()
    moe_sum_patch = patch_moe_sum()
    moe_silu_patch = patch_moe_silu()
    moe_quant_patch = patch_moe_quant()
    moe_act_quant_patch = patch_moe_act_quant()
    copy_patch = patch_block_copy()
    accept_patch = patch_gdn_accept(tuning)
    fused_patch = patch_gdn_fused()
    gdn_state_q8_patch = patch_gdn_state_q8()
    gdn_state_q8_pack_patch = patch_gdn_state_q8_pack()
    mamba_zero_new_patch = patch_mamba_zero_new()
    meta_once_patch = patch_gdn_meta_once()
    spec_prefix_patch = patch_gdn_spec_prefix()
    gdn_norm_patch = patch_gdn_norm()
    draft_meta_patch = patch_draft_meta()
    draft_early_stop_patch = patch_draft_early_stop()
    conv_syncfree_patch = patch_conv_syncfree()
    marks_patch = patch_marks()
    gdn_tril_patch = patch_gdn_tril()
    gdn_tril_ilp_patch = patch_gdn_tril_ilp()
    vision_cache_patch = patch_vision_cache()
    vision_ahead_patch = patch_vision_ahead(tuning)
    prefill_hold_patch = patch_prefill_hold()
    moe_add_fused_patch = patch_moe_add_fused()
    router_chain_patch = patch_router_chain()
    humming_int3_patch = patch_humming_int3()
    mixed_full_patch = patch_mixed_full()
    gdn_mixed_full_patch = patch_gdn_mixed_full()
    gdn_ba_views_patch = patch_gdn_ba_views()
    gdn_tail_zero_patch = patch_gdn_tail_zero()
    mm_processor_only_patch = patch_mm_processor_only()
    no_blank_patch = patch_no_blank()
    sample_topk_patch = patch_sample_topk()
    small_kernels_patch = patch_small_kernels()
    token_pin_patch = patch_token_pin()
    l2_pin_patch = patch_l2_pin()
    keep_end_patch = patch_cache_keep_end()
    mamba_count_patch = patch_mamba_count_floor()
    alloc_guard_patch = patch_alloc_guard()
    slim_patch = patch_slim_state()
    hc_fused_patch = patch_hc_fused()
    bf16c_patch = patch_bf16c()
    gdn_cp_split_patch = patch_gdn_cp_split()
    gdn_commit_stream_patch = patch_gdn_commit_stream()
    dense_hole_patch = patch_dense_hole(tuning)
    weight_checksum_patch = patch_weight_checksum()
    weight_cache_backend_patch = patch_weight_cache_backend_guard()
    host_kv_patch = patch_host_kv()
    paused_patch = patch_paused()
    prefill_wip_patch = patch_prefill_wip()
    align_tail_patch = patch_align_tail()
    gdn_align_tiles_patch = patch_gdn_align_tiles()
    no_adopt_patch = patch_hisparse_no_adopt()
    tail_restore_patch = patch_hisparse_tail_restore()
    prefill_fill_patch = patch_prefill_fill()
    mem_offload_patch = patch_mem_offload()
    embed_hot_patch = patch_embed_hot()
    ple_qhot_patch = patch_ple_qhot()
    pdl_patch = patch_pdl()
    gdn_decode_tail_patch = patch_gdn_decode_tail()
    step_timers_patch = patch_step_timers()
    dummy_weights_patch = patch_dummy_weights()
    park_patch = patch_park()
    w4a8_patch = patch_w4a8()
    print(f"SELFCHECK_FATAL {'on' if selfcheck_fatal() else 'off'}: a knob whose own startup or first-use check "
          f"fails {'stops the serve with one SELFCHECK FAIL line' if selfcheck_fatal() else 'prints one SELFCHECK FAIL line and stands aside'}"
          f" ({SELFCHECK_FATAL_ENV})", flush=True)
    phase("patches applied")
    env, environment_check = runtime_environment(
        deep_preload_validation=not fast_start,
        tuning=tuning,
    )
    phase("environment ready")
    overlapped = start_self_checks(env, model_dir)
    phase("self-checks started")
    if not overlapped:
        qsa_check = qsa_self_check(env, model_dir)
    weight_cache_patch = patch_weight_cache_daemon(tuning, env)
    phase("weight cache ready")
    if not overlapped:
        bf16c_fast_check = check_bf16c_fast(env)
        bf16c_mix_check = check_bf16c_mix(env)
        dense_hole_check = check_dense_hole(env)
    if fast_start:
        ipc_check = {
            "skipped": True,
            "reason": "diagnostic-only duplicate of live PLE startup",
        }
        cli_check = {
            "skipped": True,
            "reason": "the exact pinned server command is exercised directly",
        }
    else:
        ipc_check = diagnose_pidfd_ipc(env)
        cli_check = validate_cli_contract(env)
    if load_dummy_enabled():
        model_check = {"skipped": True, "reason": f"{LOAD_DUMMY_ENV}: no checkpoint shard is read"}
        print(f"LOAD_DUMMY the serve runs on DUMMY weights ({LOAD_DUMMY_ENV}=1): vLLM --load-format dummy for the "
              f"target and the draft, no checkpoint verification, no shard prefetch, PLE_HOT filled with "
              f"deterministic rows; the outputs are noise and only the machinery is real", flush=True)
    else:
        model_check = verify_model(model_dir, full_file_hashes=not fast_start)
    phase("model verified")
    server_identity, readiness, tuning = serve_or_fall_back(
        model_dir, env, tuning, setup_deadline
    )
    try:
        joined = join_self_checks(env) or SELF_CHECKS["box"]
    except BaseException:
        stop_owned_server()
        raise
    if overlapped:
        qsa_check, bf16c_fast_check, bf16c_mix_check = joined["qsa"], joined["fast"], joined["mix"]
        dense_hole_check = joined.get("hole") or {"armed": False}
    phase("setup joined the self-checks")
    harvest_compile_cache()
    join_pyc_seed("the harvest")
    PYC_RESULTS["harvest"] = harvest_pyc_cache()
    server_identity = update_ready_identity(server_identity)
    if RUNTIME_PROFILE != "day0":
        found = re.search(PLE_INIT_PATTERN,
                          SERVER_LOG.read_text(encoding="utf-8", errors="replace"))
        print(f"VLLM_RUNTIME_PROFILE {RUNTIME_PROFILE} PLE table: "
              + (found.group(0) if found else "NO PLE initialization line in the server log"),
              flush=True)
    if fp8_patch["armed"]:
        fp8_patch["server_log"] = re.findall(
            FP8_DENSE_LOG_PATTERN, SERVER_LOG.read_text(encoding="utf-8", errors="replace"))
        print("FP8_DENSE in the server log: "
              + (" | ".join(fp8_patch["server_log"]) or "NO FP8_DENSE line in the server log"),
              flush=True)
    if fast_start:
        preflight = {
            "mode": "fast",
            "skipped": True,
            "reason": "the authorized one-game smoke is the integrated serving test",
            "served_model": SERVED_MODEL_NAME,
            "models_endpoint": readiness["models"],
            "skipped_requests": [
                "synthetic text",
                "synthetic tool call",
                "synthetic vision",
                "C1 throughput",
                "C4 throughput",
                "pre-game MTP telemetry",
            ],
            "final_mtp_telemetry_source": "vllm-metrics-final.prom from teardown",
        }
    else:
        preflight = run_preflights(setup_deadline)
    write_json(PREFLIGHT_PATH, preflight)
    if fast_start:
        process_gpu = {
            "deferred": True,
            "reason": "teardown records final process ownership and GPU state",
        }
    else:
        server_identity = update_ready_identity(server_identity)
        process_gpu = capture_process_and_gpu(server_identity)
    persisted = persist_analyzer_environment(env)
    provenance = {
        "source_dataset": SOURCE_DATASET,
        "source_identity_sha256": sha256_file(BUNDLE_DIR / "SOURCE_IDENTITY.json"),
        "runtime_dataset": RUNTIME_DATASET,
        "runtime_profile": RUNTIME_PROFILE,
        "model_hf_repo": MODEL_HF_REPO,
        "model_hf_revision": MODEL_HF_REVISION,
        "vllm_image": VLLM_IMAGE,
        "vllm_index_digest": VLLM_INDEX_DIGEST,
        "vllm_version": VLLM_VERSION,
        "vllm_tuning": tuning,
        "fast_start": fast_start,
        "fast_start_environment": FAST_START_ENV,
        "setup_started_epoch": setup_started_epoch,
        "setup_total_timeout_seconds": SETUP_TOTAL_TIMEOUT,
        "server_ready_timeout_seconds": SERVER_READY_TIMEOUT,
        "inventory": inventory,
        "runtime_storage": storage_check,
        "runtime": runtime_check,
        "ple_patch": ple_patch,
        "ple_host_patch": ple_host_patch,
        "ple_hot_patch": ple_hot_patch,
        "moe_route_patch": moe_route_patch,
        "moe_share_patch": moe_share_patch,
        "sched_patch": sched_patch,
        "group_patch": group_patch,
        "mtp_patch": mtp_patch,
        "draft_vocab_patch": draft_vocab_patch,
        "target_vocab_patch": target_vocab_patch,
        "mtp_weights_patch": mtp_weights_patch,
        "vision_weights_patch": vision_weights_patch,
        "expert_cut_patch": expert_cut_patch,
        "expert_hot_patch": expert_hot_patch,
        "mtp_last_rows_patch": last_rows_patch,
        "shared_stream_patch": shared_stream_patch,
        "piecewise_large_patch": piecewise_patch,
        "draft_conf_log_patch": draft_conf_log_patch,
        "relax_accept_patch": relax_accept_patch,
        "block_verify_patch": block_verify_patch,
        "accept_lambda_patch": accept_lambda_patch,
        "adaptive_gate_patch": adaptive_gate_patch,
        "draft_budget_gate": draft_budget_row,
        "draft_budget_patch": draft_budget_patch,
        "offload_sync_patch": offload_sync_patch,
        "kv8_patch": kv8_patch,
        "keep_patch": keep_patch,
        "fp8_dense_patch": fp8_patch,
        "router_pad_patch": pad_patch,
        "router_fused_patch": router_fused_patch,
        "moe_taaf_patch": moe_taaf_patch,
        "moe_block_patch": moe_block_patch,
        "moe_sum_patch": moe_sum_patch,
        "moe_silu_patch": moe_silu_patch,
        "moe_quant_patch": moe_quant_patch,
        "moe_act_quant_patch": moe_act_quant_patch,
        "block_copy_patch": copy_patch,
        "gdn_accept_patch": accept_patch,
        "gdn_fused_patch": fused_patch,
        "gdn_state_q8_patch": gdn_state_q8_patch,
        "gdn_state_q8_pack_patch": gdn_state_q8_pack_patch,
        "mamba_zero_new_patch": mamba_zero_new_patch,
        "gdn_meta_once_patch": meta_once_patch,
        "gdn_spec_prefix_patch": spec_prefix_patch,
        "gdn_norm_patch": gdn_norm_patch,
        "draft_meta_patch": draft_meta_patch,
        "draft_early_stop_gate": draft_early_stop_row,
        "draft_early_stop_patch": draft_early_stop_patch,
        "conv_syncfree_patch": conv_syncfree_patch,
        "marks_patch": marks_patch,
        "gdn_tril_patch": gdn_tril_patch,
        "gdn_tril_ilp_patch": gdn_tril_ilp_patch,
        "vision_cache_patch": vision_cache_patch,
        "vision_ahead_patch": vision_ahead_patch,
        "prefill_hold_patch": prefill_hold_patch,
        "moe_add_fused_patch": moe_add_fused_patch,
        "router_chain_patch": router_chain_patch,
        "humming_int3_patch": humming_int3_patch,
        "mixed_full_patch": mixed_full_patch,
        "gdn_mixed_full_patch": gdn_mixed_full_patch,
        "gdn_ba_views_patch": gdn_ba_views_patch,
        "gdn_tail_zero_patch": gdn_tail_zero_patch,
        "gdn_decode_tail_patch": gdn_decode_tail_patch,
        "pdl_patch": pdl_patch,
        "step_timers_patch": step_timers_patch,
        "qsa_self_check": qsa_check,
        "dummy_weights_patch": dummy_weights_patch,
        "park_patch": park_patch,
        "mm_processor_only_patch": mm_processor_only_patch,
        "no_blank_patch": no_blank_patch,
        "sample_topk_patch": sample_topk_patch,
        "small_kernels_patch": small_kernels_patch,
        "token_pin_patch": token_pin_patch,
        "l2_pin_patch": l2_pin_patch,
        "keep_end_patch": keep_end_patch,
        "mamba_count_patch": mamba_count_patch,
        "alloc_guard_patch": alloc_guard_patch,
        "host_kv_patch": host_kv_patch,
        "paused_patch": paused_patch,
        "prefill_wip_patch": prefill_wip_patch,
        "align_tail_patch": align_tail_patch,
        "gdn_align_tiles_patch": gdn_align_tiles_patch,
        "no_adopt_patch": no_adopt_patch,
        "tail_restore_patch": tail_restore_patch,
        "prefill_fill_patch": prefill_fill_patch,
        "mem_offload_patch": mem_offload_patch,
        "embed_hot_patch": embed_hot_patch,
        "ple_qhot_patch": ple_qhot_patch,
        "slim_patch": slim_patch,
        "hc_fused_patch": hc_fused_patch,
        "bf16c_patch": bf16c_patch,
        "bf16c_fast_check": bf16c_fast_check,
        "gdn_cp_split_patch": gdn_cp_split_patch,
        "gdn_commit_stream_patch": gdn_commit_stream_patch,
        "weight_checksum_patch": weight_checksum_patch,
        "weight_cache_backend_patch": weight_cache_backend_patch,
        "weight_cache_patch": weight_cache_patch,
        "model": model_check,
        "environment": environment_check,
        "ipc_permission": ipc_check,
        "cli_contract": cli_check,
        "server_identity": server_identity,
        "readiness": readiness,
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "process_gpu": process_gpu,
        "persisted_analyzer_environment": persisted,
        "server_log_sha256_at_preflight": sha256_file(SERVER_LOG),
        "server_log_tail": tail_log(300),
        "setup_elapsed_seconds": time.monotonic() - setup_started,
        "completed_epoch": time.time(),
    }
    if bf16c_mix_check != {"armed": False}:
        provenance["bf16c_mix_check"] = bf16c_mix_check
    if dense_hole_patch != {"armed": False}:
        provenance["dense_hole_patch"] = dense_hole_patch
        provenance["dense_hole_check"] = dense_hole_check
    provenance["pyc"] = {key: value for key, value in PYC_RESULTS.items() if key not in ("trace", "compile")}
    provenance["compile_seed"] = PYC_RESULTS.get("compile")
    provenance["init_trace"] = {**(PYC_RESULTS.get("trace") or {}), **finish_init_trace()}
    write_json(SETUP_PROVENANCE, provenance)
    summary = {
        "mode": "fast" if fast_start else "full",
        "ready_seconds": readiness["ready_seconds"],
        "pid": server_identity["pid"],
    }
    if not fast_start:
        summary.update(
            {
                "c1_tokens_per_second": preflight["c1_throughput"][
                    "completion_tokens_per_second"
                ],
                "c4_tokens_per_second": preflight["c4_throughput"][
                    "aggregate_completion_tokens_per_second"
                ],
                "mtp_draft_tokens": preflight["native_mtp"]["draft_tokens"],
                "mtp_accepted_tokens": preflight["native_mtp"]["accepted_tokens"],
            }
        )
    print("VLLM_SETUP_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


def _setup_timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError(f"The {SETUP_TOTAL_TIMEOUT}s setup/preflight ceiling expired.")


QUIET_SETTINGS = (
    ("TAAF_VLLM_PROFILER_DIR", ""), ("TAAF_VLLM_PROFILER_STEPS", ""), ("TAAF_VLLM_PER_REQUEST_SPEC_METRICS", "none"),
    ("TAAF_VLLM_PAUSED_DIFF_LOG_S", ""), ("TAAF_VLLM_PAUSED_PARK_PROBE", "0"), ("TAAF_VLLM_VISION_TALLY", "0"),
    ("TAAF_VLLM_EXPERT_TALLY", "0"), ("TAAF_VLLM_HOST_KV_DEVICE_TALLY", "0"), ("TAAF_VLLM_HOST_KV_GATHER_TALLY", "0"),
    ("TAAF_VLLM_HOST_KV_PCIE_TALLY", "0"), ("TAAF_VLLM_HOST_KV_UNION_TALLY", "0"), ("TAAF_VLLM_HOST_KV_PICK_LOG", "0"),
    ("TAAF_VLLM_PICK_DUMP", "0"), ("TAAF_VLLM_DRAFT_CONF_LOG", "0"), ("TAAF_VLLM_DRAFT_CAPTURE", "0"),
    ("TAAF_VLLM_HOST_KV_HITS_SHADOW", "0"), ("TAAF_VLLM_HOST_KV_HITS_PROBE", "0"), ("TAAF_VLLM_MAMBA_HIT_CHECK", "0"),
    ("TAAF_VLLM_GDN_Q8_FIRST", "0"), ("TAAF_VLLM_STEP_TIMERS_PREP", ""), ("TAAF_VLLM_STEP_TIMERS_HC", ""),
)
QUIET_TIMER_NEEDS = ("TAAF_VLLM_SPEC_INDEX", "TAAF_VLLM_HOST_KV_PREFETCH_SPEC")
QUIET_FORCED = (("TAAF_VLLM_TRIM_HOLE_NO_RUN_LOG", "0"),)


def quiet_wanted(env=None) -> bool:
    env = os.environ if env is None else env
    said = str(env.get(QUIET_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{QUIET_ENV}={said!r} is not one of '', '0', '1'.")
    if said:
        return said == "1"
    return str(env.get(QUIET_RERUN_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def apply_quiet(env=None) -> dict[str, Any]:
    env = os.environ if env is None else env
    if not quiet_wanted(env):
        return {"quiet": False}
    removed = []
    for name, off in QUIET_SETTINGS:
        held = str(env.get(name) or "").strip()
        if not held or held.lower() == off:
            continue
        if off:
            env[name] = off
        else:
            env.pop(name, None)
        removed.append(f"{name[len('TAAF_VLLM_'):]}={held}")
    for name, off in QUIET_FORCED:
        held = str(env.get(name) or "").strip()
        if held != off:
            env[name] = off
            removed.append(f"{name[len('TAAF_VLLM_'):]}={held or 'default'}")
    needs = [name for name in QUIET_TIMER_NEEDS if str(env.get(name) or "").strip() not in ("", "0")]
    timers = str(env.get(STEP_TIMERS_ENV) or "").strip()
    kept = []
    if timers and timers != "0":
        if needs:
            kept.append(f"STEP_TIMERS={timers} (armed for {', '.join(name[len('TAAF_VLLM_'):] for name in needs)})")
        else:
            env[STEP_TIMERS_ENV] = "0"
            removed.append(f"STEP_TIMERS={timers}")
    reason = f"{QUIET_ENV}=1" if str(env.get(QUIET_ENV) or "").strip() == "1" else f"{QUIET_RERUN_ENV} is set"
    print(f"QUIET ({reason}): removed {', '.join(removed) or 'nothing'}"
          + (f"; kept {', '.join(kept)}" if kept else "")
          + "; hot spans, hot reads and the hole checks stay as the build sets them", flush=True)
    return {"quiet": True, "removed": removed, "kept": kept}


if __name__ == "__main__":
    try:
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("The pinned Kaggle setup requires Linux SIGALRM support.")
        signal.signal(signal.SIGALRM, _setup_timeout_handler)
        signal.alarm(SETUP_TOTAL_TIMEOUT)
        main()
        signal.alarm(0)
    except BaseException as exc:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "server_log_tail": tail_log(400),
            "cleanup": stop_owned_server(),
            "failed_epoch": time.time(),
        }
        write_json(FAILURE_PATH, failure)
        print("VLLM_SETUP_FAILED", json.dumps(failure, sort_keys=True), flush=True)
        raise
