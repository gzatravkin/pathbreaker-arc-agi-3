"$PYTHON" - <<'PYSETUP'
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVING_OWNER = 'grigoriyzatravkin'
SERVING_SLUG = 'arc3-serving-unified59h'
SERVING_SCRIPT = 'serving_setup.py'
RUNTIME_OWNER = 'grigoriyzatravkin'
RUNTIME_SLUG = 'arc3-vllm030-runtime-flat'
MODEL_OWNER = 'dfranzen'
MODEL_SLUG = 'intel-qwen3.8-flash-next-w4a16-autoround'
SERVED_MODEL_NAME = 'Qwen/Qwen3.8-Flash-Next-NVFP4'
TOOL_CALL_PARSER = 'qwen3_coder'
REASONING_PARSER = 'qwen3'
SERVING_ENV = {'TAAF_MODEL_PATH': '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1', 'TAAF_MODEL_REPO': 'Intel/Qwen3.8-Flash-Next-W4A16-AutoRound', 'TAAF_MODEL_UNPINNED': '1', 'TAAF_VLLM_ADAPTIVE_VERIFY': '0', 'TAAF_VLLM_ALIGN_TAIL': '1', 'TAAF_VLLM_ALLOC_GUARD': '1', 'TAAF_VLLM_API_MALLOC': 'arena2', 'TAAF_VLLM_BF16C': '1', 'TAAF_VLLM_BF16C_FAST2': '1', 'TAAF_VLLM_BF16C_GROUPS': 'vision', 'TAAF_VLLM_BF16C_MIX': '1', 'TAAF_VLLM_BF16C_MIX_DECODE': '1', 'TAAF_VLLM_BF16C_ROWS': '512', 'TAAF_VLLM_BLOCK_COPY_CHUNK_MIB': '64', 'TAAF_VLLM_BLOCK_VERIFY': '1', 'TAAF_VLLM_CACHE_KEEP': '1', 'TAAF_VLLM_CACHE_KEEP_END': '0', 'TAAF_VLLM_CAPTURE_SPARSE': '128:40', 'TAAF_VLLM_COMPILE_HARVEST': '1', 'TAAF_VLLM_COMPILE_SEED': '1', 'TAAF_VLLM_CONV_SYNCFREE': '1', 'TAAF_VLLM_DENSE_HOLE': '1', 'TAAF_VLLM_DRAFT_FUSED_META': '1', 'TAAF_VLLM_DRAFT_HEAD_FP8': '1', 'TAAF_VLLM_GEMMA_NORM_FUSED': '1', 'TAAF_VLLM_MARLIN_WORKSPACE_REUSE': '1', 'TAAF_VLLM_DRAFT_TOPK_TRITON': '1', 'TAAF_VLLM_HISPARSE_INVALIDATE_FUSED': '1', 'TAAF_VLLM_SAMPLE_GLUE_FUSED': '1', 'TAAF_VLLM_DRAFT_MODEL': '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1', 'TAAF_VLLM_DRAFT_SAMPLE': '1', 'TAAF_VLLM_DRAFT_SAMPLE_NARROW': '1', 'TAAF_VLLM_DRAFT_SAMPLE_TOP': '20', 'TAAF_VLLM_DRAFT_SUMMIN': '0', 'TAAF_VLLM_DRAFT_VOCAB': '1', 'TAAF_VLLM_DRAFT_VOCAB_PAD': '64', 'TAAF_VLLM_EARLY_READ': '1', 'TAAF_VLLM_EMBED_HOST': '1', 'TAAF_VLLM_EMBED_HOT': '8192', 'TAAF_VLLM_ENABLE_PREFIX_CACHING': '1', 'TAAF_VLLM_EXPERTS_KEPT': '400', 'TAAF_VLLM_EXPERT_CUT_AT_LOAD': '1', 'TAAF_VLLM_EXPERT_CUT_COALESCE_MIB': '64', 'TAAF_VLLM_EXPERT_CUT_WORKERS': '32', 'TAAF_VLLM_EXPERT_ORDER': 'usage', 'TAAF_VLLM_FORK_BRANCH': '1', 'TAAF_VLLM_FORK_LIVE': '1', 'TAAF_VLLM_FORK_SEATS': '0', 'TAAF_VLLM_FP8_DENSE': 'gdn', 'TAAF_VLLM_FP8_DENSE_QUANT': 'ptpc', 'TAAF_VLLM_FP8_TRITON_BAND': '2560x6144:257-576', 'TAAF_VLLM_GDN_ACCEPT': '1', 'TAAF_VLLM_GDN_COMMIT_CAP': '2', 'TAAF_VLLM_GDN_COMMIT_SHAPE': '32,8,1', 'TAAF_VLLM_GDN_FUSED': 'tiled', 'TAAF_VLLM_GDN_HALVES': '0', 'TAAF_VLLM_GDN_META_ONCE': '1', 'TAAF_VLLM_GDN_META_SLICE': '1', 'TAAF_VLLM_GDN_MIXED_LEAN': '1', 'TAAF_VLLM_GDN_NORM_STRIDED': '1', 'TAAF_VLLM_GDN_PREP_ONCE': '1', 'TAAF_VLLM_GDN_SLOT_FILL': '1', 'TAAF_VLLM_GDN_SPEC_PREFIX': '1', 'TAAF_VLLM_GDN_STATE_INPLACE': '0', 'TAAF_VLLM_GDN_STATE_Q8': '0', 'TAAF_VLLM_GDN_STATE_Q8_PACK': '0', 'TAAF_VLLM_GDN_TILED_SHAPE': '64,2,1', 'TAAF_VLLM_GDN_VERIFY_AHEAD': '1', 'TAAF_VLLM_GPU_MEMORY_PERMILLE': '930', 'TAAF_VLLM_HC_DECODE': '300-660', 'TAAF_VLLM_HC_DECODE_DOWN': 'cublas', 'TAAF_VLLM_HC_DECODE_DOWN_MIN_ROWS': '300', 'TAAF_VLLM_HC_DECODE_UP4': '64,32,4,3,32', 'TAAF_VLLM_HC_FUSED': '1', 'TAAF_VLLM_HC_MIXED_F1': '1', 'TAAF_VLLM_HISPARSE_COPY_ASYNC': '1', 'TAAF_VLLM_HISPARSE_PLAN_ONCE': '1', 'TAAF_VLLM_HISPARSE_PLAN_ONCE_CAPTURED': '1', 'TAAF_VLLM_HISPARSE_SYNCFREE': '1', 'TAAF_VLLM_HOST_KV': '1', 'TAAF_VLLM_HOST_KV_ADMIT': 'real', 'TAAF_VLLM_HOST_KV_DEVICE_TALLY': '0', 'TAAF_VLLM_HOST_KV_DRAFT_STOCK': '0', 'TAAF_VLLM_HOST_KV_DRAFT_THROUGH': '1', 'TAAF_VLLM_HOST_KV_FLAT_COPY': '1', 'TAAF_VLLM_HOST_KV_FLAT_LINES': '1', 'TAAF_VLLM_HOST_KV_FLAT_PROGRAMS': '96', 'TAAF_VLLM_HOST_KV_FRESH_FIX': '0', 'TAAF_VLLM_HOST_KV_FUSED': '1', 'TAAF_VLLM_HOST_KV_GATHER_EARLY': '1', 'TAAF_VLLM_HOST_KV_GATHER_EARLY_CAPTURED': '1', 'TAAF_VLLM_HOST_KV_GATHER_SPREAD': '4', 'TAAF_VLLM_HOST_KV_GATHER_TALLY': '0', 'TAAF_VLLM_HOST_KV_GIB': '84', 'TAAF_VLLM_HOST_KV_GROUP': '2', 'TAAF_VLLM_HOST_KV_HITS_DRAFT': '1', 'TAAF_VLLM_HOST_KV_HITS_FIRST': '1', 'TAAF_VLLM_HOST_KV_HITS_FIRST_PROGRAMS': '96', 'TAAF_VLLM_HOST_KV_HITS_PURE': '1', 'TAAF_VLLM_HOST_KV_HIT_SKIP': '2', 'TAAF_VLLM_HOST_KV_HOLD_CLEAN': '1', 'TAAF_VLLM_HOST_KV_HOT_RELEASE': '1', 'TAAF_VLLM_HOST_KV_HOT_ROWS': '12288', 'TAAF_VLLM_HOST_KV_HOT_SPLIT': '1', 'TAAF_VLLM_HOST_KV_INDEXER_WIDE': '1', 'TAAF_VLLM_HOST_KV_LAST_SLICE': '1', 'TAAF_VLLM_HOST_KV_MARK_FILL': '1', 'TAAF_VLLM_HOST_KV_MIXED_CAPTURE': '1', 'TAAF_VLLM_HOST_KV_ONE_HASH': '1', 'TAAF_VLLM_HOST_KV_ONE_LAUNCH': '1', 'TAAF_VLLM_HOST_KV_PLAN_AHEAD': '1', 'TAAF_VLLM_HOST_KV_PLAN_FUSED': '1', 'TAAF_VLLM_HOST_KV_PLE_GIB': '37', 'TAAF_VLLM_HOST_KV_ROWS': '9984', 'TAAF_VLLM_HOST_KV_RUN_MERGE': '0', 'TAAF_VLLM_HOST_KV_SINGLE_FUSED': '1', 'TAAF_VLLM_HOST_KV_SOURCES_FUSED': '1', 'TAAF_VLLM_HOST_KV_SPARSE_STAGE': '1', 'TAAF_VLLM_HOST_KV_SPILL_FLAG': '1', 'TAAF_VLLM_HOST_KV_SPILL_ONESORT': '1', 'TAAF_VLLM_HOST_KV_SPILL_SKIP': '1', 'TAAF_VLLM_HOST_KV_SPILL_TRUST': '1', 'TAAF_VLLM_HOST_KV_STAGE_MIB': '320', 'TAAF_VLLM_HOST_KV_STAGE_TILE': '64', 'TAAF_VLLM_HOST_KV_TAKE_ROWS': '0', 'TAAF_VLLM_HOST_KV_UNION_BITMAP': '1', 'TAAF_VLLM_HOST_KV_UNION_SWAP': '1', 'TAAF_VLLM_HOT_KEEP': '1', 'TAAF_VLLM_HOT_READS': '23,35,47', 'TAAF_VLLM_HOT_SPANS': '1', 'TAAF_VLLM_IMAGE_MAX_PIXELS': '4194304', 'TAAF_VLLM_INDEXER_KV_DTYPE': 'fp8', 'TAAF_VLLM_INT8_FAST': '1', 'TAAF_VLLM_KV4_WRITE_FUSED': '1', 'TAAF_VLLM_KV_AUTO': '1', 'TAAF_VLLM_KV_CACHE_DTYPE': 'int4_per_token_head', 'TAAF_VLLM_KV_CACHE_MEMORY_BYTES': '28806644240', 'TAAF_VLLM_L2_PIN': '96', 'TAAF_VLLM_LOCAL_TIER': '0', 'TAAF_VLLM_MAMBA_COUNT_FLOOR': '1', 'TAAF_VLLM_MAMBA_JUNCTION': '1', 'TAAF_VLLM_MAMBA_SSM_DTYPE': 'bfloat16', 'TAAF_VLLM_MAMBA_ZERO_NEW': '1', 'TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE': '592', 'TAAF_VLLM_MAX_MODEL_LEN': '204800', 'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS': '4224', 'TAAF_VLLM_MAX_NUM_SEQS': '117', 'TAAF_VLLM_MEM_LEDGER': '1', 'TAAF_VLLM_MIXED_FULL': '1', 'TAAF_VLLM_MIXED_PREP_SHARED': '1', 'TAAF_VLLM_MIXED_TAIL_POOL': '1', 'TAAF_VLLM_MM_PROCESSOR_ONLY': '1', 'TAAF_VLLM_MOE_ACT_QUANT': '1', 'TAAF_VLLM_MOE_ADD_FUSED': '1', 'TAAF_VLLM_MOE_BLOCK_M': '1817-2072:48,2201-2328:48,2900-4224:48', 'TAAF_VLLM_MOE_QUANT_FUSED': '1', 'TAAF_VLLM_MOE_SILU_EPI': '1', 'TAAF_VLLM_MOE_SILU_PREMIX': '1', 'TAAF_VLLM_MOE_SUM_EPI': '1', 'TAAF_VLLM_MTP_ATTN_ROWS': '1', 'TAAF_VLLM_MTP_INDEX_ROWS': '1', 'TAAF_VLLM_MTP_LAST_ROWS': '1', 'TAAF_VLLM_MTP_OWN_READS': '1', 'TAAF_VLLM_MTP_TOKENS': '4', 'TAAF_VLLM_MTP_WEIGHTS': 'drafter_h200_snapB_dense.safetensors,drafter_h200_snapB_experts_int4.safetensors', 'TAAF_VLLM_NO_BLANK_LINES': '0', 'TAAF_VLLM_NO_VIDEO': '1', 'TAAF_VLLM_OFFLOAD_LOAD_SYNC': '1', 'TAAF_VLLM_OMP_THREADS': '1', 'TAAF_VLLM_PARK': '1', 'TAAF_VLLM_PARK_DISK': '1', 'TAAF_VLLM_PAUSED': '1', 'TAAF_VLLM_PAUSED_BRANCH': '1', 'TAAF_VLLM_PAUSED_CLIENT_VIEW': '1', 'TAAF_VLLM_PAUSED_CLIENT_VIEW_SERVER_TOKENS': '1', 'TAAF_VLLM_PAUSED_DIFF_LOG_S': '30', 'TAAF_VLLM_PAUSED_LEAN': '1', 'TAAF_VLLM_PAUSED_MM_DROP': '1', 'TAAF_VLLM_PAUSED_PARK_PROBE': '1', 'TAAF_VLLM_PAUSED_POOL': '1', 'TAAF_VLLM_PAUSED_PRESSURE_MEMORY_ONLY': '1', 'TAAF_VLLM_PAUSED_REGISTER': '0', 'TAAF_VLLM_PAUSED_REPLY_SLACK': '12288', 'TAAF_VLLM_PAUSED_RESUME_UNSTOPPED': '1', 'TAAF_VLLM_PAUSED_SEAT_PRESSURE': '1', 'TAAF_VLLM_PAUSED_TEXT_GUARD': '1', 'TAAF_VLLM_PDL': '1', 'TAAF_VLLM_PER_REQUEST_SPEC_METRICS': 'summary', 'TAAF_VLLM_PIECEWISE_LARGE': '4224', 'TAAF_VLLM_PIECEWISE_LARGE_STEP': '128', 'TAAF_VLLM_PLE_DF11': '1', 'TAAF_VLLM_PLE_DF11_ASYNC': '1', 'TAAF_VLLM_PLE_DF11_STARTUP': '1', 'TAAF_VLLM_PLE_DF11_STARTUP_THREADS': '32', 'TAAF_VLLM_PLE_EXACT_PIN': '1', 'TAAF_VLLM_PLE_HOT': '0', 'TAAF_VLLM_PLE_PACKED': '1', 'TAAF_VLLM_PLE_QHOT': '1048576', 'TAAF_VLLM_PLE_QSTORE': 'int6', 'TAAF_VLLM_POOL_LEDGER': '1', 'TAAF_VLLM_PREFILL_FILL': '1', 'TAAF_VLLM_PREFILL_HOLD': '1', 'TAAF_VLLM_PREFILL_HOLD_ALIGNED': '1', 'TAAF_VLLM_PREFILL_HOLD_AUTO': '1', 'TAAF_VLLM_PREFILL_HOLD_MS': '1000', 'TAAF_VLLM_PREFILL_HOLD_TOKENS': '1600', 'TAAF_VLLM_PREFIX_RETENTION': '0', 'TAAF_VLLM_PREP_ADMIT': '1', 'TAAF_VLLM_PREP_GDN': '1', 'TAAF_VLLM_PREP_NOSYNC': '1', 'TAAF_VLLM_PROFILER_DIR': '/kaggle/working/vllm-profile', 'TAAF_VLLM_PROFILER_STEPS': '60', 'TAAF_VLLM_PYC_HARVEST': '1', 'TAAF_VLLM_PYC_SEED': '1', 'TAAF_VLLM_QSA_DECODE_CONFIG': '64,4,1', 'TAAF_VLLM_QSA_DECODE_FUSED': '1', 'TAAF_VLLM_QSA_DECODE_LEAN': '1', 'TAAF_VLLM_QSA_DECODE_WARPS': '4', 'TAAF_VLLM_QSA_HALVES': '0', 'TAAF_VLLM_QSA_INDEX_COMPACT': '1', 'TAAF_VLLM_QSA_KV8': '1', 'TAAF_VLLM_QSA_PREFILL': '1', 'TAAF_VLLM_QSA_PREFILL_CONFIG': '64,4,3', 'TAAF_VLLM_QSA_PREFILL_LEAN': '1', 'TAAF_VLLM_QSA_PREFILL_LEAN_CHECKED': '1', 'TAAF_VLLM_QSA_SELFCHECK': '1', 'TAAF_VLLM_QSA_SPLITK_FUSED': '1', 'TAAF_VLLM_QSA_STAGE_REUSE': '1', 'TAAF_VLLM_QSA_STAGE_REUSE_MIB': '88', 'TAAF_VLLM_QSA_TILE': '4', 'TAAF_VLLM_QSA_TILE_SKIP_EMPTY': '1', 'TAAF_VLLM_QSA_TILE_UNION_FUSED': '1', 'TAAF_VLLM_QUANTIZATION': 'inc', 'TAAF_VLLM_READY_POLL_S': '0.5', 'TAAF_VLLM_READY_TIMEOUT_S': '3000', 'TAAF_VLLM_ROUTER_CHAIN': 'gate_up', 'TAAF_VLLM_ROUTER_FUSED': '1', 'TAAF_VLLM_RUNTIME': 'v030', 'TAAF_VLLM_SAMPLE_TOPK': '32', 'TAAF_VLLM_SAMPLE_TOPK_DRAFT': '1', 'TAAF_VLLM_SCHED_SPEC_RESERVE': '1', 'TAAF_VLLM_SELFCHECK_FATAL': '0', 'TAAF_VLLM_SELFCHECK_OVERLAP': '2', 'TAAF_VLLM_SETUP_TIMEOUT_S': '3300', 'TAAF_VLLM_STABLE_THINK_SYSTEM': '1', 'TAAF_VLLM_STEP_TIMERS': '0', 'TAAF_VLLM_TAIL_SOFT': 'hard', 'TAAF_VLLM_TARGET_VOCAB': '1', 'TAAF_VLLM_TRIM_HOLE': '0', 'TAAF_VLLM_TRIM_HOLE_ALIGN': '3584', 'TAAF_VLLM_TRIM_HOLE_CHECKS': '0', 'TAAF_VLLM_TRIM_HOLE_KEYS': '', 'TAAF_VLLM_TRIM_HOLE_MAX_POS': '376832', 'TAAF_VLLM_TRIM_HOLE_MIN': '3584', 'TAAF_VLLM_TRIM_HOLE_REPLY': '1', 'TAAF_VLLM_TRIM_HOLE_RETRY': '20', 'TAAF_VLLM_TRIM_HOLE_SYNC_TEXT': '1', 'TAAF_VLLM_TRIM_HOLE_TEXT': '1', 'TAAF_VLLM_VERDICT_CACHE': '1', 'TAAF_VLLM_VISION_AHEAD': '1', 'TAAF_VLLM_VISION_CACHE': '1024', 'TAAF_VLLM_WEIGHTS_PREFETCH': '8', 'TAAF_VLLM_WEIGHTS_PREFETCH_GIB': '24', 'VLLM_MARLIN_INPUT_DTYPE': 'int8', 'VLLM_SERVER_DEV_MODE': '1', 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB': '64', 'VLLM_USE_SIMPLE_KV_OFFLOAD': '1', 'TAAF_VLLM_HOST_KV_ONE_CUDA': '1', 'TAAF_VLLM_HOT_KEEP_WARM': '1', 'TAAF_VLLM_TAIL_KEPT_LOGITS': '1', 'TAAF_VLLM_GDN_DECODE_TAIL_ZERO': '1', 'TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION': '1', 'TAAF_VLLM_SHARED_STREAM_TOKENS': '600', 'TAAF_VLLM_OPEN_THINK_TEMPLATE': '1', 'TAAF_VLLM_EXPERT_HOT': '1'}
VLLM_HOST = '127.0.0.1'
VLLM_PORT = 1234
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
ANALYZER_CONTEXT_WINDOW = 150000
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SETUP_LOG = WORKING_DIR / 'setup-diagnostics.log'
SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
PATCH_DIRNAME = 'vllm-patches'
IDENTITY_NAME = 'SOURCE_IDENTITY.json'
SERVING_SETUP_SHA256 = 'e89e2877ba6f756797817f6f560f4926c5dcf96a0104a725334e565f5dca8d72'
SERVING_REQUIRES = {'serving_setup.py': ('TAAF_VLLM_CACHE_KEEP_END', 'TAAF_VLLM_PAUSED_REGISTER', 'TAAF_VLLM_QUANTIZATION', 'TAAF_VLLM_DRAFT_MODEL', 'TAAF_VLLM_PLE_EXACT_PIN', 'TAAF_VLLM_HOST_KV_PLE_GIB', 'TAAF_VLLM_HOST_KV_FUSED', 'TAAF_VLLM_HOST_KV_HOT_SPLIT', 'TAAF_VLLM_ALIGN_TAIL', 'TAAF_VLLM_EMBED_HOST', 'TAAF_VLLM_MEM_LEDGER', 'TAAF_VLLM_GDN_META_ONCE', 'TAAF_VLLM_CONV_SYNCFREE', 'TAAF_VLLM_GDN_META_SLICE', 'TAAF_VLLM_GDN_NORM_STRIDED', 'TAAF_VLLM_GDN_SLOT_FILL', 'TAAF_VLLM_GDN_SPEC_PREFIX', 'TAAF_VLLM_HOST_KV_GATHER_TALLY', 'TAAF_VLLM_HOST_KV_LAST_SLICE', 'TAAF_VLLM_HOST_KV_SOURCES_FUSED', 'TAAF_VLLM_HOST_KV_SPILL_FLAG', 'TAAF_VLLM_HOST_KV_UNION_BITMAP', 'TAAF_VLLM_MM_PROCESSOR_ONLY', 'TAAF_VLLM_PAUSED_BRANCH', 'TAAF_VLLM_PAUSED_CLIENT_VIEW', 'TAAF_VLLM_PLE_HOT', 'TAAF_VLLM_PROFILER_DIR', 'TAAF_VLLM_HOST_KV_MIXED_CAPTURE', 'TAAF_VLLM_HOST_KV_GROUP', 'TAAF_VLLM_HOST_KV_PICK_LOG', 'TAAF_VLLM_INT8_FAST', 'TAAF_VLLM_QSA_TILE', 'TAAF_VLLM_QSA_DECODE_FUSED', 'TAAF_VLLM_QSA_DECODE_WARPS', 'TAAF_VLLM_MIXED_FULL', 'TAAF_VLLM_PDL', 'TAAF_VLLM_BF16C', 'TAAF_VLLM_BF16C_ROWS', 'TAAF_VLLM_MTP_WEIGHTS', 'TAAF_VLLM_API_MALLOC', 'TAAF_VLLM_BF16C_MIX', 'TAAF_VLLM_BF16C_MIX_DECODE', 'TAAF_VLLM_COMPILE_HARVEST', 'TAAF_VLLM_EARLY_READ', 'TAAF_VLLM_HOST_KV_PICK_LOG_AT', 'TAAF_VLLM_HOST_KV_PREFETCH_PREV', 'TAAF_VLLM_HOST_KV_PREFETCH_PREV_ROWS', 'TAAF_VLLM_INIT_TRACE', 'TAAF_VLLM_LOCAL_TIER', 'TAAF_VLLM_MEMTRACE', 'TAAF_VLLM_MEMTRACE_S', 'TAAF_VLLM_PAUSED_MM_DROP', 'TAAF_VLLM_PLE_DF11', 'TAAF_VLLM_PREP_GDN', 'TAAF_VLLM_PREP_NOSYNC', 'TAAF_VLLM_PYC_HARVEST', 'TAAF_VLLM_READY_POLL_S', 'TAAF_VLLM_SELFCHECK_OVERLAP', 'TAAF_VLLM_STEP_TIMERS_PREP', 'TAAF_VLLM_VERDICT_CACHE', 'TAAF_VLLM_EXPERT_CUT_AT_LOAD', 'TAAF_VLLM_PLE_DF11_STARTUP', 'TAAF_VLLM_HOST_KV_PCIE_TALLY', 'TAAF_VLLM_HOST_KV_RUN_MERGE', 'TAAF_VLLM_COMPILE_SEED', 'TAAF_VLLM_PYC_SEED', 'TAAF_VLLM_HOST_KV_ONE_LAUNCH', 'TAAF_VLLM_HOST_KV_PREFETCH_PREV_ONCE', 'TAAF_VLLM_FORK_BRANCH', 'TAAF_VLLM_KV_AUTO', 'TAAF_VLLM_CAPTURE_SPARSE', 'TAAF_VLLM_PIECEWISE_LARGE_STEP'), 'ple_host.py': ('TAAF_VLLM_PLE_EXACT_PIN',), 'align_tail.py': ('ALIGN_TAIL live',), 'mem_offload.py': ('TAAF_VLLM_EMBED_HOST', 'TAAF_VLLM_MEM_LEDGER'), 'qsa_hisparse.py': ('TAAF_VLLM_HOST_KV_UNION_BITMAP', 'TAAF_VLLM_HOST_KV_SPILL_FLAG', 'TAAF_VLLM_HOST_KV_GATHER_TALLY', 'TAAF_VLLM_HOST_KV_LAST_SLICE', 'TAAF_VLLM_HOST_KV_SOURCES_FUSED', 'TAAF_VLLM_HOST_KV_MIXED_CAPTURE', 'TAAF_VLLM_HOST_KV_GROUP', 'prefill_mirror', 'last_rows_clamped', 'TAAF_VLLM_HOST_KV_PICK_LOG_AT', 'TAAF_VLLM_HOST_KV_PLAN_AHEAD', 'TAAF_VLLM_HOST_KV_PLAN_FUSED', 'TAAF_VLLM_HOST_KV_PREFETCH_PREV', 'TAAF_VLLM_HOST_KV_PREFETCH_PREV_ROWS'), 'gdn_norm.py': ('GDN_NORM_STRIDED live',), 'conv_syncfree.py': ('CONV_SYNCFREE live',), 'mm_processor_only.py': ('MM_PROCESSOR_ONLY live',), 'paused.py': ('TAAF_VLLM_PAUSED_BRANCH', 'TAAF_VLLM_PAUSED_CLIENT_VIEW', 'TAAF_VLLM_PAUSED_DIFF_LOG_S', 'TAAF_VLLM_MEMTRACE', 'TAAF_VLLM_MEMTRACE_DEEP', 'TAAF_VLLM_MEMTRACE_S', 'TAAF_VLLM_PAUSED_LEAN', 'TAAF_VLLM_POOL_LEDGER'), 'ple_hot.py': ('TAAF_VLLM_PLE_HOT', 'TAAF_VLLM_PLE_DF11', 'TAAF_VLLM_PLE_DF11_ASYNC'), 'qsa_kv8.py': ('TAAF_VLLM_INT8_FAST', 'decode_fused_int8', '_qsa_prefill_kv8_tile_kernel', 'TAAF_VLLM_QSA_SPLITK_FUSED', 'TAAF_VLLM_QSA_TILE_SKIP_EMPTY', 'TAAF_VLLM_QSA_TILE_UNION_FUSED'), 'bf16c.py': ('TAAF_VLLM_BF16C_MIX', 'TAAF_VLLM_BF16C_MIX_DECODE'), 'init_trace.py': ('TAAF_VLLM_INIT_TRACE',), 'shard_prefetch.py': ('TAAF_VLLM_PLE_DF11',), 'token_pin.py': ('TAAF_VLLM_PREP_ADMIT',), 'gdn_fused.py': ('TAAF_VLLM_PREP_GDN', 'TAAF_VLLM_PREP_NOSYNC'), 'gdn_meta_once.py': ('TAAF_VLLM_PREP_GDN',), 'gdn_mixed_full.py': ('TAAF_VLLM_PREP_NOSYNC',), 'step_timers.py': ('TAAF_VLLM_STEP_TIMERS_PREP',), 'mixed_full_spec.py': ('TAAF_VLLM_MIXED_TAIL_POOL',), 'mixed_full.py': ('TAAF_VLLM_PIECEWISE_LARGE_STEP',)}
RUNTIME_MANIFEST = 'runtime-manifest.json'
CARRIED_KEYS = ('PYTHONPATH', 'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDACXX')

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',)}

DRY_UNLESS_RERUN = False
DRY_MARKER = WORKING_DIR / 'dry_build.marker'
THINK_OPEN = False
THINK_OPEN_OLD = '{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}'
THINK_OPEN_NEW = "{%- if loop.last and content.startswith('<think>') %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- elif preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}"
WEIGHTS_PREFETCH = 32
WARM_CACHE = False
WARM_CACHE_DATASET = ''
WARM_CACHE_BYTECODE = False
SERVING_POLL_S = 1
PREBAKED_STAMP = 'PRUNE_COMPLETE.json'
PREFETCH_AHEAD_BYTES = 24 * (1 << 30)
WEIGHTS_EVICT = False
EVICT_BEHIND = 2
READ_PIECE = 64 << 20
READ_BLOCK = 8 << 20
PROBE_BYTES = 1 << 30
LOADER_PROGRESS = r'checkpoint shards: +[0-9]+% Completed \| ([0-9]+)/([0-9]+) \['
THINK_OPEN_WRAPPER = '''
import pathlib, runpy, subprocess, sys
main, old, new = sys.argv[1:4]
real = subprocess.Popen
def opened(command, *args, **kwargs):
    if isinstance(command, list) and '--chat-template' in command:
        held = pathlib.Path(command[command.index('--chat-template') + 1])
        text = held.read_text(encoding='utf-8')
        if text.count(old) == 1:
            try:
                if held.is_symlink():
                    held.unlink()
                held.write_text(text.replace(old, new), encoding='utf-8')
                print('THINK_OPEN: the served chat template leaves a begun thought open, written in '
                      'place at ' + str(held), flush=True)
            except OSError as exc:
                print('THINK_OPEN: NOT applied, the template path is not writable: ' + repr(exc),
                      flush=True)
        elif 'loop.last and content.startswith' in text:
            print('THINK_OPEN: the served chat template already leaves a begun thought open',
                  flush=True)
        else:
            print('THINK_OPEN: NOT applied, the served chat template holds the assistant line '
                  + str(text.count(old)) + ' times', flush=True)
    return real(command, *args, **kwargs)
subprocess.Popen = opened
sys.argv = [main]
runpy.run_path(main, run_name='__main__')
'''

MEMGUARD_RELEASE_MIB = int(os.environ.get('TAAF_MEMGUARD_RELEASE_MIB') or '1024')
MEMGUARD_WRAPPER = '''
import os, pathlib, runpy, subprocess, sys
main, floor = sys.argv[1], sys.argv[2]
os.environ['TAAF_VLLM_MEM_RELEASE_MIB'] = floor
NL = chr(10)
OLD = NL.join(['def ledger_tick(dev: Any) -> str:',
               '    line = ledger_line(dev, "live")',
               '    if _STATE["warm"]:',
               '        dev.reset_peaks()',
               '    return line', ''])
NEW = NL.join(['def ledger_tick(dev: Any) -> str:',
               '    line = ledger_line(dev, "live")',
               '    if _STATE["warm"]:',
               '        dev.reset_peaks()',
               '        floor = float(os.environ.get("TAAF_VLLM_MEM_RELEASE_MIB") or 0) * MIB',
               '        before = dev.free_bytes() if floor > 0 else 0',
               '        if floor > 0 and before < floor:',
               '            dev.release()',
               '            after = dev.free_bytes()',
               '            line += chr(10) + (f"MEM_RELEASE t_s={uptime():.0f} free_before_mib={before / MIB:.1f} "',
               '                               f"free_after_mib={after / MIB:.1f} floor_mib={floor / MIB:.0f}")',
               '    return line', ''])
real = subprocess.Popen
state = {'said': None}
def patch_helper(env):
    roots = [r for r in ((env or {}).get('PYTHONPATH', '') + os.pathsep + os.environ.get('PYTHONPATH', '')).split(os.pathsep) if r]
    for root in roots:
        path = pathlib.Path(root) / 'vllm' / 'taaf_mem_offload.py'
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        if 'MEM_RELEASE t_s=' in text:
            return 'already in ' + str(path)
        if text.count(OLD) != 1:
            return 'NOT applied, ledger_tick appears ' + str(text.count(OLD)) + ' times in ' + str(path)
        try:
            if path.is_symlink():
                path.unlink()
            path.write_text(text.replace(OLD, NEW), encoding='utf-8')
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except Exception as exc:
            path.write_text(text, encoding='utf-8')
            return 'NOT applied, ' + str(path) + ' could not be rewritten: ' + repr(exc)
        return 'applied in ' + str(path)
    return None
def opened(command, *args, **kwargs):
    if state['said'] is None and isinstance(command, (list, tuple)):
        try:
            said = patch_helper(kwargs.get('env'))
        except Exception as exc:
            said = 'NOT applied, the patch raised ' + repr(exc)
        if said is not None:
            state['said'] = said
            print('MEMGUARD: ' + said + '; the served worker frees its cached blocks when free memory drops under '
                  + floor + ' MiB', flush=True)
    env = kwargs.get('env')
    if env is not None:
        env['TAAF_VLLM_MEM_RELEASE_MIB'] = floor
    return real(command, *args, **kwargs)
subprocess.Popen = opened
sys.argv = [main]
runpy.run_path(main, run_name='__main__')
'''

def _is_competition_rerun() -> bool:
    return os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {
        '1', 'true', 'yes', 'y', 'on'}


if DRY_UNLESS_RERUN and not _is_competition_rerun():
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    DRY_MARKER.write_text('dry build: no vLLM, no games\n', encoding='utf-8')
    print('duck.kaggle: DRY BUILD -- KAGGLE_IS_COMPETITION_RERUN is not set, so no '
          'runtime is unpacked, no weights are read and no server is started. This '
          'version exists to be submitted; the scored rerun sets the flag and runs '
          'in full.', flush=True)
    raise SystemExit(0)


import concurrent.futures
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import sysconfig
import tarfile
import time
from pathlib import Path, PurePosixPath

WARM_CACHE_NAME = 'vllm-warm-caches.tar.gz'
WARM_CACHE_HEAD = 'WARM_CACHE_KEY.json'
WARM_CACHE_PLAN = 'warm-cache-plan.json'
WARM_CACHE_CAP = 2 << 30
WARM_ROOT_NAMES = ('CACHE_ROOT', 'COMPILE_CACHE_ROOT')
WARM_PYCACHE = 'pycache'
WARM_READ_PIECE = 64 << 20
WARM_ROOT_LINE = r'^([A-Z_]*ROOT) = Path\("([^"]+)"\)$'


def warm_roots(serving_text: str) -> dict:
    return {name: Path(path) for name, path in re.findall(WARM_ROOT_LINE, serving_text, re.M)}


def warm_site(runtime_root) -> Path:
    version = f'python{sys.version_info[0]}.{sys.version_info[1]}'
    return Path(runtime_root) / 'usr' / 'local' / 'lib' / version / 'dist-packages'


def warm_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def warm_gpu() -> str:
    try:
        probe = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f'unknown ({type(exc).__name__})'
    names = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    return ' + '.join(names) or f'unknown (nvidia-smi exited {probe.returncode})'


def warm_key(runtime_manifest, roots: dict, gpu: str, runtime_root) -> dict:
    return {
        'runtime_manifest_sha256': warm_sha256(runtime_manifest),
        'gpu': gpu,
        'python': f'{sys.implementation.cache_tag} {sys.version.split()[0]}',
        'roots': {name: str(path) for name, path in sorted(roots.items())},
        'runtime_root': str(runtime_root),
    }


def warm_units(roots: dict) -> list:
    units = []
    for name, root in sorted(roots.items()):
        root = Path(root)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            whole = child.is_dir() and not child.is_symlink()
            paths = sorted(child.rglob('*')) if whole else [child]
            members, size = [], 0
            for path in paths:
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    size += path.stat().st_size
                except OSError:
                    continue
                members.append((path, name + '/' + path.relative_to(root).as_posix()))
            if members:
                units.append((name + '/' + child.name, members, size))
    return units


def warm_mb(size: int) -> str:
    return f'{size / 2**20:.0f} MB'


def pack_warm_caches(roots: dict, archive, key: dict, cap: int = WARM_CACHE_CAP, say=print) -> dict:
    began = time.time()
    archive = Path(archive)
    units = warm_units(roots)
    taken, left, held = [], [], 0
    for unit in sorted(units, key=lambda unit: (unit[2], unit[0])):
        if held + unit[2] > cap:
            left.append(unit)
            continue
        taken.append(unit)
        held += unit[2]
    if not taken:
        say('WARM_CACHE pack: nothing to pack -- '
            + ('the cache roots hold no files' if not units
               else f'every cache folder is over the {warm_mb(cap)} cap'))
        return {}
    partial = archive.with_name(archive.name + '.partial')
    files, gone = {}, 0
    with tarfile.open(partial, 'w:gz', compresslevel=1) as tar:
        head = json.dumps({'key': key, 'packed_epoch': int(time.time())},
                          sort_keys=True).encode('utf-8')
        info = tarfile.TarInfo(WARM_CACHE_HEAD)
        info.size = len(head)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(head))
        for label, members, _ in taken:
            for path, name in members:
                try:
                    tar.add(str(path), arcname=name, recursive=False)
                except OSError:
                    gone += 1
                    continue
                files[label] = files.get(label, 0) + 1
    os.replace(partial, archive)
    size = archive.stat().st_size
    count = sum(files.values())
    say(f'WARM_CACHE packed {count} files, {warm_mb(held)} of cache, into {archive} '
        f'({warm_mb(size)}) in {time.time() - began:.1f} s: '
        + ', '.join(f'{label} {files.get(label, 0)} files {warm_mb(unit_size)}'
                    for label, _, unit_size in taken)
        + (f'; {gone} file(s) vanished while packing' if gone else '')
        + (f'; LEFT OUT over the {warm_mb(cap)} cap: '
           + ', '.join(f'{label} {warm_mb(unit_size)}' for label, _, unit_size in left)
           if left else ''))
    return {'archive': str(archive), 'files': count, 'bytes': size, 'cache_bytes': held,
            'units': [label for label, _, _ in taken], 'left_out': [label for label, _, _ in left]}


def warm_read_ahead(path, threads: int = 16) -> float:
    began = time.time()
    size = Path(path).stat().st_size

    def one(start: int) -> None:
        handle = os.open(str(path), os.O_RDONLY | getattr(os, 'O_BINARY', 0))
        try:
            os.lseek(handle, start, os.SEEK_SET)
            left = min(WARM_READ_PIECE, size - start)
            while left > 0:
                block = os.read(handle, min(8 << 20, left))
                if not block:
                    break
                left -= len(block)
        finally:
            os.close(handle)

    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, range(0, size, WARM_READ_PIECE)))
    return time.time() - began


def warm_find(mount) -> Path | None:
    mount = Path(mount)
    if not mount.is_dir():
        return None
    found = sorted(mount.rglob(WARM_CACHE_NAME))
    return found[0] if found else None


def warm_target(roots: dict, name: str):
    top, _, rest = name.partition('/')
    parts = PurePosixPath(rest).parts
    if top not in roots or not rest or rest.startswith('/') or '..' in parts or '\\' in rest:
        return None
    return Path(roots[top]).joinpath(*parts)


def restore_warm_caches(archive, key: dict, roots: dict, say=print) -> dict:
    began = time.time()
    try:
        tar = tarfile.open(archive, 'r:gz')
    except (OSError, tarfile.TarError) as exc:
        say(f'WARM_CACHE cold: {archive} does not open as a gzip tar ({exc!r})')
        return {}
    restored, refused, size = {}, 0, 0
    with tar:
        head = tar.next()
        if head is None or head.name != WARM_CACHE_HEAD or not head.isfile():
            say(f'WARM_CACHE cold: {archive} does not begin with {WARM_CACHE_HEAD}, so nothing '
                f'says what it was packed for')
            return {}
        said = json.loads(tar.extractfile(head).read().decode('utf-8'))
        packed_for = said.get('key') or {}
        wrong = [f'{field}: packed for {packed_for.get(field)!r}, this run has {key.get(field)!r}'
                 for field in sorted(set(packed_for) | set(key))
                 if packed_for.get(field) != key.get(field)]
        if wrong:
            say('WARM_CACHE cold: the archive does not match this run -- ' + '; '.join(wrong))
            return {}
        for member in iter(tar.next, None):
            target = warm_target(roots, member.name)
            if target is None or not (member.isfile() or member.isdir()):
                refused += 1
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            landing = target.with_name(target.name + '.warm-partial')
            source = tar.extractfile(member)
            with open(landing, 'wb') as sink:
                while True:
                    block = source.read(8 << 20)
                    if not block:
                        break
                    sink.write(block)
            os.chmod(landing, (member.mode & 0o777) | 0o600)
            os.utime(landing, (member.mtime, member.mtime))
            os.replace(landing, target)
            label = member.name.split('/', 2)[0] + '/' + member.name.split('/', 2)[1]
            restored[label] = restored.get(label, 0) + 1
            size += member.size
    count = sum(restored.values())
    say(f'WARM_CACHE restored {count} files, {warm_mb(size)}, from {archive} in '
        f'{time.time() - began:.1f} s (packed at epoch {said.get("packed_epoch")}): '
        + (', '.join(f'{label} {number}' for label, number in sorted(restored.items()))
           or 'the archive held no cache files')
        + (f'; {refused} member(s) refused as outside the cache roots' if refused else ''))
    return {'files': count, 'bytes': size, 'units': sorted(restored), 'refused': refused,
            'seconds': time.time() - began}


def warm_compile(sources: list, prefix, say=print, timeout: int = 600) -> dict:
    prefix = Path(prefix)
    found = [str(Path(source)) for source in sources if Path(source).is_dir()]
    if not found:
        say(f'WARM_CACHE bytecode: none of {[str(source) for source in sources]} is a directory, '
            f'nothing compiled')
        return {}
    began = time.time()
    env = dict(os.environ)
    env['PYTHONPYCACHEPREFIX'] = str(prefix)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        done = subprocess.run(
            [sys.executable, '-m', 'compileall', '-q', '-j', '0', '--invalidation-mode',
             'timestamp', '-x', r'[/\\]tests?[/\\]', *found],
            env=env, capture_output=True, text=True, timeout=timeout)
        code = done.returncode
    except subprocess.TimeoutExpired:
        code = 'timeout'
    written = sum(1 for _ in prefix.rglob('*.pyc')) if prefix.is_dir() else 0
    say(f'WARM_CACHE bytecode: compiled {", ".join(found)} into {prefix} in '
        f'{time.time() - began:.0f} s, {written} .pyc files (compileall exited {code})')
    return {'pyc': written, 'returncode': code}


def warm_plan(roots: dict, key: dict, bytecode: bool, runtime_root) -> dict:
    return {'key': key, 'roots': {name: str(path) for name, path in sorted(roots.items())},
            'bytecode': bool(bytecode),
            'sources': [str(warm_site(runtime_root)), sysconfig.get_paths()['stdlib']],
            'pycache': str(Path(roots['COMPILE_CACHE_ROOT']) / WARM_PYCACHE)}


def warm_pack_from_plan(working_dir, say=print) -> dict:
    working_dir = Path(working_dir)
    plan_path = working_dir / WARM_CACHE_PLAN
    if not plan_path.is_file():
        say(f'WARM_CACHE pack: no {WARM_CACHE_PLAN} in {working_dir}, so the setup never reached '
            f'the server and there is nothing warm to keep')
        return {}
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('bytecode'):
        warm_compile(plan.get('sources') or [], plan['pycache'], say=say)
    roots = {name: Path(path) for name, path in (plan.get('roots') or {}).items()}
    return pack_warm_caches(roots, working_dir / WARM_CACHE_NAME, plan.get('key') or {}, say=say)


def note(line: str) -> None:
    stamped = f'{time.strftime("%H:%M:%S")} {line}'
    print(stamped, flush=True)
    try:
        with SETUP_LOG.open('a', encoding='utf-8') as handle:
            handle.write(stamped + chr(10))
    except OSError:
        pass


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug,
                         Path('/kaggle/input/datasets') / owner / slug,
                         Path('/kaggle/input/models') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


SERVING_MOUNT = resolve_kaggle_dataset_path(SERVING_OWNER, SERVING_SLUG)
RUNTIME_MOUNT = resolve_kaggle_dataset_path(RUNTIME_OWNER, RUNTIME_SLUG)
MODEL_MOUNT = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)
SERVING_MAIN = SERVING_MOUNT / SERVING_SCRIPT


def note_environment() -> None:
    note(f'serving mount: {SERVING_MOUNT} exists={SERVING_MOUNT.exists()}')
    note(f'serving script: {SERVING_MAIN} exists={SERVING_MAIN.is_file()}')
    note(f'runtime mount: {RUNTIME_MOUNT} exists={RUNTIME_MOUNT.exists()}')
    note(f'model mount: {MODEL_MOUNT} exists={MODEL_MOUNT.exists()}')
    note(f'served model name: {SERVED_MODEL_NAME}')
    note(f'serving profile: {json.dumps(SERVING_ENV, sort_keys=True)}')
    try:
        meminfo = Path('/proc/meminfo').read_text(encoding='utf-8').split('\n')[:3]
        usage = shutil.disk_usage(str(WORKING_DIR))
        note(f'host: cpus={os.cpu_count()} {" ".join(line.strip() for line in meminfo)} '
             f'disk_free_gb={usage.free / 2**30:.0f} at {WORKING_DIR}')
    except Exception as exc:
        note(f'host probe failed: {type(exc).__name__}: {exc}')
    if shutil.which('nvidia-smi'):
        probe = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'], capture_output=True, text=True)
        note(f'nvidia-smi rc={probe.returncode} out={probe.stdout.strip()!r} '
             f'err={probe.stderr.strip()[:200]!r}')
    else:
        note('nvidia-smi is not on PATH')
    note(f'expected gpu type: {os.getenv("KAGGLE_GPU_TYPE", "rtx-pro-6000")!r} '
         f'count={os.getenv("KAGGLE_GPU_COUNT", "1")!r}')


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', 'rtx-pro-6000').strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', '1')
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    note(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}')


def assert_inputs_present() -> None:
    wanted = {
        'serving script': SERVING_MAIN,
        'patch directory': SERVING_MOUNT / PATCH_DIRNAME,
        'source identity': SERVING_MOUNT / IDENTITY_NAME,
        'runtime manifest': RUNTIME_MOUNT / RUNTIME_MANIFEST,
        'model directory': MODEL_MOUNT,
    }
    absent = [f'{label} at {path}' for label, path in wanted.items() if not path.exists()]
    if absent:
        raise FileNotFoundError(
            'the serving inputs this build names are not mounted: ' + '; '.join(absent))
    for label, path in sorted(wanted.items()):
        note(f'input present: {label} -> {path}')
    mounted = hashlib.sha256(SERVING_MAIN.read_bytes()).hexdigest()
    stamped = str(json.loads((SERVING_MOUNT / IDENTITY_NAME).read_text(encoding='utf-8')).get('serving_setup_sha256') or '')
    if mounted != SERVING_SETUP_SHA256 or stamped != SERVING_SETUP_SHA256:
        raise RuntimeError(f'{SERVING_MAIN} is {mounted[:16]} and its identity stamps {stamped[:16]}; this profile '
                           f'was built for {SERVING_SETUP_SHA256[:16]}, so the mounted serving dataset is another version')
    lacking = [f'{name}:{knob}' for name, knobs in sorted(SERVING_REQUIRES.items()) for knob in knobs
               if knob not in (SERVING_MOUNT / name).read_text(encoding='utf-8', errors='replace')]
    if lacking:
        raise RuntimeError(f'the mounted serving dataset does not carry the code these knobs name: {lacking}')
    note(f'serving dataset matches this profile: serving_setup.py {mounted[:16]}, '
         f'{sum(len(knobs) for knobs in SERVING_REQUIRES.values())} knob names found in their files')


def serving_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(name): str(text) for name, text in SERVING_ENV.items()})
    env['TAAF_KAGGLE_BUNDLE_DIR'] = str(SERVING_MOUNT)
    env['TAAF_KAGGLE_WORKING_DIR'] = str(WORKING_DIR)
    env['TAAF_KAGGLE_SETUP_ENV'] = os.environ['TAAF_KAGGLE_SETUP_ENV']
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    hook = Path('/tmp/taaf_logit_bias_hook')
    hook.mkdir(parents=True, exist_ok=True)
    (hook / 'sitecustomize.py').write_text('import importlib.abc\nimport importlib.machinery\nimport sys\n\nTARGET = "vllm.sampling_params"\n\n\ndef patch(module):\n    cls = getattr(module, "SamplingParams", None)\n    original = getattr(cls, "_validate_spec_decode", None)\n    if original is None:\n        print("LOGIT_BIAS_SPEC not armed: SamplingParams._validate_spec_decode is missing", file=sys.stderr, flush=True)\n        return\n    if getattr(original, "taaf_logit_bias", False):\n        return\n\n    def _validate_spec_decode(self, speculative_config):\n        bias = self.logit_bias\n        if not bias:\n            return original(self, speculative_config)\n        self.logit_bias = None\n        try:\n            return original(self, speculative_config)\n        finally:\n            self.logit_bias = bias\n\n    _validate_spec_decode.taaf_logit_bias = True\n    cls._validate_spec_decode = _validate_spec_decode\n    print("LOGIT_BIAS_SPEC armed: SamplingParams._validate_spec_decode lets logit_bias through with speculative "\n          "decoding; min_p is still refused", file=sys.stderr, flush=True)\n\n\nclass _Loader(importlib.abc.Loader):\n    def __init__(self, inner):\n        self.inner = inner\n\n    def create_module(self, spec):\n        return self.inner.create_module(spec)\n\n    def exec_module(self, module):\n        self.inner.exec_module(module)\n        patch(module)\n\n    def __getattr__(self, name):\n        return getattr(self.inner, name)\n\n\nclass _Finder(importlib.abc.MetaPathFinder):\n    def find_spec(self, name, path=None, target=None):\n        if name != TARGET:\n            return None\n        spec = importlib.machinery.PathFinder.find_spec(name, path)\n        if spec is None or spec.loader is None:\n            return spec\n        spec.loader = _Loader(spec.loader)\n        return spec\n\n\nif not any(isinstance(finder, _Finder) for finder in sys.meta_path):\n    sys.meta_path.insert(0, _Finder())\n', encoding='utf-8')
    env['PYTHONPATH'] = os.pathsep.join([str(hook)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    note(f'LOGIT_BIAS_SPEC hook written to {hook}/sitecustomize.py, first on the server PYTHONPATH')
    prefix = warm_pycache_prefix()
    if prefix is not None:
        env['PYTHONPYCACHEPREFIX'] = str(prefix)
    return env


def warm_cache_roots() -> dict:
    try:
        return warm_roots(SERVING_MAIN.read_text(encoding='utf-8'))
    except OSError:
        return {}


def warm_pycache_prefix() -> Path | None:
    if not (WARM_CACHE and WARM_CACHE_BYTECODE):
        return None
    named = warm_cache_roots()
    return named['COMPILE_CACHE_ROOT'] / WARM_PYCACHE if 'COMPILE_CACHE_ROOT' in named else None


def prepare_warm_cache() -> None:
    if not WARM_CACHE:
        return
    try:
        named = warm_cache_roots()
        wanted = [*WARM_ROOT_NAMES, 'RUNTIME_ROOT']
        if any(name not in named for name in wanted):
            note(f'WARM_CACHE off: {SERVING_MAIN} names {sorted(named)}, not every one of '
                 f'{wanted}, so this run neither restores nor packs a cache')
            return
        roots = {name: named[name] for name in WARM_ROOT_NAMES}
        key = warm_key(RUNTIME_MOUNT / RUNTIME_MANIFEST, roots, warm_gpu(), named['RUNTIME_ROOT'])
        plan = warm_plan(roots, key, WARM_CACHE_BYTECODE, named['RUNTIME_ROOT'])
        (WORKING_DIR / WARM_CACHE_PLAN).write_text(json.dumps(plan, indent=2, sort_keys=True),
                                                   encoding='utf-8')
        note(f'WARM_CACHE key {json.dumps(key, sort_keys=True)}; the teardown packs '
             + ', '.join(f'{name}={path}' for name, path in sorted(roots.items()))
             + f' into {WORKING_DIR / WARM_CACHE_NAME}'
             + (f'; the server reads Python bytecode from {plan["pycache"]} and the teardown '
                f'compiles {", ".join(plan["sources"])} into it' if WARM_CACHE_BYTECODE else ''))
        if not WARM_CACHE_DATASET:
            note('WARM_CACHE cold: no ARC_KAGGLE_WARM_CACHE_DATASET is named, so this run starts '
                 'from empty caches and only packs its own')
            return
        mount = resolve_kaggle_dataset_path(*WARM_CACHE_DATASET.split('/', 1))
        archive = warm_find(mount)
        if archive is None:
            note(f'WARM_CACHE cold: no {WARM_CACHE_NAME} under {mount} (exists={mount.exists()})')
            return
        seconds = warm_read_ahead(archive)
        note(f'WARM_CACHE read {archive} ({archive.stat().st_size / 2**20:.0f} MB) in '
             f'{seconds:.1f} s')
        restore_warm_caches(archive, key, roots, say=note)
    except Exception as exc:
        note(f'WARM_CACHE failed and the server starts with whatever landed: '
             f'{type(exc).__name__}: {exc}')


ENGINE_DEAD_MARKS = ('Engine core initialization failed',
                     'EngineCore failed to start',
                     'Failed core proc(s)')


def engine_death(log_path: Path) -> str:
    if not log_path.is_file():
        return ''
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''
    for mark in ENGINE_DEAD_MARKS:
        if mark in text:
            lines = [line for line in text.splitlines()
                     if 'Error: ' in line and 'Traceback' not in line
                     and not any(m in line for m in ENGINE_DEAD_MARKS)]
            return (lines[-1] if lines else mark)[-400:]
    return ''


def serving_python() -> str:
    if sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12):
        return sys.executable
    for candidate in ('/usr/bin/python3.12', shutil.which('python3.12') or ''):
        if candidate and Path(candidate).is_file():
            note(f'the serving runtime is built for CPython 3.12 and this kernel runs '
                 f'{sys.version_info.major}.{sys.version_info.minor}; {SERVING_MAIN.name} and the '
                 f'server run under {candidate}')
            return candidate
    raise RuntimeError(f'the serving runtime is built for CPython 3.12, this kernel runs '
                       f'{sys.version_info.major}.{sys.version_info.minor} and no python3.12 is on '
                       f'this worker')


def game_loop_pythonpath(pythonpath: str) -> tuple[str, list[str]]:
    own = f'python{sys.version_info.major}.{sys.version_info.minor}'
    kept, dropped = [], []
    for entry in [one for one in pythonpath.split(os.pathsep) if one]:
        versions = re.findall(r'(?:^|/)(python\d+\.\d+)(?=/|$)', entry)
        (dropped if versions and versions[-1] != own else kept).append(entry)
    return os.pathsep.join(kept), dropped


def run_serving_main() -> None:
    note(f'running {SERVING_MAIN} with TAAF_KAGGLE_BUNDLE_DIR={SERVING_MOUNT}')
    python = serving_python()
    command = [python, '-B', str(SERVING_MAIN)]
    if THINK_OPEN:
        command = [python, '-B', '-c', THINK_OPEN_WRAPPER, str(SERVING_MAIN),
                   THINK_OPEN_OLD, THINK_OPEN_NEW]
        note('THINK_OPEN: the serving script runs under a wrapper that hands vLLM the chat '
             'template with begun thoughts left open')
    if MEMGUARD_RELEASE_MIB > 0:
        if THINK_OPEN:
            raise RuntimeError('MEMGUARD and THINK_OPEN both wrap the serving script; this build allows one')
        command = [python, '-B', '-c', MEMGUARD_WRAPPER, str(SERVING_MAIN), str(MEMGUARD_RELEASE_MIB)]
        note(f'MEMGUARD: the serving script runs under a wrapper that makes the worker free cached blocks '
             f'whenever free device memory drops under {MEMGUARD_RELEASE_MIB} MiB')
    child = subprocess.Popen(command, env=serving_environment(), cwd=str(WORKING_DIR))
    while child.poll() is None:
        cause = engine_death(SERVER_LOG)
        if cause:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
            raise RuntimeError(f'the vLLM engine died during init and {SERVING_MAIN} '
                               f'would have waited out its own timeout: {cause}')
        time.sleep(SERVING_POLL_S)
    if child.returncode != 0:
        raise RuntimeError(f'{SERVING_MAIN} exited {child.returncode}; the server this '
                           f'build describes was never started')


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def tail_server_log(lines: int = 80) -> str:
    if not SERVER_LOG.exists():
        return ''
    return '\n'.join(SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def assert_serves_what_we_ask_for() -> None:
    listing = request_json(f'{VLLM_BASE_URL}/models', timeout=30)
    names = [str(row.get('id') or '') for row in (listing.get('data') or [])]
    if SERVED_MODEL_NAME not in names:
        raise RuntimeError(
            f'the server answers with {names}, and every request this run makes names '
            f'{SERVED_MODEL_NAME!r}, so no call would be answered')
    note(f'vLLM server ready and serving {names}')


def generated_tokens() -> float:
    with urllib.request.urlopen(f'http://{VLLM_HOST}:{VLLM_PORT}/metrics', timeout=30) as answer:
        text = answer.read().decode('utf-8', 'replace')
    return sum(float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
               if line.startswith('vllm:generation_tokens_total'))


def run_vllm_api_smoke_test() -> None:
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
        'temperature': 0.0,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    try:
        response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=180)
    except TimeoutError:
        before = generated_tokens()
        time.sleep(20)
        made = generated_tokens() - before
        if made <= 0:
            raise
        note(f'smoke test: the request waited past 180 s behind other traffic, and the server '
             f'generated {made:.0f} tokens in the next 20 s, so it serves; no smoke answer is shown')
        return
    generated = response['choices'][0]['message'].get('content', '').strip()
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER QWEN SMOKE TEST REAL MODEL OUTPUT', flush=True)
    print('Generated:', generated, flush=True)
    print('=' * 88 + '\n', flush=True)


def run_think_open_probe() -> None:
    root = f'http://{VLLM_HOST}:{VLLM_PORT}'
    thought = 'I add the two numbers:'
    messages = [{'role': 'user', 'content': 'What is 2 + 2? Answer with the number.'},
                {'role': 'assistant', 'content': '<think>' + chr(10) + thought}]
    asked = {'model': SERVED_MODEL_NAME, 'messages': messages,
             'add_generation_prompt': False, 'continue_final_message': True}
    tokens = request_json(f'{root}/tokenize', payload=asked, timeout=60)
    shown = request_json(f'{root}/detokenize', timeout=60, payload={
        'model': SERVED_MODEL_NAME, 'tokens': tokens.get('tokens') or []})
    prompt = str(shown.get('prompt') or '')
    tail = prompt.rsplit('<|im_start|>assistant', 1)[-1]
    note('THINK_OPEN probe, the end of the prompt the server builds: ' + repr(tail[-200:]))
    if not prompt.rstrip().endswith(thought) or '</think>' in tail:
        raise RuntimeError('THINK_OPEN is on but the served template closed the begun thought')
    reply = request_json(f'{VLLM_BASE_URL}/chat/completions', timeout=180, payload={
        **asked, 'temperature': 0.0, 'max_tokens': 256})
    message = reply['choices'][0]['message']
    said = str(message.get('reasoning') or message.get('reasoning_content') or '')
    content = str(message.get('content') or '')
    note('THINK_OPEN probe reply: reasoning ' + repr(said[:200]) + ', content '
         + repr(content[:120]))
    if not said.strip() or '<think>' in content or '</think>' in content:
        raise RuntimeError('THINK_OPEN is on but the reply parser did not keep the continued '
                           'thought as reasoning: ' + repr(said[:120]) + ' / ' + repr(content[:120]))
    budget = 24
    begun = 'I will go through the numbers one by one:'
    longer = [{'role': 'user', 'content': 'List every prime number below 300, then answer with '
               'how many there are.'},
              {'role': 'assistant', 'content': '<think>' + chr(10) + begun}]
    cut = request_json(f'{VLLM_BASE_URL}/chat/completions', timeout=180, payload={
        'model': SERVED_MODEL_NAME, 'messages': longer, 'add_generation_prompt': False,
        'continue_final_message': True, 'temperature': 0.0, 'max_tokens': 400,
        'thinking_token_budget': budget})
    choice = cut['choices'][0]
    thought = str(choice['message'].get('reasoning') or choice['message'].get('reasoning_content') or '')
    answer = str(choice['message'].get('content') or '')
    note(f'THINK_BUDGET probe ({budget} tokens on a continued thought): reasoning '
         f'{len(thought)} chars ending {thought[-120:]!r}, content {answer[:120]!r}, '
         f'finish {choice.get("finish_reason")!r}, completion tokens '
         f'{(cut.get("usage") or {}).get("completion_tokens")!r}')
    if '<think>' in answer or '</think>' in answer:
        raise RuntimeError('the thinking budget closed the thought but the parser left the '
                           'markers in the content: ' + repr(answer[:160]))
    if not answer.strip() and len(thought) > 20 * budget:
        raise RuntimeError(f'thinking_token_budget={budget} did not close a continued thought: '
                           f'{len(thought)} chars of reasoning and no content')


def served_files(inputs: Path, model_root: Path, kept: str) -> tuple[list[Path], int, int]:
    manifests = sorted(model_root.rglob('MODEL_MANIFEST.json')) if model_root.is_dir() else []
    configs = [path.parent for path in sorted(model_root.rglob('config.json'))
               if any(path.parent.glob('*.safetensors'))] if model_root.is_dir() else []
    model_dir = manifests[0].parent if manifests else configs[0] if configs else model_root
    baked = None
    for stamp in sorted(inputs.rglob(PREBAKED_STAMP)) if inputs.is_dir() else []:
        try:
            said = json.loads(stamp.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if str(said.get('experts_kept') or '') == str(kept):
            baked = stamp.parent
            break
    chosen: dict[str, Path] = {}
    if baked is not None:
        for path in sorted(baked.iterdir()):
            if path.is_file() and path.name != PREBAKED_STAMP:
                chosen[path.name] = path
    from_baked = len(chosen)
    if model_dir.is_dir():
        for path in sorted(model_dir.iterdir()):
            if path.is_file() and path.name not in chosen:
                chosen[path.name] = path
    return [chosen[name] for name in sorted(chosen)], from_baked, len(chosen) - from_baked


def natural_key(name: str) -> list:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'([0-9]+)', name)]


def loader_position(text: str, least: int = 1) -> tuple[int, int] | None:
    import re

    found = [(int(done), int(total)) for done, total in re.findall(LOADER_PROGRESS, text)
             if int(total) >= least]
    return found[-1] if found else None


def read_piece(path: Path, start: int, length: int) -> str:
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        return f'{path}: {exc!r}'
    try:
        done = 0
        while done < length:
            block = os.pread(handle, min(READ_BLOCK, length - done), start + done)
            if not block:
                break
            done += len(block)
    except OSError as exc:
        return f'{path}: {exc!r}'
    finally:
        os.close(handle)
    return ''


def drop_cached(path: Path) -> int:
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        return path.stat().st_size
    except (OSError, AttributeError):
        return 0
    finally:
        os.close(handle)


def log_tail(path: Path, size: int = 1 << 16) -> str:
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - size))
            return handle.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def pieces_of(sizes: list[int], piece: int) -> list[tuple[int, int, int]]:
    return [(index, start, min(piece, size - start))
            for index, size in enumerate(sizes) for start in range(0, size, piece)]


def loaded_bytes(sizes: list[int], position: int, total: int) -> int:
    done = min(len(sizes), max(0, round(position * len(sizes) / total))) if total else 0
    return sum(sizes[:done])


def one_stream(chosen: list[tuple[Path, int]]) -> str:
    read, spent = 0, 0.0
    for path, size in chosen:
        length = min(size, PROBE_BYTES - read)
        if length <= 0:
            break
        began = time.time()
        if not read_piece(path, 0, length):
            read += length
        spent += time.time() - began
    return f'{read / 1e9:.2f} GB at {read / max(0.001, spent) / 1e9:.2f} GB/s' if read else 'nothing read'


def cgroup_memory(base: Path = Path('/sys/fs/cgroup')) -> str:
    try:
        ceiling = (base / 'memory.max').read_text(encoding='utf-8').strip()
        used = int((base / 'memory.current').read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return 'cgroup memory unreadable'
    parts = {}
    try:
        for line in (base / 'memory.stat').read_text(encoding='utf-8').splitlines():
            name, _, number = line.partition(' ')
            if name in ('anon', 'file', 'shmem') and number.strip().isdigit():
                parts[name] = int(number)
    except OSError:
        pass
    limit = f'{int(ceiling) / 2**30:.1f}' if ceiling.isdigit() else ceiling
    shown = ', '.join(f'{name} {size / 2**30:.1f}' for name, size in sorted(parts.items()))
    return f'cgroup {used / 2**30:.1f} of {limit} GiB' + (f' ({shown})' if shown else '')


def prefetch_left_out(weights: list[Path]) -> tuple[list[Path], list[str]]:
    import re

    ple = str(SERVING_ENV.get('TAAF_VLLM_PLE_DF11') or '') == '1'
    cut = str(SERVING_ENV.get('TAAF_VLLM_EXPERT_CUT_AT_LOAD') or '') == '1'
    if not (ple or cut):
        return weights, []
    expert = re.compile(r'^(?:model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\.mlp\.experts\.\d+\.')
    names: dict[Path, list[str]] = {}
    for folder in sorted({path.parent for path in weights}):
        index = folder / 'model.safetensors.index.json'
        try:
            mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
        except (OSError, ValueError, KeyError):
            continue
        for tensor, file in mapping.items():
            names.setdefault(folder / file, []).append(tensor)
    kept, left = [], []
    for path in weights:
        held = names.get(path)
        if held and ple and all('ple_embedding.ngram_embedding' in one for one in held):
            left.append(f'{path.name} (PLE, read by the DF11 encoder)')
        elif held and cut and any(expert.match(one) for one in held):
            left.append(f'{path.name} (experts, read as kept byte ranges by the loader)')
        else:
            kept.append(path)
    return kept, left


def prefetch_weights(threads: int) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from itertools import accumulate

    began = time.time()
    files, baked, mount = served_files(Path('/kaggle/input'), MODEL_MOUNT,
                                       str(SERVING_ENV.get('TAAF_VLLM_EXPERTS_KEPT') or ''))
    small = [path for path in files if not path.name.endswith('.safetensors')]
    weights = sorted((path for path in files if path.name.endswith('.safetensors')),
                     key=lambda path: natural_key(path.name))
    weights, left_out = prefetch_left_out(weights)
    if left_out:
        note(f'WEIGHTS_PREFETCH leaves {len(left_out)} file(s) to their own readers: ' + ', '.join(left_out))
    sizes = [path.stat().st_size for path in weights]
    whole = sum(sizes)
    note(f'WEIGHTS_PREFETCH started: {len(weights)} weight files, {whole / 1e9:.1f} GB '
         f'({baked} files from the prebake folder, {mount} from the model mount), {threads} '
         f'threads reading {READ_PIECE >> 20} MB pieces, at most '
         f'{PREFETCH_AHEAD_BYTES / 1e9:.0f} GB ahead of the loader')
    held = [(path, size) for path, size in zip(weights, sizes)
            if (path.parent / PREBAKED_STAMP).is_file()]
    carried = [(path, size) for path, size in zip(weights, sizes)
               if not (path.parent / PREBAKED_STAMP).is_file()]
    note(f'WEIGHTS_PREFETCH one stream: the model mount {one_stream(carried[-1:])}; the '
         f'prebake dataset {one_stream(held[:4])}')
    pieces = pieces_of(sizes, READ_PIECE)
    offsets = [0, *accumulate(sizes)]
    failed: list[str] = []
    running: dict = {}
    marks: list[str] = []
    upto = read_bytes = passes = reported = 0
    held_back = 0.0
    evicted = dropped = 0
    last_position = -1
    last_seen = None
    position, total = 0, len(weights)
    loader_noted = False
    started = mark_at = time.time()
    mark_bytes = 0
    finished_at = None
    with ThreadPoolExecutor(threads) as pool:
        failed += [said for said in pool.map(lambda path: read_piece(path, 0, path.stat().st_size),
                                             small) if said]
        while True:
            tail = log_tail(SERVER_LOG)
            seen = loader_position(tail, max(1, len(weights) // 2))
            position, total = seen if seen else (0, len(weights))
            if seen and position < last_position:
                passes += 1
            last_position = position if seen else last_position
            if seen and not loader_noted:
                loader_noted = True
                note(f'WEIGHTS_PREFETCH loader began after {time.time() - began:.0f} s with '
                     f'{read_bytes / 1e9:.1f} GB read; {cgroup_memory()}')
            if seen and seen != last_seen and seen[0] in (1, total // 2, total):
                marks.append(f'pass {passes + 1} {seen[0]}/{total} at {time.time() - began:.0f} s')
            last_seen = seen
            if WEIGHTS_EVICT and seen and passes == 0:
                behind = min(len(weights), max(0, round(position * len(weights) / total))) - EVICT_BEHIND
                while evicted < behind:
                    dropped += drop_cached(weights[evicted])
                    evicted += 1
            limit = (loaded_bytes(sizes, position, total) if passes == 0 else whole) \
                + PREFETCH_AHEAD_BYTES
            while upto < len(pieces) and len(running) < threads:
                index, start, length = pieces[upto]
                if offsets[index] + start >= limit:
                    break
                running[pool.submit(read_piece, weights[index], start, length)] = length
                upto += 1
            blocked = upto < len(pieces) and len(running) < threads
            for future in [one for one in running if one.done()]:
                said = future.result()
                if said:
                    failed.append(said)
                read_bytes += running.pop(future)
            while reported < 10 and read_bytes >= whole * (reported + 1) / 10:
                reported += 1
                now = time.time()
                note(f'WEIGHTS_PREFETCH {reported * 10}%: {read_bytes / 1e9:.1f} GB in '
                     f'{now - started:.0f} s, {(read_bytes - mark_bytes) / 1e9 / max(0.001, now - mark_at):.2f}'
                     f' GB/s since the last mark, {held_back:.0f} s held back by the window; loader '
                     + (f'pass {passes + 1} at {position}/{total}' if seen else 'not started')
                     + '; ' + cgroup_memory())
                mark_bytes, mark_at = read_bytes, now
            if finished_at is None and upto >= len(pieces) and not running:
                finished_at = time.time()
            if 'Model loading took' in tail or time.time() - began > 3600:
                break
            time.sleep(0.2)
            if blocked:
                held_back += 0.2
    spent = max(0.001, (finished_at or time.time()) - started)
    note(f'WEIGHTS_PREFETCH done: {read_bytes / 1e9:.1f} GB read in {spent:.0f} s '
         f'({read_bytes / 1e9 / spent:.2f} GB/s, {held_back:.0f} s of it held back by the '
         f'window); loader {", ".join(marks) or "not seen"}; {len(failed)} unreadable'
         + (f'; {dropped / 1e9:.1f} GB of {evicted} loaded file(s) dropped from the page cache behind '
            f'the loader' if WEIGHTS_EVICT else '')
         + (f': {failed[0]}' if failed else '') + '; ' + cgroup_memory())


def start_prefetch() -> None:
    import threading

    if WEIGHTS_PREFETCH <= 0:
        return
    threading.Thread(target=prefetch_weights, args=(WEIGHTS_PREFETCH,), daemon=True).start()


STARTUP_MARKS = (
    r'Filesystem type for checkpoints: [^\n]{0,120}',
    r'MTP_OWN_READS[^\n]{0,120}',
    r'kv cache group sizes [^\n]{0,80}',
    r'Prefetching checkpoint files into page cache finished in [0-9.]+s',
    r'Loading safetensors checkpoint shards: 100% Completed \| [0-9]+/[0-9]+ \[[0-9:]+',
    r'Loading weights took [0-9.]+ seconds',
    r'Model loading took [^\n]{0,80}',
    r'Memory profiling takes [^\n]{0,80}',
    r'torch.compile takes [^\n]{0,80}',
    r'Graph capturing finished in [^\n]{0,80}',
    r'init engine \(profile, create kv cache, warmup model\) took [0-9.]+ seconds',
    r'Available KV cache memory: [^\n]{0,60}',
    r'GPU KV cache size: [^\n]{0,60}',
    r'Maximum concurrency for [^\n]{0,80}',
)


def note_startup() -> None:
    import re

    try:
        text = SERVER_LOG.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        note(f'VLLM_STARTUP the server log could not be read: {exc!r}')
        return
    import re as _re

    offload = _re.search(r'SimpleCPUOffloadConnector: role=WORKER, per_rank=([0-9.]+) GB', text)
    note('KV_OFFLOAD ' + (f'ON: {offload.group(1)} GB of host memory hold evicted KV blocks'
                          if offload else 'OFF: the server runs without host KV offloading'))
    for mark in STARTUP_MARKS:
        seen = []
        for found in re.findall(mark, text):
            if found not in seen:
                seen.append(found)
        note('VLLM_STARTUP ' + (' | '.join(seen) if seen else 'no line like ' + mark))


note_environment()
try:
    assert_expected_cuda_gpu()
    assert_inputs_present()
    start_prefetch()
    prepare_warm_cache()
    run_serving_main()
    note_startup()
    assert_serves_what_we_ask_for()
    run_vllm_api_smoke_test()
    if THINK_OPEN:
        run_think_open_probe()
except BaseException as exc:
    note(f'SETUP FAILED: {type(exc).__name__}: {exc}')
    note('server log tail:' + chr(10) + tail_server_log())
    raise
note('setup complete: the server answered the smoke test')
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
missing = [name for name in CARRIED_KEYS if not str(existing_setup_env.get(name) or '').strip()]
if missing:
    raise RuntimeError(
        f'{SERVING_MAIN} left {missing} unset in {setup_env_path}, so the game loop '
        f'would run against a different interpreter environment than the server')
kept_path, server_only = game_loop_pythonpath(existing_setup_env['PYTHONPATH'])
if server_only:
    existing_setup_env['PYTHONPATH'] = kept_path
    note(f'the game loop runs under python{sys.version_info.major}.{sys.version_info.minor}, so '
         f'{len(server_only)} PYTHONPATH entries built for another interpreter stay with the server '
         f'only: {server_only}; the game loop imports from {kept_path or "its own site-packages"}')
setup_env = {
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    **{'ARC_CONFIG': 'swift-own-baked-150k-v030-prod4', 'ARC_PLUGINS': 'round_end', 'ARC_PLUGIN_OVERRIDE': '', 'LOCAL_ANALYZER_APP_NAME': 'ARC3 Kaggle Harness', 'LOCAL_ANALYZER_SEED': '-1', 'ARC_PROVIDER_ORDER': '', 'ARC_RUN_COST_CAP_USD': '0.0', 'LOCAL_ANALYZER_CONTEXT_WINDOW': '150000', 'LOCAL_ANALYZER_MAX_OUTPUT': '0', 'ARC_KV_BUDGET_TOKENS': '0', 'ARC_KV_GATE_REPLY_TOKENS': '16000', 'ARC_PRESS_RATE': '0.0', 'ARC_EYES_UPSCALES': '', 'ARC_EYES_CASES': '0', 'ARC_EYES_CHARS': '0', 'LOCAL_ANALYZER_ENABLE_THINKING': '1', 'LOCAL_ANALYZER_REASONING_EFFORT': '', 'LOCAL_ANALYZER_THINKING_BUDGET': '0', 'LOCAL_ANALYZER_TEMPERATURE': '0.6', 'LOCAL_ANALYZER_TOP_P': '0.95', 'LOCAL_ANALYZER_TOP_K': '20', 'LOCAL_ANALYZER_PRESENCE_PENALTY': '0.0', 'LOCAL_ANALYZER_MIN_P': '-1.0', 'LOCAL_ANALYZER_YIELD_SECONDS': '60.0', 'LOCAL_ANALYZER_TOOL_STEPS': '0', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '30', 'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '1024', 'ARC_MEMORY_WINDOW_TURNS': '30', 'ARC_MEMORY_WINDOW_TRIM': '1', 'ARC_MEMORY_TRIM_LOW_WATER': '0.8', 'ARC_MEMORY_OLD_REASONING': 'keep', 'ARC_MEMORY_KEEP_REASONING_TURNS': '1', 'ARC_MEMORY_KEPT_REASONING_TOKENS': '16000', 'ARC_LOG_PREFIX_REQUESTS': '0', 'ARC_MEMORY_KEEP_CAPPED_TEXT': '0', 'ARC_MEMORY_STICKY': '0', 'ARC_MEMORY_SAY_ONCE': '0', 'ARC_MEMORY_COUNT_TEXT': '0', 'ARC_MEMORY_COUNT_SERVED': '0', 'ARC_TOOL_SURFACE_ABOVE': '0', 'ARC_SNAPSHOT_ON': '0', 'ARC_RESUME_FROM_PREVIOUS_ARM': '0', 'ARC_RESUME_FROM': '', 'ARC_REPLAY_FROM': '', 'ARC_REPLAY_LIVE_S': '0', 'ARC_RESTART_LINE': '0', 'ARC_MEMORY_IMAGE_TOKENS': '120', 'ARC_MODEL_NOTE_CHARS': '2000', 'ARC_NOOP_GUARD': '', 'ARC_GAME_OVER_LINES': '1', 'ARC_MULTIMODAL_CONTEXT': 'current_grid', 'ARC_MULTIMODAL_UPSCALE': '4', 'ARC_MAX_IMAGES_PER_REQUEST': '1', 'ARC_CONTEXT_PAD_TOKENS': '0', 'ARC_EARLY_ABORT_SHARE': '0.0', 'ARC_PAUSE_TOKENS': '0', 'ARC_CHAMPION_STINT_TOKENS': '0', 'ARC_PAUSE_LEVELS': '2', 'ARC_DEAD_CLOCK_STOP': '1', 'ARC_DEAD_ENDPOINT_AFTER': '6', 'ARC_DEAD_ENDPOINT_RETRY_S': '120.0', 'ARC_ENDPOINT_REVIVE_S': '900.0', 'ARC_WATCHDOG_POLL_S': '20.0', 'ARC_WATCHDOG_DEAD_AFTER': '6', 'ARC_WATCHDOG_STALLED_AFTER': '9', 'ARC_WATCHDOG_SELFTEST_S': '0.0', 'ARC_WATCHDOG_PROFILE_AT_S': '0.0', 'ARC_WATCHDOG_PROFILE_FOR_S': '20.0', 'ARC_WATCHDOG_MAX_RESTARTS': '0', 'ARC_WATCHDOG_READY_S': '0.0', 'ARC_SERVE_STATS_S': '120.0', 'ARC_VALID_ARGUMENTS': '1', 'ARC_FRESH_STEP_SUMMARY': '1', 'ARC_ACTION_COST_ON': '0', 'ARC_ANIMATION_RELAY_ON': '0', 'ARC_ANIMATION_RELAY_WHY': '1', 'ARC_OBJECT_TABLE_ON': '0', 'ARC_OBJECT_TABLE_ROWS': '40', 'ARC_ASK_AROUND_ON': '0', 'ARC_ASK_AROUND_TURN': '5', 'ARC_ASK_AROUND_EVERY': '0', 'ARC_ASK_AROUND_ANSWERERS': '3', 'ARC_ASK_AROUND_CHARS': '200', 'ARC_ASK_AROUND_CONTEXT_CHARS': '9000', 'ARC_ASK_AROUND_SECONDS': '45.0', 'ARC_ASK_AROUND_ANSWERS_PER_TURN': '2', 'ARC_ASK_AROUND_HISTORY_CHARS': '3000', 'ARC_ASK_AROUND_PICTURES': '1', 'ARC_ASK_AROUND_SHOW_BOARDS': '0', 'ARC_ASK_TO_THINK_ON': '0', 'ARC_ASK_TO_THINK_EFFORT': 'low', 'ARC_ASK_TO_THINK_RAISED': 'xhigh', 'ARC_ASK_TO_THINK_MARKER': 'THINK', 'ARC_AUTO_DELEGATE_ON': '0', 'ARC_AUTO_DELEGATE_FRAMES': '1', 'ARC_AUTO_DELEGATE_EVERY': '2', 'ARC_AUTO_DELEGATE_CHARS': '500', 'ARC_AUTO_DELEGATE_EACH': '2', 'ARC_BOARD_VIDEO_ON': '0', 'ARC_BOARD_VIDEO_FRAMES': '6', 'ARC_BOARD_VIDEO_UPSCALE': '4', 'ARC_BOARD_VIDEO_MIN_CELLS': '8', 'ARC_BOARD_VIDEO_FORM': 'describe', 'ARC_BOARD_VIDEO_RETRIEVAL': '0', 'ARC_BOARD_VIDEO_GRID_LINES': '0', 'ARC_BOARD_VIDEO_CHARS': '350', 'ARC_BOARD_VIDEO_SECONDS': '30.0', 'ARC_BOARD_VIDEO_WHERE': 'tool_result', 'ARC_CLOCK_HALVES_ON': '0', 'ARC_CLOCK_TARGET_ON': '0', 'ARC_CLOCK_TARGET_LEVELS': '4', 'ARC_COMPACT_WINDOW_ON': '0', 'ARC_COMPACT_WINDOW_BOIL_CHARS': '0', 'ARC_COMPACT_WINDOW_BOIL_SECONDS': '60.0', 'ARC_COMPACT_WINDOW_CHARS': '1400', 'ARC_CORE_PRIORS_ON': '0', 'ARC_COUNTER_MASK_ON': '0', 'ARC_COUNTER_MASK_EVERY': '4', 'ARC_COUNTER_MASK_SETTLE': '2', 'ARC_COUNTER_MASK_OBJECT_LOOKS': '1', 'ARC_COUNTER_MASK_REPORT': '1', 'ARC_COUNTER_MASK_VIEW': '0', 'ARC_FACT_MEMORY_ON': '0', 'ARC_FACT_MEMORY_AGENT_WRITES': '0', 'ARC_FACT_MEMORY_CHARS': '8000', 'ARC_FACT_MEMORY_SECONDS': '90.0', 'ARC_FACT_MEMORY_SHOWN': '0', 'ARC_FACT_MEMORY_CONTEXT': '0', 'ARC_FACT_MEMORY_MAX_TOKENS': '1500', 'ARC_FACT_MEMORY_PREFIX': '1', 'ARC_FACT_MEMORY_REWRITE_VOICE': '0', 'ARC_FACT_MEMORY_REPLACES_NOTES': '0', 'ARC_FACT_MEMORY_PICTURES': '1', 'ARC_FACT_MEMORY_LEVEL_REWRITE': '1', 'ARC_FACT_MEMORY_PRUNE': '0', 'ARC_FACT_MEMORY_COST_LINE': '1', 'ARC_FACT_MEMORY_FORKS': '1', 'ARC_FACT_MEMORY_CHECK_THINKING': '0', 'ARC_FACT_MEMORY_EXPIRE': '0', 'ARC_FEW_BOARDS_ON': '0', 'ARC_FEW_BOARDS_BOARDS': '4', 'ARC_FIELD_NOTES_ON': '0', 'ARC_FIELD_NOTES_CHUNK': '60000', 'ARC_FIELD_NOTES_TURNS': '6', 'ARC_FIELD_NOTES_CHARS': '1200', 'ARC_FIELD_NOTES_KEEP': '6000', 'ARC_FIELD_NOTES_CATCHUP': '2', 'ARC_FIELD_NOTES_FOLD': '4', 'ARC_FIELD_NOTES_WHERE': 'system', 'ARC_FIELD_NOTES_SECONDS': '180.0', 'ARC_FIELD_NOTES_RETRIES': '2', 'ARC_FIRST_LEVEL_EXPLORER_ON': '0', 'ARC_FIRST_LEVEL_EXPLORER_EFFORT': 'low', 'ARC_INSTRUCTIONS_ONCE_ON': '0', 'ARC_INSTRUCTIONS_ONCE_NOTES': '1', 'ARC_KEPT_NAMESPACE_ON': '0', 'ARC_KEPT_NAMESPACE_FUNCTIONS': '40', 'ARC_KEPT_NAMESPACE_CODE': '1', 'ARC_KEPT_NAMESPACE_CODE_CHARS': '12000', 'ARC_KEPT_NAMESPACE_REMOVE_FUNC': '1', 'ARC_LEAN_PROMPT_ON': '0', 'ARC_LEAN_PROMPT_NOTES': '1', 'ARC_LESSONS_ON': '0', 'ARC_LESSONS_CHARS': '300', 'ARC_LESSONS_SECONDS': '60.0', 'ARC_LESSONS_FILE': 'lessons.txt', 'ARC_LESSONS_SHOW': '12', 'ARC_LEVEL_RECAP_ON': '0', 'ARC_LEVEL_SEAM_ON': '0', 'ARC_LEVEL_STORAGE_ON': '0', 'ARC_LEVEL_STORAGE_CHARS': '80000', 'ARC_LEVEL_STORAGE_GAME_CHARS': '80000', 'ARC_LOOK_ON': '0', 'ARC_LOOK_UPSCALE': '8', 'ARC_LOOK_KEEP': '2', 'ARC_LOOK_LEVEL_START': '1', 'ARC_LOOSE_LABELS_ON': '0', 'ARC_LOST_STORY_ON': '0', 'ARC_LOST_STORY_VIEWS': '3', 'ARC_LOST_STORY_KEPT': '0', 'ARC_LOW_EFFORT_ON': '0', 'ARC_LOW_EFFORT_LEVEL': 'low', 'ARC_MAESTRO_ON': '0', 'ARC_MAESTRO_LEVELS': '2', 'ARC_MAESTRO_CHARS': '1500', 'ARC_MAESTRO_EXPLORE_TOKENS': '10000', 'ARC_MEMORY_RESET_ON': '0', 'ARC_MEMORY_RESET_THRESHOLD_TOKENS': '120000', 'ARC_MEMORY_RESET_SECONDS': '180.0', 'ARC_MEMORY_RESET_THINKING_TOKENS': '4000', 'ARC_MEMORY_RESET_CHARS': '1500', 'ARC_MEMORY_RESET_FUNCTIONS': '40', 'ARC_MEMORY_RESET_CODE_CHARS': '8000', 'ARC_MEMORY_RESET_INPLACE': '0', 'ARC_MODEL_RESET_ON': '0', 'ARC_MODEL_RESET_REASON': '0', 'ARC_MORE_MODULES_ON': '0', 'ARC_MORE_MODULES_MODULES': 'difflib,textwrap,time,hashlib', 'ARC_NEW_SIGHT_ON': '0', 'ARC_NEW_SIGHT_PER_LEVEL': '3', 'ARC_NEW_SIGHT_FROM_LEVEL': '2', 'ARC_NO_PRESS_TURN_ON': '0', 'ARC_NOTES_SCRIBE_ON': '0', 'ARC_NOTES_SCRIBE_EVERY': '3', 'ARC_NOTES_SCRIBE_CHARS': '400', 'ARC_NOTES_SCRIBE_SECONDS': '60.0', 'ARC_NOTES_SCRIBE_BOARDS': '4', 'ARC_OPENING_PROBE_ON': '0', 'ARC_OPENING_PROBE_PRESSES': '4', 'ARC_OVERSEER_ON': '0', 'ARC_OVERSEER_MINUTES': '20.0', 'ARC_OVERSEER_NOTE': '1', 'ARC_OVERSEER_KILL_MINUTES': '0.0', 'ARC_PERCEIVER_ON': '0', 'ARC_PERCEIVER_ASSIST': '1', 'ARC_PERCEIVER_OPENING': '0', 'ARC_PERCEIVER_OPENING_CHARS': '2500', 'ARC_PERCEIVER_AGENT_BOARD': '0', 'ARC_PERCEIVER_AGENT_VIEW': '0', 'ARC_PERCEIVER_CHANGES': '1', 'ARC_PERCEIVER_GATE': '1', 'ARC_PERCEIVER_ASK': '1', 'ARC_PERCEIVER_ASK_FRAMES': '0', 'ARC_PERCEIVER_CHARS': '1200', 'ARC_PERCEIVER_SECONDS': '60.0', 'ARC_PERCEIVER_TURNS': '0', 'ARC_PERCEIVER_BOARDS': '6', 'ARC_PERCEIVER_UPSCALE': '12', 'ARC_PERCEIVER_RULERS': '5', 'ARC_PERCEIVER_PICTURES': '1', 'ARC_PERCEIVER_REPEAT_LOOK': '1', 'ARC_PERCEIVER_THOUGHTS': '6000', 'ARC_PERCEIVER_RECALL': '0', 'ARC_PERCEIVER_LETTERS': '1', 'ARC_PERCEIVER_SHAPES': '400', 'ARC_PRESS_MAP_ON': '0', 'ARC_PRESS_MAP_FIRST': '20', 'ARC_PRESS_MAP_LATER': '10', 'ARC_PRESS_MAP_DEEPER': '0', 'ARC_PRESS_MAP_PER_KIND': '3', 'ARC_PRESS_MAP_RETRY': '1', 'ARC_PRESS_MAP_WALK_FIRST': '30', 'ARC_PRESS_MAP_WALK_LATER': '0', 'ARC_PRESS_MAP_IDLE_STOP': '3', 'ARC_PRESS_MAP_OWN_TURNS': '1', 'ARC_PRESS_MAP_EXPLAIN': '1', 'ARC_PRESS_MAP_EXPLAIN_TOKENS': '140', 'ARC_PRESS_MAP_MODE': 'script', 'ARC_PRESS_NUDGE_ON': '0', 'ARC_PRESS_NUDGE_CALLS': '40', 'ARC_PRESS_NUDGE_AGAIN': '20', 'ARC_PRESS_PRESSURE_ON': '0', 'ARC_PRESS_PRESSURE_TRUTH': '1', 'ARC_PRESS_PRESSURE_IDLE': '1', 'ARC_PRESS_PRESSURE_LEVEL_PROBE': '1', 'ARC_PRESS_PRESSURE_WITHHOLD': '0', 'ARC_PRESS_PRESSURE_WITHHOLD_LEVELS': '0', 'ARC_PRESSABLE_ONLY_ON': '0', 'ARC_PROBE_BURST_ON': '0', 'ARC_PROBE_BURST_LEVELS': '2', 'ARC_PROBE_BURST_FIRST': '5', 'ARC_PROBE_BURST_COUNT': '3', 'ARC_PROBE_BURST_MOST': '20', 'ARC_PROBE_BURST_TOKENS': '900', 'ARC_PROBE_BURST_READ_TOKENS': '700', 'ARC_PROBE_BURST_MIN_LEFT': '120', 'ARC_SANDBOX_SHAPES_ON': '0', 'ARC_SEAM_CONSULT_ON': '0', 'ARC_SEAM_CONSULT_CHARS': '1000', 'ARC_SEAM_CONSULT_SECONDS': '45.0', 'ARC_SEAM_CONSULT_EFFORT': '', 'ARC_SEAM_CONSULT_LEVELS_SHOWN': '1', 'ARC_SEAM_CONSULT_UPSCALE': '4', 'ARC_SEAM_CONSULT_CLEAR': '1', 'ARC_SET_MODEL_ON': '0', 'ARC_SET_MODEL_ONCE': '1', 'ARC_SHORT_PROMPT_ON': '0', 'ARC_SHORT_PROMPT_DEAD': '1', 'ARC_SHORT_PROMPT_REPEATS': '1', 'ARC_SHORT_PROMPT_HOUSEKEEPING': '1', 'ARC_SHORT_PROMPT_METHOD': '1', 'ARC_SHORT_PROMPT_UNSHOWN': '1', 'ARC_SPOILERS_ON': '0', 'ARC_SQUEEZE_THOUGHTS_ON': '0', 'ARC_SQUEEZE_THOUGHTS_CHARS': '1000', 'ARC_SQUEEZE_THOUGHTS_SECONDS': '60.0', 'ARC_SQUEEZE_THOUGHTS_RATIO': '0.0', 'ARC_SQUEEZE_THOUGHTS_MOST': '0.0', 'ARC_SQUEEZE_THOUGHTS_MARKED': '1', 'ARC_SQUEEZE_THOUGHTS_AHEAD': '0', 'ARC_STALL_GUARD_ON': '0', 'ARC_STALL_GUARD_BUDGET_S': '60.0', 'ARC_STALL_GUARD_FRESH_BUDGET_S': '0.0', 'ARC_STALL_GUARD_BUDGET_TOKENS': '0', 'ARC_STALL_GUARD_FORCING': '0', 'ARC_STALL_GUARD_CLOSE': 'I have thought enough; I act now on my best hypothesis.', 'ARC_STALL_GUARD_DRAFT_CHARS': '0', 'ARC_STALL_GUARD_FRESH_TURNS': '2', 'ARC_START_THOUGHT_ON': '0', 'ARC_START_THOUGHT_FIRST': "It's the first level. I won't overthink it. I need to explore quickly, learn the mechanics that are still not clear, and try different approaches to win the level as soon as possible.", 'ARC_START_THOUGHT_LATER': "All right, I start a new level. Here we may have new mechanics we need to learn. Stuff may behave differently than on the levels before. Let's quickly explore everything and not overthink; we should act a lot and learn by exploration.", 'ARC_START_THOUGHT_INSIDE': '0', 'ARC_START_THOUGHT_EVERY': '0', 'ARC_START_THOUGHT_ROUND': 'All right, I start a new round. I should hurry and try to learn as much as possible if anything is not clear. I will prefer to act and learn from my actions rather than just overthink everything; that should be more time efficient.\n\nWhat the last result showed:', 'ARC_START_THOUGHT_PHASES': '0', 'ARC_START_THOUGHT_EXPLORE': 'All right, a new level. Let me explore it first: I try what I have not tried yet and see what each thing does. Hm, let me look at the board. First,', 'ARC_START_THOUGHT_EXPLORE_LATER': 'All right, a new level: the same game, possibly with something new. First I look for what is new here and try it; the way I won the last level may not be enough. Hm, what is new on this board compared to the last one? Let me look. First,', 'ARC_START_THOUGHT_OBVIOUS': 'I should check if there is anything obvious to win. I need to think about which series of presses has a good chance to win. Hm', 'ARC_START_THOUGHT_ROLES': '1', 'ARC_START_THOUGHT_DECIDE': 'Interesting. Do I know enough about this level to finish it? I have three choices. SOLVE: the level is complex and I understand every mechanic, the goal and the controls, so I write a program that searches for the winning path and plays it. WIN: I can see the winning line, so I press all of it now. LEARN: something that matters is still unclear, so I find out exactly that with the fewest presses. From these three, now, in one word, my choice is', 'ARC_START_THOUGHT_PRESS': 'I have looked enough without pressing. Now I need to think which press will tell me the most, and press it. Hm', 'ARC_START_THOUGHT_CONSULT': '0', 'ARC_START_THOUGHT_CONSULT_EVERY': '3', 'ARC_START_THOUGHT_CONSULT_FIRST': '3', 'ARC_START_THOUGHT_CONSULT_TURNS': '2', 'ARC_START_THOUGHT_CONSULT_FROM': '1', 'ARC_START_THOUGHT_CONSULT_TOKENS': '600', 'ARC_START_THOUGHT_NEWS': '0', 'ARC_STUCK_BREAKER_ON': '0', 'ARC_STUCK_BREAKER_SEARCH_SECONDS': '30.0', 'ARC_STUCK_BREAKER_SEARCH_TOKENS': '20000', 'ARC_STUCK_BREAKER_SEARCH_LEVELS': '1', 'ARC_STUCK_BREAKER_SEARCH_ACTIONS': '100000', 'ARC_STUCK_BREAKER_NUDGE_TOKENS': '0', 'ARC_STUCK_BREAKER_FRAMES': '30', 'ARC_STUCK_BREAKER_TRIALS': '0', 'ARC_STUCK_BREAKER_PARALLEL': '0', 'ARC_STUCK_BREAKER_FIRST_TOKENS': '0', 'ARC_STUCK_BREAKER_SECOND_TOKENS': '50000', 'ARC_STUCK_BREAKER_PHASES': 'base:300+cells8:300', 'ARC_STUCK_BREAKER_LATER_TOKENS': '-1', 'ARC_STUCK_BREAKER_WHILE_HELD': '0', 'ARC_STUCK_BREAKER_PHASE_PRESSES': '0', 'ARC_STUCK_BREAKER_REPORT_PHASE': '0', 'ARC_STUCK_BREAKER_STUDY_OWN': '0', 'ARC_STUCK_BREAKER_STUDY_OWN_LEVELS': '0', 'ARC_STUCK_BREAKER_STUDY_SHOW': 'answer', 'ARC_STUCK_BREAKER_CARRY': '1', 'ARC_STUCK_BREAKER_STUDY_EFFORT': 'xhigh', 'ARC_STUCK_BREAKER_STUDY_SEARCH': '1', 'ARC_STUCK_BREAKER_STUDY_BUDGET': '16000', 'ARC_STUCK_BREAKER_RETHINK': '0', 'ARC_STUCK_BREAKER_RETHINK_FROM': '2', 'ARC_STUCK_BREAKER_RETHINK_RESETS': '5', 'ARC_STUCK_BREAKER_RETHINK_TOKENS': '20000', 'ARC_STUCK_BREAKER_RETHINK_MOST': '2', 'ARC_STUCK_BREAKER_STUDY_OPEN': '0', 'ARC_STUCK_BREAKER_STUDY_SECONDS': '900.0', 'ARC_STUCK_BREAKER_STUDY_EXPLAIN': '1', 'ARC_STUCK_BREAKER_STUDY_EXPLAIN_LAST': '16', 'ARC_STUCK_BREAKER_STUDY_EXPLAIN_TOKENS': '140', 'ARC_STUCK_BREAKER_SEARCH_LINE': '1', 'ARC_STUCK_BREAKER_MISS_EFFORT': 'none', 'ARC_STUCK_BREAKER_STUDY_MIN_LEFT': '150.0', 'ARC_STUCK_BREAKER_STUDY_CALL': '1', 'ARC_SUBAGENT_ON': '0', 'ARC_SUBAGENT_BUDGET': '40', 'ARC_SUBAGENT_TURNS': '50', 'ARC_SUBAGENT_CHARS': '5000', 'ARC_SUBAGENT_BRIEF_CHARS': '0', 'ARC_SUBAGENT_LEARNS_CHARS': '0', 'ARC_SUBAGENT_EFFORT_CEILING': 'low', 'ARC_SUBAGENT_MAX_OUTPUT': '0', 'ARC_SUBAGENT_SECONDS': '120.0', 'ARC_SUBAGENT_EXPECTS': '5', 'ARC_SUBAGENT_GRADED': '0', 'ARC_SUBAGENT_SEES_PICTURE': '0', 'ARC_SUBAGENT_WALL': '0.0', 'ARC_SUBAGENT_FRAMES_SHOWN': '10', 'ARC_SUBAGENT_FRAMES_SHEET': '1', 'ARC_SUBAGENT_SHEET_UPSCALE': '4', 'ARC_SUBAGENT_OPENS': '3', 'ARC_SUBAGENT_ZOOM_UPSCALE': '8', 'ARC_SUBAGENT_HISTORY_BOARDS': '6', 'ARC_THINK_ECONOMY_ON': '0', 'ARC_TIME_BANK_ON': '0', 'ARC_TIME_BANK_BASE': '15.0', 'ARC_TIME_BANK_BONUS': '10.0', 'ARC_TIME_BANK_QUIT_AFTER': '12.0', 'ARC_TIME_BANK_IDLE': '5.0', 'ARC_TRANSITION_STORY_ON': '0', 'ARC_TURN_STATICS_ON': '0', 'ARC_TURN_STATICS_HOISTED': '1', 'ARC_TURN_STATICS_SAID_ABOVE': '1', 'ARC_UNDO_ACTION_ON': '0', 'ARC_VERBATIM_REPLIES_ON': '0', 'ARC_WIDE_CONTEXT_ON': '0', 'ARC_WIDE_CONTEXT_TOKENS': '50000', 'ARC_WON_BOARDS_ON': '0', 'ARC_WON_BOARDS_LEVELS': '1', 'ARC_WON_BOARDS_PICTURE': '0', 'ARC_WON_BOARDS_IN_TURN': '0', 'ARC_WORLD_MODEL_ON': '0', 'ARC_WORLD_MODEL_CHARS': '900', 'ARC_WORLD_MODEL_SECONDS': '120.0', 'ARC_WORLD_WRITER_ON': '0', 'ARC_WORLD_WRITER_MAX_OUTPUT': '8000', 'ARC_WORLD_WRITER_SECONDS': '180.0', 'ARC_WORLD_WRITER_SOLVE_SECONDS': '20.0', 'ARC_WORLD_WRITER_SOLVE_PRESSES': '20000', 'ARC_WORLD_WRITER_THINKING_TOKENS': '0', 'ARC_BOARD_PROPOSERS': '8', 'ARC_BOARD_PROPOSER_MODE': 'list', 'ARC_BOARD_PROPOSER_MAX_TOKENS': '600', 'ARC_BOARD_PROPOSER_TEMPERATURE': '0.8', 'ARC_BOARD_JUDGE_MAX_TOKENS': '400', 'ARC_BOARD_JUDGE_TEMPERATURE': '0.3', 'ARC_BOARD_PERCEIVER_MAX_TOKENS': '800', 'ARC_BOARD_THEORIST_MAX_TOKENS': '600', 'ARC_BOARD_GOAL_MAX_TOKENS': '600', 'ARC_BOARD_PERCEIVE_EVERY': '10', 'ARC_BOARD_PERCEIVE_NEW_SHAPES': '3', 'ARC_BOARD_THEORIST_EVERY': '10', 'ARC_BOARD_CALL_TIMEOUT_S': '60.0', 'ARC_BOARD_MODEL_LEN': '32768', 'ARC_BOARD_IMAGE_TOKENS': '120', 'ARC_BOARD_RECENT_EVENTS': '8', 'ARC_QUAD_HYPOTHESES_MAX_TOKENS': '4000', 'ARC_QUAD_JUDGE_MAX_TOKENS': '4000', 'ARC_QUAD_STRATEGY_MAX_TOKENS': '4000', 'ARC_QUAD_PLAN_MOVES': '10', 'ARC_QUAD_HISTORY': '10', 'ARC_QUAD_FILMSTRIP': '12', 'ARC_CMD_COMMANDER_EFFORT': 'xhigh', 'ARC_CMD_WORKER_EFFORT': 'medium', 'ARC_CMD_COMMANDER_MAX_TOKENS': '12000', 'ARC_CMD_WORKER_MAX_TOKENS': '25000', 'ARC_CMD_DOCUMENT_CHARS': '4000', 'ARC_CMD_BUDGET_MAX': '20', 'ARC_CMD_COMMANDER_SEES': 'pictures', 'ARC_CMD_MISSION_MINUTES': '0.0', 'ARC_CMD_MISSION_SHARE': '0.0', 'ARC_CMD_REPORT_EFFORT': 'low', 'ARC_CMD_WORKER_EFFORT_CAP': 'medium', 'ARC_CMD_REASONING_TAIL': '600', 'ARC_CMD_CODE_BUDGET_MAX': '30', 'ARC_CMD_ASCII_CELLS': '1024', 'ARC_CMD_WORKER_TURNS': '8', 'ARC_CMD_FRAMES': '12', 'ARC_CMD_RECORD_LINES': '24', 'ARC_CMD_RULER_SCALE': '8', 'ARC_CMD_MISSIONS_MAX': '0', 'ARC_CMD_COMMANDER_WINDOW': '6000', 'ARC_CMD_CALL_TIMEOUT_S': '1000.0', 'ARC_CMD_IDLE_MISSIONS': '3', 'ARC_CARRIED_MAIN': '', 'ARC_SERVE_SWEEP_MINUTES': '0.0', 'ARC_SERVE_SWEEP_STAGES': '', 'ARC_MTP_TUNE_MODE': 'eval', 'ARC_MTP_TUNE_TRAIN_MINUTES': '60.0', 'ARC_MTP_TUNE_EVAL_RECORDS': '150', 'ARC_MTP_TUNE_KV_GIB': '12.0', 'ARC_MTP_TUNE_TRAIN_BASES': '3072', 'ARC_MTP_TUNE_CONTEXT': '32768', 'ARC_MTP_TUNE_INIT': '', 'ARC_MTP_TUNE_SKIP_RECORDS': '0', 'ARC_ROLLING_LANES': '0', 'ARC_SUBMISSION_SESSION_MINUTES': '540.0', 'ARC_ROLLING_EXTEND': '0', 'ARC_CHAMPION_STINT_MINUTES': '0.0', 'ARC_CHAMPIONS': '0', 'ARC_CHAMPION_SEATS': '0', 'ARC_CHAMPION_ROUNDS': '0', 'ARC_CHAMPION_ASK': '0', 'ARC_CHAMPION_RANK': 'levels', 'ARC_CHAMPION_ROLLING': '0', 'ARC_CHAMPION_SLOWING': '2.0', 'ARC_SUB_FRAMES': '1', 'ARC_SUB_FRAMES_MAX_BYTES': '20000000'},
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
}
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
note(f'worker rows written last, so this run is the configuration this build names, '
     f'not the serving script defaults: {len(setup_env)} rows')

PYSETUP