import dataclasses
import hashlib
import importlib
import json
import os
import threading
import traceback
import time
from pathlib import Path
from typing import Any

import torch

EMBED_ENV = "TAAF_VLLM_EMBED_HOST"
EMBED_HOT_ENV = "TAAF_VLLM_EMBED_HOT"
VISION_ENV = "TAAF_VLLM_VISION_HOST"
LEDGER_ENV = "TAAF_VLLM_MEM_LEDGER"
LEDGER_EVERY_ENV = "TAAF_VLLM_MEM_LEDGER_S"
LEDGER_EVERY_DEFAULT = 30.0
LEDGER_READS = 4
KV_AUTO_ENV = "TAAF_VLLM_KV_AUTO"
WARM_STEP_ENV = "TAAF_VLLM_WARM_STEP_CHECK"
WARM_STEP_RESERVE_ENV = "TAAF_VLLM_WARM_STEP_RESERVE_MIB"
WARM_STEP_CONTEXT_ENV = "TAAF_VLLM_WARM_STEP_CONTEXT"
WARM_STEP_RESERVE_MIB = 768
WARM_STEP_CONTEXT = 118272
SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70
PLAN_NAME = "taaf_kv_auto_plan.json"
PLAN_SCHEMA = 1
REFUSED_NAME = "taaf_kv_auto_plan.refused.json"
OVERRUN_MARGIN = 256 * (1 << 20)
MEMORY_KNOBS = frozenset("TAAF_VLLM_" + name for name in (
    "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN", "MTP_TOKENS", "MTP_ATTN_ROWS", "MTP_LAST_ROWS",
    "SCHED_SPEC_RESERVE", "ENABLE_PREFIX_CACHING", "MAMBA_COUNT_FLOOR", "CACHE_KEEP", "CACHE_KEEP_END",
    "KV_OFFLOADING_GIB", "BLOCK_COPY_CHUNK_MIB",
    "MAX_CUDAGRAPH_CAPTURE_SIZE", "PIECEWISE_LARGE", "PIECEWISE_LARGE_STEP", "CAPTURE_SPARSE", "ENFORCE_EAGER",
    "MIXED_FULL", "MIXED_FULL_LARGEST", "MIXED_FULL_SEQS", "MIXED_PREP_SHARED", "MIXED_TAIL_POOL",
    "KV_CACHE_DTYPE", "INDEXER_KV_DTYPE", "MAMBA_SSM_DTYPE", "QSA_KV8", "QSA_LAYOUT",
    "HOST_KV", "HOST_KV_ROWS", "HOST_KV_GIB", "HOST_KV_HOT_ROWS", "HOST_KV_HOT_SPLIT", "HOST_KV_SLOTS",
    "HOST_KV_STAGE_MIB", "HOST_KV_PLE_GIB", "HOST_KV_GROWTH_GIB", "HOST_KV_MIRROR_ENTRIES", "HOST_KV_RECENT",
    "HOST_KV_FIT", "HOST_KV_GROUP", "HOST_KV_INDEXER_WIDE", "HOST_KV_SPARSE_STAGE", "HOST_KV_PREFETCH_PREV_ROWS",
    "HOST_KV_MIXED_CAPTURE",
    "BF16C", "BF16C_FAST2", "BF16C_GROUPS", "BF16C_MIX", "BF16C_MIX_DECODE", "BF16C_ROWS", "BF16C_PIPE",
    "EMBED_HOST", "VISION_HOST", "MTP_WEIGHTS", "DRAFT_MODEL", "DRAFT_VOCAB", "DRAFT_VOCAB_FILE", "DRAFT_VOCAB_PAD", "DRAFT_HEAD_FP8",
    "EXPERTS_KEPT", "MOE_CUT", "MOE_BACKEND", "QUANTIZATION", "FP8_DENSE", "FP8_DENSE_QUANT",
    "PLE_HOT", "PLE_HOT_GIB", "PLE_PACKED", "PLE_DF11", "PLE_GROUP_CAP", "PLE_PREFILL_GROUPS", "RUNTIME"))
ALIGN = 256
MIB = 1 << 20
HEAD_NAMES = ("lm_head", "head")

_HELD: list[Any] = []
_STATE: dict[str, Any] = {"ledger": None, "warm": False, "start": time.monotonic(),
                          "auto": {}}


def on(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def aligned(nbytes: int) -> int:
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


class Cuda:
    def __init__(self, device: torch.device):
        self.device = device

    def pinned(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    def host_view(self, host: torch.Tensor) -> torch.Tensor:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        return get_accelerator_view_from_cpu_tensor(host)

    def empty(self, nbytes: int) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def stream(self) -> Any:
        return torch.cuda.Stream(device=self.device)

    def event(self) -> Any:
        return torch.cuda.Event()

    def current(self) -> Any:
        return torch.cuda.current_stream(self.device)

    def use(self, stream: Any) -> Any:
        return torch.cuda.stream(stream)

    def release(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def sync(self) -> None:
        torch.cuda.synchronize(self.device)

    def reset_peaks(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def expandable(self) -> None:
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if setter is None:
            torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        else:
            setter("expandable_segments:True")

    def cap(self, fraction: float) -> None:
        torch.cuda.set_per_process_memory_fraction(fraction, self.device)

    def pools(self) -> dict[str, int]:
        out = {"default_reserved": 0, "default_allocated": 0, "graph_reserved": 0, "graph_allocated": 0,
               "expandable_reserved": 0}
        for segment in torch.cuda.memory_snapshot():
            pool = tuple(segment.get("segment_pool_id") or (0, 0))
            kind = "default" if pool == (0, 0) else "graph"
            out[f"{kind}_reserved"] += int(segment.get("total_size", 0))
            out[f"{kind}_allocated"] += int(segment.get("allocated_size", 0))
            if segment.get("is_expandable"):
                out["expandable_reserved"] += int(segment.get("total_size", 0))
        return out

    def identity(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(self.device)
        driver = getattr(torch._C, "_cuda_getDriverVersion", lambda: 0)()
        return {"device": props.name, "total": int(props.total_memory), "driver": int(driver),
                "torch": torch.__version__, "cuda": torch.version.cuda or ""}

    def free_bytes(self) -> int:
        return int(torch.cuda.mem_get_info(self.device)[0])

    def ledger(self) -> dict[str, int]:
        torn = 1
        for _ in range(LEDGER_READS):
            before = torch.cuda.memory_stats(self.device).get("reserved_bytes.all.current", 0)
            free, total = torch.cuda.mem_get_info(self.device)
            stats = torch.cuda.memory_stats(self.device)
            if stats.get("reserved_bytes.all.current", 0) == before:
                torn = 0
                break
        return {"free": int(free), "total": int(total), "torn": torn,
                "allocated": int(stats.get("allocated_bytes.all.current", 0)),
                "reserved": int(stats.get("reserved_bytes.all.current", 0)),
                "peak_allocated": int(stats.get("allocated_bytes.all.peak", 0)),
                "peak_reserved": int(stats.get("reserved_bytes.all.peak", 0)),
                "retries": int(stats.get("num_alloc_retries", 0)),
                "ooms": int(stats.get("num_ooms", 0))}


def nbytes_of(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def heads(models: list[Any]) -> set[int]:
    found = set()
    for model in models:
        if model is None:
            continue
        for name, module in model.named_modules(remove_duplicate=False):
            if name.rsplit(".", 1)[-1] in HEAD_NAMES:
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor):
                    found.add(weight.data_ptr())
    return found


def embeddings(models: list[Any]) -> list[tuple[str, Any]]:
    found, seen = [], set()
    for model in models:
        if model is None:
            continue
        for name, module in model.named_modules():
            if name.rsplit(".", 1)[-1] != "embed_tokens":
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter) or getattr(weight, "_vllm_is_uva_offloaded", False):
                continue
            if weight.data_ptr() in seen:
                continue
            seen.add(weight.data_ptr())
            found.append((name, module))
    return found


def offload_embeddings(models: list[Any], dev: Any) -> dict[str, Any]:
    tied = heads(models)
    moved, skipped = [], []
    total = 0
    for name, module in embeddings(models):
        weight = module.weight
        if weight.data_ptr() in tied:
            skipped.append({"name": name, "why": "the output head reads the same table every step"})
            continue
        host = dev.pinned(weight.shape, weight.dtype)
        host.copy_(weight.data)
        _HELD.append(host)
        weight.data = dev.host_view(host)
        weight._vllm_is_uva_offloaded = True
        size = nbytes_of(host)
        total += size
        row = {"name": name, "bytes": size, "shape": list(host.shape), "dtype": str(host.dtype)}
        if str(os.environ.get(EMBED_HOT_ENV) or "").strip() not in ("", "0"):
            row["hot"] = importlib.import_module("vllm.taaf_embed_hot").install(module, host, dev.device)
        moved.append(row)
    return {"moved": moved, "skipped": skipped, "bytes": total}


class Stager:
    def __init__(self, blocks: list[Any], dev: Any):
        self.dev = dev
        self.host: list[torch.Tensor] = []
        self.sizes: list[int] = []
        layouts = []
        for block in blocks:
            entries, offset = [], 0
            for _, param in block.named_parameters():
                size = nbytes_of(param)
                entries.append((param, offset, size))
                offset = aligned(offset + size)
            host = dev.pinned(torch.Size([max(offset, ALIGN)]), torch.uint8)
            for param, start, size in entries:
                host[start:start + size].view(param.dtype).view(param.shape).copy_(param.data)
            self.host.append(host)
            self.sizes.append(offset)
            layouts.append(entries)
        self.slot_bytes = max(self.sizes) if self.sizes else 0
        self.slots = [dev.empty(self.slot_bytes), dev.empty(self.slot_bytes)]
        for index, entries in enumerate(layouts):
            slot = self.slots[index % 2]
            for param, start, size in entries:
                param.data = slot[start:start + size].view(param.dtype).view(param.shape)
        self.stream = dev.stream()
        self.ready = [dev.event(), dev.event()]
        self.free = [dev.event(), dev.event()]
        self.pending: int | None = None
        self.copies = 0
        self.handles = []
        for index, block in enumerate(blocks):
            self.handles.append(block.register_forward_pre_hook(self.before(index)))
            self.handles.append(block.register_forward_hook(self.after(index)))

    @property
    def host_bytes(self) -> int:
        return sum(self.sizes)

    @property
    def staging_bytes(self) -> int:
        return 2 * self.slot_bytes

    def issue(self, index: int) -> None:
        slot = index % 2
        with self.dev.use(self.stream):
            self.stream.wait_event(self.free[slot])
            self.slots[slot][:self.sizes[index]].copy_(self.host[index][:self.sizes[index]], non_blocking=True)
            self.ready[slot].record(self.stream)
        self.pending = index
        self.copies += 1

    def before(self, index: int):
        def hook(module, args):
            if self.pending != index:
                self.issue(index)
            self.dev.current().wait_event(self.ready[index % 2])
            self.pending = None
            if index + 1 < len(self.host):
                self.issue(index + 1)
        return hook

    def after(self, index: int):
        def hook(module, args, output):
            self.free[index % 2].record(self.dev.current())
        return hook


def tower(model: Any) -> tuple[Any, str]:
    if model is None:
        return None, "no model"
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "visual":
            continue
        blocks = getattr(module, "blocks", None)
        if isinstance(blocks, torch.nn.ModuleList) and len(blocks):
            return module, name
        return None, f"{name} is {type(module).__name__} with no block list"
    return None, "the model has no vision tower"


def stage_vision(model: Any, dev: Any, config: Any = None) -> dict[str, Any]:
    compilation = getattr(config, "compilation_config", None)
    if compilation is not None and (getattr(compilation, "cudagraph_mm_encoder", False)
                                    or getattr(compilation, "compile_mm_encoder", False)):
        raise RuntimeError(f"{VISION_ENV} swaps the vision blocks' weights in through forward hooks, which a "
                           f"captured or compiled encoder never calls; turn the encoder graphs and compile off.")
    visual, said = tower(model)
    if visual is None:
        return {"staged": False, "why": said}
    blocks = list(visual.blocks)
    stager = Stager(blocks, dev)
    _HELD.append(stager)
    return {"staged": True, "tower": said, "blocks": len(blocks), "host_bytes": stager.host_bytes,
            "staging_bytes": stager.staging_bytes, "net_bytes": stager.host_bytes - stager.staging_bytes}


def uptime() -> float:
    return time.monotonic() - _STATE["start"]


def driver_of(held: dict[str, int]) -> int:
    return held["total"] - held["free"] - held["reserved"]


def ledger_line(dev: Any, tag: str) -> str:
    held = dev.ledger()
    return (f"MEM_LEDGER tag={tag} t_s={uptime():.0f} free_mib={held['free'] / MIB:.1f} "
            f"total_mib={held['total'] / MIB:.1f} "
            f"allocated_mib={held['allocated'] / MIB:.1f} reserved_mib={held['reserved'] / MIB:.1f} "
            f"driver_mib={driver_of(held) / MIB:.1f} "
            f"peak_allocated_mib={held['peak_allocated'] / MIB:.1f} peak_reserved_mib={held['peak_reserved'] / MIB:.1f} "
            f"warm={int(_STATE['warm'])} torn={held.get('torn', 0)} retries={held['retries']} ooms={held['ooms']}")


def ledger_every() -> float:
    said = str(os.environ.get(LEDGER_EVERY_ENV) or "").strip()
    return float(said) if said else LEDGER_EVERY_DEFAULT


def ledger_tick(dev: Any) -> str:
    line = ledger_line(dev, "live")
    if _STATE["warm"]:
        dev.reset_peaks()
    return line


def start_ledger(dev: Any) -> None:
    if _STATE["ledger"] is not None:
        return
    every = ledger_every()

    def loop() -> None:
        while True:
            time.sleep(every)
            try:
                print(ledger_tick(dev), flush=True)
            except Exception as failure:
                print(f"MEM_LEDGER tag=error {type(failure).__name__}: {failure}", flush=True)
                return

    thread = threading.Thread(target=loop, name="taaf-mem-ledger", daemon=True)
    thread.start()
    _STATE["ledger"] = thread


def capture_line(graphs: str, desc: Any, before: dict[str, int], after: dict[str, int], seconds: float) -> str:
    mode = getattr(getattr(desc, "cg_mode", None), "name", str(getattr(desc, "cg_mode", "-")))
    return (f"MEM_CAPTURE graphs={graphs} mode={mode} tokens={getattr(desc, 'num_tokens', -1)} "
            f"reqs={getattr(desc, 'num_reqs', -1)} "
            f"d_reserved_mib={(after['reserved'] - before['reserved']) / MIB:.1f} "
            f"d_allocated_mib={(after['allocated'] - before['allocated']) / MIB:.1f} "
            f"d_driver_mib={(driver_of(after) - driver_of(before)) / MIB:.1f} "
            f"d_used_mib={(before['free'] - after['free']) / MIB:.1f} "
            f"free_mib={after['free'] / MIB:.1f} ms={seconds * 1000:.0f}")


class CaptureMarks:
    def __init__(self, dev: Any, graphs: str):
        self.dev, self.graphs = dev, graphs
        self.desc: Any = None
        self.before: dict[str, int] | None = None
        self.at = 0.0
        self.lines: list[str] = []

    def mark(self, desc: Any) -> None:
        self.dev.sync()
        now, at = self.dev.ledger(), time.perf_counter()
        if self.desc is not None and self.before is not None:
            line = capture_line(self.graphs, self.desc, self.before, now, at - self.at)
            self.lines.append(line)
            print(line, flush=True)
        self.desc, self.before, self.at = desc, now, at


def wrap_capture(base: Any, dev: Any) -> Any:
    def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        marks = CaptureMarks(dev, type(self).__name__)

        def forward(desc, warmup):
            if warmup:
                marks.mark(desc)
            return create_forward_fn(desc, warmup=warmup)

        result = base(self, forward, progress_bar_desc)
        marks.mark(None)
        return result

    capture.taaf_mem = True
    return capture


def install_capture_lines(dev: Any, manager: Any = None) -> bool:
    if manager is None:
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager as manager
    if getattr(manager.capture, "taaf_mem", False):
        return False
    manager.capture = wrap_capture(manager.capture, dev)
    return True


def plan_path() -> Path:
    return Path(__file__).with_name(PLAN_NAME)


def refused_path() -> Path:
    return plan_path().with_name(REFUSED_NAME)


def config_facts(config: Any) -> dict[str, Any]:
    compilation = getattr(config, "compilation_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    model = getattr(config, "model_config", None)
    cache = getattr(config, "cache_config", None)
    speculative = getattr(config, "speculative_config", None)
    return {"cudagraph_mode": str(getattr(compilation, "cudagraph_mode", "")),
            "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
            "model": str(getattr(model, "model", "")),
            "max_model_len": getattr(model, "max_model_len", None),
            "cache_dtype": str(getattr(cache, "cache_dtype", "")),
            "block_size": getattr(cache, "block_size", None),
            "spec_tokens": getattr(speculative, "num_speculative_tokens", None)}


def knob_facts(environ: Any) -> dict[str, str]:
    return {name: str(value) for name, value in sorted(environ.items()) if name in MEMORY_KNOBS}


def fingerprint(facts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def boot_facts(config: Any, dev: Any, environ: Any = None) -> dict[str, Any]:
    return {"schema": PLAN_SCHEMA, "config": config_facts(config), "knobs": knob_facts(environ or os.environ),
            "device": dev.identity()}


def read_refusal(path: Path, print_: str, plan: dict[str, Any]) -> str | None:
    if not path.is_file():
        return None
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if held.get("fingerprint") != print_:
        return None
    growth = int(held.get("growth", 0))
    over, margin = overrun(growth, plan_terms(plan))
    if over <= margin:
        return None
    return (f"plan refused: a boot of fingerprint {print_} grew {mib(growth)} MiB, {mib(over)} MiB past this "
            f"plan's {mib(growth - over)} MiB")


def read_plan(path: Path, print_: str, refused: Path | None = None) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, f"no plan at {path.name}"
    held = json.loads(path.read_text(encoding="utf-8"))
    if held.get("schema") != PLAN_SCHEMA:
        return None, f"plan schema {held.get('schema')} is not {PLAN_SCHEMA}"
    if held.get("fingerprint") != print_:
        return None, f"plan fingerprint {held.get('fingerprint')} is not this boot's {print_}"
    if not held.get("boots"):
        return None, "plan holds no measured boot"
    why = read_refusal(refused, print_, held) if refused is not None else None
    if why is not None:
        return None, why
    return held, "plan"


def overrun(measured_growth: int, sized: dict[str, int]) -> tuple[int, int]:
    return measured_growth - int(sized["growth"]), max(int(sized.get("spread", 0)), OVERRUN_MARGIN)


def plan_terms(plan: dict[str, Any]) -> dict[str, int]:
    boots = plan["boots"]
    growth = [int(boot["growth"]) for boot in boots]
    torch_play = [int(boot["play_torch"]) for boot in boots]
    driver_play = [int(boot["play_driver"]) for boot in boots]
    play = [one + two for one, two in zip(torch_play, driver_play)]
    return {"boots": len(boots), "growth": max(growth), "play_torch": max(torch_play),
            "play_driver": max(driver_play), "spread": max(growth) - min(growth) + max(play) - min(play)}


def auto_kv(free_sizing: int, transient: int, terms: dict[str, int]) -> dict[str, int]:
    torch_play = max(terms["play_torch"], transient)
    kv = free_sizing - terms["growth"] - torch_play - terms["play_driver"] - terms["spread"]
    return {**terms, "free_sizing": free_sizing, "transient": transient, "torch_play_used": torch_play, "kv": kv}


def cap_bytes(total: int, driver_warm: int, sized: dict[str, int]) -> int:
    return total - driver_warm - sized["play_driver"] - sized["spread"]


def mib(nbytes: int) -> str:
    return f"{nbytes / MIB:.1f}"


def wrap_determine(base: Any, dev: Any) -> Any:
    def determine_available_memory(self):
        explicit = int(self.cache_config.kv_cache_memory_bytes or 0)
        if explicit <= 0:
            raise RuntimeError(f"{KV_AUTO_ENV} sizes the pool after vLLM's explicit-size profile run, so it needs "
                               f"kv_cache_memory_bytes as the size of a boot without a plan.")
        dev.release()
        dev.reset_peaks()
        result = base(self)
        dev.sync()
        held = dev.ledger()
        transient = max(held["peak_allocated"] - held["allocated"], 0)
        dev.release()
        held = dev.ledger()
        facts = boot_facts(self.vllm_config, dev)
        print_ = fingerprint(facts)
        print(f"MEM_KVAUTO facts fingerprint={print_} knobs="
              f"{','.join(f'{name[10:]}={value}' for name, value in facts['knobs'].items()) or 'none'}", flush=True)
        state = _STATE["auto"]
        state.update(fingerprint=print_, total=held["total"], free_sizing=held["free"],
                     used_sizing=held["total"] - held["free"], reserved_sizing=held["reserved"],
                     driver_sizing=driver_of(held), transient=transient, explicit=explicit, sized=None)
        plan, why = read_plan(plan_path(), print_, refused_path())
        if plan is None:
            print(f"MEM_KVAUTO sizing=explicit fingerprint={print_} kv_mib={mib(explicit)} "
                  f"free_sizing_mib={mib(held['free'])} transient_mib={mib(transient)} why={why!r}; this boot "
                  f"measures what a plan needs", flush=True)
            return result
        sized = auto_kv(held["free"], transient, plan_terms(plan))
        if sized["kv"] < 64 * MIB:
            raise RuntimeError(f"{KV_AUTO_ENV}: the plan leaves {mib(sized['kv'])} MiB for the KV pool "
                               f"({sized}); the measured footprint does not fit this device.")
        state["sized"] = sized
        print(f"MEM_KVAUTO sizing=plan fingerprint={print_} boots={sized['boots']} kv_mib={mib(sized['kv'])} "
              f"= free_sizing_mib={mib(sized['free_sizing'])} - growth_mib={mib(sized['growth'])} "
              f"- play_torch_mib={mib(sized['torch_play_used'])} (plan {mib(sized['play_torch'])}, "
              f"profile transient {mib(transient)}) - play_driver_mib={mib(sized['play_driver'])} "
              f"- spread_mib={mib(sized['spread'])}; explicit_mib={mib(explicit)}", flush=True)
        self.cache_config.kv_cache_memory_bytes = sized["kv"]
        return result - explicit + sized["kv"]

    determine_available_memory.taaf_mem = True
    return determine_available_memory


def kv_bytes(kv_cache_config: Any) -> int:
    sizes = {int(getattr(tensor, "size", 0)) for tensor in (getattr(kv_cache_config, "kv_cache_tensors", None) or [])
             if not getattr(tensor, "host_resident", False)}
    return max(sizes, default=0)


def wrap_initialize(base: Any, dev: Any, auto: bool) -> Any:
    def initialize_from_config(self, kv_cache_config):
        result = base(self, kv_cache_config)
        _STATE["auto"]["kv"] = kv_bytes(kv_cache_config)
        if auto:
            dev.expandable()
        print(f"MEM_KVINIT kv_mib={mib(_STATE['auto']['kv'])} blocks={getattr(kv_cache_config, 'num_blocks', -1)} "
              f"expandable={int(auto)}" + ("; default-pool segments made after the KV pool map pages on demand"
                                           if auto else ""), flush=True)
        print(ledger_line(dev, "kv"), flush=True)
        return result

    initialize_from_config.taaf_mem = True
    return initialize_from_config


def warm_line(held: dict[str, int], pools: dict[str, int]) -> str:
    return (f"MEM_WARM t_s={uptime():.0f} free_mib={mib(held['free'])} reserved_mib={mib(held['reserved'])} "
            f"allocated_mib={mib(held['allocated'])} driver_mib={mib(driver_of(held))} "
            + " ".join(f"{name}_mib={mib(value)}" for name, value in pools.items()))


def budget_line(held: dict[str, int]) -> str:
    budget = os.environ.get("TAAF_VLLM_MAX_NUM_BATCHED_TOKENS", "").strip() or "-"
    return (f"MEM_BUDGET step_budget={budget} "
            f"transient_warm_mib={mib(max(held['peak_allocated'] - held['allocated'], 0))} "
            f"allocated_warm_mib={mib(held['allocated'])} free_warm_mib={mib(held['free'])}: the largest step of the "
            f"warm-up and graph captures took the transient above what stays allocated; a larger step budget "
            f"raises it and lowers what is free for play")


def finish_auto(dev: Any, held: dict[str, int]) -> dict[str, Any]:
    state = _STATE["auto"]
    kv = int(state.get("kv", 0))
    used_warm = held["total"] - held["free"]
    measured = {"growth": used_warm - state["used_sizing"] - kv,
                "torch_growth": held["reserved"] - state["reserved_sizing"] - kv,
                "driver_growth": driver_of(held) - state["driver_sizing"],
                "reserved_warm": held["reserved"], "driver_warm": driver_of(held), "free_warm": held["free"]}
    state["measured"] = measured
    print(f"MEM_KVAUTO measured fingerprint={state['fingerprint']} kv_mib={mib(kv)} "
          f"used_sizing_mib={mib(state['used_sizing'])} used_warm_mib={mib(used_warm)} "
          f"growth_mib={mib(measured['growth'])} torch_growth_mib={mib(measured['torch_growth'])} "
          f"driver_growth_mib={mib(measured['driver_growth'])} transient_mib={mib(state['transient'])} "
          f"reserved_warm_mib={mib(measured['reserved_warm'])} driver_warm_mib={mib(measured['driver_warm'])} "
          f"free_warm_mib={mib(held['free'])}", flush=True)
    sized = state.get("sized")
    if not sized:
        print("MEM_CAP off: no plan sized this pool, so no overrun is defined", flush=True)
        return measured
    need = sized["torch_play_used"] + sized["play_driver"]
    cap = cap_bytes(held["total"], measured["driver_warm"], sized)
    print(f"MEM_KVAUTO check growth_mib={mib(measured['growth'])} plan_growth_mib={mib(sized['growth'])} "
          f"free_warm_mib={mib(held['free'])} play_need_mib={mib(need)} spread_mib={mib(sized['spread'])} "
          f"short_mib={mib(max(need - held['free'], 0))}", flush=True)
    over, margin = overrun(measured["growth"], sized)
    if over > margin:
        refusal = {"fingerprint": state["fingerprint"], "growth": measured["growth"], "plan_growth": sized["growth"],
                   "over": over, "margin": margin}
        try:
            refused_path().write_text(json.dumps(refusal) + "\n", encoding="utf-8")
            kept = f"the next boot of this fingerprint refuses the plan ({refused_path().name})"
        except OSError as failure:
            kept = f"the refusal could not be written ({type(failure).__name__}: {failure})"
        print(f"MEM_KVAUTO OVERRUN growth_mib={mib(measured['growth'])} plan_growth_mib={mib(sized['growth'])} "
              f"over_mib={mib(over)} margin_mib={mib(margin)}: this boot grew past the plan that sized its KV pool, "
              f"so play has that much less headroom than planned; {kept}; rebuild the plan from this run's log",
              flush=True)
    dev.cap(cap / held["total"])
    print(f"MEM_CAP armed cap_mib={mib(cap)} total_mib={mib(held['total'])} "
          f"driver_warm_mib={mib(measured['driver_warm'])} "
          f"driver_reserve_mib={mib(sized['play_driver'] + sized['spread'])} "
          f"torch_headroom_mib={mib(cap - held['reserved'])}; a torch allocation past the cap raises "
          f"OutOfMemoryError at its call site", flush=True)
    return measured


def wrap_warm(base: Any, dev: Any, auto: bool) -> Any:
    def compile_or_warm_up_model(self):
        result = base(self)
        if auto:
            dev.release()
        else:
            dev.sync()
        held = dev.ledger()
        _STATE["warm"] = True
        print(warm_line(held, dev.pools()), flush=True)
        print(budget_line(held), flush=True)
        if auto:
            finish_auto(dev, held)
        if on(WARM_STEP_ENV):
            warm_step(getattr(self, "model_runner", None), dev)
        dev.reset_peaks()
        return result

    compile_or_warm_up_model.taaf_mem = True
    return compile_or_warm_up_model


def install_worker_hooks(dev: Any, auto: bool, worker: Any = None) -> bool:
    if worker is None:
        from vllm.v1.worker.gpu_worker import Worker as worker
    if getattr(worker.compile_or_warm_up_model, "taaf_mem", False):
        return False
    if auto:
        worker.determine_available_memory = wrap_determine(worker.determine_available_memory, dev)
    worker.initialize_from_config = wrap_initialize(worker.initialize_from_config, dev, auto)
    worker.compile_or_warm_up_model = wrap_warm(worker.compile_or_warm_up_model, dev, auto)
    return True


def device_of(model: Any) -> torch.device:
    for param in model.parameters():
        if param.device.type == "cuda":
            return param.device
    return torch.device("cuda", torch.cuda.current_device())


def apply(model: Any, speculator: Any = None, config: Any = None, dev: Any = None) -> dict[str, Any]:
    auto = on(KV_AUTO_ENV)
    embed, vision, ledger = on(EMBED_ENV), on(VISION_ENV), on(LEDGER_ENV) or auto
    if not (embed or vision or ledger):
        return {}
    hooks = dev is None
    dev = dev or Cuda(device_of(model))
    models = [model, getattr(speculator, "model", None)]
    report: dict[str, Any] = {}
    if ledger:
        print(ledger_line(dev, "loaded"), flush=True)
    before = dev.free_bytes()
    if embed:
        report["embed"] = offload_embeddings(models, dev)
    if vision:
        report["vision"] = stage_vision(model, dev, config)
    if embed or vision:
        dev.release()
        report["freed_measured"] = dev.free_bytes() - before
        moved = report.get("embed", {}).get("bytes", 0)
        staged = report.get("vision", {})
        print(f"MEM_OFFLOAD embed_host={int(embed)} vision_host={int(vision)} "
              f"embed_bytes={moved} embed_tables={len(report.get('embed', {}).get('moved', []))} "
              f"embed_skipped={len(report.get('embed', {}).get('skipped', []))} "
              f"vision_host_bytes={staged.get('host_bytes', 0)} vision_staging_bytes={staged.get('staging_bytes', 0)} "
              f"vision_blocks={staged.get('blocks', 0)} vision_why={staged.get('why', '-')!r} "
              f"freed_measured_bytes={report['freed_measured']}", flush=True)
    if ledger:
        print(ledger_line(dev, "offloaded"), flush=True)
        start_ledger(dev)
    if ledger and hooks:
        report["hooks"] = {"capture": install_capture_lines(dev), "worker": install_worker_hooks(dev, auto)}
        print(f"MEM_LEDGER hooks capture={int(report['hooks']['capture'])} worker={int(report['hooks']['worker'])} "
              f"kv_auto={int(auto)}: one MEM_CAPTURE line per captured graph size, MEM_KVINIT after the KV pool, "
              f"MEM_WARM after warmup, live lines report peaks since the previous line once warm", flush=True)
    return report


def selfcheck_fail(knob: str, detail: str) -> None:
    print(f"SELFCHECK FAIL {knob}: {detail}", flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def warm_step_limits(runner: Any) -> tuple[int, int, int]:
    budget = int(runner.max_num_tokens)
    context = int(os.environ.get(WARM_STEP_CONTEXT_ENV) or WARM_STEP_CONTEXT)
    context = max(min(context, int(runner.max_model_len) - budget), 0)
    reserve = int(float(os.environ.get(WARM_STEP_RESERVE_ENV) or WARM_STEP_RESERVE_MIB) * 2 ** 20)
    return budget, context, reserve


def warm_step_dispatch(runner: Any) -> Any:
    manager = getattr(runner, "cudagraph_manager", None)
    if manager is None or not callable(getattr(manager, "dispatch", None)):
        return None
    base = manager.dispatch
    used = []

    def piecewise(num_reqs, num_tokens, *args, **kwargs):
        desc = base(num_reqs, num_tokens, *args, **kwargs)
        mode = getattr(desc.cg_mode, "name", str(desc.cg_mode))
        if mode != "FULL":
            used.append(mode)
            return desc
        for one in (getattr(manager, "_candidates", None) or {}).get((num_tokens, desc.num_active_loras), ()):
            if getattr(one.cg_mode, "name", "") == "PIECEWISE":
                used.append("PIECEWISE")
                return one
        used.append("NONE")
        return dataclasses.replace(desc, cg_mode=type(desc.cg_mode)["NONE"], num_reqs=num_reqs)

    manager.dispatch = piecewise
    return manager, base, used


def warm_step_run(runner: Any, budget: int, context: int) -> list[str]:
    held = warm_step_dispatch(runner)
    try:
        with torch.inference_mode():
            runner._dummy_run(budget, context_len=context)
    finally:
        if held is not None:
            held[0].dispatch = held[1]
    return held[2] if held is not None else []


def warm_step(runner: Any, dev: Any) -> dict[str, Any]:
    if runner is None or not hasattr(runner, "_dummy_run"):
        print(f"MEM_WARM_STEP not run: the worker has no model runner with a dummy run; {WARM_STEP_ENV} checks "
              f"nothing", flush=True)
        return {"ran": False}
    budget, wanted, reserve = warm_step_limits(runner)
    tried = []
    for context in ([wanted, 0] if wanted else [0]):
        dev.release()
        dev.reset_peaks()
        before = dev.ledger()
        try:
            modes = warm_step_run(runner, budget, context)
            dev.sync()
        except torch.OutOfMemoryError as failure:
            dev.release()
            detail = (f"a {budget}-token step of the target and the drafter at {context} tokens of context does not "
                      f"fit beside the KV pool: {str(failure).splitlines()[0][:300]}")
            print(f"MEM_WARM_STEP out of memory: {detail}", flush=True)
            selfcheck_fail(WARM_STEP_ENV, detail)
            return {"ran": True, "fits": False, "why": "oom", "context": context}
        except Exception as failure:
            dev.release()
            frames = traceback.extract_tb(failure.__traceback__)
            where = " <- ".join(f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
                                for frame in frames[::-1][:4])
            tried.append(f"context {context}: {type(failure).__name__}: {str(failure)[:200]} at {where or 'unknown'}")
            print(f"MEM_WARM_STEP could not run at context {context} ({type(failure).__name__}: "
                  f"{str(failure)[:300]}) at {where or 'unknown'}"
                  + ("; trying again with no context" if context else f"; {WARM_STEP_ENV} measured nothing this boot"),
                  flush=True)
            continue
        after = dev.ledger()
        grown = max(after["peak_reserved"] - before["reserved"], 0)
        left = before["free"] - grown
        row = {"ran": True, "budget": budget, "context": context, "modes": modes,
               "transient": after["peak_allocated"] - before["allocated"], "grown": grown, "left": left,
               "reserve": reserve, "fits": left >= reserve, "tried": tried}
        print(f"MEM_WARM_STEP budget={budget} context={context} graph={'/'.join(modes) or 'eager'} "
              f"transient_mib={mib(row['transient'])} reserved_growth_mib={mib(grown)} "
              f"free_before_mib={mib(before['free'])} left_mib={mib(left)} reserve_mib={mib(reserve)}: a full-budget "
              f"step of the target and the drafter "
              + ("fits with the reserve to spare" if row["fits"] else "leaves less than the reserve free"), flush=True)
        dev.release()
        if not row["fits"]:
            selfcheck_fail(WARM_STEP_ENV, f"a {budget}-token step at {context} tokens of context leaves "
                                          f"{mib(left)} MiB free, under the {mib(reserve)} MiB reserve")
        return row
    return {"ran": False, "why": "could not run", "tried": tried}
