import ctypes
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

SOURCE_NAMES = ("host_kv_resolve.cu", "taaf_host_kv_resolve.cu")


def source_path():
    here = Path(__file__).resolve()
    for name in SOURCE_NAMES:
        if here.with_name(name).is_file():
            return here.with_name(name)
    return here.with_name(SOURCE_NAMES[0])


SOURCE = source_path()
THREADS_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_THREADS"
DEFINES_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_DEFINES"
NVCC_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_NVCC"
NVCC_FLAGS_ENV = "TAAF_VLLM_HOST_KV_ONE_CUDA_NVCC_FLAGS"
THREADS = 1024
MARK = -(1 << 31)

FIELDS = (
    ("void*", "union_p"), ("long long", "union_stride"), ("void*", "layout"), ("int", "parts"), ("int", "lanes"),
    ("void*", "request_ids"), ("void*", "states"), ("int", "state_count"), ("void*", "src_table"),
    ("long long", "src_stride"), ("int", "src_reqs"), ("int", "src_blocks"), ("int", "src_bs"),
    ("void*", "res_table"), ("long long", "res_stride"), ("int", "res_reqs"), ("int", "res_blocks"),
    ("int", "res_bs"), ("long long", "res_null"), ("void*", "hot_table"), ("long long", "hot_table_stride"),
    ("int", "hot_bs"), ("long long", "attn_block_stride"), ("void*", "dgi"), ("void*", "lru"), ("int", "region"),
    ("long long", "host_total"), ("long long", "hot_total"), ("void*", "attn"), ("long long", "attn_stride"),
    ("void*", "hot_out"), ("void*", "host_out"), ("void*", "dev_out"), ("void*", "counts_out"), ("void*", "idle"),
    ("void*", "stats"), ("void*", "tally"), ("void*", "guard"), ("void*", "pcie"), ("void*", "marks"),
    ("void*", "devs"), ("void*", "plan_host"), ("void*", "plan_dev"), ("void*", "plan_count"),
    ("long long", "plan_stride"), ("int", "piece"), ("void*", "fresh"), ("long long", "fresh_stride"),
    ("int", "pcie_calls"), ("int", "pcie_rows"), ("void*", "prof"), ("long long", "ext_union"),
    ("long long", "ext_src"), ("long long", "ext_res"), ("long long", "ext_hot"), ("long long", "ext_attn"),
    ("long long", "ext_fresh"), ("long long", "ext_lists"), ("long long", "ext_counts"), ("long long", "ext_plan"),
    ("long long", "ext_marks"), ("long long", "ext_dgi"), ("long long", "ext_lru"), ("long long", "state_rows"),
    ("void*", "viol"), ("long long", "ext_req"), ("long long", "ext_idle"), ("long long", "ext_pcount"))
CTYPES = {"void*": ctypes.c_void_p, "long long": ctypes.c_longlong, "int": ctypes.c_int}
C_TYPES = {torch.int32: "int", torch.int64: "long long", torch.int16: "short", torch.uint8: "unsigned char"}


class Args(ctypes.Structure):
    _fields_ = [(name, CTYPES[kind]) for kind, name in FIELDS]


def struct_text():
    return "struct Args {\n" + "".join(f"    {kind} {name};\n" for kind, name in FIELDS) + "};\n"


PROF_BUFFER = {}
_STATE = {"modules": {}, "libs": None, "build_path": None, "viol": {}}


def pow2(count):
    return 1 << max(int(count) - 1, 0).bit_length()


def driver():
    if _STATE["libs"] is None:
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        _STATE["libs"] = ctypes.CDLL(name)
    return _STATE["libs"]


def check(code, what):
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA driver error {code}")


def nvcc_path():
    said = os.environ.get(NVCC_ENV)
    if said and Path(said).exists():
        return said
    found = shutil.which("nvcc")
    if found:
        return found
    for pattern in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-*/bin/nvcc"):
        hits = sorted(glob.glob(pattern), reverse=True)
        if hits:
            return hits[0]
    return None


def nvrtc_library():
    names = ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so") if sys.platform != "win32" else ()
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    roots = [Path(torch.__file__).parent / "lib"]
    for entry in sys.path:
        roots.append(Path(entry) / "nvidia")
        roots.append(Path(entry) / "torch" / "lib")
    patterns = ("libnvrtc.so*", "nvrtc64_*.dll")
    for root in roots:
        for pattern in patterns:
            for hit in sorted(glob.glob(str(root / "**" / pattern), recursive=True)):
                if "builtins" in hit or "alt" in hit:
                    continue
                try:
                    return ctypes.CDLL(hit)
                except OSError:
                    pass
    return None


def defines(config):
    return [f"-D{key}={value}" for key, value in config.items()]


def nvcc_flags():
    return tuple(str(os.environ.get(NVCC_FLAGS_ENV) or "").split())


def compile_command(nvcc, arch, config, out, src):
    extra = nvcc_flags()
    own = [flag for before, flag in zip(("",) + extra, extra) if flag.startswith("-O") and not before.startswith("-X")]
    optimize = [] if own else ["-O3"]
    return [nvcc, "-cubin", f"-arch=sm_{arch}", "--std=c++17", *optimize, "-lineinfo", *extra, *defines(config),
            "-o", str(out), str(src)]


def compile_nvcc(nvcc, source, arch, config):
    out = Path(tempfile.mkdtemp(prefix="resolve_cuda_")) / "resolve.cubin"
    src = out.with_suffix(".cu")
    src.write_text(source)
    command = compile_command(nvcc, arch, config, out, src)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("nvcc failed: " + (done.stderr or done.stdout)[-2000:])
    return out.read_bytes(), "nvcc-cubin"


def compile_nvrtc(source, arch, config):
    lib = nvrtc_library()
    if lib is None:
        raise RuntimeError("no nvcc and no libnvrtc found")
    program = ctypes.c_void_p()
    check(lib.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"resolve.cu", 0, None, None),
          "nvrtcCreateProgram")
    last = None
    for target in (f"--gpu-architecture=sm_{arch}", f"--gpu-architecture=compute_{arch}"):
        options = [target, "--std=c++17", *defines(config)]
        array = (ctypes.c_char_p * len(options))(*[option.encode() for option in options])
        code = lib.nvrtcCompileProgram(program, len(options), array)
        size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(program, log)
        if code == 0:
            cubin = target.endswith(f"sm_{arch}")
            getsize, get = (lib.nvrtcGetCUBINSize, lib.nvrtcGetCUBIN) if cubin else (lib.nvrtcGetPTXSize,
                                                                                    lib.nvrtcGetPTX)
            getsize(program, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            check(get(program, buffer), "nvrtcGet")
            return buffer.raw, "nvrtc-cubin" if cubin else "nvrtc-ptx"
        last = log.value.decode(errors="replace")
        program = ctypes.c_void_p()
        check(lib.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"resolve.cu", 0, None, None),
              "nvrtcCreateProgram")
    raise RuntimeError("nvrtc failed: " + str(last)[-2000:])


def build_key(config, device):
    index = torch.cuda.current_device() if device.index is None else device.index
    return index, tuple(sorted(config.items())), nvcc_flags()


def is_built(config, device):
    return build_key(config, device) in _STATE["modules"]


def build(config, device):
    index = torch.cuda.current_device() if device.index is None else device.index
    arch = "%d%d" % torch.cuda.get_device_capability(index)
    key = build_key(config, device)
    held = _STATE["modules"].get(key)
    if held is not None:
        return held
    source = source_path().read_text().replace("//ARGS//", struct_text())
    nvcc = nvcc_path()
    try:
        if nvcc is None:
            raise RuntimeError("no nvcc")
        image, how = compile_nvcc(nvcc, source, arch, config)
    except Exception as failure:
        image, how = compile_nvrtc(source, arch, config)
        print(f"HOST_KV resolve cuda: nvcc unavailable ({str(failure)[:120]}), built with {how}", flush=True)
    _STATE["build_path"] = how
    lib = driver()
    torch.zeros(1, device=device)
    module = ctypes.c_void_p()
    check(lib.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
    function = ctypes.c_void_p()
    check(lib.cuModuleGetFunction(ctypes.byref(function), module, b"one_resolve"), "cuModuleGetFunction")
    smem = 12 * config["KB"]
    check(lib.cuFuncSetAttribute(function, 8, smem), "cuFuncSetAttribute")
    held = (module, function, smem)
    _STATE["modules"][key] = held
    counter(device)
    return held


def threads():
    said = str(os.environ.get(THREADS_ENV) or "").strip()
    if not said:
        return THREADS
    if not said.isdigit() or int(said) not in (128, 256, 512, 1024):
        raise ValueError(f"{THREADS_ENV} must be one of 128, 256, 512, 1024, found {said!r}")
    return int(said)


def counter(device):
    index = torch.cuda.current_device() if device.index is None else device.index
    held = _STATE["viol"].get(index)
    if held is None:
        held = _STATE["viol"][index] = torch.zeros(1, dtype=torch.int64, device=torch.device("cuda", index))
    return held


def pointer(tensor):
    return tensor.data_ptr() if tensor is not None else 0


def extent(tensor):
    if tensor is None or tensor.numel() == 0:
        return 0
    if tensor.dim() == 2:
        return (tensor.shape[0] - 1) * tensor.stride(0) + tensor.shape[1]
    return int(tensor.numel())


def floor_extent(*tensors):
    return min(extent(one) for one in tensors)


TABLE_TYPES = (torch.int32, torch.int64)
FLAT_TYPES = (torch.int32,)
CHECKED = (
    ("union", TABLE_TYPES, 2, False), ("request_ids", TABLE_TYPES, 1, False), ("states", tuple(C_TYPES), 1, False),
    ("src_table", TABLE_TYPES, 2, False), ("res_table", TABLE_TYPES, 2, True), ("hot_table", TABLE_TYPES, 2, False),
    ("dgi", FLAT_TYPES, 0, False), ("lru", (torch.int16,), 0, False), ("attn", FLAT_TYPES, 2, False),
    ("hot_out", FLAT_TYPES, 0, False), ("host_out", FLAT_TYPES, 0, False), ("dev_out", FLAT_TYPES, 0, False),
    ("counts_out", FLAT_TYPES, 0, False), ("idle", (torch.bool, torch.uint8), 0, True),
    ("stats", (torch.int64,), 0, False), ("tally", (torch.int64,), 0, True), ("guard", (torch.int64,), 0, True),
    ("pcie", (torch.int64,), 0, True), ("fresh", (torch.int8,), 2, True), ("layout", (torch.int64,), 0, False))


def unsupported(scratch, lanes, **tensors):
    device = tensors["union"].device
    if device.type != "cuda":
        return "the union is not on a CUDA device"
    for name, kinds, rank, optional in CHECKED:
        held = tensors.get(name)
        if held is None:
            if optional:
                continue
            return f"{name} is absent"
        if held.device != device:
            return f"{name} is on {held.device}, the union on {device}"
        if held.dtype not in kinds:
            return f"{name} is {held.dtype}, not one of {[str(one) for one in kinds]}"
        if rank == 2 and (held.dim() != 2 or held.stride(1) != 1):
            return f"{name} rows are not unit-stride"
        if rank != 2 and not held.is_contiguous():
            return f"{name} is not contiguous"
    if int(scratch["lanes"]) < int(lanes):
        return f"the scratch holds {scratch['lanes']} lanes, the launch {lanes}"
    for name in ("marks", "devs"):
        if scratch[name].numel() < int(scratch["lanes"]) * int(scratch["kb"]):
            return f"scratch {name} holds fewer entries than its lanes need"
    for name in ("plan_host", "plan_dev"):
        if scratch[name].numel() < int(scratch["lanes"]) * int(scratch["columns"]):
            return f"scratch {name} holds fewer entries than its lanes need"
    if scratch["plan_count"].numel() < int(scratch["lanes"]):
        return "scratch plan_count holds fewer entries than its lanes"
    if tensors["request_ids"].numel() < int(lanes):
        return f"request_ids holds {tensors['request_ids'].numel()} entries, the launch {lanes} lanes"
    if tensors.get("idle") is not None and tensors["idle"].numel() < int(lanes):
        return f"idle holds {tensors['idle'].numel()} entries, the launch {lanes} lanes"
    return ""


def launch_config(union, request_ids, states, src_table, res_table, hot_table, src_bs, res_bs, lru, hot_cache, idle,
                  tally, guard, guarded, pcie, fresh, scratch, skip, nthreads=None):
    nthreads = nthreads or threads()
    kb = int(scratch["kb"])
    hb = -(-int(lru.shape[1]) // nthreads) * nthreads
    log = kb.bit_length() - 1
    if kb < 1 or kb & (kb - 1):
        raise ValueError(f"resolve cuda needs a power-of-two block, found kb={kb}")
    if int(lru.shape[1]) > 32767:
        raise ValueError(f"resolve cuda keeps hot slots in 16 bits, found region={int(lru.shape[1])}")
    if kb > 8192 or kb * 12 > 99 * 1024 - 256 or hb // nthreads > 64 or -(-kb // nthreads) > 64:
        raise ValueError(f"resolve cuda does not fit kb={kb} hb={hb} threads={nthreads}")
    config = dict(KB=kb, HB=hb, LOG=log, THREADS=nthreads, HAS_RES=int(res_table is not None),
                  HAS_IDLE=int(idle is not None), TALLY=int(tally is not None),
                  GUARD=int(bool(guarded) and guard is not None), PCIE=int(pcie is not None),
                  FRESH=int(fresh is not None), SKIP=int(bool(skip)))
    config.update(SRC_BS=int(src_bs), HOT_BS=int(hot_cache.shape[1]), RES_BS=max(int(res_bs), 1))
    for item in filter(None, str(os.environ.get(DEFINES_ENV) or "").split(",")):
        key, _, value = item.partition("=")
        config[key] = value
    config["UN_WIDE"] = int(union.dtype == torch.int64)
    config["UN_T"] = C_TYPES[union.dtype]
    config["RQ_T"] = C_TYPES[request_ids.dtype]
    config["HT_T"] = C_TYPES[hot_table.dtype]
    config["ST_T"] = C_TYPES[states.dtype]
    config["SR_T"] = C_TYPES[src_table.dtype]
    config["RS_T"] = C_TYPES[res_table.dtype] if res_table is not None else "int"
    return config


def resolve_launch(union, layout, parts, request_ids, states, src_table, src_bs, res_table, res_bs, res_null,
                   hot_table, attn_block_stride, dgi, lru, host_cache, hot_cache, attn, hot_out, host_out, dev_out,
                   counts_out, idle, stats, tally, guard, pcie, scratch, piece, guarded, skip=True, fresh=None,
                   pcie_calls=0, pcie_rows=1, nthreads=None, viol=None):
    lanes = union.shape[0]
    if lanes == 0:
        return 0
    nthreads = nthreads or threads()
    config = launch_config(union, request_ids, states, src_table, res_table, hot_table, src_bs, res_bs, lru, hot_cache,
                           idle, tally, guard, guarded, pcie, fresh, scratch, skip, nthreads)
    device = union.device
    _, function, smem = build(config, device)
    own = (viol is None or viol.dtype != torch.int64 or viol.device != device or viol.numel() < 1
           or not viol.is_contiguous())
    region = int(lru.shape[1])
    lists = floor_extent(hot_out, host_out, dev_out)
    fields = dict(
        union_p=pointer(union), union_stride=union.stride(0), layout=pointer(layout), parts=int(parts), lanes=lanes,
        request_ids=pointer(request_ids), states=pointer(states), state_count=states.numel(),
        src_table=pointer(src_table), src_stride=src_table.stride(0), src_reqs=src_table.shape[0],
        src_blocks=src_table.shape[1], src_bs=int(src_bs), res_table=pointer(res_table),
        res_stride=res_table.stride(0) if res_table is not None else 0,
        res_reqs=res_table.shape[0] if res_table is not None else 0,
        res_blocks=res_table.shape[1] if res_table is not None else 0, res_bs=max(int(res_bs), 1),
        res_null=int(res_null), hot_table=pointer(hot_table), hot_table_stride=hot_table.stride(0),
        hot_bs=int(hot_cache.shape[1]), attn_block_stride=int(attn_block_stride), dgi=pointer(dgi), lru=pointer(lru),
        region=region, host_total=host_cache.shape[0], hot_total=hot_cache.shape[0] * hot_cache.shape[1],
        attn=pointer(attn), attn_stride=attn.stride(0), hot_out=pointer(hot_out), host_out=pointer(host_out),
        dev_out=pointer(dev_out), counts_out=pointer(counts_out), idle=pointer(idle), stats=pointer(stats),
        tally=pointer(tally), guard=pointer(guard), pcie=pointer(pcie), marks=pointer(scratch["marks"]),
        devs=pointer(scratch["devs"]), plan_host=pointer(scratch["plan_host"]), plan_dev=pointer(scratch["plan_dev"]),
        plan_count=pointer(scratch["plan_count"]), plan_stride=int(scratch["columns"]), piece=max(int(piece), 1),
        fresh=pointer(fresh), fresh_stride=fresh.stride(0) if fresh is not None else 0, pcie_calls=int(pcie_calls),
        pcie_rows=int(pcie_rows), prof=pointer(PROF_BUFFER.get("t")), ext_union=extent(union),
        ext_src=extent(src_table), ext_res=extent(res_table), ext_hot=extent(hot_table), ext_attn=extent(attn),
        ext_fresh=extent(fresh), ext_lists=lists, ext_counts=extent(counts_out),
        ext_plan=floor_extent(scratch["plan_host"], scratch["plan_dev"]),
        ext_marks=floor_extent(scratch["marks"], scratch["devs"]), ext_dgi=extent(dgi), ext_lru=extent(lru),
        state_rows=min(extent(dgi), extent(lru)) // region, viol=pointer(counter(device) if own else viol),
        ext_req=extent(request_ids), ext_idle=extent(idle), ext_pcount=extent(scratch["plan_count"]))
    args = Args(**fields)
    holder = ctypes.c_void_p(ctypes.addressof(args))
    params = (ctypes.c_void_p * 1)(holder)
    stream = torch.cuda.current_stream(device).cuda_stream
    check(driver().cuLaunchKernel(function, lanes, 1, 1, nthreads, 1, 1, smem, ctypes.c_void_p(stream), params,
                                  None), "cuLaunchKernel")
    return lanes


def build_path():
    return _STATE["build_path"]


def violations():
    torch.cuda.synchronize()
    return sum(int(held.item()) for held in _STATE["viol"].values())
