import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

ENV = "TAAF_VLLM_INIT_TRACE"
PATH_ENV = "TAAF_VLLM_INIT_TRACE_PATH"
PRINT_ENV = "TAAF_VLLM_INIT_TRACE_PRINT"
IMPORTS_ENV = "TAAF_VLLM_INIT_TRACE_IMPORTS"
DEFAULT_PATH = os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "init_trace.json")
PREFIX = "INIT_T"
SLOW_IMPORT_S = 0.25
TOP = 30
SERIES_CAP = 600
WATCH = frozenset((
    "torch", "numpy", "transformers", "vllm", "triton", "flashinfer", "tokenizers", "safetensors", "fastapi",
    "uvicorn", "pydantic", "openai", "xgrammar", "PIL", "cv2", "scipy", "ray", "compressed_tensors",
    "mistral_common", "lark", "msgspec", "zmq", "cloudpickle", "einops", "aiohttp", "huggingface_hub",
))

_LOCAL = threading.local()
_LOCK = threading.RLock()
MOUNT_ROOT = "/kaggle/input"
MOUNT_DEPTH = 5
_MOUNT = {"opens": {}, "lists": {}}
_STATE = {"fd": None, "pid": None, "role": "proc", "imports": [], "compiled": [0, 0.0], "aggs": {},
          "dumped": set(), "missing": set(), "path": None, "installed": False}


def trace_path():
    return os.environ.get(PATH_ENV) or DEFAULT_PATH


def events_path(path=None):
    base = str(path or trace_path())
    return base[:-5] + ".jsonl" if base.endswith(".json") else base + ".jsonl"


def _role_from_name(text):
    if "EngineCore" in text:
        return "enginecore"
    if "Worker" in text or "worker" in text:
        return "worker"
    return None


def role():
    if _STATE["role"] != "proc" and _STATE["pid"] == os.getpid():
        return _STATE["role"]
    found = None
    mp = sys.modules.get("multiprocessing.process")
    if mp is not None:
        try:
            found = _role_from_name(mp.current_process().name)
        except Exception:
            found = None
    if found is None:
        try:
            with open("/proc/self/cmdline", "rb") as handle:
                text = handle.read().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            text = " ".join(sys.argv or [])
        found = _role_from_name(text)
        if found is None and ("api_server" in text or " serve " in text or "vllm.entrypoints" in text):
            found = "api"
    _STATE["pid"] = os.getpid()
    _STATE["role"] = found or "proc"
    return _STATE["role"]


def _open():
    pid = os.getpid()
    if _STATE["fd"] is not None and _STATE["pid"] == pid:
        return _STATE["fd"]
    if _STATE["pid"] != pid:
        _STATE["role"] = "proc"
        _STATE["imports"] = []
        _STATE["aggs"] = {}
        _STATE["dumped"] = set()
        _STATE["compiled"] = [0, 0.0]
    _STATE["pid"] = pid
    target = events_path()
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        _STATE["fd"] = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    except OSError:
        _STATE["fd"] = -1
    return _STATE["fd"]


def emit(name, t0=None, t1=None, **extra):
    now = time.time()
    t1 = now if t1 is None else t1
    t0 = t1 if t0 is None else t0
    row = {"pid": os.getpid(), "role": role(), "name": name, "t0": round(t0, 4), "t1": round(t1, 4)}
    row.update(extra)
    fd = _open()
    if fd is not None and fd >= 0:
        try:
            os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
        except OSError:
            pass
    if os.environ.get(PRINT_ENV) == "1":
        tail = "".join(f" {key}={value}" for key, value in extra.items() if not isinstance(value, (list, dict)))
        try:
            sys.stderr.write(f"{PREFIX} pid={row['pid']} role={row['role']} {name} t={t1:.3f} "
                             f"dur={t1 - t0:.3f}{tail}\n")
            sys.stderr.flush()
        except Exception:
            pass
    return row


def mark(name, **extra):
    return emit(name, None, None, **extra)


def process_start():
    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as handle:
            raw = handle.read()
        ticks = int(raw[raw.rindex(")") + 2:].split()[19])
        up = time.clock_gettime(time.CLOCK_BOOTTIME)
        return time.time() - (up - ticks / os.sysconf("SC_CLK_TCK"))
    except Exception:
        return None


def _flush_aggs(label=None):
    with _LOCK:
        for key in [key for key in _STATE["aggs"] if label is None or key == label]:
            count, total, first, last, longest, series = _STATE["aggs"].pop(key)
            extra = {"n": count, "total_s": round(total, 3), "max_s": round(longest, 3)}
            if series:
                extra["series"] = series
            emit(key + " (all calls)", first, last, **extra)


def _add_agg(label, t0, t1, series=False):
    with _LOCK:
        held = _STATE["aggs"].get(label)
        if held is None:
            held = _STATE["aggs"][label] = [0, 0.0, t0, t1, 0.0, []]
        held[0] += 1
        held[1] += t1 - t0
        held[3] = t1
        held[4] = max(held[4], t1 - t0)
        if series and len(held[5]) < SERIES_CAP:
            held[5].append([round(t0, 3), round(t1 - t0, 3)])


def _compiled_mark(where):
    count, seconds = _STATE["compiled"]
    emit("source_compiles " + where, None, None, n=count, total_s=round(seconds, 3))


def _done(label, t0, agg, after, error=False):
    t1 = time.time()
    if agg:
        _add_agg(label, t0, t1)
    else:
        emit(label, t0, t1, **({"error": True} if error else {}))
    if after is not None:
        try:
            after()
        except Exception:
            pass


class _TimedAsync:
    def __init__(self, held, label):
        self.held, self.label = held, label

    async def __aenter__(self):
        t0 = time.time()
        try:
            return await self.held.__aenter__()
        finally:
            emit(self.label + " (enter)", t0, time.time())

    async def __aexit__(self, *exc):
        mark(self.label + " (exit begins)")
        return await self.held.__aexit__(*exc)


class _TimedSync:
    def __init__(self, held, label):
        self.held, self.label = held, label

    def __enter__(self):
        t0 = time.time()
        try:
            return self.held.__enter__()
        finally:
            emit(self.label + " (enter)", t0, time.time())

    def __exit__(self, *exc):
        mark(self.label + " (exit begins)")
        return self.held.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self.held, name)


def _wrap(func, label, agg=False, after=None):
    import inspect
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                _done(label, t0, agg, after)
        return wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
        except BaseException:
            _done(label, t0, agg, after, error=True)
            raise
        kind = type(result).__name__
        if kind == "_AsyncGeneratorContextManager":
            return _TimedAsync(result, label)
        if kind == "_GeneratorContextManager":
            return _TimedSync(result, label)
        _done(label, t0, agg, after)
        return result
    return wrapper


def _missing(what):
    with _LOCK:
        if what in _STATE["missing"]:
            return
        _STATE["missing"].add(what)
    mark("wrap_missing " + what)


def wrap_attr(holder, attr, label, agg=False, after=None):
    import inspect
    try:
        raw = inspect.getattr_static(holder, attr)
    except AttributeError:
        _missing(label)
        return False
    try:
        if isinstance(raw, staticmethod):
            setattr(holder, attr, staticmethod(_wrap(raw.__func__, label, agg, after)))
        elif isinstance(raw, classmethod):
            setattr(holder, attr, classmethod(_wrap(raw.__func__, label, agg, after)))
        elif callable(raw):
            setattr(holder, attr, _wrap(raw, label, agg, after))
        else:
            _missing(label)
            return False
    except (AttributeError, TypeError):
        _missing(label)
        return False
    return True


def wrap_methods(module, names, agg=(), after=None):
    after = after or {}
    classes = [value for value in list(vars(module).values())
               if isinstance(value, type) and getattr(value, "__module__", None) == module.__name__]
    for name in list(names) + list(agg):
        owners = [cls for cls in classes if name in vars(cls)]
        if not owners:
            _missing(f"{module.__name__}.*.{name}")
        for cls in owners:
            wrap_attr(cls, name, f"{cls.__name__}.{name}", agg=name in agg, after=after.get(name))


def wrap_globals(module, names, after=None):
    after = after or {}
    for name in names:
        if name in vars(module):
            wrap_attr(module, name, f"{module.__name__.rsplit('.', 1)[-1]}.{name}", after=after.get(name))
        else:
            _missing(f"{module.__name__}.{name}")


def _hook_torch(module):
    _compiled_mark("after import torch")
    cuda = getattr(module, "cuda", None)
    if cuda is None or not hasattr(cuda, "_lazy_init"):
        _missing("torch.cuda._lazy_init")
        return
    original = cuda._lazy_init

    @functools.wraps(original)
    def lazy_init():
        fresh = not cuda.is_initialized()
        t0 = time.time()
        try:
            return original()
        finally:
            if fresh and cuda.is_initialized():
                emit("cuda_context_init (torch.cuda._lazy_init)", t0, time.time())

    cuda._lazy_init = lazy_init
    graphs = sys.modules.get("torch.cuda.graphs")
    holder = getattr(graphs, "CUDAGraph", None)
    if holder is None or not hasattr(holder, "capture_begin") or not hasattr(holder, "capture_end"):
        _missing("torch.cuda.graphs.CUDAGraph.capture_begin/capture_end")
        return
    begin, end = holder.capture_begin, holder.capture_end

    @functools.wraps(begin)
    def capture_begin(self, *args, **kwargs):
        _LOCAL.capture_started = time.time()
        return begin(self, *args, **kwargs)

    @functools.wraps(end)
    def capture_end(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            started = getattr(_LOCAL, "capture_started", None)
            if started is not None:
                _add_agg("cuda_graph_capture", started, time.time(), series=True)

    holder.capture_begin, holder.capture_end = capture_begin, capture_end


def _hook_transformers_auto(module):
    wrap_methods(module, ["from_pretrained"])


def _hooks():
    def worker_dump(reason):
        return lambda: (_flush_aggs(), mount_report(reason), dump_imports(reason))

    return {
        "torch": _hook_torch,
        "transformers.models.auto.tokenization_auto": _hook_transformers_auto,
        "transformers.models.auto.processing_auto": _hook_transformers_auto,
        "transformers.models.auto.configuration_auto": _hook_transformers_auto,
        "transformers.models.auto.image_processing_auto": _hook_transformers_auto,
        "transformers.models.auto.video_processing_auto": _hook_transformers_auto,
        "vllm.tokenizers.registry": lambda m: wrap_globals(m, ["get_tokenizer", "cached_get_tokenizer"]),
        "vllm.entrypoints.openai.api_server": lambda m: wrap_globals(
            m, ["run_server", "run_server_worker", "build_async_engine_client",
                "build_async_engine_client_from_engine_args", "init_app_state", "build_app"]),
        "vllm.entrypoints.launcher": lambda m: wrap_globals(m, ["serve_http"]),
        "uvicorn.server": lambda m: wrap_methods(m, ["startup"]),
        "vllm.v1.engine.async_llm": lambda m: wrap_methods(m, ["__init__", "from_vllm_config"]),
        "vllm.v1.engine.utils": lambda m: (wrap_globals(m, ["launch_core_engines", "wait_for_engine_startup"]),
                                           wrap_methods(m, ["__init__"])),
        "vllm.v1.engine.core": lambda m: wrap_methods(
            m, ["__init__", "_initialize_kv_caches", "run_engine_core"]),
        "vllm.v1.executor.multiproc_executor": lambda m: wrap_methods(
            m, ["_init_executor", "wait_for_ready", "worker_main"]),
        "vllm.v1.worker.gpu_worker": lambda m: (
            wrap_methods(m, ["init_device", "determine_available_memory", "initialize_from_config"],
                         after={}),
            wrap_methods(m, ["load_model"], after={"load_model": worker_dump("after worker load_model")}),
            wrap_methods(m, ["compile_or_warm_up_model"],
                         after={"compile_or_warm_up_model": worker_dump("after worker warmup")}),
            wrap_globals(m, ["kernel_warmup", "warmup_kernels"])),
        "vllm.v1.worker.gpu.model_runner": lambda m: wrap_methods(
            m, ["load_model", "profile_run", "capture_model", "initialize_kv_cache"], agg=["_dummy_run"],
            after={"capture_model": lambda: _flush_aggs("_dummy_run")}),
        "vllm.v1.worker.gpu.cudagraph_utils": lambda m: wrap_methods(
            m, ["capture"], after={"capture": lambda: _flush_aggs("cuda_graph_capture")}),
        "vllm.model_executor.model_loader.default_loader": lambda m: wrap_methods(m, ["load_model", "load_weights"]),
        "vllm.model_executor.model_loader.utils": lambda m: wrap_globals(
            m, ["initialize_model", "process_weights_after_loading"]),
        "vllm.model_executor.warmup.kernel_warmup": lambda m: wrap_globals(m, ["kernel_warmup"]),
        "triton.compiler.compiler": lambda m: wrap_attr(m, "compile", "triton.compile", agg=True),
    }


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.hooks = _hooks()
        self.imports = bool(_import_roles())

    def find_spec(self, fullname, path=None, target=None):
        for finder in sys.meta_path:
            if finder is self:
                continue
            probe = getattr(finder, "find_spec", None)
            if probe is None:
                continue
            spec = probe(fullname, path, target)
            if spec is None:
                continue
            loader = spec.loader
            if loader is not None and not isinstance(loader, type) and hasattr(loader, "exec_module"):
                try:
                    loader.exec_module = self._timed(fullname, loader.exec_module)
                except (AttributeError, TypeError):
                    pass
            return spec
        return None

    def _timed(self, fullname, original):
        def exec_module(module):
            stack = getattr(_LOCAL, "stack", None)
            if stack is None:
                stack = _LOCAL.stack = []
            entry = [time.time(), 0.0]
            stack.append(entry)
            try:
                return original(module)
            finally:
                end = time.time()
                stack.pop()
                total = end - entry[0]
                if stack:
                    stack[-1][1] += total
                self._finished(fullname, entry[0], end, total, total - entry[1], module)
        return exec_module

    def _finished(self, fullname, t0, t1, total, own, module):
        if self.imports:
            _STATE["imports"].append((fullname, total, own))
        top = fullname.split(".", 1)[0]
        if fullname == top and top in WATCH:
            emit("import " + fullname, t0, t1)
        elif fullname.startswith("vllm.taaf_") or total >= SLOW_IMPORT_S:
            emit("import " + fullname, t0, t1, own_s=round(own, 3))
        hook = self.hooks.get(fullname)
        if hook is not None:
            try:
                hook(module)
            except Exception as failure:
                mark("hook_error " + fullname, error=f"{type(failure).__name__}: {failure}"[:200])
        if fullname == "vllm" or fullname == "transformers":
            _compiled_mark("after import " + fullname)


def _import_roles():
    said = str(os.environ.get(IMPORTS_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ()
    return ("all",) if said in ("1", "true", "yes", "on", "all") else tuple(part.strip() for part in said.split(","))


def dump_imports(reason):
    roles = _import_roles()
    mine = role()
    if not roles or ("all" not in roles and mine not in roles):
        return None
    with _LOCK:
        rows = list(_STATE["imports"])
    if not rows:
        return None
    base = os.path.dirname(trace_path()) or "."
    name = f"importtime_{mine}.txt" if "all" not in roles else f"importtime_{mine}_{os.getpid()}.txt"
    ordered = sorted(rows, key=lambda row: -row[2])
    count, seconds = _STATE["compiled"]
    summary = (f"IMPORTTIME {mine} pid={os.getpid()} {reason}: {len(rows)} modules, top{TOP} self "
               f"{sum(row[2] for row in ordered[:TOP]):.2f} s of {sum(row[2] for row in rows):.2f} s self total, "
               f"{count} modules compiled from source in {seconds:.2f} s")
    lines = [summary, f"{'self_s':>9} {'cumulative_s':>13}  module"]
    lines += [f"{own:9.4f} {total:13.4f}  {fullname}" for fullname, total, own in ordered[:TOP]]
    lines += ["", "all modules by self time:"]
    lines += [f"{own:9.4f} {total:13.4f}  {fullname}" for fullname, total, own in ordered]
    try:
        with open(os.path.join(base, name), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        return None
    mark("importtime_dump", file=name, modules=len(rows), top30_self_s=round(sum(row[2] for row in ordered[:TOP]), 3),
         compiled_n=count, compiled_s=round(seconds, 3))
    try:
        sys.stderr.write(summary + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    return summary


def _count_source_compiles():
    for cls in importlib.machinery.SourceFileLoader.__mro__:
        original = vars(cls).get("source_to_code")
        if original is None:
            continue

        def source_to_code(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                _STATE["compiled"][0] += 1
                _STATE["compiled"][1] += time.perf_counter() - t0

        try:
            cls.source_to_code = source_to_code
        except (AttributeError, TypeError):
            pass
        return


def _mount_key(path):
    parts = path.split("/")
    return "/".join(parts[:MOUNT_DEPTH + 1])


def _audit(event, args):
    if event == "open":
        target = args[0]
        if isinstance(target, bytes):
            target = os.fsdecode(target)
        if not isinstance(target, str) or not target.startswith(MOUNT_ROOT):
            return
        key = _mount_key(target)
        try:
            size = os.stat(target).st_size
        except OSError:
            size = 0
        with _LOCK:
            held = _MOUNT["opens"].setdefault(key, [0, 0])
            held[0] += 1
            held[1] += size
    elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
        target = args[0]
        if isinstance(target, bytes):
            target = os.fsdecode(target)
        if isinstance(target, str) and target.startswith(MOUNT_ROOT):
            with _LOCK:
                _MOUNT["lists"][event] = _MOUNT["lists"].get(event, 0) + 1


def mount_report(where):
    with _LOCK:
        opens = {key: list(value) for key, value in _MOUNT["opens"].items()}
        lists = dict(_MOUNT["lists"])
    if not opens and not lists:
        return None
    files = sum(value[0] for value in opens.values())
    size = sum(value[1] for value in opens.values())
    top = sorted(opens.items(), key=lambda item: -item[1][0])[:12]
    return mark("mount_reads " + where, files=files, bytes=size, listings=sum(lists.values()),
                by_dir={key: {"files": value[0], "bytes": value[1]} for key, value in top}, listing_calls=lists)


def install_mount_audit():
    if not _STATE.get("audit"):
        _STATE["audit"] = True
        sys.addaudithook(_audit)


def install():
    if _STATE["installed"]:
        return
    _STATE["installed"] = True
    began = time.time()
    started = process_start()
    if started is not None:
        emit("process exec", started, started, argv=" ".join((sys.argv or [])[:3])[:160], ppid=os.getppid())
    emit("python startup to sitecustomize", started if started is not None else began, began)
    _count_source_compiles()
    install_mount_audit()
    sys.meta_path.insert(0, _Finder())
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=lambda: _STATE.update(fd=None))
    import atexit
    atexit.register(lambda: (_flush_aggs(), _compiled_mark("at exit"), mount_report("at exit"),
                             dump_imports("at exit")))


def consolidate(source=None, target=None):
    source = source or events_path()
    target = target or trace_path()
    rows = []
    try:
        with open(source, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return None
    rows.sort(key=lambda row: (row["t0"], row["t1"]))
    processes = {}
    for row in rows:
        held = processes.setdefault(str(row["pid"]), {"role": "proc", "first": row["t0"], "last": row["t1"]})
        if row["role"] != "proc":
            held["role"] = row["role"]
        held["first"] = min(held["first"], row["t0"])
        held["last"] = max(held["last"], row["t1"])
    document = {"version": 1, "events": rows, "processes": processes}
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
    os.replace(temporary, target)
    return document


def _chain():
    own = os.path.dirname(os.path.abspath(__file__))
    for entry in list(sys.path):
        if not entry or os.path.abspath(entry) == own:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_taaf_next_sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return


if __name__ == "sitecustomize" and os.environ.get(ENV, "").strip() not in ("", "0"):
    try:
        install()
    except Exception as _failure:
        sys.stderr.write(f"{PREFIX} install failed: {type(_failure).__name__}: {_failure}\n")
    try:
        _chain()
    except Exception:
        pass
