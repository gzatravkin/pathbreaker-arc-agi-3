import json
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PLAN_NAME = "expert_cut_plan.json"
EXPERT_TENSOR = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(\d+)\.(.+)$")
GATE_TENSOR = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
MTP_TENSOR = re.compile(r"^mtp\.layers\.\d+\.mlp\.(?:gate\.weight|experts\.(?:down_proj|gate_up_proj))$")
MTP_EXPERT_TENSOR = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts\.)(\d+)\.(.+)$")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
TORCH_DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
                "I16": "int16", "U16": "uint16", "F16": "float16", "BF16": "bfloat16", "I32": "int32",
                "U32": "uint32", "F32": "float32", "I64": "int64", "U64": "uint64", "F64": "float64"}
WORKERS_DEFAULT = 32
WINDOW_MIB_DEFAULT = 2048
COALESCE_MIB_DEFAULT = 64
GAP_KIB_DEFAULT = 1024
INSTALLED_MARK = "_taaf_expert_cut_installed"


def safetensors_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        head = json.loads(handle.read(size).decode("utf-8"))
    head.pop("__metadata__", None)
    return head, 8 + size


def make_plan(rows_by_layer, mtp_rows, workers=WORKERS_DEFAULT, window_mib=WINDOW_MIB_DEFAULT,
              coalesce_mib=COALESCE_MIB_DEFAULT, gap_kib=GAP_KIB_DEFAULT):
    return {"experts_kept": len(mtp_rows), "workers": int(workers), "window_mib": int(window_mib),
            "coalesce_mib": int(coalesce_mib), "gap_kib": int(gap_kib),
            "rows_by_layer": {str(layer): [int(one) for one in rows] for layer, rows in rows_by_layer.items()},
            "mtp_rows": [int(one) for one in mtp_rows]}


class Plan:
    def __init__(self, value):
        self.rows_by_layer = {int(layer): [int(one) for one in rows]
                              for layer, rows in value["rows_by_layer"].items()}
        self.keep_by_layer = {layer: {expert: index for index, expert in enumerate(rows)}
                              for layer, rows in self.rows_by_layer.items()}
        self.mtp_rows = [int(one) for one in value["mtp_rows"]]
        self.mtp_keep = {expert: index for index, expert in enumerate(self.mtp_rows)}
        self.workers = int(value.get("workers") or WORKERS_DEFAULT)
        self.window = int(value.get("window_mib") or WINDOW_MIB_DEFAULT) << 20
        self.coalesce = int(value.get("coalesce_mib", COALESCE_MIB_DEFAULT)) << 20
        self.gap = int(value.get("gap_kib", GAP_KIB_DEFAULT)) << 10

    @classmethod
    def read(cls, directory):
        return cls(json.loads((Path(directory) / PLAN_NAME).read_text(encoding="utf-8")))


def entries(head, plan):
    rows = []
    experts = gates = mtp = 0
    left = []
    for name, spec in head.items():
        found = EXPERT_TENSOR.match(name)
        head_found = None if found else MTP_EXPERT_TENSOR.match(name)
        if found or head_found:
            prefix, expert, tail = ((found.group(1), found.group(3), found.group(4))
                                    if found else head_found.groups())
            keep = plan.keep_by_layer[int(found.group(2))] if found else plan.mtp_keep
            if int(expert) not in keep:
                continue
            rows.append((f"{prefix}{keep[int(expert)]}.{tail}", name, spec, None, True))
            experts += 1
            mtp += 0 if found else 1
            continue
        gate = GATE_TENSOR.match(name)
        if gate or MTP_TENSOR.match(name):
            picked = plan.rows_by_layer[int(gate.group(1))] if gate else plan.mtp_rows
            if spec["shape"][0] != 512:
                raise RuntimeError(f"{name} does not lead with 512: {spec['shape']}")
            cut = dict(spec)
            cut["shape"] = [len(picked), *spec["shape"][1:]]
            rows.append((name, name, cut, picked, True))
            gates += 1 if gate else 0
            mtp += 0 if gate else 1
            continue
        if (spec.get("shape") and spec["shape"][0] == 512 and ".mlp." in name and "shared_expert" not in name
                and ("experts" in name or "gate" in name)):
            left.append(name)
        rows.append((name, name, spec, None, False))
    rows.sort(key=lambda row: row[0])
    return rows, (experts, gates, mtp), left


def scan(paths, plan):
    experts = gates = mtp = 0
    left = []
    for path in paths:
        head, _ = safetensors_header(path)
        _, counts, over = entries(head, plan)
        experts += counts[0]
        gates += counts[1]
        mtp += counts[2]
        left += over
    return experts, gates, mtp, left


class Slice:
    def __init__(self, spec):
        self.spec = spec

    def get_shape(self):
        return list(self.spec["shape"])

    def get_dtype(self):
        return self.spec["dtype"]


def row_runs(picked):
    runs = []
    for row in picked:
        if runs and runs[-1][0] + runs[-1][1] == row:
            runs[-1][1] += 1
        else:
            runs.append([row, 1])
    return runs


def _row_bytes(spec):
    step = DTYPE_BYTES[spec["dtype"]]
    for size in spec["shape"][1:]:
        step *= size
    return step


class Batch:
    def __init__(self, owner):
        import numpy
        self.numpy = numpy
        self.owner = owner
        self.lock = threading.Lock()
        self.items = {}
        plan = owner.plan
        rank = {name: index for index, name in enumerate(owner.names)}
        pieces = []
        for row in owner.rows:
            new, name, spec, picked, cut = row
            if not cut and spec["dtype"] not in TORCH_DTYPES:
                continue
            size = owner._size(row)
            item = {"array": numpy.empty(size, dtype=numpy.uint8), "left": 0, "done": threading.Event(),
                    "error": None, "row": row}
            self.items[new] = item
            base = owner.body + owner.head[name]["data_offsets"][0]
            spans = []
            if picked is None:
                spans.append((base, size, 0))
            else:
                step = _row_bytes(spec)
                at = 0
                for first, count in row_runs(picked):
                    spans.append((base + first * step, count * step, at))
                    at += count * step
            for offset, length, at in spans:
                while length > 0:
                    take = min(length, plan.coalesce)
                    pieces.append((offset, take, new, at))
                    item["left"] += 1
                    offset += take
                    at += take
                    length -= take
            if not item["left"]:
                item["done"].set()
        pieces.sort(key=lambda piece: piece[0])
        self.reads = []
        for offset, take, new, at in pieces:
            last = self.reads[-1] if self.reads else None
            if last is not None and offset - last["end"] <= plan.gap and offset + take - last["start"] <= plan.coalesce:
                last["parts"].append((offset, take, new, at))
                last["end"] = offset + take
                last["rank"] = min(last["rank"], rank[new])
            else:
                self.reads.append({"start": offset, "end": offset + take, "parts": [(offset, take, new, at)],
                                   "rank": rank[new]})
        self.reads.sort(key=lambda read: read["rank"])
        self.wanted = sum(item["array"].nbytes for item in self.items.values())
        self.fetched = sum(read["end"] - read["start"] for read in self.reads)
        self.pending = len(self.reads)
        self.failed = None
        self.closed = False
        self.began = time.time()
        self.pool = ThreadPoolExecutor(max_workers=max(1, plan.workers), thread_name_prefix="expert-cut-batch")
        for read in self.reads:
            self.pool.submit(self._run, read)

    def _settle(self, item, failure):
        with self.lock:
            if failure is not None:
                item["error"] = failure
                item["done"].set()
                return
            item["left"] -= 1
            if item["left"] == 0:
                item["done"].set()

    def _run(self, read):
        numpy = self.numpy
        try:
            if self.closed:
                return
            buffer = numpy.empty(read["end"] - read["start"], dtype=numpy.uint8)
            self.owner._fill(self.owner._handle(), read["start"], memoryview(buffer))
            for offset, take, new, at in read["parts"]:
                item = self.items[new]
                lead = offset - read["start"]
                numpy.copyto(item["array"][at:at + take], buffer[lead:lead + take])
                self._settle(item, None)
        except BaseException as failure:
            if self.closed:
                return
            self.failed = failure
            print(f"EXPERT_CUT_READ {Path(self.owner.path).name}: the read of bytes {read['start']}..{read['end']} "
                  f"failed: {failure!r}", flush=True)
            for _, _, new, _ in read["parts"]:
                self._settle(self.items[new], failure)
        finally:
            with self.lock:
                self.pending -= 1
                last = self.pending == 0
            if last and not self.closed:
                self._report()

    def _report(self):
        took = max(time.time() - self.began, 1e-9)
        plan = self.owner.plan
        print(f"EXPERT_CUT_READ {Path(self.owner.path).name}: {self.wanted / 2**30:.2f} GiB of tensors in "
              f"{len(self.reads)} reads of up to {plan.coalesce >> 20} MiB ({(self.fetched - self.wanted) / 2**20:.0f} "
              f"MiB of gaps up to {plan.gap >> 10} KiB read between them) by {plan.workers} threads in {took:.1f} s, "
              f"{self.fetched / took / 1e9:.2f} GB/s{'' if self.failed is None else ', FAILED'}", flush=True)

    def take(self, name):
        item = self.items.get(name)
        if item is None or item["array"] is None:
            return None
        item["done"].wait()
        if item["error"] is not None:
            raise item["error"]
        array, item["array"] = item["array"], None
        import torch
        spec = item["row"][2]
        dtype = getattr(torch, TORCH_DTYPES[spec["dtype"]])
        if array.size == 0:
            return torch.empty(tuple(spec["shape"]), dtype=dtype)
        return torch.from_numpy(array).view(dtype).reshape(tuple(spec["shape"]))

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)


class CutOpen:
    def __init__(self, real, path, plan, rows, body, head):
        self.real = real
        self.path = str(path)
        self.plan = plan
        self.body = body
        self.rows = rows
        self.names = [row[0] for row in rows]
        self.head = head
        self.cut = {row[0]: row for row in rows if row[4]}
        self.order = [row[0] for row in rows if row[4]]
        self.pool = None
        self.pending = {}
        self.cursor = 0
        self.inflight = 0
        self.local = threading.local()
        self.handles = []
        self.lock = threading.Lock()
        self.batch = None

    def _handle(self):
        found = getattr(self.local, "handle", None)
        if found is None:
            found = self.local.handle = open(self.path, "rb", buffering=0)
            with self.lock:
                self.handles.append(found)
        return found

    def _fill(self, handle, offset, view):
        done = 0
        while done < len(view):
            handle.seek(offset + done)
            got = handle.readinto(view[done:])
            if not got:
                raise EOFError(f"{self.path} ended {len(view) - done} B early")
            done += got

    def _size(self, row):
        _, _, spec, picked, _ = row
        start, end = spec["data_offsets"]
        if picked is None:
            return end - start
        step = DTYPE_BYTES[spec["dtype"]]
        for size in spec["shape"][1:]:
            step *= size
        return step * len(picked)

    def _read(self, row):
        _, name, spec, picked, _ = row
        handle = self._handle()
        origin = self.head[name]["data_offsets"]
        out = bytearray(self._size(row))
        view = memoryview(out)
        if picked is None:
            self._fill(handle, self.body + origin[0], view)
            return out
        step = DTYPE_BYTES[spec["dtype"]]
        for size in spec["shape"][1:]:
            step *= size
        at = 0
        for first, count in row_runs(picked):
            take = count * step
            self._fill(handle, self.body + origin[0] + first * step, view[at: at + take])
            at += take
        return out

    def _top_up(self):
        while self.cursor < len(self.order):
            name = self.order[self.cursor]
            row = self.cut[name]
            size = self._size(row)
            if self.inflight and self.inflight + size > self.plan.window:
                break
            if self.pool is None:
                self.pool = ThreadPoolExecutor(max_workers=max(1, self.plan.workers),
                                               thread_name_prefix="expert-cut-read")
            self.pending[name] = (self.pool.submit(self._read, row), size)
            self.inflight += size
            self.cursor += 1

    def keys(self):
        return list(self.names)

    def __iter__(self):
        return iter(self.names)

    def get_tensor(self, name):
        if self.plan.coalesce:
            if self.batch is None:
                self.batch = Batch(self)
            got = self.batch.take(name)
            if got is not None:
                return got
        row = self.cut.get(name)
        if row is None:
            return self.real.get_tensor(name)
        import torch
        self._top_up()
        held = self.pending.pop(name, None)
        if held is None:
            data = self._read(row)
        else:
            data = held[0].result()
            self.inflight -= held[1]
        self._top_up()
        spec = row[2]
        count = 1
        for size in spec["shape"]:
            count *= size
        dtype = getattr(torch, TORCH_DTYPES[spec["dtype"]])
        if count == 0:
            return torch.empty(tuple(spec["shape"]), dtype=dtype)
        return torch.frombuffer(data, dtype=torch.uint8).view(dtype).reshape(tuple(spec["shape"]))

    def get_slice(self, name):
        row = self.cut.get(name)
        if row is None:
            return self.real.get_slice(name)
        return Slice(row[2])

    def __getattr__(self, name):
        return getattr(self.real, name)

    def __enter__(self):
        self.real.__enter__()
        return self

    def __exit__(self, *exc):
        if self.batch is not None:
            self.batch.close()
            self.batch = None
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        for handle in self.handles:
            handle.close()
        self.handles = []
        self.pending = {}
        return self.real.__exit__(*exc)


class Opener:
    def __init__(self, original):
        self.original = original
        self.plans = {}

    def plan_for(self, directory):
        key = str(directory)
        if key not in self.plans:
            self.plans[key] = Plan.read(directory) if (Path(directory) / PLAN_NAME).is_file() else None
        return self.plans[key]

    def __call__(self, filename, *args, **kwargs):
        real = self.original(filename, *args, **kwargs)
        plan = self.plan_for(Path(str(filename)).parent)
        if plan is None:
            return real
        head, body = safetensors_header(filename)
        rows, _, _ = entries(head, plan)
        if not any(row[4] for row in rows) and len(rows) == len(head):
            return real
        return CutOpen(real, filename, plan, rows, body, head)


def install(namespace):
    current = namespace.get("safe_open")
    if current is None or isinstance(current, Opener):
        return False
    namespace["safe_open"] = Opener(current)
    namespace[INSTALLED_MARK] = True
    return True
