import contextlib
import functools
import hashlib
import heapq
import json
import mmap
import os
import queue
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import torch

HOT_ENV = "TAAF_VLLM_PLE_HOT"
GIB_ENV = "TAAF_VLLM_PLE_HOT_GIB"
DIR_ENV = "TAAF_VLLM_PLE_HOT_DIR"
THREADS_ENV = "TAAF_VLLM_PLE_HOT_THREADS"
EVERY_ENV = "TAAF_VLLM_PLE_HOT_EVERY"
WRITE_MIB_ENV = "TAAF_VLLM_PLE_HOT_WRITE_MIB"
PREWARM_ENV = "TAAF_VLLM_PLE_HOT_PREWARM"
DISK_ENV = "TAAF_VLLM_PLE_HOT_DISK_GIB"
READ_ENV = "TAAF_VLLM_PLE_HOT_READ"
PRERESOLVE_ENV = "TAAF_VLLM_PLE_HOT_PRERESOLVE"
PACKED_ENV = "TAAF_VLLM_PLE_PACKED"
DF11_ENV = "TAAF_VLLM_PLE_DF11"
DF11_GROUP = 32
DF11_LIMIT = 12
DF11_ESCAPE = 256
DF11_SAMPLE_ROWS = 1 << 20
DF11_MAGIC = b"TAAFDF11"
DF11_VERSION = 1
DF11_FILE_ENV = "TAAF_VLLM_PLE_DF11_FILE"
DF11_CHECKPOINT_ENV = "TAAF_VLLM_PLE_DF11_CHECKPOINT"
DF11_FILE_NAME = "ple_df11.taafdf11"
DF11_ROOT = "/kaggle/input"
DF11_DEPTH = 4
DF11_HEADER_RESERVE = 1 << 20
DF11_PROBE_ROWS = 64
DF11_CHECK_ROWS = 4
DF11_TENSOR_MARK = "ple_embedding.ngram_embedding"
DF11_ASYNC_ENV = "TAAF_VLLM_PLE_DF11_ASYNC"
DF11_STARTUP_ENV = "TAAF_VLLM_PLE_DF11_STARTUP"
DF11_STARTUP_THREADS_ENV = "TAAF_VLLM_PLE_DF11_STARTUP_THREADS"
DF11_STARTUP_PROCS_ENV = "TAAF_VLLM_PLE_DF11_STARTUP_PROCS"
DF11_STARTUP_ENCODER_ENV = "TAAF_VLLM_PLE_DF11_STARTUP_ENCODER"
DF11_STARTUP_ENCODERS = ("auto", "cuda", "cpu")
DF11_STARTUP_ORDER_ENV = "TAAF_VLLM_PLE_DF11_STARTUP_ORDER"
DF11_STARTUP_ORDER_TIMEOUT_ENV = "TAAF_VLLM_PLE_DF11_STARTUP_ORDER_TIMEOUT"
DF11_STARTUP_ORDERS = ("together", "after_weights")
DF11_STARTUP_ORDER_DEFAULT = "after_weights"
DF11_STARTUP_ORDER_TIMEOUT_DEFAULT = 1800
DF11_STARTUP_THREADS_DEFAULT = 16
DF11_STARTUP_PROCS_DEFAULT = 8
DF11_STARTUP_CHECK_ROWS = 256
DF11_TASK_ROWS = 65536
DF11_SAMPLE_CHUNKS = 64
DF11_SAMPLE_CHUNK_ROWS = 16384
DF11_READ_THREADS_ENV = "TAAF_VLLM_PLE_DF11_READ_THREADS"
DF11_DROP_ENV = "TAAF_VLLM_PLE_DF11_DROP_BEHIND"
DF11_READ_PIECE = 4 << 20
DF11_READ_THREADS_DEFAULT = 32
DF11_STALL_S = 60.0
DF11_PROGRESS_S = 30.0
PACK_THREADS = 8
ESCAPE_DEVICE_BYTES = 256 << 20
PRE_CHECKS = 3
PRE_TELL_EVERY = 2000
DUMMY_ENV = "TAAF_VLLM_LOAD_DUMMY"
DUMMY_DISK_MIB_ENV = "TAAF_VLLM_PLE_HOT_DUMMY_DISK_MIB"
DUMMY_DISK_MIB = 256
DUMMY_BASE_ROWS = 8192
DUMMY_SEED = 20261008
DUMMY_FILL_THREADS = 16
READS = ("mmap", "pread")
DEFAULT_GIB = 16
DEFAULT_THREADS = 64
DEFAULT_EVERY = 512
DEFAULT_WRITE_MIB = 2048
DEFAULT_DISK_GIB = 48
PROGRESS_EVERY = 8 << 30
TELL_EVERY = 1 << 21
PACK_ROWS = 32768
EXP_WINDOW = 15
ESCAPE_BYTES = 9
ALIGN = 4096
CHUNK = 64 << 20
FLUSH = 1 << 30
DROP_EVERY = 256 << 20
DIR_CANDIDATES = ("/kaggle/tmp", "/tmp")
UNFIT_FS = frozenset({"tmpfs", "ramfs", "devtmpfs", "hugetlbfs", "nfs", "nfs4", "cifs", "smb3", "9p",
                      "fuse.sshfs", "ceph", "glusterfs", "lustre"})
EMPTY = -1
TOMB = -2
KEY_MASK = 0xFFFFFFFF
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                      "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
_CLASSES = {}
_PACKED_CLASSES = {}
_PRE = {"plans": None, "replays": 0, "checked": 0, "seconds": 0.0, "told": False}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def say(line):
    print(f"PLE_HOT {line}", file=sys.stderr, flush=True)


def tell(line):
    print(f"PLE_PACKED {line}", file=sys.stderr, flush=True)


def mode():
    said = str(os.environ.get(HOT_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said in ("1", "true", "yes", "on", "flat"):
        return "flat"
    if said == "mount":
        return "mount"
    raise ValueError(f"{HOT_ENV} must be unset, 0, 1, flat or mount, found {said!r}")


def preresolve():
    said = str(os.environ.get(PRERESOLVE_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said in ("1", "true", "yes", "on"):
        if not mode():
            raise ValueError(f"{PRERESOLVE_ENV} resolves the rows of {HOT_ENV}; set {HOT_ENV} as well")
        return True
    raise ValueError(f"{PRERESOLVE_ENV} must be a boolean, found {said!r}")


def packed():
    said = str(os.environ.get(PACKED_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said not in ("1", "true", "yes", "on"):
        raise ValueError(f"{PACKED_ENV} must be a boolean, found {said!r}")
    if mode():
        raise ValueError(f"{PACKED_ENV} holds the whole PLE table packed in pinned host memory and {HOT_ENV} holds a "
                         f"row cache in its place; set one of them")
    return True


def df11():
    said = str(os.environ.get(DF11_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said not in ("1", "true", "yes", "on"):
        raise ValueError(f"{DF11_ENV} must be a boolean, found {said!r}")
    if not packed():
        raise ValueError(f"{DF11_ENV} changes how {PACKED_ENV} packs the whole PLE table; set {PACKED_ENV} as well")
    return True


def integer(name, default, low, high):
    said = str(os.environ.get(name) or "").strip()
    if not said:
        return default
    if not said.isdigit() or not low <= int(said) <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}, found {said!r}")
    return int(said)


def number(name, default, low, high):
    said = str(os.environ.get(name) or "").strip()
    if not said:
        return default
    try:
        value = float(said)
    except ValueError:
        value = float("nan")
    if not low < value <= high:
        raise ValueError(f"{name} must be a number above {low} and at most {high}, found {said!r}")
    return value


def read_kind():
    said = str(os.environ.get(READ_ENV) or "").strip().lower() or READS[0]
    if said not in READS:
        raise ValueError(f"{READ_ENV} must be unset or one of {list(READS)}, found {said!r}")
    return said


def dummy_load():
    said = str(os.environ.get(DUMMY_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{DUMMY_ENV} must be a boolean, found {said!r}")


def dummy_records(width, rows=None):
    rows = DUMMY_BASE_ROWS if rows is None else int(rows)
    half = int(width) // 2
    rng = np.random.default_rng(DUMMY_SEED)
    record = np.empty((rows, 1 + half + int(width)), dtype=np.uint8)
    record[:, 0] = 100 + (np.arange(rows) % 16)
    record[:, 1: 1 + half] = rng.integers(1, 16, size=(rows, half), dtype=np.uint8) | (
        rng.integers(1, 16, size=(rows, half), dtype=np.uint8) << 4)
    record[:, 1 + half:] = rng.integers(0, 256, size=(rows, int(width)), dtype=np.uint8)
    return record


def dummy_expected(base, ranks):
    return base[np.asarray(ranks, dtype=np.int64) % base.shape[0]]


def settings():
    return {"mode": mode(), "dummy": dummy_load(), "read": read_kind(), "gib": integer(GIB_ENV, DEFAULT_GIB, 1, 1024),
            "threads": integer(THREADS_ENV, DEFAULT_THREADS, 1, 1024),
            "every": integer(EVERY_ENV, DEFAULT_EVERY, 1, 1 << 30),
            "write_mib": integer(WRITE_MIB_ENV, DEFAULT_WRITE_MIB, 64, 1 << 20),
            "disk_gib": number(DISK_ENV, DEFAULT_DISK_GIB, 0, 4096),
            "dir": str(os.environ.get(DIR_ENV) or "").strip(),
            "prewarm": str(os.environ.get(PREWARM_ENV) or "").strip()}


def slots_for(gib, row_bytes, rows):
    return max(1, min(int(rows) - 1, int(gib) * 2**30 // int(row_bytes)))


def index_bytes(slots):
    return (1 << max(10, (2 * int(slots) - 1).bit_length())) * 8 + int(slots) * 9


def aligned(size):
    region = mmap.mmap(-1, max(int(size), ALIGN))
    return region, np.frombuffer(region, dtype=np.uint8)


def ceil_to(value, unit):
    return (value + unit - 1) // unit * unit


def pwrite_all(fd, view, offset):
    done = 0
    while done < len(view):
        if hasattr(os, "pwrite"):
            n = os.pwrite(fd, view[done:], offset + done)
        else:
            os.lseek(fd, offset + done, os.SEEK_SET)
            n = os.write(fd, view[done:])
        if n <= 0:
            raise OSError(f"write of {len(view) - done} bytes at {offset + done} returned {n}")
        done += n


def pread_into(fd, view, offset):
    if hasattr(os, "preadv"):
        return os.preadv(fd, [view], offset)
    os.lseek(fd, offset, os.SEEK_SET)
    data = os.read(fd, len(view))
    view[: len(data)] = data
    return len(data)


def open_file(path, flags, direct):
    flags |= getattr(os, "O_BINARY", 0)
    if direct and hasattr(os, "O_DIRECT"):
        try:
            return os.open(path, flags | os.O_DIRECT, 0o644), True
        except OSError:
            pass
    return os.open(path, flags, 0o644), False


def drop_cache(fd):
    if hasattr(os, "posix_fadvise"):
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def direct_works(fd):
    region, _ = aligned(ALIGN)
    try:
        pread_into(fd, memoryview(region)[:ALIGN], 0)
    except OSError:
        return False
    return True


def fs_type(path):
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    target = os.path.realpath(path)
    best, kind = "", ""
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        point = parts[1].replace("\\040", " ")
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) > len(best):
            best, kind = point, parts[2]
    return kind


def cgroup_memory():
    seen = {"current": None, "file": None, "dirty": None}
    try:
        seen["current"] = int(Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8").strip()) / 2**30
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text(encoding="utf-8").splitlines():
            name, _, count = line.partition(" ")
            if name in ("file", "file_dirty") and count.strip().isdigit():
                seen["file" if name == "file" else "dirty"] = int(count) / 2**30
    except OSError:
        pass
    return seen


def cgroup_file_gib():
    return cgroup_memory()["file"]


def gib_text(value):
    return "unknown" if value is None else f"{value:.1f} GiB"


def memory_text():
    seen = cgroup_memory()
    return (f"cgroup memory current {gib_text(seen['current'])} file {gib_text(seen['file'])} "
            f"dirty {gib_text(seen['dirty'])}")


def process_storage_written():
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            name, _, count = line.partition(":")
            if name == "write_bytes" and count.strip().isdigit():
                return int(count)
    except OSError:
        return None
    return None


class ScratchWrites:
    def __init__(self):
        self.total = 0
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            self.total += int(count)

    def text(self):
        kernel = process_storage_written()
        return (f"this process wrote {self.total / 2**30:.2f} GiB to the flat file"
                f"{'' if kernel is None else f' ({kernel / 2**30:.2f} GiB of storage writes in /proc/self/io)'}")


SCRATCH = ScratchWrites()


def flat_dir(need, said):
    reasons = []
    for candidate in ([said] if said else list(DIR_CANDIDATES)):
        path = Path(candidate) / "taaf_ple_hot"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            reasons.append(f"{candidate}: {exc}")
            continue
        kind = fs_type(path)
        if kind in UNFIT_FS:
            reasons.append(f"{candidate} is {kind}")
            continue
        free = shutil.disk_usage(path).free
        if free < need + 2**30:
            reasons.append(f"{candidate} has {free / 2**30:.1f} GiB free")
            continue
        return path, kind
    raise RuntimeError(f"PLE_HOT flat: no local disk holds the {need / 2**30:.1f} GiB PLE file ({'; '.join(reasons)}); "
                       f"name one with {DIR_ENV}, or choose {HOT_ENV}=mount explicitly")


class RowIndex:
    def __init__(self, slots):
        self.bits = max(10, (2 * int(slots) - 1).bit_length())
        self.size = 1 << self.bits
        self.mask = self.size - 1
        self.table = np.full(self.size, EMPTY, dtype=np.int64)
        self.live = 0
        self.dead = 0
        self.rebuilds = 0

    def _home(self, keys):
        return ((keys.astype(np.uint64) * GOLDEN) >> np.uint64(64 - self.bits)).astype(np.int64)

    def _where(self, keys):
        pos = self._home(keys)
        found = np.full(keys.shape[0], -1, dtype=np.int64)
        todo = np.arange(keys.shape[0])
        while todo.size:
            at = pos[todo]
            held = self.table[at]
            hit = (held >= 0) & ((held & KEY_MASK) == keys[todo])
            found[todo[hit]] = at[hit]
            go = ~hit & (held != EMPTY)
            todo = todo[go]
            pos[todo] = (at[go] + 1) & self.mask
        return found

    def find(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        at = self._where(keys)
        return np.where(at >= 0, self.table[np.maximum(at, 0)] >> 32, -1)

    def insert(self, keys, slots):
        keys = np.asarray(keys, dtype=np.int64)
        packed = (np.asarray(slots, dtype=np.int64) << 32) | keys
        pos = self._home(keys)
        placed = np.zeros(keys.shape[0], dtype=bool)
        todo = np.arange(keys.shape[0])
        while todo.size:
            at = pos[todo]
            free = self.table[at] < 0
            if free.any():
                spots, first = np.unique(at[free], return_index=True)
                winners = todo[free][first]
                self.dead -= int(np.count_nonzero(self.table[spots] == TOMB))
                self.table[spots] = packed[winners]
                placed[winners] = True
            todo = todo[~placed[todo]]
            pos[todo] = (pos[todo] + 1) & self.mask
        self.live += keys.shape[0]
        if self.live + self.dead > 0.7 * self.size:
            self.rebuild()

    def remove(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        at = self._where(keys)
        if (at < 0).any():
            raise RuntimeError(f"PLE_HOT index lost {int(np.count_nonzero(at < 0))} of {keys.shape[0]} rows it was asked to drop")
        self.table[at] = TOMB
        self.live -= keys.shape[0]
        self.dead += keys.shape[0]

    def rebuild(self):
        held = self.table[self.table >= 0]
        self.table.fill(EMPTY)
        self.live = 0
        self.dead = 0
        self.rebuilds += 1
        if held.size:
            self.insert(held & KEY_MASK, held >> 32)


class RowCache:
    def __init__(self, store, slots):
        self.store = store
        self.slots = int(slots)
        self.zero = self.slots
        self.row_bytes = store.shape[1]
        self.row_of = np.full(self.slots, -1, dtype=np.int32)
        self.ref = np.zeros(self.slots, dtype=np.uint8)
        self.stamp = np.full(self.slots, -1, dtype=np.int32)
        self.hand = 0
        self.filled = 0
        self.evictions = 0
        self.index = RowIndex(self.slots)
        store[self.zero] = 0

    def lookup(self, rows, step):
        slots = self.index.find(rows)
        hit = slots >= 0
        held = slots[hit]
        self.ref[held] = 1
        self.stamp[held] = step
        missing = rows[~hit]
        if not missing.size:
            return slots, hit, missing, missing
        fresh = self._claim(missing.shape[0], step)
        self.row_of[fresh] = missing
        self.ref[fresh] = 1
        self.stamp[fresh] = step
        self.index.insert(missing, fresh)
        slots[~hit] = fresh
        return slots, hit, missing, fresh

    def forget(self, slots):
        rows = self.row_of[slots].astype(np.int64)
        kept = rows >= 0
        self.index.remove(rows[kept])
        self.row_of[slots] = -1
        self.ref[slots] = 0
        self.stamp[slots] = -1

    def _claim(self, count, step):
        parts = []
        take = min(count, self.slots - self.filled)
        if take:
            parts.append(np.arange(self.filled, self.filled + take, dtype=np.int64))
            self.stamp[parts[0]] = step
            self.filled += take
        need = count - take
        swept = 0
        victims = []
        while need:
            width = min(self.slots, max(4 * need, 65536))
            window = (np.arange(width, dtype=np.int64) + self.hand) % self.slots
            usable = (self.ref[window] == 0) & (self.stamp[window] < step)
            pick = np.flatnonzero(usable)[:need]
            end = int(pick[-1]) + 1 if pick.size == need else width
            self.ref[window[:end]] = 0
            self.stamp[window[pick]] = step
            victims.append(window[pick])
            need -= pick.size
            self.hand = (self.hand + end) % self.slots
            swept += end
            if need and swept > 2 * self.slots + width:
                raise RuntimeError(f"PLE_HOT: one step needs {count} new rows but the {self.slots}-row cache "
                                   f"holds too many rows this step already uses; raise {GIB_ENV}")
        if victims:
            gone = np.concatenate(victims)
            rows = self.row_of[gone].astype(np.int64)
            self.index.remove(rows[rows >= 0])
            self.row_of[gone] = -1
            self.evictions += gone.shape[0]
            parts.append(gone)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]


NIBBLES = np.stack([np.arange(256, dtype=np.uint8) & 15, np.arange(256, dtype=np.uint8) >> 4], axis=1)


class RowCodec:
    def __init__(self, width):
        if int(width) % 2:
            raise ValueError(f"the row codec needs an even number of BF16 values, found {width}")
        self.width = int(width)
        self.half = self.width // 2
        self.row_bytes = 2 * self.width
        self.record = 1 + self.half + self.width

    def encode(self, data, first):
        count = data.shape[0]
        words = torch.from_numpy(np.ascontiguousarray(data)).view(torch.int16)
        exp = (words >> 7) & 255
        sign_mantissa = (((words >> 8) & 128) | (words & 127)).to(torch.uint8)
        base = (exp.amax(dim=1) - (EXP_WINDOW - 1)).clamp_(min=0)
        code = exp - base[:, None] + 1
        outside = code < 1
        code.masked_fill_(outside, 0)
        record = torch.empty((count, self.record), dtype=torch.uint8)
        record[:, 0] = base.to(torch.uint8)
        record[:, 1: 1 + self.half] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        record[:, 1 + self.half:] = sign_mantissa
        at = outside.nonzero()
        keys = (at[:, 0] + int(first)) * self.width + at[:, 1]
        return record.numpy(), keys.numpy(), exp[outside].to(torch.uint8).numpy()

    def decode(self, record, rows, escapes):
        count = record.shape[0]
        code = NIBBLES[record[:, 1: 1 + self.half]].reshape(count, self.width)
        exp = code.astype(np.uint16) + (record[:, :1].astype(np.uint16) - np.uint16(1))
        escaped = code == 0
        if escaped.any():
            where_row, where_col = np.nonzero(escaped)
            exp[where_row, where_col] = escapes.lookup(rows[where_row] * self.width + where_col)
        low = record[:, 1 + self.half:].astype(np.uint16)
        word = ((low & np.uint16(128)) << np.uint16(8)) | (exp << np.uint16(7)) | (low & np.uint16(127))
        return word.astype("<u2", copy=False).view(np.uint8).reshape(count, self.row_bytes)


class EscapeLog:
    def __init__(self):
        self.parts = []
        self.keys = np.zeros(0, dtype=np.int64)
        self.exps = np.zeros(0, dtype=np.uint8)
        self.count = 0

    def add(self, keys, exps):
        if keys.shape[0]:
            self.parts.append((keys, exps))
            self.count += int(keys.shape[0])

    def seal(self):
        if self.parts:
            keys = np.concatenate([one[0] for one in self.parts] + [self.keys])
            exps = np.concatenate([one[1] for one in self.parts] + [self.exps])
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.exps = exps[order]
            self.parts = []

    def nbytes(self):
        return int(self.count) * ESCAPE_BYTES

    def lookup(self, keys):
        if not self.keys.shape[0]:
            raise RuntimeError("PLE_HOT: a row holds an escaped exponent but the escape table is empty")
        at = np.minimum(np.searchsorted(self.keys, keys), self.keys.shape[0] - 1)
        if not (self.keys[at] == keys).all():
            raise RuntimeError("PLE_HOT: an escaped exponent is missing from the escape table")
        return self.exps[at]


class Tiers:
    def __init__(self, rows, record, disk_bytes, hot=None):
        self.rows = int(rows)
        self.record = int(record)
        self.disk_rows = min(self.rows, int(disk_bytes) // ALIGN * ALIGN // self.record)
        self.ram_rows = self.rows - self.disk_rows
        self.disk_file_bytes = ceil_to(self.disk_rows * self.record, ALIGN) if self.disk_rows else 0
        extra = np.zeros(0, dtype=np.int64)
        if hot is not None and self.ram_rows:
            extra = np.unique(np.asarray(hot, dtype=np.int64).reshape(-1))
            extra = extra[(extra >= 0) & (extra < self.rows)][: self.ram_rows]
        low, high = 0, self.rows
        while low < high:
            mid = (low + high) // 2
            if self.rows - mid + int(np.searchsorted(extra, mid)) <= self.ram_rows:
                high = mid
            else:
                low = mid + 1
        self.tail = low
        self.extra = extra[: int(np.searchsorted(extra, low))]
        self.listed = int(self.extra.shape[0])
        if self.listed + self.rows - self.tail != self.ram_rows:
            raise RuntimeError(f"PLE_HOT: the RAM tier plan holds {self.listed + self.rows - self.tail} rows, "
                               f"not the {self.ram_rows} it should")

    def route(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        below = np.searchsorted(self.extra, rows)
        listed = np.zeros(rows.shape[0], dtype=bool)
        if self.listed:
            listed = (below < self.listed) & (self.extra[np.minimum(below, self.listed - 1)] == rows)
        tail = rows >= self.tail
        rank = np.where(tail, self.listed + rows - self.tail, np.where(listed, below, rows - below))
        return tail | listed, rank

    def row_of_ram(self, rank):
        rank = np.asarray(rank, dtype=np.int64)
        held = self.extra[np.minimum(rank, self.listed - 1)] if self.listed else rank
        return np.where(rank < self.listed, held, self.tail + rank - self.listed)

    def row_of_disk(self, rank):
        rank = np.asarray(rank, dtype=np.int64)
        return rank + np.searchsorted(self.extra - np.arange(self.listed, dtype=np.int64), rank, side="right")


class DecodingStore:
    def __init__(self, store, codec, escapes, row_of):
        self.store = store
        self.codec = codec
        self.escapes = escapes
        self.row_of = row_of

    def __setitem__(self, slots, record):
        slots = np.asarray(slots).reshape(-1)
        record = np.asarray(record).reshape(-1, self.codec.record)
        self.store[slots] = self.codec.decode(record, self.row_of[slots].astype(np.int64), self.escapes)


class Reader:
    def __init__(self, files, row_bytes, direct):
        self.files = [str(one) for one in files]
        self.row_bytes = int(row_bytes)
        self.local = threading.local()
        self.direct = False
        if direct:
            fd, self.direct = open_file(self.files[0], os.O_RDONLY, True)
            self.direct = self.direct and direct_works(fd)
            os.close(fd)

    def _fd(self, fid):
        fds = getattr(self.local, "fds", None)
        if fds is None:
            fds = self.local.fds = {}
        fd = fds.get(fid)
        if fd is None:
            fd, _ = open_file(self.files[fid], os.O_RDONLY, self.direct)
            if not self.direct and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            fds[fid] = fd
        return fd

    def drop(self):
        if self.direct or not hasattr(os, "posix_fadvise"):
            return
        for path in self.files:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def _bounce(self, size):
        held = getattr(self.local, "bounce", None)
        if held is None or held[1].shape[0] < size:
            held = self.local.bounce = aligned(ceil_to(max(size, 1 << 16), ALIGN))
        return held

    def plan(self, fids, offsets):
        order = np.lexsort((offsets, fids))
        fids = fids[order]
        offsets = offsets[order]
        start = offsets // ALIGN * ALIGN
        stop = (offsets + self.row_bytes + ALIGN - 1) // ALIGN * ALIGN
        cut = np.ones(order.shape[0], dtype=bool)
        cut[1:] = (fids[1:] != fids[:-1]) | (start[1:] > stop[:-1])
        begins = np.flatnonzero(cut)
        ends = np.append(begins[1:], order.shape[0])
        return order, fids, offsets, start, stop, begins, ends

    def read(self, plan, slots, store, first, last, latency):
        order, fids, offsets, start, stop, begins, ends = plan
        rb = self.row_bytes
        picked = []
        held = []
        for g in range(first, last):
            b = int(begins[g])
            e = int(ends[g])
            low = int(start[b])
            high = int(stop[e - 1])
            region, bounce = self._bounce(high - low)
            fd = self._fd(int(fids[b]))
            began = time.perf_counter()
            got = pread_into(fd, memoryview(region)[: high - low], low)
            latency[g] = time.perf_counter() - began
            need = int(offsets[e - 1]) + rb - low
            if got < need:
                raise OSError(f"PLE_HOT read {got} of {need} bytes at {low} of {self.files[int(fids[b])]}")
            if e - b == 1:
                rel = int(offsets[b]) - low
                held.append(bounce[rel: rel + rb].reshape(1, rb).copy())
            else:
                rel = offsets[b:e] - low
                held.append(bounce[rel[:, None] + np.arange(rb)])
            picked.append(slots[order[b:e]])
        if picked:
            store[np.concatenate(picked)] = np.concatenate(held)


def pread_fetch(source, rows, slots, store, pool, threads):
    fids, offsets = source.locate(rows)
    plan = source.reader.plan(fids, offsets)
    _, _, _, start, stop, begins, ends = plan
    groups = begins.shape[0]
    latency = np.zeros(groups, dtype=np.float64)
    tasks = min(threads, groups)
    cuts = np.linspace(0, groups, tasks + 1).astype(np.int64)
    if tasks == 1:
        source.reader.read(plan, slots, store, 0, groups, latency)
    else:
        futures = [pool.submit(source.reader.read, plan, slots, store, int(cuts[i]), int(cuts[i + 1]), latency)
                   for i in range(tasks)]
        for future in futures:
            future.result()
    return groups, int((stop[ends - 1] - start[begins]).sum()), latency


class FlatSource:
    def __init__(self, path, rows, row_bytes, kind="mmap"):
        self.path = str(path)
        self.rows = int(rows)
        self.row_bytes = int(row_bytes)
        self.kind = "flat"
        self.read = kind
        self.map = None
        self.reader = Reader([self.path], row_bytes, kind == "pread")
        self.direct = self.reader.direct
        if kind == "mmap":
            self.handle = open(self.path, "rb")
            self.map = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
            if hasattr(self.map, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                self.map.madvise(mmap.MADV_RANDOM)
            self.table = np.frombuffer(self.map, dtype=np.uint8)[: self.rows * self.row_bytes].reshape(
                self.rows, self.row_bytes)

    def locate(self, rows):
        return np.zeros(rows.shape[0], dtype=np.int64), rows.astype(np.int64) * self.row_bytes

    def _gather(self, rows, slots, store, first, last, latency, index):
        began = time.perf_counter()
        store[slots[first:last]] = self.table[rows[first:last]]
        latency[index] = (time.perf_counter() - began) / max(1, last - first)

    def fetch(self, rows, slots, store, pool, threads):
        if self.map is None:
            return pread_fetch(self, rows, slots, store, pool, threads)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        slots = slots[order]
        count = rows.shape[0]
        tasks = min(threads, count)
        cuts = np.linspace(0, count, tasks + 1).astype(np.int64)
        latency = np.zeros(tasks, dtype=np.float64)
        if tasks == 1:
            self._gather(rows, slots, store, 0, count, latency, 0)
        else:
            futures = [pool.submit(self._gather, rows, slots, store, int(cuts[i]), int(cuts[i + 1]), latency, i)
                       for i in range(tasks)]
            for future in futures:
                future.result()
        begin = rows * self.row_bytes
        pages = 1 + (begin // ALIGN != (begin + self.row_bytes - 1) // ALIGN)
        return count, int(pages.sum()) * ALIGN, latency

    def drop(self):
        if self.map is not None and hasattr(self.map, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            self.map.madvise(mmap.MADV_DONTNEED)
        self.reader.drop()

    def close(self):
        if self.map is not None:
            self.table = None
            self.map.close()
            self.handle.close()
            self.map = None


class DummyFlat(FlatSource):
    def __init__(self, path, rows, row_bytes, kind="mmap"):
        super().__init__(path, rows, row_bytes, kind)

    def fetch(self, rows, slots, store, pool, threads):
        return super().fetch(np.asarray(rows, dtype=np.int64) % self.rows, slots, store, pool, threads)


class MountSource:
    def __init__(self, model_dir, tail, rows, row_bytes, dtype, parts):
        self.rows = int(rows)
        self.row_bytes = int(row_bytes)
        self.shard_size = (self.rows + int(parts) - 1) // int(parts)
        index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text(encoding="utf-8"))
        wanted = {}
        for name, file in index["weight_map"].items():
            head, sep, rest = name.partition(tail + ".shard_")
            number = rest[: -len(".weight")]
            if sep and rest.endswith(".weight") and number.isdigit() and (not head or head.endswith(".")):
                wanted.setdefault(int(number), []).append((name, file))
        files = []
        bases = np.full(int(parts), -1, dtype=np.int64)
        fid_of = np.full(int(parts), -1, dtype=np.int64)
        headers = {}
        for shard in range(int(parts)):
            first = shard * self.shard_size
            expect = max(0, min(self.shard_size, self.rows - first))
            names = wanted.get(shard, [])
            if expect == 0:
                continue
            if len(names) != 1:
                raise RuntimeError(f"PLE_HOT mount: shard {shard} of {tail} names {len(names)} checkpoint tensors, not one")
            name, file = names[0]
            if file not in headers:
                headers[file] = self._header(Path(model_dir) / file)
                files.append(str(Path(model_dir) / file))
            start, meta = headers[file]
            kind = SAFETENSORS_DTYPES.get(meta[name]["dtype"])
            if kind != dtype:
                raise RuntimeError(f"PLE_HOT mount: {name} is {meta[name]['dtype']}, the table is {dtype}; "
                                   f"only the flat file converts")
            if list(meta[name]["shape"]) != [expect, row_bytes // torch.empty((), dtype=dtype, device="cpu").element_size()]:
                raise RuntimeError(f"PLE_HOT mount: {name} has shape {meta[name]['shape']}, expected {expect} rows")
            bases[shard] = start + int(meta[name]["data_offsets"][0])
            fid_of[shard] = files.index(str(Path(model_dir) / file))
        self.bases = bases
        self.fid_of = fid_of
        self.reader = Reader(files, row_bytes, True)
        self.direct = self.reader.direct
        self.path = str(model_dir)
        self.kind = "mount"

    @staticmethod
    def _header(path):
        with open(path, "rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            return 8 + size, json.loads(handle.read(size))

    def locate(self, rows):
        rows = rows.astype(np.int64)
        shard = rows // self.shard_size
        return self.fid_of[shard], self.bases[shard] + (rows - shard * self.shard_size) * self.row_bytes

    def fetch(self, rows, slots, store, pool, threads):
        return pread_fetch(self, rows, slots, store, pool, threads)

    def drop(self):
        self.reader.drop()

    def close(self):
        return None


class TierSource:
    def __init__(self, flat, tiers, ram, codec, escapes, row_of):
        self.flat = flat
        self.tiers = tiers
        self.ram = ram
        self.codec = codec
        self.escapes = escapes
        self.row_of = row_of
        self.kind = "flat"
        self.read = flat.read if flat is not None else "ram"
        self.direct = flat.direct if flat is not None else False
        self.path = flat.path if flat is not None else ""
        self.ram_served = 0
        self.disk_served = 0
        self.next_tell = 0

    def _from_ram(self, sink, ranks, slots):
        sink[slots] = self.ram[ranks]

    def fetch(self, rows, slots, store, pool, threads):
        began = time.perf_counter()
        sink = DecodingStore(store, self.codec, self.escapes, self.row_of)
        in_ram, rank = self.tiers.route(rows)
        from_ram = np.flatnonzero(in_ram)
        from_disk = np.flatnonzero(~in_ram)
        futures = []
        if from_ram.shape[0]:
            tasks = max(1, min(int(threads), from_ram.shape[0] // 256))
            cuts = np.linspace(0, from_ram.shape[0], tasks + 1).astype(np.int64)
            futures = [pool.submit(self._from_ram, sink, rank[from_ram[cuts[i]: cuts[i + 1]]],
                                   slots[from_ram[cuts[i]: cuts[i + 1]]]) for i in range(tasks)]
        reads, nbytes, latency = 0, 0, None
        if from_disk.shape[0]:
            reads, nbytes, latency = self.flat.fetch(rank[from_disk], slots[from_disk], sink, pool, threads)
        for future in futures:
            future.result()
        self.ram_served += int(from_ram.shape[0])
        self.disk_served += int(from_disk.shape[0])
        if self.ram_served + self.disk_served >= self.next_tell:
            self.next_tell = self.ram_served + self.disk_served + TELL_EVERY
            say(f"tiers: misses served so far {self.ram_served:,} rows from the RAM tier and {self.disk_served:,} "
                f"from the disk tier; last batch {int(from_ram.shape[0]):,} + {int(from_disk.shape[0]):,} rows decoded "
                f"in {(time.perf_counter() - began) * 1e3:.2f} ms")
        return reads, nbytes, latency

    def drop(self):
        if self.flat is not None:
            self.flat.drop()

    def close(self):
        if self.flat is not None:
            self.flat.close()
        self.ram = None


class FlatWriter:
    def __init__(self, path, size, pool_bytes, limit=None, label=""):
        self.path = str(path)
        self.size = int(size)
        self.padded = ceil_to(self.size, ALIGN)
        self.limit = None if limit is None else int(limit)
        self.label = str(label)
        if self.limit is not None and self.padded > self.limit:
            raise RuntimeError(f"PLE_HOT flat file {self.path} would be {self.padded / 2**30:.2f} GiB, over the "
                               f"{self.limit / 2**30:.2f} GiB {DISK_ENV} budget; nothing was written")
        self.verify = os.path.isfile(self.path) and os.path.getsize(self.path) == self.padded
        flags = os.O_RDWR | os.O_CREAT | (0 if self.verify else os.O_TRUNC)
        self.fd, self.direct = open_file(self.path, flags, True)
        if not self.verify:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(self.fd, 0, self.padded)
            else:
                os.ftruncate(self.fd, self.padded)
        if self.direct and not direct_works(self.fd):
            os.close(self.fd)
            self.fd, self.direct = open_file(self.path, os.O_RDWR, False)
        self.free = queue.Queue()
        for _ in range(max(1, int(pool_bytes) // CHUNK)):
            self.free.put(aligned(CHUNK))
        self.scratch = aligned(CHUNK) if self.verify else None
        self.work = queue.Queue()
        self.edges = {}
        self.lock = threading.Lock()
        self.error = None
        self.wrote = 0
        self.kept = 0
        self.reported = 0
        self.began = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name="ple-hot-writer", daemon=True)
        self.thread.start()

    def _raise(self):
        if self.error is not None:
            raise RuntimeError(f"PLE_HOT flat file {self.path} failed: {self.error!r}") from self.error

    def put(self, row_start, data, row_bytes):
        self._raise()
        flat = data.reshape(-1)
        low = int(row_start) * int(row_bytes)
        high = low + flat.shape[0]
        head = min(high, ceil_to(low, ALIGN))
        tail = max(head, high // ALIGN * ALIGN)
        if low % ALIGN:
            self._edge(low, flat[: head - low])
        at = head
        while at < tail:
            n = min(CHUNK, tail - at)
            region, view = self.free.get()
            self._raise()
            view[:n] = flat[at - low: at - low + n]
            self.work.put((at, region, view, n, True))
            at += n
        if tail < high:
            self._edge(tail, flat[tail - low:])

    def _edge(self, offset, piece):
        block = offset // ALIGN
        need = min(ALIGN, self.size - block * ALIGN)
        with self.lock:
            held = self.edges.get(block)
            if held is None:
                held = self.edges[block] = [aligned(ALIGN), 0]
            region, view = held[0]
            rel = offset - block * ALIGN
            view[rel: rel + piece.shape[0]] = piece
            held[1] += piece.shape[0]
            if held[1] < need:
                return
            del self.edges[block]
        self.work.put((block * ALIGN, region, view, ALIGN, False))

    def _same(self, offset, n):
        region, view = self.scratch
        got = pread_into(self.fd, memoryview(region)[:n], offset)
        return got == n, view

    def _run(self):
        since = 0
        while True:
            item = self.work.get()
            if item is None:
                return
            offset, region, view, n, pooled = item
            try:
                if self.error is None:
                    if self.verify:
                        full, old = self._same(offset, n)
                        if full and np.array_equal(old[:n], view[:n]):
                            self.kept += n
                            self._progress()
                            continue
                    if self.limit is not None and self.wrote + n > self.limit:
                        raise RuntimeError(f"PLE_HOT flat file {self.path}: a write of {n} bytes would take this file "
                                           f"past its {self.limit / 2**30:.2f} GiB {DISK_ENV} budget")
                    pwrite_all(self.fd, memoryview(region)[:n], offset)
                    self.wrote += n
                    SCRATCH.add(n)
                    self._progress()
                    since += n
                    if not self.direct and since >= FLUSH:
                        drop_cache(self.fd)
                        since = 0
            except BaseException as exc:
                self.error = exc
            finally:
                if pooled:
                    self.free.put((region, view))

    def _progress(self):
        done = self.wrote + self.kept
        if done < self.reported + PROGRESS_EVERY:
            return
        self.reported = done // PROGRESS_EVERY * PROGRESS_EVERY
        seconds = time.perf_counter() - self.began
        say(f"{self.label}: flat file {done / 2**30:.1f} of {self.size / 2**30:.1f} GiB in {seconds:.1f} s "
            f"({done / max(seconds, 1e-9) / 1e9:.2f} GB/s); {memory_text()}; {SCRATCH.text()}")

    def close(self):
        self.work.put(None)
        self.thread.join()
        self._raise()
        if self.edges:
            raise RuntimeError(f"PLE_HOT flat file {self.path}: {len(self.edges)} edge block(s) never received all their rows")
        os.fsync(self.fd)
        if not self.direct:
            drop_cache(self.fd)
        os.close(self.fd)
        return time.perf_counter() - self.began


class Tally:
    def __init__(self, every, label):
        self.every = int(every)
        self.label = label
        self.reset(1)

    def reset(self, first):
        self.first = first
        self.steps = 0
        self.lookups = 0
        self.distinct = 0
        self.missed_lookups = 0
        self.misses = 0
        self.reads = 0
        self.bytes = 0
        self.latency = []
        self.stall = []
        self.job = []
        self.ids_wait = []

    def add(self, step, stats, stall, cache):
        self.steps += 1
        self.lookups += stats["lookups"]
        self.distinct += stats["distinct"]
        self.missed_lookups += stats["missed_lookups"]
        self.misses += stats["misses"]
        self.reads += stats["reads"]
        self.bytes += stats["bytes"]
        if stats["latency"] is not None:
            self.latency.append(stats["latency"])
        self.stall.append(stall)
        self.job.append(stats["job"])
        self.ids_wait.append(stats["ids_wait"])
        if self.steps >= self.every:
            self.emit(step, cache)

    def emit(self, step, cache):
        if not self.steps:
            return
        lat = np.concatenate(self.latency) * 1e3 if self.latency else np.zeros(1)
        stall = np.asarray(self.stall) * 1e3
        job = np.asarray(self.job) * 1e3
        wait = np.asarray(self.ids_wait) * 1e3
        hits = 100.0 * (1 - self.missed_lookups / max(1, self.lookups))
        say(f"{self.label} steps {self.first}-{step}: lookups {self.lookups:,} distinct {self.distinct:,} "
            f"hits {hits:.2f}% misses {self.misses:,} rows ({self.missed_lookups:,} lookups) reads {self.reads:,} "
            f"bytes read {self.bytes / 2**20:.1f} MiB; miss latency p50 {np.percentile(lat, 50):.3f} ms "
            f"p95 {np.percentile(lat, 95):.3f} ms; stall per step mean {stall.mean():.3f} ms "
            f"p95 {np.percentile(stall, 95):.3f} ms max {stall.max():.3f} ms; job p50 {np.percentile(job, 50):.3f} ms "
            f"p95 {np.percentile(job, 95):.3f} ms; ids wait p50 {np.percentile(wait, 50):.3f} ms; cache "
            f"{cache.filled * cache.row_bytes / 2**30:.2f}/{cache.slots * cache.row_bytes / 2**30:.2f} GiB "
            f"filled, evictions {cache.evictions:,}, index rebuilds {cache.index.rebuilds}")
        self.reset(step + 1)


class Engine:
    def __init__(self, cache, source, rows, host_ids, host_slots, threads, every, label):
        self.cache = cache
        self.source = source
        self.rows = int(rows)
        self.host_ids = host_ids
        self.host_slots = host_slots
        self.ids = host_ids.numpy()
        self.out = host_slots.numpy()
        self.threads = int(threads)
        self.pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="ple-hot-read")
        self.runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-hot-job")
        self.tally = Tally(every, label)
        self.pending = None
        self.step = 0
        self.unflushed = 0

    def fetch(self, rows, slots):
        if not rows.size:
            return 0, 0, None
        reads, nbytes, latency = self.source.fetch(rows, slots, self.cache.store, self.pool, self.threads)
        self.unflushed += nbytes
        if self.unflushed >= DROP_EVERY:
            self.source.drop()
            self.unflushed = 0
        return reads, nbytes, latency

    def resolve(self, n, step):
        ids = self.ids[:n]
        uniq, inverse = np.unique(ids, return_inverse=True)
        valid = (uniq >= 0) & (uniq < self.rows)
        slots = np.full(uniq.shape[0], self.cache.zero, dtype=np.int64)
        found, hit, missing, fresh = self.cache.lookup(uniq[valid], step)
        try:
            reads, nbytes, latency = self.fetch(missing, fresh)
        except BaseException:
            self.cache.forget(fresh)
            raise
        slots[valid] = found
        self.out[:n] = slots[inverse]
        missed_lookups = 0
        if missing.size:
            missed = np.zeros(uniq.shape[0], dtype=bool)
            missed[np.flatnonzero(valid)[~hit]] = True
            missed_lookups = int(np.count_nonzero(missed[inverse]))
        return {"lookups": int(n), "distinct": int(uniq.shape[0]), "misses": int(missing.shape[0]),
                "missed_lookups": missed_lookups, "reads": reads, "bytes": nbytes, "latency": latency}

    def resolve_now(self, n):
        if self.pending is not None:
            raise RuntimeError("PLE_HOT: a pre-resolve started while a prefetch was never finished")
        if n > self.ids.shape[0]:
            raise RuntimeError(f"PLE_HOT: {n} lookups exceed the {self.ids.shape[0]}-entry staging buffer")
        self.step += 1
        began = time.perf_counter()
        stats = self.resolve(n, self.step)
        stats["ids_wait"] = 0.0
        stats["job"] = time.perf_counter() - began
        self.tally.add(self.step, stats, 0.0, self.cache)
        return stats

    def _job(self, n, wait, launch, step):
        began = time.perf_counter()
        if wait is not None:
            wait()
        ready = time.perf_counter()
        stats = self.resolve(n, step)
        if launch is not None:
            launch()
        stats["ids_wait"] = ready - began
        stats["job"] = time.perf_counter() - ready
        return stats

    def submit(self, n, wait=None, launch=None, sync=False):
        if self.pending is not None:
            raise RuntimeError("PLE_HOT: a prefetch started while the previous one was never finished")
        if n > self.ids.shape[0]:
            raise RuntimeError(f"PLE_HOT: {n} lookups exceed the {self.ids.shape[0]}-entry staging buffer")
        self.step += 1
        if sync:
            stats = self._job(n, wait, launch, self.step)
            self.pending = (None, stats)
        else:
            self.pending = (self.runner.submit(self._job, n, wait, launch, self.step), None)

    def wait(self):
        if self.pending is None:
            raise RuntimeError("PLE_HOT: a lookup was finished that never started")
        future, stats = self.pending
        began = time.perf_counter()
        self.pending = None
        if future is not None:
            stats = future.result()
        stall = time.perf_counter() - began
        self.tally.add(self.step, stats, stall, self.cache)
        return stall


class Store:
    def __init__(self, *, rows, width, dtype, label, model_dir, tail, parts, config, pinned):
        if config.get("dummy") and config["mode"] == "mount":
            raise RuntimeError(f"{DUMMY_ENV} fills the flat PLE tiers; {HOT_ENV}=mount has no checkpoint to read")
        self.rows = int(rows)
        self.width = int(width)
        self.dtype = dtype
        self.row_bytes = self.width * torch.empty((), dtype=dtype, device="cpu").element_size()
        self.label = label
        self.config = config
        self.slots = slots_for(config["gib"], self.row_bytes, self.rows)
        size = (self.slots + 1) * self.row_bytes
        self.region, flat = aligned(size)
        self.flat = torch.from_numpy(flat)
        if pinned:
            result = torch.cuda.cudart().cudaHostRegister(self.flat.data_ptr(), size, 0)
            if result.value != 0:
                raise RuntimeError(f"PLE_HOT: cudaHostRegister of the {size / 2**30:.2f} GiB row cache failed: {result}")
            if not self.flat.is_pinned():
                raise RuntimeError("PLE_HOT: CUDA does not report the registered row cache as pinned")
        self.weight = self.flat.view(dtype).view(self.slots + 1, self.width)
        self.cache = RowCache(self.flat.numpy().reshape(self.slots + 1, self.row_bytes), self.slots)
        self.lock = threading.Lock()
        self.seen = {}
        self.engine = None
        self.source = None
        self.writer = None
        self.build = None
        self.built = False
        self.prewarm = self._prewarm_rows(config["prewarm"])
        self.prewarmed = 0
        self.tiers = None
        self.ram = None
        self.codec = None
        self.escapes = None
        if config["mode"] == "mount":
            self.source = MountSource(model_dir, tail, self.rows, self.row_bytes, dtype, parts)
            where = f"pread from the mounted checkpoint {model_dir} ({len(self.source.reader.files)} file(s))"
            held = f"{self.rows:,} rows x {self.row_bytes} B = {self.rows * self.row_bytes / 2**30:.2f} GiB stay off host RAM"
        else:
            if dtype != torch.bfloat16:
                raise RuntimeError(f"PLE_HOT flat stores rows with the BF16 exponent codec; this table is {dtype}")
            self.codec = RowCodec(self.width)
            self.escapes = EscapeLog()
            budget = int(config["disk_gib"] * 2**30)
            self.tiers = Tiers(self.rows, self.codec.record, budget, self.prewarm)
            self.ram = np.empty((self.tiers.ram_rows, self.codec.record), dtype=np.uint8)
            raw = self.rows * self.row_bytes
            packed = self.rows * self.codec.record
            held = (f"{self.rows:,} rows x {self.row_bytes} B = {raw / 2**30:.2f} GiB raw, {packed / 2**30:.2f} GiB "
                    f"packed ({self.codec.record} B a row, ratio {self.codec.record / self.row_bytes:.4f} before "
                    f"escapes); RAM tier {self.tiers.ram_rows * self.codec.record / 2**30:.2f} GiB pageable "
                    f"({self.tiers.ram_rows:,} rows: {self.tiers.listed:,} listed + {self.rows - self.tiers.tail:,} "
                    f"from row {self.tiers.tail:,} up); disk file {self.tiers.disk_file_bytes / 2**30:.2f} GiB "
                    f"({self.tiers.disk_rows:,} rows) under the {config['disk_gib']:g} GiB {DISK_ENV} budget")
            if config.get("dummy"):
                where = self._dummy_tiers()
            elif self.tiers.disk_rows:
                directory, kind = flat_dir(self.tiers.disk_file_bytes, config["dir"])
                digest = hashlib.sha1(f"{model_dir}|{label}|{dtype}|{self.codec.record}".encode("utf-8")).hexdigest()[:12]
                self.path = directory / f"ple_{digest}_{self.tiers.disk_rows}x{self.codec.record}.bin"
                self.writer = FlatWriter(self.path, self.tiers.disk_rows * self.codec.record,
                                         config["write_mib"] * 2**20, limit=budget, label=label)
                where = (f"a flat file {self.path} on {kind or 'an unknown fs'} "
                         f"({'verifying the existing file' if self.writer.verify else 'written as the shards load'}, "
                         f"O_DIRECT {'on' if self.writer.direct else 'off, page cache dropped every GiB'}) "
                         f"and the RAM tier")
            else:
                where = "the RAM tier"
        say(f"{label}: {held}; row cache {self.slots:,} slots = {size / 2**30:.2f} GiB "
            f"{'pinned' if pinned else 'unpinned'} + index {index_bytes(self.slots) / 2**30:.2f} GiB; misses read "
            f"from {where}; {config['threads']} read threads; tally every {config['every']} steps; "
            f"prewarm {self.prewarm.shape[0]:,} rows")

    def _dummy_tiers(self):
        began = time.perf_counter()
        base = dummy_records(self.codec.width)
        period = base.shape[0]
        ram = self.ram
        chunk = np.tile(base, (64, 1))
        starts = list(range(0, ram.shape[0], chunk.shape[0]))

        def fill(at):
            end = min(at + chunk.shape[0], ram.shape[0])
            ram[at:end] = chunk[: end - at]

        with ThreadPoolExecutor(max_workers=DUMMY_FILL_THREADS) as pool:
            list(pool.map(fill, starts))
        flat = None
        where = "the RAM tier"
        file_rows = 0
        if self.tiers.disk_rows:
            mib = integer(DUMMY_DISK_MIB_ENV, DUMMY_DISK_MIB, 1, 4096)
            file_rows = min(self.tiers.disk_rows, mib * 2**20 // self.codec.record) // period * period
            if file_rows < period:
                raise RuntimeError(f"PLE_HOT dummy: a {mib} MiB stand-in file holds fewer than {period} records")
            directory, kind = flat_dir(file_rows * self.codec.record, self.config["dir"])
            self.path = directory / f"ple_dummy_{file_rows}x{self.codec.record}.bin"
            with open(self.path, "wb") as handle:
                for at in range(0, file_rows, chunk.shape[0]):
                    handle.write(chunk[: min(chunk.shape[0], file_rows - at)].tobytes())
            flat = DummyFlat(self.path, file_rows, self.codec.record, self.config["read"])
            where = (f"a stand-in file {self.path} on {kind or 'an unknown fs'} of {file_rows:,} records "
                     f"({file_rows * self.codec.record / 2**20:.0f} MiB written) read at rank modulo {file_rows:,} "
                     f"for all {self.tiers.disk_rows:,} disk-tier rows, and the RAM tier")
        self.escapes.seal()
        self.seen = {0: self.rows}
        self.source = TierSource(flat, self.tiers, self.ram, self.codec, self.escapes, self.cache.row_of)
        self.build = {"seconds": time.perf_counter() - began, "wrote": file_rows * self.codec.record,
                      "kept": 0, "direct": False}
        self.built = True
        say(f"{self.label}: DUMMY tiers built in {self.build['seconds']:.1f} s: RAM tier "
            f"{self.tiers.ram_rows * self.codec.record / 2**30:.2f} GiB filled with {period} deterministic "
            f"records repeated by rank, {self.build['wrote'] / 2**20:.0f} MiB written to disk, no escaped exponents, "
            f"no shard read; {memory_text()}")
        return where

    def _prewarm_rows(self, said):
        if not said:
            return np.zeros(0, dtype=np.int64)
        path = Path(said)
        rows = np.load(path) if path.suffix == ".npy" else np.asarray(json.loads(path.read_text(encoding="utf-8")))
        rows = np.unique(rows.astype(np.int64).reshape(-1))
        rows = rows[(rows >= 0) & (rows < self.rows)]
        return rows[: self.slots // 2]

    def put_shard(self, start, tensor, tp_start, tp_end):
        if self.built:
            raise RuntimeError(f"PLE_HOT {self.label}: a shard arrived after the table was sealed")
        count = int(tensor.shape[0])
        low = max(int(start), int(tp_start))
        high = min(int(start) + count, int(tp_end))
        if low >= high:
            return 0
        part = tensor.narrow(0, low - int(start), high - low).to(device="cpu", dtype=self.dtype).contiguous()
        data = part.view(torch.uint8).numpy().reshape(high - low, self.row_bytes)
        first = low - int(tp_start)
        with self.lock:
            if first in self.seen:
                raise RuntimeError(f"PLE_HOT {self.label}: rows from {first} arrived twice")
            self.seen[first] = high - low
            if self.prewarm.size:
                lo, hi = np.searchsorted(self.prewarm, [first, first + high - low])
                if hi > lo:
                    _, _, rows, fresh = self.cache.lookup(self.prewarm[lo:hi], 0)
                    self.cache.store[fresh] = data[rows - first]
                    self.prewarmed += int(rows.shape[0])
        if self.tiers is not None:
            self._pack(first, data)
        return high - low

    def _pack(self, first, data):
        for at in range(0, data.shape[0], PACK_ROWS):
            piece = data[at: at + PACK_ROWS]
            record, keys, exps = self.codec.encode(piece, first + at)
            in_ram, rank = self.tiers.route(np.arange(first + at, first + at + piece.shape[0], dtype=np.int64))
            to_ram = np.flatnonzero(in_ram)
            to_disk = np.flatnonzero(~in_ram)
            if to_ram.shape[0]:
                start = int(rank[to_ram[0]])
                if not np.array_equal(rank[to_ram], np.arange(start, start + to_ram.shape[0])):
                    raise RuntimeError(f"PLE_HOT {self.label}: the RAM tier ranks of rows from {first + at} are not consecutive")
                self.ram[start: start + to_ram.shape[0]] = record[to_ram]
            if to_disk.shape[0]:
                start = int(rank[to_disk[0]])
                if not np.array_equal(rank[to_disk], np.arange(start, start + to_disk.shape[0])):
                    raise RuntimeError(f"PLE_HOT {self.label}: the disk ranks of rows from {first + at} are not consecutive")
                self.writer.put(start, record[to_disk], self.codec.record)
            with self.lock:
                self.escapes.add(keys, exps)

    def seal(self):
        if self.built:
            return
        with self.lock:
            covered = 0
            for first in sorted(self.seen):
                if first != covered:
                    raise RuntimeError(f"PLE_HOT {self.label}: rows {covered}-{first} never arrived from the checkpoint")
                covered += self.seen[first]
            if covered != self.rows:
                raise RuntimeError(f"PLE_HOT {self.label}: {covered:,} of {self.rows:,} rows arrived from the checkpoint")
            if self.tiers is not None:
                self.escapes.seal()
                seconds = 0.0
                wrote = kept = 0
                direct = False
                flat = None
                if self.writer is not None:
                    seconds = self.writer.close()
                    wrote, kept, direct = self.writer.wrote, self.writer.kept, self.writer.direct
                    flat = FlatSource(self.path, self.tiers.disk_rows, self.codec.record, self.config["read"])
                    self.writer = None
                self.build = {"seconds": seconds, "wrote": wrote, "kept": kept, "direct": direct}
                self.source = TierSource(flat, self.tiers, self.ram, self.codec, self.escapes, self.cache.row_of)
                raw = self.rows * self.row_bytes
                packed = self.rows * self.codec.record + self.escapes.nbytes()
                say(f"{self.label}: tiers sealed in {seconds:.1f} s: disk file {self.tiers.disk_file_bytes / 2**30:.2f} GiB "
                    f"({wrote / 2**30:.2f} GiB written, {kept / 2**30:.2f} GiB already right), RAM tier "
                    f"{self.tiers.ram_rows * self.codec.record / 2**30:.2f} GiB, {self.escapes.count:,} escaped exponents "
                    f"in {self.escapes.nbytes() / 2**30:.3f} GiB of RAM; packed {packed / 2**30:.2f} of {raw / 2**30:.2f} "
                    f"GiB raw, ratio {packed / max(raw, 1):.4f}; misses read by {self.source.read}"
                    f"{' (page cache dropped every 256 MiB)' if self.source.read == 'mmap' else ''}, O_DIRECT "
                    f"{'on' if self.source.direct else 'off'}; {memory_text()}; {SCRATCH.text()}")
            say(f"{self.label}: {self.prewarmed:,} rows prewarmed from the loading shards")
            self.built = True

    def arm(self, lookups, pinned):
        host_ids = torch.empty(int(lookups), dtype=torch.int64, device="cpu", pin_memory=pinned)
        host_slots = torch.empty(int(lookups), dtype=torch.int64, device="cpu", pin_memory=pinned)
        self.engine = Engine(self.cache, self.source, self.rows, host_ids, host_slots, self.config["threads"],
                             self.config["every"], self.label)
        return self.engine

    def ready(self):
        if not self.built:
            self.seal()
            self.engine.source = self.source


def private_region(size):
    size = max(int(size), ALIGN)
    flags = getattr(mmap, "MAP_PRIVATE", None)
    region = mmap.mmap(-1, size) if flags is None else mmap.mmap(-1, size, flags=flags)
    return region, np.frombuffer(region, dtype=np.uint8)


def packed_ops():
    from vllm import taaf_ple_packed
    return taaf_ple_packed


class PackedTable:
    def __init__(self, *, rows, width, dtype, label, pinned, dummy):
        if dtype != torch.bfloat16:
            raise RuntimeError(f"{PACKED_ENV} packs BF16 rows with the exponent codec; this table is {dtype}")
        began = time.perf_counter()
        self.rows = int(rows)
        self.width = int(width)
        self.dtype = dtype
        self.label = label
        self.codec = RowCodec(self.width)
        self.size = self.rows * self.codec.record
        self.region, flat = private_region(self.size)
        self.host = torch.from_numpy(flat[: self.size])
        self.pinned = False
        if pinned:
            result = torch.cuda.cudart().cudaHostRegister(self.host.data_ptr(), self.size, 0)
            if result.value != 0:
                raise RuntimeError(f"PLE_PACKED: cudaHostRegister of the {self.size / 2**30:.2f} GiB packed table "
                                   f"failed: {result}")
            self.pinned = True
            if not self.host.is_pinned():
                raise RuntimeError("PLE_PACKED: CUDA does not report the registered packed table as pinned")
        self.table = flat[: self.size].reshape(self.rows, self.codec.record)
        self.stand_in = torch.zeros((1, self.width), dtype=dtype, device="cpu", pin_memory=bool(pinned))
        self.escapes = EscapeLog()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=PACK_THREADS, thread_name_prefix="ple-packed-pack")
        self.seen = {}
        self.done = 0
        self.reported = 0
        self.built = False
        self.count = 0
        self.escape_rows = None
        self.escape_exps = None
        self.ops = None
        self.told = False
        self.dummy = bool(dummy)
        if self.dummy:
            self._dummy_fill()
        raw = self.rows * 2 * self.width
        tell(f"{label}: {self.rows:,} rows x {2 * self.width} B = {raw / 2**30:.2f} GiB raw are held as one "
             f"{self.size / 2**30:.2f} GiB packed host buffer ({self.codec.record} B a row), "
             f"{'registered with CUDA' if self.pinned else 'unregistered'} in {time.perf_counter() - began:.1f} s; "
             f"the GPU gathers and decodes each step's rows from it; "
             f"{'filled with deterministic DUMMY records' if self.dummy else 'packed as the shards load'} by "
             f"{PACK_THREADS} threads; {memory_text()}")

    def _dummy_fill(self):
        chunk = np.tile(dummy_records(self.width), (64, 1))

        def fill(at):
            end = min(at + chunk.shape[0], self.rows)
            self.table[at:end] = chunk[: end - at]

        list(self.pool.map(fill, range(0, self.rows, chunk.shape[0])))
        self.seen = {0: self.rows}

    def put_shard(self, start, tensor, tp_start, tp_end):
        if self.built:
            raise RuntimeError(f"PLE_PACKED {self.label}: a shard arrived after the table was sealed")
        count = int(tensor.shape[0])
        low = max(int(start), int(tp_start))
        high = min(int(start) + count, int(tp_end))
        if low >= high:
            return 0
        part = tensor.narrow(0, low - int(start), high - low).to(device="cpu", dtype=self.dtype).contiguous()
        data = part.view(torch.uint8).numpy().reshape(high - low, 2 * self.width)
        first = low - int(tp_start)
        with self.lock:
            if first in self.seen:
                raise RuntimeError(f"PLE_PACKED {self.label}: rows from {first} arrived twice")
            if first + high - low > self.rows:
                raise RuntimeError(f"PLE_PACKED {self.label}: rows {first}-{first + high - low} fall past the "
                                   f"{self.rows:,}-row table")
            self.seen[first] = high - low
        list(self.pool.map(self._pack, [(first + at, data[at: at + PACK_ROWS])
                                        for at in range(0, data.shape[0], PACK_ROWS)]))
        with self.lock:
            self.done += high - low
            packed_bytes = self.done * self.codec.record
            if packed_bytes >= self.reported + PROGRESS_EVERY:
                self.reported = packed_bytes // PROGRESS_EVERY * PROGRESS_EVERY
                tell(f"{self.label}: packed {packed_bytes / 2**30:.1f} of {self.size / 2**30:.1f} GiB, "
                     f"{self.escapes.count:,} escaped exponents so far; {memory_text()}")
        return high - low

    def _pack(self, piece):
        first, data = piece
        record, keys, exps = self.codec.encode(data, first)
        self.table[first: first + data.shape[0]] = record
        with self.lock:
            self.escapes.add(keys, exps)

    def seal(self, device):
        if self.built:
            return
        began = time.perf_counter()
        self.pool.shutdown(wait=True)
        with self.lock:
            covered = 0
            for first in sorted(self.seen):
                if first != covered:
                    raise RuntimeError(f"PLE_PACKED {self.label}: rows {covered}-{first} never arrived from the checkpoint")
                covered += self.seen[first]
            if covered != self.rows:
                raise RuntimeError(f"PLE_PACKED {self.label}: {covered:,} of {self.rows:,} rows arrived from the "
                                   f"checkpoint")
            self.escapes.seal()
            count = int(self.escapes.count)
            rows = self.escapes.keys // self.width if count else np.zeros(1, dtype=np.int64)
            exps = self.escapes.exps if count else np.zeros(1, dtype=np.uint8)
            kind = np.int32 if self.rows < 2**31 else np.int64
            rows = torch.from_numpy(np.ascontiguousarray(rows.astype(kind)))
            exps = torch.from_numpy(np.ascontiguousarray(exps))
            nbytes = rows.numel() * rows.element_size() + exps.numel()
            on_device = nbytes <= ESCAPE_DEVICE_BYTES or torch.device(device).type == "cpu"
            self.escape_rows = rows.to(device) if on_device else rows.pin_memory()
            self.escape_exps = exps.to(device) if on_device else exps.pin_memory()
            self.count = count
            self.escapes = None
            self.ops = packed_ops()
            self.built = True
        tell(f"{self.label}: sealed in {time.perf_counter() - began:.1f} s: {self.rows:,} rows in "
             f"{self.size / 2**30:.2f} GiB, {count:,} escaped exponents in {nbytes / 2**20:.1f} MiB "
             f"{'on ' + str(torch.device(device)) if on_device else 'of pinned host memory'}; packed "
             f"{(self.size + nbytes) / 2**30:.2f} of {self.rows * 2 * self.width / 2**30:.2f} GiB raw; {memory_text()}")

    def ready(self, device):
        if self.built:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"PLE_PACKED {self.label}: the table must be sealed by an eager step before any "
                               f"CUDA graph capture")
        self.seal(device)

    def gather(self, ids, out, start, end):
        self.ops.gather(self.host, ids, out, self.escape_rows, self.escape_exps, self.count, start, end)
        if not self.told:
            self.told = True
            tell(f"live: {self.label}: the first gather decoded {ids.numel():,} lookups on the GPU straight from the "
                 f"pinned packed table; no host work runs per step")
        return out

    def close(self):
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            self.pinned = False


def reserved_region(size):
    size = max(int(size), ALIGN)
    flags = getattr(mmap, "MAP_PRIVATE", None)
    if flags is None:
        region = mmap.mmap(-1, size)
    else:
        spare = getattr(mmap, "MAP_NORESERVE", 0x4000 if sys.platform.startswith("linux") else 0)
        region = mmap.mmap(-1, size, flags=flags | spare)
    return region, np.frombuffer(region, dtype=np.uint8)


def huffman_lengths(counts):
    live = [int(one) for one in np.flatnonzero(counts)]
    if len(live) == 1:
        return {live[0]: 1}
    heap = [(int(counts[one]), one, (one,)) for one in live]
    heapq.heapify(heap)
    depth = dict.fromkeys(live, 0)
    tick = len(counts)
    while len(heap) > 1:
        light, _, first = heapq.heappop(heap)
        heavy, _, second = heapq.heappop(heap)
        for one in first + second:
            depth[one] += 1
        heapq.heappush(heap, (light + heavy, tick, first + second))
        tick += 1
    return depth


def code_lengths(counts, limit=DF11_LIMIT):
    counts = np.asarray(counts, dtype=np.int64)
    if np.count_nonzero(counts) > 1 << int(limit):
        raise ValueError(f"{np.count_nonzero(counts)} symbols do not fit codes of at most {limit} bits")
    floor = 0
    while True:
        depth = huffman_lengths(np.where(counts > 0, np.maximum(counts, floor), 0))
        if max(depth.values()) <= limit:
            break
        floor = max(2 * floor, int(counts.sum()) >> int(limit), 1)
    lengths = np.zeros(counts.shape[0], dtype=np.uint8)
    for one, size in depth.items():
        lengths[one] = size
    return lengths


class Df11Book:
    def __init__(self, lengths, limit=DF11_LIMIT):
        lengths = np.asarray(lengths, dtype=np.int64)
        limit = int(limit)
        if lengths.shape != (DF11_ESCAPE + 1,) or lengths.min() < 0 or lengths.max() > limit or not lengths[DF11_ESCAPE]:
            raise ValueError(f"a DF11 codebook holds {DF11_ESCAPE + 1} code lengths of at most {limit} bits with the "
                             f"escape coded")
        if float(np.sum(np.exp2(-lengths[lengths > 0].astype(np.float64)))) > 1.0:
            raise ValueError("the DF11 code lengths break the Kraft inequality")
        self.limit = limit
        self.lengths = lengths.astype(np.uint8)
        self.sample_bits = None
        self.sampled = 0
        order = sorted((int(size), one) for one, size in enumerate(lengths) if size)
        self.symbols = np.array([one for _, one in order], dtype=np.int64)
        self.first = np.zeros(limit + 1, dtype=np.int64)
        self.count = np.zeros(limit + 1, dtype=np.int64)
        self.offset = np.zeros(limit + 1, dtype=np.int64)
        codes = np.zeros(DF11_ESCAPE + 1, dtype=np.int64)
        code, previous = 0, order[0][0]
        for at, (size, one) in enumerate(order):
            if at:
                code = (code + 1) << (size - previous)
            previous = size
            codes[one] = code
            if not self.count[size]:
                self.first[size] = code
                self.offset[size] = at
            self.count[size] += 1
        reverse = np.zeros(DF11_ESCAPE + 1, dtype=np.int64)
        self.lut = np.zeros(1 << limit, dtype=np.int32)
        for size, one in order:
            reverse[one] = int(format(int(codes[one]), f"0{size}b")[::-1], 2)
            self.lut[reverse[one] + (np.arange(1 << (limit - size), dtype=np.int64) << size)] = (one << 4) | size
        escaped = lengths[:DF11_ESCAPE] == 0
        code = np.where(escaped, reverse[DF11_ESCAPE], reverse[:DF11_ESCAPE])
        bits = np.where(escaped, lengths[DF11_ESCAPE], lengths[:DF11_ESCAPE])
        self.entries = {"cpu": torch.from_numpy((code | (bits << 16) | (escaped.astype(np.int64) << 24)).astype(np.int32))}

    @classmethod
    def from_rows(cls, data, limit=DF11_LIMIT, sample=DF11_SAMPLE_ROWS):
        picked = np.ascontiguousarray(data[:: max(1, -(-data.shape[0] // int(sample)))])
        counts = np.zeros(DF11_ESCAPE + 1, dtype=np.int64)
        counts[:DF11_ESCAPE] = np.bincount(((picked.view("<u2") >> 7) & 255).ravel(), minlength=DF11_ESCAPE)
        counts[DF11_ESCAPE] = 1
        book = cls(code_lengths(counts, limit), limit)
        book.sample_bits = float((counts * book.lengths).sum() / max(1, counts[:DF11_ESCAPE].sum()))
        book.sampled = int(picked.shape[0])
        return book

    def row_bytes(self, width):
        return -(-int(width) * int(self.lengths.max()) // 8)

    def encode(self, data, first, width, group, device="cpu"):
        device = torch.device(device)
        table = self.entries.get(str(device))
        if table is None:
            table = self.entries[str(device)] = self.entries["cpu"].to(device)
        count = data.shape[0]
        span = -(-int(width) * int(self.lengths.max()) // 64) + 1
        words = torch.from_numpy(np.ascontiguousarray(data)).to(device).view(torch.int16).reshape(count, width)
        exp = (words >> 7) & 255
        signs = (((words >> 8) & 128) | (words & 127)).to(torch.uint8)
        entry = table[exp.int()]
        bits = (entry >> 16) & 255
        ends = bits.cumsum(1, dtype=torch.int32)
        pos = ends - bits
        sizes = (ends[:, -1] + 7) >> 3
        code = (entry & 0xFFFF).long()
        shift = (pos & 63).long()
        at = ((pos >> 6) + (torch.arange(count, dtype=torch.int32, device=device) * span)[:, None]).reshape(-1).long()
        block = torch.zeros(count * span, dtype=torch.int64, device=device)
        block.index_add_(0, at, (code << shift).reshape(-1))
        block.index_add_(0, at + 1, ((code >> 1) >> (63 - shift)).reshape(-1))
        keep = torch.arange(span * 8, dtype=torch.int32, device=device)[None, :] < sizes[:, None]
        stream = block.view(torch.uint8).reshape(count, span * 8)[keep]
        starts = torch.zeros(count, dtype=torch.int64, device=device)
        torch.cumsum(sizes[:-1].long(), 0, out=starts[1:])
        escaped = (entry >> 24) != 0
        hit = escaped.nonzero()
        keys = (hit[:, 0] + int(first)) * width + hit[:, 1]
        return (signs.cpu().numpy(), sizes.to(torch.uint8).cpu().numpy(), stream.cpu().numpy(),
                starts[::group].cpu().numpy(), keys.cpu().numpy(), exp[escaped].to(torch.uint8).cpu().numpy())


def df11_decode(book, signs, lengths, bases, stream, escape_keys, escape_exps, group, ids):
    ids = np.asarray(ids, dtype=np.int64)
    rows, width = signs.shape
    count = ids.shape[0]
    inside = (ids >= 0) & (ids < rows)
    row = np.where(inside, ids, 0)
    head = row // group * group
    at = bases[row // group].astype(np.int64)
    for member in range(group - 1):
        at += np.where(head + member < row, lengths[np.minimum(head + member, rows - 1)], 0).astype(np.int64)
    data = np.asarray(stream, dtype=np.uint8)
    size_bytes = data.shape[0]
    bit = at * 8
    exps = np.zeros((count, width), dtype=np.int64)
    last = book.symbols.shape[0] - 1
    for column in range(width):
        code = np.zeros(count, dtype=np.int64)
        found = np.full(count, -1, dtype=np.int64)
        size = np.zeros(count, dtype=np.int64)
        for length in range(1, book.limit + 1):
            spot = bit + length - 1
            index = spot >> 3
            byte = np.where(index < size_bytes, data[np.minimum(index, max(size_bytes - 1, 0))], 0).astype(np.int64)
            code = (code << 1) | ((byte >> (spot & 7)) & 1)
            rank = code - book.first[length]
            hit = (found < 0) & (rank >= 0) & (rank < book.count[length])
            found = np.where(hit, book.symbols[np.clip(book.offset[length] + rank, 0, last)], found)
            size = np.where(hit, length, size)
        if (found < 0).any():
            raise ValueError("a DF11 row holds bits that are no code of its book")
        exps[:, column] = found
        bit += size
    escaped = exps == DF11_ESCAPE
    if escaped.any():
        where_row, where_col = np.nonzero(escaped)
        keys = row[where_row] * width + where_col
        slot = np.minimum(np.searchsorted(escape_keys, keys), max(0, escape_keys.shape[0] - 1))
        if not escape_keys.shape[0] or not (escape_keys[slot] == keys).all():
            raise RuntimeError("PLE_DF11: an escaped exponent is missing from the escape table")
        exps[where_row, where_col] = escape_exps[slot]
    low = signs[row].astype(np.int64)
    word = ((low & 128) << 8) | (exps << 7) | (low & 127)
    word[~inside] = 0
    return word.astype(np.uint16)


class Df11Table:
    def __init__(self, *, rows, width, dtype, label, pinned, dummy, book=None, reserve=None):
        if dtype != torch.bfloat16:
            raise RuntimeError(f"{DF11_ENV} packs BF16 rows with the exponent code; this table is {dtype}")
        began = time.perf_counter()
        self.rows = int(rows)
        self.width = int(width)
        self.dtype = dtype
        self.label = label
        self.group = DF11_GROUP
        self.groups = -(-self.rows // self.group)
        self.lengths_at = self.rows * self.width
        self.bases_at = ceil_to(self.lengths_at + self.rows, 8)
        self.size = self.bases_at + 8 * self.groups
        self.region, flat = private_region(self.size)
        self.host = torch.from_numpy(flat[: self.size])
        self.signs = flat[: self.lengths_at].reshape(self.rows, self.width)
        self.lengths = flat[self.lengths_at: self.lengths_at + self.rows]
        self.bases = flat[self.bases_at: self.size].view(np.int64)
        self.want_pinned = bool(pinned)
        self.pinned = self._register(self.host, self.size, "table") if pinned else False
        self.device = torch.device("cuda") if pinned and torch.cuda.is_available() else torch.device("cpu")
        self.device_lock = threading.Lock() if self.device.type == "cuda" else contextlib.nullcontext()
        self.stream_pinned = False
        self.book = None
        self.stream_region = None
        self.stream = None
        self.reserved = 0
        self.used = 0
        self.pending = {}
        self.stand_in = torch.zeros((1, self.width), dtype=dtype, device="cpu", pin_memory=bool(pinned))
        self.escapes = EscapeLog()
        self.escape_keys = None
        self.escape_values = None
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=PACK_THREADS, thread_name_prefix="ple-df11-pack")
        self.seen = {}
        self.done = 0
        self.reported = 0
        self.built = False
        self.count = 0
        self.escape_rows = None
        self.escape_exps = None
        self.lut = None
        self.ops = None
        self.told = False
        self.preloaded = False
        self.expected = None
        self.claimed = {}
        self.real_tensors = 0
        self.stubbed = 0
        self.header = None
        self.first_put = None
        self.filler = None
        self.dummy = bool(dummy)
        if book is not None:
            self._open(book, reserve)
        if self.dummy:
            self._dummy_fill()
        tell(f"{label}: DF11: {self.rows:,} rows x {2 * self.width} B = {self.rows * 2 * self.width / 2**30:.2f} GiB "
             f"raw are held as {self.size / 2**30:.2f} GiB of sign+mantissa bytes, row lengths and one offset per "
             f"{self.group} rows, {'registered with CUDA' if self.pinned else 'unregistered'}, beside a Huffman "
             f"stream of the exponents encoded on {self.device.type} as the shards load, in "
             f"{time.perf_counter() - began:.1f} s; {memory_text()}")

    def _register(self, tensor, size, what):
        result = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), size, 0)
        if result.value != 0:
            raise RuntimeError(f"PLE_DF11: cudaHostRegister of the {size / 2**30:.2f} GiB packed {what} failed: "
                               f"{result}")
        if not tensor.is_pinned():
            raise RuntimeError(f"PLE_DF11: CUDA does not report the registered packed {what} as pinned")
        return True

    def _open(self, book, reserve=None):
        self.book = book
        self.reserved = (int(reserve) if reserve is not None else self.rows * book.row_bytes(self.width)) + ALIGN
        self.stream_region, self.stream = reserved_region(self.reserved)
        if book.sample_bits is not None:
            tell(f"{self.label}: DF11 codebook from {book.sampled:,} sampled rows: "
                 f"{int(np.count_nonzero(book.lengths[:DF11_ESCAPE]))} exponents coded, longest code "
                 f"{int(book.lengths.max())} bits, {book.sample_bits:.3f} bits an exponent on the sample")

    def _dummy_fill(self):
        base = RowCodec(self.width).decode(dummy_records(self.width), np.arange(DUMMY_BASE_ROWS), EscapeLog())
        book = Df11Book.from_rows(base)
        self._open(book, DUMMY_BASE_ROWS * book.row_bytes(self.width))
        signs, sizes, stream, heads, _, _ = book.encode(base, 0, self.width, self.group)
        self.stream[: stream.shape[0]] = stream
        self.used = int(stream.shape[0])
        tile = base.shape[0]

        def fill(at):
            end = min(at + tile, self.rows)
            self.signs[at:end] = signs[: end - at]
            self.lengths[at:end] = sizes[: end - at]
            heads_in = -(-(end - at) // self.group)
            self.bases[at // self.group: at // self.group + heads_in] = heads[:heads_in]

        list(self.pool.map(fill, range(0, self.rows, tile)))
        self.seen = {0: self.rows}
        self.done = self.rows

    def put_shard(self, start, tensor, tp_start, tp_end):
        if self.built:
            raise RuntimeError(f"PLE_DF11 {self.label}: a shard arrived after the table was sealed")
        count = int(tensor.shape[0])
        low = max(int(start), int(tp_start))
        high = min(int(start) + count, int(tp_end))
        if low >= high:
            return 0
        if self.preloaded:
            return self._claim(int(start), count, bool(getattr(tensor, "is_meta", False)), high - low)
        if self.first_put is None:
            self.first_put = time.perf_counter()
        part = tensor.narrow(0, low - int(start), high - low).to(device="cpu", dtype=self.dtype).contiguous()
        data = part.view(torch.uint8).numpy().reshape(high - low, 2 * self.width)
        first = low - int(tp_start)
        end = first + high - low
        with self.lock:
            if first in self.seen:
                raise RuntimeError(f"PLE_DF11 {self.label}: rows from {first} arrived twice")
            if end > self.rows:
                raise RuntimeError(f"PLE_DF11 {self.label}: rows {first}-{end} fall past the {self.rows:,}-row table")
            self.seen[first] = high - low
            if self.book is None:
                self._open(Df11Book.from_rows(data))
        lead = min(end, ceil_to(first, self.group))
        tail = end if end == self.rows else max(lead, end // self.group * self.group)
        step = max(self.group, PACK_ROWS // self.group * self.group)
        with self.lock:
            for edge, stop in ((first, lead), (tail, end)):
                if stop > edge:
                    self.pending[edge] = data[edge - first: stop - first].copy()
        list(self.pool.map(self._pack, [(at, data[at - first: min(at + step, tail) - first])
                                        for at in range(lead, tail, step)]))
        with self.lock:
            self.done += high - low
            raw = self.done * 2 * self.width
            if raw >= self.reported + PROGRESS_EVERY:
                self.reported = raw // PROGRESS_EVERY * PROGRESS_EVERY
                tell(f"{self.label}: DF11 packed {self.done:,} of {self.rows:,} rows, {self.used / 2**30:.1f} GiB of "
                     f"exponent codes, {self.escapes.count:,} escaped exponents so far; {memory_text()}")
        return high - low

    def _claim(self, start, count, stub, taken):
        with self.lock:
            if self.expected is not None and self.expected.get(start) != count:
                raise RuntimeError(f"PLE_DF11 {self.label}: the loader placed {count:,} rows at row {start:,} and the "
                                   f"precomputed table holds no such shard; it was baked for another checkpoint "
                                   f"layout")
            if start in self.claimed:
                raise RuntimeError(f"PLE_DF11 {self.label}: rows from {start} arrived twice")
            self.claimed[start] = count
            if not stub:
                self.real_tensors += 1
        return taken

    def _pack(self, piece):
        first, data = piece
        with self.device_lock:
            signs, sizes, stream, heads, keys, exps = self.book.encode(data, first, self.width, self.group, self.device)
        self._place(first, data.shape[0], (signs, sizes, stream, heads, keys, exps))

    def _place(self, first, count, encoded):
        signs, sizes, stream, heads, keys, exps = encoded
        with self.lock:
            at = self.used
            self.used += int(stream.shape[0])
            if self.used > self.reserved:
                raise RuntimeError(f"PLE_DF11 {self.label}: the exponent codes outgrew their {self.reserved:,} B")
            self.escapes.add(keys, exps)
        self.signs[first: first + count] = signs
        self.lengths[first: first + count] = sizes
        self.stream[at: at + stream.shape[0]] = stream
        self.bases[first // self.group: first // self.group + heads.shape[0]] = heads + at

    def await_fill(self):
        pending, self.filler = self.filler, None
        if pending is None:
            return
        gate = getattr(self, "gate", None)
        if gate is not None:
            gate.release("seal")
        began = time.perf_counter()
        pending.result()
        waited = time.perf_counter() - began
        tell(f"{self.label}: DF11 background fill joined, the load waited {waited:.1f} s for it")

    def seal(self, device):
        if self.built:
            return
        self.await_fill()
        began = time.perf_counter()
        self.pool.shutdown(wait=True)
        if self.preloaded:
            self._check_claims()
        with self.lock:
            covered = 0
            for first in sorted(self.seen):
                if first != covered:
                    raise RuntimeError(f"PLE_DF11 {self.label}: rows {covered}-{first} never arrived from the checkpoint")
                covered += self.seen[first]
            if covered != self.rows:
                raise RuntimeError(f"PLE_DF11 {self.label}: {covered:,} of {self.rows:,} rows arrived from the "
                                   f"checkpoint")
            runs = []
            for first in sorted(self.pending):
                piece = self.pending[first]
                if runs and runs[-1][0] + runs[-1][1].shape[0] == first:
                    runs[-1] = (runs[-1][0], np.concatenate([runs[-1][1], piece]))
                else:
                    runs.append((first, piece))
            self.pending = {}
        for first, data in runs:
            stop = first + data.shape[0]
            if first % self.group or (stop % self.group and stop != self.rows):
                raise RuntimeError(f"PLE_DF11 {self.label}: the shard edges {first}-{stop} do not close whole groups")
            self._pack((first, data))
        if self.device.type == "cuda":
            self.book.entries = {"cpu": self.book.entries["cpu"]}
            torch.cuda.empty_cache()
        self._finish(device)
        if not self.preloaded and self.first_put is not None:
            tell(f"{self.label}: DF11 path=encode: {time.perf_counter() - self.first_put:.1f} s from the first shard "
                 f"of the checkpoint to the sealed table")
        tell(f"{self.label}: DF11 sealed in {time.perf_counter() - began:.1f} s: {self.rows:,} rows in "
             f"{(self.size + self.used) / 2**30:.2f} GiB ({(self.size + self.used) / self.rows:.2f} B a row: "
             f"{self.size / 2**30:.2f} GiB sign+mantissa, lengths and offsets, {self.used / 2**30:.2f} GiB of "
             f"exponent codes), {self.count:,} escaped exponents; of {self.rows * 2 * self.width / 2**30:.2f} GiB "
             f"raw; {memory_text()}")

    def _check_claims(self):
        if self.expected is None:
            return
        wanted = self.expected
        if self.claimed != wanted:
            lacking = sorted(set(wanted) - set(self.claimed))[:4]
            raise RuntimeError(f"PLE_DF11 {self.label}: the loader delivered {len(self.claimed)} of the "
                               f"{len(wanted)} shards of the precomputed table (missing rows from {lacking})")
        tell(f"{self.label}: DF11 path=precomputed: {len(self.claimed)} checkpoint shards claimed by the loader, "
             f"{len(self.claimed) - self.real_tensors} as unread stand-ins and {self.real_tensors} as real tensors "
             f"{'(the raw bytes were read and dropped)' if self.real_tensors else '(no raw byte of the table was read)'}")

    def _finish(self, device):
        with self.lock:
            self.escapes.seal()
            count = int(self.escapes.count)
            self.escape_keys = self.escapes.keys
            self.escape_values = self.escapes.exps
            rows = self.escapes.keys // self.width if count else np.zeros(1, dtype=np.int64)
            exps = self.escapes.exps if count else np.zeros(1, dtype=np.uint8)
            kind = np.int32 if self.rows < 2**31 else np.int64
            rows = torch.from_numpy(np.ascontiguousarray(rows.astype(kind)))
            exps = torch.from_numpy(np.ascontiguousarray(exps))
            nbytes = rows.numel() * rows.element_size() + exps.numel()
            on_device = nbytes <= ESCAPE_DEVICE_BYTES or torch.device(device).type == "cpu"
            self.escape_rows = rows.to(device) if on_device else rows.pin_memory()
            self.escape_exps = exps.to(device) if on_device else exps.pin_memory()
            self.stream_t = torch.from_numpy(self.stream)
            if self.want_pinned and not self.stream_pinned:
                self.stream_pinned = self._register(self.stream_t, ceil_to(max(self.used, 1), ALIGN), "exponent codes")
            self.signs_t = torch.from_numpy(self.signs)
            self.lengths_t = torch.from_numpy(self.lengths)
            self.bases_t = torch.from_numpy(self.bases)
            self.lut = torch.from_numpy(self.book.lut).to(device)
            self.count = count
            self.escapes = None
            self.built = True

    def ready(self, device):
        if self.built:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"PLE_DF11 {self.label}: the table must be sealed by an eager step before any "
                               f"CUDA graph capture")
        self.seal(device)

    def decode(self, ids):
        return df11_decode(self.book, self.signs, self.lengths, self.bases, self.stream[: self.used],
                           self.escape_keys, self.escape_values, self.group, ids)

    def gather(self, ids, out, start, end):
        if self.ops is None:
            self.ops = packed_ops()
        self.ops.df11_gather(self.signs_t, self.lengths_t, self.bases_t, self.stream_t, self.lut, ids, out,
                             self.escape_rows, self.escape_exps, self.count, start, end, self.group, self.book.limit)
        if not self.told:
            self.told = True
            tell(f"live: {self.label}: the first DF11 gather decoded {ids.numel():,} lookups on the GPU straight from "
                 f"the pinned packed table; no host work runs per step")
        return out

    def close(self):
        pending, self.filler = self.filler, None
        if pending is not None:
            try:
                pending.result()
            except BaseException:
                pass
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            self.pinned = False
        if self.stream_pinned:
            torch.cuda.cudart().cudaHostUnregister(self.stream_t.data_ptr())
            self.stream_pinned = False


class PartsReader:
    def __init__(self, paths):
        self.paths = [Path(one) for one in paths]
        if not self.paths:
            raise FileNotFoundError("PLE_DF11: no file part to read")
        self.name = str(self.paths[0])
        self.size = sum(one.stat().st_size for one in self.paths)
        self.at = 0
        self.handle = open(self.paths[0], "rb")

    def _advance(self):
        if self.at + 1 >= len(self.paths):
            return False
        self.handle.close()
        self.at += 1
        self.handle = open(self.paths[self.at], "rb")
        return True

    def readinto(self, view):
        while True:
            got = self.handle.readinto(view)
            if got or not self._advance():
                return got

    def read(self, count):
        out = bytearray(count)
        done = 0
        while done < count:
            got = self.readinto(memoryview(out)[done:])
            if not got:
                break
            done += got
        return bytes(out[:done])

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_exact(handle, view):
    view = memoryview(view).cast("B")
    done = 0
    while done < len(view):
        got = handle.readinto(view[done:])
        if not got:
            raise RuntimeError(f"PLE_DF11: {handle.name} ended {len(view) - done:,} B early")
        done += got


def df11_parts(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.parent.glob(path.name + ".p[0-9][0-9][0-9]"))


def find_df11_file():
    said = str(os.environ.get(DF11_FILE_ENV) or "").strip()
    if said.lower() in ("0", "false", "no", "off"):
        return None
    if said:
        return Path(said) if df11_parts(said) else None
    root = Path(DF11_ROOT)
    if not root.is_dir():
        return None
    for depth in range(DF11_DEPTH + 1):
        pattern = "/".join(["*"] * depth + [DF11_FILE_NAME + "*"])
        for found in sorted(root.glob(pattern)):
            base = found.with_name(DF11_FILE_NAME)
            if df11_parts(base):
                return base
    return None


def df11_header_bytes(table, extra=None, reserve=0):
    count = int(table.count)
    value = {"version": DF11_VERSION, "rows": table.rows, "width": table.width, "group": table.group,
             "limit": table.book.limit, "lengths": table.book.lengths.tolist(), "size": table.size,
             "used": table.used, "escapes": count}
    value.update(extra or {})
    raw = json.dumps(value, sort_keys=True).encode("utf-8")
    if reserve:
        if len(raw) > reserve:
            raise RuntimeError(f"PLE_DF11: the header takes {len(raw):,} B, over its {reserve:,} B reserve")
        raw = raw + b" " * (reserve - len(raw))
    return raw


def write_df11(table, path, extra=None):
    if not table.built:
        raise RuntimeError("PLE_DF11: only a sealed table can be written")
    count = int(table.count)
    header = df11_header_bytes(table, extra)
    with open(path, "wb") as handle:
        handle.write(DF11_MAGIC)
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(memoryview(table.host.numpy()))
        handle.write(memoryview(table.stream[: table.used]))
        handle.write(np.ascontiguousarray(table.escape_keys[:count], dtype="<i8").tobytes())
        handle.write(np.ascontiguousarray(table.escape_values[:count], dtype=np.uint8).tobytes())
    return Path(path).stat().st_size


def read_df11_header(path):
    with PartsReader(df11_parts(path)) as handle:
        if handle.read(len(DF11_MAGIC)) != DF11_MAGIC:
            raise RuntimeError(f"PLE_DF11: {path} is not a packed DF11 table")
        return json.loads(handle.read(int.from_bytes(handle.read(8), "little")).decode("utf-8"))


def df11_drop():
    if not str(os.environ.get(DF11_DROP_ENV) or "").strip():
        return True
    return df11_flag(DF11_DROP_ENV)


def df11_flag(name):
    said = str(os.environ.get(name) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said not in ("1", "true", "yes", "on"):
        raise ValueError(f"{name} must be a boolean, found {said!r}")
    return True


def df11_async():
    return df11_flag(DF11_ASYNC_ENV)


def df11_read_threads():
    wanted = str(os.environ.get(DF11_READ_THREADS_ENV) or "").strip()
    if wanted:
        return integer(DF11_READ_THREADS_ENV, 1, 1, 256)
    return DF11_READ_THREADS_DEFAULT if df11_async() else 1


def df11_header_end(handle, path):
    if handle.read(len(DF11_MAGIC)) != DF11_MAGIC:
        raise RuntimeError(f"PLE_DF11: {path} is not a packed DF11 table")
    size = int.from_bytes(handle.read(8), "little")
    header = json.loads(handle.read(size).decode("utf-8"))
    return header, len(DF11_MAGIC) + 8 + size


def df11_table_from(header, path, *, label, pinned):
    if header.get("version") != DF11_VERSION or header.get("group") != DF11_GROUP:
        raise RuntimeError(f"PLE_DF11: {path} holds version {header.get('version')} with groups of "
                           f"{header.get('group')}; this reader takes version {DF11_VERSION}, groups of {DF11_GROUP}")
    table = Df11Table(rows=header["rows"], width=header["width"], dtype=torch.bfloat16, label=label,
                      pinned=pinned, dummy=False, book=Df11Book(header["lengths"], header["limit"]),
                      reserve=header["used"])
    if table.size != header["size"]:
        table.close()
        raise RuntimeError(f"PLE_DF11: {path} lays out {header['size']:,} B of rows, this reader {table.size:,}")
    return table


def df11_adopt(table, header, keys, exps):
    table.used = int(header["used"])
    table.escapes.add(keys.astype(np.int64), exps)
    table.seen = {0: table.rows}
    table.done = table.rows
    table.header = header
    shards = header.get("shards")
    if shards:
        table.preloaded = True
        table.expected = {int(one["start"]): int(one["rows"]) for one in shards}
    return table


def read_df11(path, *, label, pinned=False):
    with PartsReader(df11_parts(path)) as handle:
        header, _ = df11_header_end(handle, path)
        table = df11_table_from(header, path, label=label, pinned=pinned)
        try:
            read_exact(handle, table.host.numpy())
            read_exact(handle, table.stream[: header["used"]])
            count = int(header["escapes"])
            keys = np.empty(count, dtype="<i8")
            exps = np.empty(count, dtype=np.uint8)
            read_exact(handle, keys)
            read_exact(handle, exps)
            if handle.read(1):
                raise RuntimeError(f"PLE_DF11: {path} runs past its escape table")
        except BaseException:
            table.close()
            raise
    return df11_adopt(table, header, keys, exps)


class PartFiles:
    def __init__(self, paths, drop=False):
        self.paths = [Path(one) for one in paths]
        self.sizes = [one.stat().st_size for one in self.paths]
        self.size = sum(self.sizes)
        self.drop = bool(drop) and hasattr(os, "posix_fadvise")

    def read_into(self, offset, view):
        view = memoryview(view).cast("B")
        done = 0
        base = 0
        for path, size in zip(self.paths, self.sizes):
            if done == len(view):
                break
            if offset + done >= base + size:
                base += size
                continue
            local = offset + done - base
            take = min(len(view) - done, size - local)
            with open(path, "rb", buffering=0) as handle:
                got = 0
                while got < take:
                    handle.seek(local + got)
                    count = handle.readinto(view[done + got: done + take])
                    if not count:
                        raise RuntimeError(f"PLE_DF11: {path} ended {take - got:,} B early")
                    got += count
                if self.drop:
                    os.posix_fadvise(handle.fileno(), local, take, os.POSIX_FADV_DONTNEED)
            done += take
            base += size
        if done != len(view):
            raise RuntimeError(f"PLE_DF11: {self.paths[0]} ended {len(view) - done:,} B early")


def df11_open(path, *, label, pinned=False):
    with PartsReader(df11_parts(path)) as handle:
        header, head_end = df11_header_end(handle, path)
    table = df11_table_from(header, path, label=label, pinned=pinned)
    count = int(header["escapes"])
    keys = np.empty(count, dtype="<i8")
    exps = np.empty(count, dtype=np.uint8)
    table.header = header
    table.preloaded = bool(header.get("shards"))
    table.expected = ({int(one["start"]): int(one["rows"]) for one in header["shards"]}
                      if header.get("shards") else None)
    return table, head_end, keys, exps


def df11_fill(path, table, head_end, keys, exps, threads, drop):
    began = time.perf_counter()
    header = table.header
    files = PartFiles(df11_parts(path), drop=drop)
    spans = [table.host.numpy(), table.stream[: header["used"]], keys, exps]
    expected = head_end + sum(memoryview(one).cast("B").nbytes for one in spans)
    if files.size != expected:
        raise RuntimeError(f"PLE_DF11: {path} holds {files.size:,} B where its header lays out {expected:,}")
    jobs = []
    at = head_end
    for span in spans:
        flat = memoryview(span).cast("B")
        for first in range(0, flat.nbytes, DF11_READ_PIECE):
            piece = flat[first: first + DF11_READ_PIECE]
            jobs.append((at + first, piece))
        at += flat.nbytes
    with ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="ple-df11-read") as pool:
        for outcome in pool.map(lambda job: files.read_into(*job), jobs):
            pass
    df11_adopt(table, header, keys, exps)
    seconds = time.perf_counter() - began
    tell(f"{table.label}: DF11 filled {files.size / 2**30:.2f} GiB from {len(files.paths)} part(s) with {threads} "
         f"reader thread(s) in {seconds:.1f} s ({files.size / max(seconds, 1e-9) / 1e9:.2f} GB/s)"
         f"{', dropping each piece from the page cache behind the read' if files.drop else ''}")
    return seconds


def safetensors_header(path):
    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(size).decode("utf-8")), 8 + size


def checkpoint_shards(directory, mark=DF11_TENSOR_MARK):
    directory = Path(directory)
    weights = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    found = []
    for name in sorted(set(weights.values())):
        if not any(mark in key and weights[key] == name for key in weights):
            continue
        head, base = safetensors_header(directory / name)
        for key, spec in head.items():
            if key == "__metadata__" or mark not in key:
                continue
            rows, width = spec["shape"]
            found.append({"name": key, "file": name, "rows": int(rows), "width": int(width),
                          "dtype": spec["dtype"], "offset": base + int(spec["data_offsets"][0])})
    marks = [re.search(r"shard_(\d+)", one["name"]) for one in found]
    if not found or not all(marks):
        raise RuntimeError(f"PLE_DF11: {directory} holds {len(found)} tensor(s) named like {mark}, not numbered shards")
    order = sorted(zip((int(one.group(1)) for one in marks), range(len(found))))
    if [one[0] for one in order] != list(range(len(found))):
        raise RuntimeError(f"PLE_DF11: the shard numbers of {directory} are not 0..{len(found) - 1}")
    shards, start = [], 0
    for _, at in order:
        one = dict(found[at])
        one["start"] = start
        start += one["rows"]
        shards.append(one)
    return shards


def shard_identity(shards):
    return [{key: one[key] for key in ("name", "start", "rows", "width", "dtype")} for one in shards]


def shard_fingerprint(shards):
    return hashlib.sha256(json.dumps(shard_identity(shards), sort_keys=True).encode("utf-8")).hexdigest()


def probe_rows(shards, seed=0, per_shard=DF11_CHECK_ROWS):
    rng = np.random.default_rng(seed)
    rows = []
    for one in shards:
        picks = {0, one["rows"] - 1}
        picks.update(int(rng.integers(0, one["rows"])) for _ in range(max(0, per_shard - len(picks))))
        rows.extend(one["start"] + at for at in sorted(picks))
    return rows


def raw_rows(directory, shards, global_rows):
    out = np.zeros((len(global_rows), shards[0]["width"]), dtype=np.uint16)
    handles = {}
    try:
        for at, row in enumerate(global_rows):
            one = next(item for item in shards if item["start"] <= row < item["start"] + item["rows"])
            handle = handles.get(one["file"])
            if handle is None:
                handle = handles[one["file"]] = open(Path(directory) / one["file"], "rb")
            handle.seek(one["offset"] + (row - one["start"]) * one["width"] * 2)
            out[at] = np.frombuffer(handle.read(one["width"] * 2), dtype="<u2")
    finally:
        for handle in handles.values():
            handle.close()
    return out


def row_digest(words):
    return hashlib.sha256(np.ascontiguousarray(words, dtype="<u2").tobytes()).hexdigest()


def stub_names(table):
    return {one["name"] for one in table.header["shards"]}


class WeightGate:
    def __init__(self, label):
        self.label = label
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.directory = None
        self.expected = set()
        self.closed = set()
        self.signal = None

    def release(self, signal):
        with self.lock:
            if self.event.is_set():
                return False
            self.signal = signal
            self.event.set()
            return True

    def opened(self, filename):
        path = Path(str(filename))
        with self.lock:
            if self.directory is None:
                self.directory = path.parent
                self.expected = {one.name for one in path.parent.glob("*.safetensors")} | {path.name}
            return path.parent == self.directory

    def finished(self, filename):
        path = Path(str(filename))
        with self.lock:
            if path.parent != self.directory:
                return
            self.closed.add(path.name)
            done = self.expected <= self.closed
        if done:
            self.release("every shard of the weight directory read")

    def wait(self, timeout):
        if self.event.wait(timeout):
            return self.signal
        self.release(f"no signal within {timeout} s")
        return self.signal


class StubbedOpen:
    def __init__(self, real, names, table, gate=None, filename=None):
        self.real = real
        self.names = names
        self.table = table
        self.gate = gate
        self.filename = filename
        self.counted = gate is not None and gate.opened(filename)
        self.stubbed = 0

    def get_tensor(self, name):
        if name not in self.names:
            return self.real.get_tensor(name)
        piece = self.real.get_slice(name)
        self.stubbed += 1
        self.table.stubbed += 1
        return torch.empty(tuple(piece.get_shape()), dtype=SAFETENSORS_DTYPES[piece.get_dtype()], device="meta")

    def __getattr__(self, name):
        return getattr(self.real, name)

    def __enter__(self):
        self.real.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self.real.__exit__(*exc)
        finally:
            if self.counted and exc[0] is None:
                self.gate.finished(self.filename)

    def __iter__(self):
        return iter(self.real)


class StubbedOpener:
    def __init__(self, original, names, table, gate=None):
        self.original = original
        self.names = names
        self.table = table
        self.gate = gate

    def __call__(self, *args, **kwargs):
        filename = args[0] if args else kwargs.get("filename")
        return StubbedOpen(self.original(*args, **kwargs), self.names, self.table, self.gate, filename)


def install_stub(table, gate=None):
    try:
        from vllm.model_executor.model_loader import weight_utils
    except Exception as failure:
        tell(f"{table.label}: DF11 stand-in not installed ({type(failure).__name__}: {failure}); the loader reads "
             f"the raw tensors and the table drops them")
        return False
    current = getattr(weight_utils, "safe_open", None)
    if current is None:
        tell(f"{table.label}: DF11 stand-in not installed: vLLM's weight_utils has no safe_open to wrap")
        return False
    if isinstance(current, StubbedOpener):
        current = current.original
    weight_utils.safe_open = StubbedOpener(current, stub_names(table), table, gate)
    return True


def verify_precomputed(table, directory, found):
    header = table.header
    mine = [(int(row), digest) for row, digest in header.get("probes", [])]
    if not mine:
        raise RuntimeError("the file carries no probe rows to check its own bytes against")
    ids = np.array([row for row, _ in mine], dtype=np.int64)
    got = table.decode(ids)
    for at, (row, digest) in enumerate(mine):
        if row_digest(got[at]) != digest:
            raise RuntimeError(f"row {row} decodes to bytes other than the ones recorded at the bake")
    ids = np.array(probe_rows(found, seed=int(time.time())), dtype=np.int64)
    if not np.array_equal(table.decode(ids), raw_rows(directory, found, ids.tolist())):
        raise RuntimeError("rows read from the checkpoint differ from the same rows decoded from the file")
    return len(mine), len(ids)


def load_precomputed_async(path, header, label, directory, found, began):
    threads = df11_read_threads()
    drop = df11_drop()
    table, head_end, keys, exps = df11_open(path, label=label, pinned=True)

    def background():
        df11_fill(path, table, head_end, keys, exps, threads, drop)
        try:
            probes, rows = verify_precomputed(table, directory, found)
        except Exception as failure:
            raise RuntimeError(f"PLE_DF11 {label}: the precomputed table at {path} failed its checks after the "
                               f"loader had already been given unread stand-ins for the raw rows, so there is no "
                               f"encode fallback ({type(failure).__name__}: {failure})") from failure
        tell(f"{label}: DF11 background checks agree: {probes} recorded probe rows and {rows} rows read back from "
             f"{directory}")

    try:
        installed = install_stub(table)
    except BaseException:
        table.close()
        raise
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-df11-fill")
    table.filler = pool.submit(background)
    pool.shutdown(wait=False)
    tell(f"{label}: DF11 path=precomputed (async): {path} ({sum(one.stat().st_size for one in df11_parts(path)) / 2**30:.2f}"
         f" GiB in {len(df11_parts(path))} part(s)) is filled by {threads} reader thread(s) beside the weight load "
         f"after {time.perf_counter() - began:.1f} s of header work; raw table reads "
         f"{'replaced by unread stand-ins' if installed else 'not replaced'}; the table is joined at its seal")
    return table


def load_precomputed(rows, width, dtype, label):
    path = find_df11_file()
    if path is None:
        if df11_startup_on():
            return load_startup(rows, width, dtype, label)
        tell(f"{label}: DF11 path=encode: no {DF11_FILE_NAME} under {DF11_ROOT} and {DF11_FILE_ENV} names none; "
             f"the table is encoded from the checkpoint as it loads")
        return None
    began = time.perf_counter()
    table = None
    try:
        directory = str(os.environ.get(DF11_CHECKPOINT_ENV) or "").strip()
        if not directory:
            raise RuntimeError(f"{DF11_CHECKPOINT_ENV} names no checkpoint to check the table against")
        header = read_df11_header(path)
        shards = header.get("shards")
        if not shards:
            raise RuntimeError("the file carries no shard list of the checkpoint it was baked from")
        if int(header["rows"]) != int(rows) or int(header["width"]) != int(width) or dtype != torch.bfloat16:
            raise RuntimeError(f"the file holds {header['rows']:,} x {header['width']} rows, the model asks for "
                               f"{int(rows):,} x {int(width)} {dtype}")
        found = checkpoint_shards(directory)
        if shard_identity(found) != shard_identity(shards):
            raise RuntimeError(f"the checkpoint at {directory} has {len(found)} PLE shard(s) that differ in name, "
                               f"start, rows, width or dtype from the {len(shards)} the file was baked from")
        if header.get("fingerprint") != shard_fingerprint(found):
            raise RuntimeError("the shard fingerprint in the file does not match its own shard list")
        if df11_async():
            return load_precomputed_async(path, header, label, directory, found, began)
        threads = df11_read_threads()
        if threads > 1:
            table, head_end, keys, exps = df11_open(path, label=label, pinned=True)
            df11_fill(path, table, head_end, keys, exps, threads, df11_drop())
        else:
            table = read_df11(path, label=label, pinned=True)
        probes, checked = verify_precomputed(table, directory, found)
        installed = install_stub(table)
        tell(f"{label}: DF11 path=precomputed: {path} ({sum(one.stat().st_size for one in df11_parts(path)) / 2**30:.2f}"
             f" GiB in {len(df11_parts(path))} part(s)) read in {time.perf_counter() - began:.1f} s; header, "
             f"{probes} recorded probe rows and {checked} rows read back from {directory} agree; raw table reads "
             f"{'replaced by unread stand-ins' if installed else 'not replaced'}")
        return table
    except Exception as failure:
        if table is not None:
            table.close()
        tell(f"{label}: DF11 path=encode FALLBACK after {time.perf_counter() - began:.1f} s: the precomputed table at "
             f"{path} was refused ({type(failure).__name__}: {failure}); the table is encoded from the checkpoint as "
             f"it loads")
        return None


def read_rows(path, offset, low, high, width):
    out = np.empty((high - low, 2 * width), dtype=np.uint8)
    with open(path, "rb") as handle:
        handle.seek(offset + low * 2 * width)
        view = memoryview(out).cast("B")
        done = 0
        while done < len(view):
            got = handle.readinto(view[done:])
            if not got:
                raise RuntimeError(f"{path} ended {len(view) - done:,} B early")
            done += got
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return out


def segments_of(shards, directory, first, last):
    found = []
    for one in shards:
        low = max(first, one["start"])
        high = min(last, one["start"] + one["rows"])
        if low < high:
            found.append((str(Path(directory) / one["file"]), one["offset"], low - one["start"], high - one["start"],
                          one["width"]))
    return found


def read_global(segments):
    parts = [read_rows(path, offset, low, high, width) for path, offset, low, high, width in segments]
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def sample_book(shards, directory, threads=1):
    total = sum(one["rows"] for one in shards)
    chunk = min(DF11_SAMPLE_CHUNK_ROWS, total)
    spans = [(total - chunk) * at // max(1, DF11_SAMPLE_CHUNKS - 1) for at in range(DF11_SAMPLE_CHUNKS)]

    def pick(first):
        return read_global(segments_of(shards, directory, first, first + chunk))

    if int(threads) > 1:
        with ThreadPoolExecutor(max_workers=int(threads), thread_name_prefix="ple-df11-sample") as pool:
            picks = list(pool.map(pick, spans))
    else:
        picks = [pick(first) for first in spans]
    data = np.concatenate(picks)
    return Df11Book.from_rows(data, sample=data.shape[0]), data


def df11_startup_on():
    return df11_flag(DF11_STARTUP_ENV)


def startup_encoder(table):
    said = str(os.environ.get(DF11_STARTUP_ENCODER_ENV) or "").strip().lower() or "auto"
    if said not in DF11_STARTUP_ENCODERS:
        raise ValueError(f"{DF11_STARTUP_ENCODER_ENV} must be one of {list(DF11_STARTUP_ENCODERS)}, found {said!r}")
    if said == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{DF11_STARTUP_ENCODER_ENV}=cuda but no CUDA device is present")
    if said == "auto":
        return "cuda" if table.device.type == "cuda" else "cpu"
    return said


_STARTUP = {}


def startup_init(lengths, limit):
    torch.set_num_threads(1)
    _STARTUP["book"] = Df11Book(np.asarray(lengths, dtype=np.int64), limit)


def startup_encode(task):
    first, width, segments = task
    data = read_global(segments)
    return first, data.shape[0], _STARTUP["book"].encode(data, first, width, DF11_GROUP, "cpu")


def startup_tasks(shards, directory, rows, width, task_rows):
    return [(first, width, segments_of(shards, directory, first, min(first + task_rows, rows)))
            for first in range(0, rows, task_rows)]


def startup_order():
    said = str(os.environ.get(DF11_STARTUP_ORDER_ENV) or "").strip().lower() or DF11_STARTUP_ORDER_DEFAULT
    if said not in DF11_STARTUP_ORDERS:
        raise ValueError(f"{DF11_STARTUP_ORDER_ENV} must be one of {list(DF11_STARTUP_ORDERS)}, found {said!r}")
    return said


def startup_wait(table, gate, timeout):
    began = time.perf_counter()
    signal = gate.wait(timeout)
    tell(f"{table.label}: DF11 startup reader released after {time.perf_counter() - began:.1f} s ({signal})")


def startup_fill(table, directory, shards, *, threads, procs, encoder, task_rows=None):
    began = time.perf_counter()
    task_rows = task_rows or DF11_TASK_ROWS
    rows, width = table.rows, table.width
    book, _ = sample_book(shards, directory, threads)
    with table.lock:
        table._open(book)
    sampled = time.perf_counter() - began
    tasks = startup_tasks(shards, directory, rows, width, task_rows)
    progress = [0, began]
    if encoder == "cpu" and procs > 0:
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        with context.Pool(procs, initializer=startup_init, initargs=(book.lengths.tolist(), book.limit)) as pool:
            for first, count, encoded in pool.imap(startup_encode, tasks):
                table._place(first, count, encoded)
                table.done += count
                startup_tell(table, progress, began)
    else:
        device = table.device if encoder == "cuda" else torch.device("cpu")
        ahead = max(2, int(threads) * 2)
        with ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="ple-df11-startup-read") as pool:
            queued = []
            for first, _, segments in tasks:
                while len(queued) >= ahead:
                    startup_drain(table, queued, device, progress, began)
                queued.append((first, pool.submit(read_global, segments)))
            while queued:
                startup_drain(table, queued, device, progress, began)
    table.escapes.seal()
    table.seen = {0: rows}
    table.done = rows
    return sampled


def startup_drain(table, queued, device, progress, began):
    first, pending = queued.pop(0)
    data = pending.result()
    if device.type == "cuda":
        with table.device_lock:
            encoded = table.book.encode(data, first, table.width, table.group, device)
    else:
        encoded = table.book.encode(data, first, table.width, table.group, device)
    table._place(first, data.shape[0], encoded)
    table.done += data.shape[0]
    startup_tell(table, progress, began)


def startup_tell(table, progress, began):
    now = time.perf_counter()
    raw = table.done * 2 * table.width
    if raw >= progress[0] + PROGRESS_EVERY or (now - progress[1] >= 30.0 and table.done < table.rows):
        progress[0] = raw // PROGRESS_EVERY * PROGRESS_EVERY
        progress[1] = now
        tell(f"{table.label}: DF11 startup encoded {table.done:,} of {table.rows:,} rows ({table.done / table.rows:.1%}), "
             f"{table.used / 2**30:.1f} GiB of exponent codes, {table.escapes.count:,} escaped exponents, "
             f"{(now - began) / 60:.1f} min spent")


def startup_check(table, directory, shards, count=DF11_STARTUP_CHECK_ROWS):
    ids = np.array(probe_rows(shards, seed=int(time.time())), dtype=np.int64)
    ids = np.unique(np.concatenate([ids, np.random.default_rng(3).integers(0, table.rows, size=count)]))
    got = df11_decode(table.book, table.signs, table.lengths, table.bases, table.stream[: table.used],
                      table.escapes.keys, table.escapes.exps, table.group, ids)
    if not np.array_equal(got, raw_rows(directory, shards, ids.tolist())):
        raise RuntimeError("rows decoded from the table encoded at startup differ from the checkpoint")
    return int(ids.shape[0])


def load_startup(rows, width, dtype, label):
    began = time.perf_counter()
    table = None
    try:
        directory = str(os.environ.get(DF11_CHECKPOINT_ENV) or "").strip()
        if not directory:
            raise RuntimeError(f"{DF11_CHECKPOINT_ENV} names no checkpoint to encode the table from")
        if dtype != torch.bfloat16:
            raise RuntimeError(f"the model asks for {dtype} rows, the DF11 encoder takes BF16")
        found = checkpoint_shards(directory)
        held = sum(one["rows"] for one in found)
        if held != int(rows) or found[0]["width"] != int(width):
            raise RuntimeError(f"the checkpoint holds {held:,} x {found[0]['width']} rows, the model asks for "
                               f"{int(rows):,} x {int(width)}")
        if any(one["dtype"] != "BF16" or one["width"] != found[0]["width"] for one in found):
            raise RuntimeError("the PLE shards are not all BF16 of one width")
        order = startup_order()
        timeout = integer(DF11_STARTUP_ORDER_TIMEOUT_ENV, DF11_STARTUP_ORDER_TIMEOUT_DEFAULT, 0, 86400)
        threads = integer(DF11_STARTUP_THREADS_ENV, DF11_STARTUP_THREADS_DEFAULT, 1, 128)
        procs = integer(DF11_STARTUP_PROCS_ENV, DF11_STARTUP_PROCS_DEFAULT, 0, 64)
        table = Df11Table(rows=rows, width=width, dtype=dtype, label=label, pinned=True, dummy=False)
        encoder = startup_encoder(table)
        table.header = {"shards": shard_identity(found)}
        table.preloaded = True
        table.expected = {int(one["start"]): int(one["rows"]) for one in found}

        gate = WeightGate(label) if order == "after_weights" else None
        table.gate = gate

        def background():
            if gate is not None:
                startup_wait(table, gate, timeout)
            sampled = startup_fill(table, directory, found, threads=threads, procs=procs, encoder=encoder)
            checked = startup_check(table, directory, found)
            tell(f"{label}: DF11 startup encode finished in {time.perf_counter() - began:.1f} s on {encoder} "
                 f"({sampled:.1f} s of it the codebook sample); {checked} rows decoded from the table agree with "
                 f"{directory}")

        installed = install_stub(table, gate)
        if gate is not None and not installed:
            gate.release("no weight-load stand-in to signal from")
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-df11-startup")
        table.filler = pool.submit(background)
        pool.shutdown(wait=False)
        by = f" by {procs} process(es)" if encoder == "cpu" and procs else ""
        tell(f"{label}: DF11 path=startup: no precomputed table; the checkpoint's {len(found)} PLE shards are read by "
             f"{threads} thread(s) and encoded on {encoder}{by} {'after the weight load (order=after_weights)' if gate is not None else 'beside the weight load'}; raw table reads "
             f"{'replaced by unread stand-ins' if installed else 'not replaced'}; the table is joined at its seal, "
             f"which an eager step runs before any request is served")
        return table
    except Exception as failure:
        if table is not None:
            table.close()
        tell(f"{label}: DF11 path=encode FALLBACK after {time.perf_counter() - began:.1f} s: the startup encode was "
             f"refused ({type(failure).__name__}: {failure}); the table is encoded from the checkpoint as it loads")
        return None


def full_graph_needs_breaks():
    return bool(mode())


_SEGMENTED = {}


def segmented_class():
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    made = _SEGMENTED.get(BreakableCUDAGraphCapture)
    if made is None:
        class SegmentedCapture(BreakableCUDAGraphCapture):
            def add_eager(self, fn):
                return fn()

            def add_break(self, fn):
                return super().add_eager(fn)

        made = _SEGMENTED[BreakableCUDAGraphCapture] = SegmentedCapture
    return made


def segmented_capture(pool=None):
    return segmented_class()(pool=pool)


def hosts_hot_rows(manager):
    return "spec_decode" not in type(manager).__module__


def new_graph(pool=None, manager=None):
    import torch

    if full_graph_needs_breaks() and hosts_hot_rows(manager) and not preresolve():
        say("every full CUDA graph is captured as segments around the host resolve of its rows; attention stays "
            "inside the segments and the replay is one call")
        return segmented_capture(pool)
    return torch.cuda.CUDAGraph()


class Plan:
    __slots__ = ("engine", "compute", "input_ids", "query_start_loc", "context", "n", "slots", "captured_ids",
                 "label")

    def __init__(self, engine, compute, input_ids, query_start_loc, context, n, slots, captured_ids, label):
        self.engine = engine
        self.compute = compute
        self.input_ids = input_ids
        self.query_start_loc = query_start_loc
        self.context = context
        self.n = n
        self.slots = slots
        self.captured_ids = captured_ids
        self.label = label


@contextlib.contextmanager
def _recording(graph, pool, stream):
    import torch

    plans = []
    graph._taaf_plans = plans
    _PRE["plans"] = plans
    try:
        with torch.cuda.graph(graph, pool, stream=stream):
            yield
    finally:
        _PRE["plans"] = None
    if plans and not _PRE["told"]:
        _PRE["told"] = True
        say(f"pre-resolve: a captured graph holds {len(plans)} PLE layers whose rows are resolved on the host "
            f"before each replay ({plans[0].n} lookups per layer); the graph itself only gathers rows from "
            "the persistent slot buffers")


def capturing(graph, pool, stream):
    import torch

    if isinstance(graph, torch.cuda.CUDAGraph):
        if preresolve():
            return _recording(graph, pool, stream)
        return torch.cuda.graph(graph, pool, stream=stream)
    return graph


def _stage(plans):
    import torch

    staged = []
    for plan in plans:
        ids = plan.compute(plan.input_ids, plan.query_start_loc, plan.context)
        if ids.numel() != plan.n:
            raise RuntimeError(f"PLE_HOT pre-resolve: layer {plan.label} produced {ids.numel()} lookups, "
                               f"its graph was captured for {plan.n}")
        plan.engine.host_ids[:plan.n].copy_(ids.reshape(-1), non_blocking=True)
        staged.append(ids)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    event.synchronize()
    futures = [plan.engine.runner.submit(plan.engine.resolve_now, plan.n) for plan in plans]
    for future in futures:
        future.result()
    for plan in plans:
        plan.slots.copy_(plan.engine.host_slots[:plan.n], non_blocking=True)
    return staged


def _check(plans, staged):
    import torch

    for plan, ids in zip(plans, staged):
        if not torch.equal(plan.captured_ids, ids):
            raise RuntimeError(f"PLE_HOT pre-resolve: the n-gram ids layer {plan.label} computed inside the graph "
                               "differ from the ids resolved before the replay; its inputs are not the "
                               "persistent buffers the graph reads")
    _PRE["checked"] += 1
    say(f"pre-resolve check {_PRE['checked']} of {PRE_CHECKS}: the n-gram ids of {len(plans)} layers computed "
        "inside the replayed graph equal the ids resolved before it")


def replay(graph):
    plans = getattr(graph, "_taaf_plans", None)
    if not plans:
        graph.replay()
        return
    began = time.perf_counter()
    staged = _stage(plans)
    _PRE["seconds"] += time.perf_counter() - began
    graph.replay()
    _PRE["replays"] += 1
    if _PRE["checked"] < PRE_CHECKS:
        _check(plans, staged)
    if _PRE["replays"] % PRE_TELL_EVERY == 1:
        say(f"pre-resolve: {_PRE['replays']} replays, {_PRE['seconds'] / _PRE['replays'] * 1e3:.2f} ms of "
            f"host resolve before each, {len(plans)} layers in the last")


def _break_tools():
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.utils.torch_utils import weak_ref_tensor
    return BreakableCUDAGraphCapture, weak_ref_tensor


def ple_break(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        capture_class, weak = _break_tools()
        capture = capture_class.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)
        weak_args = tuple(weak(a) for a in args)
        weak_kwargs = {k: weak(v) for k, v in kwargs.items()}
        return getattr(capture, "add_break", capture.add_eager)(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper


def hot_class(base):
    made = _CLASSES.get(base)
    if made is not None:
        return made
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CompilationMode, get_current_vllm_config

    config = settings()
    vllm_config = get_current_vllm_config()
    if vllm_config.compilation_config.mode != CompilationMode.NONE:
        raise RuntimeError(f"{HOT_ENV} resolves PLE rows on the host each step, which torch.compile cannot trace; "
                           f"it runs with breakable CUDA graphs or eager mode only")

    class Qwen4ExpPLEHotEmbedding(base):
        def __init__(self, *args, **kwargs):
            self._taaf_prefix = str(kwargs.get("prefix", ""))
            super().__init__(*args, **kwargs)
            if self.tp_size != 1 or self.etp_data_parallel_size != 1:
                raise RuntimeError(f"{HOT_ENV} serves one GPU; this PLE table is split across {self.tp_size} ranks")
            lookups = int(kwargs.get("max_total_tokens", 0)) * int(kwargs.get("num_ngram_heads", 1))
            if lookups <= 0:
                raise RuntimeError(f"{HOT_ENV} needs max_total_tokens to size its staging buffers")
            if config["dummy"]:
                self.weight.dummy_weight_value = 0
            self._taaf_engine = self._taaf_hot.arm(lookups, True)
            self._taaf_slots = torch.empty(lookups, dtype=torch.int64, device=self._uva_weight.device)
            self._taaf_event = torch.cuda.Event()

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            model = get_current_vllm_config().model_config
            parts = int(getattr(model.hf_text_config, "split_ngram_parts", 512))
            prefix = self._taaf_prefix
            tail = prefix[prefix.find("layers."):] if "layers." in prefix else prefix
            self._taaf_hot = Store(rows=num_embeddings, width=embedding_dim, dtype=dtype, label=prefix,
                                   model_dir=model.model, tail=tail, parts=parts, config=config, pinned=True)
            return self._taaf_hot.weight

        def weight_loader(self, param, loaded_weight, checkpoint_start=None):
            if checkpoint_start is None:
                raise RuntimeError(f"{HOT_ENV} loads the PLE table only from checkpoint row shards")
            self._taaf_hot.put_shard(checkpoint_start, loaded_weight, self.shard_indices.org_vocab_start_index,
                                     self.shard_indices.org_vocab_end_index)

        taaf_source = None

        @ple_break
        def start_prefetch(self, hidden_states, ngram_ids):
            if torch.cuda.is_current_stream_capturing() and _PRE["plans"] is not None:
                self._taaf_hot.ready()
                source = self.taaf_source
                if source is None:
                    raise RuntimeError(f"{PRERESOLVE_ENV} needs the n-gram inputs of the layer, and the "
                                       "ngram_embedding edit that hands them over did not run")
                compute, input_ids, query_start_loc, context = source
                n = ngram_ids.numel()
                shape = tuple(ngram_ids.shape)
                stream = self._prefetch_stream
                stream.wait_stream(torch.cuda.current_stream())
                slots = self._taaf_slots[:n]
                with torch.cuda.stream(stream):
                    self._lookup(slots.view(shape), output=self._prefetch_buffer[: shape[0]])
                _PRE["plans"].append(Plan(self._taaf_engine, compute, input_ids, query_start_loc, context, n,
                                          slots, ngram_ids, self._taaf_prefix))
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{HOT_ENV} resolves PLE rows on the host every step, so its prefetch cannot sit "
                                   f"inside a plain torch.cuda.graph capture; capture the step with "
                                   f"BreakableCUDAGraphCapture so the prefetch and its finalize run as eager breaks")
            self._taaf_hot.ready()
            engine = self._taaf_engine
            shape = tuple(ngram_ids.shape)
            n = ngram_ids.numel()
            stream = self._prefetch_stream
            event = self._taaf_event
            stream.wait_stream(torch.cuda.current_stream())
            ngram_ids.record_stream(stream)
            with torch.cuda.stream(stream):
                engine.host_ids[:n].copy_(ngram_ids.reshape(-1), non_blocking=True)
                event.record(stream)
            slots = self._taaf_slots[:n]
            output = self._prefetch_buffer[: shape[0]]

            def launch():
                with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                    slots.copy_(engine.host_slots[:n], non_blocking=True)
                    self._lookup(slots.view(shape), output=output)

            engine.submit(n, event.synchronize, launch, sync=BreakableCUDAGraphCapture.current() is not None)

        @ple_break
        def _finalize_prefetch(self, prefetch_output, output):
            if not torch.cuda.is_current_stream_capturing():
                self._taaf_engine.wait()
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
            active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
            embeddings = self._reduce_etp_embeddings(active_output)
            embeddings = self._select_embeddings(embeddings, output.shape[0], slot_offset)
            output.copy_(embeddings.flatten(-2))

    _CLASSES[base] = Qwen4ExpPLEHotEmbedding
    return Qwen4ExpPLEHotEmbedding


def packed_class(base):
    made = _PACKED_CLASSES.get(base)
    if made is not None:
        return made

    class Qwen4ExpPLEPackedEmbedding(base):
        def __init__(self, *args, **kwargs):
            self._taaf_prefix = str(kwargs.get("prefix", ""))
            super().__init__(*args, **kwargs)
            if self.tp_size != 1 or self.etp_data_parallel_size != 1:
                raise RuntimeError(f"{PACKED_ENV} serves one GPU; this PLE table is split across {self.tp_size} ranks")
            if dummy_load():
                self.weight.dummy_weight_value = 0

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            kind = qstore()
            if kind:
                self._taaf_packed = (load_quantized(num_embeddings, embedding_dim, dtype, self._taaf_prefix, kind)
                                     if df11() else None)
                if self._taaf_packed is None:
                    self._taaf_packed = QTable(rows=num_embeddings, width=embedding_dim, dtype=dtype,
                                               label=self._taaf_prefix, pinned=True, dummy=dummy_load(), kind=kind)
                return self._taaf_packed.stand_in
            self._taaf_packed = load_precomputed(num_embeddings, embedding_dim, dtype, self._taaf_prefix)                 if df11() and not dummy_load() else None
            if self._taaf_packed is not None:
                return self._taaf_packed.stand_in
            table_class = Df11Table if df11() else PackedTable
            self._taaf_packed = table_class(rows=num_embeddings, width=embedding_dim, dtype=dtype,
                                            label=self._taaf_prefix, pinned=True, dummy=dummy_load())
            return self._taaf_packed.stand_in

        def weight_loader(self, param, loaded_weight, checkpoint_start=None):
            if checkpoint_start is None:
                raise RuntimeError(f"{PACKED_ENV} loads the PLE table only from checkpoint row shards")
            self._taaf_packed.put_shard(checkpoint_start, loaded_weight, self.shard_indices.org_vocab_start_index,
                                        self.shard_indices.org_vocab_end_index)

        def _lookup(self, input_ids, output=None):
            expected_shape = (*input_ids.shape, self.embedding_dim)
            if output is None:
                output = torch.empty(expected_shape, dtype=self.weight.dtype, device=input_ids.device)
            elif (tuple(output.shape) != expected_shape or output.dtype != self.weight.dtype
                  or output.device != input_ids.device or not output.is_contiguous()):
                raise ValueError("PLE prefetch output must be contiguous and match the input shape, weight dtype, "
                                 "and input device")
            table = self._taaf_packed
            table.ready(input_ids.device)
            return table.gather(input_ids.reshape(-1), output, self.shard_indices.org_vocab_start_index,
                                self.shard_indices.org_vocab_end_index)

    _PACKED_CLASSES[base] = Qwen4ExpPLEPackedEmbedding
    return Qwen4ExpPLEPackedEmbedding


def embedding_class(chosen, pinned_host):
    whole = packed()
    if chosen is not pinned_host or not (whole or mode()):
        return chosen
    return packed_class(pinned_host) if whole else hot_class(pinned_host)


QHOT_ENV = "TAAF_VLLM_PLE_QHOT"
QHOT_FILE_ENV = "TAAF_VLLM_PLE_QHOT_FILE"
QHOT_FILE = "taaf_ple_qhot_rows.npy"
QSTORE_ENV = "TAAF_VLLM_PLE_QSTORE"
QSTORE_LEVELS = {"int8": 127.0, "int6": 31.0}
QSTORE_GROUP = 64
QSTORE_TINY = 2.0 ** -24
QSTORE_CHUNK_ENV = "TAAF_VLLM_PLE_QSTORE_CHUNK_ROWS"
QSTORE_CHUNK_ROWS = 1 << 21
QSTORE_SLICE_ROWS = 1 << 16


def qstore():
    said = str(os.environ.get(QSTORE_ENV) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return ""
    if said not in QSTORE_LEVELS:
        raise ValueError(f"{QSTORE_ENV} must be unset, off, {' or '.join(QSTORE_LEVELS)}, found {said!r}")
    return said


def q_group(width):
    return QSTORE_GROUP if int(width) % QSTORE_GROUP == 0 else int(width)


def q_row_bytes(width, kind):
    if kind == "int6":
        if int(width) % 4:
            raise ValueError(f"6-bit codes pack four columns into three bytes; a row of {width} does not split so")
        return int(width) * 3 // 4
    return int(width)


def q_table_bytes(rows, width, kind):
    return int(rows) * (q_row_bytes(width, kind) + 2 * (int(width) // q_group(width)))


def q_step(kind):
    return float(np.float32(1.0) / np.float32(QSTORE_LEVELS[kind]))


def q_reference(data, kind):
    rows, width = data.shape
    group = q_group(width)
    held = data.float().reshape(rows, width // group, group)
    levels = QSTORE_LEVELS[kind]
    scale = held.abs().amax(dim=-1, keepdim=True).clamp(min=QSTORE_TINY) * q_step(kind)
    return (torch.round(held / scale).clamp(-levels, levels) * scale).reshape(rows, width).to(data.dtype)


def q_encode(data, kind):
    rows, width = data.shape
    group = q_group(width)
    levels = QSTORE_LEVELS[kind]
    held = data.float().reshape(rows, width // group, group)
    peak = held.abs().amax(dim=-1, keepdim=True)
    if not bool(torch.isfinite(peak).all()):
        raise RuntimeError(f"{QSTORE_ENV}: a PLE row holds a value that is not finite")
    level = torch.round(held / (peak.clamp(min=QSTORE_TINY) * q_step(kind))).clamp(-levels, levels).reshape(rows, width)
    signed_zero = (level == 0) & torch.signbit(level)
    level = level.to(torch.int32)
    if kind == "int6":
        code = torch.where(signed_zero, 0, level + 32).reshape(rows, width // 4, 4)
        word = code[..., 0] | (code[..., 1] << 6) | (code[..., 2] << 12) | (code[..., 3] << 18)
        packed = torch.stack((word & 255, (word >> 8) & 255, (word >> 16) & 255), dim=-1)
        codes = packed.reshape(rows, width * 3 // 4).to(torch.uint8)
    else:
        codes = torch.where(signed_zero, -128, level).to(torch.int8).view(torch.uint8)
    peaks = peak.reshape(rows, width // group).to(torch.bfloat16).view(torch.int16)
    return codes, peaks


def q_decode(codes, peaks, kind, width):
    rows = codes.shape[0]
    group = q_group(width)
    levels = QSTORE_LEVELS[kind]
    if kind == "int6":
        raw = codes.to(torch.int32).reshape(rows, width // 4, 3)
        word = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
        code = torch.stack([(word >> (6 * at)) & 63 for at in range(4)], dim=-1).reshape(rows, width)
        signed_zero = code == 0
        level = (code - 32).float()
    else:
        code = codes.view(torch.int8).to(torch.int32)
        signed_zero = code == -128
        level = code.float()
    scale = peaks.view(torch.bfloat16).float().clamp(min=QSTORE_TINY) * q_step(kind)
    value = (level.reshape(rows, width // group, group) * scale[..., None]).reshape(rows, width).to(torch.bfloat16)
    bits = value.view(torch.int16).clone()
    bits[signed_zero] = -32768
    return bits.view(torch.bfloat16)


def q_scales(kind, device):
    peak = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().to(device)
    return (peak.clamp(min=QSTORE_TINY) * q_step(kind)).contiguous()


class QTable:
    def __init__(self, *, rows, width, dtype, label, pinned, dummy, kind):
        if dtype != torch.bfloat16:
            raise RuntimeError(f"{QSTORE_ENV} quantizes BF16 rows; this table is {dtype}")
        began = time.perf_counter()
        self.kind = kind
        self.rows = int(rows)
        self.width = int(width)
        self.dtype = dtype
        self.label = label
        self.group = q_group(self.width)
        self.groups = self.width // self.group
        self.row_bytes = q_row_bytes(self.width, kind)
        self.codes_size = self.rows * self.row_bytes
        self.peaks_at = ceil_to(self.codes_size, 8)
        self.size = self.peaks_at + self.rows * self.groups * 2
        self.region, flat = private_region(self.size)
        self.host = torch.from_numpy(flat[: self.size])
        self.codes = flat[: self.codes_size].reshape(self.rows, self.row_bytes)
        self.peaks = flat[self.peaks_at: self.size].view(np.int16).reshape(self.rows, self.groups)
        self.codes_t = self.host[: self.codes_size]
        if kind == "int8":
            self.codes_t = self.codes_t.view(torch.int8)
        self.peaks_t = self.host[self.peaks_at: self.size].view(torch.int16)
        self.pinned = False
        if pinned:
            result = torch.cuda.cudart().cudaHostRegister(self.host.data_ptr(), self.size, 0)
            if result.value != 0:
                raise RuntimeError(f"{QSTORE_ENV}: cudaHostRegister of the {self.size / 2**30:.2f} GiB table failed: "
                                   f"{result}")
            if not self.host.is_pinned():
                raise RuntimeError(f"{QSTORE_ENV}: CUDA does not report the registered table as pinned")
            self.pinned = True
        self.device = torch.device("cuda") if pinned and torch.cuda.is_available() else torch.device("cpu")
        self.stand_in = torch.zeros((1, self.width), dtype=dtype, device="cpu", pin_memory=bool(pinned))
        self.lock = threading.Lock()
        self.seen = {}
        self.done = 0
        self.built = False
        self.scales = None
        self.ops = None
        self.told = False
        self.preloaded = False
        self.expected = None
        self.claimed = {}
        self.real_tensors = 0
        self.stubbed = 0
        self.header = None
        self.filler = None
        self.dummy = bool(dummy)
        if self.dummy:
            self._dummy_fill()
        tell(f"{label}: QSTORE {kind}: {self.rows:,} rows x {2 * self.width} B = "
             f"{self.rows * 2 * self.width / 2**30:.2f} GiB raw are held as {self.size / 2**30:.2f} GiB "
             f"({self.row_bytes} B of codes and {2 * self.groups} B of BF16 group peaks a row, one peak per "
             f"{self.group} columns), {'registered with CUDA' if self.pinned else 'unregistered'}, in "
             f"{time.perf_counter() - began:.1f} s; {memory_text()}")

    def place(self, first, data):
        codes, peaks = q_encode(data, self.kind)
        count = codes.shape[0]
        torch.from_numpy(self.codes[first: first + count]).copy_(codes)
        torch.from_numpy(self.peaks[first: first + count]).copy_(peaks)

    def _dummy_fill(self):
        base = RowCodec(self.width).decode(dummy_records(self.width), np.arange(DUMMY_BASE_ROWS), EscapeLog())
        self.place(0, torch.from_numpy(base.view(np.int16)).view(torch.bfloat16))
        tile = base.shape[0]
        for at in range(tile, self.rows, tile):
            end = min(at + tile, self.rows)
            self.codes[at:end] = self.codes[: end - at]
            self.peaks[at:end] = self.peaks[: end - at]
        self.seen = {0: self.rows}
        self.done = self.rows

    def put_shard(self, start, tensor, tp_start, tp_end):
        if self.built:
            raise RuntimeError(f"{QSTORE_ENV} {self.label}: a shard arrived after the table was sealed")
        count = int(tensor.shape[0])
        low = max(int(start), int(tp_start))
        high = min(int(start) + count, int(tp_end))
        if low >= high:
            return 0
        if self.preloaded:
            return self._claim(int(start), count, bool(getattr(tensor, "is_meta", False)), high - low)
        first = low - int(tp_start)
        with self.lock:
            if first in self.seen:
                raise RuntimeError(f"{QSTORE_ENV} {self.label}: rows from {first} arrived twice")
            if first + high - low > self.rows:
                raise RuntimeError(f"{QSTORE_ENV} {self.label}: rows {first}-{first + high - low} fall past the "
                                   f"{self.rows:,}-row table")
            self.seen[first] = high - low
        part = tensor.narrow(0, low - int(start), high - low)
        for at in range(0, high - low, QSTORE_SLICE_ROWS):
            self.place(first + at, part[at: at + QSTORE_SLICE_ROWS].to(device=self.device, dtype=self.dtype))
        with self.lock:
            self.done += high - low
        return high - low

    def _claim(self, start, count, stub, taken):
        with self.lock:
            if self.expected is not None and self.expected.get(start) != count:
                raise RuntimeError(f"{QSTORE_ENV} {self.label}: the loader placed {count:,} rows at row {start:,} and "
                                   f"the precomputed table holds no such shard")
            if start in self.claimed:
                raise RuntimeError(f"{QSTORE_ENV} {self.label}: rows from {start} arrived twice")
            self.claimed[start] = count
            if not stub:
                self.real_tensors += 1
        return taken

    def await_fill(self):
        pending, self.filler = self.filler, None
        if pending is None:
            return
        began = time.perf_counter()
        hook = getattr(self, "on_wait", None)
        if hook is not None and not pending.done():
            hook()
        while True:
            try:
                pending.result(timeout=DF11_STALL_S)
                break
            except FutureTimeout:
                if hook is not None:
                    hook()
                tell(f"{self.label}: QSTORE seal still waits for the background conversion after "
                     f"{time.perf_counter() - began:.0f} s: {self.done:,} of {self.rows:,} rows placed; {memory_text()}")
        tell(f"{self.label}: QSTORE background conversion joined, the load waited {time.perf_counter() - began:.1f} s "
             f"for it")

    def seal(self, device):
        if self.built:
            return
        self.await_fill()
        began = time.perf_counter()
        if self.preloaded and self.expected is not None and self.claimed != self.expected:
            lacking = sorted(set(self.expected) - set(self.claimed))[:4]
            raise RuntimeError(f"{QSTORE_ENV} {self.label}: the loader delivered {len(self.claimed)} of the "
                               f"{len(self.expected)} shards (missing rows from {lacking})")
        with self.lock:
            covered = 0
            for first in sorted(self.seen):
                if first != covered:
                    raise RuntimeError(f"{QSTORE_ENV} {self.label}: rows {covered}-{first} never arrived")
                covered += self.seen[first]
            if covered != self.rows:
                raise RuntimeError(f"{QSTORE_ENV} {self.label}: {covered:,} of {self.rows:,} rows arrived")
            self.scales = q_scales(self.kind, device)
            self.ops = packed_ops()
            self.hot = self.build_hot(device)
            self.built = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tell(f"{self.label}: QSTORE {self.kind} sealed in {time.perf_counter() - began:.1f} s: {self.rows:,} rows in "
             f"{self.size / 2**30:.2f} GiB ({self.size / self.rows:.2f} B a row) of "
             f"{self.rows * 2 * self.width / 2**30:.2f} GiB raw; {memory_text()}")

    def ready(self, device):
        if self.built:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{QSTORE_ENV} {self.label}: the table must be sealed by an eager step before any CUDA "
                               f"graph capture")
        self.seal(device)

    def decode(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        inside = (ids >= 0) & (ids < self.rows)
        row = np.where(inside, ids, 0)
        out = q_decode(torch.from_numpy(self.codes[row].copy()), torch.from_numpy(self.peaks[row].copy()),
                       self.kind, self.width).view(torch.int16).numpy().view(np.uint16).copy()
        out[~inside] = 0
        return out

    def build_hot(self, device):
        said = str(os.environ.get(QHOT_ENV) or "").strip()
        if said in ("", "0") or not torch.cuda.is_available():
            return None
        if not said.isdigit() or int(said) <= 0:
            raise RuntimeError(f"{QHOT_ENV}={said!r} is not a positive row count")
        path = Path(os.environ.get(QHOT_FILE_ENV) or Path(__file__).with_name(QHOT_FILE))
        ranked = np.load(path).astype(np.int64)
        ranked = ranked[(ranked >= 0) & (ranked < self.rows)][: int(said)]
        if not ranked.size:
            return None
        capacity = 1 << int(4 * ranked.size - 1).bit_length()
        table, slot_of = self.ops.qhot_table(ranked, capacity)
        placed = int((slot_of >= 0).sum())
        hot_codes = torch.from_numpy(np.ascontiguousarray(self.codes[ranked])).to(device)
        hot_peaks = torch.from_numpy(np.ascontiguousarray(self.peaks[ranked])).to(device)
        size = hot_codes.numel() + hot_peaks.numel() * 2 + capacity * 8
        tell(f"{self.label}: PLE_QHOT holds {placed:,} of {ranked.size:,} ranked rows on the GPU "
             f"({size / 2**20:.0f} MiB with a {capacity:,}-slot table); misses read the pinned {self.kind} table")
        return torch.from_numpy(table).to(device), hot_codes, hot_peaks

    def gather(self, ids, out, start, end):
        if self.ops is None:
            self.ops = packed_ops()
        if getattr(self, "hot", None) is not None:
            self.ops.q_gather_hot(self.codes_t, self.peaks_t, self.scales, ids, out, start, end, self.row_bytes,
                                  self.group, self.kind == "int6", self.hot)
        else:
            self.ops.q_gather(self.codes_t, self.peaks_t, self.scales, ids, out, start, end, self.row_bytes,
                              self.group, self.kind == "int6")
        if not self.told:
            self.told = True
            tell(f"live: {self.label}: the first QSTORE gather dequantized {ids.numel():,} lookups on the GPU straight "
                 f"from the pinned {self.kind} table; no host work runs per step")
        return out

    def close(self):
        pending, self.filler = self.filler, None
        if pending is not None:
            try:
                pending.result()
            except BaseException:
                pass
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
            self.pinned = False


class Df11Chunks:
    def __init__(self, path, header, head_end, threads, drop):
        self.header = header
        self.rows = int(header["rows"])
        self.width = int(header["width"])
        self.group = int(header["group"])
        self.used = int(header["used"])
        self.size = int(header["size"])
        self.count = int(header["escapes"])
        self.head_end = int(head_end)
        self.lengths_at = self.rows * self.width
        self.bases_at = ceil_to(self.lengths_at + self.rows, 8)
        self.files = PartFiles(df11_parts(path), drop=drop)
        expected = self.head_end + self.size + self.used + 9 * self.count
        if self.files.size != expected:
            raise RuntimeError(f"PLE_DF11: {path} holds {self.files.size:,} B where its header lays out {expected:,}")
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="ple-qstore-read")
        groups = -(-self.rows // self.group)
        self.bases = np.empty(groups, dtype=np.int64)
        self.keys = np.empty(self.count, dtype="<i8")
        self.exps = np.empty(self.count, dtype=np.uint8)
        tail = self.head_end + self.size + self.used
        self._read([(self.head_end + self.bases_at, self.bases), (tail, self.keys), (tail + 8 * self.count, self.exps)])
        self.bytes_read = self.bases.nbytes + self.keys.nbytes + self.exps.nbytes

    def _read(self, spans):
        jobs = []
        for offset, array in spans:
            flat = memoryview(array).cast("B")
            for first in range(0, flat.nbytes, DF11_READ_PIECE):
                jobs.append((offset + first, flat[first: first + DF11_READ_PIECE]))
        return [self.pool.submit(self.files.read_into, *job) for job in jobs]

    def submit(self, first, end):
        lengths = np.empty(end - first, dtype=np.uint8)
        for job in self._read([(self.head_end + self.lengths_at + first, lengths)]):
            job.result()
        head = first // self.group
        sizes = np.add.reduceat(lengths.astype(np.int64), np.arange(0, end - first, self.group))
        starts = self.bases[head: head + sizes.shape[0]]
        local = np.zeros(sizes.shape[0], dtype=np.int64)
        np.cumsum(sizes[:-1], out=local[1:])
        total = int(sizes.sum())
        stream = np.empty(total + 8, dtype=np.uint8)
        stream[total:] = 0
        breaks = np.flatnonzero(starts[1:] != starts[:-1] + sizes[:-1]) + 1
        spans = []
        for low, high in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [sizes.shape[0]]))):
            at, size = int(local[low]), int(local[high - 1] + sizes[high - 1] - local[low])
            if size:
                spans.append((self.head_end + self.size + int(starts[low]), stream[at: at + size]))
        signs = np.empty((end - first, self.width), dtype=np.uint8)
        jobs = self._read([(self.head_end + first * self.width, signs)] + spans)
        return {"first": first, "end": end, "signs": signs, "lengths": lengths, "stream": stream, "heads": local,
                "runs": len(spans), "jobs": jobs}

    def wait(self, chunk, stalled=None):
        began = time.perf_counter()
        for job in chunk.pop("jobs"):
            while True:
                try:
                    job.result(timeout=DF11_STALL_S)
                    break
                except FutureTimeout:
                    if stalled is not None:
                        stalled(chunk["first"], time.perf_counter() - began)
        self.bytes_read += chunk["signs"].nbytes + chunk["lengths"].nbytes + chunk["stream"].nbytes - 8
        return chunk

    def decode(self, chunk, low, high, lut, limit, device, ops):
        first = chunk["first"]
        count = high - low
        heads = chunk["heads"]
        head, tail = low // self.group, -(-high // self.group)
        at = int(heads[head])
        stop = int(heads[tail]) if tail < heads.shape[0] else chunk["stream"].shape[0] - 8
        keys = self.keys
        lo, hi = np.searchsorted(keys, [(first + low) * self.width, (first + high) * self.width])
        escapes = int(hi - lo)
        rows = (keys[lo:hi] // self.width - first - low).astype(np.int32) if escapes else np.zeros(1, np.int32)
        exps = self.exps[lo:hi] if escapes else np.zeros(1, np.uint8)
        put = lambda array: torch.from_numpy(np.ascontiguousarray(array)).to(device)
        out = torch.empty((count, self.width), dtype=torch.bfloat16, device=device)
        ids = torch.arange(count, dtype=torch.int64, device=device)
        ops.df11_gather_staged(put(chunk["signs"][low:high]).reshape(-1), put(chunk["lengths"][low:high]),
                               put(heads[head:tail] - at), put(chunk["stream"][at: stop + 8]), lut, ids, out, put(rows),
                               put(exps), escapes, 0, count, self.group, limit)
        return out

    def close(self):
        self.pool.shutdown(wait=True)


def q_chunk_rows(group):
    rows = integer(QSTORE_CHUNK_ENV, QSTORE_CHUNK_ROWS, 32, 1 << 26)
    return max(group, rows // group * group)


def q_convert(path, table, header, head_end, threads, drop, directory, found, device):
    began = time.perf_counter()
    reader = Df11Chunks(path, header, head_end, threads, drop)
    book = Df11Book(header["lengths"], header["limit"])
    ops = packed_ops()
    probes = {int(row): digest for row, digest in header.get("probes", [])}
    picked = sorted(set(probe_rows(found, seed=int(time.time()))))
    checks = {}
    decode_s = 0.0
    try:
        side = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        with (torch.cuda.stream(side) if side is not None else contextlib.nullcontext()):
            lut = torch.from_numpy(book.lut).to(device)
            step = q_chunk_rows(reader.group)
            starts = list(range(0, reader.rows, step))
            ahead = reader.submit(0, min(step, reader.rows)) if starts else None
            told = time.perf_counter()

            def stalled(first, waited):
                tell(f"{table.label}: QSTORE {table.kind} convert STALL: the read of rows from {first:,} has waited "
                     f"{waited:.0f} s; {table.done:,} of {reader.rows:,} rows placed, "
                     f"{reader.bytes_read / 2**30:.2f} GiB read in {time.perf_counter() - began:.0f} s; {memory_text()}")

            for at, first in enumerate(starts):
                chunk = reader.wait(ahead, stalled)
                if time.perf_counter() - told >= DF11_PROGRESS_S:
                    told = time.perf_counter()
                    spent = told - began
                    tell(f"{table.label}: QSTORE {table.kind} convert progress: {first:,} of {reader.rows:,} rows, "
                         f"{reader.bytes_read / 2**30:.2f} GiB read in {spent:.0f} s "
                         f"({reader.bytes_read / max(spent, 1e-9) / 1e9:.2f} GB/s), {decode_s:.0f} s decoding; "
                         f"{memory_text()}")
                end = chunk["end"]
                ahead = reader.submit(starts[at + 1], min(starts[at + 1] + step, reader.rows)) \
                    if at + 1 < len(starts) else None
                ticked = time.perf_counter()
                for low in range(0, end - first, QSTORE_SLICE_ROWS):
                    high = min(low + QSTORE_SLICE_ROWS, end - first)
                    decoded = reader.decode(chunk, low, high, lut, book.limit, device, ops)
                    for row in [one for one in probes if first + low <= one < first + high]:
                        words = decoded[row - first - low].view(torch.int16).cpu().numpy().view(np.uint16)
                        if row_digest(words) != probes[row]:
                            raise RuntimeError(f"row {row} decodes to bytes other than the ones recorded at the bake")
                    for row in [one for one in picked if first + low <= one < first + high]:
                        checks[row] = decoded[row - first - low].view(torch.int16).cpu().numpy().view(np.uint16).copy()
                    table.place(first + low, decoded)
                    del decoded
                decode_s += time.perf_counter() - ticked
                with table.lock:
                    table.seen[first] = end - first
                    table.done += end - first
            if side is not None:
                side.synchronize()
            del lut
    finally:
        reader.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ids = np.array(sorted(checks), dtype=np.int64)
    raw = raw_rows(directory, found, ids.tolist())
    mine = np.stack([checks[one] for one in ids.tolist()])
    if not np.array_equal(mine, raw):
        raise RuntimeError("rows read from the checkpoint differ from the same rows decoded from the DF11 file")
    stored = table.decode(ids)
    wanted = q_reference(torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16), table.kind)
    if not np.array_equal(stored, wanted.view(torch.int16).numpy().view(np.uint16)):
        raise RuntimeError("rows dequantized from the table differ from the round trip of the checkpoint rows")
    seconds = time.perf_counter() - began
    tell(f"{table.label}: QSTORE {table.kind} converted {reader.rows:,} rows from {reader.bytes_read / 2**30:.2f} GiB "
         f"of the DF11 file in {seconds:.1f} s ({reader.bytes_read / max(seconds, 1e-9) / 1e9:.2f} GB/s read, "
         f"{decode_s:.1f} s of it decoding and quantizing on {device.type}); {len(probes)} recorded probe rows and "
         f"{len(ids)} checkpoint rows agree, and their stored codes give the round trip bit for bit; {memory_text()}")
    return seconds


def load_prebuilt(rows, width, dtype, label, kind):
    if q6_mode() == "off" or dummy_load():
        return None
    paths = q6_find_parts(str(os.environ.get(Q6_DIR_ENV) or "").strip() or DF11_ROOT)
    if not paths:
        return None
    began = time.perf_counter()
    directory = str(os.environ.get(DF11_CHECKPOINT_ENV) or "").strip()
    if not directory:
        raise RuntimeError(f"{QSTORE_ENV}: {DF11_CHECKPOINT_ENV} names no checkpoint to check the prebuilt table against")
    if dtype != torch.bfloat16:
        raise RuntimeError(f"{QSTORE_ENV} quantizes BF16 rows; this table is {dtype}")
    found = checkpoint_shards(directory)
    parts, why = q6_complete(paths, rows, width, kind, shard_fingerprint(found))
    if parts is None:
        selfcheck_fail(Q6_ENV, f"{label}: the mounted prebuilt {kind} parts cannot serve this checkpoint: {why}")
        tell(f"{label}: QSTORE {kind} path=prebuilt stands aside: {why}; the table is built by conversion instead")
        return None
    table = QTable(rows=rows, width=width, dtype=dtype, label=label, pinned=True, dummy=False, kind=kind)
    table.header = {"shards": found, "probes": []}
    table.preloaded = True
    table.expected = {int(one["start"]): int(one["rows"]) for one in found}
    threads = q6_threads()
    size = sum(q6_layout(header)[2] - Q6_HEAD for _, header in parts)
    skip_rows = q6_skip_rows()
    if skip_rows:
        tell(f"{label}: QSTORE {kind} path=prebuilt: WARNING the check of {len(probe_rows(found))} checkpoint rows "
             f"against the prebuilt table is SKIPPED ({Q6_SKIP_ROWS_ENV}=1); the parts' baked probe digests and "
             f"the checkpoint's shard fingerprint from its headers are still checked")

    def background():
        try:
            started = time.perf_counter()

            def progress(total):
                table.done = min(table.rows, int(table.rows * total / max(size, 1)))

            q6_load_into(parts, table.codes, table.peaks, threads, progress=progress,
                         say=lambda text: tell(f"{label}: QSTORE {kind} {text}"))
            read_s = time.perf_counter() - started
            probes = q6_check_probes(parts, table.codes, table.peaks)
            if skip_rows:
                checked = (f"the checkpoint row check is SKIPPED ({Q6_SKIP_ROWS_ENV}=1): no row of the checkpoint's "
                           f"PLE files was read, only their headers, so a checkpoint whose PLE rows are absent or "
                           f"zero cannot be told from a good one")
            else:
                ids = np.array(sorted(set(probe_rows(found, seed=int(time.time())))), dtype=np.int64)
                raw = raw_rows(directory, found, ids.tolist())
                wanted = q_reference(torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16), kind)
                if not np.array_equal(table.decode(ids), wanted.view(torch.int16).numpy().view(np.uint16)):
                    raise RuntimeError("rows dequantized from the prebuilt table differ from the round trip of the "
                                       "checkpoint rows")
                checked = f"{len(ids)} checkpoint rows give the round trip bit for bit"
            with table.lock:
                table.seen = {0: table.rows}
                table.done = table.rows
            tell(f"{label}: QSTORE {kind} path=prebuilt read {size / 2**30:.2f} GiB of codes and peaks into the "
                 f"table in {read_s:.1f} s ({size / max(read_s, 1e-9) / 1e9:.2f} GB/s); {probes} baked probe rows "
                 f"read back their digests and {checked}; {memory_text()}")
        except Exception as failure:
            raise RuntimeError(f"{QSTORE_ENV} {label}: the prebuilt table could not be read after the loader had "
                               f"been given unread stand-ins for the raw rows ({type(failure).__name__}: {failure})")                 from failure

    def waiting():
        if not q6_drop_on():
            return
        before = cgroup_file_gib()
        files = drop_checkpoint_cache(directory)
        after = cgroup_file_gib()
        freed = "unknown" if before is None or after is None else f"{before - after:.1f} GiB"
        tell(f"{label}: QSTORE {kind} prebuilt: the weights are loaded and the seal waits, so the page cache of {files} "
             f"checkpoint file(s) in {directory} is dropped ({Q6_DROP_ENV}): cgroup file {gib_text(before)} -> "
             f"{gib_text(after)}, {freed} freed")

    table.on_wait = waiting
    try:
        installed = install_stub(table)
    except BaseException:
        table.close()
        raise
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-qstore-prebuilt")
    table.filler = pool.submit(background)
    pool.shutdown(wait=False)
    tell(f"{label}: QSTORE {kind} path=prebuilt: {len(parts)} part(s) ({', '.join(str(one) for one, _ in parts)}) "
         f"hold the {kind} codes and peaks of all {int(rows):,} rows for this checkpoint; {threads} reader thread(s) "
         f"in {q6_piece() / 2**20:g} MiB pieces read them straight into the pinned table beside the weight load after {time.perf_counter() - began:.1f} s "
         f"of header work, each piece dropped from the page cache once read; no DF11 decode; raw table reads "
         f"{'replaced by unread stand-ins' if installed else 'not replaced'}; the table is joined at its seal")
    return table


def load_quantized(rows, width, dtype, label, kind):
    prebuilt = load_prebuilt(rows, width, dtype, label, kind)
    if prebuilt is not None:
        return prebuilt
    path = find_df11_file()
    if path is None or dummy_load():
        return None
    began = time.perf_counter()
    directory = str(os.environ.get(DF11_CHECKPOINT_ENV) or "").strip()
    if not directory:
        raise RuntimeError(f"{QSTORE_ENV}: {DF11_CHECKPOINT_ENV} names no checkpoint to check the table against")
    header = read_df11_header(path)
    shards = header.get("shards")
    if not shards:
        raise RuntimeError(f"{QSTORE_ENV}: {path} carries no shard list of the checkpoint it was baked from")
    if int(header["rows"]) != int(rows) or int(header["width"]) != int(width) or dtype != torch.bfloat16:
        raise RuntimeError(f"{QSTORE_ENV}: {path} holds {header['rows']:,} x {header['width']} rows, the model asks "
                           f"for {int(rows):,} x {int(width)} {dtype}")
    if header.get("version") != DF11_VERSION or header.get("group") != DF11_GROUP:
        raise RuntimeError(f"{QSTORE_ENV}: {path} holds DF11 version {header.get('version')} with groups of "
                           f"{header.get('group')}")
    found = checkpoint_shards(directory)
    if shard_identity(found) != shard_identity(shards):
        raise RuntimeError(f"{QSTORE_ENV}: the checkpoint at {directory} has PLE shards that differ from the "
                           f"{len(shards)} the file was baked from")
    if header.get("fingerprint") != shard_fingerprint(found):
        raise RuntimeError(f"{QSTORE_ENV}: the shard fingerprint in {path} does not match its own shard list")
    with PartsReader(df11_parts(path)) as handle:
        _, head_end = df11_header_end(handle, path)
    table = QTable(rows=rows, width=width, dtype=dtype, label=label, pinned=True, dummy=False, kind=kind)
    table.header = header
    table.preloaded = True
    table.expected = {int(one["start"]): int(one["rows"]) for one in shards}
    threads = df11_read_threads()
    drop = df11_drop()

    def background():
        try:
            q_convert(path, table, header, head_end, threads, drop, directory, found, table.device)
        except Exception as failure:
            raise RuntimeError(f"{QSTORE_ENV} {label}: the conversion of {path} failed after the loader had been "
                               f"given unread stand-ins for the raw rows ({type(failure).__name__}: {failure})") \
                from failure

    try:
        installed = install_stub(table)
    except BaseException:
        table.close()
        raise
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-qstore-convert")
    table.filler = pool.submit(background)
    pool.shutdown(wait=False)
    tell(f"{label}: QSTORE {kind} path=convert: {path} ({sum(one.stat().st_size for one in df11_parts(path)) / 2**30:.2f}"
         f" GiB in {len(df11_parts(path))} part(s)) is read part by part by {threads} reader thread(s), decoded and "
         f"quantized on {table.device.type} beside the weight load after {time.perf_counter() - began:.1f} s of "
         f"header work; no DF11 table is held; raw table reads "
         f"{'replaced by unread stand-ins' if installed else 'not replaced'}; the table is joined at its seal")
    return table


Q6_MAGIC = b"TAAFQ6P1"
Q6_VERSION = 1
Q6_HEAD = 1 << 16
Q6_FILE_NAME = "ple_q6.taafq6"
Q6_ENV = "TAAF_VLLM_PLE_QSTORE_PREBUILT"
Q6_DIR_ENV = "TAAF_VLLM_PLE_QSTORE_PREBUILT_DIR"
Q6_DROP_ENV = "TAAF_VLLM_PLE_Q6_DROP_CHECKPOINT"
Q6_SKIP_ROWS_ENV = "TAAF_VLLM_PLE_Q6_SKIP_CHECKPOINT_ROWS"
Q6_DEPTH = 5
Q6_PIECE = 64 << 20
Q6_PIECE_ENV = "TAAF_VLLM_PLE_Q6_PIECE_MIB"
Q6_THREADS_ENV = "TAAF_VLLM_PLE_Q6_THREADS"
Q6_THREADS = 32
Q6_BAKE_ROWS = 1 << 20


def q6_mode():
    said = str(os.environ.get(Q6_ENV) or "").strip().lower()
    if said in ("", "auto", "1", "on", "true"):
        return "auto"
    if said in ("0", "off", "false", "no"):
        return "off"
    raise ValueError(f"{Q6_ENV} must be unset, auto, 1 or 0, found {said!r}")


def q6_part_name(index, count):
    return f"{Q6_FILE_NAME}.part{index}of{count}"


def q6_bounds(rows, index, count):
    if not 0 <= index < count:
        raise ValueError(f"part {index} of {count} does not exist")
    return rows * index // count, rows * (index + 1) // count


def q6_layout(header):
    count = int(header["end"]) - int(header["first"])
    codes_at = Q6_HEAD
    peaks_at = codes_at + count * int(header["row_bytes"])
    return codes_at, peaks_at, peaks_at + count * int(header["groups"]) * 2


def q6_row_digest(codes, peaks):
    return hashlib.sha256(np.ascontiguousarray(codes).tobytes() + np.ascontiguousarray(peaks).tobytes()).hexdigest()


def q6_write_header(handle, header):
    raw = json.dumps(header, sort_keys=True).encode("utf-8")
    if len(raw) + 16 > Q6_HEAD:
        raise RuntimeError(f"PLE_Q6: the header takes {len(raw):,} B, over its {Q6_HEAD - 16:,} B reserve")
    handle.seek(0)
    handle.write(Q6_MAGIC + len(raw).to_bytes(8, "little") + raw + b" " * (Q6_HEAD - 16 - len(raw)))


def q6_read_header(path):
    with open(path, "rb") as handle:
        if handle.read(8) != Q6_MAGIC:
            raise RuntimeError(f"PLE_Q6: {path} is not a prebuilt PLE part")
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size).decode("utf-8"))
    if header.get("version") != Q6_VERSION:
        raise RuntimeError(f"PLE_Q6: {path} is version {header.get('version')}, this reader knows {Q6_VERSION}")
    return header


def q6_bake(directory, index, count, out_dir, kind="int6", df11_probes=None, rows_at_once=Q6_BAKE_ROWS, say=print):
    began = time.perf_counter()
    shards = checkpoint_shards(directory)
    rows = sum(one["rows"] for one in shards)
    width = shards[0]["width"]
    if any(one["width"] != width or one["dtype"] != "BF16" for one in shards):
        raise RuntimeError("PLE_Q6: the checkpoint's PLE shards are not all BF16 rows of one width")
    first, end = q6_bounds(rows, index, count)
    group = q_group(width)
    header = {"version": Q6_VERSION, "kind": kind, "rows": rows, "width": width, "group": group,
              "groups": width // group, "row_bytes": q_row_bytes(width, kind), "first": first, "end": end,
              "index": index, "count": count, "fingerprint": shard_fingerprint(shards),
              "shards": shard_identity(shards)}
    codes_at, peaks_at, size = q6_layout(header)
    wanted = sorted({row for row in probe_rows(shards, seed=0) if first <= row < end}
                    | {first, end - 1})
    df11 = {int(row): digest for row, digest in (df11_probes or []) if first <= int(row) < end}
    probes, checked = [], 0
    path = Path(out_dir) / q6_part_name(index, count)
    handles = {}
    try:
        with open(path, "wb") as out:
            out.truncate(size)
            for low in range(first, end, rows_at_once):
                high = min(low + rows_at_once, end)
                words = np.empty((high - low, width), dtype="<u2")
                at = low
                while at < high:
                    one = next(item for item in shards if item["start"] <= at < item["start"] + item["rows"])
                    stop = min(high, one["start"] + one["rows"])
                    handle = handles.get(one["file"])
                    if handle is None:
                        handle = handles[one["file"]] = open(Path(directory) / one["file"], "rb")
                    view = memoryview(words[at - low: stop - low]).cast("B")
                    handle.seek(one["offset"] + (at - one["start"]) * width * 2)
                    done = 0
                    while done < view.nbytes:
                        got = handle.readinto(view[done:])
                        if not got:
                            raise RuntimeError(f"PLE_Q6: {one['file']} ended inside {one['name']}")
                        done += got
                    at = stop
                for row in [one for one in df11 if low <= one < high]:
                    if row_digest(words[row - low]) != df11[row]:
                        raise RuntimeError(f"PLE_Q6: checkpoint row {row} differs from the row the DF11 file recorded")
                    checked += 1
                codes, peaks = q_encode(torch.from_numpy(words.view(np.int16)).view(torch.bfloat16), kind)
                codes = codes.numpy()
                peaks = peaks.numpy()
                out.seek(codes_at + (low - first) * header["row_bytes"])
                out.write(codes.tobytes())
                out.seek(peaks_at + (low - first) * header["groups"] * 2)
                out.write(peaks.tobytes())
                for row in [one for one in wanted if low <= one < high]:
                    probes.append([row, q6_row_digest(codes[row - low], peaks[row - low])])
                say(f"PLE_Q6 bake part {index} of {count}: rows {low:,}-{high:,} of {first:,}-{end:,} in "
                    f"{time.perf_counter() - began:.0f} s")
            header.update(probes=probes, df11_checked=checked, df11_probes=len(df11), size=size)
            q6_write_header(out, header)
    finally:
        for handle in handles.values():
            handle.close()
    if checked != len(df11):
        raise RuntimeError(f"PLE_Q6: {checked} of the {len(df11)} DF11 probe rows in this part were checked")
    say(f"PLE_Q6 bake part {index} of {count}: {end - first:,} rows, {size / 2**30:.2f} GiB, {len(probes)} probe "
        f"digests, {checked} DF11 probe rows equal the checkpoint, {time.perf_counter() - began:.0f} s")
    return path, header


def q6_find_parts(root=DF11_ROOT, depth=Q6_DEPTH):
    root = Path(root)
    if not root.is_dir():
        return []
    found = {}
    for level in range(depth + 1):
        for path in sorted(root.glob("/".join(["*"] * level + [Q6_FILE_NAME + ".part*of*"]))):
            found.setdefault(path.name, path)
    return [found[name] for name in sorted(found)]


def q6_complete(paths, rows, width, kind, fingerprint):
    parts = []
    for path in paths:
        header = q6_read_header(path)
        if (header["kind"], int(header["rows"]), int(header["width"]), header["fingerprint"]) != \
                (kind, int(rows), int(width), fingerprint):
            return None, f"{path.name} is {header['kind']} {header['rows']:,} x {header['width']} of another checkpoint"
        if path.stat().st_size != q6_layout(header)[2]:
            return None, f"{path.name} holds {path.stat().st_size:,} B, its header lays out {q6_layout(header)[2]:,}"
        parts.append((path, header))
    parts.sort(key=lambda one: int(one[1]["first"]))
    covered = 0
    for path, header in parts:
        if int(header["first"]) != covered:
            return None, f"rows {covered:,}-{header['first']:,} are in no part"
        covered = int(header["end"])
    if covered != int(rows):
        return None, f"the parts cover {covered:,} of {int(rows):,} rows"
    return parts, ""


_Q6_HANDLES = threading.local()


def q6_threads():
    said = str(os.environ.get(Q6_THREADS_ENV) or "").strip()
    if not said:
        return Q6_THREADS
    if not said.isdigit() or not 1 <= int(said) <= 256:
        raise ValueError(f"{Q6_THREADS_ENV} must be a thread count from 1 to 256, found {said!r}")
    return int(said)


def q6_piece():
    said = str(os.environ.get(Q6_PIECE_ENV) or "").strip()
    if not said:
        return Q6_PIECE
    if not said.isdigit() or not 1 <= int(said) <= 1024:
        raise ValueError(f"{Q6_PIECE_ENV} must be a piece size in MiB from 1 to 1024, found {said!r}")
    return int(said) << 20


def _q6_handle(path):
    held = getattr(_Q6_HANDLES, "files", None)
    if held is None:
        held = _Q6_HANDLES.files = {}
    handle = held.get(str(path))
    if handle is None:
        handle = held[str(path)] = open(path, "rb", buffering=0)
    return handle


def _q6_read_into(path, offset, target):
    view = memoryview(target).cast("B")
    handle = _q6_handle(path)
    done = 0
    while done < view.nbytes:
        got = os.preadv(handle.fileno(), [view[done:]], offset + done) if hasattr(os, "preadv") else None
        if got is None:
            handle.seek(offset + done)
            got = handle.readinto(view[done:])
        if not got:
            raise RuntimeError(f"PLE_Q6: {path} ended {view.nbytes - done:,} B early at {offset + done:,}")
        done += got
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(handle.fileno(), offset, view.nbytes, os.POSIX_FADV_DONTNEED)
    return view.nbytes


def q6_skip_rows():
    said = str(os.environ.get(Q6_SKIP_ROWS_ENV) or "").strip().lower()
    if said in ("", "0", "off", "false", "no"):
        return False
    if said in ("1", "on", "true"):
        return True
    raise ValueError(f"{Q6_SKIP_ROWS_ENV} must be unset, 1 or 0, found {said!r}")


def q6_drop_on():
    said = str(os.environ.get(Q6_DROP_ENV) or "").strip().lower()
    if said in ("", "1", "on", "true"):
        return True
    if said in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"{Q6_DROP_ENV} must be unset, 1 or 0, found {said!r}")


def drop_checkpoint_cache(directory):
    if not hasattr(os, "posix_fadvise"):
        return 0
    count = 0
    for path in sorted(Path(directory).glob("*.safetensors")):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        finally:
            os.close(fd)
    return count


def q6_load_into(parts, codes, peaks, threads=DF11_READ_THREADS_DEFAULT, progress=None, say=None):
    jobs = []
    piece = q6_piece()
    for path, header in parts:
        first, end = int(header["first"]), int(header["end"])
        codes_at, peaks_at, _ = q6_layout(header)
        for target, offset in ((codes[first:end], codes_at), (peaks[first:end], peaks_at)):
            flat = target.reshape(-1).view(np.uint8)
            for low in range(0, flat.nbytes, piece):
                jobs.append((path, offset + low, flat[low: low + piece]))
    size = sum(job[2].nbytes for job in jobs)
    total = 0
    began = time.perf_counter()
    told = began
    with ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="ple-q6-read") as pool:
        pending = [pool.submit(_q6_read_into, *job) for job in jobs]
        for future in pending:
            waited = time.perf_counter()
            while True:
                try:
                    total += future.result(timeout=DF11_STALL_S)
                    break
                except FutureTimeout:
                    if say is not None:
                        say(f"prebuilt STALL: a {piece / 2**20:g} MiB piece has waited "
                            f"{time.perf_counter() - waited:.0f} s; {total / 2**30:.2f} of {size / 2**30:.2f} GiB read "
                            f"in {time.perf_counter() - began:.0f} s; {memory_text()}")
            if progress is not None:
                progress(total)
            if say is not None and time.perf_counter() - told >= DF11_PROGRESS_S:
                told = time.perf_counter()
                say(f"prebuilt read {total / 2**30:.2f} of {size / 2**30:.2f} GiB in {told - began:.0f} s "
                    f"({total / max(told - began, 1e-9) / 1e9:.2f} GB/s), each piece dropped from the page cache; "
                    f"{memory_text()}")
    return total


def q6_check_probes(parts, codes, peaks):
    count = 0
    for path, header in parts:
        for row, digest in header["probes"]:
            if q6_row_digest(codes[row], peaks[row]) != digest:
                raise RuntimeError(f"PLE_Q6: row {row} of {path.name} reads back other bytes than the bake wrote")
            count += 1
    return count
