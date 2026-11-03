import collections
import glob
import hashlib
import json
import os
import re
import threading
import time

GIB = 1 << 30
SKIP_ENV = "TAAF_VLLM_WEIGHTS_PREFETCH_SKIP_MARK"
DEFAULT_CHUNK = 16 << 20
PROGRESS_BYTES = 16 * GIB
POLL_S = 0.5
UNSEEN_NOTICE_S = 300.0
JOIN_S = 30.0
STAGE_INDEX = "STAGE.json"
STAGE_QUEUE_BYTES = 512 << 20
STAGE_SYNC_BYTES = 1 << 30
DF11_FILE_NAME = "ple_df11.taafdf11"
DF11_FILE_ENV = "TAAF_VLLM_PLE_DF11_FILE"
DF11_DEPTH = 4
MAPS_LINE = re.compile(rb"^\S+ \S+ \S+ \S+ \d+\s+(.*)$")


def say(line):
    print(f"SHARD_PREFETCH {line}", flush=True)


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", os.path.basename(path))]


def checkpoint_files(directory, skip_mark=None):
    directory = str(directory)
    found = [name for name in os.listdir(directory) if name.endswith(".safetensors")]
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            weights = json.load(handle)["weight_map"]
        named = set(weights.values())
        if skip_mark:
            named = {name for name in named
                     if not all(skip_mark in key for key, held in weights.items() if held == name)}
            say(f"files holding only tensors named like {skip_mark} are left unread: the packed table replaces them")
        found = [name for name in found if name in named]
    rows = []
    for name in sorted(found, key=natural_key):
        real = os.path.realpath(os.path.join(directory, name))
        rows.append((real, os.stat(real).st_size))
    return rows


def mapped_position(index_of, proc="/proc"):
    best = -1
    try:
        pids = [name for name in os.listdir(proc) if name.isdigit()]
    except OSError:
        return best
    for pid in pids:
        try:
            with open(f"{proc}/{pid}/maps", "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if b".safetensors" not in data:
            continue
        for line in data.split(b"\n"):
            if b".safetensors" not in line:
                continue
            match = MAPS_LINE.match(line)
            if match is None:
                continue
            path = match.group(1).decode("utf-8", "surrogateescape")
            if path.endswith(" (deleted)"):
                path = path[:-len(" (deleted)")]
            found = index_of.get(path)
            if found is not None and found > best:
                best = found
    return best


def drop_cache(path):
    if not hasattr(os, "posix_fadvise"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


class Prefetcher:
    def __init__(self, files, threads, window_bytes, chunk_bytes=DEFAULT_CHUNK, position=None, evict=drop_cache,
                 alive=None, report=say, poll_s=POLL_S, progress_bytes=PROGRESS_BYTES, clock=time.monotonic,
                 stager=None):
        self.stager = stager
        self.early_pending = False
        self.served = None
        self.files = [(str(path), int(size)) for path, size in files]
        self.threads = int(threads)
        self.chunk = int(chunk_bytes)
        largest = max((size for _, size in self.files), default=0)
        self.window = max(int(window_bytes), largest + self.chunk)
        self.clamped = self.window != int(window_bytes)
        self.position = position or (lambda: -1)
        self.evict = evict
        self.alive = alive or (lambda: True)
        self.report = report
        self.poll_s = poll_s
        self.progress_bytes = int(progress_bytes)
        self.clock = clock
        self.starts = []
        total = 0
        for _, size in self.files:
            self.starts.append(total)
            total += size
        self.total = total
        self.chunks = [(index, offset, min(self.chunk, size - offset))
                       for index, (_, size) in enumerate(self.files) for offset in range(0, size, self.chunk)]
        self.cv = threading.Condition()
        self.next = 0
        self.read = 0
        self.skipped = 0
        self.done = [0] * len(self.files)
        self.inflight = [0] * len(self.files)
        self.touched = [False] * len(self.files)
        self.dropped = [False] * len(self.files)
        self.loader = -1
        self.first_seen = None
        self.next_progress = self.progress_bytes
        self.last_mark = None
        self.stopped = False
        self.finished = False
        self.failed = None
        self.waiting = 0
        self.threads_left = self.threads
        self.workers = []
        self.monitor = None
        self.began = None
        self.peak_ahead = 0

    def start(self):
        self.began = self.clock()
        self.last_mark = (self.began, 0)
        self.report(f"start: {self.threads} threads, window {self.window / GIB:.1f} GiB"
                    f"{' (raised to hold the largest shard)' if self.clamped else ''}, "
                    f"chunk {self.chunk >> 20} MiB, {len(self.files)} files, {self.total / GIB:.2f} GiB")
        self.monitor = threading.Thread(target=self._watch, name="shard-prefetch-watch", daemon=True)
        self.monitor.start()
        for number in range(self.threads):
            worker = threading.Thread(target=self._work, name=f"shard-prefetch-{number}", daemon=True)
            worker.start()
            self.workers.append(worker)
        return self

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()
        deadline = self.clock() + JOIN_S
        for thread in [*self.workers, self.monitor]:
            if thread is not None:
                thread.join(max(0.0, deadline - self.clock()))
        self._drop_touched()
        self._finish("stopped" if self.next < len(self.chunks) or self.failed else "loader done")
        if self.stager is not None:
            self.stager.stop()

    def ahead_bytes(self):
        with self.cv:
            return self._ahead()

    def _ahead(self):
        return sum(self.done[max(self.loader, 0):])

    def _fail(self, what):
        with self.cv:
            first = self.failed is None
            if first:
                self.failed = what
            self.stopped = True
            self.cv.notify_all()
        if first:
            self.report(f"error, prefetching stops and the serve goes on: {what}")

    def _finish(self, reason):
        with self.cv:
            if self.finished:
                return
            self.finished = True
            seconds = self.clock() - self.began
            line = (f"end ({reason}): {self.read / GIB:.2f} GiB read, {self.skipped / GIB:.2f} GiB left to the loader, "
                    f"{seconds:.1f} s, mean {self.read / max(seconds, 1e-9) / 1e9:.2f} GB/s, "
                    f"peak {self.peak_ahead / GIB:.1f} GiB ahead of the loader")
        self.report(line)

    def _drop_touched(self):
        for index, (path, _) in enumerate(self.files):
            if self.touched[index] and not self.dropped[index]:
                self._drop(index)

    def _drop(self, index):
        try:
            self.evict(self.files[index][0])
        except Exception as exc:
            self.report(f"could not drop {os.path.basename(self.files[index][0])} from the page cache: {exc!r}")
        self.dropped[index] = True

    def _watch(self):
        try:
            while True:
                with self.cv:
                    if self.stopped:
                        return
                if not self.alive():
                    with self.cv:
                        self.stopped = True
                        self.cv.notify_all()
                    self.report("the server is gone, prefetching stops")
                    return
                seen = int(self.position())
                due = []
                with self.cv:
                    if seen > self.loader:
                        if self.first_seen is None and seen >= 0:
                            self.first_seen = self.clock() - self.began
                            self.report(f"loader first seen on file {seen + 1}/{len(self.files)} "
                                        f"({os.path.basename(self.files[seen][0])}) after {self.first_seen:.0f} s")
                        self.loader = seen
                    elif self.first_seen is None and self.clock() - self.began > UNSEEN_NOTICE_S \
                            and not self.finished:
                        self.first_seen = float("inf")
                        self.report(f"the loader has not been seen after {UNSEEN_NOTICE_S:.0f} s; "
                                    f"nothing is dropped and the window stays at the first file")
                    for index in range(max(self.loader, 0)):
                        if self.dropped[index] or self.inflight[index]:
                            continue
                        if self.touched[index]:
                            due.append(index)
                        else:
                            self.dropped[index] = True
                    self.cv.notify_all()
                for index in due:
                    self._drop(index)
                time.sleep(self.poll_s)
        except Exception as exc:
            self._fail(f"watcher {exc!r}")

    def _claim(self):
        with self.cv:
            while True:
                if self.stopped:
                    return None
                floor = max(self.loader, 0)
                while self.next < len(self.chunks) and self.chunks[self.next][0] < floor:
                    self.skipped += self.chunks[self.next][2]
                    self.next += 1
                if self.next >= len(self.chunks):
                    return None
                index, offset, length = self.chunks[self.next]
                if self.starts[index] + offset + length - self.starts[floor] <= self.window:
                    self.next += 1
                    self.inflight[index] += 1
                    self.touched[index] = True
                    return index, offset, length
                self.waiting += 1
                self.cv.wait(self.poll_s)
                self.waiting -= 1

    def _work(self):
        buffer = bytearray(self.chunk)
        view = memoryview(buffer)
        try:
            while True:
                claimed = self._claim()
                if claimed is None:
                    break
                index, offset, length = claimed
                try:
                    got = self._pull(self.files[index][0], offset, length, view)
                finally:
                    with self.cv:
                        self.inflight[index] -= 1
                if got != length:
                    raise OSError(f"{self.files[index][0]} gave {got} of {length} bytes at {offset}")
                if self.stager is not None:
                    self.stager.offer(index, offset, view[:length])
                self._landed(index, length)
        except Exception as exc:
            self._fail(f"{exc!r}")
        self._exit()

    def _pull(self, path, offset, length, view):
        with open(path, "rb", buffering=0) as handle:
            handle.seek(offset)
            got = 0
            while got < length:
                count = handle.readinto(view[got:length])
                if not count:
                    break
                got += count
        return got

    def _landed(self, index, length):
        line = None
        with self.cv:
            self.done[index] += length
            self.read += length
            ahead = self._ahead()
            self.peak_ahead = max(self.peak_ahead, ahead)
            if self.read >= self.next_progress:
                self.next_progress += self.progress_bytes
                now = self.clock()
                then, was = self.last_mark
                self.last_mark = (now, self.read)
                seen = (f"loader on file {self.loader + 1}/{len(self.files)}" if self.loader >= 0
                        else "loader not seen yet")
                line = (f"{self.read / GIB:.1f}/{self.total / GIB:.1f} GiB in {now - self.began:.0f} s, "
                        f"{(self.read - was) / max(now - then, 1e-9) / 1e9:.2f} GB/s now, "
                        f"{self.read / max(now - self.began, 1e-9) / 1e9:.2f} GB/s mean; "
                        f"{ahead / GIB:.1f} GiB read ahead of the loader, {seen}, "
                        f"{self.waiting}/{self.threads} threads waiting on the window")
            self.cv.notify_all()
        if line:
            self.report(line)

    def _exit(self):
        with self.cv:
            self.threads_left -= 1
            last = self.threads_left == 0
        if last and not self.failed and not self.stopped:
            self._finish("all read")


def configured(env):
    said = str(env.get("TAAF_VLLM_WEIGHTS_PREFETCH") or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return None
    threads = int(said)
    window = float(str(env.get("TAAF_VLLM_WEIGHTS_PREFETCH_GIB") or "").strip() or 24)
    chunk = int(str(env.get("TAAF_VLLM_WEIGHTS_PREFETCH_CHUNK_MIB") or "").strip() or 16)
    if not 1 <= threads <= 256 or not 1 <= window <= 1024 or not 1 <= chunk <= 1024:
        raise ValueError(f"threads {threads}, window {window} GiB, chunk {chunk} MiB are out of range")
    return threads, int(window * GIB), chunk << 20


def launch(directories, env, alive=None):
    try:
        setting = configured(env)
        if setting is None:
            return None
        threads, window, chunk = setting
        files = []
        skip = str(env.get(SKIP_ENV) or "").strip() or None
        for directory in directories:
            files.extend(checkpoint_files(directory, skip))
        if not files:
            say("no safetensors files found, nothing to prefetch")
            return None
        index_of = {path: number for number, (path, _) in enumerate(files)}
        position = (lambda: mapped_position(index_of)) if os.path.isdir("/proc/self") else None
        if position is None:
            say("no /proc here: the loader cannot be followed, the window stays at the first file")
        return Prefetcher(files, threads, window, chunk, position=position, alive=alive).start()
    except Exception as exc:
        say(f"error before the start, no prefetch and the serve goes on: {exc!r}")
        return None


def file_key(path):
    return hashlib.sha1(os.path.dirname(str(path)).encode("utf-8", "surrogateescape")).hexdigest()[:10]


def tier_path(root, source):
    return os.path.join(str(root), file_key(source), os.path.basename(str(source)))


def read_stage_index(root):
    try:
        with open(os.path.join(str(root), STAGE_INDEX), encoding="utf-8") as handle:
            held = json.load(handle)
        return held if isinstance(held, dict) else {}
    except (OSError, ValueError):
        return {}


def staged_copy(root, source, index=None):
    index = read_stage_index(root) if index is None else index
    entry = index.get(str(source))
    if not isinstance(entry, dict):
        return None
    local = tier_path(root, source)
    try:
        now = os.stat(source)
        held = os.stat(local)
    except OSError:
        return None
    if int(entry.get("size", -1)) != now.st_size or held.st_size != now.st_size \
            or int(entry.get("mtime_ns", -1)) != now.st_mtime_ns:
        return None
    return local


class Stager:
    def __init__(self, root, files, queue_bytes=STAGE_QUEUE_BYTES, sync_bytes=STAGE_SYNC_BYTES, report=say):
        self.root = str(root)
        self.files = [(str(path), int(size)) for path, size in files]
        self.queue_bytes = int(queue_bytes)
        self.sync_bytes = int(sync_bytes)
        self.report = report
        self.index = read_stage_index(self.root)
        self.have = {number for number, (path, _) in enumerate(self.files)
                     if staged_copy(self.root, path, self.index) is not None}
        self.cv = threading.Condition()
        self.queue = collections.deque()
        self.queued = 0
        self.offered = 0
        self.dropped = 0
        self.written = 0
        self.stopped = False
        self.handles = {}
        self.seen = {}
        self.counted = {number: 0 for number in range(len(self.files))}
        self.since_sync = 0
        self.completed = set(self.have)
        self.failed = None
        self.began = None
        self.thread = None

    def start(self):
        os.makedirs(self.root, exist_ok=True)
        self.began = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="shard-stage", daemon=True)
        self.thread.start()
        self.report(f"stage: copying what is read into {self.root}, {len(self.have)} of {len(self.files)} files "
                    f"already staged there, queue {self.queue_bytes >> 20} MiB")
        return self

    def offer(self, index, offset, view):
        if index in self.completed or self.failed is not None:
            return False
        length = len(view)
        with self.cv:
            self.offered += length
            if self.stopped or self.queued + length > self.queue_bytes:
                self.dropped += length
                return False
            self.queue.append((index, offset, bytes(view)))
            self.queued += length
            self.cv.notify_all()
        return True

    def _open(self, index):
        handle = self.handles.get(index)
        if handle is None:
            final = tier_path(self.root, self.files[index][0])
            os.makedirs(os.path.dirname(final), exist_ok=True)
            handle = open(final + ".part", "w+b")
            handle.truncate(self.files[index][1])
            self.handles[index] = handle
            self.seen[index] = set()
        return handle

    def _write(self, index, offset, data):
        if index in self.completed:
            return
        handle = self._open(index)
        if offset in self.seen[index]:
            return
        handle.seek(offset)
        handle.write(data)
        self.seen[index].add(offset)
        self.counted[index] += len(data)
        self.written += len(data)
        self.since_sync += len(data)
        if self.since_sync >= self.sync_bytes:
            self._sync()
        if self.counted[index] >= self.files[index][1]:
            self._seal(index)

    def _sync(self):
        self.since_sync = 0
        for handle in self.handles.values():
            handle.flush()
            if hasattr(os, "fdatasync"):
                os.fdatasync(handle.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def _seal(self, index):
        handle = self.handles.pop(index)
        handle.flush()
        os.fsync(handle.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        handle.close()
        final = tier_path(self.root, self.files[index][0])
        os.replace(final + ".part", final)
        source = os.stat(self.files[index][0])
        self.index[self.files[index][0]] = {"size": source.st_size, "mtime_ns": source.st_mtime_ns}
        scratch = os.path.join(self.root, STAGE_INDEX + ".tmp")
        with open(scratch, "w", encoding="utf-8") as out:
            json.dump(self.index, out)
        os.replace(scratch, os.path.join(self.root, STAGE_INDEX))
        self.completed.add(index)

    def _run(self):
        try:
            while True:
                with self.cv:
                    while not self.queue and not self.stopped:
                        self.cv.wait(0.5)
                    if not self.queue:
                        return
                    index, offset, data = self.queue.popleft()
                    self.queued -= len(data)
                self._write(index, offset, data)
        except Exception as failure:
            self.failed = repr(failure)
            self.report(f"stage: error, staging stops and the serve goes on: {failure!r}")

    def stop(self):
        with self.cv:
            if self.thread is None:
                return
            self.stopped = True
            self.cv.notify_all()
        self.thread.join(30.0)
        for index, handle in list(self.handles.items()):
            try:
                handle.close()
                os.unlink(tier_path(self.root, self.files[index][0]) + ".part")
            except OSError:
                pass
        self.handles = {}
        seconds = time.monotonic() - (self.began or time.monotonic())
        self.report(f"stage: {len(self.completed)} of {len(self.files)} files staged complete "
                    f"({sum(self.files[n][1] for n in self.completed) / GIB:.2f} GiB), "
                    f"{self.written / GIB:.2f} GiB written this run in {seconds:.0f} s, "
                    f"{self.dropped / GIB:.2f} GiB of {self.offered / GIB:.2f} GiB offered were not queued "
                    f"because the disk was behind{', error ' + self.failed if self.failed else ''}")
        self.thread = None


def _timed_parallel(jobs, work, threads):
    todo = collections.deque(jobs)
    failures = []

    def run():
        while not failures:
            try:
                one = todo.popleft()
            except IndexError:
                return
            try:
                work(one)
            except Exception as failure:
                failures.append(failure)

    began = time.monotonic()
    workers = [threading.Thread(target=run, daemon=True) for _ in range(max(1, int(threads)))]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures:
        raise failures[0]
    return time.monotonic() - began


def probe_tiers(source, root, gib, threads, report=say, chunk=None):
    chunk = chunk or DEFAULT_CHUNK
    size = os.path.getsize(source)
    total = min(size, int(gib * GIB)) // chunk * chunk
    if total <= 0:
        report(f"probe: {source} is too small to time")
        return None
    count = total // chunk
    buffers = [bytearray(chunk) for _ in range(count)]

    def read_from(path):
        def one(number):
            with open(path, "rb", buffering=0) as handle:
                handle.seek(number * chunk)
                view = memoryview(buffers[number])
                got = 0
                while got < chunk:
                    step = handle.readinto(view[got:])
                    if not step:
                        raise OSError(f"{path} ended early")
                    got += step
        return one

    def write_one(number):
        with open(scratch, "r+b", buffering=0) as handle:
            handle.seek(number * chunk)
            handle.write(buffers[number])

    os.makedirs(root, exist_ok=True)
    scratch = os.path.join(str(root), "probe.bin")
    out = {}
    try:
        out["input_s"] = _timed_parallel(range(count), read_from(source), threads)
        with open(scratch, "wb") as handle:
            handle.truncate(total)
        began = time.monotonic()
        _timed_parallel(range(count), write_one, threads)
        with open(scratch, "r+b") as handle:
            os.fsync(handle.fileno())
        out["write_s"] = time.monotonic() - began
        if hasattr(os, "posix_fadvise"):
            descriptor = os.open(scratch, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        out["read_s"] = _timed_parallel(range(count), read_from(scratch), threads)
    finally:
        try:
            os.unlink(scratch)
        except OSError:
            pass
    out["bytes"] = total
    out["input_gbs"] = total / max(out["input_s"], 1e-9) / 1e9
    out["tmp_write_gbs"] = total / max(out["write_s"], 1e-9) / 1e9
    out["tmp_read_gbs"] = total / max(out["read_s"], 1e-9) / 1e9
    report(f"probe: {total / GIB:.1f} GiB with {threads} threads: input read {out['input_gbs']:.2f} GB/s, "
           f"{root} write {out['tmp_write_gbs']:.2f} GB/s, read back {out['tmp_read_gbs']:.2f} GB/s")
    return out


def _df11_parts(base):
    base = str(base)
    if os.path.isfile(base):
        return [base]
    return sorted(glob.glob(glob.escape(base) + ".p[0-9][0-9][0-9]"))


def df11_rows(env, root="/kaggle/input"):
    said = str(env.get(DF11_FILE_ENV) or "").strip()
    if said.lower() in ("0", "false", "no", "off"):
        return None, []
    if said:
        found = _df11_parts(said)
        base = said
    else:
        found, base = [], None
        if os.path.isdir(str(root)):
            for depth in range(DF11_DEPTH + 1):
                pattern = os.path.join(glob.escape(str(root)), *(["*"] * depth + [DF11_FILE_NAME + "*"]))
                for hit in sorted(glob.glob(pattern)):
                    candidate = os.path.join(os.path.dirname(hit), DF11_FILE_NAME)
                    found = _df11_parts(candidate)
                    if found:
                        base = candidate
                        break
                if found:
                    break
    if not found:
        return None, []
    return base, [(os.path.realpath(one), os.stat(one).st_size) for one in found]


def launch_rows(rows, threads, window, chunk, alive=None, stager=None, served=None, early=False):
    index_of = {path: number for number, (path, _) in enumerate(rows)}
    position = (lambda: mapped_position(index_of)) if os.path.isdir("/proc/self") else None
    if position is None:
        say("no /proc here: the loader cannot be followed, the window stays at the first file")
    if stager is not None:
        stager.start()
    made = Prefetcher(rows, threads, window, chunk, position=position, alive=alive, stager=stager)
    made.served = served
    made.early_pending = bool(early)
    return made.start()
