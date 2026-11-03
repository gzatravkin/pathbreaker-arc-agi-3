import hashlib
import json
import multiprocessing
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ple_hot as hot  # noqa: E402

TASK_ROWS = hot.DF11_TASK_ROWS
SAMPLE_CHUNKS = hot.DF11_SAMPLE_CHUNKS
SAMPLE_CHUNK_ROWS = hot.DF11_SAMPLE_CHUNK_ROWS
BENCH_ROWS = 16384
REPORT_EVERY_S = 30.0
PART_BYTES = 16 << 30
_STATE = {}


def say(text):
    print(f"BAKE {time.strftime('%H:%M:%S')} [df11] {text}", flush=True)


read_rows = hot.read_rows
segments_of = hot.segments_of
read_global = hot.read_global


def init_worker(lengths, limit):
    torch.set_num_threads(1)
    _STATE["book"] = hot.Df11Book(np.asarray(lengths, dtype=np.int64), limit)


def encode_task(task):
    first, width, segments = task
    data = read_global(segments)
    return first, data.shape[0], _STATE["book"].encode(data, first, width, hot.DF11_GROUP, "cpu")


sample_book = hot.sample_book


def measure_rate(book, data, width):
    torch.set_num_threads(1)
    rows = min(BENCH_ROWS, data.shape[0])
    sample = np.ascontiguousarray(data[:rows])
    book.encode(sample[:256], 0, width, hot.DF11_GROUP, "cpu")
    began = time.perf_counter()
    book.encode(sample, 0, width, hot.DF11_GROUP, "cpu")
    return rows / (time.perf_counter() - began)


def estimate(rows, width, procs, rate, read_gib_per_s=(0.15, 0.6)):
    encode_s = rows / (rate * procs)
    raw_gib = rows * 2 * width / 2**30
    read_s = (raw_gib / read_gib_per_s[1], raw_gib / read_gib_per_s[0])
    low = max(encode_s, read_s[0])
    high = max(encode_s, read_s[1])
    return {"rate_per_process": rate, "encode_s": encode_s, "raw_gib": raw_gib, "read_s": read_s,
            "low_s": low, "high_s": high}


class PartsWriter:
    def __init__(self, base, part_bytes):
        self.base = Path(base)
        self.part = int(part_bytes)
        self.handles = {}
        self.end = 0

    def name(self, index):
        return self.base.with_name(f"{self.base.name}.p{index:03d}")

    def handle(self, index):
        found = self.handles.get(index)
        if found is None:
            path = self.name(index)
            found = self.handles[index] = open(path, "r+b" if path.exists() else "w+b")
        return found

    def write_at(self, offset, data):
        view = memoryview(data).cast("B")
        done = 0
        while done < len(view):
            index, inner = divmod(offset + done, self.part)
            take = min(len(view) - done, self.part - inner)
            handle = self.handle(index)
            handle.seek(inner)
            handle.write(view[done: done + take])
            done += take
        self.end = max(self.end, offset + len(view))

    def read_at(self, offset, count):
        out = bytearray()
        while len(out) < count:
            index, inner = divmod(offset + len(out), self.part)
            take = min(count - len(out), self.part - inner)
            handle = self.handle(index)
            handle.flush()
            handle.seek(inner)
            got = handle.read(take)
            if len(got) != take:
                raise RuntimeError(f"{self.name(index)} ended early")
            out += got
        return bytes(out)

    def close(self, total):
        last = (total - 1) // self.part
        for index in range(last + 1):
            handle = self.handle(index)
            handle.flush()
            handle.truncate(self.part if index < last else total - last * self.part)
        for handle in self.handles.values():
            handle.close()
        self.handles = {}
        return [self.name(index) for index in range(last + 1)]


class Streamed:
    def __init__(self, base, shards, book, part_bytes=PART_BYTES):
        self.rows = sum(one["rows"] for one in shards)
        self.width = shards[0]["width"]
        self.group = hot.DF11_GROUP
        self.groups = -(-self.rows // self.group)
        self.lengths_at = self.rows * self.width
        self.bases_at = hot.ceil_to(self.lengths_at + self.rows, 8)
        self.size = self.bases_at + 8 * self.groups
        self.host_at = len(hot.DF11_MAGIC) + 8 + hot.DF11_HEADER_RESERVE
        self.stream_at = self.host_at + self.size
        self.book = book
        self.shards = shards
        self.writer = PartsWriter(base, part_bytes)
        self.lengths = np.zeros(self.rows, dtype=np.uint8)
        self.bases = np.zeros(self.groups, dtype=np.int64)
        self.used = 0
        self.escape_keys = []
        self.escape_exps = []
        self.escapes = 0
        self.done = 0

    def add(self, first, count, encoded):
        signs, sizes, stream, heads, keys, exps = encoded
        at = self.used
        self.used += int(stream.shape[0])
        self.writer.write_at(self.host_at + first * self.width, np.ascontiguousarray(signs))
        self.writer.write_at(self.stream_at + at, np.ascontiguousarray(stream))
        self.lengths[first: first + count] = sizes
        self.bases[first // self.group: first // self.group + heads.shape[0]] = heads + at
        if keys.shape[0]:
            self.escape_keys.append(keys)
            self.escape_exps.append(exps)
            self.escapes += int(keys.shape[0])
        self.done += count

    def finish(self, extra, directory):
        if self.done != self.rows:
            raise RuntimeError(f"{self.done:,} of {self.rows:,} rows were encoded")
        if self.escapes:
            keys = np.concatenate(self.escape_keys)
            exps = np.concatenate(self.escape_exps)
            order = np.argsort(keys, kind="stable")
            keys, exps = keys[order], exps[order]
        else:
            keys, exps = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8)
        self.escape_keys = []
        self.escape_exps = []
        self.writer.write_at(self.host_at + self.lengths_at, self.lengths)
        self.writer.write_at(self.host_at + self.bases_at, self.bases)
        self.writer.write_at(self.stream_at + self.used, np.ascontiguousarray(keys, dtype="<i8"))
        self.writer.write_at(self.stream_at + self.used + 8 * keys.shape[0], np.ascontiguousarray(exps))
        total = self.stream_at + self.used + 9 * keys.shape[0]
        shape = types.SimpleNamespace(rows=self.rows, width=self.width, group=self.group, book=self.book,
                                      size=self.size, used=self.used, count=int(keys.shape[0]))
        header = hot.df11_header_bytes(shape, extra, reserve=hot.DF11_HEADER_RESERVE)
        self.writer.write_at(0, hot.DF11_MAGIC + len(header).to_bytes(8, "little") + header)
        self.keys, self.exps = keys, exps
        self.total = total
        self.directory = directory
        return total

    def decode_from_disk(self, ids):
        out = np.zeros((len(ids), self.width), dtype=np.uint16)
        for at, row in enumerate(ids):
            head = row // self.group * self.group
            end = min(head + self.group, self.rows)
            sizes = self.lengths[head:end]
            signs = np.frombuffer(self.writer.read_at(self.host_at + head * self.width, (end - head) * self.width),
                                  dtype=np.uint8).reshape(end - head, self.width)
            base = int(self.bases[row // self.group])
            stream = np.frombuffer(self.writer.read_at(self.stream_at + base, int(sizes.sum())), dtype=np.uint8)
            inside = (self.keys >= head * self.width) & (self.keys < end * self.width)
            out[at] = hot.df11_decode(self.book, signs, sizes, np.zeros(1, dtype=np.int64), stream,
                                      self.keys[inside] - head * self.width, self.exps[inside], self.group,
                                      np.array([row - head]))[0]
        return out

    def close(self):
        return self.writer.close(self.total)


def spread_rows(rows, count, seed=7):
    rng = np.random.default_rng(seed)
    edge = [0, rows - 1]
    inner = rng.integers(0, rows, size=max(0, count - len(edge)))
    return sorted(set(int(one) for one in edge) | set(int(one) for one in inner))


def build(directory, base, procs, part_bytes=PART_BYTES, task_rows=TASK_ROWS, report=say, expect_rows=None,
          check_rows=512):
    began = time.time()
    shards = hot.checkpoint_shards(directory)
    rows = sum(one["rows"] for one in shards)
    width = shards[0]["width"]
    if any(one["dtype"] != "BF16" or one["width"] != width for one in shards):
        raise SystemExit("the PLE shards are not all BF16 of one width")
    if expect_rows is not None and rows != int(expect_rows):
        raise SystemExit(f"the PLE shards hold {rows:,} rows, the serve expects {int(expect_rows):,}")
    raw_gib = rows * 2 * width / 2**30
    report(f"{len(shards)} PLE shards, {rows:,} rows x {width} BF16 = {raw_gib:.1f} GiB raw, {procs} encode "
           f"process(es)")
    book, sample = sample_book(shards, directory)
    report(f"codebook from {book.sampled:,} rows spread over the table: "
           f"{int(np.count_nonzero(book.lengths[:hot.DF11_ESCAPE]))} exponents coded, longest "
           f"{int(book.lengths.max())} bits, {book.sample_bits:.3f} bits an exponent on the sample")
    rate = measure_rate(book, sample, width)
    plan = estimate(rows, width, procs, rate)
    report(f"estimate: {rate:,.0f} rows/s a process x {procs} = {rate * procs:,.0f} rows/s -> encode "
           f"{plan['encode_s'] / 60:.0f} min; reading {plan['raw_gib']:.0f} GiB at 0.15-0.6 GiB/s "
           f"{plan['read_s'][0] / 60:.0f}-{plan['read_s'][1] / 60:.0f} min; the two overlap, so "
           f"{plan['low_s'] / 60:.0f}-{plan['high_s'] / 60:.0f} min")
    del sample
    out = Streamed(base, shards, book, part_bytes)
    tasks = [(first, width, segments_of(shards, directory, first, min(first + task_rows, rows)))
             for first in range(0, rows, task_rows)]
    context = multiprocessing.get_context("spawn")
    last = time.time()
    done_before = 0
    with context.Pool(procs, initializer=init_worker, initargs=(book.lengths.tolist(), book.limit)) as pool:
        for first, count, encoded in pool.imap_unordered(encode_task, tasks):
            out.add(first, count, encoded)
            if time.time() - last >= REPORT_EVERY_S:
                spent = time.time() - began
                speed = (out.done - done_before) / (time.time() - last)
                done_before, last = out.done, time.time()
                report(f"{out.done:,} of {rows:,} rows ({out.done / rows:.1%}), {speed:,.0f} rows/s, "
                       f"{out.used / 2**30:.1f} GiB of exponent codes, {out.escapes:,} escaped exponents, "
                       f"{spent / 60:.1f} min spent, eta {(rows - out.done) / max(speed, 1) / 60:.0f} min")
    rows_probe = spread_rows(rows, hot.DF11_PROBE_ROWS)
    raw = hot.raw_rows(directory, shards, rows_probe)
    probes = [[row, hot.row_digest(raw[at])] for at, row in enumerate(rows_probe)]
    extra = {"shards": hot.shard_identity(shards), "fingerprint": hot.shard_fingerprint(shards), "probes": probes,
             "baked_rows_per_s": round(rate * procs), "part_bytes": int(part_bytes)}
    total = out.finish(extra, directory)
    report(f"encoded {rows:,} rows in {(time.time() - began) / 60:.1f} min: {total / 2**30:.2f} GiB "
           f"({total / rows:.2f} B a row) of {raw_gib:.2f} GiB raw, {out.escapes:,} escaped exponents")
    check = spread_rows(rows, check_rows, seed=11)
    back = out.decode_from_disk(check)
    if not np.array_equal(back, hot.raw_rows(directory, shards, check)):
        raise SystemExit("rows decoded from the written file differ from the checkpoint")
    report(f"read back {len(check)} rows from the written file and from the checkpoint: bit-identical")
    files = out.close()
    return {"files": [str(one) for one in files], "bytes": total, "rows": rows, "width": width,
            "shards": len(shards), "seconds": round(time.time() - began, 1), "escapes": out.escapes,
            "fingerprint": extra["fingerprint"], "raw_bytes": rows * 2 * width,
            "digest": hashlib.sha256(json.dumps(probes).encode("utf-8")).hexdigest()}
