import os
import time

ENV = "TAAF_VLLM_PARK"
RAM_ENV = "TAAF_VLLM_PARK_RAM_GIB"
HOLD_ENV = "TAAF_VLLM_PARK_HOLD_S"
VERIFY_ENV = "TAAF_VLLM_PARK_VERIFY"
DISK_ENV = "TAAF_VLLM_PARK_DISK"
DISK_GIB_ENV = "TAAF_VLLM_PARK_DISK_GIB"
HOST_DISK_ENV = "TAAF_VLLM_PARK_HOST_DISK"
DEFAULT_DISK_DIR = "/kaggle/tmp/park"
DEFAULT_DISK_GIB = 32.0
DEFAULT_RAM_GIB = 8.0
DEFAULT_HOLD_S = 600.0
REPORT_S = 60.0
GIB = 1 << 30
MIB = 1 << 20
_WORKER = {"state": None, "failed": None}
_ENGINE = {"installed": False}
_STATUS = {}


def say(text):
    print(f"PARK {text}", flush=True)


def _number(env, default):
    said = str(os.environ.get(env) or "").strip()
    return float(said) if said else default


def verify_on():
    return str(os.environ.get(VERIFY_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def disk_dir():
    said = str(os.environ.get(DISK_ENV) or "").strip()
    if said.lower() in ("", "0", "false", "no", "off"):
        return ""
    return DEFAULT_DISK_DIR if said.lower() in ("1", "true", "yes", "on") else said


def host_disk_on():
    return str(os.environ.get(HOST_DISK_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def host_disk_dir():
    return disk_dir() or DEFAULT_DISK_DIR


def ram_bytes():
    if disk_dir():
        return int(_number(DISK_GIB_ENV, DEFAULT_DISK_GIB) * GIB)
    return int(_number(RAM_ENV, DEFAULT_RAM_GIB) * GIB)


def hold_seconds():
    return _number(HOLD_ENV, DEFAULT_HOLD_S)


def statuses():
    if not _STATUS:
        from vllm.v1.request import RequestStatus
        _STATUS["waiting"] = RequestStatus.WAITING
        _STATUS["paused"] = RequestStatus.WAITING_FOR_STREAMING_REQ
        _STATUS["running"] = RequestStatus.RUNNING
    return _STATUS


def layout(config, host=False):
    count = int(config.hisparse_host_num_blocks if host else config.num_blocks)
    regions, seen = [], set()
    for tensor in config.kv_cache_tensors:
        if bool(getattr(tensor, "host_resident", False)) != host:
            continue
        size = int(tensor.block_stride)
        span_blocks = count * size
        if int(tensor.layer_stride) >= span_blocks:
            start, span = int(tensor.offset), len(tensor.layers) * int(tensor.layer_stride)
        else:
            start, span = 0, span_blocks
        if span_blocks <= 0 or span % span_blocks:
            raise ValueError(f"a KV cache tensor of {len(tensor.layers)} layer(s) spans {span} bytes, not a whole "
                             f"number of {count}-block regions of {size} bytes")
        for index in range(span // span_blocks):
            begin = start + index * span_blocks
            if begin in seen:
                continue
            seen.add(begin)
            regions.append((begin, size))
    return regions


def block_bytes(config, host=False):
    return sum(size for _, size in layout(config, host))


def runs(sources, targets):
    made = []
    for source, target in zip(sources, targets):
        if made and made[-1][0] + made[-1][2] == source and made[-1][1] + made[-1][2] == target:
            made[-1][2] += 1
        else:
            made.append([int(source), int(target), 1])
    return made


class Entry:
    __slots__ = ("group", "slot", "hashes", "tokens")

    def __init__(self, group, hashes, tokens):
        self.group = group
        self.slot = None
        self.hashes = hashes
        self.tokens = tokens


class Record:
    __slots__ = ("key", "rid", "entries", "pins", "boundary", "tokens", "at", "host")

    def __init__(self, key, rid, entries, pins, boundary, tokens, host=None):
        self.key = key
        self.rid = rid
        self.entries = entries
        self.pins = pins
        self.boundary = boundary
        self.tokens = tokens
        self.at = time.monotonic()
        self.host = host


class HostPark:
    __slots__ = ("path", "entries", "bytes", "ms")

    def __init__(self, path, entries, moved, spent):
        self.path = path
        self.entries = entries
        self.bytes = moved
        self.ms = spent


class Hold:
    __slots__ = ("key", "rid", "blocks", "pins", "until", "seen", "boundary")

    def __init__(self, key, rid, blocks, pins, until, boundary):
        self.key = key
        self.rid = rid
        self.blocks = blocks
        self.pins = pins
        self.until = until
        self.seen = None
        self.boundary = boundary


class Park:
    def __init__(self, scheduler):
        manager = scheduler.kv_cache_manager
        self.scheduler = scheduler
        self.coordinator = manager.coordinator
        self.managers = list(self.coordinator.single_type_managers)
        self.pool = manager.block_pool
        groups = list(scheduler.kv_cache_config.kv_cache_groups)
        cacheable = [index for index, group in enumerate(groups) if group.kv_cache_spec.prefix_cacheable]
        self.saved = [index for index in cacheable if self.managers[index].block_pool is self.pool]
        self.pinned = [index for index in cacheable if self.managers[index].block_pool is not self.pool]
        self.sizes = [int(one.block_size) for one in self.managers]
        self.block_bytes = block_bytes(scheduler.kv_cache_config)
        self.slots_total = ram_bytes() // self.block_bytes if self.block_bytes else 0
        self.free_slots = list(range(self.slots_total - 1, -1, -1))
        self.hold_s = hold_seconds()
        self.records = {}
        self.holds = {}
        self.tally = {"parks": 0, "park_blocks": 0, "park_bytes": 0, "park_ms": 0.0, "refused": 0, "resumes": 0,
                      "resume_copied": 0, "resume_in_place": 0, "resume_bytes": 0, "resume_ms": 0.0,
                      "resume_refused": 0, "dropped": 0, "holds_released": 0, "holds_expired": 0}
        self.reported = time.monotonic()
        self.changed = False
        kinds = {}
        for index in self.saved:
            kinds[type(self.managers[index]).__name__] = kinds.get(type(self.managers[index]).__name__, 0) + 1
        self.host_group = None
        if host_disk_on():
            self.arm_host_disk(scheduler.kv_cache_config)
        say(f"armed: a park copies a paused session's prefix-cached GPU blocks at its newest registered boundary "
            f"({', '.join(f'{count} {name}' for name, count in kinds.items())} group(s)) into a pinned RAM store "
            f"of {self.slots_total} slots ({self.slots_total * self.block_bytes / GIB:.2f} GiB, "
            f"{self.block_bytes / MIB:.2f} MiB a block) and pins its {len(self.pinned)} host-pool group(s)' pages; "
            f"a resume copies back what the pool evicted, re-registers it under the same hashes and holds it until "
            f"the session's next request is admitted (at most {self.hold_s:.0f} s)")

    def arm_host_disk(self, config):
        hosts = [index for index in self.pinned
                 if getattr(getattr(self.managers[index], "coordinator", None), "host_manager", None)
                 is self.managers[index]]
        why = ""
        if len(hosts) != 1:
            why = f"{len(hosts)} host-pool group(s) found, not one"
        elif getattr(config, "hisparse_host_num_blocks", None) is None:
            why = "the KV config names no host-pool block count"
        elif getattr(config, "hisparse_shared_host_pool", False):
            why = "the host pool is a region shared between ranks"
        if not why:
            try:
                self.host_block_bytes = block_bytes(config, host=True)
            except ValueError as failure:
                why = str(failure)
        if not why and not self.host_block_bytes:
            why = "the host pool has no block bytes"
        if why:
            say(f"host disk: {HOST_DISK_ENV} is set but not armed ({why}); a park pins the host pages as before")
            return
        self.host_group = hosts[0]
        self.host_pool = self.managers[self.host_group].block_pool
        self.host_dir = host_disk_dir()
        self.host_files = 0
        self.tally.update(host_saved=0, host_saved_bytes=0, host_saved_ms=0.0, host_save_failed=0, host_in_place=0,
                          host_read=0, host_read_bytes=0, host_read_ms=0.0, host_fallbacks=0)
        say(f"host disk: a park writes the session's host-pool pages ({self.host_block_bytes / MIB:.2f} MiB a page, "
            f"group {self.host_group}) to a file in {self.host_dir} and releases them to the host pool; a resume "
            f"reads back the pages the pool reused, re-registers them under the same hashes and holds them with the "
            f"GPU blocks; a missing page or a full host pool falls back to recomputing the prefix")

    def save_host(self, executor, kept):
        extras = getattr(self.host_pool, "cached_block_hashes_by_block", None) or {}
        entries = []
        for position, block in enumerate(kept):
            if block.block_hash is None:
                return None, "an unhashed host page"
            entry = Entry(self.host_group, (block.block_hash, *sorted(extras.get(block.block_id, ()))),
                          block.block_hash_num_tokens)
            entry.slot = position
            entries.append(entry)
        self.host_files += 1
        path = os.path.join(self.host_dir, f"host-{os.getpid()}-{self.host_files}.bin")
        try:
            done = host_rpc(executor, True, path, [block.block_id for block in kept], None, self.host_block_bytes)
        except Exception as failure:
            remove(path)
            return None, f"{type(failure).__name__}: {failure}"
        return HostPark(path, entries, int(done["bytes"]), float(done["ms"])), None

    def device_blocks(self, rid):
        count = 0
        for index, manager in enumerate(self.managers):
            if manager.block_pool is not self.pool:
                continue
            count += sum(1 for block in manager.req_to_blocks.get(rid, ()) if not block.is_null)
            tail = (getattr(manager, "_retained_tail_blocks", None) or {}).get(rid)
            if tail is not None:
                count += 1
        return count

    def find(self, rid):
        requests = self.scheduler.requests
        held = requests.get(rid)
        if held is not None:
            return held
        prefix = f"{rid}-"
        return next((request for name, request in requests.items() if name.startswith(prefix)), None)

    def gauges(self):
        return {"sessions_parked": len(self.records), "holds": len(self.holds),
                "slots_used": self.slots_total - len(self.free_slots), "slots_total": self.slots_total,
                "store_gib": round(self.slots_total * self.block_bytes / GIB, 2)}

    def refuse(self, why, **extra):
        self.tally["refused"] += 1
        self.changed = True
        return dict({"parked": False, "why": why}, **extra, **self.gauges())

    def lookup(self, request):
        name = "find_longest_cache_hit"
        groups = getattr(self.coordinator, "attention_groups", None) or ()
        classes = list(dict.fromkeys([group.manager_cls for group in groups] or [type(one) for one in self.managers]))
        found = {}
        held = [(cls, cls.__dict__.get(name), getattr(cls, name)) for cls in classes]

        def note(ids, blocks):
            for index, listed in zip(ids, blocks):
                kept = found.setdefault(index, {})
                for block in listed:
                    if not block.is_null:
                        kept[block.block_id] = block

        def watch(bound):
            def probe(cls, *args, **kwargs):
                blocks, length = bound(*args, **kwargs)
                ids = kwargs.get("kv_cache_group_ids", args[2] if len(args) > 2 else ())
                note(ids, blocks)
                if kwargs.get("drop_eagle_block"):
                    whole, _ = bound(*args, **dict(kwargs, drop_eagle_block=False))
                    note(ids, whole)
                return blocks, length
            return classmethod(probe)

        try:
            for cls, _, bound in held:
                setattr(cls, name, watch(bound))
            _, boundary, _ = self.coordinator.find_longest_cache_hit(request.block_hashes, int(request.num_tokens))
        finally:
            for cls, own, _ in held:
                if own is not None:
                    setattr(cls, name, own)
                elif name in cls.__dict__:
                    delattr(cls, name)
        return {index: list(kept.values()) for index, kept in found.items()}, boundary

    def park(self, executor, rid, key):
        started = time.perf_counter()
        self.settle()
        request = self.find(rid)
        if request is None:
            return self.refuse("no_request", rid=rid)
        if request.status != statuses()["paused"]:
            return self.refuse("not_paused", rid=rid, status=getattr(request.status, "name", str(request.status)))
        if key in self.records:
            self.drop(key, "parked again")
        found, boundary = self.lookup(request)
        tokens = int(request.num_tokens)
        if boundary <= 0:
            return self.refuse("no_boundary", rid=rid, tokens=tokens)
        entries, seen = [], set()
        extras = self.pool.cached_block_hashes_by_block
        counts = {}
        for index in self.saved:
            for block in found.get(index, ()):
                if block.block_id in seen:
                    continue
                if block.block_hash is None:
                    return self.refuse("unhashed_block", rid=rid, group=index)
                seen.add(block.block_id)
                hashes = (block.block_hash, *sorted(extras.get(block.block_id, ())))
                entry = Entry(index, hashes, block.block_hash_num_tokens)
                entry.slot = block.block_id
                entries.append(entry)
                name = type(self.managers[index]).__name__
                counts[name] = counts.get(name, 0) + 1
        pins, host_kept = [], []
        for index in self.pinned:
            kept = list({block.block_id: block for block in found.get(index, ())}.values())
            if index == self.host_group:
                host_kept = kept
                continue
            pins.append((index, kept))
        if len(entries) > len(self.free_slots):
            return self.refuse("store_full", rid=rid, need=len(entries), free=len(self.free_slots))
        sources = [entry.slot for entry in entries]
        slots = [self.free_slots.pop() for _ in entries]
        for entry, slot in zip(entries, slots):
            entry.slot = slot
        for index, kept in pins:
            self.managers[index].block_pool.touch(kept)
        try:
            copied = rpc(executor, True, sources, slots, self.slots_total)
        except Exception as failure:
            for index, kept in pins:
                self.managers[index].block_pool.free_blocks(kept)
            self.free_slots.extend(reversed(slots))
            return self.refuse(f"copy_failed: {type(failure).__name__}: {failure}", rid=rid)
        host = None
        if self.host_group is not None:
            host, why = self.save_host(executor, host_kept) if host_kept else (None, "")
            if host is None:
                self.host_pool.touch(host_kept)
                pins.append((self.host_group, host_kept))
                if why:
                    self.tally["host_save_failed"] += 1
                    say(f"host disk: park {key} {rid} kept its {len(host_kept)} host page(s) pinned in the pool, "
                        f"the write failed ({why})")
            else:
                self.tally["host_saved"] += len(host.entries)
                self.tally["host_saved_bytes"] += host.bytes
                self.tally["host_saved_ms"] += host.ms
        held = self.device_blocks(request.request_id)
        self.records[key] = Record(key, rid, entries, pins, boundary, tokens, host)
        spent = (time.perf_counter() - started) * 1000.0
        moved = len(entries) * self.block_bytes
        self.tally["parks"] += 1
        self.tally["park_blocks"] += len(entries)
        self.tally["park_bytes"] += moved
        self.tally["park_ms"] += spent
        self.changed = True
        pinned = sum(len(kept) for _, kept in pins)
        say(f"park {key} {rid}: boundary {boundary} of {tokens} tokens ({tokens - boundary} recomputed on resume "
            f"before the new turn); saved {len(entries)} block(s) "
            f"({', '.join(f'{count} {name}' for name, count in counts.items())}) = {moved / MIB:.1f} MiB in "
            f"{copied['ms']:.1f} ms of copy, {spent:.1f} ms in all; {pinned} host page(s) pinned; "
            + (f"{len(host.entries)} host page(s) = {host.bytes / MIB:.1f} MiB written to {host.path} in "
               f"{host.ms:.1f} ms and released to the host pool; " if host is not None else "")
            + f"the session holds "
            f"{held} GPU block(s); store {self.slots_total - len(self.free_slots)}/{self.slots_total} slots")
        return dict({"parked": True, "rid": rid, "boundary": boundary, "tokens": tokens, "blocks": len(entries),
                     "kinds": counts, "bytes": moved, "copy_ms": round(copied["ms"], 3), "ms": round(spent, 3),
                     "host_pages": pinned, "gpu_blocks": held}, **self.host_fields(host), **self.gauges())

    def restore(self, executor, key, rid):
        started = time.perf_counter()
        self.settle()
        record = self.records.get(key)
        if record is None:
            self.tally["resume_refused"] += 1
            return dict({"restored": False, "why": "no_record", "key": key}, **self.gauges())
        host_touched, host_copies = [], []
        if record.host is not None:
            host_touched, host_copies = self.split(self.host_pool, record.host.entries)
            self.host_pool.touch(host_touched)
            host_free = self.host_pool.get_num_free_blocks()
            if len(host_copies) > host_free:
                self.host_pool.free_blocks(host_touched)
                self.tally["resume_refused"] += 1
                return dict({"restored": False, "why": "no_room", "pool": "host", "key": key,
                             "need": len(host_copies), "free": host_free}, **self.gauges())
        table = self.pool.cached_block_hash_to_block
        touched, seen, copies = [], set(), []
        for entry in record.entries:
            found = [table.get_one_block(one) for one in entry.hashes]
            if all(block is not None for block in found):
                for block in found:
                    if block.block_id not in seen:
                        seen.add(block.block_id)
                        touched.append(block)
            else:
                copies.append(entry)
        self.pool.touch(touched)
        free = self.pool.get_num_free_blocks()
        if len(copies) > free:
            self.pool.free_blocks(touched)
            if record.host is not None:
                self.host_pool.free_blocks(host_touched)
            self.tally["resume_refused"] += 1
            return dict({"restored": False, "why": "no_room", "key": key, "need": len(copies), "free": free},
                        **self.gauges())
        fresh = self.pool.get_new_blocks(len(copies)) if copies else []
        host_fresh, read = [], None
        if host_copies:
            host_fresh = self.host_pool.get_new_blocks(len(host_copies))
            try:
                read = host_rpc(executor, False, record.host.path, [block.block_id for block in host_fresh],
                                [entry.slot for entry in host_copies], self.host_block_bytes,
                                len(record.host.entries))
            except Exception as failure:
                self.pool.free_blocks(fresh)
                self.pool.free_blocks(touched)
                self.host_pool.free_blocks(host_fresh)
                self.host_pool.free_blocks(host_touched)
                why = f"{type(failure).__name__}: {failure}"
                self.tally["host_fallbacks"] += 1
                self.drop(key, f"host pages unreadable, the session recomputes its prefix ({why})")
                self.tally["resume_refused"] += 1
                return dict({"restored": False, "why": f"host_pages_missing: {why}", "key": key}, **self.gauges())
        try:
            copied = rpc(executor, False, [entry.slot for entry in copies], [block.block_id for block in fresh],
                         self.slots_total)
        except Exception as failure:
            self.pool.free_blocks(fresh)
            self.pool.free_blocks(touched)
            if record.host is not None:
                self.host_pool.free_blocks(host_fresh)
                self.host_pool.free_blocks(host_touched)
            self.drop(key, "copy failed")
            self.tally["resume_refused"] += 1
            return dict({"restored": False, "why": f"copy_failed: {type(failure).__name__}: {failure}", "key": key},
                        **self.gauges())
        for entry, block in zip(copies, fresh):
            self.pool._insert_block_hash(entry.hashes[0], block, entry.tokens)
            for one in entry.hashes[1:]:
                self.pool._insert_block_hash(one, block, None)
        pins = record.pins
        if record.host is not None:
            for entry, block in zip(host_copies, host_fresh):
                self.host_pool._insert_block_hash(entry.hashes[0], block, entry.tokens)
                for one in entry.hashes[1:]:
                    self.host_pool._insert_block_hash(one, block, None)
            pins = [*pins, (self.host_group, host_touched + host_fresh)]
            remove(record.host.path)
            self.tally["host_in_place"] += len(record.host.entries) - len(host_copies)
            if read is not None:
                self.tally["host_read"] += len(host_copies)
                self.tally["host_read_bytes"] += int(read["bytes"])
                self.tally["host_read_ms"] += float(read["ms"])
        del self.records[key]
        self.free_slots.extend(entry.slot for entry in record.entries)
        stale = self.holds.pop(rid, None)
        if stale is not None:
            self.pool.free_blocks(stale.blocks)
            self.unpin(stale.pins)
        self.holds[rid] = Hold(key, rid, touched + fresh, pins, time.monotonic() + self.hold_s,
                               record.boundary)
        spent = (time.perf_counter() - started) * 1000.0
        moved = len(copies) * self.block_bytes
        self.tally["resumes"] += 1
        self.tally["resume_copied"] += len(copies)
        self.tally["resume_in_place"] += len(record.entries) - len(copies)
        self.tally["resume_bytes"] += moved
        self.tally["resume_ms"] += spent
        self.changed = True
        say(f"resume {key} -> {rid}: boundary {record.boundary} of {record.tokens} tokens; "
            f"{len(record.entries) - len(copies)} block(s) still in the pool, {len(copies)} copied back "
            f"({moved / MIB:.1f} MiB in {copied['ms']:.1f} ms of copy, {spent:.1f} ms in all"
            + (f", {copied['verified']} checksummed equal" if copied.get("verified") else "") + ") and re-registered; "
            + (f"{len(record.host.entries) - len(host_copies)} host page(s) still in the host pool, "
               f"{len(host_copies)} read back from {record.host.path} "
               f"({(read or {}).get('bytes', 0) / MIB:.1f} MiB in {(read or {}).get('ms', 0.0):.1f} ms) and "
               f"re-registered; " if record.host is not None else "")
            + f"held until {rid} is admitted; parked {(time.monotonic() - record.at):.0f} s")
        return dict({"restored": True, "rid": rid, "boundary": record.boundary, "tokens": record.tokens,
                     "copied": len(copies), "in_place": len(record.entries) - len(copies), "bytes": moved,
                     "copy_ms": round(copied["ms"], 3), "ms": round(spent, 3),
                     "verified": int(copied.get("verified") or 0)},
                    **({"host_in_place": len(record.host.entries) - len(host_copies), "host_read": len(host_copies),
                        "host_read_bytes": int((read or {}).get("bytes", 0)),
                        "host_read_ms": round(float((read or {}).get("ms", 0.0)), 3)}
                       if record.host is not None else {}), **self.gauges())

    def split(self, pool, entries):
        table = pool.cached_block_hash_to_block
        touched, seen, copies = [], set(), []
        for entry in entries:
            found = [table.get_one_block(one) for one in entry.hashes]
            if all(block is not None for block in found):
                for block in found:
                    if block.block_id not in seen:
                        seen.add(block.block_id)
                        touched.append(block)
            else:
                copies.append(entry)
        return touched, copies

    def host_fields(self, host):
        if host is None:
            return {}
        return {"host_disk_pages": len(host.entries), "host_disk_bytes": host.bytes,
                "host_disk_ms": round(host.ms, 3), "host_disk_path": host.path}

    def unpin(self, pins):
        for index, kept in pins:
            if kept:
                self.managers[index].block_pool.free_blocks(kept)

    def drop(self, key, why):
        record = self.records.pop(key, None)
        if record is None:
            return dict({"dropped": False, "key": key}, **self.gauges())
        self.free_slots.extend(entry.slot for entry in record.entries)
        self.unpin(record.pins)
        if record.host is not None:
            remove(record.host.path)
        self.tally["dropped"] += 1
        self.changed = True
        say(f"drop {key}: {why}; {len(record.entries)} slot(s) and the pinned host pages let go")
        return dict({"dropped": True, "key": key}, **self.gauges())

    def settle(self):
        if not self.holds:
            return
        now = time.monotonic()
        live = statuses()
        for rid, hold in list(self.holds.items()):
            if hold.seen is None:
                request = self.find(rid)
                if request is not None:
                    hold.seen = request.request_id
            else:
                request = self.scheduler.requests.get(hold.seen)
            admitted = request is not None and (request.status == live["running"] or request.num_computed_tokens > 0)
            gone = request is None and hold.seen is not None
            if not (admitted or gone or now >= hold.until):
                continue
            del self.holds[rid]
            self.pool.free_blocks(hold.blocks)
            self.unpin(hold.pins)
            self.changed = True
            if admitted or gone:
                self.tally["holds_released"] += 1
            else:
                self.tally["holds_expired"] += 1
                say(f"hold {hold.key} -> {rid}: expired after {self.hold_s:.0f} s before the request was admitted; "
                    f"its blocks stay cached but may be evicted")

    def report(self, force=False):
        now = time.monotonic()
        if not force and now - self.reported < REPORT_S:
            return
        if not force and not (self.changed or self.records or self.holds):
            return
        self.reported = now
        self.changed = False
        shown = {name: round(value, 1) if isinstance(value, float) else value for name, value in self.tally.items()}
        say(f"tally {dict(shown, **self.gauges())}")

    def begin(self):
        self.settle()
        self.report()


def state(scheduler):
    held = getattr(scheduler, "_taaf_park", None)
    if held is None:
        held = Park(scheduler)
        scheduler._taaf_park = held
    return held


def rpc(executor, outward, sources, targets, slots_total):
    if not sources:
        return {"ms": 0.0, "bytes": 0}
    answer = executor.collective_rpc(worker_copy, args=(bool(outward), list(sources), list(targets),
                                                        int(slots_total)))
    if isinstance(answer, list):
        answer = answer[0]
    return answer


def remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def host_rpc(executor, outward, path, blocks, positions, block_bytes, total=None):
    answer = executor.collective_rpc(worker_host, args=(bool(outward), str(path), list(blocks),
                                                        None if positions is None else list(positions),
                                                        int(block_bytes), total))
    if isinstance(answer, list):
        answer = answer[0]
    return answer


def _engine_park(self, rid, key):
    return state(self.scheduler).park(self.model_executor, str(rid), str(key))


def _engine_restore(self, key, rid):
    return state(self.scheduler).restore(self.model_executor, str(key), str(rid))


def _engine_drop(self, key):
    return state(self.scheduler).drop(str(key), "dropped by the client")


def _engine_stats(self):
    held = state(self.scheduler)
    held.settle()
    return dict(held.tally, **held.gauges())


def attach(scheduler):
    if _ENGINE["installed"]:
        return
    from vllm.v1.engine.core import EngineCore
    EngineCore.taaf_park = _engine_park
    EngineCore.taaf_park_restore = _engine_restore
    EngineCore.taaf_park_drop = _engine_drop
    EngineCore.taaf_park_stats = _engine_stats
    _ENGINE["installed"] = True


def begin(scheduler):
    state(scheduler).begin()


class Worker:
    def __init__(self, runner):
        import torch
        self.bind(runner)
        self.slots = ram_bytes() // self.block_bytes if self.block_bytes else 0
        started = time.monotonic()
        disk = disk_dir()
        self.path = None
        if disk:
            os.makedirs(disk, exist_ok=True)
            self.path = os.path.join(disk, f"park-store-{os.getpid()}.bin")
            self.store = torch.from_file(self.path, shared=True, size=max(1, self.slots * self.block_bytes),
                                         dtype=torch.int8)
        else:
            self.store = torch.empty(max(1, self.slots * self.block_bytes), dtype=torch.int8)
        if self.cuda and not disk:
            result = torch.cuda.cudart().cudaHostRegister(self.store.data_ptr(), self.store.nbytes, 0)
            code = getattr(result, "value", result)
            if code != 0:
                raise RuntimeError(f"PARK could not pin its {self.store.nbytes / GIB:.2f} GiB RAM store: "
                                   f"cudaHostRegister {result}")
        self.verify = verify_on()
        self.sums = {}
        self.hosts = []
        offset = 0
        for _, size in self.planned:
            self.hosts.append(self.store[offset:offset + self.slots * size].view(self.slots, size))
            offset += self.slots * size
        say(f"worker: {len(self.regions)} block region(s) of the KV allocation, {self.block_bytes / MIB:.2f} MiB a "
            f"block; {'disk' if self.path else 'RAM'} store {self.slots} slot(s), "
            f"{self.slots * self.block_bytes / GIB:.2f} GiB "
            f"{('in ' + self.path + ', pageable, no RAM pinned') if self.path else ('pinned' if self.cuda else 'unpinned')} "
            f"in {time.monotonic() - started:.1f} s"
            + ("; every copied block is checksummed on the card at park and compared after it comes back"
               if self.verify else ""))

    def bind(self, runner):
        import torch
        config = runner.kv_cache_config
        caches = [cache for cache in runner.kv_caches if isinstance(cache, torch.Tensor)]
        if not caches:
            raise RuntimeError("PARK found no device KV cache tensor to copy blocks from")
        storages = {cache.untyped_storage().data_ptr() for cache in caches}
        if len(storages) != 1:
            raise RuntimeError(f"PARK addresses a block as one offset into the KV backing allocation, but the device "
                               f"caches sit in {len(storages)} allocations")
        first = caches[0]
        raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(first.untyped_storage())
        count = int(config.num_blocks)
        planned = layout(config)
        regions = []
        for begin, size in planned:
            end = begin + count * size
            if end > raw.numel():
                raise RuntimeError(f"PARK: a region at byte {begin} of {count} blocks x {size} bytes ends past the "
                                   f"{raw.numel()}-byte KV allocation")
            regions.append(raw[begin:end].view(count, size))
        sizes = [size for _, size in planned]
        if hasattr(self, "planned") and sizes != [size for _, size in self.planned]:
            raise RuntimeError(f"PARK: the KV cache was rebuilt with blocks of {sizes} bytes, the RAM store holds "
                               f"{[size for _, size in self.planned]}")
        self.planned = planned
        self.regions = regions
        self.block_bytes = sum(sizes)
        self.device = raw.device
        self.cuda = raw.device.type == "cuda"

    def checksums(self, blocks):
        import torch
        index = torch.tensor(blocks, dtype=torch.long, device=self.device)
        total = None
        for region in self.regions:
            words = region.index_select(0, index).view(torch.int64)
            weights = torch.arange(1, words.shape[1] + 1, dtype=torch.int64, device=self.device)
            part = (words * weights).sum(dim=1)
            total = part if total is None else total * 1000003 + part
        return [int(value) for value in total.tolist()]

    def copy(self, outward, sources, targets):
        import torch
        if self.cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for source, target, length in runs(sources, targets):
            for region, host in zip(self.regions, self.hosts):
                if outward:
                    host[target:target + length].copy_(region[source:source + length], non_blocking=self.cuda)
                else:
                    region[target:target + length].copy_(host[source:source + length], non_blocking=self.cuda)
        if self.cuda:
            torch.cuda.synchronize(self.device)
        spent = (time.perf_counter() - started) * 1000.0
        checked = 0
        if self.verify and sources:
            if outward:
                self.sums.update(zip(targets, self.checksums(sources)))
            else:
                landed = self.checksums(targets)
                wrong = [(slot, block) for slot, block, value in zip(sources, targets, landed)
                         if self.sums.get(slot) != value]
                if wrong:
                    raise RuntimeError(f"PARK verify: {len(wrong)} of {len(targets)} block(s) came back with other "
                                       f"bytes than they left with (slot, block) {wrong[:4]}")
                checked = len(targets)
        return {"ms": spent, "bytes": len(sources) * self.block_bytes, "verified": checked}


def runner_of(worker):
    held = getattr(worker, "model_runner", None)
    if held is None:
        held = getattr(getattr(worker, "worker", None), "model_runner", None)
    return held


def worker_init(runner):
    try:
        held = _WORKER["state"]
        if held is None:
            _WORKER["state"] = Worker(runner)
        else:
            held.bind(runner)
            say(f"worker: the KV cache was rebuilt; the {held.slots}-slot RAM store now copies from it")
        _WORKER["failed"] = None
    except Exception as failure:
        _WORKER["failed"] = f"{type(failure).__name__}: {failure}"
        say(f"worker: the RAM store is not armed ({_WORKER['failed']}); every park is refused")


def worker_copy(worker, outward, sources, targets, slots_total):
    held = _WORKER["state"]
    if held is None:
        if _WORKER.get("failed"):
            raise RuntimeError(f"PARK: the RAM store is not armed: {_WORKER['failed']}")
        say("worker: the RAM store was not allocated with the KV cache, so the first copy allocates it")
        worker_init(runner_of(worker))
        held = _WORKER["state"]
        if held is None:
            raise RuntimeError(f"PARK: the RAM store is not armed: {_WORKER['failed']}")
    if int(slots_total) != held.slots:
        raise RuntimeError(f"PARK: the scheduler counts {slots_total} RAM slots and the worker {held.slots}; their "
                           f"block layouts disagree")
    if len(sources) != len(targets):
        raise RuntimeError(f"PARK: {len(sources)} source block(s) for {len(targets)} target(s)")
    if outward:
        bad = [slot for slot in targets if not 0 <= slot < held.slots]
    else:
        bad = [slot for slot in sources if not 0 <= slot < held.slots]
    if bad:
        raise RuntimeError(f"PARK: slots {bad[:4]} are outside the {held.slots}-slot store")
    return held.copy(bool(outward), list(sources), list(targets))


def host_regions(runner):
    import torch
    config = runner.kv_cache_config
    if getattr(config, "hisparse_shared_host_pool", False):
        raise RuntimeError("the host pool is a region shared between ranks")
    count = int(config.hisparse_host_num_blocks)
    pools = {}
    context = getattr(getattr(runner, "compilation_config", None), "static_forward_context", None) or {}
    for layer in context.values():
        runtime = getattr(getattr(layer, "hisparse_cache", None), "runtime", None)
        held = getattr(runtime, "registered_host_pool", None)
        if isinstance(held, torch.Tensor):
            pools[held.data_ptr()] = held
    if len(pools) != 1:
        raise RuntimeError(f"found {len(pools)} registered host pools, not one")
    raw = next(iter(pools.values())).view(torch.int8).view(-1)
    regions = []
    for begin, size in layout(config, host=True):
        end = begin + count * size
        if end > raw.numel():
            raise RuntimeError(f"a host region at byte {begin} of {count} blocks x {size} bytes ends past the "
                               f"{raw.numel()}-byte host pool")
        regions.append(raw[begin:end].view(count, size))
    return regions


def worker_host(worker, outward, path, blocks, positions, block_bytes, total):
    import torch
    regions = host_regions(runner_of(worker))
    if sum(region.shape[1] for region in regions) != block_bytes:
        raise RuntimeError(f"the scheduler counts {block_bytes} bytes a host page and the worker "
                           f"{sum(region.shape[1] for region in regions)}")
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    started = time.perf_counter()
    index = torch.tensor(blocks, dtype=torch.long)
    if outward:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "wb") as handle:
            for region in regions:
                handle.write(memoryview(region.index_select(0, index).numpy()).cast("B"))
        os.replace(temporary, path)
        moved = len(blocks) * block_bytes
    else:
        total = int(total)
        if len(positions) != len(blocks):
            raise RuntimeError(f"{len(positions)} file page(s) for {len(blocks)} host block(s)")
        if any(not 0 <= one < total for one in positions):
            raise RuntimeError(f"file pages {positions[:4]} are outside the file's {total}")
        size = os.path.getsize(path)
        if size != total * block_bytes:
            raise RuntimeError(f"{path} holds {size} bytes, not {total} pages of {block_bytes}")
        picked = torch.tensor(positions, dtype=torch.long)
        whole = list(positions) == list(range(total))
        with open(path, "rb") as handle:
            for region in regions:
                buffer = torch.empty((total, region.shape[1]), dtype=torch.int8)
                got = handle.readinto(memoryview(buffer.numpy()).cast("B"))
                if got != buffer.numel():
                    raise RuntimeError(f"{path} ended after {got} of {buffer.numel()} bytes of a region")
                region.index_copy_(0, index, buffer if whole else buffer.index_select(0, picked))
        moved = total * block_bytes
    return {"ms": (time.perf_counter() - started) * 1000.0, "bytes": moved}
