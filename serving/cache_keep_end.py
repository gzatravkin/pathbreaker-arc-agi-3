import dataclasses
import os
import time

from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

KEPT = 2
WINDOW_STEPS = 4
REPORT_S = 60.0
SEEN_KEYS = 8192
RAM_ENV = "TAAF_VLLM_CACHE_KEEP_END_RAM"
RAM_MODES = ("1", "pin")
PAUSED_ENV = "TAAF_VLLM_PAUSED_REGISTER"
PAUSED_STATUS = "WAITING_FOR_STREAMING_REQ"
_SAID: set = set()


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


@dataclasses.dataclass
class Spare:
    boundary: int
    blocks: list


@dataclasses.dataclass
class Snapshot:
    boundary: int
    blocks: list
    keys: list


@dataclasses.dataclass
class Tally:
    armed: int = 0
    registered: int = 0
    released: int = 0
    tight: int = 0
    dropped: int = 0
    hits: int = 0
    reused: int = 0
    finished: int = 0
    attention_keys: int = 0
    paused_keys: int = 0
    paused_sessions: int = 0
    ram_queued: int = 0
    ram_blocks: int = 0
    ram_evicted: int = 0
    ram_none: int = 0
    ram_full: int = 0
    ram_on_cpu: int = 0
    pin_moved: int = 0
    pin_stale: int = 0


def paused_on():
    return str(os.environ.get(PAUSED_ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def ram_mode():
    said = str(os.environ.get(RAM_ENV) or "").strip()
    if said in ("", "0"):
        return ""
    if said not in RAM_MODES:
        raise ValueError(f"{RAM_ENV} must be one of {', '.join(RAM_MODES)} or unset, found {said!r}")
    return said


def install_ram_store(tails):
    from vllm.v1.simple_kv_offload import manager as offload
    owner = offload.SimpleCPUOffloadScheduler
    original = owner._queue_finished_eager_store
    holder = getattr(original, "taaf_reply_ram", None)
    if holder is not None:
        holder[0] = tails
        return
    holder = [tails]

    def queue(self, request, block_ids):
        original(self, request, block_ids)
        holder[0].ram_store(self, request, offload.TransferMeta)

    queue.taaf_reply_ram = holder
    owner._queue_finished_eager_store = queue


def refusal(manager):
    coordinator = manager.coordinator
    pool = getattr(coordinator, "block_pool", None)
    if not manager.enable_caching or pool is None:
        return "prefix caching is off"
    if not getattr(coordinator, "enable_partial_hash_hits", False):
        return "the coordinator takes no hits finer than its scheduler block"
    mambas = [m for m in coordinator.single_type_managers if type(m) is MambaManager]
    if not mambas:
        return "the model has no Mamba cache group"
    if any(m.mamba_cache_mode != "align" or not m.enable_caching for m in mambas):
        return "a Mamba group is not cached in align mode"
    if any(not hasattr(m, "_retain_tail_copy") for m in mambas):
        return "CACHE_KEEP's retained-copy hook is absent from MambaManager"
    if any(m.block_size <= pool.hash_block_size for m in mambas):
        return "the match unit is not finer than the Mamba block"
    return None


class ReplyTails:
    def __init__(self, manager):
        coordinator = manager.coordinator
        self.pool = coordinator.block_pool
        self.unit = self.pool.hash_block_size
        self.mambas = [m for m in coordinator.single_type_managers if type(m) is MambaManager]
        self.attention = [m for m in coordinator.single_type_managers
                          if type(m) is FullAttentionManager and m.enable_caching
                          and m.kv_cache_spec.prefix_cacheable and m.block_size > self.unit]
        self.reserve = max(4 * len(self.mambas), self.pool.num_gpu_blocks // 16)
        self.spares: dict[str, Spare] = {}
        self.held: dict[str, list[Snapshot]] = {}
        self.sends: dict[str, tuple[int, dict[int, int]]] = {}
        self.keys: dict = {}
        self.seen: set = set()
        self.tally = Tally()
        self.total = Tally()
        self.reported = time.monotonic()
        self.connector = None
        self.ram = ram_mode()
        self.pinned: dict[str, Snapshot] = {}
        self.paused = paused_on()
        self.keyed: dict[str, int] = {}

    @classmethod
    def attach(cls, manager):
        why = refusal(manager)
        if why is not None:
            say("inert", f"CACHE_KEEP_END inert on this scheduler: {why}")
            return None
        made = cls(manager)
        say("armed", f"CACHE_KEEP_END scheduler: {len(made.mambas)} Mamba group(s) get one spare "
                     f"block each while a decoding request is within {WINDOW_STEPS} steps of its "
                     f"next {made.unit}-token boundary; the GPU writes the exact recurrent state "
                     f"there and the spare is registered under that boundary once the step is "
                     f"processed; the newest {KEPT} snapshots of a request stay cached, older ones "
                     f"go back to the free list; a finished request's attention blocks are keyed "
                     f"at every {made.unit}-token boundary of its reply; spares are skipped while "
                     f"fewer than {made.reserve} blocks are free")
        if made.paused:
            say("paused", f"CACHE_KEEP_END paused sessions: a request waiting for its next streaming chunk has "
                          f"its attention blocks keyed at every {made.unit}-token boundary it has confirmed "
                          f"({PAUSED_ENV}), so a second request sharing its history matches up to its newest "
                          f"reply snapshot instead of its last full block")
        if made.ram:
            install_ram_store(made)
            say("ram", f"CACHE_KEEP_END RAM armed ({RAM_ENV}={made.ram}): when a request finishes, its "
                       f"newest snapshot at least one unit below its confirmed end and the attention "
                       f"blocks holding that boundary and the end are queued to the RAM tier with the "
                       f"connector's finished stores, keyed at every unit boundary between them"
                       + ("; the snapshot before the newest is the one pinned, so it is the reachable "
                          "one when the request ends" if made.ram == "pin" else ""))
        return made

    def on_schedule(self, scheduler):
        window = WINDOW_STEPS * (1 + int(getattr(scheduler, "num_spec_tokens", 0) or 0))
        self.connector = getattr(scheduler, "connector", None)
        running = set()
        for request in scheduler.running:
            rid = request.request_id
            running.add(rid)
            if rid not in self.seen:
                self.seen.add(rid)
                self._observe(request)
            confirmed = request.num_computed_tokens - request.num_output_placeholders
            spare = self.spares.get(rid)
            if spare is not None and confirmed >= spare.boundary:
                self._register(request, self.spares.pop(rid))
                self.sends[rid] = (0, {})
                spare = None
            if spare is None:
                self._arm(request, window)
        self.seen &= running
        if self.paused:
            self._key_paused(scheduler)
        sent = self.sends
        self.sends = {}
        self._report()
        return sent or None

    def on_free(self, request):
        rid = request.request_id
        released = []
        confirmed = min(request.num_computed_tokens - request.num_output_placeholders,
                        request.num_tokens)
        top = max(confirmed, 0) // self.unit * self.unit
        spare = self.spares.pop(rid, None)
        if spare is not None:
            if spare.boundary + self.unit <= top:
                self._register(request, spare)
            else:
                released.extend(spare.blocks)
                self.tally.released += 1
        self._attention_keys(request, top)
        self.keyed.pop(rid, None)
        self.pinned.pop(rid, None)
        for snapshot in self.held.pop(rid, []):
            if snapshot.boundary > top - self.unit:
                self._drop(snapshot)
        self.sends.pop(rid, None)
        self.seen.discard(rid)
        self.tally.finished += 1
        return released

    def _arm(self, request, window):
        computed = request.num_computed_tokens
        if computed < request.num_prompt_tokens or computed < request.num_tokens - 1:
            return
        boundary = (computed // self.unit + 1) * self.unit
        if boundary - computed > window:
            return
        if boundary > request.num_prompt_tokens + request.max_tokens:
            return
        if self.pool.get_num_free_blocks() - len(self.mambas) < self.reserve:
            self.tally.tight += 1
            return
        blocks = self.pool.get_new_blocks(len(self.mambas))
        self.spares[request.request_id] = Spare(boundary, blocks)
        self.sends[request.request_id] = (
            boundary, {m.kv_cache_group_id: b.block_id for m, b in zip(self.mambas, blocks)})
        self.tally.armed += 1

    def _register(self, request, spare):
        rid = request.request_id
        keys = []
        for manager, block in zip(self.mambas, spare.blocks):
            keys.append(self.pool.cache_partial_block(
                request=request, block=block, num_tokens=spare.boundary,
                kv_cache_group_id=manager.kv_cache_group_id, block_size=manager.block_size,
                replace_existing_hashes=True))
        held = self.held.setdefault(rid, [])
        if self.ram == "pin":
            self._pin_previous(rid, held)
        else:
            for manager, block in zip(self.mambas, spare.blocks):
                manager._retain_tail_copy(rid, block)
        self.pool.free_blocks(spare.blocks)
        held.append(Snapshot(spare.boundary, spare.blocks, keys))
        while len(held) > KEPT:
            pinned = self.pinned.get(rid)
            victim = next(one for one in held if one is not pinned)
            held.remove(victim)
            self._drop(victim)
        unit_hash = request.block_hashes[spare.boundary // self.unit - 1]
        self.keys[unit_hash] = (spare.boundary, request.num_prompt_tokens)
        while len(self.keys) > SEEN_KEYS:
            self.keys.pop(next(iter(self.keys)))
        self.tally.registered += 1
        say("registered", f"CACHE_KEEP_END first reply snapshot: request {request.request_id} "
                          f"at {spare.boundary} tokens ({spare.boundary - request.num_prompt_tokens} "
                          f"into its reply), {len(spare.blocks)} Mamba block(s) registered")

    def _pin_previous(self, rid, held):
        previous = held[-1] if held else None
        if previous is None or previous is self.pinned.get(rid):
            return
        if not self._holds(previous):
            self.tally.pin_stale += 1
            return
        self.pool.touch(previous.blocks)
        for manager, block in zip(self.mambas, previous.blocks):
            manager._retain_tail_copy(rid, block)
        self.pool.free_blocks(previous.blocks)
        self.pinned[rid] = previous
        self.tally.pin_moved += 1

    def _holds(self, snapshot):
        return all(key is not None and self.pool.cached_block_hash_to_block.contain(key, block.block_id)
                   for key, block in zip(snapshot.keys, snapshot.blocks))

    def _top(self, request):
        confirmed = min(request.num_computed_tokens - request.num_output_placeholders,
                        request.num_tokens)
        return max(confirmed, 0) // self.unit * self.unit

    def _reachable(self, rid, top):
        found = [one for one in self.held.get(rid, []) if one.boundary <= top - self.unit]
        return max(found, key=lambda one: one.boundary) if found else None

    def _boundary_blocks(self, request, boundary, top):
        hashes = request.block_hashes
        first = max(boundary, request.num_prompt_tokens // self.unit * self.unit + self.unit)
        out = []
        for manager in self.attention:
            blocks = manager.req_to_blocks.get(request.request_id)
            if not blocks:
                continue
            for edge in range(first, top + 1, self.unit):
                if edge % manager.block_size == 0:
                    continue
                index = edge // manager.block_size
                if index >= len(blocks) or edge // self.unit > len(hashes):
                    break
                block = blocks[index]
                if block.is_null:
                    continue
                key = make_block_hash_with_group_id(hashes[edge // self.unit - 1],
                                                    manager.kv_cache_group_id)
                self.pool._insert_block_hash(key, block, num_tokens=edge)
                if all(block is not other for other in out):
                    out.append(block)
        return out

    def ram_store(self, offloader, request, transfer):
        if offloader._gpu_block_pool is None or getattr(offloader, "_lazy_mode", False):
            return
        rid = request.request_id
        top = self._top(request)
        snapshot = self._reachable(rid, top)
        if snapshot is None:
            self.tally.ram_none += 1
            return
        if not self._holds(snapshot):
            self.tally.ram_evicted += 1
            return
        cpu = offloader.cpu_block_pool
        if all(cpu.cached_block_hash_to_block.get_one_block(key) is not None for key in snapshot.keys):
            self.tally.ram_on_cpu += 1
            return
        in_flight = offloader._in_flight_store_gpu_blocks
        ids = []
        for block in [*snapshot.blocks, *self._boundary_blocks(request, snapshot.boundary, top)]:
            if block.is_null or block.block_id in in_flight or block.block_id in ids:
                continue
            ids.append(block.block_id)
        if not ids:
            return
        if len(ids) >= cpu.get_num_free_blocks():
            self.tally.ram_full += 1
            return
        gpu = offloader._gpu_block_pool
        stored = cpu.get_new_blocks(len(ids))
        meta = [{} for _ in ids] if offloader.enable_kv_cache_events else None
        offloader._pending_finished_stores.append(transfer(ids, [one.block_id for one in stored], meta))
        in_flight.update(ids)
        gpu.touch([gpu.blocks[one] for one in ids])
        self.tally.ram_queued += 1
        self.tally.ram_blocks += len(ids)

    def _drop(self, snapshot):
        for block, key in zip(snapshot.blocks, snapshot.keys):
            if key is None or not self.pool.cached_block_hash_to_block.contain(key, block.block_id):
                continue
            self.pool._maybe_evict_cached_block(block)
            if block.ref_cnt == 0 and not block.is_null:
                self.pool.free_block_queue.remove(block)
                self.pool.free_block_queue.prepend_n([block])
        self.tally.dropped += 1

    def _attention_keys(self, request, top):
        first = request.num_prompt_tokens // self.unit * self.unit + self.unit
        self.tally.attention_keys += self._keys_between(request, first, top)

    def _keys_between(self, request, first, top):
        hashes = request.block_hashes
        count = 0
        for manager in self.attention:
            blocks = manager.req_to_blocks.get(request.request_id)
            if not blocks:
                continue
            for boundary in range(first, top + 1, self.unit):
                if boundary % manager.block_size == 0:
                    continue
                index = boundary // manager.block_size
                if index >= len(blocks) or boundary // self.unit > len(hashes):
                    break
                block = blocks[index]
                if block.is_null:
                    continue
                key = make_block_hash_with_group_id(hashes[boundary // self.unit - 1],
                                                    manager.kv_cache_group_id)
                self.pool._insert_block_hash(key, block, num_tokens=boundary)
                count += 1
        return count

    def _key_paused(self, scheduler):
        requests = getattr(scheduler, "requests", None) or {}
        for rid, request in requests.items():
            if getattr(request.status, "name", "") != PAUSED_STATUS:
                continue
            confirmed = min(request.num_computed_tokens - request.num_output_placeholders,
                            request.num_tokens)
            top = max(confirmed - 1, 0) // self.unit * self.unit
            done = self.keyed.get(rid)
            if done is None or done > top:
                done = request.num_prompt_tokens // self.unit * self.unit
            if top > done:
                made = self._keys_between(request, done + self.unit, top)
                self.tally.paused_keys += made
                self.tally.paused_sessions += 1
            self.keyed[rid] = max(top, done)
        if len(self.keyed) > len(requests):
            for rid in [one for one in self.keyed if one not in requests]:
                del self.keyed[rid]

    def _observe(self, request):
        hit = request.num_computed_tokens
        if hit <= 0 or hit % self.unit or hit // self.unit > len(request.block_hashes):
            return
        found = self.keys.get(request.block_hashes[hit // self.unit - 1])
        if found is None:
            return
        boundary, prompt = found
        self.tally.hits += 1
        self.tally.reused += max(boundary - (prompt // self.unit * self.unit - self.unit), 0)
        say("hit", f"CACHE_KEEP_END first resume from a reply snapshot: request "
                   f"{request.request_id} hit {hit} tokens, {boundary - prompt} of them past the "
                   f"previous request's prompt")

    def _ram_line(self):
        reader = getattr(self.connector, "get_boundary_store_stats", None)
        stats = reader() if callable(reader) else None
        if stats is None:
            return None
        counts = dataclasses.asdict(stats) if dataclasses.is_dataclass(stats) else dict(vars(stats))
        line = "CACHE_KEEP_END RAM tier boundary hand-offs since start: " + ", ".join(
            f"{name} {value}" for name, value in counts.items())
        manager = getattr(self.connector, "scheduler_manager", None)
        pool = getattr(manager, "cpu_block_pool", None)
        if pool is not None and hasattr(manager, "num_cpu_blocks"):
            line += f"; RAM blocks free {pool.get_num_free_blocks()} of {manager.num_cpu_blocks}"
        return line

    def _report(self):
        now = time.monotonic()
        if now - self.reported < REPORT_S:
            return
        tally = self.tally
        if (tally.armed or tally.registered or tally.hits or tally.tight or tally.ram_queued
                or tally.paused_keys):
            span = now - self.reported
            print(f"CACHE_KEEP_END last {span:.0f} s: {tally.armed} spare(s) armed, "
                  f"{tally.registered} reply snapshot(s) registered, {tally.released} released "
                  f"unwritten, {tally.tight} skipped for a tight pool, {tally.dropped} dropped as "
                  f"stale; {tally.finished} request(s) freed with {tally.attention_keys} reply "
                  f"attention key(s); {tally.paused_sessions} paused-session pass(es) keyed "
                  f"{tally.paused_keys} attention boundary(ies); {tally.hits} new request(s) resumed from a reply snapshot, "
                  f"{tally.reused} prompt tokens past the stock prompt-tail hit; now "
                  f"{len(self.spares)} spare(s) pending, "
                  f"{sum(len(v) for v in self.held.values())} snapshot(s) held by running "
                  f"requests", flush=True)
            if self.ram:
                print(f"CACHE_KEEP_END RAM last {span:.0f} s: {tally.ram_queued} finished request(s) "
                      f"queued a reply snapshot to the RAM tier ({tally.ram_blocks} blocks), "
                      f"{tally.ram_evicted} found it evicted, {tally.ram_none} had none at least a "
                      f"unit below the end, {tally.ram_on_cpu} already on the tier, {tally.ram_full} "
                      f"refused by a full tier; pin moved {tally.pin_moved}, stale {tally.pin_stale}",
                      flush=True)
            ram = self._ram_line()
            if ram:
                print(ram, flush=True)
        for field in dataclasses.fields(Tally):
            setattr(self.total, field.name,
                    getattr(self.total, field.name) + getattr(tally, field.name))
        self.tally = Tally()
        self.reported = now
