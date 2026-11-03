import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

SETTINGS = Path(__file__).with_name("taaf_slim_state.json")
FIELD = "taaf_slim"
REPORT_S = 60.0
WAIT_KEEP_S = 5.0
WAIT_KEEP_PER_STEP = 16
WAIT_ORDER_AGE = 1000.0
CONTEXTS = (25_000, 30_000, 40_000)
_SAID: set = set()


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def settings():
    try:
        held = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def on_step(scheduler):
    held = getattr(scheduler, FIELD, False)
    if held is False:
        held = Slim.attach(scheduler, settings())
        setattr(scheduler, FIELD, held)
    if held is not None:
        held.step(scheduler)


def make_room(scheduler):
    held = getattr(scheduler, FIELD, None)
    if not held or not held.soft:
        return False
    return held.rescue(scheduler)


def cdiv(a, b):
    return -(-a // b)


def refusal(scheduler, chosen):
    manager = getattr(scheduler, "kv_cache_manager", None)
    coordinator = getattr(manager, "coordinator", None)
    if coordinator is None or getattr(coordinator, "block_pool", None) is None:
        return "the scheduler has no KV cache coordinator"
    mambas = [m for m in coordinator.single_type_managers if type(m) is MambaManager]
    if not mambas:
        return "the model has no Mamba cache group"
    if (chosen.get("keep_soft") or chosen.get("tail_soft")) and any(
            not hasattr(m, "_retained_state_block_idx") or not hasattr(m, "_retained_tail_blocks")
            for m in mambas):
        named = "KEEP_SOFT" if chosen.get("keep_soft") else "TAIL_SOFT"
        return f"{named} was armed but CACHE_KEEP's retained-state hooks are absent from MambaManager"
    return None


@dataclasses.dataclass
class Tally:
    short: int = 0
    released: int = 0
    blocks: int = 0
    rescued: int = 0


@dataclasses.dataclass
class TailTally:
    short: int = 0
    released: int = 0
    blocks: int = 0
    front: int = 0


@dataclasses.dataclass
class WaitTally:
    requests: int = 0
    refreshes: int = 0
    empty: int = 0
    blocks: int = 0
    tokens: int = 0


class Slim:
    def __init__(self, scheduler, chosen):
        coordinator = scheduler.kv_cache_manager.coordinator
        self.pool = coordinator.block_pool
        managers = list(coordinator.single_type_managers)
        self.mambas = [m for m in managers if type(m) is MambaManager]
        self.attention = [m for m in managers if type(m) is FullAttentionManager]
        self.other = [m for m in managers if m not in self.mambas and m not in self.attention]
        groups = scheduler.kv_cache_config.kv_cache_groups
        self.kinds = {m.kv_cache_group_id: Counter(name.rsplit(".", 1)[-1]
                                                   for name in groups[m.kv_cache_group_id].layer_names)
                      for m in self.mambas}
        self.fixed = sum(self._pages(m, scheduler.vllm_config) for m in self.other)
        self.soft = bool(chosen.get("keep_soft"))
        self.tail = int(chosen.get("tail_soft") or 0)
        self.hard = self.tail < 0
        self.tail_tally = TailTally()
        self.tail_total = TailTally()
        self.ride = bool(chosen.get("ple_ride"))
        self.pins = all(hasattr(m, "_retained_tail_blocks") for m in self.mambas)
        self.floor = int(chosen.get("floor", 0))
        self.every = float(chosen.get("report_s", REPORT_S))
        self.tally = Tally()
        self.total = Tally()
        self.reported = None
        self.wait_order = bool(chosen.get("wait_order"))
        self.wait_keep = bool(chosen.get("wait_keep")) or self.wait_order
        self.wait_every = float(chosen.get("wait_keep_s", WAIT_KEEP_S))
        self.wait_age = float(chosen.get("wait_order_age", WAIT_ORDER_AGE))
        self.watched = {}
        self.cached = {}
        self.arrived = {}
        self.reordered = 0
        self.wait_tally = WaitTally()
        self.wait_total = WaitTally()

    @staticmethod
    def _pages(manager, vllm_config):
        spec = manager.kv_cache_spec
        try:
            return max(1, cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes))
        except Exception:
            return 1

    @classmethod
    def attach(cls, scheduler, chosen):
        why = refusal(scheduler, chosen)
        if why is not None:
            say("inert", f"SLIM_STATE inert on this scheduler: {why}")
            return None
        made = cls(scheduler, chosen)
        made.announce()
        return made

    def per_request(self, tokens, pinned=False):
        state = len(self.mambas) * (2 if pinned else 1)
        return sum(cdiv(tokens, m.block_size) for m in self.attention) + self.fixed + state

    def capacity(self, pinned=False):
        blocks = self.pool.num_gpu_blocks - 1
        return [blocks // self.per_request(tokens, pinned) for tokens in CONTEXTS]

    def announce(self):
        mixed = [gid for gid, kinds in sorted(self.kinds.items()) if len(kinds) > 1]
        layout = "; ".join(f"group {gid}: " + " + ".join(f"{count} {kind}"
                                                        for kind, count in sorted(kinds.items()))
                           for gid, kinds in sorted(self.kinds.items()))
        contexts = "/".join(f"{tokens:,}" for tokens in CONTEXTS)
        bare = "/".join(str(self.per_request(tokens)) for tokens in CONTEXTS)
        fits = "/".join(str(count) for count in self.capacity())
        line = (f"SLIM_STATE scheduler: {len(self.mambas)} Mamba cache group(s) ({layout}); "
                + (f"group(s) {mixed} hold more than one kind of Mamba layer, so a short-conv state "
                   f"shares their blocks"
                   if mixed else "every Mamba cache group holds one kind of layer")
                + f"; modeled per running request at {contexts} tokens: {bare} blocks with one "
                f"state block a group, so the {self.pool.num_gpu_blocks - 1}-block pool holds "
                f"{fits} requests")
        if self.pins:
            pinned = "/".join(str(self.per_request(tokens, True)) for tokens in CONTEXTS)
            held = "/".join(str(count) for count in self.capacity(True))
            line += (f"; with CACHE_KEEP's pinned snapshot beside every running state, {pinned} "
                     f"blocks and {held} requests")
        if self.hard:
            line += ("; TAIL_SOFT=hard sends every running request's CACHE_KEEP retained snapshots to the front "
                     "of the free queue still cached at each step, so no snapshot is held while its session runs")
        elif self.tail:
            line += (f"; TAIL_SOFT sends CACHE_KEEP's retained snapshots, newest-admitted request first, to the "
                     f"front of the free queue still cached whenever fewer than {self.tail} blocks besides "
                     f"unpinned resident pages are free, so they are reused before any resident page")
        if self.soft:
            line += ("; KEEP_SOFT sends those pins to the evictable cache, newest-admitted request "
                     "first, before any running request is preempted for blocks and whenever the "
                     "oldest waiting request may not fit"
                     + (f" or fewer than {self.floor} blocks are free" if self.floor else ""))
        say("armed", line)
        if self.wait_order:
            say("order-armed", f"WAIT_ORDER: the waiting queue is served fewest uncached prompt tokens "
                               f"first, each request's count falling by {self.wait_age:g} tokens a "
                               f"second it has waited, so a turn that resumes from the cache is "
                               f"admitted before a cold prompt and no prompt waits forever")
        if self.wait_keep:
            say("wait-armed", f"WAIT_KEEP: a waiting request's cached prefix goes to the back of the "
                              f"eviction queue when it arrives and every {self.wait_every:g} s while it "
                              f"waits, at most {WAIT_KEEP_PER_STEP} requests a step, so the blocks it "
                              f"will resume from are the last a new allocation takes")
        if self.ride and not mixed:
            say("ride-missed", "SLIM_STATE: PLE_RIDE was armed in the setup but every Mamba cache "
                               "group holds one kind of layer, so the short-conv state still has "
                               "a group of its own")

    def step(self, scheduler):
        if self.tail:
            self.yield_tails(scheduler)
        if self.soft:
            self.yield_pins(scheduler)
        if self.wait_keep:
            self.keep_waiting(scheduler)
        self.report(scheduler)

    def waiting(self, scheduler):
        for queue in (getattr(scheduler, "waiting", None), getattr(scheduler, "skipped_waiting", None)):
            if queue is not None:
                yield from queue

    def keep_waiting(self, scheduler):
        now = time.monotonic()
        manager = scheduler.kv_cache_manager
        present = set()
        budget = WAIT_KEEP_PER_STEP
        for request in self.waiting(scheduler):
            rid = request.request_id
            present.add(rid)
            if request.num_computed_tokens or budget <= 0:
                continue
            last = self.watched.get(rid)
            if last is not None and now - last < self.wait_every:
                continue
            if last is None:
                self.wait_tally.requests += 1
                self.arrived[rid] = now
            self.watched[rid] = now
            budget -= 1
            blocks, hit, _ = manager.get_computed_blocks(request)
            self.cached[rid] = hit
            moved = self.to_back(blocks.blocks)
            self.wait_tally.refreshes += 1
            self.wait_tally.blocks += moved
            self.wait_tally.tokens += hit
            if not hit:
                self.wait_tally.empty += 1
            elif moved:
                say("wait-first", f"WAIT_KEEP first refresh: request {rid} waits with {hit} of its "
                                  f"{request.num_tokens} tokens cached; {moved} block(s) moved to the "
                                  f"back of the eviction queue")
        for rid in [rid for rid in self.watched if rid not in present]:
            del self.watched[rid]
            self.cached.pop(rid, None)
            self.arrived.pop(rid, None)
        if self.wait_order:
            self.order_waiting(scheduler, now)

    def order_waiting(self, scheduler, now):
        queue = getattr(scheduler, "waiting", None)
        if queue is None or not hasattr(queue, "appendleft") or len(queue) < 2:
            return
        held = list(queue)

        def need(request):
            rid = request.request_id
            waited = now - self.arrived.get(rid, now)
            return request.num_tokens - self.cached.get(rid, 0) - self.wait_age * waited

        ordered = sorted(held, key=need)
        if [one.request_id for one in ordered] != [one.request_id for one in held]:
            queue.clear()
            queue.extend(ordered)
            self.reordered += 1

    def to_back(self, groups):
        state_groups = {m.kv_cache_group_id for m in self.mambas}
        ordered = [block for gid, blocks in enumerate(groups) if gid not in state_groups
                   for block in reversed(list(blocks))]
        ordered += [block for gid, blocks in enumerate(groups) if gid in state_groups
                    for block in reversed(list(blocks))]
        queue = self.pool.free_block_queue
        moved = 0
        for block in ordered:
            if block.is_null or block.ref_cnt != 0 or block.prev_free_block is None                     or block.next_free_block is None:
                continue
            queue.remove(block)
            queue.append(block)
            moved += 1
        return moved

    def waiting_need(self, scheduler):
        if len(scheduler.running) >= scheduler.max_num_running_reqs or not scheduler.waiting:
            return 0
        try:
            request = scheduler.waiting.peek_request()
        except (IndexError, AttributeError):
            return 0
        return self.per_request(request.num_tokens) + len(self.mambas)

    def yield_pins(self, scheduler):
        want = self.floor + self.waiting_need(scheduler)
        if self.pool.get_num_free_blocks() >= want:
            return
        self.tally.short += 1
        for request in reversed(scheduler.running):
            if self.pool.get_num_free_blocks() >= want:
                break
            got = self.unpin(request.request_id)
            if got:
                self.tally.released += 1
                self.tally.blocks += got

    def spare(self):
        return self.pool.get_num_free_blocks() - len(getattr(self.pool, "_reuse_watchers", None) or ())

    def yield_tails(self, scheduler):
        if self.hard:
            self.drop_tails(scheduler)
            return
        want = self.tail + self.waiting_need(scheduler)
        if self.spare() >= want:
            return
        self.tail_tally.short += 1
        for request in reversed(scheduler.running):
            if self.spare() >= want:
                break
            got, front = self.release(request.request_id, front=True)
            if got:
                self.tail_tally.released += 1
                self.tail_tally.blocks += got
                self.tail_tally.front += front

    def drop_tails(self, scheduler):
        first = self.mambas[0]
        for request in scheduler.running:
            rid = request.request_id
            if rid not in first._retained_tail_blocks and rid not in first._retained_state_block_idx:
                continue
            got, front = self.release(rid, front=True)
            if got:
                self.tail_tally.released += 1
                self.tail_tally.blocks += got
                self.tail_tally.front += front

    def rescue(self, scheduler):
        for request in reversed(scheduler.running):
            got = self.unpin(request.request_id)
            if got:
                self.tally.released += 1
                self.tally.blocks += got
                self.tally.rescued += 1
                return True
        return False

    def pinned(self, request_id):
        held = []
        for manager in self.mambas:
            index = manager._retained_state_block_idx.get(request_id)
            blocks = manager.req_to_blocks.get(request_id)
            if index is not None and blocks is not None and index < len(blocks) \
                    and not blocks[index].is_null:
                held.append(blocks[index])
            tail = manager._retained_tail_blocks.get(request_id)
            if tail is not None:
                held.append(tail)
        return held

    def unpin(self, request_id):
        return self.release(request_id)[0]

    def release(self, request_id, front=False):
        held = self.pinned(request_id)
        offered = {id(entry[2]) for manager in self.mambas
                   for entry in getattr(manager, "_pending_boundary_state_offloads", ())}
        if any(id(block) in offered for block in held):
            return 0, 0
        released = []
        for manager in self.mambas:
            index = manager._retained_state_block_idx.pop(request_id, None)
            blocks = manager.req_to_blocks.get(request_id)
            if index is not None and blocks is not None and index < len(blocks) \
                    and not blocks[index].is_null:
                released.append(blocks[index])
                blocks[index] = manager._null_block
            tail = manager._retained_tail_blocks.pop(request_id, None)
            if tail is not None:
                released.append(tail)
        if not released:
            return 0, 0
        if not front:
            self.pool.free_blocks(released)
            return len(released), 0
        ahead, elsewhere = [], []
        for block in released:
            if getattr(block, "pool", None) not in (None, self.pool):
                elsewhere.append(block)
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                ahead.append(block)
        if ahead:
            self.pool.free_block_queue.prepend_n(ahead)
        if elsewhere:
            self.pool.free_blocks(elsewhere)
        return len(released), len(ahead)

    def holdings(self, running):
        state = pinned = 0
        for manager in self.mambas:
            kept = getattr(manager, "_retained_state_block_idx", {})
            tails = getattr(manager, "_retained_tail_blocks", {})
            for request in running:
                rid = request.request_id
                blocks = manager.req_to_blocks.get(rid) or ()
                live = sum(1 for block in blocks if not block.is_null)
                index = kept.get(rid)
                held = int(index is not None and index < len(blocks) and not blocks[index].is_null)
                state += live - held
                pinned += held + int(rid in tails)

        def live(managers):
            return sum(1 for manager in managers for request in running
                       for block in manager.req_to_blocks.get(request.request_id) or ()
                       if not block.is_null)

        return {"running": len(running), "state": state, "pinned": pinned,
                "transient": max(0, state - len(self.mambas) * len(running)),
                "attention": live(self.attention), "other": live(self.other)}

    def report(self, scheduler):
        running = list(scheduler.running)
        now = time.monotonic()
        if not running or (self.reported is not None and now - self.reported < self.every):
            return
        held = self.holdings(running)
        unit = getattr(scheduler.kv_cache_manager, "taaf_unit", None)
        spares = sum(len(spare.blocks) for spare in (getattr(unit, "spares", None) or {}).values())
        count = held["running"]
        total = held["state"] + held["pinned"] + spares
        tally = self.tally
        since = "first look" if self.reported is None else f"last {now - self.reported:.0f} s"
        print(f"SLIM_STATE {since}: {count} running request(s) hold {total} Mamba state block(s) "
              f"in {len(self.mambas)} group(s), {total / count:.2f} a request: "
              f"{held['state'] / count:.2f} running ({held['transient']} transient at a block "
              f"crossing), {held['pinned'] / count:.2f} pinned, {spares / count:.2f} reply spare; "
              f"plus {held['attention'] / count:.2f} attention and {held['other'] / count:.2f} other "
              f"block(s) a request; {self.pool.get_num_free_blocks()} of "
              f"{self.pool.num_gpu_blocks - 1} blocks free"
              + (f"; KEEP_SOFT released the pins of {tally.released} request(s), {tally.blocks} "
                 f"block(s): {tally.rescued} in place of a preemption, the rest on {tally.short} "
                 f"step(s) a waiting request may not have fit" if self.soft else "")
              + (f"; TAIL_SOFT released the retained snapshots of {self.tail_tally.released} request(s), "
                 f"{self.tail_tally.blocks} block(s), {self.tail_tally.front} at the front of the free queue, on "
                 + ("every step (hard)" if self.hard else f"{self.tail_tally.short} step(s) with fewer than {self.tail} spare blocks")
                 + f"; {self.spare()} spare"
                 if self.tail else ""),
              flush=True)
        for field in dataclasses.fields(TailTally):
            setattr(self.tail_total, field.name,
                    getattr(self.tail_total, field.name) + getattr(self.tail_tally, field.name))
        self.tail_tally = TailTally()
        for field in dataclasses.fields(Tally):
            setattr(self.total, field.name,
                    getattr(self.total, field.name) + getattr(tally, field.name))
        self.tally = Tally()
        if self.wait_keep:
            waited = self.wait_tally
            print(f"WAIT_KEEP {since}: {waited.requests} new waiting request(s), {waited.refreshes} "
                  f"refresh(es), {waited.empty} with nothing cached; {waited.blocks} cached block(s) "
                  f"moved to the back of the eviction queue, {waited.tokens} cached prompt tokens "
                  f"seen at refresh; {len(self.watched)} waiting now"
                  + (f"; WAIT_ORDER reordered the queue on {self.reordered} step(s)"
                     if self.wait_order else ""), flush=True)
            self.reordered = 0
            for field in dataclasses.fields(WaitTally):
                setattr(self.wait_total, field.name,
                        getattr(self.wait_total, field.name) + getattr(waited, field.name))
            self.wait_tally = WaitTally()
        self.reported = now
