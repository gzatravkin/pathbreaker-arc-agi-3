import os
import sys
import time
from collections import OrderedDict

ENV = "TAAF_VLLM_VISION_CACHE"
EVERY_S = 60.0
MIB = 1 << 20
SKIP_PREFIX = "tmp_"
AHEAD_MODULE = "vllm.taaf_vision_ahead"
_STATE = {"cache": None, "budget": None}


def budget_bytes():
    raw = str(os.environ.get(ENV) or "").strip()
    if not raw:
        return 0
    try:
        mib = int(raw)
    except ValueError:
        return 0
    return mib * MIB if mib > 0 else 0


def say(line):
    print(line, flush=True)


def _nbytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def _own(tensor):
    try:
        storage = int(tensor.untyped_storage().nbytes())
    except Exception:
        return tensor
    return tensor.clone() if storage > _nbytes(tensor) else tensor


class Cache:
    def __init__(self, budget, out=say, clock=time.monotonic, every_s=EVERY_S):
        self.budget = int(budget)
        self.out = out
        self.clock = clock
        self.every_s = every_s
        self.held = OrderedDict()
        self.bytes = 0
        self.window = 0
        self.last = clock()
        self.totals = {"hits": 0, "misses": 0, "stored": 0, "evicted": 0, "too_big": 0, "hit_bytes": 0}
        self.reset_window()

    def reset_window(self):
        self.counts = {"hits": 0, "misses": 0, "stored": 0, "evicted": 0, "too_big": 0, "hit_bytes": 0}

    def count(self, name, value=1):
        self.counts[name] += value
        self.totals[name] += value

    def store(self, key, tensor):
        if key in self.held:
            self.bytes -= _nbytes(self.held.pop(key))
        tensor = _own(tensor)
        size = _nbytes(tensor)
        if size > self.budget:
            self.count("too_big")
            return False
        while self.held and self.bytes + size > self.budget:
            _, old = self.held.popitem(last=False)
            self.bytes -= _nbytes(old)
            self.count("evicted")
        self.held[key] = tensor
        self.bytes += size
        self.count("stored")
        return True

    def take(self, key):
        tensor = self.held.pop(key, None)
        if tensor is None:
            self.count("misses")
            return None
        size = _nbytes(tensor)
        self.bytes -= size
        self.count("hits")
        self.count("hit_bytes", size)
        return tensor

    def clear(self):
        self.held.clear()
        self.bytes = 0

    def line(self):
        lookups = self.counts["hits"] + self.counts["misses"]
        total = self.totals["hits"] + self.totals["misses"]
        return (f"vision_cache window={self.window} hits={self.counts['hits']} misses={self.counts['misses']} "
                f"hit_share={self.counts['hits'] / max(lookups, 1):.4f} stored={self.counts['stored']} "
                f"evicted={self.counts['evicted']} too_big={self.counts['too_big']} "
                f"hit_mib={self.counts['hit_bytes'] / MIB:.1f} entries={len(self.held)} "
                f"held_mib={self.bytes / MIB:.1f} budget_mib={self.budget / MIB:.0f} "
                f"run_hits={self.totals['hits']} run_hit_share={self.totals['hits'] / max(total, 1):.4f}")

    def tick(self):
        now = self.clock()
        if now - self.last >= self.every_s:
            self.out(self.line())
            self.window += 1
            self.last = now
            self.reset_window()


def cache():
    budget = budget_bytes()
    held = _STATE["cache"]
    if budget <= 0:
        return None
    if held is None or _STATE["budget"] != budget:
        held = _STATE["cache"] = Cache(budget)
        _STATE["budget"] = budget
        say(f"VISION_CACHE armed: encoder outputs the scheduler frees are kept on the GPU by mm hash, up to "
            f"{budget / MIB:.0f} MiB, least recently stored first out; an image scheduled again takes its "
            f"cached embedding instead of running the encoder; one vision_cache line a minute")
    return held


def release(encoder_cache, mm_hash):
    tensor = encoder_cache.encoder_outputs.pop(mm_hash, None)
    ahead = sys.modules.get(AHEAD_MODULE)
    if ahead is not None and tensor is not None:
        ahead.released(tensor)
    held = cache()
    if held is None or tensor is None or str(mm_hash).startswith(SKIP_PREFIX):
        return
    held.store(mm_hash, tensor)
    held.tick()


def restore(encoder_cache, mm_hash):
    held = cache()
    if held is None:
        return False
    tensor = held.take(mm_hash)
    held.tick()
    if tensor is None:
        return False
    encoder_cache.encoder_outputs[mm_hash] = tensor
    return True


def clear():
    held = _STATE["cache"]
    if held is not None:
        held.clear()
