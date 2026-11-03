import os

ENV = "TAAF_VLLM_ALIGN_TAIL"
REPORT_EVERY = 2000
_TALLY = {"splits": 0, "spared": 0}


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def refusal(scheduler):
    coordinator = scheduler.kv_cache_manager.coordinator
    retention = getattr(coordinator, "retention_interval", None)
    if retention != 0:
        return (f"prefix_cache_retention_interval is {retention!r}, so every full Mamba block is registered and "
                f"each needs its own boundary state")
    if scheduler.mamba_has_prefill_checkpoint_blocks:
        return "the Mamba groups export internal prefill checkpoints, which the stock split already places"
    if not callable(getattr(coordinator, "get_replay_boundaries", None)):
        return "the KV cache coordinator has no get_replay_boundaries"
    units = {manager.cache_hit_alignment_tokens for manager in coordinator.single_type_managers
             if type(manager).__name__ == "MambaManager"}
    if len(units) != 1:
        return f"the Mamba managers disagree on the cache-hit alignment ({sorted(units)})"
    return None


def loose(scheduler):
    held = getattr(scheduler, "_taaf_align_tail", None)
    if held is None:
        reason = refusal(scheduler)
        held = scheduler._taaf_align_tail = reason is None
        block = scheduler.cache_config.block_size
        if held:
            print(f"ALIGN_TAIL live: a prefill chunk stops only where a Mamba state is registered (replay "
                  f"boundaries, junction, prompt tail, last cacheable block) and otherwise fills the token budget "
                  f"across {block}-token block boundaries", flush=True)
        else:
            print(f"ALIGN_TAIL stands aside, the stock block-aligned split runs: {reason}", flush=True)
    return held


def unit(scheduler):
    return next(manager.cache_hit_alignment_tokens
                for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
                if type(manager).__name__ == "MambaManager")


def kept_ends(marks, alignment, block_size):
    return tuple(mark // alignment * alignment // block_size * block_size for mark in marks)


def stops(scheduler, request, block_size, start, end):
    marks = list(scheduler.kv_cache_manager.coordinator.get_replay_boundaries(request))
    if request.shared_prefix_boundary:
        marks.append(request.shared_prefix_boundary)
    found = kept_ends(marks, unit(scheduler), block_size)
    _TALLY["splits"] += 1
    if (start // block_size + 1) * block_size < end:
        _TALLY["spared"] += 1
    if _TALLY["splits"] % REPORT_EVERY == 1:
        print(f"ALIGN_TAIL tally: prefill splits={_TALLY['splits']} offered past the next block "
              f"boundary={_TALLY['spared']}", flush=True)
    return found
