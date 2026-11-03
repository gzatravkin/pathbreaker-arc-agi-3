import os
import sys
import time

ENV = "TAAF_VLLM_PREFILL_HOLD"
TOKENS_ENV = "TAAF_VLLM_PREFILL_HOLD_TOKENS"
MS_ENV = "TAAF_VLLM_PREFILL_HOLD_MS"
TOKENS = 1024
MS = 300.0
REPORT_EVERY = 1000
SCHEDULABLE = ("WAITING", "PREEMPTED")
_STATE = {"limits": None, "first": {}, "said": False, "error_said": False,
          "tally": {"steps": 0, "held": 0, "held_tokens": 0, "released_tokens": 0, "released_wait": 0,
                    "released_idle": 0, "errors": 0}}


def say(message):
    print(f"PREFILL_HOLD {message}", file=sys.stderr, flush=True)


def limits():
    if _STATE["limits"] is None:
        tokens = int(os.environ.get(TOKENS_ENV) or TOKENS)
        ms = float(os.environ.get(MS_ENV) or MS)
        _STATE["limits"] = (tokens, ms / 1000.0)
    return _STATE["limits"]


def _status(request):
    status = getattr(request, "status", None)
    return getattr(status, "name", str(status))


def pending(scheduler):
    found = []
    for queue in (getattr(scheduler, "waiting", None) or (), getattr(scheduler, "skipped_waiting", None) or ()):
        for request in queue:
            if _status(request) not in SCHEDULABLE:
                continue
            left = int(request.num_tokens) - int(request.num_computed_tokens)
            if left > 0:
                found.append((request.request_id, left))
    return found


def decide(found, decoding, now, first, tokens, wait):
    live = {rid for rid, _ in found}
    for rid in [rid for rid in first if rid not in live]:
        first.pop(rid)
    for rid, _ in found:
        first.setdefault(rid, now)
    if not found:
        return False, None
    if not decoding:
        return False, "released_idle"
    if sum(left for _, left in found) >= tokens:
        return False, "released_tokens"
    if now - min(first[rid] for rid, _ in found) >= wait:
        return False, "released_wait"
    return True, "held"


def report():
    tokens, wait = limits()
    tally = _STATE["tally"]
    say(f"tally {tally}; a step without {tokens} prefill tokens waiting and no request older than "
        f"{wait * 1000:.0f} ms runs decode only")


def hold(scheduler, scheduled_running):
    if _AUTO["on"]:
        return _AUTO["held"]
    tally = _STATE["tally"]
    tally["steps"] += 1
    try:
        tokens, wait = limits()
        found = pending(scheduler)
        held, cause = decide(found, bool(scheduled_running), time.monotonic(), _STATE["first"], tokens, wait)
    except Exception as failure:
        tally["errors"] += 1
        if not _STATE["error_said"]:
            _STATE["error_said"] = True
            say(f"stood aside for this step: {type(failure).__name__}: {failure}; the waiting queue is scheduled "
                f"as in stock")
        return False
    if cause is not None:
        tally[cause] += 1
    if held:
        tally["held_tokens"] += sum(left for _, left in found)
        if not _STATE["said"]:
            _STATE["said"] = True
            say(f"first hold: {len(found)} request(s) with {sum(left for _, left in found)} prefill tokens wait "
                f"beside {len(scheduled_running)} running; released at {tokens} tokens or {wait * 1000:.0f} ms")
    if tally["steps"] % REPORT_EVERY == 0:
        report()
    return held


AUTO_ENV = "TAAF_VLLM_PREFILL_HOLD_AUTO"
CAP_ENV = "TAAF_VLLM_PREFILL_HOLD_CAP_MS"
FIXED_ENV = "TAAF_VLLM_PREFILL_HOLD_F_MS"
PER_TOKEN_ENV = "TAAF_VLLM_PREFILL_HOLD_P_US"
ALIGNED_ENV = "TAAF_VLLM_PREFILL_HOLD_ALIGNED"
FILL_SHARE_ENV = "TAAF_VLLM_PREFILL_HOLD_FILL_SHARE"
WAIT_WEIGHT_ENV = "TAAF_VLLM_PREFILL_HOLD_WAIT_WEIGHT"
CHUNK_RELEASE_ENV = "TAAF_VLLM_PREFILL_HOLD_CHUNK_RELEASE"
CAP_MS = 10000.0
FIXED_MS = 38.0
PER_TOKEN_US = 27.0
FILL_SHARE = 0.9
WAIT_WEIGHT = 1.0
LANE_BUCKET = 8
DECODE_EMA = 0.05
FIT_DECAY = 0.995
FIT_MIN = 20
FIT_SPREAD = 200.0
TALLY_EVERY_S = 60.0
CLOCK_MARK = "_taaf_prefill_hold_clock"
_AUTO = {"on": False, "held": False, "limits": None, "shape": None, "first": {}, "steps": [], "done_at": None,
         "decode": {}, "fit": [0.0, 0.0, 0.0, 0.0, 0.0], "began": None, "said_at": None, "said": False,
         "minute": None, "good": None, "kept": 0}


def auto_limits():
    if _AUTO["limits"] is None:
        _AUTO["limits"] = (float(os.environ.get(CAP_ENV) or CAP_MS) / 1000.0,
                           float(os.environ.get(FIXED_ENV) or FIXED_MS),
                           float(os.environ.get(PER_TOKEN_ENV) or PER_TOKEN_US) / 1000.0)
    return _AUTO["limits"]


def chunk_release():
    return str(os.environ.get(CHUNK_RELEASE_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def shape():
    if _AUTO["shape"] is None:
        aligned = str(os.environ.get(ALIGNED_ENV) or "").strip().lower() in ("1", "on", "true", "yes")
        share = min(1.0, max(0.1, float(os.environ.get(FILL_SHARE_ENV) or FILL_SHARE)))
        weight = max(0.01, float(os.environ.get(WAIT_WEIGHT_ENV) or WAIT_WEIGHT))
        _AUTO["shape"] = (aligned, share, weight)
    return _AUTO["shape"]


def fresh_minute():
    return {"steps": 0, "mixed": 0, "prefill": 0, "held": 0, "fill": 0, "cost": 0, "cap": 0, "idle": 0,
            "chunk": 0, "waits": [], "lanes": 0, "errors": 0}


def new_tokens(request):
    return max(int(request.num_tokens_with_spec) + int(request.num_output_placeholders)
               - int(request.num_computed_tokens), 0)


def step_cost():
    _, prior_fixed, prior_per_token = auto_limits()
    weight, sx, sy, sxx, sxy = _AUTO["fit"]
    if weight < FIT_MIN:
        return prior_fixed, prior_per_token, "prior"
    mean_x, mean_y = sx / weight, sy / weight
    spread = sxx / weight - mean_x * mean_x
    if spread < FIT_SPREAD * FIT_SPREAD:
        per_token, source = prior_per_token, "mean"
    else:
        per_token, source = max((sxy / weight - mean_x * mean_y) / spread, 0.0), "fit"
    fixed = mean_y - per_token * mean_x
    if fixed > 0.0:
        _AUTO["good"] = fixed
        if fixed >= prior_fixed:
            return fixed, per_token, source
        return prior_fixed, per_token, f"{source} gave {fixed:.1f} ms, floor {prior_fixed:.1f}"
    good = _AUTO["good"]
    held = prior_fixed if good is None else max(good, prior_fixed)
    _AUTO["kept"] += 1
    return held, per_token, f"{source} gave {fixed:.1f} ms, F held at {held:.1f}"


def learn(lanes, prefill, ms):
    bucket = lanes // LANE_BUCKET
    decode = _AUTO["decode"]
    if prefill <= 0:
        held = decode.get(bucket)
        decode[bucket] = ms if held is None else held + DECODE_EMA * (ms - held)
        return
    if bucket not in decode:
        return
    extra = ms - decode[bucket]
    fit = _AUTO["fit"]
    for index in range(5):
        fit[index] *= FIT_DECAY
    fit[0] += 1.0
    fit[1] += prefill
    fit[2] += extra
    fit[3] += prefill * prefill
    fit[4] += prefill * extra


def choose(pending, decoding, decode_tokens, budget, now, first, fixed_ms, cap_s, take=None, chunk=False,
           share=FILL_SHARE, weight=WAIT_WEIGHT):
    live = {rid for rid, _ in pending}
    for rid in [rid for rid in first if rid not in live]:
        first.pop(rid)
    for rid, _ in pending:
        first.setdefault(rid, now)
    if not pending:
        return False, None
    if not decoding:
        return False, "idle"
    if take is None:
        if sum(left for _, left in pending) >= budget - decode_tokens:
            return False, "fill"
    else:
        if chunk and chunk_release():
            return False, "chunk"
        if take >= share * (budget - decode_tokens):
            return False, "fill"
    waits = [now - first[rid] for rid, _ in pending]
    if max(waits) >= cap_s:
        return False, "cap"
    if weight * sum(waits) * 1000.0 >= decoding * fixed_ms:
        return False, "cost"
    return True, "held"


def aligned_take(scheduler, room):
    block = int(getattr(getattr(scheduler, "cache_config", None), "block_size", 0) or 0)
    aligned = bool(getattr(scheduler, "need_mamba_block_aligned_split", False)) and block > 0
    order = [(request, True) for request in getattr(scheduler, "running", None) or () if request.is_prefill_chunk]
    for queue in (getattr(scheduler, "waiting", None) or (), getattr(scheduler, "skipped_waiting", None) or ()):
        order += [(request, False) for request in queue if _status(request) in SCHEDULABLE]
    take, chunk = 0, False
    for request, running in order:
        if room <= 0:
            break
        left = new_tokens(request) if running else int(request.num_tokens) - int(request.num_computed_tokens)
        if left <= 0:
            continue
        chunk = chunk or running
        if left <= room:
            piece = left
        elif aligned:
            start = int(request.num_computed_tokens)
            piece = max((start + room) // block * block - start, 0)
        else:
            piece = room
        take += piece
        room -= piece
    return take, chunk


def plan(scheduler, token_budget):
    _AUTO["on"] = True
    _AUTO["held"] = False
    now = time.monotonic()
    if _AUTO["began"] is None:
        _AUTO["began"] = _AUTO["said_at"] = now
        _AUTO["minute"] = fresh_minute()
        arm_clock(scheduler)
    minute = _AUTO["minute"]
    try:
        cap_s, _, _ = auto_limits()
        fixed_ms, _, _ = step_cost()
        pending = pending_prefill(scheduler)
        decoding, decode_tokens = 0, 0
        for request in getattr(scheduler, "running", None) or ():
            if not request.is_prefill_chunk:
                decoding += 1
                decode_tokens += new_tokens(request)
        aligned, share, weight = shape()
        take, chunk = (aligned_take(scheduler, int(token_budget) - decode_tokens) if aligned and pending
                       else (None, False))
        held, cause = choose(pending, decoding, decode_tokens, int(token_budget), now, _AUTO["first"], fixed_ms,
                             cap_s, take, chunk, share, weight)
    except Exception as failure:
        minute["errors"] += 1
        if not _STATE["error_said"]:
            _STATE["error_said"] = True
            say(f"auto stood aside for this step: {type(failure).__name__}: {failure}; prefill is scheduled as in "
                f"stock")
        return False
    if cause == "held":
        minute["held"] += 1
        if not _AUTO["said"]:
            _AUTO["said"] = True
            say(f"auto first hold: {len(pending)} session(s) with {sum(left for _, left in pending)} prefill tokens "
                f"wait beside {decoding} decoding lanes ({decode_tokens} decode tokens of a {token_budget}-token "
                f"step); released when the step fills, when the summed wait reaches lanes x {fixed_ms:.1f} ms, or "
                f"after {cap_s:.1f} s"
                + (f"; fill counts the {take} tokens the block-aligned split would take now, a share of "
                   f"{share:.2f} fills, a running chunk is {'never held' if chunk_release() else 'held like the rest'}, the wait weighs {weight:.2f}"
                   if take is not None else ""))
    elif cause is not None:
        minute[cause] += 1
        minute["waits"].extend(now - _AUTO["first"].pop(rid, now) for rid, _ in pending)
    _AUTO["held"] = held
    if now - _AUTO["said_at"] >= TALLY_EVERY_S:
        tally_line(now, decoding)
    return held


def pending_prefill(scheduler):
    found = pending(scheduler)
    for request in getattr(scheduler, "running", None) or ():
        if request.is_prefill_chunk:
            left = new_tokens(request)
            if left > 0:
                found.append((request.request_id, left))
    return found


def scheduled(scheduler, num_scheduled_tokens, new_reqs, resumed_reqs, total):
    if not _AUTO["on"] or not total:
        return None
    try:
        fresh = {request.request_id for request in [*new_reqs, *resumed_reqs]}
        prefill, lanes = 0, 0
        for rid, count in num_scheduled_tokens.items():
            request = scheduler.requests.get(rid)
            if rid in fresh or (request is not None and request.is_prefill_chunk):
                prefill += int(count)
            else:
                lanes += 1
        _AUTO["steps"].append((time.monotonic(), lanes, prefill, int(total)))
        del _AUTO["steps"][:-8]
        minute = _AUTO["minute"]
        minute["steps"] += 1
        minute["lanes"] += lanes
        if prefill:
            minute["mixed"] += 1
            minute["prefill"] += prefill
    except Exception:
        _AUTO["minute"]["errors"] += 1
    return None


def finished(scheduler_output, now=None):
    now = time.monotonic() if now is None else now
    total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    if not total:
        return None
    steps = _AUTO["steps"]
    while steps and steps[0][3] != total:
        steps.pop(0)
    if not steps:
        return None
    started, lanes, prefill, _ = steps.pop(0)
    begin = started if _AUTO["done_at"] is None else max(started, _AUTO["done_at"])
    _AUTO["done_at"] = now
    ms = (now - begin) * 1000.0
    learn(lanes, prefill, ms)
    return ms


def arm_clock(scheduler):
    owner = type(scheduler)
    if getattr(owner, CLOCK_MARK, False) or not hasattr(owner, "update_from_output"):
        return False
    base = owner.update_from_output

    def clocked(self, scheduler_output, *args, **kwargs):
        try:
            finished(scheduler_output)
        except Exception:
            _AUTO["minute"]["errors"] += 1
        return base(self, scheduler_output, *args, **kwargs)

    owner.update_from_output = clocked
    setattr(owner, CLOCK_MARK, True)
    return True


def tally_line(now, decoding):
    minute = _AUTO["minute"]
    fixed_ms, per_token, source = step_cost()
    waits = sorted(minute["waits"])
    mean_wait = 1000.0 * sum(waits) / len(waits) if waits else 0.0
    p90 = 1000.0 * waits[min(int(0.9 * len(waits)), len(waits) - 1)] if waits else 0.0
    say(f"auto [{(now - _AUTO['began']) / 60.0:.1f} min]: {minute['steps']} steps, {minute['mixed']} mixed with "
        f"{minute['prefill'] / max(minute['mixed'], 1):.0f} prefill tokens each, mean {minute['lanes'] / max(minute['steps'], 1):.0f} "
        f"decoding lanes; {minute['held']} steps held; released by fill {minute['fill']}, cost {minute['cost']}, "
        f"cap {minute['cap']}, idle {minute['idle']}; wait of a released session mean {mean_wait:.0f} ms, p90 "
        f"{p90:.0f} ms; chunk releases {minute['chunk']}; F {fixed_ms:.1f} ms a mixed step ({source}), "
        f"{per_token * 1000.0:.1f} us a prefill token, "
        f"L x F {decoding * fixed_ms / 1000.0:.2f} lane-s at {decoding} lanes; errors {minute['errors']}; "
        f"F held {_AUTO['kept']} times since the start")
    _AUTO["minute"] = fresh_minute()
    _AUTO["said_at"] = now
