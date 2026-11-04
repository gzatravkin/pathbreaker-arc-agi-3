import importlib
import sys
import os
import time

import numpy as np

MIXED_FULL_ENV = "TAAF_VLLM_MIXED_FULL"
DUMMY_PREFILL_SEQS = 2
HELPER_MODULE = "vllm.taaf_mixed_full"
PICK_MODULE = "vllm.taaf_qsa_hisparse"
_SAID = set()
_STATE = {"settled": 0, "vetoed": {}, "fell_back": 0, "replayed": 0, "captured": 0, "decode_full": 0, "last": 0.0}
REPORT_S = 15.0


def armed():
    return str(os.environ.get(MIXED_FULL_ENV) or "").strip().lower() == "1"


def report():
    now = time.monotonic()
    if now - _STATE["last"] < REPORT_S:
        return
    _STATE["last"] = now
    print(f"MIXED_FULL replay settled={_STATE['settled']} replayed={_STATE['replayed']} "
          f"fell_back={_STATE['fell_back']} decode_full={_STATE['decode_full']} vetoed={dict(_STATE['vetoed'])}",
          flush=True)


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def owns(manager):
    return any(held.__name__ == "ModelCudaGraphManager" for held in type(manager).__mro__)


def planner():
    return importlib.import_module(HELPER_MODULE)


def dummy_lens(bucket, qlen, prefill_seqs=DUMMY_PREFILL_SEQS):
    tokens = int(bucket.tokens)
    qlen = int(qlen)
    prefill = max(1, min(int(prefill_seqs), int(bucket.seqs), int(bucket.reqs) - 1))
    floor = qlen + 1
    decodes = max(0, min(int(bucket.reqs) - prefill, (tokens // 2) // qlen))
    rest = tokens - decodes * qlen
    head = [floor] * (prefill - 1)
    tail = rest - floor * (prefill - 1)
    if tail < floor:
        raise ValueError(f"a {tokens}-token bucket cannot hold {prefill} prefill sequences beside {decodes} decodes")
    return [qlen] * decodes + head + [tail]


def mixed_descriptors(manager, num_tokens, num_active_loras, descriptor, full):
    if not armed() or num_active_loras or not owns(manager):
        return []
    plan = planner().ensure_plan(manager.decode_query_len)
    if num_tokens not in plan.sizes:
        return []
    bucket = plan.bucket_for(num_tokens)
    if bucket is None or bucket.tokens != num_tokens or bucket.reqs < 2:
        return []
    return [descriptor(cg_mode=full, num_tokens=num_tokens, num_reqs=bucket.reqs, num_active_loras=0)]


def is_mixed(desc, full):
    return (desc.cg_mode == full and desc.uniform_token_count is None and desc.max_query_len is None
            and desc.num_ubatches == 1)


def refuse(manager, desc, full):
    if getattr(manager, "_taaf_no_full", False) and desc.cg_mode == full:
        return True
    return bool(getattr(manager, "_taaf_no_mixed_full", False)) and is_mixed(desc, full)


def pick_streak(runner, batch_state):
    step = getattr(sys.modules.get(PICK_MODULE), "pick_engine_step", None)
    return bool(step is not None and step(runner, batch_state))


def off_graph(runner, batch_state, desc):
    from vllm.config import CUDAGraphMode

    manager = runner.cudagraph_manager
    if manager is None or desc.cg_mode != CUDAGraphMode.FULL:
        return desc
    lens = [int(x) for x in batch_state.num_scheduled_tokens]
    manager._taaf_no_full = True
    try:
        held = manager.dispatch(len(lens), int(batch_state.num_tokens), None, desc.num_active_loras,
                                max_query_len=max(lens))
    finally:
        manager._taaf_no_full = False
    _STATE["vetoed"]["pick_log"] = _STATE["vetoed"].get("pick_log", 0) + 1
    report()
    say("pick_log", f"MIXED_FULL pick_log: a pick-log streak step of {len(lens)} requests and {sum(lens)} tokens "
                    f"leaves its {desc.num_tokens}-token full graph for the {held.cg_mode.name.lower()} path, so the "
                    "QSA layers run their recording hooks; only streak steps leave the full graphs")
    return held


def dummy_batch(input_batch_cls, num_reqs, num_tokens, input_buffers, max_query_len, mixed, qlen):
    batch = input_batch_cls.make_dummy(num_reqs, num_tokens, input_buffers, max_query_len=max_query_len)
    if not mixed and not armed():
        return batch
    helper = planner()
    helper.set_graph_step(mixed)
    if not mixed:
        return batch
    plan = helper.ensure_plan(qlen)
    bucket = plan.bucket_for(num_tokens)
    lens = dummy_lens(bucket, qlen)
    if len(lens) > num_reqs:
        raise ValueError(f"the mixed dummy of the {num_tokens}-token bucket needs {len(lens)} requests, "
                         f"the descriptor pads to {num_reqs}")
    checked, why = plan.fit(lens, num_tokens)
    if checked is None or checked.tokens != num_tokens:
        raise ValueError(f"the mixed dummy of the {num_tokens}-token bucket does not fit it: {why}")
    apply_lens(batch, input_buffers, lens, num_reqs, num_tokens)
    say("dummy", f"MIXED_FULL capture: the {num_tokens}-token bucket is captured on a mixed dummy of "
                 f"{plan.split(lens)} decode requests of {qlen} tokens and {len(lens) - plan.split(lens)} prefill "
                 f"sequences, padded to {num_reqs} requests; every mixed bucket records the prefill path")
    return batch


def apply_lens(batch, input_buffers, lens, padded_reqs, num_tokens):
    import torch

    real = len(lens)
    sched = np.zeros(padded_reqs, dtype=np.int32)
    sched[:real] = np.asarray(lens, dtype=np.int32)
    qsl = np.empty(padded_reqs + 1, dtype=np.int32)
    qsl[0] = 0
    np.cumsum(sched, out=qsl[1:])
    seq = input_buffers.seq_lens
    seq[:padded_reqs] = torch.from_numpy(sched).to(seq.device)
    seq[padded_reqs:] = 0
    start = input_buffers.query_start_loc
    start[:padded_reqs + 1] = torch.from_numpy(qsl).to(start.device)
    start[padded_reqs + 1:] = num_tokens
    batch.num_scheduled_tokens[:] = sched[:batch.num_scheduled_tokens.shape[0]]
    batch.query_start_loc_np[:] = qsl[:batch.query_start_loc_np.shape[0]]
    batch.seq_lens_cpu_upper_bound.copy_(torch.from_numpy(sched.copy())[:batch.seq_lens_cpu_upper_bound.shape[0]])
    batch.logits_indices.copy_((batch.query_start_loc[1:] - 1).clamp(min=0))
    return batch


FORK_MARK = "taaf_fork_of"


def fork_admitted(scheduler_output):
    for one in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        extra = getattr(getattr(one, "sampling_params", None), "extra_args", None) or {}
        if extra.get(FORK_MARK):
            return True
    return False


def step_rows(batch_state, scheduler_output):
    drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    return [len(drafts.get(req_id, ())) for req_id in batch_state.req_ids], batch_state.is_prefilling_np


def real_step(batch_state, scheduler_output):
    if scheduler_output is None or not len(getattr(batch_state, "req_ids", None) or ()):
        return False
    return int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0


def settle(runner, batch_state, desc, scheduler_output=None):
    if not armed():
        return desc
    helper = planner()
    helper.set_graph_step(False)
    if batch_state is None:
        return desc
    if pick_streak(runner, batch_state):
        return off_graph(runner, batch_state, desc)
    from vllm.config import CUDAGraphMode

    manager = runner.cudagraph_manager
    if manager is None or not is_mixed(desc, CUDAGraphMode.FULL):
        return desc
    lens = [int(x) for x in batch_state.num_scheduled_tokens]
    computed = [int(x) for x in runner.req_states.num_computed_tokens_np[batch_state.idx_mapping_np]]
    if fork_admitted(scheduler_output):
        bucket, why = None, "fork_branch_admitted"
    else:
        bucket, why = helper.admit(lens, desc.num_tokens, computed, step_rows(batch_state, scheduler_output))
    _STATE["settled"] += 1
    if bucket is not None:
        _STATE["replayed"] += 1
        report()
        helper.set_graph_step(True, desc.num_tokens if real_step(batch_state, scheduler_output) else 0)
        return desc
    if why == "decode_only":
        held = decode_graph(manager, lens, desc.num_active_loras, CUDAGraphMode.FULL)
        if held is not None:
            _STATE["decode_full"] += 1
            report()
            say("decode-full", f"MIXED_FULL decode_full: a decode step of {len(lens)} requests and {sum(lens)} tokens "
                               f"under the {desc.num_tokens}-token mixed graph replays the {held.num_tokens}-token "
                               f"full decode graph ({held.num_reqs} requests), not the piecewise graph")
            return held
    _STATE["fell_back"] += 1
    _STATE["vetoed"][why] = _STATE["vetoed"].get(why, 0) + 1
    report()
    say(f"veto-{why.split(':')[0]}", f"MIXED_FULL veto: a step of {len(lens)} requests and {sum(lens)} tokens does "
                                    f"not replay the {desc.num_tokens}-token full graph ({why}); it keeps its "
                                    "piecewise graph")
    manager._taaf_no_mixed_full = True
    try:
        return manager.dispatch(len(lens), int(batch_state.num_tokens), None, desc.num_active_loras,
                                max_query_len=max(lens))
    finally:
        manager._taaf_no_mixed_full = False


def decode_graph(manager, lens, num_active_loras, full):
    qlen = int(manager.decode_query_len)
    if num_active_loras or not lens or any(int(x) != qlen for x in lens):
        return None
    reqs, tokens = len(lens), sum(int(x) for x in lens)
    captured = getattr(manager, "graphs", None)
    dropped = manager.__dict__.get("_taaf_dropped", ())
    held = [d for d in manager._capture_descs.get(full, ())
            if d.cg_mode == full and d.uniform_token_count == qlen and d.num_ubatches == 1
            and not d.num_active_loras and d.num_reqs is not None and d.num_reqs >= reqs and d.num_tokens >= tokens
            and d not in dropped and (captured is None or d in captured)]
    return min(held, key=lambda d: d.num_tokens, default=None)


WITNESS_MODULES = ("vllm.taaf_qsa_graph", "vllm.taaf_gdn_mixed_full")


def witnesses():
    found = {}
    for name in WITNESS_MODULES:
        held = sys.modules.get(name)
        read = getattr(held, "witness", None)
        if read is not None:
            found.update(read())
    return found


def reasons():
    found = {}
    for name in WITNESS_MODULES:
        held = sys.modules.get(name)
        read = getattr(held, "reasons", None)
        if read is not None:
            found.update(read())
    return found


def refusal(before, after, why=None):
    stood = [name for name, (attempts, built) in after.items()
             if attempts - before.get(name, (0, 0))[0] != built - before.get(name, (0, 0))[1]]
    if not stood:
        return ""
    why = reasons() if why is None else why
    return ("a backend stood aside from the bucketed dummy: "
            + ", ".join(f"{name} ({why.get(name) or 'no reason recorded'})" for name in stood))


def probe(desc, create_forward_fn):
    before = witnesses()
    try:
        create_forward_fn(desc, warmup=True)
    except Exception as failure:
        return f"{type(failure).__name__}: {str(failure)[:200]}"
    return refusal(before, witnesses())


def drop(manager, victims, full):
    gone = set(victims)
    held = manager._capture_descs.get(full)
    if held is not None:
        held[:] = [desc for desc in held if desc not in gone]
    for listed in manager._candidates.values():
        listed[:] = [desc for desc in listed if desc not in gone]
    manager.__dict__.setdefault("_taaf_dropped", set()).update(gone)


def screen(manager, descs, create_forward_fn, full):
    if not armed():
        return descs
    cleared = manager.__dict__.setdefault("_taaf_cleared", set())
    victims = {}
    for desc in list(descs):
        if not is_mixed(desc, full) or desc in cleared:
            continue
        why = probe(desc, create_forward_fn)
        if why:
            victims[desc] = why
        else:
            cleared.add(desc)
    if victims:
        drop(manager, victims, full)
        reasons = {}
        for why in victims.values():
            reasons[why] = reasons.get(why, 0) + 1
        for why, count in reasons.items():
            say(f"screen-{why}", f"MIXED_FULL screen: {count} of {count + len(cleared)} mixed bucket graphs "
                                      f"are not captured as FULL and keep their piecewise graph ({why})")
        _STATE["screened_out"] = _STATE.get("screened_out", 0) + len(victims)
    else:
        say("screen-clear", f"MIXED_FULL screen: all {len(cleared)} mixed bucket graphs built every backend's "
                            "bucketed metadata on their dummy")
    return manager._capture_descs.get(full, descs)
