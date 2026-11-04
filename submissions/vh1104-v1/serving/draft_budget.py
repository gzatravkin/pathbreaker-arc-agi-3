import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENV = "TAAF_VLLM_DRAFT_BUDGET"
DEFAULT_CONF = 0.75
_STATE: dict = {"specs": {}, "said": set(), "per_device": {}}
_TALLY: dict = {"steps": {}, "at": 0.0}
TALLY_S = 30.0


def _sibling(name):
    held = sys.modules.get(f"taaf_{name}")
    if held is not None:
        return held
    try:
        import importlib
        return importlib.import_module(f"vllm.taaf_{name}")
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for candidate in (here / f"taaf_{name}.py", here / f"{name}.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(f"taaf_{name}", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"taaf_{name}"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"{ENV} needs taaf_{name} beside taaf_draft_budget")


def policy():
    return _sibling("draft_budget_policy")


def kernel():
    return _sibling("draft_budget_kernel")


def _say(key, line):
    if key not in _STATE["said"]:
        _STATE["said"].add(key)
        print(line, flush=True)


ARMED = str(os.environ.get(ENV) or "").strip() not in ("", "0")
EARLY_STOP = str(os.environ.get("TAAF_VLLM_DRAFT_EARLY_STOP") or "").strip() not in ("", "0")
BUDGET_CHECK = str(os.environ.get("TAAF_VLLM_DRAFT_BUDGET_CHECK") or "").strip() not in ("", "0")
_CHECKS = [0, 0]


def armed():
    return ARMED


def spec(depth):
    said = str(os.environ.get(ENV) or "").strip()
    key = (said, depth)
    held = _STATE["specs"].get(key)
    if held is None:
        found = policy().parse_knob(said, boot_depth=depth)
        held = found if found is not None else policy().parse_knob("full", boot_depth=depth)
        if held["depth"] > depth:
            raise RuntimeError(f"{ENV}={said!r} verifies up to {held['depth']} drafts a lane but the drafter "
                               f"makes {depth}; boot with num_speculative_tokens >= {held['depth']}")
        _STATE["specs"][key] = held
    return held


def conf_for(device, max_num_reqs, depth):
    key = (str(device), int(depth))
    held = _STATE["per_device"].get(key)
    if held is None:
        held = torch.full((int(max_num_reqs) + 1, int(depth)), DEFAULT_CONF, dtype=torch.float32, device=device)
        _STATE["per_device"][key] = held
        _say(("armed", key), f"DRAFT_BUDGET armed: confidence buffer {tuple(held.shape)} on {device}; the drafter "
                             f"writes the probability of each drafted token, the allocator ranks slots by it")
    if held.shape[0] < int(max_num_reqs) + 1:
        raise RuntimeError(f"{ENV}: the drafter sized its confidence buffer for {held.shape[0] - 1} requests and the "
                           f"manager asks for {int(max_num_reqs)}")
    return held


def arm_speculator(owner, device):
    owner._taaf_budget_conf = conf_for(device, owner.max_num_reqs, owner.num_speculative_steps)
    return owner._taaf_budget_conf


def record(conf, draft_step, values, ids, sampled, idx_mapping, temperature):
    read_idx = idx_mapping.clamp(min=0)
    temp = temperature[read_idx].float()
    temp = torch.where(temp > 0, temp, torch.ones_like(temp))
    probs = torch.softmax(values.float() / temp.unsqueeze(-1), dim=-1)
    hit = ids == sampled.reshape(-1, 1).to(ids.dtype)
    chosen = (probs * hit.to(probs.dtype)).sum(dim=-1)
    trash = conf.shape[0] - 1
    write_idx = torch.where(idx_mapping >= 0, idx_mapping, torch.full_like(idx_mapping, trash))
    step_vec = draft_step.expand_as(write_idx) if draft_step.dim() == 0 else draft_step
    conf[write_idx, step_vec] = chosen


def reset(manager, req_idx):
    conf = _state(manager).get("conf")
    if conf is not None:
        conf[req_idx].fill_(DEFAULT_CONF)
    early = _early_stop(manager)
    if early is not None:
        from vllm.taaf_draft_early_stop import reset as early_reset

        early_reset(early, req_idx)


def _state(manager):
    held = vars(manager).get("_taaf_budget")
    if held is None:
        held = vars(manager)["_taaf_budget"] = {}
    return held


def _prepare(manager):
    state = _state(manager)
    if "kernel" in state:
        return state
    device = manager.req_states.device
    depth = manager.num_speculative_steps
    pol = policy()
    state["conf"] = conf_for(device, manager.req_states.max_num_reqs, depth)
    state["meta"] = torch.zeros(2, dtype=torch.int32, device=device)
    state["out"] = torch.zeros(max(32, 1 << (int(manager.req_states.max_num_reqs) - 1).bit_length()),
                               dtype=torch.int32, device=device)
    state["tables"] = {}
    state["kernel"] = kernel()
    state["pol"] = pol
    return state


def _table(state, name, depth):
    held = state["tables"].get(name)
    if held is None:
        pol = state["pol"]
        entry = pol.load_tables()["tables"][name]
        edges, keys = pol.table_arrays(entry, device=state["conf"].device)
        if keys.shape[0] < depth:
            pad = keys[-1:].expand(depth - keys.shape[0], keys.shape[1])
            keys = torch.cat([keys, pad], dim=0)
        held = state["tables"][name] = (edges.contiguous(), keys.contiguous())
    return held


def note_capture_sizes(sizes, manager=None):
    pol = policy()
    kept = tuple(sorted({int(s) for s in sizes if 0 < int(s) <= pol.CAPTURE_SIZES[-1]}))
    if kept:
        _STATE["sizes"] = kept
        _say(("sizes", kept), f"DRAFT_BUDGET lands decode row totals on the {len(kept)} captured sizes up to "
                              f"{kept[-1]}")
    if manager is not None:
        prewarm(manager)


def prewarm(manager):
    state = _prepare(manager)
    chosen = spec(manager.num_speculative_steps)
    depth = int(chosen["depth"])
    edges, keys = _table(state, chosen["table"], depth)
    top = max(32, 1 << (int(manager.req_states.max_num_reqs) - 1).bit_length())
    size = 32
    while size <= top:
        caps = torch.zeros(size, dtype=torch.int32, device=state["conf"].device)
        idx = torch.zeros(size, dtype=torch.int32, device=state["conf"].device)
        state["meta"].zero_()
        state["kernel"].allocate(state["conf"][:, :depth], caps, state["meta"], edges, keys, out=state["out"], idx=idx)
        size *= 2
    _say(("prewarm", id(manager)), f"DRAFT_BUDGET compiled the allocation kernel for 32..{top} lanes at depth {depth}")


def tally(verification, mixed, total):
    key = f"{'m' if mixed else 'd'}{int(verification)}:{int(total)}"
    held = _TALLY["steps"]
    held[key] = held.get(key, 0) + 1
    now = time.monotonic()
    if now - _TALLY["at"] >= TALLY_S:
        _TALLY["at"] = now
        top = sorted(held.items(), key=lambda one: -one[1])[:8]
        held.clear()
        print("DRAFT_BUDGET tally (d decode-only, m mixed; lanes:rows a step): "
              + " ".join(f"{k}={v}" for k, v in top), flush=True)


def lane_budget(chosen, verification, scheduled_total, max_logits):
    pol = policy()
    rows = pol.rows_for_lanes(chosen, int(verification), _STATE.get("sizes") or pol.CAPTURE_SIZES)
    budget = max(0, rows - int(verification))
    budget = min(budget, int(scheduled_total), int(max_logits))
    return budget


def get_num_tokens(manager, num_tokens_per_req, draft_tokens):
    req_ids = list(num_tokens_per_req)
    num_reqs = len(req_ids)
    scheduled_tokens = np.fromiter(num_tokens_per_req.values(), dtype=np.int32, count=num_reqs)
    scheduled_drafts = np.fromiter((len(draft_tokens.get(req_id, ())) for req_id in req_ids), dtype=np.int32,
                                   count=num_reqs)
    chosen = spec(manager.num_speculative_steps)
    capped = np.minimum(scheduled_drafts, chosen["depth"])
    non_draft = scheduled_tokens - scheduled_drafts
    verification = int((scheduled_drafts > 0).sum())
    max_logits = max(0, manager._max_total_logits - num_reqs * manager.num_bonus_tokens)
    mixed = bool((non_draft > manager.num_bonus_tokens).any())
    if chosen["name"] == "full" or mixed:
        budget = min(int(capped.sum()), max_logits)
    else:
        budget = lane_budget(chosen, verification, int(capped.sum()), max_logits)
    early = _early_stop(manager)
    if early is not None:
        from vllm.taaf_draft_early_stop import rows_bound

        budget = min(budget, rows_bound(early, verification))
    tally(verification, mixed, int(non_draft.sum()) + budget)
    manager._batch_budget = (
        {req_id: int(n) for req_id, n in zip(req_ids, scheduled_drafts, strict=True)},
        {req_id: int(n) for req_id, n in zip(req_ids, non_draft, strict=True)},
        int(budget),
    )
    return int(non_draft.sum()) + int(budget)


def _early_stop(manager):
    if not EARLY_STOP:
        return None
    from vllm.taaf_draft_early_stop import held

    return held(manager.req_states.device)


def _check(manager, idx_mapping, capacities, draft_budget, num_reqs, depth):
    caps = [int(c) for c in capacities[:num_reqs].tolist()]
    slots = [int(s) for s in idx_mapping[:num_reqs].tolist()]
    held = getattr(manager, "_batch_budget", None)
    scheduled = list(held[0].values()) if held else []
    total = sum(caps)
    problems = []
    if total != int(draft_budget):
        problems.append(f"sum of capacities {total} != budget {int(draft_budget)}")
    if any(c < 0 or c > depth for c in caps):
        problems.append(f"capacity outside 0..{depth}")
    if any(s < 0 for s in slots):
        problems.append("negative slot")
    if scheduled and total > sum(scheduled):
        problems.append(f"capacities {total} exceed scheduled drafts {sum(scheduled)}")
    if scheduled and len(scheduled) == num_reqs and any(c > d for c, d in zip(caps, scheduled)):
        problems.append("a lane's capacity exceeds its scheduled drafts (same row order assumed)")
    _CHECKS[0] += 1
    uneven = len(set(caps)) > 1
    if problems or (uneven and _CHECKS[1] < 5):
        _CHECKS[1] += int(uneven and not problems)
        print(f"DRAFT_BUDGET_CHECK {'FAIL' if problems else 'uneven'} step {_CHECKS[0]}: {'; '.join(problems) or 'ok'} | "
              f"lanes {num_reqs} budget {int(draft_budget)} depth {depth} capacities {caps} slots {slots} "
              f"scheduled {scheduled} non_draft {list(held[1].values()) if held else None}", flush=True)
    if problems:
        raise RuntimeError("DRAFT_BUDGET_CHECK: " + "; ".join(problems))


def assign(manager, idx_mapping, capacities, draft_budget, num_reqs):
    state = _prepare(manager)
    chosen = spec(manager.num_speculative_steps)
    depth = int(chosen["depth"])
    edges, keys = _table(state, chosen["table"], depth)
    pinned = torch.tensor([int(num_reqs), int(draft_budget)], dtype=torch.int32, pin_memory=True)
    state["meta"].copy_(pinned, non_blocking=True)
    idx = idx_mapping if idx_mapping.dtype in (torch.int32, torch.int64) else idx_mapping.to(torch.int64)
    out = state["kernel"].allocate(state["conf"][:, :depth], capacities[:num_reqs].to(torch.int32).contiguous(),
                                   state["meta"], edges, keys, out=state["out"], idx=idx)
    capacities[:num_reqs].copy_(out[:num_reqs])
    if BUDGET_CHECK:
        _check(manager, idx_mapping, capacities, draft_budget, num_reqs, depth)
    _say(("first", chosen["name"], depth), f"DRAFT_BUDGET first allocation: {num_reqs} lanes, budget "
                                           f"{int(draft_budget)} draft rows, depth {depth}, table {chosen['table']}, "
                                           f"preset {chosen['name']}")
