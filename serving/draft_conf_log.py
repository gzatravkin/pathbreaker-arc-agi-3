import atexit
import os
import threading
import time

import torch

ENV = "TAAF_VLLM_DRAFT_CONF_LOG"
INTERVAL_ENV = "TAAF_VLLM_DRAFT_CONF_LOG_INTERVAL_S"
DEFAULT_INTERVAL_S = 30.0
BUCKETS = 20
RING_SIZE = 128

_lock = threading.Lock()
_states: dict[tuple[str, int, int], dict] = {}


def say(text: str) -> None:
    print(f"DRAFT_CONF_LOG {text}", flush=True)


def _interval_s() -> float:
    said = str(os.environ.get(INTERVAL_ENV) or "").strip()
    if not said:
        return DEFAULT_INTERVAL_S
    return max(1.0, float(said))


def _new_state(max_num_reqs: int, num_steps: int, device: torch.device) -> dict:
    trash = max_num_reqs
    return {
        "max_num_reqs": max_num_reqs,
        "num_steps": num_steps,
        "device": device,
        "trash": trash,
        "pending_prob": torch.full((trash + 1, num_steps), -1.0, dtype=torch.float32, device=device),
        "pending_margin": torch.zeros((trash + 1, num_steps), dtype=torch.float32, device=device),
        "proposed": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "accepted": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "observed": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "margin_sum": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "pending_score": torch.full((trash + 1, num_steps), -1.0, dtype=torch.float32, device=device),
        "score_proposed": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "score_observed": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "score_accepted": torch.zeros((num_steps, BUCKETS), dtype=torch.float64, device=device),
        "ring": torch.full((RING_SIZE, 3), -1.0, dtype=torch.float32, device=device),
        "ring_pos": 0,
        "last_report": time.monotonic(),
        "steps_arange": torch.arange(num_steps, dtype=torch.int64, device=device),
    }


def arm(max_num_reqs: int, num_steps: int, device: torch.device) -> dict:
    key = (str(device), int(max_num_reqs), int(num_steps))
    with _lock:
        held = _states.get(key)
        if held is not None:
            return held
        made = _new_state(max_num_reqs, num_steps, device)
        _states[key] = made
    atexit.register(_report, made, True)
    say(f"armed: max_num_reqs={max_num_reqs} num_speculative_steps={num_steps} buckets={BUCKETS} "
        f"interval_s={_interval_s()}; records softmax confidence of each drafted token and whether "
        f"the verifier accepted it, read-only and never changes sampling or acceptance")
    return made


def _bucket_index(prob: torch.Tensor) -> torch.Tensor:
    clamped = prob.clamp(0.0, 1.0)
    return (clamped * BUCKETS).long().clamp_(max=BUCKETS - 1)


def log_proposed(
    state: dict,
    draft_step: torch.Tensor,
    logits: torch.Tensor,
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    valid = idx_mapping >= 0
    trash = state["trash"]
    write_idx = torch.where(valid, idx_mapping, torch.full_like(idx_mapping, trash))
    read_idx = idx_mapping.clamp(min=0)
    temp = temperature[read_idx].float()
    temp = torch.where(temp > 0, temp, torch.ones_like(temp))
    scaled = logits.float() / temp.unsqueeze(-1)
    top2 = scaled.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    if sampled is None:
        chosen_logprob = top2[:, 0] - torch.logsumexp(scaled, dim=-1)
    else:
        chosen = sampled.clamp(min=0).view(-1, 1)
        chosen_logprob = torch.log_softmax(scaled, dim=-1).gather(1, chosen).squeeze(1)
    prob = chosen_logprob.exp()
    step_vec = draft_step.expand_as(write_idx) if draft_step.dim() == 0 else draft_step
    state["pending_prob"][write_idx, step_vec] = prob
    state["pending_margin"][write_idx, step_vec] = margin
    bucket = _bucket_index(prob)
    flat = (step_vec * BUCKETS + bucket).reshape(-1)
    weight = valid.to(state["proposed"].dtype).reshape(-1)
    state["proposed"].view(-1).index_add_(0, flat, weight)
    previous = state["pending_score"][write_idx, (step_vec - 1).clamp(min=0)].clamp(min=0.0)
    score = torch.where(step_vec > 0, previous, torch.ones_like(previous)) * prob
    state["pending_score"][write_idx, step_vec] = score
    score_flat = (step_vec * BUCKETS + _bucket_index(score)).reshape(-1)
    state["score_proposed"].view(-1).index_add_(0, score_flat, weight)
    lane = 0
    if prob.shape[0] > lane:
        row = state["ring_pos"] % RING_SIZE
        step_scalar = draft_step if draft_step.dim() == 0 else draft_step[lane]
        state["ring"][row, 0] = step_scalar.float()
        state["ring"][row, 1] = prob[lane]
        state["ring"][row, 2] = margin[lane]
        state["ring_pos"] = state["ring_pos"] + 1


def log_verified(
    state: dict,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    valid = idx_mapping >= 0
    trash = state["trash"]
    read_idx = torch.where(valid, idx_mapping, torch.full_like(idx_mapping, trash))
    num_accepted = (num_sampled - 1).clamp(min=0)
    num_admitted = num_accepted + num_rejected
    steps = state["steps_arange"]
    observed = (steps[None, :] <= num_accepted[:, None]) & (steps[None, :] < num_admitted[:, None])
    observed = observed & valid[:, None]
    accepted = (steps[None, :] < num_accepted[:, None]) & observed
    probs = state["pending_prob"][read_idx]
    margins = state["pending_margin"][read_idx]
    bucket = _bucket_index(probs.clamp(min=0.0))
    flat = (steps[None, :].expand_as(bucket) * BUCKETS + bucket).reshape(-1)
    observed_weight = observed.to(state["observed"].dtype).reshape(-1)
    accepted_weight = accepted.to(state["accepted"].dtype).reshape(-1)
    margin_weight = (margins * observed.to(margins.dtype)).reshape(-1).to(state["margin_sum"].dtype)
    state["observed"].view(-1).index_add_(0, flat, observed_weight)
    state["accepted"].view(-1).index_add_(0, flat, accepted_weight)
    state["margin_sum"].view(-1).index_add_(0, flat, margin_weight)
    scores = state["pending_score"][read_idx].clamp(min=0.0)
    score_flat = (steps[None, :].expand_as(scores) * BUCKETS + _bucket_index(scores)).reshape(-1)
    state["score_observed"].view(-1).index_add_(0, score_flat, observed_weight)
    state["score_accepted"].view(-1).index_add_(0, score_flat, accepted_weight)
    _maybe_report(state)


def _capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _maybe_report(state: dict, force: bool = False) -> None:
    now = time.monotonic()
    if not force and now - state["last_report"] < _interval_s():
        return
    if _capturing():
        return
    state["last_report"] = now
    _report(state, force)


def _survival_cells(state: dict) -> list[str]:
    proposed = state["score_proposed"].cpu().tolist()
    observed = state["score_observed"].cpu().tolist()
    accepted = state["score_accepted"].cpu().tolist()
    cells = []
    for step in range(state["num_steps"]):
        for bucket in range(BUCKETS):
            if proposed[step][bucket] <= 0:
                continue
            cells.append(
                f"step={step} score=[{bucket / BUCKETS:.2f},{(bucket + 1) / BUCKETS:.2f}) "
                f"proposed={proposed[step][bucket]:.0f} observed={observed[step][bucket]:.0f} "
                f"accepted={accepted[step][bucket]:.0f}"
            )
    return cells


def _report(state: dict, force: bool) -> None:
    if _capturing():
        return
    proposed = state["proposed"].cpu().tolist()
    accepted = state["accepted"].cpu().tolist()
    observed = state["observed"].cpu().tolist()
    margin_sum = state["margin_sum"].cpu().tolist()
    cells = []
    for step in range(state["num_steps"]):
        for bucket in range(BUCKETS):
            p = proposed[step][bucket]
            if p <= 0:
                continue
            o = observed[step][bucket]
            a = accepted[step][bucket]
            m = margin_sum[step][bucket]
            lo, hi = bucket / BUCKETS, (bucket + 1) / BUCKETS
            rate = a / o if o > 0 else float("nan")
            mean_margin = m / o if o > 0 else float("nan")
            cells.append(
                f"step={step} conf=[{lo:.2f},{hi:.2f}) proposed={p:.0f} observed={o:.0f} "
                f"accepted={a:.0f} accept_rate={rate:.3f} margin_sum={m:.3f} "
                f"mean_margin={mean_margin:.3f}"
            )
    if cells:
        say(("shutdown " if force else "") + "calibration " + " | ".join(cells))
    else:
        say(("shutdown " if force else "") + "calibration no samples yet")
    survival = _survival_cells(state)
    if survival:
        say(("shutdown " if force else "") + "survival " + " | ".join(survival))
    ring = state["ring"].cpu().tolist()
    pos = state["ring_pos"]
    recent = [ring[i % RING_SIZE] for i in range(max(0, pos - 5), pos)]
    shown = [f"(step={s:.0f},conf={p:.3f},margin={m:.3f})" for s, p, m in recent if p >= 0.0]
    if shown:
        say(("shutdown " if force else "") + "ring " + " ".join(shown))
