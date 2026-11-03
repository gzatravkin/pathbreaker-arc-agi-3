import atexit
import os
import threading
import time

import torch

ENV = "TAAF_VLLM_DRAFT_SUMMIN"
INTERVAL_ENV = "TAAF_VLLM_DRAFT_SUMMIN_INTERVAL_S"
TAUS_ENV = "TAAF_VLLM_DRAFT_SUMMIN_TAUS"
DEFAULT_INTERVAL_S = 60.0
DEFAULT_TAUS = (0.7, 1.0, 1.3, 1.6)
TOP = 20
VOCAB_CACHE = "_taaf_draft_vocab"

_lock = threading.Lock()
_state: dict | None = None


def say(text: str) -> None:
    print(f"DRAFT_SUMMIN {text}", flush=True)


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _interval_s() -> float:
    said = str(os.environ.get(INTERVAL_ENV) or "").strip()
    return max(1.0, float(said)) if said else DEFAULT_INTERVAL_S


def taus() -> tuple[float, ...]:
    said = str(os.environ.get(TAUS_ENV) or "").strip()
    if not said:
        return DEFAULT_TAUS
    values = tuple(float(one) for one in said.split(","))
    if not values or any(not 0.05 <= one <= 20.0 for one in values):
        raise RuntimeError(f"{TAUS_ENV}={said!r} must be comma-separated multipliers in 0.05..20")
    return values


def state() -> dict | None:
    return _state


def arm(max_num_reqs: int, num_steps: int, device: torch.device) -> dict:
    global _state
    with _lock:
        if _state is not None:
            return _state
        trash = max_num_reqs
        scales = taus()
        n = len(scales)
        _state = {
            "trash": trash,
            "num_steps": num_steps,
            "taus": scales,
            "tau_tensor": torch.tensor(scales, dtype=torch.float32, device=device),
            "q_ids": torch.full((trash + 1, num_steps, TOP), -1, dtype=torch.int64, device=device),
            "q_logits": torch.zeros((trash + 1, num_steps, TOP), dtype=torch.float32, device=device),
            "pending_greedy": torch.zeros((trash + 1, num_steps), dtype=torch.float32, device=device),
            "pending_summin": torch.zeros((trash + 1, num_steps, n), dtype=torch.float32, device=device),
            "pending_fresh": torch.zeros((trash + 1, num_steps), dtype=torch.float32, device=device),
            "pending_exact": torch.zeros((trash + 1, num_steps), dtype=torch.float32, device=device),
            "rows": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "exact_rows": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "greedy_sum": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "summin_sum": torch.zeros((num_steps, n), dtype=torch.float64, device=device),
            "accepted": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "unscored": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "nonfinite": torch.zeros(num_steps, dtype=torch.float64, device=device),
            "steps_arange": torch.arange(num_steps, dtype=torch.int64, device=device),
            "last_report": time.monotonic(),
        }
    atexit.register(report, _state, True)
    say(f"armed: max_num_reqs={max_num_reqs} num_speculative_steps={num_steps} top={TOP} "
        f"taus={','.join(str(one) for one in scales)} (multipliers of each request's temperature) "
        f"interval_s={_interval_s()}; scores each verified draft position by p(greedy draft) and by "
        f"sum min(p, q_tau) over the top-{TOP} of the target and of the drafter, read-only and never "
        f"changes sampling or acceptance")
    return _state


def vocab_ids(model, width: int) -> torch.Tensor | None:
    built = vars(model).get(VOCAB_CACHE)
    if built is None or built[1].shape[0] != width:
        return None
    return built[1]


def log_q(held: dict, draft_step: torch.Tensor, logits: torch.Tensor, idx_mapping: torch.Tensor,
          model) -> None:
    valid = idx_mapping >= 0
    write_idx = torch.where(valid, idx_mapping, torch.full_like(idx_mapping, held["trash"]))
    values, cols = logits.float().topk(TOP, dim=-1)
    index = vocab_ids(model, logits.shape[-1])
    ids = cols if index is None else index[cols]
    earlier = torch.ones(TOP, TOP, dtype=torch.bool, device=ids.device).tril(-1)
    repeated = ((ids.unsqueeze(-1) == ids.unsqueeze(-2)) & earlier).any(-1)
    values = values.masked_fill(repeated, float("-inf"))
    step_vec = draft_step.expand_as(write_idx) if draft_step.dim() == 0 else draft_step
    held["q_ids"][write_idx, step_vec] = ids
    held["q_logits"][write_idx, step_vec] = values


def _target_probs(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
                  top_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, ids = logits.float().topk(TOP, dim=-1)
    temp = torch.where(temperature > 0, temperature, torch.ones_like(temperature)).float()
    scaled = values / temp.unsqueeze(-1)
    rank = torch.arange(TOP, device=logits.device).unsqueeze(0)
    k = torch.where(top_k > 0, top_k, torch.full_like(top_k, TOP)).clamp(max=TOP)
    scaled = scaled.masked_fill(rank >= k.unsqueeze(-1), float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    before = probs.cumsum(dim=-1) - probs
    keep = before < top_p.float().unsqueeze(-1)
    keep[:, 0] = True
    probs = probs * keep
    probs = probs / probs.sum(dim=-1, keepdim=True)
    exact = (top_k > 0) & (top_k <= TOP)
    return probs, ids, exact


def observe_target(logits: torch.Tensor, ex_idx: torch.Tensor, ex_pos: torch.Tensor,
                   temperature_table: torch.Tensor, top_k_table: torch.Tensor,
                   top_p_table: torch.Tensor, live: torch.Tensor | None = None) -> None:
    held = _state
    if held is None:
        return
    trash = held["trash"]
    steps = held["num_steps"]
    valid = (ex_idx >= 0) & (ex_idx < trash)
    if live is not None:
        valid = valid & live
    req = torch.where(valid, ex_idx, torch.full_like(ex_idx, trash)).long()
    count = torch.zeros(trash + 1, dtype=torch.int64, device=logits.device)
    count.index_add_(0, req, torch.ones_like(req))
    pos = ex_pos.long()
    verifies = valid & (pos < count[req] - 1) & (pos < steps)
    step = pos.clamp(min=0, max=steps - 1)
    read = req.clamp(max=trash - 1)
    p, p_ids, exact = _target_probs(logits, temperature_table[read], top_k_table[read],
                                    top_p_table[read])
    q_ids = held["q_ids"][req, step]
    q_logits = held["q_logits"][req, step]
    match = p_ids.unsqueeze(-1) == q_ids.unsqueeze(-2)
    greedy = (p * match[:, :, 0].float()).sum(-1)
    temp = torch.where(temperature_table[read] > 0, temperature_table[read],
                       torch.ones_like(temperature_table[read])).float()
    scale = temp.unsqueeze(-1) * held["tau_tensor"].unsqueeze(0)
    q = torch.softmax(q_logits.unsqueeze(1) / scale.unsqueeze(-1), dim=-1)
    paired = torch.minimum(p.unsqueeze(1).unsqueeze(-1), q.unsqueeze(2))
    summin = (paired * match.unsqueeze(1).float()).sum(dim=(-1, -2))
    write = torch.where(verifies, req, torch.full_like(req, trash))
    held["pending_greedy"][write, step] = greedy
    held["pending_summin"][write, step] = summin
    held["pending_fresh"][write, step] = verifies.float()
    held["pending_exact"][write, step] = (verifies & exact).float()


def fold(held: dict, idx_mapping: torch.Tensor, num_sampled: torch.Tensor,
         num_rejected: torch.Tensor) -> None:
    valid = idx_mapping >= 0
    read = torch.where(valid, idx_mapping, torch.full_like(idx_mapping, held["trash"]))
    accepted_n = (num_sampled - 1).clamp(min=0)
    admitted = accepted_n + num_rejected
    steps = held["steps_arange"]
    observed = (steps[None, :] <= accepted_n[:, None]) & (steps[None, :] < admitted[:, None]) & valid[:, None]
    fresh = held["pending_fresh"][read] > 0
    greedy = held["pending_greedy"][read].double()
    summin = held["pending_summin"][read].double()
    finite = torch.isfinite(greedy) & torch.isfinite(summin).all(-1)
    picked = observed & fresh
    held["nonfinite"] += (picked & ~finite).double().sum(0)
    picked = picked & finite
    scored = picked.double()
    zero = torch.zeros((), dtype=torch.float64, device=greedy.device)
    held["rows"] += scored.sum(0)
    held["exact_rows"] += (scored * held["pending_exact"][read].double()).sum(0)
    held["greedy_sum"] += torch.where(picked, greedy, zero).sum(0)
    held["summin_sum"] += torch.where(picked.unsqueeze(-1), summin, zero).sum(0)
    held["accepted"] += ((steps[None, :] < accepted_n[:, None]) & picked).double().sum(0)
    held["unscored"] += (observed & ~fresh).double().sum(0)
    held["pending_fresh"][read] = 0.0
    _maybe_report(held)


def _capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _maybe_report(held: dict) -> None:
    now = time.monotonic()
    if now - held["last_report"] < _interval_s() or _capturing():
        return
    held["last_report"] = now
    report(held, False)


def _chain(rates: list[float]) -> float:
    total, alive = 1.0, 1.0
    for one in rates:
        alive *= one
        total += alive
    return total


def report(held: dict, final: bool) -> None:
    if _capturing():
        return
    rows = held["rows"].cpu().tolist()
    exact = held["exact_rows"].cpu().tolist()
    greedy = held["greedy_sum"].cpu().tolist()
    summin = held["summin_sum"].cpu().tolist()
    accepted = held["accepted"].cpu().tolist()
    unscored = held["unscored"].cpu().tolist()
    nonfinite = held["nonfinite"].cpu().tolist()
    cells, greedy_rates, tau_rates = [], [], [[] for _ in held["taus"]]
    for step in range(held["num_steps"]):
        n = rows[step]
        g = greedy[step] / n if n else float("nan")
        a = accepted[step] / n if n else float("nan")
        greedy_rates.append(g)
        per_tau = []
        for at, tau in enumerate(held["taus"]):
            s = summin[step][at] / n if n else float("nan")
            tau_rates[at].append(s)
            per_tau.append(f"tau{tau}={s:.4f}")
        cells.append(f"step={step} rows={n:.0f} exact={exact[step]:.0f} unscored={unscored[step]:.0f} "
                     f"nonfinite={nonfinite[step]:.0f} "
                     f"accepted={a:.4f} greedy_p={g:.4f} " + " ".join(per_tau))
    chains = [f"greedy={_chain(greedy_rates):.4f}"] + [
        f"tau{tau}={_chain(tau_rates[at]):.4f}" for at, tau in enumerate(held["taus"])]
    say(("final" if final else "window") + ": conditional per position | " + " | ".join(cells)
        + " || tokens a step from the conditional rates: " + " ".join(chains))
