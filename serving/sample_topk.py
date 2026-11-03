import os
import sys
import types

import numpy as np
import torch

ENV = "TAAF_VLLM_SAMPLE_TOPK"
DRAFT_ENV = "TAAF_VLLM_SAMPLE_TOPK_DRAFT"
KEPT_ENV = "TAAF_VLLM_TAIL_KEPT_LOGITS"
TARGET_VOCAB_MODULE = "vllm.taaf_target_vocab"
BLOCK_MODULE = "vllm.taaf_block_verify"
LAMBDA_MODULE = "vllm.taaf_accept_lambda"
CU_NP = "_taaf_cu_np"
EVERY = 2000
_tally = {"taken": 0, "tail": 0, "stock": 0}
_reasons: dict[str, int] = {}
_rows = {"top_rows": 0, "top_reqs": 0, "stock_rows": 0, "stock_reqs": 0, "split_steps": 0}
_misses = {"device": None}
_said = set()


def count():
    raw = os.environ.get(ENV, "").strip()
    value = int(raw) if raw.isdigit() else 0
    return value if 2 <= value <= 256 and value & (value - 1) == 0 else 0


K = count()
DRAFT = bool(K) and str(os.environ.get(DRAFT_ENV) or "").strip() == "1"


def say(key, line):
    if key not in _said:
        _said.add(key)
        print(f"SAMPLE_TOPK {line}", flush=True)


def _report():
    total = sum(_tally.values())
    if total % EVERY == 0:
        drafts = ""
        if DRAFT:
            held = _misses["device"]
            missed = int(held.item()) if held is not None else 0
            drafts = (f" draft_rows top={_rows['top_rows']} ({_rows['top_reqs']} requests) "
                      f"stock={_rows['stock_rows']} ({_rows['stock_reqs']} requests) split_steps="
                      f"{_rows['split_steps']} drafts_outside_their_top={missed}")
        print(f"SAMPLE_TOPK tally taken={_tally['taken']} tail={_tally['tail']} stock={_tally['stock']} "
              f"reasons={dict(sorted(_reasons.items()))}{drafts}", flush=True)


def refuse(reason):
    _tally["stock"] += 1
    _reasons[reason] = _reasons.get(reason, 0) + 1
    _report()


def took(where):
    _tally[where] += 1
    _report()


def draft_ids():
    block = sys.modules.get(BLOCK_MODULE)
    state = getattr(block, "_state", None)
    return state.get("ids") if isinstance(state, dict) else None


def fits(sampler, idx_mapping_np):
    states = sampler.sampling_states
    temperature = states.temperature.np[idx_mapping_np]
    top_k = states.top_k.np[idx_mapping_np]
    return (temperature == 0.0) | ((top_k >= 1) & (top_k <= K))


def step_refusal(sampler, draft_logits, idx_mapping_np):
    states = sampler.sampling_states
    if K == 0:
        return "off"
    if draft_logits is not None:
        if not DRAFT:
            return "draft_logits"
        ids = draft_ids()
        if ids is None:
            return "draft_ids"
        if (not draft_logits.is_contiguous() or ids.dim() != 3 or ids.shape[0] != draft_logits.shape[0] + 1
                or ids.shape[1] != draft_logits.shape[1]):
            return "draft_layout"
    if states.vocab_size <= K:
        return "vocab"
    if np.any(states.min_p.np[idx_mapping_np] != 0.0):
        return "min_p"
    return None


def refusal(sampler, draft_logits, idx_mapping_np):
    reason = step_refusal(sampler, draft_logits, idx_mapping_np)
    if reason is not None:
        return reason
    if draft_logits is not None:
        return None if fits(sampler, idx_mapping_np).all() else "top_k"
    states = sampler.sampling_states
    temperature = states.temperature.np[idx_mapping_np]
    top_k = states.top_k.np[idx_mapping_np]
    if np.any((temperature != 0.0) & (top_k > K)):
        return "top_k"
    return None


def eager_refusal(rs, logits, draft_logits, idx_mapping_np):
    sampler = rs.sampler
    drafted = draft_logits is not None and DRAFT
    reason = (step_refusal if drafted else refusal)(sampler, draft_logits, idx_mapping_np)
    if reason is not None:
        return reason
    if rs.watermark_key is not None:
        return "watermark"
    if rs.synthetic_conditional_rates is not None or (rs.use_block_verification and not drafted):
        return "method"
    if drafted and getattr(rs, "enable_adaptive_verification", False):
        return "adaptive"
    if logits.ndim != 2 or logits.stride(-1) != 1 or not logits.is_cuda:
        return "shape"
    if sampler.sampling_states.max_num_logprobs(idx_mapping_np) != -1:
        return "logprobs"
    if np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np]):
        return "logit_bias"
    if np.any(sampler.penalties_state.use_penalty[idx_mapping_np]):
        return "penalty"
    if np.any(sampler.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
        return "bad_words"
    budget = sampler.thinking_budget_state
    if budget.enabled and np.any(budget.use_thinking_budget[idx_mapping_np]):
        return "thinking_budget"
    return None


def kept_owner(model, vocab):
    if vars(model).get(vocab.CACHE) or str(os.environ.get(KEPT_ENV) or "").strip() != "1":
        return model
    inner = getattr(model, "language_model", None)
    if inner is None or not vars(inner).get(vocab.CACHE):
        say("kept_aside", f"TAIL_KEPT_LOGITS stands aside: neither {type(model).__name__} nor a language_model "
                          f"under it holds the target vocabulary's kept head, so the tail scores the whole head")
        return model
    say("kept_inner", f"TAIL_KEPT_LOGITS fires: the sampling tail reads the kept head that "
                      f"{type(inner).__name__} built under the {type(model).__name__} wrapper")
    return inner


def kept_logits(model, hidden):
    vocab = sys.modules.get(TARGET_VOCAB_MODULE)
    owner = model if vocab is None else kept_owner(model, vocab)
    built = vars(owner).get(vocab.CACHE) if vocab is not None else None
    processor = getattr(owner, "logits_processor", None)
    if not built or processor is None or processor.soft_cap is not None or processor.scale != 1.0:
        return None
    rows, index = built
    return torch.nn.functional.linear(hidden.to(rows.dtype), rows)[:, :index.numel()], index


def real_ids(ids, call):
    module = sys.modules.get(LAMBDA_MODULE)
    if module is None:
        return call()
    held = module.use_ids(ids)
    try:
        return call()
    finally:
        module.use_ids(held)


def candidates(logits, draft_sampled, ids=None):
    from vllm.taaf_sample_topk_kernel import topk

    values, columns = topk(logits, K)
    if ids is not None:
        columns = ids[columns]
    small = values.new_empty((logits.shape[0], K + 1))
    small[:, :K] = values
    small[:, K] = float("-inf")
    scored_by = torch.cat((columns[:1], columns[:-1]))
    hit = scored_by == draft_sampled.to(columns.dtype)[:, None]
    local = torch.where(hit.any(dim=1), hit.to(torch.int8).argmax(dim=1),
                        torch.full_like(draft_sampled, K, dtype=torch.int64))
    local = torch.where(draft_sampled < 0, draft_sampled.to(torch.int64), local)
    return small, columns, local.to(draft_sampled.dtype)


def scored(small, expanded_idx_mapping, temperature, top_k, top_p):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature

    apply_temperature(small, expanded_idx_mapping, temperature)
    return apply_top_k_top_p(small, top_k[expanded_idx_mapping].clamp(max=K), top_p[expanded_idx_mapping])


def tokens(columns, sampled, cu_num_logits):
    steps = sampled.shape[1]
    rows = (cu_num_logits[:-1].to(torch.int64)[:, None]
            + torch.arange(steps, device=sampled.device, dtype=torch.int64)[None, :])
    rows = rows.clamp(min=0, max=columns.shape[0] - 1)
    return columns[rows, sampled.clamp(min=0, max=K - 1)].to(sampled.dtype)


def sample(rejection_sample, logits, draft_sampled, cu, pos, idx, ex_idx, ex_pos, temperature, top_k, top_p, seeds,
           steps, use_fp64):
    kept, ids = logits if isinstance(logits, tuple) else (logits, None)
    small, columns, local = candidates(kept, draft_sampled, ids)
    probs_in = scored(small, ex_idx, temperature, top_k, top_p)
    sampled, num_sampled = real_ids(draft_sampled, lambda: rejection_sample(
        probs_in, None, local, cu, pos, idx, ex_idx, ex_pos, temperature, seeds, steps, None, use_fp64=use_fp64,
        use_block_verification=False))
    return probs_in, tokens(columns, sampled, cu), num_sampled


def pick(logits):
    from vllm.taaf_sample_topk_kernel import topk

    kept, ids = logits if isinstance(logits, tuple) else (logits, None)
    values, columns = topk(kept, K)
    return values, (columns if ids is None else ids[columns])


def union_layout(columns, processed, ids_table, ex_idx, ex_pos, steps):
    rows = columns.shape[0]
    top = int(ids_table.shape[2])
    trash = int(ids_table.shape[0]) - 1
    req = ex_idx.to(torch.int64)
    step = ex_pos.to(torch.int64)
    live = (step < steps) & (req >= 0) & (req < trash)
    req_at = torch.where(live, req, trash)
    step_at = torch.where(live, step, 0)
    drafted = ids_table[req_at, step_at].to(torch.int64)
    drafted = torch.where(live[:, None] & (drafted >= 0), drafted, -1)
    match = drafted[:, :, None] == columns[:, None, :]
    shared = match.any(dim=2)
    slots = torch.arange(top, device=columns.device, dtype=torch.int64)
    local = torch.where(shared, match.to(torch.int8).argmax(dim=2), K + slots[None, :])
    local = torch.where(drafted >= 0, local, -1)
    width = K + top + 1
    union = columns.new_full((rows, width), -1)
    union[:, :K] = columns
    union[:, K:K + top] = torch.where(shared | (drafted < 0), -1, drafted)
    target = processed.new_full((rows, width), float("-inf"))
    target[:, :K] = processed[:, :K]
    return types.SimpleNamespace(union=union, target=target, drafted=drafted, local=local, req=torch.where(
        live, req, 0), req_at=req_at, step_at=step_at, live=live, width=width, top=top, trash=trash)


def compact_draft(draft_logits, plan, steps):
    flat = draft_logits.view(-1)
    base = plan.req * draft_logits.stride(0) + plan.step_at * draft_logits.stride(1)
    values = flat[base[:, None] + plan.drafted.clamp(min=0)].to(torch.float32)
    values = torch.where(plan.drafted >= 0, values, float("-inf"))
    out = torch.full((plan.trash + 1, steps, plan.width), float("-inf"), dtype=torch.float32,
                     device=draft_logits.device)
    rows = plan.req_at[:, None].expand(-1, plan.top)
    cols = plan.step_at[:, None].expand(-1, plan.top)
    out.index_put_((rows, cols, torch.where(plan.local >= 0, plan.local, plan.width - 1)), values)
    return out


def local_drafts(draft_sampled, plan, ex_idx, ex_pos):
    tok = draft_sampled.to(torch.int64)
    nxt = tok[1:]
    hit = plan.drafted[:-1] == nxt[:, None]
    found = hit.any(dim=1)
    at = plan.local[:-1].gather(1, hit.to(torch.int8).argmax(dim=1, keepdim=True)).squeeze(1)
    in_top = plan.union[:-1, :K] == nxt[:, None]
    at = torch.where(found, at, torch.where(in_top.any(dim=1), in_top.to(torch.int8).argmax(dim=1),
                                            plan.width - 1))
    same = (ex_idx[1:] == ex_idx[:-1]) & (ex_pos[1:].to(torch.int64) == ex_pos[:-1].to(torch.int64) + 1)
    moved = torch.where(nxt < 0, nxt, torch.where(same, at, 0))
    head = torch.where(tok[:1] < 0, tok[:1], 0)
    missed = (~found & same & (nxt >= 0) & plan.live[:-1]).sum()
    held = _misses["device"]
    if held is None or held.device != missed.device:
        held = _misses["device"] = torch.zeros((), dtype=torch.int64, device=missed.device)
    held.add_(missed)
    return torch.cat((head, moved)).to(draft_sampled.dtype)


def compact_ids(ids_table, plan):
    out = torch.full_like(ids_table, -1)
    out[plan.req_at, plan.step_at] = plan.local.to(out.dtype)
    return out


def union_tokens(union, sampled, cu_num_logits):
    steps = sampled.shape[1]
    rows = (cu_num_logits[:-1].to(torch.int64)[:, None]
            + torch.arange(steps, device=sampled.device, dtype=torch.int64)[None, :])
    rows = rows.clamp(min=0, max=union.shape[0] - 1)
    got = union[rows, sampled.clamp(min=0, max=union.shape[1] - 1)]
    return torch.where(got >= 0, got, union[rows, 0]).to(sampled.dtype)


def fused_plan(columns, processed, ids_table, ex_idx, ex_pos, steps, draft_logits, draft_sampled):
    if str(os.environ.get("TAAF_VLLM_SAMPLE_GLUE_FUSED") or "").strip() != "1":
        return None
    ints = (torch.int32, torch.int64)
    if not (columns.is_cuda and columns.dtype == torch.int64 and processed.dim() == 2 and processed.stride(1) == 1
            and ids_table.dtype in ints and ex_idx.dtype in ints and ex_pos.dtype in ints
            and draft_sampled.dtype in ints and draft_logits.is_contiguous() and draft_logits.dim() == 3
            and ids_table.dim() == 3 and columns.shape[0] > 0 and processed.shape[1] >= K):
        say("glue_stock", "SAMPLE_GLUE_FUSED stands aside for this step's layout: the eager union, compact draft "
                          "and local draft ops run")
        return None
    from vllm.taaf_small_kernels_triton import glue

    held = _misses["device"]
    if held is None or held.device != columns.device:
        held = _misses["device"] = torch.zeros((), dtype=torch.int64, device=columns.device)
    union, target, local, req_at, step_at, live, req, drafts, moved, width, top, trash = glue(
        columns, processed, ids_table, ex_idx, ex_pos, draft_logits, draft_sampled, steps, K, held)
    plan = types.SimpleNamespace(union=union, target=target, drafted=None, local=local, req=req, req_at=req_at,
                                 step_at=step_at, live=live, width=width, top=top, trash=trash)
    say("glue_fused", "SAMPLE_GLUE_FUSED fires: the union layout, compact draft rows and local draft ids of a "
                      "sampled-draft verification come from one Triton kernel in place of about 90 eager ops")
    return plan, drafts, moved


def sample_draft(rejection_sample, logits, draft_logits, draft_sampled, cu, pos, idx, ex_idx, ex_pos, temperature,
                 top_k, top_p, seeds, steps, use_fp64, use_block, picked=None):
    values, columns = pick(logits) if picked is None else picked
    small = values.new_empty((values.shape[0], K + 1))
    small[:, :K] = values
    small[:, K] = float("-inf")
    processed = scored(small, ex_idx, temperature, top_k, top_p)
    ids_table = draft_ids()
    fused = fused_plan(columns, processed, ids_table, ex_idx, ex_pos, steps, draft_logits, draft_sampled)
    if fused is not None:
        plan, drafts, local = fused
    else:
        plan = union_layout(columns, processed, ids_table, ex_idx, ex_pos, steps)
        drafts = compact_draft(draft_logits, plan, steps)
        local = local_drafts(draft_sampled, plan, ex_idx, ex_pos)
    block = sys.modules.get(BLOCK_MODULE)
    held = block._state["ids"]
    if use_block:
        block._state["ids"] = compact_ids(ids_table, plan)
    try:
        sampled, num_sampled = real_ids(draft_sampled, lambda: rejection_sample(
            plan.target, drafts, local, cu, pos, idx, ex_idx, ex_pos, temperature, seeds, steps, None,
            use_fp64=use_fp64, use_block_verification=use_block))
    finally:
        block._state["ids"] = held
    return plan.target, union_tokens(plan.union, sampled, cu), num_sampled


def request_rows(cu_np, reqs):
    starts = cu_np[reqs]
    lengths = cu_np[reqs + 1] - starts
    sub_cu = np.zeros(len(reqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=sub_cu[1:])
    rows = np.repeat(starts - sub_cu[:-1], lengths) + np.arange(int(sub_cu[-1]), dtype=np.int64)
    return rows, sub_cu


def verify_draft(rs, logits, draft_logits, draft_sampled, pos, cu, idx, idx_np, ex_idx, ex_pos, rejection_sample):
    sampler = rs.sampler
    states = sampler.sampling_states
    steps = rs.num_speculative_steps
    fit = fits(sampler, idx_np)
    if not fit.any():
        refuse("top_k")
        return None
    tables = (states.temperature.gpu, states.top_k.gpu, states.top_p.gpu, states.seeds.gpu)
    use_fp64, use_block = sampler.use_fp64_gumbel, rs.use_block_verification
    rows = logits.shape[0]
    if fit.all():
        out = sample_draft(rejection_sample, logits, draft_logits, draft_sampled, cu, pos, idx, ex_idx, ex_pos,
                           *tables, steps, use_fp64, use_block)
        counted(rows, len(idx_np), 0, 0)
        return out
    cu_np = getattr(rs, CU_NP, None)
    if cu_np is None or len(cu_np) != len(idx_np) + 1 or int(cu_np[-1]) != rows:
        refuse("layout")
        return None
    cu_np = np.asarray(cu_np, dtype=np.int64)
    top_reqs, stock_reqs = np.flatnonzero(fit), np.flatnonzero(~fit)
    top_rows, top_cu = request_rows(cu_np, top_reqs)
    stock_rows, stock_cu = request_rows(cu_np, stock_reqs)
    device = logits.device

    def put(array, dtype):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype, non_blocking=True)

    a_rows, b_rows = put(top_rows, torch.int64), put(stock_rows, torch.int64)
    a_reqs, b_reqs = put(top_reqs, torch.int64), put(stock_reqs, torch.int64)
    values, columns = pick(logits)
    _, a_sampled, a_num = sample_draft(
        rejection_sample, logits, draft_logits, draft_sampled[a_rows], put(top_cu, cu.dtype), pos[a_rows],
        idx[a_reqs], ex_idx[a_rows], ex_pos[a_rows], *tables, steps, use_fp64, use_block,
        picked=(values[a_rows], columns[a_rows]))
    b_logits, b_ex_idx, b_ex_pos = logits[b_rows], ex_idx[b_rows], ex_pos[b_rows]
    b_drafts, b_pos, b_idx, b_cu = draft_sampled[b_rows], pos[b_rows], idx[b_reqs], put(stock_cu, cu.dtype)
    processed = sampler.apply_sampling_params(b_logits, b_ex_idx, b_idx, idx_np[stock_reqs], b_pos, b_drafts,
                                              b_ex_pos)
    b_sampled, b_num = rejection_sample(processed, draft_logits, b_drafts, b_cu, b_pos, b_idx, b_ex_idx, b_ex_pos,
                                        tables[0], tables[3], steps, None, use_fp64=use_fp64,
                                        use_block_verification=use_block)
    sampled = a_sampled.new_empty((len(idx_np), a_sampled.shape[1]))
    sampled.index_copy_(0, a_reqs, a_sampled)
    sampled.index_copy_(0, b_reqs, b_sampled.to(sampled.dtype))
    num_sampled = a_num.new_empty(len(idx_np))
    num_sampled.index_copy_(0, a_reqs, a_num)
    num_sampled.index_copy_(0, b_reqs, b_num.to(num_sampled.dtype))
    _rows["split_steps"] += 1
    counted(len(top_rows), len(top_reqs), len(stock_rows), len(stock_reqs))
    return processed, sampled, num_sampled


def counted(top_rows, top_reqs, stock_rows, stock_reqs):
    _rows["top_rows"] += top_rows
    _rows["top_reqs"] += top_reqs
    _rows["stock_rows"] += stock_rows
    _rows["stock_reqs"] += stock_reqs
    say("draft", f"first step with sampled drafts: {top_rows} logits rows of {top_reqs} requests verified over the "
                 f"target's top {K} and the drafter's own top ids in one compact row, {stock_rows} rows of "
                 f"{stock_reqs} requests the stock way over the whole vocabulary")
    took("taken")


def verify(rs, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
           expanded_idx_mapping, expanded_local_pos):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    reason = eager_refusal(rs, logits, draft_logits, idx_mapping_np)
    if reason is not None:
        if reason != "off":
            refuse(reason)
        return None
    if draft_logits is not None:
        return verify_draft(rs, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                            expanded_idx_mapping, expanded_local_pos, rejection_sample)
    states = rs.sampler.sampling_states
    out = sample(rejection_sample, logits, draft_sampled, cu_num_logits, pos, idx_mapping, expanded_idx_mapping,
                 expanded_local_pos, states.temperature.gpu, states.top_k.gpu, states.top_p.gpu, states.seeds.gpu,
                 rs.num_speculative_steps, rs.sampler.use_fp64_gumbel)
    say("eager", f"first eager step: {logits.shape[0]} logits rows of width {logits.shape[1]} sampled over the top "
                 f"{K} candidates and one -inf column")
    took("taken")
    return out
