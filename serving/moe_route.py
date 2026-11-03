import atexit
import json
import os
import re
import time

import torch

FILE_ENV = "TAAF_VLLM_MOE_ROUTE_FILE"
READ_EVERY_S = 1.0
STOCK_PREFERENCE = 1e-5
FLUSH_EVERY_S = 3.0
MODES = ("off", "log", "share")
OFF = {"mode": "off", "tag": "", "log": "", "k_keep": 0, "ratio": 1.0, "cap": 0,
       "max_tokens": 256, "virtual_lanes": [], "virtual_steps": 4, "virtual_tokens": 0,
       "simulate": [], "topp": 0.0, "group": 0, "prefill_topk": 0, "mtp_topk": 0}
SHAPING = ("topp", "group", "prefill_topk", "mtp_topk")
MAIN_LAYERS = 48
LAYER = re.compile(r"layers\.(\d+)\.")
COLUMNS = ("layer", "tokens", "forwards", "stock_sum", "stock_sumsq", "stock_min", "stock_max",
           "routed_sum", "routed_sumsq", "routed_min", "routed_max", "moved_slots")
VIRTUAL_COLUMNS = ("layer", "lanes", "steps", "simulation", "k_keep", "ratio", "cap", "batches",
                   "stock_sum", "routed_sum", "moved_slots")
ROUTE_RANGE = "taaf_moe_route"
EXPERTS_RANGE = "taaf_routed_experts"

_held = {"read_at": float("-inf"), "flushed_at": time.monotonic(), "raw": None,
         "settings": dict(OFF), "said": set()}
_pending: dict = {}
_virtual: dict = {}
_window: dict = {}
_rows: dict = {}


def say(key, text) -> None:
    if key in _held["said"]:
        return
    _held["said"].add(key)
    print(f"MOE_ROUTE {text}", flush=True)


def parsed(raw: str, path: str = "") -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("the route file is not a JSON object")
    mode = str(data.get("mode") or "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    out = {"mode": mode, "tag": str(data.get("tag") or ""),
           "log": str(data.get("log") or "") or f"{path or 'moe_route.json'}.log.jsonl",
           "k_keep": int(data.get("k_keep", OFF["k_keep"])),
           "ratio": float(data.get("ratio", OFF["ratio"])),
           "cap": int(data.get("cap", OFF["cap"])),
           "max_tokens": int(data.get("max_tokens", OFF["max_tokens"])),
           "virtual_lanes": sorted({int(lane) for lane in data.get("virtual_lanes") or []}),
           "virtual_steps": int(data.get("virtual_steps", OFF["virtual_steps"])),
           "virtual_tokens": int(data.get("virtual_tokens", OFF["virtual_tokens"])),
           "simulate": [[int(k_keep), float(ratio), int(cap)]
                        for k_keep, ratio, cap in data.get("simulate") or []],
           "topp": float(data.get("topp", OFF["topp"])),
           "group": int(data.get("group", OFF["group"])),
           "prefill_topk": int(data.get("prefill_topk", OFF["prefill_topk"])),
           "mtp_topk": int(data.get("mtp_topk", OFF["mtp_topk"]))}
    if not 0.0 <= out["topp"] <= 1.0:
        raise ValueError(f"topp {out['topp']} is outside [0, 1]")
    if min(out["group"], out["prefill_topk"], out["mtp_topk"]) < 0:
        raise ValueError("group, prefill_topk and mtp_topk must not be negative")
    if not 0.0 <= out["ratio"] <= 1.0:
        raise ValueError(f"ratio {out['ratio']} is outside [0, 1]")
    if min(out["k_keep"], out["cap"], out["max_tokens"]) < 0:
        raise ValueError("k_keep, cap and max_tokens must not be negative")
    if any(not 0.0 <= ratio <= 1.0 or min(k_keep, cap) < 0
           for k_keep, ratio, cap in out["simulate"]):
        raise ValueError("every simulated share needs k_keep and cap >= 0 and a ratio in [0, 1]")
    if out["virtual_lanes"] and (out["virtual_steps"] < 1 or min(out["virtual_lanes"]) < 1
                                 or out["virtual_tokens"] < max(out["virtual_lanes"])):
        raise ValueError("virtual lanes need virtual_steps >= 1 and virtual_tokens covering the "
                         "widest lane count")
    return out


def describe(held: dict) -> str:
    virtual = (f", virtual batches of {held['virtual_lanes']} lanes x {held['virtual_steps']} "
               f"{held['virtual_tokens']}-token forwards simulating {held['simulate']}"
               if held["virtual_lanes"] else "")
    shaping = "".join(f" {key}={held[key]}" for key in SHAPING if held[key])
    if held["mode"] == "share":
        return (f"mode=share k_keep={held['k_keep']} ratio={held['ratio']} cap={held['cap']} "
                f"max_tokens={held['max_tokens']}{shaping} tag={held['tag']!r}{virtual}")
    return f"mode={held['mode']}{shaping} tag={held['tag']!r}{virtual}"


def shaping(held: dict) -> bool:
    return (0.0 < held["topp"] < 1.0 or held["group"] > 1 or held["prefill_topk"] > 0
            or held["mtp_topk"] > 0)


def main_layer(name) -> bool:
    found = LAYER.search(str(name))
    return bool(found) and int(found.group(1)) < MAIN_LAYERS


def find_rows(device):
    import gc

    found = []
    for held in gc.get_objects():
        kind = type(held)
        if kind.__name__ != "GPUModelRunner" or "worker.gpu" not in kind.__module__:
            continue
        starts = getattr(getattr(held, "input_buffers", None), "query_start_loc", None)
        decode_len = getattr(held, "decode_query_len", None)
        if (isinstance(starts, torch.Tensor) and starts.dim() == 1 and starts.dtype == torch.int32
                and starts.device == torch.device(device) and isinstance(decode_len, int)
                and decode_len >= 1):
            found.append((starts, decode_len))
    if len(found) != 1:
        return None, f"{len(found)} model runners with a request-start buffer were found, not one"
    return found[0], ""


def rows_for(device):
    if device in _rows:
        return _rows[device]
    held, why = find_rows(device)
    _rows[device] = held
    if held is None:
        print(f"MOE_ROUTE row boundaries unknown ({why}): topp, group and prefill_topk stay off "
              f"on every row", flush=True)
    else:
        print(f"MOE_ROUTE rows from the model runner's request starts ({held[1]} decode rows a "
              f"request): topp and group act on decode rows, prefill_topk on prefill rows",
              flush=True)
    return held


def ordered(values: torch.Tensor) -> torch.Tensor:
    return torch.sort(values, dim=-1, descending=True, stable=True).indices


def row_kinds(rows, tokens: int, device):
    index = torch.arange(tokens, device=device)
    none = torch.zeros(tokens, dtype=torch.bool, device=device)
    if rows is None:
        return none, none, index * 0, index * 0, 1
    starts, decode_len = rows
    starts = starts.to(device).long()
    ends = starts[1:]
    request = (ends.unsqueeze(0) <= index.unsqueeze(1)).sum(dim=1)
    real = request < ends.numel()
    safe = request.clamp(max=ends.numel() - 1)
    start = torch.where(real, starts[safe], 0)
    end = torch.where(real, starts[safe + 1], 0)
    length = end - start
    return (real & (length >= 1) & (length <= decode_len), real & (length > decode_len), start,
            end, int(decode_len))


def trim(out_ids, out_w, mass_on, mass: float, keep_top, renormalize: bool):
    tokens, width = out_ids.shape
    order = ordered(out_w)
    rank = torch.empty_like(order).scatter_(
        1, order, torch.arange(width, device=out_ids.device).expand(tokens, width).contiguous())
    ahead = (rank.unsqueeze(1) < rank.unsqueeze(2))
    before = (out_w.unsqueeze(1) * ahead).sum(dim=2)
    total = out_w.sum(dim=1, keepdim=True)
    by_mass = (before < mass * total) | (rank == 0)
    by_top = rank.float() < keep_top.unsqueeze(1)
    keep = torch.where(mass_on.unsqueeze(1), by_mass, by_top)
    count = keep.sum(dim=1)
    trimmed = (count < width).unsqueeze(1)
    top1 = out_ids.gather(1, order[:, :1])
    kept_total = (out_w * keep).sum(dim=1, keepdim=True)
    scaled = out_w / kept_total if renormalize else out_w * (total / kept_total)
    new_ids = torch.where(trimmed, torch.where(keep, out_ids, top1), out_ids)
    new_w = torch.where(trimmed, torch.where(keep, scaled, torch.zeros_like(scaled)), out_w)
    return new_ids, new_w, count


def pipeline(logits, weights, ids, held: dict, rows=None, draft: bool = False,
             renormalize: bool = True, batch: bool = True):
    tokens, width = int(ids.shape[0]), int(ids.shape[1])
    experts = int(logits.shape[-1])
    device = ids.device
    ids_long = ids.long()
    weights_f = weights.float()
    is_decode, is_prefill, start, end, decode_len = row_kinds(None if draft else rows, tokens,
                                                              device)
    live = ((ids_long >= 0) & (ids_long < experts) & torch.isfinite(weights_f)).all(dim=1)
    if rows is not None and not draft:
        live &= is_decode | is_prefill
    is_decode &= live
    is_prefill &= live
    batch_on = batch and held["mode"] == "share" and held["cap"] > 0
    size = int(min(max(held["group"], 1), decode_len))
    index = torch.arange(tokens, device=device)
    first = start + torch.div(index - start, size, rounding_mode="floor") * size
    last = torch.minimum(end, first + size)
    group_on = is_decode & (size >= 2) & (last - first >= 2)
    out_ids, out_w = ids_long, weights_f
    if batch_on or bool(group_on.any()):
        ratio = float(held["ratio"])
        probs = torch.softmax(logits.float(), dim=-1)
        keep_n = max(0, min(int(held["k_keep"]), width))
        kept = torch.zeros(tokens, width, dtype=torch.bool, device=device)
        kept.scatter_(1, ordered(weights_f)[:, :keep_n], True)
        demand = torch.zeros(experts, device=device).index_add_(0, ids_long[live].reshape(-1),
                                                                 weights_f[live].reshape(-1))
        core = torch.zeros(experts, device=device).index_fill_(
            0, ids_long[kept & live.unsqueeze(1)], 1.0)
        score = demand + core * (demand.sum() + 1.0)
        inside = torch.zeros(experts, dtype=torch.bool, device=device)
        inside[ordered(score)[:max(width, min(int(held["cap"]), experts))]] = True
        inside &= demand > 0
        owner = first.unsqueeze(1).expand(tokens, width)[group_on]
        demand_g = torch.zeros(tokens, experts, device=device).index_put_(
            (owner.reshape(-1), ids_long[group_on].reshape(-1)), weights_f[group_on].reshape(-1),
            accumulate=True)
        core_g = torch.zeros(tokens, experts, device=device).index_put_(
            (owner.reshape(-1), ids_long[group_on].reshape(-1)),
            kept[group_on].reshape(-1).float(), accumulate=True) > 0
        score_g = demand_g + torch.where(core_g, demand_g.sum(dim=1, keepdim=True) + 1.0,
                                         torch.zeros((), device=device))
        score_g = torch.where(demand_g > 0, score_g, torch.full((), -1.0, device=device))
        chosen_g = ordered(score_g)[:, :width]
        inside_g = torch.zeros(tokens, experts, dtype=torch.bool, device=device).scatter_(
            1, chosen_g, demand_g.gather(1, chosen_g) > 0)
        inside_row = inside_g[first.clamp(0, tokens - 1)] & group_on.unsqueeze(1)
        one = torch.ones((), device=device)
        factor = torch.where(~inside, torch.full((), ratio if batch_on else 1.0, device=device),
                             one).unsqueeze(0)
        factor = factor * torch.where(~inside_row & group_on.unsqueeze(1),
                                      torch.full((), ratio, device=device), one)
        adjusted = probs * factor
        safe_ids = torch.where(live.unsqueeze(1), ids_long, 0)
        stock_col = torch.zeros_like(probs, dtype=torch.bool).scatter(1, safe_ids, True)
        adjusted = torch.where(stock_col, adjusted * (1.0 + STOCK_PREFERENCE), adjusted)
        kept_col = torch.zeros_like(stock_col).scatter(1, safe_ids, kept)
        adjusted = torch.where(kept_col, torch.full((), -1.0, device=device), adjusted)
        candidates = ordered(adjusted)[:, :width]
        free_rank = (torch.cumsum((~kept).long(), dim=1) - 1).clamp(min=0)
        picked = torch.where(kept, ids_long, candidates.gather(1, free_rank))
        subset = (picked.unsqueeze(-1) == ids_long.unsqueeze(-2)).any(-1).all(-1)
        use_new = (group_on | batch_on) & ~subset & live
        fresh = probs.gather(1, picked)
        if renormalize:
            fresh = fresh / fresh.sum(dim=-1, keepdim=True)
        out_ids = torch.where(use_new.unsqueeze(1), picked, ids_long)
        out_w = torch.where(use_new.unsqueeze(1), fresh, weights_f)
    full = torch.full((tokens,), float(width), device=device)
    if draft:
        mass_on = torch.zeros(tokens, dtype=torch.bool, device=device)
        keep_top = torch.where(live, float(held["mtp_topk"]), full) \
            if held["mtp_topk"] > 0 else full
    else:
        mass_on = is_decode & (0.0 < held["topp"] < 1.0)
        keep_top = torch.where(is_prefill & (held["prefill_topk"] > 0),
                               torch.full((), float(held["prefill_topk"]), device=device), full)
    out_ids, out_w, count = trim(out_ids, out_w, mass_on, float(held["topp"]), keep_top,
                                 renormalize)
    counts = {"topp": count[mass_on],
              "prefill": count[is_prefill] if 0 < held["prefill_topk"] < width and not draft
              else count[:0],
              "mtp": count[live] if draft and 0 < held["mtp_topk"] < width else count[:0]}
    return out_w.to(weights.dtype), out_ids.to(ids.dtype), counts


def experts_range():
    return torch.profiler.record_function(EXPERTS_RANGE)


def settings(now: float | None = None) -> dict:
    now = time.monotonic() if now is None else now
    if now - _held["read_at"] < READ_EVERY_S:
        return _held["settings"]
    _held["read_at"] = now
    path = str(os.environ.get(FILE_ENV) or "")
    try:
        raw = open(path, encoding="utf-8").read() if path else ""
    except OSError:
        raw = ""
    if raw == _held["raw"]:
        return _held["settings"]
    _held["raw"] = raw
    try:
        fresh = parsed(raw, path) if raw.strip() else dict(OFF)
    except (ValueError, TypeError) as failure:
        print(f"MOE_ROUTE the route file {path} is unreadable ({failure}); routing stays stock",
              flush=True)
        fresh = dict(OFF)
    if fresh != _held["settings"]:
        flush(force=True)
        _window.clear()
        _held["settings"] = fresh
        print(f"MOE_ROUTE now {describe(fresh)}, read from {path}", flush=True)
    return _held["settings"]


def capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def unshareable(router, experts: int) -> str:
    scoring = str(getattr(router, "scoring_func", "softmax"))
    if scoring != "softmax":
        return f"the router scores with {scoring}, and the re-pick compares softmax scores"
    held = int(getattr(router, "global_num_experts", experts) or experts)
    if held != experts:
        return f"the router logits have {experts} columns while the router holds {held} experts"
    return ""


def batch_mask(ids: torch.Tensor, weights: torch.Tensor, kept: torch.Tensor, experts: int,
               cap: int) -> torch.Tensor:
    demand = torch.zeros(experts, dtype=torch.float32, device=ids.device).index_add_(
        0, ids.reshape(-1).long(), weights.reshape(-1).float())
    core = torch.zeros(experts, dtype=torch.bool, device=ids.device)
    if kept.shape[1]:
        core = core.index_fill(0, kept.reshape(-1), True)
    width = max(int(ids.shape[1]), min(int(cap), experts))
    score = demand + core.float() * (demand.sum() + 1.0)
    chosen = torch.topk(score, width).indices
    inside = torch.zeros(experts, dtype=torch.bool, device=ids.device).index_fill(0, chosen, True)
    return inside & (demand > 0)


def share(logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, k_keep: int,
          ratio: float, cap: int, renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, width = int(ids.shape[0]), int(ids.shape[1])
    experts = int(logits.shape[-1])
    keep = max(0, min(int(k_keep), width))
    if keep >= width or tokens == 0:
        return weights, ids
    probs = torch.softmax(logits.float(), dim=-1)
    order = torch.argsort(weights, dim=-1, descending=True, stable=True)
    kept = ids.long().gather(1, order)[:, :keep]
    inside = batch_mask(ids, weights, kept, experts, cap)
    adjusted = torch.where(inside.unsqueeze(0), probs, probs * float(ratio))
    stock = torch.zeros_like(probs, dtype=torch.bool).scatter(1, ids.long(), True)
    adjusted = torch.where(stock, adjusted * (1.0 + STOCK_PREFERENCE), adjusted)
    if keep:
        adjusted = adjusted.scatter(1, kept, -1.0)
    rest = torch.topk(adjusted, width - keep, dim=-1).indices
    picked = torch.cat([kept, rest], dim=1)
    fresh = probs.gather(1, picked)
    if renormalize:
        fresh = fresh / fresh.sum(dim=-1, keepdim=True)
    same = (picked.unsqueeze(-1) == ids.long().unsqueeze(-2)).any(-1).all(-1, keepdim=True)
    return (torch.where(same, weights, fresh.to(weights.dtype)),
            torch.where(same, ids, picked.to(ids.dtype)))


def distinct(ids: torch.Tensor, experts: int) -> torch.Tensor:
    flat = ids.reshape(-1).long().clamp(0, experts - 1)
    return torch.zeros(experts, dtype=torch.int64, device=ids.device).index_fill(0, flat, 1).sum()


def note(layer: str, before: torch.Tensor, after: torch.Tensor, experts: int) -> None:
    stock = distinct(before, experts)
    if after is before:
        routed, moved = stock, torch.zeros_like(stock)
    else:
        routed = distinct(after, experts)
        moved = (~(after.unsqueeze(-1) == before.unsqueeze(-2)).any(-1)).sum().to(torch.int64)
    _pending.setdefault((str(layer), int(before.shape[0])), []).append(
        torch.stack([stock, routed, moved]))


def window(layer: str, held: dict, logits: torch.Tensor, weights: torch.Tensor,
           ids: torch.Tensor, renormalize: bool) -> None:
    if int(ids.shape[0]) != held["virtual_tokens"]:
        _window.pop(layer, None)
        return
    widest = max(held["virtual_lanes"])
    batch = _window.setdefault(layer, [])
    batch.append((logits[:widest], weights[:widest], ids[:widest]))
    if len(batch) < held["virtual_steps"]:
        return
    _window[layer] = []
    experts = int(logits.shape[-1])
    for lanes in held["virtual_lanes"]:
        joined = [torch.cat([part[index][:lanes] for part in batch]) for index in range(3)]
        stock = distinct(joined[2], experts)
        if not held["simulate"]:
            _virtual.setdefault((layer, lanes, -1), []).append(
                torch.stack([stock, stock, torch.zeros_like(stock)]))
        for index, (k_keep, ratio, cap) in enumerate(held["simulate"]):
            _, picked = share(joined[0], joined[1], joined[2], k_keep, ratio, cap, renormalize)
            moved = (~(picked.unsqueeze(-1) == joined[2].unsqueeze(-2)).any(-1)).sum()
            _virtual.setdefault((layer, lanes, index), []).append(
                torch.stack([stock, distinct(picked, experts), moved.to(torch.int64)]))


def virtual_rows(held: dict) -> list:
    rows = []
    for (layer, lanes, index), parts in _virtual.items():
        part = torch.stack(parts).cpu().double()
        k_keep, ratio, cap = held["simulate"][index] if index >= 0 else (0, 1.0, 0)
        rows.append([layer, lanes, held["virtual_steps"], index, k_keep, ratio, cap, len(parts),
                     float(part[:, 0].sum()), float(part[:, 1].sum()), int(part[:, 2].sum())])
    _virtual.clear()
    return rows


def flush(force: bool = False) -> None:
    now = time.monotonic()
    if not _pending and not _virtual:
        _held["flushed_at"] = now
        return
    if not force and now - _held["flushed_at"] < FLUSH_EVERY_S:
        return
    held = _held["settings"]
    keys = list(_pending)
    counts = [len(_pending[key]) for key in keys]
    stacked = (torch.stack([row for key in keys for row in _pending[key]]).cpu() if keys
               else torch.zeros(0, 3, dtype=torch.int64))
    _pending.clear()
    virtual = virtual_rows(held)
    _held["flushed_at"] = now
    rows, start = [], 0
    for (layer, tokens), count in zip(keys, counts):
        part = stacked[start:start + count].double()
        start += count
        stock, routed = part[:, 0], part[:, 1]
        rows.append([layer, tokens, count,
                     float(stock.sum()), float((stock * stock).sum()),
                     int(stock.min()), int(stock.max()),
                     float(routed.sum()), float((routed * routed).sum()),
                     int(routed.min()), int(routed.max()), int(part[:, 2].sum())])
    line = {"t": round(time.time(), 3), "pid": os.getpid(), "tag": held["tag"],
            "mode": held["mode"], "k_keep": held["k_keep"], "ratio": held["ratio"],
            "cap": held["cap"], "max_tokens": held["max_tokens"],
            **{key: held[key] for key in SHAPING}, "columns": list(COLUMNS),
            "rows": rows, "virtual_columns": list(VIRTUAL_COLUMNS), "virtual": virtual}
    path = held["log"]
    try:
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError as failure:
        print(f"MOE_ROUTE could not append to {path}: {failure}", flush=True)
        return
    print(f"MOE_ROUTE wrote {sum(counts)} layer-forward record(s) over {len(rows)} "
          f"(layer, tokens) row(s) and {sum(row[7] for row in virtual)} virtual batch "
          f"record(s) of {describe(held)} to {path}", flush=True)


def route(layer, router, logits, weights, ids):
    with torch.profiler.record_function(ROUTE_RANGE):
        return routed(layer, router, logits, weights, ids)


def routed(layer, router, logits, weights, ids):
    if capturing():
        say("capture", "a CUDA graph capture took the stock routing; its replays keep it "
                       "whatever the route file says, so only eager forwards are logged or re-picked")
        return weights, ids
    held = settings()
    reshaped = shaping(held)
    if (held["mode"] == "off" and not reshaped) or ids.dim() != 2 or int(ids.shape[0]) == 0:
        return weights, ids
    experts = int(logits.shape[-1])
    out_weights, out_ids = weights, ids
    if held["virtual_lanes"] and not unshareable(router, experts):
        window(str(layer), held, logits, weights, ids, bool(getattr(router, "renormalize", True)))
        say(("virtual", held["tag"]),
            f"virtual batches of {held['virtual_lanes']} lanes x {held['virtual_steps']} "
            f"consecutive {held['virtual_tokens']}-token forwards first collected on {layer}")
    if reshaped:
        why = unshareable(router, experts)
        if why:
            say(("unshareable", why), f"shaping skipped on {layer}: {why}")
        else:
            draft = not main_layer(layer)
            rows = None if draft else rows_for(ids.device)
            out_weights, out_ids, _ = pipeline(
                logits, weights, ids, held, rows, draft, bool(getattr(router, "renormalize", True)),
                batch=int(ids.shape[0]) <= held["max_tokens"])
            say(("shaping", held["tag"], draft),
                f"{'draft' if draft else 'main'} layer {layer} first reshaped: "
                + " ".join(f"{key}={held[key]}" for key in SHAPING if held[key]))
    elif held["mode"] == "share" and int(ids.shape[0]) <= held["max_tokens"]:
        why = unshareable(router, experts)
        if why:
            say(("unshareable", why), f"share skipped on {layer}: {why}")
        else:
            out_weights, out_ids = share(logits, weights, ids, held["k_keep"], held["ratio"],
                                         held["cap"], bool(getattr(router, "renormalize", True)))
            say(("share", held["tag"]),
                f"share first re-picked on {layer}: {int(ids.shape[0])} tokens, top "
                f"{held['k_keep']} of {int(ids.shape[1])} kept, the rest from a batch set of at "
                f"most {held['cap']} of {experts} experts when within {held['ratio']} of the "
                f"original score")
    note(layer, ids, out_ids, experts)
    flush()
    return out_weights, out_ids


atexit.register(flush, True)
