from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mtp_ref as ref  # noqa: E402
import mtp_tune_worker as base  # noqa: E402

FULL = "full"
DRAFT_CHUNK = 512
TARGET_CHUNK = 1024
SAMPLE_EXPERTS = 16
PROGRESS_EVERY = 20
MARGIN_S = 240.0
PACK_TAILS = (".weight_packed", ".weight_scale", ".weight_shape")
OWN_HEAD = ("lm_head.weight",)
OWN_EMBED = ("embed_tokens.weight",)
C512_FILE = "mtp_c512.safetensors"
D_FILE = "mtp_tuned.safetensors"


def say(text: str) -> None:
    print(f"DRAFT_AB {time.strftime('%H:%M:%S')} {text}", flush=True)


@dataclass
class Source:
    name: str
    dense: dict[str, torch.Tensor]
    raw: dict[str, torch.Tensor]
    gate_up: torch.Tensor
    down: torch.Tensor
    note: str
    lm_head: torch.Tensor | None = None
    embed: torch.Tensor | None = None


@dataclass
class Drafter:
    label: str
    dense: dict[str, torch.Tensor]
    experts: str
    allowed: torch.Tensor | None
    how: str
    servable: bool = True


def dense_raw(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items()
            if not ref.EXPERT_SPLIT.match(name) and not ref.EXPERT_FUSED.match(name)}


def source_of(name: str, tensors: dict[str, torch.Tensor], shape: ref.Shape, note: str) -> Source:
    head, report = ref.build_head(dict(tensors), shape)
    return Source(name, dict(head.dense_state()), dense_raw(tensors), head.gate_up.detach(), head.down.detach(),
                  f"{note}; {report['dense']} dense, {report['experts']} experts ({report['experts_layout']})")


def load_intel(model_dir: Path, config: dict[str, Any]) -> Source:
    inventory = ref.mtp_inventory(model_dir)
    if not inventory:
        raise RuntimeError(f"{model_dir} carries no mtp.* tensors, so the checkpoint ships no MTP head")
    kinds = sorted({spec["dtype"] for spec in inventory.values()})
    tensors = ref.read_tensors(model_dir, sorted(inventory), inventory, "cpu")
    return source_of("intel", tensors, ref.Shape.from_config(config),
                     f"{len(inventory)} mtp.* tensors of {model_dir.name}, dtypes {kinds}")


def int4_group(config: dict[str, Any]) -> int:
    held = config.get("quantization_config") or {}
    if held.get("quant_method") != "compressed-tensors" or held.get("format") != "pack-quantized":
        raise RuntimeError(f"the drafter's quantization is {held.get('quant_method')}/{held.get('format')}, "
                           f"not compressed-tensors pack-quantized")
    groups = [one.get("weights") or {} for one in (held.get("config_groups") or {}).values()]
    if len(groups) != 1:
        raise RuntimeError(f"the drafter names {len(groups)} quantization groups, not one")
    weights = groups[0]
    wanted = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group"}
    wrong = {key: weights.get(key) for key, value in wanted.items() if weights.get(key) != value}
    if wrong or not int(weights.get("group_size") or 0):
        raise RuntimeError(f"the drafter's experts are not symmetric int4 by groups: {wrong or weights}")
    return int(weights["group_size"])


def unpacked(raw: dict[str, torch.Tensor], group: int) -> tuple[dict[str, torch.Tensor], int]:
    out: dict[str, torch.Tensor] = {}
    count = 0
    for name, value in raw.items():
        if name.endswith(".weight_packed"):
            stem = name[:-len(".weight_packed")]
            for tail in PACK_TAILS[1:]:
                if stem + tail not in raw:
                    raise RuntimeError(f"{name} has no {stem + tail} beside it")
            out[stem + ".weight"] = ref.unpack_int4(value, raw[stem + ".weight_scale"],
                                                    raw[stem + ".weight_shape"].tolist(), group)
            count += 1
        elif name.endswith((".weight_zero_point", ".weight_g_idx")):
            raise RuntimeError(f"{name}: an asymmetric or reordered int4 pack is not read here")
        elif not name.endswith(PACK_TAILS):
            out[name] = value
    return out, count


def own_tensor(folder: Path, tails: tuple[str, ...]) -> torch.Tensor | None:
    index = folder / "model.safetensors.index.json"
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    names = [name for name in weight_map if name.endswith(tails) and "visual" not in name and "mtp." not in name]
    if len(names) != 1:
        return None
    from safetensors import safe_open
    with safe_open(str(folder / weight_map[names[0]]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(names[0])


def load_albucino(draft_dir: Path, target_shape: ref.Shape) -> Source:
    config = json.loads((draft_dir / "config.json").read_text(encoding="utf-8"))
    group = int4_group(config)
    shape = ref.Shape.from_config(config)
    for field in ("hidden", "hc", "heads", "kv_heads", "head_dim", "experts", "top_k", "inter"):
        if getattr(shape, field) != getattr(target_shape, field):
            raise RuntimeError(f"the drafter's {field} is {getattr(shape, field)}, the target's "
                               f"{getattr(target_shape, field)}")
    inventory = ref.mtp_inventory(draft_dir)
    if not inventory:
        raise RuntimeError(f"{draft_dir} carries no mtp.* tensors")
    raw = ref.read_tensors(draft_dir, sorted(inventory), inventory, "cpu")
    tensors, count = unpacked(raw, group)
    source = source_of("albucino", tensors, shape,
                       f"{len(inventory)} mtp.* tensors of {draft_dir.name}, {count} int4 g{group} expert "
                       f"matrices unpacked to BF16")
    source.lm_head = own_tensor(draft_dir, OWN_HEAD)
    source.embed = own_tensor(draft_dir, OWN_EMBED)
    return source


def compare_sources(one: Source, two: Source) -> dict[str, Any]:
    names = sorted(set(one.raw) & set(two.raw))
    equal = [name for name in names if one.raw[name].shape == two.raw[name].shape
             and one.raw[name].dtype == two.raw[name].dtype and torch.equal(one.raw[name], two.raw[name])]
    chosen = sorted(random.Random(0).sample(range(one.gate_up.shape[0]),
                                            min(SAMPLE_EXPERTS, one.gate_up.shape[0])))
    errors = []
    if one.gate_up.shape == two.gate_up.shape:
        for expert in chosen:
            for left, right in ((one.gate_up, two.gate_up), (one.down, two.down)):
                a, b = left[expert].float(), right[expert].float()
                errors.append(float((a - b).norm() / b.norm().clamp(min=1e-12)))
    errors.sort()
    return {"dense_names": len(names), "dense_bitwise_equal": len(equal),
            "dense_differ": [name for name in names if name not in equal][:6],
            "expert_relative_error_median": round(errors[len(errors) // 2], 4) if errors else None,
            "expert_relative_error_max": round(errors[-1], 4) if errors else None,
            "experts_sampled": len(chosen)}


def kept_rows(saliency: Path, keep: int, experts: int) -> list[int]:
    order = [int(one) for one in json.loads(saliency.read_text(encoding="utf-8"))["mtp"]]
    if sorted(order) != list(range(experts)):
        raise RuntimeError(f"{saliency.name} does not rank the head's {experts} experts")
    return sorted(order[:keep])


def remap_tuned(tuned_file: Path, onto: Source, saliency: Path | None) -> tuple[dict[str, torch.Tensor], list[int],
                                                                                dict[str, Any]]:
    from safetensors.torch import load_file
    tuned = load_file(str(tuned_file))
    dense = {name: value.clone() for name, value in onto.dense.items()}
    rows: list[int] = []
    report: dict[str, Any] = {"file": tuned_file.name,
                              "sha256": hashlib.sha256(tuned_file.read_bytes()).hexdigest()[:16]}
    drift = []
    for name, value in tuned.items():
        if name not in dense:
            raise RuntimeError(f"{tuned_file.name} carries {name}, which the head does not")
        held = dense[name]
        if tuple(value.shape) == tuple(held.shape):
            drift.append(float((value.float() - held).norm() / held.norm().clamp(min=1e-12)))
            dense[name] = value.float()
            continue
        if not name.endswith("mlp.gate.weight") or value.shape[1:] != held.shape[1:] or value.shape[0] > held.shape[0]:
            raise RuntimeError(f"{tuned_file.name}: {name} is {tuple(value.shape)}, the head "
                               f"{tuple(held.shape)}, and only a router may hold fewer rows")
        if saliency is None:
            raise RuntimeError(f"{name} routes {value.shape[0]} of {held.shape[0]} experts and no saliency order "
                               f"names which")
        rows = kept_rows(saliency, int(value.shape[0]), int(held.shape[0]))
        picked = held[rows]
        cosine = F.cosine_similarity(value.float(), picked, dim=-1)
        shifted = F.cosine_similarity(value.float(), held[[rows[(index + 1) % len(rows)]
                                                           for index in range(len(rows))]], dim=-1)
        report["router"] = {"rows": int(value.shape[0]), "of": int(held.shape[0]),
                            "cosine_at_kept_median": round(float(cosine.median()), 4),
                            "cosine_at_kept_min": round(float(cosine.min()), 4),
                            "cosine_shifted_median": round(float(shifted.median()), 4)}
        full = held.clone()
        full[rows] = value.float()
        dense[name] = full
    drift.sort()
    report["tensors"] = len(tuned)
    report["relative_change_median"] = round(drift[len(drift) // 2], 4) if drift else None
    return dense, rows, report


class Installer:
    def __init__(self, head: ref.MTPHead, sources: dict[str, Source], current: str):
        self.head = head
        self.sources = sources
        self.current = current
        self.switches = 0

    def __call__(self, drafter: Drafter) -> None:
        device = self.head.gate_up.device
        base.load_dense(self.head, drafter.dense)
        if self.current != drafter.experts:
            with torch.no_grad():
                self.head.gate_up.copy_(self.sources[drafter.experts].gate_up, non_blocking=True)
                self.head.down.copy_(self.sources[drafter.experts].down, non_blocking=True)
            self.current = drafter.experts
            self.switches += 1
        self.head.allowed = drafter.allowed.to(device) if drafter.allowed is not None else None


class Bench:
    def __init__(self, head: ref.MTPHead, lm_head: torch.Tensor, embed: torch.Tensor,
                 vocabs: dict[str, list[int] | None], cfg: dict[str, Any]):
        self.head = head
        self.lm_head = lm_head
        self.embed = embed
        self.cfg = cfg
        self.device = lm_head.device
        self.vocabs = {name: (torch.tensor(ids, device=self.device) if ids is not None else None)
                       for name, ids in vocabs.items()}
        self.temperatures = [float(cfg["temperature"]), *[float(one) for one in cfg.get("eval_temperatures") or []
                                                          if float(one) != float(cfg["temperature"])]]
        self.drafts = int(cfg["ab_drafts"])
        self.rows: dict[tuple[int, str], torch.Tensor] = {}

    def plan(self, ids: torch.Tensor, start: int, seed: int) -> torch.Tensor:
        count = ids.shape[0]
        bases = torch.arange(max(0, start - 1 - self.drafts), count - 2, device=ids.device)
        limit = int(self.cfg["eval_bases"])
        if bases.numel() > limit:
            picker = torch.Generator().manual_seed(seed)
            keep = torch.randperm(bases.numel(), generator=picker)[:limit].sort().values.to(ids.device)
            bases = bases[keep]
        return bases

    def targets(self, sample: torch.Tensor, at: torch.Tensor) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
        parts: dict[float, list[tuple[torch.Tensor, torch.Tensor]]] = {one: [] for one in self.temperatures}
        for start in range(0, at.numel(), TARGET_CHUNK):
            logits = F.linear(sample[at[start:start + TARGET_CHUNK]], self.lm_head)
            for temperature in self.temperatures:
                parts[temperature].append(ref.game_distribution(logits, temperature, base.GAME["top_k"],
                                                                base.GAME["top_p"]))
        return {temperature: (torch.cat([one[0] for one in held], 0), torch.cat([one[1] for one in held], 0))
                for temperature, held in parts.items()}

    def vocab_rows(self, lm_head: torch.Tensor, name: str) -> torch.Tensor:
        held = (id(lm_head), name)
        if held not in self.rows:
            ids = self.vocabs[name]
            self.rows[held] = lm_head if ids is None else lm_head.index_select(0, ids)
        return self.rows[held]

    def drafted(self, out: torch.Tensor, lm_head: torch.Tensor, name: str) -> torch.Tensor:
        rows = self.vocab_rows(lm_head, name)
        picks = [F.linear(out[one:one + DRAFT_CHUNK].to(rows.dtype), rows).argmax(-1)
                 for one in range(0, out.shape[0], DRAFT_CHUNK)]
        chosen = torch.cat(picks, 0)
        ids = self.vocabs[name]
        return chosen if ids is None else ids[chosen]

    def record(self, ids: torch.Tensor, multi: torch.Tensor, sample: torch.Tensor, start: int,
               drafters: list[Drafter], install, heads: dict[str, tuple[torch.Tensor, torch.Tensor]], seed: int = 0):
        count = ids.shape[0]
        first_target = start - 1
        bases = self.plan(ids, start, seed)
        if bases.numel() == 0:
            return {}
        cfg = self.cfg
        low = max(0, int(bases[0]) - int(cfg["context"]))
        low -= low % self.head.shape.ratio
        reach = int(cfg.get("grad_context") or 0)
        grad_from = max(0, count - 1 - low - reach) if reach else 0
        wanted = (bases[:, None] + torch.arange(1, self.drafts + 1, device=ids.device)[None, :]).clamp(max=count - 2)
        at = wanted.flatten().unique()
        row_of = torch.full((count,), -1, dtype=torch.long, device=ids.device)
        row_of[at] = torch.arange(at.numel(), device=ids.device)
        targets = self.targets(sample, at)
        steps = [bases + step + 1 for step in range(self.drafts)]
        masks = [(target >= first_target) & (target <= count - 2) for target in steps]
        full = masks[0]
        for one in masks[1:]:
            full = full & one
        first = row_of[(bases + 1).clamp(max=count - 2)]
        out: dict[str, dict[tuple[str, float], tuple[list[torch.Tensor], torch.Tensor]]] = {}
        for drafter in drafters:
            install(drafter)
            lm_head, embed = heads.get(drafter.experts, (self.lm_head, self.embed))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
                outs = self.head.forward_steps(multi[low:], ids[low:], embed, bases, self.drafts, first_pos=low,
                                               chunk=int(cfg["attn_chunk"]),
                                               share_index=bool(cfg["share_index"]), grad_from=grad_from)
            views = {}
            for name in self.vocabs:
                picks = [self.drafted(one, lm_head, name) for one in outs]
                for temperature, (probs, tids) in targets.items():
                    rates = []
                    for step, pick in enumerate(picks):
                        row = row_of[steps[step].clamp(max=count - 2)]
                        rates.append((probs[row] * (tids[row] == pick[:, None])).sum(-1)[full])
                    views[(name, temperature)] = (rates, probs[first][:, 0][full])
            out[drafter.label] = views
        return out


def record_seed(record: dict[str, Any]) -> int:
    return int(hashlib.sha256(str(record.get("id") or record.get("stem")).encode("utf-8")).hexdigest()[:8], 16)


def run_pass(name: str, bench: Bench, extractor: base.Extractor, rows: list[dict[str, Any]],
             drafters: list[Drafter], install, heads, main: tuple[str, float]) -> dict[str, Any]:
    tallies: dict[tuple[str, str, float], ref.Tally] = {}
    games: dict[tuple[str, str], ref.Tally] = {}
    counts = {"records": 0, "unread": 0, "no_reply": 0, "out_of_memory": 0}
    began = time.monotonic()
    for index, record in enumerate(rows):
        got = extractor.hidden(record)
        if got is None:
            counts["unread"] += 1
            continue
        ids, multi, sample = got
        start = base.reply_start(extractor.tokenizer, ids, record)
        if start is None:
            counts["no_reply"] += 1
            continue
        try:
            views = bench.record(ids, multi, sample, start, drafters, install, heads, record_seed(record))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            counts["out_of_memory"] += 1
            say(f"{name}: out of GPU memory on a {len(ids):,}-token record; it is left out for every drafter "
                f"({counts['out_of_memory']} so far)")
            continue
        got = ids = multi = sample = None
        if not views:
            continue
        counts["records"] += 1
        for label, held in views.items():
            for (vocab, temperature), (rates, best) in held.items():
                tallies.setdefault((label, vocab, temperature), ref.Tally(bench.drafts)).add(rates, best)
            rates, best = held[main]
            games.setdefault((label, str(record.get("game"))), ref.Tally(bench.drafts)).add(rates, best)
        if counts["records"] == 1 or index % PROGRESS_EVERY == PROGRESS_EVERY - 1 or index + 1 == len(rows):
            say(f"{name} {index + 1}/{len(rows)} ({time.monotonic() - began:.0f} s): " + "; ".join(
                f"{one.label} {tallies[(one.label, *main)].summary().get('tokens_per_step')}"
                for one in drafters if (one.label, *main) in tallies) + f" tokens a step at {bench.drafts} drafts "
                f"({main[0]}, T{main[1]:g})")
    return {"counts": counts, "seconds": round(time.monotonic() - began, 1),
            "views": {f"{label}|{vocab}|{temperature:g}": tally.summary()
                      for (label, vocab, temperature), tally in tallies.items()},
            "games": {f"{label}|{game}": tally.summary() for (label, game), tally in games.items()}}


def view(result: dict[str, Any], label: str, vocab: str, temperature: float) -> dict[str, Any]:
    return result["views"].get(f"{label}|{vocab}|{temperature:g}") or {}


def table(results: list[dict[str, Any]], drafters: dict[str, Drafter], vocabs: list[str],
          temperatures: list[float]) -> list[dict[str, Any]]:
    rows = []
    main_t = temperatures[0]
    for result in results:
        for label in result["labels"]:
            views = {vocab: view(result, label, vocab, main_t) for vocab in vocabs}
            chosen = max((one for one in vocabs if one != FULL and views[one].get("bases")),
                         key=lambda one: views[one]["tokens_per_step"], default=None)
            rows.append({"label": label, "pass": result["name"], "best_vocab": chosen, "views": views,
                         "other_temperatures": {f"{vocab}|{temperature:g}": view(result, label, vocab, temperature)
                                                .get("tokens_per_step")
                                                for vocab in vocabs for temperature in temperatures[1:]},
                         "how": drafters[label].how, "servable": drafters[label].servable,
                         "records": result["counts"]["records"]})
    return rows


def table_line(row: dict[str, Any], drafts: int, temperature: float) -> str:
    chosen = row["best_vocab"]
    held = row["views"].get(chosen) or {}
    positions = " / ".join(f"{one:.3f}" for one in held.get("per_position") or [])
    conditional = " / ".join(f"{one:.3f}" for one in held.get("teacher_forced") or [])
    others = ", ".join(f"{vocab} {(row['views'][vocab] or {}).get('tokens_per_step')}"
                       for vocab in row["views"] if vocab != chosen)
    return (f"{row['label']:<11} {held.get('tokens_per_step')} tokens a step at {drafts} drafts with {chosen} "
            f"(T{temperature:g}); per position {positions} (given the one before: {conditional}); "
            f"{others}; {held.get('bases', 0):,} positions over {row['records']} records; {row['how']}")


def decide(rows: list[dict[str, Any]], served_now: tuple[str, str]) -> dict[str, Any]:
    candidates = [row for row in rows if row["servable"] and row["best_vocab"]]
    if not candidates:
        return {}
    best = max(candidates, key=lambda row: row["views"][row["best_vocab"]]["tokens_per_step"])
    now = next((row for row in rows if row["label"] == served_now[0]), None)
    baseline = (now["views"].get(served_now[1]) or {}).get("tokens_per_step") if now else None
    return {"label": best["label"], "vocab": best["best_vocab"],
            "tokens_per_step": best["views"][best["best_vocab"]]["tokens_per_step"],
            "served_now": f"{served_now[0]} with {served_now[1]}", "served_now_tokens_per_step": baseline,
            "how": best["how"]}


def remaining_s(cfg: dict[str, Any]) -> float | None:
    hard = float(cfg.get("hard_minutes") or 0.0)
    if hard <= 0:
        return None
    return hard * 60.0 - (time.time() - float(cfg.get("driver_started") or time.time()))


def export(head: ref.MTPHead, like: dict[str, torch.Tensor], path: Path, meta: dict[str, str]) -> dict[str, Any]:
    from safetensors.torch import save_file
    tensors = {name: value.cpu() for name, value in ref.export_tensors(head, like).items()}
    save_file(tensors, str(path), metadata=meta)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    say(f"wrote {path.name}: {len(tensors)} tensor(s), {path.stat().st_size / 2**20:.1f} MiB, sha256 {digest[:16]}")
    return {"file": path.name, "tensors": len(tensors), "sha256": digest}


def held_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [one for one in rows if one["split"] == "train"]
    held = [one for one in rows if one["split"] == "heldout"]
    if cfg["eval_records"] and len(held) > int(cfg["eval_records"]):
        held = random.Random(int(cfg["seed"])).sample(held, int(cfg["eval_records"]))
    held.sort(key=lambda one: (one["run"], one["game"], int(one.get("step") or 0), int(one.get("call") or 0)))
    return train, held


def build_drafters(cfg: dict[str, Any], sources: dict[str, Source],
                   refused: dict[str, str]) -> tuple[dict[str, Drafter], dict[str, Any]]:
    wanted = [one.strip() for one in cfg["ab_heads"].split(",") if one.strip()]
    drafters: dict[str, Drafter] = {}
    notes: dict[str, Any] = {}
    if "albucino" in wanted and "albucino" in sources:
        drafters["albucino"] = Drafter("albucino", sources["albucino"].dense, "albucino", None,
                                       "keep TAAF_VLLM_DRAFT_MODEL on the albucino drafter")
    if "intel" in wanted and "intel" in sources:
        drafters["intel"] = Drafter("intel", sources["intel"].dense, "intel", None,
                                    "unset TAAF_VLLM_DRAFT_MODEL: the W4A16 checkpoint's own BF16 MTP layer drafts")
    tuned = [one for one in wanted if one.startswith("tuned")]
    if not tuned:
        return drafters, notes
    onto = sources.get("intel") or sources.get("albucino")
    if onto is None:
        for one in tuned:
            refused[one] = "neither the W4A16 checkpoint's head nor the albucino drafter loaded, so no experts"
        return drafters, notes
    try:
        dense, rows, report = remap_tuned(Path(cfg["tuned_file"]), onto,
                                          Path(cfg["saliency"]) if cfg.get("saliency") else None)
    except Exception as failure:
        for one in tuned:
            refused[one] = f"{type(failure).__name__}: {failure}"
        return drafters, notes
    notes["tuned"] = report
    allowed = None
    if rows:
        allowed = torch.zeros(onto.gate_up.shape[0], dtype=torch.bool)
        allowed[rows] = True
    if "tuned" in wanted:
        if "intel" in sources:
            drafters["tuned"] = Drafter("tuned", dense, "intel", None,
                                        f"unset TAAF_VLLM_DRAFT_MODEL and set TAAF_VLLM_MTP_WEIGHTS={C512_FILE}")
        else:
            refused["tuned"] = "the W4A16 checkpoint's head did not load: " + refused.get("intel", "")
    if "tuned-int4" in wanted:
        if "albucino" in sources:
            drafters["tuned-int4"] = Drafter("tuned-int4", dense, "albucino", None,
                                             f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set "
                                             f"TAAF_VLLM_MTP_WEIGHTS={C512_FILE}")
        else:
            refused["tuned-int4"] = "the albucino drafter did not load: " + refused.get("albucino", "")
    if "tuned-k400" in wanted:
        if allowed is None:
            refused["tuned-k400"] = "the tuned router already routes every expert of the head"
        else:
            drafters["tuned-k400"] = Drafter("tuned-k400", dense, onto.name, allowed,
                                             "not servable by weights alone: the router keeps only the experts "
                                             "it was tuned with", servable=False)
    return drafters, notes


def main() -> int:
    cfg = json.loads(sys.argv[1])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    began = time.monotonic()
    rows = base.records(Path(cfg["corpus"]))
    train, held = held_rows(rows, cfg)
    say(f"corpus: {len(rows)} record(s); held out {len(held)} from games "
        f"{sorted({one['game'] for one in held})} ({sum(one['reply_tokens'] for one in held):,} reply tokens), "
        f"{len(train)} to tune on")
    model_dir = Path(cfg["model_dir"])
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    shape = ref.Shape.from_config(config)
    llm = base.engine(str(model_dir), cfg["serve_args"], cfg)
    found = base.attach(llm)
    device = torch.device(cfg.get("device") or "cuda")
    lm_head = found["lm_head"].detach().to(device)
    embed = found["embed"].detach().to(device)
    if lm_head.dim() != 2 or not lm_head.dtype.is_floating_point or lm_head.shape[1] != shape.hidden:
        raise SystemExit(f"the target's lm_head is {tuple(lm_head.shape)} {lm_head.dtype}, not a float table")
    sources: dict[str, Source] = {}
    refused: dict[str, str] = {}
    loaders = (("intel", lambda: load_intel(model_dir, config)),
               ("albucino", lambda: load_albucino(Path(cfg["draft_model"]), ref.Shape.from_config(config))))
    for name, load in loaders:
        if name == "albucino" and not cfg.get("draft_model"):
            refused[name] = "the serve names no TAAF_VLLM_DRAFT_MODEL"
            continue
        try:
            took = time.monotonic()
            sources[name] = load()
            say(f"source {name}: {sources[name].note}, read in {time.monotonic() - took:.0f} s")
        except Exception as failure:
            refused[name] = f"{type(failure).__name__}: {failure}"
            say(f"source {name} CANNOT LOAD: {refused[name]}")
    report: dict[str, Any] = {"config": {k: v for k, v in cfg.items() if k != "serve_args"}, "refused": refused,
                              "sources": {name: one.note for name, one in sources.items()}}
    heads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if "intel" in sources and "albucino" in sources:
        report["albucino_against_intel"] = compare_sources(sources["albucino"], sources["intel"])
        say(f"albucino against the W4A16 checkpoint's head: {report['albucino_against_intel']}")
    if "albucino" in sources:
        own = sources["albucino"]
        for tag, mine, target in (("lm_head", own.lm_head, lm_head), ("embed", own.embed, embed)):
            same = mine is not None and tuple(mine.shape) == tuple(target.shape) and \
                torch.equal(mine.to(device, target.dtype), target)
            report[f"albucino_{tag}_equals_target"] = same if mine is not None else None
            say(f"albucino's own {tag}: " + ("none shipped, the target's is used" if mine is None else
                                            "bitwise the target's" if same else "DIFFERS from the target's, so "
                                            "albucino drafts with its own"))
        if (own.lm_head is not None and not report["albucino_lm_head_equals_target"]) or \
                (own.embed is not None and not report["albucino_embed_equals_target"]):
            heads["albucino"] = (own.lm_head.to(device, lm_head.dtype) if own.lm_head is not None else lm_head,
                                 own.embed.to(device, embed.dtype) if own.embed is not None else embed)
        own.lm_head = own.embed = None
    drafters, notes = build_drafters(cfg, sources, refused)
    report.update(notes)
    if "tuned" in notes:
        say(f"tuned head {notes['tuned']}")
    for label, why in sorted(refused.items()):
        say(f"{label} CANNOT LOAD on this target: {why}")
    if not drafters:
        (out / "draft_ab_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        raise SystemExit("no drafter could be built")
    first = next(iter(sources.values()))
    head = ref.MTPHead(shape, first.dense, first.gate_up, first.down).to(device)
    if device.type == "cuda":
        for source in sources.values():
            source.gate_up = source.gate_up.pin_memory()
            source.down = source.down.pin_memory()
    install = Installer(head, sources, first.name)
    like = (sources.get("intel") or first).raw
    if "tuned" in drafters or "tuned-int4" in drafters:
        install(drafters.get("tuned") or drafters["tuned-int4"])
        report["export_c512"] = export(head, like, out / C512_FILE,
                                       {"source": str(cfg.get("model_repo") or ""), "router": "512 rows"})
    vocab_files = dict(cfg.get("ab_vocab_files") or {})
    vocabs: dict[str, list[int] | None] = {FULL: None}
    for name, path in vocab_files.items():
        vocabs[name] = json.loads(Path(path).read_text(encoding="utf-8"))["ids"]
    bench = Bench(head, lm_head, embed, vocabs, cfg)
    names = list(vocabs)
    main_vocab = next((one for one in names if one != FULL), FULL)
    main = (main_vocab, bench.temperatures[0])
    extractor = base.Extractor(llm, llm.get_tokenizer())
    order = [label for label in ("albucino", "intel", "tuned", "tuned-k400", "tuned-int4") if label in drafters]
    say(f"pass 1: {len(held)} held-out records, drafters {order}, vocabularies {names}, temperatures "
        f"{bench.temperatures}, {bench.drafts} drafts, at most {cfg['eval_bases']} positions a record")
    first_pass = run_pass("pass 1", bench, extractor, held, [drafters[one] for one in order], install, heads, main)
    first_pass.update({"name": "pass 1", "labels": order})
    results = [first_pass]
    report["passes"] = results
    rows_now = table(results, drafters, names, bench.temperatures)
    report["table"] = rows_now
    (out / "draft_ab_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    minutes = float(cfg.get("ab_tune_minutes") or 0.0)
    left = remaining_s(cfg)
    if minutes > 0 and left is not None:
        afford = (left - MARGIN_S - first_pass["seconds"] * 3 / (1 + len(order))) / 60.0
        if afford < minutes:
            say(f"(d) gets {max(0.0, afford):.1f} of the {minutes:g} tuning minutes asked: {left / 60:.1f} min are "
                f"left before the carried main's hard stop and the second pass needs its share")
            minutes = max(0.0, afford)
    if minutes >= 1.0:
        start = cfg["ab_tune_from"]
        if start == "best":
            servable = [row for row in rows_now if row["servable"] and row["best_vocab"]]
            start = max(servable, key=lambda row: row["views"][row["best_vocab"]]["tokens_per_step"])["label"] \
                if servable else ""
        if start not in drafters or not drafters[start].servable:
            say(f"(d) skipped: no servable head to tune from ({start or 'none'})")
        else:
            install(drafters[start])
            tokenizer = llm.get_tokenizer()
            header = tokenizer.encode(base.ASSISTANT_HEADER, add_special_tokens=False) \
                if cfg.get("span") == "assistant" else []
            end_id = int(tokenizer.convert_tokens_to_ids(base.END_TOKEN)) if header else -1
            train_vocab = json.loads(Path(cfg["draft_vocab"]).read_text(encoding="utf-8"))["ids"] \
                if cfg.get("draft_vocab") else None
            trainer = base.Trainer(head, {"embed": embed, "lm_head": lm_head}, train_vocab, cfg, None, header, end_id)
            say(f"(d): tuning {start} for {minutes:.1f} min on {len(train)} same-weights records, targets "
                f"{trainer.target} (KL {trainer.kl:g}) at T{trainer.temperature:g}, draft vocabulary "
                f"{Path(cfg['draft_vocab']).name if cfg.get('draft_vocab') else 'full'}")
            rounds = base.train_rounds(cfg, head, trainer, extractor, train, minutes)
            tuned = {name: value.detach().clone() for name, value in head.dense_state().items()}
            del trainer
            torch.cuda.empty_cache()
            experts = drafters[start].experts
            how = (f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set TAAF_VLLM_MTP_WEIGHTS={D_FILE}"
                   if experts == "albucino" else f"unset TAAF_VLLM_DRAFT_MODEL and set TAAF_VLLM_MTP_WEIGHTS={D_FILE}")
            drafters["d"] = Drafter("d", tuned, experts, None, f"{how} (tuned from {start})")
            report["tune"] = {"from": start, **rounds,
                              "export": export(head, like, out / D_FILE,
                                               {"source": str(cfg.get("model_repo") or ""), "from": start,
                                                "trained_records": str(rounds["trained"])})}
            second = ["d"]
            if experts == "intel" and "albucino" in sources:
                drafters["d-int4"] = Drafter("d-int4", tuned, "albucino", None,
                                             f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set "
                                             f"TAAF_VLLM_MTP_WEIGHTS={D_FILE}")
                second.append("d-int4")
            say(f"pass 2: the same {len(held)} held-out records, drafters {second}")
            later = run_pass("pass 2", bench, extractor, held, [drafters[one] for one in second], install, heads, main)
            later.update({"name": "pass 2", "labels": second})
            results.append(later)
            rows_now = table(results, drafters, names, bench.temperatures)
            report["table"] = rows_now
    else:
        say("(d) not run: no tuning minutes")
    say(f"TABLE ({main[0]} is the serve's vocabulary now; each line names the vocabulary that drafts best for it):")
    for row in rows_now:
        say("TABLE " + table_line(row, bench.drafts, bench.temperatures[0]))
    for result in results:
        for label in result["labels"]:
            games = sorted(key.split("|", 1)[1] for key in result["games"] if key.startswith(label + "|"))
            say(f"BY GAME {label:<11} " + ", ".join(
                f"{game} {result['games'][label + '|' + game].get('tokens_per_step')}" for game in games)
                + f" ({main[0]}, T{main[1]:g})")
    decision = decide(rows_now, ("albucino", main_vocab))
    report["decision"] = decision
    if decision:
        say(f"NEXT RUN: {decision['label']} with {decision['vocab']}, {decision['tokens_per_step']} tokens a step "
            f"against {decision['served_now_tokens_per_step']} for what the pure serve drafts with now "
            f"({decision['served_now']}); {decision['how']}"
            + ("" if decision["vocab"] == main_vocab else f"; TAAF_VLLM_DRAFT_VOCAB_FILE={decision['vocab']}"))
    report["extraction"] = extractor.stats
    report["minutes"] = round((time.monotonic() - began) / 60, 1)
    (out / "draft_ab_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    say(f"done in {report['minutes']} min; report {out / 'draft_ab_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
