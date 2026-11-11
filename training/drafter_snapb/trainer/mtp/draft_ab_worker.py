from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import capture_snapshot as cap  # noqa: E402
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
CALL_TAIL = re.compile(r"_s\d+_c\d+$")
INTEL_NEEDED = ("intel", "tuned")


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
    for field in ("hidden", "hc", "heads", "kv_heads", "head_dim", "top_k", "inter"):
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


def live_bases(bases: torch.Tensor, given, drafts: int, count: int) -> torch.Tensor:
    if given is None or len(given) < 3:
        raise RuntimeError("ab_bases=live reads the drafter rows (.dr) of a decode capture, and this record has none")
    verify = given[0].float().sum(-1) > 0.5
    verdict = given[2][:, 1]
    rows = (bases[:, None] + 1 + torch.arange(drafts, device=bases.device)[None, :]).clamp(max=count - 1)
    keep = verify[rows].all(-1) & (verdict[(bases + 1).clamp(max=count - 1)] != 1) & (bases + drafts < count - 1)
    return bases[keep]


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
        self.sampled_taus = [float(one) for one in cfg.get("sampled_taus") or []]
        self.rows: dict[tuple[int, str], torch.Tensor] = {}
        self.calib: dict[tuple[str, str], Any] = {}
        self.agree: dict[tuple[str, str], list[int]] = {}

    def plan(self, ids: torch.Tensor, start: int, seed: int, given=None) -> torch.Tensor:
        count = ids.shape[0]
        bases = torch.arange(max(0, start - 1 - self.drafts), count - 2, device=ids.device)
        if self.cfg.get("ab_bases") == "live":
            bases = live_bases(bases, given, self.drafts, count)
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
            rows = lm_head if ids is None else lm_head.index_select(0, ids)
            view = name.partition("+")[2]
            self.rows[held] = ref.head_view(rows, view) if view.startswith("fp8") else rows
        return self.rows[held]

    def drafted(self, out: torch.Tensor, lm_head: torch.Tensor, name: str,
                temperature: float | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        rows = self.vocab_rows(lm_head, name)
        picks, confs = [], []
        for one in range(0, out.shape[0], DRAFT_CHUNK):
            logits = F.linear(out[one:one + DRAFT_CHUNK].to(rows.dtype), rows)
            picks.append(logits.argmax(-1))
            if temperature is not None:
                confs.append(ref.draft_conf(logits, temperature))
        chosen = torch.cat(picks, 0)
        ids = self.vocabs[name]
        chosen = chosen if ids is None else ids[chosen]
        return chosen if temperature is None else (chosen, torch.cat(confs, 0))

    def sampled(self, out: torch.Tensor, lm_head: torch.Tensor, name: str, probs: torch.Tensor,
                tids: torch.Tensor, temperature: float) -> torch.Tensor:
        rows = self.vocab_rows(lm_head, name)
        ids = self.vocabs[name]
        column = None
        if ids is not None:
            column = torch.full((int(lm_head.shape[0]),), -1, dtype=torch.long, device=ids.device)
            column[ids] = torch.arange(ids.numel(), device=ids.device)
        kept = []
        for one in range(0, out.shape[0], DRAFT_CHUNK):
            logits = F.linear(out[one:one + DRAFT_CHUNK].to(rows.dtype), rows)
            held = tids[one:one + DRAFT_CHUNK]
            cols = held if column is None else column[held]
            kept.append(ref.sampled_acceptance(logits, probs[one:one + DRAFT_CHUNK], cols, temperature))
        return torch.cat(kept, 0)

    def record(self, ids: torch.Tensor, multi: torch.Tensor, sample: torch.Tensor | None, start: int,
               drafters: list[Drafter], install, heads: dict[str, tuple[torch.Tensor, torch.Tensor]], seed: int = 0,
               given: tuple[torch.Tensor, torch.Tensor] | None = None):
        count = ids.shape[0]
        first_target = start - 1
        bases = self.plan(ids, start, seed, given)
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
        if given is not None:
            if len(self.temperatures) != 1:
                raise ValueError("stored targets hold one temperature; evaluate other temperatures on a live target")
            targets = {self.temperatures[0]: (given[0][at].float(), given[1][at].long())}
        else:
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
            chosen: dict[str, list[torch.Tensor]] = {}
            for name in self.vocabs:
                drafted = [self.drafted(one, lm_head, name, self.temperatures[0]) for one in outs]
                picks = [one[0] for one in drafted]
                chosen[name] = picks
                plain = name.partition("+")[0]
                if plain != name and plain in chosen:
                    tally = self.agree.setdefault((drafter.label, name), [0, 0])
                    tally[0] += sum(int((one[full] == two[full]).sum()) for one, two in zip(picks, chosen[plain]))
                    tally[1] += int(full.sum()) * len(picks)
                for temperature, (probs, tids) in targets.items():
                    rates = []
                    for step, pick in enumerate(picks):
                        row = row_of[steps[step].clamp(max=count - 2)]
                        rates.append((probs[row] * (tids[row] == pick[:, None])).sum(-1)[full])
                    views[(name, temperature)] = (rates, probs[first][:, 0][full])
                    if temperature == self.temperatures[0]:
                        self.calib.setdefault((drafter.label, name), cap.Calib()).add(
                            torch.cat([one[1][full] for one in drafted]).float().cpu().numpy(),
                            torch.cat(rates).float().cpu().numpy())
                    for tau in self.sampled_taus:
                        kept = []
                        for step, one in enumerate(outs):
                            row = row_of[steps[step].clamp(max=count - 2)]
                            kept.append(self.sampled(one, lm_head, name, probs[row], tids[row],
                                                     temperature * tau)[full])
                        views[(f"{name} sampled x{tau:g}", temperature)] = (kept, probs[first][:, 0][full])
            out[drafter.label] = views
        return out


CONTEXT_BANDS = ((10000, "0-10K"), (30000, "10-30K"), (60000, "30-60K"), (10**12, "60K+"))


def context_band(count: int) -> str:
    return next(label for limit, label in CONTEXT_BANDS if count <= limit)


def cell_table(title: str, cells: dict[str, dict[str, Any]], label: str) -> list[str]:
    games = sorted({key.split("|")[1] for key in cells if key.startswith(label + "|")})
    lines = [f"{title} {label}: game | " + " | ".join(band for _, band in CONTEXT_BANDS)
             + " (first-position conditional / tokens a step / positions)"]
    for game in games:
        parts = []
        for _, band in CONTEXT_BANDS:
            held = cells.get(f"{label}|{game}|{band}") or {}
            first = (held.get("teacher_forced") or [None])[0]
            parts.append(f"{first} / {held.get('tokens_per_step')} / {held.get('bases', 0)}" if held.get("bases")
                         else "-")
        lines.append(f"{title} {label}: {game} | " + " | ".join(parts))
    return lines


def length_band(count: int) -> str:
    return next(label for limit, label in base.LENGTHS if count <= limit)


def record_seed(record: dict[str, Any]) -> int:
    return int(hashlib.sha256(str(record.get("id") or record.get("stem")).encode("utf-8")).hexdigest()[:8], 16)


MARGIN_BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))


def margin_tallies(book, label, rates, best, steps):
    if best is None:
        return
    for low, high in MARGIN_BANDS:
        mask = (best >= low) & (best < high)
        if bool(mask.any()):
            book.setdefault((label, f"top1 {low:g}-{min(high, 1.0):g}"), ref.Tally(steps)).add(
                [one[mask] for one in rates], best[mask])


def run_pass(name: str, bench: Bench, extractor: base.Extractor | None, rows: list[dict[str, Any]],
             drafters: list[Drafter], install, heads, main: tuple[str, float], fetch=None) -> dict[str, Any]:
    tallies: dict[tuple[str, str, float], ref.Tally] = {}
    games: dict[tuple[str, str], ref.Tally] = {}
    lengths: dict[tuple[str, str], ref.Tally] = {}
    cells: dict[tuple[str, str, str], ref.Tally] = {}
    margins: dict[tuple[str, str], ref.Tally] = {}
    counts = {"records": 0, "unread": 0, "no_reply": 0, "out_of_memory": 0}
    bench.calib, bench.agree = {}, {}
    began = time.monotonic()
    for index, record in enumerate(rows):
        given = None
        if fetch is None:
            got = extractor.hidden(record)
            if got is None:
                counts["unread"] += 1
                continue
            ids, multi, sample = got
            start = base.reply_start(extractor.tokenizer, ids, record)
            if start is None:
                counts["no_reply"] += 1
                continue
        else:
            got = fetch(record)
            if got is None:
                counts["unread"] += 1
                continue
            ids, multi, start, given = got
            sample = None
        try:
            stored = {"given": given} if given is not None else {}
            views = bench.record(ids, multi, sample, start, drafters, install, heads, record_seed(record), **stored)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            counts["out_of_memory"] += 1
            say(f"{name}: out of GPU memory on a {len(ids):,}-token record; it is left out for every drafter "
                f"({counts['out_of_memory']} so far)")
            continue
        ids_len = len(ids)
        got = ids = multi = sample = given = None
        if not views:
            continue
        counts["records"] += 1
        for label, held in views.items():
            for (vocab, temperature), (rates, best) in held.items():
                tallies.setdefault((label, vocab, temperature), ref.Tally(bench.drafts)).add(rates, best)
            rates, best = held[main]
            games.setdefault((label, str(record.get("game"))), ref.Tally(bench.drafts)).add(rates, best)
            sampled = held.get((f"{main[0]} sampled x1", main[1]))
            if sampled is not None:
                games.setdefault((label + SAMPLED, str(record.get("game"))), ref.Tally(bench.drafts)).add(*sampled)
            lengths.setdefault((label, length_band(int(ids_len))), ref.Tally(bench.drafts)).add(rates, best)
            cells.setdefault((label, str(record.get("game")), context_band(int(ids_len))),
                             ref.Tally(bench.drafts)).add(rates, best)
            margin_tallies(margins, label, rates, best, bench.drafts)
        if counts["records"] == 1 or index % PROGRESS_EVERY == PROGRESS_EVERY - 1 or index + 1 == len(rows):
            say(f"{name} {index + 1}/{len(rows)} ({time.monotonic() - began:.0f} s): " + "; ".join(
                f"{one.label} {tallies[(one.label, *main)].summary().get('tokens_per_step')}"
                for one in drafters if (one.label, *main) in tallies) + f" tokens a step at {bench.drafts} drafts "
                f"({main[0]}, T{main[1]:g})")
    return {"counts": counts, "seconds": round(time.monotonic() - began, 1),
            "views": {f"{label}|{vocab}|{temperature:g}": tally.summary()
                      for (label, vocab, temperature), tally in tallies.items()},
            "games": {f"{label}|{game}": tally.summary() for (label, game), tally in games.items()},
            "lengths": {f"{label}|{band}": tally.summary() for (label, band), tally in lengths.items()},
            "cells": {f"{label}|{game}|{band}": tally.summary() for (label, game, band), tally in cells.items()},
            "margins": {f"{label}|{band}": tally.summary() for (label, band), tally in margins.items()},
            "calibration": {f"{label}|{vocab}": calib.summary() for (label, vocab), calib in bench.calib.items()},
            "agreement": {f"{label}|{vocab}": round(same / total, 5) if total else None
                          for (label, vocab), (same, total) in bench.agree.items()}}


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
                         "records": result["counts"]["records"],
                         "calibration": (result.get("calibration") or {}).get(f"{label}|{chosen}") or {}})
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
            f"{others}; {held.get('bases', 0):,} positions over {row['records']} records; "
            f"{cap.calib_text(row.get('calibration') or {}, brief=True)}; {row['how']}")


def elasticity(conditional: list[float]) -> float:
    rate = sum(conditional) / len(conditional) if conditional else 0.0
    tokens = sum(rate ** power for power in range(len(conditional) + 1))
    slope = sum(power * rate ** (power - 1) for power in range(1, len(conditional) + 1))
    return slope * rate / tokens if tokens else 0.0


def saved_ms(name: str, size: int | None, plain_size: int | None, cfg: dict[str, Any]) -> float:
    whole = float(cfg.get("lm_head_ms") or 0.0)
    view = name.partition("+")[2]
    if view.startswith("fp8"):
        return whole / 2.0
    if size and plain_size:
        return whole * max(0.0, 1.0 - size / plain_size)
    return 0.0


def head_lines(result: dict[str, Any], labels: list[str], vocabs: dict[str, list[int] | None], temperature: float,
               cfg: dict[str, Any]) -> list[str]:
    step = float(cfg.get("step_ms") or 0.0)
    names = [one for one in vocabs if one != FULL]
    lines = []
    if not float(cfg.get("lm_head_ms") or 0.0) or len(names) < 2:
        return lines
    for label in labels:
        for name in names[1:]:
            plain = name.partition("+")[0]
            base_name = plain if plain != name and plain in vocabs else names[0]
            one = view(result, label, name, temperature)
            two = view(result, label, base_name, temperature)
            if not one.get("bases") or not two.get("bases"):
                continue
            loss = 1.0 - one["tokens_per_step"] / two["tokens_per_step"]
            size = len(vocabs[name]) if vocabs.get(name) is not None else None
            plain_size = len(vocabs[base_name]) if vocabs.get(base_name) is not None else None
            saving = saved_ms(name, size, plain_size, cfg)
            allowed = saving / step if step else 0.0
            give = elasticity(two.get("teacher_forced") or [])
            agree = (result.get("agreement") or {}).get(f"{label}|{name}")
            lines.append(
                f"LM_HEAD {label} {name} against {base_name}: {one['tokens_per_step']} against "
                f"{two['tokens_per_step']} tokens a step ({-loss * 100:+.2f}%), conditional "
                f"{one.get('teacher_forced')} against {two.get('teacher_forced')}, picks agree "
                f"{agree if agree is not None else '-'}; saves {saving:.2f} of {step:.1f} ms a step, so it pays "
                f"while tokens a step fall less than {allowed * 100:.2f}% (conditional acceptance "
                f"{allowed / give * 100 if give else 0:.2f}% relative); "
                + ("PAYS" if step and loss < allowed else "DOES NOT PAY"))
    return lines


SAMPLED = " sampled"
THEORY_GAIN = 0.01
THEORY_FIRST = 0.010
THEORY_GAMES = 2
THEORY_POSITIONS = 10000


def theory_lines(result: dict[str, Any], base_label: str, arms: list[str], main: tuple[str, float],
                 suffix: str = "") -> list[str]:
    games = sorted({key.split("|", 1)[1] for key in result.get("games", {})
                    if key.startswith(base_label + suffix + "|")})
    two = view(result, base_label, *main)
    lines = []
    for arm in arms:
        one = view(result, arm, *main)
        if not one.get("bases") or not two.get("bases"):
            continue
        gain = one["tokens_per_step"] / two["tokens_per_step"] - 1.0
        first = (one.get("teacher_forced") or [0.0])[0] - (two.get("teacher_forced") or [0.0])[0]
        by_game = {}
        for game in games:
            mine = result["games"].get(f"{arm}{suffix}|{game}") or {}
            theirs = result["games"].get(f"{base_label}{suffix}|{game}") or {}
            if mine.get("tokens_per_step") and theirs.get("tokens_per_step"):
                by_game[game] = mine["tokens_per_step"] / theirs["tokens_per_step"] - 1.0
        if len(by_game) < THEORY_GAMES or two["bases"] < THEORY_POSITIONS:
            verdict = f"TOO SMALL (needs {THEORY_GAMES} games and {THEORY_POSITIONS:,} positions)"
        elif all(value >= THEORY_GAIN for value in by_game.values()):
            verdict = "CONFIRMED"
        elif gain < THEORY_GAIN and first < THEORY_FIRST:
            verdict = "REFUTED"
        else:
            verdict = "UNSETTLED"
        lines.append(f"DECODE DATA{suffix} {arm} against {base_label}: {gain * 100:+.2f}% tokens a step, position 1 "
                     f"{first:+.4f}, by game " + ", ".join(f"{game} {value * 100:+.2f}%" for game, value in by_game.items())
                     + f"; {two['bases']:,} positions over {len(by_game)} held-out games; {verdict}")
    return lines


def reply_counts(reader) -> Any:
    import numpy as np
    counts = None
    for entry in reader.entries("train"):
        held = reader.mapped(int(entry["seg"]))["ids"]
        part = np.asarray(held[int(entry["start"]):int(entry["count"])], dtype=np.int64)
        got = np.bincount(part, minlength=1)
        if counts is None or len(counts) < len(got):
            grown = np.zeros(len(got), dtype=np.int64)
            if counts is not None:
                grown[:len(counts)] = counts
            counts = grown
        counts[:len(got)] += got
    return counts


def load_vocabs(cfg: dict[str, Any], reader=None) -> dict[str, list[int] | None]:
    vocabs: dict[str, list[int] | None] = {FULL: None}
    counts = None
    for name, path in dict(cfg.get("ab_vocab_files") or {}).items():
        ids = json.loads(Path(path).read_text(encoding="utf-8"))["ids"]
        view = name.partition("+")[2]
        if view.startswith("top"):
            if reader is None or not view[3:].isdigit():
                say(f"vocabulary {name} needs a snapshot to rank its ids by use and a number after top; left out")
                continue
            counts = reply_counts(reader) if counts is None else counts
            used = [(int(counts[one]) if one < len(counts) else 0, -one, one) for one in ids]
            ids = sorted(one for _, _, one in sorted(used, reverse=True)[:int(view[3:])])
            say(f"vocabulary {name}: the {len(ids)} ids of {Path(path).name} the train replies use most")
        vocabs[name] = ids
    return vocabs


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


def sequences(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for one in rows:
        stem = CALL_TAIL.sub("", str(one.get("stem") or one.get("id") or ""))
        groups.setdefault((str(one.get("run")), str(one.get("game")), stem), []).append(one)
    out = []
    for key in sorted(groups):
        out.append(sorted(groups[key], key=lambda one: (int(one.get("step") or 0), int(one.get("call") or 0))))
    return out


def windows(rows: list[dict[str, Any]], size: int, limit: int, seed: int) -> list[dict[str, Any]]:
    picker = random.Random(seed)
    groups = sequences(rows)
    picker.shuffle(groups)
    if limit:
        out: list[dict[str, Any]] = []
        for group in groups:
            if len(out) >= limit:
                break
            take = min(size, len(group), limit - len(out))
            start = picker.randrange(len(group) - take + 1)
            out += group[start:start + take]
        return out
    pieces = [group[start:start + size] for group in groups for start in range(0, len(group), size)]
    picker.shuffle(pieces)
    return [one for piece in pieces for one in piece]


def held_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [one for one in rows if one["split"] == "train"]
    held = [one for one in rows if one["split"] == "heldout"]
    if int(cfg.get("train_block") or 0) > 0:
        train = windows(train, int(cfg["train_block"]), 0, int(cfg["seed"]))
    if int(cfg.get("eval_block") or 0) > 0:
        return train, windows(held, int(cfg["eval_block"]), int(cfg["eval_records"] or len(held)), int(cfg["seed"]))
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


SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
EVAL_ESTIMATE_S = 300.0


def game_targets(sample: torch.Tensor, lm_head: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    probs, ids = [], []
    for start in range(0, sample.shape[0], TARGET_CHUNK):
        logits = F.linear(sample[start:start + TARGET_CHUNK], lm_head)
        p, i = ref.game_distribution(logits, temperature, base.GAME["top_k"], base.GAME["top_p"])
        probs.append(p.to(torch.float16))
        ids.append(i.to(torch.int32))
    return torch.cat(probs, 0), torch.cat(ids, 0)


def load_drafters(cfg: dict[str, Any], shape: ref.Shape) -> tuple[dict[str, Source], dict[str, Drafter], dict]:
    sources = {"albucino": load_albucino(Path(cfg["draft_model"]), shape)}
    refused: dict[str, str] = {}
    drafters, notes = build_drafters(cfg, sources, refused)
    notes["refused"] = refused
    if "albucino" not in drafters:
        raise SystemExit(f"the albucino drafter did not build: {refused}")
    return sources, drafters, notes


def own_heads(sources: dict[str, Source], lm_head: torch.Tensor, embed: torch.Tensor, device: torch.device,
              notes: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    own = sources["albucino"]
    heads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    same = {}
    for tag, mine, target in (("lm_head", own.lm_head, lm_head), ("embed", own.embed, embed)):
        same[tag] = mine is None or (tuple(mine.shape) == tuple(target.shape)
                                     and torch.equal(mine.to(device, target.dtype), target))
    if not all(same.values()):
        heads["albucino"] = (own.lm_head.to(device, lm_head.dtype) if own.lm_head is not None else lm_head,
                             own.embed.to(device, embed.dtype) if own.embed is not None else embed)
    own.lm_head = own.embed = None
    notes["albucino_head_equals_target"] = same
    return heads


def snap_preflight(cfg: dict[str, Any], config: dict[str, Any], shape: ref.Shape, sources: dict[str, Source],
                   start: Drafter, device: torch.device, folder: Path) -> dict[str, Any]:
    import mtp_snapshot as snap
    began = time.monotonic()
    vocab = int((config.get("text_config") or {}).get("vocab_size") or config["vocab_size"])
    count = 1024
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lm_head = (torch.randn(vocab, shape.hidden, device=device) * 0.02).to(torch.bfloat16)
    embed = (torch.randn(vocab, shape.hidden, device=device) * 0.02).to(torch.bfloat16)
    albucino = sources["albucino"]
    head = ref.MTPHead(shape, start.dense, albucino.gate_up, albucino.down, train_experts=True).to(device)
    train_vocab = json.loads(Path(cfg["draft_vocab"]).read_text(encoding="utf-8"))["ids"] \
        if cfg.get("draft_vocab") else None
    trainer = base.Trainer(head, {"embed": embed, "lm_head": lm_head}, train_vocab, cfg)
    ids = torch.randint(10, vocab, (count,), device=device)
    multi = torch.randn(count, shape.hc * shape.hidden, device=device).to(torch.bfloat16)
    given = game_targets(torch.randn(count, shape.hidden, device=device).to(torch.bfloat16), lm_head,
                         float(cfg["temperature"]))
    opt, params = base.optimizer(cfg, head)

    def step(one_trainer, one_opt, one_params) -> tuple[torch.Tensor, float]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        clock = time.monotonic()
        torch.manual_seed(int(cfg["seed"]))
        result = one_trainer.pass_one(ids, multi, None, count - 200, int(cfg["steps"]), True, 128,
                                      str(cfg.get("span") or "reply"), float(cfg.get("late") or 0.0), given)
        if result is None:
            raise RuntimeError("the preflight record planned no position")
        loss = sum(weight * one for weight, one in zip(one_trainer.weights, result[0]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(one_params, float(cfg["clip"]))
        one_opt.step()
        one_opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return loss.detach(), time.monotonic() - clock

    before = head.expert_weights(0)[0].detach().clone()
    loss, _ = step(trainer, opt, params)
    moved = float((head.expert_weights(0)[0].detach() - before).norm())
    if not moved > 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError(f"one expert step moved the experts by {moved} with loss {float(loss)}")
    _, expert_s = step(trainer, opt, params)
    dense_head = ref.MTPHead(shape, start.dense, albucino.gate_up, albucino.down).to(device)
    dense_trainer = base.Trainer(dense_head, {"embed": embed, "lm_head": lm_head}, train_vocab, cfg)
    dense_opt, dense_params = base.optimizer(cfg, dense_head)
    step(dense_trainer, dense_opt, dense_params)
    _, dense_s = step(dense_trainer, dense_opt, dense_params)
    del dense_head, dense_trainer, dense_opt, dense_params
    slower = expert_s / max(dense_s, 1e-6)
    if slower > float(cfg.get("snap_expert_slowdown") or 10.0):
        raise RuntimeError(f"a training step with the experts takes {expert_s:.2f} s against {dense_s:.2f} s for the "
                           f"dense tensors alone ({slower:.1f}x): the experts arm would see that many times fewer "
                           f"records in its minutes")
    group = int4_group(json.loads((Path(cfg["draft_model"]) / "config.json").read_text(encoding="utf-8")))
    weight = head.expert_weights(0)[0].detach().to(torch.bfloat16)
    int4_error = float((ref.int4_roundtrip(weight, group).float() - weight.float()).norm() / weight.float().norm())
    q, scale = snap.quantize_rows(multi, shape.hc)
    int8_error = float((snap.dequantize_rows(q, scale, shape.hc).float() - multi.float()).norm()
                       / multi.float().norm())
    probe = folder.parent / (folder.name + "_preflight")
    writer = snap.Writer(probe, shape.hc, shape.hc * shape.hidden, base.GAME["top_k"])
    writer.add({"id": "p", "split": "train"}, ids.tolist(), count - 200, lambda at: multi[at:],
               lambda at: (given[0][at:], given[1][at:]))
    writer.close({})
    back_ids, back_multi, _, (back_p, back_t) = snap.Reader(probe, device).fetch(writer.entries[0])
    same = torch.equal(back_ids, ids) and torch.equal(back_t, given[1]) and \
        torch.equal(back_multi, snap.dequantize_rows(q, scale, shape.hc))
    shutil.rmtree(probe, ignore_errors=True)
    if not same or int8_error > 0.02 or int4_error > 0.25:
        raise RuntimeError(f"the snapshot does not read back what it wrote ({same}) or the int8 / int4 error "
                           f"({int8_error:.4f} / {int4_error:.4f}) is out of bounds")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    del head, trainer, opt, params, loss, lm_head, embed, multi, q, scale, back_multi
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = {"seconds": round(time.monotonic() - began, 1), "peak_gib": round(peak, 1),
           "int8_error": round(int8_error, 5), "int4_error": round(int4_error, 4), "expert_step": moved,
           "step_s_experts": round(expert_s, 3), "step_s_dense": round(dense_s, 3)}
    say(f"SNAP PREFLIGHT ok before the target loads: one fp32 expert step (all {len(albucino.gate_up)} experts, "
        f"fused AdamW) "
        f"moved expert 0 by {moved:.2e}, a step {expert_s:.2f} s against {dense_s:.2f} s for the dense tensors alone "
        f"({slower:.1f}x), peak {peak:.1f} GiB of the GPU; int8 rows {int8_error:.4f} and int4 g{group} "
        f"{int4_error:.4f} relative error; the snapshot store read back what it wrote; {out['seconds']} s")
    return out


def snap_extract(cfg: dict[str, Any]) -> int:
    import mtp_snapshot as snap
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    began = time.monotonic()
    rows = base.records(Path(cfg["corpus"]))
    train, held = held_rows(rows, cfg)
    folder = Path(cfg["snap_dir"])
    room = shutil.disk_usage(folder if folder.exists() else folder.parent).free
    budget = min(float(cfg["snap_gib"]) * snap.GiB, room - 4 * snap.GiB)
    held_gib, ceiling_gib = snap.cgroup_held_gib()
    say(f"SNAP corpus: {len(rows)} record(s); {len(held)} held out from games "
        f"{sorted({one['game'] for one in held})}, {len(train)} to tune on; snapshot under {folder}, "
        f"{room / snap.GiB:.1f} GiB free there, budget {budget / snap.GiB:.1f} GiB; cgroup holds "
        f"{held_gib if held_gib is None else round(held_gib, 1)} of {ceiling_gib} GiB")
    if budget < 8 * snap.GiB:
        raise SystemExit(f"{folder} leaves {budget / snap.GiB:.1f} GiB for the snapshot; nothing loaded")
    model_dir = Path(cfg["model_dir"])
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    shape = ref.Shape.from_config(config)
    if not cfg.get("draft_model"):
        raise SystemExit("the serve names no albucino drafter, so the snapshot has no head to check int8 against")
    device = torch.device(cfg.get("device") or "cuda")
    sources, drafters, notes = load_drafters(cfg, shape)
    start_label = "tuned-int4" if "tuned-int4" in drafters else "albucino"
    say(f"SNAP drafters before the target loads: {sorted(drafters)}; {sources['albucino'].note}")
    notes["preflight"] = snap_preflight(cfg, config, shape, sources, drafters[start_label], device, folder)
    llm = base.engine(str(model_dir), cfg["serve_args"], cfg)
    found = base.attach(llm)
    lm_head = found["lm_head"].detach().to(device)
    embed = found["embed"].detach().to(device)
    tokenizer = llm.get_tokenizer()
    header = tokenizer.encode(base.ASSISTANT_HEADER, add_special_tokens=False)
    end_id = int(tokenizer.convert_tokens_to_ids(base.END_TOKEN))
    temperature = float(cfg["temperature"])
    train_bits = int(cfg.get("snap_train_bits") or 8)
    writer = snap.Writer(folder, shape.hc, shape.hc * shape.hidden, base.GAME["top_k"], train_bits)
    torch.save(lm_head.cpu(), folder / snap.LM_HEAD)
    torch.save(embed.cpu(), folder / snap.EMBED)
    heads = own_heads(sources, lm_head, embed, device, notes)
    labels = [one for one in ("albucino", "tuned-int4") if one in drafters]
    head = ref.MTPHead(shape, sources["albucino"].dense, sources["albucino"].gate_up,
                       sources["albucino"].down).to(device)
    install = Installer(head, sources, "albucino")
    vocabs = load_vocabs(cfg)
    bench = Bench(head, lm_head, embed, vocabs, cfg)
    main_view = (next((one for one in vocabs if one != FULL), FULL), bench.temperatures[0])
    kinds = ("bf16", "int8", "int4") if train_bits == 4 else ("bf16", "int8")
    parity = {label: {kind: ref.Tally(bench.drafts) for kind in kinds} for label in labels}
    extractor = base.Extractor(llm, tokenizer)
    order = [*held, *train]
    limit_s = float(cfg["snap_extract_minutes"]) * 60.0
    stopped = "the corpus ran out"
    counts = {"held": 0, "train": 0, "unread": 0, "no_reply": 0}
    extract_began = time.monotonic()
    for index, record in enumerate(order):
        if record["split"] == "train":
            if writer.bytes >= budget:
                stopped = f"the snapshot reached its {budget / snap.GiB:.1f} GiB budget"
                break
            held_now, ceiling = snap.cgroup_held_gib()
            if held_now is not None and ceiling is not None and held_now > ceiling - float(cfg["snap_margin_gib"]):
                stopped = f"the cgroup holds {held_now:.1f} of {ceiling:.1f} GiB"
                break
            if time.monotonic() - extract_began > limit_s:
                stopped = f"the {cfg['snap_extract_minutes']:g} extraction minutes are spent"
                break
        got = extractor.hidden(record)
        if got is None:
            counts["unread"] += 1
            continue
        ids_t, multi, sample = got
        start = base.reply_start(tokenizer, ids_t, record)
        if start is None:
            counts["no_reply"] += 1
            continue
        if record["split"] == "heldout" and counts["held"] < int(cfg["snap_parity"]):
            seen = {"bf16": bench.record(ids_t, multi, sample, start, [drafters[one] for one in labels], install,
                                         heads, record_seed(record))}
            for kind in kinds[1:]:
                seen[kind] = bench.record(ids_t, snap.stored_rows(multi, shape.hc, 4 if kind == "int4" else 8),
                                          sample, start, [drafters[one] for one in labels], install, heads,
                                          record_seed(record))
            for label in labels:
                for kind, views in seen.items():
                    if label in views:
                        rates, best = views[label][main_view]
                        parity[label][kind].add(rates, best)
        writer.add(record, ids_t.tolist(), start, lambda first: multi[first:],
                   lambda first: game_targets(sample[first:], lm_head, temperature))
        counts["held" if record["split"] == "heldout" else "train"] += 1
        got = ids_t = multi = sample = None
        if index % 50 == 49:
            held_now, ceiling = snap.cgroup_held_gib()
            say(f"SNAP {counts['held']} held-out and {counts['train']} train record(s) in "
                f"{len(writer.segments)} segment(s) ({writer.extended} extended one), "
                f"{writer.bytes / snap.GiB:.1f} GiB, "
                f"{extractor.stats['computed'] / max(1e-9, extractor.stats['seconds']):,.0f} tok/s computed, "
                f"cgroup {held_now if held_now is None else round(held_now, 1)} of {ceiling} GiB, "
                f"{(time.monotonic() - extract_began) / 60:.1f} min")
    parity_out = {label: {kind: tally.summary() for kind, tally in kinds.items()} for label, kinds in parity.items()}
    for label, measured in parity_out.items():
        say(f"SNAP INT8 {label}: {measured['bf16'].get('tokens_per_step')} tokens a step from the BF16 hidden states, "
            f"{measured['int8'].get('tokens_per_step')} from the same states through int8"
            + (f", {measured['int4'].get('tokens_per_step')} through int4 g{snap.group_for(shape.hc * shape.hidden, shape.hc)} (the train rows)"
               if "int4" in measured else "")
            + f" ({measured['bf16'].get('bases', 0):,} positions over {cfg['snap_parity']} held-out records, "
            f"{main_view[0]}, T{main_view[1]:g})")
    index = writer.close({"header": header, "end_id": end_id, "temperature": temperature,
                          "top_k": base.GAME["top_k"], "top_p": base.GAME["top_p"], "model_dir": str(model_dir),
                          "parity": parity_out, "counts": counts, "stopped": stopped,
                          "extraction": extractor.stats, "notes": notes})
    say(f"SNAP done: {counts['held']} held-out and {counts['train']} train record(s), {len(writer.segments)} "
        f"segment(s), {index['bytes'] / snap.GiB:.1f} GiB, stopped because {stopped}; "
        f"{(time.monotonic() - began) / 60:.1f} min")
    if float(cfg.get("snap_save_gib") or 0) > 0:
        took = time.monotonic()
        saved = snap.save_subset(folder, out / "mtp_snap_saved", int(float(cfg["snap_save_gib"]) * snap.GiB),
                                 [folder / snap.LM_HEAD, folder / snap.EMBED, model_dir / "config.json"])
        index["saved"] = saved
        say(f"SNAP SAVED to {out / 'mtp_snap_saved'}: {saved['heldout']} held-out and {saved['train']} train "
            f"record(s) in {saved['segments']} of {saved['of']} segment(s), {saved['bytes'] / snap.GiB:.1f} GiB with "
            f"lm_head, embedding and config, in {time.monotonic() - took:.0f} s; a later kernel trains from it with "
            f"ARC_MTP_TUNE_SNAP_FROM and loads no target")
    (out / "snapshot_extract_report.json").write_text(json.dumps({k: v for k, v in index.items() if k != "entries"},
                                                                 indent=1, default=str), encoding="utf-8")
    if counts["held"] == 0 or counts["train"] == 0:
        raise SystemExit("the snapshot holds no held-out or no train record")
    return 0


def expert_scale_dtype(draft_dir: Path, layer: int = 0) -> torch.dtype:
    inventory = ref.mtp_inventory(draft_dir)
    kinds = {spec["dtype"] for name, spec in inventory.items()
             if name.startswith(f"mtp.layers.{layer}.mlp.experts.") and name.endswith(".weight_scale")}
    if len(kinds) != 1 or next(iter(kinds)) not in SAFETENSORS_DTYPES:
        raise RuntimeError(f"the drafter's expert scales are {sorted(kinds)}, not one known dtype")
    return SAFETENSORS_DTYPES[next(iter(kinds))]


def export_experts(gate_up: torch.Tensor, down: torch.Tensor, inter: int, group: int, scale_dtype: torch.dtype,
                   path: Path, layer: int = 0) -> dict[str, Any]:
    from safetensors.torch import save_file
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(gate_up.shape[0]):
        for proj, weight in (("gate_proj", gate_up[expert, :inter]), ("up_proj", gate_up[expert, inter:]),
                             ("down_proj", down[expert])):
            packed, scale, shape = ref.pack_int4(weight.float(), group, scale_dtype)
            stem = f"mtp.layers.{layer}.mlp.experts.{expert}.{proj}"
            tensors[stem + ".weight_packed"] = packed.cpu()
            tensors[stem + ".weight_scale"] = scale.cpu()
            tensors[stem + ".weight_shape"] = shape
    save_file(tensors, str(path), metadata={"format": "pack-quantized", "group_size": str(group)})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    say(f"wrote {path.name}: {len(tensors)} tensor(s), {path.stat().st_size / 2**20:.1f} MiB, sha256 {digest[:16]}")
    return {"file": path.name, "tensors": len(tensors), "sha256": digest}


def arm_specs(said: str) -> list[tuple[str, float | None, str]]:
    out = []
    for one in [part.strip() for part in str(said).split(",") if part.strip()]:
        kind, _, rate = one.partition("@")
        lr = float(rate) if rate else None
        label = ("B" if kind == "experts" else "A") + (f"-lr{lr:g}" if lr is not None else "")
        out.append((kind, lr, label))
    return out


def epochs(entries: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    picker = random.Random(seed)
    out: list[dict[str, Any]] = []
    for _ in range(max(1, count)):
        one = list(entries)
        picker.shuffle(one)
        out += one
    return out


def select_entries(held: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not held:
        return []
    step = max(1, len(held) // count)
    return held[::step][:count]


def curve_arms(said: str) -> list[tuple[str, float]]:
    out = []
    for one in [part.strip() for part in str(said).replace(";", ",").split(",") if part.strip()]:
        name, _, share = one.partition("=")
        out.append((name.strip(), float(share)))
    return out


def curve_plans(train: list[dict[str, Any]], arms: list[tuple[str, float]], updates: int, accum: int,
                seed: int) -> dict[str, dict[str, Any]]:
    order = list(train)
    random.Random(seed).shuffle(order)
    asked = updates or max(1, len(order) // accum)
    plans = {}
    for name, share in arms:
        calls = order[:max(1, min(len(order), round(share * len(order))))]
        plans[name] = {"share": share, "calls": calls, "updates": asked}
    return plans


def curve_stream(plan: dict[str, Any], accum: int, seed: int) -> list[dict[str, Any]]:
    need = plan["updates"] * accum
    length = need + need // 10 + accum
    return epochs(plan["calls"], -(-length // len(plan["calls"])), seed)[:length]


def should_stop(bests: list[float], evals: int, gain: float) -> bool:
    return len(bests) > evals and bests[-1] - bests[-1 - evals] < gain


def curve_table(arms: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, arm in arms.items():
        kept = arm.get("keep_best") or {}
        curve = [one for one in kept.get("curve") or [] if one.get("tokens_per_step") is not None]
        if not curve:
            rows.append({"arm": name, "failed": arm.get("failed")})
            continue
        best = max(curve, key=lambda one: one["tokens_per_step"])
        first, last = curve[0], curve[-1]
        rows.append({"arm": name, "share": arm.get("share"), "unique_calls": arm.get("unique_calls"),
                     "updates": arm.get("updates"), "evals": len(curve),
                     "start": first["tokens_per_step"], "best": best["tokens_per_step"],
                     "best_update": best["update"], "final": last["tokens_per_step"],
                     "start_4": first.get("tokens_per_step_4"), "best_4": best.get("tokens_per_step_4"),
                     "final_4": last.get("tokens_per_step_4"),
                     "probe_start": first.get("probe_tokens_per_step"),
                     "probe_at_best": best.get("probe_tokens_per_step"),
                     "probe_final": last.get("probe_tokens_per_step"), "stopped": kept.get("stopped")})
    return rows


class CurveTrack:
    def __init__(self, path: Path, arm: str, unique: int, accum: int, probe: list[dict[str, Any]], gain: float,
                 evals: int, began: float):
        self.path = path
        self.arm = arm
        self.unique = unique
        self.accum = accum
        self.probe = probe
        self.gain = gain
        self.evals = evals
        self.began = began
        self.bests: list[float] = []

    def row(self, update: int, held: dict[str, Any] | None, probe: dict[str, Any] | None) -> dict[str, Any]:
        def at(views, count):
            positions = (views or {}).get("per_position") or []
            return round(1.0 + sum(positions[:count]), 4) if len(positions) >= count else None
        return {"arm": self.arm, "update": update, "calls_seen": update * self.accum,
                "unique_calls_in_arm": self.unique, "heldout_tokens_per_step_3": at(held, 3),
                "heldout_tokens_per_step_4": at(held, 4), "heldout_per_position": (held or {}).get("per_position"),
                "probe_calls": len(self.probe), "probe_tokens_per_step_3": at(probe, 3),
                "probe_tokens_per_step_4": at(probe, 4), "probe_per_position": (probe or {}).get("per_position"),
                "elapsed_min": round((time.monotonic() - self.began) / 60, 2)}

    def write(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        say(f"CURVE {row['arm']} update {row['update']} calls {row['calls_seen']} of {row['unique_calls_in_arm']} "
            f"unique: held-out {row['heldout_tokens_per_step_3']} at 3 drafts, {row['heldout_tokens_per_step_4']} "
            f"at 4; train probe {row['probe_tokens_per_step_3']} at 3, {row['probe_tokens_per_step_4']} at 4; "
            f"{row['elapsed_min']} min")

    def stop(self, score: float | None) -> str | None:
        if score is None:
            return None
        self.bests.append(max([score, *self.bests[-1:]]))
        if should_stop(self.bests, self.evals, self.gain):
            return (f"held-out best moved {self.bests[-1] - self.bests[-1 - self.evals]:.4f} over the last "
                    f"{self.evals} evals, under {self.gain:g}")
        return None


EVAL_STEPS = 4


class KeepBest:
    def __init__(self, label: str, head: ref.MTPHead, trainer, select: list[dict[str, Any]], fetch, steps: int,
                 bases: int, track: CurveTrack | None = None):
        self.label = label
        self.head = head
        self.trainer = trainer
        self.select = select
        self.fetch = fetch
        self.steps = steps
        self.bases = bases
        self.params = list(head.trainable())
        self.large = {id(one) for one in head.expert_parameters()}
        self.best: tuple[float, int] | None = None
        self.saved: list[torch.Tensor] | None = None
        self.curve: list[dict[str, Any]] = []
        self.views: dict[str, Any] = {}
        self.track = track
        self.stopped: str | None = None

    def measure(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        steps = max(self.steps, EVAL_STEPS)
        books = {"main": ref.Tally(steps), "greedy": ref.Tally(steps), "x0.7": ref.Tally(steps)}
        calib = cap.Calib()
        for entry in entries:
            ids, multi, start, given = self.fetch(entry)
            self.trainer.calib_last = None
            result = self.trainer.pass_one(ids, multi, None, start, steps, False, self.bases, given=given)
            if result is None:
                continue
            if self.trainer.calib_last and self.trainer.calib_last[0]:
                confs, hits = self.trainer.calib_last
                calib.add(torch.cat(confs).float().cpu().numpy(), torch.cat(hits).float().cpu().numpy())
            books["main"].add(result[1], result[2])
            more = result[4]
            if " greedy" in more:
                books["greedy"].add(*more[" greedy"])
            elif getattr(self.trainer, "target", "") != "tv":
                books["greedy"].add(result[1], result[2])
            if " sampled x0.7" in more:
                books["x0.7"].add(*more[" sampled x0.7"])
        views = {name: book.summary() for name, book in books.items()}
        views["calib"] = calib.summary()
        return views

    def score(self) -> float | None:
        self.views = self.measure(self.select)
        main = self.views["main"]
        if not main.get("bases"):
            return None
        reached = main["per_position"][:self.steps]
        return round(1.0 + sum(reached), 4)

    def eval_line(self, update: int) -> str:
        def positions(name):
            held = self.views.get(name) or {}
            return "/".join(f"{one:.3f}" for one in held.get("teacher_forced") or []) or "-"
        recent = getattr(self.trainer, "recent", None) or {}
        running = (recent.get("running") or {}).get("teacher_forced") or []
        train = "/".join(f"{one:.3f}" for one in running) or "-"
        return (f"EVAL {self.label} update {update}: summin x0.7 {positions('x0.7')} summin x1 {positions('main')} "
                f"greedy {positions('greedy')} tok/step greedy {(self.views.get('greedy') or {}).get('tokens_per_step')} "
                f"x1 {(self.views.get('main') or {}).get('tokens_per_step')} | "
                f"{cap.calib_text(self.views.get('calib') or {}, brief=True)} | train loss "
                f"{recent.get('loss', float('nan')):.4f} train summin x1 {train} | {len(self.select)} picking records")

    def __call__(self, update: int) -> None:
        began = time.monotonic()
        score = self.score()
        entry = {"update": update, "tokens_per_step": score, "views": self.views}
        self.curve.append(entry)
        if self.track is not None:
            probe = self.measure(self.track.probe)["main"] if self.track.probe else None
            row = self.track.row(update, self.views.get("main"), probe)
            entry.update({"tokens_per_step_4": row["heldout_tokens_per_step_4"],
                          "probe_tokens_per_step": row["probe_tokens_per_step_3"],
                          "probe_tokens_per_step_4": row["probe_tokens_per_step_4"]})
            self.track.write(row)
            self.stopped = self.stopped or self.track.stop(score)
        if score is not None and (self.best is None or score > self.best[0]):
            self.best = (score, update)
            self.saved = [one.detach().to(torch.bfloat16 if id(one) in self.large else one.dtype).clone()
                          for one in self.params]
        say(self.eval_line(update))
        say(f"arm {self.label} checkpoint at update {update}: {score} tokens a step on {len(self.select)} held-out "
            f"records; best {self.best[0] if self.best else None} at update {self.best[1] if self.best else None} "
            f"({time.monotonic() - began:.0f} s)")

    def restore(self) -> dict[str, Any]:
        if self.saved is not None:
            with torch.no_grad():
                for param, kept in zip(self.params, self.saved):
                    param.copy_(kept.to(param.dtype))
            self.saved = None
        return {"best_update": self.best[1] if self.best else None,
                "best_tokens_per_step": self.best[0] if self.best else None, "curve": self.curve,
                "select_records": len(self.select), "stopped": self.stopped}


def snap_train(cfg: dict[str, Any]) -> int:
    import mtp_snapshot as snap
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    began = time.monotonic()
    device = torch.device(cfg.get("device") or "cuda")
    folder = Path(cfg.get("snap_from") or cfg["snap_dir"])
    reader = snap.Reader(folder, device)
    index = reader.index
    train, held = reader.entries("train"), reader.entries("heldout")
    free, total = torch.cuda.mem_get_info()
    say(f"SNAP TRAIN {len(train)} train and {len(held)} held-out record(s) from {folder} "
        f"({index['bytes'] / snap.GiB:.1f} GiB); no target model loaded; GPU {free / 2**30:.1f} of "
        f"{total / 2**30:.1f} GiB free")
    lm_head = torch.load(folder / snap.LM_HEAD).to(device)
    embed = torch.load(folder / snap.EMBED).to(device)
    config_file = folder / "config.json"
    if not config_file.is_file():
        config_file = Path(index["model_dir"]) / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    shape = ref.Shape.from_config(config)
    sources, drafters, notes = load_drafters(cfg, shape)
    heads = own_heads(sources, lm_head, embed, device, notes)
    start_label = "tuned-int4" if "tuned-int4" in drafters else "albucino"
    start = drafters[start_label]
    albucino = sources["albucino"]
    draft_dir = Path(cfg["draft_model"])
    group = int4_group(json.loads((draft_dir / "config.json").read_text(encoding="utf-8")))
    scale_dtype = expert_scale_dtype(draft_dir)
    train_vocab = json.loads(Path(cfg["draft_vocab"]).read_text(encoding="utf-8"))["ids"] \
        if cfg.get("draft_vocab") else None
    span = str(cfg.get("span") or "reply")
    header = list(index["header"]) if span == "assistant" else []
    end_id = int(index["end_id"]) if header else -1
    found = {"embed": embed, "lm_head": lm_head}
    if start.experts in heads:
        found["draft_lm_head"], found["draft_embed"] = heads[start.experts]
    like = albucino.raw
    arms = arm_specs(cfg["snap_arms"])
    plans: dict[str, dict[str, Any]] = {}
    if cfg.get("snap_curve"):
        shares = curve_arms(cfg["snap_curve"])
        plans = curve_plans(train, shares, int(cfg.get("snap_updates") or 0), int(cfg["accum"]), int(cfg["seed"]))
        arms = [("dense", None, name) for name, _ in shares]
        for name, plan in plans.items():
            say(f"curve arm {name}: the first {len(plan['calls'])} of {len(train)} train calls in a fixed shuffled "
                f"order, {plan['updates']} updates at {int(cfg['accum'])} calls each")
    trained: list[str] = []
    select = select_entries(held, int(cfg.get("snap_select_records") or 0)) \
        if int(cfg.get("snap_select_every") or 0) > 0 else []
    if select:
        chosen = {id(one) for one in select}
        held = [one for one in held if id(one) not in chosen]
        say(f"keep-best: every {cfg['snap_select_every']} updates each arm is measured on {len(select)} held-out "
            f"records and its best state is the one exported; the final table uses the other {len(held)}")
    report: dict[str, Any] = {"config": {k: v for k, v in cfg.items() if k != "serve_args"},
                              "snapshot": {k: v for k, v in index.items() if k != "entries"}, "notes": notes,
                              "start": start_label, "arms": {}, "select": [one["key"] for one in select]}
    for at, (kind, rate, label) in enumerate(arms):
        arm = label
        arm_cfg = {**cfg, "lr": rate} if rate is not None else cfg
        left = remaining_s(cfg)
        minutes = float(cfg["snap_minutes"])
        if left is not None:
            afford = (left - MARGIN_S - EVAL_ESTIMATE_S) / 60.0 / (len(arms) - at)
            if afford < minutes:
                say(f"arm {arm} gets {max(0.0, afford):.1f} of the {minutes:g} minutes asked: {left / 60:.1f} min "
                    f"are left before the carried main's hard stop")
                minutes = max(0.0, afford)
        if minutes <= 0:
            say(f"arm {arm} skipped: no minutes left")
            continue
        try:
            arm_report = train_arm(cfg, arm_cfg, kind, label, at, minutes, shape, start, start_label, albucino, found,
                                   train_vocab, header, end_id, train, reader, like, group, scale_dtype, sources,
                                   drafters, trained, device, out, select, plans.get(label), began)
        except Exception as failure:
            arm_report = {"failed": f"{type(failure).__name__}: {failure}"}
            say(f"ARM FAILED {arm}: {type(failure).__name__}: {failure}; the other arms and the evaluation go on")
            print(traceback.format_exc(), flush=True)
            torch.cuda.empty_cache()
        report["arms"][arm] = arm_report
        (out / "snapshot_train_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if plans:
        report["curve_table"] = curve_table(report["arms"])
        (out / "snapshot_train_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        for row in report["curve_table"]:
            say(f"CURVE TABLE {json.dumps(row)}")
    report["train_probe"] = train_probe(cfg, train)
    finish_eval(cfg, shape, albucino, sources, drafters, start_label, trained, lm_head, embed, held, reader, heads,
                report, out, began, device)
    return 0


def train_arm(cfg, arm_cfg, kind, label, at, minutes, shape, start, start_label, albucino, found, train_vocab, header,
              end_id, train, reader, like, group, scale_dtype, sources, drafters, trained, device, out,
              select=(), plan=None, began=0.0) -> dict:
    experts = kind == "experts"
    head = ref.MTPHead(shape, start.dense, albucino.gate_up, albucino.down, train_experts=experts).to(device)
    trainer = base.Trainer(head, found, train_vocab, arm_cfg, None, header, end_id)
    free, _ = torch.cuda.mem_get_info()
    what = f"dense tensors and all {len(albucino.gate_up)} experts" if experts else "dense tensors only"
    say(f"arm {label}: from {start_label}, {what}, {minutes:.1f} min over {len(train)} stored records, "
        f"lr {arm_cfg['lr']:g}" + (f", experts lr {float(arm_cfg.get('expert_lr') or arm_cfg['lr']):g}" if experts
                                   else "") + f"; {free / 2**30:.1f} GiB of the GPU free")
    track = None
    stream = epochs(train, int(cfg["snap_epochs"]), int(cfg["seed"]) + at)
    if plan is not None:
        calls = plan["calls"]
        wanted = int(cfg.get("curve_probe") or 0)
        probe = calls[::max(1, len(calls) // max(1, wanted))][:wanted]
        track = CurveTrack(out / "curve.jsonl", label, len(calls), int(cfg["accum"]), probe,
                           float(cfg["snap_stop_gain"]), int(cfg["snap_stop_evals"]), began)
        stream = curve_stream(plan, int(cfg["accum"]), int(cfg["seed"]) + at)
    keeper = KeepBest(label, head, trainer, list(select), reader.fetch, int(cfg["steps"]), int(cfg["eval_bases"]),
                      track) if select else None
    rounds = base.train_rounds(arm_cfg, head, trainer, None, stream, minutes, fetch=reader.fetch, checkpoint=keeper,
                               every=int(cfg.get("snap_select_every") or 0) if keeper else 0,
                               updates=plan["updates"] if plan is not None else 0)
    if plan is not None:
        rounds.update({"share": plan["share"], "unique_calls": len(plan["calls"])})
    if keeper is not None:
        rounds["keep_best"] = keeper.restore()
        say(f"arm {label}: exports its state at update {rounds['keep_best']['best_update']} of {rounds['updates']} "
            f"({rounds['keep_best']['best_tokens_per_step']} tokens a step on the {len(select)} picking records)")
    dense = {name: value.detach().clone() for name, value in head.dense_state().items()}
    named = f"mtp_snap{label}.safetensors"
    how = f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set TAAF_VLLM_MTP_WEIGHTS={named}"
    arm_report = {**rounds, "export": export(head, like, out / named,
                                             {"source": str(cfg.get("model_repo") or ""), "from": start_label,
                                              "arm": label, "lr": str(arm_cfg["lr"]),
                                              "trained_records": str(rounds["trained"])})}
    if experts:
        gate_up, down = (one.to(torch.bfloat16) for one in head.experts_stacked())
        del trainer, head
        torch.cuda.empty_cache()
        before = albucino.gate_up.to(device)
        drift = float((gate_up.float() - before.float()).norm() / before.float().norm())
        before = None
        rounded_up = torch.stack([ref.int4_roundtrip(one, group) for one in gate_up])
        rounded_down = torch.stack([ref.int4_roundtrip(one, group) for one in down])
        packed_name = f"mtp_snap{label}_experts.safetensors"
        arm_report["expert_relative_change"] = round(drift, 4)
        arm_report["experts_export"] = export_experts(gate_up, down, shape.inter, group, scale_dtype, out / packed_name)
        sources[label] = Source(label, dense, {}, gate_up.cpu(), down.cpu(), "trained experts in BF16")
        sources[label + "-int4"] = Source(label + "-int4", dense, {}, rounded_up.cpu(), rounded_down.cpu(),
                                          f"trained experts through int4 g{group}")
        drafters[label] = Drafter(label, dense, label, None, "not servable as is: the experts are BF16", servable=False)
        drafters[label + "-int4"] = Drafter(label + "-int4", dense, label + "-int4", None,
                                            f"{how}, and the drafter's experts from {packed_name}")
        trained += [label, label + "-int4"]
        say(f"arm {label}: the experts moved {drift:.4f} of their norm")
    else:
        drafters[label] = Drafter(label, dense, "albucino", None, how)
        trained.append(label)
        del trainer, head
    torch.cuda.empty_cache()
    return arm_report


def train_probe(cfg: dict[str, Any], train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = int(cfg.get("train_probe_records") or 0)
    if count <= 0 or not train:
        return []
    step = max(1, len(train) // count)
    return train[::step][:count]


def finish_eval(cfg, shape, albucino, sources, drafters, start_label, trained, lm_head, embed, held, reader, heads,
                report, out, began, device) -> None:
    head = ref.MTPHead(shape, albucino.dense, albucino.gate_up, albucino.down).to(device)
    if device.type == "cuda":
        for source in sources.values():
            if not source.gate_up.is_pinned():
                source.gate_up = source.gate_up.pin_memory()
                source.down = source.down.pin_memory()
    install = Installer(head, sources, "albucino")
    vocabs = load_vocabs(cfg, reader)
    bench = Bench(head, lm_head, embed, vocabs, cfg)
    names = list(vocabs)
    main_vocab = next((one for one in names if one != FULL), FULL)
    main = (main_vocab, bench.temperatures[0])
    order = list(dict.fromkeys(["albucino", start_label, *trained]))
    say(f"SNAP EVAL: {len(held)} held-out records from the snapshot, drafters {order}, {bench.drafts} drafts, at most "
        f"{cfg['eval_bases']} positions a record")
    if cfg.get("kv_only"):
        order = [start_label]
    if cfg.get("kv_only") and cfg.get("kv_quant_views"):
        say("kv_only with KV views: the table pass is the bf16 view, so it is not run twice")
    else:
        result = run_pass("snap eval", bench, None, held, [drafters[one] for one in order], install, heads, main,
                          fetch=reader.fetch)
        result.update({"name": "snap eval", "labels": order})
        rows_now = table([result], drafters, names, bench.temperatures)
        report.update({"eval": result, "table": rows_now})
        say(f"TABLE ({main[0]}; every head drafts from the same int8 snapshot of the cut target's hidden states):")
        for row in rows_now:
            say("TABLE " + table_line(row, bench.drafts, bench.temperatures[0]))
        heads_said = head_lines(result, order, vocabs, bench.temperatures[0], cfg)
        report["lm_head"] = heads_said
        for line in heads_said:
            say(line)
        for label in order:
            for key, title in (("games", "BY GAME"), ("lengths", "BY PROMPT")):
                parts = sorted(one.split("|", 1)[1] for one in result[key] if one.startswith(label + "|"))
                say(f"{title} {label:<11} " + ", ".join(
                    f"{part} {result[key][label + '|' + part].get('tokens_per_step')}" for part in parts)
                    + f" ({main[0]}, T{main[1]:g})")
        decision = decide(rows_now, ("albucino", main_vocab))
        report["decision"] = decision
        if decision:
            say(f"NEXT RUN: {decision['label']} with {decision['vocab']}, {decision['tokens_per_step']} tokens a step "
                f"against {decision['served_now_tokens_per_step']} for albucino; {decision['how']}")
        theory = theory_lines(result, start_label, list(trained), main) if start_label != "albucino" else []
        if start_label != "albucino" and result["views"].get(f"{start_label}|{main[0]} sampled x1|{main[1]:g}"):
            theory += theory_lines(result, start_label, list(trained), (f"{main[0]} sampled x1", main[1]), SAMPLED)
        report["decode_data"] = theory
        for line in theory:
            say(line)
    views = [str(one) for one in cfg.get("kv_quant_views") or []]
    if views:
        report["kv_views"] = {}
        for mode in views:
            head.kv_quant = mode
            result = run_pass(f"kv view {mode or 'bf16'}", bench, None, held, [drafters[start_label]], install, heads,
                              main, fetch=reader.fetch)
            view = result["views"].get(f"{start_label}|{main[0]}|{main[1]:g}") or {}
            games = {one.split("|", 1)[1]: (value.get("teacher_forced") or [None])[0]
                     for one, value in result["games"].items() if one.startswith(start_label + "|")}
            report["kv_views"][mode or "bf16"] = {"view": view, "first_position_by_game": games,
                                                  "sampled": {key: value for key, value in result["views"].items()
                                                              if "sampled" in key}}
            report["kv_views"][mode or "bf16"]["cells"] = result["cells"]
            for line in cell_table(f"CELLS kv {mode or 'bf16'} held-out", result["cells"], start_label):
                say(line)
            say(f"BY MARGIN kv {mode or 'bf16'} {start_label}: " + ", ".join(
                f"{key.split('|', 1)[1]} {value.get('teacher_forced')} ({value.get('bases')})"
                for key, value in sorted(result.get("margins", {}).items()) if key.startswith(start_label + "|")))
            say(f"KV VIEW {mode or 'bf16'} {start_label} on {len(held)} held-out ({main[0]}, T{main[1]:g}, greedy): "
                f"{view.get('tokens_per_step')} tokens a step, conditional {view.get('teacher_forced')}; first "
                f"position by game " + ", ".join(f"{game} {rate}" for game, rate in sorted(games.items())))
        head.kv_quant = ""
        (out / "snapshot_train_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    probe = report.get("train_probe") or []
    if probe:
        report["train_probe"] = {"records": len(probe)}
        for mode in [str(one) for one in cfg.get("kv_quant_views") or []] or [""]:
            head.kv_quant = mode
            result = run_pass(f"train probe {mode or 'bf16'}", bench, None, probe, [drafters[start_label]], install,
                              heads, main, fetch=reader.fetch)
            view = result["views"].get(f"{start_label}|{main[0]}|{main[1]:g}") or {}
            games = {one.split("|", 1)[1]: (value.get("teacher_forced") or [None])[0]
                     for one, value in result["games"].items() if one.startswith(start_label + "|")}
            report["train_probe"][mode or "bf16"] = {"view": view, "first_position_by_game": games,
                                                     "cells": result["cells"]}
            for line in cell_table(f"CELLS train-probe {mode or 'bf16'}", result["cells"], start_label):
                say(line)
            say(f"TRAIN PROBE {mode or 'bf16'} {start_label} on {len(probe)} train records ({main[0]}, "
                f"T{main[1]:g}, greedy): {view.get('tokens_per_step')} tokens a step, conditional "
                f"{view.get('teacher_forced')}; first position by game "
                + ", ".join(f"{game} {rate}" for game, rate in sorted(games.items())))
        head.kv_quant = ""
    report["reads"] = {"records": reader.reads, "seconds": round(reader.seconds, 1)}
    report["minutes"] = round((time.monotonic() - began) / 60, 1)
    (out / "snapshot_train_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    say(f"SNAP TRAIN done in {report['minutes']} min; report {out / 'snapshot_train_report.json'}")


def main() -> int:
    cfg = json.loads(sys.argv[1])
    if cfg.get("mode") == "snap":
        return snap_extract(cfg) if cfg.get("phase") == "extract" else snap_train(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    began = time.monotonic()
    rows = base.records(Path(cfg["corpus"]))
    train, held = held_rows(rows, cfg)
    say(f"corpus: {len(rows)} record(s); held out {len(held)} in {len(sequences(held))} call sequence(s) from games "
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
    wanted = [one.strip() for one in cfg["ab_heads"].split(",") if one.strip()]
    for name, load in loaders:
        if name == "albucino" and not cfg.get("draft_model"):
            refused[name] = "the serve names no TAAF_VLLM_DRAFT_MODEL"
            continue
        if name == "intel" and not any(one in INTEL_NEEDED for one in wanted):
            say(f"source intel not read: no drafter of {wanted} drafts with the checkpoint's own head or experts")
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
    if ("tuned" in drafters or "tuned-int4" in drafters) and "router" in notes.get("tuned", {}):
        install(drafters.get("tuned") or drafters["tuned-int4"])
        report["export_c512"] = export(head, like, out / C512_FILE,
                                       {"source": str(cfg.get("model_repo") or ""), "router": "512 rows"})
    vocabs = load_vocabs(cfg)
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
            found = {"embed": embed, "lm_head": lm_head}
            if drafters[start].experts in heads:
                found["draft_lm_head"], found["draft_embed"] = heads[drafters[start].experts]
            trainer = base.Trainer(head, found, train_vocab, cfg, None, header, end_id)
            say(f"(d): tuning {start} for {minutes:.1f} min on {len(train)} same-weights records, targets "
                f"{trainer.target} (KL {trainer.kl:g}) at T{trainer.temperature:g}, draft vocabulary "
                f"{Path(cfg['draft_vocab']).name if cfg.get('draft_vocab') else 'full'}, {trainer.steps} drafts "
                f"weighted {trainer.weights}, chain {trainer.chain:g}, learning rate on the "
                f"{cfg.get('schedule') or 'records'} clock, drafting with "
                f"{'its own' if 'draft_lm_head' in found else 'the target'}'s lm_head and embedding")
            rounds = base.train_rounds(cfg, head, trainer, extractor, train, minutes)
            tuned = {name: value.detach().clone() for name, value in head.dense_state().items()}
            del trainer
            torch.cuda.empty_cache()
            experts = drafters[start].experts
            named = str(cfg.get("ab_tuned_name") or D_FILE)
            how = (f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set TAAF_VLLM_MTP_WEIGHTS={named}"
                   if experts == "albucino" else f"unset TAAF_VLLM_DRAFT_MODEL and set TAAF_VLLM_MTP_WEIGHTS={named}")
            drafters["d"] = Drafter("d", tuned, experts, None, f"{how} (tuned from {start})")
            report["tune"] = {"from": start, **rounds,
                              "export": export(head, like, out / named,
                                               {"source": str(cfg.get("model_repo") or ""), "from": start,
                                                "trained_records": str(rounds["trained"])})}
            second = ["d"]
            if experts == "intel" and "albucino" in sources:
                drafters["d-int4"] = Drafter("d-int4", tuned, "albucino", None,
                                             f"keep TAAF_VLLM_DRAFT_MODEL on albucino and set "
                                             f"TAAF_VLLM_MTP_WEIGHTS={named}")
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
