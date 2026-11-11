from __future__ import annotations

import base64
import dataclasses
import gzip
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mtp_ref as ref  # noqa: E402

GAME = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
PHASES = ((10, "steps 0-10"), (40, "steps 11-40"), (10**9, "steps 41+"))
LENGTHS = ((20000, "under 20K"), (60000, "20K-60K"), (10**12, "60K+"))
INIT_SEARCH = Path("/kaggle/input")
MIN_TRAIN_BASES = 128
MAX_TRAIN_MISSES = 40
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n"
END_TOKEN = "<|im_end|>"
CAPTURE: list[tuple[torch.Tensor, torch.Tensor]] = []


def say(text: str) -> None:
    print(f"MTP_TUNE {time.strftime('%H:%M:%S')} {text}", flush=True)


def shard_text(corpus: Path, shard: dict[str, Any]) -> bytes:
    packed = corpus / shard["name"]
    if packed.is_file():
        blob = packed.read_bytes()
        if hashlib.sha256(blob).hexdigest() != shard["sha256"]:
            raise SystemExit(f"{packed.name} does not hash to its manifest entry; the corpus is damaged")
        return gzip.decompress(blob)
    plain = packed.with_suffix("") if packed.suffix == ".gz" else packed
    if not plain.is_file():
        raise SystemExit(f"neither {packed.name} nor {plain.name} is under {corpus}")
    if not shard.get("text_sha256"):
        raise SystemExit(f"{plain.name} arrived unpacked and the manifest holds no text_sha256 to check it by")
    text = plain.read_bytes()
    if hashlib.sha256(text).hexdigest() != shard["text_sha256"]:
        raise SystemExit(f"{plain.name} does not hash to its manifest entry; the corpus is damaged")
    return text


def records(corpus: Path) -> list[dict[str, Any]]:
    manifest = json.loads((corpus / "MTP_CORPUS.json").read_text(encoding="utf-8"))
    out = []
    unpacked = 0
    for shard in manifest["shards"]:
        unpacked += not (corpus / shard["name"]).is_file()
        out += [json.loads(line) for line in shard_text(corpus, shard).decode("utf-8").splitlines() if line]
    say(f"corpus {len(manifest['shards'])} shards, {unpacked} mounted unpacked, {len(out)} records, hashes match")
    return out


def engine(model_dir: str, serve_args: list[str], cfg: dict[str, Any]):
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    known, unknown = parser.parse_known_args(["--model", model_dir, *serve_args])
    args = EngineArgs.from_cli_args(known)
    free, total = torch.cuda.mem_get_info()
    asked = float(getattr(args, "gpu_memory_utilization", 0.9) or 0.9)
    fits = math.floor((free - 2**30) / total * 100) / 100
    say(f"GPU {free / 2**30:.2f} of {total / 2**30:.2f} GiB free once the server stopped; the serve asked for "
        f"{asked:g} of it, the engine takes {min(asked, fits):g}")
    moved = {"speculative_config": None, "enforce_eager": True, "max_num_seqs": 1,
             "max_num_batched_tokens": int(cfg["prefill_chunk"]), "enable_chunked_prefill": True,
             "kv_transfer_config": None, "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
             "kv_cache_memory_bytes": int(cfg["kv_gib"] * 2**30),
             "gpu_memory_utilization": min(asked, fits), "prefix_match_unit": None,
             "distributed_executor_backend": "uni", "async_scheduling": False,
             "kv_offloading_size": None}
    for name, value in moved.items():
        if hasattr(args, name):
            setattr(args, name, value)
    say(f"engine from the serve's own flags ({len(unknown)} frontend word(s) ignored); overrides {moved}; "
        f"prefix caching {getattr(args, 'enable_prefix_caching', None)}, max_model_len "
        f"{getattr(args, 'max_model_len', None)}, quantization {getattr(args, 'quantization', None)}")
    stock = EngineArgs(model=model_dir)
    kwargs = {}
    for field in dataclasses.fields(args):
        if not field.init or field.name == "model":
            continue
        value = getattr(args, field.name)
        try:
            same = bool(value == getattr(stock, field.name))
        except Exception:
            same = False
        if not same or field.name in moved:
            kwargs[field.name] = value
    say(f"engine keyword(s) passed: {sorted(kwargs)}")
    began = time.monotonic()
    llm = LLM(model=model_dir, **kwargs)
    say(f"engine up in {(time.monotonic() - began) / 60:.1f} min")
    return llm


def attach(llm) -> dict[str, Any]:
    def grab(model):
        mixer = None
        for name, module in model.named_modules():
            if name.endswith("hyper_connection_mixer") and ".mtp" not in name and "mtp." not in name:
                mixer = (name, module)
        if mixer is None:
            raise RuntimeError("no hyper_connection_mixer in the target model")

        stock = mixer[1].combine_and_mix

        def captured(*args, **kwargs):
            output = stock(*args, **kwargs)
            CAPTURE.append((output[0].detach().clone(), output[1].detach().clone()))
            return output

        mixer[1].combine_and_mix = captured
        embed = lm_head = None
        for name, param in model.named_parameters():
            if name.endswith("embed_tokens.weight") and "visual" not in name and embed is None:
                embed = param
            if name.endswith("lm_head.weight"):
                lm_head = param
        return {"mixer": mixer[0], "embed": embed, "lm_head": lm_head}

    found = llm.apply_model(grab)[0]
    say(f"hooked {found['mixer']}; embed {tuple(found['embed'].shape)} {found['embed'].dtype}; "
        f"lm_head {tuple(found['lm_head'].shape)} {found['lm_head'].dtype}")
    return found


def picture(url: str):
    from PIL import Image
    head, _, data = url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


class Extractor:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.prev_ids: list[int] = []
        self.prev_multi: torch.Tensor | None = None
        self.prev_sample: torch.Tensor | None = None
        self.stats = {"records": 0, "tokens": 0, "computed": 0, "reruns": 0, "mismatch": 0, "seconds": 0.0}
        self.chunks = 0
        self.rows: list[int] = []

    def run(self, record: dict[str, Any], fresh: bool = False):
        from vllm import SamplingParams
        if fresh:
            self.llm.reset_prefix_cache()
        prompt: dict[str, Any] = {"prompt": record["text"]}
        if record["images"]:
            prompt["multi_modal_data"] = {"image": [picture(one) for one in record["images"]]}
        CAPTURE.clear()
        began = time.monotonic()
        out = self.llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)[0]
        torch.cuda.synchronize()
        spent = time.monotonic() - began
        ids = list(out.prompt_token_ids)
        cached = int(getattr(out, "num_cached_tokens", 0) or 0)
        self.chunks = len(CAPTURE)
        self.rows = [int(one[0].shape[0]) for one in CAPTURE]
        multi = torch.cat([one[0] for one in CAPTURE], 0) if CAPTURE else None
        sample = torch.cat([one[1] for one in CAPTURE], 0) if CAPTURE else None
        CAPTURE.clear()
        return ids, cached, multi, sample, spent

    def hidden(self, record: dict[str, Any]):
        ids, cached, multi, sample, spent = self.run(record)
        new = 0 if multi is None else multi.shape[0]
        if new != len(ids) - cached or (cached and (self.prev_multi is None or cached > len(self.prev_ids)
                                                    or self.prev_ids[:cached] != ids[:cached])):
            self.stats["reruns"] += 1
            ids, cached, multi, sample, more = self.run(record, fresh=True)
            spent += more
            if multi is None or multi.shape[0] != len(ids) - cached or cached:
                self.stats["mismatch"] += 1
                if self.stats["mismatch"] <= 3:
                    say(f"capture mismatch on a fresh run: prompt {len(ids)} tokens, {cached} cached, captured "
                        f"{'nothing' if multi is None else f'{multi.shape[0]} rows'} in {self.chunks} forward(s) "
                        f"of {self.rows} rows")
                return None
        if cached:
            multi = torch.cat([self.prev_multi[:cached], multi], 0)
            sample = torch.cat([self.prev_sample[:cached], sample], 0)
        self.prev_ids, self.prev_multi, self.prev_sample = ids, multi, sample
        self.stats["records"] += 1
        self.stats["tokens"] += len(ids)
        self.stats["computed"] += len(ids) - cached
        self.stats["seconds"] += spent
        return torch.tensor(ids, device=multi.device), multi, sample


def reply_start(tokenizer, ids: torch.Tensor, record: dict[str, Any]) -> int | None:
    start = ids.shape[0] - int(record["reply_tokens"])
    said = tokenizer.decode(ids[start:].tolist(), skip_special_tokens=False)
    return start if said == record["text"][-int(record["reply_chars"]):] else None


FOCUS_MODES = ("", "target", "verdict", "drafter")


def focus_rows(given, count: int, cfg: dict[str, Any]) -> torch.Tensor | None:
    mode = str(cfg.get("focus") or "")
    if not mode or given is None:
        return None
    at = float(cfg.get("focus_at") or 0.95)
    keep = float(cfg.get("focus_keep") or 0.0)
    probs = given[0]
    ones = torch.ones(count, dtype=torch.float32, device=probs.device)
    if mode == "target":
        top = probs[:count, 0].float()
        return torch.where(top >= at, torch.full_like(ones, keep), ones)
    if len(given) < 3:
        return None
    later = torch.cat([given[2][1:count], torch.full((1, 2), -1.0, device=probs.device)], 0)
    confident = later[:, 1] == 1
    if mode == "drafter":
        confident = confident & (later[:, 0] >= at)
    return torch.where(confident, torch.full_like(ones, keep), ones)


def vocab_rows(lm_head: torch.Tensor, vocab: list[int] | None):
    rows = torch.tensor(vocab, device=lm_head.device) if vocab else None
    weights = lm_head.detach().index_select(0, rows) if rows is not None else lm_head.detach()
    size = lm_head.shape[0]
    column = torch.full((size,), -1, dtype=torch.long, device=lm_head.device)
    if rows is not None:
        column[rows] = torch.arange(rows.numel(), device=lm_head.device)
    else:
        column = torch.arange(size, device=lm_head.device)
    return rows, weights, column


def assistant_ranges(ids: torch.Tensor, header: list[int], end_id: int) -> list[tuple[int, int]]:
    count = ids.shape[0]
    width = len(header)
    if not header or count <= width:
        return []
    windows = ids.unfold(0, width, 1)
    starts = (windows == torch.tensor(header, device=ids.device)).all(-1).nonzero().flatten() + width
    ends = (ids == end_id).nonzero().flatten()
    out = []
    for first in starts.tolist():
        at = int(torch.searchsorted(ends, torch.tensor(first, device=ids.device)))
        last = int(ends[at]) if at < ends.numel() else count - 1
        if first <= last:
            out.append((first - 1, min(last - 1, count - 2)))
    return out


class Trainer:
    def __init__(self, head: ref.MTPHead, found: dict[str, Any], vocab: list[int] | None, cfg: dict[str, Any],
                 serve_vocab: list[int] | None = None, header: list[int] | None = None, end_id: int = -1):
        self.head = head
        self.cfg = cfg
        self.embed = (found["draft_embed"] if found.get("draft_embed") is not None else found["embed"]).detach()
        lm_head = found["lm_head"].detach()
        self.full = lm_head
        drafting = found["draft_lm_head"].detach() if found.get("draft_lm_head") is not None else lm_head
        self.vocab, self.rows, self.column = vocab_rows(drafting, vocab)
        self.extra_vocab = None
        if serve_vocab is not None and serve_vocab != vocab:
            ids, weights, _ = vocab_rows(drafting, serve_vocab)
            self.extra_vocab = (ids, weights)
        self.steps = int(cfg["steps"])
        self.weights = [float(one) for one in cfg["step_weights"]][:self.steps]
        self.temperature = float(cfg.get("temperature") or GAME["temperature"])
        self.temperatures = [float(one) for one in cfg.get("eval_temperatures") or [] if float(one) != self.temperature]
        self.target = str(cfg.get("target") or "game")
        self.sampled_taus = [float(one) for one in cfg.get("sampled_taus") or []]
        self.kl = float(cfg.get("kl") or 0.0)
        self.chain = float(cfg.get("chain") or 0.0)
        self.header = list(header or [])
        self.end_id = int(end_id)
        self.calib_last: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    def targets(self, sample: torch.Tensor, first: int, last: int, temperature: float | None = None):
        probs, ids = [], []
        for start in range(first, last + 1, 1024):
            stop = min(last + 1, start + 1024)
            logits = F.linear(sample[start:stop], self.full)
            p, i = ref.game_distribution(logits, self.temperature if temperature is None else temperature,
                                         GAME["top_k"], GAME["top_p"])
            probs.append(p)
            ids.append(i)
        return torch.cat(probs, 0), torch.cat(ids, 0)

    def targets_at(self, sample: torch.Tensor, at: torch.Tensor, temperature: float):
        probs, ids = [], []
        for start in range(0, at.numel(), 1024):
            logits = F.linear(sample[at[start:start + 1024]], self.full)
            p, i = ref.game_distribution(logits, temperature, GAME["top_k"], GAME["top_p"])
            probs.append(p)
            ids.append(i)
        return torch.cat(probs, 0), torch.cat(ids, 0)

    def suffixes(self) -> list[tuple[str, float | None, bool]]:
        out = [(f" T{one:g}", one, False) for one in self.temperatures]
        out += [(f" sampled x{one:g}", None, False) for one in self.sampled_taus]
        if self.extra_vocab is not None:
            out += [(" serve-vocab", None, True)]
            out += [(f" serve-vocab T{one:g}", one, True) for one in self.temperatures]
        return out

    def plan(self, ids: torch.Tensor, start: int, steps: int, max_bases: int, span: str, late: float,
             allow: torch.Tensor | None = None):
        count = ids.shape[0]
        first_target = start - 1
        if span == "reply" and not late:
            bases = torch.arange(max(0, first_target - steps), count - 2, device=ids.device)
            if allow is not None:
                bases = bases[allow[(bases + 1).clamp(max=count - 1)]]
            if bases.numel() > max_bases:
                keep = torch.randperm(bases.numel(), device=ids.device)[:max_bases].sort().values
                bases = bases[keep]
            return bases, None
        ranges = [(first_target, count - 2)]
        if span == "assistant":
            ranges += assistant_ranges(ids, self.header, self.end_id)
        floor = max(0, count - 2 - int(self.cfg["context"]))
        generated = torch.zeros(count, dtype=torch.bool, device=ids.device)
        for first, last in ranges:
            if last >= max(first, floor):
                generated[max(first, floor):last + 1] = True
        bases = (generated.nonzero().flatten() - 1).clamp(min=0).unique()
        bases = bases[bases <= count - 3]
        if allow is not None:
            bases = bases[allow[(bases + 1).clamp(max=count - 1)]]
        if bases.numel() > max_bases:
            weight = ((bases.double() + 1) / count) ** float(late) if late else torch.ones_like(bases, dtype=torch.double)
            keep = torch.multinomial(weight, max_bases, replacement=False).sort().values
            bases = bases[keep]
        return bases, generated

    def pass_one(self, ids: torch.Tensor, multi: torch.Tensor, sample: torch.Tensor | None, start: int,
                 steps: int, train: bool, max_bases: int, span: str = "reply", late: float = 0.0,
                 given: tuple[torch.Tensor, torch.Tensor] | None = None):
        count = ids.shape[0]
        first_target = start - 1
        focus = focus_rows(given, count, self.cfg) if train else None
        allow = focus > 0 if focus is not None and not float(self.cfg.get("focus_keep") or 0.0) else None
        bases, generated = self.plan(ids, start, steps, max_bases, span, late, allow)
        if bases.numel() == 0:
            return None
        cap = int(self.cfg["context"])
        ratio = self.head.shape.ratio
        low = max(0, int(bases[0]) - cap)
        low -= low % ratio
        reach = int(self.cfg.get("grad_context") or 0)
        grad_from = max(0, count - 1 - low - reach) if reach else 0
        if given is not None and self.temperatures and not train:
            raise ValueError("stored targets hold one temperature; evaluate other temperatures on a live target")
        if generated is None:
            probs, tids = (given[0][first_target:count - 1].float(), given[1][first_target:count - 1].long()) \
                if given is not None else self.targets(sample, first_target, count - 2)
            row_of = None
            others = {} if train or given is not None else \
                {one: self.targets(sample, first_target, count - 2, one) for one in self.temperatures}
        else:
            wanted = (bases[:, None] + torch.arange(1, steps + 1, device=ids.device)[None, :]).clamp(max=count - 2)
            at = wanted.flatten().unique()
            row_of = torch.full((count,), -1, dtype=torch.long, device=ids.device)
            row_of[at] = torch.arange(at.numel(), device=ids.device)
            probs, tids = (given[0][at].float(), given[1][at].long()) if given is not None \
                else self.targets_at(sample, at, self.temperature)
            others = {} if train or given is not None else \
                {one: self.targets_at(sample, at, one) for one in self.temperatures}
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = self.head.forward_steps(multi[low:], ids[low:], self.embed, bases, steps, first_pos=low,
                                               chunk=int(self.cfg["attn_chunk"]),
                                               share_index=bool(self.cfg["share_index"]), grad_from=grad_from)
        losses, rates, masks = [], [], []
        extra: dict[str, list[torch.Tensor]] = {name: [] for name, _, _ in self.suffixes()}
        greedy: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        hits: list[torch.Tensor] = []
        alive = None
        for step, out in enumerate(outs):
            target = bases + step + 1
            if row_of is None:
                mask = (target >= first_target) & (target <= count - 2)
                row = (target - first_target).clamp(min=0, max=probs.shape[0] - 1)
            else:
                mask = (target <= count - 2) & generated[target.clamp(max=count - 1)]
                row = row_of[target.clamp(max=count - 2)]
            p, t = probs[row], tids[row]
            logits = F.linear(out.float(), self.rows.float())
            rate = ref.acceptance(logits.detach(), p, t, self.vocab)
            if self.target == "tv":
                rate = ref.sampled_acceptance(logits.detach(), p, self.column[t], self.temperature)
            rates.append(rate)
            masks.append(mask)
            if train:
                weight = torch.ones_like(rate, dtype=torch.float32) if alive is None or not self.chain                     else alive.float().clamp(min=0.0) ** self.chain
                share = mask.float()
                if focus is not None:
                    term = focus[target.clamp(max=count - 1)]
                    weight = weight * term
                    share = share * term
                if self.target == "argmax":
                    col = self.column[t[:, 0]]
                    keep = mask & (col >= 0)
                    ce = F.cross_entropy(logits, col.clamp(min=0), reduction="none")
                    held = keep.float() if focus is None else keep * term
                    loss = (ce * keep * weight).sum() / held.sum().clamp(min=1e-6 if focus is not None else 1)
                    if self.kl:
                        soft = ref.soft_ce(logits, p, self.column[t])
                        loss = loss + self.kl * (soft * mask * weight).sum() / share.sum().clamp(
                            min=1e-6 if focus is not None else 1)
                    losses.append(loss)
                elif self.target == "tv":
                    kept = ref.sampled_acceptance(logits, p, self.column[t], self.temperature)
                    loss = ((1.0 - kept) * mask * weight).sum() / share.sum().clamp(min=1e-6 if focus is not None
                                                                                    else 1)
                    if self.kl:
                        soft = ref.soft_ce(logits, p, self.column[t])
                        loss = loss + self.kl * (soft * mask * weight).sum() / share.sum().clamp(
                            min=1e-6 if focus is not None else 1)
                    losses.append(loss)
                else:
                    ce = ref.soft_ce(logits, p, self.column[t])
                    losses.append((ce * mask * weight).sum() / share.sum().clamp(min=1e-6 if focus is not None
                                                                                  else 1))
                alive = rate.detach() if alive is None else alive * rate.detach()
            else:
                hit = rate
                if self.target == "tv":
                    hit = ref.acceptance(logits.detach(), p, t, self.vocab)
                    greedy.append(hit)
                confs.append(ref.draft_conf(logits.detach(), self.temperature)[mask])
                hits.append(hit[mask])
                served = None
                for name, temperature, serve in self.suffixes():
                    if serve and served is None:
                        served = F.linear(out.float(), self.extra_vocab[1].float())
                    pick = served if serve else logits
                    held_p, held_t = (p, t) if temperature is None else (others[temperature][0][row],
                                                                           others[temperature][1][row])
                    if name.startswith(" sampled x"):
                        tau = float(name[len(" sampled x"):])
                        extra[name].append(ref.sampled_acceptance(logits.detach(), held_p, self.column[held_t],
                                                                  self.temperature * tau))
                        continue
                    extra[name].append(ref.acceptance(pick.detach(), held_p, held_t,
                                                      self.extra_vocab[0] if serve else self.vocab))
        full = masks[-1]
        for one in masks:
            full = full & one
        first_rows = (bases + 1 - first_target).clamp(min=0, max=probs.shape[0] - 1) if row_of is None \
            else row_of[(bases + 1).clamp(max=count - 2)]
        best = probs[first_rows][:, 0]
        more = {}
        for name, temperature, _ in self.suffixes():
            if extra[name]:
                ceiling = best if temperature is None else others[temperature][0][first_rows][:, 0]
                more[name] = ([one[full] for one in extra[name]], ceiling[full])
        if greedy:
            more[" greedy"] = ([one[full] for one in greedy], best[full])
        if not train:
            self.calib_last = (confs, hits)
        return losses, [one[full] for one in rates], best[full], [(one[m], m) for one, m in zip(rates, masks)], more


def schedule(update: int, total: int, warm: int, clock: float | None = None) -> float:
    if update < warm:
        return (update + 1) / warm
    progress = min(1.0, (update - warm) / max(1, total - warm)) if clock is None else min(1.0, max(0.0, clock))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def phase_of(record: dict[str, Any]) -> str:
    step = int(record.get("step") or 0)
    return next(label for limit, label in PHASES if step <= limit)


def length_of(record: dict[str, Any]) -> str:
    tokens = int(record.get("prompt_tokens") or 0)
    return next(label for limit, label in LENGTHS if tokens <= limit)


def groups_of(record: dict[str, Any]) -> list[str]:
    return [f"run {record.get('run')}", f"phase {phase_of(record)}", f"prompt {length_of(record)}",
            f"game {record.get('game')}"]


def chain(summary: dict[str, Any], drafts: int) -> float | None:
    positions = summary.get("per_position") or []
    return round(1.0 + sum(positions[:drafts]), 4) if positions else None


def breakdown(groups: dict[str, dict[str, Any]], labels: list[str]) -> list[str]:
    lines = []
    for key in sorted(groups):
        held = groups[key]
        parts = [f"{label} {chain(held[label].summary(), 3)}" for label in labels]
        lines.append(f"{key}: {held[labels[0]].bases} positions; tokens a step at three drafts " + ", ".join(parts))
    return lines


def evaluate(name: str, trainer: Trainer, extractor: Extractor, rows: list[dict[str, Any]], steps: int,
             swap: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    tallies = {label: ref.Tally(steps) for label in swap}
    names = [one for one, _, _ in trainer.suffixes()]
    extras = {label + one: ref.Tally(steps) for label in swap for one in names}
    groups: dict[str, dict[str, Any]] = {}
    began = time.monotonic()
    short = 0
    for index, record in enumerate(rows):
        got = extractor.hidden(record)
        if got is None:
            continue
        ids, multi, sample = got
        start = reply_start(extractor.tokenizer, ids, record)
        if start is None:
            continue
        for label, state in swap.items():
            load_dense(trainer.head, state)
            try:
                result = trainer.pass_one(ids, multi, sample, start, steps, False, int(trainer.cfg["eval_bases"]))
            except torch.OutOfMemoryError:
                result = None
                short += 1
                torch.cuda.empty_cache()
                say(f"{name}: out of GPU memory evaluating {label} on a {len(ids):,}-token record ({short} so far); "
                    f"that record is left out of {label}")
            if result is not None:
                tallies[label].add(result[1], result[2])
                for one, (rates, best) in result[4].items():
                    extras[label + one].add(rates, best)
                for key in groups_of(record):
                    groups.setdefault(key, {one: ref.Tally(steps) for one in swap})[label].add(result[1], result[2])
        if index % 25 == 24:
            say(f"{name} {index + 1}/{len(rows)} " + " | ".join(
                f"{label} {tallies[label].summary()}" for label in swap)
                + f" ({time.monotonic() - began:.0f}s)")
            for label, tally in extras.items():
                say(f"{name} {index + 1}/{len(rows)} {label} {tally.summary()}")
        if index % 100 == 99 and index + 1 < len(rows):
            for line in breakdown(groups, list(swap)):
                say(f"{name} so far ({index + 1}/{len(rows)}) by {line}")
    lines = breakdown(groups, list(swap))
    for line in lines:
        say(f"{name} by {line}")
    out = {label: tally.summary() for label, tally in tallies.items()}
    out.update({label: tally.summary() for label, tally in extras.items()})
    out["groups"] = {key: {label: tally.summary() for label, tally in held.items()} for key, held in groups.items()}
    out["out_of_memory"] = short
    return out


def warm_source(source: dict[str, torch.Tensor], said: str) -> dict[str, torch.Tensor]:
    path = Path(said)
    if not path.is_absolute():
        found = sorted(INIT_SEARCH.rglob(said)) if INIT_SEARCH.is_dir() else []
        if len(found) != 1:
            raise SystemExit(f"init {said} matches {len(found)} file(s) under {INIT_SEARCH}, not one")
        path = found[0]
    from safetensors.torch import load_file
    tuned = load_file(str(path))
    out = dict(source)
    for name, value in tuned.items():
        if name not in out:
            raise SystemExit(f"init {path.name} carries {name}, which the checkpoint's draft model does not")
        if tuple(value.shape) != tuple(out[name].shape) or value.dtype != out[name].dtype:
            raise SystemExit(f"init {path.name}: {name} is {tuple(value.shape)} {value.dtype}, the checkpoint "
                             f"{tuple(out[name].shape)} {out[name].dtype}")
        out[name] = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    say(f"warm start: {len(tuned)} of the draft model's tensors from {path.name} (sha256 {digest}); the stock head "
        f"stays as the 'stock' label of every evaluation")
    return out


def load_dense(head: ref.MTPHead, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in state.items():
            head.w[ref.key(name)].copy_(value)


def optimizer(cfg: dict[str, Any], head: ref.MTPHead) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    params = head.trainable()
    experts = [one for one in head.expert_parameters() if one.requires_grad]
    held = {id(one) for one in experts}
    dense = [one for one in params if id(one) not in held]
    groups = [{"params": dense, "lr": float(cfg["lr"]), "scale": 1.0}]
    if experts:
        ratio = float(cfg.get("expert_lr") or cfg["lr"]) / float(cfg["lr"])
        groups.append({"params": experts, "lr": float(cfg["lr"]) * ratio, "scale": ratio})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True), params


def train_rounds(cfg: dict[str, Any], head: ref.MTPHead, trainer: Trainer, extractor: Extractor | None,
                 train: list[dict[str, Any]], minutes: float, fetch=None, checkpoint=None,
                 every: int = 0) -> dict[str, Any]:
    if checkpoint is not None and every > 0:
        checkpoint(0)
    last_checked = 0
    opt, params = optimizer(cfg, head)
    accum = int(cfg["accum"])
    total = max(1, len(train) // accum)
    budget = float(minutes) * 60
    tally = ref.Tally(int(cfg["steps"]))
    since, since_losses = ref.Tally(int(cfg["steps"])), []
    window: list[float] = []
    update = 0
    trained = 0
    bases = int(cfg["train_bases"])
    misses = 0
    train_began = time.monotonic()
    free, total_memory = torch.cuda.mem_get_info()
    say(f"training starts with {free / 2**30:.2f} of {total_memory / 2**30:.2f} GiB of the GPU free, "
        f"{bases} positions a record")
    for index, record in enumerate(train):
        if time.monotonic() - train_began > budget:
            say(f"training budget of {minutes:g} min spent after {index} record(s)")
            break
        given = None
        if fetch is None:
            got = extractor.hidden(record)
            if got is None:
                continue
            ids, multi, sample = got
            start = reply_start(extractor.tokenizer, ids, record)
            if start is None:
                extractor.stats["mismatch"] += 1
                continue
        else:
            got = fetch(record)
            if got is None:
                continue
            ids, multi, start, given = got
            sample = None
        length = len(ids)
        short = counted = False
        try:
            result_one = trainer.pass_one(ids, multi, sample, start, int(cfg["steps"]), True, bases,
                                          str(cfg.get("span") or "reply"), float(cfg.get("late") or 0.0),
                                          given)
            if result_one is not None:
                losses, rates, best, _, _ = result_one
                loss = sum(w * one for w, one in zip(trainer.weights, losses)) / accum
                loss.backward()
                window.append(float(loss.detach()) * accum)
                since_losses.append(float(loss.detach()) * accum)
                tally.add(rates, best)
                since.add(rates, best)
                trained += 1
                counted = True
        except torch.OutOfMemoryError:
            short = True
        got = ids = multi = sample = given = result_one = losses = loss = None
        if short:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            misses += 1
            bases = max(MIN_TRAIN_BASES, bases // 2)
            free, _ = torch.cuda.mem_get_info()
            say(f"out of GPU memory on a {length:,}-token record (miss {misses}); {free / 2**30:.2f} GiB free "
                f"after emptying the cache; training on {bases} positions a record from here")
            if misses > MAX_TRAIN_MISSES:
                say(f"{misses} records ran out of memory; training stops")
                break
            continue
        if not counted:
            continue
        if trained % accum == 0:
            clock = (time.monotonic() - train_began) / budget if cfg.get("schedule") == "time" and budget > 0 else None
            for group in opt.param_groups:
                group["lr"] = float(cfg["lr"]) * group.get("scale", 1.0) * schedule(update, total,
                                                                                  int(cfg["warmup"]), clock)
            norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["clip"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            if checkpoint is not None and every > 0 and update % every == 0:
                trainer.recent = {"loss": sum(since_losses) / max(1, len(since_losses)), "running": since.summary()}
                since, since_losses = ref.Tally(int(cfg["steps"])), []
                checkpoint(update)
                last_checked = update
            if update % int(cfg["log_every"]) == 0:
                source = (f"extract {extractor.stats['computed'] / max(1e-9, extractor.stats['seconds']):,.0f} tok/s"
                          if extractor is not None else "from the snapshot")
                say(f"train {index + 1}/{len(train)} update {update} loss {sum(window) / len(window):.4f} "
                    f"grad {float(norm):.3f} lr {opt.param_groups[0]['lr']:.2e} running {tally.summary()} "
                    f"{source} elapsed {(time.monotonic() - train_began) / 60:.1f} min")
                window = []
                tally = ref.Tally(int(cfg["steps"]))
    if checkpoint is not None and every > 0 and update != last_checked:
        checkpoint(update)
    spent = (time.monotonic() - train_began) / 60
    say(f"trained {trained} record(s) in {update} update(s), {spent:.1f} min"
        + (f"; extraction {extractor.stats}" if extractor is not None else ""))
    return {"trained": trained, "updates": update, "minutes": round(spent, 1), "misses": misses}


def main() -> int:
    cfg = json.loads(sys.argv[1])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    began = time.monotonic()
    rows = records(Path(cfg["corpus"]))
    train = [one for one in rows if one["split"] == "train"]
    held = [one for one in rows if one["split"] == "heldout"]
    if cfg["eval_records"] and len(held) > cfg["eval_records"]:
        held = random.Random(int(cfg["seed"])).sample(held, int(cfg["eval_records"]))
        held.sort(key=lambda one: (one["run"], one["stem"]))
    if cfg.get("skip_records"):
        train = train[int(cfg["skip_records"]):]
    if cfg["train_records"]:
        train = train[:int(cfg["train_records"])]
    say(f"corpus: {len(rows)} record(s); training on {len(train)} "
        f"({sum(one['reply_tokens'] for one in train):,} reply tokens), evaluating on {len(held)} held-out "
        f"({sum(one['reply_tokens'] for one in held):,} reply tokens)")
    model_dir = Path(cfg["model_dir"])
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    inventory = ref.mtp_inventory(model_dir)
    kinds: dict[str, int] = {}
    for spec in inventory.values():
        kinds[spec["dtype"]] = kinds.get(spec["dtype"], 0) + 1
    say(f"checkpoint MTP tensors: {len(inventory)} by dtype {kinds}; e.g. " + "; ".join(
        f"{name} {spec['dtype']} {spec['shape']}" for name, spec in list(sorted(inventory.items()))[:4]))
    source = ref.read_tensors(model_dir, sorted(inventory), inventory, "cpu")
    stock = None
    if cfg.get("init"):
        stock_head, _ = ref.build_head(dict(source), ref.Shape.from_config(config), bool(cfg["train_experts"]))
        stock = {name: value.detach().clone() for name, value in stock_head.dense_state().items()}
        del stock_head
        source = warm_source(source, cfg["init"])
    head, report = ref.build_head(dict(source), ref.Shape.from_config(config), bool(cfg["train_experts"]))
    head = head.cuda()
    say(f"MTP head built: {report}")
    vocab = None
    if cfg.get("draft_vocab"):
        vocab = json.loads(Path(cfg["draft_vocab"]).read_text(encoding="utf-8"))["ids"]
        say(f"draft vocabulary: {len(vocab)} ids from {Path(cfg['draft_vocab']).name}, drafts are argmax over these "
            f"rows as the serve does")
    serve_vocab = None
    if cfg.get("serve_vocab"):
        serve_vocab = json.loads(Path(cfg["serve_vocab"]).read_text(encoding="utf-8"))["ids"]
        say(f"the running serve's draft vocabulary ({len(serve_vocab)} ids, {Path(cfg['serve_vocab']).name}) is "
            f"evaluated beside it as 'serve-vocab'")
    llm = engine(str(model_dir), cfg["serve_args"], cfg)
    found = attach(llm)
    tokenizer = llm.get_tokenizer()
    extractor = Extractor(llm, tokenizer)
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False) if cfg.get("span") == "assistant" else []
    end_id = int(tokenizer.convert_tokens_to_ids(END_TOKEN)) if header else -1
    if header:
        say(f"positions from every assistant turn: header {ASSISTANT_HEADER!r} is token(s) {header}, a turn ends at "
            f"{END_TOKEN} ({end_id})")
    trainer = Trainer(head, found, vocab, cfg, serve_vocab, header, end_id)
    say(f"targets {trainer.target} (KL weight {trainer.kl:g}) at temperature {trainer.temperature:g}, top_k "
        f"{GAME['top_k']}, top_p {GAME['top_p']}; held-out acceptance also at {trainer.temperatures or 'no other'} "
        f"temperature(s); window {cfg['context']} tokens with gradients through the newest "
        f"{cfg.get('grad_context') or 'all'}; training positions from {cfg.get('span') or 'reply'} text, weighted "
        f"by (position / length) ** {float(cfg.get('late') or 0):g}")
    original = {name: value.clone() for name, value in head.dense_state().items()}
    eval_steps = max(int(cfg["steps"]), int(cfg["eval_steps"]))
    result: dict[str, Any] = {"config": {k: v for k, v in cfg.items() if k not in ("serve_args",)},
                              "corpus": {"train": len(train), "heldout": len(held)}}
    if cfg["mode"] == "eval":
        result["eval"] = evaluate("EVAL", trainer, extractor, held, eval_steps,
                                  {"before": original, **({"stock": stock} if stock else {})})
        say(f"EVAL before {result['eval']['before']}")
        say(f"extraction {extractor.stats}")
        (out / "mtp_tune_report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        return 0
    rounds = train_rounds(cfg, head, trainer, extractor, train, float(cfg["train_minutes"]))
    tuned = {name: value.clone() for name, value in head.dense_state().items()}
    trained, update = rounds["trained"], rounds["updates"]
    exported = {name: value.cpu() for name, value in
                ref.export_tensors(head, source, include_experts=bool(cfg["train_experts"])).items()}
    from safetensors.torch import save_file
    meta = {"source": str(cfg.get("model_repo") or ""), "trained_records": str(trained),
            "updates": str(update), "steps": str(cfg["steps"])}
    path = out / "mtp_tuned.safetensors"
    save_file(exported, str(path), metadata=meta)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    say(f"wrote {path.name}: {len(exported)} tensor(s), {path.stat().st_size / 2**20:.1f} MiB, sha256 {digest[:16]}")
    result["export"] = {"file": path.name, "tensors": len(exported), "sha256": digest}
    if cfg.get("vocab_file"):
        paired = out / Path(cfg["vocab_file"]).name
        paired.write_bytes(Path(cfg["vocab_file"]).read_bytes())
        result["export"]["draft_vocab"] = paired.name
        say(f"wrote {paired.name} beside it: the draft vocabulary this head was tuned for "
            f"(sha256 {hashlib.sha256(paired.read_bytes()).hexdigest()[:16]})")
    result["eval"] = evaluate("EVAL", trainer, extractor, held, eval_steps,
                              {**({"stock": stock} if stock else {}), "before": original, "after": tuned})
    say(f"EVAL before {result['eval']['before']}")
    say(f"EVAL after  {result['eval']['after']}")
    for label in sorted(key for key in result["eval"] if " " in key):
        say(f"EVAL {label} {result['eval'][label]}")
    result["extraction"] = extractor.stats
    result["minutes"] = round((time.monotonic() - began) / 60, 1)
    (out / "mtp_tune_report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
