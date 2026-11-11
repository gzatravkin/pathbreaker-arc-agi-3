import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOP = 20
SMALL = 0.01
BASE_CHUNK = 1024
PARTS = ("outside the draft vocabulary", "outside the drafter's top-20", f"0 < q < {SMALL:g}", f"q >= {SMALL:g}")
BLOCK_LAMBDAS = (1.0, 1.5, 2.0)
BLOCK_SAMPLES = 64
TV_SAMPLES = 256
STARTED = time.monotonic()


def say(text):
    print(f"DIAG [{(time.monotonic() - STARTED) / 60:5.1f} min] {text}", flush=True)


def tokens_each(rates):
    return 1.0 + np.cumprod(rates, axis=-1).sum(-1)


def paired(diff, games):
    diff = np.asarray(diff, dtype=np.float64)
    naive = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    means = [diff[games == one].mean() for one in sorted(set(games.tolist()))]
    cluster = np.std(means, ddof=1) / math.sqrt(len(means)) if len(means) > 1 else float("nan")
    return float(diff.mean()), float(naive), float(cluster), len(means)


def position_stats(logits, probs, cols, temperature):
    values, index = logits.float().topk(TOP, dim=-1)
    q = torch.softmax(values / temperature, dim=-1)
    p = probs.float()
    valid = cols >= 0
    match = (cols[:, :, None] == index[:, None, :]) & valid[:, :, None]
    q_at = (match.float() * q[:, None, :]).sum(-1)
    accept = torch.minimum(p, q_at).sum(-1)
    short = (p - q_at).clamp(min=0.0)
    parts = torch.stack([(short * ~valid).sum(-1), (short * (valid & (q_at <= 0))).sum(-1),
                         (short * ((q_at > 0) & (q_at < SMALL))).sum(-1), (short * (q_at >= SMALL)).sum(-1)], -1)
    entropy_q = -(q * q.clamp(min=1e-12).log()).sum(-1)
    entropy_p = -(p * p.clamp(min=1e-12).log()).sum(-1)
    p_at_q = (match.float() * p[:, :, None]).sum(1)
    return accept, parts, entropy_q, entropy_p, (q, p_at_q, p, q_at, match)


def main():
    import block_verify as bv
    import capture_snapshot as cap
    import dflash_eval
    import draft_ab_worker as worker
    import mtp_ref as ref
    import mtp_snapshot as snap
    import mtp_tune
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--vocab", default=str(HERE / "draft_vocab.json"))
    parser.add_argument("--out", default="/kaggle/working/diag")
    parser.add_argument("--heldout", type=int, default=7)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--train-minutes", type=float, default=10.0)
    parser.add_argument("--every-minutes", type=float, default=2.0)
    parser.add_argument("--squeeze", type=float, default=0.7)
    parser.add_argument("--kl", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weights", default="1,0.66,0.38,0.16")
    parser.add_argument("--probe", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--train-snapshot", default="")
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--export", default="mtp_big.safetensors")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--export-every", action="store_true")
    parser.add_argument("--eval-heads", default="")
    parser.add_argument("--train-bases", type=int, default=512)
    parser.add_argument("--select", choices=("pick", "pick+report"), default="pick")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    capture, model = Path(args.capture), Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((capture / "snapshot.json").read_text(encoding="utf-8"))
    hc, width = int(index["hc"]), int(index["width"])
    group = snap.group_for(width, hc)
    by_rid = cap.session_games(capture)
    owner = [cap.owner_of(held.get("rid", ""), by_rid) for held in index["segments"]]
    games = sorted(set(owner) - set(cap.NOT_GAMES))
    held_games = cap.heldout_of(games, args.heldout / max(1, len(games)))
    held_sorted = sorted(held_games)
    pick_games = set(held_sorted[: len(held_sorted) // 2])
    report_games = held_games - pick_games
    train_games = set(games) - held_games
    say(f"{capture}: {len(index['segments'])} segments of {len(games)} games; train {sorted(train_games)}; held out "
        f"{held_sorted} (keep-best picks on {sorted(pick_games)}, the after-training numbers are read on "
        f"{sorted(report_games)})")

    embed, lm_head = dflash_eval.load_target_rows(model, device)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    shape = ref.Shape.from_config(config)
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))["ids"]
    cfg = {**mtp_tune.DEFAULTS, "draft_model": args.draft_model, "tuned_file": args.tuned,
           "ab_heads": "albucino,tuned-int4" + ("" if args.no_bf16 else ",tuned"), "draft_vocab": args.vocab,
           "temperature": args.temperature, "target": "tv", "kl": args.kl, "lr": args.lr, "steps": args.steps,
           "step_weights": [float(one) for one in args.weights.split(",")], "context": args.context,
           "attn_chunk": 32, "train_bases": args.train_bases, "accum": 4, "warmup": 20, "schedule": "time",
           "grad_context": 8192, "log_every": 25, "saliency": "", "clip": args.clip, "share_index": False}
    sources = {"albucino": worker.load_albucino(Path(args.draft_model), shape)}
    if not args.no_bf16:
        sources["intel"] = worker.load_intel(model, config)
        say(f"intel source: {sources['intel'].note}")
    say(f"albucino source: {sources['albucino'].note}")
    refused = {}
    drafters, notes = worker.build_drafters(cfg, sources, refused)
    say(f"drafters {sorted(drafters)}; refused {refused}")
    heads = worker.own_heads(sources, lm_head, embed, device, notes)
    albucino = sources["albucino"]
    head = ref.MTPHead(shape, albucino.dense, albucino.gate_up, albucino.down).to(device)
    install = worker.Installer(head, {name: source for name, source in sources.items()}, "albucino")
    found = {"embed": embed, "lm_head": lm_head}
    if "albucino" in heads:
        found["draft_lm_head"], found["draft_embed"] = heads["albucino"]
    trainer = worker.base.Trainer(head, found, vocab, cfg, None, [], -1)
    temperature = float(trainer.temperature)
    column = trainer.column

    def rows_for(drafter):
        own_head, own_embed = heads.get(drafter.experts, (lm_head, embed))
        _, rows, _ = worker.base.vocab_rows(own_head.detach(), vocab)
        return rows, own_embed.detach()

    packed = {}
    for seg, held in enumerate(index["segments"]):
        rows = int(held["rows"])
        stem = capture / f"seg{seg:05d}"
        if owner[seg] not in games or not Path(f"{stem}.dr").is_file():
            continue
        dr = np.fromfile(f"{stem}.dr", dtype=np.dtype(snap.DR_FIELDS), count=rows)
        verdict = np.where(dr["acc"] == snap.DR_NONE, -1.0, (dr["acc"] == 1).astype(np.float32))
        packed[seg] = {
            "rows": rows, "game": owner[seg],
            "ids": torch.from_numpy(np.fromfile(f"{stem}.ids", dtype=np.int32, count=rows).astype(np.int64)).to(device),
            "q": torch.from_numpy(np.fromfile(f"{stem}.q", dtype=np.uint8, count=rows * width // 2)
                                  .reshape(rows, width // 2)).to(device),
            "s": torch.from_numpy(np.fromfile(f"{stem}.s", dtype=np.float16, count=rows * width // group)
                                  .reshape(rows, width // group)).to(device),
            "tp": torch.from_numpy(np.fromfile(f"{stem}.tp", dtype=np.float16, count=rows * TOP)
                                   .reshape(rows, TOP)).to(device),
            "ti": torch.from_numpy(np.fromfile(f"{stem}.ti", dtype=np.int32, count=rows * TOP)
                                   .reshape(rows, TOP).astype(np.int64)).to(device),
            "verdict": torch.from_numpy(np.stack([np.zeros(rows, np.float32), verdict], 1)).to(device)}
        one = packed[seg]
        one["given"] = (one["tp"], one["ti"], one["verdict"])
        one["live"] = worker.live_bases(torch.arange(0, rows - 2, device=device), one["given"], args.steps, rows)
        picker = torch.Generator().manual_seed(args.seed + seg)
        take = torch.randperm(one["live"].numel(), generator=picker)[:args.probe].sort().values.to(device)
        one["probe"] = one["live"][take]
    say(f"{len(packed)} segments on the GPU, {sum(one['live'].numel() for one in packed.values()):,} live step starts, "
        f"probe {args.probe} a segment for the curves")

    def multi_of(one):
        return snap.dequantize_rows4(one["q"], one["s"], group)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    def measure(segs, which, rows, own_embed, with_parts=False):
        accept, greedy, parts, ent_q, ent_p, game = [], [], [], [], [], []
        block = {lam: {"block": [], "token": [], "tv_block": [], "tv_token_best": [], "tv_token_lam": []}
                 for lam in BLOCK_LAMBDAS}
        above = []
        with torch.no_grad():
            for seg in segs:
                one = packed[seg]
                bases = one[which]
                if bases.numel() == 0:
                    continue
                multi = multi_of(one)
                for at in range(0, bases.numel(), BASE_CHUNK):
                    part = bases[at:at + BASE_CHUNK]
                    low = max(0, int(part[0]) - args.context)
                    low -= low % head.shape.ratio
                    end = min(one["rows"], int(part[-1]) + args.steps + 2)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        outs = head.forward_steps(multi[low:end], one["ids"][low:end], own_embed, part, args.steps,
                                                  first_pos=low, chunk=32, share_index=False)
                    a_steps, g_steps, p_steps, q_steps, e_steps = [], [], [], [], []
                    chain = [[], [], [], [], []]
                    for step, made in enumerate(outs):
                        logits = F.linear(made.float(), rows.float())
                        at_rows = part + step + 1
                        a, pp, hq, hp, extra = position_stats(logits, one["tp"][at_rows], column[one["ti"][at_rows]],
                                                              temperature)
                        a_steps.append(a)
                        g_steps.append(extra[1][:, 0])
                        p_steps.append(pp)
                        q_steps.append(hq)
                        e_steps.append(hp)
                        for held, value in zip(chain, extra):
                            held.append(value)
                    accept.append(torch.stack(a_steps, -1).cpu().numpy())
                    greedy.append(torch.stack(g_steps, -1).cpu().numpy())
                    if with_parts:
                        parts.append(torch.stack(p_steps, 1).cpu().numpy())
                        ent_q.append(torch.stack(q_steps, -1).cpu().numpy())
                        ent_p.append(torch.stack(e_steps, -1).cpu().numpy())
                        qs, pqs, ps, qps, matches = chain
                        for lam in BLOCK_LAMBDAS:
                            made_block, share = bv.block_tokens(qs, pqs, ps, qps, lam, BLOCK_SAMPLES, generator)
                            best_tv, lam_tv = bv.token_tv(qs[0], pqs[0], ps[0], qps[0], matches[0], lam)
                            held = block[lam]
                            held["block"].append(made_block.cpu().numpy())
                            held["token"].append(bv.token_tokens(qs, pqs, lam).cpu().numpy())
                            held["tv_block"].append(bv.first_token_tv(qs, pqs, ps, qps, matches[0], lam, TV_SAMPLES,
                                                                      generator).cpu().numpy())
                            held["tv_token_best"].append(best_tv.cpu().numpy())
                            held["tv_token_lam"].append(lam_tv.cpu().numpy())
                        above.append(share.cpu().numpy())
                    game.append(np.array([one["game"]] * part.numel()))
                del multi
        return {"accept": np.concatenate(accept), "greedy": np.concatenate(greedy), "game": np.concatenate(game),
                **({"parts": np.concatenate(parts), "ent_q": np.concatenate(ent_q), "ent_p": np.concatenate(ent_p),
                    "above": np.concatenate(above),
                    "block": {lam: {name: np.concatenate(values) for name, values in held.items()}
                              for lam, held in block.items()}} if with_parts else {})}

    segs_all = sorted(packed)
    segs_train = [seg for seg in segs_all if packed[seg]["game"] in train_games]
    segs_pick = [seg for seg in segs_all if packed[seg]["game"] in pick_games]
    segs_report = [seg for seg in segs_all if packed[seg]["game"] in report_games]
    report = {"games": {"train": sorted(train_games), "pick": sorted(pick_games), "report": sorted(report_games)}}

    start = drafters["tuned-int4"]
    install(start)
    rows_int4, embed_int4 = rows_for(start)
    if not args.train_only:
        got = measure(segs_all, "live", rows_int4, embed_int4, with_parts=True)
        held_mask = np.isin(got["game"], sorted(held_games))
        report["A"] = {}
        for label, mask in (("all games", np.ones(len(got["game"]), dtype=bool)), ("held-out games", held_mask)):
            acc = got["accept"][mask]
            parts = got["parts"][mask]
            short = 1.0 - acc
            lines = []
            for step in range(args.steps):
                total = short[:, step].mean()
                shares = parts[:, step, :].mean(0) / max(total, 1e-12)
                lines.append({"position": step + 1, "accept": round(float(acc[:, step].mean()), 4),
                              "deficit": round(float(total), 4),
                              "deficit_share": {name: round(float(one), 4) for name, one in zip(PARTS, shares)},
                              "entropy_q": round(float(got["ent_q"][mask][:, step].mean()), 4),
                              "entropy_p": round(float(got["ent_p"][mask][:, step].mean()), 4)})
            report["A"][label] = lines
            say(f"A mtp_k400d with int4 experts, T{temperature:g}, drafter top-{TOP} sampled, {mask.sum():,} live starts of "
                f"{label}:")
            for line in lines:
                say(f"A   position {line['position']}: accept {line['accept']:.3f}, deficit sum max(0, p - q) "
                    f"{line['deficit']:.3f} of which " + ", ".join(f"{name} {share * 100:.1f}%" for name, share in
                                                                 line["deficit_share"].items())
                    + f"; entropy q {line['entropy_q']:.3f} nats against p {line['entropy_p']:.3f}")

        strict_token = got["block"][1.0]["token"]
        check = float(np.abs(strict_token - tokens_each(got["accept"])).max())
        share = got["above"].mean(0)
        report["D"] = {"share_ratio_above_one": [round(float(one), 4) for one in share], "formula_check": check,
                       "rules": {}}
        say(f"D share of draft positions with p(X)/q(X) > 1, X drawn from q (only there can block verification win "
            f"anything back): {' '.join(f'{one:.3f}' for one in share)} at positions 1-4; token-wise strict from the chain "
            f"code against A's rates: max difference {check:.2e}")
        for lam in BLOCK_LAMBDAS:
            held = got["block"][lam]
            mean, naive, cluster, count = paired(held["block"] - strict_token, got["game"])
            made_token = float(held["token"].mean())
            line = {"block": round(float(held["block"].mean()), 4), "token": round(made_token, 4),
                    "strict": round(float(strict_token.mean()), 4), "block_minus_strict": round(mean, 4),
                    "se_naive": round(naive, 4), "se_by_game": round(cluster, 4),
                    "tv_first_block": round(float(held["tv_block"].mean()), 4),
                    "tv_first_token_best_residual": round(float(held["tv_token_best"].mean()), 4),
                    "tv_first_token_lambda_residual": round(float(held["tv_token_lam"].mean()), 4)}
            report["D"]["rules"][f"lambda {lam:g}"] = line
            tag = "D" if lam == 1.0 else "E"
            say(f"{tag} lambda {lam:g}: block verification {line['block']:.4f} tokens a step against token-wise "
                f"{line['token']:.4f} (strict token-wise {line['strict']:.4f}); block - strict {mean:+.4f}, SE {naive:.4f} "
                f"over starts, {cluster:.4f} over {count} games; TV of the first emitted token to p: block "
                f"{line['tv_first_block']:.4f}, token-wise {line['tv_first_token_best_residual']:.4f} with the best "
                f"residual and {line['tv_first_token_lambda_residual']:.4f} with the lambda residual"
                + (" (lambda 1 is exact: the block TV is the Monte Carlo floor)" if lam == 1.0 else ""))

        if "tuned" in drafters:
            install(drafters["tuned"])
            rows_bf16, embed_bf16 = rows_for(drafters["tuned"])
            other = measure(segs_all, "live", rows_bf16, embed_bf16)
            install(start)
            diff = tokens_each(other["accept"]) - tokens_each(got["accept"])
            mean, naive, cluster, count = paired(diff, got["game"])
            report["B"] = {"int4": [round(float(one), 4) for one in got["accept"].mean(0)],
                           "bf16": [round(float(one), 4) for one in other["accept"].mean(0)],
                           "tokens_int4": round(float(tokens_each(got["accept"]).mean()), 4),
                           "tokens_bf16": round(float(tokens_each(other["accept"]).mean()), 4),
                           "paired_diff": round(mean, 4), "se_naive": round(naive, 4), "se_by_game": round(cluster, 4)}
            say(f"B sampled acceptance on the same {len(diff):,} live starts: int4 experts (albucino, as served) "
                f"{report['B']['int4']} = {report['B']['tokens_int4']:.4f} tokens a step; BF16 experts (the checkpoint's "
                f"own MTP) {report['B']['bf16']} = {report['B']['tokens_bf16']:.4f}; paired BF16 - int4 {mean:+.4f} "
                f"tokens a step, SE {naive:.4f} over starts, {cluster:.4f} over {count} games")
        else:
            say(f"B skipped: no BF16-expert drafter ({refused.get('tuned', 'the --no-bf16 flag')})")
    if args.eval_heads:
        sets = {"pick": segs_pick, "report": segs_report, "other": segs_train}
        results, base = {}, None
        for path in [one for one in args.eval_heads.split(",") if one]:
            built, _ = worker.build_drafters({**cfg, "tuned_file": path}, sources, {})
            install(built["tuned-int4"])
            rows, own_embed = rows_for(built["tuned-int4"])
            got = {name: measure(segs, "live", rows, own_embed) for name, segs in sets.items()}
            line = {}
            for name, held in got.items():
                line[name] = {"sampled": round(float(tokens_each(held["accept"]).mean()), 4),
                              "greedy": round(float(tokens_each(held["greedy"]).mean()), 4), "starts": len(held["game"])}
                if base is not None:
                    for kind, key in (("sampled", "accept"), ("greedy", "greedy")):
                        mean, naive, cluster, count = paired(tokens_each(held[key]) - tokens_each(base[name][key]),
                                                             held["game"])
                        line[name][kind + "_minus_first"] = {"mean": round(mean, 4), "se_starts": round(naive, 4),
                                                             "se_games": round(cluster, 4), "games": count}
            base = base or got
            results[path] = line
            say(f"EVAL {path}: " + "; ".join(f"{name} sampled {one['sampled']} greedy {one['greedy']} on {one['starts']:,} "
                                             f"live starts" + (f" (sampled {one['sampled_minus_first']['mean']:+.4f} SE "
                                             f"{one['sampled_minus_first']['se_starts']:.4f}/{one['sampled_minus_first']['se_games']:.4f}, "
                                             f"greedy {one['greedy_minus_first']['mean']:+.4f} SE "
                                             f"{one['greedy_minus_first']['se_starts']:.4f}/{one['greedy_minus_first']['se_games']:.4f} "
                                             f"against the first head)" if "sampled_minus_first" in one else "")
                                             for name, one in line.items()))
        (out / "eval_heads.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        say(f"wrote {out / 'eval_heads.json'}")
        return
    if args.no_train:
        (out / "diag_mtp.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        say(f"wrote {out / 'diag_mtp.json'} (no training: --no-train)")
        return

    trainer.rows = rows_int4 * args.squeeze
    train_label = "the other capture games (not trained on)" if args.train_snapshot else "train games"
    curve, best = [], {"score": -1.0}
    params = list(head.trainable())

    def evaluate(update):
        rows = trainer.rows
        train = measure(segs_train, "probe", rows, trainer.embed)
        pick = measure(segs_pick, "probe", rows, trainer.embed)
        rep = measure(segs_report, "probe", rows, trainer.embed)
        point = {"update": update, "minutes": round((time.monotonic() - began) / 60, 2),
                 "train": round(float(tokens_each(train["accept"]).mean()), 4),
                 "pick": round(float(tokens_each(pick["accept"]).mean()), 4),
                 "report": round(float(tokens_each(rep["accept"]).mean()), 4),
                 "train_accept": [round(float(one), 4) for one in train["accept"].mean(0)],
                 "pick_accept": [round(float(one), 4) for one in pick["accept"].mean(0)]}
        curve.append(point)
        score = point["pick"] if args.select == "pick" else (point["pick"] + point["report"]) / 2
        better = score > best["score"]
        if args.export_every and update:
            stem = Path(args.export).stem
            worker.export(head, sources["albucino"].raw, out / f"{stem}_u{update:05d}.safetensors",
                          {"from": "tuned-int4", "update": str(update), "lr": str(args.lr), "clip": str(args.clip)})
        if better:
            best.update({"score": score, "point": point,
                         "state": [one.detach().clone() for one in params],
                         "report_accept": rep["accept"], "report_game": rep["game"]})
            if update:
                worker.export(head, sources["albucino"].raw, out / args.export,
                              {"from": "tuned-int4", "best_update": str(update), "squeeze": str(args.squeeze),
                               "kl": str(args.kl), "lr": str(args.lr), "clip": str(args.clip)})
        (out / "curve.json").write_text(json.dumps({"curve": curve, "best_update": best["point"]["update"]}, indent=1),
                                        encoding="utf-8")
        say(f"C update {update} ({point['minutes']:.1f} min): tokens a step on {train_label} {point['train']:.4f}, "
            f"held-out picking games {point['pick']:.4f}, held-out report games {point['report']:.4f}; "
            f"acceptance on {train_label} {point['train_accept']}, picking {point['pick_accept']}"
            + ("; best so far" if better else ""))
        return rep

    base_rep = measure(segs_report, "probe", rows_int4, trainer.embed)
    say(f"C mtp_k400d as served (logits x1) on the report probe: {tokens_each(base_rep['accept']).mean():.4f} tokens a "
        f"step; the training starts from it with logits x{args.squeeze:g} (drafting at T{temperature / args.squeeze:.3f})")
    entries = []
    if args.train_snapshot:
        reader = snap.Reader(Path(args.train_snapshot), device, window=args.window)
        entries = list(reader.index["entries"])
        fetch = reader.fetch
        source = (f"{len({one['game'] for one in entries})} games of the prefill snapshot {args.train_snapshot} "
                  f"({sum(int(one['rows']) for one in reader.index['segments']):,} rows, context window {args.window})")
    else:
        for seg in segs_train:
            one = packed[seg]
            for first, stop in cap.turns_from_targets(capture, seg, one["rows"], TOP):
                entries.append({"seg": seg, "start": first, "count": stop})
        cache = {seg: multi_of(packed[seg]) for seg in segs_train}

        def fetch(entry):
            one = packed[entry["seg"]]
            count = int(entry["count"])
            return (one["ids"][:count], cache[entry["seg"]][:count], int(entry["start"]),
                    tuple(part[:count] for part in one["given"]))
        source = f"{len(segs_train)} train games of {capture}"

    order = []
    for epoch in range(200):
        shuffled = list(entries)
        random.Random(args.seed + epoch).shuffle(shuffled)
        order += shuffled
    say(f"C training: {len(entries)} reply turns of {source}, TV plus {args.kl:g} x soft CE to "
        f"the target's top-{TOP}, position weights {cfg['step_weights']}, lr {args.lr:g}, clip {args.clip:g}, dense only, "
        f"{args.train_minutes:g} min of training; curves every {args.every_minutes:g} min")
    began = time.monotonic()
    timing = {"last": time.monotonic(), "spent": 0.0}

    def checkpoint(update):
        if update and time.monotonic() - timing["last"] < args.every_minutes * 60:
            return
        held = time.monotonic()
        evaluate(update)
        timing["spent"] += time.monotonic() - held
        timing["last"] = time.monotonic()

    trained = worker.base.train_rounds(cfg, head, trainer, None, order, args.train_minutes + 3.0, fetch=fetch,
                                       checkpoint=checkpoint, every=1)
    if not curve or curve[-1]["update"] != trained["updates"]:
        evaluate(trained["updates"])
    first = curve[0]
    chosen = best["point"]
    with torch.no_grad():
        for param, kept in zip(params, best["state"]):
            param.copy_(kept)
    exported = worker.export(head, sources["albucino"].raw, out / args.export,
                             {"from": "tuned-int4", "best_update": str(chosen["update"]),
                              "trained_records": str(trained.get("trained")), "squeeze": str(args.squeeze),
                              "kl": str(args.kl), "lr": str(args.lr)})
    diff = tokens_each(best["report_accept"]) - tokens_each(base_rep["accept"])
    mean, naive, cluster, count = paired(diff, best["report_game"])
    report["C"] = {"curve": curve, "best_update": chosen["update"], "served_report": round(float(
        tokens_each(base_rep["accept"]).mean()), 4), "best_report": chosen["report"],
        "paired_best_minus_served": round(mean, 4), "se_naive": round(naive, 4), "se_by_game": round(cluster, 4),
        "eval_minutes": round(timing["spent"] / 60, 2), "trained": trained, "export": exported,
        "train_source": source}
    say(f"C best on the picking games at update {chosen['update']} ({chosen['minutes']:.1f} min): picking "
        f"{chosen['pick']:.4f} against {first['pick']:.4f} at the start; on the report games {chosen['report']:.4f} "
        f"against {first['report']:.4f} at the start (logits x{args.squeeze:g}) and "
        f"{report['C']['served_report']:.4f} for mtp_k400d as served; paired best - served {mean:+.4f} tokens a step, "
        f"SE {naive:.4f} over starts, {cluster:.4f} over {count} games; train games {first['train']:.4f} -> "
        f"{curve[-1]['train']:.4f}")
    name = "diag_mtp_train.json" if args.train_only else "diag_mtp.json"
    (out / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
    say(f"wrote {out / name}")


if __name__ == "__main__":
    main()
