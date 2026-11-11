from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import mtp_corpus  # noqa: E402

MEMBER = re.compile(r"(?:.*/)?(?P<game>[a-z0-9]{4})-[0-9a-f]+_p(?P<lane>\d+)_requests\.jsonl")


def say(text: str) -> None:
    print(f"mtp_long_corpus {time.strftime('%H:%M:%S')} {text}", flush=True)


def digest(message: dict[str, Any]) -> str:
    return hashlib.sha1(mtp_corpus.canonical(message).encode("utf-8")).hexdigest()


def survey(job: tuple[str, str, bytes]) -> tuple[str, str, list[dict[str, Any]]]:
    run, name, blob = job
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    calls: list[dict[str, Any]] = []
    previous: list[str] | None = None
    number = 0
    for index, row in enumerate(rows):
        if row.get("event") != "response" or not isinstance(row.get("messages"), list):
            continue
        number += 1
        prompt = row["messages"]
        hashes = [digest(one) for one in prompt]
        later = next((one for one in rows[index + 1:] if one.get("event") == "request"), None)
        reply = mtp_corpus.reply_after(prompt, later.get("messages") or []) if later is not None else None
        usage = row.get("usage") or {}
        extends = previous is not None and len(hashes) > len(previous) and hashes[:len(previous)] == previous \
            and prompt[len(previous)].get("role") == "assistant"
        calls.append({"row": index, "call": number, "step": int(row.get("analysis_step") or 0),
                      "prompt": int(usage.get("prompt_tokens") or 0),
                      "completion": int(usage.get("completion_tokens") or 0),
                      "extends": bool(extends), "usable": reply is not None and int(usage.get("completion_tokens") or 0) > 0
                      and prompt[-1].get("role") in ("user", "tool")})
        previous = hashes
    return run, name, calls


def new_tokens(calls: list[dict[str, Any]], at: int, first: int) -> int:
    one = calls[at]
    whole = one["prompt"] + one["completion"]
    if at == first or not one["extends"]:
        return whole
    before = calls[at - 1]
    return max(one["completion"], whole - before["prompt"] - before["completion"])


def block_cost(calls: list[dict[str, Any]], first: int, length: int) -> tuple[int, int, int]:
    stop = min(len(calls), first + length)
    cost = sum(new_tokens(calls, at, first) for at in range(first, stop))
    replies = sum(calls[at]["completion"] for at in range(first, stop) if calls[at]["usable"])
    breaks = sum(1 for at in range(first + 1, stop) if not calls[at]["extends"])
    return cost, replies, breaks


def choose(talks: dict[str, list[tuple[str, str, list[dict[str, Any]]]]], calls: int, games: int, per_game_talks: int,
           min_block: int, seed: int) -> tuple[dict[tuple[str, str], list[int]], dict[str, Any]]:
    rng = random.Random(seed)
    names = sorted(talks)
    rng.shuffle(names)
    names = sorted(names[:games])
    each = max(min_block, math.ceil(calls / max(1, len(names))))
    chosen: dict[tuple[str, str], list[int]] = {}
    totals = {"cost": 0, "replies": 0, "calls": 0, "breaks": 0, "blocks": 0}
    for game in names:
        options = [one for one in talks[game] if len(one[2]) >= min_block]
        rng.shuffle(options)
        runs_seen: set[str] = set()
        picked = []
        for one in options:
            if one[0] in runs_seen:
                continue
            runs_seen.add(one[0])
            picked.append(one)
            if len(picked) >= per_game_talks:
                break
        if not picked:
            continue
        length = max(min_block, math.ceil(each / len(picked)))
        for run, name, game_calls in picked:
            starts = [at for at, one in enumerate(game_calls) if at == 0 or not one["extends"]]
            starts = [at for at in starts if at + min_block <= len(game_calls)] or [0]
            scored = []
            for at in starts:
                cost, replies, breaks = block_cost(game_calls, at, length)
                scored.append((cost / max(1, replies), at, cost, replies, breaks))
            scored.sort()
            pool = scored[:max(1, math.ceil(len(scored) / 3))]
            _, at, cost, replies, breaks = rng.choice(pool)
            stop = min(len(game_calls), at + length)
            rows = [one["row"] for one in game_calls[at:stop] if one["usable"]]
            chosen[(run, name)] = rows
            totals["cost"] += cost
            totals["replies"] += replies
            totals["calls"] += len(rows)
            totals["breaks"] += breaks
            totals["blocks"] += 1
    return chosen, {**totals, "games": len(names), "calls_per_game": each}


def render(job: tuple[str, str, bytes, list[int], str]) -> list[dict[str, Any]]:
    run, name, blob, wanted, split = job
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    base = Path(name).name.removesuffix("_requests.jsonl")
    game = base.split("-")[0]
    keep = set(wanted)
    out = []
    call = 0
    for index, row in enumerate(rows):
        if row.get("event") != "response" or not isinstance(row.get("messages"), list):
            continue
        call += 1
        if index not in keep:
            continue
        later = next((one for one in rows[index + 1:] if one.get("event") == "request"), None)
        reply = mtp_corpus.reply_after(row["messages"], later.get("messages") or []) if later is not None else None
        if reply is None:
            out.append({"skip": "the next request does not carry the reply"})
            continue
        step = int(row.get("analysis_step") or 0)
        record = mtp_corpus.request_record(run, f"{base}_s{step}_c{call}", split, row, reply, step, call, game)
        out.append(record)
    return out


def batched(pool, work, jobs, size: int):
    batch = []
    for job in jobs:
        batch.append(job)
        if len(batch) >= size:
            yield from pool.map(work, batch, chunksize=1)
            batch = []
    if batch:
        yield from pool.map(work, batch, chunksize=1)


def members(paths: list[Path], keep) -> Any:
    for path in paths:
        run = path.parent.name
        began = time.monotonic()
        seen = 0
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                found = MEMBER.fullmatch(member.name)
                if not member.isfile() or not found or not keep(run, member.name, found["game"]):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                seen += 1
                yield run, member.name, found["game"], handle.read()
        say(f"{run}: {seen} game files read in {time.monotonic() - began:.0f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long runs of consecutive calls per game from archived request logs, "
                                                 "chosen so a prefix-reusing extract computes each token once.")
    parser.add_argument("--archives", nargs="+", type=Path, required=True)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--calls", type=int, default=3500)
    parser.add_argument("--games", type=int, default=90)
    parser.add_argument("--talks-per-game", type=int, default=2)
    parser.add_argument("--min-block", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    excluded = {one.strip()[:4] for one in args.exclude.split(",") if one.strip()}
    began = time.monotonic()
    talks: dict[str, list[tuple[str, str, list[dict[str, Any]]]]] = {}
    with Pool(args.workers) as pool:
        jobs = ((run, name, blob) for run, name, _, blob in
                members(args.archives, lambda run, name, game: game not in excluded))
        for run, name, game_calls in batched(pool, survey, jobs, args.workers):
            game = MEMBER.fullmatch(name)["game"]
            talks.setdefault(game, []).append((run, name, game_calls))
    surveyed = sum(len(calls) for many in talks.values() for _, _, calls in many)
    say(f"survey: {len(talks)} games, {sum(len(many) for many in talks.values())} game files, {surveyed:,} calls in "
        f"{time.monotonic() - began:.0f} s; {len(excluded)} games kept out")
    chosen, totals = choose(talks, args.calls, args.games, args.talks_per_game, args.min_block, args.seed)
    say(f"plan: {totals['calls']:,} calls in {totals['blocks']} blocks of {totals['games']} games "
        f"({totals['calls_per_game']} calls a game asked); {totals['replies']:,} reply tokens; about "
        f"{totals['cost'] / 1e6:.1f}M tokens to compute ({totals['cost'] / max(1, totals['replies']):.2f} a reply token), "
        f"{totals['breaks']} breaks inside blocks")
    if args.plan_only:
        return 0
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    with Pool(args.workers, initializer=mtp_corpus.init_worker,
              initargs=(str(args.template), str(args.tokenizer), [])) as pool:
        jobs = ((run, name, blob, chosen[(run, name)], "train") for run, name, _, blob in
                members(args.archives, lambda run, name, game: (run, name) in chosen))
        for built in batched(pool, render, jobs, args.workers):
            for one in built:
                if "skip" in one:
                    skipped[one["skip"]] = skipped.get(one["skip"], 0) + 1
                elif mtp_corpus.path_hits(one):
                    skipped["a string reads as a local path"] = skipped.get("a string reads as a local path", 0) + 1
                else:
                    records.append(one)
    records.sort(key=mtp_corpus.order_key)
    about = {"runs": len({one["run"] for one in records}), "games": len({one["game"] for one in records}),
             "source": "request logs, long runs of consecutive calls per game",
             "selection": {"calls": args.calls, "games": args.games, "talks_per_game": args.talks_per_game,
                           "min_block": args.min_block, "seed": args.seed, "excluded": sorted(excluded),
                           "archives": [path.parent.name for path in args.archives], "plan": totals,
                           "reply_tokens": sum(int(one["reply_tokens"]) for one in records),
                           "prompt_tokens": sum(int(one.get("prompt_tokens") or 0) for one in records)}}
    stats = mtp_corpus.summarize(records, skipped, 0)
    manifest = mtp_corpus.write(records, args.out, {**about, "stats": stats})
    say(f"wrote {len(records):,} records of {about['games']} games from {about['runs']} runs in "
        f"{len(manifest['shards'])} shards, {sum(one['bytes'] for one in manifest['shards']) / 2**20:.1f} MiB; "
        f"skipped {skipped or 0}; prompt text shared with the previous record {stats.get('text_chars_shared_with_previous', 0):,} "
        f"of {stats.get('text_chars', 0):,} chars; {time.monotonic() - began:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
