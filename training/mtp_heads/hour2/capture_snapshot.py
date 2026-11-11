import hashlib
import json
import os
import re
import shutil
from pathlib import Path

INDEX = "snapshot.json"
SESSIONS = "sessions.jsonl"
PARTS = ("ids", "q", "s", "tp", "ti")
DR_PART = "dr"
DR_FIELDS = [("id", "<i4"), ("conf", "<f2"), ("acc", "u1")]
NO_DRAFT = 255
CARRIED = ("lm_head.pt", "embed.pt", "config.json")
GAME = re.compile(r"(?:^|[-_/])([a-z0-9]{4})-[0-9a-f]{8}(?=$|[-_/#])")
SHORT_GAME = re.compile(r"^([a-z]{2}\d{2})(?:[-_]|$)")
UNOWNED = (("_warmup", "warmup"), ("chatcmpl-", "chat"), ("cmpl-", "cmpl"))
NOT_GAMES = ("forum", "?", "chat", "cmpl", "warmup")
HELDOUT_SHARE = 0.2
BANDS = (0.5, 0.7, 0.8, 0.9, 0.95)
BAND_LABELS = ("<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">=0.95")


def is_capture(folder):
    path = Path(folder) / INDEX
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("source") == "decode-capture"


def external(rid):
    head, _, tail = str(rid).rpartition("-")
    return head if head and len(tail) == 8 else str(rid)


def game_of(key):
    key = str(key or "")
    found = GAME.search(key) or SHORT_GAME.match(key)
    if found:
        return found.group(1)
    return "forum" if key.startswith("forum") else key or "?"


def owner_of(rid, by_rid):
    rid = str(rid or "")
    if external(rid) in by_rid:
        return by_rid[external(rid)]
    return next((label for prefix, label in UNOWNED if rid.startswith(prefix)), "?")


def session_games(folder):
    games = {}
    path = Path(folder) / SESSIONS
    if not path.is_file():
        return games
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        games[str(row.get("rid"))] = game_of(row.get("key"))
    return games


def heldout_of(games, share=HELDOUT_SHARE, named=()):
    if named:
        return set(named)
    ranked = sorted(set(games) - set(NOT_GAMES), key=lambda game: hashlib.sha256(game.encode()).hexdigest())
    return set(ranked[:max(1, round(len(ranked) * share))]) if ranked else set()


def turns_from_targets(folder, seg, rows, topk, min_rows=8):
    import numpy as np
    path = Path(folder) / f"seg{seg:05d}.tp"
    if not path.is_file() or rows <= 0:
        return []
    tp = np.memmap(path, dtype=np.float16, mode="r", shape=(rows, topk))
    generated = np.asarray(tp[:, 0]) > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], generated.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= min_rows]


class Calib:
    def __init__(self):
        import numpy as np
        self.count = np.zeros(len(BANDS) + 1)
        self.conf = np.zeros(len(BANDS) + 1)
        self.accept = np.zeros(len(BANDS) + 1)

    def add(self, conf, accept):
        import numpy as np
        conf = np.asarray(conf, dtype=np.float64).ravel()
        accept = np.asarray(accept, dtype=np.float64).ravel()
        band = np.digitize(conf, BANDS)
        size = len(BANDS) + 1
        self.count += np.bincount(band, minlength=size)
        self.conf += np.bincount(band, weights=conf, minlength=size)
        self.accept += np.bincount(band, weights=accept, minlength=size)

    def summary(self):
        total = float(self.count.sum())
        if not total:
            return {"positions": 0}
        buckets = [{"band": label, "share": round(float(count) / total, 4),
                    "conf": round(float(conf / count), 4), "accept": round(float(accept / count), 4)}
                   for label, count, conf, accept in zip(BAND_LABELS, self.count, self.conf, self.accept) if count]
        return {"positions": int(total), "conf": round(float(self.conf.sum()) / total, 4),
                "accept": round(float(self.accept.sum()) / total, 4),
                "gap": round(float(self.conf.sum() - self.accept.sum()) / total, 4),
                "ece": round(float(abs(self.conf - self.accept).sum()) / total, 4), "buckets": buckets}


def calib_text(summary, brief=False):
    if not summary.get("positions"):
        return "calib -"
    if brief:
        return (f"calib gap {summary['gap']:+.3f} ece {summary['ece']:.3f} (conf {summary['conf']:.3f}, accept "
                f"{summary['accept']:.3f})")
    cells = " ".join(f"{one['band']} {one['share']:.2f}:{one['conf']:.2f}/{one['accept']:.2f}"
                     for one in summary["buckets"])
    return (f"calib conf {summary['conf']:.3f} accept {summary['accept']:.3f} gap {summary['gap']:+.3f} "
            f"ece {summary['ece']:.3f} [band share:conf/accept {cells}]")


def draft_rows(folder, seg, rows):
    import numpy as np
    path = Path(folder) / f"seg{seg:05d}.{DR_PART}"
    if not path.is_file() or rows <= 0:
        return None
    held = np.fromfile(path, dtype=np.dtype(DR_FIELDS))
    return held[:rows] if len(held) >= rows else None


def draft_depths(acc):
    import numpy as np
    accepted = acc == 1
    run = np.cumsum(accepted)
    run = run - np.maximum.accumulate(np.where(accepted, 0, run))
    depth = np.zeros(len(acc), dtype=np.int64)
    depth[accepted] = run[accepted]
    rejected = np.flatnonzero(acc == 0)
    depth[rejected] = np.where(rejected > 0, run[np.maximum(rejected - 1, 0)], 0) + 1
    return depth


def live_drafts(capture, segments, entries, topk, at=0.95):
    import numpy as np
    out = {}
    for split in ("train", "heldout"):
        calib, depths = Calib(), {}
        easy = {"positions": 0, "easy": 0, "easy_accept": 0.0, "hard_accept": 0.0}
        for seg in sorted({int(one["seg"]) for one in entries if one["split"] == split}):
            rows = int(segments[seg]["rows"])
            drafts = draft_rows(capture, seg, rows)
            if drafts is None:
                continue
            acc = drafts["acc"]
            seen = acc != NO_DRAFT
            known = seen & (drafts["conf"].astype(np.float32) >= 0)
            calib.add(drafts["conf"][known].astype(np.float64), acc[known] == 1)
            depth = draft_depths(acc)
            for one in np.unique(depth[seen]):
                held = seen & (depth == one)
                tally = depths.setdefault(int(one), [0, 0])
                tally[0] += int((acc[held] == 1).sum())
                tally[1] += int(held.sum())
            path = Path(capture) / f"seg{seg:05d}.tp"
            if path.is_file():
                top = np.asarray(np.memmap(path, dtype=np.float16, mode="r", shape=(rows, topk))[:, 0], np.float32)
                where = np.flatnonzero(seen[1:]) + 1
                first = top[where - 1]
                verdict = acc[where] == 1
                valid = first > 0
                light = valid & (first >= at)
                easy["positions"] += int(valid.sum())
                easy["easy"] += int(light.sum())
                easy["easy_accept"] += float(verdict[light].sum())
                easy["hard_accept"] += float(verdict[valid & ~light].sum())
        positions = easy["positions"]
        hard = positions - easy["easy"]
        rejected = positions - easy["easy_accept"] - easy["hard_accept"]
        out[split] = {"calibration": calib.summary(),
                      "conditional_by_depth": {str(one): round(tally[0] / tally[1], 4)
                                               for one, tally in sorted(depths.items()) if tally[1]},
                      "drafted_by_depth": {str(one): tally[1] for one, tally in sorted(depths.items())},
                      "target_top1": {"at": at, "positions": positions,
                                      "easy_share": round(easy["easy"] / positions, 4) if positions else None,
                                      "easy_accept": round(easy["easy_accept"] / easy["easy"], 4)
                                      if easy["easy"] else None,
                                      "hard_accept": round(easy["hard_accept"] / hard, 4) if hard else None,
                                      "rejections_in_hard": round((hard - easy["hard_accept"]) / rejected, 4)
                                      if rejected else None}}
    return out


def live_text(split, held):
    depth = " / ".join(f"{one}:{rate:.3f}" for one, rate in held["conditional_by_depth"].items()) or "-"
    top = held["target_top1"]
    return (f"LIVE DRAFTS {split}: conditional acceptance by depth {depth}; target top-1 >= {top['at']}: "
            f"{top['easy_share']} of {top['positions']} drafted positions accept {top['easy_accept']}, the rest "
            f"accept {top['hard_accept']}, {top['rejections_in_hard']} of rejections below; "
            + calib_text(held["calibration"]))


def spread(items, count):
    if count <= 0 or len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(at * step)] for at in range(count)]


def convert(capture, reference, out, heldout=(), share=HELDOUT_SHARE, link=True, max_heldout=0):
    capture, reference, out = Path(capture), Path(reference), Path(out)
    index = json.loads((capture / INDEX).read_text(encoding="utf-8"))
    ref_index = json.loads((reference / INDEX).read_text(encoding="utf-8"))
    if int(ref_index["width"]) != int(index["width"]) or int(ref_index["hc"]) != int(index["hc"]):
        raise SystemExit(f"the reference snapshot is {ref_index['hc']}x{ref_index['width']}, the capture "
                         f"{index['hc']}x{index['width']}")
    by_rid = session_games(capture)
    entries = []
    for seg, held in enumerate(index["segments"]):
        rid = held.get("rid") or next((one.get("rid") for one in index["entries"] if int(one["seg"]) == seg), "")
        game = owner_of(rid, by_rid)
        if game == "warmup":
            continue
        for at, (start, stop) in enumerate(turns_from_targets(capture, seg, int(held["rows"]), int(index["topk"]))):
            entries.append({"key": f"{rid}#r{at}", "split": "train", "game": game, "run": "capture", "step": 0,
                            "call": at, "seg": seg, "count": stop, "start": start, "rid": rid})
    if not entries:
        for entry in index["entries"]:
            entries.append({**entry, "game": owner_of(entry.get("rid", ""), by_rid)})
    held = heldout_of({one["game"] for one in entries}, share, heldout)
    for entry in entries:
        entry["split"] = "heldout" if entry["game"] in held else "train"
    if max_heldout:
        kept = {id(one) for one in spread([one for one in entries if one["split"] == "heldout"], max_heldout)}
        entries = [one for one in entries if one["split"] == "train" or id(one) in kept]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    used = sorted({int(one["seg"]) for one in entries})
    remap = {seg: at for at, seg in enumerate(used)}
    drafted = bool(used) and all((capture / f"seg{seg:05d}.{DR_PART}").is_file() for seg in used)
    live = live_drafts(capture, index["segments"], entries, int(index["topk"])) if drafted else None
    for seg in used:
        for part in PARTS + ((DR_PART,) if drafted else ()):
            source, target = capture / f"seg{seg:05d}.{part}", out / f"seg{remap[seg]:05d}.{part}"
            if link:
                try:
                    os.symlink(source.resolve(), target)
                    continue
                except OSError:
                    pass
            shutil.copyfile(source, target)
    for name in CARRIED:
        if (reference / name).is_file():
            if link:
                try:
                    os.symlink((reference / name).resolve(), out / name)
                    continue
                except OSError:
                    pass
            shutil.copyfile(reference / name, out / name)
    segments = [{**index["segments"][seg], "bits": 4, "source_seg": seg} for seg in used]
    for entry in entries:
        entry["seg"] = remap[int(entry["seg"])]
        entry["count"] = min(int(entry["count"]), int(segments[entry["seg"]]["rows"]))
    entries = [one for one in entries if one["count"] > int(one["start"])]
    saved = {key: value for key, value in ref_index.items() if key not in ("segments", "entries", "bytes", "parity")}
    saved.update({"hc": index["hc"], "width": index["width"], "topk": index["topk"], "segments": segments,
                  "entries": entries, "bytes": index.get("bytes", 0), "source": "decode-capture",
                  "temperature": index.get("temperature"), "top_k": index.get("top_k"), "top_p": index.get("top_p"),
                  "heldout_games": sorted(held), "capture_counts": index.get("counts", {}),
                  "draft_rows": drafted, **({"live_drafts": live} if live else {})})
    (out / INDEX).write_text(json.dumps(saved), encoding="utf-8")
    games = sorted({one["game"] for one in entries})
    if live:
        for split, one in live.items():
            print(live_text(split, one), flush=True)
    return {"segments": len(segments), "entries": len(entries), "rows": sum(int(one["rows"]) for one in segments),
            "games": games, "heldout": sorted(held),
            "heldout_entries": sum(one["split"] == "heldout" for one in entries), "draft_rows": drafted,
            "turns_by_game": {game: sum(one["game"] == game for one in entries) for game in games}}


def report(capture, heldout=(), share=HELDOUT_SHARE):
    capture = Path(capture)
    index = json.loads((capture / INDEX).read_text(encoding="utf-8"))
    by_rid = session_games(capture)
    entries = []
    for seg, held in enumerate(index["segments"]):
        game = owner_of(held.get("rid", ""), by_rid)
        if game != "warmup":
            entries.append({"seg": seg, "game": game, "rows": int(held["rows"])})
    held = heldout_of({one["game"] for one in entries}, share, heldout)
    for entry in entries:
        entry["split"] = "heldout" if entry["game"] in held else "train"
    live = live_drafts(capture, index["segments"], entries, int(index["topk"]))
    for split, one in live.items():
        print(live_text(split, one), flush=True)
    games = sorted({one["game"] for one in entries})
    return {"segments": len(entries), "games": games, "heldout": sorted(held),
            "rows_by_owner": {game: sum(one["rows"] for one in entries if one["game"] == game) for game in games},
            "live_drafts": live}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--heldout", default="", help="comma list of held-out games; default a hashed fifth")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--report", action="store_true", help="only the live drafter table of the capture")
    args = parser.parse_args()
    named = [one for one in args.heldout.split(",") if one]
    if args.report or not args.out:
        print(json.dumps(report(args.capture, named), indent=1))
    else:
        print(json.dumps(convert(args.capture, args.reference, args.out, named, link=not args.copy), indent=1))
