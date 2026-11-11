from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
from pathlib import Path

PART = re.compile(r"TAAF_EXPERT_PART pid=(\d+) (\d+)/(\d+) (\S+)")
EXPERTS = 512


def decode_dumps(log: Path) -> list[dict]:
    dumps: list[tuple[str, int, dict[int, str]]] = []
    open_dump: dict[str, tuple[str, int, dict[int, str]]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        found = PART.search(line)
        if not found:
            continue
        pid, index, total, chunk = found.group(1), int(found.group(2)), int(found.group(3)), found.group(4)
        if index == 0 or pid not in open_dump or open_dump[pid][1] != total:
            open_dump[pid] = (pid, total, {})
            dumps.append(open_dump[pid])
        open_dump[pid][2][index] = chunk
    out = []
    for pid, total, parts in dumps:
        if sorted(parts) != list(range(total)):
            continue
        rows = json.loads(gzip.decompress(base64.b64decode("".join(parts[i] for i in range(total)))))
        out.append({"pid": pid, "rows": rows, "decisions": sum(next(iter(rows.values())))})
    return out


def layer_key(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    found = re.search(r"\.layers\.(\d+)\.", name)
    if not found:
        raise ValueError(f"no layer number in {name}")
    return found.group(1)


def usage_order(rows: dict, tie: dict) -> dict:
    order = {}
    for name, counts in rows.items():
        key = layer_key(name)
        if len(counts) < EXPERTS:
            raise ValueError(f"{name} counts {len(counts)} experts, wanted {EXPERTS}")
        place = {expert: rank for rank, expert in enumerate(tie[key])}
        order[key] = sorted(range(EXPERTS), key=lambda expert: (-counts[expert], place[expert]))
    missing = set(tie) - set(order)
    if missing:
        raise ValueError(f"counts lack layers {sorted(missing)}")
    return order


def lost(rows: dict, order: dict, keep: int) -> list[float]:
    shares = []
    for name, counts in rows.items():
        kept = set(order[layer_key(name)][:keep])
        total = sum(counts[:EXPERTS])
        shares.append(sum(counts[e] for e in range(EXPERTS) if e not in kept) / total)
    return shares


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--tie", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dumps = decode_dumps(args.log)
    if not dumps:
        raise SystemExit(f"no complete TAAF_EXPERT_PART dump in {args.log}")
    best = max(dumps, key=lambda one: one["decisions"])
    tie = json.loads(args.tie.read_text(encoding="utf-8"))
    order = usage_order(best["rows"], tie)
    ordered = {key: order[key] for key in sorted(order, key=lambda k: (k == "mtp", int(k) if k.isdigit() else 0))}
    args.out.write_text(json.dumps(ordered, separators=(",", ":")), encoding="utf-8")
    print(f"{args.out}: {len(ordered)} layers ranked by {best['decisions']:,} routing decisions a layer "
          f"(pid {best['pid']}, {len(dumps)} complete dumps in {args.log.name})")
    for keep in (480, 448, 416, 400, 384, 352, 320):
        mine, theirs = lost(best["rows"], order, keep), lost(best["rows"], tie, keep)
        print(f"  keep {keep}: usage loses {100 * sum(mine) / len(mine):.2f}% (worst layer {100 * max(mine):.2f}%), "
              f"tie-break order {100 * sum(theirs) / len(theirs):.2f}%")


if __name__ == "__main__":
    main()
