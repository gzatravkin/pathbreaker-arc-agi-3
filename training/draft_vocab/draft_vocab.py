import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

TOKENIZER = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "qwen3.8-flash-next" / "tokenizer.json"
REGIONS = ("reasoning", "content", "tool_call")
THINK_END = "</think>"
TOOL_START = "<tool_call>"


def reply_texts(reply: dict):
    for key in ("reasoning", "reasoning_content", "content"):
        if isinstance(reply.get(key), str) and reply.get(key):
            yield reply[key]
    for call in reply.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                yield arguments
                continue
        if isinstance(arguments, dict):
            yield (f"<tool_call>\n<function={function.get('name', '')}>\n"
                   + "".join(f"<parameter={name}>\n{value}\n</parameter>\n"
                             for name, value in arguments.items())
                   + "</function>\n</tool_call>")


def request_log_replies(path: Path):
    seen = set()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for message in row.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            key = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                yield message


def generated_texts(run: Path):
    for path in sorted((run / "call_requests").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        yield from reply_texts(row.get("reply") or {})
    for path in sorted((run / "req").glob("*_requests.jsonl")):
        for message in request_log_replies(path):
            yield from reply_texts(message)


def game_of(name: str) -> str:
    return name.split("-", 1)[0]


def count_by_game(runs, tokenizer) -> dict[str, collections.Counter]:
    games: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for run in runs:
        for path in sorted((run / "req").glob("*_requests.jsonl")):
            batch = [text for message in request_log_replies(path) for text in reply_texts(message)]
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                games[game_of(path.name)].update(encoding.ids)
    return games


def count(runs, tokenizer):
    counts = collections.Counter()
    texts = 0
    for run in runs:
        batch = list(generated_texts(run))
        texts += len(batch)
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            counts.update(encoding.ids)
    return counts, texts


def base_ids(tokenizer, rows: int) -> set[int]:
    special = set(tokenizer.get_added_tokens_decoder())
    special |= {index for index in range(tokenizer.get_vocab_size(with_added_tokens=True))
                if tokenizer.id_to_token(index) and len(tokenizer.id_to_token(index)) == 1}
    return {index for index in special if index < rows}


def outside_share(games: dict[str, collections.Counter], names, kept: set[int]) -> float:
    total = sum(sum(games[name].values()) for name in names)
    out = sum(n for name in names for index, n in games[name].items() if index not in kept)
    return out / max(total, 1)


def split_check(games: dict[str, collections.Counter], first: int, keep: set[int], splits: int, size: int) -> None:
    import random

    names = sorted(games)
    if len(names) < 2 * size:
        raise SystemExit(f"--split-games {size} needs {2 * size} games, the runs hold {len(names)}")
    rng = random.Random(7)
    rows: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(splits):
        picked = rng.sample(names, 2 * size)
        train, test = picked[:size], picked[size:]
        seen = set().union(*(set(games[name]) for name in train))
        rows["ids of the train games"].append(outside_share(games, test, seen))
        rows[f"ids of the train games + ids below {first}"].append(outside_share(games, test, seen | set(range(first))))
        if keep:
            rows[f"ids of the train games + ids below {first} + kept vocabularies"].append(
                outside_share(games, test, seen | set(range(first)) | keep))
    for label, values in rows.items():
        values.sort()
        print(f"draft_vocab: {splits} splits of {size} train / {size} new games, {label}: outside on the new games "
              f"median {values[len(values) // 2]:.3%}, range {values[0]:.3%}-{values[-1]:.3%}")
    worst = sorted(((outside_share(games, [name], set().union(*(set(games[o]) for o in names if o != name))
                                   | set(range(first)) | keep), name) for name in names), reverse=True)
    print(f"draft_vocab: leave one game out, the other games' ids + ids below {first} + kept vocabularies: "
          + ", ".join(f"{name} {value:.3%}" for value, name in worst[:5])
          + f"; median {worst[len(worst) // 2][0]:.3%}")


def corpus_tails(corpus: Path):
    manifest = json.loads((corpus / "MTP_CORPUS.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        for line in gzip.decompress((corpus / shard["name"]).read_bytes()).decode("utf-8").splitlines():
            if line:
                record = json.loads(line)
                yield record["split"], record["text"][-int(record["reply_chars"]):]


def region_of(tail: str, start: int) -> str:
    end = tail.find(THINK_END)
    if end < 0 or start < end:
        return "reasoning" if end >= 0 else "content"
    tool = tail.find(TOOL_START, end)
    return "tool_call" if 0 <= tool <= start else "content"


def count_corpus(corpus: Path, tokenizer):
    counts = {split: {region: collections.Counter() for region in REGIONS} for split in ("train", "heldout")}
    texts = collections.Counter()
    batch: list[tuple[str, str]] = []

    def flush():
        for (split, tail), encoding in zip(batch, tokenizer.encode_batch([one for _, one in batch],
                                                                         add_special_tokens=False)):
            for index, (start, _) in zip(encoding.ids, encoding.offsets):
                counts[split][region_of(tail, start)][index] += 1
        batch.clear()

    for split, tail in corpus_tails(corpus):
        texts[split] += 1
        batch.append((split, tail))
        if len(batch) >= 256:
            flush()
    flush()
    return counts, texts


def coverage(counts: dict[str, collections.Counter], kept: set[int]) -> dict[str, dict[str, float]]:
    out = {}
    for region in (*REGIONS, "all"):
        held = (sum((counts[one] for one in REGIONS), collections.Counter()) if region == "all"
                else counts[region])
        total = sum(held.values())
        outside = sum(n for index, n in held.items() if index not in kept)
        out[region] = {"tokens": total, "outside": outside, "outside_share": round(outside / max(total, 1), 5)}
    return out


def choose(counts: collections.Counter, tokenizer, size: int, rows: int,
           after: tuple[collections.Counter, ...] = ()) -> list[int]:
    chosen = base_ids(tokenizer, rows)
    for ranked in (counts, *after):
        for index, _ in ranked.most_common():
            if len(chosen) >= size:
                break
            chosen.add(index)
    return sorted(chosen)[:size]


def record_of(ids: list[int], rows: int, built_from: list[str], name: str = "") -> dict:
    return {"name": name or f"arc{len(ids) // 1024}k", "size": len(ids), "vocab_rows": rows,
            "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "built_from": built_from,
            "ids": ids}


def write(record: dict, out: str) -> None:
    Path(out).write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
    print(f"draft_vocab: wrote {out} {record['ids_sha256'][:16]}")


def corpus_main(args, tokenizer) -> None:
    corpus = Path(args.corpus)
    counts, texts = count_corpus(corpus, tokenizer)
    print(f"draft_vocab: corpus {corpus.name}: {texts['train']} train and {texts['heldout']} held-out replies")
    report: dict = {"corpus": corpus.name, "replies": dict(texts), "vocabularies": {}}
    for path in args.against:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        kept = set(record["ids"])
        report["vocabularies"][record.get("name") or Path(path).stem] = {
            "size": len(kept), "ids_sha256": record.get("ids_sha256"),
            **{split: coverage(counts[split], kept) for split in counts}}
    size = args.size or max((len(json.loads(Path(one).read_text(encoding="utf-8"))["ids"]) for one in args.against),
                            default=32768)
    train = sum(counts["train"].values(), collections.Counter())
    extra = []
    for one in args.also:
        held, _ = count_corpus(Path(one), tokenizer)
        extra.append(sum((sum(held[split].values(), collections.Counter()) for split in held), collections.Counter()))
    ids = choose(train, tokenizer, size, args.rows, tuple(extra))
    sources = [corpus.name, *(Path(one).name for one in args.also)]
    rebuilt = record_of(ids, args.rows, sources, args.name)
    kept = set(ids)
    report["vocabularies"][f"rebuilt from {corpus.name} train" + "".join(f" then {one}" for one in sources[1:])] = {
        "size": len(ids), "ids_sha256": rebuilt["ids_sha256"],
        **{split: coverage(counts[split], kept) for split in counts}}
    for name, held in report["vocabularies"].items():
        for split in ("train", "heldout"):
            print(f"draft_vocab: {name} ({held['size']} ids) on {split}: " + ", ".join(
                f"{region} {held[split][region]['outside_share']:.3%} of {held[split][region]['tokens']:,} outside"
                for region in (*REGIONS, "all")))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    if args.out:
        write(rebuilt, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", nargs="*", default=[])
    parser.add_argument("--hold-out", nargs="*", default=[])
    parser.add_argument("--corpus", default="")
    parser.add_argument("--against", nargs="*", default=[])
    parser.add_argument("--also", nargs="*", default=[])
    parser.add_argument("--report", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--rows", type=int, default=248320)
    parser.add_argument("--out", default="")
    parser.add_argument("--tokenizer", default=str(TOKENIZER))
    parser.add_argument("--union", action="store_true",
                        help="keep every id the --build runs generated, every id of the --keep vocabularies and "
                             "every id below --first, with no size cap")
    parser.add_argument("--keep", nargs="*", default=[])
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--split-games", type=int, default=0,
                        help="check generalization: build from this many games' ids, count misses on as many others")
    parser.add_argument("--splits", type=int, default=30)
    args = parser.parse_args()
    if bool(args.build) == bool(args.corpus):
        parser.error("name exactly one source: --build runs or --corpus")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    if args.corpus:
        corpus_main(args, tokenizer)
        return
    if args.union:
        keep = set()
        for path in args.keep:
            keep |= set(json.loads(Path(path).read_text(encoding="utf-8"))["ids"])
        runs = [Path(run) for run in args.build]
        if args.split_games:
            split_check(count_by_game(runs, tokenizer), args.first, keep, args.splits, args.split_games)
        counts, texts = count(runs, tokenizer)
        ids = sorted(index for index in base_ids(tokenizer, args.rows) | keep | set(range(args.first)) | set(counts)
                     if index < args.rows)
        print(f"draft_vocab: union of {texts} generated texts ({sum(counts.values())} tokens, {len(counts)} distinct), "
              f"{len(keep)} kept ids, ids below {args.first}: {len(ids)} ids")
        if args.out:
            sources = [*(run.name for run in runs), *(Path(path).name for path in args.keep)]
            write(record_of(ids, args.rows, sources + ([f"first{args.first}"] if args.first else []), args.name),
                  args.out)
        return
    if not args.out:
        parser.error("--build needs --out")
    counts, texts = count([Path(run) for run in args.build], tokenizer)
    ids = choose(counts, tokenizer, args.size or 32768, args.rows)
    total = sum(counts.values())
    inside = sum(counts[index] for index in ids)
    print(f"draft_vocab: {texts} generated texts, {total} tokens, {len(counts)} distinct, "
          f"{len(ids)} kept, {inside / max(total, 1):.4%} of the build tokens inside")
    if args.hold_out:
        held, held_texts = count([Path(run) for run in args.hold_out], tokenizer)
        held_total = sum(held.values())
        kept = set(ids)
        held_inside = sum(n for index, n in held.items() if index in kept)
        print(f"draft_vocab: held out {held_texts} texts, {held_total} tokens, "
              f"{held_inside / max(held_total, 1):.4%} inside")
    write(record_of(ids, args.rows, [Path(run).name for run in args.build], args.name), args.out)


if __name__ == "__main__":
    main()
