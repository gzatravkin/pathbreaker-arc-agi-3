from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

BUNDLED_SOURCES = Path(__file__).resolve().parents[1]
if str(BUNDLED_SOURCES) not in sys.path:
    sys.path.insert(0, str(BUNDLED_SOURCES))

from solver.delivery.bundle import LOCAL_PATH_RE  # noqa: E402

CALL = re.compile(r"(?P<game>[a-z0-9]+)-[0-9a-f]+_p(?P<lane>\d+)_(?:(?P<kind>[A-Za-z_]+?)_)?s(?P<step>\d+)_c(?P<call>\d+)")
MANIFEST = "MTP_CORPUS.json"
ESCAPE_LIKE = re.compile(r"[A-Za-z]:\\[ntr0](?![A-Za-z0-9_.$-])")
SHARD_RECORDS = 400
EFFORTS = (None, "low", "medium")

_state: dict[str, Any] = {}


def chat_environment():
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def template_messages(messages: list[dict[str, Any]], reasoning_key: str) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        one = dict(message)
        if one.get("role") == "assistant":
            said = one.pop("reasoning", None)
            if said is None:
                said = one.get("reasoning_content")
            one.pop("reasoning_content", None)
            if reasoning_key and said:
                one[reasoning_key] = said
            calls = []
            for call in one.get("tool_calls") or []:
                function = dict(call.get("function") or {})
                raw = function.get("arguments")
                if isinstance(raw, str):
                    try:
                        function["arguments"] = json.loads(raw) if raw.strip() else {}
                    except json.JSONDecodeError:
                        function["arguments"] = {"code": raw}
                calls.append({**call, "function": function})
            if calls:
                one["tool_calls"] = calls
        out.append(one)
    return out


def sent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(messages)
    while out and out[-1].get("role") == "assistant":
        out.pop()
    return out


def images_of(messages: list[dict[str, Any]]) -> list[str]:
    found = []
    for message in messages:
        body = message.get("content")
        if isinstance(body, list):
            for part in body:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    found.append(str((part.get("image_url") or {}).get("url") or ""))
    return found


def reply_message(reply: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": reply.get("content") or "",
            "reasoning": reply.get("reasoning") or reply.get("reasoning_content") or "",
            "tool_calls": reply.get("tool_calls") or []}


def render(template, messages, tools, effort, thinking, add_generation_prompt):
    kwargs: dict[str, Any] = {"enable_thinking": thinking}
    if thinking and effort:
        kwargs["reasoning_effort"] = effort
    return template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                           **kwargs)


def variants(tool_sets: list[Any]):
    for reasoning_key in ("reasoning_content", ""):
        for tools in tool_sets:
            for effort in EFFORTS:
                yield reasoning_key, tools, effort


def init_worker(template_path: str, tokenizer_path: str, tool_sets: list[Any]) -> None:
    from tokenizers import Tokenizer
    _state["template"] = chat_environment().from_string(Path(template_path).read_text(encoding="utf-8"))
    _state["tokenizer"] = Tokenizer.from_file(tokenizer_path)
    _state["tools"] = tool_sets


def count(text: str) -> int:
    return len(_state["tokenizer"].encode(text, add_special_tokens=False).ids)


def build_record(job: tuple[str, str, str]) -> dict[str, Any] | None:
    run, path, split = job
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"skip": "unreadable"}
    reply = raw.get("reply") if isinstance(raw, dict) else None
    if not isinstance(reply, dict) or not isinstance(raw.get("messages"), list):
        return {"skip": "no reply"}
    usage = reply.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    if completion <= 0:
        return {"skip": "empty reply"}
    prompt = sent(raw["messages"])
    if not prompt or prompt[-1].get("role") not in ("user", "tool"):
        return {"skip": "prompt does not end with a user turn"}
    pictures = images_of(prompt)
    image_tokens = int(((usage.get("prompt_tokens_details") or {}).get("multimodal_tokens") or {}).get("image") or 0)
    recorded = int(usage.get("prompt_tokens") or 0)
    thinking = bool(reply.get("reasoning") or reply.get("reasoning_content")) or \
        int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) > 0
    template = _state["template"]
    best = None
    for reasoning_key, tools, effort in variants(_state["tools"]):
        try:
            head = render(template, template_messages(prompt, reasoning_key), tools, effort, thinking, True)
        except Exception:
            continue
        predicted = count(head) - len(pictures) + image_tokens
        miss = abs(predicted - recorded)
        if best is None or miss < best[0]:
            best = (miss, reasoning_key, tools, effort, head)
        if miss == 0:
            break
    if best is None:
        return {"skip": "template refused every variant"}
    miss, reasoning_key, tools, effort, head = best
    whole = render(template, template_messages(prompt + [reply_message(reply)], reasoning_key), tools, effort,
                   thinking, False)
    if not whole.startswith(head):
        return {"skip": "the rendered reply does not extend the rendered prompt"}
    tail = whole[len(head):]
    if tail.endswith("\n"):
        tail = tail[:-1]
    text = head + tail
    stem = Path(path).stem
    found = CALL.fullmatch(stem)
    digest = hashlib.sha256((text + "\x00".join(pictures)).encode("utf-8")).hexdigest()
    return {"id": digest[:24], "run": run, "stem": stem, "game": found["game"] if found else stem.split("-")[0],
            "lane": int(found["lane"]) if found else 0, "kind": (found["kind"] or "main") if found else "other",
            "step": int(found["step"]) if found else 0, "call": int(found["call"]) if found else 0,
            "split": split, "text": text, "reply_chars": len(tail), "reply_tokens": count(tail),
            "prompt_tokens": recorded, "completion_tokens": completion, "image_tokens": image_tokens,
            "images": pictures, "prompt_miss": miss, "effort": effort, "tools": bool(tools),
            "thinking": thinking, "reasoning_kept": bool(reasoning_key)}


def canonical(message: dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, ensure_ascii=False)


def reply_after(prompt: list[dict[str, Any]], later: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not prompt:
        return None
    last = canonical(prompt[-1])
    before = canonical(prompt[-2]) if len(prompt) > 1 else None
    for index in range(len(later) - 2, -1, -1):
        if later[index + 1].get("role") != "assistant" or canonical(later[index]) != last:
            continue
        if before is not None and index > 0 and canonical(later[index - 1]) != before:
            continue
        return later[index + 1]
    return None


def render_with(template, messages, tools, kwargs, add_generation_prompt):
    return template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


def request_record(run: str, stem: str, split: str, row: dict[str, Any], reply: dict[str, Any],
                   step: int, call: int, game: str) -> dict[str, Any]:
    usage = row.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    if completion <= 0:
        return {"skip": "empty reply"}
    prompt = list(row["messages"])
    if not prompt or prompt[-1].get("role") not in ("user", "tool"):
        return {"skip": "prompt does not end with a user turn"}
    pictures = images_of(prompt)
    image_tokens = int(((usage.get("prompt_tokens_details") or {}).get("multimodal_tokens") or {}).get("image") or 0)
    recorded = int(usage.get("prompt_tokens") or 0)
    kwargs = dict(row.get("chat_template_kwargs") or {})
    tools = row.get("tools") or None
    template = _state["template"]
    best = None
    for reasoning_key in ("reasoning_content", ""):
        try:
            head = render_with(template, template_messages(prompt, reasoning_key), tools, kwargs, True)
        except Exception:
            continue
        miss = abs(count(head) - len(pictures) + image_tokens - recorded)
        if best is None or miss < best[0]:
            best = (miss, reasoning_key, head)
        if miss == 0:
            break
    if best is None:
        return {"skip": "template refused every variant"}
    miss, reasoning_key, head = best
    whole = render_with(template, template_messages(prompt + [reply_message(reply)], reasoning_key), tools, kwargs,
                        False)
    if not whole.startswith(head):
        return {"skip": "the rendered reply does not extend the rendered prompt"}
    tail = whole[len(head):]
    if tail.endswith("\n"):
        tail = tail[:-1]
    text = head + tail
    digest = hashlib.sha256((text + "\x00".join(pictures)).encode("utf-8")).hexdigest()
    thinking = kwargs.get("enable_thinking", True) is not False
    return {"id": digest[:24], "run": run, "stem": stem, "game": game, "lane": 0, "kind": "main", "step": step,
            "call": call, "split": split, "text": text, "reply_chars": len(tail), "reply_tokens": count(tail),
            "prompt_tokens": recorded, "completion_tokens": completion, "image_tokens": image_tokens,
            "images": pictures, "prompt_miss": miss, "effort": kwargs.get("reasoning_effort"), "tools": bool(tools),
            "thinking": thinking, "reasoning_kept": bool(reasoning_key),
            "finish_reason": str(row.get("finish_reason") or "")}


def build_requests(job: tuple[str, str, str]) -> list[dict[str, Any]]:
    run, path, split = job
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return [{"skip": "unreadable"}]
    base = Path(path).name.removesuffix("_requests.jsonl")
    game = base.split("-")[0]
    out = []
    call = 0
    for index, row in enumerate(rows):
        if row.get("event") != "response" or not isinstance(row.get("messages"), list):
            continue
        call += 1
        later = next((one for one in rows[index + 1:] if one.get("event") == "request"), None)
        if later is None:
            out.append({"skip": "the game's last call: no later request carries its reply"})
            continue
        reply = reply_after(row["messages"], later.get("messages") or [])
        if reply is None:
            out.append({"skip": "the next request does not carry the reply"})
            continue
        step = int(row.get("analysis_step") or 0)
        stem = f"{base}_s{step}_c{call}"
        out.append(request_record(run, stem, split, row, reply, step, call, game))
    return out


def plan_requests(roots: list[Path], every: int, named: list[str]) -> tuple[list[tuple[str, str, str]], dict]:
    files = [(root.name, path) for root in roots for path in sorted(root.glob("*_requests.jsonl"))]
    games = sorted({path.name.split("-")[0] for _, path in files})
    held = heldout_games(games, every, named)
    jobs = [(run, str(path), "heldout" if path.name.split("-")[0] in held else "train") for run, path in files]
    return jobs, {"runs": sorted({run for run, _ in files}), "games": games, "heldout_games": sorted(held),
                  "source": "request logs"}


def same_weights(run: Path, mark: str, reads: str) -> bool:
    log = run / "setup-diagnostics.log"
    if not log.is_file():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    return mark in text and f"draft model read {reads} tensors" in text


def heldout_games(games: list[str], every: int, named: list[str]) -> set[str]:
    if named:
        return set(named)
    return {game for index, game in enumerate(sorted(games)) if every and index % every == every - 1}


def plan(root: Path, mark: str, reads: str, every: int, named: list[str]) -> tuple[list[tuple[str, str, str]], dict]:
    runs = sorted(one.parent for one in root.glob("*/call_requests") if one.is_dir()
                  and same_weights(one.parent, mark, reads))
    calls = [(run.name, path) for run in runs for path in sorted((run / "call_requests").glob("*.json"))]
    games = sorted({(CALL.fullmatch(path.stem) or {"game": path.stem.split("-")[0]})["game"]
                    for _, path in calls})
    held = heldout_games(games, every, named)
    jobs = []
    for run, path in calls:
        found = CALL.fullmatch(path.stem)
        game = found["game"] if found else path.stem.split("-")[0]
        jobs.append((run, str(path), "heldout" if game in held else "train"))
    return jobs, {"runs": [run.name for run in runs], "games": games, "heldout_games": sorted(held)}


def order_key(record: dict[str, Any]) -> tuple:
    return (record["run"], record["game"], record["stem"].split("_p")[0], record["lane"], record["kind"],
            record["step"], record["call"])


def path_hits(record: dict[str, Any]) -> list[str]:
    return sorted({value[hit.start():hit.end() + 12] for value in record.values() if isinstance(value, str)
                   for hit in LOCAL_PATH_RE.finditer(value) if not ESCAPE_LIKE.match(value, hit.start())})


def write(records: list[dict[str, Any]], out: Path, about: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, len(records), SHARD_RECORDS):
        name = f"part-{start // SHARD_RECORDS:04d}.jsonl.gz"
        part = records[start:start + SHARD_RECORDS]
        body = "".join(json.dumps(one, ensure_ascii=False) + "\n" for one in part)
        hits = sorted({hit for one in part for hit in path_hits(one)})
        if hits:
            raise SystemExit(f"{name} would carry local path(s) {hits[:5]}; nothing written")
        text = body.encode("utf-8")
        blob = gzip.compress(text, compresslevel=6, mtime=0)
        (out / name).write_bytes(blob)
        shards.append({"name": name, "records": len(records[start:start + SHARD_RECORDS]),
                       "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                       "text_sha256": hashlib.sha256(text).hexdigest(), "text_bytes": len(text)})
    manifest = {**about, "shards": shards}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def summarize(records: list[dict[str, Any]], skipped: dict[str, int], duplicates: int) -> dict[str, Any]:
    out: dict[str, Any] = {"records": len(records), "skipped": skipped, "duplicates": duplicates}
    for split in ("train", "heldout"):
        part = [one for one in records if one["split"] == split]
        out[split] = {"records": len(part), "reply_tokens": sum(one["reply_tokens"] for one in part),
                      "prompt_tokens": sum(one["prompt_tokens"] for one in part),
                      "games": len({one["game"] for one in part})}
    if records:
        misses = [one["prompt_miss"] for one in records]
        drift = [one["reply_tokens"] - one["completion_tokens"] for one in records]
        out["prompt_exact"] = sum(1 for one in misses if one == 0)
        out["prompt_within_8"] = sum(1 for one in misses if one <= 8)
        out["reply_drift_median"] = statistics.median(drift)
        out["reply_drift_p90_abs"] = sorted(abs(one) for one in drift)[int(0.9 * (len(drift) - 1))]
        shared = 0
        previous = None
        for one in records:
            if previous is not None and previous["run"] == one["run"] and previous["game"] == one["game"]:
                limit = min(len(previous["text"]), len(one["text"]))
                same = 0
                while same < limit and previous["text"][same] == one["text"][same]:
                    same += 1
                shared += same
            previous = one
        out["text_chars"] = sum(len(one["text"]) for one in records)
        out["text_chars_shared_with_previous"] = shared
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recorded game calls into MTP training sequences.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--requests", nargs="*", type=Path, default=[])
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-mark", default="qwen3-8-swift-nvfp4")
    parser.add_argument("--mtp-reads", default="2431")
    parser.add_argument("--heldout-every", type=int, default=6)
    parser.add_argument("--heldout", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--drop-path-like", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.root) == bool(args.requests):
        parser.error("name exactly one source: --root (call_requests runs) or --requests (request-log runs)")
    named = [one for one in args.heldout.split(",") if one]
    if args.requests:
        tool_sets: list[Any] = []
        jobs, about = plan_requests(args.requests, args.heldout_every, named)
        worker, chunks = build_requests, 1
    else:
        from solver.agent.core import duck
        from solver.agent.core.prompt import TOOL_NAME
        tool_sets = [duck.tool_schema(TOOL_NAME, "", False, lean=True, kept=True),
                     duck.tool_schema(TOOL_NAME, "", False, lean=True, kept=False), None]
        jobs, about = plan(args.root, args.model_mark, args.mtp_reads, args.heldout_every, named)
        worker, chunks = build_record, 8
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"mtp_corpus: {len(about['runs'])} run(s), {len(jobs)} job(s), held-out games {about['heldout_games']}",
          flush=True)
    with Pool(args.workers, initializer=init_worker,
              initargs=(str(args.template), str(args.tokenizer), tool_sets)) as pool:
        built = pool.map(worker, jobs, chunksize=chunks)
    if args.requests:
        built = [one for many in built for one in many]
    skipped: dict[str, int] = {}
    seen: set[str] = set()
    records = []
    duplicates = 0
    for one in built:
        if one is None or "skip" in one:
            reason = (one or {}).get("skip", "none")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if one["id"] in seen:
            duplicates += 1
            continue
        if args.drop_path_like and path_hits(one):
            reason = "a string reads as a local path"
            skipped[reason] = skipped.get(reason, 0) + 1
            print(f"mtp_corpus: {one['stem']} of {one['run']} left out, it carries {path_hits(one)[:3]}", flush=True)
            continue
        seen.add(one["id"])
        records.append(one)
    records.sort(key=order_key)
    stats = summarize(records, skipped, duplicates)
    manifest = write(records, args.out, {**about, "stats": stats})
    print(json.dumps(stats, indent=1), flush=True)
    print(f"mtp_corpus: wrote {len(manifest['shards'])} shard(s), "
          f"{sum(one['bytes'] for one in manifest['shards']) / 2**20:.1f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
