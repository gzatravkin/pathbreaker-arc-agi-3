import json
import os
import re
import threading
import time
from typing import Any

import requests

MARK = "taaf_hot_spans"
KEY_HEADER = "X-Session-ID"
RECORD_ENV = "ARC3_HOT_RECORD"
RETURN_ENV = "ARC3_HOT_RETURN"
RETURN_PASSES_ENV = "ARC3_HOT_RETURN_PASSES"
HELPER_ENV = "ARC3_HELPER_HOT"
RETURN_HEAD = ("Your own notes from earlier in this game, removed from your context by the trim. They are old: the "
               "current board and step follow below them.")
HELPER_HEAD = ("What the player keeps re-reading: the text its thinking relies on right now. It may hold a wrong "
               "belief; check it against the game.")
LEVEL = re.compile(r"\blevel (\d+)\b", re.I)
TEMPLATE = re.compile(r"<\|im_start\|>(?:system|user|assistant|tool)?\n?|<\|[a-z_]+\|>"
                      r"|</?(?:tool_call|tool_response|think|tts_[a-z_]+)>|<function=[^>\n]*>|<parameter=[^>\n]*>"
                      r"|</parameter>|</function>")
LINE_CHARS = 24
LONG_CHARS = 240
_lock = threading.Lock()
_said: set[str] = set()
_started = time.monotonic()


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip() == "1"


def passes_on(owner: Any, everywhere: str, passes: str) -> bool:
    if flag(everywhere):
        return True
    named = {int(part) for part in str(os.environ.get(passes) or "").split(",") if part.strip().isdigit()}
    said = getattr(owner, "_burn_said", None)
    return bool(named and said and said[0]) and said[1] in named


def say_once(kind: str, text: str) -> None:
    with _lock:
        if kind in _said:
            return
        _said.add(kind)
    print(text, flush=True)


def query(base_url: str, model: str, key: str, text_tokens: int = 0, below: int | None = None,
          timeout: float = 5.0) -> dict[str, Any] | None:
    if not key or not base_url:
        return None
    marks: dict[str, Any] = {MARK: 1}
    if text_tokens > 0:
        marks["text_tokens"] = int(text_tokens)
    if below is not None:
        marks["below"] = max(0, int(below))
    body = {"model": model, "prompt": ".", "max_tokens": 1, "temperature": 0.0, "vllm_xargs": marks}
    began = time.monotonic()
    try:
        answer = requests.post(f"{base_url.rstrip('/')}/completions", json=body, timeout=timeout,
                               headers={"Content-Type": "application/json", KEY_HEADER: key})
        answer.raise_for_status()
        result = json.loads(answer.json()["choices"][0]["text"])
    except Exception as exc:
        say_once("query_failed", f"HOT SPANS query failed for {key}: {type(exc).__name__}: {str(exc)[:200]}; the "
                                 f"run goes on without hot spans for that call")
        return None
    if not isinstance(result, dict):
        return None
    result["client_ms"] = round((time.monotonic() - began) * 1000.0, 2)
    if result.get("why"):
        say_once("why_" + str(result["why"])[:40], f"HOT SPANS answered without spans for {key}: {result['why']}")
    return result


def record(path: str, game: str, turn: int, step: int, cut: bool, prompt_tokens: int,
           answer: dict[str, Any]) -> None:
    if str(os.environ.get("TAAF_RUN_AS_SUBMISSION") or "").strip() == "1":
        return
    row = {"game": game, "turn": turn, "step": step, "minute": round((time.monotonic() - _started) / 60.0, 2), "cut": cut,
           "prompt_tokens": prompt_tokens,
           **{name: answer.get(name) for name in ("rows", "region", "mapped", "unmapped", "computed", "align", "below",
                                                   "listed", "text_tokens", "worker_ms", "call_ms", "client_ms",
                                                   "why")},
           "blocks": answer.get("blocks") or [], "spans": answer.get("spans") or []}
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    say_once("record", f"HOT SPANS recorder first used by {game} at turn {turn}{' (a cut)' if cut else ''}: "
                       f"{len(row['spans'])} spans and {len(row['blocks'])} blocks of {row['rows']} hot rows, "
                       f"{row['client_ms']} ms; every {env_int('ARC3_HOT_EVERY', 3)}rd player turn and every cut go "
                       f"to {path}")


def _argument_text(raw: Any) -> list[str]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return [str(raw)]
    if isinstance(value, dict):
        return [item if isinstance(item, str) else json.dumps(item) for item in value.values()]
    return [str(raw)]


def message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    parts = [content] if isinstance(content, str) else [
        part.get("text") or "" for part in content or () if isinstance(part, dict) and part.get("type") == "text"]
    for call in message.get("tool_calls") or ():
        parts += _argument_text((call.get("function") or {}).get("arguments") or "")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return "\n".join(str(part) for part in [reasoning, *parts] if part)


def plain(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", TEMPLATE.sub("", text))


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("<image>", " ")).strip()


def lines_of(text: str) -> list[str]:
    return [one for one in (squeeze(line) for line in text.split("\n")) if len(one) >= LINE_CHARS]


def fragments(answer: dict[str, Any] | None, gone: list[Any] | None = None,
              kept: list[Any] | None = None) -> list[dict[str, Any]]:
    found = []
    for span in (answer or {}).get("spans") or []:
        text = plain(str(span.get("text") or "")).strip("\n")
        if not squeeze(text):
            continue
        found.append({"text": text, "start": int(span.get("text_start", span.get("start", 0)) or 0),
                      "rows": int(span.get("rows") or 0)})
    if gone is None:
        return sorted(found, key=lambda one: one["start"])
    gone_text = [squeeze(message_text(message)) for message in gone]
    kept_text = "\n".join(squeeze(message_text(message)) for message in kept or ())
    chosen = []
    for one in found:
        lost = [line for line in lines_of(one["text"]) if line not in kept_text]
        at = next((index for index, text in enumerate(gone_text) if any(line in text for line in lost)), None)
        if at is not None:
            source = gone[at]
            chosen.append(dict(one, at=at, role=str(source.get("role") or "") if isinstance(source, dict) else ""))
    return sorted(chosen, key=lambda one: one["start"])


def notes(pieces: list[dict[str, Any]], gone: list[Any], kept: list[Any] | None = None) -> list[dict[str, Any]]:
    own = {squeeze(line) for message in gone if isinstance(message, dict) and message.get("role") == "assistant"
           for line in message_text(message).split("\n")}
    still = {squeeze(line) for message in kept or () for line in message_text(message).split("\n")}
    seen: set[str] = set()
    shown = []
    for one in pieces:
        lines = []
        for raw in one["text"].split("\n"):
            line = raw.rstrip()
            key = squeeze(line)
            if key and key in own and key not in still and len(line) <= LONG_CHARS and key not in seen:
                seen.add(key)
                lines.append(line)
        if lines:
            shown.append(dict(one, text="\n".join(lines)))
    return shown


def notes_block(pieces: list[dict[str, Any]], levels: list[int | None]) -> str:
    parts = [RETURN_HEAD]
    for one, level in zip(pieces, levels):
        label = f"[your earlier turn, level {level}]" if level else "[your earlier turn]"
        parts.append(label + "\n" + "\n".join("> " + line for line in one["text"].split("\n")))
    return "\n\n".join(parts)


def level_before(messages: list[Any], at: int) -> int | None:
    for message in reversed(messages[:at + 1]):
        if isinstance(message, dict) and message.get("role") in ("user", "tool"):
            found = LEVEL.findall(message_text(message))
            if found:
                return int(found[-1])
    return None


def block(head: str, pieces: list[dict[str, Any]], levels: list[int | None] | None = None) -> str:
    lines = [head]
    for index, one in enumerate(pieces):
        level = levels[index] if levels else None
        lines.append((f"(level {level}) " if level else "") + one["text"])
    return "\n\n".join(lines)


def capped(pieces: list[dict[str, Any]], tokens: int) -> list[dict[str, Any]]:
    shown, used = [], 0
    for one in pieces:
        if shown and used + len(one["text"]) > tokens * 4:
            break
        shown.append(one)
        used += len(one["text"])
    return shown
