"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        cached = self.__dict__.get("_ascii_text")
        if cached is None:
            cached = format_grid_ascii(self.grid)
            object.__setattr__(self, "_ascii_text", cached)
        return cached

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame
    result: dict[str, Any] = field(default_factory=dict)


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
    )


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    return HistoryEntry(
        action=str(payload.get("action", "")).strip(), frame=frame,
        result=dict(payload["result"]) if isinstance(payload.get("result"), dict) else {},
    )


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
        "result": dict(getattr(entry, "result", {}) or {}),
    }


def _memfix_knob(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no", "off"}


class _StateFile:
    __slots__ = ("count", "last", "end", "size", "mtime", "current", "loaded")

    def __init__(self) -> None:
        self.count = 0
        self.last = None
        self.end = 0
        self.size = -1
        self.mtime = -1
        self.current = None
        self.loaded = []


_STATE_FILES: dict[str, _StateFile] = {}
_STATE_LOCK = threading.Lock()


def _state_key(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _entry_round_trip(entry: HistoryEntry) -> tuple[str, HistoryEntry | None]:
    text = json.dumps(history_entry_to_payload(entry))
    back = history_entry_from_payload(json.loads(text))
    if back is not None and back.frame == entry.frame:
        back = HistoryEntry(action=back.action, frame=entry.frame, result=back.result)
    return text, back


def _frame_round_trip(frame: Frame | None) -> tuple[str, Frame | None]:
    text = json.dumps(frame_to_payload(frame))
    back = frame_from_payload(json.loads(text))
    if back is not None and back == frame:
        back = frame
    return text, back


def _write_incremental(path: Path, current_frame: Frame | None, history: list[HistoryEntry]) -> None:
    key = _state_key(path)
    with _STATE_LOCK:
        state = _STATE_FILES.get(key)
        if state is None:
            state = _STATE_FILES[key] = _StateFile()
    count = state.count
    if (_stamp(path) != (state.size, state.mtime) or count > len(history)
            or (count and history[count - 1] is not state.last)):
        count = 0
    pieces: list[str] = []
    loaded: list[HistoryEntry] = []
    for entry in history[count:]:
        text, back = _entry_round_trip(entry)
        pieces.append(text)
        if back is not None:
            loaded.append(back)
    current_text, current = _frame_round_trip(current_frame)
    tail = ('], "current_frame": ' + current_text + "}").encode("ascii")
    if count == 0:
        head = ('{"history": [' + ", ".join(pieces)).encode("ascii")
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp_path, "wb") as handle:
            handle.write(head + tail)
        tmp_path.replace(path)
        state.end = len(head)
        state.loaded = loaded
    else:
        chunk = "".join(", " + piece for piece in pieces).encode("ascii")
        with open(path, "r+b") as handle:
            handle.seek(state.end)
            handle.write(chunk + tail)
            handle.truncate()
        state.end += len(chunk)
        state.loaded.extend(loaded)
    state.size, state.mtime = _stamp(path) or (-1, -1)
    state.count = len(history)
    state.last = history[-1] if history else None
    state.current = current


def forget_runtime_state(path: Path) -> None:
    with _STATE_LOCK:
        _STATE_FILES.pop(_state_key(path), None)


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    if not path.exists():
        return None, []
    if _memfix_knob("ARC3_STATE_INCREMENTAL"):
        state = _STATE_FILES.get(_state_key(path))
        if state is not None and _stamp(path) == (state.size, state.mtime):
            return state.current, list(state.loaded)
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return current_frame, history_entries


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _memfix_knob("ARC3_STATE_INCREMENTAL"):
        _write_incremental(path, current_frame, history)
        return
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
