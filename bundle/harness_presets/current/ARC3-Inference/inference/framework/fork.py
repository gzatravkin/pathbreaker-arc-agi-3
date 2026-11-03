from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import arcengine

DIR_KEY = "ARC3_FORK_DIR"
PHASE_KEY = "ARC3_FORK_PHASE"
BUDGET_KEY = "ARC3_FORK_BUDGET_TOKENS"
ARM_KEY = "ARC3_FORK_ARM"
WARM_KEY = "ARC3_FORK_WARM_WAIT_S"
STOP_AT_LEVEL_KEY = "ARC3_FORK_STOP_AT_LEVEL"
LEAN_KEY = "ARC3_LEAN_SYSTEM"
META_MARK = "\n\nThe arcade's metadata for this game, as its API returned it:\n"
PLAN_NAME = "plan.json"
SOURCE_NAME = "source"
REQUESTS_SUFFIX = "_requests.jsonl"
EVENTS_SUFFIX = "_events.jsonl"
DEFAULT_BUDGET = 35000.0
PINS_HEAD = "Kept by the harness from earlier in this game:"
DEBRIEF_PIN = re.compile(r"^- Your debrief after completing level \d+: .*$", re.M)
RESET_PIN = re.compile(r"^- You pressed RESET on level .*$", re.M)
RETAINED = re.compile(r"Retained your function (\w+)\(")
KEPT = re.compile(r"Kept for later calls: ([^\n]+)")
STEP = re.compile(r"Current state: step (\d+), level (\d+)")
MOUSE = re.compile(r"MOUSE\(row=(-?\d+), col=(-?\d+)\)")
_PLAN_LOCK = threading.Lock()
WALL_EVERY_S = 600.0
_WALLS: dict[str, Any] = {"start": None, "said": 0.0, "copies": {}, "broke": {}}
ARMS_EVERY_S = 300.0
_ARMS: dict[str, Any] = {"start": None, "said": 0.0, "ducks": {}}


def ducks() -> Any:
    try:
        from inference.agent import duck_arms
    except ImportError:
        return None
    return duck_arms if duck_arms.on() else None


def judge() -> Any:
    try:
        from inference.framework import judge_arms
    except ImportError:
        return None
    return judge_arms


def phase() -> str:
    said = os.environ.get(PHASE_KEY, "").strip().lower()
    if said == "restore" and os.environ.get(DIR_KEY, "").strip():
        return said
    return ""


def restoring() -> bool:
    return phase() == "restore"


def folder() -> Path:
    return Path(os.environ[DIR_KEY].strip())


def setting(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return float(default)


def arm() -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", os.environ.get(ARM_KEY, "").strip()) or "arm"


def name(game_id: str, pass_index: int) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', game_id)}_p{int(pass_index)}"


def record_path(stem: str, label: str) -> Path:
    return folder() / f"{stem}.{label}.json"


def game_id(session: Any) -> str:
    run = session.game.game_run
    return run.game_id if run is not None else str(session.game_index)


def stem_of(session: Any) -> str:
    return name(game_id(session), session.pass_index)


def grid(game: Any) -> tuple[tuple[int, ...], ...]:
    data = game.current_state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def grid_sha(rows: Any) -> str:
    return hashlib.sha1(json.dumps([list(row) for row in rows]).encode("utf-8")).hexdigest()


def generated(analyzer: Any) -> int:
    return int(getattr(analyzer, "_session_generated_tokens", 0) or 0)


def total(analyzer: Any) -> int:
    return int(getattr(analyzer, "_session_total_tokens", 0) or 0)


def progress(session: Any) -> dict[str, Any]:
    run = session.game.game_run
    return {
        "levels": int(run.levels_completed),
        "actions": len(run.history),
        "actions_per_level": [int(count) for count in run.actions_per_level],
        "state": run.state,
        "generated_tokens": generated(session.analyzer),
        "total_tokens": total(session.analyzer),
    }


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def source_path(source: str, suffix: str) -> Path:
    return folder() / SOURCE_NAME / f"{source}{suffix}"


def action_of(event: dict[str, Any]) -> list[Any]:
    label = str(event.get("action_name") or "")
    data = event.get("action_data")
    if not isinstance(data, dict):
        found = MOUSE.search(str(event.get("action_display") or ""))
        data = {"x": int(found.group(2)), "y": int(found.group(1))} if label == "ACTION6" and found else {}
    return [label, dict(data)]


def source_actions(events: list[dict[str, Any]]) -> list[tuple[list[Any], Any, bool]]:
    out = []
    for event in events:
        if event.get("type") != "action" or not event.get("action_name"):
            continue
        out.append((action_of(event), event.get("board"), bool(event.get("automatic"))))
    return out


def turn_request(rows: list[dict[str, Any]], turn: int) -> int | None:
    found = [index for index, row in enumerate(rows) if row.get("event") == "request"
             and int(row.get("analysis_step") or -1) == turn and not row.get("fork")]
    if not found:
        return None
    return min(found, key=lambda index: (int(rows[index].get("request_index_within_turn") or 0), index))


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text") or "") for part in content or () if isinstance(part, dict))


def strip_pins(message: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(message)
    content = out.get("content")
    if isinstance(content, list):
        out["content"] = [part for part in content if not (isinstance(part, dict) and part.get("type") == "text"
                                                            and str(part.get("text") or "").startswith(PINS_HEAD))]
    return out


def pins_of(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    text = "\n".join(text_of(message) for message in messages)
    debriefs = list(dict.fromkeys(DEBRIEF_PIN.findall(text)))
    resets = list(dict.fromkeys(RESET_PIN.findall(text)))
    return debriefs, resets


def python_calls(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    for row in rows:
        for message in row.get("messages") or []:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    key = str(call.get("id") or "")
                    if not key or key in calls:
                        continue
                    try:
                        code = json.loads((call.get("function") or {}).get("arguments") or "{}").get("code", "")
                    except (ValueError, AttributeError):
                        code = ""
                    calls[key] = str(code or "")
            elif message.get("role") == "tool":
                key = str(message.get("tool_call_id") or "")
                if key and key not in results:
                    results[key] = text_of(message)
    return [(calls[key], results.get(key, "")) for key in calls]


def defined(code: str) -> dict[str, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(code, node)
            if segment is not None:
                start = min([node.lineno] + [one.lineno for one in node.decorator_list])
                lines = code.splitlines()
                found[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    return found


def kept_functions(rows: list[dict[str, Any]]) -> dict[str, str]:
    kept: dict[str, str] = {}
    for code, result in python_calls(rows):
        names = RETAINED.findall(result) + [one for line in KEPT.findall(result) for one in re.findall(r"(\w+)\(", line)]
        if not names:
            continue
        sources = defined(code)
        for one in names:
            if one in sources:
                kept[one] = sources[one]
    return kept


def replay(session: Any, actions: list[tuple[list[Any], Any, bool]]) -> int | None:
    first = None
    for index, ((label, data), board, automatic) in enumerate(actions, start=1):
        action = arcengine.ActionInput(id=arcengine.GameAction[label], data=dict(data))
        session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0,
                                flush_viewer_payload=False, automatic=automatic)
        if first is None and board is not None and grid(session.game) != tuple(tuple(int(c) for c in row) for row in board):
            first = index
    return first


def no_warmup(solver: Any) -> None:
    lock = getattr(solver, "_warmup_lock", None)
    if lock is None:
        solver._warmup_remaining = 0
        return
    with lock:
        solver._warmup_remaining = 0


def warm_wait() -> float:
    return setting(WARM_KEY, 0.0)


def warm_path(gid: str) -> Path:
    return folder() / f"{name(gid, 0)}.warm"


def mark_warm(session: Any, why: str) -> None:
    if int(session.pass_index) != 0 or getattr(session, "fork_warm_marked", False) or warm_wait() <= 0:
        return
    session.fork_warm_marked = True
    try:
        warm_path(game_id(session)).write_text(why, encoding="utf-8")
    except OSError as exc:
        print(f"fork warm: {stem_of(session)} could not mark itself warm: {exc!r}", flush=True)
        return
    print(f"fork warm: {stem_of(session)} {why}; its copies start now", flush=True)


def wait_warm(session: Any) -> None:
    if int(session.pass_index) == 0 or getattr(session, "fork_warm_waited", False) or warm_wait() <= 0:
        return
    session.fork_warm_waited = True
    began = time.monotonic()
    path = warm_path(game_id(session))
    while not path.exists() and time.monotonic() - began < warm_wait():
        time.sleep(2.0)
    print(f"fork warm: {stem_of(session)} waited {time.monotonic() - began:.0f}s for copy 0 "
          f"({'it is warm' if path.exists() else 'timed out, starting cold'})", flush=True)


def restore(session: Any) -> bool | None:
    done = _restore(session)
    if done is False:
        mark_warm(session, "was not played")
    return done


def lean_prompt(analyzer: Any, prompt: str) -> str:
    if os.environ.get(LEAN_KEY, "").strip() != "1":
        return prompt
    try:
        from inference.agent import lean_system
    except ImportError:
        return prompt
    if prompt.startswith(lean_system.TEXT[:80]):
        return prompt
    module = sys.modules.get(type(analyzer).__module__)
    spot = prompt.find(META_MARK)
    tail = prompt[spot:] if spot >= 0 else ""
    extra = module._prefill_system() if module is not None and hasattr(module, "_prefill_system") else ""
    short = lean_system.system(prompt, str(getattr(analyzer, "_forum_key", "") or "")) + extra + tail
    print(f"fork restore: the short system message replaces the recorded one ({len(short)} characters instead of "
          f"{len(prompt)})", flush=True)
    return short


def seat(analyzer: Any, session: Any, rows: list[dict[str, Any]], at: int, fresh: bool,
         level_start: bool = False) -> dict[str, Any]:
    messages = rows[at]["messages"]
    analyzer._session_runtime_dir = session.state_path.parent
    analyzer._step_env_callback = None
    analyzer._game_meta_told = True
    analyzer._system_prompt = lean_prompt(analyzer, str(messages[0].get("content") or ""))
    if fresh:
        analyzer._history_messages = []
        return {"history": 0, "kept": 0, "debriefs": 0, "resets": 0, "notes": 0}
    analyzer._history_messages = copy.deepcopy(messages[1:-1])
    analyzer._fork_opener = strip_pins(messages[-1])
    analyzer._kept_functions = kept_functions(rows[:at + 1])
    debriefs, resets = pins_of(messages)
    analyzer._knob_debriefs, analyzer._knob_resets = debriefs, resets
    found = STEP.search(text_of(messages[-1]))
    if found:
        analyzer._knob_level = int(found.group(2)) - (1 if level_start else 0)
        analyzer._knob_picture_level = int(found.group(2))
    analyzer._knob_start = 0
    notes = 0
    update = getattr(analyzer, "_update_summarized_knowledge_from_assistant", None)
    if update is not None:
        seen = set()
        for row in rows[:at + 1]:
            for message in row.get("messages") or []:
                if message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if not isinstance(content, str) or not content.strip() or content in seen:
                    continue
                seen.add(content)
                try:
                    update(content)
                    notes += 1
                except Exception as exc:
                    print(f"fork restore: a note update failed: {exc!r}", flush=True)
    return {"history": len(analyzer._history_messages), "kept": len(analyzer._kept_functions),
            "debriefs": len(debriefs), "resets": len(resets), "notes": notes}


def _restore(session: Any) -> bool | None:
    if not restoring():
        return None
    stem = stem_of(session)
    try:
        plan = json.loads((folder() / PLAN_NAME).read_text(encoding="utf-8"))
        item = plan.get(stem)
        if not item or item.get("budget_tokens") is None or not item.get("source") or item.get("turn") is None:
            print(f"fork restore: {stem} has no source, turn and token budget in the plan; not played", flush=True)
            return False
        no_warmup(session.solver)
        source, turn, fresh = str(item["source"]), int(item["turn"]), bool(item.get("fresh"))
        rows = jsonl(source_path(source, REQUESTS_SUFFIX))
        at = turn_request(rows, turn)
        if at is None:
            print(f"fork restore: {source} has no recorded request at turn {turn}; {stem} not played", flush=True)
            return False
        found = STEP.search(text_of(rows[at]["messages"][-1]))
        if found is None:
            print(f"fork restore: the turn {turn} request of {source} does not name its step; {stem} not played",
                  flush=True)
            return False
        done = int(found.group(1)) - 1
        actions = source_actions(jsonl(source_path(source, EVENTS_SUFFIX)))
        if len(actions) < done:
            print(f"fork restore: {source} events hold {len(actions)} actions, the turn needs {done}; {stem} not "
                  f"played", flush=True)
            return False
        began = time.monotonic()
        session.seed_initial_history()
        diverged = replay(session, actions[:done])
        run = session.game.game_run
        level = int(found.group(2))
        if diverged is not None or len(run.history) != done or int(run.levels_completed) + 1 != level:
            print(f"fork restore: {stem} replaying {done} actions of {source} does NOT reach turn {turn} (first "
                  f"differing board {diverged}, {len(run.history)} actions, level {int(run.levels_completed) + 1} "
                  f"against {level}); not played", flush=True)
            return False
        analyzer = session.analyzer
        seated = seat(analyzer, session, rows, at, fresh, bool(item.get("level_start")))
        if item.get("helper"):
            helper_path = folder() / SOURCE_NAME / str(item["helper"])
            try:
                analyzer._fork_helper = json.loads(helper_path.read_text(encoding="utf-8"))
                print(f"fork restore: {stem} takes the helper of {source} recorded in {helper_path.name}", flush=True)
            except (OSError, ValueError) as exc:
                print(f"fork restore: {stem} has no readable helper {helper_path.name}: {exc!r}", flush=True)
        session.analysis_step = turn - 1
        session.token_baseline = generated(analyzer)
        session.game_token_baseline = generated(analyzer)
        session.started_at = time.monotonic()
        session.fork_restored = True
        session.fork_clock = time.monotonic()
        session.fork_base = generated(analyzer)
        session.fork_budget = float(item["budget_tokens"])
        session.fork_levels0 = int(run.levels_completed)
        session.fork_actions0 = len(run.history)
        session.fork_source = (source, turn, at)
        walls_join(session)
        arms_join(session, item)
        if judge() is not None:
            judge().restored(session, stem)
        if ducks() is not None:
            ducks().restored(session, stem)
        print(f"fork restore: {stem} arm {arm()} from {source} turn {turn}: replayed {done} actions in "
              f"{time.monotonic() - began:.1f}s, level {level}, boards match; "
              + ("a fresh agent with no history" if fresh else
                 f"{seated['history']} history messages, {seated['kept']} kept functions, {seated['debriefs']} "
                 f"debriefs and {seated['resets']} reset reasons pinned, {seated['notes']} replies read for notes")
              + f"; plays {session.fork_budget:.0f} generated tokens"
              + (", stopping at the level" if stop_at_level() else ""), flush=True)
        return True
    except Exception as exc:
        print(f"fork restore: {stem} failed: {exc!r}; not played", flush=True)
        return False


def first_request_check(owner: Any, messages: list[dict[str, Any]]) -> None:
    held = owner.__dict__.pop("_fork_check", None)
    if held is None:
        return
    source, turn, recorded = held
    sent = [{"role": one.get("role"), "content": one.get("content"), "tool_calls": one.get("tool_calls")}
            for one in messages]
    want = [{"role": one.get("role"), "content": one.get("content"), "tool_calls": one.get("tool_calls")}
            for one in recorded]
    if sent == want:
        print(f"fork check: the first request after the restore from {source} turn {turn} equals the recorded one "
              f"({len(sent)} messages)", flush=True)
        return
    first = next((index for index, (a, b) in enumerate(zip(sent, want)) if a != b), min(len(sent), len(want)))
    print(f"fork check: the first request after the restore from {source} turn {turn} differs from the recorded one "
          f"at message {first} of {len(sent)} (recorded {len(want)})", flush=True)


def before_turn(session: Any, analysis_step: int, retry: Any) -> None:
    if not restoring():
        return
    held = getattr(session, "fork_source", None)
    if held is not None and not getattr(session, "fork_checked", False):
        session.fork_checked = True
        source, turn, at = held
        rows = jsonl(source_path(source, REQUESTS_SUFFIX))
        session.analyzer._fork_check = (source, turn, rows[at]["messages"])
    held = getattr(session, "fork_baselines", None)
    if held is not None:
        session.token_baseline, session.game_token_baseline = held
        session.fork_baselines = None
    if warm_wait() > 0:
        session.fork_turns = getattr(session, "fork_turns", 0) + 1
        if session.fork_turns >= 2:
            mark_warm(session, "finished its first turn")
        wait_warm(session)
    if judge() is not None:
        judge().before_turn(session, retry)
    if ducks() is not None:
        ducks().before_turn(session, retry)


def over(session: Any) -> bool:
    if not restoring():
        return False
    budget = getattr(session, "fork_budget", None)
    if budget is None:
        return False
    walls_curve()
    arms_board()
    if broke(session) and stop_at_level():
        return True
    if generated(session.analyzer) - session.fork_base >= budget:
        session.fork_budget_hit = True
        return True
    if judge() is not None and judge().over(session):
        session.fork_budget_hit = True
        return True
    return False


def stop_at_level() -> bool:
    return os.environ.get(STOP_AT_LEVEL_KEY, "").strip().lower() in ("1", "true", "yes", "on")


def burn_arm(session: Any) -> str:
    passes = {int(part) for part in os.environ.get("ARC3_BURN_PASSES", "").split(",") if part.strip().isdigit()}
    also = {part.strip() for part in os.environ.get("ARC3_BURN_ALSO", "").split(",") if part.strip()}
    return "burn" if int(session.pass_index) in passes or stem_of(session) in also else "cut"


def wall_of(session: Any) -> str:
    return f"{game_id(session).split('-')[0]} L{int(session.fork_levels0) + 1}"


def arms_join(session: Any, item: dict[str, Any]) -> None:
    with _PLAN_LOCK:
        if _ARMS["start"] is None:
            _ARMS["start"] = _ARMS["said"] = time.monotonic()
        _ARMS["ducks"][stem_of(session)] = {"label": str(item.get("duck_label") or arm()),
                                            "group": str(item.get("duck_group") or stem_of(session)),
                                            "session": session, "won": None, "ended": None}


def _middle(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def arms_board(force: bool = False) -> None:
    with _PLAN_LOCK:
        if _ARMS["start"] is None or (not force and time.monotonic() - _ARMS["said"] < ARMS_EVERY_S):
            return
        _ARMS["said"] = time.monotonic()
        ducks_now = list(_ARMS["ducks"].values())
    minutes = (time.monotonic() - _ARMS["start"]) / 60.0
    arms: dict[str, list[dict[str, Any]]] = {}
    for one in ducks_now:
        arms.setdefault(one["label"], []).append(one)
    base = {one["group"]: one for one in arms.get("A", [])}
    lines = []
    for label in sorted(arms, key=lambda name: (name not in ("A", "A2"), name)):
        rows = arms[label]
        won = [one["won"] for one in rows if one["won"]]
        ended = sum(one["ended"] is not None for one in rows)
        spent = sum(max(0, generated(one["session"].analyzer) - int(one["session"].fork_base)) for one in rows)
        line = (f"{label} won {len(won)}/{len(rows)} (ended {ended}), {len(won) / max(1, spent) * 1e5:.2f} wins per "
                f"100K tokens of {spent // 1000}K")
        if won:
            line += (f", to the win median {_middle([one['tokens'] for one in won]) / 1000:.1f}K tokens "
                     f"{_middle([one['actions'] for one in won]):.0f} actions {_middle([one['minutes'] for one in won]):.1f} min")
        if label != "A" and base:
            pairs = [(one, base[one["group"]]) for one in rows if one["group"] in base
                     and (one["won"] or one["ended"]) and (base[one["group"]]["won"] or base[one["group"]]["ended"])]
            ahead = sum(bool(mine["won"]) and not theirs["won"] for mine, theirs in pairs)
            behind = sum(bool(theirs["won"]) and not mine["won"] for mine, theirs in pairs)
            line += f"; against A on {len(pairs)} settled points +{ahead} -{behind}"
        lines.append(line)
    print(f"fork arms [{minutes:.0f} min]:\n  " + "\n  ".join(lines), flush=True)


def walls_join(session: Any) -> None:
    with _PLAN_LOCK:
        if _WALLS["start"] is None:
            _WALLS["start"] = time.monotonic()
            _WALLS["said"] = time.monotonic()
        _WALLS["copies"].setdefault(burn_arm(session), {})[stem_of(session)] = wall_of(session)


def walls_curve(force: bool = False) -> None:
    with _PLAN_LOCK:
        if _WALLS["start"] is None or (not force and time.monotonic() - _WALLS["said"] < WALL_EVERY_S):
            return
        _WALLS["said"] = time.monotonic()
        minutes = (time.monotonic() - _WALLS["start"]) / 60.0
        parts = []
        for side in sorted(_WALLS["copies"]):
            copies = _WALLS["copies"][side]
            broken = _WALLS["broke"].get(side, {})
            walls = set(copies.values())
            walls_broken = {copies[stem] for stem in broken}
            tokens = sorted(one["tokens"] for one in broken.values())
            middle = tokens[len(tokens) // 2] if tokens else 0
            parts.append(f"{side} {len(broken)}/{len(copies)} copies, {len(walls_broken)}/{len(walls)} walls, "
                         f"median {middle} tokens to the break [{', '.join(sorted(walls_broken)) or '-'}]")
        print(f"fork walls [{minutes:.0f} min]: " + " | ".join(parts), flush=True)


def broke(session: Any) -> bool:
    if getattr(session, "fork_broke", None) is not None:
        return True
    levels0 = getattr(session, "fork_levels0", None)
    if levels0 is None or int(session.game.game_run.levels_completed) <= levels0:
        return False
    session.fork_broke = {"tokens": generated(session.analyzer) - session.fork_base,
                          "minutes": round((time.monotonic() - session.fork_clock) / 60.0, 2),
                          "actions": len(session.game.game_run.history) - session.fork_actions0}
    with _PLAN_LOCK:
        _WALLS["broke"].setdefault(burn_arm(session), {})[stem_of(session)] = session.fork_broke
        if stem_of(session) in _ARMS["ducks"]:
            _ARMS["ducks"][stem_of(session)]["won"] = dict(session.fork_broke)
    print(f"fork wall: {stem_of(session)} ({burn_arm(session)}) won {wall_of(session)} after "
          f"{session.fork_broke['tokens']} generated tokens, {session.fork_broke['actions']} actions, "
          f"{session.fork_broke['minutes']:.1f} min", flush=True)
    return True


def reason(session: Any) -> str:
    run = session.game.game_run
    if getattr(session, "fork_broke", None) is not None and stop_at_level():
        return "wall_broken"
    if getattr(session, "fork_budget_hit", False):
        return "token_budget"
    if run.state in ("won", "crashed"):
        return run.state
    if session.stop_event.is_set():
        return "cancelled"
    if session.runtime_limit_reached():
        return "time_limit"
    if session.token_limit_reached():
        return "token_limit"
    soft = session.solver.soft_time_remaining_seconds()
    if soft is not None and soft <= 0:
        return "soft_end"
    return run.state


def note(session: Any, label: str, why: str, since_epoch: float, since_generated: int) -> None:
    gid = game_id(session)
    stem = stem_of(session)
    try:
        source, turn, _ = getattr(session, "fork_source", (None, None, None))
        row = {"game_id": gid, "pass_index": int(session.pass_index), "stem": stem, "label": label, "reason": why,
               "source": source, "turn": turn, "end_epoch": time.time(), "played_s": time.time() - since_epoch,
               "post_fork_generated": generated(session.analyzer) - int(since_generated),
               "note": session.game.game_run.solver_note, "wall": getattr(session, "fork_broke", None),
               "levels_at_fork": getattr(session, "fork_levels0", None), **progress(session)}
        record_path(stem, label).write_text(json.dumps(row), encoding="utf-8")
        print(f"fork {label}: {stem} {why}, levels {row['levels']}, actions {row['actions']}, "
              f"{row['post_fork_generated']} generated after the fork in {row['played_s'] / 60:.1f} min", flush=True)
    except Exception as exc:
        print(f"fork {label}: {stem} NOT recorded: {exc!r}", flush=True)


def ended(session: Any) -> None:
    if not restoring():
        return
    mark_warm(session, "ended")
    if not getattr(session, "fork_restored", False):
        return
    if judge() is not None:
        judge().ended(session, folder(), {"levels": session.fork_levels0, "actions_at_fork": session.fork_actions0,
                                          "generated_after": generated(session.analyzer) - session.fork_base})
    if ducks() is not None:
        ducks().ended(session)
    note(session, arm(), reason(session), time.time() - (time.monotonic() - session.fork_clock), session.fork_base)
    with _PLAN_LOCK:
        if stem_of(session) in _ARMS["ducks"]:
            _ARMS["ducks"][stem_of(session)]["ended"] = reason(session)
        left = sum(one["ended"] is None for one in _ARMS["ducks"].values())
    arms_board(force=left == 0)
