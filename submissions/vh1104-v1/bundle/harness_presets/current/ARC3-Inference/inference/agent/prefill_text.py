import os
import re
from collections import defaultdict

PRINT_ENV = "ARC3_COMPACT_PRINT"
TURN_ENV = "ARC3_COMPACT_TURN"
SAME_BOARD_ENV = "ARC3_SAME_BOARD_TEXT"

ECHO = re.compile(r"^\[action\] (.+?) -> (.+)$")
BATCH = re.compile(r"^(\[action\] batch of \d+ actions?, \d+ executed: )(.+?)((?: \| stop_reason: .*)?)$")
HEX = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{16}(?![0-9A-Za-z])")
PAIR = r"[\[\(]\s*(-?\d+),\s*(-?\d+)\s*[\]\)]"
PAIR_LIST = re.compile(r"[\[\(]\s*(?:" + PAIR + r"\s*,\s*){3,}" + PAIR + r"\s*,?\s*[\]\)]")
GRID = re.compile(r"^(\s*-?\d+:?\s+)?([A-Za-z.#]{12,})\s*$")
RUN = re.compile(r"(.)\1{3,}")
COMMA = re.compile(r",\s(?=[\[\(\'\"\d-])")

PRINT_LEGEND = ("- Printed output reaches you compacted by the harness, nothing dropped: a sorted list of four or "
                "more [<a>, <b>] pairs reads cells{<a>:<b> ...}, where <x>-<y> is a range and consecutive <a> "
                "sharing the same <b> merge; a printed row of letters equal to the row above reads same; a run of one "
                "letter reads <letter>×<count>; consecutive action echoes with one verdict read "
                "[action] <name>×<count> ... -> <verdict> (each); a 16-hex hash reads #<n>, the same alias for the "
                "same hash all game.")

_SAID: set = set()
_ARMS: list = []


def env(name: str, default: str = "") -> str:
    if not _ARMS:
        try:
            from inference.agent import knob_arms
        except ImportError:
            knob_arms = None
        _ARMS.append(knob_arms)
    arms = _ARMS[0]
    return str(arms.env(name, default) if arms is not None else os.environ.get(name, default))


def flag(name: str) -> bool:
    return env(name, "0").strip().lower() in ("1", "true", "yes", "on")


def say_once(key: str, text: str) -> None:
    if key in _SAID:
        return
    _SAID.add(key)
    print(text, flush=True)


def runs(names: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(names):
        j = i
        while j + 1 < len(names) and names[j + 1] == names[i]:
            j += 1
        out.append(names[i] + (f"×{j - i + 1}" if j > i else ""))
        i = j + 1
    return out


def collapse_echoes(lines: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(lines):
        m = ECHO.match(lines[i])
        if not m or BATCH.match(lines[i]):
            batch = BATCH.match(lines[i])
            if batch:
                names = [one.strip() for one in batch.group(2).split(", ")]
                out.append(batch.group(1) + ", ".join(runs(names)) + batch.group(3))
            else:
                out.append(lines[i])
            i += 1
            continue
        verdict, names = m.group(2), []
        while i < len(lines):
            k = ECHO.match(lines[i])
            if not k or BATCH.match(lines[i]) or k.group(2) != verdict:
                break
            names.append(k.group(1))
            i += 1
        if len(names) == 1:
            out.append(f"[action] {names[0]} -> {verdict}")
        else:
            out.append(f"[action] {' '.join(runs(names))} -> {verdict} (each)")
    return out


def alias_hashes(text: str, aliases: dict) -> str:
    def name(m):
        if m.group(0) not in aliases:
            aliases[m.group(0)] = f"#{len(aliases) + 1}"
        return aliases[m.group(0)]
    return HEX.sub(name, text)


def span(values: list[int]) -> str:
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(f"{values[i]}-{values[j]}" if j > i else f"{values[i]}")
        i = j + 1
    return ",".join(out)


def cells(text: str) -> str:
    def repl(m):
        pairs = [(int(a), int(b)) for a, b in re.findall(PAIR, m.group(0))]
        if any(pairs[k] >= pairs[k + 1] for k in range(len(pairs) - 1)):
            return m.group(0)
        rows = defaultdict(list)
        for r, c in pairs:
            rows[r].append(c)
        merged = []
        for r in sorted(rows):
            cols = span(rows[r])
            if merged and merged[-1][1] == cols and r == merged[-1][2] + 1:
                merged[-1][2] = r
            else:
                merged.append([r, cols, r])
        return "cells{" + " ".join((f"{a}-{b}" if b > a else f"{a}") + ":" + c for a, c, b in merged) + "}"
    return PAIR_LIST.sub(repl, text)


def grids(lines: list[str]) -> list[str]:
    out, prev = [], None
    for line in lines:
        m = GRID.match(line)
        if not m:
            out.append(line)
            prev = None
            continue
        label, body = m.group(1) or "", m.group(2)
        out.append(label + ("same" if body == prev else RUN.sub(lambda x: f"{x.group(1)}×{len(x.group(0))}", body)))
        prev = body
    return out


def compact_print(text: str, aliases: dict | None = None) -> str:
    if not text:
        return text
    lines = collapse_echoes(text.split("\n"))
    lines = grids(lines)
    out = "\n".join(lines)
    out = alias_hashes(out, aliases if aliases is not None else {})
    out = cells(out)
    return COMMA.sub(",", out)


COUNT = re.compile(r"^The code executed \d+ actions? in the previous sequence\.$")
EXECUTED = re.compile(r"^Executed actions(?: \(first 10(?: of \d+)?\))?: (.*)\.$")
SAME_LEVEL = "You are still on the same level."
GAME_OVER_HEAD = "GAME OVER occurred during the previous sequence."
GAME_OVER_LINE = ("GAME OVER: the attempt failed; the level restarted from its start and finished levels are kept. "
                  "Diagnose it with the GAME OVER rules of the system message before acting again.")
RESUBMIT_HEAD = "Do NOT re-submit this same series of moves"
FATAL = re.compile(r"^The game over occurred immediately after action (.+?)\. Apply the BAR RULE above")
DELTA_OLD = "Board change since your previous turn - a heuristic read by the harness; check it in python."
DELTA_NEW = "Board change since your previous turn:"
RETAINED_TAIL = (" These are your previous definitions, available to call directly. Their presence does not mean "
                 "they are correct; revise them if new evidence contradicts their assumptions.")
TRACE = re.compile(r"^(Per-action trace(?: of the previous sequence)?: )(.+?)\.?$")
TRACE_ITEM = re.compile(r"^\d+: (.+?) -> (.+)$")
ANIMATION = re.compile(r"^The last executed action \(`(.+?)`\) animated over (\d+) frames?(?:\.| and left the board area "
                       r"unchanged, but it was not a no-op:) (\d+) cells? changed and changed back around rows "
                       r"(\S+), cols (\S+)\.(?: `last_animation_timeline` shows them\.| Whatever this action did is "
                       r"visible only there - read `last_animation_timeline` before concluding it failed\.)"
                       r"(?: It lasted \d+ engine frames?, about ([\d.]+) s at the game's \d+ fps\.)?$")
DELTA_IMAGE_LONG = ("The picture below draws, as in the sprite catalog, the objects that look different after this "
                    "turn's actions: changed or turned ones before and after, new types, and types that are gone.")
DELTA_IMAGE_SHORT = "Objects that look different:"
LEVEL_SHORT = "You completed the previous level; `current_frame` and the image show the next level's starting board."

TURN_LEGEND = (
    "- Turn messages are kept short. The actions your code ran are echoed in its own printed output. Valid actions are "
    "restated only when they change, retained functions only when they are new or changed, and the board image only "
    "when the board differs from the last image you were shown.\n"
    "- A picture captioned Objects that look different: draws, as in the sprite catalog, the objects that look "
    "different after the turn's actions: changed or turned ones before and after, new types, and types that are "
    "gone.\n"
    "- The board change, board digest and object pictures are heuristic reads by the harness, not ground truth: check "
    "them in python. Retained functions are your earlier definitions, callable directly; their presence does not mean "
    "they are correct, so revise them when new evidence contradicts their assumptions.\n"
    "- GAME OVER rules. A game over means the attempt FAILED (not that the run is finished); the game was "
    "automatically RESET: the current level restarted from its initial state and previously completed levels are kept. "
    "It is caused either by the specific action taken (e.g., moving onto a hazard cell) or by a step/time budget "
    "running out, usually shown by a bar at the grid border that shrinks with each action. Before acting again, "
    "diagnose the cause by inspecting the board right before the death: `history[-1].action` is 'RESET', "
    "`history[-2].action` is the fatal action, `history[-2].frame` is the death frame (the board after the fatal "
    "action), and `history[-3].frame` is the board immediately before the fatal action - check what cell the fatal "
    "move targeted there, and inspect the budget bar in the death frame. BAR RULE: the bar shrinking with each action "
    "is NORMAL - a slightly shorter bar is NOT evidence of the cause of death. The budget caused the death ONLY if the "
    "bar is fully (or almost fully) depleted in the death frame. If it IS depleted: reach the goal in fewer actions, or "
    "look for a way to restore the bar (e.g., collecting an item that refills it). If the bar clearly had budget "
    "remaining: the fatal action itself caused the death (hazard or forbidden move) - do NOT retry the same path; try a "
    "DIFFERENT approach. Do NOT re-submit the series of moves that died: the game is deterministic, so replaying it "
    "will die again. Change the plan BEFORE the fatal point - a different route, action, or timing.")


def trace_runs(body: str) -> str | None:
    items = []
    for part in body.split("; "):
        m = TRACE_ITEM.match(part.strip().rstrip("."))
        if not m:
            return None
        items.append((m.group(1), m.group(2)))
    groups = []
    for name, verdict in items:
        if groups and groups[-1][1] == verdict:
            groups[-1][0].append(name)
        else:
            groups.append(([name], verdict))
    return "; ".join(" ".join(runs(names)) + f" -> {verdict}" + (" (each)" if len(names) > 1 else "")
                     for names, verdict in groups)


def animation_line(m: re.Match) -> str:
    action, frames, cells, rows, cols, seconds = m.groups()
    unchanged = "left the board area unchanged" in m.group(0)
    head = f"Animation of `{action}`: {frames} frames"
    head += f" ({seconds} s)" if seconds else ""
    if unchanged:
        return (f"{head}; the board area ended unchanged but it was not a no-op: {cells} cells changed and back at "
                f"rows {rows}, cols {cols}; read `last_animation_timeline`.")
    return f"{head}; {cells} cells changed and back at rows {rows}, cols {cols} (`last_animation_timeline`)."


def compact_turn(text: str, echo_ok: bool, level_full: str = "") -> str:
    if level_full and level_full in text:
        text = text.replace(level_full, LEVEL_SHORT)
    text = text.replace(RETAINED_TAIL, "")
    out = []
    for line in text.split("\n"):
        if echo_ok and (COUNT.match(line) or (EXECUTED.match(line) and " on " not in line)):
            continue
        if line == SAME_LEVEL or line.startswith(RESUBMIT_HEAD):
            continue
        if line.startswith(GAME_OVER_HEAD):
            out.append(GAME_OVER_LINE)
            continue
        fatal = FATAL.match(line)
        if fatal:
            out.append(f"The fatal action was {fatal.group(1)}.")
            continue
        if line == DELTA_OLD:
            out.append(DELTA_NEW)
            continue
        trace = TRACE.match(line)
        if trace:
            short = trace_runs(trace.group(2))
            out.append(trace.group(1) + short + "." if short else line)
            continue
        animation = ANIMATION.match(line)
        if animation:
            out.append(animation_line(animation))
            continue
        out.append(line)
    return "\n".join(out)


def perception_legends() -> list[str]:
    try:
        from inference.agent import perception
    except ImportError:
        return []
    return (([perception.SCENE_LEGEND] if perception.flag(perception.SCENE_ENV) else [])
            + ([perception.SHORT_DELTA_LEGEND] if perception.flag(perception.SHORT_DELTA_ENV) else []))


def system_lines(compact_print_on: bool, compact_turn_on: bool) -> str:
    lines = ([PRINT_LEGEND] if compact_print_on else []) + ([TURN_LEGEND] if compact_turn_on else [])
    lines += perception_legends()
    return ("\n\nHow the harness shows things\n" + "\n".join(lines)) if lines else ""


RETENTION = re.compile(r"Retained your function (.+?) for later Python calls(?: (?:on this level|in this turn|in this game))?\.")
VALID_LINE = re.compile(r"^Valid actions right now: .*$", re.M)
GRID_CAPTION = "Current grid image:"
UNCHANGED_CAPTION = "Current grid image: unchanged since the last image above, not sent again."
CONTINUE = "Continue."


def compact_retention(text: str) -> str:
    sigs = RETENTION.findall(text)
    if not sigs:
        return text
    rest = " ".join(RETENTION.sub("", text).split())
    return "Kept for later calls: " + ", ".join(sigs) + "." + (" " + rest if rest else "")


def texts_of(message) -> list[str]:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return [content]
    return [part.get("text") or "" for part in content or () if isinstance(part, dict) and part.get("type") == "text"]


def first_image(message) -> str | None:
    content = message.get("content") if isinstance(message, dict) else None
    for part in content if isinstance(content, list) else ():
        if isinstance(part, dict) and part.get("type") == "image_url":
            return str((part.get("image_url") or {}).get("url") or "") or None
    return None


def after_trim(messages: list, opener: dict, turn: bool = True, same_board: bool = True) -> dict:
    at = next((k for k, message in enumerate(messages) if message is opener), None)
    if at is None:
        return {}
    earlier = [message for message in messages[:at] if isinstance(message, dict) and message.get("role") == "user"]
    content = opener.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
    fired: dict = {}
    for k, part in enumerate(parts if turn else ()):
        if isinstance(part, dict) and part.get("type") == "text" and RETAINED_TAIL in (part.get("text") or ""):
            parts[k] = {**part, "text": part["text"].replace(RETAINED_TAIL, "")}
            fired["retained tail"] = True
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text") == DELTA_IMAGE_LONG:
            parts[k] = {**part, "text": DELTA_IMAGE_SHORT}
            fired["picture caption"] = True
    shown = None
    for message in reversed(earlier if turn else ()):
        found = VALID_LINE.findall("\n".join(texts_of(message)))
        if found:
            shown = found[-1]
            break
    if shown is not None:
        for k, part in enumerate(parts):
            if not (isinstance(part, dict) and part.get("type") == "text"):
                continue
            lines = (part.get("text") or "").split("\n")
            kept = [line for line in lines if line != shown]
            if len(kept) != len(lines):
                parts[k] = {**part, "text": "\n".join(kept)}
                fired["valid actions"] = True
    url = first_image({"content": parts})
    before = next((message for message in reversed(earlier) if first_image(message)), None)
    if same_board and url and before is not None and first_image(before) == url:
        spot = next(k for k, part in enumerate(parts) if isinstance(part, dict) and part.get("type") == "image_url")
        caption = parts[spot - 1] if spot > 0 else None
        if isinstance(caption, dict) and caption.get("type") == "text" and (caption.get("text") or "").endswith(GRID_CAPTION):
            parts[spot - 1] = {**caption, "text": caption["text"][:-len(GRID_CAPTION)] + UNCHANGED_CAPTION}
            del parts[spot]
            fired["same image"] = True
    if fired:
        opener["content"] = parts
    return fired
