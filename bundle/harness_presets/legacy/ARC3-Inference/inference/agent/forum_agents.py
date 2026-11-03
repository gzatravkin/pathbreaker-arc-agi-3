import ast
import difflib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ACTIVE = None

SESSION_HEADER = "X-Session-ID"
RELEASE_MARK = "taaf_release"
UNREACHABLE_STOP = "␞␞forum-stop␞␞"
NAMES = ("Ada", "Basil", "Cora", "Dmitri", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kira", "Lev",
         "Mina", "Nils", "Olga", "Pavel", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wim", "Xenia", "Yusuf",
         "Zoe", "Anton", "Bella", "Cyril", "Dana", "Emil", "Fiona", "Gleb", "Hana", "Igor", "Jana", "Karl", "Lina",
         "Marco", "Nadia", "Oskar", "Petra", "Rafael", "Sofia", "Timur", "Ulla", "Viktor", "Wanda", "Yara", "Zeno",
         "Alina", "Boris", "Clara", "Denis", "Eva", "Felix", "Gaia", "Henrik", "Irina", "Jakob", "Katya", "Leon",
         "Maya", "Noah", "Olivia", "Pablo", "Ruth", "Simon", "Thea", "Uri", "Valya", "Walter", "Xavier", "Yan",
         "Zara", "Arne", "Bruno", "Celia", "Dario", "Edith", "Fedor", "Gemma", "Hector", "Ilya", "Juno", "Kai",
         "Lotte", "Milo", "Nora", "Otto")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,23}$")
SECTION_RE = re.compile(r"^\[([A-Z][A-Z _]*?)(?::\s*([^\]]*))?\]\s*$")
TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\S+) \|")
DROPPED_SECTIONS = {"SYSTEM PROMPT", "MODEL RESPONSE META", "ANALYZER STATUS"}
CLOUD_KINDS = {"THINKING", "TOOL CALL"}
WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
PARTS = {"all": None, "thinking": {"THINKING"}, "code": {"TOOL CALL"}, "results": {"TOOL RESULT"},
         "prompt": {"USER PROMPT"}, "system": {"SYSTEM PROMPT"}}

DOCS = """About the competition (ARC Prize 2026, ARC-AGI-3):
- Every game is an unfamiliar turn-based puzzle on a grid of at most 64x64 cells, 16 colours. Nobody is told the rules or the goal; a player discovers them by acting and watching.
- Each game has several levels. A later level usually reuses the mechanics of earlier ones and adds something new.
- Actions: UP, DOWN, LEFT, RIGHT, INTERACT, CLICK (a cell), UNDO; RESET restarts the level. The same names mean the same thing in every game.
- Score of one level = min(1.15, (human actions / player actions)^2) if the level is won, else 0. Level k weighs k (the 5th level counts five times the 1st). Every action counts, RESET included. Fewer actions on a won level is worth a lot; a level not won is worth nothing.
- The scored set is hidden and its games differ from these. What transfers is a way of thinking, a debugging habit, a tool, a search method, not a game's solution.

About the players:
- Each game is played by one player: the same language model as you, alone, with one tool, Python, which sees the board as text and segmented objects and can call action([...]) to play. It also sees an image of the board each turn.
- A player cannot see the forum or other games. The only thing that reaches it is a note you send: the note is added, with your name, to the message that opens its next turn. It then decides on its own whether to follow it.
- A player's context is cut when it grows long; old turns disappear from its memory.

About you:
- You cannot act in any game. You can read every game (its turns, the model's thinking, its code and the code's output, any board), run Python on any game's boards, talk on the forum, and send notes.
- Reading is not free. Every character you read is processed (prefilled) on the same GPU the players use, so a helper that reads whole games slows every player down. Read narrowly: search first, read few turns, ask for one part (thinking, code or results) instead of all, look at one board instead of many.
- Your memory is limited to __LIMIT__ tokens of conversation. Large reads fill it fast.
- You have your own seat. When your conversation reaches about __HANDOFF__ tokens you will be asked to write a handoff with write_handoff. Then this instance of you ends, and a fresh instance of you, with your name and seat, continues from your handoff alone. Write in the handoff everything your successor needs: what you are doing and why, which games and threads you follow, what you have checked and found, what you promised others, what to do next.
"""

NOTE_HEADER = ("Notes from helper agents who are reading your game (they cannot act; check each against your own "
               "evidence before relying on it):")
SIMILAR_EARLY = ("Similarity rests on the thinking and code written so far; below about 15K tokens of thinking and code on either side "
                 "it shifts a lot as the games reveal more of their mechanics, so treat early matches as weak hints.")
SIGNED_LINE = ("Your posts, threads and notes are signed by the harness with your name, seat and arrival number. Do not "
               "write a signature or any helper's name as the author of your text; the harness adds the true one.")
SENIOR_LINE = ("Every helper has an arrival number (#1 is the first seat ever created; a successor instance keeps its "
               "seat's number). Respect the senior helpers, those with lower numbers, who have been on the forum longer: "
               "weigh their verified findings and coordination proposals first, while still checking the evidence yourself.")
THREAD_LINE = ("Start a new thread for each new topic (one game, one family of mechanics, one tool trick, one coordination "
               "question) instead of appending to an existing long thread, so other helpers read only the threads they "
               "need; give each thread a title that names its specific topic.")
SIMILAR_LINE = ("When show_game lists similar games, look at the mechanics of those games, especially the ones further "
                "along (more levels won), and consider whether what was learned there transfers, if there is any real "
                "overlap.")

SYSTEM = """You are one of many helper agents in a shared workspace during a live run of puzzle games. The players are other instances of the same model, each alone on its game. Your goal: as many levels as possible get won across all games, with as few actions as possible.

Nobody assigns you a role. Decide for yourself what is most useful, using the forum to coordinate with the other helpers. Bad or confident-but-wrong advice to a player costs it actions; a note that is checked against the evidence in its own game is worth much more than a guess.

__SIMILAR__

__SIGNED__

__SENIOR__

__THREADS__

Tools:
- show_game: one game at a glance: its current board as an image, every action so far, the last few actions with what each changed, the player's latest thinking, the levels it has won, and the other games most similar to it by the distinctive words in their players' thinking and code.
- list_games: every game with its levels, actions, tokens, time and notes.
- read_turns: a game's turns (the player's thinking, code, code output, the prompt it got).
- view_board: one board of a game, as text or as an image.
- search: a regular expression over every game's turns and the forum.
- run_python: Python on a game's boards with the player's own frame API; action() is disabled.
- forum_list, forum_read, forum_post, forum_new_thread: the forum.
- send_note: a short note to one player, delivered at the start of its next turn. A few player turns later you get news of what followed: whether its thinking mentioned the note, levels won, actions spent, and how often its actions did nothing before and after.
- send_function: a Python function (with a trial call) and a short message to one player: its code is shown to the player and the function is already defined in the player's Python. Counts as a note.
- claim, release: say that you look after a game, or that you no longer do; list_games and show_game show who holds each game.
- news: what happened since you last looked (also appended automatically after your tool calls).
- write_handoff: your handoff to the next instance of you in your seat.

""".replace("__SIMILAR__", SIMILAR_LINE).replace("__SIGNED__", SIGNED_LINE).replace("__SENIOR__", SENIOR_LINE).replace("__THREADS__", THREAD_LINE) + DOCS


def system_text(limit: int, handoff: int) -> str:
    return SYSTEM.replace("__LIMIT__", f"{limit:,}").replace("__HANDOFF__", f"{handoff:,}")


def _now() -> float:
    return time.time()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _cap(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[cut: {len(text) - limit} more characters; narrow the request]"


def _quote(text: str) -> str:
    text = str(text)
    if len(text) > 1500:
        text = text[:1500] + f" [... {len(text) - 1500} more chars in forum/events.jsonl]"
    return "\n".join("    | " + line for line in text.splitlines())


NAME_ALT = r"(?:%s)\d*" % "|".join(sorted(NAMES, key=len, reverse=True))
SIGN_LINE_RE = re.compile(r"^[\s\-—–~*_>]*(?:signed|sig|from|by|regards|cheers)?[\s:,\-]*(?:by\s+)?"
                          r"(%s)(?:'s)?(?:\s*[,(\[]?\s*(?:seat|helper|#)\s*#?\d+\s*[)\]]?)*(?:\s*,?\s*#\d+)?"
                          r"(?:\s+here)?[\s.!:;)*_\]]*$" % NAME_ALT, re.IGNORECASE)
SIGN_HEAD_RE = re.compile(r"^[\s*_>]*(?:(?:signed|from)\s*[:,\-]?\s*(%s)(?:\s*[,(]\s*seat\s*\d+\s*\)?)?(?:\s*,?\s*#\d+)?"
                          r"|(%s)\s*(?:,\s*seat\s*\d+|\(\s*seat\s*\d+\s*\))(?:\s*,?\s*#\d+)?(?:\s+here)?"
                          r"|(%s)\s+here(?:\s*\(\s*seat\s*\d+\s*\))?)[\s.:;,!\-—]*" % (NAME_ALT, NAME_ALT, NAME_ALT),
                          re.IGNORECASE)
TITLE_SIGN_RE = re.compile(r"\s*[(\[]\s*(?:by\s+)?(%s)(?:\s*,\s*seat\s*\d+)?(?:\s*,?\s*#\d+)?\s*[)\]]" % NAME_ALT,
                           re.IGNORECASE)


def strip_signature(text: str) -> tuple[str, list[str]]:
    lines = str(text).strip().splitlines()
    dropped: list[str] = []
    while lines and SIGN_LINE_RE.match(lines[-1]) and len(lines[-1]) <= 80:
        dropped.append(lines.pop().strip())
        while lines and not lines[-1].strip():
            lines.pop()
    while lines and SIGN_LINE_RE.match(lines[0]) and len(lines[0]) <= 80:
        dropped.append(lines.pop(0).strip())
        while lines and not lines[0].strip():
            lines.pop(0)
    if lines:
        head = SIGN_HEAD_RE.match(lines[0])
        if head and head.end() > 0 and lines[0][head.end():].strip():
            dropped.append(lines[0][:head.end()].strip())
            lines[0] = lines[0][head.end():]
    return "\n".join(lines).strip(), dropped


def strip_title(title: str) -> str:
    return TITLE_SIGN_RE.sub("", str(title)).strip() or str(title).strip()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def near(one: str, two: str) -> float:
    matcher = difflib.SequenceMatcher(None, one, two, autojunk=False)
    if matcher.real_quick_ratio() < 0.5 or matcher.quick_ratio() < 0.5:
        return 0.0
    return matcher.ratio()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _functions_kept() -> bool:
    return _env_flag("ARC3_PERSISTENT_FUNCTIONS")


def rename_function(source: str, old: str, new: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == old:
            node.name = new
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new
    return ast.unparse(tree)


def signature_of(source: str, name: str) -> str:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return f"{name}({ast.unparse(node.args)})"
    return f"{name}(...)"


def inject(analyzer: Any, name: str, source: str, author: str) -> tuple[str, str]:
    kept = getattr(analyzer, "_kept_functions", None)
    if not isinstance(kept, dict):
        kept = {}
        analyzer._kept_functions = kept
    final = name
    if final in kept and kept[final] != source:
        base = f"{name}_{re.sub(r'[^a-z0-9]', '', author.lower()) or 'helper'}"
        final, number = base, 2
        while final in kept:
            final, number = f"{base}{number}", number + 1
        source = rename_function(source, name, final)
    kept[final] = source
    return final, source


def _noops(moves: list[Any]) -> int:
    return sum(1 for entry in moves if (getattr(entry, "result", None) or {}).get("no_op"))


def words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text) if len(set(word.lower())) >= 3]


class _Turn(dict):
    __slots__ = ("game", "index", "start")

    def __missing__(self, key: str) -> Any:
        if key != "sections":
            raise KeyError(key)
        return self.game.sections_of(self)


def _sections(lines: list[str]) -> list[list[Any]]:
    sections: list[list[Any]] = []
    for line in lines:
        head = SECTION_RE.match(line)
        if head:
            sections.append([head.group(1).strip(), head.group(2) or "", []])
        elif sections:
            sections[-1][2].append(line)
    return sections


def _family(key: str) -> str:
    return key.rpartition("_p")[0] or key


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


def _tools(seat: str = "") -> list[dict[str, Any]]:
    game = {"type": "string", "description": "the game key exactly as list_games shows it"}
    return [
        _tool("show_game", (f"[{seat}] " if seat else "") + "One game at a glance: the current board as an image, every action so far (repeats "
              "compressed), the last 3 actions with what each changed, the player's latest thinking, and the levels "
              "won with their action counts, and the most similar other games.", {"game": game}, ["game"]),
        _tool("list_games", "Every game: levels won, actions, tokens, minutes, state, notes waiting and delivered.", {}, []),
        _tool("read_turns", "Turns of one game, oldest first. Without from_turn, the latest turns.",
              {"game": game, "from_turn": {"type": "integer"}, "count": {"type": "integer", "description": "1-6"},
               "part": {"type": "string", "enum": sorted(PARTS)}}, ["game"]),
        _tool("view_board", "One board of a game. step = the number of actions taken so far when the board was "
              "seen (0 = start); omit for the current board.",
              {"game": game, "step": {"type": "integer"}, "format": {"type": "string", "enum": ["text", "image", "both"]}},
              ["game"]),
        _tool("search", "Regular expression over every game's turns and the forum; up to 25 matching lines.",
              {"pattern": {"type": "string"}, "game": {"type": "string", "description": "optional, one game only"}},
              ["pattern"]),
        _tool("run_python", "Run Python on one game's boards. Globals as the player has them: current_frame, "
              "previous_frame, history, transitions, last_transition, frame_diff(before, after); every frame has "
              ".ascii, .segmentation, .step, .level, .shape. action() is disabled. print() what you need.",
              {"game": game, "code": {"type": "string"}}, ["game", "code"]),
        _tool("forum_list", "Every forum thread: id, title, author, posts, last activity.", {}, []),
        _tool("forum_read", "Posts of one thread, from post number from_post (default 0).",
              {"thread": {"type": "integer"}, "from_post": {"type": "integer"}}, ["thread"]),
        _tool("forum_new_thread", "Open a thread with a first post.",
              {"title": {"type": "string"}, "text": {"type": "string"}}, ["title", "text"]),
        _tool("forum_post", "Post in a thread.", {"thread": {"type": "integer"}, "text": {"type": "string"}},
              ["thread", "text"]),
        _tool("send_note", "A note to one player, added to the message that opens its next turn. Short and checkable.",
              {"game": game, "text": {"type": "string"}}, ["game", "text"]),
        _tool("claim", "Say that you look after one game; others see it in list_games and show_game. A game "
              "held by another helper cannot be claimed until it is released.", {"game": game}, ["game"]),
        _tool("release", "Stop looking after a game you claimed.", {"game": game}, ["game"]),
        _tool("send_function", "A Python function plus a short message to one player, delivered like a note. The "
              "code defines exactly one top-level function, name, followed by a trial: a line calling it on this "
              "game's boards. The trial runs here first; nothing is sent if the code does not compile or the trial "
              "raises. The player sees the message and the code, and finds the function already defined in its "
              "Python, kept between calls; it decides whether to call it.",
              {"game": game, "name": {"type": "string"}, "code": {"type": "string"},
               "message": {"type": "string"}}, ["game", "name", "code", "message"]),
        _tool("news", "Everything that happened since you last looked.", {}, []),
        _tool("write_handoff", "Your handoff to the next instance of you in your seat: everything it needs to "
              "continue your work. At most 8000 characters. This instance ends after it.",
              {"text": {"type": "string"}}, ["text"]),
    ]


class _Game:

    def __init__(self, session: Any, key: str) -> None:
        self.session = session
        self.key = key
        self.levels_seen = 0
        self.level_tokens = 0
        self.level_actions = 0
        self.level_time = session.started_at
        self.ended = False
        self.notes_waiting: list[dict[str, Any]] = []
        self.notes_delivered = 0
        self.turns: list[_Turn] = []
        self.parsed_bytes = 0
        self.partial = b""
        self.section_kind: str | None = None
        self.parse_lock = threading.Lock()
        self.cloud: Counter[str] = Counter()
        self.cloud_chars = 0
        self.freed = False
        self.calls = 0
        self.pending: list[dict[str, Any]] = []

    def moves(self) -> list[Any]:
        return [entry for entry in list(getattr(self.session, "history_entries", []) or []) if getattr(entry, "action", "")]

    def run(self) -> Any:
        return self.session.game.game_run

    def tokens(self) -> int:
        value = getattr(self.session.analyzer, "generated_tokens", 0)
        try:
            return int(value() if callable(value) else value)
        except Exception:
            return 0

    def status(self) -> dict[str, Any]:
        run = self.run()
        return {
            "game": self.key,
            "levels": int(getattr(run, "levels_completed", 0) or 0) if run is not None else 0,
            "total": int(getattr(run, "number_of_levels", 0) or 0) if run is not None else 0,
            "actions": int(self.session.action_count),
            "per_level": list(getattr(run, "actions_per_level", []) or []) if run is not None else [],
            "tokens": self.tokens(),
            "minutes": (time.monotonic() - self.session.started_at) / 60.0,
            "state": str(getattr(run, "state", "?")) if run is not None else "?",
        }

    def frames(self) -> tuple[Any, list[Any]]:
        history = list(self.session.history_entries)
        return self.session.current_frame(), history

    def parse(self) -> list[dict[str, Any]]:
        path = Path(self.session.transcript_path)
        with self.parse_lock:
            try:
                size = path.stat().st_size
            except OSError:
                return self.turns
            if size <= self.parsed_bytes:
                return self.turns
            with open(path, "rb") as handle:
                handle.seek(self.parsed_bytes)
                chunk = handle.read(size - self.parsed_bytes)
            offset = self.parsed_bytes - len(self.partial)
            self.parsed_bytes = size
            raw_lines = (self.partial + chunk).split(b"\n")
            self.partial = raw_lines.pop()
            for raw in raw_lines:
                start = offset
                offset += len(raw) + 1
                line = raw.decode("utf-8", errors="replace")
                found = TURN_RE.match(line)
                if found:
                    turn = _Turn(turn=int(found.group(1)), action=int(found.group(2)), clock=found.group(3))
                    turn.game, turn.index, turn.start = self, len(self.turns), start
                    self.turns.append(turn)
                    self.section_kind = None
                    continue
                if not self.turns:
                    continue
                head = SECTION_RE.match(line)
                if head:
                    self.section_kind = head.group(1).strip()
                elif self.section_kind is not None and self.section_kind in CLOUD_KINDS:
                    self.cloud.update(words(line))
                    self.cloud_chars += len(line) + 1
            return self.turns

    def _bounds(self, turns: list[_Turn]) -> list[tuple[_Turn, int, int]]:
        with self.parse_lock:
            complete = self.parsed_bytes - len(self.partial)
            out = []
            for turn in turns:
                after = self.turns[turn.index + 1].start if turn.index + 1 < len(self.turns) else complete
                out.append((turn, turn.start, after))
            return out

    def sections_of(self, turn: _Turn) -> list[list[Any]]:
        (_, start, end), = self._bounds([turn])
        with open(Path(self.session.transcript_path), "rb") as handle:
            handle.seek(start)
            data = handle.read(end - start)
        return _sections(data.decode("utf-8", errors="replace").split("\n")[1:-1])

    def turn_sections(self) -> Any:
        bounds = self._bounds(list(self.parse()))
        if not bounds:
            return
        with open(Path(self.session.transcript_path), "rb") as handle:
            for turn, start, end in bounds:
                handle.seek(start)
                data = handle.read(end - start)
                yield turn, _sections(data.decode("utf-8", errors="replace").split("\n")[1:-1])

    def cloud_copy(self) -> dict[str, int]:
        self.parse()
        with self.parse_lock:
            return dict(self.cloud)

    def cloud_tokens(self) -> int:
        with self.parse_lock:
            return int(self.cloud_chars / 3.6)


class Similar:

    def __init__(self, refresh_s: float, max_df: float) -> None:
        self.refresh_s = refresh_s
        self.max_df = max_df
        self.lock = threading.Lock()
        self.built = 0.0
        self.seconds = 0.0
        self.vectors: dict[str, dict[str, float]] = {}

    def refresh(self, games: list[_Game], drop: set[str], force: bool = False) -> None:
        with self.lock:
            if not force and self.built and _now() - self.built < self.refresh_s:
                return
            began = time.perf_counter()
            clouds = {game.key: game.cloud_copy() for game in games}
            df: Counter[str] = Counter()
            for cloud in clouds.values():
                df.update(cloud.keys())
            count = len(clouds)
            limit = max(2.0, self.max_df * count)
            idf = {word: math.log((1 + count) / seen) for word, seen in df.items()
                   if 2 <= seen <= limit and word not in drop}
            vectors = {}
            for key, cloud in clouds.items():
                vector = {word: (1.0 + math.log(tf)) * idf[word] for word, tf in cloud.items() if word in idf}
                norm = math.sqrt(sum(value * value for value in vector.values()))
                vectors[key] = {word: value / norm for word, value in vector.items()} if norm else {}
            self.vectors = vectors
            self.built = _now()
            self.seconds = time.perf_counter() - began

    def similar(self, key: str, top: int, shown: int) -> list[tuple[float, str, list[str]]]:
        vectors = self.vectors
        base = vectors.get(key) or {}
        out = []
        for other, vector in vectors.items():
            if other == key or _family(other) == _family(key) or not vector or not base:
                continue
            small, big = (base, vector) if len(base) <= len(vector) else (vector, base)
            shared = sorted(((base[word] * vector[word], word) for word in small if word in big), reverse=True)
            score = sum(value for value, _word in shared)
            if score > 0:
                out.append((score, other, [word for _value, word in shared[:shown]]))
        return sorted(out, reverse=True)[:top]


class Forum:

    def __init__(self, root: Path, base_url: str, model: str, helpers: int, end_at: float) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "helpers").mkdir(exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.count = helpers
        self.started = _now()
        self.end_at = end_at
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.games: dict[str, _Game] = {}
        self.members: list[dict[str, Any]] = []
        self.threads: list[dict[str, Any]] = []
        self.news: list[dict[str, Any]] = []
        self.stats = {"requests": 0, "errors": 0, "prompt": 0, "cached": 0, "generated": 0, "posts": 0, "threads": 0,
                      "notes": 0, "functions": 0, "delivered": 0, "reads": 0, "read_chars": 0, "python": 0, "images": 0}
        self.python_slots = threading.Semaphore(_env_int("ARC3_FORUM_PYTHON_SLOTS", 6))
        self.context_tokens = _env_int("ARC3_FORUM_CONTEXT_TOKENS", 100000)
        self.handoff_at = _env_int("ARC3_FORUM_HANDOFF_AT", 90000)
        self.max_output = _env_int("ARC3_FORUM_MAX_OUTPUT", 4096)
        self.read_chars = _env_int("ARC3_FORUM_READ_CHARS", 12000)
        self.arrival_s = _env_float("ARC3_FORUM_ARRIVAL_S", 4.0)
        self.upscale = _env_int("ARC3_FORUM_IMAGE_UPSCALE", 7)
        self.api_key = os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        self.sessions = os.environ.get("ARC3_FORUM_SESSIONS", "1").strip().lower() not in ("", "0", "false", "off")
        self.grow = os.environ.get("ARC3_FORUM_GROW_ON_END", "0").strip().lower() not in ("", "0", "false", "off")
        self.max_helpers = _env_int("ARC3_FORUM_MAX_HELPERS", 0)
        self.grown = 0
        self.claims: dict[str, int] = {}
        self.outcome_turns = _env_int("ARC3_FORUM_OUTCOME_TURNS", 3)
        self.template_games = _env_int("ARC3_FORUM_TEMPLATE_GAMES", 3)
        self.function_chars = _env_int("ARC3_FORUM_FUNCTION_CHARS", 4000)
        self.template_similarity = _env_float("ARC3_FORUM_TEMPLATE_SIMILARITY", 0.8)
        self.similar_top = _env_int("ARC3_FORUM_SIMILAR_TOP", 5)
        self.similar_words = _env_int("ARC3_FORUM_SIMILAR_WORDS", 8)
        self.similar = Similar(_env_float("ARC3_FORUM_SIMILAR_REFRESH_S", 20.0),
                               _env_float("ARC3_FORUM_SIMILAR_MAX_DF", 0.5))
        self.events_path = self.root / "events.jsonl"
        self.threads_out: list[threading.Thread] = []

    def minute(self) -> str:
        done = (_now() - self.started) / 60.0
        left = max(0.0, (self.end_at - _now()) / 60.0)
        return f"minute {done:.0f} since the forum opened, {left:.0f} minutes of play left in this run"

    def label(self, member_id: int) -> str:
        with self.lock:
            member = self.members[member_id] if 0 <= member_id < len(self.members) else None
        if member is None:
            return f"seat {member_id}"
        return f"{member['name']} (seat {member['id']}, #{member['order']})"

    def holder(self, key: str) -> str:
        with self.lock:
            held = self.claims.get(key)
        return self.label(held) if held is not None else ""

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"t": round(_now() - self.started, 1), "kind": kind, **fields}
        with self.lock:
            with open(self.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def say(self, text: str) -> None:
        print(f"forum [{(_now() - self.started) / 60:.1f} min]: {text}", flush=True)

    def add_news(self, text: str, audience: set[int] | None = None) -> None:
        with self.lock:
            self.news.append({"seq": len(self.news), "t": _now(), "text": text, "audience": audience})

    def add_game(self, session: Any, key: str) -> None:
        game = _Game(session, key)
        with self.lock:
            self.games[game.key] = game
        self.record("game", game=game.key)

    def game(self, key: Any) -> _Game | None:
        key = str(key or "").strip()
        with self.lock:
            if key in self.games:
                return self.games[key]
            matches = [game for name, game in self.games.items() if name.startswith(key)]
        return matches[0] if len(matches) == 1 else None

    def refresh_similar(self, force: bool = False) -> None:
        with self.lock:
            games = list(self.games.values())
            drop = ({name.lower() for name in NAMES} | {one["name"].lower() for one in self.members}
                    | set(words(NOTE_HEADER + " " + PLAYER_LINE)))
        self.similar.refresh(games, drop, force)

    def watch(self) -> None:
        last_line = 0.0
        similar_failed = False
        while not self.stop_event.wait(5.0):
            if self.similar_top > 0:
                try:
                    self.refresh_similar()
                except Exception as exc:
                    if not similar_failed:
                        similar_failed = True
                        self.say(f"similar games: refresh failed, {type(exc).__name__}: {exc}")
            with self.lock:
                games = list(self.games.values())
            for game in games:
                try:
                    status = game.status()
                except Exception:
                    continue
                if status["levels"] > game.levels_seen:
                    tokens = status["tokens"] - game.level_tokens
                    actions = status["actions"] - game.level_actions
                    minutes = (time.monotonic() - game.level_time) / 60.0
                    text = (f"{game.key} won level {status['levels']} of {status['total']}: {actions} actions and "
                            f"{tokens / 1000:.1f}k tokens on that level, {minutes:.1f} min")
                    self.add_news(text)
                    self.record("level", game=game.key, level=status["levels"], actions=actions, tokens=tokens)
                    self.say(text)
                    game.levels_seen = status["levels"]
                    game.level_tokens = status["tokens"]
                    game.level_actions = status["actions"]
                    game.level_time = time.monotonic()
                if not game.ended and status["state"] not in ("playing", "?"):
                    self.mark_ended(game, status)
            if _now() - last_line >= 60.0:
                last_line = _now()
                with self.lock:
                    alive = sum(1 for one in self.members if one.get("alive"))
                    stats = dict(self.stats)
                self.say(f"{alive} helpers alive ({self.count} at start + {self.grown} on freed lanes), "
                         f"similar-games index {len(self.similar.vectors)} games in {self.similar.seconds:.2f}s, "
                         f"{len(self.threads)} threads, {stats['posts']} posts, {stats['notes']} "
                         f"notes ({stats['delivered']} delivered); {stats['requests']} requests, {stats['errors']} "
                         f"errors, prompt {stats['prompt'] / 1e6:.2f}M ({stats['cached'] / 1e6:.2f}M cached), "
                         f"generated {stats['generated'] / 1e6:.2f}M; reads {stats['reads']} "
                         f"({stats['read_chars'] / 1e6:.2f}M chars), python {stats['python']}, images {stats['images']}")
                self.record("stats", alive=alive, grown=self.grown, similar_s=round(self.similar.seconds, 3), **stats)

    def mark_ended(self, game: _Game, status: dict[str, Any]) -> None:
        with self.lock:
            if game.ended:
                return
            game.ended = True
        text = f"{game.key} ended ({status['state']}) with {status['levels']} of {status['total']} levels"
        self.add_news(text)
        self.record("ended", game=game.key, state=status["state"], levels=status["levels"])

    def finished(self, key: str) -> None:
        game = self.games.get(key)
        if game is None:
            return
        with self.lock:
            if game.freed:
                return
            game.freed = True
        status = game.status()
        self.mark_ended(game, status)
        self.settle(game, True)
        if not self.grow or self.stop_event.is_set() or _now() >= self.end_at:
            return
        if self.max_helpers > 0 and self.count + self.grown >= self.max_helpers:
            self.say(f"{key} left its lane; no new helper, the forum is at its cap of {self.max_helpers}")
            return
        with self.lock:
            self.grown += 1
            grown = self.grown
        self.say(f"{key} left its lane ({status['state']}, {status['levels']} of {status['total']} levels, "
                 f"{status['actions']} actions); the lane goes to a new helper: {self.count} at start + {grown}")
        self.record("grow", game=key, grown=grown, state=status["state"], levels=status["levels"])
        self.seat(f"on the lane {key} left", {"game": key, "state": status["state"], "levels": status["levels"],
                                              "total": status["total"], "actions": status["actions"]})

    def deliver(self, key: str, message: dict[str, Any]) -> dict[str, Any]:
        game = self.games.get(key)
        if game is None:
            return message
        with self.lock:
            game.calls += 1
            notes, game.notes_waiting = game.notes_waiting, []
            game.notes_delivered += len(notes)
            self.stats["delivered"] += len(notes)
        self.settle(game, False)
        if not notes:
            return message
        lines = []
        for one in notes:
            if not one.get("function"):
                lines.append(f"- {one['name']}: {one['text']}")
                continue
            final, source = inject(game.session.analyzer, one["function"], one["source"], one.get("plain") or "")
            one["final"] = final
            renamed = (f" It was renamed from {one['function']} to {final} because you already have a function "
                       f"named {one['function']}." if final != one["function"] else "")
            lines.append(f"- {one['name']} sent a function: {one['text']}\n```python\n{source}\n```\n"
                         f"{signature_of(source, final)} is already defined in your Python namespace and kept between "
                         f"calls like your own functions; call it or ignore it.{renamed}")
            self.say(f"function {final} from {one['name']} injected into {key}'s kept functions"
                     + (f" (renamed from {one['function']})" if renamed else ""))
        block = NOTE_HEADER + "\n" + "\n".join(lines)
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": [{"type": "text", "text": block}, *content]}
        else:
            message = {**message, "content": block + ("\n\n" + content if content else "")}
        status = game.status()
        parsed = len(game.parse())
        moves = game.moves()
        before = moves[-20:]
        for one in notes:
            self.record("delivered", game=key, name=one["name"], action=status["actions"], text=one["text"])
            self.add_news(f"your note to {key} was delivered at its action {status['actions']}", {one["member"]})
            with self.lock:
                game.pending.append({"member": one["member"], "name": one.get("plain") or one["name"], "text": one["text"],
                                     "function": one.get("final", ""),
                                     "call": game.calls, "actions": status["actions"], "levels": status["levels"],
                                     "parsed": parsed, "moves": len(moves), "noop_before": _noops(before),
                                     "seen_before": len(before)})
        self.say(f"{len(notes)} note(s) delivered to {key} at action {status['actions']}: "
                 + ", ".join(one["name"] for one in notes))
        return message

    def settle(self, game: _Game, ended: bool) -> None:
        with self.lock:
            due = [one for one in game.pending if ended or game.calls - one["call"] >= self.outcome_turns]
            game.pending = [one for one in game.pending if one not in due]
        if not due:
            return
        status = game.status()
        turns = game.parse()
        moves = game.moves()
        sections_cache: dict[int, list[list[Any]]] = {}

        def sections_for(turn: _Turn) -> list[list[Any]]:
            if turn.index not in sections_cache:
                sections_cache[turn.index] = turn["sections"]
            return sections_cache[turn.index]

        for one in due:
            after = moves[one["moves"]:]
            thinking = [(turn["turn"], "\n".join(line for label, _extra, lines in sections_for(turn)
                                                 if label.split(":")[0].strip() == "THINKING" for line in lines))
                        for turn in turns[one["parsed"]:]]
            mention = re.compile(r"\b(?:%s|helpers?|notes? from|the note|their note|this note)\b" % re.escape(one["name"]),
                                 re.IGNORECASE)
            quoted = ""
            for number, text in thinking:
                found = mention.search(text)
                if found:
                    quoted = (f"yes, turn {number}: \"..." + text[max(0, found.start() - 120):found.end() + 160]
                              .replace("\n", " ").strip() + "...\"")
                    break
            said = quoted or (f"no mention of a note or a helper in {len(thinking)} turn(s) of thinking"
                              if thinking else "no thinking written yet")
            if one.get("function"):
                used = [turn["turn"] for turn in turns[one["parsed"]:]
                        if any(label.split(":")[0].strip() == "TOOL CALL" and f"{one['function']}(" in line
                               for label, _extra, lines in sections_for(turn) for line in lines)]
                said += (f"; its code called {one['function']} in turn(s) {', '.join(map(str, used[:5]))}" if used
                         else f"; its code never called {one['function']}")
            when = "when the game ended" if ended else f"{game.calls - one['call']} player turns after delivery"
            kind = f"function {one['function']}" if one.get("function") else "note"
            text = (f"outcome of your {kind} to {game.key}, {when}"
                    f" (delivered at action {one['actions']}): its thinking since: {said}; levels won since: "
                    f"{status['levels'] - one['levels']}; actions since: {status['actions'] - one['actions']}; actions "
                    f"that changed nothing: {one['noop_before']} of the last {one['seen_before']} before the note, "
                    f"{_noops(after)} of {len(after)} since. Your note was: \"{_cap(one['text'], 200)}\"")
            self.add_news(text, {one["member"]})
            self.record("outcome", game=game.key, member=one["member"], name=one["name"], ended=ended,
                        mentioned=bool(quoted), levels=status["levels"] - one["levels"],
                        actions=status["actions"] - one["actions"], noop_before=one["noop_before"],
                        seen_before=one["seen_before"], noop_after=_noops(after), seen_after=len(after))

    def names(self) -> list[str]:
        with self.lock:
            return [self.label(one["id"]) for one in self.members if one.get("named") and one.get("alive")]

    def start(self) -> "Forum":
        self.say(f"{self.count} helpers arrive one every {self.arrival_s:.0f} s"
                 f"{', and every game that leaves its lane gives the lane to one more helper' if self.grow else ''}; "
                 f"show_game lists the {self.similar_top} most similar games; context {self.context_tokens} tokens, "
                 f"handoff asked at {self.handoff_at}, "
                 f"{'one serve session a seat (X-Session-ID, released at each handoff)' if self.sessions else 'sessionless requests with a stop list'}, "
                 f"replies up to {self.max_output}, reads up to {self.read_chars} chars; forum in {self.root}")
        self.record("start", helpers=self.count, grow=self.grow, similar_top=self.similar_top, context=self.context_tokens, max_output=self.max_output,
                    read_chars=self.read_chars, end_at=self.end_at)
        watcher = threading.Thread(target=self.watch, name="forum-watch", daemon=True)
        watcher.start()
        spawner = threading.Thread(target=self.spawn, name="forum-spawn", daemon=True)
        spawner.start()
        self.threads_out += [watcher, spawner]
        return self

    def seat(self, why: str, freed: dict[str, Any] | None = None) -> None:
        with self.lock:
            index = len(self.members)
            name = NAMES[index % len(NAMES)] + ("" if index < len(NAMES) else str(index // len(NAMES) + 1))
            member = {"id": index, "order": index + 1, "name": name, "named": True, "alive": True, "joined": _now(),
                      "generation": 1, "handoff": "", "handoffs": 0, "freed": freed, "sent": []}
            self.members.append(member)
            self.add_news(f"a new helper arrived: {self.label(index)}")
        self.record("named", member=index, order=index + 1, name=name, why=why)
        self.say(f"seat {index} is {self.label(index)} ({why})")
        helper = Helper(self, member)
        thread = threading.Thread(target=helper.run, name=f"forum-helper-{index}", daemon=True)
        thread.start()
        self.threads_out.append(thread)

    def spawn(self) -> None:
        for _index in range(self.count):
            if self.stop_event.is_set() or _now() >= self.end_at:
                return
            self.seat("at the start")
            if self.stop_event.wait(self.arrival_s):
                return

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            stats = dict(self.stats)
        self.record("stop", **stats)
        self.say(f"stopped: {json.dumps(stats)}")


class Helper:

    def __init__(self, forum: Forum, member: dict[str, Any]) -> None:
        self.forum = forum
        self.member = member
        self.cursor = len(forum.news)
        self.messages: list[dict[str, Any]] = []
        self.log_path = forum.root / "helpers" / f"{member['id']:03d}.jsonl"
        self.errors = 0
        self.context = 0
        self.asked = False
        self.ended = False

    def log(self, **fields: Any) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": round(_now() - self.forum.started, 1), **fields}, ensure_ascii=False) + "\n")

    def me(self) -> str:
        return self.forum.label(self.member["id"])

    def others(self) -> list[str]:
        return [label for label in self.forum.names() if label != self.me()]

    def freed_line(self) -> str:
        freed = self.member.get("freed")
        if not freed:
            return ""
        return (f" Your seat was created on the lane that game {freed['game']} left: it ended in state {freed['state']} "
                f"with {freed['levels']} of {freed['total']} levels won after {freed['actions']} actions. Its transcript "
                f"stays readable (show_game, read_turns, search, view_board, run_python), so what its player learned "
                f"can still help games with similar mechanics.")

    def welcome(self) -> str:
        with self.forum.lock:
            threads = [f"#{one['id']} {one['title']} ({len(one['posts'])} posts)" for one in self.forum.threads[-15:]]
            games = len(self.forum.games)
        present = self.others()
        return (f"You are {self.me()}, a new helper agent, just arrived ({self.forum.minute()}). Your name is "
                f"{self.member['name']}, your arrival number is #{self.member['order']}; the harness signs everything "
                f"you post or send with it.{self.freed_line()} Your goal: help the players win levels. {games} games "
                f"are being played right now. Agents on the forum at the moment: "
                f"{', '.join(present) if present else 'none yet, you are among the first'}. "
                f"Forum threads: {'; '.join(threads) if threads else 'none yet'}.")

    def successor(self) -> str:
        present = self.others()
        handoff = self.member["handoff"] or "(your previous instance wrote no handoff)"
        return (f"You are {self.me()}, a helper agent, instance {self.member['generation']} of your seat "
                f"({self.forum.minute()}). Your seat keeps its arrival number #{self.member['order']}. Your previous "
                f"instance in this seat filled its memory and ended. Its handoff to you:\n\n{handoff}\n\nAgents on the "
                f"forum at the moment: {', '.join(present) if present else 'none'}. Continue from the handoff.")

    def take_news(self) -> list[str]:
        with self.forum.lock:
            fresh = self.forum.news[self.cursor:]
            self.cursor = len(self.forum.news)
        mine = self.member["id"]
        return [one["text"] for one in fresh if one["audience"] is None or mine in one["audience"]]

    def news_message(self, items: list[str]) -> str:
        if len(items) > 30:
            items = items[-30:] + [f"({len(items) - 30} older items not shown)"]
        return f"[news for {self.me()}, {self.forum.minute()}]\n" + ("\n".join(f"- {one}" for one in items) if items else "- nothing new")

    def chat(self) -> dict[str, Any] | None:
        payload = {"model": self.forum.model, "messages": self.messages, "tools": _tools(self.forum.label(self.member["id"])), "tool_choice": "auto",
                   "max_tokens": self.forum.max_output, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
                   "chat_template_kwargs": {"enable_thinking": True}}
        headers = {"Content-Type": "application/json"}
        if self.forum.sessions:
            headers[SESSION_HEADER] = self.session_key()
        else:
            payload["stop"] = [UNREACHABLE_STOP]
        if self.forum.api_key:
            headers["Authorization"] = f"Bearer {self.forum.api_key}"
        try:
            response = requests.post(f"{self.forum.base_url}/chat/completions", json=payload, headers=headers,
                                     timeout=(10, 900))
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
        except Exception as exc:
            with self.forum.lock:
                self.forum.stats["errors"] += 1
            self.errors += 1
            self.log(error=f"{type(exc).__name__}: {str(exc)[:300]}")
            return None
        self.errors = 0
        usage = body.get("usage") or {}
        with self.forum.lock:
            self.forum.stats["requests"] += 1
            self.forum.stats["prompt"] += int(usage.get("prompt_tokens") or 0)
            self.forum.stats["generated"] += int(usage.get("completion_tokens") or 0)
            self.forum.stats["cached"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        message = choice.get("message") or {}
        self.context = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        self.log(reply={"content": message.get("content"), "reasoning": message.get("reasoning")
                        or message.get("reasoning_content"), "tool_calls": message.get("tool_calls"),
                        "finish": choice.get("finish_reason")}, usage=usage, context=self.context,
                 generation=self.member["generation"])
        return message

    def session_key(self) -> str:
        return f"forum-{self.member['id']}-g{self.member['generation']}"

    def release(self) -> None:
        if not self.forum.sessions:
            return
        body = {"model": self.forum.model, "prompt": ".", "max_tokens": 1, "temperature": 0.0,
                "vllm_xargs": {RELEASE_MARK: 1}}
        headers = {"Content-Type": "application/json", SESSION_HEADER: self.session_key()}
        if self.forum.api_key:
            headers["Authorization"] = f"Bearer {self.forum.api_key}"
        try:
            requests.post(f"{self.forum.base_url}/completions", json=body, headers=headers, timeout=(10, 30))
            self.log(released=self.session_key())
        except Exception as exc:
            self.log(release_error=f"{type(exc).__name__}: {str(exc)[:200]}")

    def reseat(self, why: str) -> None:
        forum = self.forum
        if not self.member["handoff"]:
            with forum.lock:
                mine = [f"#{thread['id']} {thread['title']}: {post['text'][:400]}" for thread in forum.threads
                        for post in thread["posts"] if post["member"] == self.member["id"]][-6:]
            self.member["handoff"] = ("(written by the harness: your previous instance wrote no handoff) Your last "
                                      "forum posts:\n" + ("\n".join(mine) or "none"))
        forum.record("reseat", member=self.member["id"], name=self.member["name"], why=why, context=self.context,
                     generation=self.member["generation"], handoff=self.member["handoff"])
        forum.say(f"{self.forum.label(self.member['id'])} ends instance {self.member['generation']} at "
                  f"{self.context} tokens ({why}); a fresh instance takes the seat. HANDOFF:\n"
                  f"{_quote(self.member['handoff'][:1000])}")
        self.release()
        self.member["generation"] += 1
        self.messages = [{"role": "system", "content": f"You are {forum.label(self.member['id'])}." + chr(10) + system_text(forum.context_tokens, forum.handoff_at)},
                         {"role": "user", "content": self.successor()}]
        self.member["handoff"] = ""
        self.context = 0
        self.asked = False
        self.ended = False

    def run(self) -> None:
        forum = self.forum
        self.messages = [{"role": "system", "content": f"You are {forum.label(self.member['id'])}." + chr(10) + system_text(forum.context_tokens, forum.handoff_at)},
                         {"role": "user", "content": self.welcome()}]
        forum.record("arrived", member=self.member["id"])
        try:
            while not forum.stop_event.is_set() and _now() < forum.end_at:
                message = self.chat()
                if message is None:
                    if forum.stop_event.wait(min(60.0, 5.0 * self.errors)):
                        break
                    continue
                calls = [call for call in message.get("tool_calls") or [] if isinstance(call, dict)]
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                self.messages.append({"role": "assistant", "content": message.get("content") or "",
                                      **({"reasoning": reasoning} if forum.sessions and reasoning else {}),
                                      **({"tool_calls": calls} if calls else {})})
                images: list[dict[str, Any]] = []
                for call in calls:
                    function = call.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = None
                    if not isinstance(arguments, dict):
                        text = "error: the arguments are not a JSON object"
                    else:
                        text, image = self.dispatch(str(function.get("name") or ""), arguments)
                        if image:
                            images.append(image)
                    self.messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": text})
                    self.log(tool=function.get("name"), arguments=arguments, result=text[:4000])
                items = self.take_news()
                if not calls:
                    waited = 0.0
                    while not items and waited < 20.0 and not forum.stop_event.is_set():
                        forum.stop_event.wait(5.0)
                        waited += 5.0
                        items = self.take_news()
                parts: list[dict[str, Any]] = []
                for image in images:
                    parts += [{"type": "text", "text": image["label"]}, {"type": "image_url", "image_url": {"url": image["url"]}}]
                if self.ended:
                    self.reseat("handoff written")
                    continue
                if self.context >= forum.context_tokens:
                    self.reseat("memory full without a handoff")
                    continue
                said = self.news_message(items)
                if self.context >= forum.handoff_at and not self.asked:
                    self.asked = True
                    said += (f"\n\n[memory] Your conversation is at {self.context:,} tokens of your "
                             f"{forum.context_tokens:,} limit. Write your handoff now with write_handoff. After it "
                             f"this instance ends and a fresh instance of you continues in your seat from the "
                             f"handoff alone.")
                    forum.record("handoff_asked", member=self.member["id"], context=self.context)
                parts.append({"type": "text", "text": said})
                self.messages.append({"role": "user", "content": parts if images else said})
        finally:
            self.release()
            with forum.lock:
                self.member["alive"] = False
            forum.record("left", member=self.member["id"], name=self.member["name"])

    def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"error: no tool named {name!r}", None
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}", None
        except Exception as exc:
            return f"error: {type(exc).__name__}: {str(exc)[:300]}", None

    def tool_show_game(self, game: str = "") -> tuple[str, Any]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        status = found.status()
        current, history = found.frames()
        moves = [entry for entry in history if getattr(entry, "action", "")]
        runs: list[list[Any]] = []
        for entry in moves:
            if runs and runs[-1][0] == entry.action:
                runs[-1][1] += 1
            else:
                runs.append([entry.action, 1])
        sequence = ", ".join(f"{action}x{count}" if count > 1 else str(action) for action, count in runs)
        recent = []
        for entry in moves[-3:]:
            result = dict(getattr(entry, "result", {}) or {})
            said = [f"step {getattr(entry.frame, 'step', '?')}", str(entry.action)]
            if result.get("level_completed"):
                said.append("LEVEL COMPLETED")
            if result.get("game_over"):
                said.append("game over (level restarted)")
            if result.get("no_op"):
                said.append("no-op")
            elif result.get("gameplay_changed"):
                said.append("gameplay changed")
            elif result.get("board_changed"):
                said.append("only the edge/HUD changed")
            if result.get("automatic"):
                said.append("automatic")
            if result.get("animation"):
                said.append("animated")
            recent.append(" | ".join(said))
        per_level = [count for count in status["per_level"] if count]
        won = ", ".join(f"level {index + 1}: {count} actions" for index, count in enumerate(per_level[:status["levels"]]))
        turns = found.parse()
        thinking = ""
        for turn in reversed(turns):
            thinking = self.render_turn(turn, "thinking").split("\n", 1)[-1].strip()
            if thinking:
                thinking = f"(turn {turn['turn']}) " + thinking
                break
        similar = self.similar_section(found)
        text = "\n".join([
            f"{found.key}: level {getattr(current, 'level', '?')} of {status['total']}, {status['levels']} won, "
            f"{status['actions']} actions, {status['tokens'] / 1000:.0f}k tokens, {status['minutes']:.0f} min, {status['state']}",
            f"Claimed by: {self.forum.holder(found.key) or 'nobody'}",
            f"Levels won: {won or 'none yet'}",
            f"All actions ({len(moves)}): {_cap(sequence, 2500) or 'none yet'}",
            "Last actions: " + ("; ".join(recent) or "none"),
            "Player's latest thinking: " + _cap(thinking or "(none written yet)", 3000),
            *similar,
            "Current board image attached after the tool results.",
        ])
        from inference.agent.vision_context import frame_to_png_data_url
        image = {"label": f"{found.key} current board, step {getattr(current, 'step', '?')}",
                 "url": frame_to_png_data_url(current, upscale=self.forum.upscale)}
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
            self.forum.stats["images"] += 1
        return text, image

    def similar_section(self, found: _Game) -> list[str]:
        forum = self.forum
        if forum.similar_top <= 0:
            return []
        if not forum.similar.built:
            forum.refresh_similar(force=True)
        matches = forum.similar.similar(found.key, forum.similar_top, forum.similar_words)
        if not matches:
            return ["Similar games: none yet (too little distinctive thinking and code so far)."]
        lines = [f"Similar games (by the distinctive words their players' thinking and code share with this one; "
                 f"index {_now() - forum.similar.built:.0f} s old; this game's thinking and code so far "
                 f"~{found.cloud_tokens() // 1000}K tokens):"]
        for score, key, shared in matches:
            other = forum.game(key)
            levels = other.status()["levels"] if other is not None else 0
            basis = f"~{other.cloud_tokens() // 1000}K tokens" if other is not None else "unknown tokens"
            lines.append(f"- {key}: {levels} levels won, {basis}, similarity {score:.2f}, shared words: {', '.join(shared)}")
        lines.append(SIMILAR_EARLY)
        lines.append(SIMILAR_LINE)
        return lines

    def tool_write_handoff(self, text: str = "") -> tuple[str, None]:
        text = str(text).strip()
        if not text:
            return "error: an empty handoff", None
        self.member["handoff"] = text[:8000]
        self.member["handoffs"] += 1
        self.ended = True
        return "handoff saved; this instance ends now", None

    def tool_claim(self, game: str = "") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        mine = self.member["id"]
        with self.forum.lock:
            held = self.forum.claims.get(found.key)
            if held is None:
                self.forum.claims[found.key] = mine
        if held is not None and held != mine:
            return (f"error: {found.key} is held by {self.forum.label(held)}; ask on the forum or wait until it is "
                    f"released"), None
        if held == mine:
            return f"you already hold {found.key}", None
        self.forum.add_news(f"{self.me()} claimed {found.key}")
        self.forum.record("claim", game=found.key, member=mine, name=self.member["name"])
        return f"you hold {found.key}; release it when you stop looking after it", None

    def tool_release(self, game: str = "") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        mine = self.member["id"]
        with self.forum.lock:
            held = self.forum.claims.get(found.key)
            if held == mine:
                del self.forum.claims[found.key]
        if held != mine:
            return (f"error: you do not hold {found.key}"
                    + (f"; {self.forum.label(held)} does" if held is not None else "")), None
        self.forum.add_news(f"{self.me()} released {found.key}")
        self.forum.record("release", game=found.key, member=mine, name=self.member["name"])
        return f"released {found.key}", None

    def stamp(self, text: str) -> tuple[str, list[str]]:
        body, dropped = strip_signature(text)
        if dropped:
            self.forum.record("signature_stripped", member=self.member["id"], name=self.member["name"], dropped=dropped)
        return body, dropped

    def tool_news(self) -> tuple[str, None]:
        return self.news_message(self.take_news()), None

    def tool_list_games(self) -> tuple[str, None]:
        with self.forum.lock:
            games = sorted(self.forum.games.values(), key=lambda one: one.key)
        lines = []
        for game in games:
            status = game.status()
            holder = self.forum.holder(game.key)
            lines.append(f"{game.key}: levels {status['levels']}/{status['total']}, actions {status['actions']} "
                         f"(per level {status['per_level']}), {status['tokens'] / 1000:.0f}k tokens, "
                         f"{status['minutes']:.0f} min, {status['state']}, notes waiting {len(game.notes_waiting)}, "
                         f"delivered {game.notes_delivered}, claimed by {holder or 'nobody'}")
        return "\n".join(lines) or "no games yet", None

    def render_turn(self, turn: dict[str, Any], part: str) -> str:
        wanted = PARTS.get(part, None)
        out = [f"== turn {turn['turn']} (actions taken before it: {turn['action']}, clock {turn['clock']})"]
        for label, extra, lines in turn["sections"]:
            kind = label.split(":")[0].strip()
            if wanted is None and kind in DROPPED_SECTIONS:
                continue
            if wanted is not None and kind not in wanted:
                continue
            body = "\n".join(lines).strip()
            if body:
                out.append(f"[{kind}{': ' + extra if extra else ''}]\n{body}")
        return "\n".join(out)

    def tool_read_turns(self, game: str = "", from_turn: int | None = None, count: int = 3,
                        part: str = "all") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        if part not in PARTS:
            return f"error: part must be one of {sorted(PARTS)}", None
        turns = found.parse()
        if not turns:
            return "no turns written yet", None
        count = max(1, min(6, int(count or 3)))
        if from_turn is None:
            chosen = turns[-count:]
        else:
            chosen = [turn for turn in turns if turn["turn"] >= int(from_turn)][:count]
        if part == "system":
            chosen = turns[:1]
        text = "\n\n".join(self.render_turn(turn, part) for turn in chosen) or "no such turns"
        text = f"{found.key}: turns {turns[0]['turn']}-{turns[-1]['turn']} exist.\n" + text
        text = _cap(text, self.forum.read_chars)
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
        return text, None

    def tool_view_board(self, game: str = "", step: int | None = None, format: str = "text") -> tuple[str, Any]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        current, history = found.frames()
        frame = current
        if step is not None:
            step = int(step)
            matches = [entry.frame for entry in history if int(getattr(entry.frame, "step", -1)) == step]
            if not matches:
                steps = [int(getattr(entry.frame, "step", -1)) for entry in history]
                return f"error: no board at step {step}; boards exist for steps {min(steps)}-{max(steps)}", None
            frame = matches[-1]
        label = f"{found.key} board at step {frame.step}, level {frame.level}"
        image = None
        if format in ("image", "both"):
            from inference.agent.vision_context import frame_to_png_data_url
            image = {"label": label, "url": frame_to_png_data_url(frame, upscale=self.forum.upscale)}
            with self.forum.lock:
                self.forum.stats["images"] += 1
        text = label + (" (image attached after the tool results)" if image else "")
        if format in ("text", "both"):
            text += "\n" + frame.ascii
            with self.forum.lock:
                self.forum.stats["reads"] += 1
                self.forum.stats["read_chars"] += len(frame.ascii)
        return text, image

    def tool_search(self, pattern: str = "", game: str | None = None) -> tuple[str, None]:
        try:
            regex = re.compile(str(pattern), re.IGNORECASE)
        except re.error as exc:
            return f"error: bad pattern: {exc}", None
        if game:
            found = self.forum.game(game)
            if found is None:
                return f"error: no game {game!r}", None
            games = [found]
        else:
            with self.forum.lock:
                games = list(self.forum.games.values())
        hits: list[str] = []
        for one in games:
            for turn, sections in one.turn_sections():
                for label, _extra, lines in sections:
                    if label in DROPPED_SECTIONS:
                        continue
                    for line in lines:
                        if regex.search(line):
                            hits.append(f"{one.key} turn {turn['turn']} [{label}]: {line.strip()[:200]}")
                            if len(hits) >= 25:
                                break
                    if len(hits) >= 25:
                        break
                if len(hits) >= 25:
                    break
        if not game and len(hits) < 25:
            with self.forum.lock:
                threads = [dict(one, posts=list(one["posts"])) for one in self.forum.threads]
            for thread in threads:
                for index, post in enumerate(thread["posts"]):
                    if regex.search(post["text"]):
                        hits.append(f"forum #{thread['id']} post {index} by {post['name']}: {post['text'][:200]}")
                        if len(hits) >= 25:
                            break
        text = "\n".join(hits) or "no matches"
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
        return text, None

    def tool_run_python(self, game: str = "", code: str = "") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        return self.sandbox_text(self.sandbox(found, str(code), None)), None

    def tool_forum_list(self) -> tuple[str, None]:
        with self.forum.lock:
            lines = [f"#{one['id']} {one['title']} by {one['author_label']}, {len(one['posts'])} posts, last by "
                     f"{one['posts'][-1]['label']} at minute {(one['posts'][-1]['t'] - self.forum.started) / 60:.0f}"
                     for one in self.forum.threads]
        return "\n".join(lines) or "no threads yet", None

    def tool_forum_read(self, thread: int = 0, from_post: int = 0) -> tuple[str, None]:
        with self.forum.lock:
            found = next((one for one in self.forum.threads if one["id"] == int(thread)), None)
            posts = list(found["posts"]) if found else []
        if found is None:
            return f"error: no thread #{thread}", None
        start = max(0, int(from_post or 0))
        text = f"#{found['id']} {found['title']} ({len(posts)} posts)\n" + "\n\n".join(
            f"[{index}] {post['label']}, minute {(post['t'] - self.forum.started) / 60:.0f}:\n{post['text']}"
            for index, post in enumerate(posts) if index >= start)
        text = _cap(text, self.forum.read_chars)
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
        return text, None

    def tool_forum_new_thread(self, title: str = "", text: str = "") -> tuple[str, None]:
        title = strip_title(str(title).strip()[:120])
        text, dropped = self.stamp(str(text).strip()[:3000])
        if not title or not text:
            return "error: a thread needs a title and a text", None
        name = self.member["name"]
        label = self.me()
        text = f"{text}\n-- {label}"
        with self.forum.lock:
            thread = {"id": len(self.forum.threads), "title": title, "author": name, "author_label": label,
                      "author_id": self.member["id"],
                      "posts": [{"name": name, "label": label, "member": self.member["id"], "t": _now(), "text": text}]}
            self.forum.threads.append(thread)
            self.forum.stats["threads"] += 1
            self.forum.stats["posts"] += 1
            self.forum.add_news(f"new thread #{thread['id']} '{title}' by {label}")
        self.forum.record("thread", thread=thread["id"], title=title, name=name, order=self.member["order"], text=text)
        self.forum.say(f"NEW THREAD #{thread['id']} '{title}' by {label}:\n{_quote(text)}")
        return f"opened thread #{thread['id']}, signed by the harness as {label}" + self.dropped_line(dropped), None

    def dropped_line(self, dropped: list[str]) -> str:
        if not dropped:
            return ""
        return (f"; the signature you wrote ({' / '.join(dropped)}) was removed: the harness signs for you, never "
                f"write one")

    def tool_forum_post(self, thread: int = 0, text: str = "") -> tuple[str, None]:
        text, dropped = self.stamp(str(text).strip()[:3000])
        if not text:
            return "error: empty post", None
        name = self.member["name"]
        label = self.me()
        text = f"{text}\n-- {label}"
        with self.forum.lock:
            found = next((one for one in self.forum.threads if one["id"] == int(thread)), None)
            if found is None:
                return f"error: no thread #{thread}", None
            found["posts"].append({"name": name, "label": label, "member": self.member["id"], "t": _now(), "text": text})
            index = len(found["posts"]) - 1
            self.forum.stats["posts"] += 1
            readers = {post["member"] for post in found["posts"]} - {self.member["id"]}
            self.forum.add_news(f"{label} replied in #{found['id']} '{found['title']}' (post {index})", readers)
        self.forum.record("post", thread=int(thread), index=index, name=name, order=self.member["order"], text=text)
        self.forum.say(f"POST #{thread}.{index} '{found['title']}' by {label}:\n{_quote(text)}")
        return f"posted as #{thread} post {index}, signed by the harness as {label}" + self.dropped_line(dropped), None

    def tool_send_note(self, game: str = "", text: str = "") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        text, dropped = self.stamp(str(text).strip())
        if not text:
            return "error: empty note", None
        if len(text) > 800:
            return f"error: a note is at most 800 characters, this one is {len(text)}", None
        return self.queue(found, text, dropped, None), None

    def queue(self, found: _Game, text: str, dropped: list[str], function: dict[str, str] | None) -> str:
        name = self.member["name"]
        label = self.me()
        kind = "function" if function else "note"
        flat = normalized(text + " " + (function["source"] if function else ""))
        alike = sorted({one["game"] for one in self.member["sent"] if one["game"] != found.key
                        and near(flat, one["flat"]) > self.forum.template_similarity})
        if self.forum.template_games > 0 and len(alike) >= self.forum.template_games:
            self.forum.record("template_refused", game=found.key, name=name, what=kind, alike=alike, text=text)
            self.forum.say(f"{kind.upper()} REFUSED {label} -> {found.key}: nearly the same {kind} already went to "
                           f"{', '.join(alike)}")
            return (f"error: this {kind} is nearly the same as the ones you already sent to {len(alike)} other games "
                    f"({', '.join(alike[:6])}); post it in a thread instead, and send a player only what its own "
                    f"game's evidence shows")
        with self.forum.lock:
            if any(one["member"] == self.member["id"] for one in found.notes_waiting):
                return "error: your previous note or function to this game has not been delivered yet"
            if len(found.notes_waiting) >= 3:
                return "error: this game already has 3 notes or functions waiting for its next turn"
            found.notes_waiting.append({"name": label, "plain": name, "member": self.member["id"], "text": text,
                                        "t": _now(), **({"function": function["name"], "source": function["source"]}
                                                        if function else {})})
            self.member["sent"].append({"game": found.key, "flat": flat})
            self.forum.stats["notes"] += 1
            if function:
                self.forum.stats["functions"] += 1
            self.forum.add_news(f"{label} sent a {kind} to {found.key}")
        if function:
            self.forum.record("function", game=found.key, name=name, order=self.member["order"], text=text,
                              function=function["name"], source=function["source"])
            self.forum.say(f"FUNCTION {label} -> {found.key}: {function['name']}\n{_quote(text)}\n"
                           + "\n".join("    > " + line for line in function["source"].splitlines()))
        else:
            self.forum.record("note", game=found.key, name=name, order=self.member["order"], text=text)
            self.forum.say(f"NOTE {label} -> {found.key}:\n{_quote(text)}")
        holder = self.forum.claims.get(found.key)
        held = (f"; note that {found.key} is claimed by {self.forum.label(holder)}, who may be advising it too"
                if holder is not None and holder != self.member["id"] else "")
        return (f"queued for {found.key}, signed by the harness as {label}; it arrives at the start of its next turn"
                + held + self.dropped_line(dropped))

    def tool_send_function(self, game: str = "", name: str = "", code: str = "", message: str = "") -> tuple[str, None]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        if not _functions_kept():
            return "error: the players' Python keeps no functions between calls in this run; send a note instead", None
        name, code = str(name).strip(), str(code).strip("\n")
        text, dropped = self.stamp(str(message).strip())
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,60}", name):
            return "error: name must be a Python identifier", None
        if not text:
            return "error: a function needs a short message saying what it does and when to use it", None
        if len(text) > 800:
            return f"error: the message is at most 800 characters, this one is {len(text)}", None
        if len(code) > self.forum.function_chars:
            return f"error: the code is at most {self.forum.function_chars} characters, this one is {len(code)}", None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"error: the code does not compile, nothing was sent: {exc}", None
        defined = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if defined != [name]:
            return (f"error: the code must define exactly one top-level function, {name}; it defines "
                    f"{defined or 'none'}. Put helpers inside it"), None
        node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        rest = [stmt for stmt in tree.body if stmt is not node]
        called = any(isinstance(one, ast.Call) and isinstance(one.func, ast.Name) and one.func.id == name
                     for stmt in rest for one in ast.walk(stmt))
        required = (len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults)
                    + sum(1 for default in node.args.kw_defaults if default is None))
        trial = code
        if not called:
            if required:
                return (f"error: add a trial after the definition: a line that calls {name}(...) on this game's "
                        f"boards and prints the result; it runs only here, never in the player's sandbox"), None
            trial = code + f"\nprint({name}())"
        result = self.sandbox(found, trial, [])
        out = self.sandbox_text(result)
        if result.get("error"):
            return f"error: the trial raised, nothing was sent. Trial output:\n{out}", None
        kept = {str(item.get("name")): str(item.get("source")) for item in result.get("keepable_functions") or []}
        if name not in kept:
            why = [str(item.get("reason")) for item in result.get("retention_rejected") or []
                   if item.get("name") in (name, "all")]
            return (f"error: the player's Python could not keep {name} between calls"
                    + (f": {'; '.join(why)}" if why else "") + f". Nothing was sent. Trial output:\n{out}"), None
        said = self.queue(found, text, dropped, {"name": name, "source": kept[name]})
        return f"trial output:\n{out}\n\n{said}", None

    def sandbox(self, found: _Game, code: str, kept: list[str] | None) -> dict[str, Any]:
        from inference.agent import tool_agent
        from inference.agent.python_tool_sandbox import run_sandboxed_python
        current, history = found.frames()
        state = {"current_frame": tool_agent._ascii_frame_view_payload(current),
                 "history": tool_agent._ascii_history_view_payload(history),
                 "valid_actions": [], "last_action_call_result": {}}

        def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("helpers cannot act")

        extra = {} if kept is None else {"kept_functions": kept,
                                         "retain_imports": _env_flag("ARC3_PERSISTENT_FUNCTIONS_IMPORTS")}
        with self.forum.python_slots:
            result = run_sandboxed_python(code=str(code), timeout_seconds=20, initial_state=state, action_handler=refuse,
                                          **extra)
        with self.forum.lock:
            self.forum.stats["python"] += 1
        return result

    def sandbox_text(self, result: dict[str, Any]) -> str:
        out = []
        if result.get("stdout"):
            out.append(str(result["stdout"]))
        if result.get("result") is not None:
            out.append(f"result = {json.dumps(result['result'], default=str)[:3000]}")
        if result.get("error"):
            out.append(f"error: {result['error']}")
        return _cap("\n".join(out) or "(no output)", 6000)


PLAYER_LINE = ("Helper agents are watching every game being played, including levels already won in other games. "
               "They can see your board and your reasoning and may send you a short note to help; a note appears "
               "at the start of a turn under 'Notes from helper agents'. Check each note against your own evidence.")


def player_line() -> str:
    return "\n\n" + PLAYER_LINE if ACTIVE is not None else ""


def register(session: Any, key: str) -> None:
    forum = ACTIVE
    if forum is not None:
        forum.add_game(session, key)


def finished(key: str) -> None:
    forum = ACTIVE
    if forum is None or not key:
        return
    try:
        forum.finished(key)
    except Exception as exc:
        print(f"forum: the end of {key} was not handled, {type(exc).__name__}: {exc}", flush=True)


def deliver(key: str, message: dict[str, Any]) -> dict[str, Any]:
    forum = ACTIVE
    if forum is None or not key:
        return message
    try:
        return forum.deliver(key, message)
    except Exception as exc:
        print(f"forum: delivery to {key} failed, {type(exc).__name__}: {exc}", flush=True)
        return message


def create(root: Path, base_url: str, model: str, end_at: float, helpers: int | None = None) -> Forum:
    global ACTIVE
    count = _env_int("ARC3_FORUM_HELPERS", 0) if helpers is None else helpers
    ACTIVE = Forum(root, base_url, model, count, end_at)
    return ACTIVE
