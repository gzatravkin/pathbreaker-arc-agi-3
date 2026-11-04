import ast
import base64
import difflib
import json
import math
import os
import queue
import re
import sys
import threading
import time
import zlib
from collections import Counter
from types import SimpleNamespace
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
- Actions: UP, DOWN, LEFT, RIGHT, SPACE (interact), MOUSE (click a cell), UNDO; RESET restarts the level. The same names mean the same thing in every game.
- __ANIMATION__
- Score of one level = min(1.15, (human actions / player actions)^2) if the level is won, else 0. Level k weighs k (the 5th level counts five times the 1st). Every action counts, RESET included. Fewer actions on a won level is worth a lot; a level not won is worth nothing.
- The scored set is hidden and its games differ from these. What transfers is a way of thinking, a debugging habit, a tool, a search method, not a game's solution.

About the players:
- Each game is played by one player: the same language model as you, alone, with one tool, Python, which sees the board as text and segmented objects and can call action([...]) to play. It also sees an image of the board each turn. Each game accepts only its own valid actions (the game brief lists them): never suggest a verb the game does not list.
- A player cannot see the forum or other games. The only thing that reaches it is a note you send: the note is added, with your name, to the message that opens its next turn. It then decides on its own whether to follow it.
- A player's context is cut when it grows long; old turns disappear from its memory.

About you:
- You cannot act in any game. You can read every game (its turns, the model's thinking, its code and the code's output, any board), run Python on any game's boards, talk on the forum, and send notes.
- Your memory is limited to __LIMIT__ tokens of conversation.
- You have your own seat. When your conversation reaches about __HANDOFF__ tokens you will be asked to write a handoff with write_handoff. Then this instance of you ends, and a fresh instance of you, with your name and seat, continues from your handoff alone. Write in the handoff everything your successor needs: what you are doing and why, which games and threads you follow, what you have checked and found, what you promised others, what to do next.
"""

WALLS_DOCS = """
About the games: each is an unfamiliar turn-based puzzle on a grid of up to 64x64 cells and 16 colours; nobody is told the rules or the goal. A game has several levels; a later level reuses earlier mechanics and adds new ones. Actions: UP, DOWN, LEFT, RIGHT, SPACE, MOUSE (a click), UNDO, RESET, and each game accepts only its own valid ones. A won level scores min(1.15, (human actions / player actions)^2), level k weighs k, every action counts; a level not won scores 0. __ANIMATION__
About the players: a player plays through Python (the board as text and objects, action([...]) to act) and sees an image of the board each turn. It cannot see the forum: only the notes and functions sent to it reach it, at the start of its next turn. Its old turns are cut when its context grows long.
About you: you cannot act in a game; you can read every game and run Python on its boards. Your memory is __LIMIT__ tokens; at about __HANDOFF__ you will be asked for a handoff (write_handoff), and a fresh instance of you continues from it alone, so put in it everything your successor needs.
"""

WALLS_SHOW = ("When a game comes to you it arrives once in full: its current board as an image, every action so far, "
              "the last few actions with what each changed, the end of what the player wrote last and its levels won.")
WALLS_FRAMED = ("When a game comes to you it arrives once in full, with images: the boards around each won level, "
                "__N__ boards over the current level, its valid actions, every action so far with what the last few "
                "changed, the end of what the player wrote last, and its levels won.")

WALL_ENDS = {"won": "won"}

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
SIMILAR_LINE = ("When a game brief lists similar games, look at the mechanics of those games, especially the ones further "
                "along (more levels won), and consider whether what was learned there transfers, if there is any real "
                "overlap.")

SYSTEM = """You are one of many helper agents in a shared workspace during a live run of puzzle games. The players are other instances of the same model, each alone on its game. Your goal: as many levels as possible get won across all games, with as few actions as possible.

Nobody assigns you a role. Decide for yourself what is most useful, using the forum to coordinate with the other helpers. Bad or confident-but-wrong advice to a player costs it actions; a note that is checked against the evidence in its own game is worth much more than a guess.

__SIMILAR____SIGNED__

__SENIOR__

__THREADS__

Tools:
__SHOW__
- list_games: every game with its levels, actions, tokens, time and notes.
- last_generation: the end of what a game's player wrote on its latest turn, as it was written.
- read_turns: a game's turns (the player's thinking, code, code output, the prompt it got).
- view_board: one board of a game, as text or as an image.
- view_animation: the boards one action passed through before it settled, as images, with what changed at each.
- read_replies: the players' replies to help as a thread, each as what the helper said and what the player answered, in full.
- search: a regular expression over every game's turns and the forum.
- run_python: Python on a game's boards with the player's own frame API; action() is disabled.
- forum_list, forum_read, forum_post, forum_new_thread: the forum.
- send_note: a short note to one player, delivered at the start of its next turn. A few player turns later you get news of what followed: whether its thinking mentioned the note, levels won, actions spent, and how often its actions did nothing before and after.
- send_function: a Python function (with a trial call) and a short message to one player: its code is shown to the player and the function is already defined in the player's Python. It does not use up a note: send one whenever code would save the player work.
- claim, release: say that you look after a game, or that you no longer do; list_games shows who holds each game, and a claim sends you the game's brief.
- news: what happened since you last looked (also appended automatically after your tool calls).
- write_handoff: your handoff to the next instance of you in your seat.

"""
SHOW_SIMILAR = ("- game brief (not a tool): when a game comes to you, assigned or claimed, the game arrives once in full: "
                "its current board as an image, every action so far, the last few actions with what each changed, the "
                "end of what the player wrote last, the levels it has won, and the other games most similar to it by the "
                "distinctive words in their players' thinking and code.")
SHOW_PLAIN = ("- game brief (not a tool): when a game comes to you, assigned or claimed, the game arrives once in full: "
              "its current board as an image, every action so far, the last few actions with what each changed, the end "
              "of what the player wrote last and the levels it has won.")

SHOW_FRAMED = ("game brief (not a tool): when a game comes to you, assigned or claimed, the game arrives once in full, "
               "with images: the board before and after each level the player won with the winning action, __N__ boards "
               "spread over the current level ending at the current one, the game's valid actions, every action so far, "
               "the last few actions with what each changed, the end of what the player wrote last, and the levels "
               "won. Read it before anything else on that game; later, view_board, last_generation and run_python "
               "show what changed.")


def _show_framed(frames: int) -> str:
    return SHOW_FRAMED.replace("__N__", str(frames))


def _similar_parts(text: str, similar: bool) -> str:
    return (text.replace("__SIMILAR__", SIMILAR_LINE + "\n\n" if similar else "")
            .replace("__SHOW__", SHOW_SIMILAR if similar else SHOW_PLAIN))


FLAT_SYSTEM = SYSTEM
SYSTEM = _similar_parts(FLAT_SYSTEM, True).replace("__SIGNED__", SIGNED_LINE).replace("__SENIOR__", SENIOR_LINE).replace("__THREADS__", THREAD_LINE) + DOCS


ANIMATION_LINE = ("One action, or a RESET, can play an animation of several boards before the board settles; "
                  "view_animation shows its boards and how long it plays. The games were made for people as well, "
                  "and an animation takes a person time to watch: a long one is a likely sign that it shows "
                  "something the player needs to win, so look at its boards.")


def _reset_animation() -> bool:
    return _env_flag("ARC3_RESET_ANIMATION")


def _game_meta_lines(game: Any) -> list[str]:
    if not _env_flag("ARC3_META_RAW"):
        return []
    try:
        raw = game.session.step_env({"query": "game_meta"})
        text = str(raw.get("text") or "") if isinstance(raw, dict) else ""
    except Exception:
        return []
    return [f"The arcade's metadata for this game, as its API returned it: {text}"] if text else []


def _animation_docs(text: str) -> str:
    if _reset_animation():
        return text.replace("__ANIMATION__", ANIMATION_LINE)
    return text.replace("- __ANIMATION__\n", "").replace(" __ANIMATION__", "")


def system_text(limit: int, handoff: int, similar: bool = True) -> str:
    text = SYSTEM if similar else (_similar_parts(FLAT_SYSTEM, False).replace("__SIGNED__", SIGNED_LINE)
                                   .replace("__SENIOR__", SENIOR_LINE).replace("__THREADS__", THREAD_LINE) + DOCS)
    return _animation_docs(text.replace("__LIMIT__", f"{limit:,}").replace("__HANDOFF__", f"{handoff:,}"))


def _now() -> float:
    return time.time()


LOG_STREAM = None
HOME_STDOUT = sys.stdout
LOG_LINES: "queue.Queue[str]" = queue.Queue(maxsize=max(1, int(os.environ.get("ARC3_FORUM_LOG_QUEUE") or 20000)))
LOG_DROPPED = [0]
LOG_START = threading.Lock()
LOG_THREAD: list[threading.Thread] = []


def _log_writer() -> None:
    while True:
        line = LOG_LINES.get()
        try:
            print(line, file=LOG_STREAM or HOME_STDOUT or sys.__stdout__, flush=True)
        except Exception:
            LOG_DROPPED[0] += 1


def _log(line: str) -> None:
    if not LOG_THREAD:
        with LOG_START:
            if not LOG_THREAD:
                LOG_THREAD.append(threading.Thread(target=_log_writer, name="forum-log", daemon=True))
                LOG_THREAD[0].start()
    try:
        LOG_LINES.put_nowait(line)
    except queue.Full:
        LOG_DROPPED[0] += 1


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


def _fit(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind(".\n"), head.rfind("\n"))
    kept = head[:end + 1].rstrip() if end >= limit // 2 else head.rstrip()
    return kept, f" (trimmed from {len(text)} to {len(kept)} characters: the limit is {limit}; put the essential first)"


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def near(one: str, two: str) -> float:
    matcher = difflib.SequenceMatcher(None, one, two, autojunk=False)
    if matcher.real_quick_ratio() < 0.5 or matcher.quick_ratio() < 0.5:
        return 0.0
    return matcher.ratio()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_flag_default(name: str, default: bool) -> bool:
    said = os.environ.get(name, "").strip().lower()
    return default if not said else said not in ("0", "false", "no", "off")


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


def _family(key: str) -> str:
    return key.rpartition("_p")[0] or key


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


def _tools(seat: str = "", similar: bool = True) -> list[dict[str, Any]]:
    game = {"type": "string", "description": "the game key exactly as list_games shows it"}
    return [
        _tool("list_games", "Every game: levels won, actions, tokens, minutes, state, notes waiting and delivered.", {}, []),
        _tool("last_generation", "The end of what one game's player wrote on its latest turn, as it was written. "
              "Read this before you send anything: it says what the player is working on right now.",
              {"game": game, "tokens": {"type": "integer", "description": "how much of the end to show, in tokens"}},
              ["game"]),
        _tool("read_turns", "Turns of one game, oldest first. Without from_turn, the latest turns.",
              {"game": game, "from_turn": {"type": "integer"}, "count": {"type": "integer", "description": "1-6"},
               "part": {"type": "string", "enum": sorted(PARTS)}}, ["game"]),
        _tool("read_replies", "The players' replies to help, oldest first, as a forum thread: each entry is what a helper "
              "sent (note or function) and what the player answered, both in full. game narrows it to one game; "
              "from_entry pages through a long thread.",
              {"game": {"type": "string", "description": "optional, one game only"},
               "from_entry": {"type": "integer"}}, []),
        _tool("view_animation", "The boards one action of a game passed through before it settled (its animation), as "
              "images, with which cells changed at each of them. step = that action's number; omit it for the latest "
              "animated action. Only a few animations of each game stay cached, the latest of each kind; the game "
              "brief lists every animated action and marks the cached ones."
              + (" The RESET that started the game and the latest RESET after a death are kept too (step 0 is the "
                 "start), and every animation says how long it plays." if _reset_animation() else ""),
              {"game": game, "step": {"type": "integer"}}, ["game"]),
        _tool("view_board", "One board of a game. step = the number of actions taken so far when the board was "
              "seen (0 = start); omit for the current board.",
              {"game": game, "step": {"type": "integer"}, "format": {"type": "string", "enum": ["text", "image", "both"]}},
              ["game"]),
        _tool("search", "Regular expression over every game's turns and the forum; up to 25 matching lines.",
              {"pattern": {"type": "string"}, "game": {"type": "string", "description": "optional, one game only"}},
              ["pattern"]),
        _tool("run_python", _frame_objects_text(
              "Run Python on one game's boards. Globals as the player has them: current_frame, "
              "previous_frame, history, transitions, last_transition, frame_diff(before, after), valid_actions, "
              "last_action_call_result, and the player's own kept functions; every frame has "
              ".ascii, .segmentation, .step, .level, .shape. action() is disabled. print() what you need."),
              {"game": game, "code": {"type": "string"}}, ["game", "code"]),
        _tool("forum_list", "Every forum thread: id, title, author, posts, last activity.", {}, []),
        _tool("forum_read", "Posts of one thread, from post number from_post (default 0).",
              {"thread": {"type": "integer"}, "from_post": {"type": "integer"}}, ["thread"]),
        _tool("forum_new_thread", "Open a thread with a first post.",
              {"title": {"type": "string"}, "text": {"type": "string"}}, ["title", "text"]),
        _tool("forum_post", "Post in a thread.", {"thread": {"type": "integer"}, "text": {"type": "string"}},
              ["thread", "text"]),
        _tool("send_note", "A note to one player, added to the message that opens its next turn. Short and checkable, "
              "at most __NOTE_CHARS__ characters.",
              {"game": game, "text": {"type": "string"}}, ["game", "text"]),
        _tool("claim", "Say that you look after one game; others see it in list_games, and the answer carries the game's brief. A game "
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


LIST_BY_ROLE = {"king": "Every game, one line each: levels won, actions, state, its prince and vassal.",
                "prince": "The games of your domain: levels won, actions, tokens, minutes, state, notes, vassal.",
                "vassal": "Your games: levels won, actions, tokens, minutes, state, notes."}
DIRECT_LIST = "Every game, one line each: levels won, actions, state, its vassal."
WALL_LIST_BY_ROLE = {"king": "Every game, one line each: levels won, actions, state, and whether it is at a wall and whose.",
                     "vassal": "The games of your walls: levels won, actions, tokens, minutes, state, notes."}
KIND_PROPERTY = {"type": "string", "description": "optional: your own short label for the kind of help this is, so "
                 "the king can measure which kinds of help work"}


def _list_text(role: str, direct: bool, walls: bool) -> str:
    if walls:
        return WALL_LIST_BY_ROLE[role]
    if direct and role == "king":
        return DIRECT_LIST
    return LIST_BY_ROLE[role]


def _court_tools(seat: str, role: str, direct: bool = False, walls: bool = False, similar: bool = True,
                 per: int = 2, turns: int = 5) -> list[dict[str, Any]]:
    tools = _tools(seat, similar)
    if walls:
        dropped = ("claim", "release") + (("send_note", "send_function") if role == "king" else ())
        tools = [tool for tool in tools if tool["function"]["name"] not in dropped]
    for tool in tools:
        function = tool["function"]
        if function["name"] == "list_games":
            function["description"] = _list_text(role, direct, walls)
        if function["name"] in ("send_note", "send_function"):
            function["parameters"]["properties"]["kind"] = KIND_PROPERTY
            if walls:
                function["description"] += (" Only to a game at a wall" + (" assigned to you" if role == "vassal" else "")
                                            + "; anything else is refused.")
    seat_ref = {"type": "string", "description": "the name or the seat number of the helper"}
    games = {"type": "array", "items": {"type": "string"}, "description": "game keys exactly as list_games shows them"}
    text = {"type": "string"}
    extra = []
    if role == "king" and walls:
        extra += [_tool("read_helper", "The end of what one vassal generated lately, as it was written: its "
                        "thinking, its text and the tools it called with their arguments, newest last. Read it to "
                        "see how a vassal works, the way last_generation shows a player.",
                        {"helper": seat_ref, "tokens": {"type": "integer", "description": "how much of the end, in tokens"}},
                        ["helper"])]
        extra += [_tool("assign", f"Give a vassal walls: games now at a wall, at most {per} per vassal. A wall belongs to "
                        "one vassal at a time; giving it to another moves it. With replace, the vassal's walls become "
                        "exactly these and the walls it loses have no vassal until you assign them.",
                        {"vassal": seat_ref, "games": games, "replace": {"type": "boolean"}}, ["vassal", "games"]),
                  _tool("order", "A standing order to one of your vassals: delivered at once and kept with its seat, "
                        "so its successor instances get it too.", {"vassal": seat_ref, "text": text}, ["vassal", "text"]),
                  _tool("metrics", "Whether help breaks walls: walls that got help against walls that got none, walls "
                        f"won within {turns} player turns after a help, per vassal and per kind of help.", {}, []),
                  _tool("edit_page", "Replace the text of your page (verified rules, how to break walls). Every new "
                        "seat reads it first, under the current walls.", {"text": text}, ["text"])]
    elif role == "king" and direct:
        extra += [_tool("assign", "Give a vassal games. A game belongs to one vassal at a time; giving it to another "
                        "moves it. With replace, the vassal's games become exactly these.",
                        {"vassal": seat_ref, "games": games, "replace": {"type": "boolean"}}, ["vassal", "games"]),
                  _tool("order", "A standing order to one of your vassals: delivered at once and kept with its seat, "
                        "so its successor instances get it too.", {"vassal": seat_ref, "text": text}, ["vassal", "text"]),
                  _tool("metrics", "The swarm's numbers and the effectiveness of help: per vassal, per kind of help, "
                        "and the levels won soon after a note or function.", {}, []),
                  _tool("edit_page", "Replace the text of your page (verified rules, how to help well). Every new seat "
                        "reads it first, under the current division of games.", {"text": text}, ["text"])]
    elif role == "king":
        extra += [_tool("assign", "Give a prince games for its domain. A game belongs to one prince at a time; giving "
                        "it to another moves it, and a vassal of the old prince loses it. With replace, the prince's "
                        "domain becomes exactly these games.",
                        {"prince": seat_ref, "games": games, "replace": {"type": "boolean"}}, ["prince", "games"]),
                  _tool("order", "A standing order to one of your princes: delivered at once and kept with its seat, "
                        "so its successor instances get it too.", {"prince": seat_ref, "text": text}, ["prince", "text"]),
                  _tool("metrics", "The swarm's numbers and the effectiveness of help: per prince with its vassals, "
                        "per kind of help, and the levels won soon after a note or function.", {}, []),
                  _tool("edit_page", "Replace the text of your page (verified rules, how to help well). Every new seat "
                        "reads it first, under the current domains.", {"text": text}, ["text"])]
    if role == "prince":
        extra += [_tool("assign", "Give one of your vassals games from your domain. With replace, its games become "
                        "exactly these.", {"vassal": seat_ref, "games": games, "replace": {"type": "boolean"}},
                        ["vassal", "games"]),
                  _tool("order", "A standing order to one of your vassals: delivered at once and kept with its seat.",
                        {"vassal": seat_ref, "text": text}, ["vassal", "text"])]
    if role in ("prince", "vassal"):
        lord = "king" if role == "prince" or direct or walls else "prince"
        said = ("what you sent, whether the player used it, whether the wall broke, with evidence." if walls
                else "what you did, what worked and what did not, with evidence.")
        extra.append(_tool("report", f"A message to your {lord}: {said}", {"text": text}, ["text"]))
    if walls:
        extra.append(_tool("walls", ("The live walls: game, level, the tokens and actions spent on that level, minutes "
                                     "since it became a wall, its vassal, the notes and functions delivered to it and "
                                     "whether the player used them; and the walls closed lately." if role == "king" else
                                     "Your walls: game, level, the tokens and actions spent on that level, minutes since "
                                     "it became a wall, the notes and functions delivered to it and whether the player "
                                     "used them."), {}, []))
    extra.append(_tool("read_page", ("The king's page: the current walls with their vassals and the rules the king has "
                                     "verified." if walls else
                                     "The king's page: the current domains and the rules the king has verified."), {}, []))
    at = next(index for index, tool in enumerate(tools) if tool["function"]["name"] == "news")
    tools[at]["function"]["description"] = ("What happened in your part of the court since you last looked: alerts, "
                                            "orders, reports, outcomes of help, and the numbers that changed.")
    tools = tools[:at] + extra + tools[at:]
    if walls:
        gone = ("news", "read_turns", "read_page") if role == "king" else ("news",)
        tools = [tool for tool in tools if tool["function"]["name"] not in gone]
    return tools


FORK_QUESTION = ("This is your whole game so far: the boards above run from the start of the game to now, each after "
                 "the action named above it, with every level you won and every death included, and after them the "
                 "animations of your wins and deaths. Look at the whole of it. Is there something you are missing, "
                 "something you have forgotten, something you should do and have not? Answer in plain text, in this "
                 "order. First, what the boards show for certain, each fact with the action number of the board that "
                 "shows it. Then what you have not tried yet or have not looked at. Last, the one next thing to do: "
                 "the exact actions, each one of this game's valid actions, and which board would tell you it worked "
                 "and which that it failed. You take no action in this reply.")
WALL_WORTH = ("The players are competent and read their own frames: send what they lack, a reading of their frames "
              "they missed, an untested hypothesis their evidence allows, a fresh look when they loop. Work fast: your "
              "first note on a wall goes out within three turns (one fact you measured, with its action number, and "
              "one test of one to six actions), then deepen and send more. Read the player's last generation and "
              "actions first and never propose what it already did. Mark each claim measured, with its action "
              "number, or a hypothesis. Answer a player's question within a turn or two, even partly; read its "
              "replies (read_replies) and drop an idea its evidence refutes. Send code (send_function) when it does "
              "work the player has not done: a solver, a check of its hypothesis over every transition, a simulator. "
              "Never suggest an action the game does not list.")
ROLE_TEXT = {
    "king": ("Your role: king. You answer for the swarm's overall effectiveness: levels won across all games with few "
             "actions, and the share of the players' GPU the helpers spend. You assign domains (games) to your princes "
             "with assign, give them orders with order, and check their effectiveness with metrics; every report a "
             "prince sends reaches you. Above all, look for the best way to make the helpers effective, not only for a "
             "division of games: study which kinds of help were followed by levels (for example a mechanic verified in "
             "a solved similar game, budget and clock facts, functions, general habits), which were ignored or followed "
             "by worse play, and which only spent the shared GPU; turn what you learn into orders, and keep your page "
             "(edit_page) current with the rules you have verified, since every new seat reads it first. You keep "
             "every reading tool and may help games yourself. " + WALL_WORTH),
    "prince": ("Your role: prince. You carry out the king's orders and report to the king with report: what you did, "
               "what worked and what did not, with evidence. You hold a domain of games (list_games shows it). Help "
               "those games yourself, give your vassals games from your domain with assign and orders with order, and "
               "read their reports in your news. Vassals are created as games finish, so you may start with none. " + WALL_WORTH),
    "vassal": ("Your role: vassal. You carry out your prince's orders and report to your prince with report: what you "
               "did, what worked and what did not, with evidence. You look after the games your prince assigned you "
               "(list_games shows them); help those games, and you may read any game. " + WALL_WORTH),
}
DIRECT_KING = ("Your role: king. You answer for the swarm's overall effectiveness: levels won across all games with few "
               "actions, and the share of the players' GPU the helpers spend. You command your vassals directly: you "
               "give them games with assign and orders with order, and check their effectiveness with metrics; every "
               "report a vassal sends reaches you. Above all, look for the best way to make the helpers effective, not "
               "only for a division of games: study which kinds of help were followed by levels (for example a mechanic "
               "verified in another game, budget and clock facts, functions, general habits), which were ignored or "
               "followed by worse play, and which only spent the shared GPU; turn what you learn into orders, and keep "
               "your page (edit_page) current with the rules you have verified, since every new seat reads it first. "
               "You keep every reading tool and may help games yourself. " + WALL_WORTH)
WALL_KING = ("Your role: king. The swarm's job is to break walls, the levels the players are stuck on, so that every "
             "game is won with few player actions. The tool walls lists every live wall: its game and level, the tokens "
             "and actions its player spent on that level, the minutes since it became a wall, its vassal, and the help "
             "delivered to it. The harness hands walls out; you can move them with assign, at most __PER__ per vassal, "
             "and give vassals orders with order. The vassals write to the players; you do not. Every report a vassal "
             "sends reaches you, and read_helper, read_replies and metrics show how they work. Your page (edit_page) "
             "is what every new seat reads first. The vassals are told to work fast, send what they have early and "
             "deepen it after. How you run the swarm is yours to decide. "
             "The forum score at the top of metrics is how people judge the swarm: it rises when a wall's level is "
             "won within 10 player turns after a help, more the sooner, more for a level that had cost the player "
             "more, twice for a function the player's code called; it falls for every 10 minutes a wall waits for its "
             "first help and for every 10 minutes a wall stands unbroken.")
WALL_VASSAL = ("Your role: vassal. You work only on your walls (walls lists them): find why the player has not won "
               "its level, check your idea against that game's evidence, and send the player what will win it. Carry "
               "out the king's orders and report to him (report): what you sent, whether it was used, whether the "
               "wall broke. You keep a wall until its level is won or its game ends; if it has stood 40 minutes and "
               "you are out of ideas, say so in a report. Between notes keep working on your wall. " + WALL_WORTH)
ROLE_TOOLS = {
    "king": ("- assign: give a prince games for its domain.\n- order: a standing order to one prince.\n"
             "- metrics: the swarm's numbers and the effectiveness of help per prince and per kind of help.\n"
             "- edit_page: replace the text of your page.\n"),
    "prince": ("- assign: give one of your vassals games from your domain.\n- order: a standing order to one of your "
               "vassals.\n- report: a message to the king.\n"),
    "vassal": "- report: a message to your prince.\n",
}
DIRECT_TOOLS = {
    "king": ("- assign: give a vassal games.\n- order: a standing order to one vassal.\n"
             "- metrics: the swarm's numbers and the effectiveness of help per vassal and per kind of help.\n"
             "- edit_page: replace the text of your page.\n"),
    "vassal": "- report: a message to your king.\n",
}
WALL_TOOLS = {
    "king": ("- assign: give a vassal walls, at most __PER__ each.\n- order: a standing order to one vassal.\n"
             "- read_helper: the end of what one vassal generated lately: thinking, text and tool calls.\n"
             "- metrics: whether help breaks walls, per vassal and per kind of help.\n"
             "- walls: every live wall with its vassal and the help it got.\n"
             "- edit_page: replace the text of your page.\n"),
    "vassal": "- report: a message to your king.\n- walls: your walls with the help they got.\n",
}
COURT_LINE = "The helpers are organized as a court: one king, princes under the king, and vassals under each prince."
DIRECT_LINE = "The helpers are organized as a court: one king and the vassals under the king."
GROW_LINE = ("Every game that finishes and leaves its lane creates one new helper: a vassal for the prince with the "
             "fewest vassals, at most __VASSALS__ per prince, at most __SEATS__ helpers in all.")
DIRECT_GROW_LINE = ("Every game that finishes and leaves its lane creates one new helper: a vassal for the king, at most "
                    "__SEATS__ helpers in all.")
FIXED_LINE = "The number of helpers is fixed at __SEATS__: a game that finishes leaves its lane to the players."

COURT_SYSTEM = """You are one of the helper agents in a shared workspace during a live run of puzzle games. The players are other instances of the same model, each alone on its game. Your goal: as many levels as possible get won across all games, with as few actions as possible.

__COURT__ Your role, your superior, your games and your standing orders belong to your seat and are given to you below; a successor instance in your seat keeps them. __GROWTH__ Bad or confident-but-wrong advice to a player costs it actions; a note that is checked against the evidence in its own game is worth much more than a guess.

__ROLE__

__SIMILAR____SIGNED__

__THREADS__

Tools:
__SHOW__
- list_games: __LIST__
- last_generation: the end of what a game's player wrote on its latest turn, as it was written.
- read_turns: a game's turns (the player's thinking, code, code output, the prompt it got).
- view_board: one board of a game, as text or as an image.
- view_animation: the boards one action passed through before it settled, as images, with what changed at each.
- read_replies: the players' replies to help as a thread, each as what the helper said and what the player answered, in full.
- search: a regular expression over every game's turns and the forum.
- run_python: Python on a game's boards with the player's own frame API; action() is disabled.
- forum_list, forum_read, forum_post, forum_new_thread: the forum.
- send_note: a short note to one player, delivered at the start of its next turn, with an optional label for the kind of help. A few player turns later you get news of what followed: whether its thinking mentioned the note, levels won, actions spent, and how often its actions did nothing before and after.
- send_function: a Python function (with a trial call) and a short message to one player: its code is shown to the player and the function is already defined in the player's Python. It does not use up a note: send one whenever code would save the player work.
- claim, release: say that you look after a game, or that you no longer do; list_games shows who holds each game, and a claim sends you the game's brief.
__ROLE_TOOLS__- read_page: the king's page, with the current domains and the rules the king has verified.
- news: what happened in your part of the court since you last looked, also appended after your tool calls: alerts, orders and reports at once, the rest every few calls, numbers only when they change.
- write_handoff: your handoff to the next instance of you in your seat.

"""

WALLS_SYSTEM = """You are a helper agent during a live run of puzzle games; each game is played by one player, another instance of your model, alone. The helpers' job is to break walls. A wall is a game given to a vassal: from the first minute the harness hands out the games that have spent the most on their current level, and a game whose player asked for help, or that has spent __WALL__ on one level, goes first to the next free vassal. A vassal keeps its wall until its level is won or its game ends. Games no vassal holds get nothing from the helpers.__ASKS__

__COURT__ __GROWTH__ Your seat keeps your role, walls and orders across handoffs.

__ROLE__

__SIMILAR__The harness signs your posts and notes with your name and seat; do not sign them yourself. Start a new thread, titled by its topic, for each new topic rather than appending to a long one.

__WALLSHOW__

"""


ASKS_LINE = ("A player that asks for help (Advisors.Ask) puts its game at a wall at once, and its question reaches the "
             "wall's vassal and the king. Every player also answers each note or function it receives with short "
             "feedback (Advisors.Reply), which reaches the helper who sent it.")


def wall_rule(tokens: int, actions: int) -> str:
    parts = ([f"at least {tokens:,} tokens"] if tokens > 0 else []) + ([f"at least {actions} actions"] if actions > 0 else [])
    return " or ".join(parts) or "a long time"


def court_system_text(limit: int, handoff: int, role: str, vassals: int, seats: int, direct: bool = False,
                      walls: bool = False, grow: bool = True, similar: bool = True, per: int = 2, turns: int = 5,
                      wall_tokens: int = 0, wall_actions: int = 0, player_api: bool = False) -> str:
    if walls:
        role_text = {"king": WALL_KING, "vassal": WALL_VASSAL}[role]
        role_tools = WALL_TOOLS[role]
    elif direct:
        role_text = DIRECT_KING if role == "king" else ROLE_TEXT["vassal"].replace("prince", "king")
        role_tools = DIRECT_TOOLS[role]
    else:
        role_text = ROLE_TEXT[role] if similar else ROLE_TEXT[role].replace("a solved similar game", "another game")
        role_tools = ROLE_TOOLS[role]
    growth = (DIRECT_GROW_LINE if direct or walls else GROW_LINE) if grow else FIXED_LINE
    text = (_similar_parts(WALLS_SYSTEM if walls else COURT_SYSTEM, similar)
            .replace("__COURT__", DIRECT_LINE if direct or walls else COURT_LINE).replace("__GROWTH__", growth)
            .replace("__ROLE__", role_text).replace("__ROLE_TOOLS__", role_tools)
            .replace("__LIST__", _list_text(role, direct, walls)).replace("__WALL__", wall_rule(wall_tokens, wall_actions))
            .replace("__WALLSHOW__", WALLS_SHOW).replace("__SIGNED__", SIGNED_LINE).replace("__THREADS__", THREAD_LINE)
            .replace("__ASKS__", " " + ASKS_LINE if player_api else "")
            .replace("__PER__", str(per)).replace("__TURNS__", str(turns))
            .replace("__VASSALS__", str(vassals)).replace("__SEATS__", str(seats)))
    if walls and role == "king":
        text = "\n".join(line for line in text.split("\n")
                         if not line.startswith(("- send_note:", "- send_function:")))
    docs = WALLS_DOCS if walls else DOCS
    return _animation_docs(text + docs.replace("__LIMIT__", f"{limit:,}").replace("__HANDOFF__", f"{handoff:,}"))


ROLE_SIGN_RE = re.compile(r"^[\s\-—–~*_>]*(?:(?:signed|from|by)\s*[:,\-]?\s*)?(%s)(?:'s)?\s*[(\[]\s*(?:king|prince|vassal)\b"
                          r"[^)\]\n]{0,40}[)\]][\s.!:;*_]*$" % NAME_ALT, re.IGNORECASE)


def strip_role_signature(text: str) -> tuple[str, list[str]]:
    lines = str(text).strip().splitlines()
    dropped: list[str] = []
    while lines and ROLE_SIGN_RE.match(lines[-1]) and len(lines[-1]) <= 80:
        dropped.append(lines.pop().strip())
        while lines and not lines[-1].strip():
            lines.pop()
    while lines and ROLE_SIGN_RE.match(lines[0]) and len(lines[0]) <= 80:
        dropped.append(lines.pop(0).strip())
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip(), dropped


def image_tokens(url: str) -> int:
    marker = "base64,"
    index = url.find(marker)
    try:
        raw = base64.b64decode(url[index + len(marker):index + len(marker) + 32], validate=True)
    except Exception:
        return 0
    if index < 0 or len(raw) < 24 or raw[12:16] != b"IHDR":
        return 0
    width, height = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return -(-width // 32) * -(-height // 32) + 2


def _keys(keys: list[str], shown: int = 4) -> str:
    keys = list(keys)
    return ", ".join(keys[:shown]) + (f", +{len(keys) - shown}" if len(keys) > shown else "")


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
        self.turns: list[dict[str, Any]] = []
        self.parsed_bytes = 0
        self.partial = ""
        self.parse_lock = threading.Lock()
        self.animation_lock = threading.Lock()
        self.animations: list[dict[str, Any]] = []
        self.animation_log: list[tuple[int, str, int, str]] = []
        self.pinned_animations: list[dict[str, Any]] = []
        self.reset_animations: list[dict[str, Any]] = []
        self.history_seen = 0
        self.fork_busy = False
        self.fork_t = 0.0
        self.forks_done = 0
        self.fork_waiting: list[dict[str, Any]] = []
        self.fork_calls: list[int] = []
        self.animation_limit = 0
        self.animation_errors = 0
        self.cloud: Counter[str] = Counter()
        self.cloud_chars = 0
        self.freed = False
        self.calls = 0
        self.pending: list[dict[str, Any]] = []
        self.level_calls: list[int] = []
        self.turn_levels = 0
        self.stuck_mark = 0
        self.wall: int | None = None
        self.last_batch: list[dict[str, Any]] = []
        self.asks: list[dict[str, Any]] = []

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

    def keep_animation(self, record: dict[str, Any]) -> None:
        try:
            chain = list(record.get("chain") or ())
            if len(chain) <= 2 or self.animation_limit <= 0:
                return
            frames = [(len(grid), len(grid[0]) if grid else 0,
                       zlib.compress(bytes(int(cell) & 0xFF for row in grid for cell in row))) for grid in chain]
            entry = {"step": int(record.get("step") or 0), "level": record.get("level"),
                     "action": str(record.get("action_display") or "?"), "frames": frames}
            if _reset_animation():
                from inference.utils.animation import animation_lasted
                entry.update({"lasted": animation_lasted(record.get("summary")), "cause": record.get("cause"),
                              "before": record.get("before")})
            kind = (entry["action"].split("(")[0], len(frames))
            with self.animation_lock:
                if self.animation_log and self.animation_log[-1][0] == entry["step"]:
                    self.animation_log.pop()
                self.animation_log.append((entry["step"], entry["action"], len(frames) - 1, entry.get("lasted") or ""))
                del self.animation_log[:-500]
                if entry.get("cause") is not None:
                    start = [one for one in self.reset_animations if one.get("cause") == "start"]
                    self.reset_animations = (start if entry["cause"] != "start" else []) + [entry]
                    return
                kept = [one for one in self.animations if one["step"] != entry["step"]]
                same = [one for one in kept if (one["action"].split("(")[0], len(one["frames"])) == kind]
                if same:
                    kept.remove(same[0])
                elif len(kept) >= self.animation_limit:
                    kept.pop(0)
                self.animations = kept + [entry]
        except Exception:
            self.animation_errors += 1

    def pin_animations(self, limit: int = 8) -> None:
        try:
            entries = list(self.session.history_entries)
        except Exception:
            return
        start, self.history_seen = self.history_seen, len(entries)
        for entry in entries[start:]:
            result = dict(getattr(entry, "result", {}) or {})
            if not (result.get("level_completed") or result.get("game_over")):
                continue
            step = int(getattr(getattr(entry, "frame", None), "step", -1))
            with self.animation_lock:
                found = next((one for one in self.animations if one["step"] == step), None)
                if found is None or any(one["step"] == step for one in self.pinned_animations):
                    continue
                pinned = dict(found)
                pinned["why"] = "level won" if result.get("level_completed") else "death"
                self.pinned_animations = (self.pinned_animations + [pinned])[-limit:]

    def animation_steps(self) -> list[int]:
        with self.animation_lock:
            return sorted({one["step"] for one in self.animations + self.pinned_animations + self.reset_animations})

    def animation_summary(self, limit: int = 40) -> str:
        with self.animation_lock:
            log = list(self.animation_log)
            cached = {one["step"] for one in self.animations + self.pinned_animations + self.reset_animations}
        if not log:
            return ""
        shown = log[-limit:]
        marks: dict[int, str] = {}
        try:
            for entry in self.frames()[1]:
                result = dict(getattr(entry, "result", {}) or {})
                if getattr(entry, "action", "") and (result.get("level_completed") or result.get("game_over")):
                    marks[int(getattr(entry.frame, "step", -1))] = (
                        "LEVEL WON" if result.get("level_completed") else "death, level restarted")
        except Exception:
            marks = {}
        lines = [f"step {step} {action}: {count} boards" + (f" over {lasted}" if lasted else "")
                 + (f", {marks[step]}" if step in marks else "")
                 + (" [cached, view_animation]" if step in cached else "")
                 for step, action, count, lasted in shown]
        more = f" (the latest {len(shown)} of {len(log)})" if len(log) > len(shown) else ""
        return (f"Animated actions{more}; only those marked cached can still be viewed:" + chr(10)
                + chr(10).join(f"  {line}" for line in lines))

    def animation(self, step: int | None) -> dict[str, Any] | None:
        with self.animation_lock:
            found = [one for one in self.pinned_animations + self.animations + self.reset_animations
                     if step is None or one["step"] == step]
            if step is None:
                found = sorted(found, key=lambda one: one["step"])
        if not found:
            return None
        entry = found[-1]
        chain = []
        for rows, cols, packed in entry["frames"]:
            flat = zlib.decompress(packed)
            chain.append(tuple(tuple(flat[row * cols:(row + 1) * cols]) for row in range(rows)))
        return {"step": entry["step"], "level": entry["level"], "action_display": entry["action"], "chain": chain,
                **{key: entry[key] for key in ("lasted", "cause", "before") if entry.get(key)}}

    def generation_tail(self, tokens: int) -> str:
        turns = self.parse()
        if not turns:
            return ""
        limit = max(1, tokens) * 4
        for turn in reversed(turns):
            parts = []
            for label, extra, lines in turn["sections"]:
                kind = label.split(":")[0].strip()
                if kind not in CLOUD_KINDS:
                    continue
                body = "\n".join(lines).strip()
                if body:
                    parts.append(f"[{kind}{': ' + extra if extra else ''}]\n{body}")
            body = "\n".join(parts)
            last = turn["sections"][-1][0].split(":")[0].strip() if turn["sections"] else ""
            if turn is turns[-1] and last in CLOUD_KINDS and self.partial.strip():
                body = (body + "\n" + self.partial).strip()
            if not body:
                continue
            head = f"(turn {turn['turn']}, after {turn['action']} actions"
            head += ", the latest one recorded and possibly still growing)" if turn is turns[-1] else ")"
            if len(body) > limit:
                body = "...(earlier part left out)\n" + body[-limit:]
            return head + "\n" + body
        return ""

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

    def player(self) -> Any:
        return getattr(self.session, "analyzer", None)

    def valid_actions(self) -> list[str]:
        try:
            return [str(one) for one in getattr(self.player(), "_current_valid_actions", None) or []]
        except Exception:
            return []

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
            self.parsed_bytes = size
            text = self.partial + chunk.decode("utf-8", errors="replace")
            lines = text.split("\n")
            self.partial = lines.pop()
            for line in lines:
                found = TURN_RE.match(line)
                if found:
                    self.turns.append({"turn": int(found.group(1)), "action": int(found.group(2)),
                                       "clock": found.group(3), "sections": []})
                    continue
                if not self.turns:
                    continue
                head = SECTION_RE.match(line)
                sections = self.turns[-1]["sections"]
                if head:
                    sections.append([head.group(1).strip(), head.group(2) or "", []])
                elif sections:
                    sections[-1][2].append(line)
                    if sections[-1][0] in CLOUD_KINDS:
                        self.cloud.update(words(line))
                        self.cloud_chars += len(line) + 1
            return self.turns

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
        self.context_tokens = _env_int("ARC3_FORUM_CONTEXT_TOKENS", 140000)
        self.handoff_at = _env_int("ARC3_FORUM_HANDOFF_AT", 126000)
        self.max_output = _env_int("ARC3_FORUM_MAX_OUTPUT", 10000)
        self.animations_kept = max(0, _env_int("ARC3_FORUM_ANIMATIONS", 3))
        self.grown_live = max(0, _env_int("ARC3_FORUM_GROWN_CONTEXT_LIVE", 0))
        self.grown_window = max(8192, _env_int("ARC3_FORUM_GROWN_CONTEXT_WINDOW", 200000))
        self.grown_keep = max(4096, min(_env_int("ARC3_FORUM_GROWN_CONTEXT_KEEP", 57344), self.grown_window // 2))
        self.grown_since: float | None = None
        self.animation_frames = max(2, _env_int("ARC3_FORUM_ANIMATION_FRAMES", 8))
        self.read_chars = _env_int("ARC3_FORUM_READ_CHARS", 12000)
        self.arrival_s = _env_float("ARC3_FORUM_ARRIVAL_S", 4.0)
        self.upscale = _env_int("ARC3_FORUM_IMAGE_UPSCALE", 7)
        self.api_key = os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        self.sessions = os.environ.get("ARC3_FORUM_SESSIONS", "1").strip().lower() not in ("", "0", "false", "off")
        self.grow = os.environ.get("ARC3_FORUM_GROW_ON_END", "0").strip().lower() not in ("", "0", "false", "off")
        self.max_helpers = _env_int("ARC3_FORUM_MAX_HELPERS", 0)
        self.grown = 0
        self.claims: dict[str, int] = {}
        self.briefs: dict[int, list[str]] = {}
        self.reply_log: list[dict[str, Any]] = []
        self.helper_objects: dict[int, Any] = {}
        self.recent_levels: list[tuple[float, str, int]] = []
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
        self.walls_mode = _env_flag("ARC3_FORUM_WALLS")
        self.hier = _env_flag("ARC3_FORUM_HIERARCHY") or self.walls_mode
        self.vassals_per = max(0, _env_int("ARC3_FORUM_VASSALS", 3))
        self.princes_asked = max(0, _env_int("ARC3_FORUM_PRINCES", max(0, helpers - 1)))
        self.princes = min(self.princes_asked, max(0, helpers - 1)) if self.hier and not self.walls_mode else 0
        self.start_vassals = (max(0, _env_int("ARC3_FORUM_START_VASSALS", max(0, helpers - 1 - self.princes)))
                              if self.hier else 0)
        if self.hier and helpers > 0:
            self.count = 1 + self.princes + self.start_vassals
        self.wall_tokens = _env_int("ARC3_FORUM_WALL_TOKENS", 40000)
        self.wall_actions = _env_int("ARC3_FORUM_WALL_ACTIONS", 150)
        self.walls_per = max(1, _env_int("ARC3_FORUM_WALLS_PER_VASSAL", 2))
        self.note_chars = max(200, _env_int("ARC3_FORUM_NOTE_CHARS", 1500))
        self.wall_wait_s = max(0, _env_int("ARC3_FORUM_WALL_WAIT_S", 60))
        self.forks_max = max(0, _env_int("ARC3_FORUM_FORKS", 0))
        self.fork_frames = max(2, _env_int("ARC3_FORUM_FORK_FRAMES", 100))
        self.fork_upscale = max(1, _env_int("ARC3_FORUM_FORK_UPSCALE", 6))
        self.fork_anim_frames = max(0, _env_int("ARC3_FORUM_FORK_ANIMATION_FRAMES", 6))
        self.fork_gap_s = max(0, _env_int("ARC3_FORUM_FORK_GAP_S", 1200))
        self.fork_tokens = max(256, _env_int("ARC3_FORUM_FORK_MAX_TOKENS", 20000))
        self.fork_reply_wait_s = max(0, _env_int("ARC3_FORUM_FORK_REPLY_WAIT_S", 900))
        self.fork_hold_s = _env_float("ARC3_FORUM_FORK_HOLD_S", 3.0)
        self.min_helpers = max(1, _env_int("ARC3_FORUM_MIN_HELPERS", 5))
        self.follow_games = _env_int("ARC3_FORUM_HELPERS_FOLLOW_GAMES", 1) > 0
        self.forks_running = 0
        self.score_every_s = max(0, _env_int("ARC3_FORUM_SCORE_EVERY_S", 300))
        self.score_turns = max(1, _env_int("ARC3_FORUM_SCORE_TURNS", 10))
        self.score_half = max(1, _env_int("ARC3_FORUM_SCORE_HALF_TURNS", 3))
        self.score_read = max(1, _env_int("ARC3_FORUM_SCORE_READ_TOKENS", 100000))
        self.page_every_s = max(0, _env_int("ARC3_FORUM_PAGE_EVERY_S", 600))
        self.wall_fill = _env_flag_default("ARC3_FORUM_WALL_FILL", True)
        self.wall_fill_tokens = max(0, _env_int("ARC3_FORUM_WALL_FILL_MIN_TOKENS", 0))
        self.fork_min_tokens = max(0, _env_int("ARC3_FORUM_FORK_MIN_TOKENS", 15000))
        self.walls: list[dict[str, Any]] = []
        self.domain: dict[str, int] = {}
        self.fief: dict[str, int] = {}
        self.page = {"text": "", "t": 0.0, "edits": 0}
        self.helps: list[dict[str, Any]] = []
        self.effects: list[str] = []
        self.level_times: list[float] = []
        self.registered = 0
        self.deferred: list[tuple[str, dict[str, Any]]] = []
        self.spawned = False
        self.budget_warned = False
        self.worse = 0
        self.stuck_actions = _env_int("ARC3_FORUM_STUCK_ACTIONS", 150)
        self.budget_alert_min = _env_float("ARC3_FORUM_BUDGET_ALERT_MIN", 10.0)
        self.news_every = max(1, _env_int("ARC3_FORUM_NEWS_EVERY", 3))
        self.news_lines = max(1, _env_int("ARC3_FORUM_NEWS_LINES", 8))
        self.effect_turns = max(1, _env_int("ARC3_FORUM_EFFECT_TURNS", 5))
        self.show_frames = max(0, _env_int("ARC3_FORUM_SHOW_FRAMES", 0))
        self.show_generation = max(0, _env_int("ARC3_FORUM_SHOW_GENERATION", 4000))
        self.last_generation = max(0, _env_int("ARC3_FORUM_LAST_GENERATION", 2000))
        self.player_api = _env_flag("ARC3_FORUM_PLAYER_API")
        self.answers_chars = 600

    def show_upscale(self) -> int:
        override = _env_int("ARC3_FORUM_SHOW_UPSCALE", 0)
        if override > 0:
            return override
        try:
            from inference.agent.vision_context import current_grid_image_upscale
            return int(current_grid_image_upscale())
        except Exception:
            return self.upscale

    def minute(self) -> str:
        done = (_now() - self.started) / 60.0
        left = max(0.0, (self.end_at - _now()) / 60.0)
        return f"minute {done:.0f} since the forum opened, {left:.0f} minutes of play left in this run"

    def label(self, member_id: int) -> str:
        with self.lock:
            member = self.members[member_id] if 0 <= member_id < len(self.members) else None
        if member is None:
            return f"seat {member_id}"
        if self.hier:
            return f"{member['name']} ({member['role']}, seat {member['id']})"
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
        _log(f"forum [{(_now() - self.started) / 60:.1f} min]: {text}")

    def brief(self, member: int, game: str) -> None:
        with self.lock:
            waiting = self.briefs.setdefault(member, [])
            if game not in waiting:
                waiting.append(game)

    def take_briefs(self, member: int) -> list[str]:
        with self.lock:
            return self.briefs.pop(member, [])

    def add_news(self, text: str, audience: set[int] | None = None, **meta: Any) -> None:
        with self.lock:
            self.news.append({"seq": len(self.news), "t": _now(), "text": text, "audience": audience, **meta})

    def add_game(self, session: Any, key: str) -> None:
        game = _Game(session, key)
        game.animation_limit = self.animations_kept
        if self.animations_kept:
            try:
                session.animation_hook = game.keep_animation
            except Exception as exc:
                self.say(f"{key}: animations are not kept ({type(exc).__name__})")
            started = getattr(session, "reset_animation_record", None)
            if isinstance(started, dict):
                game.keep_animation(started)
        prince = None
        with self.lock:
            self.games[game.key] = game
            if self.hier and self.princes > 0:
                prince = 1 + self.registered % self.princes
                self.domain[game.key] = prince
            self.registered += 1
        self.record("game", game=game.key, **({"prince": prince} if prince is not None else {}))

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
        last_stats_said = 0.0
        last_score = _now()
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
                self.poll(game)
            if self.walls_mode and self.wall_fill:
                self.fill_walls()
            if self.forks_max:
                self.schedule_forks()
            if self.hier:
                self.check_budget()
            if self.walls_mode and self.score_every_s and _now() - last_score >= self.score_every_s:
                last_score = _now()
                try:
                    score, lines = self.forum_score()
                    self.say("FORUM_SCORE " + f"{score:+.2f}" + chr(10) + chr(10).join(lines))
                    self.record("score", score=round(score, 2), lines=lines)
                except Exception as exc:
                    self.say(f"FORUM_SCORE failed: {type(exc).__name__}: {exc}")
            if _now() - last_line >= 60.0:
                last_line = _now()
                with self.lock:
                    alive = sum(1 for one in self.members if one.get("alive"))
                    stats = dict(self.stats)
                if _now() - last_stats_said >= 300.0:
                    last_stats_said = _now()
                    self.say(f"{alive} helpers alive ({self.count} at start + {self.grown} on freed lanes), "
                             f"similar-games index {len(self.similar.vectors)} games in {self.similar.seconds:.2f}s, "
                             f"{len(self.threads)} threads, {stats['posts']} posts, {stats['notes']} "
                             f"notes ({stats['delivered']} delivered); {stats['requests']} requests, {stats['errors']} "
                             f"errors, prompt {stats['prompt'] / 1e6:.2f}M ({stats['cached'] / 1e6:.2f}M cached), "
                             f"generated {stats['generated'] / 1e6:.2f}M; reads {stats['reads']} "
                             f"({stats['read_chars'] / 1e6:.2f}M chars), python {stats['python']}, images {stats['images']}; "
                             f"forum log lines dropped {LOG_DROPPED[0]}")
                self.record("stats", alive=alive, grown=self.grown, similar_s=round(self.similar.seconds, 3),
                            log_dropped=LOG_DROPPED[0], **stats)

    def poll(self, game: _Game) -> None:
        try:
            status = game.status()
        except Exception:
            return
        if status["levels"] > game.levels_seen:
            tokens = status["tokens"] - game.level_tokens
            actions = status["actions"] - game.level_actions
            minutes = (time.monotonic() - game.level_time) / 60.0
            text = (f"{game.key} won level {status['levels']} of {status['total']}: {actions} actions and "
                    f"{tokens / 1000:.1f}k tokens on that level, {minutes:.1f} min")
            self.add_news(text, kind="level", game=game.key)
            with self.lock:
                self.recent_levels.append((_now(), game.key, status["levels"]))
                del self.recent_levels[:-50]
            if self.walls_mode:
                self.end_wall(game, status)
            if self.hier:
                with self.lock:
                    self.level_times += [_now()] * (status["levels"] - game.levels_seen)
                game.stuck_mark = 0
            self.record("level", game=game.key, level=status["levels"], actions=actions, tokens=tokens)
            game.levels_seen = status["levels"]
            game.level_tokens = status["tokens"]
            game.level_actions = status["actions"]
            game.level_time = time.monotonic()
        if not game.ended and status["state"] not in ("playing", "?"):
            self.mark_ended(game, status)
        game.pin_animations()
        if self.walls_mode:
            self.check_wall(game, status)
        elif self.hier:
            self.check_stuck(game, status)

    def mark_ended(self, game: _Game, status: dict[str, Any]) -> None:
        with self.lock:
            if game.ended:
                return
            game.ended = True
        text = f"{game.key} ended ({status['state']}) with {status['levels']} of {status['total']} levels"
        self.add_news(text, kind="ended", game=game.key)
        self.record("ended", game=game.key, state=status["state"], levels=status["levels"])
        if self.walls_mode:
            self.end_wall(game, status)

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
        self.trim_helpers()
        if self.hier:
            if self.grow:
                self.grow_vassal(key, status)
            return
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

    def player_context(self) -> tuple[int, int] | None:
        if self.grown_live <= 0:
            return None
        if self.grown_since is None:
            with self.lock:
                live = sum(1 for game in self.games.values() if not game.freed)
                if not self.games or live > self.grown_live:
                    return None
                self.grown_since = _now()
            self.say(f"{live} games still play, at most {self.grown_live}: every player's context now grows to "
                     f"{self.grown_window} tokens before a trim, and a trim keeps about {self.grown_keep}")
            self.record("grown_context", live=live, window=self.grown_window, keep=self.grown_keep)
        return self.grown_window, self.grown_keep

    def deliver(self, key: str, message: dict[str, Any]) -> dict[str, Any]:
        game = self.games.get(key)
        if game is None:
            return message
        with self.lock:
            game.calls += 1
            notes, game.notes_waiting = game.notes_waiting, []
            forks, game.fork_waiting = game.fork_waiting, []
            stale = [one for one in forks if one.get("mark", game.level_tokens) != game.level_tokens]
            forks = [one for one in forks if one not in stale]
        for one in stale:
            self.record("fork_stale", game=game.key, calls=game.calls - one["calls"])
        with self.lock:
            game.notes_delivered += len(notes)
            self.stats["delivered"] += len(notes)
        if self.hier:
            self.turn_levels(game)
        self.settle(game, False)
        if forks:
            block = self.fork_block(game, forks)
            content = message.get("content")
            if isinstance(content, list):
                message = {**message, "content": [{"type": "text", "text": block}, *content]}
            else:
                message = {**message, "content": block + ("\n\n" + content if content else "")}
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
        block = NOTE_HEADER + "\n" + "\n".join(lines) + ("\n" + NOTE_REPLY_LINE if self.player_api else "")
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": [{"type": "text", "text": block}, *content]}
        else:
            message = {**message, "content": block + ("\n\n" + content if content else "")}
        status = game.status()
        parsed = len(game.parse())
        moves = game.moves()
        before = moves[-20:]
        batch = []
        for one in notes:
            self.record("delivered", game=key, name=one["name"], action=status["actions"], text=one["text"])
            self.add_news(f"your note to {key} was delivered at its action {status['actions']}", {one["member"]},
                          kind="delivered", game=key)
            with self.lock:
                game.pending.append({"member": one["member"], "name": one.get("plain") or one["name"], "text": one["text"],
                                     "function": one.get("final", ""),
                                     "call": game.calls, "actions": status["actions"], "levels": status["levels"],
                                     "parsed": parsed, "moves": len(moves), "noop_before": _noops(before),
                                     "seen_before": len(before)})
                if self.hier:
                    game.pending[-1]["help"] = len(self.helps)
                    if game.wall is not None:
                        self.walls[game.wall]["helps"].append(len(self.helps))
                    self.helps.append({"member": one["member"], "prince": self.prince_of(one["member"]),
                                       "kind": "function" if one.get("function") else "note",
                                       "topic": one.get("kind", ""), "game": key, "call": game.calls,
                                       "read": None, "mentioned": None, "worse": None, "t": _now(),
                                       "wall": game.wall, "used": None, "replies": [], "text": one["text"],
                                       "spent": status["tokens"] - game.level_tokens})
                batch.append({"member": one["member"], "name": one.get("plain") or one["name"],
                              "kind": "function" if one.get("function") else "note", "text": one["text"],
                              "help": len(self.helps) - 1 if self.hier else None, "call": game.calls})
        with self.lock:
            game.last_batch = batch
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
        for one in due:
            after = moves[one["moves"]:]
            thinking = [(turn["turn"], "\n".join(line for label, _extra, lines in turn["sections"]
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
            used: list[int] = []
            if one.get("function"):
                used = [turn["turn"] for turn in turns[one["parsed"]:]
                        if any(label.split(":")[0].strip() == "TOOL CALL" and f"{one['function']}(" in line
                               for label, _extra, lines in turn["sections"] for line in lines)]
                said += (f"; its code called {one['function']} in turn(s) {', '.join(map(str, used[:5]))}" if used
                         else f"; its code never called {one['function']}")
            when = "when the game ended" if ended else f"{game.calls - one['call']} player turns after delivery"
            kind = f"function {one['function']}" if one.get("function") else "note"
            text = (f"outcome of your {kind} to {game.key}, {when}"
                    f" (delivered at action {one['actions']}): its thinking since: {said}; levels won since: "
                    f"{status['levels'] - one['levels']}; actions since: {status['actions'] - one['actions']}; actions "
                    f"that changed nothing: {one['noop_before']} of the last {one['seen_before']} before the note, "
                    f"{_noops(after)} of {len(after)} since. Your note was: \"{_cap(one['text'], 200)}\"")
            if self.hier:
                self.court_outcome(game, one, text, bool(quoted), bool(thinking and mention.search(thinking[0][1])),
                                   status["levels"] - one["levels"], _noops(after), len(after),
                                   bool(used) if one.get("function") else bool(quoted))
            else:
                self.add_news(text, {one["member"]})
            self.record("outcome", game=game.key, member=one["member"], name=one["name"], ended=ended,
                        mentioned=bool(quoted), levels=status["levels"] - one["levels"],
                        actions=status["actions"] - one["actions"], noop_before=one["noop_before"],
                        seen_before=one["seen_before"], noop_after=_noops(after), seen_after=len(after))

    def names(self) -> list[str]:
        with self.lock:
            return [self.label(one["id"]) for one in self.members if one.get("named") and one.get("alive")]

    def fit_serve(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=(5, 10),
                                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None)
            limit = max(int(one.get("max_model_len") or 0) for one in response.json().get("data") or [{}])
        except Exception as exc:
            self.say(f"the serve's context limit is unknown ({type(exc).__name__}); context stays {self.context_tokens}")
            return
        room = limit - self.max_output - 2048
        if limit and self.context_tokens > room:
            was = self.context_tokens
            self.context_tokens = max(8192, room)
            self.handoff_at = min(self.handoff_at, self.context_tokens * 9 // 10)
            self.say(f"the serve holds {limit} tokens a request, so the context drops from {was} to "
                     f"{self.context_tokens} and the handoff to {self.handoff_at} (replies up to {self.max_output})")

    def start(self) -> "Forum":
        self.fit_serve()
        self.say(f"{self.count} helpers arrive one every {self.arrival_s:.0f} s"
                 f"{', and every game that leaves its lane gives the lane to one more helper' if self.grow else ''}; "
                 f"game briefs list the {self.similar_top} most similar games; context {self.context_tokens} tokens, "
                 f"handoff asked at {self.handoff_at}, "
                 f"{'one serve session a seat (X-Session-ID, released at each handoff)' if self.sessions else 'sessionless requests with a stop list'}, "
                 f"replies up to {self.max_output}, reads up to {self.read_chars} chars; forum in {self.root}")
        self.record("start", helpers=self.count, grow=self.grow, similar_top=self.similar_top, context=self.context_tokens, max_output=self.max_output,
                    read_chars=self.read_chars, end_at=self.end_at)
        if self.hier:
            self.say(f"court: 1 king, {self.princes} princes and {self.start_vassals} vassals at the start, "
                     + (f"a new vassal on every freed lane up to {self.vassals_per} a {'prince' if self.princes else 'king'}"
                        if self.grow else "no growth on freed lanes")
                     + f", at most {self.seat_cap()} seats; news per role, alerts at once, the rest every {self.news_every} "
                     f"calls, at most {self.news_lines} lines; "
                     + (f"walls at {wall_rule(self.wall_tokens, self.wall_actions)} on one level, at most "
                        f"{self.walls_per} walls a vassal" + (f" (ARC3_FORUM_PRINCES={self.princes_asked} ignored: walls "
                                                               f"run a king over vassals)" if self.princes_asked and
                                                              os.environ.get("ARC3_FORUM_PRINCES") else "")
                        if self.walls_mode else f"stuck alert every {self.stuck_actions} actions without a level")
                     + f"; help effect counted over {self.effect_turns} player turns; similar games "
                     + (f"top {self.similar_top}" if self.similar_top > 0 else "off"))
            self.record("court", princes=self.princes, vassals=self.vassals_per, cap=self.seat_cap(),
                        start_vassals=self.start_vassals, grow=self.grow, walls=self.walls_mode,
                        wall_tokens=self.wall_tokens, wall_actions=self.wall_actions, walls_per=self.walls_per)
        watcher = threading.Thread(target=self.watch, name="forum-watch", daemon=True)
        watcher.start()
        spawner = threading.Thread(target=self.spawn, name="forum-spawn", daemon=True)
        spawner.start()
        self.threads_out += [watcher, spawner]
        return self

    def seat(self, why: str, freed: dict[str, Any] | None = None, role: str = "", lord: int | None = None) -> None:
        with self.lock:
            index = len(self.members)
            name = NAMES[index % len(NAMES)] + ("" if index < len(NAMES) else str(index // len(NAMES) + 1))
            member = {"id": index, "order": index + 1, "name": name, "named": True, "alive": True, "joined": _now(),
                      "generation": 1, "handoff": "", "handoffs": 0, "freed": freed, "sent": []}
            if self.hier:
                role = role or ("king" if index == 0 else "prince" if index <= self.princes else "vassal")
                if lord is None and role == "vassal" and self.princes:
                    lord = 1 + (index - 1 - self.princes) % self.princes
                lord = lord if lord is not None else (None if role == "king" else 0)
                member.update({"role": role, "lord": lord, "orders": [], "generated": 0})
            self.members.append(member)
            self.add_news(f"a new helper arrived: {self.label(index)}", kind="arrived", member=index)
        self.record("named", member=index, order=index + 1, name=name, why=why,
                    **({"role": member["role"], "lord": member["lord"]} if self.hier else {}))
        self.say(f"seat {index} is {self.label(index)} ({why})")
        if self.hier and member["role"] == "vassal":
            self.first_fief(member)
        helper = Helper(self, member)
        self.helper_objects[index] = helper
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
        if self.hier:
            with self.lock:
                self.spawned = True
                waiting, self.deferred = self.deferred, []
            for key, status in waiting:
                self.grow_vassal(key, status)

    def prince_of(self, member_id: int | None) -> int | None:
        with self.lock:
            member = self.members[member_id] if member_id is not None and 0 <= member_id < len(self.members) else None
        if member is None or member.get("role") == "king":
            return None
        return member["id"] if member.get("role") == "prince" else member.get("lord")

    def games_of(self, member_id: int) -> list[str]:
        with self.lock:
            member = self.members[member_id] if 0 <= member_id < len(self.members) else None
            if member is None:
                return []
            if member.get("role") == "king":
                return sorted(self.games)
            if member.get("role") == "prince":
                return sorted(key for key, owner in self.domain.items() if owner == member_id)
            return sorted(key for key, owner in self.fief.items() if owner == member_id)

    def vassals_of(self, prince: int) -> list[dict[str, Any]]:
        with self.lock:
            return [one for one in self.members if one.get("role") == "vassal" and one.get("lord") == prince]

    def seat_cap(self) -> int:
        if not self.grow:
            cap = 1 + self.princes + self.start_vassals
        else:
            cap = 1 + self.princes + max(self.start_vassals, (self.princes or 1) * self.vassals_per)
        return min(cap, self.max_helpers) if self.max_helpers > 0 else cap

    def grow_vassal(self, key: str, status: dict[str, Any]) -> None:
        if self.stop_event.is_set() or _now() >= self.end_at:
            return
        freed = {"game": key, "state": status["state"], "levels": status["levels"], "total": status["total"],
                 "actions": status["actions"]}
        with self.lock:
            if not self.spawned:
                self.deferred.append((key, status))
                return
            counts = [(len(self.vassals_of(one["id"])), one["id"]) for one in self.members
                      if one.get("role") == ("prince" if self.princes else "king")]
            open_ = [pair for pair in counts if pair[0] < self.vassals_per]
            full = len(self.members) >= self.seat_cap() or (
                self.follow_games and self.helpers_alive() >= self.helper_limit())
            if open_ and not full:
                self.grown += 1
                grown = self.grown
                prince = min(open_)[1]
                self.say(f"{key} left its lane ({status['state']}, {status['levels']} of {status['total']} levels, "
                         f"{status['actions']} actions); a new vassal for {self.label(prince)}: {self.count} at start "
                         f"+ {grown}")
                self.record("grow", game=key, grown=grown, state=status["state"], levels=status["levels"],
                            prince=prince)
                self.seat(f"on the lane {key} left", freed, "vassal", prince)
                return
        self.say(f"{key} left its lane; no new vassal, the court is at its cap of {self.seat_cap()} seats"
                 + ("" if open_ else f" ({self.vassals_per} vassals for every prince)"))
        self.record("grow_refused", game=key, seats=len(self.members), cap=self.seat_cap())

    def helpers_alive(self) -> int:
        with self.lock:
            return sum(1 for one in self.members if one.get("alive") and not one.get("retired"))

    def helper_limit(self) -> int:
        with self.lock:
            live = sum(1 for game in self.games.values() if not game.freed)
        return max(self.min_helpers, live)

    def trim_helpers(self) -> None:
        if not self.follow_games:
            return
        limit = self.helper_limit()
        with self.lock:
            alive = [one for one in self.members if one.get("alive") and not one.get("retired")]
            youngest = sorted((one for one in alive if one.get("role") == "vassal"), key=lambda one: -one["order"])
            leaving = youngest[:max(0, len(alive) - limit)]
            for one in leaving:
                one["retired"] = True
        for one in leaving:
            self.say(f"{self.label(one['id'])} leaves: {limit} games or helpers at most "
                     f"(the games still playing, at least {self.min_helpers}); the youngest helpers go first")
            self.record("retire", member=one["id"], name=one["name"], limit=limit)
            self.add_news(f"{self.label(one['id'])} leaves the court: fewer games play now, so the youngest "
                          f"helpers leave; its walls go back to the harness", {0}, kind="retire")

    def first_fief(self, member: dict[str, Any]) -> None:
        if self.walls_mode:
            self.assign_walls()
            return
        prince = member["lord"]
        with self.lock:
            if self.princes:
                domain = [key for key, owner in sorted(self.domain.items()) if owner == prince and not self.games[key].ended]
            else:
                domain = [key for key in sorted(self.games) if not self.games[key].ended]
            free = [key for key in domain if key not in self.fief]
            share = math.ceil(len(domain) / (len(self.vassals_of(prince)) + 1)) if domain else 0
            taken = free[:share]
            for key in taken:
                self.fief[key] = member["id"]
        if taken:
            self.record("assign", by=prince, to=member["id"], games=taken, first=True)
            self.add_news(f"{self.label(member['id'])} arrived and got {len(taken)} game(s) of your domain at first: "
                          f"{_keys(taken)}; reassign them with assign", {prince}, kind="assign")

    def turn_levels(self, game: _Game) -> None:
        try:
            levels = game.status()["levels"]
        except Exception:
            return
        won = []
        with self.lock:
            if levels > game.turn_levels:
                game.level_calls += [game.calls - 1] * (levels - game.turn_levels)
                won = list(range(game.turn_levels + 1, levels + 1))
                game.turn_levels = levels
            helps = [one for one in self.helps if one["game"] == game.key]
        for level in won:
            recent = [one for one in helps if one["call"] <= game.calls - 1 < one["call"] + self.effect_turns]
            if not recent:
                continue
            last = recent[-1]
            turns = game.calls - last["call"]
            topic = f", kind '{last['topic']}'" if last["topic"] else ""
            read = {True: "yes", False: "no", None: "not known yet"}[last["read"]]
            text = (f"{game.key} won level {level} {turns} player turn(s) after a {last['kind']} by "
                    f"{self.label(last['member'])}{topic} (read next turn: {read})")
            with self.lock:
                self.effects.append(text)
            if not self.walls_mode:
                self.add_news(text, {0}, kind="effect", game=game.key)
            self.record("effect", game=game.key, level=level, turns=turns, member=last["member"], help_kind=last["kind"],
                        topic=last["topic"])

    def help_counts(self, help: dict[str, Any]) -> tuple[int, int, bool]:
        game = self.games.get(help["game"])
        if game is None:
            return 0, 0, False
        calls = list(game.level_calls)
        before = sum(1 for call in calls if help["call"] - self.effect_turns <= call < help["call"])
        after = sum(1 for call in calls if help["call"] <= call < help["call"] + self.effect_turns)
        return before, after, game.ended or game.calls - help["call"] >= self.effect_turns

    def help_table(self, helps: list[dict[str, Any]]) -> dict[str, Any]:
        row = {"sent": len(helps), "read": 0, "settled": 0, "before": 0, "after": 0, "worse": 0, "score": 0}
        for one in helps:
            before, after, done = self.help_counts(one)
            if one["read"] is not None:
                row["settled"] += 1
                row["read"] += 1 if one["read"] else 0
            row["worse"] += 1 if one["worse"] else 0
            if done:
                row["before"] += before
                row["after"] += after
                if one["read"]:
                    row["score"] += after - before
        return row

    def check_stuck(self, game: _Game, status: dict[str, Any]) -> None:
        if self.stuck_actions <= 0 or game.ended or status["state"] not in ("playing", "?", "NOT_FINISHED"):
            return
        since = status["actions"] - game.level_actions
        mark = since // self.stuck_actions
        if mark <= game.stuck_mark:
            return
        game.stuck_mark = mark
        text = (f"ALERT {game.key} is stuck: {since} actions without a level (level {status['levels'] + 1} of "
                f"{status['total']}, {status['actions']} actions in all)")
        self.add_news(text, kind="stuck", game=game.key, alert=True)
        self.record("stuck", game=game.key, since=since, levels=status["levels"])

    def check_budget(self) -> None:
        left = (self.end_at - _now()) / 60.0
        if self.budget_warned or left > self.budget_alert_min:
            return
        self.budget_warned = True
        self.add_news(f"ALERT the play clock is nearly out: {left:.0f} minutes left in this run", kind="budget",
                      alert=True)
        self.record("budget", left=round(left, 1))

    def check_wall(self, game: _Game, status: dict[str, Any]) -> None:
        if game.ended or game.wall is not None or status["state"] not in ("playing", "?", "NOT_FINISHED"):
            return
        tokens = status["tokens"] - game.level_tokens
        actions = status["actions"] - game.level_actions
        hit = [name for name, spent, limit in (("tokens", tokens, self.wall_tokens), ("actions", actions, self.wall_actions))
               if limit > 0 and spent >= limit]
        if hit:
            self.open_wall(game, status, hit[0])

    def fork_slots(self) -> int:
        return 0 if self.forks_max <= 0 else min(self.forks_max, 1 + self.grown // 2)

    def schedule_forks(self) -> None:
        with self.lock:
            free = self.fork_slots() - self.forks_running
            games = [game for game in self.games.values() if not game.ended and not game.freed and not game.fork_busy]
        if free <= 0:
            return
        ranked = []
        for game in games:
            player = game.player()
            if player is None or not getattr(player, "_forum_request", None) or _now() - game.fork_t < self.fork_gap_s:
                continue
            status = game.status()
            if status["state"] not in ("playing", "?", "NOT_FINISHED"):
                continue
            spent = status["tokens"] - game.level_tokens
            asked = bool(game.asks) and game.asks[-1]["t"] > game.fork_t
            if not (asked or spent >= self.fork_min_tokens):
                continue
            ranked.append(((0 if asked else 1, game.forks_done, -spent), game))
        for _, game in sorted(ranked, key=lambda one: one[0])[:free]:
            with self.lock:
                game.fork_busy = True
                self.forks_running += 1
            threading.Thread(target=self.run_fork, args=(game,), name=f"forum-fork-{game.key}", daemon=True).start()

    def fork_message(self, game: _Game) -> tuple[dict[str, Any], int, int]:
        from inference.agent.vision_context import frame_to_png_data_url
        entries = [entry for entry in list(game.session.history_entries) if getattr(entry, "frame", None) is not None]
        marked = {0, len(entries) - 1} | {index for index, entry in enumerate(entries)
                                          if (getattr(entry, "result", {}) or {}).get("level_completed")
                                          or (getattr(entry, "result", {}) or {}).get("game_over")}
        others = [index for index in range(len(entries)) if index not in marked]
        spare = max(0, self.fork_frames - len(marked))
        count = min(spare, len(others))
        chosen = sorted(marked | {others[int(i * len(others) / count)] for i in range(count)})
        parts: list[dict[str, Any]] = []
        previous = None
        for index in chosen:
            entry = entries[index]
            result = dict(getattr(entry, "result", {}) or {})
            grid = entry.frame.grid
            if index == 0:
                line = "Start of the game, before any action."
            else:
                before = entries[index - 1].frame.grid
                cells = sum(1 for ra, rb in zip(before, grid) for x, y in zip(ra, rb) if x != y)
                line = f"Action {getattr(entry.frame, 'step', '?')}: {getattr(entry, 'action', '')}. {cells} cells changed."
                line += " Level won." if result.get("level_completed") else ""
                line += " Died: game over, the level restarted." if result.get("game_over") else ""
                if previous is not None and index - previous > 1:
                    line = f"({index - previous - 1} actions not shown.) " + line
            parts += [{"type": "text", "text": line},
                      {"type": "image_url", "image_url": {"url": frame_to_png_data_url(entry.frame, upscale=self.fork_upscale)}}]
            previous = index
        shown = len(chosen)
        animated = 0
        with game.animation_lock:
            pinned = [dict(one) for one in game.pinned_animations]
        for one in pinned:
            record = game.animation(one["step"])
            if record is None or self.fork_anim_frames <= 0:
                continue
            chain = record["chain"]
            wanted = min(len(chain), self.fork_anim_frames)
            picks = sorted({round(i * (len(chain) - 1) / max(1, wanted - 1)) for i in range(wanted)})
            for k in picks:
                parts += [{"type": "text", "text": f"Animation of action {one['step']} ({one.get('why', '')}): board {k} "
                                                   f"of {len(chain) - 1}."},
                          {"type": "image_url", "image_url": {"url": frame_to_png_data_url(
                              SimpleNamespace(grid=chain[k]), upscale=self.fork_upscale)}}]
                animated += 1
        parts.append({"type": "text", "text": FORK_QUESTION})
        return {"role": "user", "content": parts}, shown, animated

    def fork_view(self, game: _Game, player: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        seen = getattr(player, "_forum_reply", None)
        deadline = _now() + (self.fork_reply_wait_s if seen is not None else 0)
        while _now() < deadline and not game.ended:
            got = getattr(player, "_forum_reply", None)
            if got is not None and got is not seen:
                return got[0], got[1], "reply"
            time.sleep(0.05)
        messages, tools, _turn = player._forum_request
        return messages, tools, "prompt"

    def run_fork(self, game: _Game) -> None:
        began, calls, mark = _now(), game.calls, game.level_tokens
        hold = threading.Event()
        first: list[float] = []
        player = None
        key = None
        try:
            player = game.player()
            message, shown, animated = self.fork_message(game)
            key = str(getattr(player, "_rotation_key", None) or getattr(player, "_diag_name", "") or game.key)
            FIRST_CHUNK[key] = lambda: (first.append(_now()), hold.set())
            player._fork_hold = hold
            messages, tools, at = self.fork_view(game, player)
            sent = _now()
            reply = player._fork_complete("forum", list(messages) + [message], tools, max_tokens=self.fork_tokens)
            if at == "reply":
                at += (f", player held {first[0] - sent:.1f} s until the fork's first token" if first and
                       first[0] - sent <= self.fork_hold_s else ", player released before the fork's first token")
            text = player._fork_text(reply.message)
            usage = reply.usage if isinstance(reply.usage, dict) else {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            self.record("fork", game=game.key, boards=shown, animation=animated, prompt=usage.get("prompt_tokens"),
                        cached=cached, completion=usage.get("completion_tokens"), seconds=round(_now() - began, 1),
                        at=at, text=text)
            self.say(f"FORK {game.key} at the player's {at}: {shown} boards and {animated} animation boards, prompt "
                     f"{usage.get('prompt_tokens')} ({cached} cached), {usage.get('completion_tokens')} generated, "
                     f"{_now() - began:.0f} s" + (f":{chr(10)}{_quote(text)}" if text else ", no text"))
            if text:
                with self.lock:
                    game.fork_waiting.append({"t": began, "calls": calls, "boards": shown, "animation": animated,
                                              "text": text, "mark": mark})
        except Exception as exc:
            self.say(f"FORK {game.key} failed: {type(exc).__name__}: {exc}")
            self.record("fork_failed", game=game.key, error=f"{type(exc).__name__}: {exc}")
        finally:
            hold.set()
            if key is not None:
                FIRST_CHUNK.pop(key, None)
            if player is not None:
                player.__dict__.pop("_fork_hold", None)
            with self.lock:
                game.fork_busy = False
                game.fork_t = _now()
                game.forks_done += 1
                self.forks_running -= 1

    def fork_block(self, game: _Game, forks: list[dict[str, Any]]) -> str:
        lines = []
        for one in forks:
            lines.append(f"You were forked {game.calls - one['calls']} turn(s) ago ({(_now() - one['t']) / 60:.0f} min ago): "
                         f"a copy of you, with this whole conversation, was shown {one['boards']} boards of your game "
                         + (f"and {one['animation']} boards of the animations of your wins and deaths " if one["animation"] else "")
                         + f"to help you. Here is what it wanted to tell you:{chr(10)}{one['text']}")
            self.record("fork_delivered", game=game.key, calls=game.calls - one["calls"])
            game.fork_calls.append(game.calls)
        return (chr(10) * 2).join(lines)

    def fill_walls(self) -> None:
        with self.lock:
            seats = sum(1 for one in self.members if one.get("role") == "vassal" and one.get("alive"))
            open_ = sum(1 for wall in self.walls if not wall["end"])
            games = [game for game in self.games.values() if not game.ended and not game.freed and game.wall is None]
        free = seats * self.walls_per - open_
        if free <= 0:
            return
        ranked = []
        for game in games:
            status = game.status()
            if status["state"] not in ("playing", "?", "NOT_FINISHED"):
                continue
            tokens = status["tokens"] - game.level_tokens
            if tokens >= self.wall_fill_tokens:
                ranked.append((tokens, status["actions"] - game.level_actions, game, status))
        ranked.sort(key=lambda one: (one[0], one[1]), reverse=True)
        for _, _, game, status in ranked[:free]:
            self.open_wall(game, status, "rank")

    def open_wall(self, game: _Game, status: dict[str, Any], by: str) -> None:
        tokens = status["tokens"] - game.level_tokens
        actions = status["actions"] - game.level_actions
        hit = [by]
        with self.lock:
            if game.wall is not None or game.ended:
                return
            wall = {"id": len(self.walls), "game": game.key, "level": status["levels"] + 1, "total": status["total"],
                    "t": _now(), "by": hit[0], "tokens0": game.level_tokens, "actions0": game.level_actions,
                    "calls0": game.calls, "vassal": None, "vassals": [], "helps": [], "end": "", "t_end": 0.0,
                    "end_call": None, "tokens": tokens, "actions": actions, "asks": []}
            self.walls.append(wall)
            game.wall = wall["id"]
        text = (f"ALERT new wall: {game.key} level {wall['level']} of {wall['total']}: {tokens / 1000:.0f}k tokens and "
                f"{actions} actions on this level without winning it"
                + (", and its player asked the helpers for help" if by == "ask" else "")
                + (" (the most spent on one level among the games no vassal holds, and a vassal had a free "
                   "wall place)" if by == "rank" else ""))
        self.add_news(text, {0}, kind="wall", game=game.key, alert=True)
        self.record("wall", game=game.key, wall=wall["id"], level=wall["level"], tokens=tokens, actions=actions,
                    by=hit[0])
        self.say(text)
        self.assign_walls()

    def end_wall(self, game: _Game, status: dict[str, Any]) -> None:
        with self.lock:
            if game.wall is None:
                return
            wall = self.walls[game.wall]
            game.wall = None
            won = status["levels"] >= wall["level"]
            wall.update({"end": "won" if won else "ended", "t_end": _now(), "end_call": game.calls,
                         "tokens": status["tokens"] - wall["tokens0"], "actions": status["actions"] - wall["actions0"]})
            vassal = wall["vassal"]
            if vassal is not None and self.fief.get(game.key) == vassal:
                del self.fief[game.key]
            helps = [self.helps[index] for index in wall["helps"]]
        minutes = (wall["t_end"] - wall["t"]) / 60.0
        if won:
            last = helps[-1] if helps else None
            after = (f"; {len(helps)} help(s) delivered to it, the last a {last['kind']} by {self.label(last['member'])} "
                     f"{game.calls - last['call']} player turn(s) before the win" if last
                     else "; no help was delivered to it")
            text = (f"WALL BROKEN {game.key} level {wall['level']} won {minutes:.0f} min after it became a wall, "
                    f"{wall['tokens'] / 1000:.0f}k tokens and {wall['actions']} actions on the level{after}")
        else:
            text = (f"WALL CLOSED {game.key} level {wall['level']}: the game ended ({status['state']}) without winning "
                    f"it, {minutes:.0f} min after it became a wall, {len(helps)} help(s) delivered to it")
        self.add_news(text, {0} | ({vassal} if vassal is not None else set()), kind="wall", game=game.key, alert=True)
        self.record("wall_end", game=game.key, wall=wall["id"], won=won, minutes=round(minutes, 1),
                    tokens=wall["tokens"], actions=wall["actions"], helps=len(helps), vassal=vassal)
        self.say(text)
        self.assign_walls()

    def wall_priority(self, wall: dict[str, Any]) -> tuple:
        game = self.games.get(wall["game"])
        spent = 0
        if game is not None:
            try:
                spent = game.status()["tokens"] - game.level_tokens
            except Exception:
                spent = 0
        return (0 if wall["asks"] or wall["by"] == "ask" else 1, 1 if wall["by"] == "rank" else 0, -spent, wall["t"])

    def wall_load(self, member_id: int) -> int:
        with self.lock:
            return sum(1 for wall in self.walls if not wall["end"] and wall["vassal"] == member_id)

    def assign_walls(self) -> None:
        given = []
        with self.lock:
            vassals = [one["id"] for one in self.members if one.get("role") == "vassal" and one.get("alive")]
            waiting = [one for one in self.walls if not one["end"] and one["vassal"] is None]
            for wall in sorted(waiting, key=self.wall_priority):
                free = sorted((self.wall_load(vassal), vassal) for vassal in vassals
                              if self.wall_load(vassal) < self.walls_per)
                if not free:
                    break
                vassal = free[0][1]
                wall["vassal"] = vassal
                wall["vassals"].append(vassal)
                self.fief[wall["game"]] = vassal
                given.append((wall, vassal))
        for wall, vassal in given:
            self.brief(vassal, wall["game"])
            self.add_news(f"the harness gave you the wall {wall['game']} level {wall['level']}: "
                          f"{self.wall_line(wall)}", {vassal}, kind="assign", game=wall["game"])
            self.add_news(f"the harness gave the wall {wall['game']} to {self.label(vassal)}; move it with assign",
                          {0}, kind="harness_assign", game=wall["game"])
            self.record("assign", by=None, to=vassal, games=[wall["game"]], harness=True)
            self.say(f"WALL {wall['game']} -> {self.label(vassal)} (harness)")

    def help_used(self, help: dict[str, Any]) -> str:
        if help["used"] is None:
            return "use not known yet"
        if help["kind"] == "function":
            return "called by the player" if help["used"] else "never called"
        return "mentioned in its thinking" if help["used"] else "not mentioned in its thinking"

    def wall_line(self, wall: dict[str, Any]) -> str:
        game = self.games.get(wall["game"])
        if wall["end"] or game is None:
            tokens, actions, minutes = wall["tokens"], wall["actions"], (wall["t_end"] - wall["t"]) / 60.0
        else:
            status = game.status()
            tokens, actions = status["tokens"] - wall["tokens0"], status["actions"] - wall["actions0"]
            minutes = (_now() - wall["t"]) / 60.0
        with self.lock:
            helps = [self.helps[index] for index in wall["helps"]]
            waiting = len(game.notes_waiting) if game is not None else 0
            vassal = self.members[wall["vassal"]]["name"] if wall["vassal"] is not None else "none"
        said = "; ".join(f"{one['kind']} by {self.members[one['member']]['name']}"
                         + (f" ('{one['topic']}')" if one["topic"] else "")
                         + f", {one['call'] - wall['calls0']} player turns into the wall, {self.help_used(one)}"
                         + (f"; it said \"{_cap(one.get('text', ''), 120)}\" and the player answered "
                            f"\"{_cap(one['replies'][-1], 160)}\"" if one.get("replies") else "")
                         for one in helps[-4:])
        line = (f"{wall['game']} level {wall['level']} of {wall['total']}: {tokens / 1000:.0f}k tokens and {actions} "
                f"actions on this level, " + (f"a wall for {minutes:.0f} min" if not wall["end"] else
                                               f"{WALL_ENDS.get(wall['end'], 'game ended')} after {minutes:.0f} min as a wall")
                + f"; vassal {vassal}; help delivered {len(helps)}"
                + (f" ({len(helps) - 4} older not shown)" if len(helps) > 4 else "") + (f": {said}" if said else "")
                + (f"; {waiting} waiting for its next turn" if waiting else "")
                + (f"; the player asked {len(wall['asks'])} time(s), last: \"{_cap(wall['asks'][-1]['text'], 200)}\""
                   if wall.get("asks") else ""))
        return line

    def walls_text(self, member_id: int | None) -> str:
        with self.lock:
            live = [wall for wall in self.walls if not wall["end"] and (member_id is None or wall["vassal"] == member_id)]
            closed = [wall for wall in self.walls if wall["end"] and (member_id is None or member_id in wall["vassals"])][-5:]
        head = (f"Walls now: {len(live)}" if member_id is None else f"Your walls now: {len(live)}")
        lines = [head] + [f"- {self.wall_line(wall)}" for wall in live]
        if closed:
            lines.append("Closed lately:")
            lines += [f"- {self.wall_line(wall)}" for wall in closed]
        lines += self.study_lines()
        return "\n".join(lines)

    def study_lines(self, top: int = 8) -> list[str]:
        with self.lock:
            games = [game for game in self.games.values() if not game.ended and not game.freed and game.wall is None]
            won = list(self.recent_levels)
        near = []
        for game in games:
            status = game.status()
            if status["state"] not in ("playing", "?", "NOT_FINISHED"):
                continue
            near.append((status["tokens"] - game.level_tokens, status["actions"] - game.level_actions, game.key,
                         status["levels"] + 1, status["total"]))
        near.sort(reverse=True)
        lines = ["Closest to a wall (not at a wall yet, the most spent on the current level), to study ahead:"]
        lines += [f"- {key} level {level} of {total}: {tokens / 1000:.0f}k tokens, {actions} actions on it"
                  for tokens, actions, key, level, total in near[:top]] or ["- none"]
        lines.append("Levels won lately (their winning press is in view_animation while it stays cached):")
        lines += [f"- {key} won level {level} at minute {(t - self.started) / 60:.0f}" for t, key, level in won[-top:]] \
            or ["- none yet"]
        return lines

    def wall_broken_after(self, help: dict[str, Any]) -> bool:
        wall = self.walls[help["wall"]] if help.get("wall") is not None else None
        return bool(wall and wall["end"] == "won" and wall["end_call"] - help["call"] < self.effect_turns)

    def wall_row(self, helps: list[dict[str, Any]]) -> dict[str, int]:
        mine = [one for one in helps if one.get("wall") is not None]
        return {"sent": len(mine), "used": sum(1 for one in mine if one["used"]),
                "settled": sum(1 for one in mine if one["used"] is not None),
                "broken": len({one["wall"] for one in mine if self.wall_broken_after(one)}),
                "worse": sum(1 for one in mine if one["worse"])}

    def wall_counts(self) -> dict[str, Any]:
        with self.lock:
            walls = list(self.walls)
        helped = [wall for wall in walls if wall["helps"]]
        bare = [wall for wall in walls if not wall["helps"]]
        return {"all": len(walls), "open": sum(1 for wall in walls if not wall["end"]),
                "unassigned": sum(1 for wall in walls if not wall["end"] and wall["vassal"] is None),
                "won": sum(1 for wall in walls if wall["end"] == "won"),
                "ended": sum(1 for wall in walls if wall["end"] == "ended"),
                "helped": len(helped), "helped_won": sum(1 for wall in helped if wall["end"] == "won"),
                "bare": len(bare), "bare_won": sum(1 for wall in bare if wall["end"] == "won"),
                "minutes_helped": [round((wall["t_end"] - wall["t"]) / 60.0) for wall in helped if wall["end"] == "won"],
                "minutes_bare": [round((wall["t_end"] - wall["t"]) / 60.0) for wall in bare if wall["end"] == "won"]}

    def wall_numbers(self, member: dict[str, Any]) -> list[str]:
        if member["role"] == "king":
            counts = self.wall_counts()
            with self.lock:
                helps = list(self.helps)
                vassals = [one for one in self.members if one.get("role") == "vassal"]
            row = self.wall_row(helps)
            spent = sum(int(one.get("generated") or 0) for one in vassals)
            score, parts = self.forum_score()
            return [f"forum score {score:+.2f} (metrics explains it); {parts[-1]}",
                    f"walls: {counts['open']} open ({counts['unassigned']} without a vassal), {counts['won']} broken, "
                    f"{counts['ended']} closed by the game's end; helper tokens generated: king "
                    f"{int(member.get('generated') or 0) // 1000}k, vassals {spent // 1000}k",
                    f"help to walls: {row['sent']} delivered, {row['used']} of {row['settled']} used by the player, "
                    f"{row['broken']} walls won within {self.effect_turns} player turns after a help; walls won with help "
                    f"{counts['helped_won']} of {counts['helped']}, without help {counts['bare_won']} of {counts['bare']}"]
        with self.lock:
            helps = [one for one in self.helps if one["member"] == member["id"]]
        row = self.wall_row(helps)
        mine = self.games_of(member["id"])
        return [f"your walls: {len(mine)}" + (f" ({_keys(mine)})" if mine else "") + f"; your help: {row['sent']} "
                f"delivered, {row['used']} of {row['settled']} used by the player, {row['broken']} walls won within "
                f"{self.effect_turns} player turns after it"]

    def forum_score(self) -> tuple[float, list[str]]:
        with self.lock:
            walls = [dict(wall) for wall in self.walls]
            helps = [dict(one) for one in self.helps]
            games = list(self.games.values())
            members = [dict(one) for one in self.members]
        now = _now()
        credit = idle = unused = standing = 0.0
        blame: dict[Any, list[float]] = {}

        def charge(who: Any, slot: int, value: float) -> None:
            blame.setdefault(who, [0.0, 0.0, 0.0, 0.0, 0.0])[slot] += value
        standing_walls = 0
        credited: list[str] = []
        idle_walls = 0
        first: list[float] = []
        for wall in walls:
            mine = [helps[index] for index in wall["helps"] if index < len(helps)]
            if wall["end"] == "won":
                shares = []
                for one in mine:
                    gap = wall["end_call"] - one["call"]
                    if 0 <= gap <= self.score_turns:
                        weight = min(2.0, 1.0 + float(one.get("spent") or 0) / 40000.0)
                        called = one["kind"] == "function" and one["used"]
                        shares.append(weight * (2.0 if called else 1.0) * 0.5 ** (gap / self.score_half))
                best = max(shares) if shares else 0.0
                if best:
                    credit += best
                    charge(wall["vassal"] if wall["vassal"] is not None else (wall["vassals"] or [None])[-1], 0, best)
                    credited.append(f"{wall['game']} +{best:.2f}")
            if wall["vassal"] is not None or wall["vassals"]:
                stop = min([one["t"] for one in mine] or [wall["t_end"] if wall["end"] else now])
                if mine:
                    first.append((stop - wall["t"]) / 60.0)
                blocks = int((stop - wall["t"]) // 600)
                if blocks:
                    idle += 0.125 * blocks
                    idle_walls += 1
                    charge(wall["vassal"] if wall["vassal"] is not None else (wall["vassals"] or [None])[-1], 1,
                           0.125 * blocks)
            if wall["end"] != "won":
                blocks = int(((wall["t_end"] if wall["end"] else now) - wall["t"]) // 600)
                if blocks:
                    standing += 0.1 * blocks
                    standing_walls += 1
                    charge(wall["vassal"] if wall["vassal"] is not None else (wall["vassals"] or [None])[-1], 2,
                           0.1 * blocks)
            called = 0.25 * sum(1 for one in mine if one["kind"] == "function" and one["used"])
            unused += called
            if called:
                charge(wall["vassal"] if wall["vassal"] is not None else (wall["vassals"] or [None])[-1], 3, called)
        score = credit + unused - idle - standing
        functions = [one for one in helps if one["kind"] == "function"]
        counts = self.wall_counts()
        forks = sum(len(game.fork_calls) for game in games)
        fork_wins = sum(1 for game in games for call in game.fork_calls
                        if any(0 <= won - call < self.effect_turns for won in game.level_calls))
        first.sort()
        lines = [f"credit +{credit:.2f} from {len(credited)} walls won within {self.score_turns} player turns after a "
                 f"help (each help worth 1 + (tokens the level had cost when it came) / 40k, at most 2, x2 for a "
                 f"function the player's code called, halved for every {self.score_half} turns between it and the won level; a wall pays "
                 f"its best help in that window)" + (f": {', '.join(credited[-8:])}" if credited else ""),
                 f"idle -{idle:.2f}: {idle_walls} walls held 10+ minutes before their first help (0.125 per 10 minutes)",
                 f"functions +{unused:.2f}: 0.25 for each function the player's code called",
                 f"standing -{standing:.2f}: {standing_walls} walls not broken, 0.1 per 10 minutes each from the minute "
                 f"they were handed out, whoever's fault it is",
                 f"walls won with help {counts['helped_won']} of {counts['helped']}, without help {counts['bare_won']} "
                 f"of {counts['bare']}; minutes from wall to first help, median "
                 + (f"{first[len(first) // 2]:.0f}" if first else "none yet"),
                 f"functions {len(functions)}, called by the player {sum(1 for one in functions if one['used'])}; "
                 f"forks delivered {forks}, a level within {self.effect_turns} player turns after {fork_wins}"]
        rows = []
        for who, (won, waited, stood, called, read) in sorted(
                blame.items(), key=lambda pair: -(pair[1][0] + pair[1][3] - pair[1][1] - pair[1][2] - pair[1][4])):
            name = self.label(who) if who is not None else "walls with no vassal"
            rows.append(f"{name} {won + called - waited - stood - read:+.2f} (levels +{won:.2f}, functions "
                        f"+{called:.2f}, waiting -{waited:.2f}, standing -{stood:.2f})")
        lines.append("by holder: " + ("; ".join(rows) if rows else "nothing yet"))
        return score, lines

    def wall_metrics_text(self) -> str:
        counts = self.wall_counts()
        with self.lock:
            helps = list(self.helps)
            vassals = [one for one in self.members if one.get("role") == "vassal"]
            walls = list(self.walls)

        def median(values: list[int]) -> str:
            ordered = sorted(values)
            return f"{ordered[len(ordered) // 2]} min" if ordered else "none"

        score, parts = self.forum_score()
        lines = [f"Forum score {score:+.2f}:", *[f"- {one}" for one in parts],
                 f"Walls: {counts['all']} in all, {counts['open']} open ({counts['unassigned']} without a vassal), "
                 f"{counts['won']} broken (level won), {counts['ended']} closed by the game's end.",
                 f"Walls that got help: {counts['helped']}, won {counts['helped_won']}, median minutes from wall to win "
                 f"{median(counts['minutes_helped'])}; walls without help: {counts['bare']}, won {counts['bare_won']}, "
                 f"median {median(counts['minutes_bare'])}.",
                 f"Per vassal (walls now | walls held in all | help delivered | used by the player | walls won within "
                 f"{self.effect_turns} player turns after its help | tokens generated):"]
        king = [one for one in helps if one["member"] == 0]
        row = self.wall_row(king)
        lines.append(f"- {self.label(0)}: - | - | {row['sent']} | {row['used']}/{row['settled']} | {row['broken']} | "
                     f"{int(self.members[0].get('generated') or 0) // 1000}k")
        for vassal in vassals:
            row = self.wall_row([one for one in helps if one["member"] == vassal["id"]])
            held = sum(1 for wall in walls if vassal["id"] in wall["vassals"])
            lines.append(f"- {self.label(vassal['id'])}: {self.wall_load(vassal['id'])} | {held} | {row['sent']} | "
                         f"{row['used']}/{row['settled']} | {row['broken']} | {int(vassal.get('generated') or 0) // 1000}k")
        lines.append(f"Per kind of help (delivered to walls | used by the player | walls won within {self.effect_turns} "
                     f"player turns after it | followed by worse play):")
        groups: dict[str, list[dict[str, Any]]] = {}
        for one in helps:
            groups.setdefault(one["kind"], []).append(one)
        labels: dict[str, list[dict[str, Any]]] = {}
        for one in helps:
            if one["topic"]:
                labels.setdefault(one["topic"], []).append(one)
        shown = [(name, mine) for name, mine in sorted(groups.items())]
        shown += [(f"label '{name}'", mine) for name, mine in sorted(labels.items(), key=lambda pair: -len(pair[1]))[:8]]
        for name, mine in shown:
            row = self.wall_row(mine)
            lines.append(f"- {name}: {row['sent']} | {row['used']}/{row['settled']} | {row['broken']} | {row['worse']}")
        if not shown:
            lines.append("- none yet")
        lines += self.reply_lines(helps)
        closed = [wall for wall in walls if wall["end"]][-6:]
        lines.append("Walls closed lately:")
        lines += [f"- {self.wall_line(wall)}" for wall in closed] or ["- none yet"]
        lines.append(f"A help counts toward a broken wall when the wall's level was won within {self.effect_turns} "
                     f"player turns after the help was delivered; used means the player's code called a function, or "
                     f"its thinking mentioned a note.")
        return "\n".join(lines)

    def reply_lines(self, helps: list[dict[str, Any]]) -> list[str]:
        if not self.player_api:
            return []
        replied = [one for one in helps if one.get("replies")][-8:]
        return [f"The players' replies to help (latest {len(replied)} of {sum(1 for one in helps if one.get('replies'))}), "
                "each as what the helper said, then what the player answered:"] + [
            f"- {one['game']}: {self.label(one['member'])} said ({one['kind']}): \"{_cap(one['text'], 240)}\"" + chr(10)
            + f"  the player answered ({self.help_used(one)}): \"{_cap(one['replies'][-1], 240)}\"" for one in replied] + (
            [] if replied else ["- none yet"])

    def advisor(self, key: str, kind: str, text: str) -> str:
        game = self.games.get(key)
        text = str(text).strip()
        if game is None:
            return "Advisors: this game is not followed by the helpers; nothing was sent."
        if not text:
            return "Advisors: the text is empty; nothing was sent."
        text = _cap(text, self.answers_chars)
        if kind == "reply":
            return self.player_reply(game, text)
        if kind == "ask":
            return self.player_ask(game, text)
        return "Advisors: unknown call; nothing was sent."

    def player_reply(self, game: _Game, text: str) -> str:
        with self.lock:
            batch = list(game.last_batch)
            for one in batch:
                if one["help"] is not None and one["help"] < len(self.helps):
                    self.helps[one["help"]]["replies"].append(text)
        if not batch:
            return "Advisors: no helper note has reached you yet; nothing was sent."
        for one in batch:
            self.add_news(f"REPLY on {game.key}, {game.calls - one['call']} player turn(s) after your {one['kind']}:"
                          + chr(10) + f"  you said: \"{_cap(one['text'], 600)}\"" + chr(10)
                          + f"  the player answered: \"{text}\"", {one["member"]}, kind="reply", game=game.key,
                          alert=True)
        with self.lock:
            self.reply_log.append({"t": _now(), "game": game.key, "answer": text,
                                   "advice": [{"member": one["member"], "kind": one["kind"], "text": one["text"],
                                               "turns": game.calls - one["call"]} for one in batch]})
        self.record("reply", game=game.key, members=[one["member"] for one in batch], text=text,
                    advice=[{"member": one["member"], "name": one["name"], "kind": one["kind"], "text": one["text"],
                             "turns": game.calls - one["call"]} for one in batch])
        self.say(f"REPLY {game.key} -> {', '.join(one['name'] for one in batch)}:\n{_quote(text)}")
        return f"Advisors: your reply was delivered to {', '.join(sorted({one['name'] for one in batch}))}."

    def player_ask(self, game: _Game, text: str) -> str:
        if game.ended:
            return "Advisors: this game has ended; nothing was sent."
        status = game.status()
        with self.lock:
            game.asks.append({"t": _now(), "text": text, "call": game.calls, "level": status["levels"] + 1})
        if self.walls_mode:
            if game.wall is None:
                self.open_wall(game, status, "ask")
            with self.lock:
                wall = self.walls[game.wall] if game.wall is not None else None
                if wall is not None:
                    wall["asks"].append({"t": _now(), "text": text, "call": game.calls})
                audience = {0} | ({wall["vassal"]} if wall is not None and wall["vassal"] is not None else set())
        elif self.hier:
            audience = {0} | {owner for owner in (self.domain.get(game.key), self.fief.get(game.key)) if owner is not None}
        else:
            audience = None
        self.add_news(f"ASK from the player of {game.key} (level {status['levels'] + 1} of {status['total']}, "
                      f"{status['actions'] - game.level_actions} actions on it): {text}", audience, kind="ask",
                      game=game.key, alert=True)
        self.record("ask", game=game.key, level=status["levels"] + 1, text=text,
                    audience=sorted(audience) if audience is not None else None)
        self.say(f"ASK {game.key}:\n{_quote(text)}")
        return ("Advisors: your question was delivered to the helpers; an answer, if they find one, arrives as a "
                "helper note at the start of a later turn.")

    def court_outcome(self, game: _Game, one: dict[str, Any], text: str, mentioned: bool, read: bool, levels: int,
                      noop_after: int, seen_after: int, used: bool | None = None) -> None:
        rate_before = one["noop_before"] / one["seen_before"] if one["seen_before"] else 0.0
        rate_after = noop_after / seen_after if seen_after else 0.0
        worse = levels == 0 and seen_after >= 3 and rate_after > rate_before + 0.2
        with self.lock:
            if "help" in one and one["help"] < len(self.helps):
                self.helps[one["help"]].update({"read": read, "mentioned": mentioned, "worse": worse, "used": used})
            if worse:
                self.worse += 1
        author = one["member"]
        self.add_news(text, {author}, kind="outcome", game=game.key)
        prince = self.domain.get(game.key)
        if prince is not None and prince != author:
            self.add_news(f"help in your domain: {one['name']}'s {'function' if one.get('function') else 'note'} to "
                          f"{game.key}: read next turn {'yes' if read else 'no'}, levels since {levels}, actions that "
                          f"changed nothing {noop_after} of {seen_after} since", {prince}, kind="domain_outcome",
                          game=game.key)
        if worse:
            chiefs = {author} | ({self.prince_of(author)} - {None})
            self.add_news(f"ALERT play got worse after {self.label(author)}'s note to {game.key}: actions that changed "
                          f"nothing {noop_after} of {seen_after} since, {one['noop_before']} of {one['seen_before']} "
                          f"before, no level", chiefs, kind="worse", game=game.key, alert=True)

    def page_text(self) -> str:
        with self.lock:
            princes = [one for one in self.members if one.get("role") == "prince"]
            lines = [f"[the king's page, {self.minute()}]"]
            if self.page["edits"]:
                lines.append(f"Written by {self.label(0)} at minute {(self.page['t'] - self.started) / 60:.0f}:")
                lines.append(self.page["text"])
            else:
                lines.append("The king has not written the page yet.")
            if self.walls_mode or not self.princes:
                vassals = self.vassals_of(0)
                live = [wall for wall in self.walls if not wall["end"]]
                if self.walls_mode:
                    lines.append(f"Walls open now: {len(live)}, without a vassal {sum(1 for one in live if one['vassal'] is None)}.")
                lines.append(("Vassals with their walls:" if self.walls_mode else "Vassals with their games:"))
                for vassal in vassals:
                    mine = sorted(key for key, owner in self.fief.items() if owner == vassal["id"])
                    lines.append(f"- {self.label(vassal['id'])}: {_keys(mine) if mine else 'none'}")
                if not vassals:
                    lines.append("- none yet")
                return "\n".join(lines)
            unassigned = len([key for key in self.games if key not in self.domain])
            lines.append("Domains:")
            for prince in princes:
                mine = [key for key, owner in self.domain.items() if owner == prince["id"]]
                vassals = [f"{one['name']} {sum(1 for owner in self.fief.values() if owner == one['id'])}"
                           for one in self.vassals_of(prince["id"])]
                lines.append(f"- {self.label(prince['id'])}: {len(mine)} games, "
                             f"{sum(1 for key in mine if self.games[key].ended)} finished; vassals with their games: "
                             f"{', '.join(vassals) or 'none yet'}")
            if unassigned:
                lines.append(f"- not in any domain: {unassigned} games")
        return "\n".join(lines)

    def resolve(self, ref: Any) -> dict[str, Any] | None:
        text = str(ref or "").strip()
        with self.lock:
            if re.fullmatch(r"(?:seat\s*)?\d+", text, re.IGNORECASE):
                index = int(re.sub(r"\D", "", text))
                return self.members[index] if 0 <= index < len(self.members) else None
            for member in self.members:
                if member["name"].lower() == text.lower() or self.label(member["id"]).lower() == text.lower():
                    return member
            head = text.split("(")[0].strip().lower()
            found = [member for member in self.members if member["name"].lower() == head]
        return found[0] if len(found) == 1 else None

    def game_stats(self, keys: list[str]) -> dict[str, int]:
        out = {"games": len(keys), "levels": 0, "finished": 0, "stuck": 0}
        for key in keys:
            game = self.games.get(key)
            if game is None:
                continue
            try:
                out["levels"] += game.status()["levels"]
            except Exception:
                pass
            out["finished"] += 1 if game.ended else 0
            out["stuck"] += 1 if game.stuck_mark and not game.ended else 0
        return out

    def tokens_of(self, prince: int | None) -> int:
        with self.lock:
            return sum(int(one.get("generated") or 0) for one in self.members
                       if (one["id"] if one.get("role") != "vassal" else one.get("lord")) == prince
                       or (prince is None and one.get("role") == "king"))

    def numbers(self, member: dict[str, Any]) -> list[str]:
        if self.walls_mode:
            return self.wall_numbers(member)
        role = member["role"]
        stats = self.game_stats(self.games_of(member["id"]))
        if role == "king":
            with self.lock:
                recent = sum(1 for moment in self.level_times if _now() - moment <= 600.0)
                princes = [one for one in self.members if one.get("role") == "prince"]
                helps = list(self.helps)
                seats = len(self.members)
            spent = ", ".join([f"king {self.tokens_of(None) // 1000}k"]
                              + [f"{one['name']} {self.tokens_of(one['id']) // 1000}k" for one in princes])
            row = self.help_table(helps)
            minutes = min(10.0, max(1.0, (_now() - self.started) / 60.0))
            return [f"swarm: {stats['levels']} levels won, {recent / minutes:.1f} levels/min over the last "
                    f"{minutes:.0f} min, {stats['finished']} of {stats['games']} games finished, {stats['stuck']} stuck, "
                    f"{seats} of {self.seat_cap()} seats; helper tokens generated (prince with its vassals): {spent}",
                    f"help: {row['sent']} notes and functions delivered, {row['read']} of {row['settled']} read on the "
                    f"next turn, levels within {self.effect_turns} player turns after help {row['after']} against "
                    f"{row['before']} before, {row['worse']} followed by worse play; metrics has the table"]
        if role == "prince":
            vassals = self.vassals_of(member["id"])
            spent = ", ".join(f"{one['name']} {int(one.get('generated') or 0) // 1000}k" for one in [member] + vassals)
            with self.lock:
                helps = [one for one in self.helps if self.domain.get(one["game"]) == member["id"]]
            row = self.help_table(helps)
            return [f"your domain: {stats['games']} games, {stats['levels']} levels won, {stats['finished']} finished, "
                    f"{stats['stuck']} stuck; tokens generated: {spent}",
                    f"help in your domain: {row['sent']} delivered, {row['read']} of {row['settled']} read on the next "
                    f"turn, levels within {self.effect_turns} player turns after {row['after']} against {row['before']} "
                    f"before"]
        with self.lock:
            helps = [one for one in self.helps if one["member"] == member["id"]]
        row = self.help_table(helps)
        return [f"your games: {stats['games']}, {stats['levels']} levels won, {stats['finished']} finished, "
                f"{stats['stuck']} stuck; your help: {row['sent']} delivered, {row['read']} of {row['settled']} read on "
                f"the next turn, levels within {self.effect_turns} player turns after {row['after']} against "
                f"{row['before']} before"]

    def metrics_text(self) -> str:
        if self.walls_mode:
            return self.wall_metrics_text()
        with self.lock:
            princes = [one for one in self.members if one.get("role") == "prince"]
            helps = list(self.helps)
            effects = list(self.effects[-10:])
            king = self.members[0] if self.members else None
        head = "seats | games | levels won | finished | stuck | delivered | read next turn | levels after vs before | attribution | tokens"
        lines = self.numbers(king)[:1] if king else []
        lines.append(f"Per prince with its vassals ({head}):")

        def row_text(name: str, seats: int, keys: list[str], mine: list[dict[str, Any]], tokens: int) -> str:
            stats = self.game_stats(keys)
            row = self.help_table(mine)
            return (f"- {name}: {seats} | {stats['games']} | {stats['levels']} | {stats['finished']} | {stats['stuck']} | "
                    f"{row['sent']} | {row['read']}/{row['settled']} | {row['after']} vs {row['before']} | "
                    f"{row['score']:+d} | {tokens // 1000}k")

        if king:
            lines.append(row_text(self.label(0), 1, [], [one for one in helps if one["prince"] is None],
                                  self.tokens_of(None)))
        for prince in princes:
            lines.append(row_text(self.label(prince["id"]), 1 + len(self.vassals_of(prince["id"])),
                                  self.games_of(prince["id"]), [one for one in helps if one["prince"] == prince["id"]],
                                  self.tokens_of(prince["id"])))
        if not self.princes:
            for vassal in self.vassals_of(0):
                lines.append(row_text(self.label(vassal["id"]), 1, self.games_of(vassal["id"]),
                                      [one for one in helps if one["member"] == vassal["id"]],
                                      int(vassal.get("generated") or 0)))
        lines.append("Per kind of help (delivered | read next turn | levels after vs before | followed by worse play | "
                     "attribution):")
        groups: dict[str, list[dict[str, Any]]] = {}
        for one in helps:
            groups.setdefault(one["kind"], []).append(one)
        labels: dict[str, list[dict[str, Any]]] = {}
        for one in helps:
            if one["topic"]:
                labels.setdefault(one["topic"], []).append(one)
        for name, mine in sorted(groups.items()) + sorted(labels.items(), key=lambda pair: -len(pair[1]))[:8]:
            row = self.help_table(mine)
            tag = name if name in groups and mine is groups[name] else f"label '{name}'"
            lines.append(f"- {tag}: {row['sent']} | {row['read']}/{row['settled']} | {row['after']} vs {row['before']} | "
                         f"{row['worse']} | {row['score']:+d}")
        lines += self.reply_lines(helps)
        lines.append(f"Levels won within {self.effect_turns} player turns after a help (latest {len(effects)}):")
        lines += [f"- {text}" for text in effects] or ["- none yet"]
        lines.append(f"Attribution: levels won within {self.effect_turns} player turns after a help minus those within "
                     f"{self.effect_turns} turns before it, summed over helps the player's thinking mentioned on its "
                     f"next turn; a help counts once its turns have passed or its game ended.")
        return "\n".join(lines)

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
        self.news_calls = 0
        self.backlog: list[tuple[bool, str, str, str]] = []
        self.shown: set[str] = set()
        self.last_numbers: list[str] = []

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
                f"stays readable (read_turns, search, view_board, run_python, last_generation), so what its player learned "
                + ("can still help games with similar mechanics." if self.forum.similar_top > 0 else
                   "can still help other games."))

    def role_line(self) -> str:
        if self.forum.hier or self.member["order"] > 3:
            return ""
        return (" Your seat is one of the three senior seats: besides helping, you answer for the forum's overall "
                "effectiveness. Check whether notes and functions actually helped (did the player use them, did a "
                "level follow because of them, or would it have come anyway), say on the forum which kinds of help "
                "work and which waste the players' shared GPU, and steer the other helpers toward what works.")

    def welcome(self) -> str:
        if self.forum.hier:
            return self.court_welcome()
        with self.forum.lock:
            threads = [f"#{one['id']} {one['title']} ({len(one['posts'])} posts)" for one in self.forum.threads[-15:]]
            games = len(self.forum.games)
        present = self.others()
        return (f"You are {self.me()}, a new helper agent, just arrived ({self.forum.minute()}). Your name is "
                f"{self.member['name']}, your arrival number is #{self.member['order']}; the harness signs everything "
                f"you post or send with it.{self.freed_line()}{self.role_line()} Your goal: help the players win levels. {games} games "
                f"are being played right now. Agents on the forum at the moment: "
                f"{', '.join(present) if present else 'none yet, you are among the first'}. "
                f"Forum threads: {'; '.join(threads) if threads else 'none yet'}.")

    def successor(self) -> str:
        if self.forum.hier:
            return self.court_successor()
        present = self.others()
        handoff = self.member["handoff"] or "(your previous instance wrote no handoff)"
        return (f"You are {self.me()}, a helper agent, instance {self.member['generation']} of your seat "
                f"({self.forum.minute()}). Your seat keeps its arrival number #{self.member['order']}. Your previous "
                f"instance in this seat filled its memory and ended. Its handoff to you:\n\n{handoff}\n\nAgents on the "
                f"forum at the moment: {', '.join(present) if present else 'none'}.{self.role_line()} Continue from the handoff.")

    def take_news(self, force: bool = False) -> list[str]:
        if self.forum.hier:
            return self.court_news(force)
        with self.forum.lock:
            fresh = self.forum.news[self.cursor:]
            self.cursor = len(self.forum.news)
        mine = self.member["id"]
        return [one["text"] for one in fresh if one["audience"] is None or mine in one["audience"]]

    def news_message(self, items: list[str]) -> str:
        if len(items) > 30:
            items = items[-30:] + [f"({len(items) - 30} older items not shown)"]
        quiet = ("- nothing new. Do not wait for news: study a game (walls lists the ones closest to a wall and "
                 "the levels won lately)" if self.forum.walls_mode and self.member.get("role") == "vassal"
                 else "- nothing new")
        return f"[news for {self.me()}, {self.forum.minute()}]\n" + ("\n".join(f"- {one}" for one in items) if items else quiet)

    def chat(self) -> dict[str, Any] | None:
        payload = {"model": self.forum.model, "messages": self.messages, "tools": self.tools(), "tool_choice": "auto",
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
            if self.forum.hier:
                self.member["generated"] = int(self.member.get("generated") or 0) + int(usage.get("completion_tokens") or 0)
                self.member["read"] = int(self.member.get("read") or 0) + max(
                    0, int(usage.get("prompt_tokens") or 0)
                    - int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))
        message = choice.get("message") or {}
        spec = (body.get("metrics") or {}).get("speculative_decoding")
        if isinstance(spec, dict):
            from inference.agent import spec_live
            spec_live.observe("forum", "forum", spec)
        self.context = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        self.log(reply={"content": message.get("content"), "reasoning": message.get("reasoning")
                        or message.get("reasoning_content"), "tool_calls": message.get("tool_calls"),
                        "finish": choice.get("finish_reason")}, usage=usage, context=self.context,
                 generation=self.member["generation"],
                 **({"spec_decode": (body.get("metrics") or {}).get("speculative_decoding"), "label": "forum"}
                    if isinstance((body.get("metrics") or {}).get("speculative_decoding"), dict) else {}))
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
        self.messages = [{"role": "system", "content": self.system_message()},
                         {"role": "user", "content": self.successor()}]
        self.member["handoff"] = ""
        self.context = 0
        self.asked = False
        self.ended = False

    def run(self) -> None:
        forum = self.forum
        self.messages = [{"role": "system", "content": self.system_message()},
                         {"role": "user", "content": self.welcome()}]
        forum.record("arrived", member=self.member["id"])
        try:
            while not forum.stop_event.is_set() and _now() < forum.end_at and not self.member.get("retired"):
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
                        if isinstance(image, list):
                            images += image
                        elif image:
                            images.append(image)
                    self.messages.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": text})
                    self.log(tool=function.get("name"), arguments=arguments, result=text[:4000])
                closing = forum.hier and (self.ended or self.context >= forum.context_tokens)
                items = [] if closing else self.take_news()
                if not calls and not closing:
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
                briefed: list[dict[str, Any]] = []
                for key in [] if closing else forum.take_briefs(self.member["id"]):
                    text, shown = self.game_brief(key)
                    said += "\n\n" + text
                    briefed += shown
                if self.context >= forum.handoff_at and not self.asked:
                    self.asked = True
                    said += (f"\n\n[memory] Your conversation is at {self.context:,} tokens of your "
                             f"{forum.context_tokens:,} limit. Write your handoff now with write_handoff. After it "
                             f"this instance ends and a fresh instance of you continues in your seat from the "
                             f"handoff alone.")
                    forum.record("handoff_asked", member=self.member["id"], context=self.context)
                parts.append({"type": "text", "text": said})
                for image in briefed:
                    parts += [{"type": "text", "text": image["label"]}, {"type": "image_url", "image_url": {"url": image["url"]}}]
                self.messages.append({"role": "user", "content": parts if images or briefed else said})
        finally:
            self.release()
            with forum.lock:
                self.member["alive"] = False
                freed = []
                if self.member.get("retired"):
                    for wall in forum.walls:
                        if not wall["end"] and wall["vassal"] == self.member["id"]:
                            wall["vassal"] = None
                            forum.fief.pop(wall["game"], None)
                            freed.append(wall["game"])
            forum.record("left", member=self.member["id"], name=self.member["name"], walls=freed)
            if freed:
                forum.assign_walls()

    def game_brief(self, key: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            text, shown = self.tool_show_game(key)
        except Exception as exc:
            return f"[game brief: {key}] not available ({type(exc).__name__})", []
        if text.startswith("error:"):
            return f"[{key}] {text}", []
        shown = shown if isinstance(shown, list) else [shown] if shown else []
        text = text.replace("Images attached after the tool results", "Images attached after this message")
        text = text.replace("Current board image attached after the tool results.", "Current board image attached after this message.")
        return f"[game brief: {key}, sent once when the game came to you]\n{text}", shown

    def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        handler = getattr(self, f"tool_{name}", None) if name != "show_game" else None
        if name in COURT_TOOLS and name not in {tool["function"]["name"] for tool in self.tools()}:
            handler = None
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
        generation = found.generation_tail(self.forum.show_generation) if self.forum.show_generation else ""
        similar = self.similar_section(found)
        forum = self.forum
        if forum.walls_mode:
            with forum.lock:
                wall = forum.walls[found.wall] if found.wall is not None else None
            held = f"Wall: {forum.wall_line(wall)}" if wall is not None else "Wall: none, this game is not at a wall"
        else:
            held = f"Claimed by: {forum.holder(found.key) or 'nobody'}"
        from inference.agent.vision_context import frame_to_png_data_url
        if forum.show_frames > 0:
            images, said = self.level_frames(found, history, turns, frame_to_png_data_url)
            tail = said
        else:
            images = [{"label": f"{found.key} current board, step {getattr(current, 'step', '?')}",
                       "url": frame_to_png_data_url(current, upscale=forum.upscale)}]
            tail = ["Current board image attached after the tool results."]
        text = "\n".join([
            f"{found.key}: level {getattr(current, 'level', '?')} of {status['total']}, {status['levels']} won, "
            f"{status['actions']} actions, {status['tokens'] / 1000:.0f}k tokens, {status['minutes']:.0f} min, {status['state']}",
            *_game_meta_lines(found),
            held,
            f"Valid actions right now (the only verbs this game accepts): "
            f"{', '.join(found.valid_actions()) or 'none listed (the game is over or between turns)'}",
            *self.thread_lines(found.key),
            f"Levels won: {won or 'none yet'}",
            f"All actions ({len(moves)}): {_cap(sequence, 2500) or 'none yet'}",
            "Last actions: " + ("; ".join(recent) or "none"),
            *([found.animation_summary()] if found.animation_summary() else []),
            *(["What the player wrote last, as it was written:\n" + (generation or "(nothing yet)")]
              if self.forum.show_generation else []),
            *similar,
            *tail,
        ])
        with forum.lock:
            forum.stats["reads"] += 1
            forum.stats["read_chars"] += len(text)
            forum.stats["images"] += len(images)
        return text, images if forum.show_frames > 0 else images[0]

    def thread_lines(self, key: str) -> list[str]:
        short = key.split("-")[0].lower()
        with self.forum.lock:
            hits = [one for one in self.forum.threads if short in one["title"].lower() or key.lower() in one["title"].lower()]
        if not hits:
            return []
        return ["Forum threads about this game (forum_read):"] + [f"- #{one['id']} '{one['title']}' by {one['author_label']}, "
                                                               f"{len(one['posts'])} post(s)" for one in hits[-6:]]

    def level_frames(self, found: _Game, history: list[Any], turns: list[dict[str, Any]],
                     render: Any) -> tuple[list[dict[str, Any]], list[str]]:
        forum = self.forum
        scale = forum.show_upscale()
        images: list[dict[str, Any]] = []
        lines: list[str] = []
        count = 0
        level_start = 0
        wins = []
        for index, entry in enumerate(history):
            if not getattr(entry, "action", ""):
                continue
            count += 1
            result = dict(getattr(entry, "result", {}) or {})
            if result.get("level_completed"):
                wins.append((index, entry, count, result))
                level_start = index
        for number, (index, entry, action_no, result) in enumerate(wins, 1):
            turn = max((one["turn"] for one in turns if one["action"] < action_no), default=None)
            data = result.get("action_data") or {}
            where = (f" at x={data.get('x')}, y={data.get('y')}" if isinstance(data, dict) and "x" in data and "y" in data
                     else "")
            press = (f"level {number} won by {entry.action}{where}, action {action_no} of the game"
                     + (f", player turn {turn}" if turn is not None else ""))
            lines.append(f"- {press}")
            before = history[index - 1].frame if index > 0 else entry.frame
            images.append({"label": f"{found.key} {press}: the board before that action",
                           "url": render(before, upscale=scale)})
            images.append({"label": f"{found.key} {press}: the board after it", "url": render(entry.frame, upscale=scale)})
        level = list(history[level_start:])
        wanted = forum.show_frames
        if len(level) <= wanted:
            picks = level
        elif wanted == 1:
            picks = level[-1:]
        else:
            picks = [level[round(step * (len(level) - 1) / (wanted - 1))] for step in range(wanted)]
        for entry in picks:
            step = getattr(entry.frame, "step", "?")
            images.append({"label": f"{found.key} current level, board at step {step}"
                           + (f" after {entry.action}" if getattr(entry, "action", "") else ""),
                           "url": render(entry.frame, upscale=scale)})
        tokens = sum(image_tokens(image["url"]) for image in images)
        head = [f"Level wins ({len(wins)}), each with the board before and after the winning action:"] + (
            lines or ["- none yet"])
        head.append(f"Current level: {len(picks)} boards at even intervals over its {len(level)} recorded boards, "
                    f"ending at the latest one.")
        head.append(f"Images attached after the tool results: {len(images)}, about {tokens} prompt tokens in all "
                    f"(upscale {scale}).")
        return images, head

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

    def tool_claim(self, game: str = "") -> tuple[str, Any]:
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
        self.forum.add_news(f"{self.me()} claimed {found.key}", kind="claim", game=found.key, member=mine)
        self.forum.record("claim", game=found.key, member=mine, name=self.member["name"])
        text, shown = self.game_brief(found.key)
        text = text.replace("Images attached after this message", "Images attached after the tool results")
        text = text.replace("Current board image attached after this message.", "Current board image attached after the tool results.")
        return f"you hold {found.key}; release it when you stop looking after it\n\n{text}", shown

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
        self.forum.add_news(f"{self.me()} released {found.key}", kind="claim", game=found.key, member=mine)
        self.forum.record("release", game=found.key, member=mine, name=self.member["name"])
        return f"released {found.key}", None

    def stamp(self, text: str) -> tuple[str, list[str]]:
        body, dropped = strip_signature(text)
        if self.forum.hier:
            body, more = strip_role_signature(body)
            dropped += more
        if dropped:
            self.forum.record("signature_stripped", member=self.member["id"], name=self.member["name"], dropped=dropped)
        return body, dropped

    def tool_news(self) -> tuple[str, None]:
        return self.news_message(self.take_news(True)), None

    def tool_list_games(self) -> tuple[str, None]:
        if self.forum.hier:
            return self.court_list_games(), None
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

    def tool_last_generation(self, game: str = "", tokens: int | None = None) -> tuple[str, Any]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game named {game!r}", None
        want = self.forum.last_generation if tokens is None else max(1, min(int(tokens), 8000))
        text = found.generation_tail(want)
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
        return (f"{found.key}, what the player wrote last, as it was written:\n" + text) if text \
            else f"{found.key}: nothing written yet", None

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

    def tool_read_replies(self, game: str = "", from_entry: int = 0) -> tuple[str, None]:
        forum = self.forum
        key = ""
        if str(game or "").strip():
            found = forum.game(game)
            if found is None:
                return f"error: no game {game!r}; see list_games", None
            key = found.key
        with forum.lock:
            log = [one for one in forum.reply_log if not key or one["game"] == key]
        if not log:
            return (f"no replies yet{' from ' + key if key else ''}; a player replies with Advisors.Reply after a note "
                    f"reaches it"), None
        start = max(0, int(from_entry or 0))
        blocks = []
        for index, one in enumerate(log):
            if index < start:
                continue
            said = chr(10).join(f"{forum.label(advice['member'])} said ({advice['kind']}, {advice['turns']} player "
                                f"turn(s) before the answer):{chr(10)}{advice['text']}" for advice in one["advice"])
            blocks.append(f"[{index}] {one['game']}, minute {(one['t'] - forum.started) / 60:.0f}{chr(10)}{said}"
                          f"{chr(10)}The player answered:{chr(10)}{one['answer']}")
        head = (f"Replies to help{' on ' + key if key else ''}: {len(log)} in all, from entry {start}" + chr(10) * 2)
        text = _cap(head + (chr(10) * 2).join(blocks), forum.read_chars)
        with forum.lock:
            forum.stats["reads"] += 1
            forum.stats["read_chars"] += len(text)
        return text, None

    def tool_read_helper(self, helper: Any = None, tokens: int | None = None) -> tuple[str, None]:
        forum = self.forum
        member = forum.resolve(helper)
        if member is None:
            return f"error: no helper {helper!r}", None
        found = forum.helper_objects.get(member["id"])
        if found is None:
            return f"{forum.label(member['id'])}: nothing generated yet", None
        limit = max(1, min(int(tokens or forum.last_generation), 8000)) * 4
        parts: list[str] = []
        size = 0
        for message in reversed(list(found.messages)):
            if message.get("role") != "assistant":
                continue
            chunk = []
            if message.get("reasoning"):
                chunk.append("[THINKING]" + chr(10) + str(message["reasoning"]).strip())
            if str(message.get("content") or "").strip():
                chunk.append("[TEXT]" + chr(10) + str(message["content"]).strip())
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                chunk.append(f"[TOOL CALL: {function.get('name')}] {_cap(str(function.get('arguments') or ''), 600)}")
            if not chunk:
                continue
            text = chr(10).join(chunk)
            parts.append(text)
            size += len(text)
            if size >= limit:
                break
        if not parts:
            return f"{forum.label(member['id'])}: nothing generated yet", None
        body = (chr(10) * 2).join(reversed(parts))
        if len(body) > limit:
            body = "...(earlier part left out)" + chr(10) + body[-limit:]
        text = f"{forum.label(member['id'])}, the end of its latest work, newest last:" + chr(10) + body
        with forum.lock:
            forum.stats["reads"] += 1
            forum.stats["read_chars"] += len(text)
        return text, None

    def tool_view_animation(self, game: str = "", step: int | None = None) -> tuple[str, Any]:
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        kept = found.animation_steps()
        if not kept:
            return (f"{found.key}: no animation kept yet; an action whose board changes over several frames leaves "
                    f"one, and {self.forum.animations_kept} of them are kept, the latest of each kind (press and "
                    f"length)"), None
        record = found.animation(None if step is None else int(step))
        if record is None:
            return f"error: no animation cached for step {step}; cached steps: {', '.join(map(str, kept))}", None
        from inference.agent.vision_context import frame_to_png_data_url
        chain = record["chain"]
        wanted = min(len(chain), self.forum.animation_frames)
        picks = sorted({round(index * (len(chain) - 1) / (wanted - 1)) for index in range(wanted)})
        scale = self.forum.show_upscale()
        first = ("the board this RESET settled on (no board came before it)" if record.get("cause") == "start"
                 else "the board before the action")
        images = [{"label": f"{found.key} step {record['step']} {record['action_display']}: "
                            + (first if index == 0
                               else f"animation board {index} of {len(chain) - 1}"),
                   "url": frame_to_png_data_url(SimpleNamespace(grid=chain[index]), upscale=scale)} for index in picks]
        _, history = found.frames()
        result = next((dict(getattr(entry, "result", {}) or {}) for entry in history
                       if getattr(entry, "action", "") and int(getattr(entry.frame, "step", -1)) == record["step"]), {})
        outcome = ("LEVEL COMPLETED" if result.get("level_completed")
                   else "GAME OVER (level restarted)" if result.get("game_over") else "")
        try:
            from inference.utils.animation import build_animation_view
            timeline = json.dumps(build_animation_view(record), separators=(",", ":"))
        except Exception as exc:
            timeline = f"(no timeline: {type(exc).__name__})"
        text = chr(10).join([
            f"{found.key} step {record['step']}, level {record['level']}: {record['action_display']}"
            + (f", {outcome}" if outcome else "") + f"; {len(chain) - 1} boards before it settled"
            + (f", over {record['lasted']}" if record.get("lasted") else ""),
            f"Images attached after the tool results: {len(images)} ({first}, then animation "
            f"boards at even intervals, ending at the settled one).",
            f"Other cached animations: {', '.join(str(one) for one in kept if one != record['step']) or 'none'}",
            "Cells changed at each animation board: " + _cap(timeline, 3000),
        ])
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
            self.forum.stats["images"] += len(images)
        return text, images

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
            for turn in one.parse():
                for label, _extra, lines in turn["sections"]:
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
        with self.forum.lock:
            near = next((one for one in self.forum.threads
                         if difflib.SequenceMatcher(None, one["title"].lower(), title.lower()).ratio() >= 0.75), None)
        if near is not None:
            return (f"error: thread #{near['id']} '{near['title']}' is already about this; add to it with forum_post"), None
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
            self.forum.add_news(f"new thread #{thread['id']} '{title}' by {label}: read it with forum_read, and if you "
                                f"know something it lacks or something that contradicts it, add a post with forum_post",
                                kind="thread",
                                member=self.member["id"])
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
            self.forum.add_news(f"{label} replied in #{found['id']} '{found['title']}' (post {index})", readers,
                                kind="post")
        self.forum.record("post", thread=int(thread), index=index, name=name, order=self.member["order"], text=text)
        self.forum.say(f"POST #{thread}.{index} '{found['title']}' by {label}:\n{_quote(text)}")
        return f"posted as #{thread} post {index}, signed by the harness as {label}" + self.dropped_line(dropped), None

    def tool_send_note(self, game: str = "", text: str = "", **extra: Any) -> tuple[str, None]:
        topic = self.topic("tool_send_note", extra)
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        refused = self.wall_refusal(found)
        if refused:
            return refused, None
        text, dropped = self.stamp(str(text).strip())
        if not text:
            return "error: empty note", None
        text, cut = _fit(text, self.forum.note_chars)
        return self.queue(found, text, dropped, None, topic) + cut, None

    def queue(self, found: _Game, text: str, dropped: list[str], function: dict[str, str] | None,
              topic: str = "") -> str:
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
                                                        if function else {}), **({"kind": topic} if topic else {})})
            self.member["sent"].append({"game": found.key, "flat": flat})
            self.forum.stats["notes"] += 1
            if function:
                self.forum.stats["functions"] += 1
            self.forum.add_news(f"{label} sent a {kind} to {found.key}", kind="note", game=found.key,
                                member=self.member["id"])
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

    def tool_send_function(self, game: str = "", name: str = "", code: str = "", message: str = "",
                           **extra: Any) -> tuple[str, None]:
        topic = self.topic("tool_send_function", extra)
        found = self.forum.game(game)
        if found is None:
            return f"error: no game {game!r}; see list_games", None
        if not _functions_kept():
            return "error: the players' Python keeps no functions between calls in this run; send a note instead", None
        refused = self.wall_refusal(found)
        if refused:
            return refused, None
        name, code = str(name).strip(), str(code).strip("\n")
        text, dropped = self.stamp(str(message).strip())
        if not code.strip():
            return "error: code is missing: send the function and its trial line in code", None
        if not name:
            try:
                names = [node.name for node in ast.parse(code).body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            except SyntaxError:
                names = []
            name = names[0] if len(names) == 1 else ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,60}", name):
            return (f"error: name is missing or not a Python identifier ({name!r}); name the one top-level "
                    f"function the code defines"), None
        if not text:
            return "error: a function needs a short message saying what it does and when to use it", None
        text, cut = _fit(text, self.forum.note_chars)
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
            if required > 1:
                return (f"error: add a trial after the definition: a line that calls {name}(...) on this game's "
                        f"boards and prints the result; it runs only here, never in the player's sandbox"), None
            trial = code + (f"\nprint({name}(current_frame))" if required else f"\nprint({name}())")
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
        said = self.queue(found, text, dropped, {"name": name, "source": kept[name]}, topic)
        return f"trial output:\n{out}\n\n{said}{cut}", None

    def tools(self) -> list[dict[str, Any]]:
        forum = self.forum
        if forum.hier:
            tools = _court_tools(self.me(), self.member["role"], not forum.princes, forum.walls_mode,
                                 forum.similar_top > 0, forum.walls_per, forum.effect_turns)
        else:
            tools = _tools(self.me(), forum.similar_top > 0)
        first = tools[0]["function"]
        if not first["description"].startswith(f"[{self.me()}] "):
            first["description"] = f"[{self.me()}] " + first["description"]
        for tool in tools:
            tool["function"]["description"] = tool["function"]["description"].replace(
                "__NOTE_CHARS__", str(forum.note_chars))
        return tools

    def system_message(self) -> str:
        forum = self.forum
        if forum.hier:
            text = (f"You are {self.me()}." + chr(10)
                    + court_system_text(forum.context_tokens, forum.handoff_at, self.member["role"], forum.vassals_per,
                                        forum.seat_cap(), not forum.princes, forum.walls_mode, forum.grow,
                                        forum.similar_top > 0, forum.walls_per, forum.effect_turns, forum.wall_tokens,
                                        forum.wall_actions, forum.player_api))
        else:
            text = f"You are {self.me()}." + chr(10) + system_text(forum.context_tokens, forum.handoff_at,
                                                                   forum.similar_top > 0)
        if forum.show_frames > 0:
            framed = "- " + _show_framed(forum.show_frames)
            text = text.replace(SHOW_SIMILAR, framed).replace(SHOW_PLAIN, framed).replace(
                WALLS_SHOW, WALLS_FRAMED.replace("__N__", str(forum.show_frames)))
        return text

    def topic(self, name: str, extra: dict[str, Any]) -> str:
        unknown = [key for key in extra if key != "kind" or not self.forum.hier]
        if unknown:
            raise TypeError(f"Helper.{name}() got an unexpected keyword argument '{unknown[0]}'")
        return " ".join(str(extra.get("kind") or "").lower().split())[:40]

    def position(self) -> str:
        forum = self.forum
        role = self.member["role"]
        mine = forum.games_of(self.member["id"])
        if role == "king" and forum.walls_mode:
            vassals = [forum.label(one["id"]) for one in forum.vassals_of(0)]
            return (f"Your vassals: {', '.join(vassals) or 'none yet'} ({forum.start_vassals} in all as they arrive)."
                    f"\n{forum.walls_text(None)}\n")
        if role == "king" and not forum.princes:
            vassals = [forum.label(one["id"]) for one in forum.vassals_of(0)]
            return (f"Your vassals: {', '.join(vassals) or 'none yet'} ({forum.start_vassals} in all as they arrive). "
                    f"Each new vassal first gets a share of the open games; change the division with assign as you judge.")
        if role == "vassal" and forum.walls_mode:
            return (f"Your king: {forum.label(0)}.\n{forum.walls_text(self.member['id'])}\n"
                    + ("" if mine else "You hold no wall yet; the harness or the king assigns walls as they appear.\n"))
        if role == "king":
            with forum.lock:
                princes = [one["id"] for one in forum.members if one.get("role") == "prince"]
            return (f"Your princes so far: {', '.join(forum.label(one) for one in princes) or 'none yet'} "
                    f"({forum.princes} princes in all as they arrive). The harness first split the games in turn among "
                    f"the princes; change the domains with assign as you judge.")
        if role == "prince":
            vassals = [forum.label(one["id"]) for one in forum.vassals_of(self.member["id"])]
            return (f"Your king: {forum.label(0)}. Your domain: {len(mine)} games ({_keys(mine, 8)}). Your vassals: "
                    f"{', '.join(vassals) or 'none yet'}.")
        title = "king" if self.member["lord"] == 0 else "prince"
        return (f"Your {title}: {forum.label(self.member['lord'])}. Your games: {len(mine)}"
                + (f" ({_keys(mine, 8)})." if mine else f"; your {title} has not assigned you any yet."))

    def orders_text(self) -> str:
        orders = list(self.member.get("orders") or [])
        if not orders:
            return ""
        return " Standing orders of your seat, newest last:\n" + "\n".join(
            f"- from {one['from']} at minute {one['minute']}: {one['text']}" for one in orders) + "\n"

    def court_welcome(self) -> str:
        forum = self.forum
        with forum.lock:
            threads = [f"#{one['id']} {one['title']} ({len(one['posts'])} posts)" for one in forum.threads[-15:]]
        present = self.others()
        return (f"{forum.page_text()}\n\nYou are {self.me()}, a new helper agent, just arrived ({forum.minute()}). Your "
                f"name is {self.member['name']}; the harness signs everything you post or send with your name, role "
                f"and seat.{self.freed_line()} {self.position()}{self.orders_text()} Agents on the forum at the moment: "
                f"{', '.join(present) if present else 'none yet, you are among the first'}. "
                f"Forum threads: {'; '.join(threads) if threads else 'none yet'}.")

    def court_successor(self) -> str:
        forum = self.forum
        present = self.others()
        handoff = self.member["handoff"] or "(your previous instance wrote no handoff)"
        return (f"{forum.page_text()}\n\nYou are {self.me()}, a helper agent, instance {self.member['generation']} of "
                f"your seat ({forum.minute()}). Your seat keeps its role, {self.member['role']}, its superior, "
                f"{'its walls' if forum.walls_mode else 'its games'} and its standing orders. {self.position()}{self.orders_text()} Your previous instance in this seat "
                f"filled its memory and ended. Its handoff to you:\n\n{handoff}\n\nAgents on the forum at the moment: "
                f"{', '.join(present) if present else 'none'}. Continue from the handoff.")

    def court_item(self, item: dict[str, Any], mine: set[str]) -> tuple[bool, str, str, str] | None:
        forum = self.forum
        me = self.member["id"]
        role = self.member["role"]
        kind = item.get("kind")
        game = str(item.get("game") or "")
        text = item["text"]
        addressed = item["audience"] is not None and me in item["audience"]
        if kind in ("order", "report", "assign", "worse", "wall", "reply", "ask"):
            return (True, text, "", game) if addressed else None
        if kind == "thread":
            return (True, text, "", game) if item.get("member") != me else None
        if kind == "harness_assign":
            return (False, "", "walls given by the harness (walls)", game) if addressed else None
        if kind == "budget":
            return True, text, "", ""
        if kind == "stuck":
            return (True, text, "", game) if role != "king" and game in mine else None
        if kind in ("outcome", "domain_outcome", "post", "effect"):
            return (False, text, "", game) if addressed else None
        if kind == "delivered":
            return (False, "", "of your notes and functions delivered", game) if addressed else None
        if kind == "thread":
            if forum.walls_mode and role == "vassal":
                return None
            return (False, "", "new forum threads (forum_list)", "") if item.get("member") != me else None
        if kind == "page":
            return (False, "", "edits of the king's page (read_page)", "") if role != "king" else None
        if role == "king":
            return None
        if kind == "arrived":
            with forum.lock:
                lord = forum.members[item["member"]].get("lord") if item["member"] < len(forum.members) else None
            return (False, text, "", "") if role == "prince" and lord == me and item["member"] != me else None
        if kind in ("level", "ended", "note", "claim") and game not in mine:
            return None
        if kind == "level":
            return (False, text, "", game) if role == "vassal" else (False, "", "levels won in your domain", game)
        if kind == "ended":
            return False, text, "", game
        if kind == "note":
            if item.get("member") == me:
                return None
            return (False, text, "", game) if role == "vassal" else (False, "", "notes and functions sent to your "
                                                                      "domain by others", game)
        if kind == "claim":
            return None if item.get("member") == me else (False, "", "claims and releases in your games", game)
        return (False, text, "", game) if item["audience"] is None or addressed else None

    def court_news(self, force: bool) -> list[str]:
        forum = self.forum
        with forum.lock:
            fresh = forum.news[self.cursor:]
            self.cursor = len(forum.news)
        self.news_calls += 1
        mine = set(forum.games_of(self.member["id"]))
        for item in fresh:
            routed = self.court_item(item, mine)
            if routed:
                self.backlog.append(routed)
        due = force or self.news_calls >= forum.news_every
        if due:
            self.news_calls = 0
        if not due and not any(one[0] for one in self.backlog):
            return []
        keep: list[tuple[bool, str, str, str]] = []
        counts: dict[str, list[str]] = {}
        picked = []
        for one in [one for one in self.backlog if one[0]] + [one for one in self.backlog if not one[0]]:
            if not due and not one[0]:
                keep.append(one)
            elif one[2]:
                counts.setdefault(one[2], []).append(one[3])
            elif one[1] not in self.shown:
                picked.append(one)
        limit = forum.news_lines
        out: list[str] = []
        for one in [one for one in picked if one[0]]:
            if len(out) < limit:
                out.append(one[1])
                self.shown.add(one[1])
            else:
                keep.append(one)
        if due:
            numbers = forum.numbers(self.member)
            fresh_numbers = [line for line in numbers if line not in self.last_numbers]
            room = limit - len(out)
            if fresh_numbers and len(fresh_numbers) <= room:
                out += fresh_numbers
                self.last_numbers = numbers
            if counts and len(out) < limit:
                parts = []
                for name, games in counts.items():
                    keys = sorted({key for key in games if key})
                    parts.append(f"{len(games)} {name}" + (f" ({_keys(keys)})" if keys else ""))
                out.append("since your last news: " + "; ".join(parts))
            elif counts:
                keep += [(False, "", name, key) for name, games in counts.items() for key in games]
        for one in [one for one in picked if not one[0]]:
            if len(out) < limit:
                out.append(one[1])
                self.shown.add(one[1])
            else:
                keep.append(one)
        self.backlog = keep
        if keep:
            out.append(f"({len(keep)} more items wait for your next news)")
        return out

    def court_list_games(self) -> str:
        forum = self.forum
        role = self.member["role"]
        keys = forum.games_of(self.member["id"])
        lines = []
        for key in keys:
            game = forum.games.get(key)
            if game is None:
                continue
            status = game.status()
            with forum.lock:
                prince = forum.domain.get(key)
                vassal = forum.fief.get(key)
            names = (f"prince {forum.members[prince]['name'] if prince is not None and prince < len(forum.members) else '-'}"
                     f", vassal {forum.members[vassal]['name'] if vassal is not None else '-'}")
            if not forum.princes:
                names = f"vassal {forum.members[vassal]['name'] if vassal is not None else '-'}"
            if forum.walls_mode:
                with forum.lock:
                    wall = forum.walls[game.wall] if game.wall is not None else None
                names = (f"at a wall for {(_now() - wall['t']) / 60:.0f} min, {names}" if wall is not None
                         else "not at a wall")
            if role == "king":
                lines.append(f"{key}: levels {status['levels']}/{status['total']}, actions {status['actions']}, "
                             f"{status['state']}, {names}")
                continue
            holder = forum.holder(key)
            lines.append(f"{key}: levels {status['levels']}/{status['total']}, actions {status['actions']} "
                         f"(per level {status['per_level']}), {status['tokens'] / 1000:.0f}k tokens, "
                         f"{status['minutes']:.0f} min, {status['state']}, notes waiting {len(game.notes_waiting)}, "
                         f"delivered {game.notes_delivered}, claimed by {holder or 'nobody'}"
                         + (f", vassal {forum.members[vassal]['name'] if vassal is not None else '-'}"
                            if role == "prince" else ""))
        empty = {"king": "no games yet", "prince": "no games in your domain yet; the king assigns them",
                 "vassal": ("no walls assigned to you yet; the harness or the king assigns them as they appear"
                            if forum.walls_mode else "no games assigned to you yet; your "
                            + ("king" if self.member.get("lord") == 0 else "prince") + " assigns them")}[role]
        return "\n".join(lines) or empty

    def subject(self, ref: Any) -> tuple[dict[str, Any] | None, str]:
        forum = self.forum
        role = self.member["role"]
        princely = role == "king" and forum.princes > 0
        want = "prince" if princely else "vassal"
        mine = ([one for one in forum.members if one.get("role") == "prince"] if princely
                else forum.vassals_of(self.member["id"]))
        target = forum.resolve(ref)
        if target is None or all(one["id"] != target["id"] for one in mine):
            return None, (f"error: {str(ref)!r} is not one of your {want}s; yours: "
                          f"{', '.join(forum.label(one['id']) for one in mine) or 'none yet'}")
        return target, ""

    def tool_assign(self, prince: Any = None, vassal: Any = None, games: Any = None,
                    replace: bool = False) -> tuple[str, None]:
        forum = self.forum
        me = self.member["id"]
        king = self.member["role"] == "king" and forum.princes > 0
        target, error = self.subject(prince if prince is not None else vassal)
        if target is None:
            return error, None
        if isinstance(games, str):
            games = [one for one in re.split(r"[,\s]+", games) if one]
        keys, unknown = [], []
        for one in games or []:
            found = forum.game(one)
            if found is None:
                unknown.append(str(one))
            elif found.key not in keys:
                keys.append(found.key)
        if unknown:
            return f"error: no such games: {_keys(unknown, 8)}; nothing was assigned", None
        if forum.walls_mode:
            return self.assign_walls_to(target, keys, bool(replace)), None
        if not king and self.member["role"] != "king":
            outside = [key for key in keys if forum.domain.get(key) != me]
            if outside:
                return f"error: not in your domain: {_keys(outside, 8)}; nothing was assigned", None
        tid = target["id"]
        moved: dict[int, list[str]] = {}
        dropped: list[str] = []
        with forum.lock:
            table = forum.domain if king else forum.fief
            for key in keys:
                old = table.get(key)
                if old is not None and old != tid:
                    moved.setdefault(old, []).append(key)
                table[key] = tid
                if king and key in forum.fief and forum.members[forum.fief[key]].get("lord") != tid:
                    del forum.fief[key]
            if replace:
                for key, owner in list(table.items()):
                    if owner == tid and key not in keys:
                        del table[key]
                        dropped.append(key)
                        if king:
                            forum.fief.pop(key, None)
            held = sum(1 for owner in table.values() if owner == tid)
        where = "domain" if king else "games"
        forum.add_news(f"{self.me()} assigned you {len(keys)} game(s): {_keys(keys, 8)}"
                       + (f"; {len(dropped)} other game(s) left your {where}" if dropped else "")
                       + f"; you hold {held} now", {tid}, kind="assign")
        for old, lost in moved.items():
            forum.add_news(f"{self.me()} moved {len(lost)} game(s) from your {where} to {forum.label(tid)}: "
                           f"{_keys(lost)}", {old}, kind="assign")
        forum.record("assign", by=me, to=tid, games=keys, replace=bool(replace), dropped=dropped,
                     moved={str(old): lost for old, lost in moved.items()})
        forum.say(f"ASSIGN {self.me()} -> {forum.label(tid)}: {len(keys)} game(s){' (replace)' if replace else ''}, "
                  f"holds {held}")
        return (f"{forum.label(tid)} now holds {held} game(s) in its {where}"
                + (f"; {len(dropped)} left it" if dropped else "")), None

    def tool_order(self, prince: Any = None, vassal: Any = None, text: str = "") -> tuple[str, None]:
        forum = self.forum
        target, error = self.subject(prince if prince is not None else vassal)
        if target is None:
            return error, None
        text, dropped = self.stamp(str(text).strip())
        if not text:
            return "error: an empty order", None
        if len(text) > 1500:
            return f"error: an order is at most 1500 characters, this one is {len(text)}", None
        with forum.lock:
            target["orders"] = (list(target.get("orders") or []) + [
                {"from": self.me(), "minute": round((_now() - forum.started) / 60.0), "text": text}])[-6:]
        forum.add_news(f"ORDER from {self.me()}: {text}", {target["id"]}, kind="order")
        forum.record("order", by=self.member["id"], to=target["id"], text=text)
        forum.say(f"ORDER {self.me()} -> {forum.label(target['id'])}:\n{_quote(text)}")
        return (f"order delivered to {forum.label(target['id'])} and kept with its seat"
                + self.dropped_line(dropped)), None

    def tool_report(self, text: str = "") -> tuple[str, None]:
        forum = self.forum
        lord = self.member.get("lord")
        if lord is None:
            return "error: the king reports to nobody", None
        text, dropped = self.stamp(str(text).strip())
        if not text:
            return "error: an empty report", None
        text, cut = _fit(text, 1500)
        forum.add_news(f"REPORT from {self.me()}: {text}", {lord}, kind="report")
        forum.record("report", by=self.member["id"], to=lord, text=text)
        forum.say(f"REPORT {self.me()} -> {forum.label(lord)}:\n{_quote(text)}")
        return f"report delivered to {forum.label(lord)}" + self.dropped_line(dropped) + cut, None

    def assign_walls_to(self, target: dict[str, Any], keys: list[str], replace: bool) -> str:
        forum = self.forum
        tid = target["id"]
        with forum.lock:
            walls = {key: forum.walls[forum.games[key].wall] for key in keys if forum.games[key].wall is not None}
            off = [key for key in keys if key not in walls]
            if off:
                return (f"error: not at a wall: {_keys(off, 8)}; games that are not at a wall are not the helpers' "
                        f"business; nothing was assigned")
            kept = [] if replace else [wall["game"] for wall in forum.walls if not wall["end"] and wall["vassal"] == tid
                                       and wall["game"] not in walls]
            if len(kept) + len(walls) > forum.walls_per:
                return (f"error: a vassal holds at most {forum.walls_per} walls; {forum.label(tid)} would hold "
                        f"{len(kept) + len(walls)}; nothing was assigned")
            moved: dict[int, list[str]] = {}
            dropped = []
            for key, wall in walls.items():
                old = wall["vassal"]
                if old is not None and old != tid:
                    moved.setdefault(old, []).append(key)
                wall["vassal"] = tid
                if tid not in wall["vassals"]:
                    wall["vassals"].append(tid)
                forum.fief[key] = tid
            if replace:
                for wall in forum.walls:
                    if not wall["end"] and wall["vassal"] == tid and wall["game"] not in walls:
                        wall["vassal"] = None
                        forum.fief.pop(wall["game"], None)
                        dropped.append(wall["game"])
        lines = [forum.wall_line(wall) for wall in walls.values()]
        for key in walls:
            forum.brief(tid, key)
        forum.add_news(f"{self.me()} assigned you {len(walls)} wall(s):\n" + "\n".join(f"  {line}" for line in lines)
                       + (f"\n  you no longer hold: {_keys(dropped)}" if dropped else ""), {tid}, kind="assign")
        for old, lost in moved.items():
            forum.add_news(f"{self.me()} moved your wall(s) {_keys(lost)} to {forum.label(tid)}; stop sending help to "
                           f"them", {old}, kind="assign")
        forum.record("assign", by=self.member["id"], to=tid, games=list(walls), replace=replace, dropped=dropped,
                     moved={str(old): lost for old, lost in moved.items()})
        forum.say(f"ASSIGN {self.me()} -> {forum.label(tid)}: walls {_keys(list(walls), 8)}"
                  + (f", dropped {_keys(dropped)}" if dropped else ""))
        return (f"{forum.label(tid)} now holds {forum.wall_load(tid)} wall(s)"
                + (f"; {len(dropped)} left it and have no vassal" if dropped else ""))

    def tool_walls(self) -> tuple[str, None]:
        forum = self.forum
        if self.member["role"] != "king" and forum.walls_mode:
            until = _now() + forum.wall_wait_s
            while not forum.games_of(self.member["id"]) and _now() < until and not forum.stop_event.wait(2.0):
                pass
        text = forum.walls_text(None if self.member["role"] == "king" else self.member["id"])
        with forum.lock:
            forum.stats["reads"] += 1
            forum.stats["read_chars"] += len(text)
        return text, None

    def wall_refusal(self, found: _Game) -> str:
        forum = self.forum
        if not forum.walls_mode:
            return ""
        if self.member["role"] == "king":
            return ("error: the king does not write to the players; put a vassal on this wall with assign or give "
                    "an order")
        with forum.lock:
            wall = forum.walls[found.wall] if found.wall is not None else None
        if wall is None:
            return (f"error: {found.key} is not a wall; games that no vassal holds are not the helpers' business; "
                    f"nothing was sent")
        if self.member["role"] == "vassal" and wall["vassal"] != self.member["id"]:
            mine = forum.games_of(self.member["id"])
            return (f"error: the wall {found.key} is not yours; your walls: {_keys(mine) if mine else 'none'}; "
                    f"nothing was sent")
        return ""

    def tool_metrics(self) -> tuple[str, None]:
        text = self.forum.metrics_text()
        with self.forum.lock:
            self.forum.stats["reads"] += 1
            self.forum.stats["read_chars"] += len(text)
        return text, None

    def tool_edit_page(self, text: str = "") -> tuple[str, None]:
        forum = self.forum
        text, dropped = self.stamp(str(text).strip())
        if not text:
            return "error: an empty page", None
        if len(text) > 4000:
            return f"error: the page is at most 4000 characters, this one is {len(text)}", None
        since = _now() - float(forum.page.get("t") or 0)
        if forum.page.get("edits") and since < forum.page_every_s:
            return (f"error: the page was edited {since / 60:.0f} min ago; edit it at most once every "
                    f"{forum.page_every_s / 60:.0f} min, and give what cannot wait as an order"), None
        with forum.lock:
            forum.page.update({"text": text, "t": _now(), "edits": forum.page["edits"] + 1})
        forum.add_news(f"{self.me()} edited the king's page", kind="page")
        forum.record("page", by=self.member["id"], text=text)
        forum.say(f"PAGE edited by {self.me()}:\n{_quote(text)}")
        return "page saved; every new seat reads it first" + self.dropped_line(dropped), None

    def tool_read_page(self) -> tuple[str, None]:
        return self.forum.page_text(), None

    def sandbox(self, found: _Game, code: str, kept: list[str] | None) -> dict[str, Any]:
        from inference.agent import tool_agent
        from inference.agent.python_tool_sandbox import run_sandboxed_python
        current, history = found.frames()
        player = found.player()
        last = getattr(player, "_last_action_call_result", None)
        state = {"current_frame": tool_agent._ascii_frame_view_payload(current),
                 "history": tool_agent._ascii_history_view_payload(history),
                 "valid_actions": found.valid_actions(),
                 "last_action_call_result": dict(last) if isinstance(last, dict) else {}}
        if _env_flag("ARC3_DEATH_LEDGER"):
            try:
                state["death_ledger"] = player._death_ledger_payload()
            except Exception:
                pass

        def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("helpers cannot act")

        if kept is None:
            try:
                kept = player._retained_sources()
            except Exception:
                kept = None
        extra: dict[str, Any] = {} if kept is None else {"kept_functions": kept,
                                                          "retain_imports": _env_flag("ARC3_PERSISTENT_FUNCTIONS_IMPORTS")}
        step = getattr(player, "_step_env_callback", None)
        animated = getattr(tool_agent, "_animation_enabled", lambda: False)()
        if step is not None and animated:
            def animation(request: dict[str, Any]) -> dict[str, Any]:
                kind = str(request.get("kind") or "")
                raw = step({"query": "reset_animation" if kind.startswith("reset_") else "animation"})
                record = raw.get("record") if isinstance(raw, dict) else None
                record = record if isinstance(record, dict) else None
                if kind in ("frames", "reset_frames"):
                    return tool_agent.build_animation_frames(record)
                if not tool_agent._animation_timeline_enabled():
                    return {}
                return tool_agent.build_animation_view(record)

            extra["animation_handler"] = animation
        timeout = int(getattr(player, "_python_timeout", 20) or 20)
        with self.forum.python_slots:
            result = run_sandboxed_python(code=str(code), timeout_seconds=timeout, initial_state=state,
                                          action_handler=refuse, **extra)
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


COURT_TOOLS = {"read_helper", "assign", "order", "metrics", "edit_page", "read_page", "report", "walls", "claim", "release"}

PLAYER_LINE = ("Helper agents are watching every game being played, including levels already won in other games. "
               "They can see your board and your reasoning and may send you a short note to help; a note appears "
               "at the start of a turn under 'Notes from helper agents'. Check each note against your own evidence.")
PLAYER_API_LINE = ("Your Python also has Advisors.Reply(text) and Advisors.Ask(question). After every helper note or "
                   "function you receive, call Advisors.Reply with short feedback: whether it helped and what you tried. "
                   "When you are stuck on a level, call Advisors.Ask with your question; an answer arrives as a helper note.")
NOTE_REPLY_LINE = "Once you have tried them, give short feedback with Advisors.Reply(text)."


def player_line() -> str:
    if ACTIVE is None:
        return ""
    if hasattr(ACTIVE, "player_line"):
        return ACTIVE.player_line()
    return "\n\n" + PLAYER_LINE + (" " + PLAYER_API_LINE if ACTIVE.player_api else "")


def advisor_handler(key: str) -> Any:
    forum = ACTIVE
    if forum is None or not key or not forum.player_api:
        return None

    def handle(kind: str, text: str) -> str:
        try:
            return forum.advisor(key, kind, text)
        except Exception as exc:
            _log(f"forum: an Advisors call from {key} failed, {type(exc).__name__}: {exc}")
            return "Advisors: the message could not be delivered."

    return handle


def register(session: Any, key: str) -> None:
    forum = ACTIVE
    if forum is not None:
        forum.add_game(session, key)


FIRST_CHUNK: dict[str, Any] = {}


def fork_seats() -> int | None:
    forum = ACTIVE
    if forum is None:
        return None
    return forum.fork_slots()


def first_chunk_hook(session: str) -> Any:
    for key, hook in list(FIRST_CHUNK.items()):
        if session.endswith(f"-{key}#forum"):
            return hook
    return None


def _slots_call(name: str, *args: Any) -> Any:
    call = getattr(ACTIVE, name, None)
    if call is None:
        return None
    try:
        return call(*args)
    except Exception as exc:
        _log(f"slots: {name} failed, {type(exc).__name__}: {exc}")
        return None


def player_reply(key: str, *args: Any) -> None:
    if key:
        _slots_call("player_reply", key, *args)


def malformed(key: str, count: int) -> None:
    _slots_call("malformed", key, count)


def player_cut(key: str) -> None:
    if key:
        _slots_call("player_cut", key)


def player_session(key: str, session: str) -> None:
    if key:
        _slots_call("player_session", key, session)


def before_request(key: str, turn: int) -> str | None:
    return _slots_call("before_request", key, turn) if key else None


def take_paused(key: str) -> str | None:
    return _slots_call("take_paused", key) if key else None


def player_context() -> tuple[int, int] | None:
    forum = ACTIVE
    if forum is None:
        return None
    try:
        return forum.player_context()
    except Exception:
        return None


def finished(key: str) -> None:
    forum = ACTIVE
    if forum is None or not key:
        return
    try:
        forum.finished(key)
    except Exception as exc:
        _log(f"forum: the end of {key} was not handled, {type(exc).__name__}: {exc}")


def deliver(key: str, message: dict[str, Any]) -> dict[str, Any]:
    forum = ACTIVE
    if forum is None or not key:
        return message
    try:
        return forum.deliver(key, message)
    except Exception as exc:
        _log(f"forum: delivery to {key} failed, {type(exc).__name__}: {exc}")
        return message


def create(root: Path, base_url: str, model: str, end_at: float, helpers: int | None = None) -> Any:
    global ACTIVE
    if _env_flag("ARC3_HELPER_SLOTS"):
        from inference.agent import helper_slots
        ACTIVE = helper_slots.create(root, base_url, model, end_at)
        return ACTIVE
    count = _env_int("ARC3_FORUM_HELPERS", 0) if helpers is None else helpers
    ACTIVE = Forum(root, base_url, model, count, end_at)
    return ACTIVE


def _frame_objects_text(value):
    try:
        from inference.agent import frame_objects
    except ImportError:
        return value
    return frame_objects.text(value)
