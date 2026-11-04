import ast
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import requests

from inference.agent import forum_agents as fa

PAUSE_MARK = "taaf_pause_now"
COLORS = ("white", "light gray", "gray", "dark gray", "charcoal", "black", "magenta", "pink", "red", "blue",
          "sky blue", "yellow", "orange", "dark red", "light green", "purple")
CLICK = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")
SAFE_MODULES = ("bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math",
                "operator", "random", "re", "statistics", "string")

SYSTEM = """You are a helper for one player of a puzzle game. The player is another instance of your model, playing alone through Python: it sees the board as text and segmented objects and an image each turn, and acts with action([...]). It is stuck, and your goal is to help it win the level it is on, with as few actions as possible.

You cannot act in the game. You can read the player's boards and its own words, run Python on its boards with the player's frame API, search everything the player has written, and send it a note or a Python function. A note or function reaches the player at the start of its next turn, marked with the step it was written at; notes not yet delivered wait in order and all arrive together.

Players and helpers both stick to wrong theories and mechanics. A partly wrong hypothesis can close a level for good, and a belief once accepted is hard to give up, including your own and those of earlier helpers. If an exhaustive search or a long series of attempts found no solution, one of its assumptions is false: name the assumptions that no action has ever tested, and get the cheapest one tested. Never tell the player to stop, wait or hold its position.

The player's questions to the helpers reach you; answer each with a note. You stay with this game across its levels until your memory is full or the game ends; what you learned carries over to the next level.

Every message you get ends with a context line: how much of your 140K-token memory is used. When it reaches its limit you are given one last turn to write the player a note, and then you end.

Tools:
- send_note: a short note to the player.
- send_function: one Python function with a trial call and a short message; the player can call it from its own Python.
- run_python: Python on the player's boards (current_frame, history, segmentation), as the player sees them; action() is disabled.
- search: a regular expression over everything the player has written (its thinking, code and code output).
- view_animation: the boards one action passed through before it settled, as images, with what changed at each and how long it plays.
"""

START = ("You now help {game}: level {level} of {total}, {actions} actions so far, {spent} on this level. Below: the "
         "boards of the game, each after the action named above it with the number of cells it changed, the level "
         "wins and deaths marked, then frames of the animations of its wins and deaths, then the end of what the "
         "player has written, then every note and function earlier helpers sent this player and its answers.")
UPDATE = ("The player took a turn: {done}. The end of what it has written since:\n{tail}\nIts valid actions now: "
          "{actions}. Its board now:")
SANDBOX = ("Your run_python runs the player's own Python sandbox on the player's boards, with the same variables and "
           "imports; action() and Advisors raise an error there, because helpers cannot act. Functions you define "
           "with def that run without an error stay defined in your later run_python calls (your version wins over "
           "a player's function of the same name); variables do not persist between calls. The player's own "
           "instructions for that Python and for the actions:\n{text}\n\nThe player's valid actions now: {actions}.")
CODE_BIRTH = ("The player wrote this code; it is defined in your Python as well: call it if you want, and check it for "
              "mistakes.")
CODE_UPDATE = ("The player added or changed these functions since you last saw its code; they are defined in your "
               "Python as well: call them if you want, and check them for mistakes.")
CODE_BIRTH_NAMES = ("The player has defined these functions; they are defined in your Python as well: call them if "
                    "you want. print(player_source('<name>')) in run_python shows a function's code, to check it "
                    "for mistakes.")
CODE_UPDATE_NAMES = ("The player added or changed these functions since you last saw its code; they are defined in "
                     "your Python as well. print(player_source('<name>')) in run_python shows a function's code.")
LEVEL_WON = ("The player won level {won} of {total} in {actions} actions and now plays level {now}.{dropped} You stay "
             "with this game; its board now:")
SANDBOX_SECTIONS = ("Runtime variables inside every", "Action meanings", "Python tool guidance")
SANDBOX_FALLBACK = ("Runtime variables: current_frame is a frame view object, not a function, with only .ascii, "
                    ".segmentation, .step, .level and .shape; history is a list of entries with .action, .frame and "
                    ".result; previous_frame, transitions, last_transition and valid_actions are set as well. There "
                    "is no numpy and no raw numeric grid; the importable modules are " + ", ".join(SAFE_MODULES) + ".")
LAST = ("This is your last turn: write the player a note now with send_note. After it you end, and your slot goes "
        "to another game.")
NOTE_TAG = "<helper note, written at step {written}; you are at step {now}>"
PLAYER_LINE = ("A helper agent may be assigned to your game while you are stuck. It sees your boards and what you "
               "write, and may send you a note or a Python function; they appear at the start of a turn. Check each "
               "against your own evidence.")
PLAYER_API_LINE = ("Your Python also has Advisors.Reply(text) and Advisors.Ask(question): answer each note or function "
                   "with short feedback, and ask when you are stuck; the answer arrives as a helper note.")


def _live_stats(name: str, *args: Any) -> Any:
    try:
        from inference.agent import live_stats
    except Exception:
        return None
    return live_stats.call(name, *args)


def _effort() -> str:
    effort = os.environ.get("ARC3_REASONING_EFFORT", "").strip()
    return effort if effort in ("xhigh", "medium", "low") else ""


def _rerun_quiet() -> bool:
    return os.environ.get("TAAF_RUN_AS_SUBMISSION", "").strip() == "1"


def _env_int(name: str, default: int) -> int:
    return fa._env_int(name, default)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return fa._tool(name, description, properties, required)


TOOLS = [
    _tool("send_note", "A note to the player, delivered at the start of its next turn; notes not yet delivered wait "
          "in order and arrive together.", {"text": {"type": "string"}}, ["text"]),
    _tool("send_function", "One Python function for the player's Python, with a trial call on its boards and a short "
          "message saying what it does; it waits in order with your notes not yet delivered.",
          {"name": {"type": "string"}, "code": {"type": "string"}, "message": {"type": "string"}}, ["code", "message"]),
    _tool("run_python", "Python on the player's boards with the player's frame API; action() is disabled.",
          {"code": {"type": "string"}}, ["code"]),
    _tool("search", "A regular expression over everything the player has written: thinking, code, code output.",
          {"pattern": {"type": "string"}}, ["pattern"]),
    _tool("view_animation", "The boards one action of this game passed through before it settled (its animation), as "
          "images, with which cells changed at each and how long it plays. step = that action's number; omit it for "
          "the latest animated action; step 0 is the RESET that started the game. Only a few animations stay cached, "
          "the latest of each kind.", {"step": {"type": "integer"}}, []),
]


def helper_system() -> str:
    return SYSTEM + (chr(10) + fa.ANIMATION_LINE + chr(10) if fa._reset_animation() else "")


def clicked(text: str, before: Any) -> str:
    found = CLICK.fullmatch(text.strip())
    if not found:
        return text
    row, col = int(found.group(1)), int(found.group(2))
    if row < len(before) and col < len(before[row]):
        return f"{text} on a {COLORS[int(before[row][col]) % 16]} cell"
    return f"{text} outside the board"


def import_inside(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(one for one in tree.body if isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef)) and one.name == name)
    bound = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
    bound |= {target.id for one in ast.walk(node) if isinstance(one, (ast.Assign, ast.For, ast.With, ast.comprehension))
              for target in ast.walk(one) if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)}
    imported = {alias.asname or alias.name.split(".")[0] for one in ast.walk(node)
                if isinstance(one, (ast.Import, ast.ImportFrom)) for alias in one.names}
    used = {one.id for one in ast.walk(node) if isinstance(one, ast.Name) and isinstance(one.ctx, ast.Load)}
    missing = sorted(module for module in SAFE_MODULES if module in used and module not in bound | imported)
    if not missing:
        return source
    lines = source.splitlines()
    first = node.body[0]
    at = first.lineno - 1
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and \
            isinstance(first.value.value, str):
        at = first.end_lineno
    indent = re.match(r"\s*", lines[node.body[0].lineno - 1]).group(0)
    lines[at:at] = [f"{indent}import {module}" for module in missing]
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def _helper_gate() -> tuple[Callable[[str], bool], float] | None:
    try:
        from inference.agent import knob_arms
    except ImportError:
        knob_arms = None
    if knob_arms is not None and knob_arms.on() and knob_arms.helper_split():
        return knob_arms.helpable, knob_arms.helper_share()
    try:
        from inference.agent import duck_arms
    except ImportError:
        duck_arms = None
    if duck_arms is not None and duck_arms.on() and "H" in duck_arms.arms():
        return duck_arms.helpable, 1.0
    return None


class Slots:

    def __init__(self, root: Path, base_url: str, model: str, end_at: float) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "helpers").mkdir(exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.end_at = end_at
        self.started = time.time()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.games: dict[str, fa._Game] = {}
        self.helpers: dict[str, "Helper"] = {}
        self.history: dict[str, list[dict[str, Any]]] = {}
        self.asks: dict[str, list[dict[str, Any]]] = {}
        self.pins = [pin_entry(one) for one in os.environ.get("ARC3_SLOTS_PIN", "").split(",") if one.strip()]
        self.pinned_once: set[tuple[str, int]] = set()
        self.slots = max(0, _env_int("ARC3_SLOTS_START", 5))
        self.slot_cap = max(self.slots, _env_int("ARC3_SLOTS_MAX", 20))
        gate = _helper_gate()
        self.helpable = gate[0] if gate else None
        self.helper_share = gate[1] if gate else 1.0
        self.grow_carry = 0.0
        if gate and gate[1] < 1.0:
            self.slots = max(1, round(self.slots * gate[1]))
            self.slot_cap = max(self.slots, round(self.slot_cap * gate[1]))
        self.context_tokens = _env_int("ARC3_SLOTS_CONTEXT", 140000)
        self.last_turn_at = _env_int("ARC3_SLOTS_LAST_TURN", 120000)
        self.max_output = _env_int("ARC3_FORUM_MAX_OUTPUT", 10000)
        self.boards = _env_int("ARC3_SLOTS_BOARDS", 100)
        self.animation_frames = _env_int("ARC3_SLOTS_ANIMATION_FRAMES", 8)
        self.tail_chars = _env_int("ARC3_SLOTS_TAIL_CHARS", 5000)
        self.update_chars = _env_int("ARC3_SLOTS_UPDATE_CHARS", 1000)
        self.function_chars = _env_int("ARC3_SLOTS_FUNCTION_CHARS", 12000)
        self.code_chars = _env_int("ARC3_SLOTS_CODE_CHARS", 12000)
        self.hot_on = os.environ.get("ARC3_HELPER_HOT", "").strip() == "1"
        self.hot_tokens = _env_int("ARC3_HELPER_HOT_TOKENS", 3000)
        self.note_chars = _env_int("ARC3_SLOTS_NOTE_CHARS", 3000)
        self.upscale = max(1, _env_int("ARC3_FORUM_FORK_UPSCALE", 6))
        self.code_names = os.environ.get("ARC3_HELPER_CODE_NAMES", "").strip() == "1"
        self.last_thinking = os.environ.get("ARC3_HELPER_LAST_THINKING", "").strip() == "1"
        self.pause = os.environ.get("ARC3_SLOTS_PAUSE", "call").strip().lower()
        self.api_key = os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
        self.player_api = fa._env_flag("ARC3_FORUM_PLAYER_API")
        self.python_slots = threading.Semaphore(_env_int("ARC3_FORUM_PYTHON_SLOTS", 6))
        self.born = 0
        self.malformed_total = 0
        self.malformed_minute = 0
        self.first_actions = 0
        self.pauses_per_turn = max(0, _env_int("ARC3_SLOTS_PAUSES_PER_TURN", 1))
        self.pause_gap_s = max(0, _env_int("ARC3_SLOTS_PAUSE_GAP_S", 300))
        self.note_pause_gap_s = max(0, _env_int("ARC3_SLOTS_NOTE_PAUSE_GAP_S", 120))
        self.minute: dict[str, int] = {}
        self.pause_timeout_s = max(5, _env_int("ARC3_SLOTS_PAUSE_TIMEOUT_S", 180))
        self.background: Callable[[Callable[[], None]], None] = lambda work: threading.Thread(
            target=work, name="slots-pause", daemon=True).start()
        self.stats = {"helpers": 0, "notes": 0, "functions": 0, "delivered": 0, "interrupts": 0, "requests": 0,
                      "prompt": 0, "cached": 0, "generated": 0}

    def record(self, kind: str, **fields: Any) -> None:
        if _rerun_quiet():
            return
        event = {"t": round(time.time() - self.started, 1), "kind": kind, **fields}
        with self.lock:
            with open(self.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def say(self, text: str) -> None:
        fa._log(f"slots [{(time.time() - self.started) / 60:.1f} min]: {text}")

    def add_game(self, session: Any, key: str) -> None:
        game = fa._Game(session, key)
        game.animation_limit = max(3, _env_int("ARC3_FORUM_ANIMATIONS", 3))
        try:
            session.animation_hook = game.keep_animation
        except Exception:
            pass
        started = getattr(session, "reset_animation_record", None)
        if isinstance(started, dict):
            game.keep_animation(started)
        game.step_seen = 0
        game.level_start_tokens = 0
        game.level_start_actions = 0
        game.levels_seen = 0
        with self.lock:
            self.games[key] = game
        self.record("game", game=key)

    def start(self) -> "Slots":
        self.say(f"helper slots: {self.slots} at the start, one more for every game that ends, at most "
                 f"{self.slot_cap}; a helper stays with its game across its levels (a won level only sends it the "
                 f"news) until {self.context_tokens // 1000}K of context (last turn at {self.last_turn_at // 1000}K, "
                 f"with a last note) or the game's end; pause on a player turn: {self.pause}")
        self.record("start", slots=self.slots, cap=self.slot_cap, context=self.context_tokens, pause=self.pause)
        if self.helpable is not None:
            self.say(f"HELPER GATE: only copies marked H get a helper; slots scaled by {self.helper_share:.2f} to "
                     f"{self.slots} at the start, at most {self.slot_cap}")
        threading.Thread(target=self.watch, name="slots-watch", daemon=True).start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        for helper in list(self.helpers.values()):
            helper.end("the run ended")
        self.record("stop", **self.stats)
        self.say(f"stopped: {json.dumps(self.stats)}")

    def level_tokens(self, game: fa._Game) -> tuple[int, int]:
        status = game.status()
        if status["levels"] != game.levels_seen:
            game.levels_seen = status["levels"]
            game.level_start_tokens = status["tokens"]
            game.level_start_actions = status["actions"]
        return status["tokens"] - game.level_start_tokens, status["actions"] - game.level_start_actions

    def watch(self) -> None:
        minute = -1
        while not self.stop_event.wait(1.0):
            if time.time() >= self.end_at:
                break
            try:
                self.assign()
            except Exception as exc:
                self.say(f"assignment failed: {type(exc).__name__}: {exc}")
            now = int((time.time() - self.started) // 60)
            if now != minute:
                minute = now
                with self.lock:
                    live = sum(1 for helper in self.helpers.values() if helper.alive)
                    malformed, self.malformed_minute = self.malformed_minute, 0
                    acted, games = self.first_actions, len(self.games)
                    turns, self.minute = self.minute, {}
                self.say(f"SLOTS {self.slots} in all, {live} taken | players with an action {acted} of {games} | "
                         f"malformed tool calls dropped {malformed} this minute, {self.malformed_total} in all")
                self.say(f"TURNS last minute: players started {turns.get('player_started', 0)}, with a tool call "
                         f"{turns.get('player_called', 0)}, paused {turns.get('player_paused', 0)}, retried "
                         f"{turns.get('player_retried', 0)} (prompt {turns.get('player_retry_prompt', 0)} tokens, "
                         f"{turns.get('player_retry_computed', 0)} computed) | helpers started "
                         f"{turns.get('helper_started', 0)}, with a tool call {turns.get('helper_called', 0)}, paused "
                         f"{turns.get('helper_paused', 0)}, retried {turns.get('helper_retried', 0)} (prompt "
                         f"{turns.get('helper_retry_prompt', 0)} tokens, {turns.get('helper_retry_computed', 0)} "
                         f"computed)")
                self.record("slots", slots=self.slots, taken=live, acted=acted, games=games, malformed=malformed,
                            malformed_total=self.malformed_total, turns=turns)

    def check_levels(self) -> None:
        with self.lock:
            live = [helper for helper in self.helpers.values() if helper.alive and not helper.closing]
        for helper in live:
            status = helper.game.status()
            if int(status["levels"]) > helper.level:
                dropped = helper.take_waiting()
                helper.level = int(status["levels"])
                run = helper.game.run()
                spent = list(getattr(run, "actions_per_level", None) or [])
                actions = spent[helper.level - 1] if len(spent) >= helper.level else "?"
                helper.level_won(status, actions, len(dropped))
                self.say(f"STAYS {helper.name} on {helper.game.key}: player won level {helper.level}, context "
                         f"{helper.context // 1000}K" + (f", {len(dropped)} undelivered note(s) about it dropped"
                                                         if dropped else ""))
                self.record("level_won", helper=helper.name, game=helper.game.key, dropped=len(dropped), stayed=True,
                            level=helper.level, actions=actions, context=helper.context)

    def assign(self) -> None:
        self.check_levels()
        with self.lock:
            live = [key for key, helper in self.helpers.items() if helper.alive]
            free = self.slots - len(live)
            if free <= 0:
                return
            candidates = [game for key, game in self.games.items() if key not in live and not game.ended
                          and (self.helpable is None or self.helpable(key))]
        ranked = []
        for game in candidates:
            run = game.run()
            if run is not None and str(getattr(run, "state", "")) not in ("", "playing", "NOT_FINISHED") and \
                    getattr(run, "state", None) is not None and "play" not in str(run.state).lower():
                continue
            tokens, actions = self.level_tokens(game)
            asked = bool(self.asks.get(game.key))
            ranked.append((self.pin_rank(game), 0 if asked else 1, -tokens, game.key, game, tokens))
        ranked.sort(key=lambda one: one[:4])
        if self.pins and time.time() - self.started < 120 and any(
                level == 1 and not any(key.startswith(prefix) for key in self.games) for prefix, level in self.pins):
            ranked = [one for one in ranked if one[0] < len(self.pins)]
        for pin, _, _, key, game, tokens in ranked[:free]:
            if pin < len(self.pins):
                self.pinned_once.add((key, self.pins[pin][1]))
            self.spawn(game, f"pinned {pin + 1} of {len(self.pins)}, level {self.pins[pin][1]}" if pin < len(self.pins)
                       else "a question" if self.asks.get(key) else f"{tokens // 1000}K tokens on its level")

    def pin_rank(self, game: fa._Game) -> int:
        level = int(game.status()["levels"]) + 1
        return next((index for index, (prefix, pinned) in enumerate(self.pins)
                     if game.key.startswith(prefix) and pinned == level and (game.key, level) not in self.pinned_once),
                    len(self.pins))

    def spawn(self, game: fa._Game, why: str) -> None:
        with self.lock:
            self.born += 1
            helper = Helper(self, game, self.born, why)
            before = self.helpers.get(game.key)
            if before is not None:
                helper.waiting = before.take_waiting()
            self.helpers[game.key] = helper
            self.stats["helpers"] += 1
        threading.Thread(target=helper.run, name=f"slot-{self.born}", daemon=True).start()

    def finished(self, key: str) -> None:
        game = self.games.get(key)
        if game is None:
            return
        with self.lock:
            if game.ended:
                return
            game.ended = True
            self.grow_carry += self.helper_share
            step = int(self.grow_carry)
            self.grow_carry -= step
            grown = step > 0 and self.slots < self.slot_cap
            capped = step > 0 and not grown
            self.slots = min(self.slot_cap, self.slots + step)
            helper = self.helpers.get(key)
        status = game.status()
        if helper is not None:
            helper.end("its game ended")
        self.record("ended", game=key, state=status["state"], levels=status["levels"], slots=self.slots)
        self.say(f"{key} ended ({status['state']}, {status['levels']} of {status['total']} levels)"
                 + (f"; one more helper slot: {self.slots}" if grown else f"; slots stay at the cap {self.slots}"
                    if capped else f"; slots stay at {self.slots}"))

    def live_helper(self, key: str) -> "Helper | None":
        helper = self.helpers.get(key)
        return helper if helper is not None and helper.alive and not helper.closing else None

    def observe(self, game: fa._Game) -> dict[str, Any]:
        self.check_levels()
        status = game.status()
        helper = self.live_helper(game.key)
        if status["actions"] > game.step_seen:
            done = status["actions"] - game.step_seen
            if game.step_seen == 0:
                with self.lock:
                    self.first_actions += 1
                self.say(f"FIRST ACTION {game.key}: {done} action(s) done, "
                         f"{time.monotonic() - game.session.started_at:.0f} s into the game")
            game.step_seen = status["actions"]
            if helper is not None:
                helper.player_turn(done, status)
        return status

    def deliver(self, key: str, message: dict[str, Any]) -> dict[str, Any]:
        game = self.games.get(key)
        if game is None:
            return message
        status = self.observe(game)
        block = self.note_block(game, status)
        if not block:
            return message
        content = message.get("content")
        if isinstance(content, list):
            return {**message, "content": [{"type": "text", "text": block}, *content]}
        return {**message, "content": block + ("\n\n" + content if content else "")}

    def count(self, **values: int) -> None:
        with self.lock:
            for name, value in values.items():
                self.minute[name] = self.minute.get(name, 0) + int(value)

    def player_start(self, game: fa._Game) -> None:
        pending = getattr(game, "turn_pending", False)
        outcome = getattr(game, "turn_outcome", None)
        reason = "no reply (a timeout or a server error)" if pending else (
            outcome if outcome not in (None, "a tool call") else None)
        game.turn_pending, game.turn_retry = True, bool(reason)
        self.count(player_started=1, player_retried=1 if reason else 0)
        if reason:
            self.say(f"PLAYER RETRY {game.key}: a new request after {reason}")

    def player_reply(self, key: str, calls: int, content: bool, finish: str, paused: bool, malformed: int,
                     usage: Any) -> None:
        game = self.games.get(key)
        if game is None:
            return
        outcome = ("a tool call" if calls else "a pause" if paused else "a malformed tool call" if malformed
                   else "a reply cut at max_tokens" if finish == "length" else "a text reply with no tool call"
                   if content else "an empty reply")
        usage = usage if isinstance(usage, dict) else {}
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        retry = getattr(game, "turn_retry", False)
        game.turn_pending, game.turn_outcome = False, outcome
        self.count(player_called=1 if calls else 0, player_paused=1 if paused else 0,
                   player_retry_prompt=prompt if retry else 0, player_retry_computed=prompt - cached if retry else 0)

    def before_request(self, key: str, turn: int) -> str | None:
        game = self.games.get(key)
        if game is None:
            return None
        self.player_start(game)
        status = self.observe(game)
        if turn <= 1:
            return None
        return self.note_block(game, status)

    def take_paused(self, key: str) -> str | None:
        game = self.games.get(key)
        if game is None:
            return None
        block = self.note_block(game, game.status())
        return ("Your thinking was paused to give you this note; continue your turn.\n\n" + block) if block else None

    def malformed(self, key: str, count: int) -> None:
        with self.lock:
            self.malformed_total += int(count)
            self.malformed_minute += int(count)
        self.record("malformed", game=key, count=int(count))

    def player_cut(self, key: str) -> None:
        helper = self.live_helper(key)
        if helper is None or not self.hot_on:
            return
        block = helper.player_hot("the player's context was cut")
        if block:
            helper.post("The player's context was just cut: its oldest messages left its memory. " + block)

    def player_session(self, key: str, session: str) -> None:
        game = self.games.get(key)
        if game is not None and session:
            game.player_session = session

    def interrupt_player(self, game: fa._Game, by: str = "") -> None:
        session = getattr(game, "player_session", "")
        if not session or self.pause != "call":
            return
        step = int(game.status()["actions"])
        if getattr(game, "pause_step", None) != step:
            game.pause_step, game.pause_turn_count = step, 0
        held = ("a pause request for this lane still waits for the serve's answer"
                if getattr(game, "pause_pending", False)
                else self.pause_limit(game.pause_turn_count, getattr(game, "pause_at", None), self.note_pause_gap_s))
        if held:
            self.say(f"PLAYER PAUSE {game.key} by {by}: held back ({held}); the note waits for its tool result")
            self.record("player_pause", game=game.key, by=by, held=held)
            return
        game.pause_pending = True

        def answered(said: dict[str, Any]) -> None:
            game.pause_pending = False
            if said.get("paused"):
                if getattr(game, "pause_step", None) == step:
                    game.pause_turn_count += 1
                game.pause_at = time.time()

        self.pause_in_background(session, f"PLAYER PAUSE {game.key} by {by}", "thinking paused for a helper note",
                                 "player_pause", answered, game=game.key, by=by)

    def pause_in_background(self, session: str, line: str, done: str, kind: str,
                            answered: Callable[[dict[str, Any]], None], **fields: Any) -> None:
        sent = time.time()

        def work() -> None:
            said = pause_session(self, session)
            took = time.time() - sent
            answered(said)
            self.say(f"{line}: " + (done if said.get("paused") else f"not paused ({pause_why(said)})")
                     + f"; the serve answered in {took:.1f} s")
            self.record(kind, said=said, took=round(took, 1), **fields)

        self.background(work)

    def pause_limit(self, this_turn: int, last: float | None, gap: int | None = None) -> str:
        gap = self.pause_gap_s if gap is None else gap
        if this_turn >= self.pauses_per_turn:
            return f"already paused {this_turn} time(s) this turn, at most {self.pauses_per_turn}"
        if last is not None and time.time() - last < gap:
            return f"last pause {time.time() - last:.0f} s ago, at least {gap} s apart"
        return ""

    def note_block(self, game: fa._Game, status: dict[str, Any]) -> str | None:
        key = game.key
        helper = self.helpers.get(key)
        if helper is None:
            return None
        notes = helper.take_waiting()
        if not notes:
            return None
        lines = []
        for one in notes:
            tag = NOTE_TAG.format(written=one["step"], now=status["actions"])
            if one.get("function"):
                final, source = fa.inject(game.session.analyzer, one["function"], one["source"], helper.name)
                lines.append(f"{tag}\n{one['text']}\n```python\n{source}\n```\n{fa.signature_of(source, final)} is "
                             f"already defined in your Python namespace and kept between calls; call it or ignore it.")
            else:
                lines.append(f"{tag}\n{one['text']}")
            self.history.setdefault(key, []).append({"helper": helper.name, "kind": "function" if one.get("function")
                                                     else "note", "text": one["text"], "step": one["step"],
                                                     "delivered": status["actions"], "replies": []})
            self.record("delivered", game=key, helper=helper.name, step=one["step"], at=status["actions"],
                        help_kind="function" if one.get("function") else "note", text=one["text"])
        with self.lock:
            self.stats["delivered"] += len(notes)
        return "\n\n".join(lines) + ("\n" + fa.NOTE_REPLY_LINE if self.player_api else "")

    def advisor(self, key: str, kind: str, text: str) -> str:
        game = self.games.get(key)
        text = str(text).strip()
        if game is None or not text:
            return "Advisors: nothing was sent."
        helper = self.helpers.get(key)
        if kind == "reply":
            sent = self.history.get(key) or []
            if sent:
                sent[-1]["replies"].append(text)
            self.record("reply", game=key, text=text, helper=helper.name if helper else None)
            self.say(f"REPLY {key} -> {helper.name if helper else 'no helper'}: {fa._cap(text, 300)}")
            if helper is not None and helper.alive:
                helper.post(f"The player answered your help: \"{text}\"")
                return f"Advisors: your reply was delivered to {helper.name}."
            return "Advisors: your reply was kept for the next helper of this game."
        if kind == "ask":
            with self.lock:
                self.asks.setdefault(key, []).append({"t": time.time(), "text": text})
            self.record("ask", game=key, text=text, helper=helper.name if helper else None)
            self.say(f"ASK {key}: {fa._cap(text, 300)}")
            if helper is not None and helper.alive:
                helper.post(f"The player asks: \"{text}\"")
            return "Advisors: your question was delivered; an answer arrives as a helper note."
        return "Advisors: unknown call; nothing was sent."

    def fork_slots(self) -> int | None:
        return None

    def player_line(self) -> str:
        return "\n\n" + PLAYER_LINE + (" " + PLAYER_API_LINE if self.player_api else "")

    def player_context(self) -> tuple[int, int] | None:
        return None


class Helper:

    def __init__(self, slots: Slots, game: fa._Game, number: int, why: str) -> None:
        self.slots = slots
        self.game = game
        self.number = number
        self.name = f"helper {number}"
        self.why = why
        self.level = int(game.status()["levels"])
        self.alive = True
        self.closing = ""
        self.messages: list[dict[str, Any]] = []
        self.context = 0
        self.notes = 0
        self.functions = 0
        self.lock = threading.Lock()
        self.inbox: list[str] = []
        self.inbox_images: list[dict[str, Any]] = []
        self.wake = threading.Event()
        self.waiting: list[dict[str, Any]] = []
        self.phase = ""
        self.paused = False
        self.last_turn = False
        self.turn_pauses = 0
        self.pause_at: float | None = None
        self.pause_pending = False
        self.kept: dict[str, str] = {}
        self.kept_new: list[str] = []
        self.kept_said = False
        self.anim_said = False
        self.anim_viewed = False
        self.code_seen: dict[str, str] = {}
        self.print_aliases: dict[str, str] = {}
        self.code_said = False
        self.hot_said = False
        self.retry_reason = ""
        self.log_path = slots.root / "helpers" / f"{number:03d}.jsonl"

    def log(self, **fields: Any) -> None:
        if _rerun_quiet():
            return
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": round(time.time() - self.slots.started, 1), **fields}, ensure_ascii=False,
                                    default=str) + "\n")

    def session_key(self) -> str:
        return f"slot-{self.number}-{self.game.key}"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", fa.SESSION_HEADER: self.session_key()}
        if self.slots.api_key:
            headers["Authorization"] = f"Bearer {self.slots.api_key}"
        return headers

    def context_line(self) -> str:
        return f"context: {self.context // 1000}K of {self.slots.context_tokens // 1000}K"

    def end(self, why: str) -> None:
        if not self.alive or self.closing:
            return
        self.closing = why
        self.wake.set()

    def post(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        with self.lock:
            self.inbox.append(text)
            self.inbox_images += images or []
        self.wake.set()

    def take_waiting(self) -> list[dict[str, Any]]:
        with self.lock:
            notes, self.waiting = self.waiting, []
        return notes

    def board_image(self) -> dict[str, Any] | None:
        current, _ = self.game.frames()
        if current is None:
            return None
        from inference.agent.vision_context import frame_to_png_data_url
        return {"type": "image_url", "image_url": {"url": frame_to_png_data_url(current, upscale=self.slots.upscale)}}

    def player_turn(self, done: int, status: dict[str, Any]) -> None:
        tail = self.game.generation_tail(max(1, self.slots.update_chars // 4))[-self.slots.update_chars:]
        text = UPDATE.format(done=f"{done} action(s), now at step {status['actions']}, level {status['levels'] + 1} of "
                                  f"{status['total']}", tail=tail or "(nothing new)",
                             actions=", ".join(self.game.valid_actions()) or "(none listed)")
        with self.game.animation_lock:
            played = [one for one in self.game.animation_log if status["actions"] - done < one[0] <= status["actions"]]
        for step, action, count, lasted in played:
            text = (f"In this turn action {step} ({action}) played an animation of {count} boards"
                    + (f" over {lasted}" if lasted else "") + f"; view_animation(step={step}) shows it.\n" + text)
            self.saw_animation(step, f"in an update ({action}, {count} boards{', ' + lasted if lasted else ''})")
        code = self.player_code(True)
        if code:
            text = code + chr(10) * 2 + text
        image = self.board_image()
        self.post(text, [image] if image else [])
        if self.phase == "busy" and self.slots.pause == "call":
            self.pause_now(status["actions"])

    def player_code(self, changed_only: bool) -> str | None:
        try:
            sources = list(self.game.player()._retained_sources() or [])
        except Exception:
            sources = []
        current = {function_name(source): source for source in sources}
        fresh = {name: source for name, source in current.items()
                 if not changed_only or self.code_seen.get(name) != source}
        self.code_seen = dict(current)
        if not fresh:
            return None
        shown, listed, used = [], [], 0
        for name, source in reversed(list(fresh.items())):
            if used + len(source) <= self.slots.code_chars or not shown:
                shown.append(source.rstrip())
                used += len(source)
            else:
                line = next((one.strip() for one in source.splitlines() if one.lstrip().startswith("def ")), name)
                listed.append(line.rstrip(":").replace("def ", "", 1))
        if self.slots.code_names:
            heads = [signature_of(source) for source in fresh.values()]
            if not self.code_said:
                self.code_said = True
                self.slots.say(f"CODE NAMES {self.name} on {self.game.key}: {len(fresh)} functions named, "
                               f"{sum(len(one) for one in fresh.values())} chars of source left to player_source()")
            return (CODE_UPDATE_NAMES if changed_only else CODE_BIRTH_NAMES) + chr(10) + chr(10).join(heads)
        if not self.code_said:
            self.code_said = True
            self.slots.say(f"CODE {self.name} on {self.game.key}: {len(fresh)} functions, {used} chars shown")
        text = (CODE_UPDATE if changed_only else CODE_BIRTH) + chr(10) + "```python" + chr(10)
        text += (chr(10) * 2).join(shown) + chr(10) + "```"
        if listed:
            text += chr(10) + "Not shown for size, newest first: " + "; ".join(listed)
        return text

    def player_hot(self, why: str) -> str | None:
        if not self.slots.hot_on:
            return None
        try:
            from inference.agent import hot_fragments as hot
        except ImportError:
            return None
        answer = hot.query(self.slots.base_url, self.slots.model, getattr(self.game, "player_session", ""),
                           text_tokens=self.slots.hot_tokens)
        pieces = hot.capped(hot.fragments(answer), self.slots.hot_tokens)
        self.slots.count(helper_hot=1, helper_hot_empty=0 if pieces else 1)
        if not pieces:
            return None
        text = hot.block(hot.HELPER_HEAD, pieces)
        self.log(kind="hot", why=why, spans=len(pieces), chars=len(text))
        if not self.hot_said:
            self.hot_said = True
            self.slots.say(f"HOT {self.name} on {self.game.key}: {why}, {len(pieces)} fragment(s) of the player's hot "
                           f"text, about {len(text) // 4} tokens")
        return text

    def saw_animation(self, step: int, how: str) -> None:
        if self.anim_said:
            return
        self.anim_said = True
        self.slots.say(f"ANIMATION {self.name} on {self.game.key}: first sees an animation, step {step} {how}")

    def tool_view_animation(self, step: int | None = None, **_: Any) -> tuple[str, list[dict[str, Any]]] | str:
        game = self.game
        kept = game.animation_steps()
        if not kept:
            return "no animation kept for this game yet; an action whose board changes over several frames leaves one"
        record = game.animation(None if step is None else int(step))
        if record is None:
            return f"error: no animation cached for step {step}; cached steps: {', '.join(map(str, kept))}"
        from inference.agent.vision_context import frame_to_png_data_url
        chain = record["chain"]
        wanted = min(len(chain), self.slots.animation_frames)
        picks = sorted({round(index * (len(chain) - 1) / max(1, wanted - 1)) for index in range(wanted)})
        first = ("the board this RESET settled on (no board came before it)" if record.get("cause") == "start"
                 else "the board before the action")
        images = [{"label": f"step {record['step']} {record['action_display']}: "
                            + (first if index == 0 else f"animation board {index} of {len(chain) - 1}"),
                   "url": frame_to_png_data_url(SimpleNamespace(grid=chain[index]), upscale=self.slots.upscale)}
                  for index in picks]
        try:
            from inference.utils.animation import build_animation_view
            timeline = json.dumps(build_animation_view(record), separators=(",", ":"))
        except Exception as exc:
            timeline = f"(no timeline: {type(exc).__name__})"
        if not self.anim_viewed:
            self.anim_viewed = True
            self.slots.say(f"ANIMATION {self.name} on {game.key}: views the animation of step {record['step']}")
        return (chr(10).join([
            f"step {record['step']}, level {record['level']}: {record['action_display']}; {len(chain) - 1} boards "
            f"before it settled" + (f", over {record['lasted']}" if record.get("lasted") else ""),
            f"Images attached after the tool results: {len(images)} ({first}, then animation boards at even "
            f"intervals, ending at the settled one).",
            f"Other cached animations: {', '.join(str(one) for one in kept if one != record['step']) or 'none'}",
            "Cells changed at each animation board: " + fa._cap(timeline, 3000)]), images)

    def level_won(self, status: dict[str, Any], actions: Any, dropped: int) -> None:
        text = LEVEL_WON.format(won=status["levels"], total=status["total"], actions=actions, now=status["levels"] + 1,
                                dropped=(f" Your {dropped} note(s) not yet delivered were about the won level and "
                                         f"were dropped." if dropped else ""))
        hot = self.player_hot("a level change")
        if hot:
            text += chr(10) * 2 + hot
        image = self.board_image()
        self.post(text, [image] if image else [])
        if self.phase == "busy" and self.slots.pause == "call":
            self.pause_now(status["actions"])

    def pause_now(self, step: int = 0) -> None:
        held = ("a pause request for this helper still waits for the serve's answer" if self.pause_pending
                else self.slots.pause_limit(self.turn_pauses, self.pause_at))
        if held:
            self.slots.say(f"HELPER PAUSE {self.name} on {self.game.key} by the player's turn at step {step}: held "
                           f"back ({held}); the update waits for its next request")
            self.log(pause={"held": held})
            return
        self.pause_pending = True

        def answered(said: dict[str, Any]) -> None:
            self.pause_pending = False
            if said.get("paused"):
                self.turn_pauses += 1
                self.pause_at = time.time()

        self.slots.pause_in_background(self.session_key(), f"HELPER PAUSE {self.name} on {self.game.key} by the "
                                       f"player's turn at step {step}", "thinking paused", "helper_pause", answered,
                                       helper=self.name, game=self.game.key)

    def start_message(self) -> dict[str, Any]:
        from inference.agent.vision_context import frame_to_png_data_url
        game = self.game
        status = game.status()
        tokens, _ = self.slots.level_tokens(game)
        entries = [entry for entry in list(game.session.history_entries) if getattr(entry, "frame", None) is not None]
        marked = {0, len(entries) - 1} | {index for index, entry in enumerate(entries)
                                          if (getattr(entry, "result", {}) or {}).get("level_completed")
                                          or (getattr(entry, "result", {}) or {}).get("game_over")}
        marked = {index for index in marked if 0 <= index < len(entries)}
        recent = list(range(max(0, len(entries) - self.slots.boards), len(entries)))
        chosen = sorted((set(recent) | marked) if len(set(recent) | marked) <= self.slots.boards + len(marked)
                        else set(recent))
        parts: list[dict[str, Any]] = [{"type": "text", "text": START.format(
            game=game.key, level=status["levels"] + 1, total=status["total"], actions=status["actions"],
            spent=f"{tokens // 1000}K tokens")},
            *[{"type": "text", "text": line} for line in fa._game_meta_lines(game)],
            {"type": "text", "text": SANDBOX.format(text=player_python_text(game.player()),
                                                    actions=", ".join(game.valid_actions()) or "(none listed)")
                                     + print_legend()}]
        previous = None
        for index in chosen:
            entry = entries[index]
            result = dict(getattr(entry, "result", {}) or {})
            if index == 0:
                line = "Start of the game, before any action."
            else:
                before = entries[index - 1].frame.grid
                grid = entry.frame.grid
                cells = sum(1 for ra, rb in zip(before, grid) for x, y in zip(ra, rb) if x != y)
                line = (f"Action {getattr(entry.frame, 'step', '?')}: {clicked(str(getattr(entry, 'action', '')), before)}. "
                        f"{cells} cells changed.")
                line += " Level won." if result.get("level_completed") else ""
                line += " Died: game over, the level restarted." if result.get("game_over") else ""
                if previous is not None and index - previous > 1:
                    line = f"({index - previous - 1} actions not shown.) " + line
            parts += [{"type": "text", "text": line},
                      {"type": "image_url", "image_url": {"url": frame_to_png_data_url(entry.frame, upscale=self.slots.upscale)}}]
            previous = index
        animated = 0
        with game.animation_lock:
            started = [dict(one) for one in game.reset_animations if one.get("cause") == "start"]
            pinned = [dict(one) for one in game.pinned_animations]
        shown = [(one["step"], "the RESET that started the game") for one in started[:1]]
        shown += [(one["step"], one.get("why", "")) for one in pinned[-1:]]
        for step, why in shown:
            record = game.animation(step)
            if record is None:
                continue
            chain = record["chain"]
            lasted = f", it plays {record['lasted']}" if record.get("lasted") else ""
            wanted = min(len(chain), self.slots.animation_frames)
            for k in sorted({round(i * (len(chain) - 1) / max(1, wanted - 1)) for i in range(wanted)}):
                parts += [{"type": "text", "text": f"Animation of {why if step == 0 and record.get('cause') == 'start' else f'action {step} ({why})'}"
                                                   f"{lasted}: board {k} of {len(chain) - 1}."},
                          {"type": "image_url", "image_url": {"url": frame_to_png_data_url(
                              SimpleNamespace(grid=chain[k]), upscale=self.slots.upscale)}}]
                animated += 1
            self.saw_animation(step, f"in its birth message ({why}{lasted})")
        summary = game.animation_summary()
        if summary:
            parts.append({"type": "text", "text": summary})
        tail = game.generation_tail(max(1, self.slots.tail_chars // 4))[-self.slots.tail_chars:]
        parts.append({"type": "text", "text": "The end of what the player has written:\n" + (tail or "(nothing yet)")})
        code = self.player_code(False)
        if code:
            parts.append({"type": "text", "text": code})
        hot = self.player_hot("its birth")
        if hot:
            parts.append({"type": "text", "text": hot})
        earlier = self.slots.history.get(game.key) or []
        if earlier:
            lines = [f"- {one['helper']}, {one['kind']} written at step {one['step']}, delivered at step "
                     f"{one['delivered']}:\n{one['text']}" + "".join(f"\n  the player answered: {reply}"
                                                                     for reply in one["replies"]) for one in earlier]
            parts.append({"type": "text", "text": "Earlier helpers' notes and functions to this player, with its "
                                                  "answers:\n" + "\n".join(lines)})
        asks = self.slots.asks.get(game.key) or []
        if asks:
            parts.append({"type": "text", "text": "The player's questions:\n" + "\n".join(f"- {one['text']}" for one in asks)})
        parts.append({"type": "text", "text": self.context_line()})
        self.born_boards, self.born_animated = len(chosen), animated
        return {"role": "user", "content": parts}

    def request(self, final: bool) -> dict[str, Any] | None:
        plain = final and not self.slots.last_thinking
        if final and self.slots.last_thinking and not getattr(self.slots, "last_thinking_said", False):
            self.slots.last_thinking_said = True
            self.slots.say(f"HELPER LAST TURN THINKING {self.name} on {self.game.key}: the last turn keeps thinking "
                           f"on, so the system message and its cache stay as they were")
        payload = {"model": self.slots.model, "messages": self.messages, "tools": TOOLS,
                   "tool_choice": "required" if plain else "auto", "max_tokens": self.slots.max_output,
                   "temperature": 0.7, "top_p": 0.95, "top_k": 20,
                   "chat_template_kwargs": {"enable_thinking": not plain, "preserve_thinking": True,
                                            **({"reasoning_effort": _effort()} if _effort() else {})}}
        if _effort() and not getattr(self.slots, "effort_said", False):
            self.slots.effort_said = True
            self.slots.say(f"REASONING EFFORT helpers: every helper request asks the chat template for "
                           f"reasoning_effort={_effort()}")
        self.paused = False
        self.phase = "busy" if not final else "final"
        self.slots.count(helper_started=1, helper_retried=1 if self.retry_reason else 0)
        if self.retry_reason:
            self.slots.say(f"HELPER RETRY {self.name} on {self.game.key}: a new request after {self.retry_reason}")
        try:
            response = requests.post(f"{self.slots.base_url}/chat/completions", json=payload, headers=self.headers(),
                                     timeout=(10, 900))
            response.raise_for_status()
            body = response.json()
            choice = (body.get("choices") or [{}])[0]
        except Exception as exc:
            self.phase = ""
            self.log(error=f"{type(exc).__name__}: {str(exc)[:300]}")
            self.retry_reason = f"an error ({type(exc).__name__}: {str(exc)[:160]})"
            return None
        self.phase = ""
        usage = body.get("usage") or {}
        reply = choice.get("message") or {}
        finish = choice.get("finish_reason")
        reasoning = reply.get("reasoning") or reply.get("reasoning_content") or ""
        content = reply.get("content") or ""
        calls = [{"id": str(call.get("id") or ""), "type": "function",
                  "function": {"name": str((call.get("function") or {}).get("name") or ""),
                               "arguments": str((call.get("function") or {}).get("arguments") or "")}}
                 for call in reply.get("tool_calls") or []]
        self.paused = choice.get("stop_reason") == "taaf_pause"
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        generated = int(usage.get("completion_tokens") or 0)
        _live_stats("request", id(self), usage, finish)
        self.context = prompt + generated
        with self.slots.lock:
            self.slots.stats["requests"] += 1
            self.slots.stats["prompt"] += prompt
            self.slots.stats["cached"] += cached
            self.slots.stats["generated"] += generated
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning"] = reasoning
        if calls:
            message["tool_calls"] = calls
        paused = self.paused and not calls and not content
        retried = bool(self.retry_reason)
        self.slots.count(helper_called=1 if calls else 0, helper_paused=1 if paused else 0,
                         helper_retry_prompt=prompt if retried else 0,
                         helper_retry_computed=prompt - cached if retried else 0)
        if calls:
            self.turn_pauses = 0
        self.retry_reason = ("a pause" if paused else "" if calls else "a reply cut at max_tokens"
                             if finish == "length" else "a reply with no tool call")
        self.log(reply=message, finish=finish, usage=usage, paused=paused, context=self.context)
        return {"message": message, "paused": paused, "prompt": prompt, "cached": cached, "generated": generated}

    def run(self) -> None:
        slots = self.slots
        began = time.time()
        try:
            self.messages = [{"role": "system", "content": helper_system()}, self.start_message()]
        except Exception as exc:
            slots.say(f"BIRTH failed for {self.game.key}: {type(exc).__name__}: {exc}")
            self.alive = False
            return
        first = True
        final_tries = 0
        final_asks = 0
        try:
            while self.alive and not self.closing and not slots.stop_event.is_set():
                final = self.context >= slots.last_turn_at
                if final and not self.last_turn:
                    self.last_turn = True
                    self.add_user(LAST)
                    slots.say(f"LAST {self.name} on {self.game.key}: context {self.context // 1000}K reached the "
                              f"{slots.last_turn_at // 1000}K mark"
                              + (", its thinking was paused there" if self.paused else "")
                              + "; its last turn, without thinking, must write the player a note")
                reply = self.request(final)
                if reply is None:
                    final_tries += 1 if final else 0
                    if final and final_tries >= 3:
                        slots.say(f"LAST {self.name} on {self.game.key}: no last note, its last turn failed "
                                  f"{final_tries} times")
                        break
                    if slots.stop_event.wait(10.0):
                        break
                    continue
                if first:
                    first = False
                    slots.say(f"BIRTH {self.name} on {self.game.key} ({self.why}): {self.born_boards} boards and "
                              f"{self.born_animated} animation boards, prefill {reply['prompt'] - reply['cached']} of "
                              f"{reply['prompt']} tokens, {time.time() - began:.0f} s")
                    slots.record("birth", helper=self.name, game=self.game.key, why=self.why, boards=self.born_boards,
                                 animation=self.born_animated, prompt=reply["prompt"], cached=reply["cached"])
                self.note_interrupt_cost(reply)
                message = reply["message"]
                self.messages.append(message)
                if reply["paused"]:
                    kept = reply["generated"]
                    with slots.lock:
                        slots.stats["interrupts"] += 1
                    self.interrupted = (kept, len(message.get("reasoning") or ""))
                images: list[dict[str, Any]] = []
                for call in message.get("tool_calls") or []:
                    said = self.dispatch(call)
                    if isinstance(said, tuple):
                        said, more = said
                        images += list(more or [])
                    self.messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                          "content": f"{said}\n\n{self.context_line()}"})
                if images:
                    content: list[dict[str, Any]] = [{"type": "text", "text": "Images from your tool calls:"}]
                    for image in images:
                        content += [{"type": "text", "text": image["label"]},
                                    {"type": "image_url", "image_url": {"url": image["url"]}}]
                    self.messages.append({"role": "user", "content": content})
                if self.last_turn:
                    noted = any((call.get("function") or {}).get("name") == "send_note"
                                for call in message.get("tool_calls") or [])
                    if not noted and not str(message.get("content") or "").strip() and final_asks < 1:
                        final_asks += 1
                        slots.say(f"LAST {self.name} on {self.game.key}: its last reply sent no note; one more "
                                  f"last turn")
                        continue
                    self.last_note(message)
                    break
                if not message.get("tool_calls") and not reply["paused"] and self.context < slots.last_turn_at:
                    while not self.inbox and not self.closing and not slots.stop_event.is_set():
                        self.wake.wait(5.0)
                        self.wake.clear()
                if self.closing:
                    break
                self.flush(reply)
        finally:
            self.alive = False
            self.release()
            slots.say(f"DEATH {self.name} on {self.game.key}: {self.closing or ('context ' + str(self.context))}, "
                      f"context {self.context // 1000}K, notes {self.notes}, functions {self.functions}")
            slots.record("death", helper=self.name, game=self.game.key, why=self.closing or "context",
                         context=self.context, notes=self.notes, functions=self.functions)

    def last_note(self, message: dict[str, Any]) -> None:
        sent = [call for call in message.get("tool_calls") or []
                if (call.get("function") or {}).get("name") == "send_note"]
        if sent:
            self.slots.say(f"LAST {self.name} on {self.game.key}: its last note was sent, context "
                           f"{self.context // 1000}K")
            return
        text = str(message.get("content") or "").strip()
        if text:
            self.tool_send_note(text)
            self.slots.say(f"LAST {self.name} on {self.game.key}: its last reply called no send_note; its text "
                           f"({len(text)} chars) was sent as the note")
            return
        self.slots.say(f"LAST {self.name} on {self.game.key}: its last reply held neither a note nor text; nothing "
                       f"was sent")

    def add_user(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        text = f"{text}\n\n{self.context_line()}"
        if images:
            self.messages.append({"role": "user", "content": [{"type": "text", "text": text}, *images]})
        else:
            self.messages.append({"role": "user", "content": text})

    def flush(self, reply: dict[str, Any]) -> None:
        with self.lock:
            texts, images = self.inbox, self.inbox_images
            self.inbox, self.inbox_images = [], []
        if reply["paused"]:
            kept, chars = getattr(self, "interrupted", (0, 0))
            self.pending_interrupt = {"kept": kept, "chars": chars}
        if texts or images:
            self.add_user("\n\n".join(texts), images)
        elif not reply["message"].get("tool_calls"):
            self.add_user("(no news)")

    def note_interrupt_cost(self, reply: dict[str, Any]) -> None:
        pending = getattr(self, "pending_interrupt", None)
        if not pending:
            return
        self.pending_interrupt = None
        computed = reply["prompt"] - reply["cached"]
        self.slots.say(f"INTERRUPT {self.name} on {self.game.key}: {pending['kept']} thinking tokens kept, next "
                       f"request {reply['cached']} from cache, {computed} computed")
        self.slots.record("interrupt", helper=self.name, game=self.game.key, kept=pending["kept"],
                          cached=reply["cached"], computed=computed)

    def release(self) -> None:
        body = {"model": self.slots.model, "prompt": ".", "max_tokens": 1, "temperature": 0.0,
                "vllm_xargs": {fa.RELEASE_MARK: 1}}
        try:
            requests.post(f"{self.slots.base_url}/completions", json=body, headers=self.headers(), timeout=(5, 30))
        except Exception:
            pass

    def dispatch(self, call: dict[str, Any]) -> str:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            return "error: the arguments are not a JSON object"
        if not isinstance(arguments, dict):
            return "error: the arguments are not a JSON object"
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"error: no tool named {name!r}"
        if self.last_turn and name != "send_note":
            return f"refused: this is your last turn; only send_note is available now, {name} was not run"
        try:
            text = handler(**arguments)
        except TypeError as exc:
            text = f"error: bad arguments for {name}: {exc}"
        except Exception as exc:
            text = f"error: {type(exc).__name__}: {str(exc)[:300]}"
        self.log(tool=name, arguments=arguments, result=(text[0] if isinstance(text, tuple) else text)[:4000])
        return text

    def queue(self, item: dict[str, Any]) -> str:
        with self.lock:
            self.waiting.append(item)
            waiting = len(self.waiting)
        kind = "function" if item.get("function") else "note"
        if kind == "function":
            self.functions += 1
        else:
            self.notes += 1
        with self.slots.lock:
            self.slots.stats["functions" if kind == "function" else "notes"] += 1
        self.slots.record(kind, helper=self.name, game=self.game.key, step=item["step"], text=item["text"],
                          **({"function": item["function"], "source": item["source"]} if kind == "function" else {}))
        self.slots.say(f"{kind.upper()} {self.name} -> {self.game.key} at step {item['step']}:\n{fa._quote(item['text'])}")
        self.slots.interrupt_player(self.game, self.name)
        return ("queued; it reaches the player at its next request: at once if it is still thinking, otherwise after "
                "its current tool call" + (f"; {waiting} notes now wait for it, in the order you wrote them"
                                           if waiting > 1 else ""))

    def tool_send_note(self, text: str = "", **_: Any) -> str:
        text = str(text).strip()
        if not text:
            return "error: the note is empty"
        text = text[:self.slots.note_chars]
        return self.queue({"text": text, "step": self.game.status()["actions"]})

    def tool_send_function(self, code: str = "", message: str = "", name: str = "", **_: Any) -> str:
        code, text = str(code).strip("\n"), str(message).strip()
        if not code.strip():
            return "error: code is missing"
        if not text:
            return "error: a function needs a short message saying what it does"
        if len(code) > self.slots.function_chars:
            return f"error: the code is at most {self.slots.function_chars} characters, this one is {len(code)}"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"error: the code does not compile, nothing was sent: {exc}"
        defined = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        name = str(name).strip() or (defined[0] if len(defined) == 1 else "")
        if defined != [name]:
            return f"error: the code must define exactly one top-level function; it defines {defined or 'none'}"
        node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        rest = "\n".join(ast.get_source_segment(code, stmt) or "" for stmt in tree.body if stmt is not node)
        source = import_inside(ast.get_source_segment(code, node) or code, name)
        required = (len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults))
        trial = source + "\n" + (rest if f"{name}(" in rest else
                                 (f"print({name}(current_frame))" if required else f"print({name}())"))
        result = self.sandbox(trial)
        out = self.sandbox_text(result)
        if result.get("error"):
            return f"error: the trial raised, nothing was sent. Trial output:\n{out}"
        kept = {str(item.get("name")): str(item.get("source")) for item in result.get("keepable_functions") or []}
        if name not in kept:
            why = [str(item.get("reason")) for item in result.get("retention_rejected") or []
                   if item.get("name") in (name, "all")]
            return (f"error: the player's Python could not keep {name}" + (f": {'; '.join(why)}" if why else "")
                    + f". Nothing was sent. Trial output:\n{out}")
        said = self.queue({"text": text[:self.slots.note_chars], "step": self.game.status()["actions"],
                           "function": name, "source": kept[name]})
        return f"trial output:\n{out}\n\n{said}"

    def tool_run_python(self, code: str = "", **_: Any) -> str:
        text = self.sandbox_text(self.sandbox(str(code)))
        fresh, self.kept_new = self.kept_new, []
        return text + (f"\nkept for your later run_python calls: {', '.join(name + '()' for name in fresh)}"
                       if fresh else "")

    def tool_search(self, pattern: str = "", **_: Any) -> str:
        try:
            compiled = re.compile(str(pattern), re.IGNORECASE)
        except re.error as exc:
            return f"error: bad pattern: {exc}"
        hits = []
        for turn in self.game.parse():
            for label, _extra, lines in turn["sections"]:
                for number, line in enumerate(lines):
                    if compiled.search(line):
                        hits.append(f"turn {turn['turn']} (action {turn['action']}) [{label}] {line.strip()[:300]}")
        if not hits:
            return "no match"
        shown = hits[-60:]
        return (f"{len(hits)} matching lines" + (f", the latest {len(shown)}" if len(hits) > len(shown) else "")
                + ":\n" + "\n".join(shown))

    def sandbox(self, code: str) -> dict[str, Any]:
        from inference.agent import tool_agent
        from inference.agent.python_tool_sandbox import run_sandboxed_python
        current, history = self.game.frames()
        player = self.game.player()
        last = getattr(player, "_last_action_call_result", None)
        view = tool_agent._ascii_frame_view_payload(current)
        objects_view = getattr(tool_agent, "_frame_objects_view", None)
        if objects_view is not None and player is not None:
            view = objects_view(player, current, view)
        state = {"current_frame": view,
                 "history": tool_agent._ascii_history_view_payload(history),
                 "valid_actions": self.game.valid_actions(),
                 "last_action_call_result": dict(last) if isinstance(last, dict) else {}}

        def refuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("helpers cannot act")

        try:
            theirs = {function_name(source): source for source in player._retained_sources() or []}
        except Exception:
            theirs = {}
        offered = {**theirs, **self.kept}
        if self.slots.code_names:
            offered = {"player_source": source_function(theirs), **offered}
        timeout = int(getattr(player, "_python_timeout", 20) or 20)
        with self.slots.python_slots:
            result = run_sandboxed_python(code=str(code), timeout_seconds=timeout, initial_state=state,
                                          action_handler=refuse, kept_functions=list(offered.values()),
                                          retain_imports=True)
        mine = {str(item.get("name")): str(item.get("source")) for item in result.get("keepable_functions") or []
                if isinstance(item, dict) and theirs.get(str(item.get("name"))) != str(item.get("source"))
                and not (self.slots.code_names and str(item.get("name")) == "player_source")}
        self.kept_new = [name for name, source in mine.items() if self.kept.get(name) != source]
        if mine and not self.kept and not self.kept_said:
            self.kept_said = True
            self.slots.say(f"KEPT {self.name} on {self.game.key}: its first own function {next(iter(mine))}() stays "
                           f"defined for its later Python calls")
        self.kept = mine
        return result

    def sandbox_text(self, result: dict[str, Any]) -> str:
        out = []
        if result.get("stdout"):
            out.append(compact_print(str(result["stdout"]), self.print_aliases))
        if result.get("result") is not None:
            out.append(f"result = {json.dumps(result['result'], default=str)[:3000]}")
        if result.get("error"):
            out.append(f"error: {result['error']}")
        return fa._cap("\n".join(out) or "(no output)", 6000)


def player_python_text(player: Any) -> str:
    request = getattr(player, "_forum_request", None)
    messages = request[0] if isinstance(request, tuple) and request else None
    system = messages[0] if isinstance(messages, list) and messages else None
    text = system.get("content") if isinstance(system, dict) and system.get("role") == "system" else None
    if isinstance(text, list):
        text = "\n".join(str(part.get("text") or "") for part in text if isinstance(part, dict))
    blocks = [block.strip() for block in str(text or "").split("\n\n\n")
              if block.strip().startswith(SANDBOX_SECTIONS)]
    return "\n\n".join(blocks) if blocks else SANDBOX_FALLBACK


def signature_of(source: str) -> str:
    line = next((one.strip() for one in str(source).splitlines() if one.lstrip().startswith("def ")), "")
    return line.rstrip(":").replace("def ", "", 1) if line else function_name(source)


def source_function(sources: dict[str, str]) -> str:
    return ("def player_source(name):\n"
            f"    return {dict(sources)!r}.get(name, 'no player function of that name')\n")


def compact_print(text: str, aliases: dict[str, str]) -> str:
    try:
        from inference.agent import prefill_text
    except ImportError:
        return text
    if not prefill_text.flag(prefill_text.PRINT_ENV):
        return text
    try:
        return prefill_text.compact_print(text, aliases)
    except Exception as exc:
        print(f"COMPACT PRINT skipped a helper call: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return text


def print_legend() -> str:
    try:
        from inference.agent import prefill_text
    except ImportError:
        return ""
    if not prefill_text.flag(prefill_text.PRINT_ENV):
        return ""
    return "\n\n" + prefill_text.PRINT_LEGEND[2:]


def function_name(source: str) -> str:
    found = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(source), re.M)
    return found.group(1) if found else str(source)[:40]


def pin_entry(text: str) -> tuple[str, int]:
    prefix, _, level = text.strip().partition(":")
    return prefix.strip(), int(level) if level.strip().isdigit() else 1


def pause_why(said: dict[str, Any]) -> str:
    return str(said.get("why") or said.get("error") or said)


def pause_session(slots: Slots, session: str) -> dict[str, Any]:
    body = {"model": slots.model, "prompt": ".", "max_tokens": 1, "temperature": 0.0, "vllm_xargs": {PAUSE_MARK: 1}}
    headers = {"Content-Type": "application/json", fa.SESSION_HEADER: session}
    if slots.api_key:
        headers["Authorization"] = f"Bearer {slots.api_key}"
    try:
        response = requests.post(f"{slots.base_url}/completions", json=body, headers=headers,
                                 timeout=(5, slots.pause_timeout_s))
        text = ((response.json().get("choices") or [{}])[0].get("text") or "{}") if response.ok else "{}"
        return json.loads(text) if text.strip().startswith("{") else {"text": text[:200]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def create(root: Path, base_url: str, model: str, end_at: float) -> Slots:
    return Slots(root, base_url, model, end_at)


try:
    from inference.agent import frame_objects as _frame_objects
except ImportError:
    _frame_objects = None
if _frame_objects is not None:
    _frame_objects.apply_module(globals())
