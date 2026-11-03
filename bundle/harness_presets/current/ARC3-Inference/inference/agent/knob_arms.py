import json
import os
import re
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

PLAN_ENV = "ARC3_KNOB_PLAN"
ALL_ENV = "ARC3_KNOB_ALL"
GAME_TOKENS_ENV = "ARC3_KNOB_GAME_TOKENS"
LEVELS_GUESS_ENV = "ARC3_KNOB_LEVELS_GUESS"
DEBRIEF_TOKENS_ENV = "ARC3_KNOB_DEBRIEF_TOKENS"
DEBRIEF_THINKING_ENV = "ARC3_KNOB_DEBRIEF_THINKING"
RESETS_KEPT_ENV = "ARC3_KNOB_RESETS_KEPT"
RECORDS_ENV = "ARC3_KNOB_RECORDS"
LAMBDA_ENV = "ARC3_KNOB_LAMBDA"
PIXELS_ENV = "ARC3_KNOB_PIXELS"
SHORT_KNOBS = {"ARC3_COMPACT_PRINT": "1", "ARC3_COMPACT_TURN": "1", "ARC3_SAME_BOARD_TEXT": "1",
               "ARC3_PERCEPTION_SCENE": "1", "ARC3_PERCEPTION_SHORT_DELTA": "1", "ARC3_RETAINED_DELTA": "1"}
SMALL_PICTURES = {"ARC3_PERCEPTION_IMAGE_SCALE": "0.8"}
NOWAIT_ENV = "ARC3_KNOB_NOWAIT_BIAS"
NOWAIT_TOKENS = {"Wait": 13784, " Wait": 13428, "wait": 11158, " wait": 3655, "WAIT": 68620, " WAIT": 52565,
                 "Hmm": 77264, " Hmm": 85152, " hmm": 84485, "Actually": 50821, " Actually": 32645, "actually": 70546,
                 " actually": 3404}
INTEREST_ENV = "ARC3_KNOB_INTEREST_BIAS"
INTEREST_TOKENS = {"Interesting": 81404, " Interesting": 68278, "Interestingly": 79065, " Interestingly": 55577}
NOTICE_ENV = "ARC3_KNOB_NOTICE_BIAS"
DOSE = re.compile(r"I(\d+)")
HEDGE_ENV = "ARC3_KNOB_HEDGE_BIAS"
DULL_ENV = "ARC3_KNOB_DULL_BIAS"
NOTICE_TOKENS = {"Goal": 38663, " Goal": 38622, "goal": 32817, " goal": 5614, " goals": 8578, " understand": 3418,
                 " Understand": 68490, " understanding": 8396, " mechanic": 42672, " mechanics": 28079,
                 " mechanism": 16456, "Discovery": 65125, " Discovery": 36833, " discovery": 17789,
                 " discovered": 10782, " rule": 5728, " rules": 5426, " pattern": 5215, " hypothesis": 29094,
                 " insight": 19407, " noticed": 13293, " Observation": 83441}
INTEREST_WORDS = re.compile(r"\bInteresting(?:ly)?\b")
NOWAIT_WORDS = re.compile(r"\b(?:[Ww]ait|WAIT|Hmm|hmm|[Aa]ctually)\b")
NOTICE_WORDS = re.compile(r"\b(?:[Gg]oals?|[Uu]nderstand(?:ing)?|mechanics?|mechanism|[Dd]iscover(?:y|ed)|rules?|pattern|"
                          r"hypothesis|insight|noticed|Observation)\b")
PICTURE_SWAPS = (("- User turns include an attached image of the current ARC grid.\n",
                  "- The first turn of each level includes an attached image of the current ARC grid; later turns do "
                  "not.\n"),
                 ("- Each turn shows a picture of the board.", "- The first turn of each level shows a picture of the "
                                                               "board; later turns do not."))
OFF_LETTERS = {"v": ("ARC3_PERCEPTION_DIGEST", "ARC3_PERCEPTION_CATALOG", "ARC3_PERCEPTION_DELTA",
                      "ARC3_PERCEPTION_DELTA_IMAGE", "ARC3_PERCEPTION_LEVEL"),
               "o": ("ARC3_FRAME_OBJECTS",),
               "y": ("ARC3_LEAN_SYSTEM",),
               "h": ("ARC3_HOT_RETURN",),
               "x": ("ARC3_BATCH_NOOP_BLOCK", "ARC3_STALE_STATE_BLOCK")}
ON_LETTERS = {"G": ("ARC3_FIRST_PICTURE_ONLY",),
              "K": ("ARC3_LOOK",),
              "Y": ("ARC3_LEAN_SYSTEM",),
              "E": ("ARC3_WIN_PAUSE",),
              "C": ("ARC3_LEAN_COMPACT_PRINT",),
              "A": ("ARC3_CLICK_LOOK",),
              "O": ("ARC3_NOOP_SOFT",),
              "Z": ("ARC3_CLICK_LOOK_SHORT",),
              "V": ("ARC3_NOOP_SHORT",),
              "X": ("ARC3_TOOL_CAP_LOOK", "ARC3_LOOK"),
              "H": ("ARC3_WIN_PAUSE",)}
VALUE_LETTERS = {"T": {"ARC3_CONTEXT_DRAIN_TOKENS": "100000", "ARC3_CONTEXT_DRAIN_MAX_SHARE": "0.9"},
                 "M": {"ARC3_HOT_BUDGET": "14000"}}
WIN_TEXT = ("You won level {level}! Below are the board before your winning action, the boards the winning action "
            "passed through, and the board at the moment of the win. You can study them now: what you did right, "
            "which of your hypotheses held, what winning the level took. Or go straight on. The next level's board "
            "is shown after you call `next_level()` in Python; until then game actions are refused, and "
            "`current_frame` is the board at the moment of the win.")
WIN_OPENED = "next_level(): the next level is open. Its starting board follows."
WIN_AUTO = "The next level opened on its own after {replies} replies without next_level(). Its starting board follows."
WIN_REFUSED = "Level {level} is won and the next level is not open yet: call next_level() first. Nothing was executed."
WIN_IDLE = "next_level(): no won level is waiting; you are already on the newest level."
WIN_ASK = "Tell me, what is this game about? How did you win level {level}? What did you have to do?"
WIN_TEXT_H = 'You won level {level}! The pictures above show the board before your winning action, the boards the winning action passed through, and the board at the moment of the win. You can study them now. Once you are ready, call `next_level("<what this game is about, what this level was, and how you won it>")` in Python; until then game actions are refused, and `current_frame` is the board at the moment of the win. A person will read your description, and you are judged on how well it explains the game to an average person.'
WIN_NEEDS_TEXT = 'next_level() needs your description first: next_level("<what this game is about, what this level was, and how you won it>"). The next level is still held.'
WIN_PIN_H = '- After you completed level {level} you described the game for a person: {text}'
WIN_PIN = ("- After you completed level {level} you were asked what this game is about, how you won the level and "
           "what it took. You answered: {text}")
WIN_LABELS = ("The board before your winning action (step {step}):", "The winning action, board {index} of {total}:",
              "The board at the moment of the win (step {step}):",
              "The board before your winning action, unchanged by it until the next level loaded (step {step}):")
PICTURE_LINE = ("- The board picture comes only with the first turn of each level; later turn messages carry text "
                "only. Read the board through Python (`current_frame`) after that.")
LOOK_LINE = ("- `Look(grid)` in Python shows any grid you build as a picture in the next message, drawn like the board "
             "picture (same colours, {pixels} pixels a cell). `grid` is a list of rows, each a string of colour "
             "letters or a list of colour numbers 0-15, or a whole frame such as `current_frame`, `previous_frame` "
             "or one of `last_animation_frames`; at most 64x64 cells, at most {most} pictures a python call. A "
             "picture costs about one token per {cells} cells, far less than printing the same cells as text, and a "
             "picture is the better way to take in the overall layout.")
STEM = re.compile(r"^(.*?_p(\d+))(?:_|\.|$)")
STATE = re.compile(r"Current state: step \d+, level (\d+)(?: of (\d+))?")
CUT = re.compile(r"<tool_call>|<\|im_end\|>|<\|im_start\|>|</?think>")
RESET_LINE = ("- `RESET` restarts the current level from its first board and counts as one action; finished levels "
              "are kept. It needs a reason: `action(['RESET'], reason='<why this level must start over>')`. The "
              "harness keeps your reasons and shows them to you again later.")
BUDGET_LINE = ("- Each level has a token budget, shown at the end of every turn message: the tokens you generate on "
               "that level (thinking and replies) against the level's share of the game. Use it to pace yourself; "
               "nothing stops when it runs out.")
SYSTEM_HEAD = "\n\nAlso in this run:\n"
RESET_NEEDS_REASON = "RESET needs a reason: action(['RESET'], reason='<why this level must start over>')"
PINS_HEAD = "Kept by the harness from earlier in this game:"
BRANCH_PREFIX = "taaf-branch"
DEBRIEF_ASK = ("Level {level} is complete. Before you continue, write a short debrief for your own later use, in plain "
               "words: how you completed level {level} (what you did and why it worked), and what this game is about "
               "and how a level is won, as you understand it now. At most 120 words. Do not call any tool.")
DEBRIEF_PIN = "- Your debrief after completing level {level}: {text}"
DEBRIEF_START = "Debrief:"
INLINE_ASK = ("Level {level} is complete. Before your next action, start your reply with a short debrief for your own "
              "later use, in plain words: how you completed level {level} (what you did and why it worked), and what "
              "this game is about and how a level is won, as you understand it now. At most 120 words. Then continue "
              "playing.")
RESET_PIN = "- You pressed RESET on level {level}, giving this reason: {reason}"
_local = threading.local()
_lock = threading.Lock()
_said: set[str] = set()
_plan: dict[str, str] | None = None


def on() -> bool:
    return bool(os.environ.get(PLAN_ENV, "").strip() or os.environ.get(ALL_ENV, "").strip())


def every() -> str | None:
    letters = os.environ.get(ALL_ENV, "").strip()
    return letters or None


def plan() -> dict[str, str]:
    global _plan
    if _plan is None:
        try:
            parsed = json.loads(os.environ.get(PLAN_ENV, "") or "{}")
        except ValueError:
            parsed = {}
            print(f"KNOB PLAN unreadable: {os.environ.get(PLAN_ENV, '')[:200]!r}", flush=True)
        _plan = {str(key): str(value) for key, value in parsed.items()} if isinstance(parsed, dict) else {}
    return _plan


def number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def number_text(text: str, default: float) -> float:
    try:
        return float(text or default)
    except ValueError:
        return default


def say_once(key: str, text: str) -> None:
    with _lock:
        if key in _said:
            return
        _said.add(key)
    print(text, flush=True)


def stem_of(owner: Any) -> str:
    said = getattr(owner, "_burn_said", None)
    return str(said[0]) if said and said[0] else ""


def letters_of(owner: Any) -> str | None:
    stem = stem_of(owner)
    return plan().get(stem, every()) if stem else None


def arm(owner: Any) -> None:
    letters = letters_of(owner)
    _local.owner = owner if letters is not None else None
    _local.letters = letters
    if letters is not None:
        say_once(f"arm {stem_of(owner)}", f"knob arm: {stem_of(owner)} plays {letters or 'none'}")


class _Environ(MutableMapping):
    def __init__(self, base: Any) -> None:
        self._base = base

    def _over(self, key: str) -> str | None:
        letters = getattr(_local, "letters", None)
        if not letters:
            return None
        for letter, names in OFF_LETTERS.items():
            if letter in letters and key in names:
                return "0"
        for letter, names in ON_LETTERS.items():
            if letter in letters and key in names:
                return "1"
        for letter, values in VALUE_LETTERS.items():
            if letter in letters and key in values:
                return values[key]
        return None

    def __getitem__(self, key: str) -> str:
        value = self._over(key)
        return value if value is not None else self._base[key]

    def get(self, key: str, default: Any = None) -> Any:
        value = self._over(key)
        return value if value is not None else self._base.get(key, default)

    def __setitem__(self, key: str, value: str) -> None:
        self._base[key] = value

    def __delitem__(self, key: str) -> None:
        del self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def copy(self) -> dict[str, str]:
        return self._base.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def overlay() -> None:
    if isinstance(os.environ, _Environ):
        return
    named = [*plan().values(), every() or ""]
    if not any(letter in letters for letters in named for letter in (*OFF_LETTERS, *ON_LETTERS, *VALUE_LETTERS)):
        return
    os.environ = _Environ(os.environ)
    say_once("overlay", f"KNOB OFF letters armed: {', '.join(f'{k} turns off {len(v)} knob(s)' for k, v in OFF_LETTERS.items())}"
                        f"; ON letters: {', '.join(f'{k} turns on {names}' for k, names in ON_LETTERS.items())}")


def arm_path(path: Any) -> None:
    overlay()
    found = STEM.match(Path(str(path)).name)
    stem = found.group(1) if found else ""
    _local.owner = None
    _local.letters = plan().get(stem, every()) if stem else None


def helper_split() -> bool:
    return any("H" in letters for letters in plan().values())


def helpable(key: str) -> bool:
    return "H" in plan().get(str(key), "")


def helper_share() -> float:
    rows = list(plan().values())
    return sum("H" in letters for letters in rows) / len(rows) if rows else 1.0


def env(name: str, default: str) -> str:
    letters = getattr(_local, "letters", None)
    if letters is not None and name == "EXPOSE_RESET":
        return "on" if "R" in letters else "off"
    if letters is not None and name == "EXPOSE_UNDO":
        return "on" if "U" in letters else "off"
    if letters and name == "MULTIMODAL_UPSCALE" and "P" in letters:
        return os.environ.get(PIXELS_ENV, "8")
    if letters and name in SMALL_PICTURES and "P" in letters:
        return SMALL_PICTURES[name]
    if letters and name in SHORT_KNOBS and "S" in letters:
        return SHORT_KNOBS[name]
    return os.environ.get(name, default)


def records() -> Path | None:
    named = os.environ.get(RECORDS_ENV, "").strip()
    if named:
        return Path(named)
    working = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR", "").strip() or "/kaggle/working")
    return working / "knob_arms.jsonl" if working.is_dir() else None


def write(row: dict[str, Any]) -> None:
    path = records()
    if path is None:
        return
    with _lock, open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": round(time.time(), 1), **row}, ensure_ascii=False) + "\n")


CLICK_SHORT_LINE = ("- After a click, `@<c> [<3x3>]` gives the colour letter under the clicked cell and its 3x3 "
                    "neighbourhood before the click, rows top to bottom; `[all <c>]` means all nine cells are <c>.")
NOOP_SHORT_LINE = ("- `no change here` after an action means the board did not change in this state; the same action "
                   "may still work at another position, in another state or on another level, so do not discard it.")


CAP_LINE = '- A tool result is cut after about 512 tokens. Print only short, targeted facts. For a large visual overview use `Look(grid)`: a picture costs far less than the same rows printed, up to 8 pictures a python call.'


def system(owner: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    letters = letters_of(owner)
    if not letters or not messages or messages[0].get("role") != "system":
        return messages
    content = messages[0].get("content")
    if not isinstance(content, str):
        return messages
    pixels = max(1, int(number_text(env("MULTIMODAL_UPSCALE", "10"), 10)))
    look = LOOK_LINE.format(pixels=pixels, cells=max(1, round((32 / pixels) ** 2)), most=8 if "X" in letters else 4)
    lines = [line for letter, line in (("R", RESET_LINE), ("B", BUDGET_LINE), ("G", PICTURE_LINE), ("K", look),
                                       ("Z", CLICK_SHORT_LINE), ("V", NOOP_SHORT_LINE), ("X", look),
                                       ("X", CAP_LINE))
             if letter in letters]
    lines = list(dict.fromkeys(lines))
    swapped = content
    if "G" in letters:
        for old, new in PICTURE_SWAPS:
            swapped = swapped.replace(old, new)
    if swapped == content and (not lines or all(line in content for line in lines)):
        return messages
    head = messages[0].copy()
    missing = [line for line in lines if line not in swapped]
    head["content"] = swapped + (SYSTEM_HEAD + "\n".join(missing) if missing else "")
    return [head, *messages[1:]]


def payload_hook(owner: Any, payload: dict[str, Any]) -> None:
    letters = letters_of(owner)
    if not letters:
        return
    payload["messages"] = system(owner, payload.get("messages") or [])
    if "L" in letters:
        value = number(LAMBDA_ENV, 2.0)
        payload["vllm_xargs"] = {**(payload.get("vllm_xargs") or {}), "taaf_accept_lambda": value}
        say_once("L", f"KNOB L first used by {stem_of(owner)}: vllm_xargs taaf_accept_lambda {value}")
    if "W" in letters:
        value = number(NOWAIT_ENV, -100.0)
        payload["logit_bias"] = {**(payload.get("logit_bias") or {}),
                                 **{str(token): value for token in NOWAIT_TOKENS.values()}}
        say_once("W", f"KNOB W first used by {stem_of(owner)}: logit_bias {value} on "
                      f"{len(NOWAIT_TOKENS)} tokens {sorted(NOWAIT_TOKENS)}")
    if "I" in letters:
        dose = DOSE.search(letters)
        value = float(dose.group(1)) if dose else number(INTEREST_ENV, 1.0)
        payload["logit_bias"] = {**(payload.get("logit_bias") or {}),
                                 **{str(token): value for token in INTEREST_TOKENS.values()}}
        say_once(f"I{value:g}", f"KNOB I first used by {stem_of(owner)}: logit_bias {value:+g} on "
                      f"{len(INTEREST_TOKENS)} tokens {sorted(INTEREST_TOKENS)}")
    if "w" in letters:
        value = number(HEDGE_ENV, 2.0)
        payload["logit_bias"] = {**(payload.get("logit_bias") or {}),
                                 **{str(token): value for token in NOWAIT_TOKENS.values()}}
        say_once("w", f"KNOB w first used by {stem_of(owner)}: logit_bias {value:+g} on "
                      f"{len(NOWAIT_TOKENS)} tokens {sorted(NOWAIT_TOKENS)}")
    if "i" in letters:
        value = number(DULL_ENV, -100.0)
        payload["logit_bias"] = {**(payload.get("logit_bias") or {}),
                                 **{str(token): value for token in INTEREST_TOKENS.values()}}
        say_once("i", f"KNOB i first used by {stem_of(owner)}: logit_bias {value:+g} on "
                      f"{len(INTEREST_TOKENS)} tokens {sorted(INTEREST_TOKENS)}")
    if "N" in letters:
        value = number(NOTICE_ENV, 1.0)
        payload["logit_bias"] = {**(payload.get("logit_bias") or {}),
                                 **{str(token): value for token in NOTICE_TOKENS.values()}}
        say_once("N", f"KNOB N first used by {stem_of(owner)}: logit_bias {value:+g} on "
                      f"{len(NOTICE_TOKENS)} tokens {sorted(NOTICE_TOKENS)}")
    tally(owner, letters, payload.get("messages") or [])


WORD_TALLY: dict[str, list[int]] = {}
WORD_SAMPLES: dict[str, tuple[str, str]] = {}
WORD_TALLY_EVERY = 300


def tally(owner: Any, letters: str, messages: list[dict[str, Any]]) -> None:
    last = next((message for message in reversed(messages) if message.get("role") == "assistant"), None)
    if last is None:
        return
    said = f"{last.get('reasoning') or last.get('reasoning_content') or ''}\n{text_of(last)}"
    mark = hash(said)
    if getattr(owner, "_knob_tally_mark", None) == mark:
        return
    owner._knob_tally_mark = mark
    with _lock:
        for letter, words in (("I", INTEREST_WORDS), ("W", NOWAIT_WORDS), ("N", NOTICE_WORDS)):
            side = "+" if letter in letters else "~" if letter.lower() in letters else "-"
            dose = DOSE.search(letters) if letter == "I" and side == "+" else None
            side += dose.group(1) if dose else ""
            row = WORD_TALLY.setdefault(f"{letter}{side}", [0, 0, 0])
            found = len(words.findall(said))
            row[0] += 1
            row[1] += bool(found)
            row[2] += found
            if found:
                hit = list(words.finditer(said))[-1]
                WORD_SAMPLES[f"{letter}{side}"] = (stem_of(owner),
                                                   " ".join(said[max(0, hit.start() - 150):hit.end() + 150].split()))
        total = sum(row[0] for key, row in WORD_TALLY.items() if key.startswith("I"))
        due = total % WORD_TALLY_EVERY == 0
        shown = {key: f"{row[1]}/{row[0]} ({row[2] / max(1, row[0]):.1f} a reply)"
                 for key, row in sorted(WORD_TALLY.items())}
        samples = dict(WORD_SAMPLES) if due else {}
    if due:
        doses = " ".join(f"{key} {value}" for key, value in shown.items() if key.startswith("I+") and key != "I+")
        print(f"KNOB WORDS replies with the words, by letter on/anti/off: Interesting(ly) I+ {shown.get('I+', '-')} "
              f"{doses + ' ' if doses else ''}I~ {shown.get('I~', '-')} I- {shown.get('I-', '-')} | Wait/Hmm/Actually "
              f"W+ {shown.get('W+', '-')} W~ {shown.get('W~', '-')} W- {shown.get('W-', '-')} "
              f"| mechanic words N+ {shown.get('N+', '-')} N- {shown.get('N-', '-')}", flush=True)
        for key in sorted(samples):
            print(f"KNOB WORDS sample {key} {samples[key][0]}: ...{samples[key][1]}...", flush=True)


def picture(owner: Any, text: str) -> bool:
    if os.environ.get("ARC3_FIRST_PICTURE_ONLY", "").strip() != "1":
        return True
    found = STATE.search(text or "")
    if found is None:
        say_once("G-no-level", f"KNOB G could not read the level from a turn message of {stem_of(owner)}; "
                               f"the picture is kept")
        return True
    level = int(found.group(1))
    if getattr(owner, "_knob_picture_level", None) == level:
        say_once("G-drop", f"KNOB G first dropped a turn picture for {stem_of(owner)} on level {level}")
        return False
    owner._knob_picture_level = level
    say_once("G-keep", f"KNOB G first used by {stem_of(owner)}: kept the first picture of level {level}, later turn "
                       f"pictures of a level are dropped")
    return True


def text_of(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content or () if isinstance(part, dict)
                   and part.get("type") == "text")


def held_text(message: Any) -> str:
    parts = [text_of(message)]
    for call in (message.get("tool_calls") or ()) if isinstance(message, dict) else ():
        parts.append(str((call.get("function") or {}).get("arguments") or ""))
    return "\n".join(parts)


def budget(level: int, total: int | None) -> int:
    count = total or int(number(LEVELS_GUESS_ENV, 7))
    if level > count:
        count = level + 1
    weight = 0.5 if level == 1 else 1.5 if level == 2 else 1.0
    return int(number(GAME_TOKENS_ENV, 380000) / max(1, count) * weight)


def kilo(value: int) -> str:
    return f"{value / 1000:.0f}K" if value >= 1000 else str(value)


def budget_line(level: int, total: int | None, spent: int) -> str:
    share = budget(level, total)
    why = ""
    if level == 1:
        why = " (half the usual share: the first level is meant to be easy)"
    elif level == 2:
        why = (" (one and a half times the usual share: the second level usually brings a new mechanic, take the "
               "time to understand it)")
    left = share - spent
    tail = f"left: {kilo(left)}" if left >= 0 else f"over budget by {kilo(-left)}"
    return (f"Token budget for level {level}: about {kilo(share)} generated tokens{why}. Spent on this level so far: "
            f"{kilo(spent)}; {tail}.")


def branch_headers(said: dict[str, str]) -> dict[str, str]:
    said = dict(said)
    name = next((key for key in said if str(key).lower() == "x-session-id"), None)
    key = str(said.get(name) or "") if name else ""
    if key and not key.startswith(BRANCH_PREFIX):
        said[name] = f"{BRANCH_PREFIX}-{key}#debrief"
    return said


def debrief(owner: Any, convo: list[dict[str, Any]], tools: Any, level: int) -> str:
    thinking = os.environ.get(DEBRIEF_THINKING_ENV, "0").strip() == "1" or "Q" in (letters_of(owner) or "")
    asked = [*convo, {"role": "user", "content": DEBRIEF_ASK.format(level=level)}]
    if not thinking:
        asked.append({"role": "assistant", "content": DEBRIEF_START})
    held = getattr(owner, "_fork_tool_choice", None)
    held_payload = owner.__dict__.get("_fork_payload")
    started = time.time()
    owner._fork_tool_choice = "none"
    if not thinking:
        owner._fork_payload = {"continue_final_message": True, "add_generation_prompt": False}
    own_headers = owner.__dict__.get("_headers")
    if hasattr(owner, "_headers"):
        base = own_headers or (lambda: type(owner)._headers(owner))
        owner._headers = lambda: branch_headers(base())
    try:
        result = owner._chat_completion(asked, tools=tools,
                                        max_tokens=int(number(DEBRIEF_TOKENS_ENV, 3000 if thinking else 700)),
                                        enable_thinking=None, request_timeout_seconds=300)
    finally:
        owner._fork_tool_choice = held
        owner._fork_payload = held_payload
        if own_headers is not None:
            owner._headers = own_headers
        else:
            owner.__dict__.pop("_headers", None)
    usage = getattr(result, "usage", None) or {}
    if hasattr(owner, "_accumulate_usage_tokens"):
        owner._accumulate_usage_tokens(usage, count_toward_turn=False)
    message = getattr(result, "message", None) or {}
    content = text_of(message)
    reasoning = str(message.get("reasoning") or message.get("reasoning_content") or "")
    said = content if content.strip() else reasoning
    if "</think>" in said:
        head, tail = said.rsplit("</think>", 1)
        said = tail if tail.strip() else head
    said = re.sub(r"</?think>", " ", said).strip()
    if said.startswith(DEBRIEF_START):
        said = said[len(DEBRIEF_START):]
    text = " ".join(CUT.split(said)[0].split())[:1200]
    write({"kind": "debrief", "stem": stem_of(owner), "level": level, "text": text,
           "from": "content" if content.strip() else "reasoning" if reasoning.strip() else "nothing",
           "content_chars": len(content), "reasoning_chars": len(reasoning),
           "tokens": usage.get("completion_tokens"), "seconds": round(time.time() - started, 1)})
    if not text:
        print(f"KNOB DEBRIEF {stem_of(owner)} level {level} came back empty: content {len(content)} and reasoning "
              f"{len(reasoning)} characters", flush=True)
    print(f"KNOB DEBRIEF {stem_of(owner)} level {level} ({usage.get('completion_tokens')} tokens, "
          f"{time.time() - started:.0f} s): {text[:300]}", flush=True)
    return text


def after_trim(owner: Any, messages: list[dict[str, Any]], opener: dict[str, Any], tools: Any,
               append_transcript: Any = None) -> None:
    letters = letters_of(owner)
    if not letters:
        return
    at = next((index for index, message in enumerate(messages) if message is opener), None)
    found = STATE.search(text_of(opener)) if at is not None else None
    if found is None:
        return
    level, total = int(found.group(1)), int(found.group(2)) if found.group(2) else None
    tokens = int(getattr(owner, "_session_generated_tokens", 0) or 0)
    seen = getattr(owner, "_knob_level", None)
    if seen is None or level != seen or tokens < getattr(owner, "_knob_start", 0):
        owner._knob_level, owner._knob_start = level, tokens
    won = seen is not None and level > seen
    if "E" in letters or "H" in letters:
        win_turn(owner, messages[:at])
        if won:
            try:
                win_begin(owner, opener, level - 1, append_transcript)
            except Exception as error:  # noqa: BLE001
                owner.__dict__.pop("_win_pause", None)
                print(f"KNOB E failed for {stem_of(owner)} level {level - 1}: {type(error).__name__}: {error}",
                      flush=True)
    paused = won and getattr(owner, "_win_pause", None) is not None
    if paused and "H" in letters:
        owner._win_pause["ask"] = True
        say_once("H", f"KNOB H first used by {stem_of(owner)}: after the level {level - 1} win pictures the player is asked for next_level(description) written for a person, and the description is pinned")
    if paused and "D" in letters and "H" not in letters:
        owner._win_pause["ask"] = True
        messages.insert(at + 1, {"role": "user", "content": WIN_ASK.format(level=level - 1)})
        say_once("E+D", f"KNOB E+D first used by {stem_of(owner)}: the level {level - 1} win is followed by a user "
                        f"question, and the answer given in the pause is pinned")
    if won and ("D" in letters or "H" in letters) and not paused:
        try:
            text = debrief(owner, messages[:at + 1], tools, level - 1)
        except Exception as error:  # noqa: BLE001
            text = ""
            print(f"KNOB DEBRIEF failed for {stem_of(owner)} level {level - 1}: {type(error).__name__}: {error}",
                  flush=True)
        if text:
            owner._knob_debriefs = [*getattr(owner, "_knob_debriefs", []), DEBRIEF_PIN.format(level=level - 1,
                                                                                               text=text)]
        if text and "F" in letters and at > 1:
            dropped = at - 1
            del messages[1:at]
            at = 1
            say_once("F", f"KNOB F first used by {stem_of(owner)}: after the level {level - 1} debrief the request keeps "
                          f"only the system message and the new level's opener; {dropped} earlier messages dropped")
    if won and "J" in letters:
        content = opener.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        opener["content"] = parts + [{"type": "text", "text": INLINE_ASK.format(level=level - 1)}]
        say_once("J", f"KNOB J first used by {stem_of(owner)}: the level {level} opener asks for a debrief in the reply")
    pins = [*getattr(owner, "_knob_debriefs", [])]
    keep = int(number(RESETS_KEPT_ENV, 6))
    pins += getattr(owner, "_knob_resets", [])[-keep:] if keep > 0 else []
    history = "\n".join(held_text(message) for message in messages[1:at])
    missing = [pin for pin in pins if pin not in history]
    added = [PINS_HEAD, *missing] if missing else []
    if "B" in letters:
        line = budget_line(level, total, max(0, tokens - getattr(owner, "_knob_start", 0)))
        added.append(line)
        say_once("B", f"KNOB B first used by {stem_of(owner)}: {line}")
    if not added:
        return
    content = opener.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
    opener["content"] = parts + [{"type": "text", "text": "\n".join(added)}]
    if missing:
        say_once(f"pins {stem_of(owner)} {len(pins)}", f"KNOB PINS {stem_of(owner)}: {len(missing)} of {len(pins)} "
                                                      f"put back: {missing[-1][:200]}")
    if append_transcript is not None:
        append_transcript("KNOB ARMS", "\n".join(added))


def reset_refused(actions: list[Any], reason: Any) -> str | None:
    letters = getattr(_local, "letters", None)
    if letters and ("E" in letters or "H" in letters):
        pause = getattr(getattr(_local, "owner", None), "_win_pause", None)
        if pause and not pause["open"]:
            return WIN_REFUSED.format(level=pause["level"])
    if not letters or "R" not in letters:
        return None
    if not any(isinstance(item, dict) and str(item.get("action", "")).strip().upper() == "RESET" for item in actions):
        return None
    if str(reason or "").strip():
        return None
    say_once(f"refused {stem_of(getattr(_local, 'owner', None))}",
             f"KNOB R refused a RESET without a reason in {stem_of(getattr(_local, 'owner', None))}")
    return RESET_NEEDS_REASON


def reset_kept(actions: list[Any], reason: Any, result: Any) -> None:
    owner = getattr(_local, "owner", None)
    letters = getattr(_local, "letters", None)
    if owner is None or not letters or "R" not in letters or not isinstance(result, dict) or not result.get("executed"):
        return
    if not any(isinstance(item, dict) and str(item.get("action", "")).strip().upper() == "RESET" for item in actions):
        return
    said = " ".join(str(reason or "").split())[:600]
    level = result.get("level")
    owner._knob_resets = [*getattr(owner, "_knob_resets", []), RESET_PIN.format(level=level, reason=said)]
    write({"kind": "reset", "stem": stem_of(owner), "level": level, "reason": said})
    print(f"KNOB RESET {stem_of(owner)} level {level}: {said[:300]}", flush=True)


def _win_payload(grid: Any, step: int, level: int) -> dict[str, Any]:
    from inference.utils.grid_utils import format_grid_ascii
    rows = [list(row) for row in grid]
    return {"ascii": format_grid_ascii(grid), "step": step, "level": level,
            "shape": [len(rows), max((len(row) for row in rows), default=0)], "grid": rows}


def _win_picture(label: str, grid: Any) -> list[dict[str, Any]]:
    from types import SimpleNamespace

    from inference.agent.vision_context import frame_to_png_data_url
    url = frame_to_png_data_url(SimpleNamespace(grid=[list(row) for row in grid]))
    return [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": url}}]


def win_begin(owner: Any, opener: dict[str, Any], level: int, append_transcript: Any = None) -> None:
    callback = getattr(owner, "_step_env_callback", None)
    raw = callback({"query": "animation"}) if callback is not None else None
    record = raw.get("record") if isinstance(raw, dict) else None
    chain = list((record or {}).get("chain") or ())
    if len(chain) < 2:
        print(f"KNOB E skipped for {stem_of(owner)} level {level}: no animation of the winning action", flush=True)
        return
    step = int((record or {}).get("step") or 0)
    passed = chain[1:-1]
    keep = max(1, int(number("ARC3_WIN_PAUSE_PICTURES", 5)) - 1)
    picks = sorted({round(index * (len(passed) - 1) / max(1, keep - 1)) for index in range(keep)}) \
        if len(passed) > keep else list(range(len(passed)))
    describe = "H" in (letters_of(owner) or "")
    content = [] if describe else [{"type": "text", "text": WIN_TEXT.format(level=level)}]
    won = passed[-1] if passed else chain[0]
    if passed:
        content += _win_picture(WIN_LABELS[0].format(step=step - 1), chain[0])
        for index in picks[:-1]:
            content += _win_picture(WIN_LABELS[1].format(index=index + 1, total=len(passed)), passed[index])
        content += _win_picture(WIN_LABELS[2].format(step=step), won)
    else:
        content += _win_picture(WIN_LABELS[3].format(step=step - 1), won)
    if describe:
        content.append({"type": "text", "text": WIN_TEXT_H.format(level=level)})
    held = json.loads(json.dumps(opener.get("content")))
    opener["content"] = content
    owner._win_pause = {"level": level, "held": held, "won": _win_payload(won, step, level), "replies": 0,
                        "open": False}
    pictures = sum(part.get("type") == "image_url" for part in content)
    say_once("E", f"KNOB E first used by {stem_of(owner)}: the level {level} win is shown in {pictures} picture(s) "
                  f"from an animation of {len(chain) - 1} board(s); the next level is held until next_level()")
    print(f"KNOB E {stem_of(owner)} level {level}: paused with {pictures} picture(s)", flush=True)
    write({"kind": "win_pause", "event": "begin", "stem": stem_of(owner), "level": level, "pictures": pictures,
           "boards": len(chain) - 1})
    if append_transcript is not None:
        append_transcript("KNOB ARMS", f"E: level {level} win shown, next level held")


def win_turn(owner: Any, messages: list[dict[str, Any]]) -> None:
    pause = owner.__dict__.pop("_win_pause", None)
    if pause is None:
        return
    print(f"KNOB E {stem_of(owner)} level {pause['level']}: the turn ended without next_level(); the new turn shows "
          f"the next level", flush=True)
    write({"kind": "win_pause", "event": "turn_end", "stem": stem_of(owner), "level": pause["level"],
           "replies": pause["replies"]})
    win_answer(owner, messages, pause)


def win_answer(owner: Any, messages: list[dict[str, Any]], pause: dict[str, Any]) -> None:
    if not pause.get("ask"):
        return
    level = pause["level"]
    if pause.get("told"):
        text = _sentence_cut(pause["told"])
        print(f"KNOB H {stem_of(owner)} level {level}: pinned the next_level description: {text}", flush=True)
        write({"kind": "debrief", "stem": stem_of(owner), "level": level, "text": text,
               "from": "next_level description"})
        owner._knob_debriefs = [*getattr(owner, "_knob_debriefs", []), WIN_PIN_H.format(level=level, text=text)]
        return
    asked = WIN_ASK.format(level=level)
    at = next((index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"
               and text_of(messages[index]) == asked), None)
    replies = [message for message in messages[at + 1:] if message.get("role") == "assistant"] if at is not None else []
    source = "reply"
    said = " ".join(text_of(message) for message in replies)
    if not said.strip():
        source = "thinking"
        said = " ".join(str(message.get("reasoning") or message.get("reasoning_content") or "") for message in replies[:1])
    said = re.sub(r"</?think>", " ", said)
    text = " ".join(CUT.split(said)[0].split())[:1200]
    if not text:
        print(f"KNOB E+D {stem_of(owner)} level {level}: nothing to pin, the pause has no reply text or thinking"
              f"{'' if at is not None else ' (the question is no longer in the request)'}", flush=True)
        return
    print(f"KNOB E+D {stem_of(owner)} level {level}: pinned the {source} given in the pause: {text}", flush=True)
    write({"kind": "debrief", "stem": stem_of(owner), "level": level, "text": text, "from": f"pause {source}"})
    owner._knob_debriefs = [*getattr(owner, "_knob_debriefs", []), WIN_PIN.format(level=level, text=text)]


def win_state(owner: Any, state: dict[str, Any]) -> dict[str, Any]:
    pause = getattr(owner, "_win_pause", None)
    if not pause or pause["open"]:
        return state
    state["current_frame"] = pause["won"]
    history = state.get("history")
    if history:
        history[-1] = {**history[-1], "frame": pause["won"]}
    return state


def _sentence_cut(text: str, most: int = 2500) -> str:
    text = " ".join(text.split())
    if len(text) <= most:
        return text
    head = text[:most]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    return head[:end + 1] if end > most // 2 else head


def win_next(description: Any = None) -> tuple[bool, str]:
    owner = getattr(_local, "owner", None)
    pause = getattr(owner, "_win_pause", None)
    if not pause or pause["open"]:
        return False, WIN_IDLE
    if "H" in (letters_of(owner) or ""):
        told = " ".join(str(description or "").split())
        if len(told) < 20:
            return False, WIN_NEEDS_TEXT
        pause["told"] = told
    pause["open"] = True
    return True, WIN_OPENED


def win_after_tool(owner: Any, messages: list[dict[str, Any]]) -> None:
    pause = getattr(owner, "_win_pause", None)
    if not pause:
        return
    pause["replies"] += 1
    cap = max(1, int(number("ARC3_WIN_PAUSE_REPLIES", 4)))
    if not pause["open"] and pause["replies"] < cap:
        return
    del owner._win_pause
    win_answer(owner, messages, pause)
    held = pause["held"]
    parts = [{"type": "text", "text": held}] if isinstance(held, str) else list(held or [])
    if not pause["open"]:
        parts = [{"type": "text", "text": WIN_AUTO.format(replies=pause["replies"])}, *parts]
    owner._append_context_message(messages, {"role": "user", "content": parts})
    how = "next_level()" if pause["open"] else "on its own"
    print(f"KNOB E {stem_of(owner)} level {pause['level']}: the next level shown after {pause['replies']} "
          f"repl{'y' if pause['replies'] == 1 else 'ies'}, {how}", flush=True)
    write({"kind": "win_pause", "event": "open", "stem": stem_of(owner), "level": pause["level"],
           "replies": pause["replies"], "called": pause["open"]})
