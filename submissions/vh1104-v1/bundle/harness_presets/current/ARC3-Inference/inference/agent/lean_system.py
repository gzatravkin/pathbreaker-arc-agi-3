import os

ENV = "ARC3_LEAN_SYSTEM"
SAID = set()
TEXT = """You are playing an unfamiliar grid puzzle game through a Python tool. Solve every level of the game.

The game
- The board is 64x64 cells. Colours are letters: W=white, w=light gray, g=gray, G=dark gray, c=charcoal, B=black, M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, r=dark red, N=light green, p=purple.
- Nobody tells you the rules or the goal: find them by acting and watching what changes. Later levels usually reuse earlier mechanics and add new ones.
- Each turn shows a picture of the board and a board digest: an object list the harness computes, which can be wrong.
- Every game action counts. A level scores well when you finish it in about as many actions as a person would; you do not need the shortest possible solution. Running Python costs no actions.
- `level_completed` means one level is done. `game_over` means the attempt failed and the level restarted; finished levels are kept. `run_complete` means the whole game is solved.

Actions (only those in `valid_actions`)
- `UP`, `DOWN`, `LEFT`, `RIGHT` are directional; what they move depends on the game. `SPACE` is a game-specific action. `MOUSE` clicks one cell: integer `row` and `col`, 0-63, row grows downward, col grows rightward. `UNDO` reverses the last action.
- Act only by calling `action(<list>)` inside Python. Each item is an action name, or `{'action': 'MOUSE', 'row': <row>, 'col': <col>}`. You may batch several actions in one call and call `action` several times in one snippet; a batch stops at `game_over` or `level_completed`.
- The result reports `level_completed`, `game_over`, `board_changed`, `gameplay_changed` (ignores the 4 cells along each edge, where step budgets usually sit) and an `animation` summary when the action animated.

The python tool
- One tool, `python`, with one `code` string. Variables reset between calls. Top-level functions you define stay available for the rest of the game, with their top-level imports. Allowed imports: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string. Each call has 30 seconds and about 3,000 tokens of output.
{BOARD_API}
- `previous_frame` is the board before the latest action. `history` is a list of entries with `.action`, `.frame` (the board after that action) and `.result`. `frame_diff(<before>, <after>)` lists moved, rotated, appeared, disappeared, recoloured and resized parts (defaults: previous to current).
- `last_animation_frames` and `last_animation_timeline` hold the frames of the latest action that animated (after a death, the fatal action); `reset_animation_frames` and `reset_animation_timeline` hold those of the latest RESET. The games were made for people: a long animation probably shows something you need.

Helpers
- A helper may be assigned to your game while you are stuck. Its notes and functions arrive at the start of a turn. `Advisors.Reply(<text>)` answers them and `Advisors.Ask(<question>)` asks for help.

When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax."""
POINTS = ("`.center`, `.top`, `.bottom`, `.left`, `.right`, `.top_left`, `.top_right`, `.bottom_left`, "
          "`.bottom_right`")
LEVEL_START = ("You completed the previous level. `current_frame` shows the starting board of the next level; "
               "your functions are kept.")


def flag(name):
    return os.environ.get(name, "").strip() == "1"


def on():
    return flag(ENV)


def board_api():
    if not flag("ARC3_FRAME_OBJECTS"):
        return ("- `current_frame`: `.ascii` (rows of colour letters), `.segmentation` (objects), `.step`, `.level`, "
                "`.shape`.")
    objects = (f"`.objects` (type label -> objects with `cells`, `bbox`, `colours`, `rotation`, `history` and click "
               f"points {POINTS}), `.owner(row, col)`, `.color(row, col)`, `.step`, `.level`, `.shape`; "
               f"`current_frame.vanished` lists the objects that left this level.")
    if flag("ARC3_NO_BOARD_PRINT"):
        lines = [f"- `current_frame`: {objects} There is no board text in code.",
                 "- A python call shows at most about 100 characters of its own printed output; results of `action` "
                 "and `frame_diff` are shown whole."]
    else:
        lines = [f"- `current_frame`: `.grid` (rows of colour letters, `grid[row][col]`), {objects}"]
    if flag("ARC3_LOOKAT"):
        lines.append("- `LookAt(<row>, <col>, <h>, <w>)` shows a picture of that piece of the current board in the next "
                     "message: 8x8 cells by default, at most 16x16, at most 4 a call.")
        lines.append("- To see a part of the board, use `LookAt`, not printed `color` values. Use `color(<row>, <col>)` "
                     "and `objects` inside your code: for path finding, searches and counting over the board.")
    lines.append("- `action(<list>, show_delta=True)` also prints the board change of that call; the next turn's "
                 "change then starts after it.")
    return "\n".join(lines)


def system(prompt, who=""):
    if not on():
        return prompt
    lean = TEXT.replace("{BOARD_API}", board_api())
    if flag("ARC3_LEAN_COMPACT_PRINT"):
        board = "current_frame.grid" if flag("ARC3_FRAME_OBJECTS") else "current_frame.ascii"
        lean = lean.replace("- `previous_frame` is the board before the latest action.",
                            "- Never print or echo full board frames. Return only compact derived summaries such "
                            "as object lists, diffs, coordinates, counts, or tiny local crops.\n"
                            f"- Use `{board}` only to read a small, specific region; do not scan the whole board "
                            "with it.\n- `previous_frame` is the board before the latest action.", 1)
        if "compact" not in SAID:
            SAID.add("compact")
            print(f"LEAN COMPACT PRINT first used by {who or 'a player'}: the lean system message carries the "
                  "two compact-printing lines", flush=True)
    if not flag("ARC3_PERCEPTION_DIGEST"):
        lean = lean.replace("- Each turn shows a picture of the board and a board digest: an object list the harness "
                            "computes, which can be wrong.", "- Each turn shows a picture of the board.")
    if "lean" not in SAID:
        SAID.add("lean")
        print(f"LEAN SYSTEM first used by {who or 'a player'}: the system message is {len(lean)} characters "
              f"instead of {len(prompt)}; the game metadata line still follows it", flush=True)
    return lean


def apply_module(names):
    if on() and "LEVEL_START_USER_PROMPT" in names:
        names["LEVEL_START_USER_PROMPT"] = LEVEL_START
