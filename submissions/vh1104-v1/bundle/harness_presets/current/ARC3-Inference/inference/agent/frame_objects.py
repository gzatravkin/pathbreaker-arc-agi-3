import os
import re

ENV = "ARC3_FRAME_OBJECTS"
VANISHED_ROWS = 40
SAID = set()
API = ("`current_frame.grid` (a list of rows, each a string of colour letters), `current_frame.objects` "
       "(a dict from type label to the list of that type's objects), `current_frame.owner(row, col)` and "
       "`current_frame.color(row, col)`")
SWAPS = [
    (r"- `current_frame` exposes only `\.ascii`, `\.step`, `\.level`, `\.shape`, and `\.segmentation`\.\n",
     "- `current_frame` exposes only `.grid`, `.objects`, `.owner(row, col)`, `.color(row, col)`, `.step`, `.level`, "
     "and `.shape`.\n"),
    (r"- `current_frame\.ascii` is a single newline-delimited string[^\n]*\n",
     "- `current_frame.grid` is the whole board as a list of rows, each a string of the letter-coded ARC colour "
     "symbols; `grid[row][col]` is one cell.\n"),
    (r"- `current_frame\.segmentation` parses the board into objects\.[^\n]*\n",
     "- `current_frame.objects` maps each object type label of the board digest and catalog to the list of that "
     "type's objects; each object is a dict with `rotation`, `colours` (colour letters), `cells` (a list of "
     "`(row, col)`), `bbox` (`[r0, c0, r1, c1]`) and `history` (its `step`, `event`, `bbox`, `colours` and "
     "`rotation` on this level, matched by world position across camera scrolls; events are start, appeared, "
     "moved, turned, changed; its first entry and its latest 39). Bars, regions and lines are listed under their "
     "kind. `current_frame.vanished` lists the latest 40 objects that left on this level, each with its `key`, the "
     "same fields and its `history`. Every other "
     "frame's `.objects` is the same map without history.\n"
     "- Every object also answers `.center`, `.top`, `.bottom`, `.left`, `.right`, `.top_left`, `.top_right`, "
     "`.bottom_left` and `.bottom_right` (or `obj[\"<point>\"]`) with the `(row, col)` of its own cell nearest that "
     "point of its bbox, to click with `MOUSE`; `obj.anchor_colours` maps those points to colour letters and exists "
     "only when the letters differ.\n"),
    (r"- Each node in `segmentation\['nodes'\]`[^\n]*\n",
     "- `current_frame.owner(row, col)` is the address `(label, index)` of the object that holds that cell, so "
     "`current_frame.objects[label][index]` is that object, or `None` for background; `frame.color(row, col)` is the "
     "colour letter of one cell.\n"),
    (r"- `segmentation\['adjacency_list'\]`[^\n]*\n", ""),
    (r"- The raw numeric grid is intentionally not exposed\. Use `current_frame\.segmentation`[^\n]*\n",
     "- Read the board through " + API + "; print only what you need, never the whole board.\n"),
    (r"each frame exposes only `\.ascii`, `\.step`, `\.level`, `\.shape`, and `\.segmentation`\.",
     "each frame exposes `.grid`, `.objects`, `.owner(row, col)`, `.color(row, col)`, `.step`, `.level` and "
     "`.shape`; only `current_frame.objects` carries `history`."),
    (r"as views with `\.ascii`, `\.shape` and `\.segmentation`, the same as `current_frame`\.",
     "as views with `.grid`, `.objects`, `.owner(row, col)`, `.color(row, col)` and `.shape`."),
    (r"Crop their `\.ascii` yourself;", "Crop their `.grid` yourself;"),
    (r"- The image and `current_frame\.ascii` are two representations",
     "- The image and `current_frame.grid` are two representations"),
    (r"- Use `current_frame\.segmentation` as your primary view of the board[^\n]*\n",
     "- Read the board through " + API + "; the objects carry the labels of the board digest.\n"),
    (r"- Use `current_frame\.ascii` only to read a small, specific region[^\n]*\n", ""),
    (r"Object hashes use the same rotation-invariant shape-and-color representation as segmentation;",
     "Object hashes are a rotation-invariant shape-and-color representation;"),
    (r"`current_frame` and each `history\[\*\]\.frame` expose only `\.ascii`, `\.segmentation`, `\.step`, `\.level`, "
     r"and `\.shape` \(a `\(rows, cols\)` tuple\); ",
     "`current_frame` exposes `.grid`, `.objects`, `.owner(row, col)`, `.color(row, col)`, `.step`, `.level` and "
     "`.shape` (a `(rows, cols)` tuple), and so does each `history[*].frame`; only `current_frame.objects` carries "
     "each object's `history` on this level, and `current_frame.vanished` the objects that left; "),
    (r"The raw numeric grid is not available\. Use `\.segmentation` as the primary view; use `\.ascii` only to read a "
     r"small, specific region\. ",
     "Read the board through `.grid`, `.objects`, `.owner(row, col)` and `.color(row, col)`. "),
    (r"inspect `current_frame\.segmentation`", "inspect `current_frame.objects`"),
    (r"\(segmentation nodes, slicing, run-length checks, BFS\)", "(objects, owner, slicing of grid, BFS)"),
    (r"use `current_frame\.segmentation` as the primary view, and `current_frame\.ascii` only for a small specific "
     r"region", "read the board through " + API),
    (r"use `current_frame\.segmentation` as the primary view, and `\.ascii` only for a small specific region",
     "read the board through " + API),
    (r"Only letter-coded board views and lightweight metadata are exposed;",
     "Only letter-coded board views, labelled objects and lightweight metadata are exposed;"),
    (r"against preloaded ASCII game state\.", "against the preloaded game state."),
    (r"and `action\(actions\)` for executing one or more real environment actions\. ",
     "and `action(actions)` for executing one or more real environment actions; `action(actions, show_delta=True)` "
     "also prints the board change of exactly that call, which the next turn's opener then leaves out. "),
    (r"\.ascii, \.segmentation, \.step, \.level, \.shape\.",
     ".grid, .objects, .owner(row, col), .color(row, col), .step, .level, .shape; current_frame.objects also "
     "carries each object's history and current_frame.vanished the objects that left."),
    (r"with only \.ascii, \.segmentation, \.step, \.level and \.shape",
     "with only .grid, .objects, .owner(row, col), .color(row, col), .step, .level and .shape"),
    (r"\(current_frame, history, segmentation\)", "(current_frame, history, objects)"),
    (r"it sees the board as text and segmented objects and an image each turn",
     "it sees the board as letters, labelled objects and an image each turn"),
    (r"check anything you rely on in python\.", "any object's cells: current_frame.objects[label] / owner(r, c)."),
    (r"you read current_frame\.segmentation;", "you read current_frame.objects;"),
]
COMPILED = [(re.compile(old), new) for old, new in SWAPS]
NO_GRID_ENV = "ARC3_NO_BOARD_PRINT"
NO_GRID_SWAPS = [
    (r"`current_frame\.grid` \(a list of rows, each a string of colour letters\), ", ""),
    (r"`\.grid`, ", ""),
    (r"(?<![\w`])\.grid, ", ""),
    (r"- `current_frame\.grid` is the whole board as a list of rows[^\n]*\n",
     "- The board is not given as text in code: read it from the image, the board digest, `current_frame.objects` and "
     "`current_frame.color(row, col)`. A python call shows at most about 100 characters of its own printed output "
     "(results of `action` and `frame_diff` are shown whole), so print coordinates, counts and short notes.\n"),
    (r"Crop their `\.grid` yourself;", "Read their cells with `.color(row, col)`;"),
    (r"The image and `current_frame\.grid` are two representations",
     "The image and `current_frame.color(row, col)` are two representations"),
    (r"\(objects, owner, slicing of grid, BFS\)", "(objects, owner, color, BFS)"),
]
NO_GRID_COMPILED = [(re.compile(old), new) for old, new in NO_GRID_SWAPS]
LOOKAT_ENV = "ARC3_LOOKAT"
LOOKAT_SWAPS = [
    (r"(- `current_frame\.objects` maps each object type label[^\n]*\n)",
     r"\1- `LookAt(<row>, <col>, <h>, <w>)` shows a picture of that piece of the current board at the main picture's "
     r"scale in the next message (8x8 cells by default, at most 16x16, at most 4 pictures a python call).\n"),
]
LOOKAT_COMPILED = [(re.compile(old), new) for old, new in LOOKAT_SWAPS]
ANCHORS = ("center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right")
ANCHOR_USE = re.compile(r"[.](?:" + "|".join(ANCHORS) + r")\b|\[[\'\"](?:" + "|".join(ANCHORS) + r")[\'\"]\]")


def on():
    return os.environ.get(ENV, "").strip() == "1"


def say_once(key, line):
    if key not in SAID:
        SAID.add(key)
        print(line, flush=True)


def no_grid():
    return on() and os.environ.get(NO_GRID_ENV, "").strip() == "1"


def text(value):
    if not on() or not isinstance(value, str):
        return value
    for pattern, new in COMPILED:
        value = pattern.sub(new, value)
    if no_grid():
        for pattern, new in NO_GRID_COMPILED:
            value = pattern.sub(new, value)
    if os.environ.get(LOOKAT_ENV, "").strip() == "1" and "LookAt(" not in value:
        for pattern, new in LOOKAT_COMPILED:
            value = pattern.sub(new, value)
    return value


def apply_module(names):
    if not on():
        return 0
    changed = 0
    for key, value in list(names.items()):
        if isinstance(value, str) and ("segmentation" in value or "ascii" in value or "in python." in value):
            swapped = text(value)
            if swapped != value:
                names[key] = swapped
                changed += 1
    return changed


def anchors_of(cells, box, rows=None):
    r0, r1, c0, c1 = box
    rm, cm = (r0 + r1) / 2, (c0 + c1) / 2
    points = {"center": (rm, cm), "top": (r0, cm), "bottom": (r1, cm), "left": (rm, c0), "right": (rm, c1),
              "top_left": (r0, c0), "top_right": (r0, c1), "bottom_left": (r1, c0), "bottom_right": (r1, c1)}
    if not cells:
        return {}, None
    anchors = {name: list(min(cells, key=lambda one: ((one[0] - r) ** 2 + (one[1] - c) ** 2, one[0], one[1])))
               for name, (r, c) in points.items()}
    if not rows:
        return anchors, None
    letters = {name: rows[at[0] - r0][at[1] - c0] for name, at in anchors.items()}
    return anchors, letters if len(set(letters.values())) > 1 else None


def objects_of(found):
    from inference.agent import perception
    out, where = {}, {}
    for j, x in enumerate(found.get("inst") or ()):
        r0, r1, c0, c1 = x["box"]
        group = out.setdefault(x["type"], [])
        where[j] = (x["type"], len(group))
        cells = perception.cells_of(x.get("cells"))
        anchors, letters = anchors_of(cells, x["box"], x.get("rows"))
        one = {"rotation": perception.rotation_of(x), "colours": perception.letters_of(x), "cells": cells,
               "bbox": [r0, c0, r1, c1], "anchors": anchors}
        if letters:
            one["anchor_colours"] = letters
        group.append(one)
    for a in found.get("areas") or ():
        r0, r1, c0, c1 = a["box"]
        kind = {"hbar": "bar", "vbar": "bar"}.get(a["kind"], a["kind"])
        cells = perception.cells_of(a["cells"])
        out.setdefault(kind, []).append({"rotation": 0, "colours": [a["colour"]], "cells": cells,
                                         "bbox": [r0, c0, r1, c1], "anchors": anchors_of(cells, a["box"])[0]})
    return out, where


def used(owner, code):
    if not on() or not isinstance(code, str) or "anchors" in SAID:
        return
    hit = ANCHOR_USE.search(code)
    if hit and (".objects" in code or "owner(" in code):
        at = code.rfind(chr(10), 0, hit.start()) + 1
        line = code[at:code.find(chr(10), hit.end()) if chr(10) in code[hit.end():] else len(code)].strip()
        who = str(getattr(owner, "_forum_key", "") or (getattr(owner, "_burn_said", None) or ("a player",))[0])
        say_once("anchors", f"CLICK OBJECT first used by {who}: {line[:160]}")


def match(perception, prev, new, prev_grid, new_grid):
    return perception.match(prev, new, prev_grid, new_grid)


def tracks(perception, owner, frames):
    st = perception.state_of(owner)
    tracker = st.tracker
    tracker.feed(st, [(step, grid, 0) for step, grid in frames])
    live = {j: tracker.items[tid]["history"] for j, tid in tracker.live.items()}
    vanished = [vanished_row(item) for item in tracker.items if not item["alive"]]
    return live, vanished, tracker.digests[tracker.steps[-1]]


def vanished_row(item):
    from inference.agent import perception
    last = item["last"]
    r0, r1, c0, c1 = last["box"]
    return {"key": item["key"], "rotation": perception.rotation_of(last), "colours": perception.letters_of(last),
            "cells": perception.cells_of(last.get("cells")), "bbox": [r0, c0, r1, c1], "history": item["history"]}


def lookup(owner, grid):
    from inference.agent import perception
    _, found = perception.state_of(owner).digest_of(grid)
    return objects_of(found)[0]


def payload(owner, frame, view, history=None):
    if not on() or view is None or frame is None or not getattr(frame, "grid", None):
        return view
    try:
        from inference.agent import perception
        st = perception.state_of(owner)
        tracker = perception.tracked(st, frame, history)
        found = tracker.digests[tracker.steps[-1]]
        objects, where = objects_of(found)
        for j, (key, index) in where.items():
            tid = tracker.live.get(j)
            objects[key][index]["history"] = tracker.items[tid]["history"] if tid is not None else []
        dead = sorted((item for item in tracker.items if not item["alive"]), key=lambda item: item["history"][-1]["step"])
        vanished = [vanished_row(item) for item in dead[-VANISHED_ROWS:]]
        view = {**view, "objects": objects, "vanished": vanished}
        who = str(getattr(owner, "_forum_key", "") or (getattr(owner, "_burn_said", None) or ("a player",))[0])
        if no_grid():
            say_once("no-grid", f"NO BOARD PRINT first used by {who}: current_frame.grid is gone from every frame and "
                                f"a python call prints at most 100 characters besides action and frame_diff results")
        say_once("payload", f"FRAME OBJECTS first used by {who}: current_frame.objects has "
                            f"{sum(len(one) for one in objects.values())} object(s) of {len(objects)} type(s), "
                            f"{len(vanished)} vanished, histories over {len(tracker.steps)} step(s) of this level; "
                            f"ascii and segmentation are gone from every frame")
        return view
    except Exception as exc:
        say_once("payload-error", f"FRAME OBJECTS could not build objects: {type(exc).__name__}: {str(exc)[:200]}")
        return view
