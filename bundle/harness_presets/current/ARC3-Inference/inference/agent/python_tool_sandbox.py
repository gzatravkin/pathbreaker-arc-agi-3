"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import frame_diff as _frame_diff_module
from inference.utils import segmentation as _segmentation
from inference.utils import retained_functions as _retained_functions
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""
    FRAME_OBJECTS = False
    COLOUR_NAMES = []
    NO_BOARD_PRINT = False
    PRINT_BUDGET = 100
    LOOKAT = False
    LOOKAT_MAX = 4
    LOOKAT_SIDE = 16
    LOOK_SIDE = 64
    CUT_NOTE = "(cut: board text is not printed; use the image, objects, color(r,c))"

    __SEGMENTATION_SOURCE__

    __FRAME_DIFF_SOURCE__

    __RETAINED_FUNCTIONS_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        # pure-Python sequence comparison, no I/O. Measured across 32
        # transcripts it was imported 9 times and USED none of them - the model
        # reaches for it on "compare two things" and then writes the comparison
        # by hand. Allowing it therefore buys robustness, not capability: a
        # stray import line no longer costs the whole snippet.
        "difflib",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        # Exception types. Only four were present, so anything narrower than
        # `except Exception` failed at the except clause and took the snippet
        # with it. These are inert classes and expose nothing; SystemExit and
        # KeyboardInterrupt stay out because the sandbox uses them for its own
        # timeout and shutdown handling.
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RecursionError",
        "StopIteration",
        "UnboundLocalError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameObject(dict):
        def __init__(self, data):
            data = dict(data)
            anchors = data.pop("anchors", None) or {}
            super().__init__(data)
            self.__dict__["_anchors"] = {name: tuple(at) for name, at in anchors.items()}

        def __missing__(self, key):
            anchors = self.__dict__.get("_anchors") or {}
            if key in anchors:
                return anchors[key]
            raise KeyError(key)

        def get(self, key, default=None):
            if dict.__contains__(self, key) or key in (self.__dict__.get("_anchors") or {}):
                return self[key]
            return default

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'FrameObject' object has no attribute '{name}'") from None

    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid, objects=None, vanished=None):
            self._ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None
            self._objects = objects
            self._vanished = vanished
            self._fetched = objects is not None
            self._wrapped = False
            self._owners = None
            self._letters = None

        def _segment(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        @property
        def ascii(self):
            if FRAME_OBJECTS:
                raise AttributeError("'FrameView' object has no attribute 'ascii'")
            return self._ascii

        @property
        def segmentation(self):
            if FRAME_OBJECTS:
                raise AttributeError("'FrameView' object has no attribute 'segmentation'")
            return self._segment()

        def __getattr__(self, name):
            if FRAME_OBJECTS and NO_BOARD_PRINT and name == "grid":
                raise AttributeError("'FrameView' object has no attribute 'grid'")
            if FRAME_OBJECTS and name == "grid":
                if self.__dict__.get("_letters") is None:
                    self._letters = ["".join(COLOR_CHARS[int(v)] if 0 <= int(v) < len(COLOR_CHARS) else "?"
                                             for v in row) for row in self.__dict__.get("_grid") or ()]
                return self._letters
            if FRAME_OBJECTS and name == "objects":
                return self._objects_view()
            if FRAME_OBJECTS and name == "vanished" and self.__dict__.get("_vanished") is not None:
                return self.__dict__["_vanished"]
            if FRAME_OBJECTS and name == "owner":
                return self._owner_of
            if FRAME_OBJECTS and name == "color":
                return self._color_of
            raise AttributeError(f"'FrameView' object has no attribute '{name}'")

        def _objects_view(self):
            if not self.__dict__.get("_fetched"):
                self._fetched = True
                self._objects = _objects_fetch(self.__dict__.get("_grid") or [])
            found = self.__dict__.get("_objects")
            if found and not self.__dict__.get("_wrapped"):
                self._wrapped = True
                self._objects = found = {key: [one if isinstance(one, FrameObject) or not isinstance(one, dict)
                                               else FrameObject(one) for one in group]
                                         for key, group in found.items()}
            return found

        def _owner_of(self, row, col):
            if self._owners is None:
                self._owners = {}
                for key, found in (self._objects_view() or {}).items():
                    for index, one in enumerate(found):
                        for r, c in one.get("cells") or ():
                            self._owners.setdefault((int(r), int(c)), (key, index))
            return self._owners.get((int(row), int(col)))

        def _color_of(self, row, col):
            value = int(self._grid[int(row)][int(col)])
            return COLOR_CHARS[value] if 0 <= value < len(COLOR_CHARS) else value

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _objects_fetch(grid):
        _send({"type": "objects", "grid": grid})
        reply = _recv()
        return reply.get("objects") if reply.get("type") == "objects_result" else None


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
            objects=payload.get("objects"),
            vanished=payload.get("vanished"),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                    result=entry.get("result"),
                )
            )
        return items


    def _transitions_from_history(history):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result=entry.result,
                )
            )
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        try:
            with open("/proc/self/oom_score_adj", "w") as handle:
                handle.write("1000")
        except OSError:
            pass
        try:
            memory = int(float(os.environ.get("ARC3_SANDBOX_MEMORY_GIB") or 16) * (1 << 30))
        except ValueError:
            memory = 16 << 30
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
            (getattr(resource, "RLIMIT_AS", None), memory if memory > 0 else None),
        ):
            if limit is None or value is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        global FRAME_OBJECTS, COLOUR_NAMES, NO_BOARD_PRINT, LOOKAT, LOOKAT_MAX
        FRAME_OBJECTS = bool(initial.get("frame_objects"))
        NO_BOARD_PRINT = FRAME_OBJECTS and bool(initial.get("no_board_print"))
        LOOKAT = FRAME_OBJECTS and bool(initial.get("lookat"))
        LOOK = bool(initial.get("look"))
        LOOKAT_MAX = int(initial.get("look_max") or LOOKAT_MAX)
        COLOUR_NAMES = list(initial.get("colour_names") or [])
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        _whole = []
        _print_left = [PRINT_BUDGET]
        _print_cut = []

        def _keep_whole(value):
            _whole.append(value)
            return value

        def _cut_print(*args, sep=" ", end="\n", file=None, flush=False):
            if any(any(arg is one for one in _whole) for arg in args):
                print(*args, sep=sep, end=end)
                return
            text = str(" " if sep is None else sep).join(str(arg) for arg in args) + str("\n" if end is None else end)
            keep = text[:max(0, _print_left[0])]
            _print_left[0] -= len(text)
            if keep:
                print(keep, end="")
            if _print_left[0] < 0 and not _print_cut:
                _print_cut.append(True)
                print(("" if not keep or keep.endswith("\n") else "\n") + CUT_NOTE)

        if NO_BOARD_PRINT:
            runtime_globals["__builtins__"]["print"] = _cut_print

        _lookats = []

        def LookAt(row, col, h=8, w=8):
            frame = runtime_globals.get("current_frame")
            grid = frame.__dict__.get("_grid") if frame is not None else None
            if not grid:
                return _keep_whole("LookAt: there is no board to look at")
            if len(_lookats) >= LOOKAT_MAX:
                return _keep_whole(f"LookAt refused: at most {LOOKAT_MAX} pictures a call")
            h, w = max(1, min(LOOKAT_SIDE, int(h))), max(1, min(LOOKAT_SIDE, int(w)))
            r0 = max(0, min(len(grid) - 1, int(row)))
            c0 = max(0, min(len(grid[0]) - 1, int(col)))
            r1, c1 = min(len(grid), r0 + h) - 1, min(len(grid[0]), c0 + w) - 1
            step = getattr(frame, "step", 0)
            _lookats.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "step": step,
                             "cells": [[int(v) for v in grid[r][c0:c1 + 1]] for r in range(r0, r1 + 1)]})
            return _keep_whole(f"LookAt rows {r0}-{r1}, cols {c0}-{c1} of step {step}: its picture comes after "
                               f"this tool result")

        if LOOKAT:
            runtime_globals["LookAt"] = LookAt

        def Look(grid):
            if len(_lookats) >= LOOKAT_MAX:
                return _keep_whole(f"Look refused: at most {LOOKAT_MAX} pictures a call")
            if getattr(grid, "ascii", None) is not None:
                grid = str(grid.ascii).split("\n")
            elif getattr(grid, "grid", None) is not None:
                grid = grid.grid
            if isinstance(grid, str):
                grid = grid.split("\n")
            if isinstance(grid, (list, tuple)):
                grid = [row for row in grid if not isinstance(row, str) or row.strip()]
            try:
                rows = [list(row) for row in grid]
            except TypeError:
                return _keep_whole("Look needs a list of rows, each a string of colour letters or a list of colour "
                                   "numbers 0-15")
            if not rows or not rows[0]:
                return _keep_whole("Look: the grid is empty")
            if len(rows) > LOOK_SIDE or any(len(row) > LOOK_SIDE for row in rows):
                return _keep_whole(f"Look refused: at most {LOOK_SIDE}x{LOOK_SIDE} cells")
            if any(len(row) != len(rows[0]) for row in rows):
                return _keep_whole("Look: every row needs the same length")
            cells = []
            for row in rows:
                out = []
                for value in row:
                    if isinstance(value, str):
                        index = COLOR_CHARS.find(value) if len(value) == 1 else -1
                    else:
                        try:
                            index = int(value)
                        except (TypeError, ValueError):
                            index = -1
                    if not 0 <= index <= 15:
                        return _keep_whole(f"Look: {value!r} is not a colour letter or a colour number 0-15")
                    out.append(index)
                cells.append(out)
            h, w = len(cells), len(cells[0])
            frame = runtime_globals.get("current_frame")
            step = getattr(frame, "step", 0) if frame is not None else 0
            _lookats.append({"r0": 0, "r1": h - 1, "c0": 0, "c1": w - 1, "step": step, "cells": cells,
                             "label": f"Look: your grid of {h} rows x {w} columns"})
            return _keep_whole(f"Look: the picture of your {h}x{w} grid comes after this tool result")

        if LOOK:
            runtime_globals["Look"] = Look
        _animation_views = []

        def frame_diff(before=None, after=None):
            # Object-level diff between two frames. Defaults: previous_frame -> current_frame.
            # Returns only what changed; unchanged objects are omitted.
            if before is None:
                before = runtime_globals.get("previous_frame")
            if after is None:
                after = runtime_globals.get("current_frame")
            if before is None or after is None:
                return {"error": "need two frames (no previous_frame available?)"}
            if before.shape != after.shape:
                return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
                        "before_shape": before.shape, "after_shape": after.shape}
            return compute_frame_diff(
                before._grid,
                after._grid,
                before._segment().get("nodes", []),
                after._segment().get("nodes", []),
            )

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_call_result = state_payload.get("last_action_call_result")
            action_result = (
                dict(last_action_call_result) if isinstance(last_action_call_result, dict) else {}
            )
            # action_echo is printed by action(), then removed from its return value.
            # Keep the exposed call result identical across subsequent inspections.
            action_result.pop("action_echo", None)
            transitions = _transitions_from_history(history)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["frame_diff"] = ((lambda *args, **kwargs: _keep_whole(frame_diff(*args, **kwargs)))
                                           if NO_BOARD_PRINT else frame_diff)
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = action_result

        # Both refusals derive from BaseException, not Exception, so a
        # defensive `except Exception:` around action(...) - which models write
        # routinely - cannot swallow them. A refusal means the action did not
        # run at all, so the snippet's assumed state is already wrong and every
        # line after it rests on a false premise. One run treated 1500
        # consecutive refusals as proof that every cell on the board was inert.
        # StaleStateActionError below is deliberately catchable: it reports a
        # belief that MAY be stale rather than a verdict on the action.
        class KnownNoOpActionError(BaseException):
            pass  # raised INSTEAD of executing an action already proven inert here

        class KnownDeathActionError(BaseException):
            pass  # raised INSTEAD of executing an action that ended a prior attempt

        # Raised INSTEAD of executing an action this snippet has already run
        # from this exact board. Buggy loop code can spend a whole life bar
        # getting nowhere - a cursor that wraps turns RIGHT into a short cycle,
        # a mistaken `while pos < j: LEFT` walks away from its target forever -
        # and the terminal latch only stops that once the game actually ends,
        # measured at 20 to 30 wasted actions. BaseException like the two above:
        # the shape that produces these loops is `try: action(...) except
        # Exception:` inside a while, which would otherwise swallow the verdict
        # and keep going. No override slot: the snippet is not reading the
        # refusal, its code is running, and writing a new snippet IS the
        # response - the map is rebuilt per execution.
        class RepeatedActionInStateError(BaseException):
            pass

        # Raised when a PREVIOUS action in this snippet changed nothing, and the
        # snippet then tries to act again. Deliberately a plain Exception, not
        # BaseException: unlike the two refusals above this is a warning about
        # the snippet's beliefs rather than a verdict on the action, and a model
        # that has re-read current_frame may legitimately want to proceed.
        class StaleStateActionError(Exception):
            pass

        runtime_globals["KnownNoOpActionError"] = KnownNoOpActionError
        runtime_globals["KnownDeathActionError"] = KnownDeathActionError
        runtime_globals["StaleStateActionError"] = StaleStateActionError
        runtime_globals["RepeatedActionInStateError"] = RepeatedActionInStateError
        # armed when an action changes nothing; scoped to THIS snippet because
        # the closure is rebuilt per execution, so no reset path can be missed
        _stale_state: list[str] = []

        # Raised when a snippet tries to act after the attempt or level ended.
        # Derives from BaseException, not Exception, so a defensive
        # `try: action(...) except Exception:` around the call - which models
        # write routinely - cannot swallow it. Continuing to execute a path
        # computed against a board that no longer exists is never correct.
        class TerminalStateActionError(BaseException):
            pass

        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
        # latched when an action reports a terminal result; scoped to THIS
        # snippet because these closures are rebuilt for every execution, so
        # no reset path can be missed and the next snippet always starts clear
        _terminal_state = []

        _TERMINAL_DETAIL = {
            "game_over": (
                "the previous action reported game_over. The level has been reset to its "
                "start, so the board you planned against is gone and this action was not "
                "executed. The next turn begins from the new state."
            ),
            "level_completed": (
                "the previous action reported level_completed. A new level has been loaded, "
                "so the board you planned against is gone and this action was not executed. "
                "The next turn begins from the new state."
            ),
            "next_level": (
                "next_level() opened the next level in this snippet. Its board comes right after this tool "
                "result, so this action was not executed. Act on it in your next Python call."
            ),
            "run_complete": (
                "the previous action reported run_complete. The run is finished and this "
                "action was not executed."
            ),
        }

        def action(actions, show_delta=False, reason=None):
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            normalized_actions = _normalize_actions(actions)
            _send({
                "type": "action",
                "actions": normalized_actions,
                # the solver decides precedence: a known death or a known no-op
                # is a sharper message than "your state may be stale", and it
                # carries a concrete override
                "stale_after": _stale_state[0] if _stale_state else None,
                "show_delta": bool(show_delta),
                "reason": next((str(one).strip() for one in [reason, *[item.get("reason") for item in (
                    actions if isinstance(actions, (list, tuple)) else [actions]) if isinstance(item, dict)]]
                    if one is not None and str(one).strip()), ""),
            })
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            # Cached animations describe the previous action. Invalidate the
            # existing views so aliases also refresh, even if a guard raises.
            for view in _animation_views:
                view.invalidate()
            _refresh_state(reply.get("state") or {})
            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )
            if action_result.get("stop_reason") == "known_death":
                raise KnownDeathActionError(
                    str(action_result.get("stop_detail") or "known fatal action was not executed.")
                )
            if action_result.get("stop_reason") == "repeated_action_in_state":
                raise RepeatedActionInStateError(
                    str(action_result.get("stop_detail") or "")
                    or "this action was already run from this board in this snippet"
                )
            if action_result.get("stop_reason") == "known_noop":
                # nothing executed and no action was spent; halt the snippet so
                # the model re-plans instead of continuing on a false premise
                raise KnownNoOpActionError(
                    str(action_result.get("stop_detail") or "known no-op action was not executed.")
                )
            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):
                    # `done` means the run was won; report it as run_complete
                    _terminal_state.append(
                        "run_complete" if _flag == "done" else _flag
                    )
                    break
            _echo = action_result.pop("action_echo", None)
            if _echo:
                # into the snippet's own stdout, so a loop can see what its
                # call did without waiting for the next turn's opener
                print(_echo)
            _delta = action_result.pop("delta", None)
            if _delta:
                print(_delta)
            gameplay_changed = action_result.get("gameplay_changed")
            if gameplay_changed is True:
                # a real change means the snippet's beliefs are current again
                del _stale_state[:]
            if gameplay_changed is False and action_result.get("executed"):
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                # Arm rather than raise. The action executed and changed
                # nothing, which is often exactly what the snippet went looking
                # for - it printed the board before, and wants to print it
                # after. Killing it here throws away the second half of the
                # measurement AND the batch stop's own trace, which is computed
                # and then discarded. So let the snippet finish observing, and
                # stop it only if it tries to ACT on the stale belief.
                if not _stale_state:
                    _stale_state.append(str(_name))
            _whole.append(action_result)
            return action_result

        def _animation_fetch(kind):
            _send({"type": "animation", "request": {"kind": kind}})
            reply = _recv()
            if reply.get("type") == "animation_error":
                raise RuntimeError(str(reply.get("error", "animation failed")))
            if reply.get("type") != "animation_result":
                raise RuntimeError("Invalid animation response from sandbox host.")
            payload = reply.get("animation")
            if kind not in ("frames", "reset_frames"):
                return payload or {}
            return [
                FrameView(
                    ascii=item.get("ascii", ""),
                    step=item.get("step", 0),
                    level=item.get("level", 0),
                    shape=item.get("shape", (0, 0)),
                    grid=item.get("grid", ()),
                )
                for item in (payload or [])
            ]

        runtime_globals["action"] = action
        if initial.get("next_level"):
            def next_level(description=None):
                _send({"type": "next_level", "text": "" if description is None else str(description)})
                reply = _recv()
                if reply.get("opened") and not _terminal_state:
                    _terminal_state.append("next_level")
                return _keep_whole(str(reply.get("text", "")))

            runtime_globals["next_level"] = next_level
        if initial.get("advisors_enabled"):
            def _advisor_call(kind, text):
                _send({"type": "advisor", "kind": kind, "text": str(text)})
                reply = _recv()
                print(str(reply.get("text", "")))

            class Advisors:
                @staticmethod
                def Reply(text):
                    _advisor_call("reply", text)

                @staticmethod
                def Ask(question):
                    _advisor_call("ask", question)

            runtime_globals["Advisors"] = Advisors
        if initial.get("animation_enabled"):
            # Globals rather than attributes on last_transition: after a death
            # the harness appends a RESET entry, so last_transition is the
            # reset and not the action that animated. The record is simply
            # "the last executed action that animated", and naming it that way
            # removes the mismatch. Empty when the last executed action
            # returned a single frame.
            class _LazyAnimation:
                def __init__(self, kind, empty):
                    self._kind = kind
                    self._empty = empty
                    self._value = None

                def _get(self):
                    if self._value is None:
                        self._value = _animation_fetch(self._kind) or self._empty
                    return self._value

                def invalidate(self):
                    self._value = None

                def __len__(self):
                    return len(self._get())

                def __iter__(self):
                    return iter(self._get())

                def __getitem__(self, key):
                    return self._get()[key]

                def __bool__(self):
                    return bool(self._get())

                def __contains__(self, key):
                    return key in self._get()

                def keys(self):
                    return self._get().keys()

                def get(self, key, default=None):
                    return self._get().get(key, default)

                def __repr__(self):
                    return repr(self._get())

            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})
            _animation_views.extend((runtime_globals["last_animation_frames"],
                                     runtime_globals["last_animation_timeline"]))
            if initial.get("reset_animation_enabled"):
                runtime_globals["reset_animation_frames"] = _LazyAnimation("reset_frames", [])
                runtime_globals["reset_animation_timeline"] = _LazyAnimation("reset_timeline", {})
                _animation_views.extend((runtime_globals["reset_animation_frames"],
                                         runtime_globals["reset_animation_timeline"]))

        _refresh_state(initial.get("state") or {})

        _persistence_enabled = bool(initial.get("persistence_enabled"))
        _retention_options = {
            "retain_imports": bool(initial.get("retain_imports")),
            "repair_hints": bool(initial.get("repair_hints")),
        }
        _kept_pool = {}
        _retention_rejected = []
        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        def _collect_retention_payload():
            if not _persistence_enabled:
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _retention_rejected + rejected,
                }
            except Exception:
                # Bookkeeping failures may drop helpers, but must never replace
                # the snippet's original error/output or replay its actions.
                return {"keepable_functions": [], "retention_rejected": [
                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}

        try:
            _code_text = str(initial.get("code", ""))
            _tree = ast.parse(_code_text, "<python_tool>")
            _last_expr = None
            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                _last_expr = ast.Expression(_tree.body[-1].value)
                ast.copy_location(_last_expr, _tree.body[-1])
                _tree.body = _tree.body[:-1]
            with contextlib.redirect_stdout(stdout):
                if _tree.body:
                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
                if _last_expr is not None:
                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
                    if _value is not None:
                        (print if any(_value is one for one in _whole)
                         else runtime_globals["__builtins__"]["print"])(repr(_value))
                        if runtime_globals.get("result") is None:
                            runtime_globals["result"] = _value
            _send(
                {
                    "type": "final",
                    **_collect_retention_payload(),
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "lookats": _lookats,
                }
            )
        except (
            KnownNoOpActionError,
            RepeatedActionInStateError,
            KnownDeathActionError,
            TerminalStateActionError,
            Exception,
        ) as exc:
            # The three refusals derive from BaseException so a model's own
            # `except Exception:` cannot swallow them - but THIS handler is the
            # same construct, so without naming them explicitly they escaped it
            # too, killed the child process, and the host reported a bare
            # "Sandbox process exited unexpectedly". The refusal messages - the
            # action name, that nothing was spent, how to override - were built
            # and then never delivered. They stop the snippet either way; the
            # point is that the model is told why.
            _send(
                {
                    "type": "error",
                    **_collect_retention_payload(),
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "lookats": _lookats,
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__RETAINED_FUNCTIONS_SOURCE__\n", inspect.getsource(_retained_functions)).replace(
    "__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)
).replace(
    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_frame_diff_module)
)


log = logging.getLogger(__name__)


def _sanitize_host_error_text(text: str) -> str:
    """What the MODEL is told when the sandbox process dies.

    Deliberately fixed: the child's traceback carries host paths and internals
    that have no business in the model's context. But the text was previously
    read and thrown away entirely, so a crash left no trace anywhere - the one
    thing that knows what went wrong said nothing. It now goes to the harness
    log instead, where it is useful and nobody is prompted by it.
    """
    detail = str(text or "").strip()
    if detail:
        log.warning(
            "sandbox process died; stderr follows:\n%s",
            detail[-4000:],
        )
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
        "ARC3_SANDBOX_MEMORY_GIB": os.environ.get("ARC3_SANDBOX_MEMORY_GIB", "16"),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _knob_win_next(description=None) -> tuple[bool, str]:
    try:
        from inference.agent import knob_arms
    except ImportError:
        return False, "next_level() is not available in this run."
    return knob_arms.win_next(description)


def _knob_reset_refused(actions: list[Any], reason: Any) -> str | None:
    try:
        from inference.agent import knob_arms
    except ImportError:
        return None
    return knob_arms.reset_refused(actions, reason) if knob_arms.on() else None


def _knob_reset_kept(actions: list[Any], reason: Any, result: Any) -> None:
    try:
        from inference.agent import knob_arms
    except ImportError:
        return
    if knob_arms.on():
        knob_arms.reset_kept(actions, reason, result)


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[..., dict[str, Any]],
    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    kept_functions: list[str] | None = None,
    retain_imports: bool = False,
    repair_hints: bool = False,
    advisor_handler: Callable[[str, str], str] | None = None,
    objects_handler: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if callable(initial_state):
        initial_state = initial_state()
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "kept_functions": list(kept_functions or []),
                "persistence_enabled": kept_functions is not None,
                "retain_imports": retain_imports,
                "repair_hints": repair_hints,
                "animation_enabled": animation_handler is not None,
                "reset_animation_enabled": animation_handler is not None and os.environ.get(
                    "ARC3_RESET_ANIMATION", ""
                ).strip().lower() in ("1", "true", "yes", "on"),
                "advisors_enabled": advisor_handler is not None,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "frame_objects": os.environ.get("ARC3_FRAME_OBJECTS", "").strip() == "1",
                "no_board_print": os.environ.get("ARC3_NO_BOARD_PRINT", "").strip() == "1",
                "lookat": os.environ.get("ARC3_LOOKAT", "").strip() == "1",
                "look": os.environ.get("ARC3_LOOK", "").strip() == "1",
                "look_max": 8 if os.environ.get("ARC3_TOOL_CAP_LOOK", "").strip() == "1" else 4,
                "next_level": os.environ.get("ARC3_WIN_PAUSE", "").strip() == "1",
                "colour_names": ['white', 'light gray', 'gray', 'dark gray', 'charcoal', 'black', 'magenta', 'pink', 'red', 'blue', 'sky blue', 'yellow', 'orange', 'dark red', 'light green', 'purple'],
            },
        )
        initial_state = None

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "Animation frames are not available in this session."},
                    )
                    continue
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "animation failed in sandbox host."},
                    )
                    continue
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue
            if msg_type == "objects":
                try:
                    found = objects_handler(dict(message)) if objects_handler is not None else None
                except Exception:
                    found = None
                _send_json_line(process.stdin, {"type": "objects_result", "objects": found})
                continue
            if msg_type == "advisor":
                try:
                    answer = (advisor_handler(str(message.get("kind") or ""), str(message.get("text") or ""))
                              if advisor_handler is not None else "Advisors are not available in this run.")
                except Exception:
                    answer = "Advisors: the message could not be delivered."
                _send_json_line(process.stdin, {"type": "advisor_result", "text": str(answer)})
                continue
            if msg_type == "next_level":
                opened, text = _knob_win_next(message.get("text"))
                _send_json_line(process.stdin, {"type": "next_level_result", "opened": opened, "text": text})
                continue
            if msg_type == "action":
                refused = _knob_reset_refused(list(message.get("actions") or []), message.get("reason"))
                if refused:
                    _send_json_line(process.stdin, {"type": "action_error", "error": refused})
                    continue
                try:
                    # the message carries more than the action list: stale_after
                    # tells the solver a previous action in this snippet changed
                    # nothing, which it weighs after its own guards
                    action_result_payload = action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                        **({"show_delta": True} if message.get("show_delta") else {}),
                    )
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                _knob_reset_kept(list(message.get("actions") or []), message.get("reason"), raw_action_result)
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                action_result_payload = None
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "lookats": list(message.get("lookats") or []),
                    "keepable_functions": list(message.get("keepable_functions") or []),
                    **({"retention_rejected": message.get("retention_rejected", [])}
                       if kept_functions is not None else {}),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
