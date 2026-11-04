import random

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BG, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


def blank():
    return np.full((64, 64), BG, dtype=np.int8)


def put(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    px[y0:max(y0, min(64, y + h)), x0:max(x0, min(64, x + w))] = c


def glyph(px, x, y, pat, scale, c):
    for gy in range(3):
        for gx in range(3):
            if pat[gy * 3 + gx]:
                put(px, x + gx * scale, y + gy * scale, scale, scale, c)


def make_camera():
    return Camera(width=64, height=64, background=BG, letter_box=BG)
from collections import deque

CELL = 6
OX, OY = 2, 2
GRID = 10
ROWN = 9
SPECS = [
    dict(n=8, kind="single", goal=(1, 0, 1, 1, 1, 1, 0, 1), start=(0,) * 8,
         walls=[(x, 3) for x in range(0, 8)] + [(x, 6) for x in range(2, 10)]),
    dict(n=8, kind="single", goal=(1, 1, 0, 1, 0, 0, 1, 0), start=(0, 1, 1, 0, 0, 1, 0, 1),
         walls=[(x, 3) for x in range(0, 8)] + [(x, 6) for x in range(2, 10)]),
    dict(n=9, kind="next", goal=(1, 0, 0, 1, 1, 0, 1, 0, 1), start=(0, 1, 0, 0, 1, 1, 0, 0, 0),
         walls=[(x, 3) for x in range(1, 9)] + [(x, 6) for x in range(0, 8)]),
    dict(n=10, kind="mirror", goal=(1, 1, 0, 1, 0, 1, 1, 0, 1, 1), start=(0, 1, 1, 0, 0, 1, 0, 1, 1, 0),
         walls=[(x, 3) for x in range(0, 8)] + [(x, 6) for x in range(2, 10)] + [(5, 7), (5, 8)]),
    dict(n=10, kind="triple", goal=(0, 1, 1, 0, 1, 0, 0, 1, 1, 1), start=(1, 0, 0, 1, 0, 1, 1, 0, 0, 0),
         walls=[(x, 3) for x in range(1, 9)] + [(x, 6) for x in range(0, 8)] + [(4, 8), (5, 8)]),
]
PEEK = (0, 0)
NAMES = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def flips(kind, i, n):
    if kind == "single":
        return {i}
    if kind == "next":
        return {i, i + 1} & set(range(n))
    if kind == "mirror":
        return {i, n - 1 - i}
    cells = {i - 1, i, i + 1}
    return {c for c in cells if 0 <= c < n}


def press_set(spec):
    n = spec["n"]
    need = tuple(a ^ b for a, b in zip(spec["goal"], spec["start"]))
    best = None
    for mask in range(1 << n):
        acc = [0] * n
        for i in range(n):
            if mask >> i & 1:
                for c in flips(spec["kind"], i, n):
                    acc[c] ^= 1
        if tuple(acc) == need and (best is None or bin(mask).count("1") < bin(best).count("1")):
            best = mask
    return [i for i in range(n) if best >> i & 1]


def route(spec, a, b):
    walls = set(spec["walls"])
    prev = {a: None}
    queue = deque([a])
    while queue:
        cur = queue.popleft()
        if cur == b:
            break
        for name, d in NAMES.items():
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if nxt in prev or nxt in walls or not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            prev[nxt] = (cur, name)
            queue.append(nxt)
    out = []
    while prev[b] is not None:
        b, name = prev[b]
        out.append(name)
    return out[::-1]


def solution(spec):
    pos = PEEK
    out = []
    for i in press_set(spec):
        target = (i, ROWN)
        out += route(spec, pos, target) + ["S"]
        pos = target
    return out


SOLUTIONS = [solution(s) for s in SPECS]
LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng49(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._bits = []
        super().__init__(
            game_id="ng49-6e0d5a93",
            levels=LEVELS,
            camera=make_camera(),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level):
        self._spec = SPECS[self._current_level_index]
        self._walls = set(self._spec["walls"])
        self._bits = list(self._spec["start"])
        self._pos = PEEK
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        spec = self._spec
        px = blank()
        for (cx, cy) in self._walls:
            put(px, OX + cx * CELL, OY + cy * CELL, CELL, CELL, 3)
        put(px, OX, OY, CELL, CELL, 9)
        for i in range(spec["n"]):
            put(px, OX + i * CELL, OY + ROWN * CELL, CELL - 1, CELL - 1, 11 if self._bits[i] else 4)
        if self._pos == PEEK:
            for i, b in enumerate(spec["goal"]):
                put(px, OX + 9 + i * 5, OY + 1, 4, 4, 11 if b else 4)
        cx, cy = self._pos
        put(px, OX + cx * CELL + 1, OY + cy * CELL + 1, 3, 3, 8)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        spec = self._spec
        delta = MOVES.get(self._action.id)
        if delta is not None:
            nxt = (self._pos[0] + delta[0], self._pos[1] + delta[1])
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self._walls:
                self._pos = nxt
            self._draw()
        elif self._action.id == GameAction.ACTION5:
            cx, cy = self._pos
            if cy == ROWN and cx < spec["n"]:
                for c in flips(spec["kind"], cx, spec["n"]):
                    self._bits[c] ^= 1
                self._draw()
                if tuple(self._bits) == tuple(spec["goal"]):
                    self.next_level()
        self.complete_action()
