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

PT = (1, 1, 1, 0, 1, 0, 0, 1, 0)
PX = (1, 0, 1, 0, 1, 0, 1, 0, 1)
PL = (1, 0, 0, 1, 0, 0, 1, 1, 1)
PH = (1, 0, 1, 1, 1, 1, 1, 0, 1)
PC = (1, 1, 1, 1, 0, 0, 1, 1, 1)
PO = (1, 1, 1, 1, 0, 1, 1, 1, 1)
PZ = (1, 1, 0, 0, 1, 0, 0, 1, 1)
PV = (1, 0, 1, 1, 0, 1, 0, 1, 0)
PP = (1, 1, 1, 1, 1, 1, 1, 0, 0)
PU = (1, 0, 1, 1, 0, 1, 1, 1, 1)
PS = (0, 1, 1, 0, 1, 0, 1, 1, 0)
PD = (0, 1, 0, 1, 1, 1, 0, 1, 0)

SW, SH = 10, 9
CELL = 6
OX, OY = 2, 8

def N(p):
    return tuple(1 - v for v in p)


SPECS = [
    dict(sw=2, sh=1, gaps=[], walls=[(3, 3), (3, 4), (3, 5), (6, 6), (6, 7), (12, 2), (13, 2), (14, 2)],
         start=(1, 4), goal=(17, 4),
         keys={(2, 1): (N(PT), 8), (7, 7): (PO, 9)}, doors={(9, 4): (PT, 8)}),
    dict(sw=3, sh=1, gaps=[(9, 4)], walls=[(3, 6), (4, 6), (5, 6), (6, 2), (7, 2), (13, 1), (13, 2), (13, 3), (23, 5), (24, 5), (25, 5)],
         start=(1, 4), goal=(27, 4),
         keys={(2, 1): (N(PT), 9), (7, 7): (PX, 8), (2, 7): (PT, 14)}, doors={(19, 4): (PT, 8)}),
    dict(sw=4, sh=1, gaps=[(9, 4)], walls=[(3, 6), (4, 6), (5, 6), (6, 2), (7, 2), (13, 1), (13, 2), (13, 3), (23, 5), (24, 5), (25, 5), (15, 7), (16, 7), (35, 2), (35, 3)],
         start=(1, 4), goal=(37, 4),
         keys={(2, 1): (N(PH), 12), (7, 7): (PC, 11), (2, 7): (PH, 6),
               (12, 1): (N(PL), 6), (16, 5): (PL, 15), (11, 7): (PT, 14)},
         doors={(19, 4): (PH, 11), (29, 4): (PL, 14)}),
    dict(sw=3, sh=2, gaps=[(9, 4)], walls=[(3, 3), (3, 4), (4, 3), (6, 6), (6, 7), (14, 2), (15, 2), (14, 6), (13, 13), (14, 13), (4, 12), (5, 12), (22, 12), (23, 12), (24, 12), (6, 15), (7, 15)],
         start=(1, 6), goal=(27, 13),
         keys={(2, 1): (N(PP), 12), (7, 1): (PU, 11), (2, 7): (PP, 14),
               (2, 11): (N(PZ), 8), (7, 11): (PD, 15), (3, 15): (PZ, 9),
               (12, 1): (N(PO), 6), (15, 7): (PC, 9), (17, 5): (PO, 12)},
         doors={(4, 8): (PP, 14), (9, 13): (PZ, 12), (19, 13): (PO, 8)}),
    dict(sw=5, sh=1, gaps=[(9, 4)], walls=[(3, 6), (4, 6), (5, 6), (6, 2), (7, 2), (13, 1), (13, 2), (13, 3), (23, 5), (24, 5), (25, 5), (15, 7), (16, 7), (33, 2), (33, 3), (34, 3), (44, 6), (45, 6), (45, 5)],
         start=(1, 4), goal=(47, 4),
         keys={(2, 1): (N(PS), 12), (7, 7): (PS, 14), (2, 7): (PV, 6),
               (12, 1): (N(PD), 8), (16, 5): (PD, 9), (11, 7): (PU, 11),
               (22, 1): (N(PC), 9), (26, 7): (PC, 6), (24, 3): (PX, 15)},
         doors={(19, 4): (PS, 14), (29, 4): (PD, 15), (39, 4): (PC, 12)}),
]


def build(spec):
    w, h = spec["sw"] * SW, spec["sh"] * SH
    grid = [["." for _ in range(w)] for _ in range(h)]
    for k in range(1, spec["sw"]):
        for y in range(h):
            grid[y][SW * k - 1] = "#"
    for k in range(1, spec["sh"]):
        for x in range(w):
            grid[SH * k - 1][x] = "#"
    for x, y in spec["gaps"]:
        grid[y][x] = "."
    for x, y in spec["walls"]:
        grid[y][x] = "#"
    for x, y in spec["doors"]:
        grid[y][x] = "."
    return grid


WORLDS = [build(s) for s in SPECS]


def neighbours(spec, grid, state, delta):
    x, y, held, items, opened = state
    nx, ny = x + delta[0], y + delta[1]
    if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])) or grid[ny][nx] == "#":
        return state
    cell = (nx, ny)
    if cell in spec["doors"] and cell not in opened:
        if held is not None and held[0] == N(spec["doors"][cell][0]):
            return (nx, ny, None, items, opened | frozenset([cell]))
        return state
    if cell in items:
        pickup = items[cell]
        new = dict(items)
        if held is None:
            del new[cell]
        else:
            new[cell] = held
        return (nx, ny, pickup, new, opened)
    return (nx, ny, held, items, opened)


def key_of(state):
    x, y, held, items, opened = state
    return (x, y, held, tuple(sorted(items.items())), opened)


PLANS = [
    [(2, 1), (9, 4), (17, 4)],
    [(2, 1), (19, 4), (27, 4)],
    [(2, 1), (19, 4), (25, 4), (12, 1), (29, 4), (37, 4)],
    [(2, 1), (4, 8), (2, 11), (9, 13), (18, 13), (12, 1), (19, 13), (27, 13)],
    [(2, 1), (19, 4), (25, 4), (12, 1), (29, 4), (35, 4), (22, 1), (39, 4), (47, 4)],
]
NAMES = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def route(spec, grid, state, target):
    start = (state[0], state[1])
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == target:
            break
        for name, d in NAMES.items():
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if nxt in prev or not (0 <= nxt[1] < len(grid) and 0 <= nxt[0] < len(grid[0])):
                continue
            if grid[nxt[1]][nxt[0]] == "#":
                continue
            if nxt != target and (nxt in state[3] or (nxt in spec["doors"] and nxt not in state[4])):
                continue
            prev[nxt] = (cur, name)
            queue.append(nxt)
    out = []
    cur = target
    while prev[cur] is not None:
        cur, name = prev[cur]
        out.append(name)
    return out[::-1]


def solve(spec, grid, plan):
    state = (spec["start"][0], spec["start"][1], None, dict(spec["keys"]), frozenset())
    out = []
    for target in plan:
        for name in route(spec, grid, state, target):
            state = neighbours(spec, grid, state, NAMES[name])
            out.append(name)
    return out


SOLUTIONS = [solve(s, g, p) for s, g, p in zip(SPECS, WORLDS, PLANS)]
LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng48(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._st = None
        super().__init__(
            game_id="ng48-2f7c8e14",
            levels=LEVELS,
            camera=make_camera(),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level):
        self._idx = self._current_level_index
        self._spec = SPECS[self._idx]
        self._grid = WORLDS[self._idx]
        sx, sy = self._spec["start"]
        self._st = (sx, sy, None, dict(self._spec["keys"]), frozenset())
        self._draw(level)

    def _cell(self, px, cx, cy, kind, pat=None, colour=0):
        x, y = OX + cx * CELL, OY + cy * CELL
        if kind == "wall":
            put(px, x, y, CELL, CELL, 3)
        elif kind == "key":
            glyph(px, x, y, pat, 2, colour)
        elif kind == "door":
            put(px, x, y, CELL, CELL, 2)
            glyph(px, x, y, pat, 2, colour)
        elif kind == "goal":
            put(px, x, y, CELL, CELL, 14)
            put(px, x + 2, y + 2, 2, 2, BG)

    def _draw(self, level=None):
        level = level or self.current_level
        px = blank()
        x, y, held, items, opened = self._st
        sx, sy = x // SW, y // SH
        for cy in range(SH):
            for cx in range(SW):
                wx, wy = sx * SW + cx, sy * SH + cy
                if self._grid[wy][wx] == "#":
                    self._cell(px, cx, cy, "wall")
                cell = (wx, wy)
                if cell == self._spec["goal"]:
                    self._cell(px, cx, cy, "goal")
                if cell in self._spec["doors"] and cell not in opened:
                    pat, colour = self._spec["doors"][cell]
                    self._cell(px, cx, cy, "door", pat, colour)
                if cell in items:
                    pat, colour = items[cell]
                    self._cell(px, cx, cy, "key", pat, colour)
        put(px, 1, 0, 8, 7, 4)
        if held is not None:
            glyph(px, 2, 1, held[0], 2, held[1])
        put(px, OX + (x % SW) * CELL + 1, OY + (y % SH) * CELL + 1, 4, 4, 0)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        delta = MOVES.get(self._action.id)
        if delta is not None:
            self._st = neighbours(self._spec, self._grid, self._st, delta)
            self._draw()
            if (self._st[0], self._st[1]) == self._spec["goal"]:
                self.next_level()
        self.complete_action()
