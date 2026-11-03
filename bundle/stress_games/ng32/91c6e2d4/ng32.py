import copy

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
DICE = {
    0: (),
    1: (4,),
    2: (0, 8),
    3: (0, 4, 8),
    4: (0, 2, 6, 8),
    5: (0, 2, 4, 6, 8),
    6: (0, 2, 3, 5, 6, 8),
    7: (0, 2, 3, 4, 5, 6, 8),
    8: (0, 1, 2, 3, 5, 6, 7, 8),
    9: (0, 1, 2, 3, 4, 5, 6, 7, 8),
}
TOKENS = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}


def rect(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + w), min(64, y + h)
    if x1 > x0 and y1 > y0:
        px[y0:y1, x0:x1] = c


def dice(px, x, y, n, c, gap=2):
    for i in DICE[min(n, 9)]:
        rect(px, x + (i % 3) * gap, y + (i // 3) * gap, 1, 1, c)


def bar(px, left, y=1, c=14):
    rect(px, 2, y, 60, 1, BG)
    rect(px, 2, y, min(60, left), 1, c)


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


class Core(ARCBaseGame):
    STATE = ()

    def __init__(self, gid, count, actions, seed=0):
        self._level = None
        super().__init__(
            game_id=gid,
            levels=[canvas_level(i) for i in range(count)],
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=actions,
            win_score=count,
            seed=seed,
        )

    def on_set_level(self, level):
        self._level = level
        self.load(self._current_level_index)
        self.paint()

    def paint(self):
        px = np.full((64, 64), BG, dtype=np.int8)
        self.draw(px)
        self._level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        data = self._action.data or {}
        result = self.act(self._action.id.value, data)
        self.paint()
        if result == "win":
            self.next_level()
        elif result == "lose":
            self.lose()
        self.complete_action()

    def snap(self):
        return copy.deepcopy({k: getattr(self, k) for k in self.STATE})

    def restore(self, state):
        for k, v in copy.deepcopy(state).items():
            setattr(self, k, v)

FLOOR = 4
WALL = 3
PIP = 11
GATE = 13
EXIT = 14
HERO = 12
CELL = 6
OX = 2
OY = 8

A_SLOTS = [(3, 1), (8, 1), (5, 2), (2, 2), (7, 2), (6, 1)]
B_SLOTS = [(1, 4), (8, 5), (4, 5), (6, 4), (2, 5), (8, 4)]
GATE_1 = (4, 3)
GATE_2 = (5, 6)
EXIT_AT = (8, 7)


def make(cap, a, g1, b, g2, B):
    rows = [list("#" * 10) for _ in range(9)]
    for y in (1, 2, 4, 5):
        for x in range(1, 9):
            rows[y][x] = "."
    for x in range(1, 9):
        rows[7][x] = "."
    rows[1][1] = "S"
    rows[7][8] = "E"
    for (x, y), v in zip(A_SLOTS, a):
        rows[y][x] = str(v)
    for (x, y), v in zip(B_SLOTS, b):
        rows[y][x] = str(v)
    rows[GATE_1[1]][GATE_1[0]] = chr(96 + g1)
    rows[GATE_2[1]][GATE_2[0]] = chr(96 + g2)
    return dict(cap=cap, budget=B, grid=tuple("".join(r) for r in rows))


CONFIGS = [
    make(12, [1, 2, 1], 3, [2, 3], 5, 74),
    make(12, [2, 3, 2], 2, [1, 1], 6, 62),
    make(5, [3, 2, 2, 1], 4, [3, 1], 5, 58),
    make(6, [3, 3, 2, 2, 1], 5, [4, 2, 3], 6, 54),
    make(4, [2, 2, 1, 1, 2], 4, [2, 1, 3, 2], 4, 54),
]
LEVELS = CONFIGS
SOLUTIONS = []
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Ng32(Core):
    STATE = ("pos", "carry", "tiles", "left")

    def __init__(self, seed=0):
        self.cfg = CONFIGS[0]
        self.pos = (0, 0)
        self.carry = 0
        self.tiles = {}
        self.left = 0
        super().__init__("ng32-91c6e2d4", len(CONFIGS), [1, 2, 3, 4], seed)

    def load(self, index):
        self.cfg = CONFIGS[index]
        self.tiles = {}
        for y, line in enumerate(self.cfg["grid"]):
            for x, ch in enumerate(line):
                if ch == "S":
                    self.pos = (x, y)
                elif ch in "#E.":
                    if ch == "E":
                        self.tiles[(x, y)] = "E"
                elif ch.isdigit():
                    self.tiles[(x, y)] = int(ch)
                else:
                    self.tiles[(x, y)] = ch
        self.carry = 0
        self.left = self.cfg["budget"]

    def wall(self, cell):
        x, y = cell
        if not (0 <= x < 10 and 0 <= y < 9):
            return True
        return self.cfg["grid"][y][x] == "#"

    def moves(self):
        return ["U", "D", "L", "R"]

    def act(self, aid, data):
        d = MOVES.get(aid)
        if d is None:
            return None
        dest = (self.pos[0] + d[0], self.pos[1] + d[1])
        self.left -= 1
        if not self.wall(dest):
            t = self.tiles.get(dest)
            ok = True
            if isinstance(t, int):
                if self.carry + t <= self.cfg["cap"]:
                    self.carry += t
                    del self.tiles[dest]
            elif isinstance(t, str) and t in "abcdefghi":
                need = ord(t) - 96
                if self.carry >= need:
                    self.carry -= need
                    del self.tiles[dest]
                else:
                    ok = False
            if ok:
                self.pos = dest
                if t == "E":
                    return "win"
        if self.left <= 0:
            return "lose"
        return "ok"

    def draw(self, px):
        cfg = self.cfg
        bar(px, self.left * 60 // max(1, cfg["budget"]), 1, 14)
        rect(px, 2, 3, cfg["cap"] * 4 + 2, 3, 2)
        rect(px, 3, 4, cfg["cap"] * 4, 1, 4)
        rect(px, 3, 4, self.carry * 4, 1, PIP)
        for y in range(9):
            for x in range(10):
                gx, gy = OX + x * CELL, OY + y * CELL
                if cfg["grid"][y][x] == "#":
                    rect(px, gx, gy, CELL, CELL, WALL)
                    continue
                rect(px, gx, gy, CELL, CELL, FLOOR)
                t = self.tiles.get((x, y))
                if isinstance(t, int):
                    dice(px, gx, gy, t, PIP)
                elif t == "E":
                    rect(px, gx + 1, gy + 1, 4, 4, EXIT)
                elif isinstance(t, str):
                    rect(px, gx, gy, CELL, CELL, GATE)
                    dice(px, gx, gy, ord(t) - 96, 0)
        hx, hy = self.pos
        rect(px, OX + hx * CELL + 1, OY + hy * CELL + 1, 4, 4, HERO)

SOLUTIONS = [
    ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'D', 'L', 'L', 'L', 'L', 'D', 'D', 'L', 'L', 'L', 'D', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'D', 'D', 'R', 'R', 'R'],
    ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'D', 'L', 'L', 'L', 'L', 'D', 'D', 'D', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'D', 'D', 'R', 'R', 'R'],
    ['D', 'R', 'U', 'R', 'D', 'R', 'D', 'U', 'R', 'L', 'D', 'D', 'L', 'L', 'L', 'D', 'R', 'R', 'R', 'R', 'D', 'D', 'R', 'R', 'R'],
    ['D', 'R', 'U', 'R', 'D', 'R', 'D', 'U', 'R', 'R', 'R', 'L', 'L', 'L', 'D', 'D', 'D', 'R', 'D', 'D', 'R', 'R', 'R'],
    ['D', 'R', 'U', 'R', 'D', 'R', 'R', 'L', 'D', 'D', 'D', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'D', 'D', 'R', 'R', 'R'],
]
