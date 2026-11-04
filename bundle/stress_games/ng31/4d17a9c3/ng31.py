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

BLUE = 9
RED = 8
YELLOW = 11
ROW_Y = 5
ROW_PITCH = 11
BX = (47, 56)


def row(cap, start, target, a, b):
    return dict(cap=cap, start=start, target=target, buttons=[a, b])


CONFIGS = [
    dict(sized=True, carry=False, budget=34, rows=[
        row(9, 0, 7, (1, 1), (3, 1)),
        row(8, 0, 5, (2, 1), (3, 1)),
        row(9, 0, 6, (1, 1), (2, 1)),
        row(7, 0, 6, (2, 1), (4, 1)),
        row(8, 0, 7, (1, 1), (4, 1)),
    ]),
    dict(sized=False, carry=False, budget=24, rows=[
        row(9, 2, 8, (3, 1), (1, 1)),
        row(8, 1, 7, (4, 1), (2, 1)),
        row(9, 4, 8, (2, 1), (1, 1)),
        row(7, 0, 5, (3, 1), (2, 1)),
        row(8, 3, 7, (4, 1), (1, 1)),
    ]),
    dict(sized=False, carry=False, budget=24, rows=[
        row(9, 6, 2, (2, 1), (3, 1)),
        row(8, 5, 1, (2, 1), (3, 1)),
        row(9, 7, 3, (2, 1), (3, 1)),
        row(7, 4, 0, (2, 1), (3, 1)),
        row(8, 6, 0, (3, 1), (1, 1)),
    ]),
    dict(sized=False, carry=False, budget=30, rows=[
        row(9, 8, 2, (4, 1), (3, -1)),
        row(8, 1, 6, (3, 1), (2, -1)),
        row(9, 3, 8, (2, 1), (4, -1)),
        row(7, 5, 1, (3, 1), (4, -1)),
        row(8, 2, 5, (2, 1), (3, -1)),
    ]),
    dict(sized=False, carry=True, budget=26, rows=[
        row(9, 0, 4, (3, 1), (5, 1)),
        row(8, 0, 6, (3, 1), (2, 1)),
        row(9, 0, 2, (2, 1), (4, 1)),
        row(7, 0, 3, (3, 1), (2, 1)),
        row(8, 0, 1, (3, 1), (2, 1)),
    ]),
]
LEVELS = CONFIGS
SOLUTIONS = []


def button_box(cfg, r, k):
    pips = cfg["rows"][r]["buttons"][k][0]
    h = 4 + pips if cfg["sized"] else 8
    y = ROW_Y + r * ROW_PITCH
    return (BX[k], y + (8 - h) // 2 if h <= 8 else y - 1, 8, h)


class Ng31(Core):
    STATE = ("regs", "left")

    def __init__(self, seed=0):
        self.cfg = CONFIGS[0]
        self.regs = []
        self.left = 0
        super().__init__("ng31-4d17a9c3", len(CONFIGS), [6], seed)

    def load(self, index):
        self.cfg = CONFIGS[index]
        self.regs = [r["start"] for r in self.cfg["rows"]]
        self.left = self.cfg["budget"]

    def moves(self):
        out = []
        for r in range(len(self.cfg["rows"])):
            for k in range(2):
                x, y, w, h = button_box(self.cfg, r, k)
                out.append((x + 1, y + 1))
        return out

    def press(self, r, k):
        cfg = self.cfg
        pips, sign = cfg["rows"][r]["buttons"][k]
        cap = cfg["rows"][r]["cap"]
        raw = self.regs[r] + sign * pips
        self.regs[r] = raw % cap
        if cfg["carry"] and sign > 0 and r + 1 < len(self.regs):
            nxt = cfg["rows"][r + 1]["cap"]
            self.regs[r + 1] = (self.regs[r + 1] + raw // cap) % nxt

    def act(self, aid, data):
        if aid != 6:
            return None
        px, py = data.get("x", -1), data.get("y", -1)
        cfg = self.cfg
        for r in range(len(cfg["rows"])):
            for k in range(2):
                x, y, w, h = button_box(cfg, r, k)
                if x <= px < x + w and y <= py < y + h:
                    self.press(r, k)
                    self.left -= 1
                    if all(self.regs[i] == cfg["rows"][i]["target"] for i in range(len(self.regs))):
                        return "win"
                    if self.left <= 0:
                        return "lose"
                    return "ok"
        return None

    def draw(self, px):
        cfg = self.cfg
        bar(px, self.left * 60 // max(1, cfg["budget"]), 1, 14)
        for i, rw in enumerate(cfg["rows"]):
            y = ROW_Y + i * ROW_PITCH
            for c in range(rw["cap"]):
                rect(px, 2 + c * 5, y, 4, 6, BLUE if c < self.regs[i] else 3)
            tx = 2 + rw["target"] * 5
            rect(px, tx, y + 7, 4, 2, YELLOW)
            if rw["target"] == 0:
                rect(px, 0, y + 7, 2, 2, YELLOW)
            for k in range(2):
                x, by, w, h = button_box(cfg, i, k)
                pips, sign = rw["buttons"][k]
                rect(px, x, by, w, h, RED if sign < 0 else BLUE)
                dice(px, x + 1, by + (h - 5) // 2, pips, 0)

SOLUTIONS = [
    [(48, 7), (57, 6), (57, 6), (48, 18), (57, 17), (57, 29), (57, 29), (57, 29), (48, 40), (57, 39), (48, 51), (48, 51), (48, 51), (57, 50)],
    [(48, 6), (48, 6), (48, 17), (57, 17), (48, 28), (48, 28), (48, 39), (57, 39), (48, 50)],
    [(48, 6), (57, 6), (48, 17), (48, 17), (48, 28), (57, 28), (57, 39), (57, 50), (57, 50)],
    [(57, 6), (57, 6), (48, 17), (57, 17), (57, 17), (57, 17), (57, 28), (48, 39), (48, 50), (48, 50), (48, 50), (57, 50)],
    [(48, 6), (57, 6), (57, 6), (48, 17), (57, 17), (48, 28), (48, 39), (48, 39), (57, 39), (57, 39)],
]
