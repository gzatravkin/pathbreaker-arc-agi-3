import math

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
FUSE_C = 12
SLOW_C = 15
VALVE_C = 0
ASH = 4
WET = 10
FIRE = 11
EMBER = 8
TANK_C = 9
TANK_CORE = 10
SPRAY = 10
CHAR = 3
PIP = 11
PIP_SPENT = 3

EMPTY, FUSE, SLOW, VALVE = 0, 1, 2, 3
INTACT, BURNING, BURNT, SOAKED = 0, 1, 2, 3
SPEED = {FUSE: 1, SLOW: 3, VALVE: 1}
CRATE_BURN = 4
DIRS = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def trace(points):
    out = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = (x1 > x0) - (x1 < x0)
        dy = (y1 > y0) - (y1 < y0)
        x, y = x0, y0
        while (x, y) != (x1, y1):
            out.append((x, y))
            x += dx
            y += dy
    out.append(points[-1])
    return out


class Board:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = np.zeros((64, 64), dtype=np.int8)
        self.status = np.zeros((64, 64), dtype=np.int8)
        self.timer = {}
        self.vdir = {}
        self.decor = []
        for line in cfg.get("fuses", ()):
            for x, y in trace(line):
                self.kind[y, x] = FUSE
        for line in cfg.get("slow", ()):
            for x, y in trace(line):
                self.kind[y, x] = SLOW
        for x, y, d in cfg.get("valves", ()):
            dx, dy = DIRS[d]
            self.kind[y, x] = VALVE
            self.vdir[(x, y)] = (dx, dy)
            for k in (1, 2):
                if dx:
                    self.decor += [(x - k * dx, y - k), (x - k * dx, y + k)]
                else:
                    self.decor += [(x - k, y - k * dy), (x + k, y - k * dy)]
        self.crate_at = {}
        self.crates = []
        for i, (x, y, colour) in enumerate(cfg.get("crates", ())):
            self.crates.append(dict(x=x, y=y, colour=colour, state=INTACT, t=0))
            for j in range(3):
                for k in range(3):
                    self.crate_at[(x + k, y + j)] = i
        self.tank_at = {}
        self.tanks = []
        for i, (x, y, radius) in enumerate(cfg.get("tanks", ())):
            self.tanks.append(dict(x=x, y=y, radius=radius, state=INTACT, r=0))
            for j in range(3):
                for k in range(3):
                    self.tank_at[(x + k, y + j)] = i
        self.spray = set()
        self.sparks = cfg["sparks"]

    def flammable(self, x, y):
        return 0 <= x < 64 and 0 <= y < 64 and self.kind[y, x] != EMPTY and self.status[y, x] == INTACT

    def click(self, x, y):
        if not self.flammable(x, y):
            return False
        self.sparks -= 1
        self.status[y, x] = BURNING
        self.timer[(x, y)] = SPEED[int(self.kind[y, x])]
        return True

    def active(self):
        return bool(self.timer) or any(c["state"] == BURNING for c in self.crates) or any(
            t["state"] == BURNING for t in self.tanks)

    def _ignite(self, n, frm):
        x, y = n
        if self.flammable(x, y):
            k = int(self.kind[y, x])
            if k == VALVE and (x - frm[0], y - frm[1]) != self.vdir[(x, y)]:
                return
            self.status[y, x] = BURNING
            self.timer[(x, y)] = SPEED[k]
            return
        ci = self.crate_at.get(n)
        if ci is not None and self.crates[ci]["state"] == INTACT:
            self.crates[ci]["state"] = BURNING
            self.crates[ci]["t"] = CRATE_BURN
            return
        ti = self.tank_at.get(n)
        if ti is not None and self.tanks[ti]["state"] == INTACT:
            self.tanks[ti]["state"] = BURNING
            self.tanks[ti]["r"] = 0

    def _soak(self, tank):
        cx, cy = tank["x"] + 1, tank["y"] + 1
        r = tank["r"]
        self.spray = set()
        for y in range(max(0, cy - r), min(64, cy + r + 1)):
            for x in range(max(0, cx - r), min(64, cx + r + 1)):
                d = math.ceil(math.hypot(x - cx, y - cy))
                if d > r:
                    continue
                if d == r:
                    self.spray.add((x, y))
                if self.kind[y, x] != EMPTY and self.status[y, x] in (INTACT, BURNING):
                    self.status[y, x] = SOAKED
                    self.timer.pop((x, y), None)
                ci = self.crate_at.get((x, y))
                if ci is not None and self.crates[ci]["state"] == INTACT:
                    self.crates[ci]["state"] = SOAKED

    def tick(self):
        self.spray = set()
        for tank in self.tanks:
            if tank["state"] == BURNING:
                tank["r"] += 1
                self._soak(tank)
                if tank["r"] >= tank["radius"]:
                    tank["state"] = BURNT
        done = []
        for cell in list(self.timer):
            self.timer[cell] -= 1
            if self.timer[cell] <= 0:
                done.append(cell)
        for cell in done:
            x, y = cell
            del self.timer[cell]
            self.status[y, x] = BURNT
            if self.kind[y, x] == VALVE:
                dx, dy = self.vdir[cell]
                self._ignite((x + dx, y + dy), cell)
                continue
            for dx, dy in NEIGHBOURS:
                self._ignite((x + dx, y + dy), cell)
        for i, crate in enumerate(self.crates):
            if crate["state"] == BURNING:
                crate["t"] -= 1
                if crate["t"] <= 0:
                    crate["state"] = BURNT
                    for j in range(-1, 4):
                        for k in range(-1, 4):
                            n = (crate["x"] + k, crate["y"] + j)
                            if n in self.crate_at and self.crate_at[n] == i:
                                continue
                            if (j in (-1, 3)) and (k in (-1, 3)):
                                continue
                            self._ignite(n, (crate["x"] + 1, crate["y"] + 1))
        return self.active()

    def verdict(self):
        legend = set(self.cfg["legend"])
        burnt = [c for c in self.crates if c["state"] == BURNT]
        if any(c["colour"] not in legend for c in burnt):
            return "lose"
        if all(c["state"] == BURNT for c in self.crates if c["colour"] in legend):
            return "win"
        if self.sparks <= 0:
            return "lose"
        return "play"

    def render(self, frame_no=0):
        px = np.full((64, 64), BG, dtype=np.int8)
        ys, xs = np.nonzero(self.kind)
        for y, x in zip(ys, xs):
            st = self.status[y, x]
            k = self.kind[y, x]
            if st == INTACT:
                px[y, x] = FUSE_C if k == FUSE else SLOW_C if k == SLOW else VALVE_C
            elif st == BURNING:
                px[y, x] = FIRE
            elif st == BURNT:
                px[y, x] = ASH
            else:
                px[y, x] = WET
        for x, y in self.decor:
            if self.kind[y, x] == EMPTY:
                px[y, x] = VALVE_C
        for x, y in self.spray:
            if px[y, x] == BG:
                px[y, x] = SPRAY
        for crate in self.crates:
            x, y = crate["x"], crate["y"]
            if crate["state"] == INTACT:
                px[y:y + 3, x:x + 3] = crate["colour"]
            elif crate["state"] == BURNING:
                px[y:y + 3, x:x + 3] = FIRE if crate["t"] % 2 else EMBER
            elif crate["state"] == BURNT:
                px[y:y + 3, x:x + 3] = ASH
                px[y + 1, x + 1] = CHAR
            else:
                px[y:y + 3, x:x + 3] = WET
                px[y + 1, x + 1] = crate["colour"]
        for tank in self.tanks:
            x, y = tank["x"], tank["y"]
            if tank["state"] == INTACT:
                px[y:y + 3, x:x + 3] = TANK_C
                px[y + 1, x + 1] = TANK_CORE
            else:
                px[y:y + 3, x:x + 3] = ASH
                px[y + 1, x + 1] = TANK_C
        for i in range(self.cfg["sparks"]):
            px[1:3, 1 + 3 * i:3 + 3 * i] = PIP if i < self.sparks else PIP_SPENT
        for i, colour in enumerate(self.cfg["legend"]):
            x = 60 - 4 * i
            px[1:4, x:x + 3] = colour
        return px


LEVEL_DATA = [{'fuses': [[(6, 26), (58, 26), (58, 42), (10, 42)],
            [(16, 27), (16, 28), (30, 28), (30, 30), (20, 30), (20, 33), (18, 33)]],
  'slow': [],
  'valves': [],
  'crates': [(25, 43, 13), (7, 41, 14)],
  'tanks': [(15, 32, 16)],
  'legend': (13,),
  'sparks': 3},
 {'fuses': [[(59, 18), (7, 18), (7, 36), (23, 36)],
            [(27, 36), (39, 36), (39, 30)],
            [(39, 37), (39, 46), (51, 46)],
            [(53, 19), (53, 39), (49, 39)]],
  'slow': [],
  'valves': [],
  'crates': [(24, 35, 13), (38, 27, 13), (52, 45, 14)],
  'tanks': [(46, 38, 13)],
  'legend': (13,),
  'sparks': 3},
 {'fuses': [[(50, 8), (50, 56), (22, 56), (22, 14)]],
  'slow': [[(49, 10), (30, 10), (30, 14)]],
  'valves': [],
  'crates': [(21, 11, 14), (23, 27, 13)],
  'tanks': [(29, 15, 14)],
  'legend': (13,),
  'sparks': 3},
 {'fuses': [[(6, 14), (35, 14)],
            [(37, 14), (58, 14), (58, 46), (14, 46)],
            [(44, 45), (44, 15)],
            [(10, 15), (10, 22)]],
  'slow': [[(10, 23), (10, 40), (14, 40)]],
  'valves': [(36, 14, 'R'), (44, 30, 'U')],
  'crates': [(23, 47, 13), (11, 45, 14)],
  'tanks': [(15, 39, 11)],
  'legend': (13,),
  'sparks': 3},
 {'fuses': [[(57, 14), (28, 14)],
            [(26, 14), (5, 14), (5, 46), (21, 46)],
            [(25, 46), (49, 46)],
            [(19, 45), (19, 15)],
            [(53, 15), (53, 22)]],
  'slow': [[(53, 23), (53, 40), (49, 40)]],
  'valves': [(27, 14, 'L'), (19, 30, 'U')],
  'crates': [(22, 45, 13), (38, 47, 13), (50, 45, 14)],
  'tanks': [(46, 39, 11)],
  'legend': (13,),
  'sparks': 3},
 {'fuses': [[(6, 8), (58, 8), (58, 26), (12, 26)],
            [(16, 9), (16, 11)],
            [(57, 36), (5, 36), (5, 54), (51, 54)],
            [(53, 37), (53, 43)]],
  'slow': [[(16, 12), (16, 21), (18, 21)], [(53, 44), (53, 49), (51, 49)]],
  'valves': [],
  'crates': [(27, 27, 13), (9, 25, 14), (40, 55, 13), (52, 53, 14)],
  'tanks': [(19, 20, 10), (48, 48, 10)],
  'legend': (13,),
  'sparks': 5}]

SOLUTIONS = [
    [(42, 26)],
    [(15, 18)],
    [(50, 23)],
    [(28, 14)],
    [(34, 14)],
    [(40, 8), (18, 36)],
]


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


LEVELS = [canvas_level(i) for i in range(len(LEVEL_DATA))]


class Ng10(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._board = Board(LEVEL_DATA[0])
        self._burning = False
        super().__init__(
            game_id="ng10-f1e5a2b9",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._board = Board(LEVEL_DATA[self._current_level_index])
        self._burning = False
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        level.get_sprites_by_name("canvas")[0].pixels = self._board.render()

    def _settle(self):
        self._burning = False
        verdict = self._board.verdict()
        self._draw()
        if verdict == "win":
            self.next_level()
        elif verdict == "lose":
            self.lose()
        self.complete_action()

    def step(self) -> None:
        if not self._burning:
            if self._action.id != GameAction.ACTION6:
                self.complete_action()
                return
            data = self._action.data or {}
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            if not self._board.click(x, y):
                self._draw()
                self.complete_action()
                return
            self._burning = True
            self._draw()
            return
        if not self._board.tick():
            self._settle()
            return
        self._draw()
