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
TILE = 10
GEM = 11
HERO = 12
SEALED = 13
OPEN = 14
PATCH = 12
CELL = 7
OX = 4
OY = 7
SIZE = 8

CONFIGS = [
    dict(sdur=1, patches=0, budget=52, grid=('#1######', '#1S###11', '1#A1#AA#', '##1111E#', '1#111###', '1111#1#1', '111111##', '111#1###')),
    dict(sdur=1, patches=0, budget=60, grid=('##1#1##2', 'S1111#22', '11#121#2', '11#22221', '#2##2111', '1#2AB1#1', '222A1212', '#11#E22#')),
    dict(sdur=1, patches=2, budget=68, grid=('##111#21', 'A1ABA#1S', '12121112', '1E2##112', '111#2111', '##1#2#1#', '#1##11#1', '#2##2222')),
    dict(sdur=1, patches=3, budget=78, grid=('#####322', '12#3##13', '2##1A1CA', '13##12A1', '#3#S1122', '#132E12#', '#21#1113', '#322232A')),
    dict(sdur=1, patches=3, budget=86, grid=('12##133#', 'S#23##31', '113##3#2', '32#33#12', '23##211E', '111A1AAA', '1A131111', '3#212B1#')),
]
LEVELS = CONFIGS
SOLUTIONS = []
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Ng33(Core):
    STATE = ("pos", "dur", "gems", "patches", "left")

    def __init__(self, seed=0):
        self.cfg = None
        self.pos = (0, 0)
        self.dur = {}
        self.gems = set()
        self.patches = 0
        self.left = 0
        super().__init__("ng33-c2a85f07", max(1, len(CONFIGS)), [1, 2, 3, 4, 5], seed)

    def load(self, index):
        self.cfg = CONFIGS[index]
        self.dur = {}
        self.gems = set()
        for y, line in enumerate(self.cfg["grid"]):
            for x, ch in enumerate(line):
                if ch == "#":
                    continue
                if ch == "S":
                    self.pos = (x, y)
                    self.dur[(x, y)] = self.cfg["sdur"]
                elif ch == "E":
                    self.dur[(x, y)] = 1
                elif ch.isdigit():
                    self.dur[(x, y)] = int(ch)
                else:
                    self.dur[(x, y)] = ord(ch) - 64
                    self.gems.add((x, y))
        self.exit = next((x, y) for y, line in enumerate(self.cfg["grid"]) for x, ch in enumerate(line) if ch == "E")
        self.patches = self.cfg["patches"]
        self.left = self.cfg["budget"]

    def moves(self):
        return ["U", "D", "L", "R", "S"]

    def act(self, aid, data):
        if aid in MOVES:
            d = MOVES[aid]
            dest = (self.pos[0] + d[0], self.pos[1] + d[1])
            self.left -= 1
            if dest in self.dur and not (dest == self.exit and self.gems):
                self.dur[self.pos] -= 1
                if self.dur[self.pos] <= 0:
                    del self.dur[self.pos]
                self.pos = dest
                self.gems.discard(dest)
                if dest == self.exit:
                    return "win"
        elif aid == 5:
            self.left -= 1
            if self.patches > 0 and self.dur[self.pos] < 4:
                self.dur[self.pos] += 1
                self.patches -= 1
        else:
            return None
        if self.left <= 0:
            return "lose"
        return "ok"

    def draw(self, px):
        cfg = self.cfg
        bar(px, self.left * 60 // max(1, cfg["budget"]), 1, 14)
        for i in range(self.patches):
            rect(px, 2 + i * 5, 3, 3, 3, PATCH)
        for y in range(SIZE):
            for x in range(SIZE):
                gx, gy = OX + x * CELL, OY + y * CELL
                if cfg["grid"][y][x] == "#":
                    rect(px, gx, gy, CELL, CELL, WALL)
                    continue
                d = self.dur.get((x, y))
                if d is None:
                    continue
                rect(px, gx, gy, CELL, CELL, FLOOR)
                if (x, y) == self.exit:
                    rect(px, gx + 1, gy + 1, 5, 5, SEALED if self.gems else OPEN)
                    continue
                dice(px, gx + 1, gy + 1, d, TILE)
                if (x, y) in self.gems:
                    rect(px, gx, gy, CELL, 1, GEM)
                    rect(px, gx, gy + CELL - 1, CELL, 1, GEM)
                    rect(px, gx, gy, 1, CELL, GEM)
                    rect(px, gx + CELL - 1, gy, 1, CELL, GEM)
        hx, hy = self.pos
        gx, gy = OX + hx * CELL, OY + hy * CELL
        for ax, ay in ((0, 0), (CELL - 2, 0), (0, CELL - 2), (CELL - 2, CELL - 2)):
            rect(px, gx + ax, gy + ay, 2, 2, HERO)

SOLUTIONS = [
    ['D', 'R', 'D', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'R', 'U', 'R', 'R', 'U', 'U', 'R', 'U', 'R', 'U', 'R', 'D'],
    ['D', 'D', 'R', 'U', 'U', 'R', 'R', 'R', 'D', 'R', 'L', 'L', 'D', 'R', 'L', 'R', 'D', 'D', 'R', 'U', 'L', 'D', 'L', 'D', 'R', 'D'],
    ['D', 'D', 'L', 'L', 'D', 'R', 'R', 'U', 'U', 'L', 'L', 'L', 'U', 'S', 'U', 'D', 'L', 'L', 'S', 'U', 'R', 'D', 'L', 'L', 'L', 'D', 'D', 'D', 'R', 'U'],
    ['R', 'U', 'U', 'S', 'L', 'R', 'R', 'R', 'U', 'D', 'R', 'L', 'D', 'D', 'L', 'R', 'D', 'L', 'S', 'R', 'D', 'D', 'R', 'L', 'L', 'L', 'U', 'D', 'R', 'U', 'S', 'D', 'U', 'U', 'L'],
    ['D', 'D', 'R', 'U', 'D', 'L', 'D', 'U', 'D', 'D', 'D', 'R', 'R', 'U', 'R', 'R', 'S', 'U', 'R', 'S', 'L', 'D', 'D', 'D', 'R', 'S', 'L', 'R', 'R', 'L', 'U', 'U', 'U', 'R', 'D', 'D', 'R', 'U', 'U'],
]
