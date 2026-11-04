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

CELL = 6
OX = 2
OY = 8
FLOOR = 4
WALL = 3
GATE = 13
EXIT = 14
HERO = 12
CONFIGS = [
    dict(need=5, budget=16, grid=('S.......', '#..E....', '........', '#.......', '..#.....', '.....#..', '....#...', '.#......')),
    dict(need=1, budget=26, grid=('S###....', '........', '...#....', '....##..', '....#...', '...#....', '...E#...', '...#....')),
    dict(need=1, budget=30, grid=('S.#...##', '#.......', '#5......', '....#...', '........', '......#.', '..#..E#.', '....#..6')),
    dict(need=4, budget=28, grid=('S6...4##', '....#...', '..#....1', '.......#', '.#....E.', '#.##...#', '...#....', '#.......')),
    dict(need=4, budget=34, grid=('S.##....', '#.#.#.##', '........', '#...##3.', '....#...', '6...5..1', '....#...', '...#..E#')),
]
LEVELS = CONFIGS
SOLUTIONS = []
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def roll(aid, top, north, east):
    if aid == 1:
        return 7 - north, top, east
    if aid == 2:
        return north, 7 - top, east
    if aid == 4:
        return 7 - east, north, top
    return east, north, 7 - top


class Ng34(Core):
    STATE = ("pos", "top", "north", "east", "left")

    def __init__(self, seed=0):
        self.cfg = None
        self.pos = (0, 0)
        self.top, self.north, self.east = 1, 2, 3
        self.left = 0
        super().__init__("ng34-6e0b3d95", max(1, len(CONFIGS)), [1, 2, 3, 4], seed)

    def load(self, index):
        self.cfg = CONFIGS[index]
        self.top, self.north, self.east = 1, 2, 3
        self.left = self.cfg["budget"]
        for y, line in enumerate(self.cfg["grid"]):
            for x, ch in enumerate(line):
                if ch == "S":
                    self.pos = (x, y)

    def moves(self):
        return ["U", "D", "L", "R"]

    def act(self, aid, data):
        d = MOVES.get(aid)
        if d is None:
            return None
        self.left -= 1
        x, y = self.pos[0] + d[0], self.pos[1] + d[1]
        grid = self.cfg["grid"]
        if 0 <= x < 8 and 0 <= y < 8 and grid[y][x] != "#":
            t, n, e = roll(aid, self.top, self.north, self.east)
            ch = grid[y][x]
            need = self.cfg["need"] if ch == "E" else (int(ch) if ch.isdigit() else 0)
            if need == 0 or t == need:
                self.pos = (x, y)
                self.top, self.north, self.east = t, n, e
                if ch == "E":
                    return "win"
        if self.left <= 0:
            return "lose"
        return "ok"

    def draw(self, px):
        cfg = self.cfg
        bar(px, self.left * 60 // max(1, cfg["budget"]), 1, 14)
        for y in range(8):
            for x in range(8):
                gx, gy = OX + x * CELL, OY + y * CELL
                ch = cfg["grid"][y][x]
                if ch == "#":
                    rect(px, gx, gy, CELL, CELL, WALL)
                    continue
                rect(px, gx, gy, CELL, CELL, FLOOR)
                if ch == "E":
                    rect(px, gx, gy, CELL, CELL, EXIT)
                    dice(px, gx + 1, gy + 1, cfg["need"], 5, 1)
                elif ch.isdigit():
                    rect(px, gx, gy, CELL, CELL, GATE)
                    dice(px, gx + 1, gy + 1, int(ch), 0, 1)
        hx, hy = self.pos
        rect(px, OX + hx * CELL + 1, OY + hy * CELL + 1, 4, 4, HERO)
        for (bx, by, v) in ((52, 10, self.top), (58, 10, self.east), (52, 16, 7 - self.north)):
            rect(px, bx, by, 5, 5, 0)
            dice(px, bx + 1, by + 1, v, 5, 1)

SOLUTIONS = [
    ['R', 'D', 'R', 'R'],
    ['D', 'D', 'D', 'D', 'D', 'R', 'D', 'R', 'R'],
    ['R', 'D', 'R', 'D', 'D', 'D', 'R', 'D', 'R', 'D', 'R'],
    ['D', 'D', 'R', 'D', 'R', 'R', 'R', 'D', 'R', 'R'],
    ['R', 'D', 'D', 'R', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'R', 'D'],
]
