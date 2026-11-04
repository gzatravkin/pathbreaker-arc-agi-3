import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
FLOOR = 4
WALL = 3
WALL_EDGE = 2
S = 6
MOVES = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
PIP_SPOTS = {
    1: [(2, 2)],
    2: [(1, 1), (3, 3)],
    3: [(1, 1), (3, 1), (2, 3)],
    4: [(1, 1), (3, 1), (1, 3), (3, 3)],
}


def parse(rows):
    grid = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != " ":
                grid[(x, y)] = ch
    w = max(len(r) for r in rows)
    h = len(rows)
    return dict(w=w, h=h, g=grid, ox=(64 - w * S) // 2, oy=(64 - h * S) // 2 + 2)


def fill(px, lv, cell, col, m=0):
    x = lv["ox"] + cell[0] * S + m
    y = lv["oy"] + cell[1] * S + m
    px[y:y + S - 2 * m, x:x + S - 2 * m] = col


def dot(px, lv, cell, dx, dy, col):
    px[lv["oy"] + cell[1] * S + dy, lv["ox"] + cell[0] * S + dx] = col


def pips(px, lv, cell, n, col):
    for dx, dy in PIP_SPOTS[n]:
        dot(px, lv, cell, dx, dy, col)


def base_frame(lv):
    px = np.full((64, 64), BG, dtype=np.int8)
    for (x, y), ch in lv["g"].items():
        if ch == "#":
            fill(px, lv, (x, y), WALL)
            fill(px, lv, (x, y), WALL_EDGE, 1)
            fill(px, lv, (x, y), WALL, 2)
        else:
            fill(px, lv, (x, y), FLOOR)
    return px


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BG, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


class Game(ARCBaseGame):
    def __init__(self, gid, seed=0):
        self._lv = LEVELS_DATA[0]
        self._st = init(self._lv)
        self._hist = []
        super().__init__(
            game_id=gid,
            levels=[canvas_level(i) for i in range(len(LEVELS_DATA))],
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=ACTIONS,
            win_score=len(LEVELS_DATA),
            seed=seed,
        )

    def on_set_level(self, level):
        self._lv = LEVELS_DATA[self._current_level_index]
        self._st = init(self._lv)
        self._hist = []
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = base_frame(self._lv)
        draw(px, self._lv, self._st)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        aid = self._action.id
        if aid == GameAction.ACTION7:
            if self._hist:
                self._st = self._hist.pop()
                self._draw()
        elif aid in MOVES:
            ns = move(self._lv, self._st, MOVES[aid])
            if ns is not None:
                self._hist.append(self._st)
                self._st = ns
                self._draw()
                if won(self._lv, ns):
                    self.next_level()
        self.complete_action()

ACTIONS = [1, 2, 3, 4, 7]
TARGET_CH = {"p": 1, "q": 2, "r": 3, "t": 4}
BOX_COL = {1: 12, 2: 8, 3: 15, 4: 9}

LEVELS_DATA = [parse(r) for r in [
    ["#######",
     "#.....#",
     "#@.1.1#",
     "#....q#",
     "#######"],
    ["#########",
     "#.......#",
     "#@.1.2.r#",
     "#.......#",
     "#########"],
    ["#########",
     "#.......#",
     "#.1.1...#",
     "#@..q.1.#",
     "#...1.q.#",
     "#########"],
    ["#########",
     "#p.....r#",
     "#..#.#..#",
     "#.1.2.1.#",
     "#..#@#..#",
     "#########"],
    ["##########",
     "#........#",
     "#.1..1.1.#",
     "#@.#..#..#",
     "#..1.2...#",
     "#t.....p.#",
     "##########"],
]]


def init(lv):
    boxes, p = {}, None
    for (x, y), ch in lv["g"].items():
        if ch == "@":
            p = (x, y)
        elif ch in "1234":
            boxes[(x, y)] = int(ch)
    return (p, tuple(sorted((c, v) for c, v in boxes.items())))


def targets_of(lv):
    return {c: TARGET_CH[ch] for c, ch in lv["g"].items() if ch in TARGET_CH}


def move(lv, st, m):
    p, boxes = st
    dx, dy = DIRS[m]
    g = lv["g"]
    dest = (p[0] + dx, p[1] + dy)
    if g.get(dest, "#") == "#":
        return None
    bx = dict(boxes)
    if dest in bx:
        far = (dest[0] + dx, dest[1] + dy)
        if g.get(far, "#") == "#":
            return None
        v = bx.pop(dest)
        bx[far] = bx.get(far, 0) + v
        if bx[far] > 4:
            return None
    return (dest, tuple(sorted(bx.items())))


def won(lv, st):
    bx = dict(st[1])
    return all(bx.get(c) == v for c, v in targets_of(lv).items())


def draw(px, lv, st):
    p, boxes = st
    bx = dict(boxes)
    for c, v in targets_of(lv).items():
        fill(px, lv, c, 11, 0)
        fill(px, lv, c, FLOOR, 1)
        pips(px, lv, c, v, 11)
    for c, v in boxes:
        done = targets_of(lv).get(c) == v
        col = 14 if done else BOX_COL[v]
        fill(px, lv, c, col, 1)
        pips(px, lv, c, v, 4 if v != 4 else 1)
        if v == 4:
            pips(px, lv, c, 4, 1)
    fill(px, lv, p, 10, 1)
    fill(px, lv, p, 9, 2)


SOLUTIONS = [['R', 'R', 'R', 'U', 'R', 'D'], ['R', 'R', 'R', 'R', 'R'], ['U', 'R', 'R', 'U', 'R', 'D', 'L', 'D', 'D', 'R', 'R', 'U', 'U', 'R', 'D'], ['U', 'U', 'D', 'L', 'L', 'D', 'L', 'U', 'U', 'R', 'U', 'R', 'R', 'R', 'R', 'L', 'L', 'D', 'D', 'R', 'R', 'D', 'R', 'U', 'U'], ['U', 'U', 'R', 'D', 'D', 'U', 'U', 'R', 'R', 'R', 'R', 'R', 'D', 'D', 'D', 'L', 'L', 'L', 'L', 'L', 'U', 'L', 'D']]


class Ng92(Game):
    def __init__(self, seed=0):
        super().__init__("ng92-9a20c002", seed)
