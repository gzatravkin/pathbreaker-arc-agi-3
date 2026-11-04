import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 3
FLOOR = 1
WALL = 4
CELL = 6
OY = 8
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
KEYS = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}
_CACHE = {}


def parse(lv):
    if lv in _CACHE:
        return _CACHE[lv]
    rows = MAPS[lv][0]
    walls = set()
    marks = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch != ".":
                marks.setdefault(ch, []).append((x, y))
    info = dict(w=len(rows[0]), h=len(rows), walls=walls, marks=marks)
    _CACHE[lv] = info
    return info


def blocked(lv, cell):
    info = parse(lv)
    x, y = cell
    return not (0 <= x < info["w"] and 0 <= y < info["h"]) or cell in info["walls"]


def origin(lv, cell):
    info = parse(lv)
    ox = (64 - info["w"] * CELL) // 2
    return ox + cell[0] * CELL, OY + cell[1] * CELL


def rect(px, x, y, w, h, c):
    px[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = c


def tile(px, lv, cell, dx, dy, w, h, c):
    x, y = origin(lv, cell)
    rect(px, x + dx, y + dy, w, h, c)


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

GAME_ID = "ng99-8f6dab54"
AVAILABLE = [1, 2, 3, 4, 7]
ALLOWED = "UDLR"
COL = {"r": 8, "b": 9, "y": 11}

MAPS = [
    (["########",
      "#@.....#",
      "#..b.r.#",
      "#.....F#",
      "########"], "rb"),
    (["#########",
      "#@..y...#",
      "#.r...b.#",
      "#.....y.#",
      "#.......#",
      "#F......#",
      "#########"], "ybr"),
    (["#########",
      "#@..#..F#",
      "#...#...#",
      "#.r.#.b.#",
      "#...y...#",
      "#########"], "ryb"),
    (["#########",
      "#@..#..Y#",
      "#.r.#...#",
      "#...#.b.#",
      "#R..y..B#",
      "#########"], "rby"),
    (["###########",
      "#@..#....Y#",
      "#.b.#..r..#",
      "#.r.y.....#",
      "#.y.#.b...#",
      "#R..#...B.#",
      "###########"], "rbyr"),
]
SOLUTIONS = [
    ['D', 'R', 'R', 'R', 'R', 'D', 'R', 'U', 'L', 'D', 'R'],
    ['R', 'R', 'R', 'D', 'D', 'D', 'D', 'L', 'L', 'L', 'U', 'U', 'R', 'R', 'U', 'R', 'R', 'R', 'L', 'D', 'D', 'D', 'L', 'L', 'L', 'L', 'U', 'U', 'U', 'R', 'D', 'D', 'D', 'L'],
    ['D', 'D', 'R', 'D', 'R', 'R', 'L', 'R', 'R', 'U', 'U', 'U', 'R', 'R', 'D', 'D', 'D', 'L', 'L', 'L', 'R', 'U', 'U', 'U', 'R', 'R', 'D', 'D', 'L', 'U', 'U', 'R'],
    ['D', 'R', 'D', 'D', 'L', 'R', 'R', 'R', 'R', 'U', 'R', 'D', 'R', 'U', 'L', 'U', 'U', 'R'],
    ['D', 'D', 'R', 'L', 'D', 'D', 'U', 'U', 'R', 'R', 'R', 'R', 'D', 'R', 'D', 'R', 'R', 'U', 'L', 'L', 'U', 'U', 'U', 'R', 'R', 'R', 'D', 'L', 'L', 'D', 'L', 'L', 'L', 'L', 'D', 'D', 'L', 'L'],
]


def initial(lv):
    m = parse(lv)["marks"]
    items = set()
    for ch in "rby":
        for c in m.get(ch, []):
            items.add((c, ch))
    return (m["@"][0], None, frozenset(items), 0)


def homes(lv):
    m = parse(lv)["marks"]
    out = {}
    for ch in "FRBY":
        for c in m.get(ch, []):
            out[c] = ch.lower()
    return out


def apply(lv, s, k):
    pos, carry, items, qi = s
    order = MAPS[lv][1]
    hs = homes(lv)
    d = DIRS[k]
    dest = (pos[0] + d[0], pos[1] + d[1])
    if blocked(lv, dest):
        return s
    if dest in hs:
        if carry is not None and carry == order[qi] and hs[dest] in ("f", carry):
            return (dest, None, items, qi + 1)
        return s
    for c, col in items:
        if c == dest:
            if carry is None:
                return (dest, col, items - {(c, col)}, qi)
            return (dest, col, (items - {(c, col)}) | {(c, carry)}, qi)
    return (dest, carry, items, qi)


def won(lv, s):
    return s[3] == len(MAPS[lv][1])


def render(px, lv, s):
    pos, carry, items, qi = s
    order = MAPS[lv][1]
    for c, ch in homes(lv).items():
        if ch == "f":
            tile(px, lv, c, 0, 0, CELL, CELL, 5)
            tile(px, lv, c, 1, 1, 4, 4, 1)
            tile(px, lv, c, 2, 1, 3, 2, 14)
            tile(px, lv, c, 2, 1, 1, 4, 5)
        else:
            tile(px, lv, c, 0, 0, CELL, CELL, COL[ch])
            tile(px, lv, c, 1, 1, 4, 4, FLOOR)
            tile(px, lv, c, 2, 2, 2, 2, COL[ch])
    for c, col in items:
        tile(px, lv, c, 1, 1, 4, 4, COL[col])
        tile(px, lv, c, 2, 2, 2, 2, 1)
    tile(px, lv, pos, 1, 3, 4, 3, 5)
    if carry is None:
        tile(px, lv, pos, 1, 0, 4, 2, 2)
    else:
        tile(px, lv, pos, 1, 0, 4, 2, COL[carry])
    for i, ch in enumerate(order):
        x = 2 + i * 8
        if i < qi:
            rect(px, x, 1, 6, 5, 5)
            rect(px, x + 2, 2, 2, 3, 14)
        else:
            rect(px, x, 1, 6, 5, 0 if i == qi else 5)
            rect(px, x + 1, 2, 4, 3, COL[ch])


LEVELS = [canvas_level(i) for i in range(len(MAPS))]


class Ng99(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._lv = 0
        self._st = None
        self._hist = []
        super().__init__(
            game_id=GAME_ID,
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=AVAILABLE,
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._lv = self._current_level_index
        self._st = initial(self._lv)
        self._hist = []
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        lv = self._lv
        info = parse(lv)
        px = np.full((64, 64), BG, dtype=np.int8)
        for y in range(info["h"]):
            for x in range(info["w"]):
                tile(px, lv, (x, y), 0, 0, CELL, CELL, WALL if (x, y) in info["walls"] else FLOOR)
        render(px, lv, self._st)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        act = self._action.id
        if act == GameAction.ACTION7:
            if self._hist:
                self._st = self._hist.pop()
            self._draw()
        else:
            key = KEYS.get(act)
            if key is not None and key in ALLOWED:
                new = apply(self._lv, self._st, key)
                if new != self._st:
                    self._hist.append(self._st)
                    self._st = new
                    self._draw()
                    if won(self._lv, self._st):
                        self.next_level()
        self.complete_action()
