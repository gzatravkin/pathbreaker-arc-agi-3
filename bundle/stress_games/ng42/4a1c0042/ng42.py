from collections import deque

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
SAND = 12
PIT = 8
SWITCH = 11
GOAL = 14
AVATAR = 9
VEL_ON = 10
VEL_OFF = 4
BAR = 8
PITCH = 5
ORIGIN_X = 4
ORIGIN_Y = 8
BAR_MAX = 60
CAP = 3

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0), "S": (0, 0)}
ACTIONS = {GameAction.ACTION1: "U", GameAction.ACTION2: "D", GameAction.ACTION3: "L",
           GameAction.ACTION4: "R", GameAction.ACTION5: "S"}

MAPS = [
    [
        "###########",
        "#K........#",
        "#.....K...#",
        "#...###...#",
        "#.K.#.....#",
        "#...#...K.#",
        "#.....#...#",
        "#.K.......#",
        "#@.......G#",
        "###########",
    ],
    [
        "###########",
        "#.........#",
        "#.K.......#",
        "#.........#",
        "#....G....#",
        "#.......K.#",
        "#.........#",
        "#.K.......#",
        "#@........#",
        "###########",
    ],
    [
        "###########",
        "#....X....#",
        "#.K.......#",
        "#......X..#",
        "#....G....#",
        "#.X.......#",
        "#.......K.#",
        "#...X.....#",
        "#@........#",
        "###########",
    ],
    [
        "###########",
        "#.........#",
        "#..K.SS...#",
        "#....SS...#",
        "#.....G...#",
        "#.SS......#",
        "#.SS...K..#",
        "#.........#",
        "#@........#",
        "###########",
    ],
    [
        "###########",
        "#..X...X..#",
        "#.K.SS....#",
        "#...SS..K.#",
        "#.X...G...#",
        "#.SS...X..#",
        "#.SS......#",
        "#...X.K...#",
        "#@........#",
        "###########",
    ],
]


def parse(index):
    rows = MAPS[index]
    walls, sand, pits, keys = set(), set(), set(), []
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "S":
                sand.add((x, y))
            elif ch == "X":
                pits.add((x, y))
            elif ch == "K":
                keys.append((x, y))
            elif ch == "G":
                goal = (x, y)
            elif ch == "@":
                start = (x, y)
    return dict(w=len(rows[0]), h=len(rows), walls=walls, sand=sand, pits=pits, keys=keys, start=start,
                goal=goal)


def initial(cfg):
    return (cfg["start"][0], cfg["start"][1], 0, 0, 0)


def advance(cfg, state, key):
    x, y, vx, vy, mask = state
    dx, dy = DIRS[key]
    vx = max(-CAP, min(CAP, vx + dx))
    vy = max(-CAP, min(CAP, vy + dy))
    path = []
    px, py = x, y
    for axis in (0, 1):
        v = vx if axis == 0 else vy
        sign = 1 if v > 0 else -1
        for _ in range(abs(v)):
            nx, ny = (px + sign, py) if axis == 0 else (px, py + sign)
            if (nx, ny) in cfg["walls"] or not (0 <= nx < cfg["w"] and 0 <= ny < cfg["h"]):
                if axis == 0:
                    vx = 0
                else:
                    vy = 0
                break
            px, py = nx, ny
            path.append((px, py))
            if (px, py) in cfg["pits"]:
                return (cfg["start"][0], cfg["start"][1], 0, 0, mask), path, False
            if (px, py) in cfg["keys"]:
                mask |= 1 << cfg["keys"].index((px, py))
    if (px, py) in cfg["sand"]:
        vx -= (vx > 0) - (vx < 0)
        vy -= (vy > 0) - (vy < 0)
    full = mask == (1 << len(cfg["keys"])) - 1
    won = (px, py) == cfg["goal"] and vx == 0 and vy == 0 and full
    return (px, py, vx, vy, mask), path, won


def solve(cfg):
    start = initial(cfg)
    seen = {start: ""}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for key in "URDLS":
            nxt, _, won = advance(cfg, state, key)
            if won:
                return seen[state] + key
            if nxt not in seen:
                seen[nxt] = seen[state] + key
                queue.append(nxt)
    raise ValueError("unsolvable")


CONFIGS = [parse(i) for i in range(len(MAPS))]
SOLUTIONS = [list(solve(c)) for c in CONFIGS]
BUDGETS = [min(BAR_MAX, 2 * len(s) + 10) for s in SOLUTIONS]


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(MAPS))]


class Ng42(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(self._cfg)
        self._left = 0
        self._queue = None
        self._pending = None
        super().__init__(
            game_id="ng42-4a1c0042", levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5], win_score=len(LEVELS), seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._cfg = CONFIGS[index]
        self._st = initial(self._cfg)
        self._left = BUDGETS[index]
        self._queue = None
        self._draw(level, (self._st[0], self._st[1]))

    def _cell(self, px, x, y, colour, inset=0):
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + inset:oy + PITCH - inset, ox + inset:ox + PITCH - inset] = colour

    def _draw(self, level, at, shown=None):
        cfg = self._cfg
        x, y, vx, vy, mask = shown or self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[7, 2:2 + self._left] = BAR
        for i in range(len(cfg["keys"])):
            px[3:5, 2 + i * 4:5 + i * 4] = SWITCH if mask >> i & 1 else 2
        for cy in range(cfg["h"]):
            for cx in range(cfg["w"]):
                self._cell(px, cx, cy, WALL if (cx, cy) in cfg["walls"] else FLOOR)
        for cell in cfg["sand"]:
            self._cell(px, cell[0], cell[1], SAND)
        for cell in cfg["pits"]:
            self._cell(px, cell[0], cell[1], PIT, 1)
        for i, cell in enumerate(cfg["keys"]):
            if not mask >> i & 1:
                self._cell(px, cell[0], cell[1], SWITCH, 1)
        full = mask == (1 << len(cfg["keys"])) - 1
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL)
        if not full:
            self._cell(px, gx, gy, FLOOR, 1)
        self._cell(px, at[0], at[1], AVATAR, 1)
        cx0, cy0 = 53, 0
        px[cy0 + 3, cx0 + 3] = VEL_OFF
        for k in range(1, CAP + 1):
            px[cy0 + 3, cx0 + 3 + k] = VEL_ON if vx >= k else VEL_OFF
            px[cy0 + 3, cx0 + 3 - k] = VEL_ON if -vx >= k else VEL_OFF
            px[cy0 + 3 + k, cx0 + 3] = VEL_ON if vy >= k else VEL_OFF
            px[cy0 + 3 - k, cx0 + 3] = VEL_ON if -vy >= k else VEL_OFF
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        key = ACTIONS.get(self._action.id)
        if key is None:
            self.complete_action()
            return
        if self._queue is None:
            nxt, path, won = advance(self._cfg, self._st, key)
            self._left = max(0, self._left - 1)
            self._queue = list(path)
            self._pending = (nxt, won)
        if self._queue:
            at = self._queue.pop(0)
            self._draw(self.current_level, at, self._st)
            if self._queue:
                return
        self._st, won = self._pending
        self._queue = None
        self._draw(self.current_level, (self._st[0], self._st[1]))
        if won:
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
