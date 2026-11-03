from collections import deque

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
PIT = 8
TOKEN = 11
GOAL = 14
AVATAR = 9
CURSOR = 10
BAR = 8
PIP_ON = 11
PIP_OFF = 2
PITCH = 5
ORIGIN_X = 4
ORIGIN_Y = 8
BAR_MAX = 60

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0), "S": (0, 0)}
ACTIONS = {GameAction.ACTION1: "U", GameAction.ACTION2: "D", GameAction.ACTION3: "L",
           GameAction.ACTION4: "R", GameAction.ACTION5: "S"}

MAPS = [
    [
        "###########",
        "#.........#",
        "#...X.T...#",
        "#.........#",
        "#.T.....X.#",
        "#....X....#",
        "#.....T...#",
        "#@.......G#",
        "###########",
    ],
    [
        "###########",
        "#........G#",
        "#.#.......#",
        "##....#...#",
        "##........#",
        "#..#..##.##",
        "#.........#",
        "#@..##.#.T#",
        "###########",
    ],
    [
        "###########",
        "##.#.#...G#",
        "#.......X.#",
        "##..T#....#",
        "#.#...X...#",
        "#..X#..#..#",
        "##T#..#..##",
        "#@........#",
        "###########",
    ],
    [
        "###########",
        "#.X.#...#G#",
        "#.#..#....#",
        "#T...#....#",
        "####..X...#",
        "#.#..##.#.#",
        "#....#X..##",
        "#@##T...###",
        "###########",
    ],
    [
        "###########",
        "##..#....G#",
        "##......###",
        "#T.#.X.####",
        "#.#.XT.X..#",
        "##.X#..##.#",
        "#...#.....#",
        "#..####T.##",
        "#@......#.#",
        "###########",
    ],
]


def parse(index):
    rows = MAPS[index]
    walls, pits, tokens = set(), set(), []
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "X":
                pits.add((x, y))
            elif ch == "T":
                tokens.append((x, y))
            elif ch == "G":
                goal = (x, y)
            elif ch == "@":
                start = (x, y)
    return dict(w=len(rows[0]), h=len(rows), walls=walls, pits=pits, tokens=tokens, start=start, goal=goal)


def initial(cfg):
    sx, sy = cfg["start"]
    return (sx, sy, sx, sy, 0)


def free(cfg, cell):
    return 0 <= cell[0] < cfg["w"] and 0 <= cell[1] < cfg["h"] and cell not in cfg["walls"]


def advance(cfg, state, key):
    ax, ay, cx, cy, mask = state
    if key == "S":
        cx, cy = ax, ay
    else:
        dx, dy = DIRS[key]
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < cfg["w"] and 0 <= ny < cfg["h"]:
            cx, cy = nx, ny
    ddx, ddy = cx - ax, cy - ay
    if ddx or ddy:
        order = [(1 if ddx > 0 else -1, 0), (0, 1 if ddy > 0 else -1)]
        if not ddx:
            order = order[1:]
        elif not ddy:
            order = order[:1]
        elif abs(ddy) > abs(ddx):
            order = order[::-1]
        for sx, sy in order:
            dest = (ax + sx, ay + sy)
            if free(cfg, dest):
                ax, ay = dest
                break
    if (ax, ay) in cfg["pits"]:
        return (cfg["start"][0], cfg["start"][1], cfg["start"][0], cfg["start"][1], mask), False
    if (ax, ay) in cfg["tokens"]:
        mask |= 1 << cfg["tokens"].index((ax, ay))
    full = mask == (1 << len(cfg["tokens"])) - 1
    return (ax, ay, cx, cy, mask), ((ax, ay) == cfg["goal"] and full)


def solve(cfg):
    start = initial(cfg)
    seen = {start: ""}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for key in "URDLS":
            nxt, won = advance(cfg, state, key)
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


class Ng43(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(self._cfg)
        self._left = 0
        super().__init__(
            game_id="ng43-4a1c0043", levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5], win_score=len(LEVELS), seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._cfg = CONFIGS[index]
        self._st = initial(self._cfg)
        self._left = BUDGETS[index]
        self._draw(level)

    def _cell(self, px, x, y, colour, inset=0):
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + inset:oy + PITCH - inset, ox + inset:ox + PITCH - inset] = colour

    def _draw(self, level):
        cfg = self._cfg
        ax, ay, cx, cy, mask = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        count = len(cfg["tokens"])
        for i in range(count):
            px[4:6, 2 + i * 4:5 + i * 4] = PIP_ON if mask >> i & 1 else PIP_OFF
        for y in range(cfg["h"]):
            for x in range(cfg["w"]):
                self._cell(px, x, y, WALL if (x, y) in cfg["walls"] else FLOOR)
        for cell in cfg["pits"]:
            self._cell(px, cell[0], cell[1], PIT, 1)
        full = mask == (1 << count) - 1
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL)
        if not full:
            self._cell(px, gx, gy, FLOOR, 1)
        for i, (tx, ty) in enumerate(cfg["tokens"]):
            if not mask >> i & 1:
                self._cell(px, tx, ty, TOKEN, 1)
        self._cell(px, ax, ay, AVATAR, 1)
        ox, oy = ORIGIN_X + cx * PITCH, ORIGIN_Y + cy * PITCH
        px[oy, ox:ox + PITCH] = CURSOR
        px[oy + PITCH - 1, ox:ox + PITCH] = CURSOR
        px[oy:oy + PITCH, ox] = CURSOR
        px[oy:oy + PITCH, ox + PITCH - 1] = CURSOR
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        key = ACTIONS.get(self._action.id)
        if key is not None:
            self._st, won = advance(self._cfg, self._st, key)
            self._left = max(0, self._left - 1)
            self._draw(self.current_level)
            if won:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
