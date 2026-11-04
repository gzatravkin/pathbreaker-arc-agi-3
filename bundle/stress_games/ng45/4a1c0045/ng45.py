from collections import deque

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR_A = 3
FLOOR_B = 2
WALL = 13
TOKEN = 11
GOAL = 14
AVATAR = 9
CENTER = 0
BAR = 8
PIP_ON = 11
PIP_OFF = 2
PITCH = 4
FIELD_X = 2
FIELD_Y = 10
PAD_X = 44
PAD_Y = 10
PAD_CELL = 3
PAD_N = 5
BAR_MAX = 60

RULES = [
    ("id", "id"),
    ("rot", "rot"),
    ("tr", "tr"),
    ("id", "rot"),
    ("tr", "rot"),
]

MAPS = [
    [
        "##########",
        "#T......G#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#......T.#",
        "#........#",
        "#@.......#",
        "##########",
    ],
    [
        "##########",
        "#....##.G#",
        "#T.#...#.#",
        "##.......#",
        "#.#......#",
        "#....#...#",
        "#.#.#..#.#",
        "#........#",
        "#@.....T##",
        "##########",
    ],
    [
        "##########",
        "##..T##.G#",
        "#...#....#",
        "#......T.#",
        "#........#",
        "#.....##.#",
        "#..#.....#",
        "#.#......#",
        "#@...#.#T#",
        "##########",
    ],
    [
        "##########",
        "###.#...G#",
        "#........#",
        "#.#......#",
        "#..#.##.##",
        "#T####..T#",
        "##.......#",
        "#......T.#",
        "#@......##",
        "##########",
    ],
    [
        "##########",
        "#T.#.T..G#",
        "#..#.....#",
        "#.#.##...#",
        "#........#",
        "#.#...####",
        "#.#...#.T#",
        "#...#....#",
        "#@.#..#..#",
        "##########",
    ],
]


def transform(name, oi, oj):
    if name == "id":
        return oi, oj
    if name == "rot":
        return -oj, oi
    return oj, oi


def parse(index):
    rows = MAPS[index]
    walls, tokens = set(), []
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "T":
                tokens.append((x, y))
            elif ch == "G":
                goal = (x, y)
            elif ch == "@":
                start = (x, y)
    return dict(w=len(rows[0]), h=len(rows), walls=walls, tokens=tokens, start=start, goal=goal,
                rule=RULES[index], split=index >= 3)


def initial(cfg):
    return (cfg["start"][0], cfg["start"][1], 0)


def rule_name(cfg, x):
    left, right = cfg["rule"]
    return left if (x < cfg["w"] // 2 or not cfg["split"]) else right


def advance(cfg, state, click):
    x, y, mask = state
    oi, oj = click[0] - PAD_N // 2, click[1] - PAD_N // 2
    dx, dy = transform(rule_name(cfg, x), oi, oj)
    path = []
    px, py = x, y
    for axis, v in ((0, dx), (1, dy)):
        sign = 1 if v > 0 else -1
        for _ in range(abs(v)):
            nx, ny = (px + sign, py) if axis == 0 else (px, py + sign)
            if (nx, ny) in cfg["walls"] or not (0 <= nx < cfg["w"] and 0 <= ny < cfg["h"]):
                break
            px, py = nx, ny
            path.append((px, py))
            if (px, py) in cfg["tokens"]:
                mask |= 1 << cfg["tokens"].index((px, py))
    full = mask == (1 << len(cfg["tokens"])) - 1
    return (px, py, mask), path, ((px, py) == cfg["goal"] and full)


def pad_pixel(click):
    return (PAD_X + click[0] * PAD_CELL + 1, PAD_Y + click[1] * PAD_CELL + 1)


def pad_cell(x, y):
    i, j = (x - PAD_X) // PAD_CELL, (y - PAD_Y) // PAD_CELL
    if 0 <= x - PAD_X and 0 <= y - PAD_Y and 0 <= i < PAD_N and 0 <= j < PAD_N:
        return (i, j)
    return None


def solve(cfg):
    start = initial(cfg)
    seen = {start: []}
    queue = deque([start])
    clicks = [(i, j) for i in range(PAD_N) for j in range(PAD_N) if (i, j) != (PAD_N // 2, PAD_N // 2)]
    while queue:
        state = queue.popleft()
        for click in clicks:
            nxt, _, won = advance(cfg, state, click)
            if won:
                return seen[state] + [pad_pixel(click)]
            if nxt not in seen:
                seen[nxt] = seen[state] + [pad_pixel(click)]
                queue.append(nxt)
    raise ValueError("unsolvable")


CONFIGS = [parse(i) for i in range(len(MAPS))]
SOLUTIONS = [solve(c) for c in CONFIGS]
BUDGETS = [min(BAR_MAX, 2 * len(s) + 10) for s in SOLUTIONS]


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(MAPS))]


class Ng45(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(self._cfg)
        self._left = 0
        self._queue = None
        self._pending = None
        super().__init__(
            game_id="ng45-4a1c0045", levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6], win_score=len(LEVELS), seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._cfg = CONFIGS[index]
        self._st = initial(self._cfg)
        self._left = BUDGETS[index]
        self._queue = None
        self._draw(level, (self._st[0], self._st[1]))

    def _cell(self, px, x, y, colour, inset=0):
        ox, oy = FIELD_X + x * PITCH, FIELD_Y + y * PITCH
        px[oy + inset:oy + PITCH - inset, ox + inset:ox + PITCH - inset] = colour

    def _draw(self, level, at, mask=None):
        cfg = self._cfg
        mask = self._st[2] if mask is None else mask
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        count = len(cfg["tokens"])
        for i in range(count):
            px[4:6, 2 + i * 4:5 + i * 4] = PIP_ON if mask >> i & 1 else PIP_OFF
        for cy in range(cfg["h"]):
            for cx in range(cfg["w"]):
                if (cx, cy) in cfg["walls"]:
                    self._cell(px, cx, cy, WALL)
                else:
                    tint = FLOOR_A if (cx < cfg["w"] // 2 or not cfg["split"]) else FLOOR_B
                    self._cell(px, cx, cy, tint)
        full = mask == (1 << count) - 1
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL)
        if not full:
            self._cell(px, gx, gy, FLOOR_A, 1)
        for i, (tx, ty) in enumerate(cfg["tokens"]):
            if not mask >> i & 1:
                self._cell(px, tx, ty, TOKEN, 1)
        self._cell(px, at[0], at[1], AVATAR, 1)
        tint = FLOOR_A if (self._st[0] < cfg["w"] // 2 or not cfg["split"]) else FLOOR_B
        size = PAD_N * PAD_CELL
        px[PAD_Y:PAD_Y + size, PAD_X:PAD_X + size] = tint
        c = PAD_N // 2
        px[PAD_Y + c * PAD_CELL + 1, PAD_X + c * PAD_CELL + 1] = CENTER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id != GameAction.ACTION6:
            self.complete_action()
            return
        if self._queue is None:
            cell = pad_cell(self._action.data.get("x", -1), self._action.data.get("y", -1))
            if cell is None:
                self.complete_action()
                return
            nxt, path, won = advance(self._cfg, self._st, cell)
            self._left = max(0, self._left - 1)
            self._queue = list(path)
            self._pending = (nxt, won)
        if self._queue:
            at = self._queue.pop(0)
            self._draw(self.current_level, at)
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
