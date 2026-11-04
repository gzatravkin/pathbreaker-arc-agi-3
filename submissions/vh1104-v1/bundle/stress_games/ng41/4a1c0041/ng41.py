from collections import deque

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
TOKEN = 11
GOAL = 14
PAD = 15
AVATAR = 9
NOSE = 0
BAR = 8
PIP_ON = 11
PIP_OFF = 2
PITCH = 5
ORIGIN_X = 4
ORIGIN_Y = 8
BAR_MAX = 60

KEYS = {"U": 0, "R": 1, "D": 2, "L": 3}
STEP = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ACTIONS = {GameAction.ACTION1: "U", GameAction.ACTION4: "R", GameAction.ACTION2: "D", GameAction.ACTION3: "L"}

MAPS = [
    [
        "#########",
        "#.....#G#",
        "#.###.#.#",
        "#.#...#.#",
        "#.#.###.#",
        "#.#T....#",
        "#.#######",
        "#@T.....#",
        "#########",
    ],
    [
        "#########",
        "#T..#..G#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.###.#T#",
        "#.#...#.#",
        "#.#.###.#",
        "#@#.....#",
        "#########",
    ],
    [
        "###########",
        "#....#...G#",
        "#.##.#.##.#",
        "#.#T...#T.#",
        "#.#.####..#",
        "#.#.#T....#",
        "#.#.#.###.#",
        "#...#...#.#",
        "#.###.#.#.#",
        "#@....#...#",
        "###########",
    ],
    [
        "###########",
        "#T...#...G#",
        "#.##.#.####",
        "#.#..P....#",
        "#.#.###.#.#",
        "#.#...#.#T#",
        "#.###.#.#.#",
        "#...#...#.#",
        "#.#.#.###.#",
        "#@#...P...#",
        "###########",
    ],
    [
        "###########",
        "#.T.#P....#",
        "#.#.#.###.#",
        "#.#...#T#.#",
        "#.###.#.#.#",
        "#.#G..#.#.#",
        "#.#.###.#.#",
        "#.#.P...#.#",
        "#@#.#######",
        "#.........#",
        "###########",
    ],
]
STARTS = [0, 0, 0, 0, 0]
ABS_START = [0, 0, 0, 0, 1]


def parse(index):
    rows = MAPS[index]
    walls, tokens, pads = set(), [], set()
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "T":
                tokens.append((x, y))
            elif ch == "P":
                pads.add((x, y))
            elif ch == "G":
                goal = (x, y)
            elif ch == "@":
                start = (x, y)
    return dict(w=len(rows[0]), h=len(rows), walls=walls, tokens=tokens, pads=pads, start=start, goal=goal,
                mode=ABS_START[index], head=STARTS[index])


def initial(cfg):
    return (cfg["start"][0], cfg["start"][1], cfg["head"], cfg["mode"], 0)


def advance(cfg, state, key):
    x, y, head, mode, mask = state
    rel = KEYS[key]
    new_head = (head + rel) % 4 if mode == 0 else rel
    dx, dy = STEP[new_head]
    dest = (x + dx, y + dy)
    if dest in cfg["walls"] or not (0 <= dest[0] < cfg["w"] and 0 <= dest[1] < cfg["h"]):
        return (x, y, new_head, mode, mask), False
    if dest in cfg["tokens"]:
        mask |= 1 << cfg["tokens"].index(dest)
    if dest in cfg["pads"]:
        mode = 1 - mode
    full = mask == (1 << len(cfg["tokens"])) - 1
    return (dest[0], dest[1], new_head, mode, mask), (dest == cfg["goal"] and full)


def solve(cfg):
    start = initial(cfg)
    seen = {start: ""}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for key in "URDL":
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


class Ng41(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(self._cfg)
        self._left = 0
        super().__init__(
            game_id="ng41-4a1c0041", levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4], win_score=len(LEVELS), seed=seed,
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

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        x, y, head, mode, mask = self._st
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
                    self._cell(px, cx, cy, FLOOR, 0)
        for cell in cfg["pads"]:
            self._cell(px, cell[0], cell[1], PAD, 1)
        full = mask == (1 << count) - 1
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL, 0)
        if not full:
            self._cell(px, gx, gy, FLOOR, 1)
        for i, (tx, ty) in enumerate(cfg["tokens"]):
            if not mask >> i & 1:
                self._cell(px, tx, ty, TOKEN, 1)
        self._cell(px, x, y, AVATAR, 1)
        if mode == 0:
            ox, oy = ORIGIN_X + x * PITCH + 2, ORIGIN_Y + y * PITCH + 2
            dx, dy = STEP[head]
            px[oy + dy * 2, ox + dx * 2] = NOSE
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        key = ACTIONS.get(self._action.id)
        if key is not None:
            self._st, won = advance(self._cfg, self._st, key)
            self._left = max(0, self._left - 1)
            self._draw()
            if won:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
