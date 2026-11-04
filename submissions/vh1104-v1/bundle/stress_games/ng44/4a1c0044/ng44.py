from collections import deque

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
ROT = 15
FLP = 12
HUDC = 4
TOKEN = 11
GOAL = 14
AVATAR = 9
MARK = 10
BAR = 8
PIP_ON = 11
PIP_OFF = 2
PITCH = 5
ORIGIN_X = 4
ORIGIN_Y = 14
BAR_MAX = 60

KEYS = {"U": 0, "R": 1, "D": 2, "L": 3}
STEP = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ACTIONS = {GameAction.ACTION1: "U", GameAction.ACTION2: "D", GameAction.ACTION3: "L",
           GameAction.ACTION4: "R"}

MAPS = [
    [
        "###########",
        "#........G#",
        "#.........#",
        "#.........#",
        "#.........#",
        "#T.TO.....#",
        "#.........#",
        "#@........#",
        "###########",
    ],
    [
        "###########",
        "#...##O..G#",
        "##.T#..#..#",
        "#...#.....#",
        "#.T.#..O..#",
        "#.#...#.#.#",
        "#......T..#",
        "#@...#..#.#",
        "###########",
    ],
    [
        "###########",
        "#..T##...G#",
        "#...#.#...#",
        "#..#F.#...#",
        "#......O..#",
        "#.#.......#",
        "#.T.......#",
        "#@.....T..#",
        "###########",
    ],
    [
        "###########",
        "#.#.TF..#G#",
        "###.##....#",
        "#.#..T#..##",
        "##...O....#",
        "#.....#..##",
        "#.#....#..#",
        "#@.#.#..OT#",
        "###########",
    ],
    [
        "###########",
        "#T...#F.#G#",
        "##.....T..#",
        "##.....##O#",
        "#F....#.O.#",
        "#..O......#",
        "#...##.#T.#",
        "#@....#..##",
        "###########",
    ],
]


def parse(index):
    rows = MAPS[index]
    walls, rots, flips, tokens = set(), set(), set(), []
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "O":
                rots.add((x, y))
            elif ch == "F":
                flips.add((x, y))
            elif ch == "T":
                tokens.append((x, y))
            elif ch == "G":
                goal = (x, y)
            elif ch == "@":
                start = (x, y)
    return dict(w=len(rows[0]), h=len(rows), walls=walls, rots=rots, flips=flips, tokens=tokens, start=start,
                goal=goal)


def initial(cfg):
    return (cfg["start"][0], cfg["start"][1], 0, 0, 0)


def actual(rot, flip, key):
    k = KEYS[key]
    if flip and k in (1, 3):
        k = 4 - k
    return (k + rot) % 4


def advance(cfg, state, key):
    x, y, rot, flip, mask = state
    dx, dy = STEP[actual(rot, flip, key)]
    dest = (x + dx, y + dy)
    if dest in cfg["walls"] or not (0 <= dest[0] < cfg["w"] and 0 <= dest[1] < cfg["h"]):
        return state, False
    if dest in cfg["rots"]:
        rot = (rot + 1) % 4
    if dest in cfg["flips"]:
        flip = 1 - flip
    if dest in cfg["tokens"]:
        mask |= 1 << cfg["tokens"].index(dest)
    full = mask == (1 << len(cfg["tokens"])) - 1
    return (dest[0], dest[1], rot, flip, mask), (dest == cfg["goal"] and full)


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


class Ng44(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(self._cfg)
        self._left = 0
        super().__init__(
            game_id="ng44-4a1c0044", levels=LEVELS,
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

    def _draw(self, level):
        cfg = self._cfg
        x, y, rot, flip, mask = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        count = len(cfg["tokens"])
        for i in range(count):
            px[4:6, 2 + i * 4:5 + i * 4] = PIP_ON if mask >> i & 1 else PIP_OFF
        for cy in range(cfg["h"]):
            for cx in range(cfg["w"]):
                self._cell(px, cx, cy, WALL if (cx, cy) in cfg["walls"] else FLOOR)
        for cell in cfg["rots"]:
            self._cell(px, cell[0], cell[1], ROT, 1)
        for cell in cfg["flips"]:
            self._cell(px, cell[0], cell[1], FLP, 1)
        full = mask == (1 << count) - 1
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL)
        if not full:
            self._cell(px, gx, gy, FLOOR, 1)
        for i, (tx, ty) in enumerate(cfg["tokens"]):
            if not mask >> i & 1:
                self._cell(px, tx, ty, TOKEN, 1)
        self._cell(px, x, y, AVATAR, 1)
        for name in "URDL":
            ddx, ddy = STEP[KEYS[name]]
            bx, by = 56 + ddx * 4, 7 + ddy * 4
            px[by - 1:by + 2, bx - 1:bx + 2] = HUDC
            ex, ey = STEP[actual(rot, flip, name)]
            px[by + ey, bx + ex] = MARK
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
