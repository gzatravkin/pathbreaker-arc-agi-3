import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
BODY = 3
SINK = 14
TOKEN = 8
CRATE = 11
PIP = 0
BAR = 9
BAR_MAX = 60
EDGE_COLOURS = {1: 12, 2: 10, 3: 15}
OX = 9
OY = 12
PITCH = 13
HALF = 3

CONFIGS = [
    {'nodes': [(0, 2), (2, 2), (1, 3), (0, 1), (1, 2), (3, 1), (3, 0)], 'edges': [(0, 1, 1), (1, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1), (5, 6, 3)], 'ecol': [12, 12, 12, 12, 12, 15], 'sink': 4, 'start': 5, 'crates': {6: 1, 3: 1}, 'budget': 28},
    {'nodes': [(1, 2), (1, 0), (1, 1), (3, 2), (0, 0), (2, 0), (2, 2), (0, 1)], 'edges': [(0, 1, 2), (1, 2, 3), (0, 3, 1), (1, 4, 3), (2, 5, 2), (5, 6, 3), (4, 7, 3), (0, 7, 2), (0, 2, 3)], 'ecol': [10, 15, 10, 12, 12, 12, 12, 15, 12], 'sink': 7, 'start': 3, 'crates': {2: 2, 5: 2, 6: 2}, 'budget': 44},
    {'nodes': [(1, 3), (0, 2), (0, 0), (1, 0), (3, 3), (3, 2), (0, 1), (1, 2), (2, 1)], 'edges': [(0, 1, 1), (1, 2, 3), (2, 3, 3), (0, 4, 1), (4, 5, 2), (2, 6, 3), (1, 7, 3), (6, 8, 3), (6, 7, 1)], 'ecol': [15, 10, 12, 12, 12, 15, 12, 12, 12], 'sink': 1, 'start': 4, 'crates': {6: 3, 3: 3, 8: 1}, 'budget': 40},
    {'nodes': [(1, 2), (1, 3), (2, 1), (0, 3), (2, 2), (1, 1), (2, 0), (2, 3), (3, 3)], 'edges': [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 4, 1), (1, 5, 1), (4, 6, 1), (0, 7, 1), (1, 8, 2), (2, 6, 2), (4, 8, 3)], 'ecol': [15, 10, 12, 12, 15, 10, 15, 10, 15, 12], 'sink': 2, 'start': 6, 'crates': {4: 2, 5: 1, 1: 2, 0: 3}, 'budget': 46},
    {'nodes': [(1, 0), (0, 1), (2, 1), (2, 0), (3, 0), (2, 2), (0, 3), (0, 2), (1, 2), (1, 3)], 'edges': [(0, 1, 2), (1, 2, 3), (2, 3, 2), (3, 4, 1), (3, 5, 2), (1, 6, 3), (6, 7, 1), (0, 8, 1), (8, 9, 3), (5, 7, 3)], 'ecol': [12, 10, 12, 15, 12, 15, 15, 15, 12, 15], 'sink': 0, 'start': 8, 'crates': {5: 2, 4: 1, 2: 1, 6: 1}, 'budget': 54},
]

SOLUTIONS = [[(48, 12), (48, 25), (35, 38), (22, 38), (35, 38), (9, 38), (9, 25), (9, 38), (35, 38), (22, 38)], [(22, 38), (22, 25), (22, 38), (9, 25), (22, 38), (22, 25), (35, 12), (22, 25), (22, 38), (9, 25), (22, 38), (22, 25), (35, 12), (35, 38), (35, 12), (22, 25), (22, 38), (9, 25)], [(22, 51), (9, 38), (9, 12), (22, 12), (9, 12), (9, 38), (9, 12), (9, 25), (9, 12), (9, 38), (9, 12), (9, 25), (35, 25), (9, 25), (9, 12), (9, 38)], [(35, 25), (22, 38), (35, 25), (22, 38), (22, 51), (22, 38), (35, 25), (22, 38), (22, 51), (22, 25), (22, 51), (22, 38), (35, 25), (35, 12), (35, 38), (48, 51), (22, 51), (22, 38), (35, 25)], [(22, 12), (9, 25), (35, 25), (9, 25), (22, 12), (9, 25), (35, 25), (35, 12), (48, 12), (35, 12), (35, 25), (9, 25), (9, 51), (9, 25), (22, 12), (9, 25), (35, 25), (35, 12), (35, 38), (35, 12), (35, 25), (9, 25), (22, 12)]]


def pos(cfg, i):
    gx, gy = cfg["nodes"][i]
    return OX + gx * PITCH, OY + gy * PITCH


def edge_of(cfg, a, b):
    for k, (u, v, c) in enumerate(cfg["edges"]):
        if (u, v) == (a, b) or (u, v) == (b, a):
            return k, c
    return None


def advance(cfg, state, target):
    tok, load, crates = state
    found = edge_of(cfg, tok, target)
    if found is None or load > found[1]:
        return state, found
    crates = set(crates)
    if target in crates:
        load += cfg["crates"][target]
        crates.discard(target)
    if target == cfg["sink"]:
        load = 0
    return (target, load, frozenset(crates)), found


def solved(state):
    return not state[2] and state[1] == 0


def line(px, x0, y0, x1, y1, colour):
    n = max(abs(x1 - x0), abs(y1 - y0))
    for k in range(n + 1):
        t = k / n if n else 0
        px[int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))] = colour


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


class Ng77(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = None
        self._left = 0
        self._flash = None
        super().__init__(
            game_id="ng77-77f00001",
            levels=[canvas_level(i) for i in range(len(CONFIGS))],
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(CONFIGS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._st = (self._cfg["start"], 0, frozenset(self._cfg["crates"]))
        self._left = min(BAR_MAX, self._cfg["budget"])
        self._flash = None
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        tok, load, crates = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for k, (a, b, c) in enumerate(cfg["edges"]):
            ax, ay = pos(cfg, a)
            bx, by = pos(cfg, b)
            colour = cfg["ecol"][k]
            if self._flash == k:
                colour = TOKEN
            line(px, ax, ay, bx, by, colour)
            for j in range(c):
                t = (j + 1) / (c + 1)
                mx = int(round(ax + (bx - ax) * t))
                my = int(round(ay + (by - ay) * t))
                px[my:my + 2, mx:mx + 2] = PIP
        for i in range(len(cfg["nodes"])):
            x, y = pos(cfg, i)
            px[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1] = SINK if i == cfg["sink"] else BODY
            if i in crates:
                w = cfg["crates"][i]
                for k in range(w):
                    px[y, x - (w - 1) + 2 * k] = CRATE
        x, y = pos(cfg, tok)
        px[y - 1:y + 2, x - 1:x + 2] = TOKEN
        for k in range(load):
            px[y - 5, x - (load - 1) + 2 * k] = CRATE
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _node_at(self, x, y):
        for i in range(len(self._cfg["nodes"])):
            nx, ny = pos(self._cfg, i)
            if abs(x - nx) <= HALF and abs(y - ny) <= HALF:
                return i
        return None

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            target = self._node_at(data.get("x", -99), data.get("y", -99))
            if target is not None and target != self._st[0]:
                self._st, found = advance(self._cfg, self._st, target)
                self._flash = None
                if found is not None and self._st[0] != target:
                    self._flash = found[0]
                self._left = max(0, self._left - 1)
                self._draw()
                if solved(self._st):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()
