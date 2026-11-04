import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
OFF = 3
ON = 11
EDGE = 2
CURSOR = 8
FRAME = 14
BAR = 9
BAR_MAX = 60
OX = 8
OY = 14
PITCH = 14

DIRS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    {'nodes': [(2, 0), (3, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (3, 2)], 'edges': [(0, 1), (0, 3), (2, 3), (2, 5), (3, 6), (4, 5), (5, 6), (6, 7)], 'hollow': [], 'target': [0, 1, 2, 3, 4, 5, 6, 7], 'start': [1, 5, 7], 'cursor': 0, 'budget': 34},
    {'nodes': [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 2), (2, 2), (0, 3)], 'edges': [(0, 1), (1, 2), (2, 3), (2, 4), (3, 6), (3, 7), (5, 6)], 'hollow': [], 'target': [2, 5, 7], 'start': [0, 2, 5, 6, 7], 'cursor': 5, 'budget': 36},
    {'nodes': [(1, 0), (1, 1), (3, 1), (1, 2), (2, 2), (3, 2), (0, 3), (2, 3), (3, 3)], 'edges': [(0, 1), (1, 3), (2, 5), (3, 4), (4, 5), (4, 6), (4, 7), (5, 8), (7, 8)], 'hollow': [], 'target': [0, 1, 2, 6], 'start': [0, 3, 4, 5, 6, 7, 8], 'cursor': 3, 'budget': 42},
    {'nodes': [(2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (2, 2), (1, 3), (2, 3)], 'edges': [(0, 3), (1, 2), (1, 5), (2, 3), (3, 4), (3, 6), (4, 5), (4, 7), (6, 8), (7, 8)], 'hollow': [2, 4], 'target': [2, 3, 6, 7], 'start': [4, 8], 'cursor': 6, 'budget': 38},
    {'nodes': [(0, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (0, 3), (2, 3), (3, 3)], 'edges': [(0, 9), (1, 3), (1, 7), (2, 3), (2, 4), (2, 9), (3, 5), (4, 5), (5, 6), (5, 8), (6, 9)], 'hollow': [0, 4, 5], 'target': [0, 3, 5, 6, 8], 'start': [1, 3], 'cursor': 2, 'budget': 48},
]

SOLUTIONS = ['SLLSRRSRSUSRS', 'USRSRSRSDSLSDS', 'UUSDSRSRSDSLSLSDS', 'SUSRSUSLSDSDSRS', 'USDRSUSLSDDSUSRSRSDS']


def pos(cfg, i):
    gx, gy = cfg["nodes"][i]
    return OX + gx * PITCH, OY + gy * PITCH


def neighbours(cfg, i):
    out = []
    for a, b in cfg["edges"]:
        if a == i:
            out.append(b)
        elif b == i:
            out.append(a)
    return out


def press(cfg, lit, i):
    lit = set(lit)
    flips = list(neighbours(cfg, i))
    if i not in cfg["hollow"]:
        flips.append(i)
    for n in flips:
        lit ^= {n}
    return frozenset(lit)


def nav(cfg, i, d):
    x, y = cfg["nodes"][i]
    dx, dy = d
    best = None
    for j, (nx, ny) in enumerate(cfg["nodes"]):
        if j == i:
            continue
        along = (nx - x) * dx + (ny - y) * dy
        across = abs((nx - x) * dy) + abs((ny - y) * dx)
        if along <= 0 or across > along:
            continue
        score = along * 4 + across
        if best is None or score < best[0]:
            best = (score, j)
    return i if best is None else best[1]


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


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng76(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._lit = frozenset()
        self._cur = 0
        self._left = 0
        super().__init__(
            game_id="ng76-76f00001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._lit = frozenset(self._cfg["start"])
        self._cur = self._cfg["cursor"]
        self._left = min(BAR_MAX, self._cfg["budget"])
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for a, b in cfg["edges"]:
            ax, ay = pos(cfg, a)
            bx, by = pos(cfg, b)
            line(px, ax, ay, bx, by, EDGE)
        for i in range(len(cfg["nodes"])):
            x, y = pos(cfg, i)
            if i in cfg["target"]:
                px[y - 4:y + 5, x - 4:x + 5] = FRAME
            colour = ON if i in self._lit else OFF
            px[y - 3:y + 4, x - 3:x + 4] = colour
            if i in cfg["hollow"]:
                px[y - 1:y + 2, x - 1:x + 2] = BACKGROUND
        x, y = pos(cfg, self._cur)
        for cx, cy in ((x - 6, y - 6), (x + 6, y - 6), (x - 6, y + 6), (x + 6, y + 6)):
            px[cy, cx] = CURSOR
            px[cy, cx + (1 if cx < x else -1)] = CURSOR
            px[cy + (1 if cy < y else -1), cx] = CURSOR
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _won(self):
        return self._lit == frozenset(self._cfg["target"])

    def step(self) -> None:
        aid = self._action.id
        acted = False
        if aid in DIRS:
            self._cur = nav(self._cfg, self._cur, DIRS[aid])
            acted = True
        elif aid == GameAction.ACTION5:
            self._lit = press(self._cfg, self._lit, self._cur)
            acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if self._won():
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
