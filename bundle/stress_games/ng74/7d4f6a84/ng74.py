import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
SOIL = 13
PIP = 11
SEED = 10
STEM = 14
BLOOM = 6
BAR = 3
BAR_ON = 11
PITCH = 10
X0 = 2
SOIL_Y = 30
HEIGHT = 8

LEVELS_DEF = [
    dict(pips=[1, 2, 3], window=4, boost=0, budget=16),
    dict(pips=[3, 1, 2], window=4, boost=0, budget=16),
    dict(pips=[2, 4, 1, 3, 5], window=5, boost=0, budget=24),
    dict(pips=[2, 3, 1, 2], window=3, boost=2, budget=24),
    dict(pips=[3, 1, 2, 3, 1, 2], window=3, boost=2, budget=30),
]


class Model:
    def __init__(self, level):
        d = LEVELS_DEF[level]
        self.g = [2 * p for p in d["pips"]]
        self.window = d["window"]
        self.boost = d["boost"]
        self.n = len(self.g)
        self.cells = [None] * self.n

    def click(self, i):
        if self.cells[i] is None:
            self.cells[i] = [0, 0]
            return i
        self.cells[i] = None
        return None

    def tick(self, fresh=None):
        blooming = []
        for i, c in enumerate(self.cells):
            if c is None or i == fresh:
                continue
            if c[0] < self.g[i]:
                c[0] += 1
                if c[0] == self.g[i]:
                    blooming.append(i)
            else:
                c[1] += 1
                if c[1] >= self.window:
                    self.cells[i] = None
        while blooming and self.boost:
            i = blooming.pop(0)
            for j in (i - 1, i + 1):
                if 0 <= j < self.n and self.cells[j] is not None and self.cells[j][0] < self.g[j]:
                    self.cells[j][0] = min(self.g[j], self.cells[j][0] + self.boost)
                    if self.cells[j][0] == self.g[j]:
                        blooming.append(j)

    def done(self):
        return all(c is not None and c[0] >= self.g[i] for i, c in enumerate(self.cells))

    def key(self):
        return tuple(None if c is None else tuple(c) for c in self.cells)


def plot_x(i):
    return X0 + i * PITCH


def hit(x, y, n):
    for i in range(n):
        px = plot_x(i)
        if px <= x < px + 8 and SOIL_Y <= y < SOIL_Y + 8:
            return i
    return None


def plot_click(i):
    return (plot_x(i) + 4, SOIL_Y + 4)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVELS_DEF))]


class Ng74(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._m = Model(0)
        self._left = 0
        self._pips = []
        super().__init__(
            game_id="ng74-7d4f6a84",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = self._current_level_index
        self._m = Model(idx)
        self._pips = LEVELS_DEF[idx]["pips"]
        self._left = LEVELS_DEF[idx]["budget"]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        m = self._m
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR_ON
        for i in range(m.n):
            x = plot_x(i)
            px[SOIL_Y:SOIL_Y + 8, x:x + 8] = SOIL
            for k in range(self._pips[i]):
                px[SOIL_Y + 9, x + 2 * k] = PIP
            c = m.cells[i]
            if c is None:
                continue
            if c[0] == 0:
                px[SOIL_Y + 3:SOIL_Y + 5, x + 3:x + 5] = SEED
            else:
                h = min(HEIGHT, c[0])
                px[SOIL_Y - h:SOIL_Y, x + 3:x + 5] = STEM
                if c[0] >= m.g[i]:
                    px[SOIL_Y - h - 4:SOIL_Y - h, x + 1:x + 7] = BLOOM
                    px[SOIL_Y + 11, x:x + 8 - min(8, c[1] * 8 // m.window)] = BAR_ON
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            i = hit(int(data.get("x", -1)), int(data.get("y", -1)), self._m.n)
            fresh = None
            if i is not None:
                fresh = self._m.click(i)
            self._m.tick(fresh)
            self._left = max(0, self._left - 1)
            self._draw()
            if self._m.done():
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()


PLANS = [
    (2, 0, 1, 0, 0, None, None),
    (0, 1, 2, 1, 1, None, None),
    (4, 0, 1, 0, 3, 0, 2, 2, 2, None, None),
    (0, 1, 2, 3, 2, 2),
    (0, 2, 3, 1, 5, 4, None),
]
SOLUTIONS = [[(63, 63) if i is None else plot_click(i) for i in p] for p in PLANS]
