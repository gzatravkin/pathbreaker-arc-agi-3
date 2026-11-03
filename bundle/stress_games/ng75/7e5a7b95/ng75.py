import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FRAME = 2
WATER = 9
DROP = 10
MARK = 8
PIPE = 3
CURSOR = 11
BAR = 14
PITCH = 12
X0 = 3
BOTTOM = 40
UNIT = 3

LEVELS_DEF = [
    dict(caps=[6, 6, 6], init=[4, 0, 0], dest=[1, 2, -1], target=[0, 0, 4], budget=20),
    dict(caps=[6, 6, 6, 6], init=[5, 0, 0, 0], dest=[2, -1, 3, -1], target=[0, 0, 0, 5], budget=24),
    dict(caps=[8, 3, 8, 8], init=[7, 0, 0, 0], dest=[1, 2, 3, -1], target=[0, 0, 3, 0], budget=24),
    dict(caps=[5, 5, 6, 9], init=[4, 5, 0, 0], dest=[2, 2, 3, -1], target=[0, 0, 0, 6], budget=26),
    dict(caps=[9, 5, 6, 3, 9], init=[9, 0, 0, 0, 0], dest=[2, 2, 3, 4, -1], target=[0, 0, 0, 0, 3], budget=40),
]


class Model:
    def __init__(self, level):
        d = LEVELS_DEF[level]
        self.caps = list(d["caps"])
        self.level = list(d["init"])
        self.dest = list(d["dest"])
        self.target = list(d["target"])
        self.n = len(self.caps)
        self.cursor = 0
        self.transit = []

    def delay(self, i):
        j = self.dest[i]
        return 1 if j < 0 else max(1, abs(j - i))

    def tick(self):
        keep = []
        for amount, src, rem in self.transit:
            rem -= 1
            if rem <= 0:
                j = self.dest[src]
                if j >= 0:
                    self.level[j] = min(self.caps[j], self.level[j] + amount)
            else:
                keep.append((amount, src, rem))
        self.transit = keep

    def pour(self):
        i = self.cursor
        if self.level[i] > 0:
            self.transit.append((self.level[i], i, self.delay(i)))
            self.level[i] = 0

    def move(self, dx):
        self.cursor = max(0, min(self.n - 1, self.cursor + dx))

    def done(self):
        return not self.transit and self.level == self.target

    def key(self):
        return (self.cursor, tuple(self.level), tuple(sorted(self.transit)))


def tank_x(i):
    return X0 + i * PITCH


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVELS_DEF))]


class Ng75(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._m = Model(0)
        self._left = 0
        super().__init__(
            game_id="ng75-7e5a7b95",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = self._current_level_index
        self._m = Model(idx)
        self._left = LEVELS_DEF[idx]["budget"]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        m = self._m
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for i in range(m.n):
            x = tank_x(i)
            h = m.caps[i] * UNIT
            px[BOTTOM - h:BOTTOM, x:x + 8] = FRAME
            px[BOTTOM - h:BOTTOM - 1, x + 1:x + 7] = BACKGROUND
            wl = m.level[i] * UNIT
            if wl:
                px[BOTTOM - 1 - wl:BOTTOM - 1, x + 1:x + 7] = WATER
            ty = BOTTOM - 1 - m.target[i] * UNIT
            px[ty, x - 2:x + 1] = MARK
            px[ty, x + 7:x + 10] = MARK
            j = m.dest[i]
            py = BOTTOM + 3 + (i % 3) * 3
            px[BOTTOM:py + 1, x + 4] = PIPE
            if j >= 0:
                xj = tank_x(j) + 4
                lo, hi = min(x + 4, xj), max(x + 4, xj)
                px[py, lo:hi + 1] = PIPE
                px[BOTTOM:py + 1, xj] = PIPE
            else:
                px[py:py + 3, x + 3:x + 6] = PIPE
        for amount, src, rem in m.transit:
            j = m.dest[src]
            x0 = tank_x(src) + 4
            x1 = tank_x(j) + 4 if j >= 0 else x0
            d = m.delay(src)
            frac = (d - rem) / d
            cx = int(round(x0 + (x1 - x0) * frac))
            py = BOTTOM + 3 + (src % 3) * 3
            px[py - 1:py + 1, max(0, cx - 1):cx + 1] = DROP
            px[py + 2:py + 2 + amount, max(0, cx - 1):cx + 1] = DROP
        cx = tank_x(m.cursor)
        px[BOTTOM + 14:BOTTOM + 16, cx:cx + 8] = CURSOR
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        if aid in (GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5):
            if aid == GameAction.ACTION3:
                self._m.move(-1)
            elif aid == GameAction.ACTION4:
                self._m.move(1)
            self._m.tick()
            if aid == GameAction.ACTION5:
                self._m.pour()
            self._left = max(0, self._left - 1)
            self._draw()
            if self._m.done():
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()


SOLUTIONS = [list(p) for p in [
    "SRSL",
    "SRRSL",
    "SRSL",
    "SRSRSL",
    "SRRSRSL",
]]
