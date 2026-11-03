import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
BACK = 3
FACE = 10
PIP = 5
LAMP = 11
BAR = 14
PITCH = 13
X0 = 6
Y0 = 8
BAR_MAX = 60
FLASH = 6

LEVELS_DEF = [{'values': [1, 2, 3, 4, 4, 3, 2, 1], 'sizes': [9, 9, 10, 10, 10, 10, 9, 9], 'cols': 4, 'start': 5, 'step': 0}, {'values': [5, 5, 3, 1, 1, 3, 3, 3, 2, 2, 4, 4], 'sizes': [10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 9, 10], 'cols': 4, 'start': 6, 'step': 0}, {'values': [3, 1, 6, 5, 2, 4, 2, 2, 4, 3, 3, 4], 'sizes': [10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 9, 10], 'cols': 4, 'start': 4, 'step': 1}, {'values': [5, 6, 3, 6, 4, 2, 6, 1, 1, 3, 1, 4], 'sizes': [10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 9, 10], 'cols': 4, 'start': 2, 'step': 2}, {'values': [2, 2, 1, 1, 3, 2, 1, 1, 2, 5, 5, 4, 3, 3, 5, 4], 'sizes': [10, 10, 9, 9, 9, 9, 9, 9, 10, 10, 9, 10, 9, 10, 9, 10], 'cols': 4, 'start': 2, 'step': 1}]
SOLUTIONS = [[(8, 10), (47, 10), (21, 10), (34, 10), (8, 23), (47, 23), (21, 23), (34, 23)], [(8, 10), (47, 10), (21, 10), (8, 23), (34, 10), (21, 23), (34, 23), (47, 23), (8, 36), (34, 36), (21, 36), (47, 36)], [(8, 10), (21, 10), (8, 23), (21, 36), (21, 23), (34, 23), (47, 10), (47, 23), (8, 36), (47, 36), (34, 10), (34, 36)], [(47, 23), (8, 36), (34, 10), (34, 36), (8, 23), (21, 23), (8, 10), (21, 36), (21, 10), (47, 36), (47, 10), (34, 23)], [(34, 10), (47, 10), (8, 10), (34, 23), (21, 10), (21, 23), (8, 23), (8, 36), (47, 23), (21, 36), (47, 36), (8, 49), (34, 36), (21, 49), (34, 49), (47, 49)]]
BUDGETS = [24, 32, 32, 32, 40]


def pip_cells(count, s):
    lo, hi, mid = 2, s - 3, s // 2
    cells = {
        1: [(mid, mid)],
        2: [(lo, lo), (hi, hi)],
        3: [(lo, lo), (hi, hi), (mid, mid)],
        4: [(lo, lo), (hi, hi), (hi, lo), (lo, hi)],
        5: [(lo, lo), (hi, hi), (hi, lo), (lo, hi), (mid, mid)],
        6: [(lo, lo), (hi, hi), (hi, lo), (lo, hi), (lo, mid), (hi, mid)],
    }
    return cells[count]


def card_rect(index, cols, size):
    r, c = divmod(index, cols)
    return X0 + c * PITCH, Y0 + r * PITCH, size, size


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


class Ng84(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cards = []
        self._cols = 4
        self._gone = set()
        self._first = None
        self._pair = None
        self._flash = 0
        self._target = 0
        self._step_up = 0
        self._left = 0
        self._total = 1
        super().__init__(
            game_id="ng84-8a840001",
            levels=[canvas_level(i) for i in range(len(LEVELS_DEF))],
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS_DEF),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        cfg = LEVELS_DEF[index]
        self._cards = list(zip(cfg["values"], cfg["sizes"]))
        self._cols = cfg["cols"]
        self._target = cfg["start"]
        self._step_up = cfg["step"]
        self._gone = set()
        self._first = None
        self._pair = None
        self._flash = 0
        self._total = BUDGETS[index]
        self._left = self._total
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        bar = (self._left * BAR_MAX + self._total - 1) // self._total
        px[62:64, 2:2 + bar] = BAR
        for i in range(self._target):
            px[1:3, 2 + i * 4:4 + i * 4] = LAMP
        up = set()
        if self._first is not None:
            up.add(self._first)
        if self._pair is not None:
            up.update(self._pair)
        for i, (v, s) in enumerate(self._cards):
            if i in self._gone:
                continue
            x, y, w, h = card_rect(i, self._cols, s)
            if i in up:
                px[y:y + h, x:x + w] = FACE
                for cx, cy in pip_cells(v, s):
                    px[y + cy, x + cx] = PIP
            else:
                px[y:y + h, x:x + w] = BACK
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _hit(self, cx, cy):
        for i, (v, s) in enumerate(self._cards):
            if i in self._gone:
                continue
            x, y, w, h = card_rect(i, self._cols, s)
            if x <= cx < x + w and y <= cy < y + h:
                return i
        return None

    def step(self) -> None:
        if self._flash > 0:
            self._flash -= 1
            if self._flash == 0:
                self._pair = None
                self._first = None
                self._draw()
                if self._left == 0:
                    self.lose()
                self.complete_action()
            return
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            idx = self._hit(data.get("x", -1), data.get("y", -1))
            if idx is not None and idx != self._first:
                if self._first is None:
                    self._first = idx
                    self._draw()
                else:
                    a, b = self._first, idx
                    self._left = max(0, self._left - 1)
                    if self._cards[a][0] + self._cards[b][0] == self._target:
                        self._gone.update((a, b))
                        self._first = None
                        self._target += self._step_up
                        self._draw()
                        if len(self._gone) == len(self._cards):
                            self.next_level()
                        elif self._left == 0:
                            self.lose()
                    else:
                        self._pair = (a, b)
                        self._first = None
                        self._flash = FLASH
                        self._draw()
                        return
        self.complete_action()
