import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
TILE = 10
TARGET_BOX = 3
PIP = 5
BAR = 14
ROW_Y = 30
TARGET_Y = 8
X0 = 4
BAR_MAX = 60

CONFIGS = [{'pips': [2, 4, 4, 2, 4, 3, 4, 1], 'sizes': [5, 6, 6, 5, 6, 6, 6, 5], 'target': [4, 1, 3, 4, 2, 4, 2, 4]}, {'pips': [3, 4, 1, 1, 3, 4, 1, 2], 'sizes': [6, 6, 5, 6, 5, 6, 6, 5], 'target': [1, 1, 3, 3, 2, 1, 4, 4]}, {'pips': [4, 2, 4, 2, 4, 2, 3, 1], 'sizes': [6, 6, 6, 5, 6, 6, 5, 5], 'target': [2, 4, 2, 4, 2, 4, 1, 3]}, {'pips': [4, 3, 1, 1, 3, 4, 1, 5], 'sizes': [5, 6, 6, 5, 6, 6, 6, 6], 'target': [1, 1, 4, 3, 1, 5, 4, 3]}, {'pips': [2, 5, 3, 4, 3, 5, 2, 3], 'sizes': [6, 5, 6, 6, 5, 6, 5, 5], 'target': [4, 3, 5, 3, 5, 2, 3, 2]}]
SOLUTIONS = [[(27, 30), (27, 30), (34, 30), (34, 30), (34, 30), (34, 30), (13, 30), (55, 30)], [(6, 30), (6, 30), (20, 30), (27, 30), (34, 30), (48, 30), (48, 30), (34, 30), (55, 30)], [(20, 30), (20, 30), (27, 30), (27, 30), (41, 30), (48, 30), (13, 30), (34, 30), (48, 30), (41, 30)], [(34, 30), (13, 30), (27, 30), (55, 30), (20, 30), (55, 30), (6, 30), (6, 30), (48, 30), (55, 30)], [(6, 30), (6, 30), (6, 30), (13, 30), (13, 30), (41, 30), (41, 30), (27, 30), (27, 30), (6, 30), (34, 30)]]
BUDGETS = [24, 26, 28, 28, 30]


def pitch_of(n):
    return 56 // n


def pip_cells(count, s):
    lo, hi, mid = 1, s - 2, s // 2
    corners = [(lo, lo), (hi, hi), (hi, lo), (lo, hi)]
    if count == 1:
        return [(mid, mid)]
    if count == 2:
        return corners[:2]
    if count == 3:
        return corners[:2] + [(mid, mid)]
    if count == 4:
        return corners
    return corners + [(mid, mid)]


def tile_rect(n, slot, s):
    pitch = pitch_of(n)
    return X0 + slot * pitch, ROW_Y - s // 2, s, s


def jump(order, slot):
    n = len(order)
    p = order[slot][0]
    other = (slot + p) % n
    order = list(order)
    order[slot], order[other] = order[other], order[slot]
    return order


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


class Ng82(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._order = []
        self._target = []
        self._left = 0
        self._total = 1
        super().__init__(
            game_id="ng82-8a820001",
            levels=[canvas_level(i) for i in range(len(CONFIGS))],
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(CONFIGS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        cfg = CONFIGS[index]
        self._order = [(p, s) for p, s in zip(cfg["pips"], cfg["sizes"])]
        self._target = list(cfg["target"])
        self._total = BUDGETS[index]
        self._left = self._total
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        n = len(self._order)
        pitch = pitch_of(n)
        bar = (self._left * BAR_MAX + self._total - 1) // self._total
        px[1:3, 2:2 + bar] = BAR
        for slot, (p, s) in enumerate(self._order):
            x, y, w, h = tile_rect(n, slot, s)
            px[y:y + h, x:x + w] = TILE
            for cx, cy in pip_cells(p, s):
                px[y + cy, x + cx] = PIP
        for slot, p in enumerate(self._target):
            x = X0 + slot * pitch
            px[TARGET_Y:TARGET_Y + 5, x:x + 5] = TARGET_BOX
            for cx, cy in pip_cells(p, 5):
                px[TARGET_Y + cy, x + cx] = PIP
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            cx, cy = data.get("x", -1), data.get("y", -1)
            n = len(self._order)
            for slot, (p, s) in enumerate(self._order):
                x, y, w, h = tile_rect(n, slot, s)
                if x <= cx < x + w and y <= cy < y + h:
                    self._order = jump(self._order, slot)
                    self._left = max(0, self._left - 1)
                    self._draw()
                    if [t[0] for t in self._order] == self._target:
                        self.next_level()
                    elif self._left == 0:
                        self.lose()
                    break
        self.complete_action()
