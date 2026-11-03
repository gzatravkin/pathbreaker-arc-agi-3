import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
RIM = 3
RIM_LOCKED = 8
HAND = 15
TIP = 10
HUB = 4
TARGET = 11
CW = 14
CCW = 12
BAR = 10
BAR_MAX = 60
BOX = 13
COLS = (3, 24, 45)
ROWS = (8, 36)

DIRS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

CONFIGS = [{'speeds': [1, 1, 1, 1], 'starts': [0, 7, 6, 6], 'targets': [6, 1, 3, 4], 'len': 10}, {'speeds': [3, 3, 1, 3], 'starts': [0, 2, 7, 5], 'targets': [2, 5, 3, 0], 'len': 8}, {'speeds': [3, 3, -1, 3], 'starts': [4, 6, 2, 7], 'targets': [3, 3, 3, 5], 'len': 10}, {'speeds': [-2, 2, 1, 2, -3], 'starts': [6, 7, 2, 1, 2], 'targets': [2, 1, 4, 3, 5], 'len': 11}, {'speeds': [3, 1, -2, -2, -1], 'starts': [4, 3, 5, 0, 3], 'targets': [5, 7, 7, 2, 6], 'len': 9}]

SOLUTIONS = [[(9, 14), (9, 14), (30, 14), (9, 14), (9, 14), (51, 14), (9, 42), (9, 14), (9, 14), (9, 14)], [(30, 14), (9, 42), (9, 14), (9, 14), (51, 14), (30, 14), (30, 14), (9, 14)], [(9, 14), (9, 14), (9, 42), (9, 14), (30, 14), (9, 14), (30, 14), (51, 14), (9, 14), (30, 14)], [(9, 14), (9, 14), (9, 14), (9, 14), (9, 14), (30, 14), (30, 42), (30, 42), (30, 42), (9, 42), (51, 14)], [(9, 14), (9, 14), (9, 14), (51, 14), (30, 14), (30, 42), (9, 14), (9, 42), (9, 14)]]


def box_of(i):
    return COLS[i % 3], ROWS[i // 3]


def dial_at(cfg, x, y):
    for i in range(len(cfg["speeds"])):
        bx, by = box_of(i)
        if bx <= x < bx + BOX and by <= y < by + BOX:
            return i
    return None


def centre_of(i):
    bx, by = box_of(i)
    return bx + BOX // 2, by + BOX // 2


def speed_of(cfg, i, rule):
    s = cfg["speeds"][i]
    if rule == "allone":
        return 1 if s > 0 else -1
    if rule == "cw":
        return abs(s)
    return s


def click(cfg, pos, locked, i, rule="true"):
    locked = list(locked)
    pos = list(pos)
    locked[i] = not locked[i]
    for j in range(len(pos)):
        if not locked[j]:
            pos[j] = (pos[j] + speed_of(cfg, j, rule)) % 8
    return tuple(pos), tuple(locked)


def solved(cfg, pos, locked):
    return all(locked) and tuple(pos) == tuple(cfg["targets"])


def budget_of(cfg):
    return min(BAR_MAX, 2 * cfg["len"] + 8)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng37(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = ()
        self._locked = ()
        self._left = 0
        super().__init__(
            game_id="ng37-3a370001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = tuple(self._cfg["starts"])
        self._locked = tuple(False for _ in self._cfg["speeds"])
        self._left = budget_of(self._cfg)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for i, s in enumerate(cfg["speeds"]):
            bx, by = box_of(i)
            cx, cy = centre_of(i)
            rim = RIM_LOCKED if self._locked[i] else RIM
            px[by, bx:bx + BOX] = rim
            px[by + BOX - 1, bx:bx + BOX] = rim
            px[by:by + BOX, bx] = rim
            px[by:by + BOX, bx + BOX - 1] = rim
            tx, ty = DIRS[cfg["targets"][i]]
            px[cy + 5 * ty, cx + 5 * tx] = TARGET
            hx, hy = DIRS[self._pos[i]]
            for r in range(1, 5):
                px[cy + r * hy, cx + r * hx] = TIP if r == 4 else HAND
            px[cy, cx] = HUB
            for k in range(abs(s)):
                px[by + BOX + 1, bx + 2 * k] = CW if s > 0 else CCW
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            i = dial_at(self._cfg, int(data.get("x", -1)), int(data.get("y", -1)))
            if i is not None:
                self._pos, self._locked = click(self._cfg, self._pos, self._locked, i)
                self._left = max(0, self._left - 1)
                self._draw()
                if solved(self._cfg, self._pos, self._locked):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()
