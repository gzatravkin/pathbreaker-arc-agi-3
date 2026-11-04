import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
CELL_OFF = 3
CELL_ON = 11
PEN = 8
BAR = 14
FRAME = 2
ORIGIN_X = 2
ORIGIN_Y = 6
PITCH = 6
SIDE = 5
MINI_X = 53
MINI_Y = 6
BAR_Y = 1
BAR_X = 2
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

CONFIGS = [
    dict(n=5, holes=(), starts=((0, 0),), wide=False, path="RRRRDDLLL"),
    dict(n=5, holes=(), starts=((1, 1),), wide=False, path="RRDDLLUUU"),
    dict(n=6, holes=((2, 2), (3, 3)), starts=((0, 5),), wide=False, path="UUUURRRRDDDLL"),
    dict(n=6, holes=(), starts=((0, 0),), wide=True, path="DDRRRUUR"),
    dict(n=6, holes=(), starts=((0, 0), (3, 2)), wide=False, path="RRRDDDLL"),
    dict(n=7, holes=((3, 1), (3, 5), (6, 3)), starts=((0, 0), (4, 4)), wide=False, path="RRRDDDRRUULL"),
]


def footprint(cfg, cell):
    x, y = cell
    cells = [(x, y)]
    if cfg["wide"]:
        cells += [(x - 1, y), (x + 1, y)]
    return [c for c in cells if inside(cfg, c)]


def inside(cfg, cell):
    x, y = cell
    n = cfg["n"]
    return 0 <= x < n and 0 <= y < n and (x, y) not in cfg["holes"]


def advance(cfg, pens, lit, delta):
    dx, dy = delta
    moved = []
    for pen in pens:
        dest = (pen[0] + dx, pen[1] + dy)
        if inside(cfg, dest):
            for c in footprint(cfg, pen):
                lit ^= {c}
            moved.append(dest)
        else:
            moved.append(pen)
    return moved, lit


def target_of(cfg):
    pens = list(cfg["starts"])
    lit = set()
    for letter in cfg["path"]:
        pens, lit = advance(cfg, pens, lit, LETTERS[letter])
    return frozenset(lit)


def budget_of(cfg):
    return min(BAR_MAX, 2 * len(cfg["path"]) + 8)


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


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng01(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pens = []
        self._lit = set()
        self._target = frozenset()
        self._left = 0
        super().__init__(
            game_id="ng01-0a1b0001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pens = list(self._cfg["starts"])
        self._lit = set()
        self._target = target_of(self._cfg)
        self._left = budget_of(self._cfg)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        n = cfg["n"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[BAR_Y, BAR_X:BAR_X + self._left] = BAR
        for y in range(n):
            for x in range(n):
                if not inside(cfg, (x, y)):
                    continue
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                px[oy:oy + SIDE, ox:ox + SIDE] = CELL_ON if (x, y) in self._lit else CELL_OFF
        for x, y in self._pens:
            ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = PEN
            if cfg["wide"]:
                px[oy + 2, max(0, ox - 3):min(64, ox + 8)] = PEN
        px[MINI_Y:MINI_Y + n + 2, MINI_X:MINI_X + n + 2] = FRAME
        for y in range(n):
            for x in range(n):
                colour = BACKGROUND
                if inside(cfg, (x, y)):
                    colour = CELL_ON if (x, y) in self._target else CELL_OFF
                px[MINI_Y + 1 + y, MINI_X + 1 + x] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            self._pens, self._lit = advance(self._cfg, self._pens, self._lit, delta)
            self._left = max(0, self._left - 1)
            self._draw()
            if frozenset(self._lit) == self._target:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
