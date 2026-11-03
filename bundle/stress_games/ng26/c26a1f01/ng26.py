import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
PLAYER = 8
EXIT = 11
MARK = 10
BAR = 14
WALLS = [9, 12, 14, 15]
PITCH = 7
ORIGIN_X = 4
ORIGIN_Y = 5
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    dict(n=9, F=2, walls=[{(3, 7), (4, 6), (5, 1), (5, 7), (8, 0), (8, 6), (1, 6), (2, 8), (7, 4), (6, 2), (6, 5), (4, 5), (3, 3), (5, 3), (0, 1), (8, 8), (1, 2), (2, 1), (6, 1), (7, 3), (3, 2), (5, 2), (4, 4), (5, 5), (8, 4), (8, 1), (1, 1), (2, 3), (1, 7), (7, 2), (6, 0), (6, 6), (7, 5)}, {(4, 0), (3, 4), (4, 3), (4, 6), (5, 7), (8, 6), (2, 2), (6, 2), (6, 8), (5, 3), (2, 4), (2, 7), (1, 8), (6, 4), (6, 7), (7, 6), (3, 2), (4, 1), (4, 7), (5, 2), (4, 4), (0, 0), (1, 1), (0, 3), (1, 4), (2, 3), (2, 6), (7, 2), (6, 0), (7, 8)}], ex=(8, 8), exf=1, marks={(8, 7), (5, 0), (5, 8)}),
    dict(n=8, F=2, walls=[{(3, 7), (4, 6), (5, 1), (2, 2), (6, 2), (6, 5), (4, 2), (3, 0), (3, 3), (5, 6), (2, 4), (2, 1), (6, 4), (3, 5), (2, 0), (2, 3), (6, 6), (7, 5), (6, 3)}, {(4, 0), (3, 4), (4, 3), (3, 7), (5, 4), (5, 1), (5, 7), (0, 2), (1, 6), (1, 3), (4, 2), (3, 3), (3, 6), (7, 0), (6, 7), (7, 6), (4, 1), (3, 5), (5, 2), (1, 7), (6, 3)}], ex=(7, 7), exf=0, marks={(0, 7), (5, 2), (3, 6)}),
    dict(n=9, F=3, walls=[{(4, 0), (3, 4), (3, 1), (8, 3), (0, 5), (1, 0), (6, 2), (7, 1), (7, 7), (6, 5), (6, 8), (3, 0), (5, 6), (3, 6), (8, 5), (0, 1), (2, 4), (1, 2), (0, 4), (2, 1), (7, 0), (6, 4), (7, 6), (3, 2), (4, 1), (4, 4), (5, 5), (8, 7), (1, 1), (0, 3), (2, 0), (1, 4), (0, 6), (1, 7), (7, 2), (7, 5), (7, 8)}, {(3, 4), (4, 3), (3, 7), (5, 4), (4, 6), (5, 1), (0, 2), (8, 6), (2, 2), (1, 0), (2, 8), (6, 2), (7, 1), (7, 7), (3, 3), (4, 8), (3, 6), (0, 1), (0, 7), (2, 4), (2, 7), (1, 5), (6, 1), (7, 0), (3, 5), (5, 2), (5, 5), (8, 4), (1, 1), (0, 3), (7, 2), (6, 0), (6, 6), (7, 5)}, {(5, 4), (5, 1), (5, 7), (8, 0), (1, 3), (2, 8), (7, 4), (6, 2), (6, 5), (4, 2), (8, 2), (0, 1), (2, 4), (1, 2), (0, 4), (2, 1), (2, 7), (1, 5), (1, 8), (7, 3), (3, 2), (4, 7), (5, 2), (3, 8), (8, 4), (5, 8), (8, 7), (0, 6), (1, 7), (7, 2), (6, 6), (7, 5)}], ex=(8, 8), exf=2, marks={(6, 1), (3, 5), (8, 0)}),
    dict(n=9, F=3, walls=[{(4, 0), (4, 3), (3, 1), (5, 4), (5, 1), (8, 0), (0, 2), (8, 6), (0, 8), (7, 7), (6, 5), (6, 8), (4, 2), (3, 0), (3, 3), (3, 6), (0, 7), (1, 2), (0, 4), (2, 1), (2, 7), (1, 8), (7, 0), (6, 4), (7, 3), (6, 7), (7, 6), (4, 1), (5, 2), (4, 4), (3, 8), (5, 8), (8, 7), (1, 1), (1, 4), (0, 6), (2, 3), (2, 6), (7, 2), (6, 6), (6, 3)}, {(4, 0), (3, 4), (5, 4), (8, 0), (0, 2), (8, 3), (0, 5), (2, 2), (1, 6), (2, 5), (1, 3), (2, 8), (6, 8), (3, 0), (4, 5), (5, 6), (4, 8), (3, 6), (8, 2), (2, 4), (1, 5), (6, 7), (7, 6), (3, 2), (4, 1), (4, 7), (5, 2), (4, 4), (3, 8), (5, 5), (8, 4), (0, 0), (8, 7), (2, 0), (1, 4), (0, 6), (2, 3), (6, 6), (7, 5)}, {(3, 1), (3, 7), (4, 6), (8, 0), (0, 2), (2, 2), (1, 0), (0, 8), (7, 1), (7, 7), (6, 5), (6, 8), (4, 5), (3, 3), (8, 2), (8, 5), (1, 8), (6, 4), (7, 6), (4, 1), (5, 2), (4, 4), (8, 4), (0, 0), (8, 1), (2, 0), (0, 6), (2, 6), (7, 2), (6, 3)}], ex=(8, 8), exf=1, marks={(1, 7), (8, 4), (3, 2)}),
    dict(n=9, F=4, walls=[{(4, 0), (3, 1), (3, 7), (5, 7), (0, 5), (8, 6), (2, 5), (2, 8), (6, 5), (6, 8), (4, 2), (3, 0), (4, 5), (4, 8), (8, 5), (0, 1), (2, 4), (1, 2), (0, 4), (2, 1), (2, 7), (6, 7), (7, 6), (3, 2), (3, 5), (5, 2), (4, 4), (3, 8), (5, 5), (8, 4), (5, 8), (8, 1), (2, 0), (2, 3), (2, 6), (6, 3), (7, 8)}, {(4, 0), (3, 4), (4, 3), (3, 1), (3, 7), (4, 6), (5, 7), (8, 0), (0, 5), (1, 0), (1, 6), (7, 4), (6, 5), (3, 0), (4, 5), (3, 3), (5, 6), (4, 8), (1, 2), (0, 4), (2, 7), (6, 1), (1, 8), (3, 2), (3, 5), (4, 4), (0, 0), (5, 8), (0, 3), (2, 0), (1, 4), (2, 3), (6, 0), (6, 3), (7, 8)}, {(5, 7), (8, 0), (8, 3), (0, 5), (2, 2), (1, 0), (0, 8), (1, 3), (7, 4), (6, 2), (7, 7), (4, 2), (3, 0), (4, 5), (3, 3), (3, 6), (0, 1), (2, 4), (2, 1), (1, 5), (6, 1), (6, 4), (7, 3), (6, 7), (3, 5), (3, 8), (8, 4), (5, 8), (8, 1), (8, 7), (0, 3), (0, 6), (2, 3), (1, 7), (2, 6), (7, 2), (7, 5), (6, 3)}, {(4, 3), (5, 1), (8, 0), (0, 5), (1, 6), (0, 8), (2, 8), (7, 7), (6, 5), (6, 8), (3, 0), (3, 3), (4, 8), (8, 2), (2, 4), (1, 2), (0, 4), (2, 1), (2, 7), (1, 5), (6, 1), (1, 8), (7, 0), (6, 4), (7, 6), (4, 1), (3, 5), (5, 2), (4, 4), (3, 8), (5, 5), (5, 8), (8, 7), (1, 1), (2, 0), (7, 8)}], ex=(8, 8), exf=3, marks={(7, 7), (1, 4), (4, 3)}),
]

SOLUTIONS = ['RRRRRSDRRRDDDDLLLLLDDDRRSRURRSD', 'DRSRRSRURRDRDDDSDLDLLDSRRR', 'SSRRRRRSDDDSSLDDDSDDRSRRRS', 'DSRDSDDDDDRDRRURUUUUSSRDRSDSRSDSDDSS', 'RDSLDSSDRDSDSRSSDRRRRSSDDSRRS']


def budget_of(index):
    return min(BAR_MAX, 2 * len(SOLUTIONS[index]) + 10)


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


class Ng26(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = (0, 0)
        self._floor = 0
        self._left = 0
        super().__init__(
            game_id="ng26-c26a1f01",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = (0, 0)
        self._floor = 0
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _open(self, floor, cell):
        n = self._cfg["n"]
        x, y = cell
        return 0 <= x < n and 0 <= y < n and (x, y) not in self._cfg["walls"][floor]

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        n = cfg["n"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for f in range(cfg["F"]):
            colour = WALLS[f]
            size = 4 if f == self._floor else 2
            px[3 - (size == 4):3 - (size == 4) + size, 56 - 6 * f:56 - 6 * f + size] = colour
        wall = WALLS[self._floor]
        for y in range(n):
            for x in range(n):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if (x, y) in cfg["walls"][self._floor]:
                    px[oy:oy + 6, ox:ox + 6] = wall
                else:
                    px[oy:oy + 6, ox:ox + 6] = FLOOR
                    if (x, y) in cfg["marks"]:
                        px[oy + 2:oy + 4, ox + 2:ox + 4] = MARK
        if self._floor == cfg["exf"]:
            ex, ey = cfg["ex"]
            ox, oy = ORIGIN_X + ex * PITCH, ORIGIN_Y + ey * PITCH
            px[oy:oy + 6, ox:ox + 6] = EXIT
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 5, ox + 1:ox + 5] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        delta = MOVES.get(aid)
        acted = False
        if delta is not None:
            acted = True
            dest = (self._pos[0] + delta[0], self._pos[1] + delta[1])
            if self._open(self._floor, dest):
                self._pos = dest
        elif aid == GameAction.ACTION5:
            acted = True
            nf = (self._floor + 1) % self._cfg["F"]
            if self._open(nf, self._pos):
                self._floor = nf
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if self._floor == self._cfg["exf"] and self._pos == tuple(self._cfg["ex"]):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
