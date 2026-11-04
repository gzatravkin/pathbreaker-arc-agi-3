import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 15
PLAYER = 8
CRATE = 12
SOCKET = 11
BAR = 14
ARROW = 10
PITCH = 7
ORIGIN_X = 4
ORIGIN_Y = 5
BAR_MAX = 60
GRAVITY = [(0, 1), (-1, 0), (0, -1), (1, 0)]
BASE = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
])

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    {'n': 6, 'walls': {(0, 3), (5, 1), (2, 3), (0, 2), (5, 0), (1, 3)}, 'player': (3, 1), 'crates': [(0, 5), (3, 0)], 'targets': {(5, 4), (5, 5)}},
    {'n': 7, 'walls': {(5, 3), (6, 6), (3, 4), (6, 5)}, 'player': (0, 0), 'crates': [(0, 6), (3, 3), (5, 2)], 'targets': {(6, 1), (6, 4), (3, 0)}},
    {'n': 7, 'walls': {(4, 0), (2, 1), (1, 5), (0, 3), (1, 4), (3, 3), (1, 0)}, 'player': (6, 2), 'crates': [(1, 3), (5, 6), (6, 6)], 'targets': {(1, 1), (0, 4), (0, 5)}},
    {'n': 7, 'walls': {(3, 4), (4, 3), (3, 1), (1, 1), (0, 3), (6, 4), (4, 2), (3, 6), (3, 2), (2, 5), (1, 3)}, 'player': (1, 4), 'crates': [(5, 3), (5, 4), (5, 5), (5, 6)], 'targets': {(4, 4), (4, 0), (5, 0), (6, 0)}},
    {'n': 8, 'walls': {(7, 4), (0, 4), (3, 4), (5, 1), (1, 4), (5, 7), (0, 5), (3, 6), (1, 0), (3, 2), (4, 1)}, 'player': (6, 7), 'crates': [(1, 2), (1, 3), (2, 7), (7, 7)], 'targets': {(7, 5), (2, 6), (4, 7), (0, 0)}},
]

SOLUTIONS = ['LSUSSDSLSSSS', 'DDDSSRRRRSSSSUUUS', 'DDDLLLLSSSDLLSRRUSS', 'RUUUURRDRDSSRSSSSSSDSSLDS', 'UUUUUUULLLDSSSSSSSSSSS']


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 10


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


class Ng28(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._player = (0, 0)
        self._crates = set()
        self._g = 0
        self._left = 0
        self._falling = False
        super().__init__(
            game_id="ng28-e28c3f03",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._player = tuple(self._cfg["player"])
        self._crates = set(map(tuple, self._cfg["crates"]))
        self._g = 0
        self._falling = False
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _free(self, cell, ignore=None):
        n = self._cfg["n"]
        x, y = cell
        return (0 <= x < n and 0 <= y < n and (x, y) not in self._cfg["walls"]
                and cell not in self._crates and cell != self._player)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        n = cfg["n"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        scale = -(-budget_of(self._current_level_index) // BAR_MAX)
        px[1, 2:2 + -(-self._left // scale)] = BAR
        arrow = np.rot90(BASE, -self._g)
        for dy in range(5):
            for dx in range(5):
                if arrow[dy, dx]:
                    px[3 + dy, 56 + dx] = ARROW
        for y in range(n):
            for x in range(n):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if (x, y) in cfg["walls"]:
                    px[oy:oy + 6, ox:ox + 6] = WALL
                else:
                    px[oy:oy + 6, ox:ox + 6] = FLOOR
                    if (x, y) in cfg["targets"]:
                        px[oy:oy + 6, ox:ox + 6] = SOCKET
                        px[oy + 1:oy + 5, ox + 1:ox + 5] = FLOOR
        for x, y in self._crates:
            ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
            px[oy + 1:oy + 5, ox + 1:ox + 5] = CRATE
        x, y = self._player
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 5, ox + 1:ox + 5] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _fall_pass(self):
        dx, dy = GRAVITY[self._g]
        moved = False
        for c in sorted(self._crates, key=lambda c: -(c[0] * dx + c[1] * dy)):
            t = (c[0] + dx, c[1] + dy)
            if self._free(t):
                self._crates.remove(c)
                self._crates.add(t)
                moved = True
        return moved

    def _finish(self):
        self._left = max(0, self._left - 1)
        self._draw()
        if self._crates == set(map(tuple, self._cfg["targets"])):
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()

    def step(self) -> None:
        if self._falling:
            moved = self._fall_pass()
            self._draw()
            if not moved:
                self._falling = False
                self._finish()
            return
        aid = self._action.id
        delta = MOVES.get(aid)
        if delta is not None:
            dest = (self._player[0] + delta[0], self._player[1] + delta[1])
            if self._free(dest):
                self._player = dest
            self._finish()
        elif aid == GameAction.ACTION5:
            self._g = (self._g + 1) % 4
            self._falling = True
            self._fall_pass()
            self._draw()
        else:
            self.complete_action()
