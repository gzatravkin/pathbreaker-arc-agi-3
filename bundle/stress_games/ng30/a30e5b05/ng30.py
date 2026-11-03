import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
PLAYER = 8
EXIT = 11
BEACON = 10
BAR = 14
ORIGIN_X = 4
ORIGIN_Y = 5
PITCH = 5
RADIUS = 5
BAR_MAX = 60
LAGS = [1, 2, 3, 5, 7]

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    {'n': 12, 'walls': {(5, 4), (5, 7), (8, 3), (8, 9), (9, 8), (1, 6), (1, 3), (2, 8), (7, 1), (6, 5), (6, 11), (7, 10), (6, 8), (5, 6), (10, 2), (0, 10), (2, 1), (11, 10), (1, 8), (6, 7), (4, 1), (4, 10), (9, 3), (2, 0), (10, 7), (1, 7), (1, 10), (7, 2), (6, 3)}, 'start': (1, 0), 'ex': (10, 10)},
    {'n': 14, 'walls': {(12, 7), (4, 3), (4, 9), (4, 12), (9, 8), (8, 6), (8, 12), (13, 2), (0, 8), (11, 8), (10, 12), (11, 11), (3, 0), (12, 12), (3, 9), (5, 3), (4, 11), (9, 13), (0, 10), (2, 1), (2, 7), (6, 7), (12, 2), (12, 11), (4, 1), (3, 11), (4, 4), (8, 1), (11, 0), (0, 3), (0, 9), (10, 7), (0, 6), (0, 12), (1, 7), (2, 6), (13, 6)}, 'start': (1, 0), 'ex': (11, 12)},
    {'n': 15, 'walls': {(4, 0), (4, 3), (3, 1), (12, 10), (14, 10), (5, 7), (9, 5), (5, 10), (11, 2), (8, 6), (10, 3), (1, 0), (0, 8), (0, 14), (11, 11), (2, 14), (13, 8), (6, 8), (6, 14), (7, 13), (4, 5), (12, 6), (12, 12), (3, 6), (5, 9), (8, 2), (9, 1), (11, 4), (0, 1), (0, 7), (2, 4), (1, 2), (8, 8), (13, 4), (2, 7), (1, 5), (1, 11), (1, 8), (1, 14), (2, 13), (6, 7), (14, 2), (12, 14), (3, 11), (4, 4), (14, 8), (14, 14), (5, 11), (8, 4), (9, 3), (5, 8), (1, 1), (8, 10), (10, 1), (13, 0), (2, 3), (13, 3), (1, 7), (13, 9), (7, 5), (7, 8)}, 'start': (0, 2), 'ex': (13, 12)},
    {'n': 16, 'walls': {(8, 0), (5, 10), (10, 6), (0, 5), (11, 5), (11, 14), (2, 11), (15, 5), (6, 2), (7, 1), (5, 3), (9, 10), (2, 4), (13, 10), (15, 7), (3, 8), (5, 5), (0, 0), (9, 12), (6, 6), (3, 1), (0, 2), (13, 5), (15, 2), (10, 15), (13, 14), (15, 11), (7, 7), (5, 9), (10, 8), (13, 7), (2, 13), (14, 14), (10, 10), (13, 9), (15, 6), (2, 6), (12, 1), (12, 10), (3, 7), (9, 2), (4, 15), (1, 9), (0, 11), (2, 8), (6, 8), (12, 3), (3, 9), (10, 5), (10, 14), (0, 13), (2, 10), (6, 10), (12, 5), (3, 2), (4, 1), (3, 11), (14, 11), (10, 7), (2, 12), (6, 12), (9, 11), (0, 8), (2, 14), (7, 13), (8, 5), (11, 1), (8, 14), (7, 6), (4, 7), (5, 8), (10, 4), (9, 6), (0, 3), (13, 6), (13, 15), (6, 9), (7, 8)}, 'start': (1, 0), 'ex': (15, 14)},
    {'n': 17, 'walls': {(6, 12), (4, 0), (4, 6), (3, 16), (5, 13), (0, 2), (5, 16), (10, 0), (9, 8), (8, 6), (2, 2), (9, 14), (1, 6), (10, 9), (13, 2), (1, 3), (1, 9), (10, 12), (2, 8), (13, 5), (7, 4), (13, 8), (1, 15), (6, 8), (7, 13), (14, 6), (15, 9), (4, 5), (3, 3), (3, 9), (5, 0), (5, 6), (14, 3), (14, 9), (10, 11), (0, 10), (11, 16), (16, 0), (1, 11), (2, 10), (2, 16), (16, 9), (6, 4), (16, 12), (7, 9), (6, 7), (15, 10), (7, 12), (6, 16), (3, 2), (12, 11), (4, 7), (12, 8), (4, 4), (4, 16), (5, 5), (5, 11), (8, 4), (9, 3), (9, 9), (11, 0), (0, 3), (11, 9), (10, 13), (8, 13), (10, 10), (13, 3), (15, 6), (16, 2), (16, 5), (1, 10), (7, 2), (2, 15)}, 'start': (1, 1), 'ex': (16, 14)},
]

SOLUTIONS = ['DDRRUURRRRRDRRRDDDDDDDDLD', 'DDDDDRRDDDRRDDRRRRRRRRDDDLLU', 'DDDDRRURUUURRDDDDRRDRRDDDDDDRRRRU', 'DDDRRDRRRURRRRRDRRRDDDDLLDLDDDDRRRRD', 'RRRRRRRDRRDDDDDLLDDDLLLLDDDDDRRRRRRURRRRRR']


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 12


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


class Ng30(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._trail = []
        self._left = 0
        super().__init__(
            game_id="ng30-a30e5b05",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._trail = [tuple(self._cfg["start"])]
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _beacon(self):
        lag = LAGS[self._current_level_index]
        return self._trail[max(0, len(self._trail) - 1 - lag)]

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        n = cfg["n"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        scale = -(-budget_of(self._current_level_index) // BAR_MAX)
        px[1, 2:2 + -(-self._left // scale)] = BAR
        bx, by = self._beacon()
        player = self._trail[-1]
        for j in range(2 * RADIUS + 1):
            for i in range(2 * RADIUS + 1):
                wx, wy = bx - RADIUS + i, by - RADIUS + j
                ox, oy = ORIGIN_X + i * PITCH, ORIGIN_Y + j * PITCH
                if not (0 <= wx < n and 0 <= wy < n) or (wx, wy) in cfg["walls"]:
                    colour = WALL
                elif (wx, wy) == tuple(cfg["ex"]):
                    colour = EXIT
                else:
                    colour = FLOOR
                px[oy:oy + 4, ox:ox + 4] = colour
                if (wx, wy) == (bx, by):
                    px[oy:oy + 4, ox:ox + 4] = BEACON
                    px[oy + 1:oy + 3, ox + 1:ox + 3] = FLOOR
                if (wx, wy) == player:
                    px[oy:oy + 4, ox:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            cfg = self._cfg
            x, y = self._trail[-1]
            dest = (x + delta[0], y + delta[1])
            if 0 <= dest[0] < cfg["n"] and 0 <= dest[1] < cfg["n"] and dest not in cfg["walls"]:
                self._trail.append(dest)
            self._left = max(0, self._left - 1)
            self._draw()
            if self._trail[-1] == tuple(cfg["ex"]):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
