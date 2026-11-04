import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
DOT = 5
TOKEN = 1
BAR = 14
ORIGIN_X = 4
ORIGIN_Y = 8
PITCH = 12
BAR_MAX = 60
DOTS = {
    1: [(3, 3)],
    2: [(1, 1), (5, 5)],
    3: [(1, 1), (3, 3), (5, 5)],
    4: [(1, 1), (5, 1), (1, 5), (5, 5)],
    5: [(1, 1), (5, 1), (3, 3), (1, 5), (5, 5)],
}

CONFIGS = [
    [{'s': (0, 2), 'n': 1, 'c': 9}, {'s': (1, 2), 'n': 2, 'c': 9}, {'s': (2, 2), 'n': 3, 'c': 9}, {'s': (3, 2), 'n': 4, 'c': 9}, {'s': (4, 2), 'n': 5, 'c': 9}],
    [{'s': (0, 3), 'n': 3, 'c': 9}, {'s': (3, 4), 'n': 4, 'c': 10}, {'s': (3, 1), 'n': 5, 'c': 12}, {'s': (2, 2), 'n': 4, 'c': 10}, {'s': (0, 0), 'n': 3, 'c': 10}, {'s': (4, 2), 'n': 2, 'c': 12}, {'s': (0, 4), 'n': 1, 'c': 10}],
    [{'s': (1, 1), 'n': 5, 'c': 14}, {'s': (3, 1), 'n': 4, 'c': 14}, {'s': (4, 2), 'n': 3, 'c': 14}, {'s': (2, 4), 'n': 2, 'c': 14}, {'s': (0, 1), 'n': 1, 'c': 10}, {'s': (1, 0), 'n': 2, 'c': 12}, {'s': (4, 0), 'n': 1, 'c': 9}, {'s': (2, 0), 'n': 2, 'c': 9}, {'s': (1, 4), 'n': 3, 'c': 10}],
    [{'s': (4, 4), 'n': 3, 'c': 10}, {'s': (2, 0), 'n': 4, 'c': 14}, {'s': (2, 3), 'n': 3, 'c': 12}, {'s': (1, 1), 'n': 2, 'c': 9}, {'s': (2, 1), 'n': 3, 'c': 12}, {'s': (0, 1), 'n': 4, 'c': 9}, {'s': (3, 2), 'n': 5, 'c': 12}, {'s': (1, 0), 'n': 4, 'c': 12}, {'s': (4, 3), 'n': 5, 'c': 10}, {'s': (2, 4), 'n': 4, 'c': 14}, {'s': (4, 2), 'n': 3, 'c': 14}],
    [{'s': (0, 2), 'n': 4, 'c': 12}, {'s': (2, 0), 'n': 3, 'c': 14}, {'s': (0, 4), 'n': 2, 'c': 10}, {'s': (2, 2), 'n': 1, 'c': 10}, {'s': (1, 4), 'n': 2, 'c': 12}, {'s': (2, 3), 'n': 1, 'c': 9}, {'s': (4, 4), 'n': 2, 'c': 12}, {'s': (4, 2), 'n': 3, 'c': 12}, {'s': (3, 3), 'n': 2, 'c': 9}, {'s': (4, 3), 'n': 1, 'c': 12}, {'s': (1, 1), 'n': 2, 'c': 9}, {'s': (0, 3), 'n': 3, 'c': 12}, {'s': (3, 2), 'n': 2, 'c': 12}],
]

ORDERS = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]


def centre_of(cfg, index):
    sx, sy = cfg[index]["s"]
    return (ORIGIN_X + sx * PITCH + 3, ORIGIN_Y + sy * PITCH + 3)


SOLUTIONS = [[centre_of(cfg, j) for j in order] for cfg, order in zip(CONFIGS, ORDERS)]


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 8


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


class Ng29(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._nodes = CONFIGS[0]
        self._alive = set()
        self._token = 0
        self._left = 0
        super().__init__(
            game_id="ng29-f29d4a04",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._nodes = CONFIGS[self._current_level_index]
        self._alive = set(range(len(self._nodes)))
        self._token = 0
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        scale = -(-budget_of(self._current_level_index) // BAR_MAX)
        px[1, 2:2 + -(-self._left // scale)] = BAR
        for i, node in enumerate(self._nodes):
            if i not in self._alive:
                continue
            ox = ORIGIN_X + node["s"][0] * PITCH
            oy = ORIGIN_Y + node["s"][1] * PITCH
            if i == self._token:
                px[oy - 1:oy + 8, ox - 1:ox + 8] = TOKEN
            px[oy:oy + 7, ox:ox + 7] = node["c"]
            for dx, dy in DOTS[node["n"]]:
                px[oy + dy, ox + dx] = DOT
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _hit(self, x, y):
        for i in self._alive:
            ox = ORIGIN_X + self._nodes[i]["s"][0] * PITCH
            oy = ORIGIN_Y + self._nodes[i]["s"][1] * PITCH
            if ox <= x < ox + 7 and oy <= y < oy + 7:
                return i
        return None

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            hit = self._hit(int(data.get("x", -1)), int(data.get("y", -1)))
            if hit is not None and hit != self._token:
                if abs(self._nodes[hit]["n"] - self._nodes[self._token]["n"]) == 1:
                    self._alive.discard(self._token)
                    self._token = hit
            self._left = max(0, self._left - 1)
            self._draw()
            if len(self._alive) == 1:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
