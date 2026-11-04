import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
BLUE, RED, YEL, GRN = 9, 8, 11, 14
LEFT_KEY = GameAction.ACTION3
RIGHT_KEY = GameAction.ACTION4


def odd(item):
    return item[1] % 2 == 1


def big(item):
    return item[1] >= 4


def equal(item):
    return item[1][0] == item[1][1]


def aligned(c):
    return (BLUE, c, 0) if c % 2 else (RED, c, 1)


def shape_aligned(i, c):
    colour = (RED, BLUE, YEL, GRN)[i % 4]
    return (colour, c, 0 if c % 2 else 1)


def mixed(i, c):
    colour = (RED, BLUE, YEL, GRN)[(i * 3 + 1) % 4]
    return (colour, c, (i // 2 + i) % 2)


COUNTS1 = [1, 4, 3, 2, 5, 6, 1, 2, 3, 4, 5, 2]
COUNTS2 = [3, 2, 5, 6, 1, 4, 7, 2, 5, 8, 3, 4]
COUNTS3 = [2, 5, 3, 6, 1, 8, 7, 4, 9, 2, 5, 6]
COUNTS4 = [2, 5, 3, 6, 1, 4, 6, 3, 4, 1, 5, 2]
PAIRS5 = [(2, 2), (1, 3), (3, 3), (2, 0), (1, 1), (3, 1), (0, 0), (2, 3), (3, 3), (1, 2), (2, 2), (0, 3)]

CONFIGS = [
    dict(
        rule=odd,
        yes=[(BLUE, c, 0) for c in (1, 3, 5, 7, 9, 3)],
        no=[(RED, c, 1) for c in (2, 4, 6, 8, 2, 6)],
        queue=[aligned(c) for c in COUNTS1],
    ),
    dict(
        rule=odd,
        yes=[shape_aligned(i, c) for i, c in enumerate((1, 3, 5, 7, 9, 5))],
        no=[shape_aligned(i + 1, c) for i, c in enumerate((2, 4, 6, 8, 4, 2))],
        queue=[shape_aligned(i + 2, c) for i, c in enumerate(COUNTS2)],
    ),
    dict(
        rule=odd,
        yes=[mixed(i, c) for i, c in enumerate((1, 3, 5, 7, 9, 5, 1))],
        no=[mixed(i + 3, c) for i, c in enumerate((2, 4, 6, 8, 4, 2, 6))],
        queue=[mixed(i + 1, c) for i, c in enumerate(COUNTS3)],
    ),
    dict(
        rule=big,
        yes=[mixed(i, c) for i, c in enumerate((4, 5, 6, 4, 6, 5, 5, 4))],
        no=[mixed(i + 2, c) for i, c in enumerate((1, 2, 3, 3, 2, 1, 3, 2))],
        queue=[mixed(i + 5, c) for i, c in enumerate(COUNTS4)],
    ),
    dict(
        rule=equal,
        yes=[mixed(i, p) for i, p in enumerate(((1, 1), (2, 2), (3, 3), (2, 2), (1, 1), (3, 3)))],
        no=[mixed(i + 1, p) for i, p in enumerate(((1, 2), (3, 1), (2, 3), (1, 3), (0, 2), (3, 0)))],
        queue=[mixed(i + 2, p) for i, p in enumerate(PAIRS5)],
    ),
]
for cfg in CONFIGS:
    cfg["answers"] = ["R" if cfg["rule"](item) else "L" for item in cfg["queue"]]

SOLUTIONS = [cfg["answers"] for cfg in CONFIGS]


def item_pixels(item):
    colour, count, shape = item
    px = np.full((7, 7), colour, dtype=np.int8)
    ink = 5
    if shape == 1:
        px[1:6, 1:6] = 4
        ink = colour
    if isinstance(count, tuple):
        spots = []
        for col, n in zip((1, 5), count):
            spots += [(1 + 2 * r, col) for r in range(n)]
    else:
        spots = [(1 + 2 * (k // 3), 1 + 2 * (k % 3)) for k in range(count)]
    for r, c in spots:
        px[r, c] = ink
    return px


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


class Ng51(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = 0
        self._marks = []
        super().__init__(
            game_id="ng51-5a1c0051",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = 0
        self._marks = []
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:30] = RED
        px[1, 33:61] = GRN
        for side, items in ((2, self._cfg["no"]), (33, self._cfg["yes"])):
            for k, item in enumerate(items):
                x = side + (k % 3) * 9
                y = 4 + (k // 3) * 9
                px[y:y + 7, x:x + 7] = item_pixels(item)
        queue = self._cfg["queue"]
        if self._pos < len(queue):
            px[40:47, 28:35] = item_pixels(queue[self._pos])
        for i in range(len(queue)):
            colour = 3
            if i < len(self._marks):
                colour = GRN if self._marks[i] == "R" else RED
            px[54:57, 2 + i * 5:5 + i * 5] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        if aid in (LEFT_KEY, RIGHT_KEY) and self._pos < len(self._cfg["queue"]):
            self._marks.append("L" if aid == LEFT_KEY else "R")
            self._pos += 1
            self._draw()
            if self._pos == len(self._cfg["queue"]):
                if self._marks == self._cfg["answers"]:
                    self.next_level()
                else:
                    self.lose()
        self.complete_action()
