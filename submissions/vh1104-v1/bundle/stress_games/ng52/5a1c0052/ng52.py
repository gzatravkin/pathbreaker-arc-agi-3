import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
R, B, Y, G = 8, 9, 11, 14
ROW_Y = 1
ROW_PITCH = 10
ITEM_X = 10
ITEM_PITCH = 10
MARK_USER = 11
MARK_GIVEN = 14
STRIKES = 3


def line(idx, base, odd):
    items = [base] * 5
    items[idx] = odd
    return items, {idx}


def decoy_colour(idx, dec, base, n2, s2, colour2):
    items = [base] * 5
    items[idx] = (base[0], n2, s2)
    items[dec] = (colour2, base[1], base[2])
    return items, {idx}


def decoys_apart(idx, cdec, sdec, base, n2, colour2, shape2):
    items = [base] * 5
    items[idx] = (base[0], n2, base[2])
    items[cdec] = (colour2, base[1], base[2])
    items[sdec] = (base[0], base[1], shape2)
    return items, {idx}


def minority(pair, major, minor, colours, shapes):
    items = []
    for i in range(5):
        items.append((colours[i], minor if i in pair else major, shapes[i]))
    return items, set(pair)


def parity(counts, idx, colours, shapes):
    return [(colours[i], counts[i], shapes[i]) for i in range(5)], {idx}


CONFIGS = [
    dict(
        given=[line(2, (R, 3, 0), (B, 5, 1))],
        tests=[
            line(0, (Y, 2, 1), (G, 6, 0)),
            line(4, (B, 4, 0), (R, 1, 1)),
            line(1, (G, 5, 1), (Y, 2, 0)),
            line(3, (R, 1, 0), (B, 4, 1)),
            line(2, (Y, 6, 1), (R, 3, 0)),
        ],
    ),
    dict(
        given=[
            decoy_colour(2, 4, (R, 3, 0), 5, 1, B),
            decoy_colour(0, 3, (Y, 2, 1), 6, 0, G),
        ],
        tests=[
            decoy_colour(3, 1, (B, 4, 0), 1, 1, Y),
            decoy_colour(1, 4, (G, 5, 1), 2, 0, R),
            decoy_colour(4, 0, (R, 1, 0), 4, 1, B),
            decoy_colour(0, 2, (Y, 6, 1), 3, 0, G),
        ],
    ),
    dict(
        given=[
            decoys_apart(2, 0, 4, (R, 3, 0), 5, B, 1),
            decoys_apart(4, 1, 2, (Y, 2, 1), 6, G, 0),
        ],
        tests=[
            decoys_apart(1, 3, 0, (B, 4, 0), 1, Y, 1),
            decoys_apart(3, 0, 2, (G, 5, 1), 2, R, 0),
            decoys_apart(0, 4, 1, (R, 1, 0), 4, B, 1),
            decoys_apart(2, 1, 3, (Y, 6, 1), 3, G, 0),
        ],
    ),
    dict(
        given=[
            minority((1, 3), 2, 5, (R, B, R, Y, G), (0, 1, 1, 0, 0)),
            minority((0, 4), 4, 1, (B, B, G, R, Y), (1, 0, 0, 1, 1)),
        ],
        tests=[
            minority((2, 3), 3, 6, (Y, Y, B, B, R), (0, 0, 1, 0, 1)),
            minority((0, 2), 5, 2, (G, R, G, Y, B), (1, 1, 0, 0, 1)),
            minority((1, 4), 6, 3, (R, G, Y, B, G), (0, 1, 0, 1, 0)),
            minority((0, 3), 1, 4, (B, G, R, R, Y), (1, 0, 1, 0, 0)),
        ],
    ),
    dict(
        given=[
            parity((2, 4, 6, 3, 8), 3, (R, B, Y, G, R), (0, 1, 0, 1, 1)),
            parity((7, 1, 5, 3, 2), 4, (G, G, B, Y, R), (1, 0, 0, 1, 0)),
        ],
        tests=[
            parity((4, 8, 2, 6, 5), 4, (Y, R, B, G, B), (0, 0, 1, 1, 0)),
            parity((9, 3, 6, 1, 5), 2, (B, Y, G, R, Y), (1, 0, 0, 1, 1)),
            parity((1, 7, 3, 8, 9), 3, (R, R, Y, B, G), (0, 1, 1, 0, 0)),
            parity((6, 2, 7, 4, 8), 2, (G, B, R, Y, Y), (1, 1, 0, 0, 1)),
        ],
    ),
]


def solution_for(cfg):
    steps = []
    base = len(cfg["given"])
    for r, (items, picks) in enumerate(cfg["tests"]):
        for i in sorted(picks):
            steps.append((ITEM_X + ITEM_PITCH * i + 3, ROW_Y + ROW_PITCH * (base + r) + 3))
    steps.append("S")
    return steps


SOLUTIONS = [solution_for(cfg) for cfg in CONFIGS]


def item_pixels(item):
    colour, count, shape = item
    px = np.full((7, 7), colour, dtype=np.int8)
    ink = 5
    if shape == 1:
        px[1:6, 1:6] = 4
        ink = colour
    for k in range(count):
        px[1 + 2 * (k // 3), 1 + 2 * (k % 3)] = ink
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


class Ng52(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._marks = []
        self._strikes = 0
        super().__init__(
            game_id="ng52-5a1c0052",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[5, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._marks = [set() for _ in self._cfg["tests"]]
        self._strikes = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        rows = [(items, picks, True) for items, picks in self._cfg["given"]]
        rows += [(items, self._marks[k], False) for k, (items, _) in enumerate(self._cfg["tests"])]
        for r, (items, picks, given) in enumerate(rows):
            y = ROW_Y + ROW_PITCH * r
            px[y:y + 7, 2:5] = MARK_GIVEN if given else 3
            for i, item in enumerate(items):
                x = ITEM_X + ITEM_PITCH * i
                px[y:y + 7, x:x + 7] = item_pixels(item)
                if i in picks:
                    px[y + 8, x:x + 7] = MARK_GIVEN if given else MARK_USER
        for s in range(STRIKES - self._strikes):
            px[62:64, 2 + s * 4:5 + s * 4] = R
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        if aid == GameAction.ACTION6:
            data = self._action.data or {}
            x, y = data.get("x", -1), data.get("y", -1)
            base = len(self._cfg["given"])
            for k in range(len(self._cfg["tests"])):
                y0 = ROW_Y + ROW_PITCH * (base + k)
                if not y0 <= y < y0 + 7:
                    continue
                for i in range(5):
                    x0 = ITEM_X + ITEM_PITCH * i
                    if x0 <= x < x0 + 7:
                        self._marks[k] ^= {i}
            self._draw()
        elif aid == GameAction.ACTION5:
            if all(self._marks[k] == picks for k, (_, picks) in enumerate(self._cfg["tests"])):
                self.next_level()
            else:
                self._strikes += 1
                self._marks = [set() for _ in self._cfg["tests"]]
                self._draw()
                if self._strikes >= STRIKES:
                    self.lose()
        self.complete_action()
