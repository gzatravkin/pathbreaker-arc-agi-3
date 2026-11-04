import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
BLUE, ORANGE, SELECT = 9, 12, 11
SLOT_X = 0
SLOT_PITCH = 8
SLOT_Y = 6
TRAY_X = 4
TRAY_Y = 24
TRAY_PITCH = 9
STRIKES = 3
PREFIX = 4


def colour_of(count):
    return BLUE if count % 2 == 1 else ORANGE


def fib_wrap(a, b, n):
    seq = [a, b]
    while len(seq) < n:
        seq.append((seq[-1] + seq[-2] - 1) % 9 + 1)
    return seq


COUNT_SEQUENCES = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 3, 4, 1, 3, 4, 1, 3],
    [2, 5, 8, 2, 5, 8, 2, 5],
    [1, 9, 2, 8, 3, 7, 4, 6],
    fib_wrap(2, 5, 8),
]
CONFIGS = [dict(counts=seq) for seq in COUNT_SEQUENCES]


def glyph_cell(count, colour):
    k = 2 * (count - 1) + (0 if colour == BLUE else 1)
    return k % 6, k // 6


def tray_xy(count, colour):
    col, row = glyph_cell(count, colour)
    return TRAY_X + TRAY_PITCH * col, TRAY_Y + TRAY_PITCH * row


def solution_for(cfg):
    steps = []
    current = None
    for i in range(PREFIX, len(cfg["counts"])):
        count = cfg["counts"][i]
        glyph = (count, colour_of(count))
        if glyph != current:
            x, y = tray_xy(*glyph)
            steps.append((x + 3, y + 3))
            current = glyph
        steps.append((SLOT_X + SLOT_PITCH * i + 3, SLOT_Y + 3))
    steps.append("S")
    return steps


SOLUTIONS = [solution_for(cfg) for cfg in CONFIGS]


def glyph_pixels(count, colour):
    px = np.full((7, 7), colour, dtype=np.int8)
    for k in range(count):
        px[1 + 2 * (k // 3), 1 + 2 * (k % 3)] = 5
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


class Ng53(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._placed = {}
        self._selected = None
        self._strikes = 0
        super().__init__(
            game_id="ng53-5a1c0053",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[5, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._placed = {}
        self._selected = None
        self._strikes = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        counts = self._cfg["counts"]
        for i, count in enumerate(counts):
            x = SLOT_X + SLOT_PITCH * i
            if i < PREFIX:
                px[SLOT_Y:SLOT_Y + 7, x:x + 7] = glyph_pixels(count, colour_of(count))
            elif i in self._placed:
                px[SLOT_Y:SLOT_Y + 7, x:x + 7] = glyph_pixels(*self._placed[i])
            else:
                px[SLOT_Y:SLOT_Y + 7, x:x + 7] = 3
                px[SLOT_Y + 1:SLOT_Y + 6, x + 1:x + 6] = 4
        for n in range(1, 10):
            for colour in (BLUE, ORANGE):
                x, y = tray_xy(n, colour)
                px[y:y + 7, x:x + 7] = glyph_pixels(n, colour)
                if self._selected == (n, colour):
                    px[y - 1, x - 1:x + 8] = SELECT
                    px[y + 7, x - 1:x + 8] = SELECT
                    px[y - 1:y + 8, x - 1] = SELECT
                    px[y - 1:y + 8, x + 7] = SELECT
        for s in range(STRIKES - self._strikes):
            px[62:64, 2 + s * 4:5 + s * 4] = 8
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _click(self, x, y):
        counts = self._cfg["counts"]
        if SLOT_Y <= y < SLOT_Y + 7:
            for i in range(PREFIX, len(counts)):
                x0 = SLOT_X + SLOT_PITCH * i
                if x0 <= x < x0 + 7:
                    if self._selected is not None:
                        self._placed[i] = self._selected
                    else:
                        self._placed.pop(i, None)
            return
        for n in range(1, 10):
            for colour in (BLUE, ORANGE):
                gx, gy = tray_xy(n, colour)
                if gx <= x < gx + 7 and gy <= y < gy + 7:
                    self._selected = None if self._selected == (n, colour) else (n, colour)

    def step(self) -> None:
        aid = self._action.id
        counts = self._cfg["counts"]
        if aid == GameAction.ACTION6:
            data = self._action.data or {}
            self._click(data.get("x", -1), data.get("y", -1))
            self._draw()
        elif aid == GameAction.ACTION5:
            blanks = range(PREFIX, len(counts))
            if all(i in self._placed for i in blanks):
                if all(self._placed[i] == (counts[i], colour_of(counts[i])) for i in blanks):
                    self.next_level()
                else:
                    self._strikes += 1
                    self._placed = {}
                    self._draw()
                    if self._strikes >= STRIKES:
                        self.lose()
        self.complete_action()
