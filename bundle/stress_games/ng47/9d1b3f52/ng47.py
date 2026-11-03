import random

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BG, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


def blank():
    return np.full((64, 64), BG, dtype=np.int8)


def put(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    px[y0:max(y0, min(64, y + h)), x0:max(x0, min(64, x + w))] = c


def glyph(px, x, y, pat, scale, c):
    for gy in range(3):
        for gx in range(3):
            if pat[gy * 3 + gx]:
                put(px, x + gx * scale, y + gy * scale, scale, scale, c)


def make_camera():
    return Camera(width=64, height=64, background=BG, letter_box=BG)

POOL = ["111010010", "101010101", "100100111", "111101111", "111100111", "101111101", "010010010",
        "110010011", "101101010", "111111100", "011010110", "101101111", "010111010", "100010001"]
POOL = [tuple(int(ch) for ch in s) for s in POOL]
COLS = [8, 9, 11, 14, 12, 15]
SPECS = [
    dict(cols=4, rows=3, group=2, ident=True, swap=False, extra=3, seed=2),
    dict(cols=4, rows=4, group=2, ident=False, swap=False, extra=3, seed=4),
    dict(cols=6, rows=3, group=3, ident=False, swap=False, extra=3, seed=6),
    dict(cols=4, rows=4, group=2, ident=False, swap=True, extra=3, seed=8),
    dict(cols=6, rows=3, group=3, ident=False, swap=True, extra=3, seed=10),
]
PITCH = 10
SHOW = 8


def build(spec):
    rng = random.Random(spec["seed"])
    total = spec["cols"] * spec["rows"]
    groups = total // spec["group"]
    shapes = rng.sample(range(len(POOL)), groups)
    cards = []
    for g, s in enumerate(shapes):
        base = rng.choice(COLS)
        for _ in range(spec["group"]):
            cards.append((s, base if spec["ident"] else rng.choice(COLS)))
    rng.shuffle(cards)
    return cards


BUILT = [build(s) for s in SPECS]


def centre(spec, i):
    c, r = i % spec["cols"], i // spec["cols"]
    ox = (64 - spec["cols"] * PITCH + 1) // 2
    return ox + c * PITCH, 10 + r * PITCH


def solution(i):
    spec, cards = SPECS[i], BUILT[i]
    out = []
    by = {}
    for k, (s, c) in enumerate(cards):
        by.setdefault(s, []).append(k)
    for s in sorted(by):
        for k in by[s]:
            x, y = centre(spec, k)
            out.append((x + 4, y + 4))
    return out


SOLUTIONS = [solution(i) for i in range(len(SPECS))]
LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng47(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cards = []
        super().__init__(
            game_id="ng47-9d1b3f52",
            levels=LEVELS,
            camera=make_camera(),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level):
        self._idx = self._current_level_index
        self._spec = SPECS[self._idx]
        self._cards = list(BUILT[self._idx])
        self._done = set()
        self._up = []
        self._left = self._spec["cols"] * self._spec["rows"] // self._spec["group"] + self._spec["extra"]
        self._anim = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = blank()
        for i in range(self._left):
            put(px, 2 + i * 4, 2, 3, 3, 8)
        for i, (s, c) in enumerate(self._cards):
            x, y = centre(self._spec, i)
            shown = i in self._done or i in self._up
            frame = 14 if i in self._done else (11 if i in self._up else 3)
            put(px, x, y, 9, 9, frame)
            if shown:
                put(px, x + 1, y + 1, 7, 7, BG)
                glyph(px, x + 1, y + 1, POOL[s], 2, c)
            else:
                put(px, x + 1, y + 1, 7, 7, 4)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _hit(self, x, y):
        for i in range(len(self._cards)):
            cx, cy = centre(self._spec, i)
            if cx <= x < cx + 9 and cy <= y < cy + 9:
                return i
        return None

    def step(self):
        if self._anim > 0:
            self._anim -= 1
            if self._anim == 0:
                if self._spec["swap"] and len(self._up) >= 2:
                    a, b = self._up[0], self._up[-1]
                    self._cards[a], self._cards[b] = self._cards[b], self._cards[a]
                self._up = []
                self._draw()
                self.complete_action()
            return
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            i = self._hit(int(data.get("x", -1)), int(data.get("y", -1)))
            if i is not None and i not in self._done and i not in self._up:
                self._up.append(i)
                shapes = {self._cards[k][0] for k in self._up}
                self._draw()
                if len(shapes) > 1:
                    self._left -= 1
                    self._anim = SHOW
                    return
                if len(self._up) == self._spec["group"]:
                    self._done |= set(self._up)
                    self._up = []
                    self._draw()
                    if len(self._done) == len(self._cards):
                        self.next_level()
                        self.complete_action()
                        return
        self.complete_action()
