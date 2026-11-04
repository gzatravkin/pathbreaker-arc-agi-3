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

COLS = [8, 9, 11, 14, 12, 15]
SPECS = [
    dict(cols=4, rows=2, dup=0, same=True, hide=False, sym=True, seed=1),
    dict(cols=3, rows=3, dup=0, same=False, hide=False, sym=False, seed=2),
    dict(cols=4, rows=2, dup=2, same=False, hide=False, sym=False, seed=3),
    dict(cols=3, rows=3, dup=0, same=False, hide=True, sym=False, seed=4),
    dict(cols=4, rows=3, dup=2, same=False, hide=True, sym=False, seed=5),
]
PITCH = 14
BOX = 12
SHOW = 6


def rot(p):
    return tuple(reversed(p))


ALL = [tuple(int(ch) for ch in format(v, "09b")) for v in range(512)]
SYMS = [p for p in ALL if 3 <= sum(p) <= 7 and p == rot(p)]
ASYM = [p for p in ALL if 4 <= sum(p) <= 6 and p != rot(p)]


def build(spec):
    rng = random.Random(spec["seed"])
    n = spec["cols"] * spec["rows"]
    uniq = n - spec["dup"]
    used = set()
    shapes = []
    pool = SYMS if spec["sym"] else ASYM
    while len(shapes) < uniq + 1:
        p = rng.choice(pool)
        if p in used or rot(p) in used:
            continue
        used |= {p, rot(p)}
        shapes.append(p)
    lidshape = shapes[:uniq] + shapes[:spec["dup"]]
    rng.shuffle(lidshape)
    lidcol = [rng.choice(COLS) for _ in range(n)]
    items = [(rot(lidshape[k]), rng.choice(COLS)) for k in range(n)]
    if spec["same"]:
        items = [(rot(lidshape[k]), lidcol[k]) for k in range(n)]
    spare = (lidshape[0] if not spec["sym"] else shapes[uniq], rng.choice(COLS))
    items.append(spare)
    while True:
        order = list(range(n + 1))
        rng.shuffle(order)
        contents = [items[order[k]] for k in range(n)]
        hand = items[order[n]]
        if all(contents[k][0] != rot(lidshape[k]) for k in range(n)):
            return dict(lid=[(lidshape[k], lidcol[k]) for k in range(n)], contents=contents, hand=hand)


BUILT = [build(s) for s in SPECS]


def spot(spec, i):
    c, r = i % spec["cols"], i // spec["cols"]
    ox = (64 - spec["cols"] * PITCH + 2) // 2
    return ox + c * PITCH, 20 + r * PITCH


def solve(spec, built):
    contents = list(built["contents"])
    hand = built["hand"]
    lid = built["lid"]
    out = []
    n = len(contents)
    for _ in range(200):
        wrong = [k for k in range(n) if contents[k][0] != rot(lid[k][0])]
        if not wrong:
            return out
        fit = [k for k in wrong if hand[0] == rot(lid[k][0])]
        k = fit[0] if fit else wrong[0]
        contents[k], hand = hand, contents[k]
        x, y = spot(spec, k)
        out.append((x + 5, y + 5))
    return None


SOLUTIONS = [solve(s, b) for s, b in zip(SPECS, BUILT)]
LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng50(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._contents = []
        super().__init__(
            game_id="ng50-b83e7c06",
            levels=LEVELS,
            camera=make_camera(),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level):
        self._spec = SPECS[self._current_level_index]
        built = BUILT[self._current_level_index]
        self._lid = built["lid"]
        self._contents = list(built["contents"])
        self._hand = built["hand"]
        self._open = None
        self._anim = 0
        self._seen = not self._spec["hide"]
        self._left = min(58, 2 * len(SOLUTIONS[self._current_level_index]) + 4)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = blank()
        for i, (s, c) in enumerate(self._lid):
            x, y = spot(self._spec, i)
            put(px, x, y, BOX, BOX, 11 if self._open == i else 3)
            put(px, x + 1, y + 1, BOX - 2, BOX - 2, 4)
            glyph(px, x + 3, y + 3, s, 2, c)
        put(px, 2, 17, self._left, 1, 14)
        put(px, 26, 3, 11, 11, 2)
        put(px, 27, 4, 9, 9, BG)
        if self._seen:
            glyph(px, 27, 4, self._hand[0], 3, self._hand[1])
        else:
            put(px, 27, 4, 9, 9, 3)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _hit(self, x, y):
        for i in range(len(self._lid)):
            bx, by = spot(self._spec, i)
            if bx <= x < bx + BOX and by <= y < by + BOX:
                return i
        return None

    def step(self):
        if self._anim > 0:
            self._anim -= 1
            if self._anim == 0:
                self._open = None
                self._seen = not self._spec["hide"]
                self._draw()
                self.complete_action()
            return
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            i = self._hit(int(data.get("x", -1)), int(data.get("y", -1)))
            if i is not None:
                self._contents[i], self._hand = self._hand, self._contents[i]
                self._open = i
                self._seen = True
                self._left -= 1
                self._draw()
                if all(self._contents[k][0] == rot(self._lid[k][0]) for k in range(len(self._lid))):
                    self.next_level()
                    self.complete_action()
                    return
                if self._left <= 0:
                    self._draw()
                    self.lose()
                    self.complete_action()
                    return
                self._anim = SHOW
                return
        self.complete_action()
