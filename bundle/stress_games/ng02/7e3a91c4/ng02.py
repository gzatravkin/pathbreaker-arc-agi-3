import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
STICKY = 9
GRAIN = 12
PIP = 0
FRAME = 2
PICK = 0
ARROW = 10
MAP_EMPTY = 4
MAP_HIT = 14
MAP_WRONG = 8
MAP_SPARE = 2
CELL = 4
FX = 2
FY = 10
PANEL_X = 49
MAP_Y = 10
PIP_Y = 2
PIP_X = 2

SHAPES = {
    "o": ((0, 0),),
    "h": ((0, 0), (1, 0)),
    "v": ((0, 0), (0, 1)),
    "l": ((0, 0), (0, 1), (1, 1)),
    "g": ((0, 0),),
}
KIND = {"o": STICKY, "h": STICKY, "v": STICKY, "l": STICKY, "g": GRAIN}
FALLS = [(0, 1), (-1, 0), (0, -1), (1, 0)]

ARROW_DOWN = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=bool)
ARROWS = [ARROW_DOWN, np.rot90(ARROW_DOWN, -1), np.rot90(ARROW_DOWN, 2), np.rot90(ARROW_DOWN, 1)]


def cells_of(rows):
    out = set()
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                out.add((c, r))
    return frozenset(out)


CONFIGS = [
    dict(w=8, h=8, rows=None, shapes="o", turn=False, exact=False,
         plan=[("o", 5, 2), ("o", 5, 3), ("o", 6, 5), ("o", 5, 1), ("o", 1, 4)],
         keep=None),
    dict(w=9, h=9, rows=["", "", "", "", "", ".#####"], shapes="o", turn=False, exact=False,
         plan=[("o", 2, 7), ("o", 3, 6), ("o", 3, 7), ("o", 4, 2), ("o", 7, 1), ("o", 7, 3)],
         keep=None),
    dict(w=9, h=9, rows=None, shapes="ohv", turn=False, exact=True,
         plan=[("v", 2, 3), ("v", 6, 1), ("h", 2, 2), ("h", 5, 4), ("o", 4, 1), ("o", 3, 5)],
         keep=None),
    dict(w=8, h=8, rows=None, shapes="o", turn=True, exact=True,
         plan=[("o", 3, 2), ("o", 5, 4), ("o", 6, 5), ("o", 2, 1), ("o", 0, 0), ("o", 5, 6), ("o", 4, 3)],
         keep=None),
    dict(w=9, h=9, rows=None, shapes="og", turn=False, exact=True,
         plan=[("o", 4, 1), ("o", 4, 1), ("g", 3, 0), ("g", 3, 0), ("g", 3, 0), ("g", 4, 0), ("o", 6, 3), ("g", 5, 0)],
         keep=None),
    dict(w=10, h=10, rows=["", "", "", "", "....##", "....##"], shapes="ohvl", turn=True, exact=True,
         plan=[("v", 2, 1), ("h", 6, 7), ("l", 7, 2), ("o", 3, 6), ("h", 3, 1), ("v", 8, 6), ("o", 6, 3), ("l", 0, 2)],
         keep=None),
]


def blocked_of(cfg):
    return cells_of(cfg["rows"] or [])


def inside(cfg, cell):
    return 0 <= cell[0] < cfg["w"] and 0 <= cell[1] < cfg["h"]


def free(cfg, blocked, placed, cells):
    return all(inside(cfg, c) and c not in blocked and c not in placed for c in cells)


def shift(cells, dx, dy):
    return tuple((c + dx, r + dy) for c, r in cells)


def fall_once(cfg, blocked, placed, cells, kind, gravity):
    gx, gy = FALLS[gravity]
    moved = shift(cells, gx, gy)
    if free(cfg, blocked, placed, moved):
        return moved
    if kind != GRAIN:
        return None
    for sx, sy in ((-gy, gx), (gy, -gx)):
        side = shift(cells, sx, sy)
        slide = shift(side, gx, gy)
        if free(cfg, blocked, placed, side) and free(cfg, blocked, placed, slide):
            return slide
    return None


def settle(cfg, blocked, placed, shape, anchor, gravity):
    cells = shift(SHAPES[shape], anchor[0], anchor[1])
    if not free(cfg, blocked, placed, cells):
        return None
    kind = KIND[shape]
    while True:
        nxt = fall_once(cfg, blocked, placed, cells, kind, gravity)
        if nxt is None:
            return cells, kind
        cells = nxt


def simulate(cfg):
    blocked = blocked_of(cfg)
    placed = {}
    gravity = 0
    for shape, c, r in cfg["plan"]:
        out = settle(cfg, blocked, placed, shape, (c, r), gravity)
        if out is None:
            raise ValueError("bad plan")
        cells, kind = out
        for cell in cells:
            placed[cell] = kind
        if cfg["turn"]:
            gravity = (gravity + 1) % 4
    return placed


def target_of(cfg):
    placed = simulate(cfg)
    if cfg["keep"] is not None:
        return {c: placed[c] for c in cfg["keep"]}
    return placed


def drops_of(cfg):
    return 2 * len(cfg["plan"]) + 3


def palette_boxes(cfg):
    top = MAP_Y + cfg["h"] + 4
    boxes = []
    for i, shape in enumerate(cfg["shapes"]):
        x = PANEL_X + 7 * (i % 2)
        y = top + 8 * (i // 2)
        boxes.append((shape, x, y))
    return boxes


def arrow_origin(cfg):
    top = MAP_Y + cfg["h"] + 4 + 8 * ((len(cfg["shapes"]) + 1) // 2) + 2
    return PANEL_X + 4, top


def click_of(cfg, c, r):
    return (FX + CELL * c + 1, FY + CELL * r + 2)


def palette_click(cfg, shape):
    for s, x, y in palette_boxes(cfg):
        if s == shape:
            return (x + 3, y + 3)
    raise ValueError(shape)


def solution_of(cfg):
    steps = []
    chosen = cfg["shapes"][0]
    for shape, c, r in cfg["plan"]:
        if shape != chosen:
            steps.append(palette_click(cfg, shape))
            chosen = shape
        steps.append(click_of(cfg, c, r))
    return steps


SOLUTIONS = [solution_of(cfg) for cfg in CONFIGS]


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


class Ng02(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._blocked = frozenset()
        self._placed = {}
        self._target = {}
        self._left = 0
        self._shape = "o"
        self._gravity = 0
        self._falling = None
        super().__init__(
            game_id="ng02-7e3a91c4",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._blocked = blocked_of(self._cfg)
        self._placed = {}
        self._target = target_of(self._cfg)
        self._left = drops_of(self._cfg)
        self._shape = self._cfg["shapes"][0]
        self._gravity = 0
        self._falling = None
        self._draw(level)

    def _won(self):
        if self._cfg["exact"]:
            return self._placed == self._target
        return all(self._placed.get(c) == k for c, k in self._target.items())

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        w, h = cfg["w"], cfg["h"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[PIP_Y:PIP_Y + 2, PIP_X + 3 * i:PIP_X + 3 * i + 2] = PIP
        x0, y0 = FX - 1, FY - 1
        x1, y1 = FX + CELL * w, FY + CELL * h
        px[y1, x0:x1 + 1] = WALL
        px[y0:y1 + 1, x0] = WALL
        px[y0:y1 + 1, x1] = WALL
        if cfg["turn"]:
            px[y0, x0:x1 + 1] = WALL
        for c, r in self._blocked:
            px[FY + CELL * r:FY + CELL * r + CELL, FX + CELL * c:FX + CELL * c + CELL] = WALL
        for (c, r), kind in self._placed.items():
            px[FY + CELL * r:FY + CELL * r + CELL, FX + CELL * c:FX + CELL * c + CELL] = kind
        if self._falling is not None:
            cells, kind = self._falling
            for c, r in cells:
                px[FY + CELL * r:FY + CELL * r + CELL, FX + CELL * c:FX + CELL * c + CELL] = kind
        px[MAP_Y:MAP_Y + h + 2, PANEL_X:PANEL_X + w + 2] = FRAME
        for r in range(h):
            for c in range(w):
                cell = (c, r)
                colour = MAP_EMPTY
                want = self._target.get(cell)
                have = self._placed.get(cell)
                if cell in self._blocked:
                    colour = WALL
                elif want is not None and have == want:
                    colour = MAP_HIT
                elif want is not None and have is None:
                    colour = want
                elif have is not None:
                    colour = MAP_WRONG if cfg["exact"] or want is not None else MAP_SPARE
                px[MAP_Y + 1 + r, PANEL_X + 1 + c] = colour
        for shape, bx, by in palette_boxes(cfg):
            if shape == self._shape:
                px[by:by + 7, bx:bx + 7] = PICK
                px[by + 1:by + 6, bx + 1:bx + 6] = BACKGROUND
            for c, r in SHAPES[shape]:
                px[by + 1 + 2 * r:by + 3 + 2 * r, bx + 1 + 2 * c:bx + 3 + 2 * c] = KIND[shape]
        ax, ay = arrow_origin(cfg)
        glyph = ARROWS[self._gravity]
        px[ay:ay + 5, ax:ax + 5][glyph] = ARROW
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _click(self, x, y):
        cfg = self._cfg
        for shape, bx, by in palette_boxes(cfg):
            if bx <= x < bx + 7 and by <= y < by + 7:
                if shape != self._shape:
                    self._shape = shape
                    self._draw()
                return
        if not (FX <= x < FX + CELL * cfg["w"] and FY <= y < FY + CELL * cfg["h"]):
            return
        anchor = ((x - FX) // CELL, (y - FY) // CELL)
        cells = shift(SHAPES[self._shape], anchor[0], anchor[1])
        if not free(cfg, self._blocked, self._placed, cells):
            return
        self._falling = (cells, KIND[self._shape])
        self._draw()

    def _land(self):
        cells, kind = self._falling
        self._falling = None
        for cell in cells:
            self._placed[cell] = kind
        self._left = max(0, self._left - 1)
        if self._cfg["turn"]:
            self._gravity = (self._gravity + 1) % 4
        self._draw()
        if self._won():
            self.next_level()
        elif self._left == 0:
            self.lose()

    def step(self) -> None:
        if self._falling is not None:
            cells, kind = self._falling
            nxt = fall_once(self._cfg, self._blocked, self._placed, cells, kind, self._gravity)
            if nxt is None:
                self._land()
                self.complete_action()
                return
            self._falling = (nxt, kind)
            self._draw()
            return
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            try:
                x, y = int(data["x"]), int(data["y"])
            except (KeyError, TypeError, ValueError):
                x = y = None
            if x is not None:
                self._click(x, y)
            if self._falling is not None:
                return
        self.complete_action()
