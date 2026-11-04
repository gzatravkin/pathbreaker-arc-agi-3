import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
PAPER = 2
SEAM = 4
BAR = 14
ARROW = 12
ORIGIN_X = 2
ORIGIN_Y = 8
PITCH = 5
CELL = 3
MINI_X = 44
MINI_Y = 8
MINI_PITCH = 3


def sheet(rows, dots):
    cells = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                cells[(x, y)] = PAPER
    for (x, y), colour in dots.items():
        cells[(x, y)] = colour
    return cells


FULL = ["########"] * 8
OCTAGON = ["..####..", ".######.", "########", "########", "########", "########", ".######.", "..####.."]
RING = ["########", "########", "##....##", "##....##", "##....##", "##....##", "########", "########"]

CONFIGS = [
    dict(rows=FULL, dots={(0, 0): 8, (1, 1): 9, (2, 0): 11}, folds=[("V", 3, 0), ("H", 2, 0), ("V", 3, 0), ("H", 2, 0), ("H", 2, 0)]),
    dict(rows=FULL, dots={(1, 3): 11, (7, 4): 15, (2, 1): 14, (5, 5): 15}, folds=[("V", 7, 0), ("H", 5, 0), ("V", 2, 0), ("V", 3, 0), ("H", 4, 0), ("H", 3, 0)]),
    dict(rows=FULL, dots={(2, 6): 8, (7, 2): 11, (3, 6): 9, (6, 1): 14, (1, 5): 15}, folds=[("H", 2, 1), ("V", 7, 1), ("H", 4, 0), ("V", 2, 0), ("H", 1, 1), ("V", 4, 0)]),
    dict(rows=OCTAGON, dots={(5, 1): 11, (2, 6): 14, (4, 6): 15, (2, 4): 9, (0, 5): 8}, folds=[("V", 2, 0), ("H", 2, 1), ("H", 1, 1), ("V", 2, 0), ("V", 3, 0), ("V", 2, 0), ("H", 1, 0)]),
    dict(rows=RING, dots={(1, 2): 11, (1, 6): 15, (3, 0): 9, (2, 7): 11, (6, 1): 8}, folds=[("H", 1, 0), ("V", 4, 1), ("H", 1, 0), ("V", 3, 1), ("H", 2, 1), ("H", 1, 0), ("V", 1, 1)]),
]


def normalise(cells):
    if not cells:
        return cells
    mx = min(x for x, _ in cells)
    my = min(y for _, y in cells)
    return {(x - mx, y - my): c for (x, y), c in cells.items()}


def merge(stay, moved):
    if moved == PAPER and stay is not None:
        return stay
    return moved


def fold(cells, axis, crease, mode):
    idx = 0 if axis == "V" else 1
    out = {}
    moving = {}
    for pos, colour in cells.items():
        on_low = pos[idx] < crease
        if on_low == (mode == 0):
            moving[pos] = colour
        else:
            out[pos] = colour
    for pos, colour in moving.items():
        new = list(pos)
        new[idx] = 2 * crease - 1 - pos[idx]
        new = tuple(new)
        out[new] = merge(out.get(new), colour)
    return normalise(out)


def extent(cells):
    return max(x for x, _ in cells) + 1, max(y for _, y in cells) + 1


def target_of(cfg):
    cells = sheet(cfg["rows"], cfg["dots"])
    for axis, crease, mode in cfg["folds"]:
        cells = fold(cells, axis, crease, mode)
    return cells


def build_solution(cfg):
    out = []
    mode = 0
    for axis, crease, m in cfg["folds"]:
        if m != mode:
            out.append("S")
            mode = m
        if axis == "V":
            out.append((ORIGIN_X + crease * PITCH - 2, ORIGIN_Y + 1))
        else:
            out.append((ORIGIN_X + 1, ORIGIN_Y + crease * PITCH - 2))
    return out


SOLUTIONS = [build_solution(c) for c in CONFIGS]


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 4


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


class Ng67(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._cells = {}
        self._target = {}
        self._mode = 0
        self._left = 0
        super().__init__(
            game_id="ng67-6a670001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[5, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._cells = sheet(self._cfg["rows"], self._cfg["dots"])
        self._target = target_of(self._cfg)
        self._mode = 0
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // 62, 1 + i % 62] = BAR
        cells = self._cells
        for (x, y), colour in cells.items():
            ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
            px[oy:oy + CELL, ox:ox + CELL] = colour
            if (x + 1, y) in cells:
                px[oy:oy + CELL, ox + CELL:ox + PITCH] = SEAM
            if (x, y + 1) in cells:
                px[oy + CELL:oy + PITCH, ox:ox + CELL] = SEAM
            if (x + 1, y) in cells and (x, y + 1) in cells and (x + 1, y + 1) in cells:
                px[oy + CELL:oy + PITCH, ox + CELL:ox + PITCH] = SEAM
        for (x, y), colour in self._target.items():
            tx, ty = MINI_X + x * MINI_PITCH, MINI_Y + y * MINI_PITCH
            px[ty:ty + MINI_PITCH, tx:tx + MINI_PITCH] = colour
        ax, ay = 44, 40
        for i in range(7):
            px[ay + 3, ax + i] = ARROW
        if self._mode == 0:
            for i in range(4):
                px[ay + 3 - i, ax + 6 - i] = ARROW
                px[ay + 3 + i, ax + 6 - i] = ARROW
        else:
            for i in range(4):
                px[ay + 3 - i, ax + i] = ARROW
                px[ay + 3 + i, ax + i] = ARROW
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _crease_at(self, px, py):
        if not self._cells:
            return None
        w, h = extent(self._cells)
        rx, ry = px - ORIGIN_X, py - ORIGIN_Y
        if rx < 0 or ry < 0 or rx >= w * PITCH or ry >= h * PITCH:
            return None
        gx = rx % PITCH >= CELL
        gy = ry % PITCH >= CELL
        if gx == gy:
            return None
        if gx:
            c = rx // PITCH + 1
            return ("V", c) if 1 <= c <= w - 1 else None
        c = ry // PITCH + 1
        return ("H", c) if 1 <= c <= h - 1 else None

    def step(self) -> None:
        acted = False
        if self._action.id == GameAction.ACTION5:
            self._mode = 1 - self._mode
            acted = True
        elif self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            hit = self._crease_at(int(data.get("x", -1)), int(data.get("y", -1)))
            if hit is not None:
                self._cells = fold(self._cells, hit[0], hit[1], self._mode)
                acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if self._cells == self._target:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
