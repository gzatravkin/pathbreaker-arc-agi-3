import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
CELL_EMPTY = 4
CELL_FULL = 11
CURSOR = 8
BAR = 14
ORIGIN_X = 1
ORIGIN_Y = 8
PITCH = 4
MINI_X = 43
MINI_Y = 8
MINI_PITCH = 2
SIDE = 10

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

SHAPE_L = ((0, 0), (0, 1), (0, 2), (1, 2))
SHAPE_T = ((0, 0), (1, 0), (2, 0), (1, 1))
SHAPE_Z = ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))

CONFIGS = [
    dict(shape=SHAPE_L, turn=1, xor=False, anchors=((4, 0), (4, 4), (0, 4), (0, 1))),
    dict(shape=SHAPE_L, turn=1, xor=False, anchors=((0, 1), (3, 1), (6, 1), (6, 5), (2, 5))),
    dict(shape=SHAPE_L, turn=1, xor=True, anchors=((3, 2), (4, 3), (3, 5), (5, 5), (1, 3))),
    dict(shape=SHAPE_T, turn=-1, xor=True, anchors=((1, 1), (2, 2), (5, 1), (6, 4), (3, 6), (0, 5))),
    dict(shape=SHAPE_Z, turn=1, xor=True, anchors=((0, 0), (2, 1), (5, 2), (4, 5), (1, 5), (7, 6), (3, 3))),
]


def rotate(cells, times, turn):
    out = list(cells)
    for _ in range(times % 4):
        if turn > 0:
            out = [(2 - y, x) for x, y in out]
        else:
            out = [(y, 2 - x) for x, y in out]
    return out


def place(cfg, painted, count, anchor):
    cells = [(anchor[0] + dx, anchor[1] + dy) for dx, dy in rotate(cfg["shape"], count, cfg["turn"])]
    if any(not (0 <= x < SIDE and 0 <= y < SIDE) for x, y in cells):
        return painted, count, False
    cur = set(painted)
    if cfg["xor"]:
        for c in cells:
            cur ^= {c}
    else:
        if any(c in cur for c in cells):
            return painted, count, False
        cur |= set(cells)
    return frozenset(cur), count + 1, True


def target_of(cfg):
    painted = frozenset()
    count = 0
    for a in cfg["anchors"]:
        painted, count, ok = place(cfg, painted, count, a)
        assert ok
    return painted


def build_solution(cfg):
    cur = [0, 0]
    out = []
    for ax, ay in cfg["anchors"]:
        while cur[0] != ax:
            out.append("R" if ax > cur[0] else "L")
            cur[0] += 1 if ax > cur[0] else -1
        while cur[1] != ay:
            out.append("D" if ay > cur[1] else "U")
            cur[1] += 1 if ay > cur[1] else -1
        out.append("S")
    return out


SOLUTIONS = [build_solution(c) for c in CONFIGS]


def budget_of(index):
    return min(120, 2 * len(SOLUTIONS[index]) + 8)


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


class Ng66(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._painted = frozenset()
        self._count = 0
        self._cursor = (0, 0)
        self._target = frozenset()
        self._left = 0
        self._flash = 0
        super().__init__(
            game_id="ng66-6a660001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._painted = frozenset()
        self._count = 0
        self._cursor = (0, 0)
        self._target = target_of(self._cfg)
        self._left = budget_of(self._current_level_index)
        self._flash = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // 62, 1 + i % 62] = BAR
        for y in range(SIDE):
            for x in range(SIDE):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                px[oy:oy + 3, ox:ox + 3] = CELL_FULL if (x, y) in self._painted else CELL_EMPTY
                tx, ty = MINI_X + x * MINI_PITCH, MINI_Y + y * MINI_PITCH
                px[ty:ty + 2, tx:tx + 2] = CELL_FULL if (x, y) in self._target else CELL_EMPTY
        cx, cy = self._cursor
        ox, oy = ORIGIN_X + cx * PITCH - 1, ORIGIN_Y + cy * PITCH - 1
        colour = 7 if self._flash else CURSOR
        for i in range(0, 3 * PITCH - 1):
            for (x, y) in ((ox + i, oy), (ox + i, oy + 3 * PITCH - 1), (ox, oy + i), (ox + 3 * PITCH - 1, oy + i)):
                if 0 <= x < 64 and 0 <= y < 64:
                    px[y, x] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        self._flash = 0
        delta = MOVES.get(self._action.id)
        acted = False
        if delta is not None:
            nx, ny = self._cursor[0] + delta[0], self._cursor[1] + delta[1]
            if 0 <= nx <= SIDE - 3 and 0 <= ny <= SIDE - 3:
                self._cursor = (nx, ny)
            acted = True
        elif self._action.id == GameAction.ACTION5:
            self._painted, self._count, ok = place(self._cfg, self._painted, self._count, self._cursor)
            self._flash = 0 if ok else 1
            acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if self._painted == self._target:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
