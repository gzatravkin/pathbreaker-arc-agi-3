import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
EMPTY = 4
N = 8
BAR = 14
COLOURS = [9, 8, 11, 14]
ORIGIN_X = 1
ORIGIN_Y = 8
PITCH = 4
MINI_X = 38
MINI_Y = 8
MINI_PITCH = 3
ICON_Y = 44
ICON_PITCH = 2
ICON_STRIDE = 16
MARK = 0

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    dict(
        layers=[
            [(0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)],
            [(3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (5, 2), (6, 2)],
        ],
        goal=[(3, 4), (3, 5)],
    ),
    dict(
        layers=[
            [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1), (3, 2)],
            [(2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)],
        ],
        goal=[(4, 4), (3, 5)],
    ),
    dict(
        layers=[
            [(1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)],
            [(0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (1, 4), (2, 3)],
            [(5, 0), (5, 1), (6, 0), (6, 1), (6, 2), (7, 0), (7, 1)],
        ],
        goal=[(5, 4), (7, 6), (3, 6)],
    ),
    dict(
        layers=[
            [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)],
            [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (5, 3)],
            [(2, 5), (3, 5), (4, 4), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7)],
        ],
        goal=[(2, 5), (6, 5), (6, 1)],
    ),
    dict(
        layers=[
            [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (2, 1)],
            [(4, 5), (5, 5), (5, 6), (5, 7), (6, 6), (6, 7), (7, 6), (7, 7)],
            [(4, 1), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (7, 3)],
            [(0, 6), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)],
        ],
        goal=[(0, 7), (0, 6), (3, 6), (7, 4)],
    ),
]


def composite(layers, offsets):
    cover = {}
    for i, (cells, off) in enumerate(zip(layers, offsets)):
        for x, y in cells:
            key = ((x + off[0]) % N, (y + off[1]) % N)
            cover.setdefault(key, []).append(i)
    return {k: COLOURS[max(v)] for k, v in cover.items() if len(v) % 2 == 1}


def target_of(cfg):
    return composite(cfg["layers"], cfg["goal"])


def build_solution(cfg):
    out = []
    current = 0
    for i, off in enumerate(cfg["goal"]):
        if i != current:
            out.append((i * ICON_STRIDE + 7, ICON_Y + 7))
            current = i
        dx = off[0] % N
        dy = off[1] % N
        out += ["R"] * dx if dx <= N - dx else ["L"] * (N - dx)
        out += ["D"] * dy if dy <= N - dy else ["U"] * (N - dy)
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


class Ng69(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._off = []
        self._sel = 0
        self._target = {}
        self._left = 0
        super().__init__(
            game_id="ng69-6a690001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._off = [(0, 0)] * len(self._cfg["layers"])
        self._sel = 0
        self._target = target_of(self._cfg)
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        layers = self._cfg["layers"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // 62, 1 + i % 62] = BAR
        comp = composite(layers, self._off)
        for y in range(N):
            for x in range(N):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                px[oy:oy + 3, ox:ox + 3] = comp.get((x, y), EMPTY)
                tx, ty = MINI_X + x * MINI_PITCH, MINI_Y + y * MINI_PITCH
                px[ty:ty + 2, tx:tx + 2] = self._target.get((x, y), EMPTY)
        for i, cells in enumerate(layers):
            bx = i * ICON_STRIDE
            px[ICON_Y:ICON_Y + 16, bx:bx + 15] = EMPTY
            for x, y in cells:
                ox, oy = (x + self._off[i][0]) % N, (y + self._off[i][1]) % N
                px[ICON_Y + oy * ICON_PITCH:ICON_Y + oy * ICON_PITCH + 2, bx + ox * ICON_PITCH:bx + ox * ICON_PITCH + 2] = COLOURS[i]
            if i == self._sel:
                px[ICON_Y + 17:ICON_Y + 19, bx:bx + 15] = MARK
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        acted = False
        delta = MOVES.get(self._action.id)
        if delta is not None:
            ox, oy = self._off[self._sel]
            self._off[self._sel] = ((ox + delta[0]) % N, (oy + delta[1]) % N)
            acted = True
        elif self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            px, py = int(data.get("x", -1)), int(data.get("y", -1))
            if ICON_Y <= py < ICON_Y + 16:
                i = px // ICON_STRIDE
                if i < len(self._off) and px % ICON_STRIDE < 15 and i != self._sel:
                    self._sel = i
                    acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if composite(self._cfg["layers"], self._off) == self._target:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
