import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
INK = 9
CURSOR = 11
N = 4
ANS_X = 40
ANS_Y = 26
ANS_PITCH = 5
TEST_X = 40
TEST_Y = 4
PAIR_Y = (3, 17, 31)
PAIR_PITCH = 3
IN_X = 2
OUT_X = 24
ARROW_X = 15
BAR_Y = 60
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def grid(rows):
    return tuple(tuple(int(ch) for ch in row) for row in rows)


def flip_h(g):
    return tuple(tuple(reversed(row)) for row in g)


def gravity_down(g):
    cols = []
    for x in range(N):
        cells = [g[y][x] for y in range(N) if g[y][x]]
        cols.append([0] * (N - len(cells)) + cells)
    return tuple(tuple(cols[x][y] for x in range(N)) for y in range(N))


def heights(hs):
    return tuple(tuple(1 if y < hs[x] else 0 for x in range(N)) for y in range(N))


def pairs(rule, inputs):
    return [(g, rule(g)) for g in inputs]


CONFIGS = []


def add(rule, ex_inputs, test_input):
    CONFIGS.append(dict(rule=rule, given=pairs(rule, ex_inputs), test=test_input, target=rule(test_input)))


add(
    flip_h,
    [grid(("1100", "0011", "0011", "1100")), grid(("0011", "1111", "1111", "0011")), grid(("1100", "0000", "0000", "1100"))],
    grid(("1111", "0011", "0011", "1111")),
)
add(
    flip_h,
    [grid(("1100", "1100", "0011", "0000")), grid(("0011", "0000", "1111", "1100")), grid(("1111", "0011", "0000", "0000"))],
    grid(("0011", "0011", "1100", "1111")),
)
add(
    flip_h,
    [grid(("1000", "0100", "1010", "0001")), grid(("0110", "1000", "0001", "1100")), grid(("0001", "0010", "0100", "1001"))],
    grid(("1011", "0100", "0010", "1000")),
)
add(
    gravity_down,
    [heights((3, 1, 1, 3)), heights((2, 0, 0, 2)), heights((1, 3, 3, 1))],
    heights((2, 3, 3, 2)),
)
add(
    gravity_down,
    [grid(("1010", "0100", "1001", "0010")), grid(("0110", "1001", "0000", "1010")), grid(("1001", "0000", "0111", "0100"))],
    grid(("0101", "1010", "0010", "1000")),
)


def solution_for(cfg):
    steps = []
    cx, cy = 0, 0
    for y in range(N):
        xs = range(N) if y % 2 == 0 else range(N - 1, -1, -1)
        for x in xs:
            if not cfg["target"][y][x]:
                continue
            while cy < y:
                steps.append("D")
                cy += 1
            while cx != x:
                steps.append("R" if x > cx else "L")
                cx += 1 if x > cx else -1
            steps.append("S")
    return steps


SOLUTIONS = [solution_for(cfg) for cfg in CONFIGS]


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


def blit(px, g, x0, y0, pitch, side):
    px[y0 - 1:y0 + N * pitch, x0 - 1:x0 + N * pitch] = 4
    for y in range(N):
        for x in range(N):
            px[y0 + y * pitch:y0 + y * pitch + side, x0 + x * pitch:x0 + x * pitch + side] = INK if g[y][x] else 3


class Ng54(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._cells = [[0] * N for _ in range(N)]
        self._cursor = (0, 0)
        self._left = 0
        self._budget = 1
        super().__init__(
            game_id="ng54-5a1c0054",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._cfg = CONFIGS[index]
        self._cells = [[0] * N for _ in range(N)]
        self._cursor = (0, 0)
        self._budget = 3 * len(SOLUTIONS[index]) + 12
        self._left = self._budget
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for k, (src, dst) in enumerate(self._cfg["given"]):
            y0 = PAIR_Y[k]
            blit(px, src, IN_X, y0, PAIR_PITCH, 3)
            blit(px, dst, OUT_X, y0, PAIR_PITCH, 3)
            px[y0 + 4:y0 + 7, ARROW_X:ARROW_X + 6] = 5
            px[y0 + 5, ARROW_X:ARROW_X + 7] = 2
            px[y0 + 4, ARROW_X + 5] = 2
            px[y0 + 6, ARROW_X + 5] = 2
        blit(px, self._cfg["test"], TEST_X, TEST_Y, 5, 4)
        blit(px, self._cells, ANS_X, ANS_Y, ANS_PITCH, 4)
        cx, cy = self._cursor
        x0 = ANS_X + cx * ANS_PITCH
        y0 = ANS_Y + cy * ANS_PITCH
        px[y0 - 1, x0 - 1:x0 + 5] = CURSOR
        px[y0 + 4, x0 - 1:x0 + 5] = CURSOR
        px[y0 - 1:y0 + 5, x0 - 1] = CURSOR
        px[y0 - 1:y0 + 5, x0 + 4] = CURSOR
        width = self._left * BAR_MAX // self._budget
        px[BAR_Y, 2:2 + width] = 14
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        delta = MOVES.get(aid)
        acted = False
        if delta is not None:
            x, y = self._cursor
            self._cursor = (min(N - 1, max(0, x + delta[0])), min(N - 1, max(0, y + delta[1])))
            acted = True
        elif aid == GameAction.ACTION5:
            x, y = self._cursor
            self._cells[y][x] ^= 1
            acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if tuple(tuple(r) for r in self._cells) == self._cfg["target"]:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
