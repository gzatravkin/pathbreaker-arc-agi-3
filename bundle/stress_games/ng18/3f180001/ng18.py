import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
EMPTY = 3
ZERO = 2
DIAG = 15
FILL_ONE = [11]
FILL_TWO = [8, 9]
PIP = 2
PIP_STEP = 3
GROUP_GAP = 2
ORIGIN_X = 22
ORIGIN_Y = 22

MODES = ["sum", "sum", "sum2", "sum", "runs", "diag"]
PICS = [
    ['.aaa.', 'aaaa.', 'aaaaa', '.aaa.', '..a..'],
    ['aa...a', 'aa.a.a', '.aa...', 'aa....', 'aa.aaa', '.aaa..'],
    ['ba.aab', 'ab.a.a', '.a.baa', 'b.....', 'bab.b.', '.b.ba.'],
    ['aa...aaa', '.a.a.aa.', '..a.....', 'aa.aaa.a', '.a...aaa', '......aa', 'aa...aaa', 'aaa.a...'],
    ['.a.a.aa.', '...a..aa', '.aaa.a..', '.a.a...a', '..aa..a.', '.....a.a', 'a...aaa.', '..a.....'],
    ['aa...aa', 'a.a.a.a', 'a...aa.', '...aa.a', 'aa.aaa.', '..aaa..', '....aaa'],
]


def fills_of(mode):
    return FILL_TWO if mode == "sum2" else FILL_ONE


def parse(rows):
    n = len(rows)
    out = np.zeros((n, n), dtype=np.int8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            out[y, x] = {".": 0, "a": 1, "b": 2}[ch]
    return out


def runs_of(line):
    out, cur = [], 0
    for v in line:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out or [0]


def clues_of(mode, a):
    n = a.shape[0]
    rows, cols = [], []
    if mode == "runs":
        rows = [[(g, 0) for g in runs_of(a[y, :])] for y in range(n)]
        cols = [[(g, 0) for g in runs_of(a[:, x])] for x in range(n)]
    elif mode == "sum2":
        rows = [[(int((a[y, :] == 1).sum()), 0), (int((a[y, :] == 2).sum()), 1)] for y in range(n)]
        cols = [[(int((a[:, x] == 1).sum()), 0), (int((a[:, x] == 2).sum()), 1)] for x in range(n)]
    elif mode == "sum":
        rows = [[(int((a[y, :] > 0).sum()), 0)] for y in range(n)]
        cols = [[(int((a[:, x] > 0).sum()), 0)] for x in range(n)]
    else:
        rows = [[(int((a[y, :] > 0).sum()), 0)] for y in range(n)]
        top = [[(sum(1 for y in range(n) if x + y < n and a[y, x + y] > 0 and False) , 0)] for x in range(n)]
        top = []
        for d in range(n):
            top.append([(sum(1 for y in range(n) if 0 <= y + d < n and a[y, y + d] > 0), 0)])
        bottom = []
        for x in range(n - 1):
            d = x - (n - 1)
            bottom.append([(sum(1 for y in range(n) if 0 <= y + d < n and a[y, y + d] > 0), 0)])
        return rows, top, bottom
    return rows, cols, None


def satisfied(mode, a, clues):
    return clues_of(mode, a) == clues


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(MODES))]


def layout(n):
    pitch = 5 if n <= 8 else 4
    return pitch


class Ng18(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._mode = MODES[0]
        self._n = 5
        self._pitch = 5
        self._clues = None
        self._grid = None
        self._ox = ORIGIN_X
        self._oy = ORIGIN_Y
        super().__init__(
            game_id="ng18-3f180001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        i = self._current_level_index
        self._mode = MODES[i]
        pic = parse(PICS[i])
        self._n = pic.shape[0]
        self._pitch = layout(self._n)
        self._clues = clues_of(self._mode, pic)
        self._grid = np.zeros((self._n, self._n), dtype=np.int8)
        side = self._n * self._pitch
        self._ox = 62 - side
        self._oy = 62 - side
        self._draw(level)

    def _pips(self, px, groups, fixed, end, vertical, hue):
        fills = fills_of(self._mode)
        total = 0
        for gi, (count, colour_index) in enumerate(groups):
            total += max(1, count) * PIP_STEP + (GROUP_GAP if gi else 0)
        pos = end - total - 1
        for gi, (count, colour_index) in enumerate(groups):
            if gi:
                pos += GROUP_GAP
            colour = hue if hue is not None else fills[colour_index]
            if count == 0:
                colour = ZERO
            for _ in range(max(1, count)):
                if vertical:
                    px[pos:pos + PIP, fixed:fixed + PIP] = colour
                else:
                    px[fixed:fixed + PIP, pos:pos + PIP] = colour
                pos += PIP_STEP

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        n, p = self._n, self._pitch
        fills = fills_of(self._mode)
        for y in range(n):
            for x in range(n):
                v = int(self._grid[y, x])
                colour = fills[v - 1] if v else EMPTY
                px[self._oy + y * p:self._oy + y * p + p - 1, self._ox + x * p:self._ox + x * p + p - 1] = colour
        rows, second, third = self._clues
        for y in range(n):
            self._pips(px, rows[y], self._oy + y * p + (p - 1 - PIP) // 2, self._ox - 2, False, None)
        for x in range(n):
            if self._mode == "diag":
                self._pips(px, second[x], self._ox + x * p + (p - 1 - PIP) // 2, self._oy - 2, True, DIAG)
            else:
                self._pips(px, second[x], self._ox + x * p + (p - 1 - PIP) // 2, self._oy - 2, True, None)
        if third is not None:
            for x in range(n - 1):
                total = max(1, third[x][0][0]) * PIP_STEP
                start = self._oy + n * p + 1
                for k in range(max(1, third[x][0][0])):
                    colour = ZERO if third[x][0][0] == 0 else DIAG
                    px[start + k * PIP_STEP:start + k * PIP_STEP + PIP,
                       self._ox + x * p + (p - 1 - PIP) // 2:self._ox + x * p + (p - 1 - PIP) // 2 + PIP] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            n, p = self._n, self._pitch
            cx, cy = (x - self._ox) // p, (y - self._oy) // p
            inside = x >= self._ox and y >= self._oy and 0 <= cx < n and 0 <= cy < n
            if inside and (x - self._ox) % p != p - 1 and (y - self._oy) % p != p - 1:
                self._grid[cy, cx] = (int(self._grid[cy, cx]) + 1) % (len(fills_of(self._mode)) + 1)
                self._draw()
                if satisfied(self._mode, self._grid, self._clues):
                    self.next_level()
        self.complete_action()


def solve(index):
    pic = parse(PICS[index])
    n = pic.shape[0]
    p = layout(n)
    ox = oy = 62 - n * p
    steps = []
    for y in range(n):
        for x in range(n):
            for _ in range(int(pic[y, x])):
                steps.append((ox + x * p + 1, oy + y * p + 1))
    return steps


SOLUTIONS = [solve(i) for i in range(len(MODES))]
