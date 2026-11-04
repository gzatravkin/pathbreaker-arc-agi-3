import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
EMPTY_A = 3
EMPTY_B = 4
SEL = 0
LINE = 2
GRID = 10
PITCH = 3
LEFT_X = 1
RIGHT_X = 33
LOW_Y = 33
HIGH_Y = 1
SWATCH_Y = 4
SWATCH_X = 4
SWATCH_STEP = 8
SWATCH_SIZE = 5
CHARS = {".": 0, "r": 8, "b": 9, "y": 11, "g": 14}

PATTERNS = [
    [
        "..........",
        ".r.r......",
        ".rrr......",
        ".r.r......",
        "..........",
        ".rrrrr....",
        "..r.......",
        "..r.......",
        "..........",
        "..........",
    ],
    [
        "..........",
        ".rrr......",
        ".rbr..yyy.",
        ".rrr..yby.",
        "..........",
        "..bbbbb...",
        "..........",
        "..y.y.y.y.",
        "..........",
        "..........",
    ],
    [
        ".r..b.....",
        ".r..r..b..",
        ".b..r..b.r",
        "..........",
        ".r........",
        "...b....b.",
        "..........",
        ".r..b..b..",
        "..........",
        "..........",
    ],
    [
        "..........",
        "..rr......",
        "..rr...r..",
        "..........",
        "..........",
        ".....rrr..",
        "...r......",
        "...rr.....",
        "...r..r...",
        "..........",
    ],
    [
        "..........",
        ".rrb......",
        ".r.b..yyy.",
        ".ybb..y.b.",
        "..........",
        "..bbyy....",
        "..r..y....",
        "..r.......",
        "......rr..",
        "..........",
    ],
    [
        "..........",
        ".r.r.r....",
        "..r.r.....",
        ".r.r.r.r..",
        "..........",
        "...r.r....",
        "....r.....",
        ".r.r.r.r..",
        "..........",
        "..........",
    ],
]
PALETTES = [[8], [8, 9, 11], [8, 9], [8], [8, 9, 11], [8]]
KINDS = ["shift", "recolour", "gravity", "halo", "mirror_recolour", "neighbours"]
PERM_A = {8: 9, 9: 11, 11: 8}
PERM_B = {8: 11, 11: 9, 9: 8}


def parse(rows):
    out = np.zeros((GRID, GRID), dtype=np.int8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            out[y, x] = CHARS[ch]
    return out


def transform(kind, a):
    n = GRID
    out = np.zeros_like(a)
    if kind == "shift":
        for y in range(n):
            for x in range(n):
                out[y, (x + 3) % n] = a[y, x]
    elif kind == "recolour":
        for y in range(n):
            for x in range(n):
                out[y, x] = PERM_A.get(int(a[y, x]), 0)
    elif kind == "gravity":
        for x in range(n):
            col = [int(v) for v in a[:, x] if v]
            for i, v in enumerate(reversed(col)):
                out[n - 1 - i, x] = v
    elif kind == "halo":
        for y in range(n):
            for x in range(n):
                if a[y, x]:
                    continue
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and a[ny, nx]:
                        out[y, x] = a[ny, nx]
                        break
    elif kind == "mirror_recolour":
        for y in range(n):
            for x in range(n):
                out[y, n - 1 - x] = PERM_B.get(int(a[y, x]), 0)
    elif kind == "neighbours":
        for y in range(n):
            for x in range(n):
                c = 0
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and a[ny, nx]:
                        c += 1
                if c >= 2:
                    out[y, x] = 14
    return out


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(PATTERNS))]


def paint_grid(px, grid, x0, y0):
    for y in range(GRID):
        for x in range(GRID):
            colour = int(grid[y, x]) or (EMPTY_A if (x + y) % 2 == 0 else EMPTY_B)
            px[y0 + y * PITCH:y0 + y * PITCH + PITCH, x0 + x * PITCH:x0 + x * PITCH + PITCH] = colour


class Ng17(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._kind = KINDS[0]
        self._palette = PALETTES[0]
        self._left = np.zeros((GRID, GRID), dtype=np.int8)
        self._target = np.zeros((GRID, GRID), dtype=np.int8)
        self._brush = 0
        super().__init__(
            game_id="ng17-3f170001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        i = self._current_level_index
        self._kind = KINDS[i]
        self._palette = PALETTES[i]
        self._left = np.zeros((GRID, GRID), dtype=np.int8)
        self._target = transform(self._kind, parse(PATTERNS[i]))
        self._brush = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        paint_grid(px, self._left, LEFT_X, LOW_Y)
        paint_grid(px, transform(self._kind, self._left), RIGHT_X, LOW_Y)
        paint_grid(px, self._target, RIGHT_X, HIGH_Y)
        px[31, :] = LINE
        px[:, 31:33] = LINE
        px[31, 31:33] = LINE
        for i, colour in enumerate(self._palette):
            x = SWATCH_X + i * SWATCH_STEP
            if i == self._brush:
                px[SWATCH_Y - 2:SWATCH_Y + SWATCH_SIZE + 2, x - 2:x + SWATCH_SIZE + 2] = SEL
            px[SWATCH_Y:SWATCH_Y + SWATCH_SIZE, x:x + SWATCH_SIZE] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _click(self, x, y):
        for i in range(len(self._palette)):
            sx = SWATCH_X + i * SWATCH_STEP
            if SWATCH_Y <= y < SWATCH_Y + SWATCH_SIZE and sx <= x < sx + SWATCH_SIZE:
                self._brush = i
                return
        cx, cy = (x - LEFT_X) // PITCH, (y - LOW_Y) // PITCH
        if 0 <= cx < GRID and 0 <= cy < GRID and x >= LEFT_X and y >= LOW_Y:
            colour = self._palette[self._brush]
            self._left[cy, cx] = 0 if self._left[cy, cx] == colour else colour

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            self._click(int(data.get("x", -1)), int(data.get("y", -1)))
            self._draw()
            if np.array_equal(transform(self._kind, self._left), self._target):
                self.next_level()
        self.complete_action()


def solve(index):
    pat = parse(PATTERNS[index])
    pal = PALETTES[index]
    steps = []
    brush = 0
    for colour in pal:
        cells = [(x, y) for y in range(GRID) for x in range(GRID) if pat[y, x] == colour]
        if not cells:
            continue
        i = pal.index(colour)
        if i != brush:
            steps.append((SWATCH_X + i * SWATCH_STEP + 2, SWATCH_Y + 2))
            brush = i
        for x, y in cells:
            steps.append((LEFT_X + x * PITCH + 1, LOW_Y + y * PITCH + 1))
    return steps


SOLUTIONS = [solve(i) for i in range(len(PATTERNS))]
