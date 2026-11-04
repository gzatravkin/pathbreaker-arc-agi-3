import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
GID = "ng58-5a580001"
ACTIONS = [1, 2, 3, 4, 5]
MOVES = ["U", "D", "L", "R", "S"]
N = 9
PITCH = 6
OX = 5
OY = 8
GROUND, TREE, WET, HOUSE, ROCK, FIRE1, FIRE2, ASH = range(8)
COLOUR = {GROUND: 5, TREE: 14, WET: 10, HOUSE: 11, ROCK: 3, FIRE1: 12, FIRE2: 8, ASH: 4}
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1))

CONFIGS = [
    dict(trees=[(6, 3), (6, 6), (5, 3), (2, 8), (1, 3), (6, 8), (0, 3), (5, 4), (6, 1), (0, 0), (8, 2), (3, 7), (5, 1), (5, 6), (1, 5), (4, 1), (0, 6), (7, 1), (2, 2), (0, 1), (7, 3), (6, 0), (4, 5), (8, 3), (7, 2), (2, 6), (4, 7), (1, 4), (3, 0), (5, 8), (3, 6), (8, 1), (3, 1), (3, 4), (3, 8), (8, 7), (2, 4), (4, 3), (6, 5), (3, 2)], wet=[], houses=[(8, 0)], rocks=[(0, 8), (3, 5), (1, 6), (7, 0)], fire=[(1, 8)], cursor=(4, 4), budget=4),
    dict(trees=[(0, 0), (8, 5), (5, 3), (0, 3), (2, 0), (5, 6), (1, 4), (5, 5), (7, 4), (0, 1), (1, 2), (4, 5), (3, 1), (5, 2), (3, 0), (0, 5), (2, 8), (4, 8), (1, 8), (8, 1), (4, 2), (7, 0), (8, 3), (3, 8), (6, 4), (3, 6), (5, 1), (2, 1), (1, 7), (5, 8), (8, 2), (7, 2), (8, 8), (2, 6)], wet=[(1, 5), (3, 3), (8, 4), (6, 6), (6, 3), (3, 7)], houses=[(1, 1)], rocks=[(4, 7), (7, 8), (7, 6)], fire=[(3, 4), (2, 3)], cursor=(4, 4), budget=4),
    dict(trees=[(3, 7), (4, 5), (5, 1), (2, 4), (3, 0), (6, 0), (6, 7), (4, 6), (3, 5), (6, 4), (4, 2), (8, 6), (1, 2), (7, 3), (1, 7), (5, 3), (8, 3), (7, 2), (8, 7), (3, 4), (4, 8), (4, 3), (2, 0), (0, 1), (0, 4), (3, 1), (5, 5), (5, 0), (0, 6), (4, 1), (8, 1), (2, 5)], wet=[(0, 7), (4, 7), (4, 4), (1, 4), (2, 8), (7, 4), (6, 5), (2, 6)], houses=[(2, 7)], rocks=[(0, 0), (5, 2), (8, 4), (0, 5)], fire=[(5, 4), (6, 6)], cursor=(4, 4), budget=6),
    dict(trees=[(8, 0), (3, 4), (0, 5), (7, 1), (5, 3), (1, 7), (5, 8), (4, 3), (1, 1), (6, 4), (2, 2), (6, 7), (0, 4), (3, 7), (4, 5), (8, 6), (4, 6), (0, 2), (1, 4), (5, 6), (7, 2), (8, 8), (5, 5), (0, 6), (1, 6), (1, 0), (6, 6), (2, 0), (1, 8), (3, 6), (1, 3), (4, 8), (6, 3), (4, 1), (2, 7), (4, 4), (5, 0), (2, 8)], wet=[(4, 0), (0, 1), (6, 2), (6, 5), (1, 5), (0, 3), (8, 7), (6, 8), (7, 6), (4, 7)], houses=[(2, 5), (3, 0)], rocks=[(2, 3), (2, 6), (8, 4), (1, 2), (2, 1)], fire=[(5, 4), (8, 2)], cursor=(4, 4), budget=6),
    dict(trees=[(5, 3), (1, 7), (8, 5), (2, 2), (5, 6), (0, 1), (5, 0), (4, 0), (4, 3), (1, 5), (5, 4), (5, 7), (1, 2), (6, 2), (7, 4), (6, 6), (8, 3), (4, 6), (4, 1), (0, 2), (0, 3), (3, 6), (1, 1), (7, 7), (1, 8), (4, 8), (2, 1), (3, 7), (3, 8), (8, 2), (1, 4), (8, 4), (3, 3), (6, 7), (5, 8), (0, 4)], wet=[(7, 0), (7, 5), (6, 4), (6, 5), (6, 0), (8, 0), (2, 6), (2, 4), (3, 5), (7, 2), (0, 7), (6, 1)], houses=[(2, 8), (2, 3), (8, 1)], rocks=[(7, 6), (7, 3), (6, 3), (8, 6), (4, 2), (5, 1)], fire=[(3, 1), (2, 0), (5, 2)], cursor=(4, 4), budget=6),
]

SOLUTIONS = [['D', 'D', 'D', 'D', 'L', 'L', 'S', 'S'],
    ['S', 'S', 'S'],
    ['U', 'S', 'D', 'D', 'S', 'S', 'S'],
    ['D', 'D', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['U', 'U', 'L', 'L', 'L', 'L', 'S', 'S', 'S', 'S', 'S']]

def init(cfg):
    grid = [GROUND] * (N * N)
    for ch, cells in (("trees", TREE), ("wet", WET), ("houses", HOUSE), ("rocks", ROCK), ("fire", FIRE1)):
        for x, y in cfg[ch]:
            grid[y * N + x] = cells
    return (tuple(grid), cfg["cursor"][0], cfg["cursor"][1], cfg["budget"])


def spread(grid):
    out = list(grid)
    for y in range(N):
        for x in range(N):
            v = grid[y * N + x]
            if v == FIRE1:
                out[y * N + x] = FIRE2
            elif v == FIRE2:
                out[y * N + x] = ASH
            elif v in (TREE, WET, HOUSE):
                burning = 0
                for dx, dy in NEIGH:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and grid[ny * N + nx] in (FIRE1, FIRE2):
                        burning += 1
                if burning >= (2 if v == WET else 1):
                    out[y * N + x] = FIRE1
    return tuple(out)


def transition(cfg, state, key):
    grid, cx, cy, left = state
    if key == "U":
        return [(grid, cx, max(0, cy - 1), left)]
    if key == "D":
        return [(grid, cx, min(N - 1, cy + 1), left)]
    if key == "L":
        return [(grid, max(0, cx - 1), cy, left)]
    if key == "R":
        return [(grid, min(N - 1, cx + 1), cy, left)]
    if key == "S":
        g = list(grid)
        if g[cy * N + cx] in (TREE, WET):
            g[cy * N + cx] = GROUND
        g = tuple(g)
        frames = [(g, cx, cy, left)]
        nxt = spread(g)
        frames.append((nxt, cx, cy, left - 1))
        return frames
    return [state]


def status(cfg, state):
    grid, cx, cy, left = state
    if HOUSE not in grid and cfg["houses"]:
        return "lose"
    if not any(v in (FIRE1, FIRE2) for v in grid):
        return "win"
    if left <= 0:
        return "lose"
    return None


def render(cfg, state):
    grid, cx, cy, left = state
    px = blank()
    bar(px, 1, left * 3, 14)
    for y in range(N):
        for x in range(N):
            v = grid[y * N + x]
            bx, by = OX + x * PITCH, OY + y * PITCH
            if v == GROUND:
                px[by + 2, bx + 2] = 4
                continue
            rect(px, bx, by, 5, 5, COLOUR[v])
            if v == HOUSE:
                rect(px, bx + 2, by + 3, 1, 2, 5)
            if v == WET:
                px[by + 2, bx + 2] = 9
    bx, by = OX + cx * PITCH - 1, OY + cy * PITCH - 1
    rect(px, bx, by, 7, 1, 15)
    rect(px, bx, by + 6, 7, 1, 15)
    rect(px, bx, by, 1, 7, 15)
    rect(px, bx + 6, by, 1, 7, 15)
    return px


def rect(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    px[y0:max(y0, min(64, y + h)), x0:max(x0, min(64, x + w))] = c


def blank():
    return np.full((64, 64), BG, dtype=np.int8)


def bar(px, y, n, c, x=2):
    px[y, x:x + max(0, min(n, 60))] = c


KEYS = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}

LEVELS = [
    Level(
        sprites=[
            Sprite(
                pixels=np.full((64, 64), BG, dtype=np.int8),
                name="canvas",
                x=0,
                y=0,
                layer=0,
                blocking=BlockingMode.NOT_BLOCKED,
                interaction=InteractionMode.TANGIBLE,
                tags=["canvas"],
            )
        ],
        grid_size=(64, 64),
        name="Level %d" % (i + 1),
    )
    for i in range(len(CONFIGS))
]


class Ng58(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = None
        self._queue = None
        super().__init__(
            game_id=GID,
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=ACTIONS,
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._st = init(self._cfg)
        self._queue = None
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        level.get_sprites_by_name("canvas")[0].pixels = render(self._cfg, self._st)

    def step(self) -> None:
        if self._queue is None:
            if self._action.id == GameAction.ACTION6:
                data = self._action.data or {}
                key = (int(data.get("x", -1)), int(data.get("y", -1)))
            else:
                key = KEYS.get(self._action.id)
            self._queue = list(transition(self._cfg, self._st, key)) if key is not None else []
            self._queue.reverse()
        if self._queue:
            self._st = self._queue.pop()
            self._draw()
            if self._queue:
                return
        else:
            self._draw()
        self._queue = None
        outcome = status(self._cfg, self._st)
        if outcome == "win":
            self.next_level()
        elif outcome == "lose":
            self.lose()
        self.complete_action()
