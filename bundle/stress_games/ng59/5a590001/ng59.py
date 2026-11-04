import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
GID = "ng59-5a590001"
ACTIONS = [2, 3, 4, 5]
MOVES = ["D", "L", "R", "S"]
W = 11
A = 11
CELL = 5
OX = 2
OY = 5
BURN = 3

CONFIGS = [
    dict(drift=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], start=(7, 0), rings=[(2, 1)], budget=11),
    dict(drift=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1], start=(8, 0), rings=[(7, 1), (10, 10)], budget=17),
    dict(drift=[0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1], start=(8, 0), rings=[(2, 2), (4, 5), (1, 4)], budget=21),
    dict(drift=[0, 0, 1, 1, 0, 1, 1, 1, -1, 1, -1], start=(8, 0), rings=[(0, 1), (2, 1), (10, 4)], budget=26),
    dict(drift=[0, -1, 1, 1, 1, 0, 0, 0, 0, 0, 1], start=(5, 0), rings=[(8, 10), (9, 9), (10, 10), (10, 3)], budget=29),
]

SOLUTIONS = [['L', 'L', 'L', 'L', 'L', 'S'],
    ['L', 'S', 'S', 'L', 'L', 'S', 'L', 'R', 'S', 'R', 'R'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'S', 'S', 'R', 'R', 'S', 'D', 'L', 'S'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'S', 'R', 'S', 'D', 'S', 'S', 'R', 'R', 'R', 'R'],
    ['S', 'S', 'L', 'L', 'S', 'L', 'L', 'S', 'R', 'R', 'R', 'R', 'S', 'D', 'D', 'D', 'D', 'D', 'D', 'D']]

def init(cfg):
    return (cfg["start"][0], cfg["start"][1], 0, 0, cfg["budget"])


def tick(cfg, x, alt, heat):
    climb = 1 if heat >= 3 else (-1 if heat <= 1 else 0)
    heat = max(0, heat - 1)
    alt = max(0, min(A - 1, alt + climb))
    mid = (x, alt, heat)
    x = max(0, min(W - 1, x + cfg["drift"][alt]))
    return mid, (x, alt, heat)


def transition(cfg, state, key):
    x, alt, heat, idx, left = state
    if key == "S":
        heat = min(2 * BURN, heat + BURN)
    elif key == "D":
        heat = max(0, heat - BURN)
    elif key == "L":
        x = max(0, x - 1)
    elif key == "R":
        x = min(W - 1, x + 1)
    else:
        return [state]
    mid, end = tick(cfg, x, alt, heat)
    frames = [(mid[0], mid[1], mid[2], idx, left)]
    nx, na, nh = end
    nidx = idx
    if nidx < len(cfg["rings"]) and (nx, na) == tuple(cfg["rings"][nidx]):
        nidx += 1
    frames.append((nx, na, nh, nidx, left - 1))
    return frames


def status(cfg, state):
    x, alt, heat, idx, left = state
    if idx >= len(cfg["rings"]):
        return "win"
    if left <= 0:
        return "lose"
    return None


def render(cfg, state):
    x, alt, heat, idx, left = state
    px = blank()
    bar(px, 1, left * 2, 14)
    rect(px, OX - 1, OY - 1, W * CELL + 2, A * CELL + 2, 4)
    rect(px, OX, OY, W * CELL, A * CELL, 5)
    rect(px, OX, OY + (A - 1) * CELL + CELL - 1, W * CELL, 1, 13)
    for a in range(A):
        d = cfg["drift"][a]
        y = OY + (A - 1 - a) * CELL
        for cx in range(1, W, 3):
            bx = OX + cx * CELL
            if d > 0:
                px[y + 1, bx + 1] = 3; px[y + 2, bx + 2] = 3; px[y + 3, bx + 1] = 3
            elif d < 0:
                px[y + 1, bx + 2] = 3; px[y + 2, bx + 1] = 3; px[y + 3, bx + 2] = 3
    for k, (rx, ra) in enumerate(cfg["rings"]):
        if k < idx:
            continue
        bx, by = OX + rx * CELL, OY + (A - 1 - ra) * CELL
        c = 11 if k == idx else 2
        rect(px, bx, by, 5, 5, c)
        rect(px, bx + 1, by + 1, 3, 3, 5)
        for j in range(k - idx + 1):
            px[by + 2, bx + 1 + j] = c
    bx, by = OX + x * CELL, OY + (A - 1 - alt) * CELL
    rect(px, bx + 1, by, 3, 3, 8)
    px[by + 3, bx + 2] = 12 if heat >= 3 else 3
    px[by + 4, bx + 2] = 3
    rect(px, 59, 60 - 2 * heat * 2, 3, 2 * heat * 2, 12)
    rect(px, 58, 60 - 12 * 2, 1, 24, 2)
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


class Ng59(ARCBaseGame):
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
