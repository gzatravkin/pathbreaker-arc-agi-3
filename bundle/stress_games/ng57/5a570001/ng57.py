import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
GID = "ng57-5a570001"
ACTIONS = [6]
MOVES = []
G = 9
PITCH = 6
OX = 5
OY = 7
FADE = 3
DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, -1), "S": (0, 1)}
SLASH = {"E": "N", "N": "E", "W": "S", "S": "W"}
BACKSLASH = {"E": "S", "S": "E", "W": "N", "N": "W"}

CONFIGS = [
    dict(emit=(4, 5, 'W'), mirrors=[(2, 8), (1, 5), (3, 0), (5, 7), (4, 1), (7, 0), (0, 4), (6, 4), (8, 6)], prisms=[(1, 0)], walls=[], recv=[(0, 0), (8, 0)], budget=12),
    dict(emit=(1, 2, 'S'), mirrors=[(0, 3), (1, 7), (5, 8), (2, 3), (7, 0), (8, 1), (6, 5), (4, 6), (3, 1), (1, 4), (8, 6)], prisms=[(1, 6)], walls=[], recv=[(8, 8), (0, 6)], budget=12),
    dict(emit=(5, 1, 'E'), mirrors=[(3, 7), (0, 8), (6, 8), (4, 0), (7, 4), (1, 4), (2, 1), (5, 4), (8, 6)], prisms=[(8, 4), (8, 1)], walls=[], recv=[(8, 0), (7, 0), (1, 0)], budget=12),
    dict(emit=(0, 4, 'N'), mirrors=[(2, 2), (7, 8), (8, 6), (6, 3), (5, 1), (4, 0), (3, 2), (1, 1), (0, 2)], prisms=[(4, 8), (8, 3)], walls=[], recv=[(2, 8), (0, 0), (3, 8)], budget=14),
    dict(emit=(5, 6, 'S'), mirrors=[(5, 7), (4, 0), (2, 8), (6, 7), (0, 3), (7, 3), (1, 6), (8, 7), (3, 8), (4, 8), (2, 4), (4, 3)], prisms=[(1, 4), (8, 4)], walls=[(4, 2)], recv=[(0, 6), (6, 8), (0, 8)], budget=14),
]

SOLUTIONS = [[(13, 39), (25, 9), (25, 9), (49, 9), (49, 9)],
    [(13, 51), (13, 51), (55, 15), (31, 45), (31, 45)],
    [(49, 33), (13, 33), (37, 33), (49, 33), (37, 33)],
    [(19, 21), (25, 21), (7, 21), (7, 21), (7, 21), (19, 21)],
    [(37, 51), (19, 57), (19, 57), (43, 51), (43, 51), (19, 57)]]

def init(cfg):
    return (tuple(0 for _ in cfg["mirrors"]), tuple(0 for _ in cfg["recv"]), cfg["budget"])


def trace(cfg, orient):
    ex, ey, ed = cfg["emit"]
    mir = {tuple(m): i for i, m in enumerate(cfg["mirrors"])}
    prisms = {tuple(p) for p in cfg["prisms"]}
    walls = {tuple(w) for w in cfg["walls"]}
    recv = {tuple(r): i for i, r in enumerate(cfg["recv"])}
    rays = [((ex, ey), ed)]
    seen = set()
    cells = []
    hit = set()
    while rays:
        (x, y), d = rays.pop()
        for _ in range(60):
            dx, dy = DIRS[d]
            x, y = x + dx, y + dy
            if not (0 <= x < G and 0 <= y < G) or (x, y) in walls:
                break
            if ((x, y), d) in seen:
                break
            seen.add(((x, y), d))
            cells.append(((x, y), d))
            if (x, y) in recv:
                hit.add(recv[(x, y)])
                break
            if (x, y) in prisms:
                for nd in ("N", "S") if d in "EW" else ("E", "W"):
                    rays.append(((x, y), nd))
                break
            if (x, y) in mir and orient[mir[(x, y)]] != 2:
                table = BACKSLASH if orient[mir[(x, y)]] else SLASH
                d = table[d]
                cells.append(((x, y), d))
    return cells, hit


def transition(cfg, state, key):
    orient, charge, left = state
    if not isinstance(key, tuple):
        return [state]
    cx, cy = (key[0] - OX) // PITCH, (key[1] - OY) // PITCH
    if not (0 <= key[0] - OX and 0 <= key[1] - OY and 0 <= cx < G and 0 <= cy < G):
        return [state]
    hit_index = None
    for i, m in enumerate(cfg["mirrors"]):
        if tuple(m) == (cx, cy):
            hit_index = i
    if hit_index is None:
        return [state]
    o = list(orient)
    for i, m in enumerate(cfg["mirrors"]):
        if m[0] == cx:
            o[i] = (o[i] + 1) % 3
    _, hit = trace(cfg, o)
    new = tuple(FADE if i in hit else max(0, c - 1) for i, c in enumerate(charge))
    return [(tuple(o), new, left - 1)]


def CLICKS(cfg):
    return [(OX + m[0] * PITCH + 2, OY + m[1] * PITCH + 2) for m in cfg["mirrors"]]


def status(cfg, state):
    orient, charge, left = state
    if all(c > 0 for c in charge):
        return "win"
    if left <= 0:
        return "lose"
    return None


def render(cfg, state):
    orient, charge, left = state
    px = blank()
    bar(px, 1, left * 2, 14)
    rect(px, OX - 1, OY - 1, G * PITCH + 1, G * PITCH + 1, 4)
    for y in range(G):
        for x in range(G):
            rect(px, OX + x * PITCH, OY + y * PITCH, PITCH - 1, PITCH - 1, 5)
    cells, hit = trace(cfg, orient)
    for (x, y), d in cells:
        cx, cy = OX + x * PITCH + 2, OY + y * PITCH + 2
        dx, dy = DIRS[d]
        rect(px, cx - (2 if dx < 0 else 0) - (0), cy - (2 if dy < 0 else 0), 1, 1, 11)
        if dx:
            rect(px, cx - 2 if dx < 0 else cx, cy, 3, 1, 11)
        else:
            rect(px, cx, cy - 2 if dy < 0 else cy, 1, 3, 11)
    for x, y in cfg["walls"]:
        rect(px, OX + x * PITCH, OY + y * PITCH, PITCH - 1, PITCH - 1, 3)
    for x, y in cfg["prisms"]:
        bx, by = OX + x * PITCH, OY + y * PITCH
        rect(px, bx + 2, by, 1, 5, 15)
        rect(px, bx + 1, by + 1, 3, 3, 15)
        rect(px, bx, by + 2, 5, 1, 15)
    for i, (x, y) in enumerate(cfg["mirrors"]):
        bx, by = OX + x * PITCH, OY + y * PITCH
        if orient[i] == 2:
            px[by + 2, bx + 2] = 3
            continue
        for k in range(5):
            xx = bx + (4 - k if orient[i] == 0 else k)
            px[by + k, xx] = 0
    for i, (x, y) in enumerate(cfg["recv"]):
        bx, by = OX + x * PITCH, OY + y * PITCH
        rect(px, bx, by, 5, 5, 10)
        rect(px, bx + 1, by + 1, 3, 3, 5)
        if charge[i] > 0:
            rect(px, bx + 1, by + 4 - charge[i], 3, charge[i], 14)
    ex, ey, ed = cfg["emit"]
    bx, by = OX + ex * PITCH, OY + ey * PITCH
    rect(px, bx, by, 5, 5, 8)
    dx, dy = DIRS[ed]
    rect(px, bx + 2 + 2 * dx, by + 2 + 2 * dy, 1, 1, 0)
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


class Ng57(ARCBaseGame):
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
