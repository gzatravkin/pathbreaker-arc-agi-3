import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
GID = "ng60-5a600001"
ACTIONS = [6]
MOVES = []
N = 9
PITCH = 6
OX = 5
OY = 7
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

CONFIGS = [
    dict(magnets=[(4, 0), (0, 1), (1, 8)], balls=[(4, 4)], targets=[(1, 7)], walls=[], budget=12),
    dict(magnets=[(7, 2), (7, 5), (0, 7), (2, 1)], balls=[(7, 7), (1, 2)], targets=[(7, 6), (2, 2)], walls=[(0, 3), (2, 0)], budget=8),
    dict(magnets=[(1, 7), (1, 8), (3, 6)], balls=[(3, 8)], targets=[(2, 7)], walls=[], budget=8),
    dict(magnets=[(3, 2), (4, 6), (6, 2), (2, 4)], balls=[(7, 4), (4, 0)], targets=[(3, 3), (4, 5)], walls=[(1, 5), (2, 1), (7, 6)], budget=10),
    dict(magnets=[(7, 5), (5, 7), (3, 7), (3, 2), (4, 0)], balls=[(7, 1), (6, 0)], targets=[(7, 4), (6, 5)], walls=[(8, 5), (8, 7), (2, 0), (2, 3)], budget=12),
    dict(magnets=[(1, 5), (6, 3), (5, 5), (4, 2), (0, 3), (4, 3)], balls=[(3, 8), (6, 0), (4, 6)], targets=[(3, 8), (5, 3), (4, 6)], walls=[(6, 6), (0, 1), (1, 8), (3, 3)], budget=14),
    dict(magnets=[(7, 8), (5, 4), (5, 0), (3, 2), (7, 5), (3, 7), (7, 1)], balls=[(2, 7), (4, 5), (2, 1)], targets=[(2, 7), (6, 5), (4, 2)], walls=[(5, 7), (8, 4), (8, 7), (6, 7), (1, 2)], budget=18),
]

SOLUTIONS = [[(31, 9), (31, 9), (7, 15), (7, 15), (13, 57)],
    [(49, 39), (19, 15), (49, 21)],
    [(13, 51), (25, 45), (25, 45)],
    [(19, 33), (31, 45), (25, 21), (19, 33)],
    [(49, 39), (37, 51), (31, 9), (31, 9), (37, 51)],
    [(43, 27), (43, 27), (37, 39), (31, 21), (43, 27), (31, 21)],
    [(37, 33), (37, 9), (49, 39), (49, 15), (25, 21), (49, 15), (37, 9), (37, 33)]]

def init(cfg):
    return (tuple(0 for _ in cfg["magnets"]), tuple(tuple(b) for b in cfg["balls"]), cfg["budget"])


def pull(cfg, on, balls, i):
    x, y = balls[i]
    mags = {tuple(m): k for k, m in enumerate(cfg["magnets"])}
    walls = {tuple(w) for w in cfg["walls"]}
    best = None
    tie = False
    for dx, dy in DIRS:
        cx, cy = x, y
        dist = 0
        while True:
            cx, cy = cx + dx, cy + dy
            dist += 1
            if not (0 <= cx < N and 0 <= cy < N) or (cx, cy) in walls:
                break
            if (cx, cy) in mags:
                if on[mags[(cx, cy)]]:
                    if best is None or dist < best[0]:
                        best, tie = (dist, dx, dy), False
                    elif dist == best[0]:
                        tie = True
                break
    if best is None or tie or best[0] < 2:
        return None
    return best[1], best[2]


def settle(cfg, on, balls):
    mags = {tuple(m) for m in cfg["magnets"]}
    walls = {tuple(w) for w in cfg["walls"]}
    frames = []
    balls = list(balls)
    for _ in range(40):
        moved = False
        for i in range(len(balls)):
            d = pull(cfg, on, balls, i)
            if d is None:
                continue
            nx, ny = balls[i][0] + d[0], balls[i][1] + d[1]
            if (nx, ny) in mags or (nx, ny) in walls or (nx, ny) in balls:
                continue
            balls[i] = (nx, ny)
            moved = True
        if not moved:
            break
        frames.append(tuple(balls))
    return frames


def transition(cfg, state, key):
    on, balls, left = state
    if not isinstance(key, tuple):
        return [state]
    cx, cy = (key[0] - OX) // PITCH, (key[1] - OY) // PITCH
    if key[0] < OX or key[1] < OY:
        return [state]
    hit = None
    for i, m in enumerate(cfg["magnets"]):
        if tuple(m) == (cx, cy):
            hit = i
    if hit is None:
        return [state]
    o = list(on)
    o[hit] ^= 1
    o = tuple(o)
    frames = settle(cfg, o, balls)
    out = [(o, balls, left)]
    for f in frames:
        out.append((o, f, left))
    last = frames[-1] if frames else balls
    out.append((o, last, left - 1))
    return out


def CLICKS(cfg):
    return [(OX + m[0] * PITCH + 2, OY + m[1] * PITCH + 2) for m in cfg["magnets"]]


def status(cfg, state):
    on, balls, left = state
    if sorted(map(tuple, balls)) == sorted(tuple(t) for t in cfg["targets"]):
        return "win"
    if left <= 0:
        return "lose"
    return None


def render(cfg, state):
    on, balls, left = state
    px = blank()
    bar(px, 1, left * 2, 14)
    rect(px, OX - 1, OY - 1, N * PITCH + 1, N * PITCH + 1, 4)
    for y in range(N):
        for x in range(N):
            px[OY + y * PITCH + 2, OX + x * PITCH + 2] = 4
    for x, y in cfg["walls"]:
        rect(px, OX + x * PITCH, OY + y * PITCH, 5, 5, 2)
    for x, y in cfg["targets"]:
        bx, by = OX + x * PITCH, OY + y * PITCH
        rect(px, bx, by, 5, 5, 11)
        rect(px, bx + 1, by + 1, 3, 3, 5)
    for i, (x, y) in enumerate(cfg["magnets"]):
        bx, by = OX + x * PITCH, OY + y * PITCH
        if on[i]:
            rect(px, bx, by, 5, 5, 8)
            rect(px, bx + 1, by + 1, 3, 3, 12)
            px[by + 2, bx + 2] = 0
        else:
            rect(px, bx, by, 5, 5, 3)
            rect(px, bx + 1, by + 1, 3, 3, 5)
    for x, y in balls:
        bx, by = OX + x * PITCH, OY + y * PITCH
        rect(px, bx + 1, by + 1, 3, 3, 9)
        px[by + 2, bx + 2] = 10
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


class Ng60(ARCBaseGame):
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
