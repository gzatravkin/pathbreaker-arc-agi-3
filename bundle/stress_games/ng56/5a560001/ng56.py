import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
GID = "ng56-5a560001"
ACTIONS = [1, 2, 3, 4, 5]
MOVES = ["L", "R", "S", "U", "D"]
ROWS = 12
UNIT = 3
BASE = 50
X0 = 3
GAP = 6
TAP = 3

CONFIGS = [
    dict(w=[2, 2, 2, 2], rows=[0, 0, 0], rate=[3, 3, 2], vol=[18, 14, 12, 10], target=[7, 7, 6, 4], budget=16),
    dict(w=[2, 1, 3], rows=[0, 0], rate=[2, 2], vol=[18, 3, 0], target=[6, 6, 1], budget=14),
    dict(w=[2, 1, 2], rows=[5, 2], rate=[2, 3], vol=[12, 0, 0], target=[5, 2, 1], budget=14),
    dict(w=[2, 3, 1, 1], rows=[3, 3, 3], rate=[2, 1, 1], vol=[6, 6, 0, 9], target=[4, 4, 4, 0], budget=24),
    dict(w=[2, 1, 2, 1, 1], rows=[2, 2, 1, 2], rate=[2, 3, 3, 1], vol=[8, 5, 8, 2, 1], target=[6, 4, 4, 9, 1], budget=26),
]

SOLUTIONS = [['R', 'R', 'S', 'D', 'L', 'S', 'L', 'S', 'D', 'S'],
    ['S', 'R', 'S', 'S', 'L', 'S', 'S'],
    ['R', 'U', 'S', 'L', 'S', 'R', 'D', 'S'],
    ['U', 'S', 'U', 'S', 'R', 'S', 'U', 'R', 'U', 'D', 'D', 'D'],
    ['R', 'U', 'L', 'S', 'R', 'U', 'R', 'D', 'L', 'L', 'S', 'R', 'R', 'R', 'U', 'L', 'L', 'S', 'D', 'R', 'R', 'U', 'L', 'L', 'S', 'R', 'R', 'U']]

def init(cfg):
    return (tuple(cfg["vol"]), 0, cfg["budget"])


def heights_ok(cfg, vols):
    return all(v == t * w for v, t, w in zip(vols, cfg["target"], cfg["w"]))


def pulse(cfg, vols, cursor):
    a, row = cursor, cfg["rows"][cursor]
    b = a + 1
    w = cfg["w"]
    v = list(vols)
    frames = []
    for _ in range(cfg["rate"][cursor]):
        if v[a] * w[b] == v[b] * w[a]:
            break
        src, dst = (a, b) if v[a] * w[b] > v[b] * w[a] else (b, a)
        if v[src] - 1 < row * w[src]:
            break
        if (v[src] - 1) * w[dst] < (v[dst] + 1) * w[src]:
            break
        if v[dst] + 1 > ROWS * w[dst]:
            break
        v[src] -= 1
        v[dst] += 1
        frames.append(tuple(v))
    return frames


def transition(cfg, state, key):
    vols, cursor, left = state
    n = len(vols)
    if key == "L":
        return [(vols, max(0, cursor - 1), left)]
    if key == "R":
        return [(vols, min(n - 2, cursor + 1), left)]
    if key == "S":
        frames = pulse(cfg, vols, cursor)
        out = [(f, cursor, left) for f in frames]
        final = frames[-1] if frames else vols
        out.append((final, cursor, left - 1))
        return out
    if key == "U":
        v = list(vols)
        v[cursor] = min(ROWS * cfg["w"][cursor], v[cursor] + TAP)
        return [(tuple(v), cursor, left - 1)]
    if key == "D":
        v = list(vols)
        v[cursor + 1] = max(0, v[cursor + 1] - TAP)
        return [(tuple(v), cursor, left - 1)]
    return [state]


def status(cfg, state):
    vols, cursor, left = state
    if heights_ok(cfg, vols):
        return "win"
    if left <= 0:
        return "lose"
    return None


def layout(cfg):
    xs = []
    x = X0
    for w in cfg["w"]:
        xs.append(x)
        x += w * UNIT + 2 + GAP
    return xs


def render(cfg, state):
    vols, cursor, left = state
    px = blank()
    bar(px, 1, left * 2, 14)
    xs = layout(cfg)
    w = cfg["w"]
    for i, x in enumerate(xs):
        inner = w[i] * UNIT
        top = BASE - ROWS * UNIT
        rect(px, x, top, 1, ROWS * UNIT + 1, 2)
        rect(px, x + inner + 1, top, 1, ROWS * UNIT + 1, 2)
        rect(px, x, BASE, inner + 2, 1, 2)
        full, part = divmod(vols[i], w[i])
        rect(px, x + 1, BASE - full * UNIT, inner, full * UNIT, 9)
        if part:
            rect(px, x + 1, BASE - full * UNIT - UNIT, part * UNIT, UNIT, 9)
        ty = BASE - cfg["target"][i] * UNIT
        rect(px, x - 2, ty, 2, 1, 11)
        rect(px, x + inner + 2, ty, 2, 1, 11)
    for p, row in enumerate(cfg["rows"]):
        x1 = xs[p] + w[p] * UNIT + 2
        x2 = xs[p + 1]
        colour = 15 if p == cursor else 3
        rect(px, x1, BASE - row * UNIT - cfg["rate"][p], x2 - x1, cfg["rate"][p], colour)
    rect(px, xs[cursor] + 1, BASE - ROWS * UNIT - 4, w[cursor] * UNIT, 2, 10)
    rect(px, xs[cursor + 1] + 1, BASE + 7, w[cursor + 1] * UNIT, 2, 8)
    cx = xs[cursor] + w[cursor] * UNIT + 2 + GAP // 2
    rect(px, cx - 1, BASE + 4, 3, 1, 15)
    rect(px, cx, BASE + 3, 1, 1, 15)
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


class Ng56(ARCBaseGame):
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
