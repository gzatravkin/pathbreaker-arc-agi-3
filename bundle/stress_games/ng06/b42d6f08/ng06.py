import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
RAIL = 3
PLATE = 2
GOAL_PLATE = 0
BAR = 14
BAR_EMPTY = 4
BAD = 8
BAR_X = 2
BAR_Y = 1
BAR_MAX = 60
BASE_Y = 60
PIECE_Y = 56
PITCH = 4
HOVER_Y = 22
LEGEND_BASE = 20
FREE_CAP = 8
TOP_CLICK = 21
R, B, Y = 8, 9, 11

CONFIGS = [
    dict(
        stacks=[[(Y, 2), (B, 2), (R, 2)], [(Y, 2), (B, 2), (R, 2)], []],
        caps=[FREE_CAP, FREE_CAP, FREE_CAP],
        rule=False,
        goals={2: [(Y, 2), (Y, 2), (B, 2), (R, 2), (R, 2), (B, 2)]},
        moves=[(0, 1), (0, 1), (0, 2), (1, 0), (1, 0), (1, 0), (1, 0), (1, 2), (0, 2), (0, 2), (0, 2), (0, 2)],
    ),
    dict(
        stacks=[[(R, 2), (R, 2), (Y, 2)], [(B, 2), (B, 2), (Y, 2)], [(B, 2), (R, 2), (Y, 2)], []],
        caps=[3, 3, 3, 6],
        rule=False,
        goals={3: [(R, 2), (R, 2), (R, 2), (B, 2), (B, 2), (B, 2)]},
        moves=[(0, 3), (0, 3), (0, 3), (1, 0), (2, 0), (3, 1), (3, 2), (3, 0), (1, 3), (2, 3), (2, 3), (1, 3), (1, 3), (2, 3)],
    ),
    dict(
        stacks=[[(R, 1)], [(R, 2), (B, 3), (R, 3), (B, 2), (B, 1)], []],
        caps=[FREE_CAP, FREE_CAP, FREE_CAP],
        rule=True,
        goals={2: [(R, 2), (R, 1), (R, 3), (B, 2), (B, 3), (B, 1)]},
        moves=[(0, 2), (1, 0), (1, 0), (1, 2), (1, 0), (1, 0), (2, 1), (2, 1), (0, 2), (1, 2), (1, 2), (0, 1), (0, 2), (1, 2), (0, 2)],
    ),
    dict(
        stacks=[[(B, 2), (R, 2)], [(R, 1), (Y, 2)], [(B, 3), (R, 3), (Y, 1), (B, 1)], []],
        caps=[4, 4, 4, 5],
        rule=True,
        goals={3: [(B, 2), (B, 3), (R, 1), (R, 3), (R, 2)]},
        moves=[(0, 3), (2, 1), (0, 1), (3, 0), (1, 3), (1, 3), (2, 1), (2, 0), (3, 0), (2, 3), (1, 2), (1, 2), (0, 2), (1, 3), (0, 3), (0, 3)],
    ),
    dict(
        stacks=[[(R, 2), (Y, 2), (B, 2), (Y, 2), (B, 2)], [], [], [(R, 2), (B, 2), (B, 2), (B, 2), (B, 2)]],
        caps=[5, 5, 5, 5],
        rule=False,
        goals={1: [(R, 2), (B, 2), (B, 2), (Y, 2), (B, 2)], 2: [(R, 2), (B, 2), (B, 2), (Y, 2), (B, 2)]},
        moves=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (3, 0), (1, 0), (1, 0), (3, 0), (3, 2), (3, 2), (1, 2), (1, 2), (3, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
    ),
    dict(
        stacks=[[(Y, 2), (Y, 2), (Y, 1), (B, 1), (B, 1)], [], [], [(B, 1), (Y, 3), (Y, 3), (R, 2), (R, 2)]],
        caps=[5, 4, 4, 5],
        rule=True,
        goals={1: [(Y, 2), (Y, 3), (R, 2), (B, 1)], 2: [(Y, 2), (Y, 3), (R, 2), (B, 1)]},
        moves=[(3, 1), (3, 1), (0, 1), (0, 1), (0, 3), (0, 2), (0, 2), (3, 0), (3, 0), (2, 0), (3, 2), (1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (2, 1), (3, 1), (3, 2)],
    ),
]


def centre(n, i):
    return int(round((i + 0.5) * 64 / n))


def fits(stacks, rule, piece, j):
    if not rule or not stacks[j]:
        return True
    q = stacks[j][-1]
    return q[1] > piece[1] or q[0] == piece[0]


def clicks_of(cfg):
    n = len(cfg["stacks"])
    out = []
    for i, j in cfg["moves"]:
        out.append((centre(n, i), 40))
        out.append((centre(n, j), 40))
    return out


SOLUTIONS = [clicks_of(cfg) for cfg in CONFIGS]


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


class Ng06(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._stacks = []
        self._held = None
        self._left = 0
        self._budget = 0
        self._queue = None
        self._outcome = "ok"
        super().__init__(
            game_id="ng06-b42d6f08",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._stacks = [list(s) for s in self._cfg["stacks"]]
        self._held = None
        self._budget = 2 * len(self._cfg["moves"]) + 4
        self._left = self._budget
        self._queue = None
        self._outcome = "ok"
        self._draw(None, level)

    def _piece(self, px, cx, y, piece, height):
        half = piece[1] + 1
        px[y:y + height, cx - half:cx + half + 1] = piece[0]

    def _draw(self, flash, level=None):
        level = level or self.current_level
        cfg = self._cfg
        n = len(self._stacks)
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + BAR_MAX] = BAR_EMPTY
        width = -(-BAR_MAX * self._left // self._budget) if self._budget else 0
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + width] = BAR
        for i, stack in enumerate(self._stacks):
            cx = centre(n, i)
            cap = cfg["caps"][i]
            top = BASE_Y - PITCH * cap
            px[top:BASE_Y, cx] = RAIL
            if cap < FREE_CAP:
                px[top - 1, cx - 5:cx + 6] = RAIL
            plate = GOAL_PLATE if i in cfg["goals"] else PLATE
            if flash == i:
                plate = BAD
            px[BASE_Y:BASE_Y + 2, cx - 5:cx + 6] = plate
            for k, piece in enumerate(stack):
                self._piece(px, cx, PIECE_Y - PITCH * k, piece, 3)
        for g, target in cfg["goals"].items():
            cx = centre(n, g)
            px[LEGEND_BASE, cx - 5:cx + 6] = GOAL_PLATE
            for k, piece in enumerate(target):
                self._piece(px, cx, LEGEND_BASE - 2 - 3 * k, piece, 2)
        if self._held is not None:
            piece, src = self._held
            self._piece(px, centre(n, src), HOVER_Y, piece, 3)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _hit(self, x, y):
        n = len(self._stacks)
        if y < TOP_CLICK or y > 63:
            return None
        for i in range(n):
            if abs(x - centre(n, i)) <= 5:
                return i
        return None

    def _won(self):
        return self._held is None and all(self._stacks[g] == list(t) for g, t in self._cfg["goals"].items())

    def _resolve(self):
        if self._action.id != GameAction.ACTION6:
            return [None]
        data = self._action.data or {}
        if "x" not in data or "y" not in data:
            return [None]
        i = self._hit(int(data["x"]), int(data["y"]))
        if i is None:
            return [None]
        if self._held is None:
            if self._stacks[i]:
                self._held = (self._stacks[i].pop(), i)
            return [None]
        piece, src = self._held
        cfg = self._cfg
        self._left = max(0, self._left - 1)
        if i == src:
            self._stacks[src].append(piece)
            self._held = None
            frames = [None]
        elif len(self._stacks[i]) < cfg["caps"][i] and fits(self._stacks, cfg["rule"], piece, i):
            self._stacks[i].append(piece)
            self._held = None
            frames = [None]
        else:
            frames = [i, i, None]
        if self._won():
            self._outcome = "win"
        elif self._left == 0:
            self._outcome = "lose"
        return frames

    def step(self) -> None:
        if self._queue is None:
            self._outcome = "ok"
            self._queue = self._resolve()
        self._draw(self._queue.pop(0))
        if self._queue:
            return
        self._queue = None
        if self._outcome == "win":
            self.next_level()
        elif self._outcome == "lose":
            self.lose()
        self.complete_action()
