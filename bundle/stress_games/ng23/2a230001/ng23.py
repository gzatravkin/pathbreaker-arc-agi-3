import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
SLOT = 3
BEAM = 2
PIVOT = 2
WEIGHT = 11
PLAYER = 9
NEEDLE = 8
BALANCED = 14
BAR = 14
CELL = 6
OX = 5
OY = 14
COLS = 9
ROWS = 7
PIVOT_COL = 4

DELTAS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
ACTION_LETTER = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}

NEAR = (3, 5)
WIDE = (0, 1, 2, 3, 5, 6, 7, 8)

LEVELS_CFG = [
    dict(levers=[{'row': 3, 'cols': NEAR}], weights=[(3, 2, 2, 0), (3, 4, 2, 0), (1, 0, 2, 0), (7, 0, 2, 0), (1, 6, 1, 0), (7, 6, 3, 0)], start=(4, 5)),
    dict(levers=[{'row': 3, 'cols': NEAR}], weights=[(3, 3, 4, 0), (1, 1, 3, 0), (7, 1, 1, 0), (2, 5, 2, 0), (6, 5, 2, 0)], start=(4, 5)),
    dict(levers=[{'row': 3, 'cols': WIDE}], weights=[(1, 3, 3, 0), (0, 0, 3, 0), (8, 0, 4, 0), (2, 6, 1, 0), (6, 6, 2, 0)], start=(4, 5)),
    dict(levers=[{'row': 3, 'cols': WIDE}], weights=[(1, 4, 2, 0), (1, 3, 3, 0), (1, 5, 2, 0), (6, 0, 2, 0), (5, 6, 1, 0), (0, 5, 3, 0), (8, 0, 2, 0)], start=(4, 5)),
    dict(levers=[{'row': 1, 'cols': WIDE}, {'row': 5, 'cols': WIDE}], weights=[(2, 2, 1, 0), (0, 0, 2, 0), (3, 3, 3, 0), (2, 3, 2, 0), (6, 3, 2, 0), (1, 3, 2, 0), (7, 3, 2, 0)], start=(4, 3), cascade=[(0, 1, 2)]),
    dict(levers=[{'row': 3, 'cols': WIDE}], weights=[(3, 3, 1, 0), (6, 2, 3, 0), (1, 6, 2, 1), (8, 6, 4, 0), (2, 0, 2, 0), (7, 1, 1, 0), (7, 5, 3, 0)], start=(4, 5)),
]

SOLUTIONS = [
    'LLLDSRRRRUUSRRDDSLLUUUS',
    'LLSRRRUSRDSLUUS',
    'LLDSRRRUUSRDDSRRUUS',
    'RDSUUSRUUUUSRDDSRUUSDDS',
    'LLSRRRRUSDSRUSDSRDS',
    'LLLDSUUSRUUUUSRDDS',
]


def lever_slots(cfg):
    out = {}
    for i, lv in enumerate(cfg["levers"]):
        for col in lv["cols"]:
            for dr in (-1, 0, 1):
                out[(col, lv["row"] + dr)] = i
    return out


def effective(w):
    size, hollow = w
    return max(1, size - 1) if hollow else size


def nets(cfg, cells):
    slots = lever_slots(cfg)
    n = len(cfg["levers"])
    tor = [0] * n
    right = [0] * n
    loaded = [0] * n
    for c, w in cells.items():
        i = slots.get(c)
        if i is None:
            continue
        e = effective(w)
        tor[i] += e * (c[0] - PIVOT_COL)
        loaded[i] += 1
        if c[0] > PIVOT_COL:
            right[i] += e
    out = list(tor)
    for src, dst, factor in cfg.get("cascade", []):
        out[dst] -= factor * right[src]
    return out, loaded


def balanced(cfg, cells):
    out, loaded = nets(cfg, cells)
    return all(v == 0 for v in out) and all(v > 0 for v in loaded)


def initial(cfg):
    cells = {}
    for x, y, size, hollow in cfg["weights"]:
        cells[(x, y)] = (size, hollow)
    return cfg["start"], None, cells


def advance(cfg, st, letter):
    pos, hold, cells = st
    cells = dict(cells)
    if letter == "S":
        if hold is not None:
            if pos in cells:
                return None, None
            cells[pos] = hold
            hold = None
        elif pos in cells:
            hold = cells.pop(pos)
        else:
            return None, None
        st2 = (pos, hold, cells)
        return st2, ("win" if hold is None and balanced(cfg, cells) else None)
    dx, dy = DELTAS[letter]
    new = (pos[0] + dx, pos[1] + dy)
    if not (0 <= new[0] < COLS and 0 <= new[1] < ROWS):
        return None, None
    return (new, hold, cells), None


def budget_of(index):
    return min(60, 2 * len(SOLUTIONS[index]) + 10)


def blank_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [blank_level(i) for i in range(max(1, len(LEVELS_CFG)))]


class Ng23(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = None
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng23-2a230001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._cfg = LEVELS_CFG[index]
        self._st = initial(self._cfg)
        self._left = budget_of(index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        pos, hold, cells = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        slots = lever_slots(cfg)
        for y in range(ROWS):
            for x in range(COLS):
                x0, y0 = OX + x * CELL, OY + y * CELL
                px[y0:y0 + CELL, x0:x0 + CELL] = SLOT if (x, y) in slots else FLOOR
        for lv in cfg["levers"]:
            y0 = OY + lv["row"] * CELL + 2
            px[y0:y0 + 2, OX:OX + COLS * CELL] = BEAM
            px0 = OX + PIVOT_COL * CELL
            px[y0 - 1:y0 + 4, px0 + 1:px0 + 5] = PIVOT
        for c, w in cells.items():
            self._block(px, c, w[0])
        x0, y0 = OX + pos[0] * CELL, OY + pos[1] * CELL
        px[y0:y0 + CELL, x0:x0 + CELL] = PLAYER
        px[y0 + 1:y0 + CELL - 1, x0 + 1:x0 + CELL - 1] = FLOOR
        if hold is not None:
            self._block(px, pos, hold[0])
        out, loaded = nets(cfg, cells)
        for i, net in enumerate(out):
            gy = 5 + 4 * i
            px[gy + 1, 4:60] = SLOT
            px[gy:gy + 3, 32] = BEAM
            nx = 32 + max(-26, min(26, net * 2))
            colour = BALANCED if net == 0 and loaded[i] > 0 else NEEDLE
            px[gy:gy + 3, nx - 1:nx + 2] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    @staticmethod
    def _block(px, c, size):
        x0 = OX + c[0] * CELL + (CELL - size) // 2
        y0 = OY + c[1] * CELL + (CELL - size) // 2
        px[y0:y0 + size, x0:x0 + size] = WEIGHT

    def step(self) -> None:
        letter = ACTION_LETTER.get(self._action.id)
        if letter is not None and self._st is not None:
            st, outcome = advance(self._cfg, self._st, letter)
            if st is not None:
                self._st = st
                self._left = max(0, self._left - 1)
                self._draw()
                if outcome == "win":
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()
