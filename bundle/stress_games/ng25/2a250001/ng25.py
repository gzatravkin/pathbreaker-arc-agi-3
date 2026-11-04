import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
RAIL = 3
MARK = 11
BAR = 14
CELL = 6
OX = 2
OY = 8
W = 10
H = 9

LOOP = (
    [(x, 3) for x in range(2, 8)]
    + [(7, y) for y in range(4, 7)]
    + [(x, 6) for x in range(6, 1, -1)]
    + [(2, y) for y in range(5, 3, -1)]
)

FORKS = {
    "T": dict(cell=(4, 3), stem=(-1, 0), loop=(1, 0), spur=(0, -1), rail=(4, 2), station=(4, 1)),
    "R": dict(cell=(7, 4), stem=(0, -1), loop=(0, 1), spur=(1, 0), rail=(8, 4), station=(9, 4)),
    "B": dict(cell=(5, 6), stem=(1, 0), loop=(-1, 0), spur=(0, 1), rail=(5, 7), station=(5, 8)),
    "L": dict(cell=(2, 5), stem=(0, 1), loop=(0, -1), spur=(-1, 0), rail=(1, 5), station=(0, 5)),
}

LEVELS_CFG = [
    {'forks': {'T': 8, 'B': 9, 'R': 14}, 'trains': [(3, 8, 1), (12, 9, 1), (1, 14, 1)]},
    {'forks': {'T': 8, 'B': 9, 'R': 14}, 'trains': [(0, 8, 1), (1, 8, 1), (13, 9, 1), (7, 14, 1)]},
    {'forks': {'T': 8, 'B': 9, 'R': 14}, 'trains': [(7, 8, 2), (12, 8, 1), (9, 9, 2), (1, 14, 1)]},
    {'forks': {'T': 8, 'B': 9, 'R': 14, 'L': 12}, 'trains': [(12, 8, 1), (7, 9, 2), (8, 14, 1), (11, 12, 1), (4, 8, 2)]},
    {'forks': {'T': 8, 'B': 9, 'R': 14}, 'trains': [(0, 8, 1), (9, 9, 1), (4, 14, 1), (15, 9, 2)], 'links': [('T', 'B')]},
    {'forks': {'T': 8, 'B': 9, 'R': 14}, 'trains': [(5, 8, 1), (0, 9, 1), (13, 14, 1)], 'order': [9, 8, 14]},
]

SOLUTIONS = [
    ['S', 'S', 'S', 'S', 'S', (46, 34), 'S', 'S', 'S', 'S', (46, 34), 'S', 'S', 'S', (34, 46), (28, 28), 'S'],
    ['S', (28, 28), 'S', 'S', 'S', (28, 28), 'S', 'S', 'S', 'S', 'S', 'S', 'S', (34, 46), 'S', (46, 34), 'S'],
    ['S', 'S', (34, 46), 'S', (34, 46), (46, 34), (28, 28), 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'S', 'S', (16, 40), (34, 46), (16, 40), (28, 28), 'S', 'S', 'S', (28, 28), 'S', (34, 46), 'S', (46, 34), 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', (28, 28), 'S', 'S'],
    ['S', (34, 46), 'S', 'S', 'S', 'S', (34, 46), 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', (46, 34), 'S', 'S', 'S', (34, 46), 'S', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', (34, 46), 'S', (34, 46), (28, 28), 'S', 'S', 'S', 'S', 'S', 'S', 'S', (28, 28), 'S', 'S', 'S', (46, 34), 'S'],
]


def add(a, d):
    return (a[0] + d[0], a[1] + d[1])


def parse(cfg):
    forks = {}
    rails = set(LOOP)
    stations = {}
    for name, colour in cfg["forks"].items():
        f = FORKS[name]
        forks[f["cell"]] = dict(name=name, stem=add(f["cell"], f["stem"]), loop=add(f["cell"], f["loop"]),
                                spur=f["rail"])
        rails.add(f["rail"])
        rails.add(f["station"])
        stations[f["station"]] = colour
    order_names = sorted(cfg["forks"])
    fork_cells = [FORKS[n]["cell"] for n in order_names]
    links = {}
    for a, b in cfg.get("links", []):
        ca, cb = FORKS[a]["cell"], FORKS[b]["cell"]
        links.setdefault(ca, []).append(cb)
        links.setdefault(cb, []).append(ca)
    trains = []
    for idx, colour, speed in cfg["trains"]:
        trains.append((LOOP[idx % len(LOOP)], LOOP[(idx - 1) % len(LOOP)], colour, speed))
    return dict(forks=forks, rails=rails, stations=stations, fork_cells=fork_cells, links=links,
                trains=trains, order=cfg.get("order"), init=cfg.get("init", 0))


def initial(info):
    ts = tuple((c, p) for c, p, _, _ in info["trains"])
    return (ts, info["init"], 0, 0)


def neighbours(info, cell):
    return [add(cell, d) for d in ((0, -1), (0, 1), (-1, 0), (1, 0)) if add(cell, d) in info["rails"]]


def next_cell(info, cell, prev, sw):
    fork = info["forks"].get(cell)
    if fork is not None:
        if prev == fork["stem"]:
            idx = info["fork_cells"].index(cell)
            return fork["spur"] if sw >> idx & 1 else fork["loop"]
        return fork["stem"]
    options = [n for n in neighbours(info, cell) if n != prev]
    return options[0] if options else None


def advance(info, st, action):
    ts, sw, tick, done = st
    if action != "S":
        cell = action
        if cell not in info["forks"]:
            return None, None
        idx = info["fork_cells"].index(cell)
        sw ^= 1 << idx
        for other in info["links"].get(cell, []):
            sw ^= 1 << info["fork_cells"].index(other)
    new = []
    for i, tr in enumerate(ts):
        if tr is None:
            new.append(None)
            continue
        cell, prev = tr
        speed = info["trains"][i][3]
        if speed == 2 and tick % 2 == 1:
            new.append(tr)
            continue
        nxt = next_cell(info, cell, prev, sw)
        if nxt is None:
            return (ts, sw, tick + 1, done), "lose"
        new.append((nxt, cell))
    alive = [(i, t) for i, t in enumerate(new) if t is not None]
    for a in range(len(alive)):
        for b in range(a + 1, len(alive)):
            ia, ta = alive[a]
            ib, tb = alive[b]
            if ta[0] == tb[0]:
                return (tuple(new), sw, tick + 1, done), "lose"
            if ts[ia] is not None and ts[ib] is not None and ta[0] == ts[ib][0] and tb[0] == ts[ia][0]:
                return (tuple(new), sw, tick + 1, done), "lose"
    for i, t in alive:
        if t[0] in info["stations"]:
            colour = info["trains"][i][2]
            if colour != info["stations"][t[0]]:
                return (tuple(new), sw, tick + 1, done), "lose"
            if info["order"] is not None and info["order"][done] != colour:
                return (tuple(new), sw, tick + 1, done), "lose"
            new[i] = None
            done += 1
    st2 = (tuple(new), sw, tick + 1, done)
    return st2, ("win" if all(t is None for t in new) else None)


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


class Ng25(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._info = None
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng25-2a250001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[5, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._info = parse(LEVELS_CFG[index])
        self._st = initial(self._info)
        self._left = budget_of(index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        info = self._info
        ts, sw, tick, done = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for y in range(H):
            for x in range(W):
                x0, y0 = OX + x * CELL, OY + y * CELL
                px[y0:y0 + CELL, x0:x0 + CELL] = FLOOR
        for c in info["rails"]:
            x0, y0 = OX + c[0] * CELL, OY + c[1] * CELL
            px[y0 + 2:y0 + 4, x0 + 2:x0 + 4] = RAIL
            for n in neighbours(info, c):
                dx, dy = n[0] - c[0], n[1] - c[1]
                if dx == 1:
                    px[y0 + 2:y0 + 4, x0 + 4:x0 + 6] = RAIL
                elif dx == -1:
                    px[y0 + 2:y0 + 4, x0:x0 + 2] = RAIL
                elif dy == 1:
                    px[y0 + 4:y0 + 6, x0 + 2:x0 + 4] = RAIL
                else:
                    px[y0:y0 + 2, x0 + 2:x0 + 4] = RAIL
        for i, cell in enumerate(info["fork_cells"]):
            fork = info["forks"][cell]
            target = fork["spur"] if sw >> i & 1 else fork["loop"]
            x0, y0 = OX + cell[0] * CELL, OY + cell[1] * CELL
            dx, dy = target[0] - cell[0], target[1] - cell[1]
            px[y0 + 2 + dy * 2:y0 + 4 + dy * 2, x0 + 2 + dx * 2:x0 + 4 + dx * 2] = MARK
        for c, colour in info["stations"].items():
            x0, y0 = OX + c[0] * CELL, OY + c[1] * CELL
            px[y0:y0 + CELL, x0:x0 + CELL] = colour
            px[y0 + 1:y0 + CELL - 1, x0 + 1:x0 + CELL - 1] = FLOOR
        for i, t in enumerate(ts):
            if t is None:
                continue
            colour = info["trains"][i][2]
            x0, y0 = OX + t[0][0] * CELL, OY + t[0][1] * CELL
            px[y0 + 1:y0 + 5, x0 + 1:x0 + 5] = colour
            if info["trains"][i][3] == 2:
                px[y0 + 2:y0 + 4, x0 + 2:x0 + 4] = FLOOR
        if info["order"] is not None:
            for k, colour in enumerate(info["order"][done:]):
                px[3:6, 2 + k * 4:5 + k * 4] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _cell_at(self, x, y):
        cx, rx = divmod(x - OX, CELL)
        cy, ry = divmod(y - OY, CELL)
        if x < OX or y < OY:
            return None
        return (cx, cy)

    def step(self) -> None:
        action = None
        if self._action.id == GameAction.ACTION5:
            action = "S"
        elif self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            action = self._cell_at(int(data.get("x", -1)), int(data.get("y", -1)))
        if action is not None and self._st is not None:
            st, outcome = advance(self._info, self._st, action)
            if st is not None:
                self._st = st
                self._left = max(0, self._left - 1)
                self._draw()
                if outcome == "win":
                    self.next_level()
                elif outcome == "lose" or self._left == 0:
                    self.lose()
        self.complete_action()
