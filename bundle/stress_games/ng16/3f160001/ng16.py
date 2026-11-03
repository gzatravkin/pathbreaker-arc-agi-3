import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
PLAYER = 11
TICK = 8
GOAL = 14
GEM = 6
PORTAL = 15
BAR = 9
PAL = [9, 10, 12, 6, 8, 15, 11, 7]
N = 13
M = N - 1
PITCH = 4
ORIGIN = 6
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTION_DIR = {GameAction.ACTION1: 0, GameAction.ACTION2: 2, GameAction.ACTION3: 3, GameAction.ACTION4: 1}

CONFIGS = [
    {'seed': 100, 'braid': 3, 'glue': 'P', 'start': (12, 10), 'goal': (2, 4), 'gems': [(2, 12), (10, 4), (0, 0)], 'portals': [], 'budget': 52},
    {'seed': 200, 'braid': 3, 'glue': 'H', 'start': (6, 2), 'goal': (4, 4), 'gems': [(0, 0), (8, 6), (10, 4)], 'portals': [], 'budget': 65},
    {'seed': 300, 'braid': 3, 'glue': 'M', 'start': (8, 4), 'goal': (6, 6), 'gems': [(12, 12), (10, 10), (0, 2), (12, 6)], 'portals': [], 'budget': 75},
    {'seed': 400, 'braid': 3, 'glue': 'T', 'start': (0, 2), 'goal': (6, 6), 'gems': [(8, 12), (6, 0), (10, 6), (2, 4)], 'portals': [], 'budget': 70},
    {'seed': 500, 'braid': 3, 'glue': 'A', 'start': (4, 8), 'goal': (2, 8), 'gems': [(8, 4), (6, 0), (0, 0), (12, 2)], 'portals': [(8, 2)], 'budget': 67},
    {'seed': 601, 'braid': 3, 'glue': 'T', 'start': (10, 0), 'goal': (0, 12), 'gems': [(4, 0), (2, 0), (4, 2), (2, 8), (12, 4)], 'portals': [(0, 6), (0, 10)], 'budget': 65},
]


def lcg(state):
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


class Rng:
    def __init__(self, seed):
        self.s = lcg(seed * 7919 + 17)

    def below(self, n):
        self.s = lcg(self.s)
        return (self.s >> 8) % n


def build_walls(seed, braid):
    rng = Rng(seed)
    wall = np.ones((N, N), dtype=bool)
    seen = {(0, 0)}
    stack = [(0, 0)]
    wall[0, 0] = False
    while stack:
        cx, cy = stack[-1]
        options = []
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 7 and 0 <= ny < 7 and (nx, ny) not in seen:
                options.append((nx, ny))
        if not options:
            stack.pop()
            continue
        nx, ny = options[rng.below(len(options))]
        wall[cy * 2 + (ny - cy), cx * 2 + (nx - cx)] = False
        wall[ny * 2, nx * 2] = False
        seen.add((nx, ny))
        stack.append((nx, ny))
    placed = 0
    guard = 0
    while placed < braid and guard < 1000:
        guard += 1
        x, y = rng.below(N), rng.below(N)
        if (x + y) % 2 == 1 and wall[y, x]:
            wall[y, x] = False
            placed += 1
    return wall


def wrap(kind, edge, x, y):
    if edge == "R":
        return {"P": (0, y), "H": (0, M - y), "M": (0, y), "T": (y, M), "A": (M - y, M)}[kind]
    if edge == "L":
        return {"P": (M, y), "H": (M, M - y), "M": (M, y), "T": (y, 0), "A": (M - y, 0)}[kind]
    if edge == "U":
        return {"P": (x, M), "H": (x, M), "M": (M - x, M), "T": (0, x), "A": (0, M - x)}[kind]
    return {"P": (x, 0), "H": (x, 0), "M": (M - x, 0), "T": (M, x), "A": (M, M - x)}[kind]


def partner_edge(kind, edge):
    if kind in "PHM":
        return {"R": "L", "L": "R", "U": "D", "D": "U"}[edge]
    return {"R": "D", "D": "R", "L": "U", "U": "L"}[edge]


def edge_tile(edge, t):
    return {"R": (M, t), "L": (0, t), "U": (t, 0), "D": (t, M)}[edge]


def strip_colours(kind):
    out = {}
    order = {}
    for edge in "RLUD":
        for t in range(N):
            if (edge, t) in out:
                continue
            x, y = edge_tile(edge, t)
            tx, ty = wrap(kind, edge, x, y)
            te = partner_edge(kind, edge)
            tt = ty if te in "LR" else tx
            key = frozenset((edge, te))
            if key not in order:
                order[key] = len(order)
            colour = PAL[(order[key] * 4 + t // 4) % len(PAL)]
            out[(edge, t)] = colour
            out[(te, tt)] = colour
    return out


def move(cfg, wall, state, d):
    x, y, rot, gems = state
    dx, dy = DIRS[(d + rot) % 4]
    nx, ny = x + dx, y + dy
    if nx < 0:
        nx, ny = wrap(cfg["glue"], "L", x, y)
    elif nx > M:
        nx, ny = wrap(cfg["glue"], "R", x, y)
    elif ny < 0:
        nx, ny = wrap(cfg["glue"], "U", x, y)
    elif ny > M:
        nx, ny = wrap(cfg["glue"], "D", x, y)
    if wall[ny, nx]:
        return state
    if (nx, ny) in cfg["portals"]:
        rot = (rot + 1) % 4
    if (nx, ny) in cfg["gems"]:
        gems = gems & ~(1 << cfg["gems"].index((nx, ny)))
    return (nx, ny, rot, gems)


def solved(cfg, state):
    return state[3] == 0 and (state[0], state[1]) == cfg["goal"]


def initial(cfg):
    return (cfg["start"][0], cfg["start"][1], 0, (1 << len(cfg["gems"])) - 1)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng16(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._wall = None
        self._pos = None
        self._left = 0
        super().__init__(
            game_id="ng16-3f160001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._wall = build_walls(self._cfg["seed"], self._cfg["braid"])
        self._pos = initial(self._cfg)
        self._left = self._cfg["budget"]
        self._draw(level)

    def _cell(self, px, tx, ty, colour):
        ox, oy = ORIGIN + tx * PITCH, ORIGIN + ty * PITCH
        px[oy:oy + PITCH, ox:ox + PITCH] = colour

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + (self._left + 1) // 2] = BAR
        x, y, rot, gems = self._pos
        for ty in range(N):
            for tx in range(N):
                self._cell(px, tx, ty, WALL if self._wall[ty, tx] else FLOOR)
        for tx, ty in cfg["portals"]:
            self._cell(px, tx, ty, PORTAL)
        gx, gy = cfg["goal"]
        self._cell(px, gx, gy, GOAL)
        if gems:
            ox, oy = ORIGIN + gx * PITCH, ORIGIN + gy * PITCH
            px[oy + 1:oy + 3, ox + 1:ox + 3] = FLOOR
        for i, (tx, ty) in enumerate(cfg["gems"]):
            if gems >> i & 1:
                ox, oy = ORIGIN + tx * PITCH, ORIGIN + ty * PITCH
                px[oy + 1:oy + 3, ox + 1:ox + 3] = GEM
        self._cell(px, x, y, PLAYER)
        ox, oy = ORIGIN + x * PITCH, ORIGIN + y * PITCH
        dx, dy = DIRS[rot % 4]
        if dy == -1:
            px[oy, ox + 1:ox + 3] = TICK
        elif dy == 1:
            px[oy + 3, ox + 1:ox + 3] = TICK
        elif dx == -1:
            px[oy + 1:oy + 3, ox] = TICK
        else:
            px[oy + 1:oy + 3, ox + 3] = TICK
        colours = strip_colours(cfg["glue"])
        end = ORIGIN + N * PITCH
        for t in range(N):
            a = ORIGIN + t * PITCH
            px[a:a + PITCH, 3:5] = colours[("L", t)]
            px[a:a + PITCH, end + 1:end + 3] = colours[("R", t)]
            px[3:5, a:a + PITCH] = colours[("U", t)]
            px[end + 1:end + 3, a:a + PITCH] = colours[("D", t)]
        for i in range(bin(gems).count("1")):
            px[62, 2 + i * 3:4 + i * 3] = GEM
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        d = ACTION_DIR.get(self._action.id)
        if d is not None:
            self._pos = move(self._cfg, self._wall, self._pos, d)
            self._left = max(0, self._left - 1)
            self._draw()
            if solved(self._cfg, self._pos):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()


SOLUTIONS = ['RDDRRDLLLDDDDDDLLUUDDRRUURRR', 'UURRDDRRDDUUUUURRRUUUUUUULLLLLUULLLL', 'UURRRRRUUUUUUUUUUULLLLDDDDDDRRLLLLUULLUURR', 'DDRRLLDDLUDDDDDDRRRRDDDRRRDUULLUULLDDLL', 'UUUURRRRRRUUUUUUULDDUULLLLLLUUURRDDDD', 'LLLLLLLLDDRRDDDDLLDDLLDDRRUUUURUDDDD']
