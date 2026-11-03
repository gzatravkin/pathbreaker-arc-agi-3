import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 2
WALL = 4
BEACON = 15
HUB = 4
BEAM = 7
PLAYER = 8
EXIT = 14
BAR = 10
ORIGIN_X = 5
ORIGIN_Y = 6
PITCH = 6
SIDE = 5
BAR_MAX = 60

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

CONFIGS = [{'rows': ['S.b....', '......#', '..c....', '.......', '....a..', '.......', '.....#E'], 'beacons': {'a': (1, 4), 'b': (3, 4), 'c': (2, 4)}, 'len': 3}, {'rows': ['S...b..', '......a', '.......', '.#..#..', 'b...c..', '.....#.', '.#...aE'], 'beacons': {'a': (3, 4), 'b': (0, 4), 'c': (1, 4)}, 'len': 6}, {'rows': ['S......', '...e...', '...db..', 'a......', '..c....', '...#...', '#.....E'], 'beacons': {'a': (3, 4), 'b': (0, 3), 'c': (2, 3), 'd': (0, 4), 'e': (2, 2)}, 'len': 7}, {'rows': ['S.......', '......#.', '.b..#...', '..#.....', '..#.#e..', 'c.......', '...aa...', '.d.....E'], 'beacons': {'a': (0, 2), 'b': (3, 4), 'c': (1, 3), 'd': (1, 4), 'e': (1, 2)}, 'len': 9}, {'rows': ['S....b..', '.e..b...', '.a......', '...c....', '.....#..', '.....a..', 'd..#....', '.....#.E'], 'beacons': {'a': (3, 3), 'b': (0, 2), 'c': (1, 4), 'd': (2, 3), 'e': (0, 3)}, 'len': 10}]

SOLUTIONS = [[(7, 26), (43, 26), (43, 44)], [(13, 8), (13, 8), (13, 20), (37, 20), (43, 20), (43, 44)], [(13, 8), (13, 8), (13, 8), (13, 8), (13, 8), (43, 8), (43, 44)], [(7, 8), (7, 8), (7, 8), (7, 26), (13, 26), (13, 38), (19, 38), (19, 50), (49, 50)], [(7, 8), (13, 8), (13, 8), (19, 8), (19, 20), (19, 26), (19, 26), (19, 20), (49, 20), (49, 50)]]


def cell(cfg, pos):
    x, y = pos
    rows = cfg["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return "#"


def find(cfg, ch):
    for y, row in enumerate(cfg["rows"]):
        if ch in row:
            return (row.index(ch), y)
    return None


def solid(cfg, pos):
    ch = cell(cfg, pos)
    return ch == "#" or ch in cfg["beacons"]


def beam_index(arc, c, rule):
    if rule == "loop":
        return c % 4
    m = c % (2 * (arc - 1))
    return m if m <= arc - 1 else 2 * (arc - 1) - m


def lit_cells(cfg, c, rule="true"):
    lit = set()
    for y, row in enumerate(cfg["rows"]):
        for x, ch in enumerate(row):
            if ch in cfg["beacons"]:
                d0, arc = cfg["beacons"][ch]
                dx, dy = DIRS[(d0 + beam_index(arc, c, rule)) % 4]
                cx, cy = x + dx, y + dy
                while not solid(cfg, (cx, cy)):
                    lit.add((cx, cy))
                    cx, cy = cx + dx, cy + dy
    return lit


def route(cfg, pos, target):
    if target == pos:
        return [pos]
    dx = (target[0] > pos[0]) - (target[0] < pos[0])
    dy = (target[1] > pos[1]) - (target[1] < pos[1])
    if dx and dy:
        return None
    out = []
    cur = pos
    while cur != target:
        cur = (cur[0] + dx, cur[1] + dy)
        if solid(cfg, cur):
            return None
        out.append(cur)
    return out


def execute(cfg, pos, c, target, rule="true"):
    steps = route(cfg, pos, target)
    if steps is None:
        return None
    seen = []
    if rule == "percell":
        for k, cur in enumerate(steps):
            seen.append(cur)
            if cur in lit_cells(cfg, c + k + 1, rule):
                return cur, c + k + 1, True, seen
            if cell(cfg, cur) == "E":
                return cur, c + k + 1, False, seen
        return seen[-1], c + len(steps), False, seen
    beams = lit_cells(cfg, c, rule)
    for cur in steps:
        seen.append(cur)
        if cur in beams:
            return cur, c + 1, True, seen
        if cell(cfg, cur) == "E":
            return cur, c + 1, False, seen
    if seen[-1] in lit_cells(cfg, c + 1, rule):
        return seen[-1], c + 1, True, seen
    return seen[-1], c + 1, False, seen


def budget_of(cfg):
    return min(BAR_MAX, 2 * cfg["len"] + 8)


def cell_at(x, y):
    if x < ORIGIN_X or y < ORIGIN_Y:
        return None
    return (x - ORIGIN_X) // PITCH, (y - ORIGIN_Y) // PITCH


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


class Ng39(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = (0, 0)
        self._c = 0
        self._left = 0
        self._plan = None
        self._dead = False
        super().__init__(
            game_id="ng39-3a390001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = find(self._cfg, "S")
        self._c = 0
        self._left = budget_of(self._cfg)
        self._plan = None
        self._dead = False
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        lit = lit_cells(cfg, self._c)
        for y, row in enumerate(cfg["rows"]):
            for x, ch in enumerate(row):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if ch == "#":
                    px[oy:oy + SIDE, ox:ox + SIDE] = WALL
                elif ch in cfg["beacons"]:
                    px[oy:oy + SIDE, ox:ox + SIDE] = BEACON
                    px[oy + 2, ox + 2] = HUB
                elif ch == "E":
                    px[oy:oy + SIDE, ox:ox + SIDE] = EXIT
                else:
                    px[oy:oy + SIDE, ox:ox + SIDE] = BEAM if (x, y) in lit else FLOOR
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._dead:
            self._pos = find(self._cfg, "S")
            self._dead = False
            self._draw()
            self.complete_action()
            return
        if self._plan is None:
            if self._action.id != GameAction.ACTION6 or self._left <= 0:
                self.complete_action()
                return
            data = self._action.data or {}
            target = cell_at(int(data.get("x", -1)), int(data.get("y", -1)))
            result = execute(self._cfg, self._pos, self._c, target) if target else None
            if result is None:
                self.complete_action()
                return
            self._left -= 1
            self._plan = [list(result[3]), result]
        cells, result = self._plan
        if cells:
            self._pos = cells.pop(0)
            self._draw()
            if cells:
                return
        self._plan = None
        self._c = result[1]
        if result[2]:
            self._draw()
            if self._left == 0:
                self.lose()
                self.complete_action()
            else:
                self._dead = True
            return
        self._draw()
        if cell(self._cfg, self._pos) == "E":
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
