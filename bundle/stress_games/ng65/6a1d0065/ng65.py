import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
GOAL = 14
PALETTE = {"a": 11, "b": 12, "c": 10}
BAR = 8
PITCH = 6
OX = 2
OY = 4
N = 10
DECOY = None
DECOY_NAMES = ["global", "hop", "anyadj"]

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

DATA = [
    [
        "##########",
        "#........#",
        "#........#",
        "#.aa.....#",
        "#........#",
        "#.....gg.#",
        "#........#",
        "#........#",
        "#.b....g.#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#aa...gg.#",
        "#b.......#",
        "#........#",
        "#........#",
        "#........#",
        "#.......A#",
        "#.g......#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#.aaaa...#",
        "#........#",
        "#####....#",
        "#........#",
        "#.gggg...#",
        "#........#",
        "#........#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#.aa.....#",
        "#.bb.....#",
        "#........#",
        "#####....#",
        "#.gg.....#",
        "#.gg.....#",
        "#........#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#.a..g...#",
        "#.a..g...#",
        "#bb..g...#",
        "#........#",
        "#......c.#",
        "#...g....#",
        "#.....gg.#",
        "##########",
    ],
]
BUDGETS = [30, 30, 28, 48, 40] * 5


def parse(rows):
    walls, goals, creatures = set(), set(), []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "g":
                goals.add((x, y))
            elif ch in "abc":
                creatures.append((x, y, ch))
            elif ch in "ABC":
                creatures.append((x, y, ch.lower()))
                goals.add((x, y))
    return frozenset(walls), frozenset(goals), tuple(creatures)


CACHE = {}


def P(i):
    key = (i, tuple(DATA[i]))
    if key not in CACHE:
        CACHE[key] = parse(DATA[i])
    return CACHE[key]


def start(i):
    return P(i)[2]


def tokens(i):
    walls = P(i)[0]
    return [(x, y, d) for y in range(N) for x in range(N) if (x, y) not in walls for d in "UDLR"]


def tokens_state(i, creatures):
    return [(c[0], c[1], d) for c in creatures for d in "UDLR"]


def to_px(tok):
    x, y, d = tok
    cx, cy = OX + x * PITCH + 2, OY + y * PITCH + 2
    dx, dy = DIRS[d]
    return (cx - 2 * dx, cy - 2 * dy)


def group_of(creatures, k):
    kind = creatures[k][2]
    pos = {(c[0], c[1]): j for j, c in enumerate(creatures)}
    if DECOY == "global":
        return {j for j, c in enumerate(creatures) if c[2] == kind}
    seen = {k}
    frontier = [k]
    while frontier:
        j = frontier.pop()
        x, y = creatures[j][0], creatures[j][1]
        for dx, dy in DIRS.values():
            n = pos.get((x + dx, y + dy))
            if n is None or n in seen:
                continue
            same = creatures[n][2] == kind
            if DECOY == "anyadj":
                same = True
            if same:
                seen.add(n)
                if DECOY != "hop" or j == k:
                    frontier.append(n)
                if DECOY == "hop":
                    pass
    return seen


def step(i, creatures, tok):
    walls = P(i)[0]
    x, y, d = tok
    k = next((j for j, c in enumerate(creatures) if (c[0], c[1]) == (x, y)), None)
    if k is None:
        return creatures
    dx, dy = DIRS[d]
    group = group_of(creatures, k)
    occupied = {(c[0], c[1]) for j, c in enumerate(creatures) if j not in group}
    for j in group:
        dest = (creatures[j][0] + dx, creatures[j][1] + dy)
        if dest in walls or dest in occupied or not (0 <= dest[0] < N and 0 <= dest[1] < N):
            return creatures
    out = list(creatures)
    for j in group:
        out[j] = (creatures[j][0] + dx, creatures[j][1] + dy, creatures[j][2])
    return tuple(out)


def won(i, creatures):
    goals = P(i)[1]
    return all((c[0], c[1]) in goals for c in creatures)


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


LEVELS = [canvas_level(i) for i in range(max(1, len(DATA)))]


class Ng65(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._index = 0
        self._st = ()
        self._left = 0
        super().__init__(
            game_id="ng65-6a1d0065",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._index = self._current_level_index
        self._st = start(self._index)
        self._left = BUDGETS[self._index]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        walls, goals, _ = P(self._index)
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for y in range(N):
            for x in range(N):
                ox, oy = OX + x * PITCH, OY + y * PITCH
                colour = FLOOR
                if (x, y) in walls:
                    colour = WALL
                elif (x, y) in goals:
                    colour = GOAL
                px[oy:oy + PITCH - 1, ox:ox + PITCH - 1] = colour
        for x, y, kind in self._st:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = PALETTE[kind]
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            mx, my = int(data.get("x", -99)), int(data.get("y", -99))
            cx, cy = (mx - OX) // PITCH, (my - OY) // PITCH
            ox, oy = mx - (OX + cx * PITCH + 2), my - (OY + cy * PITCH + 2)
            if 0 <= cx < N and 0 <= cy < N and (ox, oy) != (0, 0) and abs(ox) <= 2 and abs(oy) <= 2:
                if abs(ox) >= abs(oy):
                    d = "L" if ox > 0 else "R"
                else:
                    d = "U" if oy > 0 else "D"
                nxt = step(self._index, self._st, (cx, cy, d))
                if nxt != self._st:
                    self._st = nxt
                    self._left = max(0, self._left - 1)
                    self._draw()
                    if won(self._index, self._st):
                        self.next_level()
                    elif self._left == 0:
                        self.lose()
        self.complete_action()


SOLUTIONS = [[(16, 22), (16, 28), (14, 36), (20, 36), (26, 36), (32, 36), (14, 54), (20, 54), (26, 54), (32, 54), (38, 54)], [(8, 18), (14, 18), (20, 18), (26, 18), (32, 18), (10, 22), (10, 28), (10, 34), (10, 40), (10, 46), (8, 54)], [(16, 16), (14, 24), (20, 24), (26, 24), (34, 22), (34, 28), (34, 34), (36, 42), (30, 42), (24, 42)], [(14, 18), (20, 18), (28, 16), (28, 22), (26, 30), (34, 28), (34, 34), (34, 40), (36, 48), (30, 48), (24, 48), (16, 22), (14, 30), (20, 30), (26, 30), (34, 28), (34, 34), (36, 42), (30, 42), (24, 42)], [(14, 18), (20, 18), (26, 18), (10, 28), (10, 34), (10, 40), (10, 46), (8, 54), (14, 54), (20, 54), (26, 54), (32, 54), (46, 44), (46, 38), (48, 30), (42, 30)]]
