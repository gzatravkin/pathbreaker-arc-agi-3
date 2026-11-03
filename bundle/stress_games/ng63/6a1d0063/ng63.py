import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
GOAL = 14
PLAYER = 9
COL_A = 11
COL_B = 12
BAR = 8
PITCH = 5
OX = 2
OY = 4
N = 12
DECOY = None
DECOY_NAMES = ["player", "nearest", "anycolour"]

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

DATA = [
    [
        "############",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#.a...G....#",
        "#..........#",
        "#..........#",
        "#.P......A.#",
        "#..........#",
        "#..........#",
        "############",
    ],
    [
        "############",
        "#..........#",
        "#..........#",
        "#.a.A......#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#.P........#",
        "#........G.#",
        "#..........#",
        "############",
    ],
    [
        "############",
        "#..A...A...#",
        "#.....#....#",
        "#.....#....#",
        "#.....#....#",
        "#.....#....#",
        "#.a...#...G#",
        "#.....#....#",
        "#.....#....#",
        "#.P...#....#",
        "#.....#....#",
        "############",
    ],
    [
        "############",
        "#..........#",
        "#.......G..#",
        "#.a...#....#",
        "#.....#....#",
        "#.....A....#",
        "#.....B....#",
        "#.....#....#",
        "#.b...#....#",
        "#.P...#.H..#",
        "#.....#....#",
        "############",
    ],
    [
        "############",
        "#...#...#..#",
        "#...A...#..#",
        "#...#...#.G#",
        "#.a.#...A..#",
        "#...#...#..#",
        "#.b.#...B..#",
        "#...#...#..#",
        "#...B...#..#",
        "#...#...#.H#",
        "#P..#...#..#",
        "############",
    ],
]
BUDGETS = [28, 48, 60, 56, 60] * 5


def parse(rows):
    walls, goals, beacons, creatures, player = set(), set(), {}, [], None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "g":
                goals.add((x, y))
            elif ch == "P":
                player = (x, y)
            elif ch in "ab":
                creatures.append((x, y, ch))
            elif ch in "AB":
                beacons[(x, y)] = ch.lower()
            elif ch in "GH":
                beacons[(x, y)] = "a" if ch == "G" else "b"
                goals.add((x, y))
    return frozenset(walls), frozenset(goals), beacons, tuple(creatures), player


CACHE = {}


def P(i):
    key = (i, tuple(DATA[i]))
    if key not in CACHE:
        CACHE[key] = parse(DATA[i])
    return CACHE[key]


def sign(v):
    return (v > 0) - (v < 0)


def start(i):
    walls, goals, beacons, creatures, player = P(i)
    return (player, creatures, None, None)


def tokens(i):
    return ["U", "D", "L", "R"]


def toward(walls, cells, pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if (dx, dy) == (0, 0):
        return pos
    order = [(sign(dx), 0), (0, sign(dy))] if abs(dx) >= abs(dy) else [(0, sign(dy)), (sign(dx), 0)]
    for sx, sy in order:
        if (sx, sy) == (0, 0):
            continue
        dest = (pos[0] + sx, pos[1] + sy)
        if dest not in walls and dest not in cells:
            return dest
    return pos


def step(i, state, tok):
    walls, goals, beacons, _, _ = P(i)
    player, creatures, last_a, last_b = state
    delta = DIRS[tok]
    dest = (player[0] + delta[0], player[1] + delta[1])
    cells = {(c[0], c[1]) for c in creatures}
    if dest in walls or dest in cells:
        return state
    player = dest
    last_any = None
    if player in beacons:
        if beacons[player] == "a":
            last_a = player
        else:
            last_b = player
        last_any = player
    if DECOY == "anycolour" and last_any is not None:
        last_a = last_b = last_any
    out = list(creatures)
    for k, (x, y, kind) in enumerate(out):
        cells = {(c[0], c[1]) for j, c in enumerate(out) if j != k}
        cells.add(player)
        target = last_a if kind == "a" else last_b
        if DECOY == "player":
            target = player
        elif DECOY == "nearest":
            own = [p for p, c in beacons.items() if c == kind]
            target = min(own, key=lambda p: abs(p[0] - x) + abs(p[1] - y)) if own else None
        if target is None:
            continue
        nx, ny = toward(walls, cells, (x, y), target)
        out[k] = (nx, ny, kind)
    return (player, tuple(out), last_a, last_b)


def won(i, state):
    goals = P(i)[1]
    return all((c[0], c[1]) in goals for c in state[1])


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


class Ng63(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._index = 0
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng63-6a1d0063",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
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
        walls, goals, beacons, _, _ = P(self._index)
        player, creatures, last_a, last_b = self._st
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
                px[oy:oy + PITCH, ox:ox + PITCH] = colour
        for (x, y), kind in beacons.items():
            ox, oy = OX + x * PITCH, OY + y * PITCH
            colour = COL_A if kind == "a" else COL_B
            px[oy + 1:oy + 4, ox + 1:ox + 4] = colour
            px[oy + 2, ox + 2] = FLOOR if (x, y) not in goals else GOAL
        for x, y, kind in creatures:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = COL_A if kind == "a" else COL_B
        ox, oy = OX + player[0] * PITCH, OY + player[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            letter = {v: k for k, v in DIRS.items()}[delta]
            nxt = step(self._index, self._st, letter)
            if nxt != self._st:
                self._st = nxt
                self._left = max(0, self._left - 1)
                self._draw()
                if won(self._index, self._st):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()


SOLUTIONS = [['U', 'U', 'R', 'U', 'R', 'R', 'R', 'U', 'U', 'U'], ['D', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'D', 'U', 'D', 'U'], ['U', 'U', 'R', 'U', 'U', 'U', 'U', 'U', 'U', 'R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'U', 'U', 'U', 'D', 'U'], ['R', 'U', 'U', 'U', 'R', 'R', 'R', 'U', 'R', 'U', 'U', 'U', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'U', 'U', 'U', 'U'], ['U', 'U', 'R', 'R', 'R', 'R', 'U', 'U', 'U', 'U', 'U', 'U', 'L', 'R', 'D', 'D', 'R', 'R', 'R', 'L', 'D', 'D', 'R', 'R', 'D', 'D', 'D', 'R', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U']]
