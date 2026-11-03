import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
GOAL = 14
MARK = 0
PALETTE = [8, 9, 11]
BAR = 13
PITCH = 5
OX = 2
OY = 4
N = 12
RADIUS = 4
DEAD = ("dead",)
DECOY = None
DECOY_NAMES = ["scare", "differ"]

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
        "############",
        "############",
        "############",
        "#########E.#",
        "#P...0.....#",
        "############",
        "############",
        "############",
        "############",
        "############",
        "############",
    ],
    [
        "############",
        "############",
        "############",
        "############",
        "#########E.#",
        "#P......1...#",
        "############",
        "############",
        "############",
        "############",
        "############",
        "############",
    ],
    [
        "############",
        "############",
        "############",
        "#####....###",
        "#P...0..2.E#",
        "#####....###",
        "############",
        "############",
        "############",
        "############",
        "############",
        "############",
    ],
    [
        "############",
        "############",
        "############",
        "###.......##",
        "###..0....##",
        "#P....1...E#",
        "###....2..##",
        "###.......##",
        "############",
        "############",
        "############",
        "############",
    ],
    [
        "############",
        "#P....###..#",
        "####.#.#.#.#",
        "#1...2.#.#.#",
        "#.####.#.#.#",
        "#.#..#.#.#.#",
        "#.#2.1...#.#",
        "#.####.###.#",
        "#..0...0...#",
        "###.######.#",
        "#E.....2...#",
        "############",
    ],
]
BUDGETS = [28, 30, 34, 32, 58] * 5


def parse(rows):
    walls, exits, creatures, player = set(), set(), [], None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "E":
                exits.add((x, y))
            elif ch == "P":
                player = (x, y)
            elif ch in "012":
                creatures.append((x, y, int(ch)))
    return frozenset(walls), frozenset(exits), tuple(creatures), player


CACHE = {}


def P(i):
    key = (i, tuple(DATA[i]))
    if key not in CACHE:
        CACHE[key] = parse(DATA[i])
    return CACHE[key]


def sign(v):
    return (v > 0) - (v < 0)


def start(i):
    walls, exits, creatures, player = P(i)
    return (player, 0, creatures)


def tokens(i):
    return ["U", "D", "L", "R", "S"]


def order_of(dx, dy):
    return [(sign(dx), 0), (0, sign(dy))] if abs(dx) >= abs(dy) else [(0, sign(dy)), (sign(dx), 0)]


def step(i, state, tok):
    if state == DEAD:
        return state
    walls = P(i)[0]
    player, pc, creatures = state
    cells = {(c[0], c[1]) for c in creatures}
    if tok == "S":
        pc = (pc + 1) % 3
    else:
        delta = DIRS[tok]
        dest = (player[0] + delta[0], player[1] + delta[1])
        if dest in walls or dest in cells:
            return state
        player = dest
    out = list(creatures)
    for k, (x, y, c) in enumerate(out):
        dx, dy = player[0] - x, player[1] - y
        if abs(dx) + abs(dy) > RADIUS:
            continue
        rel = (pc - c) % 3
        mode = 0
        if rel == 2:
            mode = 1
        elif rel == 1:
            mode = -1
        if DECOY == "scare":
            mode = -1 if pc != 0 else 0
        elif DECOY == "differ":
            mode = -1 if rel != 0 else 0
        if mode == 0:
            continue
        occupied = {(o[0], o[1]) for j, o in enumerate(out) if j != k}
        for sx, sy in order_of(dx * mode, dy * mode):
            if (sx, sy) == (0, 0):
                continue
            dest = (x + sx, y + sy)
            if dest == player and mode == 1:
                return DEAD
            if dest in walls or dest in occupied or dest == player:
                continue
            out[k] = (dest[0], dest[1], c)
            break
    return (player, pc, tuple(out))


def won(i, state):
    if state == DEAD:
        return False
    return state[0] in P(i)[1]


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


class Ng64(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._index = 0
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng64-6a1d0064",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
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
        walls, exits, _, _ = P(self._index)
        player, pc, creatures = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for y in range(N):
            for x in range(N):
                ox, oy = OX + x * PITCH, OY + y * PITCH
                colour = FLOOR
                if (x, y) in walls:
                    colour = WALL
                elif (x, y) in exits:
                    colour = GOAL
                px[oy:oy + PITCH, ox:ox + PITCH] = colour
        for x, y, c in creatures:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = PALETTE[c]
        ox, oy = OX + player[0] * PITCH, OY + player[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PALETTE[pc]
        px[oy + 2, ox + 2] = MARK
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        letter = None
        if self._action.id in MOVES:
            letter = {v: k for k, v in DIRS.items()}[MOVES[self._action.id]]
        elif self._action.id == GameAction.ACTION5:
            letter = "S"
        if letter is not None and self._st != DEAD:
            nxt = step(self._index, self._st, letter)
            if nxt != self._st:
                self._left = max(0, self._left - 1)
                if nxt == DEAD:
                    self.lose()
                else:
                    self._st = nxt
                    self._draw()
                    if won(self._index, self._st):
                        self.next_level()
                    elif self._left == 0:
                        self.lose()
        self.complete_action()


SOLUTIONS = [['R', 'R', 'R', 'S', 'R', 'R', 'R', 'R', 'R', 'U'], ['R', 'R', 'R', 'R', 'S', 'S', 'R', 'R', 'R', 'R', 'U'], ['R', 'R', 'R', 'S', 'R', 'U', 'S', 'R', 'R', 'R', 'D', 'R', 'R'], ['R', 'R', 'U', 'R', 'S', 'R', 'D', 'R', 'R', 'R', 'R', 'R'], ['R', 'R', 'R', 'S', 'L', 'S', 'R', 'D', 'D', 'L', 'L', 'L', 'D', 'D', 'D', 'S', 'D', 'D', 'S', 'R', 'R', 'D', 'D', 'L', 'L']]
