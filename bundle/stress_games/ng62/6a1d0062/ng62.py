import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
GOAL = 14
SEEKER = 11
AVOIDER = 10
LURE = 8
BAR = 13
PITCH = 6
OX = 2
OY = 4
N = 10
DECOY = None
DECOY_NAMES = ["any", "near"]

DATA = [
    [
        "##########",
        "#........#",
        "#........#",
        "#.a....g.#",
        "#........#",
        "#.a....g.#",
        "#........#",
        "#.a....g.#",
        "#........#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#a.......#",
        "#........#",
        "#........#",
        "#.......g#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ],
    [
        "##########",
        "#a.......#",
        "#.#####..#",
        "#.#......#",
        "#.#..g...#",
        "#.#......#",
        "#.#####..#",
        "#........#",
        "#........#",
        "##########",
    ],
    [
        "##########",
        "#........#",
        "#g.......#",
        "#....b...#",
        "#........#",
        "###......#",
        "#........#",
        "#........#",
        "#a...g...#",
        "##########",
    ],
    [
        "##########",
        "#.......g#",
        "#........#",
        "#..b.....#",
        "####.....#",
        "#........#",
        "#a.......#",
        "#........#",
        "#g.....g.#",
        "##########",
    ],
]
BUDGETS = [38, 28, 30, 24, 26]


def parse(rows):
    walls, goals, creatures = set(), set(), []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "g":
                goals.add((x, y))
            elif ch in "ab":
                creatures.append((x, y, ch))
    return frozenset(walls), frozenset(goals), tuple(creatures)


def sign(v):
    return (v > 0) - (v < 0)


def clear_line(walls, a, b):
    if a[0] == b[0]:
        step = sign(b[1] - a[1])
        y = a[1] + step
        while y != b[1]:
            if (a[0], y) in walls:
                return False
            y += step
        return True
    if a[1] == b[1]:
        step = sign(b[0] - a[0])
        x = a[0] + step
        while x != b[0]:
            if (x, a[1]) in walls:
                return False
            x += step
        return True
    return False


def start(i):
    return parse(DATA[i])[2]


def tokens(i):
    walls = parse(DATA[i])[0]
    return [(x, y) for y in range(N) for x in range(N) if (x, y) not in walls]


def step(i, creatures, tok):
    walls = parse(DATA[i])[0]
    out = list(creatures)
    if any((c[0], c[1]) == tok for c in out) and DECOY is None:
        pass
    for k, (x, y, kind) in enumerate(out):
        dx, dy = tok[0] - x, tok[1] - y
        if (dx, dy) == (0, 0):
            continue
        sees = clear_line(walls, (x, y), tok)
        if DECOY == "any":
            sees = True
        elif DECOY == "near":
            sees = abs(dx) + abs(dy) <= 8
        if not sees:
            continue
        if dx == 0 or (dy != 0 and abs(dy) > abs(dx)):
            move = (0, sign(dy))
        else:
            move = (sign(dx), 0)
        if kind == "b":
            move = (-move[0], -move[1])
        dest = (x + move[0], y + move[1])
        if dest in walls or any((c[0], c[1]) == dest for c in out):
            continue
        out[k] = (dest[0], dest[1], kind)
    return tuple(out)


def to_px(tok):
    return (OX + tok[0] * PITCH + 2, OY + tok[1] * PITCH + 2)


def natural(i):
    return sorted(parse(DATA[i])[1])


def won(i, creatures):
    goals = parse(DATA[i])[1]
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


LEVELS = [canvas_level(i) for i in range(len(DATA))]


class Ng62(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._index = 0
        self._creatures = ()
        self._lure = None
        self._left = 0
        super().__init__(
            game_id="ng62-6a1d0062",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._index = self._current_level_index
        self._creatures = start(self._index)
        self._lure = None
        self._left = BUDGETS[self._index]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        walls, goals, _ = parse(DATA[self._index])
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
        if self._lure is not None:
            ox, oy = OX + self._lure[0] * PITCH, OY + self._lure[1] * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = LURE
        for x, y, kind in self._creatures:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = SEEKER if kind == "a" else AVOIDER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            cx = (int(data.get("x", -1)) - OX) // PITCH
            cy = (int(data.get("y", -1)) - OY) // PITCH
            walls = parse(DATA[self._index])[0]
            if 0 <= cx < N and 0 <= cy < N and (cx, cy) not in walls:
                self._lure = (cx, cy)
                self._creatures = step(self._index, self._creatures, (cx, cy))
                self._left = max(0, self._left - 1)
                self._draw()
                if won(self._index, self._creatures):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()


SOLUTIONS = [[(46, 24), (46, 24), (46, 24), (46, 36), (46, 36), (46, 36), (46, 36), (46, 48), (46, 48), (46, 48), (46, 36), (46, 24), (46, 48), (46, 24), (46, 48)], [(16, 18), (22, 18), (28, 18), (34, 18), (40, 18), (46, 18), (52, 18), (52, 24), (52, 30), (52, 36)], [(16, 12), (22, 12), (28, 12), (34, 12), (40, 12), (46, 12), (46, 18), (46, 24), (22, 24), (22, 24), (34, 30)], [(40, 24), (34, 24), (16, 54), (22, 54), (28, 18), (28, 18), (28, 54), (34, 54)], [(10, 24), (28, 12), (28, 12), (28, 12), (28, 12), (28, 12), (10, 54), (10, 54), (10, 54)]]
