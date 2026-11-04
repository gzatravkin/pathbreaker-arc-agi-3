import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
PEN = 14
PLAYER = 9
SHEEP = 11
OTHER = 12
BAR = 8
PITCH = 5
OX = 2
OY = 4
N = 12
RADIUS = 3
DECOY = None

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

MAPS = [
    [
        "############",
        "#P.........#",
        "#...a......#",
        "#..........#",
        "#########..#",
        "#..........#",
        "#..........#",
        "#########..#",
        "#..........#",
        "#........pp#",
        "#........pp#",
        "############",
    ],
    [
        "############",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..#pp#....#",
        "#..#pp#....#",
        "#..####....#",
        "#..........#",
        "#....a.....#",
        "#.P........#",
        "#..........#",
        "############",
    ],
    [
        "############",
        "#..........#",
        "#.#......#.#",
        "#..a....a..#",
        "#..........#",
        "#....##....#",
        "#...#..#...#",
        "#.P........#",
        "#..........#",
        "#.....pp...#",
        "#.....pp...#",
        "############",
    ],
    [
        "############",
        "#..........#",
        "#..........#",
        "#..........#",
        "#....##....#",
        "#....##.b..#",
        "#...a......#",
        "#..........#",
        "#......#pp##",
        "#P.....#pp##",
        "#......#####",
        "############",
    ],
    [
        "############",
        "#.........p#",
        "#.#.......p#",
        "#..a.b.....#",
        "#..........#",
        "#...##.....#",
        "#...#a.....#",
        "#.P........#",
        "#..........#",
        "#pp........#",
        "#pp........#",
        "############",
    ],
]
BUDGETS = [26, 50, 54, 50, 56]


def parse(rows):
    walls, pens, creatures, player = set(), set(), [], None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "p":
                pens.add((x, y))
            elif ch == "P":
                player = (x, y)
            elif ch in "ab":
                creatures.append((x, y, ch))
    return frozenset(walls), frozenset(pens), tuple(creatures), player


def sign(v):
    return (v > 0) - (v < 0)


def free(walls, cells, cell):
    return cell not in walls and cell not in cells


def flee_step(walls, occupied, pos, threat):
    dx, dy = pos[0] - threat[0], pos[1] - threat[1]
    if abs(dx) + abs(dy) > RADIUS:
        return pos
    order = [(sign(dx), 0), (0, sign(dy))] if abs(dx) >= abs(dy) else [(0, sign(dy)), (sign(dx), 0)]
    for sx, sy in order:
        if (sx, sy) == (0, 0):
            continue
        dest = (pos[0] + sx, pos[1] + sy)
        if free(walls, occupied, dest):
            return dest
    return pos


def advance(state, delta, pens):
    walls, player, creatures = state
    dest = (player[0] + delta[0], player[1] + delta[1])
    cells = {(c[0], c[1]) for c in creatures}
    if dest not in walls and dest not in cells:
        player = dest
    out = list(creatures)
    for i, (x, y, kind) in enumerate(out):
        cells = {(c[0], c[1]) for j, c in enumerate(out) if j != i}
        cells.add(player)
        if kind == "a":
            if DECOY == "dir":
                near = abs(x - player[0]) + abs(y - player[1]) <= RADIUS
                dest = (x + delta[0], y + delta[1])
                if near and free(walls, cells, dest):
                    out[i] = (dest[0], dest[1], kind)
                continue
            if DECOY == "pen":
                near = abs(x - player[0]) + abs(y - player[1]) <= RADIUS
                if near and pens:
                    tx, ty = min(pens, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                    order = [(sign(tx - x), 0), (0, sign(ty - y))]
                    if abs(ty - y) > abs(tx - x):
                        order.reverse()
                    for sx, sy in order:
                        if (sx, sy) != (0, 0) and free(walls, cells, (x + sx, y + sy)):
                            out[i] = (x + sx, y + sy, kind)
                            break
                continue
            nx, ny = flee_step(walls, cells, (x, y), player)
            out[i] = (nx, ny, kind)
        else:
            sheep = [c for c in out if c[2] == "a"]
            if sheep:
                near = min(sheep, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
                nx, ny = flee_step(walls, cells, (x, y), (near[0], near[1]))
                out[i] = (nx, ny, kind)
    return player, tuple(out)


def solved(creatures, pens):
    return all((c[0], c[1]) in pens for c in creatures)


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


LEVELS = [canvas_level(i) for i in range(len(MAPS))]


class Ng61(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._walls = frozenset()
        self._pens = frozenset()
        self._creatures = ()
        self._player = (0, 0)
        self._left = 0
        super().__init__(
            game_id="ng61-6a1d0061",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._walls, self._pens, self._creatures, self._player = parse(MAPS[index])
        self._left = BUDGETS[index]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for y in range(N):
            for x in range(N):
                ox, oy = OX + x * PITCH, OY + y * PITCH
                if (x, y) in self._walls:
                    px[oy:oy + PITCH, ox:ox + PITCH] = WALL
                elif (x, y) in self._pens:
                    px[oy:oy + PITCH, ox:ox + PITCH] = PEN
                else:
                    px[oy:oy + PITCH, ox:ox + PITCH] = FLOOR
        for x, y, kind in self._creatures:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = SHEEP if kind == "a" else OTHER
        ox, oy = OX + self._player[0] * PITCH, OY + self._player[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            self._player, self._creatures = advance((self._walls, self._player, self._creatures), delta, self._pens)
            self._left = max(0, self._left - 1)
            self._draw()
            if solved(self._creatures, self._pens):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()


SOLUTIONS = ['RRRRRRRDDDRDDDD', 'DRRURRUURRURUUUUULLL', 'UULUUUURRRRRRRDDDDDDDR', 'URRUULUUUURRRURRDDDD', 'RRLLULUURRRRRRRDDDDDDLL']
