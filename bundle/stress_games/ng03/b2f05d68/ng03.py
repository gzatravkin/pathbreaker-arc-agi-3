import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
WIRE = 13
SPARK = 11
JOINT = 15
DOOR = 12
EXIT = 14
PLAYER = 9
LEVER = 0
HANDLE = 7
BAR = 10
CELL = 4
OX = 2
OY = 4
BAR_W = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
NEIGHBOURS = ((0, -1), (0, 1), (-1, 0), (1, 0))

CONFIGS = [
    dict(window=4, rows=[
        "###############",
        "#P..#.......#L=",
        "###.#.#####.#.=",
        "#...#.#...#.#.=",
        "#.###.#.#.#.#.=",
        "#.#...#.#.....=",
        "#.#.###.#####.=",
        "#...#...#.....=",
        "###.#.###.###.=",
        "#...#.#...#...=",
        "#.###.#.###.#.=",
        "#.....#.....#.=",
        "=D############=",
        "=E..##########=",
        "===============",
    ], path="RRDDLLDDDDRRUURRUUUURRRRRRDDDDRRUUUSUUUUUDDDLLLLUULLDDDDLLDDDDLLLLDD"),
    dict(window=4, rows=[
        "=============L#",
        "=P..........#.#",
        "=##.#####.#.#.#",
        "=.......#...#.#",
        "=.###.###.###.#",
        "=.#.#.#...#...#",
        "=.#.#.#.###.#.#",
        "=...#...#...#.#",
        "=.#.#########.#",
        "=...#.........#",
        "=.#######.###.#",
        "=.........#...#",
        "=.#############",
        "=......D..D.E.#",
        "===========####",
    ], path="RRDDLLDDDDDDDDRRRRRRRRUURRRRUUUUUUUUSUUUUUDDDDDDDDLLLLDDLLLLLLLLDDRRRRRRRRRRR"),
    dict(window=4, rows=[
        "###############",
        "#P......#.....#",
        "#######.#.#.#.#",
        "#.....#...#...#",
        "#.###.#######.#",
        "#...#.#...#...#",
        "#.#.###.#.#.###",
        "#.#...........#",
        "#.#############",
        "#.......D.D..E#",
        "#L#L####=#=####",
        "#=#=====X==####",
        "#=######=######",
        "#========######",
        "###############",
    ], path="RRRRRRDDRRUURRDDRRDDLLDDLLLLLLLLUULLDDDDSUDDRRSUUURRRRRRRRRR"),
    dict(window=4, rows=[
        "=====L#########",
        "=.....#.......#",
        "=####.#.###.#.#",
        "=...#.#.#.....#",
        "=.#.#.#.#.#####",
        "=.#...#.#...#.#",
        "=.#.###.###.#.#",
        "=.......#....P#",
        "=####.#########",
        "=.............#",
        "=D==########=D#",
        "#..L########=.#",
        "###=########=E#",
        "###===#======##",
        "#####===#######",
    ], path="LLUULLUURRUULLLLDDDDDDLLLLUURRUUUUSDDDDLLDDRRDDLLLLDDRSULUURRRRRRRRRRRRDDD"),
    dict(window=4, rows=[
        "###############",
        "#.#..........P#",
        "#.#.#.#########",
        "#.#...........#",
        "#.#.#########.#",
        "#.#.#.......#.#",
        "#.###.###.#.#.#",
        "#.....#.......#",
        "#.###=====#####",
        "#....L#=#=#####",
        "L......DDD..E.#",
        "=....###=######",
        "=#######=######",
        "=========######",
        "###############",
    ], path="LLLLLLLLDDRRRRRRRRDDDDLLUULLLLLLDDLLLLDDDSUUDLRRRSUDRRRRRRRR"),
    dict(window=3, rows=[
        "===L###########",
        "=.....#.....#P#",
        "=####.#.###.#.#",
        "=.....#...#...#",
        "=.#.#####.#.#.#",
        "=.........#...#",
        "=##.######===##",
        "=##.##L====#=##",
        "=..........DDE#",
        "=D==#######=###",
        "#..L#######=###",
        "###=#######=###",
        "###=#===#===###",
        "###===#===#####",
        "###############",
    ], path="DDLLUULLLLDDRRDDLLLLLLUURRUULLSRRDDLLDDDDDLLDDRSULUURRRRRSUUURRRRRRR"),
]

WIRED = set("+=xX")
CROSS = set("xX")


def parse(cfg):
    rows = cfg["rows"]
    info = dict(walls=set(), wires=set(), cross=set(), levers=[], doors=[], start=None, exit=None,
                w=max(len(r) for r in rows), h=len(rows))
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            cell = (c, r)
            if ch in "#=X":
                info["walls"].add(cell)
            if ch in WIRED:
                info["wires"].add(cell)
            if ch in CROSS:
                info["cross"].add(cell)
            if ch == "L":
                info["levers"].append(cell)
            if ch == "D":
                info["doors"].append(cell)
            if ch == "P":
                info["start"] = cell
            if ch == "E":
                info["exit"] = cell
    return info


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def spark_moves(info, spark):
    cell, prev = spark[0], spark[1]
    if cell in info["cross"]:
        ahead = (2 * cell[0] - prev[0], 2 * cell[1] - prev[1])
        return [ahead]
    out = []
    for d in NEIGHBOURS:
        n = add(cell, d)
        if n == prev:
            continue
        if n in info["wires"] or n in info["doors"]:
            out.append(n)
    return out


def advance(info, sparks, opened):
    moved = set()
    for spark in sparks:
        for n in spark_moves(info, spark):
            if n in info["doors"]:
                opened.add(n)
            elif n in info["wires"]:
                moved.add((n, spark[0], spark[2]))
    return frozenset(moved)


def emit(info, lever):
    out = set()
    for d in NEIGHBOURS:
        n = add(lever, d)
        if n in info["wires"]:
            out.add((n, lever, lever))
    return out


def passable(info, timers, cell):
    if cell in info["walls"] or cell in info["levers"]:
        return False
    if cell in info["doors"]:
        return timers.get(cell, 0) > 0
    return 0 <= cell[0] < info["w"] and 0 <= cell[1] < info["h"]


def tick(info, window, player, sparks, timers, action):
    timers = dict(timers)
    pulls = []
    if action in MOVES:
        dest = add(player, MOVES[action])
        if passable(info, timers, dest):
            player = dest
    else:
        busy = {spark[2] for spark in sparks}
        for d in NEIGHBOURS:
            n = add(player, d)
            if n in info["levers"] and n not in busy:
                pulls.append(n)
    opened = set()
    sparks = advance(info, sparks, opened)
    closing = False
    for door, t in list(timers.items()):
        if t > 0:
            timers[door] = t - 1
            if t == 1 and player == door:
                closing = True
    for door in opened:
        timers[door] = window
    for lever in pulls:
        sparks = sparks | emit(info, lever)
    crushed = closing and timers.get(player, 0) == 0
    return player, frozenset(sparks), timers, pulls, crushed


def budget_of(cfg):
    return 2 * len(cfg["path"]) + 12


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
SOLUTIONS = [list(cfg["path"]) for cfg in CONFIGS]


class Ng03(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._info = parse(self._cfg)
        self._player = (0, 0)
        self._sparks = frozenset()
        self._timers = {}
        self._left = 0
        super().__init__(
            game_id="ng03-b2f05d68",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._info = parse(self._cfg)
        self._player = self._info["start"]
        self._sparks = frozenset()
        self._timers = {d: 0 for d in self._info["doors"]}
        self._left = budget_of(self._cfg)
        self._draw(level)

    def _links(self, cell):
        info = self._info
        out = []
        for d in NEIGHBOURS:
            n = add(cell, d)
            if n in info["wires"] or n in info["doors"] or n in info["levers"]:
                out.append(d)
        return out

    def _paint_wire(self, px, cell, colour):
        ox, oy = OX + CELL * cell[0], OY + CELL * cell[1]
        px[oy + 1:oy + 3, ox + 1:ox + 3] = colour
        for dx, dy in self._links(cell):
            if dx == -1:
                px[oy + 1:oy + 3, ox] = colour
            elif dx == 1:
                px[oy + 1:oy + 3, ox + 3] = colour
            elif dy == -1:
                px[oy, ox + 1:ox + 3] = colour
            else:
                px[oy + 3, ox + 1:ox + 3] = colour

    def _draw(self, level=None):
        level = level or self.current_level
        info = self._info
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // BAR_W, OX + i % BAR_W] = BAR
        for c, r in info["walls"]:
            px[OY + CELL * r:OY + CELL * r + CELL, OX + CELL * c:OX + CELL * c + CELL] = WALL
        lit = {s[0] for s in self._sparks}
        busy = {s[2] for s in self._sparks}
        for cell in info["wires"]:
            self._paint_wire(px, cell, SPARK if cell in lit else WIRE)
            if cell not in info["cross"] and len(self._links(cell)) >= 3 and cell not in lit:
                ox, oy = OX + CELL * cell[0], OY + CELL * cell[1]
                px[oy + 1:oy + 3, ox + 1:ox + 3] = JOINT
        for cell in info["levers"]:
            ox, oy = OX + CELL * cell[0], OY + CELL * cell[1]
            px[oy + 3, ox:ox + 4] = LEVER
            px[oy + 2, ox + 1:ox + 3] = LEVER
            if cell in busy:
                px[oy, ox + 3] = HANDLE
                px[oy + 1, ox + 2] = HANDLE
            else:
                px[oy, ox] = HANDLE
                px[oy + 1, ox + 1] = HANDLE
        for cell in info["doors"]:
            ox, oy = OX + CELL * cell[0], OY + CELL * cell[1]
            shut = CELL - min(CELL, self._timers[cell])
            px[oy:oy + shut, ox:ox + CELL] = DOOR
        c, r = info["exit"]
        px[OY + CELL * r:OY + CELL * r + CELL, OX + CELL * c:OX + CELL * c + CELL] = EXIT
        px[OY + CELL * r + 1:OY + CELL * r + 3, OX + CELL * c + 1:OX + CELL * c + 3] = BACKGROUND
        c, r = self._player
        px[OY + CELL * r:OY + CELL * r + CELL, OX + CELL * c:OX + CELL * c + CELL] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        action = self._action.id
        if action not in MOVES and action != GameAction.ACTION5:
            self.complete_action()
            return
        info = self._info
        self._player, self._sparks, self._timers, pulls, crushed = tick(
            info, self._cfg["window"], self._player, self._sparks, self._timers, action)
        self._left = max(0, self._left - 1)
        self._draw()
        if crushed:
            self.lose()
        elif self._player == info["exit"]:
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
