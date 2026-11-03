import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
PLAYER = 9
GEM = 11
EXIT = 14
TRAP = 15
BAR = 14
ENEMY_COLOURS = (8, 12)
CELL = 6
OX = 5
OY = 7

DELTAS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0), "S": (0, 0)}
ACTION_LETTER = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}

LEVELS_CFG = [
    dict(map=['#########', '#P#.##.##', '#..##...#', '#...#E..#', '#.#...###', '#####G.##', '#..G..#.#', '#..#..#.#', '#########'], enemies=[(6, 'trail')]),
    dict(map=['#########', '#P.#..#.#', '#..#..#.#', '#....G..#', '#.#..##.#', '#..##...#', '#G...#.E#', '##.#.#..#', '#########'], enemies=[(4, 'trail')]),
    dict(map=['#########', '#P##.#..#', '#.G.....#', '#.#.#.###', '#.#...###', '###.#G.E#', '#.#G#...#', '#....#.##', '#########'], enemies=[(3, 'cut')]),
    dict(map=['#########', '#P.##...#', '#...G...#', '#...##..#', '#.....#.#', '#...E...#', '#.#.G...#', '##.G....#', '#########'], enemies=[(2, 'trail')]),
    dict(map=['#########', '#P##....#', '#E...#.##', '##....G.#', '##...#.##', '#...#G..#', '####....#', '##......#', '#########'], enemies=[(5, 'trail'), (3, 'cut')]),
    dict(map=['#########', '#P...G..#', '##..#.#.#', '##..X#..#', '#...#...#', '#.#.....#', '##E...#.#', '#.#...#G#', '#########'], enemies=[(3, 'trail')]),
]

SOLUTIONS = [
    'DDRRDRRDDDULLRSRUUU',
    'DDDDDURSLUURRRRRRDDD',
    'DRRDDDDDLRRLUUURRDRR',
    'DRRRRURDLLLDDDDDRUU',
    'DRDRRRRDDDUDLURUULLULLL',
    'RDDRRLUURRRRDDDDDDUULLDLLL',
]


def parse(cfg):
    rows = cfg["map"]
    walls, gems, traps = set(), [], set()
    start = exit_cell = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "E":
                exit_cell = (x, y)
            elif ch == "G":
                gems.append((x, y))
            elif ch == "X":
                traps.add((x, y))
    return dict(w=len(rows[0]), h=len(rows), walls=walls, gems=gems, traps=traps, start=start,
                exit=exit_cell, enemies=cfg["enemies"], nmax=max([e[0] for e in cfg["enemies"]], default=0), key=tuple(rows))


_DIST = {}


def dist_map(info, target):
    key = (info["key"], target)
    if key in _DIST:
        return _DIST[key]
    dist = {target: 0}
    frontier = [target]
    while frontier:
        nxt = []
        for x, y in frontier:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                c = (x + dx, y + dy)
                if c in dist or c in info["walls"] or not (0 <= c[0] < info["w"] and 0 <= c[1] < info["h"]):
                    continue
                dist[c] = dist[(x, y)] + 1
                nxt.append(c)
        frontier = nxt
    _DIST[key] = dist
    return dist


def initial(info):
    enemies = tuple((info["start"], True) for _ in info["enemies"])
    return (info["start"], (info["start"],), 0, enemies)


def advance(info, st, letter):
    pos, window, gems, enemies = st
    dx, dy = DELTAS[letter]
    new = (pos[0] + dx, pos[1] + dy)
    if letter != "S":
        if new in info["walls"] or not (0 <= new[0] < info["w"] and 0 <= new[1] < info["h"]):
            return None, None
    window2 = (window + (new,))[-(info["nmax"] + 1):]
    out = []
    caught = False
    for (lag, mode), (epos, alive) in zip(info["enemies"], enemies):
        if not alive:
            out.append((epos, False))
            continue
        target = window2[-(lag + 1)] if len(window2) > lag else window2[0]
        if mode == "trail":
            npos = target
        else:
            dm = dist_map(info, target)
            npos = epos
            best = dm.get(epos, 10 ** 6)
            for d in "UDLR":
                c = (epos[0] + DELTAS[d][0], epos[1] + DELTAS[d][1])
                if dm.get(c, 10 ** 6) < best:
                    best = dm[c]
                    npos = c
        if npos in info["traps"]:
            out.append((npos, False))
            continue
        if npos == new or (npos == pos and epos == new):
            caught = True
        out.append((npos, True))
    if caught:
        return (new, window2, gems, tuple(out)), "lose"
    if new in info["gems"]:
        gems |= 1 << info["gems"].index(new)
    st2 = (new, window2, gems, tuple(out))
    full = gems == (1 << len(info["gems"])) - 1
    if full and new == info["exit"]:
        return st2, "win"
    return st2, None


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


class Ng22(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._info = None
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng22-2a220001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._info = parse(LEVELS_CFG[index])
        self._st = initial(self._info)
        self._left = budget_of(index)
        self._draw(level)

    def _cell(self, px, c, colour, inset=0):
        x0, y0 = OX + c[0] * CELL + inset, OY + c[1] * CELL + inset
        size = CELL - 2 * inset
        px[y0:y0 + size, x0:x0 + size] = colour

    def _draw(self, level=None):
        level = level or self.current_level
        info = self._info
        pos, window, gems, enemies = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        full = gems == (1 << len(info["gems"])) - 1
        for y in range(info["h"]):
            for x in range(info["w"]):
                c = (x, y)
                self._cell(px, c, WALL if c in info["walls"] else FLOOR)
        for c in info["traps"]:
            self._cell(px, c, TRAP)
            self._cell(px, c, FLOOR, inset=2)
        for i, g in enumerate(info["gems"]):
            if not gems >> i & 1:
                self._cell(px, g, GEM, inset=2)
        ex = info["exit"]
        self._cell(px, ex, EXIT)
        if not full:
            self._cell(px, ex, FLOOR, inset=1)
        for i, (epos, alive) in enumerate(enemies):
            if alive:
                self._cell(px, epos, ENEMY_COLOURS[i % 2])
        self._cell(px, pos, PLAYER, inset=1)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        letter = ACTION_LETTER.get(self._action.id)
        if letter is not None and self._st is not None:
            st, outcome = advance(self._info, self._st, letter)
            if st is not None:
                self._st = st
                self._left = max(0, self._left - 1)
                self._draw()
                if outcome == "win":
                    self.next_level()
                elif outcome == "lose" or self._left == 0:
                    self.lose()
        self.complete_action()
