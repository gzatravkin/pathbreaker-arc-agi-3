import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
FLOOR = 4
PLAYER = 8
EXIT = 14
BAR = 11
LEDGER = 7
OX = 2
OY = 8
CELL = 6
COLS = 10
ROWS = 8
COLOURS = {"a": 9, "b": 12, "c": 15}
OPEN_AT_START = {"1": "a", "2": "b", "3": "c"}
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
WALLROW = "##########"

LEVELS_DEF = [
    dict(map=[WALLROW, WALLROW, "#P.a.A.E.#", WALLROW, WALLROW, WALLROW, WALLROW, WALLROW], cap=3, budget=30),
    dict(map=[WALLROW, WALLROW, "#Pa.A.bB.E", WALLROW, WALLROW, WALLROW, WALLROW, WALLROW], cap=3, budget=40),
    dict(map=[WALLROW, WALLROW, "#P1a.A.bBE", WALLROW, WALLROW, WALLROW, WALLROW, WALLROW], cap=2, budget=40),
    dict(map=[WALLROW,
              "#P..1.a.A.",
              "#########.",
              "#E.1.a....",
              WALLROW, WALLROW, WALLROW, WALLROW], cap=2, budget=60),
    dict(map=[WALLROW,
              "#P..1.a.A.",
              "#########.",
              "#E.B.b.1.a",
              WALLROW, WALLROW, WALLROW, WALLROW], cap=3, budget=60),
]


def parse(level):
    walls, gates, levers, start, exit_ = set(), {}, {}, None, None
    for y, row in enumerate(LEVELS_DEF[level]["map"]):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "E":
                exit_ = (x, y)
            elif ch in COLOURS:
                levers[(x, y)] = ch
            elif ch in OPEN_AT_START:
                gates[(x, y)] = (OPEN_AT_START[ch], True)
            elif ch.isupper():
                gates[(x, y)] = (ch.lower(), False)
    return walls, gates, levers, start, exit_


class Model:
    def __init__(self, level):
        self.walls, g, self.levers, self.start, self.exit = parse(level)
        self.gates = {c: k for c, (k, _) in g.items()}
        self.flip = {k: False for k in COLOURS}
        self.base_open = {c: o for c, (_, o) in g.items()}
        self.pos = self.start
        self.ledger = 0
        self.cap = LEVELS_DEF[level]["cap"]
        self.over = False
        self.won = False

    def open(self, cell):
        return self.base_open[cell] != self.flip[self.gates[cell]]

    def move(self, delta):
        dest = (self.pos[0] + delta[0], self.pos[1] + delta[1])
        if dest in self.walls:
            return
        if dest in self.gates and not self.open(dest):
            return
        self.pos = dest
        if dest == self.exit:
            if self.ledger <= self.cap:
                self.won = True
            else:
                self.over = True

    def press(self):
        self.ledger += 1
        if self.pos in self.levers:
            k = self.levers[self.pos]
            self.flip[k] = not self.flip[k]

    def key(self):
        return (self.pos, tuple(self.flip[k] for k in "abc"))


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVELS_DEF))]


class Ng73(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._m = Model(0)
        self._left = 0
        super().__init__(
            game_id="ng73-7c3e5f73",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._m = Model(self._current_level_index)
        self._left = LEVELS_DEF[self._current_level_index]["budget"]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        m = self._m
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for i in range(min(m.ledger, 30)):
            px[4, 2 + 2 * i] = LEDGER
        for y in range(ROWS):
            for x in range(COLS):
                ox, oy = OX + x * CELL, OY + y * CELL
                c = (x, y)
                if c in m.walls:
                    px[oy:oy + CELL, ox:ox + CELL] = WALL
                    continue
                px[oy:oy + CELL, ox:ox + CELL] = FLOOR
                if c in m.levers:
                    col = COLOURS[m.levers[c]]
                    px[oy + 2:oy + 6, ox + 1:ox + 5] = col
                    px[oy:oy + 2, ox + 2:ox + 4] = col
                elif c in m.gates:
                    col = COLOURS[m.gates[c]]
                    px[oy:oy + CELL, ox:ox + CELL] = col
                    if m.open(c):
                        px[oy + 1:oy + 5, ox + 1:ox + 5] = FLOOR
                elif c == m.exit:
                    px[oy:oy + CELL, ox:ox + CELL] = EXIT
                    px[oy + 2:oy + 4, ox + 2:ox + 4] = FLOOR
                    px[oy, ox:ox + m.cap] = BACKGROUND
        px_, py_ = m.pos
        pc = 6 if m.over else PLAYER
        px[OY + py_ * CELL + 1:OY + py_ * CELL + 5, OX + px_ * CELL + 1:OX + px_ * CELL + 5] = pc
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        delta = MOVES.get(aid)
        if delta is not None or aid == GameAction.ACTION5:
            if delta is not None:
                self._m.move(delta)
            else:
                self._m.press()
            self._left = max(0, self._left - 1)
            self._draw()
            if self._m.won:
                self.next_level()
            elif self._m.over or self._left == 0:
                self.lose()
        self.complete_action()


SOLUTIONS = [list(p) for p in [
    "RRSRRRR",
    "RSRRRRSRRR",
    "RRSRRRRSRR",
    "RRRRRSRRRDDLLLLSLLLL",
    "RRRRRSRRRDDSLLLLSLLLL",
]]
