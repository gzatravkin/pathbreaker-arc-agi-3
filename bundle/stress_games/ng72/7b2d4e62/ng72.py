import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
FLOOR = 4
PLAYER = 8
EXIT = 14
BAR = 11
OX = 2
OY = 6
CELL = 6
COLS = 10
ROWS = 9
INF = 999
COLOURS = {"a": 9, "b": 12, "c": 15, "d": 6}
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVELS_DEF = [
    dict(
        map=["##########", "##########", "##########",
             "#P.a.A..E#",
             "##########", "##########", "##########", "##########", "##########"],
        plates=dict(a=(3, INF)), budget=24),
    dict(
        map=["##########",
             "#P..b..a.#",
             "####A#####",
             "#.......BE",
             "##########", "##########", "##########", "##########", "##########"],
        plates=dict(a=(2, INF), b=(2, INF)), budget=40),
    dict(
        map=["##########", "##########",
             "###ba#####",
             "#P....ABE#",
             "##########", "##########", "##########", "##########", "##########"],
        plates=dict(a=(2, 2), b=(4, 2)), budget=30),
    dict(
        map=["##########", "##########",
             "##cba#####",
             "#P....ABCE",
             "##########", "##########", "##########", "##########", "##########"],
        plates=dict(a=(2, 2), b=(4, 2), c=(6, 2)), budget=34),
    dict(
        map=["##########", "##########",
             "###ba#####",
             "#P....AB.#",
             "########.#",
             "#EDC.....#",
             "#....dc..#",
             "##########", "##########"],
        plates=dict(a=(2, 2), b=(4, 2), c=(3, 2), d=(3, 2)), budget=60),
]

def parse(level):
    d = LEVELS_DEF[level]
    walls, gates, plates, start, exit_ = set(), {}, {}, None, None
    for y, row in enumerate(d["map"]):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "E":
                exit_ = (x, y)
            elif ch.islower():
                plates[(x, y)] = ch
            elif ch.isupper():
                gates[(x, y)] = ch.lower()
    return walls, gates, plates, start, exit_


class Model:
    def __init__(self, level):
        self.level = level
        self.walls, self.gates, self.plates, self.start, self.exit = parse(level)
        self.params = LEVELS_DEF[level]["plates"]
        self.pos = self.start
        self.t = 0
        self.events = []
        self.dead = False
        self.won = False

    def is_open(self, kind, t=None):
        t = self.t if t is None else t
        for kd, s, e in self.events:
            if kd == kind and s <= t < e:
                return True
        return False

    def act(self, delta):
        dx, dy = delta
        dest = (self.pos[0] + dx, self.pos[1] + dy)
        moved = False
        if dest not in self.walls and not (dest in self.gates and not self.is_open(self.gates[dest])):
            self.pos = dest
            moved = True
        self.t += 1
        if moved and self.pos in self.plates:
            kind = self.plates[self.pos]
            k, w = self.params[kind]
            self.events.append((kind, self.t + k, self.t + k + w))
        if self.pos in self.gates and not self.is_open(self.gates[self.pos]):
            self.dead = True
        if self.pos == self.exit:
            self.won = True

    def key(self):
        live = []
        for kd, s, e in self.events:
            if e <= self.t:
                continue
            live.append((kd, max(s - self.t, 0), INF if e - self.t > INF // 2 else e - self.t))
        return (self.pos, tuple(sorted(set(live))))


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVELS_DEF))]


class Ng72(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._m = Model(0)
        self._left = 0
        super().__init__(
            game_id="ng72-7b2d4e62",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
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
        for y in range(ROWS):
            for x in range(COLS):
                ox, oy = OX + x * CELL, OY + y * CELL
                c = (x, y)
                if c in m.walls:
                    px[oy:oy + CELL, ox:ox + CELL] = WALL
                    continue
                px[oy:oy + CELL, ox:ox + CELL] = FLOOR
                if c in m.plates:
                    kind = m.plates[c]
                    px[oy + 1:oy + 5, ox + 1:ox + 5] = COLOURS[kind]
                    k = m.params[kind][0]
                    px[oy + 2, ox:ox + k] = BACKGROUND
                elif c in m.gates:
                    kind = m.gates[c]
                    col = COLOURS[kind]
                    if m.is_open(kind):
                        px[oy:oy + CELL, ox:ox + CELL] = col
                        px[oy + 1:oy + 5, ox + 1:ox + 5] = FLOOR
                    else:
                        px[oy:oy + CELL, ox:ox + CELL] = col
                        w = m.params[kind][1]
                        if w < INF:
                            px[oy + 2, ox + 1:ox + 1 + w] = BACKGROUND
                elif c == m.exit:
                    px[oy:oy + CELL, ox:ox + CELL] = EXIT
                    px[oy + 2:oy + 4, ox + 2:ox + 4] = FLOOR
        px_, py_ = m.pos
        px[OY + py_ * CELL + 1:OY + py_ * CELL + 5, OX + px_ * CELL + 1:OX + px_ * CELL + 5] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            self._m.act(delta)
            self._left = max(0, self._left - 1)
            self._draw()
            if self._m.won:
                self.next_level()
            elif self._m.dead or self._left == 0:
                self.lose()
        self.complete_action()


SOLUTIONS = [list(p) for p in [
    "RRUURRRRR",
    "RRRRRRLLLDDRRRRR",
    "RRURDRRRR",
    "RURRDRRRRR",
    "RRURDRRRRDDDLLLULLLL",
]]
