import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
COIN = 11
PLAYER = 9
GATE = 13
GATE_PIP = 0
EXIT = 14
FLIP = 15
HUD = 8
BAR = 10
PITCH = 6
OX = 2
OY = 5
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVELS_DEF = [['#########', '#P.c...##', '#.c.##.##', '#...##2##', '######E##'], ['###########', '#P.h.h..c.#', '#.###.###.#', '#h..h....c#', '######2####', '######E####'], ['#########', '#P.c.c.h#', '#c.#.#.##', '#.c.....#', '####3####', '#c.h.c###', '####2####', '####E####'], ['###########', '#P.c.h.f..#', '#.#.###.#.#', '#c..h.h...#', '#####3#####', '#####E#####'], ['#########', '#P.c.h.f#', '#c.#.#.##', '#.c.....#', '####4####', '#cfh.c###', '####3####', '####E####']]
SOLUTIONS = [['D', 'R', 'U', 'R', 'R', 'R', 'R', 'D', 'D', 'D'], ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D', 'D', 'L', 'L', 'L', 'D', 'D'], ['D', 'D', 'R', 'U', 'U', 'R', 'R', 'R', 'L', 'D', 'D', 'S', 'D', 'U', 'D', 'D', 'R', 'L', 'D', 'D'], ['D', 'D', 'U', 'U', 'R', 'R', 'R', 'R', 'R', 'R', 'S', 'D', 'U', 'D', 'D', 'L', 'L', 'D', 'D'], ['D', 'D', 'R', 'U', 'U', 'R', 'R', 'R', 'R', 'R', 'S', 'L', 'R', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'L', 'L', 'S', 'L', 'R', 'R', 'R', 'R', 'L', 'D', 'D']]
BUDGETS = [28, 38, 48, 46, 70]


def parse(rows):
    walls, coins, gates, flips, start, exit_cell = set(), {}, {}, set(), None, None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "E":
                exit_cell = (x, y)
            elif ch == "c":
                coins[(x, y)] = 1
            elif ch == "h":
                coins[(x, y)] = 0
            elif ch == "f":
                flips.add((x, y))
            elif ch in "1234":
                gates[(x, y)] = int(ch)
    return walls, coins, gates, flips, start, exit_cell, len(rows[0]), len(rows)


def pip_cells(count):
    cells = {1: [(2, 2)], 2: [(1, 1), (3, 3)], 3: [(1, 1), (3, 3), (2, 2)], 4: [(1, 1), (3, 3), (3, 1), (1, 3)]}
    return cells[count]


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


class Ng83(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._walls = set()
        self._coins = {}
        self._gates = {}
        self._flips = set()
        self._carry = []
        self._pos = (0, 0)
        self._exit = (0, 0)
        self._w = 1
        self._h = 1
        self._left = 0
        self._total = 1
        super().__init__(
            game_id="ng83-8a830001",
            levels=[canvas_level(i) for i in range(len(LEVELS_DEF))],
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS_DEF),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        (self._walls, self._coins, self._gates, self._flips, self._pos, self._exit,
         self._w, self._h) = parse(LEVELS_DEF[index])
        self._carry = []
        self._total = BUDGETS[index]
        self._left = self._total
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        n = (self._left * BAR_MAX + self._total - 1) // self._total
        px[61:63, 2:2 + n] = BAR
        for i in range(len(self._carry)):
            px[1:3, 2 + i * 3:4 + i * 3] = HUD
        for (x, y) in self._walls:
            px[OY + y * PITCH:OY + y * PITCH + PITCH, OX + x * PITCH:OX + x * PITCH + PITCH] = WALL
        ex, ey = OX + self._exit[0] * PITCH, OY + self._exit[1] * PITCH
        px[ey:ey + 5, ex:ex + 5] = EXIT
        px[ey + 1:ey + 4, ex + 1:ex + 4] = BACKGROUND
        for (x, y) in self._flips:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            for k in range(5):
                px[oy + k, ox + k] = FLIP
                px[oy + k, ox + 4 - k] = FLIP
        for (x, y), kind in self._coins.items():
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy + 1:oy + 4, ox + 1:ox + 4] = COIN
            if kind == 0:
                px[oy + 2, ox + 2] = BACKGROUND
        for (x, y), count in self._gates.items():
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + 5, ox:ox + 5] = GATE
            for cx, cy in pip_cells(count):
                px[oy + cy, ox + cx] = GATE_PIP
        ox, oy = OX + self._pos[0] * PITCH, OY + self._pos[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        action = self._action.id
        delta = MOVES.get(action)
        changed = False
        won = False
        if delta is not None:
            dest = (self._pos[0] + delta[0], self._pos[1] + delta[1])
            inside = 0 <= dest[0] < self._w and 0 <= dest[1] < self._h
            if inside and dest not in self._walls:
                ok = True
                if dest in self._gates:
                    need = self._gates[dest]
                    if sum(self._carry) == need:
                        self._carry = [k for k in self._carry if k == 0]
                        del self._gates[dest]
                    else:
                        ok = False
                if ok:
                    self._pos = dest
                    if dest in self._coins:
                        self._carry.append(self._coins.pop(dest))
                    self._left = max(0, self._left - 1)
                    changed = True
                    won = dest == self._exit
        elif action == GameAction.ACTION5:
            self._left = max(0, self._left - 1)
            changed = True
            if self._carry and self._pos not in self._coins:
                kind = self._carry.pop()
                if self._pos in self._flips:
                    kind = 1 - kind
                self._coins[self._pos] = kind
        if changed:
            self._draw()
            if won:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
