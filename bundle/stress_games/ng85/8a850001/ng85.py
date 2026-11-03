import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
CRATE = 10
PAD = 14
PLAYER = 9
BAR = 11
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

LEVELS_DEF = [['########', '#P.....#', '#.i...t#', '#t.i...#', '########'], ['########', '#...#..#', '#..o#..#', '##P....#', '#......#', '#.i..tt#', '########'], ['########', '#t.#t.P#', '#t.....#', '#....o.#', '#.ii...#', '#......#', '########'], ['########', '#.ot#..#', '#..Poi.#', '#...#..#', '##..#.t#', '#.....t#', '##t..i.#', '########'], ['########', '#..i.t##', '##.#.P.#', '#.t#...#', '#......#', '#io#.i.#', '#tt...##', '########']]
SOLUTIONS = [['D', 'R', 'R', 'R', 'D', 'L'], ['U', 'U', 'R', 'D', 'D', 'L', 'D', 'L', 'D', 'R', 'U', 'R', 'R', 'U', 'R', 'D'], ['D', 'D', 'L', 'D', 'D', 'L', 'L', 'U', 'L', 'D', 'L', 'U', 'U', 'U', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'L', 'L'], ['D', 'D', 'D', 'R', 'R', 'U', 'U', 'R', 'U', 'U', 'L', 'D', 'D', 'D', 'R', 'D', 'D', 'L', 'L', 'U', 'R', 'L', 'L', 'U', 'U', 'U', 'L', 'L', 'U', 'R', 'D', 'R', 'R', 'R', 'U', 'R', 'D', 'D'], ['D', 'D', 'L', 'L', 'L', 'U', 'U', 'U', 'R', 'R', 'D', 'D', 'D', 'R', 'R', 'D', 'L', 'D', 'L', 'L', 'L', 'U', 'U', 'L', 'D', 'U', 'R', 'R', 'R', 'D', 'R', 'D', 'L', 'L']]
BUDGETS = [20, 40, 54, 84, 76]


def parse(rows):
    walls, crates, pads, start = set(), {}, set(), None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "i":
                crates[(x, y)] = 1
            elif ch == "o":
                crates[(x, y)] = 0
            elif ch == "t":
                pads.add((x, y))
    return walls, crates, pads, start, len(rows[0]), len(rows)


def push(walls, crates, w, h, pos, delta):
    dest = (pos[0] + delta[0], pos[1] + delta[1])
    if dest in walls or not (0 <= dest[0] < w and 0 <= dest[1] < h):
        return None
    if dest not in crates:
        return dest, crates
    crates = dict(crates)
    kind = crates.pop(dest)
    free = lambda c: c not in walls and c not in crates and 0 <= c[0] < w and 0 <= c[1] < h
    nxt = (dest[0] + delta[0], dest[1] + delta[1])
    if not free(nxt):
        return None
    if kind == 0:
        crates[nxt] = 0
        return dest, crates
    cur = nxt
    while True:
        after = (cur[0] + delta[0], cur[1] + delta[1])
        if free(after):
            cur = after
        else:
            break
    crates[cur] = 0
    return dest, crates


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


class Ng85(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._walls = set()
        self._crates = {}
        self._pads = set()
        self._pos = (0, 0)
        self._w = 1
        self._h = 1
        self._left = 0
        self._total = 1
        super().__init__(
            game_id="ng85-8a850001",
            levels=[canvas_level(i) for i in range(len(LEVELS_DEF))],
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS_DEF),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._walls, self._crates, self._pads, self._pos, self._w, self._h = parse(LEVELS_DEF[index])
        self._total = BUDGETS[index]
        self._left = self._total
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        n = (self._left * BAR_MAX + self._total - 1) // self._total
        px[1:3, 2:2 + n] = BAR
        for (x, y) in self._walls:
            px[OY + y * PITCH:OY + y * PITCH + PITCH, OX + x * PITCH:OX + x * PITCH + PITCH] = WALL
        for (x, y) in self._pads:
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + 5, ox:ox + 5] = PAD
            px[oy + 1:oy + 4, ox + 1:ox + 4] = BACKGROUND
        for (x, y), kind in self._crates.items():
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + 5, ox:ox + 5] = CRATE
            if kind == 1:
                px[oy + 1:oy + 4, ox + 1:ox + 4] = BACKGROUND
                px[oy + 2, ox + 2] = CRATE
        ox, oy = OX + self._pos[0] * PITCH, OY + self._pos[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            result = push(self._walls, self._crates, self._w, self._h, self._pos, delta)
            if result is not None:
                self._pos, self._crates = result
                self._left = max(0, self._left - 1)
                self._draw()
                if self._pads <= set(self._crates):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()
