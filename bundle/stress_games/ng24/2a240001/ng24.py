import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
PLAYER = 10
EXIT = 1
BAR = 14
CELL = 6
OX = 2
OY = 8

SHAPES = {
    0: ("1111", "1111", "1111", "1111"),
    1: ("1111", "1001", "1001", "1111"),
    2: ("0110", "0110", "1111", "1111"),
    3: ("1001", "0110", "0110", "1001"),
}

DELTAS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
ACTION_LETTER = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}

LEVELS_CFG = [
    dict(
        map=['##########', '#..#..#..#', '#.aA..#..#', '#..#..#..#', '#..#.##..#', '#..#..#..#', '#P.#.bB..#', '#..#..#.E#', '##########'],
        link={'a': ['A'], 'b': ['B']},
        attrs={'A': (8, 0), 'a': (8, 0), 'C': (11, 1), 'c': (11, 1), 'B': (12, 2), 'b': (12, 2)},
    ),
    dict(
        map=['##########', '#..#..#..#', '#..A.c#..#', '#..#..#..#', '#..#C##..#', '#..#b.#..#', '#P.#..B..#', '#.a#..#.E#', '##########'],
        link={'a': ['A'], 'c': ['C'], 'b': ['B']},
        attrs={'A': (12, 0), 'C': (8, 1), 'B': (11, 2), 'a': (11, 0), 'c': (12, 1), 'b': (8, 2)},
    ),
    dict(
        map=['##########', '#..#..#..#', '#a.A..#..#', '#..#..#..#', '#.c#C##..#', '#..#..#..#', '#P.#b.B..#', '#..#..#.E#', '##########'],
        link={'a': ['A'], 'c': ['C'], 'b': ['B']},
        attrs={'A': (12, 0), 'C': (8, 1), 'B': (11, 2), 'a': (8, 2), 'c': (11, 0), 'b': (12, 1)},
    ),
    dict(
        map=['##########', '#..#..#..#', '#..A.c#..#', '#..#..#..#', '#..#C##..#', '#..#b.#..#', '#P.#..B..#', '#.a#..#.E#', '##########'],
        link={'a': ['A'], 'c': ['C'], 'b': ['B']},
        attrs={'A': (12, 0), 'C': (8, 1), 'B': (11, 2), 'a': (11, 0), 'c': (12, 1), 'b': (8, 2)},
        second=True,
    ),
    dict(
        map=['##########', '#..#..#..#', '#..A.c#..#', '#..#..#..#', '#..#C##..#', '#..#b.#..#', '#P.#..B..#', '#.a#..#.E#', '##########'],
        link={'a': ['A'], 'c': ['C', 'A'], 'b': ['B']},
        attrs={'A': (12, 0), 'C': (8, 1), 'B': (11, 2), 'a': (11, 0), 'c': (12, 1), 'b': (8, 2)},
    ),
    dict(
        map=['##########', '#..#..#..#', '#a.A..#..#', '#..#..#..#', '#..#.##..#', '#..#..#..#', '#P.#b.B..#', '#..#..#.E#', '##########'],
        link={'a': ['A'], 'b': ['B']},
        attrs={'A': (12, 0), 'B': (11, 2), 'a': (8, 2), 'b': (12, 1)},
        timer=4,
    ),
]

SOLUTIONS = [
    'UUUURSRRDDDDRSRRDR',
    'DRSUUUUURRRSDLDDSDRRRDR',
    'UURSUULSRRRDDDDSRRRDR',
    'DRSSUUUUURRRSSDLDDSSDRRRDR',
    'DRSUUUUURRRSDLDDSDRRRDR',
    'UUUUSRRRDDDDSRRRDR',
]


def parse(cfg):
    walls, buttons, gates = set(), {}, {}
    start = exit_cell = None
    for y, row in enumerate(cfg["map"]):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch == "E":
                exit_cell = (x, y)
            elif ch.islower():
                buttons[ch] = (x, y)
            elif ch.isupper():
                gates[ch] = (x, y)
    gl = sorted(gates)
    bl = sorted(buttons)
    return dict(
        w=len(cfg["map"][0]), h=len(cfg["map"]), walls=walls, buttons=buttons, gates=gates, gl=gl, bl=bl,
        start=start, exit=exit_cell, link=cfg["link"], attrs=cfg["attrs"],
        second=cfg.get("second", False), timer=cfg.get("timer", 0),
        button_at={v: k for k, v in buttons.items()}, gate_at={v: k for k, v in gates.items()},
    )


def initial(info):
    return (info["start"], 0, 0, tuple(0 for _ in info["gl"]))


def advance(info, st, letter):
    pos, opened, counts, timers = st
    gl = info["gl"]
    if letter == "S":
        b = info["button_at"].get(pos)
        if b is None:
            return None, None
        bi = info["bl"].index(b)
        counts ^= 1 << bi
        fresh = set()
        if not info["second"] or not counts >> bi & 1:
            timers = list(timers)
            for g in info["link"][b]:
                gi = gl.index(g)
                if info["gates"][g] == pos:
                    continue
                opened ^= 1 << gi
                if opened >> gi & 1:
                    timers[gi] = info["timer"]
                    fresh.add(gi)
                else:
                    timers[gi] = 0
            timers = tuple(timers)
        new = pos
    else:
        dx, dy = DELTAS[letter]
        new = (pos[0] + dx, pos[1] + dy)
        if new in info["walls"] or not (0 <= new[0] < info["w"] and 0 <= new[1] < info["h"]):
            return None, None
        g = info["gate_at"].get(new)
        if g is not None and not opened >> gl.index(g) & 1:
            return None, None
        fresh = set()
    timers = list(timers)
    for gi, g in enumerate(gl):
        if opened >> gi & 1 and timers[gi] > 0 and gi not in fresh:
            timers[gi] -= 1
            if timers[gi] == 0:
                if info["gates"][g] == new:
                    timers[gi] = 1
                else:
                    opened &= ~(1 << gi)
    st2 = (new, opened, counts, tuple(timers))
    return st2, ("win" if new == info["exit"] else None)


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


class Ng24(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._info = None
        self._st = None
        self._left = 0
        super().__init__(
            game_id="ng24-2a240001",
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

    @staticmethod
    def _glyph(px, c, shape, colour, fill=False):
        x0, y0 = OX + c[0] * CELL + 1, OY + c[1] * CELL + 1
        for r, row in enumerate(SHAPES[shape]):
            for k, ch in enumerate(row):
                if ch == "1":
                    px[y0 + r, x0 + k] = colour

    def _draw(self, level=None):
        level = level or self.current_level
        info = self._info
        pos, opened, counts, timers = self._st
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for y in range(info["h"]):
            for x in range(info["w"]):
                x0, y0 = OX + x * CELL, OY + y * CELL
                px[y0:y0 + CELL, x0:x0 + CELL] = WALL if (x, y) in info["walls"] else FLOOR
        for b, c in info["buttons"].items():
            colour, shape = info["attrs"][b]
            self._glyph(px, c, shape, colour)
        for gi, g in enumerate(info["gl"]):
            c = info["gates"][g]
            colour, shape = info["attrs"][g]
            x0, y0 = OX + c[0] * CELL, OY + c[1] * CELL
            if opened >> gi & 1:
                px[y0:y0 + CELL, x0:x0 + CELL] = colour
                px[y0 + 1:y0 + CELL - 1, x0 + 1:x0 + CELL - 1] = FLOOR
                self._glyph(px, c, shape, colour)
                for k in range(timers[gi]):
                    dx, dy = ((1, 1), (4, 1), (1, 4), (4, 4))[k % 4]
                    px[y0 + dy, x0 + dx] = BACKGROUND
            else:
                px[y0:y0 + CELL, x0:x0 + CELL] = colour
                for r, row in enumerate(SHAPES[shape]):
                    for k, ch in enumerate(row):
                        if ch == "1":
                            px[y0 + 1 + r, x0 + 1 + k] = BACKGROUND
        ex = info["exit"]
        x0, y0 = OX + ex[0] * CELL, OY + ex[1] * CELL
        px[y0:y0 + CELL, x0:x0 + CELL] = EXIT
        px[y0 + 2:y0 + 4, x0 + 2:x0 + 4] = BACKGROUND
        x0, y0 = OX + pos[0] * CELL, OY + pos[1] * CELL
        under = px[y0 + 1:y0 + CELL - 1, x0 + 1:x0 + CELL - 1].copy()
        px[y0:y0 + CELL, x0:x0 + CELL] = PLAYER
        px[y0 + 1:y0 + CELL - 1, x0 + 1:x0 + CELL - 1] = under
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
                elif self._left == 0:
                    self.lose()
        self.complete_action()
