import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 2
WALL = 4
PIT = 13
BELT = 3
ARROW_CW = 14
ARROW_CCW = 12
PLAYER = 8
EXIT = 11
BAR = 10
ORIGIN_X = 5
ORIGIN_Y = 6
PITCH = 6
SIDE = 5
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
    GameAction.ACTION5: "S",
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

CONFIGS = [{'rows': ['Sc.b..d', '...d.b.', '.......', 'a##.a#.', '.b~.~d.', 'd..#..~', '#..~a.E'], 'belts': {'a': (1, 1), 'b': (3, 1), 'c': (0, 1), 'd': (2, 1)}, 'len': 12}, {'rows': ['S.d#cb.', 'a~d..d.', 'b#...d.', '...a...', 'b......', '.#..bab', '#...b#E'], 'belts': {'a': (1, 1), 'b': (2, 1), 'c': (3, 1), 'd': (0, 1)}, 'len': 12}, {'rows': ['S~#.d..', '....~ee', '.#e~...', '#~#.ag.', '....a.c', 'bb##...', '..bf.dE'], 'belts': {'a': (3, -1), 'b': (0, 1), 'c': (0, -1), 'd': (3, 1), 'e': (2, -1), 'f': (1, -1), 'g': (2, 1)}, 'len': 14}, {'rows': ['S#.#...b', 'dfc~f.ag', 'a#~.#b..', '..#.###.', '.ea.dh..', '#ed~....', '.~.#....', '#.bec~.E'], 'belts': {'a': (2, 1), 'b': (3, -1), 'c': (1, 1), 'd': (0, -1), 'e': (1, -1), 'f': (3, 1), 'g': (2, -1), 'h': (0, 1)}, 'len': 16}, {'rows': ['S~.dd...', '.b~.fd#d', '~.a.c#fg', 'd##~~#.f', '#....eb#', '..a.f.e#', '.....b.#', '~#....bE'], 'belts': {'a': (0, -1), 'b': (3, -1), 'c': (1, 1), 'd': (1, -1), 'e': (3, 1), 'f': (0, 1), 'g': (2, -1)}, 'len': 19}]

SOLUTIONS = ['DRRRRRRDDLDR', 'RRSDDDDSSRRD', 'DRRRUSRRDDDSDD', 'SDSSSDDRRRDDRRDR', 'DSRSSRRRURRDDSSDDDR']


def cell(cfg, pos):
    x, y = pos
    rows = cfg["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return "#"


def find(cfg, ch):
    for y, row in enumerate(cfg["rows"]):
        if ch in row:
            return (row.index(ch), y)
    return None


def belt_dir(cfg, ch, k, rule="true"):
    d0, sign = cfg["belts"][ch]
    if rule == "allcw":
        sign = 1
    return DIRS[(d0 + sign * k) % 4]


def advance(cfg, pos, k, letter, rule="true"):
    if letter == "S":
        if rule != "static":
            k = (k + 1) % 4
        return pos, k, False
    d = LETTERS[letter]
    dest = (pos[0] + d[0], pos[1] + d[1])
    if cell(cfg, dest) == "#":
        return pos, k, False
    pos = dest
    ch = cell(cfg, pos)
    if ch in cfg["belts"]:
        bd = belt_dir(cfg, ch, k, rule)
        dest = (pos[0] + bd[0], pos[1] + bd[1])
        if cell(cfg, dest) != "#":
            pos = dest
    return pos, k, cell(cfg, pos) == "~"


def budget_of(cfg):
    return min(BAR_MAX, 2 * cfg["len"] + 8)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng38(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = (0, 0)
        self._k = 0
        self._left = 0
        self._fatal = False
        super().__init__(
            game_id="ng38-3a380001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = find(self._cfg, "S")
        self._k = 0
        self._left = budget_of(self._cfg)
        self._fatal = False
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for y, row in enumerate(cfg["rows"]):
            for x, ch in enumerate(row):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if ch == "#":
                    px[oy:oy + SIDE, ox:ox + SIDE] = WALL
                elif ch == "~":
                    px[oy:oy + SIDE, ox:ox + SIDE] = PIT
                elif ch in cfg["belts"]:
                    px[oy:oy + SIDE, ox:ox + SIDE] = BELT
                    colour = ARROW_CW if cfg["belts"][ch][1] > 0 else ARROW_CCW
                    dx, dy = belt_dir(cfg, ch, self._k)
                    px[oy + 2, ox + 2] = colour
                    px[oy + 2 + dy, ox + 2 + dx] = colour
                    px[oy + 2 + 2 * dy, ox + 2 + 2 * dx] = colour
                    px[oy + 2 + dy + dx, ox + 2 + dx + dy] = colour
                    px[oy + 2 + dy - dx, ox + 2 + dx - dy] = colour
                elif ch == "E":
                    px[oy:oy + SIDE, ox:ox + SIDE] = EXIT
                else:
                    px[oy:oy + SIDE, ox:ox + SIDE] = FLOOR
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._fatal:
            self._pos = find(self._cfg, "S")
            self._fatal = False
            self._draw()
            self.complete_action()
            return
        letter = MOVES.get(self._action.id)
        if letter is None:
            self.complete_action()
            return
        self._pos, self._k, fell = advance(self._cfg, self._pos, self._k, letter)
        self._left = max(0, self._left - 1)
        self._draw()
        if fell:
            if self._left == 0:
                self.lose()
                self.complete_action()
            else:
                self._fatal = True
            return
        if cell(self._cfg, self._pos) == "E":
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
