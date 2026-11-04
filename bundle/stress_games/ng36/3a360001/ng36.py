import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 2
PLATE_ON = 11
PIP_ON = 4
PIP_OFF = 3
VOID = 13
PLAYER = 8
EXIT = 14
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

CONFIGS = [{'rows': ['S.aa#aa.', 'ab#a#.##', 'b.baa#ab', '#..a#ba#', '..aaa..b', 'a#aba#b#', 'bab#ba..', '###a.b#E'], 'plat': {'a': (1, 1), 'b': (1, 0)}, 'len': 18}, {'rows': ['S#babb', 'cd.a##', '##cb.e', 'dc#dbd', 'ed#bbc', 'a#eaaE'], 'plat': {'a': (1, 0), 'b': (2, 2), 'c': (1, 1), 'd': (2, 3), 'e': (2, 1)}, 'len': 12}, {'rows': ['Sbc.d.#', '##c#e#f', 'fgd#.h#', '##a#ihh', '#i#h.g#', 'j#g#ebh', '.a#...E'], 'plat': {'a': (2, 3), 'b': (2, 0), 'c': (3, 2), 'd': (1, 1), 'e': (3, 1), 'f': (2, 2), 'g': (1, 0), 'h': (2, 1), 'i': (3, 5), 'j': (3, 3)}, 'len': 19}, {'rows': ['S#b#cdef', 'cg#gg###', '..#fh#ii', 'fjf.bfig', 'i####jij', 'dic#ch#j', '#cc###kb', 'gg##l##E'], 'plat': {'a': (3, 4), 'b': (3, 3), 'c': (2, 2), 'd': (3, 1), 'e': (3, 5), 'f': (3, 0), 'g': (1, 0), 'h': (2, 0), 'i': (1, 1), 'j': (2, 3), 'k': (2, 1), 'l': (3, 2)}, 'len': 24}, {'rows': ['Sb#cdc#a#', '#ebbb##fg', 'fcahfca#h', '#ci.chbd.', '.#.fhghaf', 'j##ddide#', '#ecadabee', '##.##eb#E'], 'plat': {'a': (2, 0), 'b': (2, 2), 'c': (3, 5), 'd': (2, 3), 'e': (3, 2), 'f': (2, 1), 'g': (3, 0), 'h': (3, 3), 'i': (3, 4), 'j': (3, 1)}, 'len': 25}]

SOLUTIONS = ['DDRDDLDDRRURRDRRRD', 'DRRSDRRRDRDD', 'UUUURRRUUURDDDDRDDR', 'UUUUUUUUUDDDRRRRSRRRDDDD', 'UURDSRRUUDDDRLDRSRSDDRRRD']


def cell(cfg, pos):
    x, y = pos
    rows = cfg["rows"]
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return " "


def find(cfg, ch):
    for y, row in enumerate(cfg["rows"]):
        if ch in row:
            return (row.index(ch), y)
    return None


def is_on(cfg, ch, t, rule="true"):
    p, off = cfg["plat"][ch]
    if rule == "allone":
        p = 1
    return ((t + off) // p) % 2 == 0


def deadly(cfg, pos, t, rule="true"):
    ch = cell(cfg, pos)
    if ch == "#":
        return True
    return ch in cfg["plat"] and not is_on(cfg, ch, t, rule)


def advance(cfg, pos, t, letter, rule="true"):
    moved = False
    if letter in LETTERS:
        d = LETTERS[letter]
        dest = (pos[0] + d[0], pos[1] + d[1])
        if cell(cfg, dest) != " ":
            pos = dest
            moved = True
    if rule != "movesonly" or moved:
        t += 1
    return pos, t, deadly(cfg, pos, t, rule)


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


class Ng36(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = (0, 0)
        self._t = 0
        self._left = 0
        self._fatal = None
        super().__init__(
            game_id="ng36-3a360001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = find(self._cfg, "S")
        self._t = 0
        self._left = budget_of(self._cfg)
        self._fatal = None
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        for y, row in enumerate(cfg["rows"]):
            for x, ch in enumerate(row):
                if ch == " ":
                    continue
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if ch == "#":
                    px[oy:oy + SIDE, ox:ox + SIDE] = VOID
                elif ch in cfg["plat"]:
                    p = cfg["plat"][ch][0]
                    on = is_on(cfg, ch, self._t)
                    px[oy:oy + SIDE, ox:ox + SIDE] = PLATE_ON if on else VOID
                    for k in range(p):
                        xs = ox + (2 if p == 1 else (1 + 2 * k if p == 2 else 2 * k))
                        px[oy + 2, xs] = PIP_ON if on else PIP_OFF
                elif ch == "E":
                    px[oy:oy + SIDE, ox:ox + SIDE] = EXIT
                else:
                    px[oy:oy + SIDE, ox:ox + SIDE] = FLOOR
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._fatal is not None:
            self._pos = find(self._cfg, "S")
            self._fatal = None
            self._draw()
            self.complete_action()
            return
        letter = MOVES.get(self._action.id)
        if letter is None:
            self.complete_action()
            return
        self._pos, self._t, fell = advance(self._cfg, self._pos, self._t, letter)
        self._left = max(0, self._left - 1)
        self._draw()
        if fell:
            if self._left == 0:
                self.lose()
                self.complete_action()
            else:
                self._fatal = self._pos
            return
        if cell(self._cfg, self._pos) == "E":
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
