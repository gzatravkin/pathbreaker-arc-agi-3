import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 4
WATER = 9
GEM = 6
PLAYER = 8
BAR = 10
HEIGHT_COLOURS = (13, 12, 11, 14)
TOP = 3
ORIGIN_X = 5
ORIGIN_Y = 6
PITCH = 6
SIDE = 5
BAR_MAX = 60
GAUGE_X = 56

MOVES = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

CONFIGS = [{'rows': ['333233', '3#23#3', '33#322', '32#S2#', '23332#', '332333'], 'gems': [[2, 4], [0, 0], [4, 2]], 'len': 12}, {'rows': ['0#1213', '333302', '233101', '331S32', '233313', '233332'], 'gems': [[5, 1], [0, 3], [2, 5]], 'len': 15}, {'rows': ['3323102', '2230111', '2103131', '2#3S#22', '003##21', '200300#'], 'gems': [[5, 4], [5, 2], [2, 4], [2, 5]], 'len': 16}, {'rows': ['11111#1', '00#2201', '1000211', '122S301', '1211330', '2#21#31', '1101#12'], 'gems': [[2, 3], [3, 0], [4, 4], [0, 6]], 'len': 17}, {'rows': ['01303012', '10133333', '3#0310#2', '2112S331', '33102310', '21110323', '02202300'], 'gems': [[5, 1], [5, 3], [3, 2], [0, 4], [0, 1]], 'len': 19}]

SOLUTIONS = ['URDDLLLUULUU', 'UURRDDDDLLLULUL', 'ULLLDDDRRURRDDLD', 'UUUDRDDLDLLLDDLRD', 'DLLLLUUULUURRLLDDRR']


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


def height(cfg, pos):
    ch = cell(cfg, pos)
    if ch == "#":
        return None
    if ch == "S":
        return TOP
    return int(ch)


def tri(k):
    k %= 2 * TOP
    return k if k <= TOP else 2 * TOP - k


def level_at(m, gems, rule="true"):
    if rule == "permove":
        return tri(m)
    if rule == "permove3":
        return tri(m // 3)
    return tri(len(gems))


def advance(cfg, pos, m, gems, letter, rule="true"):
    start = find(cfg, "S")
    d = LETTERS[letter]
    dest = (pos[0] + d[0], pos[1] + d[1])
    h = height(cfg, dest)
    if h is None or h < level_at(m, gems, rule):
        return pos, m, gems, False
    pos = dest
    m += 1
    if list(pos) in cfg["gems"]:
        gems = gems | {pos}
    if height(cfg, pos) < level_at(m, gems, rule):
        pos = start
    return pos, m, gems, True


def done(cfg, gems):
    return len(gems) == len(cfg["gems"])


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


class Ng40(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = (0, 0)
        self._m = 0
        self._gems = frozenset()
        self._left = 0
        super().__init__(
            game_id="ng40-3a400001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos = find(self._cfg, "S")
        self._m = 0
        self._gems = frozenset()
        self._left = budget_of(self._cfg)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        lvl = level_at(self._m, self._gems)
        for y, row in enumerate(cfg["rows"]):
            for x, ch in enumerate(row):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                h = height(cfg, (x, y))
                if h is None:
                    px[oy:oy + SIDE, ox:ox + SIDE] = WALL
                else:
                    px[oy:oy + SIDE, ox:ox + SIDE] = WATER if h < lvl else HEIGHT_COLOURS[h]
        for gx, gy in cfg["gems"]:
            if (gx, gy) not in self._gems:
                ox, oy = ORIGIN_X + gx * PITCH, ORIGIN_Y + gy * PITCH
                px[oy + 1:oy + 4, ox + 1:ox + 4] = GEM
        for k in range(1, TOP + 1):
            top = 44 - 8 * k
            px[top:top + 6, GAUGE_X:GAUGE_X + 5] = WATER if lvl >= k else HEIGHT_COLOURS[k - 1]
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        letter = MOVES.get(self._action.id)
        if letter is not None:
            self._pos, self._m, self._gems, moved = advance(
                self._cfg, self._pos, self._m, self._gems, letter)
            if moved:
                self._left = max(0, self._left - 1)
            self._draw()
            if done(self._cfg, self._gems):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
