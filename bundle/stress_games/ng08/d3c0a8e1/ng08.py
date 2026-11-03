import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

VOID = 5
FRAGILE = 3
SOLID = 1
RUBBLE = 4
PLAYER = 9
PLAYER_CORE = 0
DEATH = 8
BAR = 14
CUE_TILE = 3
PITCH = 4
TILE = 3
COLS = 15
ROWS = 14
OX = 2
OY = 7
BAR_X = 1
BAR_ROW = 54
CUE_X = 58
CUE_Y = 0

PLATE_COLOURS = {"a": 6, "b": 10, "c": 7}

GLYPHS = {
    "plus": ("010", "111", "010"),
    "cross": ("101", "010", "101"),
    "ring": ("111", "101", "111"),
    "tee": ("111", "010", "010"),
    "ell": ("100", "100", "111"),
    "hook": ("110", "010", "011"),
    "bar": ("000", "111", "000"),
    "cup": ("101", "101", "111"),
}
PULSE = ((12, 11, False), (11, 0, True), (8, 11, False))

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

RING5 = ((9, 8), (10, 8), (11, 8), (12, 8), (12, 9), (12, 10), (12, 11), (11, 11), (10, 11), (9, 11), (9, 10), (9, 9))
RING6 = ((11, 9), (12, 9), (13, 9), (14, 9), (14, 10), (14, 11), (14, 12), (13, 12), (12, 12), (11, 12), (11, 11), (11, 10))
SWEEP6 = tuple((x, 6) for x in range(3, 12)) + tuple((x, 6) for x in range(10, 3, -1))


def shifted(track, k):
    return track[k:] + track[:k]


LEVEL_DATA = [
    dict(
        rows=(
            "...............",
            "...............",
            "...............",
            ".o#########a...",
            ".o##########...",
            ".o..DDD.DDD....",
            ".o##DDD.DDD....",
            ".o..DDD.DDD....",
            ".o##########...",
            ".o#########b...",
            ".o.............",
            ".o###AB###.....",
            "...............",
            "...............",
        ),
        start=(1, 6),
        cue=("plus", 4),
        goal=dict(glyph="plus", colour=4, track=((9, 11),)),
        fakes=(),
        decoys=(dict(glyph="plus", track=((4, 5),)), dict(glyph="plus", track=((8, 5),))),
        budget=132,
    ),
    dict(
        rows=(
            "...............",
            "...............",
            ".o#######......",
            ".o#######......",
            ".o.....##......",
            ".o##...##......",
            ".o.....a#......",
            ".o...DDD.DDD...",
            ".o###DDD.DDD...",
            ".o...DDD.DDD...",
            ".o#######b.....",
            ".o########.....",
            ".o.............",
            ".o###AB######..",
        ),
        start=(1, 8),
        cue=("plus", 4),
        goal=dict(glyph="plus", colour=4, track=((12, 13),)),
        fakes=(dict(glyph="cross", colour=4, track=((3, 5),)),),
        decoys=(dict(glyph="plus", track=((5, 7),)), dict(glyph="cross", track=((9, 7),))),
        budget=142,
    ),
    dict(
        rows=(
            "...............",
            "...............",
            ".........DDD...",
            "....b#...DDD...",
            "....####.DDD##.",
            "....##.......#.",
            ".a#.##.......C.",
            ".##.##DDDc#..#.",
            ".##.##DDD##..B.",
            ".##.##DDD##.##.",
            ".##.##...##..#.",
            ".##.AA...##..#.",
            ".ooooooooooooo.",
            "...............",
        ),
        start=(7, 12),
        cue=("cross", 4),
        goal=dict(glyph="cross", colour=4, track=((12, 4),)),
        fakes=(dict(glyph="plus", colour=4, track=((7, 4),)), dict(glyph="plus", colour=4, track=((12, 9),))),
        decoys=(dict(glyph="cross", track=((6, 7),)), dict(glyph="plus", track=((9, 2),))),
        budget=150,
    ),
    dict(
        rows=(
            "...............",
            "...............",
            ".######o.......",
            ".######o....DDD",
            ".##....o.###DDD",
            ".##....ooo#cDDD",
            ".#####.o.###...",
            ".####a.o.......",
            ".......o######b",
            "...DDD#o#######",
            "...DDD#o...#...",
            "...DDD.oABC##..",
            "...............",
            "...............",
        ),
        start=(7, 11),
        cue=("tee", 13),
        goal=dict(glyph="tee", colour=13, track=((12, 11),)),
        fakes=(dict(glyph="tee", colour=4, track=((6, 9),)), dict(glyph="tee", colour=15, track=((11, 10),))),
        decoys=(dict(glyph="tee", track=((3, 9),)), dict(glyph="ring", track=((12, 3),))),
        budget=162,
    ),
    dict(
        rows=(
            "...............",
            ".ooooooooooooo.",
            ".##.##.......#.",
            ".##.##DDD.DDDA.",
            ".##.##DDD.DDD#.",
            ".##.##DDD.DDDB.",
            ".##.##.......#.",
            ".a#.##.......#.",
            "....##...ooooo.",
            "....b#...o..o..",
            ".........o..o..",
            ".........oooo..",
            "...............",
            "...............",
        ),
        start=(7, 1),
        cue=("hook", 15),
        goal=dict(glyph="hook", colour=15, track=RING5),
        fakes=(dict(glyph="hook", colour=4, track=shifted(RING5, 4)), dict(glyph="ell", colour=15, track=shifted(RING5, 8))),
        decoys=(dict(glyph="hook", track=((6, 3),)), dict(glyph="plus", track=((10, 3),))),
        budget=140,
    ),
    dict(
        rows=(
            "...............",
            ".ooooooooo.....",
            ".##..AA.#......",
            ".##..##.#......",
            ".a#..#b.B......",
            "........#......",
            "........o......",
            "........o......",
            "........o......",
            "........#..oooo",
            ".c######o..o..o",
            ".#######oCoo..o",
            "...........oooo",
            "...............",
        ),
        start=(4, 1),
        cue=("cup", 13),
        goal=dict(glyph="cup", colour=13, track=RING6),
        fakes=(dict(glyph="cup", colour=4, track=shifted(RING6, 4)), dict(glyph="ell", colour=13, track=shifted(RING6, 8))),
        decoys=(dict(glyph="cup", track=SWEEP6), dict(glyph="ring", track=((10, 0),))),
        budget=150,
    ),
]

SOLUTIONS = [
    "UUURRRRRRRRRRDLLLLLLLLLLDDDDRRRRRRRRRRDLLLLLLLLLLDDRRRRRRRR",
    "UUUUUURRRRRRRDDDDLUUULLLLLLDDDDDDDRRRRRRRRDLLLLLLLLDDRRRRRRRRRRR",
    "LLLLLUUUUUULDDDDDDRRRUUUUUUUUURDDDDDDDDDRRRRUUUUURDDDDDRRRUUUUUUUUL",
    "UUUUUUUUULLLLLLDDDDDRRRRULLLUUURRRRRDDRRURRDDLLULLDDDRRRRRRRDLLLLLLLDDRRRRR",
    "UUUUUUULLDDDDDDDDLUUUUUUUULLDDDDDDLUUUUUURRRRRRRRRRRRDDDDDDDLL",
    "UUUUULLDDDLUUURRRRDDDRUUURRDDDDDDDDDUULLLLLLLDRRRRRRRRRRD",
]


def glyph_cells(name):
    return [(x, y) for y, row in enumerate(GLYPHS[name]) for x, ch in enumerate(row) if ch == "1"]


def track_at(track, t):
    return track[t % len(track)]


def decoy_cells(anchor):
    ax, ay = anchor
    return {(ax + i, ay + j) for i in range(3) for j in range(3)}


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), VOID, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVEL_DATA))]


class Ng08(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = LEVEL_DATA[0]
        self._tiles = []
        self._plates = {}
        self._gates = {}
        self._pos = (0, 0)
        self._t = 0
        self._left = 0
        self._dead = False
        super().__init__(
            game_id="ng08-d3c0a8e1",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=VOID, letter_box=VOID),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = LEVEL_DATA[self._current_level_index]
        self._tiles = []
        self._plates = {}
        self._gates = {}
        for y, row in enumerate(self._cfg["rows"]):
            line = []
            for x, ch in enumerate(row):
                if ch in "#D":
                    line.append(FRAGILE)
                elif ch in PLATE_COLOURS:
                    line.append(FRAGILE)
                elif ch == "o":
                    line.append(SOLID)
                else:
                    line.append(VOID)
                if ch in PLATE_COLOURS:
                    self._plates[(x, y)] = ch
                elif ch.lower() in PLATE_COLOURS and ch.isupper():
                    self._gates[(x, y)] = ch.lower()
            self._tiles.append(line)
        self._pos = tuple(self._cfg["start"])
        self._t = 0
        self._left = self._cfg["budget"]
        self._dead = False
        self._draw(level)

    def _goal_cell(self, t):
        return track_at(self._cfg["goal"]["track"], t)

    def _fake_cells(self, t):
        return {track_at(f["track"], t) for f in self._cfg["fakes"]}

    def _decoy_cells(self, t):
        out = set()
        for d in self._cfg["decoys"]:
            out |= decoy_cells(track_at(d["track"], t))
        return out

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), VOID, dtype=np.int8)
        for y in range(ROWS):
            for x in range(COLS):
                ox, oy = OX + x * PITCH, OY + y * PITCH
                tile = self._tiles[y][x]
                if tile == RUBBLE:
                    px[oy, ox] = RUBBLE
                    px[oy + 2, ox + 2] = RUBBLE
                elif tile != VOID:
                    px[oy:oy + TILE, ox:ox + TILE] = tile
        for (x, y), key in self._plates.items():
            if self._tiles[y][x] == RUBBLE:
                continue
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + TILE, ox:ox + TILE] = PLATE_COLOURS[key]
            px[oy + 1, ox + 1] = FRAGILE
        for (x, y), key in self._gates.items():
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + TILE, ox:ox + TILE] = PLATE_COLOURS[key]
            px[oy:oy + TILE, ox + 1] = VOID
        t = self._t
        things = [(self._cfg["goal"], self._goal_cell(t))]
        things += [(f, track_at(f["track"], t)) for f in self._cfg["fakes"]]
        for obj, (cx, cy) in things:
            ox, oy = OX + cx * PITCH, OY + cy * PITCH
            for gx, gy in glyph_cells(obj["glyph"]):
                px[oy + gy, ox + gx] = obj["colour"]
        phase = t % len(PULSE)
        bg, fg, halo = PULSE[phase]
        span = 3 * PITCH - 1
        for d in self._cfg["decoys"]:
            ax, ay = track_at(d["track"], t)
            ox, oy = OX + ax * PITCH, OY + ay * PITCH
            if halo:
                px[max(0, oy - 1):oy + span + 1, max(0, ox - 1):ox + span + 1] = bg
            px[oy:oy + span, ox:ox + span] = bg
            for gx, gy in glyph_cells(d["glyph"]):
                sy, sx = oy + gy * PITCH, ox + gx * PITCH
                px[sy:sy + TILE, sx:sx + TILE] = fg
        cx, cy = self._pos
        ox, oy = OX + cx * PITCH, OY + cy * PITCH
        if self._dead:
            px[oy:oy + TILE, ox:ox + TILE] = DEATH
            px[oy + 1, ox + 1] = VOID
        else:
            px[oy:oy + TILE, ox:ox + TILE] = PLAYER
            px[oy + 1, ox + 1] = PLAYER_CORE
        first = min(self._left, BAR_ROW)
        second = min(BAR_ROW, max(0, self._left - BAR_ROW))
        third = max(0, self._left - 2 * BAR_ROW)
        px[1, BAR_X:BAR_X + first] = BAR
        px[3, BAR_X:BAR_X + second] = BAR
        px[5, BAR_X:BAR_X + third] = BAR
        px[CUE_Y:CUE_Y + 5, CUE_X:CUE_X + 5] = CUE_TILE
        glyph, colour = self._cfg["cue"]
        for gx, gy in glyph_cells(glyph):
            px[CUE_Y + 1 + gy, CUE_X + 1 + gx] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _contact(self, cells_before, cells_after, old, new):
        return new in cells_after or (new in cells_before and old in cells_after)

    def _enterable(self, cell):
        x, y = cell
        if not (0 <= x < COLS and 0 <= y < ROWS):
            return False
        if cell in self._gates:
            return False
        return self._tiles[y][x] in (FRAGILE, SOLID)

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is None:
            self.complete_action()
            return
        old = self._pos
        new = (old[0] + delta[0], old[1] + delta[1])
        if self._enterable(new):
            if self._tiles[old[1]][old[0]] == FRAGILE:
                self._tiles[old[1]][old[0]] = RUBBLE
            self._pos = new
            key = self._plates.get(new)
            if key is not None:
                for cell in [c for c, k in self._gates.items() if k == key]:
                    del self._gates[cell]
                    self._tiles[cell[1]][cell[0]] = FRAGILE
        new = self._pos
        t0 = self._t
        self._t += 1
        t1 = self._t
        self._left = max(0, self._left - 1)
        lethal_before = self._fake_cells(t0) | self._decoy_cells(t0)
        lethal_after = self._fake_cells(t1) | self._decoy_cells(t1)
        if self._contact(lethal_before, lethal_after, old, new):
            self._dead = True
            self._draw()
            self.lose()
        elif self._contact({self._goal_cell(t0)}, {self._goal_cell(t1)}, old, new):
            self._draw()
            self.next_level()
        else:
            self._draw()
            if self._left == 0:
                self._dead = True
                self._draw()
                self.lose()
        self.complete_action()
