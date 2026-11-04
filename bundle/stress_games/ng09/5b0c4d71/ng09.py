import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

VOID = 5
WALL = 3
FLOOR = 4
ICE = 10
GLINT = 0
ARROW = 2
RING = 11
BOX = 12
BOX_EDGE = 13
PLAYER = 9
PLAYER_CORE = 0
DEAD = 8
BAR = 14
PITCH = 5
COLS = 12
ROWS = 11
OX = 2
OY = 7
BAR_X = 1
BAR_ROW = 54

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
ONE_WAY = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}
ARROWS = {
    (1, 0): ("..#..", "...#.", "#####", "...#.", "..#.."),
    (-1, 0): ("..#..", ".#...", "#####", ".#...", "..#.."),
    (0, 1): ("..#..", "..#..", "#.#.#", ".###.", "..#.."),
    (0, -1): ("..#..", ".###.", "#.#.#", "..#..", "..#.."),
}

LEVEL_DATA = [
    dict(
        rows=(
            "            ",
            " ########## ",
            " #...ii...# ",
            " #...ii...# ",
            " #...ii...# ",
            " #...ii...# ",
            " #...ii...# ",
            " #...ii...# ",
            " #...ii...# ",
            " ########## ",
            "            ",
        ),
        start=(2, 5),
        boxes=((3, 3), (3, 5), (3, 7)),
        targets=((8, 2), (8, 8), (9, 5)),
        budget=68,
    ),
    dict(
        rows=(
            "            ",
            " ########## ",
            " #...#....# ",
            " #...>iiii# ",
            " #...#....# ",
            " #...#.#..# ",
            " #...<....# ",
            " #...#iii.# ",
            " #...>....# ",
            " ########## ",
            "            ",
        ),
        start=(2, 2),
        boxes=((3, 3), (3, 5), (3, 7)),
        targets=((9, 4), (7, 6), (9, 8)),
        budget=98,
    ),
    dict(
        rows=(
            "            ",
            " ########## ",
            " #........# ",
            " #.iiiiii.# ",
            " #.ii..ii.# ",
            " #.i.##.i.# ",
            " #.ii..ii.# ",
            " #.iiiiii.# ",
            " #........# ",
            " ########## ",
            "            ",
        ),
        start=(9, 4),
        boxes=((7, 2), (9, 6), (4, 5), (5, 4)),
        targets=((4, 8), (7, 2), (9, 3), (4, 2)),
        budget=80,
    ),
    dict(
        rows=(
            "############",
            "#....#.....#",
            "#.ii.>..ii.#",
            "#.ii.#..ii.#",
            "#....#.##..#",
            "#.#.v#.....#",
            "#....#..ii.#",
            "#.ii.<..ii.#",
            "#....#.....#",
            "############",
            "            ",
        ),
        start=(4, 2),
        boxes=((9, 5), (8, 5), (7, 2)),
        targets=((6, 5), (1, 3), (7, 1)),
        budget=100,
    ),
    dict(
        rows=(
            "############",
            "#......#...#",
            "#.iiii.v.#.#",
            "#.#..#.....#",
            "#.#..#.iii.#",
            "#......#...#",
            "###.##>#.#.#",
            "#......#...#",
            "#.iiii.<...#",
            "############",
            "            ",
        ),
        start=(10, 8),
        boxes=((2, 7), (3, 3), (8, 6), (9, 5)),
        targets=((6, 7), (9, 3), (6, 5), (6, 1)),
        budget=114,
    ),
    dict(
        rows=(
            "############",
            "#....#.....#",
            "#....>iiii.#",
            "#.##.#.....#",
            "#....#..#..#",
            "#....<.....#",
            "#.#..#iii..#",
            "#....>.....#",
            "#....#.....#",
            "############",
            "            ",
        ),
        start=(6, 8),
        boxes=((2, 7), (3, 4), (1, 3)),
        targets=((10, 1), (2, 5), (1, 3)),
        budget=126,
    ),
]

SOLUTIONS = [
    "RUULURRRRDDLLDRRRRDLLLDLDRRRR",
    "DRRRLDDDLDLDRRRRRRULLLLLUULURRRRURRDULDDDRDL",
    "UULDRUUUDLDLLUULURRRRDRUDLURDRDLLLL",
    "RRURDURDLULDDDDRDDRRUUUURULLRDDLLDDLLLLDLUUUU",
    "ULLUUURDLULLDRDDDLLUUDDDLLLURRRRLLUUUUULURRRRDRDDURR",
    "UULLUUULLLDDRRDDDDLLURDRUURUUUDLLLUURRRRRDRULDDDLLLDDLLUUU",
]


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


class Ng09(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = LEVEL_DATA[0]
        self._terrain = []
        self._boxes = set()
        self._targets = set()
        self._pos = (0, 0)
        self._left = 0
        self._dir = None
        self._box_slide = None
        self._player_slide = False
        self._dead = False
        super().__init__(
            game_id="ng09-5b0c4d71",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=VOID, letter_box=VOID),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = LEVEL_DATA[self._current_level_index]
        self._terrain = [list(row) for row in self._cfg["rows"]]
        self._boxes = set(self._cfg["boxes"])
        self._targets = set(self._cfg["targets"])
        self._pos = tuple(self._cfg["start"])
        self._left = self._cfg["budget"]
        self._dir = None
        self._box_slide = None
        self._player_slide = False
        self._dead = False
        self._draw(level)

    def _cell(self, c):
        x, y = c
        if 0 <= x < COLS and 0 <= y < ROWS:
            return self._terrain[y][x]
        return " "

    def _can_enter(self, c, d):
        t = self._cell(c)
        if t in "# ":
            return False
        if t in ONE_WAY and ONE_WAY[t] != d:
            return False
        return True

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), VOID, dtype=np.int8)
        for y in range(ROWS):
            for x in range(COLS):
                t = self._terrain[y][x]
                if t == " ":
                    continue
                ox, oy = OX + x * PITCH, OY + y * PITCH
                if t == "#":
                    px[oy:oy + PITCH, ox:ox + PITCH] = WALL
                elif t == "i":
                    px[oy:oy + PITCH, ox:ox + PITCH] = ICE
                    px[oy + 1, ox + 1] = GLINT
                    px[oy + 3, ox + 3] = GLINT
                else:
                    px[oy:oy + PITCH, ox:ox + PITCH] = FLOOR
                    if t in ONE_WAY:
                        pattern = ARROWS[ONE_WAY[t]]
                        for j, row in enumerate(pattern):
                            for i, ch in enumerate(row):
                                if ch == "#":
                                    px[oy + j, ox + i] = ARROW
                if (x, y) in self._targets:
                    px[oy, ox:ox + PITCH] = RING
                    px[oy + PITCH - 1, ox:ox + PITCH] = RING
                    px[oy:oy + PITCH, ox] = RING
                    px[oy:oy + PITCH, ox + PITCH - 1] = RING
        for x, y in self._boxes:
            ox, oy = OX + x * PITCH + 1, OY + y * PITCH + 1
            px[oy:oy + 3, ox:ox + 3] = BOX_EDGE
            px[oy + 1, ox + 1] = BOX
        x, y = self._pos
        ox, oy = OX + x * PITCH + 1, OY + y * PITCH + 1
        px[oy:oy + 3, ox:ox + 3] = DEAD if self._dead else PLAYER
        px[oy + 1, ox + 1] = PLAYER_CORE
        rows = (1, 3, 5)
        left = self._left
        for r in rows:
            n = min(left, BAR_ROW)
            px[r, BAR_X:BAR_X + n] = BAR
            left -= n
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _begin(self, d):
        self._left = max(0, self._left - 1)
        p = self._pos
        c1 = (p[0] + d[0], p[1] + d[1])
        if not self._can_enter(c1, d):
            return
        if c1 in self._boxes:
            c2 = (c1[0] + d[0], c1[1] + d[1])
            if not self._can_enter(c2, d) or c2 in self._boxes:
                return
            self._boxes.discard(c1)
            self._boxes.add(c2)
            if self._cell(c2) == "i":
                self._box_slide = c2
        self._pos = c1
        self._player_slide = self._cell(c1) == "i"
        self._dir = d

    def _slide(self):
        d = self._dir
        if self._box_slide is not None:
            b = self._box_slide
            nb = (b[0] + d[0], b[1] + d[1])
            if self._can_enter(nb, d) and nb not in self._boxes and nb != self._pos:
                self._boxes.discard(b)
                self._boxes.add(nb)
                self._box_slide = nb if self._cell(nb) == "i" else None
            else:
                self._box_slide = None
        if self._player_slide:
            p = self._pos
            np_ = (p[0] + d[0], p[1] + d[1])
            if self._can_enter(np_, d) and np_ not in self._boxes:
                self._pos = np_
                self._player_slide = self._cell(np_) == "i"
            else:
                self._player_slide = False

    def _finish(self):
        self._dir = None
        self._box_slide = None
        self._player_slide = False
        if self._targets <= self._boxes:
            self._draw()
            self.next_level()
        elif self._left == 0:
            self._dead = True
            self._draw()
            self.lose()
        else:
            self._draw()
        self.complete_action()

    def step(self) -> None:
        if self._dir is None:
            d = MOVES.get(self._action.id)
            if d is None:
                self.complete_action()
                return
            self._begin(d)
            if self._dir is None or (self._box_slide is None and not self._player_slide):
                self._finish()
                return
            self._draw()
            return
        self._slide()
        if self._box_slide is None and not self._player_slide:
            self._finish()
            return
        self._draw()
