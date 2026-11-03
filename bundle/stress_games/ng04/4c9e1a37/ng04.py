import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 1
SEAM = 2
WALL = 3
PLAYER = 9
DOOR = 13
EXIT = 15
PANEL = 4
ARROW = 4
BAR = 0
PIP_OFF = 3
PIP_ON = 0
CELL = 8
VIEW = 7
VX = 4
VY = 8
BAR_W = 56
COLOURS = {"r": 8, "y": 11, "g": 14, "m": 6, "b": 10, "o": 12}
ARROWS = {">": (1, 0), "<": (-1, 0), "^": (0, -1), "v": (0, 1)}

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CHEVRON = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=bool)
CHEVRONS = {">": CHEVRON, "v": np.rot90(CHEVRON, -1), "<": np.rot90(CHEVRON, 2), "^": np.rot90(CHEVRON, 1)}

CONFIGS = [
    dict(radius=2, flash='', codes=['ry'], rows=[
        "#11##############",
        "#P#r..#.........#",
        "#.###.#.#######.#",
        "#...#........y#.#",
        "###.#.#########.#",
        "#.....#...#.....#",
        "#.#.###.#.#.###.#",
        "#.#.....#...#...#",
        "#.###########.#.#",
        "#.............#.#",
        "#.#######.#####.#",
        "#.#g....#.......#",
        "#.#####.#########",
        "#.......#.......#",
        "###.###.###.###.#",
        "#b..........#EA.#",
        "#################",
    ], path="DDRRDDRRUUUULLRRDDRRRRRRRRLLLLLLLLDDLLLLDDDDDDDDRRDDRRRRRRRRUURRRRDDLL"),
    dict(radius=2, flash='1', codes=['gbm'], rows=[
        "#################",
        "#P#b#...#.....AE#",
        "#.#.#.#.#.###.###",
        "#.#...#.#.#m#...#",
        "#.#.#.#.#.#.###.#",
        "#.....#.......#.#",
        "#.###.#########.#",
        "#...#...#.....#.#",
        "###.###.#.###.#.#",
        "#r#...#...#y....#",
        "#.###.#.#.#######",
        "#.#...#.#.......#",
        "#.#.#####.#####.#",
        "#.#.....#....g#.#",
        "#.#.#.#.#######.#",
        "#.....#.........#",
        "#################",
    ], path="DDDDRRRRDDRRDDRRDDDDRRRRLLLLUUUULLUULLUUUULLUUDDRRUURRDDDDRRRRUUDDLLUUUURRRRRR"),
    dict(radius=2, flash='', codes=['oyr'], rows=[
        "###################",
        "#P....#...........#",
        "#####.#.#########.#",
        "#...#.#.#.......#.#",
        "#.#v#.#.#.#####.#.#",
        "#.#.....#...<.#...#",
        "#.11###.#.###.###.#",
        "#.#b....#.#...#...#",
        "#.###.#.###.###.#.#",
        "#.....#.<.........#",
        "#############.###.#",
        "#g....#o........#.#",
        "###.#.###.#.#####.#",
        "#...#...<.#.#...#.#",
        "#.#######.#.#.#.#.#",
        "#.....#y..#...#.#.#",
        "#.###.#####.###.#.#",
        "#.AE#........r#...#",
        "###################",
    ], path="RRRRDDDDRRUUUURRRRRRRRRRDDDDDDDDLLLLDDLLLLLLRRDDDDLLRRUUUURRDDDDDDRRLLLLLLLLUULLLLDDRR"),
    dict(radius=2, flash='1', codes=['rg', 'bym'], rows=[
        "#################",
        "#P#.....#r#m....#",
        "#.#.#.#.#.###.#B#",
        "#.#.#...#.....#E#",
        "#.#.###.###.#.###",
        "#.#.....#...#...#",
        "#.#.#.#.#.#####.#",
        "#.#...#.#.#b....#",
        "#.#.###.#.###.#.#",
        "#.#g#...#.....#y#",
        "#.###.#.###.#.###",
        "#...#.#.A.......#",
        "###.#.#.#.#####.#",
        "#r....#o#.......#",
        "##############22#",
    ], path="DDDDDDDDDDRRDDLLRRRRUUUURRUUUULLDDLLDDUUUURRRRDDDDDDRRRRUURRUULLRRRRDDUUUULLUUUULLRRRRDD"),
    dict(radius=1, flash='1', codes=['mgob'], rows=[
        "#################",
        "#P......#.....#E#",
        "#.#.###.#.###.#A#",
        "#.#.<.#...#g#.#.#",
        "#.###.###.#.#^#.#",
        "#...#.#o....#.#.#",
        "###.#.#####.#.#.#",
        "#y#.#.....#.#.#.#",
        "#.#.#.#.#.#.#.#.#",
        "#...#.....#.#...#",
        "#.###.#.###.###.#",
        "#.#.......#.....#",
        "#.#.#####.#.#####",
        "#.#r#.....#...#b#",
        "#.###.#######.#.#",
        "#.....#m........#",
        "#################",
    ], path="RRRRRRDDRRDDRRDDDDDDDDRRDDLLLLLLRRRRRRUULLUUUUUUUUUUDDLLLLRRRRDDDDDDDDRRDDRRUUDDLLUULLUURRRRUUUUUUUUUU"),
    dict(radius=2, flash='2', codes=['ygr', 'mo'], rows=[
        "#######11########",
        "#P#y....#m#.....#",
        "#.###.#.#.#.#.#.#",
        "#.....#.#.#g#...#",
        "###.###.#.###.#.#",
        "#b#.....#...#.#.#",
        "#.#.#######.#.#.#",
        "#.......A.#...#.#",
        "#.###.###.#.###.#",
        "#...#...#.......#",
        "###.###.#.#.#####",
        "#r#...>.#.....#E#",
        "#.#.#v#.#.###.#B#",
        "#.....#.#...#.#.#",
        "#.#.###.#.#.#.#.#",
        "#g#...<.#o#.....#",
        "#################",
    ], path="DDRRRRUULLRRDDLLDDDDLLDDRRDDDDLLDDUUUUDDRRUUUULLUURRRRRRRRDDRRUUUULLUUUUDDDDRRDDDDDDLLDDDDUURRDDRRRRUUUU"),
]


def parse(cfg):
    rows = cfg["rows"]
    info = dict(cells={}, start=None, w=max(len(r) for r in rows), h=len(rows), panels={})
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "P":
                info["start"] = (c, r)
                ch = "."
            info["cells"][(c, r)] = ch
    for (c, r), ch in info["cells"].items():
        if ch in "12" and info["cells"].get((c - 1, r)) != ch:
            info["panels"][(c, r)] = ch
    return info


def at(info, cell):
    return info["cells"].get(cell, "#")


def blocked(info, cell, doors, delta):
    ch = at(info, cell)
    if ch in "#12":
        return True
    if ch in "AB":
        return ch not in doors
    if ch in ARROWS:
        return ARROWS[ch] != delta
    return False


def press(codes, progress, doors, colour):
    progress = list(progress)
    doors = set(doors)
    for k, code in enumerate(codes):
        if progress[k] == len(code):
            continue
        if code[progress[k]] == colour:
            progress[k] += 1
        elif code[0] == colour:
            progress[k] = 1
        else:
            progress[k] = 0
        if progress[k] == len(code):
            doors.add("AB"[k])
        break
    return tuple(progress), frozenset(doors)


def move(info, codes, state, delta):
    pos, progress, doors = state
    dest = (pos[0] + delta[0], pos[1] + delta[1])
    if blocked(info, dest, doors, delta):
        return state
    ch = at(info, dest)
    if ch in COLOURS:
        progress, doors = press(codes, progress, doors, ch)
    return dest, progress, doors


def budget_of(cfg):
    return 2 * len(cfg["path"]) + 40


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


LEVELS = [canvas_level(i) for i in range(max(1, len(CONFIGS)))]
SOLUTIONS = [list(cfg["path"]) for cfg in CONFIGS]


class Ng04(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._info = parse(self._cfg)
        self._now = None
        self._left = 0
        self._fresh = True
        super().__init__(
            game_id="ng04-4c9e1a37",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._info = parse(self._cfg)
        self._now = (self._info["start"], tuple(0 for _ in self._cfg["codes"]), frozenset())
        self._left = budget_of(self._cfg)
        self._fresh = True
        self._draw(level)

    def _panel_art(self, ch):
        art = np.full((CELL, 2 * CELL), WALL, dtype=np.int8)
        art[1:CELL - 1, 1:2 * CELL - 1] = PANEL
        for i, colour in enumerate(self._cfg["codes"][int(ch) - 1]):
            art[2:CELL - 2, 2 + 3 * i:4 + 3 * i] = COLOURS[colour]
        return art

    def _tile(self, px, ox, oy, cell):
        info = self._info
        ch = at(info, cell)
        if ch == "#":
            px[oy:oy + CELL, ox:ox + CELL] = WALL
            return
        if ch in "12":
            art = self._panel_art(ch)
            px[oy:oy + CELL, ox:ox + CELL] = art[:, :CELL] if cell in info["panels"] else art[:, CELL:]
            return
        px[oy:oy + CELL, ox:ox + CELL] = FLOOR
        px[oy + CELL - 1, ox:ox + CELL] = SEAM
        px[oy:oy + CELL, ox + CELL - 1] = SEAM
        if ch in COLOURS:
            px[oy + 2:oy + 6, ox + 2:ox + 6] = COLOURS[ch]
        elif ch in ARROWS:
            px[oy:oy + CELL, ox:ox + CELL][CHEVRONS[ch]] = ARROW
        elif ch in "AB":
            if ch in self._now[2]:
                for dy in (0, 5):
                    for dx in (0, 5):
                        px[oy + dy:oy + dy + 2, ox + dx:ox + dx + 2] = DOOR
            else:
                px[oy:oy + CELL, ox:ox + CELL] = DOOR
                px[oy + 1:oy + CELL - 1, ox + 2] = FLOOR
                px[oy + 1:oy + CELL - 1, ox + 5] = FLOOR
        elif ch == "E":
            px[oy + 1:oy + 6, ox + 1:ox + 6] = EXIT
            px[oy + 2:oy + 5, ox + 2:ox + 5] = FLOOR

    def _draw(self, level=None):
        level = level or self.current_level
        info = self._info
        cfg = self._cfg
        pos, progress, doors = self._now
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // BAR_W, VX + i % BAR_W] = BAR
        x = VX
        for k, code in enumerate(cfg["codes"]):
            for i, colour in enumerate(code):
                if self._fresh and str(k + 1) in cfg["flash"]:
                    shade = COLOURS[colour]
                else:
                    shade = PIP_ON if i < progress[k] else PIP_OFF
                px[6:8, x:x + 3] = shade
                x += 4
            x += 4
        radius = cfg["radius"]
        half = VIEW // 2
        for j in range(VIEW):
            for i in range(VIEW):
                if max(abs(i - half), abs(j - half)) > radius:
                    continue
                cell = (pos[0] + i - half, pos[1] + j - half)
                self._tile(px, VX + CELL * i, VY + CELL * j, cell)
        ox, oy = VX + CELL * half, VY + CELL * half
        px[oy + 2:oy + 6, ox + 1:ox + 6] = PLAYER
        px[oy + 1, ox + 2:ox + 5] = PLAYER
        px[oy + 6, ox + 2] = PLAYER
        px[oy + 6, ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is None:
            self.complete_action()
            return
        self._fresh = False
        self._now = move(self._info, self._cfg["codes"], self._now, delta)
        self._left = max(0, self._left - 1)
        self._draw()
        if at(self._info, self._now[0]) == "E":
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()
