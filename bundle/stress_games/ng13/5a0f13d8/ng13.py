import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 4
MORTAR = 3
FLOOR = 2
FLOOR_DOT = 1
BAR = 10
SLOT = 3
PAD_OUTER = 15
PAD_INNER = 7
PAD_CORE = 6
EXIT = 14
EXIT_MARK = 11
FRAME = 3
TILE = 10
RADIUS = 2
VIEW_X = 7
VIEW_Y = 12
BAR_X = 2
BAR_MAX = 60
WARP_FRAMES = 7

KEY_COLOURS = {"y": 11, "r": 8, "b": 9, "g": 12}

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

HERO = ("....44....",
        "...4004...",
        "...4004...",
        "..499994..",
        ".49c99c94.",
        ".49999994.",
        "..499994..",
        "..4d44d4..",
        "..4d4.4d4.",
        "...4...4..")

HERO_COLOURS = {"4": 4, "0": 0, "9": 9, "c": 12, "d": 13}

KEY_ART = ("..........",
           "..........",
           ".kkk......",
           "k...k.....",
           "k...kkkkkk",
           "k...k..k.k",
           ".kkk...k.k",
           "..........",
           "..........",
           "..........")

CONFIGS = [
    dict(grid=(
        "####################",
        "####################",
        "##E......##...S##.##",
        "########.##.##.##.##",
        "########.##.##.##.##",
        "##....##.##.##....##",
        "##.##.##.##.#####.##",
        "##.##.##.##.#####.##",
        "##.##....##.......##",
        "##.###########.#####",
        "##.###########.#####",
        "##.##.......##....##",
        "##.#####.##.#####.##",
        "##.#####.##.#####.##",
        "##.......##.......##",
        "#################.##",
        "#################.##",
        "##................##",
        "####################",
        "####################",
    ), budget=236),
    dict(grid=(
        "#######################",
        "#######################",
        "##E...##....##.......##",
        "#####.##.##.########.##",
        "#####.##.##.########.##",
        "##.##................##",
        "##.#################.##",
        "##.#################.##",
        "##.......##.......##.##",
        "##.########.##.##.##.##",
        "##.########.##.##.##.##",
        "##.##.......##...y##.##",
        "##.##.########Y#####.##",
        "##.##.########.#####.##",
        "##.##.......##.......##",
        "##.########.###########",
        "##.########.###########",
        "##..........##.......##",
        "##.###########.##.##.##",
        "##.###########.##.##.##",
        "##................##S##",
        "#######################",
        "#######################",
    ), budget=344),
    dict(grid=(
        "##########################",
        "##########################",
        "##E.........##....##....##",
        "###########.##.##.##.##.##",
        "###########.##.##.##.##.##",
        "##.##....##....##....##.##",
        "##.##.##.##############.##",
        "##.##.##.##############.##",
        "##.##.......##.......##.##",
        "##.##.#####.##.#####.##.##",
        "##.##.#####.##.#####.##.##",
        "##.##....##.......##.##.##",
        "##.#####.########.##.##.##",
        "##.#####.########.##.##.##",
        "##....##.##1##1......##.##",
        "##.#####.##.########.##.##",
        "##.#####.##.########.##.##",
        "##.##....##....##....##.##",
        "##.##.########.########.##",
        "##.##.########.########.##",
        "##......S...##.##.......##",
        "##.##.#####.##.##.########",
        "##.##.#####.##.##.########",
        "##..........##..........##",
        "##########################",
        "##########################",
    ), budget=371),
    dict(grid=(
        "#############################",
        "#############################",
        "##E.........##..........##.##",
        "#####.##.##.#####.##.##.##.##",
        "#####.##.##.#####.##.##.##.##",
        "##....##.##....##.##.##....##",
        "##.##.##.#####.##.##.#####.##",
        "##.##.##.#####.##.##.#####.##",
        "##.......##.R..##y##.##.##.##",
        "###########.########.##.##.##",
        "###########.########.##.##.##",
        "##.......##.##.......##....##",
        "##.#####.##.##Y########.##.##",
        "##.#####.##.##.########.##.##",
        "##....##....##.##....##.##.##",
        "#####.########.##.##.##.##.##",
        "#####.########.##.##.##.##.##",
        "##....##.......##S##.##.##.##",
        "##.#####.##.#####.##.##.##.##",
        "##.#####.##.#####.##.##.##.##",
        "##.##....##.##.......##.##.##",
        "##.########.##.##.##.##.##.##",
        "##.########.##.##.##.##.##.##",
        "##.##.......##....##r##....##",
        "##.##.########.##.########.##",
        "##.##.########.##.########.##",
        "##....##.......##..........##",
        "#############################",
        "#############################",
    ), budget=560),
    dict(grid=(
        "################################",
        "################################",
        "##E##......2##................##",
        "##.##.##.#####.##############.##",
        "##.##.##.#####.##############.##",
        "##.##1##......................##",
        "##.#################.########.##",
        "##.#################.########.##",
        "##....##..........##.......##.##",
        "#####.##.#####.##.##.##.##.##.##",
        "#####.##.#####.##.##.##.##.##.##",
        "##.##....##.......##....##.##.##",
        "##.########.#####.##.##.##.#####",
        "##.########.#####.##.##.##.#####",
        "##.......##.##2##....##.##....##",
        "##.#####.##.##.########.#####.##",
        "##.#####.##.##.########.#####.##",
        "##.##....##....##...S##....##.##",
        "##.##.##.########.########.##.##",
        "##.##.##.########.########.##.##",
        "##.##.##....##3.........##....##",
        "##.##.#####.#####.##.########.##",
        "##.##.#####.#####.##.########.##",
        "##.##.##.......##3##....##....##",
        "##.##.########.########.##.#####",
        "##.##.########.########.##.#####",
        "##.##.......##..........##....##",
        "##.#######################.##.##",
        "##.#######################.##.##",
        "##......1##...................##",
        "################################",
        "################################",
    ), budget=389),
    dict(grid=(
        "######################################",
        "######################################",
        "##E##....##.............##..........##",
        "##.##.##.##.##.##.##.##.#####.#####.##",
        "##.##.##.##.##.##.##.##.#####.#####.##",
        "##.##2##.Y..##....##.##.......##....##",
        "##.#################.#####.#####.#####",
        "##.#################.#####.#####.#####",
        "##....##..........##..........##...1##",
        "#####.##R########.########.##.########",
        "#####.##.########.########.##.########",
        "##y##....##.......##....##.##.......##",
        "##.########.########.##.##.########.##",
        "##.########.########.##.##.########.##",
        "##.......##.##.......##2##.##.......##",
        "##.#####.##.##.##############.##.#####",
        "##.#####.##.##.##############.##.#####",
        "##....##.##....##....##3...##.......##",
        "##.##.###########.##.##.##.########.##",
        "##.##.###########.##.##.##.########.##",
        "##................##...r##..........##",
        "###########.########.##############.##",
        "###########.########.##############.##",
        "##..........##...3##.##....##....##.##",
        "##.###########.#####.##.##.##.##.##.##",
        "##.###########.#####.##.##.##.##.##.##",
        "##.##.......##.......##.##....##....##",
        "##.##.#####.##.########.##.##.########",
        "##.##.#####.##.########.##.##.########",
        "##.......##.##......S##.##..........##",
        "##.#####.##.##.########.#####.##.#####",
        "##.#####.##.##.########.#####.##.#####",
        "##.##....##.##.##....##....##.##....##",
        "##.##.##.##.#####.##.#####.########.##",
        "##.##.##.##.#####.##.#####.########.##",
        "##....##.##1##....##................##",
        "######################################",
        "######################################",
    ), budget=713),
]

SOLUTIONS = [
    "DDDRRRDDDLLLDDDRRRDDDLLLLLLUUULLLDDDLLLLLLUUUUUUUUURRRDDDRRRUUUUUULLLLLL",
    "UUULLLDDDLLLLLLLLLLLLLLLUUURRRRRRRRRUUULLLLLLUUURRRRRRUUURRRDDDRRRLLLDDDRRRRRRUUUUUUUUULLLLLLLLLLLLLLLUUULLL",
    "LLLUUURRRUUUUUULLLUUURRRRRRDDDRRRRRRDDDLLLDDDRRRDDDDDDRRRUUURRRRRRUUUUUUUUUUUUUUUUUULLLDDDLLLUUULLLDDDLLLUUULLLLLLLLL",
    "DDDRRRDDDUUULLLDDDDDDRRRRRRRRRUUUUUUUUUUUUUUUUUUUUULLLUUULLLLLLDDDDDDUUUUUURRRDDDDDDDDDLLLLLLDDDDDDLLLDDDDDDLLLLLLDDDLLLUUUUUUUUURRRUUULLLUUURRRRRRDDDRRRUUUUUURRRUUULLLUUULLLLLLLLL",
    "LLLDDDRRRDDDRRRDDDLLLLLLLLLUUULLLUUULLLUUUUUULLLLLLDDDDDDDDDDDDDDDRRRRRRUUURRRRRRDDDLLLUUUUUURRRUUULLLLLLDDDLLLUUULLLUUUUUU",
    "LLLLLLUUUUUURRRDDDLLLUUULLLDDDLLLLLLLLLLLLUUULLLUUUUUUDDDDDDDDDRRRRRRRRRDDDLLLLLLLLLDDDDDDRRRUUURRRRRRDDDDDDDDDLLLUUURRRUUULLLLLLDDDLLLLLLUUULLLLLLLLLLLLDDDLLLUUULLLDDDUUULLLDDDLLLLLLDDDLLLUUUUUURRRRRRUUULLLLLLLLLDDDLLLUUULLLUUUUUU",
]


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


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


def art_array(rows, colours):
    out = np.full((len(rows), len(rows[0])), -1, dtype=np.int8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in colours:
                out[y, x] = colours[ch]
    return out


def tile_art():
    tiles = {}
    wall = np.full((TILE, TILE), WALL, dtype=np.int8)
    wall[0, :] = MORTAR
    wall[5, :] = MORTAR
    wall[0:5, 0] = MORTAR
    wall[5:10, 5] = MORTAR
    tiles["#"] = wall
    floor = np.full((TILE, TILE), FLOOR, dtype=np.int8)
    tiles["."] = floor
    dotted = floor.copy()
    dotted[4:6, 4:6] = FLOOR_DOT
    tiles[":"] = dotted
    pad = floor.copy()
    pad[1:9, 1:9] = PAD_OUTER
    pad[2:8, 2:8] = FLOOR
    pad[3:7, 3:7] = PAD_INNER
    pad[4:6, 4:6] = PAD_CORE
    tiles["pad"] = pad
    exit_tile = floor.copy()
    exit_tile[1:9, 2:8] = EXIT
    exit_tile[2:9, 3:7] = BACKGROUND
    exit_tile[5, 5] = EXIT_MARK
    tiles["E"] = exit_tile
    for ch, colour in KEY_COLOURS.items():
        key = floor.copy()
        art = art_array(KEY_ART, {"k": colour})
        key[art >= 0] = art[art >= 0]
        tiles[ch] = key
        door = np.full((TILE, TILE), colour, dtype=np.int8)
        door[0, :] = FRAME
        door[9, :] = FRAME
        door[:, 0] = FRAME
        door[:, 9] = FRAME
        door[3:5, 4:6] = WALL
        door[5:7, 4:6] = WALL
        door[6, 3:7] = WALL
        tiles[ch.upper()] = door
    return tiles


TILES = tile_art()
HERO_ART = art_array(HERO, HERO_COLOURS)


class Ng13(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._grid = []
        self._pos = (0, 0)
        self._keys = []
        self._left = 0
        self._pads = {}
        self._warp = None
        super().__init__(
            game_id="ng13-5a0f13d8",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._grid = [list(row) for row in self._cfg["grid"]]
        pads = {}
        for y, row in enumerate(self._grid):
            for x, ch in enumerate(row):
                if ch == "S":
                    self._pos = (x, y)
                    row[x] = "."
                elif ch.isdigit():
                    pads.setdefault(ch, []).append((x, y))
        self._pads = {}
        for cells in pads.values():
            self._pads[cells[0]] = cells[1]
            self._pads[cells[1]] = cells[0]
        self._keys = []
        self._left = self._cfg["budget"]
        self._warp = None
        self._draw(level)

    def _at(self, cell):
        x, y = cell
        if 0 <= y < len(self._grid) and 0 <= x < len(self._grid[y]):
            return self._grid[y][x]
        return "#"

    def step(self) -> None:
        if self._warp is not None:
            self._warp_frame()
            return
        delta = MOVES.get(self._action.id)
        if delta is None:
            self.complete_action()
            return
        self._left = max(0, self._left - 1)
        dest = (self._pos[0] + delta[0], self._pos[1] + delta[1])
        ch = self._at(dest)
        moved = False
        if ch in KEY_COLOURS:
            self._keys.append(ch)
            self._grid[dest[1]][dest[0]] = "."
            moved = True
        elif ch.isupper() and ch.lower() in KEY_COLOURS:
            if ch.lower() in self._keys:
                self._keys.remove(ch.lower())
                self._grid[dest[1]][dest[0]] = "."
                moved = True
        elif ch != "#":
            moved = True
        if moved:
            self._pos = dest
        if moved and ch == "E":
            self._draw()
            self.next_level()
            self.complete_action()
            return
        if moved and dest in self._pads:
            self._warp = dict(frame=0, dest=self._pads[dest])
            self._warp_frame()
            return
        self._draw()
        if self._left == 0:
            self.lose()
        self.complete_action()

    def _warp_frame(self):
        w = self._warp
        f = w["frame"]
        half = WARP_FRAMES
        if f == half:
            self._pos = w["dest"]
        if f < half:
            self._draw(shroud=f + 1)
        else:
            self._draw(shroud=2 * half - f)
        w["frame"] = f + 1
        if w["frame"] > 2 * half:
            self._warp = None
            if self._left == 0:
                self.lose()
            self.complete_action()

    def _tile(self, cell):
        ch = self._at(cell)
        if ch.isdigit():
            return TILES["pad"]
        if ch == "." or ch == "S":
            return TILES[":"] if (cell[0] + cell[1]) % 2 == 0 else TILES["."]
        return TILES.get(ch, TILES["#"])

    def _draw(self, level=None, shroud=0):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1:3, BAR_X:BAR_X + (self._left * BAR_MAX + self._cfg["budget"] - 1) // self._cfg["budget"]] = BAR
        for i, key in enumerate(self._keys):
            ox = BAR_X + i * 8
            px[5:11, ox:ox + 7] = SLOT
            px[6:10, ox + 1:ox + 6] = BACKGROUND
            colour = KEY_COLOURS[key]
            px[7:9, ox + 1] = colour
            px[6, ox + 2] = colour
            px[9, ox + 2] = colour
            px[7:9, ox + 3] = colour
            px[8, ox + 4:ox + 6] = colour
        size = 2 * RADIUS + 1
        for j in range(size):
            for i in range(size):
                cell = (self._pos[0] + i - RADIUS, self._pos[1] + j - RADIUS)
                ox, oy = VIEW_X + i * TILE, VIEW_Y + j * TILE
                px[oy:oy + TILE, ox:ox + TILE] = self._tile(cell)
        hx, hy = VIEW_X + RADIUS * TILE, VIEW_Y + RADIUS * TILE
        region = px[hy:hy + TILE, hx:hx + TILE]
        mask = HERO_ART >= 0
        region[mask] = HERO_ART[mask]
        if shroud > 0:
            span = size * TILE
            for k in range(shroud):
                lo, hi = k * 4, span - k * 4
                if hi <= lo:
                    break
                ring = PAD_OUTER if k % 2 == 0 else PAD_INNER
                px[VIEW_Y + lo:VIEW_Y + hi, VIEW_X + lo:VIEW_X + lo + 4] = ring
                px[VIEW_Y + lo:VIEW_Y + hi, VIEW_X + hi - 4:VIEW_X + hi] = ring
                px[VIEW_Y + lo:VIEW_Y + lo + 4, VIEW_X + lo:VIEW_X + hi] = ring
                px[VIEW_Y + hi - 4:VIEW_Y + hi, VIEW_X + lo:VIEW_X + hi] = ring
        level.get_sprites_by_name("canvas")[0].pixels = px
