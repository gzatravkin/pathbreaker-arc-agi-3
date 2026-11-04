import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
FLOOR = 1
LAVA = 12
LAVA_DOT = 8
PLAYER = 9
GEM = 11
EXIT_SHUT = 15
EXIT_OPEN = 14
EXIT_CORE = 4
LADDER = 13
ICON_BG = 4
ICON_FG = 0
BAR = 14
BAR_EMPTY = 4
DEAD = 8
CELL = 5
BAR_X = 10
BAR_Y = 2
BAR_MAX = 52
ICON_X = 1
ICON_Y = 0
MAP_TOP = 8

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
REMAP = {
    "walk": {"U": "U", "D": "D", "L": "L", "R": "R"},
    "mirror": {"U": "U", "D": "D", "L": "R", "R": "L"},
    "rotate": {"U": "R", "R": "D", "D": "L", "L": "U"},
    "swap": {"U": "U", "D": "D", "L": "L", "R": "R"},
    "slide": {"U": "U", "D": "D", "L": "L", "R": "R"},
    "gravity": {"U": "U", "D": "D", "L": "L", "R": "R"},
}

ICONS = {
    "walk": (
        "...#...",
        "..###..",
        ".#...#.",
        "##.#.##",
        ".#...#.",
        "..###..",
        "...#...",
    ),
    "mirror": (
        "...#...",
        ".#.#.#.",
        "##.#.##",
        "...#...",
        "##.#.##",
        ".#.#.#.",
        "...#...",
    ),
    "rotate": (
        ".####..",
        "#....#.",
        "#.....#",
        "#...###",
        "#....#.",
        ".#.....",
        "..###..",
    ),
    "swap": (
        "aaa.LLL",
        "aaa.LLL",
        "aaa.LLL",
        ".......",
        "LLL.aaa",
        "LLL.aaa",
        "LLL.aaa",
    ),
    "slide": (
        ".......",
        "....#..",
        "#.#.##.",
        ".......",
        "#.#.##.",
        "....#..",
        ".......",
    ),
    "gravity": (
        "...#...",
        "...#...",
        "...#...",
        ".#####.",
        "..###..",
        "...#...",
        "#######",
    ),
}

CONFIGS = [
    dict(
        rule="walk",
        grid=(
            "############",
            "#S..~...*..#",
            "#.#.~.#.##.#",
            "#.#...#..~.#",
            "#*#~~.##.~*#",
            "#...~.*#...#",
            "#~#.....~~.#",
            "#.#.#~#....#",
            "#*..#.#.#~~#",
            "#.~...#..*.E",
            "############",
        ),
    ),
    dict(
        rule="mirror",
        grid=(
            "############",
            "#~~~~~~~~~~#",
            "#S...*~*...#",
            "#~~~.~~~.~~#",
            "#*.......~.#",
            "#~~.~~~~.~*#",
            "#..*.~.....#",
            "#.~~~~.~~~.#",
            "#...*~...*.#",
            "#~~~.~~~~.~#",
            "####E#######",
        ),
    ),
    dict(
        rule="rotate",
        grid=(
            "############",
            "#.~~.......#",
            "#....~**#.~#",
            "#.#...#~...#",
            "#..~.*..~.*#",
            "#..~~#...S.#",
            "#..~..####.#",
            "#....~.~.*.#",
            "##.*#......#",
            "#....~#..~*#",
            "##E#########",
        ),
    ),
    dict(
        rule="swap",
        grid=(
            "############",
            "#.........~#",
            "##.~#.~.#~.E",
            "#..*#......#",
            "#..*.~.~.~*#",
            "#..~~..#..##",
            "#.......#..#",
            "#.~.*~..S..#",
            "#.~*~...~.~#",
            "#....~#*..*#",
            "############",
        ),
    ),
    dict(
        rule="slide",
        grid=(
            "############",
            "#.##..#..*##",
            "#....*.~#..#",
            "###**..**#.#",
            "#....#~....#",
            "#*#....*...#",
            "#.#.#.....##",
            "#......*##~#",
            "#*....#..###",
            "#.~#...S.~##",
            "#######E####",
        ),
    ),
    dict(
        rule="gravity",
        grid=(
            "############",
            "#.....*...S#",
            "####...#HH##",
            "#*.....*HH.#",
            "####H####H.#",
            "#...H.*.*H*#",
            "#####H#H####",
            "#.~*.H.H*.~#",
            "#.###H##H#.#",
            "#~..*H~~H.*E",
            "############",
        ),
    ),
]

SOLUTIONS = [
    "DDDDRRDDDLLRRUURRURLUUUURRRRRDDDDDDLLLDDRRRR",
    "LLLLRDDLLLLUURLDDDDLLUDDDRLUURRUURRRRRRRLLDDRRDDLLLDD",
    "URRDURRLLLLLDLLLDDRDLDDRRURDLDLDDRRURRRURDRR",
    "RDDRLLLUUULLLDLDUULUURULUURRRDDRRRRRDUUR",
    "ULUDURULURDRRDRURULUDRDLD",
    "LLLLLLLLLRRRRRRRRRLLLLLDDDDLRUULLRRRRRDDRRR",
]


def parse(cfg):
    cells = {}
    gems = []
    start = None
    exit_pos = None
    for y, row in enumerate(cfg["grid"]):
        for x, ch in enumerate(row):
            cells[(x, y)] = ch
            if ch == "*":
                gems.append((x, y))
            elif ch == "S":
                start = (x, y)
            elif ch == "E":
                exit_pos = (x, y)
    return dict(cells=cells, gems=tuple(gems), start=start, exit=exit_pos,
                w=len(cfg["grid"][0]), h=len(cfg["grid"]))


def blocked(world, pos, got):
    ch = world["cells"].get(pos, "#")
    if ch == "#":
        return True
    if ch == "E" and len(got) < len(world["gems"]):
        return True
    return False


def enter(world, pos, got):
    ch = world["cells"].get(pos, "#")
    if ch == "~":
        return got, "lose"
    if pos in world["gems"] and pos not in got:
        got = got | {pos}
    if ch == "E":
        return got, "win"
    return got, "ok"


def standing(world, pos):
    below = (pos[0], pos[1] + 1)
    return world["cells"].get(pos) == "H" or world["cells"].get(below, "#") in "#H"


def fall(world, pos, got, trail):
    while not standing(world, pos):
        nxt = (pos[0], pos[1] + 1)
        if blocked(world, nxt, got):
            break
        pos = nxt
        got, outcome = enter(world, pos, got)
        trail.append((pos, got))
        if outcome != "ok":
            return pos, got, outcome
    return pos, got, "ok"


def transition(world, rule, state, letter):
    pos, got = state
    move = REMAP[rule][letter]
    dx, dy = DIRS[move]
    trail = []
    if rule == "slide":
        outcome = "ok"
        while True:
            nxt = (pos[0] + dx, pos[1] + dy)
            if blocked(world, nxt, got):
                break
            pos = nxt
            before = len(got)
            got, outcome = enter(world, pos, got)
            trail.append((pos, got))
            if outcome != "ok" or len(got) > before:
                break
        return (pos, got), outcome, trail
    if rule == "gravity":
        if move == "U" and world["cells"].get(pos) != "H":
            return (pos, got), "ok", trail
        if move == "D" and world["cells"].get((pos[0], pos[1] + 1)) != "H" and world["cells"].get(pos) != "H":
            return (pos, got), "ok", trail
        nxt = (pos[0] + dx, pos[1] + dy)
        if blocked(world, nxt, got):
            return (pos, got), "ok", trail
        pos = nxt
        got, outcome = enter(world, pos, got)
        trail.append((pos, got))
        if outcome != "ok":
            return (pos, got), outcome, trail
        pos, got, outcome = fall(world, pos, got, trail)
        return (pos, got), outcome, trail
    nxt = (pos[0] + dx, pos[1] + dy)
    if blocked(world, nxt, got):
        return (pos, got), "ok", trail
    got, outcome = enter(world, nxt, got)
    trail.append((nxt, got))
    return (nxt, got), outcome, trail


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 8 if index < len(SOLUTIONS) else 80


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
LETTER = {GameAction.ACTION1: "U", GameAction.ACTION2: "D", GameAction.ACTION3: "L", GameAction.ACTION4: "R"}


class Ng07(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._world = parse(self._cfg)
        self._hero = None
        self._left = 0
        self._budget = 0
        self._queue = None
        self._outcome = "ok"
        super().__init__(
            game_id="ng07-5c9e17a4",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._world = parse(self._cfg)
        self._hero = (self._world["start"], frozenset())
        self._budget = budget_of(self._current_level_index)
        self._left = self._budget
        self._queue = None
        self._outcome = "ok"
        self._draw(self._hero, False, level)

    def _origin(self):
        w, h = self._world["w"], self._world["h"]
        return (64 - CELL * w) // 2, MAP_TOP + (56 - CELL * h) // 2

    def _looks(self, ch):
        if self._cfg["rule"] == "swap":
            if ch == "~":
                return "floor"
            if ch in ".S*E":
                return "lava" if ch in ".S*" else "exit"
        if ch == "~":
            return "lava"
        if ch == "#":
            return "wall"
        if ch == "E":
            return "exit"
        if ch == "H":
            return "ladder"
        return "floor"

    def _tile(self, px, x0, y0, look):
        if look == "wall":
            px[y0:y0 + CELL, x0:x0 + CELL] = WALL
        elif look == "lava":
            px[y0:y0 + CELL, x0:x0 + CELL] = LAVA
            px[y0 + 1, x0 + 1] = LAVA_DOT
            px[y0 + 3, x0 + 3] = LAVA_DOT
        elif look == "ladder":
            px[y0:y0 + CELL, x0:x0 + CELL] = FLOOR
            px[y0:y0 + CELL, x0 + 1] = LADDER
            px[y0:y0 + CELL, x0 + 3] = LADDER
            px[y0 + 1, x0 + 1:x0 + 4] = LADDER
            px[y0 + 3, x0 + 1:x0 + 4] = LADDER
        else:
            px[y0:y0 + CELL, x0:x0 + CELL] = FLOOR

    def _icon(self, px):
        rows = ICONS[self._cfg["rule"]]
        px[ICON_Y:ICON_Y + 7, ICON_X:ICON_X + 7] = ICON_BG
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                colour = {"#": ICON_FG, "a": FLOOR, "L": LAVA}.get(ch)
                if colour is not None:
                    px[ICON_Y + y, ICON_X + x] = colour

    def _draw(self, hero, dead, level=None):
        level = level or self.current_level
        world = self._world
        pos, got = hero
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        self._icon(px)
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + BAR_MAX] = BAR_EMPTY
        width = -(-BAR_MAX * self._left // self._budget) if self._budget else 0
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + width] = BAR
        ox, oy = self._origin()
        open_exit = len(got) == len(world["gems"])
        for (x, y), ch in world["cells"].items():
            x0, y0 = ox + x * CELL, oy + y * CELL
            look = self._looks(ch)
            if look == "exit":
                px[y0:y0 + CELL, x0:x0 + CELL] = EXIT_OPEN if open_exit else EXIT_SHUT
                px[y0 + 1:y0 + 4, x0 + 1:x0 + 4] = EXIT_CORE
                px[y0 + 2, x0 + 2] = EXIT_OPEN if open_exit else EXIT_SHUT
                continue
            self._tile(px, x0, y0, look)
            if (x, y) in world["gems"] and (x, y) not in got:
                px[y0 + 1, x0 + 2] = GEM
                px[y0 + 2, x0 + 1:x0 + 4] = GEM
                px[y0 + 3, x0 + 2] = GEM
        x0, y0 = ox + pos[0] * CELL, oy + pos[1] * CELL
        px[y0 + 1:y0 + 4, x0 + 1:x0 + 4] = DEAD if dead else PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._queue is None:
            letter = LETTER.get(self._action.id)
            if letter is None:
                self._draw(self._hero, False)
                self.complete_action()
                return
            self._hero, self._outcome, trail = transition(self._world, self._cfg["rule"], self._hero, letter)
            self._left = max(0, self._left - 1)
            self._queue = trail[:-1] + [self._hero]
        hero = self._queue.pop(0)
        self._draw(hero, not self._queue and self._outcome == "lose")
        if self._queue:
            return
        self._queue = None
        if self._outcome == "win":
            self.next_level()
        elif self._outcome == "lose" or self._left == 0:
            self.lose()
        self.complete_action()
