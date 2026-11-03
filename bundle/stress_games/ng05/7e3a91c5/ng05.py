import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
FLOOR = 1
PAD = 9
ALTAR = 15
ALTAR_GLOW = 7
LENS_RING = 2
LENS_GLOW = 10
PLAYER = 12
DOOR = 13
DOOR_INNER = 4
LIT = 11
GOOD = 14
BAD = 8
BAR = 14
BAR_EMPTY = 4
CELL = 5
BAR_X = 2
BAR_Y = 1
BAR_MAX = 60

CONFIGS = [
    dict(
        grid=(
            "####D###",
            "#......#",
            "#.4..1.#",
            "#..##..#",
            "#.2..3.#",
            "#SA....#",
            "########",
        ),
        info={"A": "O"},
        dur=3,
    ),
    dict(
        grid=(
            "##d####D#d##",
            "#..........#",
            "#.#3#..#x#.#",
            "#..........#",
            "###.##.##.##",
            "#1.......#.#",
            "###.#A#..#4#",
            "#.....#..#.#",
            "#S.2.......#",
            "############",
        ),
        info={"A": "OD"},
        dur=3,
    ),
    dict(
        grid=(
            "##d###D###d#",
            "#..........#",
            "#.x.3.#.1..#",
            "#.....#....#",
            "#.##L##.##.#",
            "#5.........#",
            "#.#.###.#.##",
            "#...2#A.x.4#",
            "#.#..#.....#",
            "#S.........#",
            "############",
        ),
        info={"A": "O", "L": "D"},
        dur=2,
    ),
    dict(
        grid=(
            "#d####D###d#",
            "#..........#",
            "#.4..x...2.#",
            "#..........#",
            "#.##.##.##.#",
            "#.1..M...5.#",
            "#..........#",
            "#.x..3.....#",
            "###.########",
            "#S.L.......#",
            "############",
        ),
        info={"L": "O", "M": "D"},
        dur=2,
    ),
    dict(
        grid=(
            "###d###D##d#",
            "#....o.....#",
            "#.3..#..x..#",
            "#.o..#...1.#",
            "#.#.##o#.#.#",
            "#..L.......#",
            "##.#.#.o##.#",
            "#.5....#2..#",
            "#...#4.#.o.#",
            "#S...A.....#",
            "############",
        ),
        info={"A": "O", "L": "HD"},
        dur=2,
    ),
    dict(
        grid=(
            "#d###d###D#d",
            "#.....o....#",
            "#.x.2.#.6.o#",
            "#.o...#....#",
            "##.#L##o##.#",
            "#4....o...5#",
            "#.#.#.#M#.##",
            "#.1.o.....x#",
            "#...#.3.#..#",
            "#S..A...o..#",
            "############",
        ),
        info={"A": "D", "L": "O", "M": "H"},
        dur=1,
    ),
]

SOLUTIONS = [
    "RSRRRRUUULULLLLDDDRRRRUULLLURRU",
    "URRRRUSULLLLRRDDDUUUUUUDDDDDDRRRRRRRUUDDLLUUULLUUUURU",
    "RRRRRUUSRUULLLUURUURRDRDLDDLLLLDDRLUURUUUDDDRRRRRDDRLUULLLLLLLLUUUURRRRRU",
    "RRUUUURRLLLRRUURRURRRDLLDDDDLLUULUUULLDRRDDRRRRRLLUUUULU",
    "RRRRSLLUURUULRRRRRUURRDDDDLLDDLLUUUULLLUUULRDDDRDDRDULLLUURRRRRRUULUUU",
    "RRRSRUURRUDLLUULUDLDDLRUURUUUDDDRDDDRULUULLLLRRRRDDRRUURRRUULULURU",
]


def parse(cfg):
    grid = cfg["grid"]
    cells = {}
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            cells[(x, y)] = ch
    pads = {}
    doors = {}
    altars = {}
    lenses = {}
    holes = set()
    start = None
    for pos, ch in cells.items():
        if ch.isdigit():
            pads[pos] = int(ch)
        elif ch == "x":
            pads[pos] = 0
        elif ch in "Dd":
            doors[pos] = ch == "D"
        elif ch in "AB":
            altars[pos] = cfg["info"][ch]
        elif ch in "LM":
            lenses[pos] = cfg["info"][ch]
        elif ch == "o":
            holes.add(pos)
        elif ch == "S":
            start = pos
    order = [p for p, k in sorted(pads.items(), key=lambda item: item[1]) if k > 0]
    return dict(cells=cells, pads=pads, doors=doors, altars=altars, lenses=lenses, holes=holes, start=start,
                order=order, w=len(grid[0]), h=len(grid))


def walkable(world, pos):
    ch = world["cells"].get(pos, "#")
    return ch not in "#Dd"


def flash_frames(world, info, dur, glow_pos, glow_colour):
    frames = [{glow_pos: glow_colour}]
    for item in info:
        if item == "O":
            for pos in world["order"]:
                frames += [{glow_pos: glow_colour, pos: LIT}] * dur
                frames.append({glow_pos: glow_colour})
            decoys = [p for p, k in world["pads"].items() if k == 0]
            if decoys:
                warn = {p: BAD for p in decoys}
                warn[glow_pos] = glow_colour
                frames += [warn] * dur
                frames.append({glow_pos: glow_colour})
        elif item == "D":
            lit = {p: (GOOD if real else BAD) for p, real in world["doors"].items()}
            lit[glow_pos] = glow_colour
            frames += [lit] * (dur + 1)
            frames.append({glow_pos: glow_colour})
        elif item == "H":
            lit = {p: BAD for p in world["holes"]}
            lit[glow_pos] = glow_colour
            frames += [lit] * (dur + 1)
            frames.append({glow_pos: glow_colour})
    return frames


def transition(world, cfg, state, action):
    pos, progress = state
    if action == "S":
        if pos in world["altars"]:
            return (pos, progress), "ok", flash_frames(world, world["altars"][pos], cfg["dur"], pos, ALTAR_GLOW)
        return (pos, progress), "ok", []
    dx, dy = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}[action]
    dest = (pos[0] + dx, pos[1] + dy)
    if dest in world["doors"]:
        if not world["doors"][dest]:
            return (dest, progress), "lose", [{dest: BAD}]
        if progress == len(world["order"]):
            return (dest, progress), "win", [{dest: GOOD}]
        return (pos, progress), "ok", [{dest: DOOR_INNER}, {}]
    if not walkable(world, dest):
        return (pos, progress), "ok", []
    if dest in world["holes"]:
        return (dest, progress), "lose", [{dest: BAD}]
    if dest in world["pads"]:
        k = world["pads"][dest]
        if k == progress + 1:
            frames = [{dest: GOOD}, {dest: GOOD}]
            if k == len(world["order"]):
                frames += [{p: GOOD for p in world["order"]}] * 3
            return (dest, k), "ok", frames
        if 0 < k <= progress:
            return (dest, progress), "ok", []
        return (dest, progress), "lose", [{dest: BAD}]
    if dest in world["lenses"]:
        return (dest, progress), "ok", flash_frames(world, world["lenses"][dest], cfg["dur"], dest, LENS_GLOW)
    return (dest, progress), "ok", []


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
LETTER = {GameAction.ACTION1: "U", GameAction.ACTION2: "D", GameAction.ACTION3: "L", GameAction.ACTION4: "R",
          GameAction.ACTION5: "S"}


class Ng05(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._world = parse(self._cfg)
        self._hero = None
        self._left = 0
        self._budget = 0
        self._queue = None
        self._outcome = "ok"
        super().__init__(
            game_id="ng05-7e3a91c5",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._world = parse(self._cfg)
        self._hero = (self._world["start"], 0)
        self._budget = budget_of(self._current_level_index)
        self._left = self._budget
        self._queue = None
        self._outcome = "ok"
        self._draw({}, level)

    def _origin(self):
        w, h = self._world["w"], self._world["h"]
        return (64 - CELL * w) // 2, 4 + (60 - CELL * h) // 2

    def _fill(self, px, pos, colour):
        ox, oy = self._origin()
        x0, y0 = ox + pos[0] * CELL, oy + pos[1] * CELL
        px[y0:y0 + CELL, x0:x0 + CELL] = colour

    def _draw(self, overlay, level=None):
        level = level or self.current_level
        world = self._world
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + BAR_MAX] = BAR_EMPTY
        width = -(-BAR_MAX * self._left // self._budget) if self._budget else 0
        px[BAR_Y:BAR_Y + 2, BAR_X:BAR_X + width] = BAR
        ox, oy = self._origin()
        for pos, ch in world["cells"].items():
            x0, y0 = ox + pos[0] * CELL, oy + pos[1] * CELL
            if ch == "#":
                self._fill(px, pos, WALL)
            elif ch in "Dd":
                self._fill(px, pos, DOOR)
                px[y0 + 1:y0 + 5, x0 + 1:x0 + 4] = DOOR_INNER
            elif pos in world["pads"]:
                self._fill(px, pos, PAD)
            elif pos in world["altars"]:
                self._fill(px, pos, ALTAR)
            else:
                self._fill(px, pos, FLOOR)
                if pos in world["lenses"]:
                    px[y0 + 1:y0 + 4, x0 + 1:x0 + 4] = LENS_RING
                    px[y0 + 2, x0 + 2] = FLOOR
        for pos, colour in overlay.items():
            if pos in world["doors"] and colour == DOOR_INNER:
                x0, y0 = ox + pos[0] * CELL, oy + pos[1] * CELL
                px[y0:y0 + CELL, x0:x0 + CELL] = DOOR_INNER
            else:
                self._fill(px, pos, colour)
        ppos = self._hero[0]
        x0, y0 = ox + ppos[0] * CELL, oy + ppos[1] * CELL
        px[y0 + 1:y0 + 4, x0 + 1:x0 + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._queue is None:
            letter = LETTER.get(self._action.id)
            if letter is None:
                self._draw({})
                self.complete_action()
                return
            self._hero, self._outcome, frames = transition(self._world, self._cfg, self._hero, letter)
            self._left = max(0, self._left - 1)
            self._queue = list(frames) + ([{}] if self._outcome == "ok" else [])
        self._draw(self._queue.pop(0))
        if self._queue:
            return
        self._queue = None
        if self._outcome == "win":
            self.next_level()
        elif self._outcome == "lose" or self._left == 0:
            self.lose()
        self.complete_action()
