import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
WALL = 3
TILE = 11
PLAYER = 9
BAR = 14
PITCH = 6
OX = 2
OY = 5
BAR_MAX = 60

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
NOTCH = {0: (0, 0), 1: (0, 4), 2: (4, 4), 3: (4, 0)}

ROWS = [
    [
        "0...#...1",
        ".........",
        "...#.#...",
        ".#.....#.",
        "....P....",
        ".#.....#.",
        "...#.#...",
        ".........",
        "3...#...2",
    ],
    [
        "..#.....2",
        ".##.###..",
        "...#.....",
        "1.##.#.#.",
        "..#.P.#.3",
        ".#..#.#..",
        ".#.##.#.#",
        "...#.....",
        "0..#.....",
    ],
    [
        "0.#...1.#",
        "..#.#...#",
        "...3.#...",
        "#..#P..#2",
        "...#.#...",
        "..#..#...",
        ".1.#..0..",
        ".#.#.##.#",
        ".........",
    ],
    [
        "3..#...0.",
        ".#.#.#.#.",
        ".#...#.1.",
        "..##.##..",
        "..2.P...3",
        ".#.##.##.",
        ".#1..0#..",
        ".##.#.#..",
        "2........",
    ],
    [
        "..#.0...3.",
        ".##.##.##.",
        "1.....#...",
        "###.#.#.#.",
        "2...P.#2..",
        ".####.#.##",
        "0...#...1.",
        ".##.###.#.",
        "..3...#...",
    ],
]
SOLUTIONS = ['UUULULLLDRRRRRURRRDDDDDDDDULLLLLDLLL', 'LDLDDDLLUUUUUDDDDRRUURURUURRRUURDDDD', 'DDDRRUUUUUUDDRRDULLDLLULLLUULDDRDDDD', 'RDDLLLUURRRRRRUUUULLDDRDDLLDDDDLLLLLUUUUUUUU', 'LUUUURLDDLLLRRRDDLLLRRRUUUURRRRRLLLLLDDDDLLLDDUURRRRRDDRRRLUUDDLLUULLLLLDDDDRR']


def parse(rows):
    walls, tiles, start = set(), {}, None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                start = (x, y)
            elif ch in "0123":
                tiles[(x, y)] = int(ch)
    return walls, tiles, start, len(rows[0]), len(rows)


def budget_of(index):
    return BUDGETS[index]


BUDGETS = [80, 80, 80, 96, 164]


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


LEVELS = [canvas_level(i) for i in range(len(ROWS))]


class Ng81(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._walls = set()
        self._tiles = {}
        self._gone = set()
        self._pos = (0, 0)
        self._w = 1
        self._h = 1
        self._next = 0
        self._left = 0
        self._total = 1
        super().__init__(
            game_id="ng81-8a810001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._walls, self._tiles, self._pos, self._w, self._h = parse(ROWS[index])
        self._gone = set()
        self._next = 0
        self._total = budget_of(index)
        self._left = self._total
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        n = (self._left * BAR_MAX + self._total - 1) // self._total
        px[1:3, 2:2 + n] = BAR
        for (x, y) in self._walls:
            px[OY + y * PITCH:OY + y * PITCH + PITCH, OX + x * PITCH:OX + x * PITCH + PITCH] = WALL
        for (x, y), kind in self._tiles.items():
            if (x, y) in self._gone:
                continue
            ox, oy = OX + x * PITCH, OY + y * PITCH
            px[oy:oy + 5, ox:ox + 5] = TILE
            ny, nx = NOTCH[kind]
            px[oy + ny, ox + nx] = BACKGROUND
        ox, oy = OX + self._pos[0] * PITCH, OY + self._pos[1] * PITCH
        px[oy + 1:oy + 4, ox + 1:ox + 4] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            dest = (self._pos[0] + delta[0], self._pos[1] + delta[1])
            if 0 <= dest[0] < self._w and 0 <= dest[1] < self._h and dest not in self._walls:
                self._pos = dest
                self._left = max(0, self._left - 1)
                if dest in self._tiles and dest not in self._gone:
                    if self._tiles[dest] == self._next % 4:
                        self._gone.add(dest)
                        self._next += 1
                    else:
                        self._gone = set()
                        self._next = 0
                self._draw()
                if len(self._gone) == len(self._tiles):
                    self.next_level()
                elif self._left == 0:
                    self.lose()
        self.complete_action()
