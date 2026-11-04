import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 3
WALL = 13
PLAYER = 1
BAR = 14
ORIGIN_X = 3
ORIGIN_Y = 10
PITCH = 7
BAR_MAX = 60
SLOTS = {"N": (3, 0), "E": (6, 3), "S": (3, 6), "W": (0, 3)}

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

CONFIGS = [
    {'rooms': [{'c': 6, 'pos': (0, 0), 'walls': {(4, 0), (3, 4), (6, 5), (1, 5), (6, 1), (0, 6), (0, 2), (6, 6), (4, 1), (5, 2)}, 'doors': {'E': 1, 'S': 3}}, {'c': 7, 'pos': (1, 0), 'walls': {(0, 1), (2, 4), (6, 2), (0, 4), (3, 4), (5, 4), (2, 0), (0, 6), (5, 0), (2, 5)}, 'doors': {'E': 2, 'S': 4, 'W': 0}}, {'c': 8, 'pos': (2, 0), 'walls': {(0, 1), (4, 4), (6, 2), (1, 2), (4, 3), (6, 6), (4, 2), (1, 0), (1, 6), (2, 5)}, 'doors': {'S': 5, 'W': 1}}, {'c': 9, 'pos': (0, 1), 'walls': {(4, 4), (2, 4), (4, 0), (0, 0), (1, 1), (2, 0), (0, 6), (2, 6), (6, 6), (5, 2)}, 'doors': {'N': 0, 'E': 4}}, {'c': 10, 'pos': (1, 1), 'walls': {(1, 2), (5, 5), (6, 5), (6, 4), (4, 2), (2, 3), (5, 0), (0, 5), (1, 0), (1, 6)}, 'doors': {'N': 1, 'E': 5, 'W': 3}}, {'c': 12, 'pos': (2, 1), 'walls': {(1, 2), (4, 0), (3, 4), (0, 0), (6, 1), (1, 1), (6, 4), (2, 3), (5, 6), (2, 5)}, 'doors': {'N': 2, 'W': 4}}], 'seq': [3, 4, 5, 2, 1]},
    {'rooms': [{'c': 6, 'pos': (0, 0), 'walls': {(6, 2), (3, 4), (0, 0), (4, 3), (4, 5), (2, 6), (5, 0), (0, 5), (6, 6), (4, 1)}, 'doors': {'N': 4, 'S': 2, 'W': 1}}, {'c': 7, 'pos': (2, 1), 'walls': {(0, 4), (2, 1), (6, 5), (0, 0), (5, 1), (0, 6), (0, 2), (5, 0), (5, 6), (2, 2)}, 'doors': {'E': 0, 'S': 3, 'W': 2}}, {'c': 8, 'pos': (1, 2), 'walls': {(4, 4), (2, 4), (6, 0), (6, 1), (4, 6), (4, 5), (5, 6), (0, 5), (2, 2), (5, 2)}, 'doors': {'N': 5, 'S': 1, 'W': 0}}, {'c': 9, 'pos': (2, 0), 'walls': {(4, 4), (0, 0), (1, 1), (5, 4), (4, 5), (2, 6), (5, 0), (5, 6), (1, 0), (4, 1)}, 'doors': {'E': 5, 'W': 1}}, {'c': 10, 'pos': (0, 1), 'walls': {(1, 2), (2, 1), (6, 1), (1, 1), (2, 3), (1, 0), (1, 6), (3, 2), (2, 5), (4, 1)}, 'doors': {'S': 0, 'W': 5}}, {'c': 12, 'pos': (1, 1), 'walls': {(5, 5), (2, 1), (4, 3), (2, 0), (5, 1), (1, 4), (0, 6), (0, 2), (3, 2), (4, 1)}, 'doors': {'N': 2, 'E': 3, 'S': 4}}], 'seq': [4, 5, 2, 1]},
    {'rooms': [{'c': 6, 'pos': (2, 2), 'walls': {(6, 2), (5, 5), (0, 4), (2, 0), (5, 1), (1, 4), (6, 4), (2, 3), (2, 6), (2, 2)}, 'doors': {'E': 3, 'S': 5, 'W': 2}}, {'c': 7, 'pos': (1, 2), 'walls': {(4, 4), (2, 4), (6, 2), (0, 4), (4, 6), (1, 4), (2, 6), (0, 5), (4, 1), (5, 2)}, 'doors': {'N': 5, 'E': 6, 'S': 2, 'W': 4}}, {'c': 8, 'pos': (0, 1), 'walls': {(6, 5), (6, 1), (2, 0), (4, 2), (1, 4), (4, 5), (5, 0), (5, 6), (1, 0), (4, 1)}, 'doors': {'N': 1, 'E': 0, 'S': 5, 'W': 3}}, {'c': 9, 'pos': (1, 0), 'walls': {(0, 1), (0, 4), (0, 0), (6, 1), (5, 4), (0, 2), (5, 6), (0, 5), (6, 0), (1, 0)}, 'doors': {'N': 0, 'E': 7, 'S': 4, 'W': 2}}, {'c': 10, 'pos': (1, 1), 'walls': {(1, 5), (0, 0), (6, 1), (1, 1), (4, 6), (2, 0), (1, 4), (2, 3), (0, 5), (5, 2)}, 'doors': {'E': 6, 'S': 1, 'W': 3}}, {'c': 12, 'pos': (2, 0), 'walls': {(6, 2), (4, 0), (0, 4), (2, 1), (4, 3), (2, 0), (1, 4), (2, 3), (0, 2), (4, 5)}, 'doors': {'N': 1, 'S': 0, 'W': 2}}, {'c': 13, 'pos': (0, 2), 'walls': {(0, 1), (2, 4), (5, 5), (2, 1), (0, 0), (1, 1), (5, 1), (0, 6), (2, 6), (3, 2)}, 'doors': {'E': 4, 'S': 1}}, {'c': 14, 'pos': (0, 0), 'walls': {(5, 5), (1, 5), (6, 1), (1, 1), (4, 6), (5, 4), (4, 2), (0, 2), (0, 5), (5, 2)}, 'doors': {'W': 3}}], 'seq': [2, 5, 1, 4, 3]},
    {'rooms': [{'c': 6, 'pos': (0, 1), 'walls': {(2, 4), (0, 4), (3, 4), (4, 3), (2, 0), (0, 6), (0, 2), (2, 5), (4, 1), (5, 2)}, 'doors': {'N': 4, 'E': 1, 'S': 8, 'W': 6}}, {'c': 7, 'pos': (1, 0), 'walls': {(4, 4), (6, 2), (6, 0), (0, 4), (1, 1), (5, 4), (0, 6), (2, 3), (2, 2), (4, 1)}, 'doors': {'N': 0, 'E': 4, 'S': 7, 'W': 3}}, {'c': 8, 'pos': (1, 1), 'walls': {(0, 1), (4, 4), (1, 2), (5, 5), (2, 0), (2, 3), (5, 6), (2, 2), (1, 6), (3, 2)}, 'doors': {'N': 4, 'E': 5, 'W': 6}}, {'c': 9, 'pos': (1, 2), 'walls': {(4, 0), (0, 4), (0, 0), (6, 1), (5, 1), (0, 2), (5, 0), (2, 2), (6, 6), (2, 5)}, 'doors': {'E': 5, 'W': 1}}, {'c': 10, 'pos': (2, 0), 'walls': {(2, 4), (6, 2), (4, 3), (6, 4), (2, 3), (5, 0), (5, 6), (6, 6), (2, 5), (5, 2)}, 'doors': {'N': 2, 'E': 6, 'S': 0, 'W': 1}}, {'c': 12, 'pos': (0, 2), 'walls': {(2, 4), (6, 2), (0, 0), (2, 0), (5, 1), (2, 3), (0, 5), (1, 0), (3, 2), (5, 2)}, 'doors': {'E': 8, 'S': 3, 'W': 2}}, {'c': 13, 'pos': (2, 2), 'walls': {(1, 5), (4, 3), (1, 1), (5, 4), (2, 0), (0, 2), (2, 6), (5, 6), (2, 2), (6, 6)}, 'doors': {'N': 2, 'E': 7, 'S': 0, 'W': 4}}, {'c': 14, 'pos': (2, 1), 'walls': {(4, 4), (0, 4), (6, 5), (1, 1), (1, 4), (2, 3), (4, 5), (5, 6), (2, 2), (4, 1)}, 'doors': {'E': 6, 'S': 8, 'W': 1}}, {'c': 15, 'pos': (0, 0), 'walls': {(0, 1), (6, 2), (1, 2), (0, 4), (3, 4), (4, 0), (5, 4), (2, 0), (4, 2), (5, 6)}, 'doors': {'E': 7, 'S': 0, 'W': 5}}], 'seq': [6, 4, 1, 3, 5, 8]},
    {'rooms': [{'c': 6, 'pos': (2, 2), 'walls': {(5, 5), (3, 4), (6, 5), (1, 1), (2, 0), (5, 1), (0, 6), (2, 3), (0, 5), (6, 6)}, 'doors': {'E': 5, 'W': 6}}, {'c': 7, 'pos': (2, 1), 'walls': {(0, 1), (2, 4), (6, 2), (6, 1), (5, 4), (0, 6), (2, 3), (6, 0), (1, 6), (5, 2)}, 'doors': {'N': 7, 'E': 4, 'S': 6, 'W': 2}}, {'c': 8, 'pos': (2, 0), 'walls': {(0, 1), (4, 4), (6, 2), (6, 5), (0, 0), (1, 1), (1, 4), (0, 6), (6, 0), (1, 6)}, 'doors': {'N': 1, 'E': 3}}, {'c': 9, 'pos': (0, 2), 'walls': {(4, 4), (6, 2), (1, 2), (0, 0), (6, 1), (5, 4), (4, 6), (2, 0), (5, 1), (1, 6)}, 'doors': {'N': 7, 'E': 4, 'S': 5, 'W': 2}}, {'c': 10, 'pos': (1, 1), 'walls': {(1, 5), (5, 4), (4, 6), (1, 4), (0, 6), (4, 5), (2, 6), (5, 6), (3, 2), (5, 2)}, 'doors': {'N': 8, 'S': 1, 'W': 3}}, {'c': 12, 'pos': (1, 2), 'walls': {(4, 4), (1, 2), (6, 0), (3, 4), (1, 5), (4, 3), (4, 6), (2, 6), (2, 2), (4, 1)}, 'doors': {'N': 0, 'E': 3, 'S': 7, 'W': 8}}, {'c': 13, 'pos': (0, 1), 'walls': {(4, 4), (2, 4), (5, 5), (3, 4), (6, 1), (1, 1), (5, 4), (4, 2), (1, 4), (6, 6)}, 'doors': {'N': 0, 'E': 8, 'S': 1, 'W': 7}}, {'c': 14, 'pos': (0, 0), 'walls': {(4, 0), (5, 5), (4, 3), (4, 6), (0, 2), (5, 6), (0, 5), (6, 0), (1, 0), (1, 6)}, 'doors': {'N': 1, 'E': 3, 'S': 6, 'W': 5}}, {'c': 15, 'pos': (1, 0), 'walls': {(4, 0), (0, 4), (1, 5), (0, 0), (6, 1), (4, 6), (2, 0), (6, 6), (1, 6), (2, 5)}, 'doors': {'E': 6, 'S': 5, 'W': 4}}], 'seq': [5, 7, 6, 8, 4, 3, 2, 1]},
]

SOLUTIONS = ['LDDDRDDDRRRDRRRURRRDDDRRRURUUUULUUUULLL', 'UUUUULLULUURRUURUULLLDDDDDD', 'UULLDDLDLLLDDRURRUUDDDLLLUUUULLDL', 'LLLUUULLLLDLLUULLDLLLLDLLULRRRRRRUUURRR', 'RRRDDDLDDRDDRDRDRRRRRRRLLLLLLDLDDLLLLLLLLLUUULL']


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 10


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


class Ng27(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._room = 0
        self._pos = (3, 3)
        self._prog = 0
        self._left = 0
        super().__init__(
            game_id="ng27-d27b2e02",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._room = 0
        self._pos = (3, 3)
        self._prog = 0
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        rooms = cfg["rooms"]
        room = rooms[self._room]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        scale = -(-budget_of(self._current_level_index) // BAR_MAX)
        px[1, 2:2 + -(-self._left // scale)] = BAR
        for i, target in enumerate(cfg["seq"]):
            colour = rooms[target]["c"]
            x0 = 2 + 4 * i
            if i < self._prog:
                px[4:7, x0:x0 + 3] = colour
            else:
                px[4:7, x0:x0 + 3] = FLOOR
                px[5, x0 + 1] = colour
        px[ORIGIN_Y - 1:ORIGIN_Y + 7 * PITCH, ORIGIN_X - 1:ORIGIN_X + 7 * PITCH] = room["c"]
        for y in range(7):
            for x in range(7):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                colour = FLOOR
                if (x, y) in room["walls"]:
                    colour = WALL
                for slot, cell in SLOTS.items():
                    if cell == (x, y) and slot in room["doors"]:
                        colour = rooms[room["doors"][slot]]["c"]
                px[oy:oy + 6, ox:ox + 6] = colour
        x, y = self._pos
        ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
        px[oy + 1:oy + 5, ox + 1:ox + 5] = PLAYER
        for i, rm in enumerate(rooms):
            mx, my = 54 + rm["pos"][0] * 3, 10 + rm["pos"][1] * 3
            px[my:my + 2, mx:mx + 2] = rm["c"]
            if i == self._room:
                px[my, mx] = PLAYER
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        delta = MOVES.get(self._action.id)
        if delta is not None:
            rooms = self._cfg["rooms"]
            room = rooms[self._room]
            dest = (self._pos[0] + delta[0], self._pos[1] + delta[1])
            if 0 <= dest[0] < 7 and 0 <= dest[1] < 7 and dest not in room["walls"]:
                self._pos = dest
                for slot, cell in SLOTS.items():
                    if cell == dest and slot in room["doors"]:
                        target = room["doors"][slot]
                        back = [s for s, j in rooms[target]["doors"].items() if j == self._room][0]
                        self._room = target
                        self._pos = SLOTS[back]
                        seq = self._cfg["seq"]
                        if self._prog < len(seq) and target == seq[self._prog]:
                            self._prog += 1
                        else:
                            self._prog = 0
            self._left = max(0, self._left - 1)
            self._draw()
            if self._prog == len(self._cfg["seq"]):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
