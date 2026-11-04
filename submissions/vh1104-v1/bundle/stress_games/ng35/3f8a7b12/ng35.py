import copy

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
DICE = {
    0: (),
    1: (4,),
    2: (0, 8),
    3: (0, 4, 8),
    4: (0, 2, 6, 8),
    5: (0, 2, 4, 6, 8),
    6: (0, 2, 3, 5, 6, 8),
    7: (0, 2, 3, 4, 5, 6, 8),
    8: (0, 1, 2, 3, 5, 6, 7, 8),
    9: (0, 1, 2, 3, 4, 5, 6, 7, 8),
}
TOKENS = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}


def rect(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + w), min(64, y + h)
    if x1 > x0 and y1 > y0:
        px[y0:y1, x0:x1] = c


def dice(px, x, y, n, c, gap=2):
    for i in DICE[min(n, 9)]:
        rect(px, x + (i % 3) * gap, y + (i // 3) * gap, 1, 1, c)


def bar(px, left, y=1, c=14):
    rect(px, 2, y, 60, 1, BG)
    rect(px, 2, y, min(60, left), 1, c)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BG, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


class Core(ARCBaseGame):
    STATE = ()

    def __init__(self, gid, count, actions, seed=0):
        self._level = None
        super().__init__(
            game_id=gid,
            levels=[canvas_level(i) for i in range(count)],
            camera=Camera(width=64, height=64, background=BG, letter_box=BG),
            available_actions=actions,
            win_score=count,
            seed=seed,
        )

    def on_set_level(self, level):
        self._level = level
        self.load(self._current_level_index)
        self.paint()

    def paint(self):
        px = np.full((64, 64), BG, dtype=np.int8)
        self.draw(px)
        self._level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        data = self._action.data or {}
        result = self.act(self._action.id.value, data)
        self.paint()
        if result == "win":
            self.next_level()
        elif result == "lose":
            self.lose()
        self.complete_action()

    def snap(self):
        return copy.deepcopy({k: getattr(self, k) for k in self.STATE})

    def restore(self, state):
        for k, v in copy.deepcopy(state).items():
            setattr(self, k, v)

YELLOW = 11
ORANGE = 12
BLUE = 9
RED = 8


def clue_of(code, guess):
    return sum(1 for a, b in zip(code, guess) if a == b)


CONFIGS = [
    dict(w=5, v=5, code=(3, 5, 5, 1, 3), rows=[(3, 2, 5, 1, 3), (3, 5, 5, 1, 1), (3, 3, 5, 3, 3)], tries=3, budget=42),
    dict(w=5, v=5, code=(3, 4, 1, 3, 3), rows=[(3, 4, 4, 3, 1), (3, 4, 1, 3, 3), (5, 4, 4, 1, 5)], tries=3, budget=46),
    dict(w=5, v=6, code=(2, 5, 3, 6, 1), rows=[(4, 5, 3, 6, 1), (2, 5, 3, 4, 1), (6, 1, 2, 6, 4)], tries=3, budget=42),
    dict(w=6, v=6, code=(6, 1, 1, 5, 3, 4), rows=[(6, 5, 1, 5, 3, 4), (6, 1, 1, 6, 3, 4), (1, 1, 1, 1, 6, 2)], tries=3, budget=48),
    dict(w=6, v=6, code=(3, 5, 1, 3, 1, 5), rows=[(3, 5, 1, 3, 6, 5), (6, 5, 1, 3, 1, 2), (6, 6, 6, 5, 6, 4)], tries=4, budget=48),
]
LEVELS = CONFIGS
SOLUTIONS = []


class Ng35(Core):
    STATE = ("wheels", "cursor", "history", "tries", "left")

    def __init__(self, seed=0):
        self.cfg = None
        self.wheels = []
        self.cursor = 0
        self.history = []
        self.tries = 0
        self.left = 0
        super().__init__("ng35-3f8a7b12", max(1, len(CONFIGS)), [1, 2, 3, 4, 5], seed)

    def load(self, index):
        self.cfg = CONFIGS[index]
        self.wheels = [1] * self.cfg["w"]
        self.cursor = 0
        self.history = [(tuple(g), clue_of(self.cfg["code"], g), 0) for g in self.cfg["rows"]]
        self.tries = self.cfg["tries"]
        self.left = self.cfg["budget"]

    def moves(self):
        return ["U", "D", "L", "R", "S"]

    def act(self, aid, data):
        cfg = self.cfg
        if aid not in (1, 2, 3, 4, 5):
            return None
        self.left -= 1
        if aid == 1:
            self.wheels[self.cursor] = self.wheels[self.cursor] % cfg["v"] + 1
        elif aid == 2:
            self.wheels[self.cursor] = (self.wheels[self.cursor] - 2) % cfg["v"] + 1
        elif aid == 3:
            self.cursor = (self.cursor - 1) % cfg["w"]
        elif aid == 4:
            self.cursor = (self.cursor + 1) % cfg["w"]
        else:
            guess = tuple(self.wheels)
            if guess == tuple(cfg["code"]):
                return "win"
            self.history.append((guess, clue_of(cfg["code"], guess), 1))
            self.tries -= 1
            if self.tries <= 0:
                return "lose"
        if self.left <= 0:
            return "lose"
        return "ok"

    def draw(self, px):
        cfg = self.cfg
        w = cfg["w"]
        bar(px, self.left * 60 // max(1, cfg["budget"]), 1, 14)
        for r, (guess, clue, mine) in enumerate(self.history[-9:]):
            y = 4 + r * 5
            for i, v in enumerate(guess):
                dice(px, 2 + i * 5, y, v, 0, 1)
            for i in range(w):
                rect(px, 36 + i * 4, y, 3, 3, YELLOW if i < clue else 3)
        for i in range(self.tries):
            rect(px, 60, 4 + i * 4, 3, 3, ORANGE)
        for i, v in enumerate(self.wheels):
            x = 2 + i * 10
            rect(px, x, 53, 9, 9, ORANGE if i == self.cursor else 3)
            rect(px, x + 1, 54, 7, 7, 4)
            dice(px, x + 2, 55, v, BLUE)

SOLUTIONS = [
    ['U', 'U', 'R', 'D', 'R', 'D', 'R', 'R', 'U', 'U', 'S'],
    ['U', 'U', 'R', 'D', 'D', 'R', 'R', 'U', 'U', 'R', 'U', 'U', 'S'],
    ['U', 'R', 'D', 'D', 'R', 'U', 'U', 'R', 'D', 'R', 'S'],
    ['D', 'R', 'R', 'R', 'D', 'D', 'R', 'U', 'U', 'R', 'U', 'U', 'U', 'S'],
    ['U', 'U', 'R', 'D', 'D', 'R', 'R', 'U', 'U', 'R', 'R', 'D', 'D', 'S'],
]
