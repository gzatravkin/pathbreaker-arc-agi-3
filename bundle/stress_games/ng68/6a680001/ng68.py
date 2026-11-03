import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
GROUND = 3
WATER = 10
STURDY = 13
FRAGILE = 12
GOAL = 11
PAD = 15
PAD_DONE = 2
HERO = 8
FACE = 0
BAR = 14
ORIGIN_X = 0
ORIGIN_Y = 12
PITCH = 4

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

CONFIGS = [
    dict(
        rows=[
            "...~~...~~~..G",
            "...~~...~~~...",
            "...~~...~~~...",
            "...~~...~~~...",
            "...~~...~~~...",
            "S..~~...~~~...",
        ],
        queue=[("o", 2), ("o", 3)],
    ),
    dict(
        rows=[
            "......~~~...~~~G",
            "......~~~...~~~.",
            "......~~~...~~~.",
            ".~~...~~~...~~~.",
            ".~~...~~~...~~~.",
            "S.....~~~...~~~.",
        ],
        queue=[("o", 2), ("o", 1), ("o", 3)],
    ),
    dict(
        rows=[
            "G...~~....",
            "....~~.K..",
            "S...~~....",
            "....~~....",
            ".~~.~~....",
            "....~~....",
        ],
        queue=[("o", 2), ("b", 2)],
    ),
    dict(
        rows=[
            "K.~~~~...~~~..",
            "~~~~~~...~~~..",
            "~~~~~~...~~~..",
            "S...~~...~~~.G",
            "....~~...~~~..",
            "....~~...~~~..",
        ],
        queue=[("b", 2), ("o", 2), ("o", 3)],
    ),
    dict(
        rows=[
            "K.~~~~...~~~..",
            "~~~~~~...~~~..",
            "~~~~~~...~~~..",
            "S...~~...~~~.G",
            "....~~...~~~..",
            "....~~...~~~..",
            "######~~~#####",
            "######.K.#####",
        ],
        queue=[("b", 2), ("o", 2), ("b", 1), ("o", 3)],
    ),
]


def parse(cfg):
    rows = cfg["rows"]
    h, w = len(rows), len(rows[0])
    water = set()
    rocks = set()
    pads = set()
    start = goal = None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "~":
                water.add((x, y))
            elif ch == "#":
                rocks.add((x, y))
            elif ch == "K":
                pads.add((x, y))
            elif ch == "S":
                start = (x, y)
            elif ch == "G":
                goal = (x, y)
    return w, h, water, pads, start, goal, rocks


def initial(cfg):
    w, h, water, pads, start, goal, rocks = parse(cfg)
    return (start, (1, 0), 0, frozenset(), frozenset(), False)


def transition(cfg, state, letter):
    w, h, water, pads, start, goal, rocks = parse(cfg)
    pos, face, qi, planks, seen, _ = state
    plank_map = dict(planks)
    if letter == "S":
        if qi >= len(cfg["queue"]):
            return state, False
        kind, length = cfg["queue"][qi]
        cells = [(pos[0] + face[0] * i, pos[1] + face[1] * i) for i in range(1, length + 1)]
        for c in cells:
            if not (0 <= c[0] < w and 0 <= c[1] < h) or c not in water or c in plank_map:
                return state, False
        for c in cells:
            plank_map[c] = kind
        return (pos, face, qi + 1, frozenset(plank_map.items()), seen, False), True
    d = LETTERS[letter]
    nxt = (pos[0] + d[0], pos[1] + d[1])
    if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
        return (pos, d, qi, planks, seen, False), True
    if nxt in rocks or (nxt in water and nxt not in plank_map):
        return (pos, d, qi, planks, seen, False), True
    if plank_map.get(pos) == "o":
        del plank_map[pos]
    seen2 = seen | {nxt} if nxt in pads else seen
    won = nxt == goal and seen2 == frozenset(pads)
    return (nxt, d, qi, frozenset(plank_map.items()), seen2, won), True


def bfs(cfg):
    start = initial(cfg)
    seen = {start}
    frontier = [(start, "")]
    while frontier:
        nxt = []
        for state, path in frontier:
            for letter in "UDLRS":
                new, changed = transition(cfg, state, letter)
                if not changed:
                    continue
                if new[5]:
                    return path + letter
                key = new[:5]
                if key not in seen:
                    seen.add(key)
                    nxt.append((new, path + letter))
        frontier = nxt
    return None


SOLUTIONS = [
    list("UUUUURRSRRRRRSRRRRRR"),
    list("UUUUURRRRRSRRSRRRRSRRRR"),
    list("URRRSRRRRULSLLLLLL"),
    list("USUUUDDDRRRSRRRRRSRRRRR"),
    list("USUUUDDDRRRSRRRRDDDSDDUUUURSRRRRR"),
]


def budget_of(index):
    return min(120, 2 * len(SOLUTIONS[index]) + 10)


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


class Ng68(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._st = initial(CONFIGS[0])
        self._left = 0
        self._deny = 0
        super().__init__(
            game_id="ng68-6a680001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._st = initial(self._cfg)
        self._left = budget_of(self._current_level_index)
        self._deny = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        w, h, water, pads, start, goal, rocks = parse(cfg)
        pos, face, qi, planks, seen, _ = self._st
        plank_map = dict(planks)
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // 62, 1 + i % 62] = BAR
        x0 = 2
        for i, (kind, length) in enumerate(cfg["queue"]):
            colour = STURDY if kind == "b" else FRAGILE
            top = 4 if i != qi else 3
            if i < qi:
                continue
            px[top:6, x0:x0 + length * 2] = colour
            x0 += length * 2 + 2
        for y in range(h):
            for x in range(w):
                ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
                if (x, y) in plank_map:
                    colour = STURDY if plank_map[(x, y)] == "b" else FRAGILE
                elif (x, y) in rocks:
                    colour = 4
                elif (x, y) in water:
                    colour = WATER
                elif (x, y) == goal:
                    colour = GOAL
                elif (x, y) in pads:
                    colour = PAD_DONE if (x, y) in seen else PAD
                else:
                    colour = GROUND
                px[oy:oy + PITCH, ox:ox + PITCH] = colour
        ox, oy = ORIGIN_X + pos[0] * PITCH, ORIGIN_Y + pos[1] * PITCH
        px[oy + 1:oy + 3, ox + 1:ox + 3] = HERO
        fx = ox + 1 + face[0] * 2 if face[0] else ox + 1
        fy = oy + 1 + face[1] * 2 if face[1] else oy + 1
        if face[0]:
            px[oy + 1:oy + 3, ox + (3 if face[0] > 0 else 0)] = FACE
        else:
            px[oy + (3 if face[1] > 0 else 0), ox + 1:ox + 3] = FACE
        if self._deny:
            px[4:6, 60:62] = 8
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        letter = {
            GameAction.ACTION1: "U",
            GameAction.ACTION2: "D",
            GameAction.ACTION3: "L",
            GameAction.ACTION4: "R",
            GameAction.ACTION5: "S",
        }.get(self._action.id)
        self._deny = 0
        if letter is not None:
            new, changed = transition(self._cfg, self._st, letter)
            self._deny = 0 if changed else 1
            self._st = new
            self._left = max(0, self._left - 1)
            self._draw()
            if new[5]:
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
