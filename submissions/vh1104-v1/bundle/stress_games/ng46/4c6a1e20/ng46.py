import random

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BG = 5
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


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


def blank():
    return np.full((64, 64), BG, dtype=np.int8)


def put(px, x, y, w, h, c):
    x0, y0 = max(0, x), max(0, y)
    px[y0:max(y0, min(64, y + h)), x0:max(x0, min(64, x + w))] = c


def glyph(px, x, y, pat, scale, c):
    for gy in range(3):
        for gx in range(3):
            if pat[gy * 3 + gx]:
                put(px, x + gx * scale, y + gy * scale, scale, scale, c)


def make_camera():
    return Camera(width=64, height=64, background=BG, letter_box=BG)

POOL = ["111010010", "101010101", "100100111", "111101111", "111100111", "101111101", "010010010",
        "110010011", "101101010", "111111100", "011010110", "101101111", "010111010", "100010001",
        "110110000", "001011111"]
POOL = [tuple(int(ch) for ch in s) for s in POOL]
COLS = [8, 9, 11, 14, 12, 15]
FRAMES = [10, 7, 12]
CHOICE = [8, 9, 11, 14, 12, 15, 6]
SPECS = [
    dict(tablets=1, rooms=5, doors=3, split=False, seed=3),
    dict(tablets=1, rooms=7, doors=4, split=True, seed=5),
    dict(tablets=2, rooms=7, doors=4, split=True, seed=7),
    dict(tablets=2, rooms=7, doors=5, split=True, seed=11),
    dict(tablets=3, rooms=9, doors=5, split=True, seed=13),
]
OX, OY, COLS_N, ROWS_N = 4, 4, 14, 13


def count(p):
    return sum(p)


def build(spec):
    rng = random.Random(spec["seed"])
    tablets = []
    eligible = [k for k in range(len(POOL)) if sum(1 for q in POOL if count(q) == count(POOL[k])) > 1]
    chosen = rng.sample(eligible, spec["tablets"])
    for j, pi in enumerate(chosen):
        tablets.append((FRAMES[j], POOL[pi], rng.choice(COLS)))
    rooms = []
    for i in range(spec["rooms"]):
        fc, p, c = tablets[i % spec["tablets"]]
        others = [q for q in POOL if q != p]
        same = [q for q in others if count(q) == count(p)]
        diff = [q for q in others if count(q) != count(p)]
        notc = [x for x in COLS if x != c]
        doors = []
        if not spec["split"]:
            doors.append((p, c))
            while len(doors) < spec["doors"]:
                doors.append((rng.choice(diff), rng.choice(notc)))
        else:
            doors.append((rng.choice(same), rng.choice(notc)))
            near = list(p)
            j = rng.randrange(9)
            near[j] ^= 1
            doors.append((tuple(near), rng.choice(notc)))
            doors.append((rng.choice(diff), c))
            while len(doors) < spec["doors"]:
                doors.append((rng.choice(diff), rng.choice(notc)))
        ok = rng.randrange(spec["doors"])
        doors[0], doors[ok] = doors[ok], doors[0]
        rooms.append(dict(wall=fc, doors=doors, ok=ok))
    return tablets, rooms


BUILT = [build(s) for s in SPECS]


def door_x(spec, k):
    n = spec["doors"]
    return round(k * (COLS_N - 2) / (n - 1))


def solution(i):
    spec = SPECS[i]
    tablets, rooms = BUILT[i]
    out = ["U"] * 11
    cx = 7
    for r in rooms:
        tx = door_x(spec, r["ok"])
        d = tx - cx
        out += ["R" if d > 0 else "L"] * abs(d) + ["U"] * 11
        cx = 7
    return out


SOLUTIONS = [solution(i) for i in range(len(SPECS))]
LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng46(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._room = 0
        self._pos = (7, 12)
        self._trail = []
        super().__init__(
            game_id="ng46-4c6a1e20",
            levels=LEVELS,
            camera=make_camera(),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level):
        self._idx = self._current_level_index
        self._spec = SPECS[self._idx]
        self._tablets, self._rooms = BUILT[self._idx]
        self._room = 0
        self._pos = (7, 12)
        self._trail = []
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = blank()
        wall = 3 if self._room == 0 else self._rooms[self._room - 1]["wall"]
        put(px, 2, 2, 60, 1, wall)
        put(px, 2, 57, 60, 1, wall)
        put(px, 2, 2, 1, 56, wall)
        put(px, 61, 2, 1, 56, wall)
        if self._room == 0:
            n = len(self._tablets)
            for j, (fc, p, gc) in enumerate(self._tablets):
                x = 32 - 8 * n + 16 * j
                put(px, x, 22, 14, 14, fc)
                put(px, x + 2, 24, 10, 10, BG)
                glyph(px, x + 3, 25, p, 3, gc)
        else:
            room = self._rooms[self._room - 1]
            for k, (p, c) in enumerate(room["doors"]):
                x = OX + door_x(self._spec, k) * 4
                put(px, x, OY, 8, 8, 3)
                put(px, x + 1, OY + 1, 6, 6, BG)
                glyph(px, x + 1, OY + 1, p, 2, c)
        for i, t in enumerate(self._trail):
            put(px, 4 + i * 5, 59, 3, 3, CHOICE[t])
        cx, cy = self._pos
        put(px, OX + cx * 4, OY + cy * 4, 3, 3, 0)
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self):
        delta = MOVES.get(self._action.id)
        if delta is not None:
            nx = min(COLS_N - 1, max(0, self._pos[0] + delta[0]))
            ny = min(ROWS_N - 1, max(1, self._pos[1] + delta[1]))
            self._pos = (nx, ny)
            if ny == 1:
                self._arrive(nx)
            self._draw()
        self.complete_action()

    def _arrive(self, cx):
        if self._room == 0:
            self._room = 1
            self._pos = (7, 12)
            return
        room = self._rooms[self._room - 1]
        pick = None
        for k in range(self._spec["doors"]):
            x0 = door_x(self._spec, k)
            if x0 <= cx <= x0 + 1:
                pick = k
        if pick is None:
            return
        self._trail.append(pick)
        if self._room == len(self._rooms):
            good = all(self._trail[i] == self._rooms[i]["ok"] for i in range(len(self._rooms)))
            if good:
                self.next_level()
            else:
                self.lose()
            return
        self._room += 1
        self._pos = (7, 12)
