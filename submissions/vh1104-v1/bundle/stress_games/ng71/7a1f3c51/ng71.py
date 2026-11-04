import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
STAND = 11
FALLING = 8
DOWN = 3
POST = 2
BAR = 14
X0 = 2
PITCH = 5
ROW_PITCH = 11
ROW_TOP = 6
POST_H = 2

LEVELS_DEF = [
    dict(rows=[dict(tiles=[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], posts=[])], slack=8),
    dict(rows=[
        dict(tiles=[(0, 1), (1, 1), (2, 1), (4, 1), (5, 1)], posts=[]),
        dict(tiles=[(0, 1), (1, 1), (3, 1), (4, 1), (5, 1), (7, 1)], posts=[]),
    ], slack=4),
    dict(rows=[
        dict(tiles=[(0, 2), (2, 1), (4, 1), (6, 2), (8, 1)], posts=[]),
        dict(tiles=[(1, 1), (2, 1), (3, 2), (6, 1), (7, 1), (9, 2)], posts=[]),
    ], slack=4),
    dict(rows=[
        dict(tiles=[(0, 1), (1, 1), (2, 1), (4, 1), (5, 1), (6, 3), (8, 1), (9, 1)], posts=[3]),
        dict(tiles=[(0, 2), (1, 1), (3, 1), (4, 3), (6, 1), (7, 1)], posts=[2, 5]),
    ], slack=4),
    dict(rows=[
        dict(tiles=[(0, 1), (1, 1), (3, 2), (5, 1), (6, 1), (8, 3)], posts=[2, 7]),
        dict(tiles=[(0, 3), (2, 1), (4, 1), (5, 1), (7, 2)], posts=[1, 6]),
        dict(tiles=[(1, 2), (3, 2), (5, 2), (7, 2)], posts=[]),
    ], slack=1),
]


def reach_targets(row, cell, h, d):
    out = []
    standing = row["state"]
    for step in range(1, h + 1):
        c = cell + d * step
        if c in row["posts"] and h <= POST_H:
            break
        if c in standing and standing[c][0] == "S":
            out.append(c)
    return out


class Model:
    def __init__(self, level):
        self.rows = []
        for r in LEVELS_DEF[level]["rows"]:
            row = dict(posts=list(r["posts"]), h={c: h for c, h in r["tiles"]},
                       state={c: ["S", 0] for c, h in r["tiles"]})
            self.rows.append(row)

    def tick(self):
        for row in self.rows:
            fresh = []
            for c, st in row["state"].items():
                if st[0] == "F":
                    fresh.append((c, st[1]))
            for c, d in fresh:
                row["state"][c] = ["D", 0]
            for c, d in fresh:
                for t in reach_targets(row, c, row["h"][c], d):
                    row["state"][t] = ["F", d]

    def push(self, r, c, d):
        row = self.rows[r]
        if c in row["state"] and row["state"][c][0] == "S":
            row["state"][c] = ["F", d]

    def done(self):
        return all(st[0] == "D" for row in self.rows for st in row["state"].values())


def hpx(h):
    return 2 * h + 1


def base_y(r):
    return ROW_TOP + r * ROW_PITCH + 9


def hit(model, level, x, y):
    for r, row in enumerate(model.rows):
        by = base_y(r)
        for c, h in row["h"].items():
            tx = X0 + c * PITCH
            if tx <= x < tx + 4 and by - hpx(h) < y <= by - 1:
                return r, c, (-1 if x - tx < 2 else 1)
    return None


def target_click(level, r, c, d):
    h = dict(LEVELS_DEF[level]["rows"][r]["tiles"])[c]
    tx = X0 + c * PITCH
    return (tx + (0 if d < 0 else 3), base_y(r) - 1)


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVELS_DEF))]


def build_solutions():
    out = []
    for lv in range(len(LEVELS_DEF)):
        out.append([(63, 63) if mv is None else target_click(lv, *mv) for mv in LEVELS_DEF[lv]["solution"]])
    return out


class Ng71(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._model = Model(0)
        self._left = 0
        self._level = 0
        super().__init__(
            game_id="ng71-7a1f3c51",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._level = self._current_level_index
        self._model = Model(self._level)
        self._left = LEVELS_DEF[self._level]["budget"]
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + min(60, self._left)] = BAR
        for r, row in enumerate(self._model.rows):
            by = base_y(r)
            px[by, 0:64] = 4
            for p in row["posts"]:
                tx = X0 + p * PITCH
                px[by - hpx(POST_H):by, tx:tx + 4] = POST
            for c, h in row["h"].items():
                tx = X0 + c * PITCH
                st, d = row["state"][c]
                if st == "S":
                    px[by - hpx(h):by, tx:tx + 4] = STAND
                elif st == "F":
                    px[by - hpx(h):by, tx:tx + 4] = FALLING
                    ax = tx + (4 if d > 0 else -2)
                    px[by - hpx(h):by - hpx(h) + 2, max(0, ax):max(0, ax) + 2] = FALLING
                else:
                    if d == 0:
                        d = 1
                    px[by - 2:by, tx:tx + 4] = DOWN
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            self._model.tick()
            found = hit(self._model, self._level, int(data.get("x", -1)), int(data.get("y", -1)))
            if found is not None:
                self._model.push(*found)
            self._left = max(0, self._left - 1)
            self._draw()
            if self._model.done():
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()

PLANS = [
    [(0, 0, 1), None, None, None, None, None],
    [(0, 0, 1), (0, 4, 1), (1, 3, 1), (1, 0, 1), (1, 7, -1), None],
    [(0, 0, 1), (1, 1, 1), (1, 9, -1), (0, 6, -1), (0, 8, -1), None],
    [(0, 0, 1), (0, 5, 1), (1, 4, -1), (1, 6, 1), (0, 4, -1), None],
    [(0, 0, 1), (0, 3, -1), (0, 8, -1), (2, 1, 1), (1, 0, 1), (1, 4, 1), (1, 7, -1), None],
]
BUDGETS = [14, 14, 14, 14, 18]
for _i, _d in enumerate(LEVELS_DEF):
    _d["solution"] = PLANS[_i]
    _d["budget"] = BUDGETS[_i]

SOLUTIONS = build_solutions()
