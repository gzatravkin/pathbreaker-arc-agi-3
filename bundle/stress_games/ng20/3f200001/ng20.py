import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
FLOOR = 4
WALL = 3
MARK = 0
BAR = 14
FRAME = 2
PITCH = 4
MINI_X = 2
MINI_Y = 2
MINI_PITCH = 2
DIRS = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
NEIGH = [(0, -1), (1, 0), (0, 1), (-1, 0)]

CONFIGS = [
    {'n': 9, 'walls': [(3, 5), (4, 1), (5, 0), (6, 8), (7, 2), (7, 7)], 'blocks': [(1, 4, 15, ''), (5, 1, 14, ''), (7, 8, 14, ''), (2, 0, 11, ''), (7, 1, 9, ''), (7, 0, 9, '')], 'target': [(0, 0, 11), (0, 1, 14), (0, 2, 14), (5, 1, 9), (7, 8, 9), (8, 0, 15)], 'budget': 22},
    {'n': 9, 'walls': [(2, 5), (3, 0), (3, 5), (5, 7), (6, 2), (6, 6), (6, 7), (8, 2)], 'blocks': [(1, 0, 15, ''), (1, 4, 12, ''), (7, 6, 9, ''), (1, 1, 11, ''), (0, 5, 9, ''), (7, 0, 9, ''), (1, 3, 12, '')], 'target': [(0, 0, 11), (0, 1, 12), (0, 2, 12), (0, 3, 15), (1, 0, 9), (4, 0, 9), (5, 0, 9)], 'budget': 26},
    {'n': 9, 'walls': [(0, 6), (3, 8), (6, 1), (7, 4), (7, 5), (8, 5)], 'blocks': [(2, 4, 12, 's'), (8, 1, 15, 's'), (5, 1, 9, 's'), (4, 7, 14, ''), (2, 2, 15, ''), (4, 2, 11, '')], 'target': [(0, 0, 11), (0, 1, 14), (1, 0, 15), (4, 0, 9), (4, 1, 15), (4, 2, 12)], 'budget': 29},
    {'n': 10, 'walls': [(0, 4), (1, 2), (2, 7), (6, 0), (6, 6), (7, 2), (7, 5), (8, 6)], 'blocks': [(8, 7, 12, 's'), (6, 9, 9, 's'), (3, 4, 11, 's'), (4, 2, 14, ''), (6, 7, 11, ''), (5, 4, 14, ''), (1, 1, 15, '')], 'target': [(0, 0, 11), (0, 1, 14), (0, 3, 15), (1, 0, 12), (1, 1, 9), (2, 1, 11), (3, 1, 14)], 'budget': 33},
    {'n': 9, 'walls': [(1, 0), (3, 4), (3, 7), (8, 3), (8, 4), (8, 5)], 'blocks': [(0, 8, 8, 'o'), (1, 4, 8, 'o'), (2, 3, 11, ''), (7, 1, 15, ''), (7, 3, 12, ''), (1, 8, 14, '')], 'target': [(0, 0, 11), (0, 1, 12), (0, 2, 14), (1, 1, 15), (7, 8, 8), (8, 8, 8)], 'budget': 29},
    {'n': 10, 'walls': [(0, 1), (3, 2), (4, 1), (4, 9), (5, 4), (5, 5), (6, 3), (7, 0)], 'blocks': [(9, 4, 8, 'o'), (5, 9, 8, 'o'), (2, 3, 11, 's'), (6, 7, 12, 's'), (2, 1, 9, ''), (6, 9, 14, ''), (8, 3, 12, '')], 'target': [(0, 0, 11), (0, 2, 12), (0, 3, 12), (0, 4, 14), (1, 0, 9), (8, 9, 8), (9, 9, 8)], 'budget': 29},
]


def tilt(cfg, pos, grp, d):
    n = cfg["n"]
    walls = set(cfg["walls"])
    opp = [("o" in b[3]) for b in cfg["blocks"]]
    sticky = [("s" in b[3]) for b in cfg["blocks"]]
    pos = list(pos)
    frames = []
    groups = {}
    for i, g in enumerate(grp):
        groups.setdefault(g, []).append(i)
    glist = sorted(groups.values(), key=lambda g: g[0])
    dirs = [(-d[0], -d[1]) if opp[g[0]] else d for g in glist]
    while True:
        moved = False
        occupied = {pos[i]: i for i in range(len(pos))}
        order = sorted(range(len(glist)), key=lambda k: -max(pos[i][0] * dirs[k][0] + pos[i][1] * dirs[k][1] for i in glist[k]))
        for k in order:
            g = glist[k]
            e = dirs[k]
            ok = True
            for i in g:
                nx, ny = pos[i][0] + e[0], pos[i][1] + e[1]
                if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in walls:
                    ok = False
                    break
                j = occupied.get((nx, ny))
                if j is not None and j not in g:
                    ok = False
                    break
            if ok:
                for i in g:
                    del occupied[pos[i]]
                for i in g:
                    pos[i] = (pos[i][0] + e[0], pos[i][1] + e[1])
                    occupied[pos[i]] = i
                moved = True
        if not moved:
            break
        frames.append(tuple(pos))
    label = list(grp)
    changed = True
    while changed:
        changed = False
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j or opp[i] or opp[j] or not (sticky[i] or sticky[j]):
                    continue
                if abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1]) == 1 and label[i] != label[j]:
                    lo, hi = min(label[i], label[j]), max(label[i], label[j])
                    label = [lo if v == hi else v for v in label]
                    changed = True
    return tuple(pos), tuple(label), frames


def colour_map(cfg, pos):
    return frozenset((p[0], p[1], b[2]) for p, b in zip(pos, cfg["blocks"]))


def initial(cfg):
    return tuple((b[0], b[1]) for b in cfg["blocks"]), tuple(range(len(cfg["blocks"])))


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(CONFIGS))]


class Ng20(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._pos = ()
        self._grp = ()
        self._left = 0
        self._queue = []
        self._final = None
        super().__init__(
            game_id="ng20-3f200001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._pos, self._grp = initial(self._cfg)
        self._left = self._cfg["budget"]
        self._queue = []
        self._final = None
        self._draw(level, self._pos)

    def _draw(self, level, pos):
        cfg = self._cfg
        n = cfg["n"]
        side = n * PITCH
        ox, oy = 63 - side, 63 - side
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 40:40 + (self._left + 1) // 2] = BAR
        px[oy:oy + side, ox:ox + side] = FLOOR
        walls = set(cfg["walls"])
        for (x, y) in walls:
            px[oy + y * PITCH:oy + y * PITCH + PITCH, ox + x * PITCH:ox + x * PITCH + PITCH] = WALL
        grp = self._grp if len(pos) == len(self._grp) else tuple(range(len(pos)))
        for i, b in enumerate(cfg["blocks"]):
            x, y = pos[i]
            bx, by = ox + x * PITCH, oy + y * PITCH
            px[by:by + 3, bx:bx + 3] = b[2]
            if "s" in b[3]:
                px[by + 1, bx + 1] = MARK
            for j in range(i):
                if grp[i] == grp[j] and abs(pos[j][0] - x) + abs(pos[j][1] - y) == 1:
                    cx, cy = pos[j]
                    x0, x1 = min(x, cx), max(x, cx)
                    y0, y1 = min(y, cy), max(y, cy)
                    if x0 != x1:
                        px[oy + y0 * PITCH:oy + y0 * PITCH + 3, ox + x0 * PITCH + 3] = b[2]
                    else:
                        px[oy + y0 * PITCH + 3, ox + x0 * PITCH:ox + x0 * PITCH + 3] = b[2]
        mside = n * MINI_PITCH
        px[MINI_Y - 1:MINI_Y + mside + 1, MINI_X - 1:MINI_X + mside + 1] = FRAME
        px[MINI_Y:MINI_Y + mside, MINI_X:MINI_X + mside] = FLOOR
        for (x, y) in walls:
            px[MINI_Y + y * MINI_PITCH:MINI_Y + y * MINI_PITCH + MINI_PITCH, MINI_X + x * MINI_PITCH:MINI_X + x * MINI_PITCH + MINI_PITCH] = WALL
        for (x, y, c) in cfg["target"]:
            px[MINI_Y + y * MINI_PITCH:MINI_Y + y * MINI_PITCH + MINI_PITCH, MINI_X + x * MINI_PITCH:MINI_X + x * MINI_PITCH + MINI_PITCH] = c
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        if self._queue:
            frame = self._queue.pop(0)
            self._draw(self.current_level, frame)
            if self._queue:
                return
            self._finish()
            return
        d = DIRS.get(self._action.id)
        if d is None:
            self.complete_action()
            return
        pos, grp, frames = tilt(self._cfg, self._pos, self._grp, d)
        self._left = max(0, self._left - 1)
        self._final = (pos, grp)
        if not frames:
            self._finish()
            return
        self._queue = frames
        frame = self._queue.pop(0)
        self._draw(self.current_level, frame)
        if not self._queue:
            self._finish()

    def _finish(self):
        self._pos, self._grp = self._final
        self._draw(self.current_level, self._pos)
        if colour_map(self._cfg, self._pos) == frozenset(self._cfg["target"]):
            last = self.is_last_level()
            self.next_level()
            if last:
                self.complete_action()
                return
            self.complete_action()
            return
        if self._left == 0:
            self.lose()
        self.complete_action()


SOLUTIONS = ['RULURDLU', 'DLULURDLUL', 'LDRDRULDRULU', 'RDLULDRULURLDL', 'DULUDRULDLUL', 'RDLURULDLULU']
