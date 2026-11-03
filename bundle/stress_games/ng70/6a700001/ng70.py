import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
SHEET = 3
BAR = 14
UP_COLOUR = 12
DOWN_COLOUR = 8
DOT_COLOURS = [8, 9, 11, 14, 15, 0]
ORIGIN_X = 6
ORIGIN_Y = 12
PITCH = 5

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

CONFIGS = [
    dict(rows=["AABBCC"] * 6, dots=[(1, 0), (3, 3), (5, 5)]),
    dict(rows=["AAABBB", "AAABBB", "AAABBB", "CCC...", "CCC...", "CCC..."], dots=[(0, 2), (4, 0), (2, 4)]),
    dict(rows=["ABBB", "ABBB", "AAAB", "AAAB"], dots=[(0, 0), (1, 0)]),
    dict(rows=["BBBBBA", "BBBBBA", "BBCCAA", "CCCCAA", "CCCCAA", "CCAAAA"], dots=[(0, 1), (0, 3), (4, 3)]),
    dict(rows=["DDDDCCC", "DDDDCCC", "DDB.ACC", "BBB.AAC", "BBBAAAC", "BBBAAAA"], dots=[(0, 1), (2, 4), (5, 1), (5, 5)]),
]


def cells_of(cfg):
    return {(x, y) for y, row in enumerate(cfg["rows"]) for x, ch in enumerate(row) if ch != "."}


def size_of(cfg):
    return len(cfg["rows"][0]), len(cfg["rows"])


def edge(a, b):
    return (a, b) if a <= b else (b, a)


def separated(cells, a, b):
    if a[0] == b[0]:
        x, y = a[0], min(a[1], b[1])
        pair = ((x - 1, y), (x, y))
    else:
        x, y = min(a[0], b[0]), a[1]
        pair = ((x, y - 1), (x, y))
    return pair if pair[0] in cells and pair[1] in cells else None


def pieces(cells, cuts):
    seen = set()
    out = []
    for start in sorted(cells):
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        while stack:
            c = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n not in cells or n in comp:
                    continue
                if d[0]:
                    x = max(c[0], n[0])
                    e = edge((x, c[1]), (x, c[1] + 1))
                else:
                    y = max(c[1], n[1])
                    e = edge((c[0], y), (c[0] + 1, y))
                if e in cuts:
                    continue
                comp.add(n)
                stack.append(n)
        seen |= comp
        out.append(comp)
    return out


def solved(cfg, cuts):
    parts = pieces(cells_of(cfg), cuts)
    if len(parts) < 2:
        return False
    if len({len(p) for p in parts}) != 1:
        return False
    return all(sum(1 for d in cfg["dots"] if d in p) == 1 for p in parts)


def needed_edges(cfg):
    lab = {}
    for y, row in enumerate(cfg["rows"]):
        for x, ch in enumerate(row):
            if ch != ".":
                lab[(x, y)] = ch
    out = set()
    for (x, y), ch in lab.items():
        if (x + 1, y) in lab and lab[(x + 1, y)] != ch:
            out.add(edge((x + 1, y), (x + 1, y + 1)))
        if (x, y + 1) in lab and lab[(x, y + 1)] != ch:
            out.add(edge((x, y + 1), (x + 1, y + 1)))
    return out


def build_solution(cfg):
    w, h = size_of(cfg)
    remaining = set(needed_edges(cfg))
    cur = (0, 0)
    out = []
    while remaining:
        incident = {v for e in remaining for v in e}
        prev = {cur: None}
        frontier = [cur]
        goal = cur if cur in incident else None
        while goal is None:
            nxt = []
            for v in frontier:
                for letter, d in LETTERS.items():
                    n = (v[0] + d[0], v[1] + d[1])
                    if 0 <= n[0] <= w and 0 <= n[1] <= h and n not in prev:
                        prev[n] = (v, letter)
                        nxt.append(n)
                        if n in incident and goal is None:
                            goal = n
            frontier = nxt
        path = []
        v = goal
        while prev[v] is not None:
            path.append(prev[v][1])
            v = prev[v][0]
        out += path[::-1]
        cur = goal
        out.append("S")
        last = None
        while True:
            options = [(letter, (cur[0] + d[0], cur[1] + d[1])) for letter, d in LETTERS.items() if edge(cur, (cur[0] + d[0], cur[1] + d[1])) in remaining]
            if not options:
                break
            options.sort(key=lambda o: o[0] != last)
            letter, n = options[0]
            remaining.discard(edge(cur, n))
            out.append(letter)
            cur = n
            last = letter
        out.append("S")
    return out[:-1]


SOLUTIONS = [build_solution(c) for c in CONFIGS]


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


class Ng70(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._cells = set()
        self._cuts = set()
        self._cursor = (0, 0)
        self._down = False
        self._left = 0
        super().__init__(
            game_id="ng70-6a700001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._cells = cells_of(self._cfg)
        self._cuts = set()
        self._cursor = (0, 0)
        self._down = False
        self._left = budget_of(self._current_level_index)
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for i in range(self._left):
            px[i // 62, 1 + i % 62] = BAR
        cells = self._cells
        cuts = self._cuts
        for x, y in cells:
            ox, oy = ORIGIN_X + x * PITCH, ORIGIN_Y + y * PITCH
            px[oy:oy + 4, ox:ox + 4] = SHEET
            if (x + 1, y) in cells and edge((x + 1, y), (x + 1, y + 1)) not in cuts:
                px[oy:oy + 4, ox + 4] = SHEET
            if (x, y + 1) in cells and edge((x, y + 1), (x + 1, y + 1)) not in cuts:
                px[oy + 4, ox:ox + 4] = SHEET
        w, h = size_of(self._cfg)
        for vx in range(w + 1):
            for vy in range(h + 1):
                around = [(vx - 1, vy - 1), (vx, vy - 1), (vx - 1, vy), (vx, vy)]
                if all(c in cells for c in around):
                    incident = [edge((vx, vy), (vx + dx, vy + dy)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                    if not any(e in cuts for e in incident):
                        px[ORIGIN_Y + vy * PITCH - 1, ORIGIN_X + vx * PITCH - 1] = SHEET
        for i, (x, y) in enumerate(self._cfg["dots"]):
            ox, oy = ORIGIN_X + x * PITCH + 1, ORIGIN_Y + y * PITCH + 1
            px[oy:oy + 2, ox:ox + 2] = DOT_COLOURS[i % len(DOT_COLOURS)]
        cx = ORIGIN_X + self._cursor[0] * PITCH - 1
        cy = ORIGIN_Y + self._cursor[1] * PITCH - 1
        colour = DOWN_COLOUR if self._down else UP_COLOUR
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx or dy) and 0 <= cx + dx < 64 and 0 <= cy + dy < 64:
                    px[cy + dy, cx + dx] = colour
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        acted = False
        delta = MOVES.get(self._action.id)
        if delta is not None:
            w, h = size_of(self._cfg)
            nx, ny = self._cursor[0] + delta[0], self._cursor[1] + delta[1]
            if 0 <= nx <= w and 0 <= ny <= h:
                if self._down and separated(self._cells, self._cursor, (nx, ny)):
                    self._cuts.add(edge(self._cursor, (nx, ny)))
                self._cursor = (nx, ny)
            acted = True
        elif self._action.id == GameAction.ACTION5:
            self._down = not self._down
            acted = True
        if acted:
            self._left = max(0, self._left - 1)
            self._draw()
            if solved(self._cfg, self._cuts):
                self.next_level()
            elif self._left == 0:
                self.lose()
        self.complete_action()
