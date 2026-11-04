import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
CELL = 4
WIRE = 3
BAR = 14
PITCH = 9
BODY = 8
OY = 7
SHORT = 8
LAMP_ON = 11
PIP = 11

N, E, S, W = 1, 2, 4, 8
VEC = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}
OPP = {N: S, S: N, E: W, W: E}
SHAPES = {"-": E | W, "|": N | S, "L": N | E, "r": E | S, "7": S | W, "J": W | N, ".": 0}
DIRCODE = {"N": N, "E": E, "S": S, "W": W}
CA, CB = 10, 12


def rot(mask):
    return ((mask << 1) | (mask >> 3)) & 15


def rotn(mask, k):
    for _ in range(k % 4):
        mask = rot(mask)
    return mask


LEVELS_CFG = [
    dict(
        n=6, mode="plain",
        grid=["......", "......", "......", ".r---.", ".L--7.", ".---J."],
        sources=[(5, 0, "E", CA)], lamps=[(3, 5, "W", CA)],
        sol=[(5, 1, 1), (5, 3, 1), (5, 4, 3), (4, 4, 2), (4, 3, 1), (4, 1, 1), (3, 1, 3), (3, 2, 1), (3, 3, 1)],
    ),
    dict(
        n=6, mode="plain",
        grid=[".L-...", "....r.", ".7|.||", "...rJ|", ".--J-J", "...-.."],
        sources=[(4, 0, "E", CA)], lamps=[(1, 5, "W", CA)],
        sol=[(4, 1, 1), (4, 2, 1), (4, 3, 1), (3, 3, 3), (3, 4, 2), (2, 4, 1), (1, 4, 3)],
    ),
    dict(
        n=6, mode="plus",
        grid=["......", "......", "..|L..", ".r---.", ".L--7.", ".---J."],
        sources=[(5, 0, "E", CA)], lamps=[(3, 5, "W", CA)],
        sol=[(5, 3, 2), (4, 2, 3), (3, 3, 1), (5, 1, 3), (5, 2, 1), (3, 1, 1), (3, 2, 1)],
    ),
    dict(
        n=6, mode="plus",
        grid=["..L...", ".-7...", ".r|-|.", "..L-7.", "..|7L.", "......"],
        sources=[(1, 0, "E", CA)], lamps=[(4, 5, "W", CA)],
        sol=[(2, 2, 2), (3, 2, 1), (3, 3, 3), (3, 4, 2), (4, 4, 1), (2, 3, 1), (2, 1, 1)],
    ),
    dict(
        n=6, mode="plain",
        grid=["..-...", ".-7.r.", "..L-J.", "..||..", ".----.", "..L.7."],
        sources=[(1, 0, "E", CA), (4, 0, "E", CB)], lamps=[(1, 5, "W", CA), (4, 5, "W", CB)],
        sol=[(1, 1, 1), (1, 2, 1), (2, 2, 1), (2, 3, 1), (2, 4, 3), (1, 4, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1)],
    ),
    dict(
        n=6, mode="turns",
        grid=[".---7.", ".r--J.", ".L---.", ".-L...", "..7.|.", ".r.-.."],
        sources=[(0, 0, "E", CA)], lamps=[(2, 5, "W", CA)],
        sol=[(0, 1, 1), (0, 3, 1), (0, 4, 3), (1, 4, 2), (1, 3, 1), (1, 1, 3), (2, 1, 1), (2, 2, 1), (2, 4, 1)],
        locked=[(3, 2)],
    ),
]


def fixed_cells(cfg):
    out = {}
    for r, c, d, col in cfg["sources"]:
        out[(r, c)] = ("src", DIRCODE[d], col)
    for r, c, d, col in cfg["lamps"]:
        out[(r, c)] = ("lamp", DIRCODE[d], col)
    return out


def build_start(cfg):
    n = cfg["n"]
    fixed = fixed_cells(cfg)
    masks = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if (r, c) in fixed:
                masks[r][c] = fixed[(r, c)][1]
            else:
                masks[r][c] = SHAPES[cfg["grid"][r][c]]
    if cfg["mode"] == "plus":
        for r, c, k in cfg["sol"]:
            for _ in range((4 - k) % 4):
                for tr, tc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                    if 0 <= tr < n and 0 <= tc < n and (tr, tc) not in fixed and masks[tr][tc]:
                        masks[tr][tc] = rot(masks[tr][tc])
    else:
        for r, c, k in cfg["sol"]:
            masks[r][c] = rotn(masks[r][c], 4 - k)
    limits = {}
    if cfg["mode"] == "turns":
        ks = {(r, c): k for r, c, k in cfg["sol"]}
        for r in range(n):
            for c in range(n):
                if (r, c) in fixed or not masks[r][c]:
                    continue
                if (r, c) in cfg.get("locked", []):
                    limits[(r, c)] = 0
                else:
                    limits[(r, c)] = ks.get((r, c), 1)
    return masks, limits


def centre(cfg, r, c):
    ox = (64 - PITCH * cfg["n"] + 1) // 2
    return ox + c * PITCH + 3, OY + r * PITCH + 3


def make_solutions():
    out = []
    for cfg in LEVELS_CFG:
        steps = []
        for r, c, k in cfg["sol"]:
            steps += [centre(cfg, r, c)] * k
        out.append(steps)
    return out


SOLUTIONS = make_solutions()


def flow(cfg, masks):
    n = cfg["n"]
    fixed = fixed_cells(cfg)
    reaches = []
    for r, c, d, col in cfg["sources"]:
        dist = {(r, c): 0}
        frontier = [(r, c)]
        t = 0
        while frontier:
            t += 1
            nxt = []
            for cr, cc in frontier:
                for d in (N, E, S, W):
                    if not masks[cr][cc] & d:
                        continue
                    nr, nc = cr + VEC[d][0], cc + VEC[d][1]
                    if not (0 <= nr < n and 0 <= nc < n) or (nr, nc) in dist:
                        continue
                    if (nr, nc) in fixed and fixed[(nr, nc)][0] == "src":
                        continue
                    if masks[nr][nc] & OPP[d]:
                        dist[(nr, nc)] = t
                        nxt.append((nr, nc))
            frontier = nxt
        reaches.append((col, dist))
    return reaches


def blank_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE, tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [blank_level(i) for i in range(len(LEVELS_CFG))]


class Ng21(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = LEVELS_CFG[0]
        self._masks = []
        self._limits = {}
        self._left = 0
        self._pending = None
        self._outcome = None
        super().__init__(
            game_id="ng21-2a210001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = LEVELS_CFG[self._current_level_index]
        self._masks, self._limits = build_start(self._cfg)
        self._left = min(60, 3 * len(SOLUTIONS[self._current_level_index]) + 10)
        self._pending = None
        self._outcome = None
        self._draw(level, self._lit_static())

    def _lit_static(self):
        lit = {}
        for col, dist in flow(self._cfg, self._masks):
            for cell in dist:
                lit[cell] = col
        return lit

    def _draw(self, level, lit, shorted=()):
        cfg = self._cfg
        n = cfg["n"]
        fixed = fixed_cells(cfg)
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, 2:2 + self._left] = BAR
        ox = (64 - PITCH * n + 1) // 2
        for r in range(n):
            for c in range(n):
                m = self._masks[r][c]
                x0, y0 = ox + c * PITCH, OY + r * PITCH
                if (r, c) in fixed:
                    kind, d, col = fixed[(r, c)]
                    px[y0:y0 + BODY, x0:x0 + BODY] = CELL
                    px[y0 + 1:y0 + 7, x0 + 1:x0 + 7] = col
                    if kind == "lamp":
                        px[y0 + 2:y0 + 6, x0 + 2:x0 + 6] = LAMP_ON if lit.get((r, c)) == col else BACKGROUND
                    self._stub(px, x0, y0, d, col if kind == "src" else lit.get((r, c), WIRE))
                    continue
                if not m:
                    continue
                px[y0:y0 + BODY, x0:x0 + BODY] = CELL
                colour = SHORT if (r, c) in shorted else lit.get((r, c), WIRE)
                px[y0 + 3:y0 + 5, x0 + 3:x0 + 5] = colour
                for d in (N, E, S, W):
                    if m & d:
                        self._stub(px, x0, y0, d, colour)
                if cfg["mode"] == "turns":
                    for i in range(self._limits.get((r, c), 0)):
                        px[y0 + i, x0] = PIP
        level.get_sprites_by_name("canvas")[0].pixels = px

    @staticmethod
    def _stub(px, x0, y0, d, colour):
        if d == N:
            px[y0:y0 + 4, x0 + 3:x0 + 5] = colour
        elif d == S:
            px[y0 + 4:y0 + 8, x0 + 3:x0 + 5] = colour
        elif d == W:
            px[y0 + 3:y0 + 5, x0:x0 + 4] = colour
        else:
            px[y0 + 3:y0 + 5, x0 + 4:x0 + 8] = colour

    def _cell_at(self, x, y):
        cfg = self._cfg
        ox = (64 - PITCH * cfg["n"] + 1) // 2
        c, rx = divmod(x - ox, PITCH)
        r, ry = divmod(y - OY, PITCH)
        if rx >= BODY or ry >= BODY or not (0 <= r < cfg["n"] and 0 <= c < cfg["n"]):
            return None
        if (r, c) in fixed_cells(cfg) or not self._masks[r][c]:
            return None
        return r, c

    def _static_frames(self):
        return [(self._lit_static(), ())]

    def _begin(self):
        if self._action.id != GameAction.ACTION6:
            return []
        data = self._action.data or {}
        cell = self._cell_at(int(data.get("x", -1)), int(data.get("y", -1)))
        if cell is None:
            return []
        cfg = self._cfg
        n = cfg["n"]
        r, c = cell
        if cfg["mode"] == "turns":
            if self._limits.get(cell, 0) <= 0:
                self._left = max(0, self._left - 1)
                if self._left == 0:
                    self._outcome = "lose"
                return self._static_frames()
            self._limits[cell] -= 1
            self._masks[r][c] = rot(self._masks[r][c])
        elif cfg["mode"] == "plus":
            fixed = fixed_cells(cfg)
            for tr, tc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= tr < n and 0 <= tc < n and (tr, tc) not in fixed and self._masks[tr][tc]:
                    self._masks[tr][tc] = rot(self._masks[tr][tc])
        else:
            self._masks[r][c] = rot(self._masks[r][c])
        self._left = max(0, self._left - 1)
        reaches = flow(cfg, self._masks)
        depth = max(max(d.values()) for _, d in reaches)
        frames = []
        for t in range(depth + 1):
            lit = {}
            for col, dist in reaches:
                for cc, dd in dist.items():
                    if dd <= t:
                        lit[cc] = col
            frames.append((lit, ()))
        seen = {}
        shorted = set()
        for i, (col, dist) in enumerate(reaches):
            for cc in dist:
                if cc in seen and seen[cc] != i:
                    shorted.add(cc)
                seen[cc] = i
        lamps_ok = all(
            any(col == lc and (lr, lc_) in dist for col, dist in reaches)
            for lr, lc_, _, lc in cfg["lamps"]
        )
        if shorted:
            frames.append((frames[-1][0], tuple(shorted)))
            self._outcome = "lose"
        elif lamps_ok:
            self._outcome = "win"
        elif self._left == 0:
            self._outcome = "lose"
        else:
            self._outcome = None
        return frames

    def step(self) -> None:
        if self._pending is None:
            self._pending = self._begin()
        if self._pending:
            lit, shorted = self._pending.pop(0)
            self._draw(self.current_level, lit, shorted)
            if self._pending:
                return
        outcome = self._outcome
        self._pending = None
        self._outcome = None
        if outcome == "win":
            self.next_level()
        elif outcome == "lose":
            self.lose()
        self.complete_action()
