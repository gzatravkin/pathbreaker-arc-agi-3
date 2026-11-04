import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
NEUTRAL = 5
ZONE = 10
FOG = 3
FOG_EDGE = 2
ENEMY_GROUND = 4
ENEMY_EDGE = 13
SELECT = 11
PIP = 0
HP_ON = 14
HP_OFF = 4
SHOT = 11
FLASH = 0
LOSS_EDGE = 8
TICK_MARK = 2
CELL = 4
FIELD_X = 0
FIELD_Y = 2
FIELD_W = 16
FIELD_H = 10
BENCH_Y = 45
TICK_FRAMES = 5
FOG_FRAMES = 16
PLAYER = 0
ENEMY = 1
TEAM = {PLAYER: (9, 15), ENEMY: (8, 13)}

ART = {
    "archer": ("..EE",
               ".EcM",
               "TTM0",
               ".3.3"),
    "sword": (".22.",
              ".cc.",
              "T220",
              "TDD0",
              "TDD0",
              ".DD.",
              ".3.3",
              ".3.3"),
    "lancer": ("0...",
               "0...",
               "02..",
               "0cc.",
               "0TT.",
               "MTTD",
               "MTTD",
               "MDD.",
               "M33.",
               "M3.3",
               "..3.",
               "...."),
    "knight": ("...YY...",
               "..2222..",
               ".22cc2..",
               "TTT2220.",
               "TYT2220.",
               "TTTDDD0.",
               ".T.DDD..",
               "..33.33."),
    "cavalry": ("....0.......",
                "...T0.......",
                "..cTT0......",
                "..TDT.0.....",
                ".MMMMMMM6...",
                "MMMMMMMM66..",
                ".M3....3M...",
                ".3.....3...."),
    "giant": ("....7777....",
              "...777777...",
              "...76776....",
              "...777777..M",
              ".7777777777M",
              "77TTTTTTT77M",
              "77DYYYYYD77M",
              "7.TTTTTTT.7M",
              "..TTTTTTT...",
              "..77...77...",
              "..77...77...",
              ".333...333.."),
}

FIXED = {"0": 0, "2": 2, "3": 3, "6": 6, "7": 7, "c": 12, "E": 14, "M": 13, "Y": 11}

STATS = {
    "archer": dict(w=1, h=1, hp=8, dmg=3, rng=3, acd=2, mcd=2, bonus={"lancer": 3, "giant": 2}, guard=()),
    "sword": dict(w=1, h=2, hp=16, dmg=4, rng=1, acd=2, mcd=1, bonus={"archer": 3, "lancer": 2}, guard=()),
    "lancer": dict(w=1, h=3, hp=18, dmg=4, rng=2, acd=2, mcd=2, bonus={"cavalry": 4, "giant": 3}, guard=()),
    "knight": dict(w=2, h=2, hp=36, dmg=4, rng=1, acd=2, mcd=2, bonus={"sword": 2}, guard=("archer",)),
    "cavalry": dict(w=3, h=2, hp=30, dmg=6, rng=1, acd=2, mcd=1, bonus={"archer": 3, "sword": 2}, guard=()),
    "giant": dict(w=3, h=3, hp=60, dmg=10, rng=1, acd=3, mcd=3, bonus={"knight": 2, "sword": 2}, guard=()),
}

ORDER = ("archer", "sword", "lancer", "knight", "cavalry", "giant")

CONFIGS = [
    dict(zone=(2, 10), bench={"lancer": 2, "sword": 3, "archer": 3, "knight": 2}, ticks=70,
         enemy=(("cavalry", 12, 0), ("cavalry", 12, 2), ("archer", 15, 4), ("archer", 15, 5), ("archer", 14, 4),
                ("knight", 12, 6), ("sword", 15, 8), ("sword", 14, 8))),
    dict(zone=(2, 10), bench={"lancer": 2, "sword": 3, "archer": 3, "knight": 2}, ticks=70,
         enemy=(("giant", 12, 3), ("sword", 13, 0), ("sword", 14, 0), ("sword", 13, 8), ("sword", 14, 8),
                ("archer", 15, 2), ("archer", 15, 6), ("archer", 15, 7), ("knight", 12, 6))),
    dict(zone=(3, 10), bench={"cavalry": 2, "giant": 1, "lancer": 2, "sword": 2, "archer": 3}, ticks=76,
         enemy=(("lancer", 12, 0), ("lancer", 13, 0), ("archer", 15, 0), ("archer", 15, 1), ("knight", 12, 4),
                ("knight", 12, 6), ("archer", 15, 5), ("archer", 14, 8), ("archer", 15, 8), ("archer", 15, 9),
                ("lancer", 11, 5))),
    dict(zone=(3, 10), bench={"giant": 1, "knight": 2, "lancer": 3, "sword": 2, "archer": 4}, ticks=80,
         enemy=(("cavalry", 12, 0), ("cavalry", 12, 2), ("lancer", 12, 4), ("lancer", 13, 4), ("archer", 15, 4),
                ("archer", 15, 5), ("giant", 12, 7), ("archer", 15, 8), ("archer", 15, 9), ("sword", 11, 4),
                ("sword", 15, 0), ("archer", 11, 6), ("archer", 11, 8))),
    dict(zone=(3, 10), bench={"cavalry": 2, "giant": 1, "knight": 2, "lancer": 3, "sword": 3, "archer": 4}, ticks=90,
         enemy=(("cavalry", 12, 0), ("archer", 15, 0), ("archer", 15, 1), ("knight", 11, 2), ("lancer", 13, 2),
                ("lancer", 14, 2), ("knight", 11, 5), ("giant", 13, 5), ("cavalry", 11, 8), ("sword", 14, 8),
                ("archer", 15, 8), ("archer", 15, 9), ("sword", 15, 3))),
]

SOLUTIONS = [
    [(18, 49), (6, 8), (6, 20), (11, 49), (2, 20), (27, 49), (2, 28), (2, 36), (4, 49), (2, 4), (2, 8), (2, 12), "S"],
    [(18, 49), (6, 20), (2, 20), (27, 49), (2, 4), (2, 36), (11, 49), (6, 28), (2, 28), (4, 49), (6, 12), (2, 12), "S"],
    [(4, 49), (10, 4), (10, 8), (10, 12), (11, 49), (6, 4), (6, 12), (44, 49), (6, 24), (29, 49), (6, 36), (18, 49),
     (2, 12), "S"],
    [(11, 49), (2, 4), (2, 16), (18, 49), (6, 8), (10, 8), (10, 28), (27, 49), (6, 16), (6, 36), (4, 49), (6, 24),
     (6, 28), (6, 32), (2, 40), "S"],
    [(18, 49), (2, 8), (10, 12), (10, 36), (11, 49), (6, 4), (6, 12), (2, 32), (4, 49), (2, 20), (10, 20), (6, 32),
     (6, 40), (40, 49), (6, 24), "S"],
]


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


def build_art():
    out = {}
    for kind, rows in ART.items():
        for side, (team, dark) in TEAM.items():
            grid = np.full((len(rows), len(rows[0])), -1, dtype=np.int8)
            for y, row in enumerate(rows):
                for x, ch in enumerate(row):
                    if ch == "T":
                        grid[y, x] = team
                    elif ch == "D":
                        grid[y, x] = dark
                    elif ch in FIXED:
                        grid[y, x] = FIXED[ch]
            out[(kind, side)] = grid
    return out


SPRITES = build_art()


class Unit:
    def __init__(self, uid, side, kind, x, y):
        s = STATS[kind]
        self.uid = uid
        self.side = side
        self.kind = kind
        self.x = x
        self.y = y
        self.w = s["w"]
        self.h = s["h"]
        self.hp = s["hp"]
        self.cool = 0

    def cells(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        return [(x + i, y + j) for i in range(self.w) for j in range(self.h)]


def gap(a, b):
    dx = max(0, b.x - (a.x + a.w - 1), a.x - (b.x + b.w - 1))
    dy = max(0, b.y - (a.y + a.h - 1), a.y - (b.y + b.h - 1))
    return dx, dy


def sign_toward(a, b):
    sx = 1 if b.x > a.x + a.w - 1 else (-1 if b.x + b.w - 1 < a.x else 0)
    sy = 1 if b.y > a.y + a.h - 1 else (-1 if b.y + b.h - 1 < a.y else 0)
    return sx, sy


def simulate(player, enemy, ticks):
    units = []
    for kind, x, y in player:
        units.append(Unit(len(units), PLAYER, kind, x, y))
    for kind, x, y in enemy:
        units.append(Unit(len(units), ENEMY, kind, x, y))
    history = []
    for _ in range(ticks):
        alive = [u for u in units if u.hp > 0]
        before = {u.uid: (u.x, u.y) for u in alive}
        shots = []
        occupied = {}
        for u in alive:
            for c in u.cells():
                occupied[c] = u.uid
        plans = []
        for u in alive:
            foes = [v for v in alive if v.side != u.side]
            if not foes:
                continue
            lane = [v for v in foes if v.y < u.y + u.h and u.y < v.y + v.h]
            if lane:
                target = min(lane, key=lambda v: (gap(u, v)[0], v.hp, v.y, v.uid))
                plans.append((gap(u, target)[0], u.side, u.uid, u, target, True))
            else:
                target = min(foes, key=lambda v: (max(gap(u, v)), v.hp, v.y, v.uid))
                plans.append((FIELD_W + max(gap(u, target)), u.side, u.uid, u, target, False))
        plans.sort(key=lambda p: (p[0], p[1], p[2]))
        for _, _, _, u, target, in_lane in plans:
            s = STATS[u.kind]
            if u.cool > 0:
                u.cool -= 1
                continue
            if in_lane and gap(u, target)[0] <= s["rng"]:
                dmg = s["dmg"] * s["bonus"].get(target.kind, 1)
                if u.kind in STATS[target.kind]["guard"]:
                    dmg = max(1, dmg // 2)
                shots.append((u.uid, target.uid, dmg))
                u.cool = s["acd"] - 1
                continue
            sx, sy = sign_toward(u, target)
            tries = [(sx, 0)] if in_lane else [(0, sy), (sx, 0)]
            for mx, my in tries:
                if mx == 0 and my == 0:
                    continue
                nx, ny = u.x + mx, u.y + my
                if nx < 0 or ny < 0 or nx + u.w > FIELD_W or ny + u.h > FIELD_H:
                    continue
                if any(occupied.get(c, u.uid) != u.uid for c in u.cells(nx, ny)):
                    continue
                for c in u.cells():
                    del occupied[c]
                u.x, u.y = nx, ny
                for c in u.cells():
                    occupied[c] = u.uid
                break
            u.cool = s["mcd"] - 1
        by_id = {u.uid: u for u in units}
        for _, tid, dmg in shots:
            by_id[tid].hp -= dmg
        history.append((before, shots, {u.uid: (u.x, u.y, u.hp) for u in alive}))
    alive = [u for u in units if u.hp > 0]
    won = not any(u.side == ENEMY for u in alive)
    return units, history, won


def bench_slots(bench):
    slots = []
    x = 1
    for kind in ORDER:
        if kind not in bench:
            continue
        w = STATS[kind]["w"] * CELL + 2
        slots.append((kind, x, w))
        x += w + 1
    return slots


class Ng12(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._revealed = set()
        self._cfg = CONFIGS[0]
        self._placed = []
        self._counts = {}
        self._selected = None
        super().__init__(
            game_id="ng12-9b2d4e61",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[5, 6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._placed = []
        self._counts = dict(self._cfg["bench"])
        self._selected = None
        self._battle = None
        self._draw_deploy(level)

    def _zone_cell(self, x, y):
        zw, zh = self._cfg["zone"]
        cx = (x - FIELD_X) // CELL
        cy = (y - FIELD_Y) // CELL
        if 0 <= cx < zw and 0 <= cy < zh and x >= FIELD_X and y >= FIELD_Y:
            return cx, cy
        return None

    def _occupant(self, cell):
        for i, (kind, x, y) in enumerate(self._placed):
            s = STATS[kind]
            if x <= cell[0] < x + s["w"] and y <= cell[1] < y + s["h"]:
                return i
        return None

    def _click(self, x, y):
        if y >= BENCH_Y - 1:
            for kind, sx, w in bench_slots(self._cfg["bench"]):
                if sx <= x < sx + w:
                    self._selected = kind
                    return
            return
        cell = self._zone_cell(x, y)
        if cell is None:
            return
        hit = self._occupant(cell)
        if hit is not None:
            kind = self._placed.pop(hit)[0]
            self._counts[kind] += 1
            return
        kind = self._selected
        if kind is None or self._counts[kind] <= 0:
            return
        s = STATS[kind]
        zw, zh = self._cfg["zone"]
        ax = cell[0] - (s["w"] - 1) // 2
        ay = cell[1] - (s["h"] - 1) // 2
        if ax < 0 or ay < 0 or ax + s["w"] > zw or ay + s["h"] > zh:
            return
        for i in range(s["w"]):
            for j in range(s["h"]):
                if self._occupant((ax + i, ay + j)) is not None:
                    return
        self._placed.append((kind, ax, ay))
        self._counts[kind] -= 1

    def step(self) -> None:
        if self._battle is not None:
            self._battle_frame()
            return
        aid = self._action.id
        if aid == GameAction.ACTION6:
            data = self._action.data or {}
            self._click(int(data.get("x", -1)), int(data.get("y", -1)))
            self._draw_deploy()
            self.complete_action()
            return
        if aid == GameAction.ACTION5:
            units, history, won = simulate(self._placed, self._cfg["enemy"], self._cfg["ticks"])
            self._revealed.add(self._current_level_index)
            self._battle = dict(units=units, history=history, won=won, frame=0)
            self._battle_frame()
            return
        self.complete_action()

    def _fog(self, px, fog_cols):
        if fog_cols <= 0:
            return
        left = FIELD_X + FIELD_W * CELL - fog_cols
        px[FIELD_Y:FIELD_Y + FIELD_H * CELL, left:FIELD_X + FIELD_W * CELL] = FOG
        px[FIELD_Y:FIELD_Y + FIELD_H * CELL, left] = FOG_EDGE

    def _ground(self, fog_cols):
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        zw, zh = self._cfg["zone"]
        fx0, fy0 = FIELD_X, FIELD_Y
        px[fy0:fy0 + FIELD_H * CELL, fx0:fx0 + FIELD_W * CELL] = NEUTRAL
        ex = self._enemy_x0()
        px[fy0:fy0 + FIELD_H * CELL, fx0 + ex * CELL:fx0 + FIELD_W * CELL] = ENEMY_GROUND
        px[fy0:fy0 + FIELD_H * CELL, fx0 + ex * CELL] = ENEMY_EDGE
        self._fog(px, fog_cols)
        for i in range(zw + 1):
            px[fy0 - 1, fx0 + i * CELL - (1 if i == zw else 0)] = TICK_MARK
        for j in range(zh + 1):
            yy = fy0 + j * CELL - (1 if j == zh else 0)
            px[yy, fx0 + zw * CELL] = TICK_MARK
        return px

    def _enemy_x0(self):
        return self._cfg.get("ezone", 11)

    def _paint(self, px, kind, side, ox, oy, flash=False):
        art = SPRITES[(kind, side)]
        h, w = art.shape
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(64, ox + w), min(64, oy + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = art[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        region = px[y0:y1, x0:x1]
        mask = sub >= 0
        region[mask] = FLASH if flash else sub[mask]

    def _draw_deploy(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        revealed = self._current_level_index in self._revealed
        px = self._ground(0 if revealed else (FIELD_W - self._enemy_x0()) * CELL)
        zw, zh = cfg["zone"]
        px[FIELD_Y:FIELD_Y + zh * CELL, FIELD_X:FIELD_X + zw * CELL] = ZONE
        for kind, x, y in self._placed:
            self._paint(px, kind, PLAYER, FIELD_X + x * CELL, FIELD_Y + y * CELL)
        if revealed:
            for kind, x, y in cfg["enemy"]:
                self._paint(px, kind, ENEMY, FIELD_X + x * CELL, FIELD_Y + y * CELL)
        for kind, sx, w in bench_slots(cfg["bench"]):
            s = STATS[kind]
            if kind == self._selected:
                px[BENCH_Y - 1, sx:sx + w] = SELECT
                px[63, sx:sx + w] = SELECT
                px[BENCH_Y - 1:64, sx] = SELECT
                px[BENCH_Y - 1:64, sx + w - 1] = SELECT
            self._paint(px, kind, PLAYER, sx + 1, BENCH_Y + 1)
            n = self._counts[kind]
            for i in range(n):
                pxx = sx + 1 + (i % ((w - 1) // 2)) * 2
                pyy = BENCH_Y + 1 + s["h"] * CELL + 2 + (i // ((w - 1) // 2)) * 2
                if pyy < 63:
                    px[pyy, pxx] = PIP
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _battle_frame(self):
        b = self._battle
        f = b["frame"]
        total = FOG_FRAMES + len(b["history"]) * TICK_FRAMES
        if f < FOG_FRAMES:
            span = (FIELD_W - self._enemy_x0()) * CELL
            fog = span - (span * (f + 1)) // FOG_FRAMES
            px = self._ground(0)
            for kind, x, y in self._cfg["enemy"]:
                self._paint(px, kind, ENEMY, FIELD_X + x * CELL, FIELD_Y + y * CELL)
            self._fog(px, fog)
            for kind, x, y in self._placed:
                self._paint(px, kind, PLAYER, FIELD_X + x * CELL, FIELD_Y + y * CELL)
        else:
            t, sub = divmod(f - FOG_FRAMES, TICK_FRAMES)
            px = self._ground(0)
            self._draw_tick(px, b, t, sub)
        b["frame"] = f + 1
        last = b["frame"] >= total
        if last:
            if b["won"]:
                self._battle = None
                self.current_level.get_sprites_by_name("canvas")[0].pixels = px
                self.next_level()
            else:
                px[FIELD_Y - 2, :] = LOSS_EDGE
                px[FIELD_Y + FIELD_H * CELL, :] = LOSS_EDGE
                self.current_level.get_sprites_by_name("canvas")[0].pixels = px
                self._battle = None
                self.lose()
            self.complete_action()
            return
        self.current_level.get_sprites_by_name("canvas")[0].pixels = px

    def _draw_tick(self, px, b, t, sub):
        before, shots, after = b["history"][t]
        units = {u.uid: u for u in b["units"]}
        frac = (sub + 1) / TICK_FRAMES
        hit = {tid for _, tid, _ in shots} if sub >= 3 else set()
        for uid, (x0, y0) in before.items():
            u = units[uid]
            x1, y1, hp = after[uid]
            ox = FIELD_X + x0 * CELL + int(round((x1 - x0) * CELL * frac))
            oy = FIELD_Y + y0 * CELL + int(round((y1 - y0) * CELL * frac))
            dying = hp <= 0 and sub == TICK_FRAMES - 1
            self._paint(px, u.kind, u.side, ox, oy, flash=(uid in hit) or dying)
            s = STATS[u.kind]
            shown = hp if sub >= 3 else self._hp_before(b, t, uid)
            width = s["w"] * CELL
            fill = max(0, (max(0, shown) * width + s["hp"] - 1) // s["hp"])
            if 0 <= oy < 64:
                px[oy, max(0, ox):min(64, ox + width)] = HP_OFF
                px[oy, max(0, ox):min(64, ox + fill)] = HP_ON
        for aid, tid, _ in shots:
            a, v = units[aid], units[tid]
            if STATS[a.kind]["rng"] <= 1:
                continue
            ax, ay, _ = after[aid]
            vx, vy, _ = after[tid]
            sx = FIELD_X + ax * CELL + a.w * CELL // 2
            sy = FIELD_Y + ay * CELL + a.h * CELL // 2
            tx = FIELD_X + vx * CELL + v.w * CELL // 2
            ty = FIELD_Y + vy * CELL + v.h * CELL // 2
            k = (sub + 1) / TICK_FRAMES
            qx = int(round(sx + (tx - sx) * k))
            qy = int(round(sy + (ty - sy) * k))
            if 0 <= qx < 64 and 0 <= qy < 64:
                px[qy, qx] = SHOT

    def _hp_before(self, b, t, uid):
        if t == 0:
            return STATS[b["units"][uid].kind]["hp"]
        prev = b["history"][t - 1][2]
        return prev[uid][2] if uid in prev else STATS[b["units"][uid].kind]["hp"]
