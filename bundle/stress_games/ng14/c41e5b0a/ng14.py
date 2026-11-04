import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

HUD_H = 18
GROUND_Y = 52
SKY = 5
HUD_BG = 4
PAGE = 1
INK = 5
SLOT_EMPTY = 3
PIP_ON = 15
GRASS = 14
EARTH = 13
FILL = 2
FILL_EDGE = 3
DOOR = 11
STAR = 1
DEAD = 8
FADED = 3
SLOT_CAP = 4
TILE_X = 2
TILE_Y = 2
SLOT_Y = 10
PITCH = 9
PIP_X = 48
PIP_Y = 2
START_X = 11
STAND = 5
EXIT_X = 61
MAN_START = 1
STEP = 2
REACH = 4

FIRE, WATER, STONE, WIND, POWER = range(5)
GLYPHS = {
    FIRE: ("..#..", ".#.#.", "#.#.#", "..#..", "..#.."),
    WATER: ("#...#", "##..#", "#.#.#", "#..##", "#...#"),
    STONE: ("#####", "#...#", "#.#.#", "#...#", "#####"),
    WIND: ("##..#", "..#.#", "#.#..", "#.#.#", "#..##"),
    POWER: ("#.#.#", ".###.", "..#..", ".###.", "#.#.#"),
}
COLOUR = {FIRE: 12, WATER: 9, STONE: 2, WIND: 0, POWER: 11}
BURST = {FIRE: (12, 8, 11), WATER: (9, 10, 9), STONE: (2, 3, 2), WIND: (0, 2, 0)}
HELPS = {"B": FIRE, "I": FIRE, "F": WATER, "C": STONE, "K": WIND, "R": WIND}
DEADLY = {("F", WIND), ("K", FIRE)}
BRAMBLE = ("ggtgg", ".gg.g", "gtggt", "g.g..")
KEG = ("..y..", ".hhh.", "mmrmm", "mrrrm", "mmrmm", ".hhh.")
KEG_COLOURS = {"y": 11, "h": 4, "m": 13, "r": 8}
MAN = (".s.", ".s.", "rrr", "rrr", "rrr", "l.l")
MAN_COLOURS = {"s": 7, "r": 6, "l": 4}

SPECS = [
    ((WIND, FIRE, STONE, WATER), ("B1", "C1", "F1", "B1")),
    ((STONE, WATER, WIND, FIRE), ("I1", "K1", "F1", "C1", "B1")),
    ((FIRE, POWER, WIND, WATER, STONE), ("B2", "C2", "B1", "R2")),
    ((WATER, STONE, POWER, FIRE, WIND), ("I2/K1", "R1", "F3", "C1")),
    ((POWER, WIND, FIRE, STONE, WATER), ("F1/C1", "B2/R2", "K1", "C3")),
    ((STONE, FIRE, WATER, POWER, WIND), ("F2/C2", "I3", "I2/K1", "R3")),
]


def parse_layers(text):
    return [(part[0], int(part[1])) for part in text.split("/")]


def base_width(kind, n):
    if kind == "R":
        return 3 + 2 * n
    if kind == "C":
        return 4 * n
    return 5


def layout(layers):
    kind, n = layers[-1]
    geoms = [(0, base_width(kind, n))]
    for kind, n in reversed(layers[:-1]):
        gx, gw = geoms[-1]
        geoms.append((gx - 1, gw + 2) if kind == "I" else (gx, gw))
    low = min(g[0] for g in geoms)
    geoms = [(g[0] - low, g[1]) for g in geoms]
    return list(reversed(geoms)), max(g[0] + g[1] for g in geoms)


def build_obstacles(spec):
    obs = []
    for text in spec[1]:
        layers = parse_layers(text)
        geoms, width = layout(layers)
        relight = len(layers) == 2 and layers[0][0] == "F" and layers[1][0] == "C"
        obs.append(dict(orig=tuple(layers), layers=list(layers), geoms=geoms, w=width, relight=relight, x=0))
    total = sum(ob["w"] for ob in obs)
    room = EXIT_X - 2 - START_X
    gap = max(6, min(12, (room - total) // max(1, len(obs) - 1)))
    x = START_X
    for ob in obs:
        ob["x"] = x
        x += ob["w"] + gap
    return obs


def casts_of(spec):
    count = 0
    for text in spec[1]:
        layers = parse_layers(text)
        runes = sum(n for kind, n in layers)
        count += 1 if runes <= SLOT_CAP else len(layers)
    return count


def budget_of(spec):
    return 2 * casts_of(spec) + 2


def solution_of(spec):
    runes = spec[0]
    steps = []
    man = MAN_START
    for ob, text in zip(build_obstacles(spec), spec[1]):
        stand = ob["x"] - STAND
        while man < stand:
            man = min(stand, man + STEP)
            steps.append("R")
        layers = parse_layers(text)
        groups = [[]]
        for kind, n in layers:
            spell = [POWER] * (n - 1) + [HELPS[kind]]
            if len(groups[-1]) + len(spell) > SLOT_CAP:
                groups.append([])
            groups[-1].extend(spell)
        for group in groups:
            for rune in group:
                i = runes.index(rune)
                steps.append((TILE_X + PITCH * i + 3, TILE_Y + 3))
            steps.append("S")
    while True:
        steps.append("R")
        man += STEP
        if man >= EXIT_X - 1:
            return steps


SOLUTIONS = [solution_of(spec) for spec in SPECS]


def stars_of(index):
    out = []
    for k in range(9):
        x = (k * 23 + index * 17 + 5) % 58 + 2
        y = HUD_H + 2 + (k * 7 + index * 5) % 24
        out.append((x, y))
    return out


def layer_pixels(kind, n, lx, lw, under):
    out = {}
    if kind == "B":
        for dy in range(4 * n):
            row = BRAMBLE[dy % 4]
            for dx in range(lw):
                ch = row[dx % 5]
                if ch == "g":
                    out[(lx + dx, GROUND_Y - 1 - dy)] = 14
                elif ch == "t":
                    out[(lx + dx, GROUND_Y - 1 - dy)] = 13
    elif kind == "I":
        top = GROUND_Y - 4 * n
        for yy in range(top, GROUND_Y):
            for xx in range(lx, lx + lw):
                if (xx, yy) in under:
                    continue
                shine = (xx - lx + yy - top) % 7 == 1 and xx - lx < lw - 1
                out[(xx, yy)] = 0 if shine else 10
    elif kind == "F":
        for dx in range(lw):
            height = 4 * n - (dx % 2)
            for dy in range(height):
                colour = 8 if dy < 2 else (11 if dy >= height - 1 else 12)
                out[(lx + dx, GROUND_Y - 1 - dy)] = colour
    elif kind == "K":
        top = GROUND_Y - len(KEG)
        for dy, row in enumerate(KEG):
            for dx, ch in enumerate(row):
                if ch != ".":
                    out[(lx + dx, top + dy)] = KEG_COLOURS[ch]
    elif kind == "R":
        d = 3 + 2 * n
        c = (d - 1) / 2.0
        top = GROUND_Y - d
        for dy in range(d):
            for dx in range(d):
                if (dx - c) ** 2 + (dy - c) ** 2 <= (c + 0.35) ** 2:
                    out[(lx + dx, top + dy)] = 2 if dx + dy < c and dx > 0 and dy > 0 else 3
    return out


def obstacle_pixels(ob, layers=None):
    layers = ob["layers"] if layers is None else layers
    geoms = ob["geoms"][len(ob["orig"]) - len(layers):]
    under = {}
    for (kind, n), (gx, gw) in reversed(list(zip(layers, geoms))):
        under.update(layer_pixels(kind, n, ob["x"] + gx, gw, under))
    return under


def top_split(ob):
    layers = ob["layers"]
    rest = obstacle_pixels(ob, layers[1:])
    gx, gw = ob["geoms"][len(ob["orig"]) - len(layers)]
    kind, n = layers[0]
    return rest, layer_pixels(kind, n, ob["x"] + gx, gw, rest), (ob["x"] + gx, gw)


def chasm_of(ob):
    if ob["orig"][-1][0] != "C":
        return None
    gx, gw = ob["geoms"][-1]
    return ob["x"] + gx, gw


def stamp(px, rows, x0, y0, colour):
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch == "#":
                px[y0 + dy, x0 + dx] = colour


def put(px, pixels):
    for (x, y), colour in pixels.items():
        if 0 <= x < 64 and HUD_H <= y < 64:
            px[y, x] = colour


def render_world(index, obs, man_x, mode, over=None, fills=None, extras=None, man_dy=0):
    over = over or {}
    fills = fills or {}
    px = np.full((64, 64), SKY, dtype=np.int8)
    for x, y in stars_of(index):
        px[y, x] = STAR
    px[GROUND_Y, :] = GRASS
    px[GROUND_Y + 1:, :] = EARTH
    for i, ob in enumerate(obs):
        span = chasm_of(ob)
        if span is None:
            continue
        cx, cw = span
        open_pit = any(kind == "C" for kind, n in ob["layers"])
        rows = fills.get(i, 0 if open_pit else 64 - GROUND_Y)
        px[GROUND_Y:, cx:cx + cw] = SKY
        for yy in range(64 - rows, 64):
            for xx in range(cx, cx + cw):
                edge = (yy - GROUND_Y) % 3 == 2 or (xx - cx) % 4 == 3
                px[yy, xx] = FILL_EDGE if edge else FILL
    px[GROUND_Y - 9, EXIT_X:EXIT_X + 3] = DOOR
    px[GROUND_Y - 9:GROUND_Y, EXIT_X] = DOOR
    px[GROUND_Y - 9:GROUND_Y, EXIT_X + 2] = DOOR
    for i, ob in enumerate(obs):
        put(px, over[i] if i in over else obstacle_pixels(ob))
    if mode is not None:
        top = GROUND_Y - len(MAN) + man_dy
        for dy, row in enumerate(MAN):
            for dx, ch in enumerate(row):
                x = man_x + dx
                if ch != "." and 0 <= x < 64 and top + dy < 64:
                    px[top + dy, x] = MAN_COLOURS[ch] if mode == "alive" else (DEAD if mode == "dead" else FADED)
    if extras:
        put(px, extras)
    return px


def draw_hud(px, runes, slot, log, budget):
    px[:HUD_H, :] = HUD_BG
    for i, rune in enumerate(runes):
        x0 = TILE_X + PITCH * i
        px[TILE_Y:TILE_Y + 7, x0:x0 + 7] = PAGE
        stamp(px, GLYPHS[rune], x0 + 1, TILE_Y + 1, INK)
    for j in range(SLOT_CAP):
        x0 = TILE_X + PITCH * j
        if j < len(slot):
            px[SLOT_Y:SLOT_Y + 7, x0:x0 + 7] = PAGE
            stamp(px, GLYPHS[slot[j]], x0 + 1, SLOT_Y + 1, INK)
        else:
            px[SLOT_Y:SLOT_Y + 7, x0:x0 + 7] = SLOT_EMPTY
    for k in range(budget):
        x = PIP_X + 3 * (k % 5)
        y = PIP_Y + 3 * (k // 5)
        px[y:y + 2, x:x + 2] = log[k] if k < len(log) else PIP_ON


def parse_spell(slot):
    out = []
    power = 0
    for rune in slot:
        if rune == POWER:
            power += 1
        else:
            out.append((rune, 1 + power))
            power = 0
    return out


def ring(cx, cy, radius, colours):
    out = {}
    for dy in range(-radius - 1, radius + 2):
        for dx in range(-radius - 1, radius + 2):
            d = (dx * dx + dy * dy) ** 0.5
            if abs(d - radius) < 0.8 or (d < radius and (dx + dy) % 3 == 0):
                out[(cx + dx, cy + dy)] = colours[(dx * 3 + dy) % len(colours)]
    return out


def square(cx, cy, size, colour):
    out = {}
    for dy in range(size):
        for dx in range(size):
            out[(cx - size // 2 + dx, cy - size // 2 + dy)] = colour
    return out


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), SKY, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(SPECS))]


class Ng14(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._index = 0
        self._obs = []
        self._cur = 0
        self._man = 0
        self._mode = "alive"
        self._slot = []
        self._mana = 0
        self._budget = 0
        self._log = []
        self._frames = None
        self._at = 0
        self._finish = None
        super().__init__(
            game_id="ng14-c41e5b0a",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=SKY, letter_box=SKY),
            available_actions=[3, 4, 5, 6, 7],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._index = self._current_level_index
        spec = SPECS[self._index]
        self._obs = build_obstacles(spec)
        self._cur = 0
        self._man = MAN_START
        self._mode = "alive"
        self._slot = []
        self._budget = budget_of(spec)
        self._mana = self._budget
        self._log = []
        self._frames = None
        self._at = 0
        self._finish = None
        self._show(self._frame(), level)

    def _show(self, px, level=None):
        level = level or self.current_level
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _frame(self, man_x=None, mode=None, over=None, fills=None, extras=None, world=None, man_dy=0):
        if world is None:
            world = render_world(self._index, self._obs, self._man if man_x is None else man_x,
                                 self._mode if mode is None else mode, over, fills, extras, man_dy)
        px = world.copy()
        draw_hud(px, SPECS[self._index][0], self._slot, self._log, self._budget)
        return px

    def step(self) -> None:
        if self._frames is None:
            self._frames = self._react()
            self._at = 0
        self._show(self._frames[self._at])
        self._at += 1
        if self._at >= len(self._frames):
            self._frames = None
            finish = self._finish
            self._finish = None
            if finish == "lose":
                self.lose()
            elif finish == "next":
                self.next_level()
            self.complete_action()

    def _react(self):
        action = self._action.id
        if action == GameAction.ACTION6:
            x = int(self._action.data.get("x", -1))
            y = int(self._action.data.get("y", -1))
            runes = SPECS[self._index][0]
            if TILE_Y <= y < TILE_Y + 7 and x >= TILE_X:
                i = (x - TILE_X) // PITCH
                if i < len(runes) and (x - TILE_X) % PITCH < 7 and len(self._slot) < SLOT_CAP:
                    self._slot.append(runes[i])
            return [self._frame()]
        if action == GameAction.ACTION7:
            if self._slot:
                self._slot.pop()
            return [self._frame()]
        if action == GameAction.ACTION5 and self._slot:
            return self._cast()
        if action == GameAction.ACTION3:
            return self._walk(max(MAN_START, self._man - STEP)) or [self._frame()]
        if action == GameAction.ACTION4:
            return self._forward()
        return [self._frame()]

    def _forward(self):
        if self._cur >= len(self._obs):
            if self._man + STEP >= EXIT_X - 1:
                frames = self._walk(64)
                frames += self._scroll()
                self._finish = "next"
                return frames
            return self._walk(self._man + STEP)
        ob = self._obs[self._cur]
        stand = ob["x"] - STAND
        if self._man < stand:
            return self._walk(min(stand, self._man + STEP))
        kind = ob["layers"][0][0]
        if kind == "C":
            frames = self._walk(chasm_of(ob)[0])
            for dy in (3, 7, 11):
                frames.append(self._frame(man_dy=dy))
            self._mode = "dead"
            frames.append(self._frame(man_dy=11))
            self._finish = "lose"
            return frames
        if kind == "F":
            frames = self._walk(ob["x"] - 2)
            frames.append(self._frame(extras=ring(self._man + 1, GROUND_Y - 4, 3, BURST[FIRE])))
            self._mode = "dead"
            frames.append(self._frame())
            self._finish = "lose"
            return frames
        return [self._frame(man_x=self._man + 1), self._frame()]

    def _impact(self, ob):
        kind, n = ob["layers"][0]
        if kind == "C":
            cx, cw = chasm_of(ob)
            return cx + cw // 2, GROUND_Y + 2
        gx, gw = ob["geoms"][len(ob["orig"]) - len(ob["layers"])]
        return ob["x"] + gx, GROUND_Y - 3

    def _flight(self, i, colour, size, target):
        frames = []
        sx, sy = self._man + 2, GROUND_Y - 5
        tx, ty = target
        for k in range(1, 5):
            t = k / 4.0
            x = int(round(sx + (tx - sx) * t))
            y = int(round(sy + (ty - sy) * t - 10 * t * (1 - t)))
            frames.append(self._frame(extras=square(x, y, size, colour)))
        return frames

    def _cast(self):
        spell = parse_spell(self._slot)
        self._slot = []
        self._mana -= 1
        self._log.append(COLOUR[spell[0][0]] if spell else COLOUR[POWER])
        i = self._cur
        frames = [self._frame()]
        if i >= len(self._obs) or self._man < self._obs[i]["x"] - STAND - REACH:
            colour = COLOUR[spell[0][0]] if spell else COLOUR[POWER]
            size = spell[0][1] + 1 if spell else 1
            frames += self._flight(i, colour, size, (self._man + 10, GROUND_Y - 1))
            frames += self._sparkle(spell[0][0] if spell else POWER, (self._man + 10, GROUND_Y - 1))
            return frames + self._drained()
        ob = self._obs[i]
        if not spell:
            hx, hy = self._man + 1, GROUND_Y - 8
            frames.append(self._frame(extras=square(hx, hy, 1, COLOUR[POWER])))
            frames.append(self._frame(extras={(hx - 1, hy - 1): COLOUR[POWER], (hx + 1, hy - 1): COLOUR[POWER]}))
        for element, size in spell:
            if not ob["layers"]:
                break
            target = self._impact(ob)
            frames += self._flight(i, COLOUR[element], size + 1, target)
            kind, n = ob["layers"][0]
            helps = HELPS[kind] == element
            if (kind, element) in DEADLY or (helps and size > n):
                return frames + self._death(i, element, target)
            if helps and size == n:
                frames += self._remove(i)
                ob["layers"].pop(0)
            elif helps:
                frames += self._partial(i, size)
            else:
                frames += self._sparkle(element, target)
        if ob["relight"] and ob["layers"] and ob["layers"][0][0] == "C":
            ob["layers"].insert(0, ob["orig"][0])
            rest, top, (lx, lw) = top_split(ob)
            low = {p: c for p, c in top.items() if p[1] >= GROUND_Y - 2}
            frames.append(self._frame(over={i: {**rest, **low}}))
            frames.append(self._frame())
        if not ob["layers"]:
            self._cur += 1
        return frames + self._drained()

    def _drained(self):
        if self._mana > 0 or self._cur >= len(self._obs):
            return []
        self._mode = "faded"
        self._finish = "lose"
        return [self._frame()]

    def _remove(self, i):
        ob = self._obs[i]
        kind, n = ob["layers"][0]
        if kind == "C":
            depth = 64 - GROUND_Y
            return [self._frame(fills={i: depth * k // 4}) for k in (1, 2, 3)]
        rest, top, span = top_split(ob)
        if kind == "B":
            first = {p: (12 if (p[0] + p[1]) % 2 else 8) for p in top}
            second = {p: 3 for p in top if (p[0] + p[1]) % 3 == 0}
        elif kind == "I":
            mid = GROUND_Y - 2 * n
            first = {p: 9 for p in top}
            second = {p: 9 for p in top if p[1] >= mid}
        elif kind == "F":
            first = {p: 2 for p in top}
            second = {(p[0], p[1] - 2): 2 for p in top if (p[0] + p[1]) % 3 == 0}
        else:
            first = {(p[0] + 2, p[1] - 2): c for p, c in top.items()}
            second = {(p[0] + 7, p[1] - 5): c for p, c in top.items()}
        return [self._frame(over={i: {**rest, **first}}), self._frame(over={i: {**rest, **second}}),
                self._frame(over={i: rest})]

    def _partial(self, i, size):
        ob = self._obs[i]
        kind, n = ob["layers"][0]
        if kind == "C":
            depth = 64 - GROUND_Y
            part = depth * size // n
            return [self._frame(fills={i: part // 2}), self._frame(fills={i: part}), self._frame(fills={i: 0}),
                    self._frame()]
        rest, top, span = top_split(ob)
        if kind == "R":
            return [self._frame(over={i: {**rest, **{(p[0] + dx, p[1]): c for p, c in top.items()}}})
                    for dx in (1, 2, 1)] + [self._frame()]
        cut = GROUND_Y - 4 * (n - size)
        burning = {p: (2 if kind == "F" else (9 if kind == "I" else 12)) if p[1] < cut else c for p, c in top.items()}
        kept = {p: c for p, c in top.items() if p[1] >= cut}
        return [self._frame(over={i: {**rest, **burning}}), self._frame(over={i: {**rest, **kept}}),
                self._frame(over={i: {**rest, **kept}}), self._frame()]

    def _sparkle(self, element, target):
        colour = COLOUR[element]
        tx, ty = target
        near = {(tx - 1, ty - 1): colour, (tx + 1, ty - 1): colour, (tx - 1, ty + 1): colour, (tx + 1, ty + 1): colour}
        far = {(tx - 2, ty - 2): colour, (tx + 2, ty - 2): colour, (tx - 2, ty + 2): colour, (tx + 2, ty + 2): colour}
        return [self._frame(extras=near), self._frame(extras=far), self._frame()]

    def _death(self, i, element, target):
        ob = self._obs[i]
        kind, n = ob["layers"][0]
        colours = (11, 12, 8) if kind == "K" else BURST[element]
        if kind == "F" and element == WIND:
            colours = BURST[FIRE]
        tx, ty = target
        cx = (tx + self._man + 1) // 2
        frames = [self._frame(extras=ring(tx, ty, 2, colours)),
                  self._frame(extras=ring(cx, GROUND_Y - 4, 4, colours)),
                  self._frame(extras=ring(self._man + 1, GROUND_Y - 4, 3, colours), mode="dead")]
        self._mode = "dead"
        if kind == "K":
            ob["layers"].pop(0)
        frames.append(self._frame())
        self._finish = "lose"
        return frames

    def _walk(self, goal):
        frames = []
        while self._man != goal:
            self._man += 1 if goal > self._man else -1
            frames.append(self._frame())
        return frames

    def _scroll(self):
        if self._index + 1 >= len(SPECS):
            return []
        here = render_world(self._index, self._obs, 0, None)
        nxt_obs = build_obstacles(SPECS[self._index + 1])
        there = render_world(self._index + 1, nxt_obs, MAN_START, "alive")
        strip = np.concatenate([here, there], axis=1)
        frames = []
        for shift in range(8, 65, 8):
            frames.append(self._frame(world=strip[:, shift:shift + 64]))
        return frames
