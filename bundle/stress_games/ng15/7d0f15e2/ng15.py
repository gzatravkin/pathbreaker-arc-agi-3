import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

TILE = 4
HUD_H = 8
VIEW_W = 16
VIEW_H = 14
FLOOR_A = 0
FLOOR_B = 1
WALL = 4
TABLE = 13
COUNTER = 3
COUNTER_TOP = 2
RUG = 15
RUG_DOT = 7
CLIENT = 10
ANGRY = 8
HAPPY = 11
WAITER = 5
FACE = 7
TIRED = 3
HUD_BG = 4
FRAME = 2
BAR = 0
SERVED = 3
OUTSIDE = 4
SPARK = 0
DISH = {"a": 8, "b": 11, "c": 9, "d": 14, "e": 6, "f": 12}
MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}
LETTERS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
SCAN = ((0, -1), (1, 0), (0, 1), (-1, 0))

MAPS = [
    (
        "######################",
        "#..........#kkkkkkkkc#",
        "#..T....T..#.........#",
        "#A.........#..k....k.#",
        "#..........r....W....#",
        "#......T...#.k...k...#",
        "#..T.......#.......r.#",
        "#..........#b.......a#",
        "#..........#######r###",
        "#.....T..............#",
        "#................T...#",
        "#.........T..........#",
        "#..T.................#",
        "#.............T......#",
        "#....................#",
        "#.....T...........T..#",
        "#.....B........C.....#",
        "######################",
    ),
    (
        "########################",
        "#......#bkkkkkka#......#",
        "#..T...#........#...T..#",
        "#......#..kc....#.B....#",
        "#......r....kk..r......#",
        "#.T....#........#....T.#",
        "#......#..rW....#......#",
        "#......####r#####......#",
        "#....T.............T...#",
        "#...........T..........#",
        "#..T.....r.............#",
        "#.........T.......T....#",
        "#....C.................#",
        "#.....##r####..........#",
        "#.....#.....#..........#",
        "#.T...#..T..#....T.....#",
        "#.....#.....#..........#",
        "#.....#.....r..........#",
        "#.A...#.....#........T.#",
        "########################",
    ),
    (
        "##########################",
        "#.....#.......A..........#",
        "#..T..#..TT.....TT....T..#",
        "#.....r..................#",
        "#.....#......T...r.......#",
        "#.T...#.............TT...#",
        "#.....#..TT..............#",
        "#.....#.........T........#",
        "#...T.r................T.#",
        "#.....#....T.............#",
        "#.....#.C...B.....TT.....#",
        "#.T...#D......T..........#",
        "#.....#........r..####r###",
        "#.....#...T.......r......#",
        "#.....#...........#.kk..b#",
        "#..T..#.......TT..#......#",
        "#.....r...........#a..kk.#",
        "#.....#..T........#..W...#",
        "#.....#..........T#kkdkkc#",
        "##########################",
    ),
    (
        "############################",
        "#kakbkc#........#..........#",
        "#...r..#..T.....#....T.....#",
        "#k.k...r........r..........#",
        "#......#.....T..#.......T..#",
        "#..k.k.#........#..T.......#",
        "#d..W..#.T......#..........#",
        "####r###........####r#######",
        "#...........T.......D......#",
        "#..T.................T.....#",
        "#................T.........#",
        "#...B...T..................#",
        "#.......r..............T...#",
        "#####r######......##r#######",
        "#..........#..T...#........#",
        "#..T.......#......#....TC..#",
        "#......T...r......#........#",
        "#..........#......r........#",
        "#.T....r...#...T..#..T.....#",
        "#..........#......#........#",
        "#.....T....#......#A.....T.#",
        "############################",
    ),
    (
        "##############################",
        "#.........#.........#........#",
        "#..T......#...T.....#...T....#",
        "#......T..r.........r........#",
        "#.........#......T..#......T.#",
        "#.T.......#..T......#........#",
        "#.........####r######....T...#",
        "#....T....#akkkkkkkc#........#",
        "#.........#.........#........#",
        "#.........rr.db.....r........#",
        "#..Tr.....#......kk.#...T....#",
        "#.........#...W.....#........#",
        "#.......T.#ke.......#........#",
        "#.........####r######......T.#",
        "#............................#",
        "#....T..........T..E.........#",
        "#......................T.....#",
        "#.T.........T....r...........#",
        "#............C......T........#",
        "#D......T..........A......T..#",
        "#...............T............#",
        "#...T........................#",
        "#.........B..T.......T.......#",
        "##############################",
    ),
    (
        "################################",
        "#.........#...........#.B......#",
        "#..T..D...#...T.......#...T....#",
        "#......T..r...........r........#",
        "#..r......#......T....#......T.#",
        "#.T.......#..T........#........#",
        "#.........####rr#######....T..C#",
        "#....T....#ckkkkkkkkdk#........#",
        "#.........#.......r...#........#",
        "#.........r..kb..W....r........#",
        "#..T......#.......fk..#...T...F#",
        "#.........#...........#........#",
        "#.......T.#ak......e..#........#",
        "#.........#####r#######......T.#",
        "#..............................#",
        "#....T..........T..............#",
        "#########r########.......T.....#",
        "#................#.............#",
        "#.T........T.....#.............#",
        "#......T.........r......T......#",
        "#............T...#.............#",
        "#....T...........#...T.......T.#",
        "#.....r..........#.............#",
        "#..T......T......#.............#",
        "#.......E........#......A.T....#",
        "################################",
    ),
]
SOLUTIONS = [
    list("ULDSRULLDRSLURRDSLULLSRRUSDLULLDRS"),
    list("USDDLSRULDLURULLULSDRURDDRULDLURUSLLURSDLULLDS"),
    list("SULULLDLUSDRURDDLURSLLUSDRURDDLSULLURULUSLDRDDRDSULULLULS"),
    list("LUSRDDRDLURDDSUULUURDLSURDDRDLURSLUUSDDSUURSLDDRRULDLURDDRUS"),
    list("RUSLDLLDRSLURDLURDLUULUUSDLLDRSDLUULURDSLULLDRUSDLUULURSLLLDSRDLUUSDDRULDS"),
    list("ULSRDRRUSDLLSRRULSRDLLULDSURDRRULDSURDLLURDSRRSLLURRDLUSLDLDDLUSDRUURRDLSULDLDDLURDDS"),
]


def parse(rows):
    lv = dict(w=len(rows[0]), h=len(rows), walls=set(), tables=set(), counters=set(), rugs=set(),
              dishes={}, clients={}, start=(1, 1))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            p = (x, y)
            if ch == "#":
                lv["walls"].add(p)
            elif ch == "T":
                lv["tables"].add(p)
            elif ch == "k":
                lv["counters"].add(p)
            elif ch == "r":
                lv["rugs"].add(p)
            elif ch == "W":
                lv["start"] = p
            elif ch in DISH:
                lv["counters"].add(p)
                lv["dishes"][p] = ch
            elif ch.lower() in DISH:
                lv["clients"][p] = ch.lower()
    lv["solid"] = lv["walls"] | lv["tables"] | lv["counters"]
    return lv


def solid(lv, p, clients):
    x, y = p
    if x < 0 or y < 0 or x >= lv["w"] or y >= lv["h"]:
        return True
    return p in lv["solid"] or p in clients


def slide(lv, p, d, clients):
    path = []
    while True:
        q = (p[0] + d[0], p[1] + d[1])
        if solid(lv, q, clients):
            return path
        p = q
        path.append(p)
        if p in lv["rugs"]:
            return path


def interact(p, facing, carry, dishes, clients):
    order = [facing] + [d for d in SCAN if d != facing]
    cells = [(p[0] + d[0], p[1] + d[1]) for d in order]
    if carry is not None:
        for q in cells:
            if clients.get(q) == carry:
                return "serve", q
        for q in cells:
            if q in clients:
                return "refuse", q
        return None, None
    for q in cells:
        if q in dishes:
            return "take", q
    return None, None


LEVEL_DATA = [parse(rows) for rows in MAPS]


def budget_of(index):
    return 2 * len(SOLUTIONS[index]) + 10


def camera_of(lv, p):
    cx = min(max(0, p[0] - VIEW_W // 2 + 1), max(0, lv["w"] - VIEW_W))
    cy = min(max(0, p[1] - VIEW_H // 2), max(0, lv["h"] - VIEW_H))
    return cx, cy


def ring(block, edge, centre):
    block[0, 1:3] = edge
    block[3, 1:3] = edge
    block[1:3, 0] = edge
    block[1:3, 3] = edge
    block[1:3, 1:3] = centre


def render(lv, p, facing, carry, dishes, clients, waiter=WAITER, tint=None, spark=None):
    px = np.full((64, 64), OUTSIDE, dtype=np.int8)
    cx, cy = camera_of(lv, p)
    for ty in range(VIEW_H):
        for tx in range(VIEW_W):
            wx, wy = cx + tx, cy + ty
            if wx >= lv["w"] or wy >= lv["h"]:
                continue
            q = (wx, wy)
            block = px[HUD_H + ty * TILE:HUD_H + ty * TILE + TILE, tx * TILE:tx * TILE + TILE]
            if q in lv["walls"]:
                block[:] = WALL
                continue
            if q in lv["tables"]:
                block[:] = TABLE
                continue
            if q in lv["counters"]:
                block[:] = COUNTER
                block[1:3, 1:3] = DISH[dishes[q]] if q in dishes else COUNTER_TOP
                if spark == q:
                    block[1:3, 1:3] = SPARK
                continue
            block[:] = FLOOR_A if (wx + wy) % 2 == 0 else FLOOR_B
            if q in lv["rugs"]:
                block[:] = RUG
                block[1, 1] = RUG_DOT
                block[2, 2] = RUG_DOT
            if q in clients:
                ring(block, tint[1] if tint and tint[0] == q else CLIENT, SPARK if spark == q else DISH[clients[q]])
            if q == p:
                ring(block, waiter, DISH[carry] if carry else waiter)
                if facing == (0, -1):
                    block[0, 1:3] = FACE
                elif facing == (0, 1):
                    block[3, 1:3] = FACE
                elif facing == (-1, 0):
                    block[1:3, 0] = FACE
                else:
                    block[1:3, 3] = FACE
    return px


def draw_hud(px, carry, tickets, left, budget):
    px[:HUD_H, :] = HUD_BG
    px[1:7, 1:7] = FRAME
    px[2:6, 2:6] = DISH[carry] if carry else HUD_BG
    for k, (kind, done) in enumerate(tickets):
        x = 9 + 4 * k
        px[1:4, x:x + 3] = SERVED if done else DISH[kind]
    n = (left * 54 + budget - 1) // budget
    px[5:7, 9:9 + n] = BAR


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), OUTSIDE, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(MAPS))]


class Ng15(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._lv = LEVEL_DATA[0]
        self._pos = (0, 0)
        self._facing = (0, 1)
        self._carry = None
        self._dishes = {}
        self._clients = {}
        self._order = []
        self._left = 0
        self._budget = 0
        self._frames = None
        self._at = 0
        self._finish = None
        super().__init__(
            game_id="ng15-7d0f15e2",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=OUTSIDE, letter_box=OUTSIDE),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = self._current_level_index
        self._lv = LEVEL_DATA[index]
        self._pos = self._lv["start"]
        self._facing = (0, 1)
        self._carry = None
        self._dishes = dict(self._lv["dishes"])
        self._clients = dict(self._lv["clients"])
        self._order = sorted(self._lv["clients"], key=lambda q: (q[1], q[0]))
        self._budget = budget_of(index)
        self._left = self._budget
        self._frames = None
        self._at = 0
        self._finish = None
        self._show(self._frame(), level)

    def _show(self, px, level=None):
        level = level or self.current_level
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _frame(self, waiter=WAITER, tint=None, spark=None):
        px = render(self._lv, self._pos, self._facing, self._carry, self._dishes, self._clients, waiter, tint, spark)
        tickets = [(self._lv["clients"][q], q not in self._clients) for q in self._order]
        draw_hud(px, self._carry, tickets, self._left, self._budget)
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
        if action not in MOVES and action != GameAction.ACTION5:
            return [self._frame()]
        self._left -= 1
        if action in MOVES:
            self._facing = MOVES[action]
            frames = []
            for q in slide(self._lv, self._pos, self._facing, self._clients):
                self._pos = q
                frames.append(self._frame())
            if not frames:
                frames.append(self._frame())
        else:
            frames = self._use()
        if not self._clients:
            self._finish = "next"
        elif self._left <= 0:
            frames.append(self._frame(waiter=TIRED))
            self._finish = "lose"
        return frames

    def _use(self):
        kind, q = interact(self._pos, self._facing, self._carry, self._dishes, self._clients)
        if kind == "take":
            frames = [self._frame(spark=q)]
            self._carry = self._dishes.pop(q)
            return frames + [self._frame()]
        if kind == "serve":
            self._carry = None
            frames = [self._frame(spark=q), self._frame(tint=(q, HAPPY))]
            del self._clients[q]
            return frames + [self._frame()]
        if kind == "refuse":
            return [self._frame(tint=(q, ANGRY)), self._frame()]
        return [self._frame()]
