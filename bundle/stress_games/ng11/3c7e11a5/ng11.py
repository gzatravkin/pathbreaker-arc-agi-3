import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
GRASS = 14
SOIL = 13
FURROW = 4
ROCK = 3
ROCK_LIGHT = 2
COIN = 11
TRACK = 3
MARK = 0
TIME = 10
SLOT = 3
STALL = 2
LEAF = 14
DRY = 3
DRY_DARK = 4
SACK = 13
SKIN = 0
HAT = 12
BODY = 9
LEGS = 4
TOOL = 8
SHOP_STRIPE = 8
MARKET_STRIPE = 9
STRIPE_LIGHT = 0
CELL = 8
FIELD_Y = 16
BAR_X = 2
BAR_MAX = 60

WHEAT = 11
CARROT = 12
BERRY = 15

MOVES = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def crop(colour, cost, grow, prices, step=1, wither=None):
    return dict(colour=colour, cost=cost, grow=grow, prices=tuple(prices), step=step, wither=wither)


CONFIGS = [
    dict(grid=("a.....#A",
               "......#.",
               "..PP....",
               "........",
               "...@....",
               "........"),
         crops=[crop(WHEAT, 2, 8, [5])], start=4, target=14, bag=4, budget=160),
    dict(grid=("a..#....",
               "...#.PP.",
               "...#.PP.",
               "........",
               "...@..#.",
               "......#A"),
         crops=[crop(WHEAT, 1, 10, [4])], start=3, target=24, bag=4, budget=260),
    dict(grid=("a......A",
               "b...#..B",
               "..PP#...",
               "..PP....",
               "....@...",
               "........"),
         crops=[crop(WHEAT, 1, 6, [3]), crop(CARROT, 3, 14, [9])], start=5, target=36, bag=4, budget=280),
    dict(grid=("P.#....a",
               "P.#....A",
               "..#..#..",
               "..#..#..",
               "P....#.@",
               "P....#.."),
         crops=[crop(BERRY, 3, 10, [8], wither=3)], start=6, target=36, bag=2, budget=300),
    dict(grid=("a..#...A",
               "b..#...B",
               "........",
               ".PP##PP.",
               "........",
               "...@#..."),
         crops=[crop(WHEAT, 1, 6, [3]), crop(CARROT, 2, 10, [3, 4, 6, 9, 12, 9, 6, 4], step=3)],
         start=4, target=60, bag=4, budget=200),
    dict(grid=("a.#.P..A",
               "b.#.P..B",
               "c.#....C",
               "..##.#..",
               "...@.#PP",
               "P....#.."),
         crops=[crop(WHEAT, 1, 5, [2, 3, 4, 3], step=5, wither=4),
                crop(CARROT, 2, 9, [5, 7, 9, 7, 5, 3], step=4, wither=3),
                crop(BERRY, 5, 14, [10, 14, 16, 12], step=6, wither=3)],
         start=5, target=58, bag=3, budget=330),
]

SOLUTIONS = [
    "ULLUUULSSDDRSURRDSLDSRDSSSRDRRRUSDLLUULLLLSSDDRSURRDSLDSRDSSSRDRRRUS",
    "ULUUULSSSDDDRRRURSURSURRDSSSSSSSSSSSLDSLDDRSDRRRDSULLLLLUUULSSSSDDDRRRURSURSURRDSRDDLSSSSSSSSSSSULSULLDSLDDRSDRRRDS",
    "LLLUUUULSSSSDDRSURRDSURRDDDLSDLLUSLUURSURRDSURRDDDLSDLLUSRRURUUURSLLLDLLSSSSDRSURRDSURRDDDLSDLLUSLUURSURRDSURRDDDLSDLLUSRRURUURS",
    "UULUURSSDLLDDDLLLUUULSULSDLSSSSSULSDDDDRRUUURRRSURSSDLLDDDLLLUUULSULSDLSSSSSULSDDDDRRUUURRRSURSSDLLDDDLLLUUULSULSDLSSSSSULSDDDDRRUUURRRS",
    "ULLLUUSSDRSURRDSLDSSSRDSSSRRURRRRRRRSDLLLLULSSSSDSRDSRRRDSRDSLLLLLDSRDSRRRDSRDSURRRRS",
    "LLUUULSSDDDRRRUULUURSDRSURSSSSDRSDRRURRRRRRRRRRRRRRRRRRRSDLLDDLLLUULSSSDDRRRUULUURSDRSDRRRDSUUULSDLSDRDSSSURSLLDDLLLUUULSSDDDRRRUULUURSDRSURSSSSDRSDRRURS",
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

SHOPS = "abc"
MARKETS = "ABC"


def stage_of(spec, planted, tick):
    age = tick - planted
    grow = spec["grow"]
    if spec["wither"] is not None and age >= grow + spec["wither"]:
        return 4
    if age >= grow:
        return 3
    return (3 * age) // grow


def price_of(spec, tick):
    prices = spec["prices"]
    return prices[(tick // spec["step"]) % len(prices)]


class Ng11(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._cells = {}
        self._plots = {}
        self._pos = (0, 0)
        self._facing = (0, 1)
        self._coins = 0
        self._bag = []
        self._tick = 0
        self._left = 0
        super().__init__(
            game_id="ng11-3c7e11a5",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._cells = {}
        self._plots = {}
        for y, row in enumerate(self._cfg["grid"]):
            for x, ch in enumerate(row):
                if ch == "@":
                    self._pos = (x, y)
                    ch = "."
                self._cells[(x, y)] = ch
                if ch == "P":
                    self._plots[(x, y)] = None
        self._facing = (0, 1)
        self._coins = self._cfg["start"]
        self._bag = []
        self._tick = 0
        self._left = self._cfg["budget"]
        self._draw(level)

    def _walkable(self, cell):
        return self._cells.get(cell) == "."

    def _interact(self):
        cfg = self._cfg
        crops = cfg["crops"]
        cell = (self._pos[0] + self._facing[0], self._pos[1] + self._facing[1])
        ch = self._cells.get(cell)
        if ch == "P":
            state = self._plots[cell]
            if state is None:
                for i, item in enumerate(self._bag):
                    if item[0] == "seed":
                        self._bag.pop(i)
                        self._plots[cell] = (item[1], self._tick)
                        break
                return
            kind, planted = state
            stage = stage_of(crops[kind], planted, self._tick)
            if stage == 4:
                self._plots[cell] = None
            elif stage == 3 and len(self._bag) < cfg["bag"]:
                self._bag.append(("crop", kind))
                self._plots[cell] = None
        elif ch is not None and ch in SHOPS:
            kind = SHOPS.index(ch)
            spec = crops[kind]
            if self._coins >= spec["cost"] and len(self._bag) < cfg["bag"]:
                self._coins -= spec["cost"]
                self._bag.append(("seed", kind))
        elif ch is not None and ch in MARKETS:
            kind = MARKETS.index(ch)
            sold = [item for item in self._bag if item == ("crop", kind)]
            if sold:
                self._coins += len(sold) * price_of(crops[kind], self._tick)
                self._bag = [item for item in self._bag if item != ("crop", kind)]

    def step(self) -> None:
        aid = self._action.id
        if aid in MOVES:
            dx, dy = MOVES[aid]
            self._facing = (dx, dy)
            dest = (self._pos[0] + dx, self._pos[1] + dy)
            if self._walkable(dest):
                self._pos = dest
        elif aid == GameAction.ACTION5:
            self._interact()
        else:
            self.complete_action()
            return
        self._tick += 1
        self._left = max(0, self._left - 1)
        self._draw()
        if self._coins >= self._cfg["target"]:
            self.next_level()
        elif self._left == 0:
            self.lose()
        self.complete_action()

    def _draw_bar(self, px, y, length, colour):
        length = max(0, min(BAR_MAX, length))
        px[y, BAR_X:BAR_X + length] = colour

    def _draw_price(self, px, ox, oy, value):
        first = min(CELL, value)
        px[oy + 6, ox:ox + first] = COIN
        if value > CELL:
            px[oy + 7, ox:ox + min(CELL, value - CELL)] = COIN

    def _draw_plant(self, px, ox, oy, stage, colour):
        if stage == 0:
            px[oy + 5, ox + 3:ox + 5] = colour
            return
        if stage == 4:
            px[oy + 5, ox + 3] = DRY_DARK
            px[oy + 4, ox + 3] = DRY
            px[oy + 3, ox + 2] = DRY
            px[oy + 4, ox + 4] = DRY_DARK
            px[oy + 5, ox + 5] = DRY
            return
        px[oy + 4:oy + 6, ox + 3] = LEAF
        if stage >= 2:
            px[oy + 4:oy + 6, ox + 4] = LEAF
            px[oy + 3, ox + 2:ox + 6] = LEAF
        if stage == 3:
            px[oy + 1:oy + 3, ox + 3:ox + 5] = colour
            px[oy + 2, ox + 2] = colour
            px[oy + 2, ox + 5] = colour

    def _draw_cell(self, px, cell, ch):
        ox, oy = cell[0] * CELL, FIELD_Y + cell[1] * CELL
        crops = self._cfg["crops"]
        if ch == "#":
            px[oy:oy + CELL, ox:ox + CELL] = ROCK
            px[oy + 1:oy + 3, ox + 2:ox + 5] = ROCK_LIGHT
            px[oy + 7, ox:ox + CELL] = FURROW
            return
        if ch == "P":
            px[oy:oy + CELL, ox:ox + CELL] = SOIL
            px[oy + 7, ox + 1:ox + 7] = FURROW
            px[oy + 1:oy + 7, ox] = FURROW
            state = self._plots[cell]
            if state is not None:
                kind, planted = state
                self._draw_plant(px, ox, oy, stage_of(crops[kind], planted, self._tick), crops[kind]["colour"])
            return
        if ch in SHOPS or ch in MARKETS:
            shop = ch in SHOPS
            kind = (SHOPS if shop else MARKETS).index(ch)
            spec = crops[kind]
            stripe = SHOP_STRIPE if shop else MARKET_STRIPE
            px[oy:oy + CELL, ox:ox + CELL] = STALL
            for i in range(CELL):
                px[oy:oy + 2, ox + i] = stripe if i % 2 == 0 else STRIPE_LIGHT
            if shop:
                px[oy + 3:oy + 5, ox + 2:ox + 6] = SACK
                px[oy + 3, ox + 3:ox + 5] = spec["colour"]
                self._draw_price(px, ox, oy, spec["cost"])
            else:
                px[oy + 3:oy + 5, ox + 3:ox + 5] = spec["colour"]
                px[oy + 4, ox + 2] = spec["colour"]
                px[oy + 4, ox + 5] = spec["colour"]
                px[oy + 2, ox + 4] = LEAF
                self._draw_price(px, ox, oy, price_of(spec, self._tick))
            return
        px[oy:oy + CELL, ox:ox + CELL] = GRASS

    def _draw_farmer(self, px):
        ox, oy = self._pos[0] * CELL, FIELD_Y + self._pos[1] * CELL
        px[oy + 1, ox + 2:ox + 6] = HAT
        px[oy + 2:oy + 4, ox + 3:ox + 5] = SKIN
        px[oy + 4:oy + 6, ox + 2:ox + 6] = BODY
        px[oy + 6, ox + 3] = LEGS
        px[oy + 6, ox + 4] = LEGS
        fx, fy = self._facing
        if fy < 0:
            px[oy, ox + 3:ox + 5] = TOOL
        elif fy > 0:
            px[oy + 7, ox + 3:ox + 5] = TOOL
        elif fx < 0:
            px[oy + 4:oy + 6, ox] = TOOL
        else:
            px[oy + 4:oy + 6, ox + 7] = TOOL

    def _draw_bag(self, px):
        crops = self._cfg["crops"]
        for i in range(self._cfg["bag"]):
            ox, oy = BAR_X + i * 8, 8
            px[oy:oy + 7, ox:ox + 7] = SLOT
            px[oy + 1:oy + 6, ox + 1:ox + 6] = BACKGROUND
            if i >= len(self._bag):
                continue
            what, kind = self._bag[i]
            colour = crops[kind]["colour"]
            if what == "seed":
                px[oy + 2:oy + 5, ox + 2:ox + 5] = SACK
                px[oy + 3, ox + 3] = colour
            else:
                px[oy + 2:oy + 5, ox + 2:ox + 5] = colour
                px[oy + 1, ox + 3] = LEAF

    def _draw(self, level=None):
        level = level or self.current_level
        cfg = self._cfg
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        target = cfg["target"]
        self._draw_bar(px, 1, target, TRACK)
        self._draw_bar(px, 2, target, TRACK)
        self._draw_bar(px, 1, self._coins, COIN)
        self._draw_bar(px, 2, self._coins, COIN)
        px[0:4, BAR_X + target] = MARK
        self._draw_bar(px, 5, (self._left * BAR_MAX + cfg["budget"] - 1) // cfg["budget"], TIME)
        self._draw_bag(px)
        for cell, ch in self._cells.items():
            self._draw_cell(px, cell, ch)
        self._draw_farmer(px)
        level.get_sprites_by_name("canvas")[0].pixels = px
