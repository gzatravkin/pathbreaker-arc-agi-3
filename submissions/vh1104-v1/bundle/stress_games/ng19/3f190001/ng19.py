import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
SLOT_EMPTY = 3
MARK = 14
LIT = 0
DIM = 3
COLOURS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
COLS = 5
BLOCK = 10
BLOCK_PITCH = 12
BOARD_X = 2
BOARD_Y = 34
SLOT_X = 3
SLOT_Y = 10
SLOT_PITCH = 7
SLOT_SIZE = 5

LEVEL_SPECS = [
    dict(k=4, mode="colour", reverse=False, seed=11),
    dict(k=5, mode="colour", reverse=False, seed=23),
    dict(k=6, mode="colour", reverse=False, seed=37),
    dict(k=6, mode="colour", reverse=True, seed=41),
    dict(k=8, mode="colour", reverse=True, seed=53),
    dict(k=6, mode="position", reverse=False, seed=67),
]


def lcg(s):
    return (s * 1103515245 + 12345) & 0x7FFFFFFF


def layout_of(spec):
    s = lcg(spec["seed"])
    order = list(range(len(COLOURS)))
    for i in range(len(order) - 1, 0, -1):
        s = lcg(s)
        j = (s >> 8) % (i + 1)
        order[i], order[j] = order[j], order[i]
    seq = []
    for _ in range(spec["k"]):
        s = lcg(s)
        seq.append((s >> 8) % len(COLOURS))
    return order, seq


def required_of(spec):
    order, seq = layout_of(spec)
    return list(reversed(seq)) if spec["reverse"] else list(seq)


def block_rect(slot):
    r, c = divmod(slot, COLS)
    x = BOARD_X + c * BLOCK_PITCH
    y = BOARD_Y + r * BLOCK_PITCH
    return x, y


def canvas_level(index):
    sprite = Sprite(
        pixels=np.full((64, 64), BACKGROUND, dtype=np.int8),
        name="canvas", x=0, y=0, layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        interaction=InteractionMode.TANGIBLE,
        tags=["canvas"],
    )
    return Level(sprites=[sprite], grid_size=(64, 64), name="Level %d" % (index + 1))


LEVELS = [canvas_level(i) for i in range(len(LEVEL_SPECS))]


class Ng19(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._spec = LEVEL_SPECS[0]
        self._order = []
        self._seq = []
        self._entered = []
        self._phase = "show"
        self._tick = 0
        super().__init__(
            game_id="ng19-3f190001",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[6],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._spec = LEVEL_SPECS[self._current_level_index]
        self._order, self._seq = layout_of(self._spec)
        self._entered = []
        self._phase = "show"
        self._tick = 0
        self._draw(level)

    def _show_length(self):
        k = self._spec["k"]
        if self._spec["mode"] == "position":
            return 3 * k
        return max(4, k)

    def _slot_x(self, i):
        return SLOT_X + i * SLOT_PITCH

    def _draw(self, level=None):
        level = level or self.current_level
        spec = self._spec
        k = spec["k"]
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        position = spec["mode"] == "position"
        lit_block = None
        if self._phase == "show" and position:
            t = self._tick - 1
            if t >= 0 and t % 3 < 2 and t // 3 < k:
                lit_block = self._seq[t // 3]
        for slot, colour_index in enumerate(self._order):
            x, y = block_rect(slot)
            colour = COLOURS[colour_index]
            if position:
                colour = LIT if colour_index == lit_block else DIM
            px[y:y + BLOCK, x:x + BLOCK] = colour
        for i in range(k):
            x = self._slot_x(i)
            px[SLOT_Y:SLOT_Y + SLOT_SIZE, x:x + SLOT_SIZE] = SLOT_EMPTY
            if self._phase == "show" and not position and self._tick > 0:
                px[SLOT_Y:SLOT_Y + SLOT_SIZE, x:x + SLOT_SIZE] = COLOURS[self._seq[i]]
        for n, colour_index in enumerate(self._entered):
            i = k - 1 - n if spec["reverse"] else n
            x = self._slot_x(i)
            px[SLOT_Y:SLOT_Y + SLOT_SIZE, x:x + SLOT_SIZE] = 11 if position else COLOURS[colour_index]
        if spec["reverse"]:
            x = self._slot_x(k - 1) + SLOT_SIZE + 1
        else:
            x = self._slot_x(0) - 2
        px[SLOT_Y - 2:SLOT_Y + SLOT_SIZE + 2, x:x + 1] = MARK
        level.get_sprites_by_name("canvas")[0].pixels = px

    def _block_at(self, x, y):
        for slot, colour_index in enumerate(self._order):
            bx, by = block_rect(slot)
            if bx <= x < bx + BLOCK and by <= y < by + BLOCK:
                return colour_index
        return None

    def step(self) -> None:
        if self._phase == "show":
            self._tick += 1
            if self._tick > self._show_length():
                self._phase = "play"
                self._tick = 0
                self._draw()
                self.complete_action()
                return
            self._draw()
            return
        if self._action.id == GameAction.ACTION6:
            data = self._action.data or {}
            hit = self._block_at(int(data.get("x", -1)), int(data.get("y", -1)))
            if hit is not None:
                self._entered.append(hit)
                self._draw()
                if len(self._entered) == self._spec["k"]:
                    if self._entered == required_of(self._spec):
                        last = self.is_last_level()
                        self.next_level()
                        if last:
                            self.complete_action()
                        return
                    self.lose()
        self.complete_action()


def solve(index):
    spec = LEVEL_SPECS[index]
    order, seq = layout_of(spec)
    out = []
    for colour_index in required_of(spec):
        x, y = block_rect(order.index(colour_index))
        out.append((x + 4, y + 4))
    return out


SOLUTIONS = [solve(i) for i in range(len(LEVEL_SPECS))]
