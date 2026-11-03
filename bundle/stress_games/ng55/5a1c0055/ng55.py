import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

BACKGROUND = 5
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
LENGTH = 6
NEEDED = 3
STRIKES = 3
LETTER_ACTIONS = {
    GameAction.ACTION1: "U",
    GameAction.ACTION2: "D",
    GameAction.ACTION3: "L",
    GameAction.ACTION4: "R",
}
LETTER_COLOUR = {"U": 11, "D": 9, "L": 14, "R": 8}
SHAPES = {
    "U": ((0, 1, 0), (1, 1, 1), (0, 0, 0)),
    "D": ((0, 0, 0), (1, 1, 1), (0, 1, 0)),
    "L": ((0, 1, 0), (1, 1, 0), (0, 1, 0)),
    "R": ((0, 1, 0), (0, 1, 1), (0, 1, 0)),
}
ACCEPT_X = 2
REJECT_X = 34
LOG_Y = 4
LOG_PITCH = 5
INPUT_X = 8
INPUT_Y = 49


def palindrome(w):
    return w == w[::-1]


def mirrored_opposite(w):
    return all(w[i] == OPPOSITE[w[LENGTH - 1 - i]] for i in range(LENGTH))


PAL_YES = ["DLRRLD", "URLLRU", "LUDDUL", "RDUUDR"]
ANTI_YES = ["UDRLUD", "DULRDU", "LRUDLR", "RLDURL"]

CONFIGS = [
    dict(rule=palindrome, yes=PAL_YES, no=["UUURRD", "UUDRUL", "URDLUR", "LUDULU"]),
    dict(rule=palindrome, yes=PAL_YES, no=["ULDLUU", "RLDURR", "UUURRD", "UUDRUL"]),
    dict(rule=palindrome, yes=PAL_YES, no=["DUDRRU", "DDLDDL", "ULDLUU", "UUURRD"]),
    dict(rule=mirrored_opposite, yes=ANTI_YES, no=["UUDRUL", "ULDLUU", "URDLUR", "LUDULU"]),
    dict(rule=mirrored_opposite, yes=ANTI_YES, no=["UUURRD", "DUDRRU", "UUDRUL", "ULDLUU"]),
    dict(rule=mirrored_opposite, yes=ANTI_YES, no=["URDLDU", "UDLRRL", "UUURRD", "UUDRUL"]),
]

SOLVE_WORDS = [
    ["UULLUU", "RDDDDR", "LURRUL"],
    ["UULLUU", "RDDDDR", "LURRUL"],
    ["UULLUU", "RDDDDR", "LURRUL"],
    ["UURLDD", "LLUDRR", "DRLRLU"],
    ["UURLDD", "LLUDRR", "DRLRLU"],
    ["UURLDD", "LLUDRR", "DRLRLU"],
]
SOLUTIONS = [[ch for word in words for ch in word + "S"] for words in SOLVE_WORDS]


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


def stamp(px, letter, x, y, scale):
    shape = SHAPES[letter]
    for r in range(3):
        for c in range(3):
            if shape[r][c]:
                px[y + r * scale:y + (r + 1) * scale, x + c * scale:x + (c + 1) * scale] = LETTER_COLOUR[letter]


class Ng55(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._cfg = CONFIGS[0]
        self._buffer = ""
        self._yes = []
        self._no = []
        self._won = []
        self._strikes = 0
        super().__init__(
            game_id="ng55-5a1c0055",
            levels=LEVELS,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            available_actions=[1, 2, 3, 4, 5, 7],
            win_score=len(LEVELS),
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._cfg = CONFIGS[self._current_level_index]
        self._buffer = ""
        self._yes = list(self._cfg["yes"])
        self._no = list(self._cfg["no"])
        self._won = []
        self._strikes = 0
        self._draw(level)

    def _draw(self, level=None):
        level = level or self.current_level
        px = np.full((64, 64), BACKGROUND, dtype=np.int8)
        px[1, ACCEPT_X:ACCEPT_X + 26] = 14
        px[1, REJECT_X:REJECT_X + 26] = 8
        for x0, words in ((ACCEPT_X, self._yes), (REJECT_X, self._no)):
            for k, word in enumerate(words[:8]):
                for i, letter in enumerate(word):
                    stamp(px, letter, x0 + 4 * i, LOG_Y + LOG_PITCH * k, 1)
        px[46, 4:60] = 3
        for i, letter in enumerate(self._buffer):
            stamp(px, letter, INPUT_X + 8 * i, INPUT_Y, 2)
        for i in range(LENGTH):
            if i >= len(self._buffer):
                px[INPUT_Y + 5, INPUT_X + 8 * i:INPUT_X + 8 * i + 6] = 3
        for s in range(NEEDED):
            px[60:63, 2 + s * 5:5 + s * 5] = 14 if s < len(self._won) else 3
        for s in range(STRIKES - self._strikes):
            px[60:63, 46 + s * 5:49 + s * 5] = 8
        level.get_sprites_by_name("canvas")[0].pixels = px

    def step(self) -> None:
        aid = self._action.id
        letter = LETTER_ACTIONS.get(aid)
        if letter is not None:
            if len(self._buffer) < LENGTH:
                self._buffer += letter
        elif aid == GameAction.ACTION7:
            self._buffer = self._buffer[:-1]
        elif aid == GameAction.ACTION5 and len(self._buffer) == LENGTH:
            word = self._buffer
            self._buffer = ""
            if word in self._won:
                pass
            elif self._cfg["rule"](word):
                self._won.append(word)
                self._yes.append(word)
            else:
                self._strikes += 1
                self._no.append(word)
            self._draw()
            if len(self._won) >= NEEDED:
                self.next_level()
            elif self._strikes >= STRIKES:
                self.lose()
        self._draw()
        self.complete_action()
