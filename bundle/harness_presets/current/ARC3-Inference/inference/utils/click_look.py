import os

from inference.utils.grid_utils import ARC_COLOR_CHARS, ARC_COLOR_LEGEND

LOOK_ENV = "ARC3_CLICK_LOOK"
SHORT_ENV = "ARC3_CLICK_LOOK_SHORT"
LIMIT = 10
NAMES = dict(part.split("=", 1) for part in ARC_COLOR_LEGEND.replace(" ", "").split(","))


def on() -> bool:
    return os.environ.get(LOOK_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def short() -> bool:
    return os.environ.get(SHORT_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def look_short(grid, row: int, col: int) -> str:
    def char(r: int, c: int) -> str:
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            return ARC_COLOR_CHARS[max(0, min(15, int(grid[r][c])))]
        return "."
    if not grid or not (0 <= row < len(grid)) or not (0 <= col < len(grid[row])):
        return "@off-board"
    cells = [char(r, c) for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)]
    if len(set(cells)) == 1:
        return f"@{char(row, col)} [all {cells[0]}]"
    return f"@{char(row, col)} [{''.join(cells[0:3])}/{''.join(cells[3:6])}/{''.join(cells[6:9])}]"


def look(grid, row: int, col: int) -> str:
    def char(r: int, c: int) -> str:
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            return ARC_COLOR_CHARS[max(0, min(15, int(grid[r][c])))]
        return "."
    if not grid or not (0 <= row < len(grid)) or not (0 <= col < len(grid[row])):
        return f"clicked ({row},{col}) is outside the board"
    here = char(row, col)
    block = "/".join("".join(char(r, c) for c in range(col - 1, col + 2)) for r in range(row - 1, row + 2))
    return (f"clicked cell was {here} ({NAMES.get(here, here)}); 3x3 before the click, rows {row - 1}-{row + 1} x "
            f"cols {col - 1}-{col + 1}: {block}")
