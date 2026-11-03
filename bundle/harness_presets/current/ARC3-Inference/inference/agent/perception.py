import base64
import io
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np
from PIL import Image, ImageDraw

DELTA_ENV = "ARC3_PERCEPTION_DELTA"
DIGEST_ENV = "ARC3_PERCEPTION_DIGEST"
CATALOG_ENV = "ARC3_PERCEPTION_CATALOG"
LEVEL_ENV = "ARC3_PERCEPTION_LEVEL"
MAP_ENV = "ARC3_PERCEPTION_MAP_TEXT"
SCENE_ENV = "ARC3_PERCEPTION_SCENE"
SHORT_DELTA_ENV = "ARC3_PERCEPTION_SHORT_DELTA"
IMAGE_SCALE_ENV = "ARC3_PERCEPTION_IMAGE_SCALE"
SHAPE_MAX = 10
SHAPE_LINES = 120
PATCH_ROWS = 10
PATCH_COLS = 24
PATCHES = 3
PATCH_TOKENS = 300
MAP_HEAD = ("map of the whole board as text = {grid}: line r<row> is {row} exactly (a cell is {cell}); uniform rows "
            "are folded, column numbers every 10 on top. The map and the shapes are exact, so there is no need to "
            "print the board:")
SHAPES_HEAD = "shapes of the types above, in the catalog's turn (. = not part of the object){cells}:"
SHAPE_CELLS = "; the cells of an object of type T<n> are current_frame.objects['T<n>'][<i>]['cells']"
PATCH_HEAD = "board before | after, rows {r0}-{r1}, cols {c0}-{c1} (each side is row[{c0}:{c2}]):"
BEFORE_HEAD = ("before = the board at step {s0}, {before}; after = {grid}. The changed places, "
               "side by side:")
ROWS_HEAD = "board after, the changed rows in full as {row}:"
CHARS = "WwgGcBMPRbSYOrNp"
NAMES = ["white", "light gray", "gray", "dark gray", "charcoal", "black", "magenta", "pink", "red", "blue", "sky blue",
         "yellow", "orange", "dark red", "light green", "purple"]
RGB = [(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0), (229, 58, 163),
       (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0), (255, 133, 27), (146, 18, 49),
       (79, 204, 48), (163, 86, 214)]
MAX_SPRITE = 10
DIGEST_HEAD = ("Board digest - a heuristic read of the current board by the harness, not ground truth; "
               "check anything you rely on in python.")
DELTA_HEAD = "Board change since your previous turn - a heuristic read by the harness; check it in python."
SHOW_DELTA_HEAD = "Board change of this action call - a heuristic read by the harness:"
DELTA_IMAGE_LINE = ("The picture below draws, as in the sprite catalog, the objects that look different after this "
                    "turn's actions: changed or turned ones before and after, new types, and types that are gone.")
CATALOG_LINE = ("The picture after this digest is the sprite catalog: each object type drawn alone under its name, "
                "at the scale noted (1:k means k cells per drawn cell).")
CAMERA_REACH = 24
CATALOG_WIDTH = 1024
CATALOG_TILES = 40


_ARMS = []


def env(name, default=""):
    if not _ARMS:
        try:
            from inference.agent import knob_arms
        except ImportError:
            knob_arms = None
        _ARMS.append(knob_arms)
    return str(_ARMS[0].env(name, default) if _ARMS[0] is not None else os.environ.get(name, default))


def flag(name):
    return env(name, "").strip() == "1"


def image_scale():
    try:
        return min(1.0, max(0.25, float(env(IMAGE_SCALE_ENV, "") or 1.0)))
    except ValueError:
        return 1.0


def any_on():
    return flag(DELTA_ENV) or flag(DIGEST_ENV)


def name_of(index):
    return f"T{index + 1}"


class Registry:
    def __init__(self):
        self.names = {}
        self.rows = {}

    def known(self, key):
        return key in self.names

    def add(self, key, rows):
        if key not in self.names:
            self.names[key] = name_of(len(self.names))
            self.rows[key] = rows
        return self.names[key]


def components(grid):
    h, w = len(grid), len(grid[0])
    comp = [[-1] * w for _ in range(h)]
    out = []
    for sr in range(h):
        for sc in range(w):
            if comp[sr][sc] != -1:
                continue
            v = grid[sr][sc]
            cid = len(out)
            cells, stack = [], [(sr, sc)]
            comp[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < h and 0 <= nc < w and comp[nr][nc] == -1 and grid[nr][nc] == v:
                        comp[nr][nc] = cid
                        stack.append((nr, nc))
            out.append({"color": v, "cells": cells, "box": bbox(cells)})
    return out, comp


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def rows_of(grid, cells):
    r0, r1, c0, c1 = bbox(cells)
    s = set(cells)
    return ["".join(CHARS[grid[r][c]] if (r, c) in s else "." for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1)]


def rot90(rows):
    return ["".join(rows[len(rows) - 1 - j][i] for j in range(len(rows))) for i in range(len(rows[0]))]


def canon(rows):
    variants, cur = [], rows
    for _ in range(4):
        variants.append("\n".join(cur))
        cur = rot90(cur)
    return min(variants)


def mirror(key):
    return canon([row[::-1] for row in key.split("\n")])


def shrink(rows, k):
    h, w = len(rows), len(rows[0])
    if h % k or w % k:
        return None
    out = []
    for r in range(0, h, k):
        line = ""
        for c in range(0, w, k):
            block = [rows[r + i][c + j] for i in range(k) for j in range(k)]
            top, n = Counter(block).most_common(1)[0]
            if n < k * k:
                return None
            line += top
        out.append(line)
    return out


def turns_between(base, rows):
    cur = base
    for k in range(4):
        if cur == rows:
            return k
        cur = rot90(cur)
    return None


def colour_of(ch):
    return NAMES[CHARS.index(ch)]


def symmetry(grid):
    h, w = len(grid), len(grid[0])
    found = []
    for label, pairs in (("left-right mirror", [((r, c), (r, w - 1 - c)) for r in range(h) for c in range(w // 2)]),
                         ("top-bottom mirror", [((r, c), (h - 1 - r, c)) for r in range(h // 2) for c in range(w)]),
                         ("180-degree turn", [((r, c), (h - 1 - r, w - 1 - c)) for r in range(h // 2) for c in range(w)])):
        if not pairs:
            continue
        joint = Counter((grid[a[0]][a[1]], grid[b[0]][b[1]]) for a, b in pairs)
        mapping, used = {}, set()
        for (x, y), n in joint.most_common():
            if x not in mapping and y not in used:
                mapping[x] = y
                used.add(y)
        hit = sum(n for (x, y), n in joint.items() if mapping.get(x) == y)
        share = hit / len(pairs)
        if share >= 0.85:
            label = label if share >= 0.97 else "near " + label
            swaps = sorted({tuple(sorted((NAMES[x], NAMES[y]))) for x, y in mapping.items() if x != y and joint[(x, y)] > 8})
            found.append(f"{label} {round(100 * share)}% of cells" + "".join(f", {a}<->{b}" for a, b in swaps))
    return found


def layout_map(grid):
    h, w = len(grid), len(grid[0])
    for k in (4, 3, 2):
        if h % k or w % k:
            continue
        rows, bad = [], 0
        for r in range(0, h, k):
            line = ""
            for c in range(0, w, k):
                block = Counter(grid[r + i][c + j] for i in range(k) for j in range(k))
                top, n = block.most_common(1)[0]
                bad += n < k * k
                line += CHARS[top]
            rows.append(line)
        if bad <= 0.08 * len(rows) * len(rows[0]):
            return rows, k
    return None, 1


def lattice(inst, idx):
    rows_start = sorted({inst[i]["box"][0] for i in idx})
    cols_start = sorted({inst[i]["box"][2] for i in idx})
    if len(rows_start) * len(cols_start) > 2 * len(idx) or len(rows_start) < 2 or len(cols_start) < 2:
        return None
    cell = {(inst[i]["box"][0], inst[i]["box"][2]): inst[i] for i in idx}
    mat = [" ".join("." if (r, c) not in cell else cell[(r, c)]["label"] for c in cols_start) for r in rows_start]
    return len(rows_start), len(cols_start), mat


def at(x):
    return f"{x['label']}@{x['box'][0]},{x['box'][2]}"


def kinds_of(grid, comps, comp, bg):
    h, w = len(grid), len(grid[0])
    kinds = {}
    for i, x in enumerate(comps):
        r0, r1, c0, c1 = x["box"]
        hh, ww, n = r1 - r0 + 1, c1 - c0 + 1, len(x["cells"])
        if x["color"] == bg:
            kinds[i] = "bg"
        elif (r0 >= h - 2 or r1 <= 1) and hh <= 2 and ww >= 8:
            kinds[i] = "hbar"
        elif (c0 >= w - 2 or c1 <= 1) and ww <= 2 and hh >= 8:
            kinds[i] = "vbar"
        elif n >= 5 and max(hh, ww) >= 8 and n <= 1.6 * max(hh, ww) and min(hh, ww) <= max(hh, ww) / 3:
            kinds[i] = "line"
        elif n > 4 * MAX_SPRITE * MAX_SPRITE // 3 or hh > 2 * MAX_SPRITE or ww > 2 * MAX_SPRITE:
            edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
            kinds[i] = "region" if edge or n > 0.25 * h * w or hh * ww > 0.5 * h * w else "big"
        else:
            kinds[i] = "part"
    texture = defaultdict(Counter)
    for i, x in enumerate(comps):
        if kinds[i] != "part" or len(x["cells"]) > 2:
            continue
        around = set()
        for r, c in x["cells"]:
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1), (r + 1, c + 1), (r - 1, c - 1),
                           (r + 1, c - 1), (r - 1, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and comp[nr][nc] != i:
                    around.add(comp[nr][nc])
        if len(around) == 1:
            j = next(iter(around))
            if kinds[j] in ("region", "bg") and len(comps[j]["cells"]) >= 60:
                kinds[i] = "texture"
                texture[j][x["color"]] += 1
    for i, x in enumerate(comps):
        if kinds[i] != "texture":
            continue
        owner = next((j for j in texture if texture[j][x["color"]] and any(
            comp[r + dr][c + dc] == j for r, c in x["cells"] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < h and 0 <= c + dc < w)), None)
        if owner is None or texture[owner][x["color"]] < 6:
            kinds[i] = "part"
    for j in list(texture):
        for col in [col for col, n in texture[j].items() if n < 6]:
            del texture[j][col]
    return kinds, texture


def clusters_of(grid, comps, comp, kinds):
    h, w = len(grid), len(grid[0])
    parent = list(range(len(comps)))
    boxes = {i: comps[i]["box"] for i in range(len(comps))}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in range(h):
        for c in range(w):
            a = comp[r][c]
            if kinds[a] != "part":
                continue
            for nr, nc in ((r + 1, c), (r, c + 1), (r + 1, c + 1), (r + 1, c - 1)):
                if nr < h and 0 <= nc < w:
                    b = comp[nr][nc]
                    if b == a or kinds[b] != "part":
                        continue
                    ra, rb = find(a), find(b)
                    if ra == rb:
                        continue
                    ba, bb = boxes[ra], boxes[rb]
                    m = (min(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), max(ba[3], bb[3]))
                    if m[1] - m[0] < MAX_SPRITE and m[3] - m[2] < MAX_SPRITE:
                        parent[rb] = ra
                        boxes[ra] = m
    clusters = defaultdict(list)
    for i in range(len(comps)):
        if kinds[i] == "part":
            clusters[find(i)].extend(comps[i]["cells"])
        elif kinds[i] == "big":
            clusters[("big", i)].extend(comps[i]["cells"])
    return list(clusters.values())


def relations(order, shown, name, by_key, inst):
    rel = []
    form = {k: canon(shown[k]) for k in order}
    by_form = {form[k]: k for k in order}
    by_shape = defaultdict(list)
    for k in order:
        by_shape[canon(["".join("#" if ch != "." else "." for ch in row) for row in shown[k]])].append(k)
    for sk, ks in by_shape.items():
        if len(ks) >= 2 and sk.count("#") >= 3 and "." in sk:
            rel.append("same shape, different colours: " + ", ".join(name[k] for k in ks))
    for k in order:
        mk = by_form.get(mirror(form[k]))
        if mk is not None and mk != k and name[mk] > name[k]:
            rel.append(f"{name[mk]} is a mirror of {name[k]}")
    for a in order:
        for b in order:
            if a == b:
                continue
            rb = form[b].split("\n")
            for f in (2, 3, 4, 5):
                small = shrink(rb, f)
                if small is not None and canon(small) == form[a]:
                    rel.append(f"{name[b]} is {name[a]} scaled x{f}")
                    break
    for x in inst:
        cols = {ch for row in x["rows"] for ch in row if ch != "."}
        if len(cols) < 2 or by_key[x["key"]] > 1:
            continue
        parts = []
        for ch in sorted(cols):
            prow = ["".join(c if c == ch else "." for c in row) for row in x["rows"]]
            prow = [r for r in prow if r.strip(".")]
            lo = min(len(r) - len(r.lstrip(".")) for r in prow)
            hi = max(len(r.rstrip(".")) for r in prow)
            pk = canon([r[lo:hi] for r in prow])
            for k in order:
                if k == x["key"]:
                    continue
                rows_k = form[k].split("\n")
                if pk == form[k] or any((small := shrink(rows_k, f)) is not None and canon(small) == pk for f in (2, 3, 4)):
                    parts.append(f"{colour_of(ch)} part looks like {name[k]}")
                    break
        if len(parts) >= 2:
            rel.append(f"{x['type']} = " + " + ".join(parts))
    return rel


def neighbour_groups(inst):
    used, groups = set(), []
    idx = sorted(range(len(inst)), key=lambda i: inst[i]["box"])
    for i in idx:
        if i in used:
            continue
        g, stack = [], [i]
        used.add(i)
        while stack:
            a = stack.pop()
            g.append(a)
            ra = inst[a]["box"]
            for b in idx:
                if b in used:
                    continue
                rb = inst[b]["box"]
                dr = max(rb[0] - ra[1], ra[0] - rb[1]) - 1
                dc = max(rb[2] - ra[3], ra[2] - rb[3]) - 1
                same = abs((ra[1] - ra[0]) - (rb[1] - rb[0])) <= 1 and abs((ra[3] - ra[2]) - (rb[3] - rb[2])) <= 1
                if dr <= 2 and dc <= 2 and same:
                    used.add(b)
                    stack.append(b)
        groups.append(g)
    return groups


def key_of(rows, n):
    hh, ww = len(rows), len(rows[0])
    if n > 4 * MAX_SPRITE * MAX_SPRITE // 3 or hh > 2 * MAX_SPRITE or ww > 2 * MAX_SPRITE:
        colours = "".join(sorted({ch for row in rows for ch in row if ch != "."}))
        return f"big {colours} {min(hh, ww)}x{max(hh, ww)}"
    return canon(rows)


def is_big(x):
    return (x["n"] > 4 * MAX_SPRITE * MAX_SPRITE // 3 or x["box"][1] - x["box"][0] >= 2 * MAX_SPRITE
            or x["box"][3] - x["box"][2] >= 2 * MAX_SPRITE)


def digest(grid, registry):
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    comps, comp = components(grid)
    border = Counter(grid[r][c] for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1))
    area = Counter(grid[r][c] for r in range(h) for c in range(w))
    bg = max(area, key=lambda v: area[v] + 2 * border[v])
    kinds, texture = kinds_of(grid, comps, comp, bg)
    inst = []
    for cells in clusters_of(grid, comps, comp, kinds):
        rows = rows_of(grid, cells)
        inst.append({"key": key_of(rows, len(cells)), "box": bbox(cells), "n": len(cells), "rows": rows,
                     "cells": [tuple(one) for one in cells]})
    by_key = Counter(x["key"] for x in inst)
    first_rows = {}
    for x in sorted(inst, key=lambda y: y["box"]):
        first_rows.setdefault(x["key"], x["rows"])
    fresh = [k for k in sorted(by_key, key=lambda k: (-by_key[k], -len(k), k)) if not registry.known(k)]
    for k in fresh:
        registry.add(k, first_rows[k])
    name = {k: registry.names[k] for k in by_key}
    shown = {k: registry.rows[k] for k in by_key}
    order = sorted(by_key, key=lambda k: (-by_key[k], len(name[k]), name[k]))
    for x in inst:
        x["type"] = name[x["key"]]
        k = turns_between(shown[x["key"]], x["rows"])
        x["label"] = x["type"] + (f"r{k * 90}" if k else "")
    out = []
    bg_tex = Counter()
    for i in range(len(comps)):
        if kinds[i] == "bg":
            bg_tex.update(texture[i])
    out.append(f"background {NAMES[bg]}" + "".join(f" with {NAMES[col]} dot texture ({n})" for col, n in bg_tex.most_common()))
    bars = {}
    for i, x in enumerate(comps):
        if kinds[i] in ("hbar", "vbar"):
            r0, r1, c0, c1 = x["box"]
            length = c1 - c0 + 1 if kinds[i] == "hbar" else r1 - r0 + 1
            where = f"row {r0}" if kinds[i] == "hbar" else f"col {c0}"
            start = c0 if kinds[i] == "hbar" else r0
            bars[(NAMES[x["color"]], where)] = bars.get((NAMES[x["color"]], where), 0) + length
            out.append(f"bar: {NAMES[x['color']]} {length} cells at {where} from {start}")
    for i, x in enumerate(comps):
        if kinds[i] == "region":
            r0, r1, c0, c1 = x["box"]
            fill = len(x["cells"]) / ((r1 - r0 + 1) * (c1 - c0 + 1))
            shape = "solid block" if fill > 0.9 else "walls/outline"
            tex = "".join(f" with {NAMES[col]} dot texture ({n})" for col, n in texture[i].most_common())
            out.append(f"region: {NAMES[x['color']]} {shape} r{r0}-{r1} c{c0}-{c1} ({len(x['cells'])} cells){tex}")
    cell_owner = {}
    for j, y in enumerate(inst):
        for r in range(y["box"][0], y["box"][1] + 1):
            for c in range(y["box"][2], y["box"][3] + 1):
                cell_owner.setdefault((r, c), j)
    for i, x in enumerate(comps):
        if kinds[i] != "line":
            continue
        r0, r1, c0, c1 = x["box"]
        touch = set()
        for r, c in x["cells"]:
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < h and 0 <= nc < w and comp[nr][nc] != i and kinds[comp[nr][nc]] == "part":
                    j = cell_owner.get((nr, nc))
                    if j is not None:
                        touch.add(at(inst[j]))
        out.append(f"line: {NAMES[x['color']]} r{r0}-{r1} c{c0}-{c1}" + (f" touching {', '.join(sorted(touch))}" if touch else ""))
    grid_lines, in_grid = [], {}
    for g in neighbour_groups(inst):
        lat = lattice(inst, g) if len(g) >= 4 else None
        if not lat:
            continue
        nr_, nc_, mat = lat
        gid = f"grid{len(grid_lines) + 1}"
        rb = min(inst[i]["box"][0] for i in g), max(inst[i]["box"][1] for i in g)
        cb = min(inst[i]["box"][2] for i in g), max(inst[i]["box"][3] for i in g)
        grid_lines.append(f"{gid} r{rb[0]}-{rb[1]} c{cb[0]}-{cb[1]} {nr_}x{nc_}: " + " / ".join(mat))
        for i in g:
            in_grid[i] = gid
    table = ["objects: type | colours | size | n | at row,col (rN: turned N degrees clockwise from the catalog)"]
    for k in order:
        rows = shown[k]
        colors = Counter(ch for row in rows for ch in row if ch != ".")
        members = sorted((i for i, x in enumerate(inst) if x["key"] == k), key=lambda i: inst[i]["box"])
        grids = sorted({in_grid[i] for i in members if i in in_grid})
        spots = []
        for i in members:
            if i in in_grid:
                continue
            x = inst[i]
            spots.append(f"{x['box'][0]},{x['box'][2]}" + x["label"][len(x["type"]):])
        where = " ".join(spots[:24]) + (f" +{len(spots) - 24} more" if len(spots) > 24 else "")
        if grids:
            where = (where + " " if where else "") + "in " + ",".join(grids)
        table.append(f"{name[k]} | {'+'.join(colour_of(ch) for ch, _ in colors.most_common(3))} | "
                     f"{len(rows)}x{len(rows[0])} | {by_key[k]} | {where}")
    if len(table) > 1:
        out += table
        if flag(MAP_ENV):
            out += shape_lines([(name[k], shown[k]) for k in order])
    out += grid_lines
    rel = relations(order, shown, name, by_key, inst)
    if rel:
        out.append("relations: " + "; ".join(rel))
    bigs = [j for j, x in enumerate(inst) if is_big(x)]
    big_set = set(bigs)
    big_cell = {}
    for j in bigs:
        y = inst[j]
        for r in range(y["box"][0], y["box"][1] + 1):
            for c in range(y["box"][2], y["box"][3] + 1):
                big_cell.setdefault((r, c), j)
    attach, inside = [], defaultdict(list)
    for i, x in enumerate(inst):
        if i in big_set:
            continue
        r0, r1, c0, c1 = x["box"]
        holder = next((j for j in bigs if inst[j]["box"][0] <= r0 and r1 <= inst[j]["box"][1]
                       and inst[j]["box"][2] <= c0 and c1 <= inst[j]["box"][3]), None)
        if holder is not None:
            inside[holder].append(i)
            continue
        hits = set()
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                j = big_cell.get((r, c))
                if j is not None and CHARS[grid[r][c]] in "".join(inst[j]["rows"]):
                    hits.add(at(inst[j]))
        if hits:
            attach.append(f"{at(x)} touches {', '.join(sorted(hits))}")
    if attach:
        out.append("attached: " + "; ".join(attach))
    for holder, members in inside.items():
        grids_in = sorted({in_grid[i] for i in members if i in in_grid})
        loose = [at(inst[i]) for i in members if i not in in_grid]
        out.append(f"inside {at(inst[holder])}: " + ", ".join(grids_in + loose))
    for line in symmetry(grid):
        out.append("symmetry: " + line)
    pairs_txt = []
    for t in sorted({x["type"] for x in inst}):
        group = [x for x in inst if x["type"] == t]
        if len(group) != 2:
            continue
        a, b = group
        if abs(a["box"][0] - b["box"][0]) <= 1 and abs((w - 1 - a["box"][3]) - b["box"][2]) <= 1:
            pairs_txt.append(f"{at(a)} and {at(b)} sit at mirrored places left/right")
        elif abs(a["box"][2] - b["box"][2]) <= 1 and abs((h - 1 - a["box"][1]) - b["box"][0]) <= 1:
            pairs_txt.append(f"{at(a)} and {at(b)} sit at mirrored places top/bottom")
    if pairs_txt:
        out.append("mirrored pairs: " + "; ".join(pairs_txt[:6]))
    tiles = []
    region_cells = sum(len(comps[i]["cells"]) for i in range(len(comps)) if kinds[i] == "region")
    if region_cells > 0.3 * h * w:
        lm, k = layout_map(grid)
        if lm is not None:
            out.append(f"layout: the whole board at 1:{k} is the first picture of the catalog")
            tiles.append((f"map 1:{k}", lm, True))
    tiles += [(name[k], shown[k], False) for k in order]
    counts = {name[k]: by_key[k] for k in order}
    areas = [{"kind": kinds[i], "colour": CHARS[x["color"]], "cells": [tuple(one) for one in x["cells"]], "box": x["box"]}
             for i, x in enumerate(comps) if kinds[i] in ("hbar", "vbar", "region", "line")]
    return {"lines": out, "tiles": tiles, "inst": inst, "bars": bars, "counts": counts, "fresh": [name[k] for k in fresh],
            "shape": (h, w), "areas": areas}


def downscale(rows):
    for k in (4, 3, 2):
        small = shrink(rows, k)
        if small is not None:
            return small, k
    return rows, 1


def catalog(tiles, px=8, pad=4):
    px = max(2, round(px * image_scale()))
    drawn = []
    for label, rows, is_map in tiles[:CATALOG_TILES]:
        rows, k = (rows, 1) if is_map else downscale(rows)
        drawn.append((label + (f" 1:{k}" if k > 1 else ""), [r[:24] for r in rows[:24]]))
    if not drawn:
        return None
    placed, x, y, line_h, width = [], pad, 0, 0, pad
    for label, rows in drawn:
        tw = max(len(rows[0]) * px, 6 * len(label)) + pad + 10
        th = len(rows) * px + 16
        if x + tw > CATALOG_WIDTH * image_scale() and x > pad:
            x, y = pad, y + line_h
            line_h = 0
        placed.append((x, y, label, rows))
        x += tw
        width = max(width, x)
        line_h = max(line_h, th)
    im = Image.new("RGB", (max(width, 20), y + line_h + 2), (40, 44, 52))
    d = ImageDraw.Draw(im)
    for x0, y0, label, rows in placed:
        d.text((x0, y0 + 1), label, fill=(255, 255, 255))
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                col = ((120, 124, 132) if (r + c) % 2 else (88, 92, 100)) if ch == "." else RGB[CHARS.index(ch)]
                d.rectangle([x0 + c * px, y0 + 14 + r * px, x0 + (c + 1) * px - 1, y0 + 14 + (r + 1) * px - 1], fill=col)
    return im


def image_part(im):
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def camera(prev_grid, new_grid, reach=CAMERA_REACH):
    a0, b0 = np.asarray(prev_grid, dtype=np.int16), np.asarray(new_grid, dtype=np.int16)
    if a0.shape != b0.shape:
        return None
    rows, cols = a0.shape
    base = float((a0 == b0).mean())
    if base >= 0.85:
        return None
    best = (0.0, 0, 0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dy == 0 and dx == 0:
                continue
            a = a0[max(0, -dy):rows + min(0, -dy), max(0, -dx):cols + min(0, -dx)]
            b = b0[max(0, dy):rows + min(0, dy), max(0, dx):cols + min(0, dx)]
            if a.size < a0.size // 2:
                continue
            same = float((a == b).mean())
            if same > best[0] + 1e-9 or (same > best[0] - 1e-9 and abs(dy) + abs(dx) < abs(best[1]) + abs(best[2])):
                best = (same, dy, dx)
    same, dy, dx = best
    if same - base > 0.15 and same > 0.85:
        return dy, dx, same, base
    return None


def overlap(a, b):
    r0, r1 = max(a[0], b[0]), min(a[1], b[1])
    c0, c1 = max(a[2], b[2]), min(a[3], b[3])
    if r1 < r0 or c1 < c0:
        return 0.0
    inter = (r1 - r0 + 1) * (c1 - c0 + 1)
    sa = (a[1] - a[0] + 1) * (a[3] - a[2] + 1)
    sb = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
    return inter / min(sa, sb)


def alike(a, b):
    sa = (a[1] - a[0] + 1) * (a[3] - a[2] + 1)
    sb = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
    return max(sa, sb) <= 4 * min(sa, sb)


def delta(prev, new, prev_grid, new_grid):
    pi, ni = prev["inst"], new["inst"]
    h, w = len(new_grid), len(new_grid[0])
    lines = []
    used_p, used_n = set(), set()
    by_key_p, by_key_n = defaultdict(list), defaultdict(list)
    for i, x in enumerate(pi):
        by_key_p[x["key"]].append(i)
    for j, x in enumerate(ni):
        by_key_n[x["key"]].append(j)
    cam = camera(prev_grid, new_grid) if (len(prev_grid), len(prev_grid[0])) == (h, w) else None
    cy, cx = (cam[0], cam[1]) if cam else (0, 0)
    real = set()
    if cam:
        for r in range(h):
            for c in range(w):
                pr, pc = r - cy, c - cx
                if 0 <= pr < h and 0 <= pc < w and prev_grid[pr][pc] != new_grid[r][c]:
                    real.add((r, c))

    def touched(box, shift=False):
        if not cam:
            return True
        r0, r1, c0, c1 = box
        if shift:
            r0, r1, c0, c1 = r0 + cy, r1 + cy, c0 + cx, c1 + cx
        return any((r, c) in real for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))

    def dist(a, b):
        world = abs(a["box"][0] + cy - b["box"][0]) + abs(a["box"][2] + cx - b["box"][2])
        screen = abs(a["box"][0] - b["box"][0]) + abs(a["box"][2] - b["box"][2])
        return min(world, screen)

    pairs = []
    for key in sorted(set(by_key_p) & set(by_key_n)):
        cand = sorted((dist(pi[i], ni[j]), i, j) for i in by_key_p[key] for j in by_key_n[key])
        for d, i, j in cand:
            if i in used_p or j in used_n or d > 24:
                continue
            used_p.add(i)
            used_n.add(j)
            pairs.append((i, j, d))
    moved, rotated = [], []
    for i, j, d in sorted(pairs, key=lambda p: ni[p[1]]["box"]):
        a, b = pi[i], ni[j]
        if cam and not touched(b["box"]):
            continue
        if d:
            dr, dc = b["box"][0] - a["box"][0], b["box"][2] - a["box"][2]
            if cam and abs(dr - cy) + abs(dc - cx) < abs(dr) + abs(dc):
                moved.append(f"{at(b)} moved {dr - cy:+d} rows {dc - cx:+d} cols in the world (on screen also the scroll)")
            else:
                moved.append(f"{at(b)} moved {dr:+d} rows {dc:+d} cols")
        k = turns_between(a["rows"], b["rows"])
        if k:
            rotated.append(f"{at(b)} turned {k * 90} degrees clockwise")
    changed = []
    for j, b in enumerate(ni):
        if j in used_n:
            continue
        best = max(((overlap(a["box"], b["box"]), i) for i, a in enumerate(pi)
                    if i not in used_p and alike(a["box"], b["box"])), default=(0, None))
        if best[0] >= 0.5 and cam and not touched(b["box"]):
            used_p.add(best[1])
            used_n.add(j)
            continue
        if best[0] >= 0.5:
            a = pi[best[1]]
            used_p.add(best[1])
            used_n.add(j)
            sa = canon(["".join("#" if ch != "." else "." for ch in r) for r in a["rows"]])
            sb = canon(["".join("#" if ch != "." else "." for ch in r) for r in b["rows"]])
            what = "recoloured" if sa == sb else "changed shape"
            was = "+".join(sorted({colour_of(ch) for r in a["rows"] for ch in r if ch != "."}))
            changed.append(f"{at(b)} {what} (was {a['type']}, {was} {len(a['rows'])}x{len(a['rows'][0])})")
    if cam:
        still = sum(1 for _, _, d in pairs if d == 0)
        lines.append(f"camera: the view scrolled {cy:+d} rows {cx:+d} cols ({round(100 * cam[2])}% of the board matches "
                     f"shifted, {round(100 * cam[3])}% unshifted); {still} object(s) stayed put on screen")
        edge = [j for j, b in enumerate(ni) if j not in used_n and (
            (cy > 0 and b["box"][0] < cy) or (cy < 0 and b["box"][1] >= h + cy)
            or (cx > 0 and b["box"][2] < cx) or (cx < 0 and b["box"][3] >= w + cx))]
        if edge:
            lines.append("scrolled into view: " + " ".join(at(ni[j]) for j in edge[:10]))
            used_n.update(edge)
        out_edge = [i for i, a in enumerate(pi) if i not in used_p and (
            (cy > 0 and a["box"][1] >= h - cy) or (cy < 0 and a["box"][0] < -cy)
            or (cx > 0 and a["box"][3] >= w - cx) or (cx < 0 and a["box"][2] < -cx))]
        if out_edge:
            lines.append("scrolled out of view: " + " ".join(f"{pi[i]['type']}@{pi[i]['box'][0]},{pi[i]['box'][2]}"
                                                             for i in out_edge[:10]))
            used_p.update(out_edge)
        came = [at(b) for j, b in enumerate(ni) if j not in used_n and touched(b["box"])]
        gone = [f"{a['type']}@{a['box'][0]},{a['box'][2]}" for i, a in enumerate(pi) if i not in used_p and touched(a["box"], True)]
        if real:
            rr = [r for r, _ in real]
            cc = [c for _, c in real]
            lines.append(f"after the scroll {len(real)} cells really changed, within r{min(rr)}-{max(rr)} c{min(cc)}-{max(cc)}")
        else:
            lines.append("after the scroll nothing else changed")
    else:
        came = [at(b) for j, b in enumerate(ni) if j not in used_n]
        gone = [f"{a['type']}@{a['box'][0]},{a['box'][2]}" for i, a in enumerate(pi) if i not in used_p]
    if moved:
        lines.append("moved: " + "; ".join(moved[:8]) + (f"; +{len(moved) - 8} more" if len(moved) > 8 else ""))
    if rotated:
        lines.append("turned: " + "; ".join(rotated[:6]) + (f"; +{len(rotated) - 6} more" if len(rotated) > 6 else ""))
    if changed:
        lines.append("changed: " + "; ".join(changed[:6]) + (f"; +{len(changed) - 6} more" if len(changed) > 6 else ""))
    if gone:
        lines.append("gone: " + " ".join(gone[:10]) + (f" +{len(gone) - 10} more" if len(gone) > 10 else ""))
    if came:
        lines.append("new: " + " ".join(came[:10]) + (f" +{len(came) - 10} more" if len(came) > 10 else ""))
    pb, nb = prev["bars"], new["bars"]
    for k in sorted(set(pb) | set(nb)):
        if pb.get(k) != nb.get(k):
            lines.append(f"bar {k[0]} at {k[1]}: {pb.get(k, 0)} -> {nb.get(k, 0)} cells")
    shape_same = (len(prev_grid), len(prev_grid[0])) == (h, w)
    cells = sum(1 for r in range(h) for c in range(w) if prev_grid[r][c] != new_grid[r][c]) if shape_same else h * w
    if not lines:
        lines.append(f"no object changed ({cells} cells differ outside objects)")
    else:
        lines.append(f"{cells} cells differ in all")
    return lines


def level_lines(new, before):
    out = []
    if new["fresh"]:
        out.append("new on this level, never seen earlier in this game: " + ", ".join(new["fresh"]))
    if before is not None:
        changes = []
        for t, n in new["counts"].items():
            if t in before["counts"] and before["counts"][t] != n:
                changes.append(f"{t} {before['counts'][t]} -> {n}")
        for t, n in before["counts"].items():
            if t not in new["counts"]:
                changes.append(f"{t} gone (was {n})")
        if changes:
            out.append("count by type against the last board of the previous level: " + "; ".join(changes))
    return out


TRACK_STEPS = 40
HISTORY_ROWS = 40
JUMP = 24
JUMP_NOTE = " (jumped: the same type left one place and came at another in one step, as by a wrap)"
SHIFT_REACH = 4
DELTA_IMAGE_ENV = "ARC3_PERCEPTION_DELTA_IMAGE"
DELTA_IMAGE_TILES = 8
DELTA_IMAGE_WIDTH = 360
DELTA_IMAGE_TOKENS = 150


def rotation_of(x):
    turn = x["label"][len(x["type"]):]
    return int(turn[1:]) if turn.startswith("r") and turn[1:].isdigit() else 0


def letters_of(x):
    return sorted({ch for row in x["rows"] for ch in row if ch != "."})


def cells_of(cells):
    return sorted([int(r), int(c)] for r, c in cells or ())


def grid_shift(a, b, reach=SHIFT_REACH):
    a0, b0 = np.asarray(a, dtype=np.int16), np.asarray(b, dtype=np.int16)
    if a0.shape != b0.shape or a0.size == 0:
        return 0, 0
    bg = int(np.bincount(a0.ravel().astype(np.int64)).argmax())
    rows, cols = a0.shape
    base, best, best_score = None, (0, 0), -1.0
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            pa = a0[max(0, -dy):rows + min(0, -dy), max(0, -dx):cols + min(0, -dx)]
            pb = b0[max(0, dy):rows + min(0, dy), max(0, dx):cols + min(0, dx)]
            mask = (pa != bg) | (pb != bg)
            total = int(mask.sum())
            if not total:
                continue
            score = int(((pa == pb) & mask).sum()) / total
            if (dy, dx) == (0, 0):
                base = score
            if score > best_score + 1e-9:
                best_score, best = score, (dy, dx)
    if base is None or best == (0, 0) or best_score < base + 0.1:
        return 0, 0
    return best


def chain_shift(before, chain):
    boards = [before, *[one for one in chain or () if one is not None and len(one)]]
    dy = dx = 0
    for a, b in zip(boards, boards[1:]):
        sy, sx = grid_shift(a, b)
        dy, dx = dy + sy, dx + sx
    return dy, dx


def scroll_of(pi, ni, prev_grid, new_grid, reach=32):
    same_shape = (len(prev_grid), len(prev_grid[0])) == (len(new_grid), len(new_grid[0]))
    cam = camera(prev_grid, new_grid) if same_shape else None
    if cam:
        return cam[0], cam[1]
    votes, by_key = {}, {}
    for x in ni:
        by_key.setdefault(x["key"], []).append(x)
    for a in pi:
        for b in by_key.get(a["key"], ()):
            move = (b["box"][0] - a["box"][0], b["box"][2] - a["box"][2])
            if abs(move[0]) + abs(move[1]) <= reach:
                votes[move] = votes.get(move, 0) + 1
    if not votes:
        return 0, 0
    move, count = max(votes.items(), key=lambda one: (one[1], -abs(one[0][0]) - abs(one[0][1])))
    if move != (0, 0) and count >= max(3, len(pi) // 3) and count > 3 * votes.get((0, 0), 0):
        return move
    return 0, 0


def match(prev, new, prev_grid, new_grid, hint=None):
    pi, ni = prev["inst"], new["inst"]
    cy, cx = hint if hint is not None else scroll_of(pi, ni, prev_grid, new_grid)
    used_p, out = set(), {}

    def size(x):
        return x["box"][1] - x["box"][0], x["box"][3] - x["box"][2]

    def shifted(b):
        return b["box"][0] - cy, b["box"][1] - cy, b["box"][2] - cx, b["box"][3] - cx

    def in_place(same_key):
        for j, b in enumerate(ni):
            if j in out:
                continue
            best = max(((overlap(a["box"], shifted(b)), -abs(a["n"] - b["n"]), i) for i, a in enumerate(pi)
                        if i not in used_p and (a["key"] == b["key"]) == same_key and size(a) == size(b)),
                       default=(0, 0, None))
            if best[0] >= 0.9:
                a = pi[best[2]]
                used_p.add(best[2])
                out[j] = (best[2], ("turned" if turns_between(a["rows"], b["rows"]) else "same") if same_key
                          else "changed")

    def dist(a, b):
        return abs(a["box"][0] + cy - b["box"][0]) + abs(a["box"][2] + cx - b["box"][2])

    def inside(x, grid):
        return (x["box"][0] > 0 and x["box"][2] > 0 and x["box"][1] < len(grid) - 1
                and x["box"][3] < len(grid[0]) - 1)

    def by_distance(most, whole=False):
        by_key_p, by_key_n = {}, {}
        for i, x in enumerate(pi):
            if i not in used_p and (not whole or inside(x, prev_grid)):
                by_key_p.setdefault(x["key"], []).append(i)
        for j, x in enumerate(ni):
            if j not in out and (not whole or inside(x, new_grid)):
                by_key_n.setdefault(x["key"], []).append(j)
        for key in sorted(set(by_key_p) & set(by_key_n)):
            for d, i, j in sorted((dist(pi[i], ni[j]), i, j) for i in by_key_p[key] for j in by_key_n[key]):
                if i in used_p or j in out or d > most:
                    continue
                used_p.add(i)
                out[j] = (i, "turned" if turns_between(pi[i]["rows"], ni[j]["rows"]) else ("moved" if d else "same"))

    in_place(True)
    by_distance(JUMP)
    in_place(False)
    for j, b in enumerate(ni):
        if j in out:
            continue
        best = max(((overlap(a["box"], shifted(b)), i) for i, a in enumerate(pi)
                    if i not in used_p and alike(a["box"], b["box"])), default=(0, None))
        if best[0] >= 0.5:
            used_p.add(best[1])
            out[j] = (best[1], "changed")
    if (cy, cx) == (0, 0):
        by_distance(float("inf"), whole=True)
    return out, [i for i in range(len(pi)) if i not in used_p], (cy, cx) if (cy, cx) != (0, 0) else None


def track_row(step, x, event, cam=None):
    r0, r1, c0, c1 = x["box"]
    row = {"step": step, "event": event, "bbox": [r0, c0, r1, c1], "colours": letters_of(x), "rotation": rotation_of(x)}
    if event in ("start", "appeared", "changed"):
        row["cells"] = cells_of(x.get("cells"))
    if cam:
        row["camera"] = list(cam)
    return row


def region_lines(first, last, cy=0, cx=0, most=4):
    before = [x for x in first.get("areas") or () if x["kind"] == "region"]
    after = [x for x in last.get("areas") or () if x["kind"] == "region"]
    used, out = set(), []
    for a in before:
        moved = (a["box"][0] + cy, a["box"][1] + cy, a["box"][2] + cx, a["box"][3] + cx)
        best = max(((overlap(moved, b["box"]), j) for j, b in enumerate(after)
                    if j not in used and b["colour"] == a["colour"]), default=(0, None))
        if best[1] is None or best[0] < 0.5:
            continue
        used.add(best[1])
        b = after[best[1]]
        if moved != tuple(b["box"]) or len(a["cells"]) != len(b["cells"]):
            r0, r1, c0, c1 = a["box"]
            s0, s1, d0, d1 = b["box"]
            out.append(f"region {colour_of(a['colour'])} rows {r0}-{r1} cols {c0}-{c1} -> rows {s0}-{s1} cols "
                       f"{d0}-{d1} ({len(a['cells'])} -> {len(b['cells'])} cells)")
    return out[:most] + ([f"+{len(out) - most} more region(s) changed"] if len(out) > most else [])


def note(item, row, most=HISTORY_ROWS):
    history = item["history"]
    history.append(row)
    if len(history) > most:
        del history[1:len(history) - most + 1]


def background_of(grid):
    h, w = len(grid), len(grid[0])
    border = Counter(grid[r][c] for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1))
    area = Counter(grid[r][c] for r in range(h) for c in range(w))
    return max(area, key=lambda v: area[v] + 2 * border[v])


def column_ruler(w, left):
    ruler = [" "] * w
    for c in range(0, w, 10):
        for i, ch in enumerate(str(c)):
            if c + i < w:
                ruler[c + i] = ch
    return " " * left + "".join(ruler).rstrip()


def code_refs():
    if os.environ.get("ARC3_FRAME_OBJECTS", "").strip() == "1" and os.environ.get("ARC3_NO_BOARD_PRINT", "").strip() == "1":
        return {"grid": "current_frame.color(<row>, <col>) cell by cell", "row": "the colour letters of that row",
                "cell": "current_frame.color(<row>, <col>)",
                "before": "next(h.frame for h in history if h.frame.step == {s0}).color(<row>, <col>)", "cells": SHAPE_CELLS}
    if os.environ.get("ARC3_FRAME_OBJECTS", "").strip() == "1":
        return {"grid": "current_frame.grid", "row": "current_frame.grid[<row>]", "cell": "current_frame.grid[<row>][<col>]",
                "before": "next(h.frame for h in history if h.frame.step == {s0}).grid", "cells": SHAPE_CELLS}
    return {"grid": "current_frame.ascii", "row": "current_frame.ascii.splitlines()[<row>]",
            "cell": "current_frame.ascii.splitlines()[<row>][<col>]",
            "before": "next(h.frame for h in history if h.frame.step == {s0}).ascii", "cells": ""}


def map_head():
    refs = code_refs()
    return MAP_HEAD.format(grid=refs["grid"], row=refs["row"], cell=refs["cell"])


def map_lines(grid):
    h, w = len(grid), len(grid[0])
    left = len(f"r{h - 1} ")
    out = [map_head(), column_ruler(w, left)]
    r = 0
    while r < h:
        if len(set(grid[r])) == 1:
            end = r
            while end + 1 < h and tuple(grid[end + 1]) == tuple(grid[r]):
                end += 1
            out.append((f"rows {r}-{end}" if end > r else f"row {r}") + f": all {CHARS[grid[r][0]]}")
            r = end + 1
            continue
        out.append(f"r{r}".ljust(left) + "".join(CHARS[v] for v in grid[r]))
        r += 1
    return out


def shape_lines(types, most=SHAPE_LINES):
    out, spent = [SHAPES_HEAD.format(cells=code_refs()["cells"])], 0
    for at, (label, rows) in enumerate(types):
        h, w = len(rows), len(rows[0])
        if h > SHAPE_MAX or w > SHAPE_MAX:
            out.append(f"{label}: {h}x{w}, too large to spell out here")
            continue
        if spent + h > most:
            out.append(f"+{len(types) - at} more type(s) not spelled out here")
            break
        if h == 1:
            out.append(f"{label}: {rows[0]}")
        else:
            out.append(f"{label}:")
            out += ["  " + row for row in rows]
        spent += h
    return out


def change_clusters(cells, reach=2):
    left, out = set(cells), []
    while left:
        seed = left.pop()
        group, stack = [seed], [seed]
        while stack:
            r, c = stack.pop()
            for dr in range(-reach, reach + 1):
                for dc in range(-reach, reach + 1):
                    one = (r + dr, c + dc)
                    if one in left:
                        left.remove(one)
                        group.append(one)
                        stack.append(one)
        out.append(group)
    return out


def merged_boxes(boxes):
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] > b[1] + 1 or b[0] > a[1] + 1 or a[2] > b[3] + 1 or b[2] > a[3] + 1:
                    continue
                box = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                if box[1] - box[0] + 1 <= PATCH_ROWS and box[3] - box[2] + 1 <= PATCH_COLS:
                    boxes[i] = box
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def patch_lines(a, b, first=None, last=None, most=PATCHES, budget=PATCH_TOKENS, before_step=None):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return []
    h, w = len(a), len(a[0])
    bars = set()
    for found in (first, last):
        for x in (found or {}).get("areas") or ():
            if x["kind"] in ("hbar", "vbar"):
                bars.update(tuple(one) for one in x["cells"])
    cells = [(r, c) for r in range(h) for c in range(w) if a[r][c] != b[r][c] and (r, c) not in bars]
    small, big_rows = [], set()
    for group in change_clusters(cells):
        r0, r1 = max(0, min(r for r, _ in group) - 1), min(h - 1, max(r for r, _ in group) + 1)
        c0, c1 = max(0, min(c for _, c in group) - 1), min(w - 1, max(c for _, c in group) + 1)
        if r1 - r0 + 1 <= PATCH_ROWS and c1 - c0 + 1 <= PATCH_COLS:
            small.append((r0, r1, c0, c1))
        else:
            big_rows.update(r for r, _ in group)
    small = merged_boxes(small)
    out, spent, chosen = [], 0, []
    for box in sorted(small, key=lambda box: (box[1] - box[0] + 1) * (box[3] - box[2] + 1))[:most]:
        cost = map_tokens(["x" * (2 * (box[3] - box[2] + 1) + 8)] * (box[1] - box[0] + 2))
        if spent + cost > budget:
            break
        chosen.append(box)
        spent += cost
    spent = 0
    refs = code_refs()
    if chosen:
        out.append(BEFORE_HEAD.format(s0=before_step, before=refs["before"].format(s0=before_step), grid=refs["grid"]))
    for r0, r1, c0, c1 in sorted(chosen):
        left = len(f"r{r1} ")
        block = [PATCH_HEAD.format(r0=r0, r1=r1, c0=c0, c1=c1, c2=c1 + 1)]
        for r in range(r0, r1 + 1):
            block.append(f"r{r}".ljust(left) + "".join(CHARS[a[r][c]] for c in range(c0, c1 + 1)) + " | "
                         + "".join(CHARS[b[r][c]] for c in range(c0, c1 + 1)))
        out += block
        spent += map_tokens(block)
    if big_rows:
        left = len(f"r{h - 1} ")
        block = [ROWS_HEAD.format(row=refs["row"]), column_ruler(w, left)]
        block += [f"r{r}".ljust(left) + "".join(CHARS[v] for v in b[r]) for r in sorted(big_rows)]
        if spent + map_tokens(block) <= budget:
            out += block
    return out


def map_tokens(lines):
    return len("\n".join(lines)) // 3


class Tracker:
    def __init__(self):
        self.level = None
        self.steps = []
        self.grids = {}
        self.digests = {}
        self.items = []
        self.live = {}
        self.events = {}
        self.hints = {}

    def feed(self, st, frames):
        for step, grid, level in frames:
            if self.steps and level == self.level and step <= self.steps[-1]:
                continue
            key, found = st.digest_of(grid)
            if level != self.level or not self.steps:
                self.begin(step, key, found, level)
            else:
                self.advance(step, key, found)

    def begin(self, step, key, found, level):
        self.level, self.steps, self.items, self.events, self.live = level, [step], [], {}, {}
        self.grids, self.digests = {step: key}, {step: found}
        for j, x in enumerate(found["inst"]):
            self.live[j] = len(self.items)
            self.items.append({"key": x["type"], "alive": True, "last": x, "history": [track_row(step, x, "start")]})

    def advance(self, step, key, found):
        prev = self.steps[-1]
        out, gone, cam = match(self.digests[prev], found, self.grids[prev], key, self.hints.get(step))
        rows, live = [], {}
        for j, x in enumerate(found["inst"]):
            if j in out and out[j][0] in self.live:
                tid = self.live[out[j][0]]
                before = self.items[tid]["last"]
                if out[j][1] != "same":
                    note(self.items[tid], track_row(step, x, out[j][1], cam))
                    rows.append((tid, out[j][1], before, x))
            else:
                tid = len(self.items)
                self.items.append({"key": x["type"], "alive": True, "last": x,
                                   "history": [track_row(step, x, "appeared", cam)]})
                rows.append((tid, "appeared", None, x))
            self.items[tid]["last"] = x
            live[j] = tid
        for i in gone:
            tid = self.live.get(i)
            if tid is None:
                continue
            item = self.items[tid]
            item["alive"] = False
            note(item, {"step": step, "event": "vanished", **({"camera": list(cam)} if cam else {})})
            rows.append((tid, "vanished", item["last"], None))
        self.live = live
        self.events[step] = {"rows": rows, "camera": cam}
        self.steps.append(step)
        self.grids[step], self.digests[step] = key, found
        for old in self.steps[:-TRACK_STEPS]:
            self.grids.pop(old, None)
            self.digests.pop(old, None)
            self.events.pop(old, None)
        self.steps = self.steps[-TRACK_STEPS:]

    def base_step(self, start):
        known = [s for s in self.steps if s <= start]
        return known[-1] if known else (self.steps[0] if self.steps else None)

    def changes(self, start, end):
        steps = [s for s in self.steps if start < s <= end and s in self.events]
        per, cy, cx = {}, 0, 0
        for s in steps:
            cam = self.events[s]["camera"]
            if cam:
                cy, cx = cy + cam[0], cx + cam[1]
            for tid, kind, before, after in self.events[s]["rows"]:
                one = per.setdefault(tid, {"kinds": [], "before": before, "after": after})
                one["kinds"].append(kind)
                one["after"] = after
                one["last"] = before
        return per, (cy, cx), steps

    def render(self, start, end):
        per, (cy, cx), steps = self.changes(start, end)
        if not steps:
            return []
        lines, moved, turned, changed, gone, came = [], [], [], [], [], []
        for tid, one in sorted(per.items()):
            kinds, before, after = one["kinds"], one["before"], one["after"]
            if after is None:
                last = one.get("last") or before
                if last is not None and "appeared" not in kinds:
                    gone.append(f"{last['type']}@{last['box'][0]},{last['box'][2]}")
                continue
            if "appeared" in kinds:
                came.append(at(after))
                continue
            if "changed" in kinds:
                sa = canon(["".join("#" if ch != "." else "." for ch in r) for r in before["rows"]])
                sb = canon(["".join("#" if ch != "." else "." for ch in r) for r in after["rows"]])
                what = "recoloured" if sa == sb else "changed shape"
                was = "+".join(sorted({colour_of(ch) for r in before["rows"] for ch in r if ch != "."}))
                changed.append(f"{at(after)} {what} (was {before['type']}, {was} {len(before['rows'])}x"
                               f"{len(before['rows'][0])})")
            if "moved" in kinds or "turned" in kinds:
                dr, dc = after["box"][0] - before["box"][0] - cy, after["box"][2] - before["box"][2] - cx
                if dr or dc:
                    how = "jumped" if abs(dr) + abs(dc) > JUMP else "moved"
                    moved.append(f"{at(after)} {how} {dr:+d} rows {dc:+d} cols" + (" in the world" if (cy or cx) else ""))
            if "turned" in kinds:
                k = turns_between(before["rows"], after["rows"])
                turned.append(f"{at(after)} turned {k * 90} degrees clockwise" if k else f"{at(after)} turned and back")
        if cy or cx:
            lines.append(f"camera: the view scrolled {cy:+d} rows {cx:+d} cols")
        moved.sort(key=lambda one: " jumped " not in one)
        for head, items, most in (("moved", moved, 8), ("turned", turned, 6), ("changed", changed, 6)):
            if items:
                lines.append(f"{head}: " + "; ".join(items[:most]) + (f"; +{len(items) - most} more" if len(items) > most else "")
                             + (JUMP_NOTE if head == "moved" and any(" jumped " in one for one in items[:most]) else ""))
        for head, items in (("gone", gone), ("new", came)):
            if items:
                lines.append(f"{head}: " + " ".join(items[:10]) + (f" +{len(items) - 10} more" if len(items) > 10 else ""))
        base = self.base_step(start)
        first, last = self.digests.get(base), self.digests.get(steps[-1])
        if first is not None and last is not None:
            pb, nb = first["bars"], last["bars"]
            for k in sorted(set(pb) | set(nb)):
                if pb.get(k) != nb.get(k):
                    lines.append(f"bar {k[0]} at {k[1]}: {pb.get(k, 0)} -> {nb.get(k, 0)} cells")
            lines.extend(region_lines(first, last, cy, cx))
            a, b = self.grids.get(base), self.grids.get(steps[-1])
            if flag(MAP_ENV) and a is not None and b is not None:
                lines.extend(patch_lines(a, b, first, last, before_step=base))
            if a is not None and b is not None and len(a) == len(b) and len(a[0]) == len(b[0]):
                cells = sum(1 for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c])
                lines.append(f"no object changed ({cells} cells differ outside objects)" if not lines
                             else f"{cells} cells differ in all")
        return lines

    def picture(self, start, end, known):
        per, _, steps = self.changes(start, end)
        tiles = []
        for tid, one in sorted(per.items()):
            kinds, before, after = one["kinds"], one["before"], one["after"]
            if "changed" in kinds and before is not None and after is not None:
                tiles.append((f"{before['type']}>{after['type']}", before["rows"]))
                tiles.append((after["type"], after["rows"]))
            elif "turned" in kinds and before is not None and after is not None:
                tiles.append((f"{before['label']}>", before["rows"]))
                tiles.append((after["label"], after["rows"]))
            elif "appeared" in kinds and after is not None and after["type"] not in known:
                tiles.append((f"new {after['type']}", after["rows"]))
                known.add(after["type"])
            elif after is None and before is not None and "appeared" not in kinds and not any(
                    other["last"]["type"] == before["type"] for other in self.items if other["alive"]):
                tiles.append((f"gone {before['type']}", before["rows"]))
        if not tiles:
            return None, 0
        tiles = tiles[:DELTA_IMAGE_TILES]
        im = catalog([(label, rows, False) for label, rows in tiles], px=6, pad=3)
        if im is None:
            return None, 0
        width, most = int(DELTA_IMAGE_WIDTH * image_scale()), int(DELTA_IMAGE_TOKENS * image_scale() ** 2)
        if im.width > width:
            scale = width / im.width
            im = im.resize((width, max(1, int(im.height * scale))))
        while vision_tokens(im) > most and im.width > 32:
            scale = (most / vision_tokens(im)) ** 0.5
            im = im.resize((max(32, int(im.width * scale)), max(1, int(im.height * scale))))
        return im, vision_tokens(im)


def vision_tokens(im):
    return (max(32, round(im.width / 32) * 32) // 32) * (max(32, round(im.height / 32) * 32) // 32)


def level_entries(frame, history, most=TRACK_STEPS):
    level = getattr(frame, "level", None)
    frames = []
    for one in reversed(list(history or ())):
        seen = getattr(one, "frame", None)
        if seen is None or not getattr(seen, "grid", None) or getattr(seen, "level", None) != level:
            break
        frames.append((int(getattr(seen, "step", 0) or 0), seen.grid, level))
        if len(frames) >= most:
            break
    frames.reverse()
    if frame is not None and getattr(frame, "grid", None) and (not frames or frames[-1][1] != frame.grid):
        frames.append((int(getattr(frame, "step", 0) or 0), frame.grid, level))
    return frames


def tracked(st, frame, history):
    st.tracker.feed(st, level_entries(frame, history))
    return st.tracker


def delta_reset(st, step):
    st.delta_from = st.turn_from = step
    st.shown = False


def tracked_delta(st, frame, history):
    tracker = tracked(st, frame, history)
    end = int(getattr(frame, "step", 0) or 0)
    start = st.delta_from if st.delta_from is not None else end
    turn = st.turn_from if st.turn_from is not None else start
    shown = st.shown
    delta_reset(st, end)
    lines = tracker.render(start, end) if start < end else []
    if lines and flag(SHORT_DELTA_ENV):
        lines = short_delta(lines)
    picture = None
    if (lines or shown) and turn < end and flag(DELTA_IMAGE_ENV):
        picture = tracker.picture(turn, end, st.pictured)
    return lines, picture


class State:
    def __init__(self):
        self.registry = Registry()
        self.last = None
        self.full_given = set()
        self.pending = None
        self.cache = {}
        self.said = set()
        self.ms = []
        self.tracker = Tracker()
        self.delta_from = None
        self.turn_from = None
        self.shown = False
        self.pictured = set()

    def digest_of(self, grid):
        key = tuple(tuple(row) for row in grid)
        hit = self.cache.get(key)
        if hit is None:
            started = time.perf_counter()
            hit = digest(key, self.registry)
            ms = (time.perf_counter() - started) * 1000
            self.ms.append(ms)
            if len(self.ms) > 200:
                del self.ms[:100]
            note_digest(ms)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[key] = hit
        return key, hit


def state_of(owner):
    st = getattr(owner, "_perception", None)
    if st is None:
        st = State()
        owner._perception = st
    return st


def full_text(d, reason, extra=(), picture=True, grid=None):
    lines = [f"{DIGEST_HEAD} ({reason})", *extra, *d["lines"]]
    if grid is not None and flag(MAP_ENV):
        lines += map_lines(grid)
    if picture and flag(CATALOG_ENV) and d["tiles"]:
        lines.append(CATALOG_LINE)
    return "\n".join(lines)


def label_index(label):
    value = 0
    for ch in label:
        value = value * 26 + (ord(ch) - 64)
    return value


def label_ranges(labels):
    labs = sorted({one for one in labels if re.fullmatch(r"[A-Z]+", one)}, key=label_index)
    rest = [one for one in labels if not re.fullmatch(r"[A-Z]+", one)]
    out, i = [], 0
    while i < len(labs):
        j = i
        while j + 1 < len(labs) and label_index(labs[j + 1]) == label_index(labs[j]) + 1:
            j += 1
        out.append(f"{labs[i]}–{labs[j]}" if j - i >= 2 else " ".join(labs[i:j + 1]))
        i = j + 1
    return " ".join(out + rest)


SCENE_POS = re.compile(r"(\d+,\d+)(r\d+)?")
SCENE_TURN = {"r90": "↻90", "r180": "↻180", "r270": "↻270"}
SCENE_BAR = re.compile(r"^bar: (.+?) (\d+) cells at (row|col) (\d+) from (\d+)$")
SCENE_INSIDE = re.compile(r"^inside (\S+?): (.*)$")


def scene(text):
    head, types, inside, other = [], OrderedDict(), [], []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s == CATALOG_LINE or s.startswith("objects: type"):
            continue
        if s.startswith("Board digest"):
            tag = re.search(r"\(([^()]*)\)\s*$", s)
            head.append("DIGEST" + (f" ({tag.group(1)})" if tag else ""))
        elif s.startswith("new on this level"):
            head.append("new types: " + label_ranges([x.strip() for x in s.split(":", 1)[1].split(",") if x.strip()]))
        elif s.startswith("count by type against the last board"):
            items = [x.strip() for x in s.split(":", 1)[1].split(";") if x.strip()]
            gone = [m.group(1) for x in items for m in [re.fullmatch(r"(\S+) gone \(was 1\)", x)] if m]
            rest = [re.sub(r" \(was (\d+)\)", r" was \1", x) for x in items
                    if not re.fullmatch(r"\S+ gone \(was 1\)", x)]
            head.append("vs last level: " + "; ".join((["gone " + label_ranges(gone)] if gone else []) + rest))
        elif s.startswith("background "):
            head.append("bg " + s[len("background "):])
        elif SCENE_BAR.match(s):
            m = SCENE_BAR.match(s)
            first, length = int(m.group(5)), int(m.group(2))
            across = "cols" if m.group(3) == "row" else "rows"
            head.append(f"bar {m.group(1)} {m.group(3)} {m.group(4)} {across} {first}-{first + length - 1}")
        elif re.match(r"^[A-Z]+ \| ", s) and len(s.split("|")) >= 5:
            cols = [c.strip() for c in s.split("|")]
            types[cols[0]] = {"desc": f"{cols[1].replace(' ', '')} {cols[2]}", "n": cols[3],
                              "pos": " | ".join(cols[4:])}
        elif SCENE_INSIDE.match(s):
            m = SCENE_INSIDE.match(s)
            inside.append((m.group(1), m.group(2)))
        else:
            other.append(s)
    used, holds = {}, []
    for container, body in inside:
        groups = OrderedDict()
        for label_turn, pos in re.findall(r"([A-Za-z]+\d*(?:r\d+)?)@(\d+,\d+)", body):
            found = re.match(r"[A-Z]+", label_turn)
            if not found:
                continue
            label = found.group(0)
            turn = label_turn[len(label):]
            used.setdefault(label, set()).add(pos)
            groups.setdefault(label, []).append(pos + SCENE_TURN.get(turn, turn))
        parts = [f"{label} {' '.join(ps)}" for label, ps in groups.items()] + re.findall(r"grid\d+", body)
        holds.append(f"{container.replace('@', ' ')} holds: " + "; ".join(parts))
    rows = []
    for label, t in types.items():
        free = [p + SCENE_TURN.get(r, r) for p, r in SCENE_POS.findall(t["pos"]) if p not in used.get(label, set())]
        tail = SCENE_POS.sub("", t["pos"]).strip()
        count = "" if t["n"] == "1" else f" ×{t['n']}"
        rows.append(f"{label} {t['desc']}{count}" + (": " + " ".join(free) if free else "") + (f" {tail}" if tail else ""))
    return "\n".join(head + (["types and free positions:"] + rows if rows else []) + holds + other)


SHORT_MOVE = re.compile(r"^(\S+@\d+,\d+) (moved|jumped) ([+-]\d+) rows ([+-]\d+) cols(.*)$")
SHORT_BAR = re.compile(r"^bar (.+?) at (row|col) (\d+): (\d+) -> (\d+) cells$")
SHORT_TURN = re.compile(r"^(\S+@\d+,\d+) turned (\d+) degrees clockwise$")
SHORT_CHANGE = re.compile(r"^(\S+@\d+,\d+) (changed shape|recoloured) \(was (\S+), (.+)\)$")


def short_move(item):
    change = SHORT_CHANGE.match(item)
    if change:
        what = "shape" if change.group(2) == "changed shape" else "colour"
        return f"{change.group(1)} {what} (was {change.group(3)} {change.group(4)})"
    m = SHORT_MOVE.match(item)
    if not m:
        turned = SHORT_TURN.match(item)
        return f"{turned.group(1)} {turned.group(2)} clockwise" if turned else item
    at, how, dr, dc, rest = m.groups()
    steps = " ".join(part for part, value in ((f"{dr} rows", dr), (f"{dc} cols", dc)) if int(value))
    return f"{at}{' jumped' if how == 'jumped' else ''} {steps or 'in place'}{rest}"


def short_delta(lines):
    out = []
    for line in lines:
        head, sep, body = line.partition(": ")
        if sep and head in ("moved", "turned", "changed"):
            out.append(f"{head}: " + "; ".join(short_move(one) for one in body.split("; ")))
            continue
        bar = SHORT_BAR.match(line)
        if bar:
            out.append(f"bar {bar.group(1)} {bar.group(2)} {bar.group(3)}: {bar.group(4)}->{bar.group(5)}")
            continue
        out.append(line[:-len(" in all")] if line.endswith(" cells differ in all") else line)
    return out


SCENE_LEGEND = ("- A board digest in short form: DIGEST (why), new types, the count against the last level, bg, bars as "
                "bar <colour> <row|col> <n> <cols|rows> <first>-<last>; then each type with colours, size, ×<count> "
                "and its free positions <row>,<col> (↻<degrees> = turned clockwise from the catalog); a container "
                "line <type> <row>,<col> holds: lists the copies inside it, so each copy's position is given once. "
                "The picture after a digest is the sprite catalog: each type drawn alone under its name, at the scale "
                "noted (1:k means k cells per drawn cell). Object cells: current_frame.objects[<label>] or "
                "owner(<row>, <col>).")
SHORT_DELTA_LEGEND = ("- A board change in short form: moved: <label>@<row>,<col> <+/-n> rows <+/-n> cols (a zero part "
                      "is left out); changed: <label>@<row>,<col> <shape|colour> (was <old label> <colours> <size>); bar <colour> "
                      "<row|col> <n>: <before>-><after> cells.")


def full_parts(d, reason, extra=(), grid=None):
    text = full_text(d, reason, extra, grid=grid)
    if flag(SCENE_ENV):
        short = scene(text)
        say_once("scene", f"PERCEPTION SCENE first used: digest {len(text)} -> {len(short)} characters")
        text = short
    parts = [{"type": "text", "text": text}]
    if flag(CATALOG_ENV):
        im = catalog(d["tiles"])
        if im is not None:
            parts.append(image_part(im))
    return parts


def reason_of(st, frame, summary):
    if st.last is None:
        return "first turn"
    summary = summary or {}
    if summary.get("stale"):
        return None
    if summary.get("game_over"):
        return "after a death and the automatic RESET"
    if summary.get("level_transition") or int(getattr(frame, "level", 0) or 0) != st.last[2]:
        return "new level"
    if "RESET" in [str(one).strip().upper() for one in summary.get("executed_actions") or []]:
        return "after RESET"
    return None


def executed_any(summary):
    if not summary or summary.get("stale"):
        return False
    try:
        return int(summary.get("executed_count") or 0) > 0
    except (TypeError, ValueError):
        return False


SAID = set()
STATS = {"digests": 0, "digest_ms": 0.0, "digest_max_ms": 0.0, "deltas": 0, "delta_ms": 0.0, "delta_tokens": 0,
         "full": 0, "full_tokens": 0, "cut": 0, "tool": 0}
EVERY = 500
TIMES = []


def say_once(key, text):
    if key not in SAID:
        SAID.add(key)
        print(text, flush=True)


def note_digest(ms):
    STATS["digests"] += 1
    STATS["digest_ms"] += ms
    STATS["digest_max_ms"] = max(STATS["digest_max_ms"], ms)
    if len(TIMES) < 10000:
        TIMES.append(ms)
    if STATS["digests"] % EVERY == 0:
        n = STATS["digests"]
        print(f"PERCEPTION tally: {n} digests, mean {STATS['digest_ms'] / n:.1f} ms, max {STATS['digest_max_ms']:.1f} ms; "
              f"{STATS['deltas']} deltas, mean {STATS['delta_ms'] / max(1, STATS['deltas']):.1f} ms and "
              f"{STATS['delta_tokens'] / max(1, STATS['deltas']):.0f} tokens; {STATS['full']} full digests "
              f"({STATS['cut']} after a cut, {STATS['tool']} in tool output), mean "
              f"{STATS['full_tokens'] / max(1, STATS['full']):.0f} tokens", flush=True)


def tokens(text):
    return len(text) // 4


def visible(messages):
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            if DIGEST_HEAD in content:
                return True
            continue
        for part in content or ():
            if isinstance(part, dict) and part.get("type") == "text" and DIGEST_HEAD in (part.get("text") or ""):
                return True
    return False


try:
    from inference.agent import frame_objects as _frame_objects
except ImportError:
    _frame_objects = None
if _frame_objects is not None:
    _frame_objects.apply_module(globals())
