import json
import os
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

ENV = "TAAF_VLLM_DRAFT_BUDGET"
DEPTH_MAX = 8
KEY_BITS = 12
KEY_MAX = (1 << KEY_BITS) - 1
LANES_MAX = 1024
INVALID = -1
EDGES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
BUCKETS = len(EDGES) + 1
CAPTURE_SIZES = tuple(range(1, 17)) + tuple(range(20, 129, 4)) + tuple(range(136, 673, 8))
TABLE_MODULES = ("taaf_draft_budget_tables.py", "draft_budget_tables.py")


def edges_tensor(edges=EDGES, device=None):
    return torch.tensor(list(edges), dtype=torch.float32, device=device)


def sanitize(conf):
    conf = conf.to(torch.float32)
    conf = torch.where(conf > 0, conf, torch.zeros_like(conf))
    return torch.where(conf < 1, conf, torch.ones_like(conf))


def running_scores(conf):
    conf = sanitize(conf)
    columns = [conf[:, 0]]
    for d in range(1, conf.shape[1]):
        columns.append(columns[-1] * conf[:, d])
    return torch.stack(columns, dim=1)


def bucket_index(score, edges):
    edges = edges.to(score.device).to(torch.float32)
    return (score.unsqueeze(-1) >= edges).sum(dim=-1).to(torch.int64)


def slot_keys(conf, caps, num_lanes, edges, table):
    depth = conf.shape[1]
    table = table.to(conf.device).to(torch.int64)
    scores = running_scores(conf)
    bucket = bucket_index(scores, edges)
    columns = torch.arange(depth, device=conf.device)
    keys = table[columns.unsqueeze(0).expand_as(bucket), bucket]
    keys = torch.cummin(keys, dim=1).values
    lanes = torch.arange(conf.shape[0], device=conf.device)
    valid = (lanes.unsqueeze(1) < num_lanes) & (columns.unsqueeze(0) < caps.to(torch.int64).unsqueeze(1))
    return torch.where(valid, keys, torch.full_like(keys, INVALID))


def reference_allocate(conf, caps, num_lanes, budget, edges, table):
    keys = slot_keys(conf, caps, num_lanes, edges, table)
    lanes, depth = keys.shape
    valid = keys >= 0
    budget = min(int(budget), int(valid.sum()))
    counts = torch.zeros(lanes, dtype=torch.int32, device=conf.device)
    if budget <= 0:
        return counts
    lane = torch.arange(lanes, device=conf.device, dtype=torch.int64).unsqueeze(1).expand(lanes, depth)
    column = torch.arange(depth, device=conf.device, dtype=torch.int64).unsqueeze(0).expand(lanes, depth)
    rank = (keys << 24) | ((DEPTH_MAX - 1 - column) << 16) | (LANES_MAX - 1 - lane)
    rank = torch.where(valid, rank, torch.full_like(rank, -1))
    picked = rank.flatten().topk(budget).indices
    chosen = torch.zeros(lanes * depth, dtype=torch.bool, device=conf.device)
    chosen[picked] = True
    return chosen.view(lanes, depth).sum(dim=1).to(torch.int32)


def capture_size_at_least(rows, sizes=CAPTURE_SIZES):
    for size in sizes:
        if size >= rows:
            return size
    return sizes[-1]


def capture_size_at_most(rows, sizes=CAPTURE_SIZES):
    best = sizes[0]
    for size in sizes:
        if size <= rows:
            best = size
    return best


def total_rows_for(lanes, rows_per_lane_x100, sizes=CAPTURE_SIZES, depth_cap=DEPTH_MAX):
    wanted = (int(lanes) * int(rows_per_lane_x100) + 99) // 100
    wanted = max(int(lanes), min(wanted, int(lanes) * (1 + int(depth_cap))))
    return min(capture_size_at_least(wanted, sizes), sizes[-1])


def budget_for(lanes, rows_per_lane_x100, sizes=CAPTURE_SIZES, depth_cap=DEPTH_MAX):
    total = total_rows_for(lanes, rows_per_lane_x100, sizes, depth_cap)
    return max(0, min(total - int(lanes), int(lanes) * int(depth_cap)))


def load_tables(path=None):
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    here = Path(__file__).resolve().parent
    for name in TABLE_MODULES:
        candidate = here / name
        if candidate.is_file():
            namespace = {}
            exec(compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec"), namespace)
            return namespace["TABLES"]
    raise FileNotFoundError(f"{ENV} needs one of {TABLE_MODULES} beside this module")


def table_arrays(entry, device=None):
    edges = torch.tensor(entry["edges"], dtype=torch.float32, device=device)
    table = torch.tensor(entry["keys"], dtype=torch.int32, device=device)
    return edges, table


def parse_knob(said, tables=None, boot_depth=None):
    said = str(said or "").strip()
    if said in ("", "0"):
        return None
    tables = tables if tables is not None else load_tables()
    if said == "full":
        depth = int(boot_depth or DEPTH_MAX)
        name = next((n for n, t in tables["tables"].items() if int(t["depth"]) == depth), tables["default"])
        return {"name": "full", "table": name, "depth": depth, "rows_per_lane_x100": 100 * (1 + depth),
                "rows_by_lanes": None}
    presets = tables.get("presets", {})
    if said in presets:
        preset = presets[said]
        return {"name": said, "table": preset["table"], "depth": int(preset["depth"]),
                "rows_per_lane_x100": preset.get("rows_per_lane_x100"), "rows_by_lanes": preset.get("rows_by_lanes")}
    if said.isascii() and said.isdigit():
        value = int(said)
        if not 100 < value <= 100 * (1 + DEPTH_MAX):
            raise ValueError(f"{ENV}={said!r}: rows per lane x100 must be above 100 and at most {100 * (1 + DEPTH_MAX)}")
        depth = int(boot_depth or tables["tables"][tables["default"]]["depth"])
        name = next((n for n, t in tables["tables"].items() if int(t["depth"]) == depth), tables["default"])
        return {"name": said, "table": name, "depth": depth, "rows_per_lane_x100": value, "rows_by_lanes": None}
    raise ValueError(f"{ENV}={said!r} is neither full, an integer of rows per lane x100 nor one of {sorted(presets)}")


def rows_for_lanes(spec, lanes, sizes=CAPTURE_SIZES):
    if spec.get("rows_by_lanes"):
        points = sorted((int(k), int(v)) for k, v in spec["rows_by_lanes"].items())
        chosen = points[0][1]
        for at, rows in points:
            if lanes >= at:
                chosen = rows
        return total_rows_for(lanes, chosen, sizes, spec["depth"])
    return total_rows_for(lanes, spec["rows_per_lane_x100"], sizes, spec["depth"])
