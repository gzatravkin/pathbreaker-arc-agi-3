import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

ENV = "TAAF_VLLM_DRAFT_EARLY_STOP"
SELECT_ENV = "TAAF_VLLM_DRAFT_EARLY_STOP_SELECT"
THRESHOLD_ENV = "TAAF_VLLM_DRAFT_EARLY_STOP_MIN_VALUE"
DEPTH_MAX = 8
QUALIFIED_BONUS = 1.0e4
REPORT_EVERY = 30.0
NO_DRAFT = -1
EDGES = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98)
DEFAULT_ACCEPT = (0.2711, 0.4149, 0.5075, 0.5994, 0.6908, 0.7774, 0.9650)
CAPTURE_SIZES = tuple(range(1, 17)) + tuple(range(20, 129, 4)) + tuple(range(136, 673, 8))
ROW_FIELDS = ("input_ids", "positions", "seq_lens", "sample_src_positions", "hidden_states", "idx_mapping")
HELD = {}
NARROW_PREV = "_taaf_draft_sample_narrow_prev"
POINT_MASS_SAID = []
PRESETS = {
    "3-5-6": "3,5:0.5,6:0.25",
    "2-4-6": "2,4:0.5,6:0.25",
    "3-6": "3,6:0.4",
    "3-8": "3,5:0.5,8:0.2",
}


@dataclass(frozen=True)
class Plan:
    base: int
    stages: tuple

    @property
    def depth(self):
        return self.stages[-1][0] if self.stages else self.base


def parse(said, boot_depth):
    said = str(said or "").strip()
    if said in ("", "0"):
        return None
    said = PRESETS.get(said, said)
    parts = [p.strip() for p in said.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{ENV}={said!r}: want 'base[,hi:fraction...]' or one of {sorted(PRESETS)}")
    base = int(parts[0])
    stages = []
    last_depth, last_fraction = base, 1.0
    for part in parts[1:]:
        hi, fraction = part.split(":")
        hi, fraction = int(hi), float(fraction)
        if hi <= last_depth:
            raise ValueError(f"{ENV}: stage depths must increase, {hi} follows {last_depth}")
        if not 0.0 < fraction <= last_fraction:
            raise ValueError(f"{ENV}: stage fractions must fall inside (0, previous], got {fraction}")
        stages.append((hi, fraction))
        last_depth, last_fraction = hi, fraction
    if base < 1 or last_depth > DEPTH_MAX:
        raise ValueError(f"{ENV}: base must be at least 1 and depths at most {DEPTH_MAX}")
    if last_depth > int(boot_depth):
        raise ValueError(f"{ENV}={said!r} drafts to depth {last_depth} but the drafter boots with {boot_depth}")
    return Plan(base, tuple(stages))


def bucket(rows, sizes=CAPTURE_SIZES):
    for size in sizes:
        if size >= rows:
            return size
    return sizes[-1]


def stage_rows(plan, num_reqs, sizes=CAPTURE_SIZES):
    out = []
    for hi, fraction in plan.stages:
        want = max(1, -(-int(num_reqs) * int(round(fraction * 1000)) // 1000))
        out.append(min(bucket(want, sizes), bucket(num_reqs, sizes)))
    return tuple(out)


class Calibration:
    def __init__(self, accept=DEFAULT_ACCEPT, edges=EDGES, tail_floor=0.0, tail_cap=0.98):
        self.edges = torch.tensor(edges, dtype=torch.float32)
        self.accept = torch.tensor(accept, dtype=torch.float32)
        self.tail_floor, self.tail_cap = tail_floor, tail_cap
        self._device = {}

    def on(self, device):
        held = self._device.get(device)
        if held is None:
            held = self._device[device] = (self.edges.to(device), self.accept.to(device))
        return held

    def conditional(self, conf):
        edges, accept = self.on(conf.device)
        index = (conf.to(torch.float32).unsqueeze(-1) >= edges).sum(dim=-1)
        return accept[index]

    def survival(self, conf):
        return torch.cumprod(self.conditional(conf), dim=-1)

    def value(self, conf, extra):
        cond = self.conditional(conf)
        survive = torch.cumprod(cond, dim=-1)[..., -1]
        rate = cond[..., -1].clamp(self.tail_floor, self.tail_cap)
        series = rate * (1.0 - rate ** int(extra)) / (1.0 - rate)
        return survive * series


def select(value, alive, k, mode="conf", threshold=0.0, noise=None):
    lanes = value.shape[0]
    take = min(int(k), lanes)
    lane = torch.arange(lanes, device=value.device, dtype=torch.float64)
    ok = alive & (value >= threshold)
    if mode == "conf":
        score = value.to(torch.float64) - lane * 1e-9
    elif mode == "random":
        score = (noise if noise is not None else torch.rand(lanes, device=value.device)).to(torch.float64)
    elif mode == "first":
        score = -lane
    else:
        raise ValueError(f"{SELECT_ENV}={mode!r} is not conf, random or first")
    score = score + ok.to(torch.float64) * QUALIFIED_BONUS
    src = score.topk(take).indices.sort().values
    return src, ok[src]


def compact_rows(rows, src, size, real, held):
    for name in ROW_FIELDS:
        tensor = getattr(rows, name)
        tensor[:size].copy_(tensor.index_select(0, src))
    rows.seq_lens[:size].mul_(real.to(rows.seq_lens.dtype))
    if held > size:
        rows.seq_lens[size:held].zero_()


def scatter_depths(out, stage_out, src, real, lo, hi):
    extra = out.shape[0]
    scratch = torch.zeros((extra + 1, hi - lo), dtype=out.dtype, device=out.device)
    scratch[:extra] = out[:, lo:hi]
    block = stage_out[: src.shape[0], lo:hi]
    target = torch.where(real, src, torch.full_like(src, extra))
    scratch.index_copy_(0, target, block)
    out[:, lo:hi] = scratch[:extra]


class State:
    def __init__(self, reached, stats, calibration, mode="conf", threshold=0.0):
        self.reached, self.stats, self.calibration = reached, stats, calibration
        self.mode, self.threshold = mode, threshold
        self.last_sizes, self.steps, self.log = (), 0, []


def arm(max_num_reqs, plan, device, calibration=None, mode=None, threshold=None):
    mode = mode or os.environ.get(SELECT_ENV, "conf")
    threshold = float(os.environ.get(THRESHOLD_ENV, "0") if threshold is None else threshold)
    state = State(
        reached=torch.full((int(max_num_reqs) + 1,), plan.depth, dtype=torch.int32, device=device),
        stats=torch.zeros((len(plan.stages), 3), dtype=torch.int64, device=device),
        calibration=calibration or Calibration(), mode=mode, threshold=threshold)
    state.plan = plan
    HELD[str(device)] = state
    return state


def arm_speculator(owner, said, device):
    if getattr(owner, "dp_size", 1) > 1:
        raise RuntimeError(f"{ENV} does not support data parallel drafting")
    if getattr(owner, "_taaf_budget_conf", None) is None:
        raise RuntimeError(f"{ENV} reads the drafter's recorded token probabilities and caps the verification rows "
                           f"through the draft budget; arm TAAF_VLLM_DRAFT_BUDGET and TAAF_VLLM_DRAFT_SAMPLE=1")
    plan = parse(said, owner.num_speculative_steps)
    if plan is None or not plan.stages:
        raise RuntimeError(f"{ENV}={said!r} names no stage; give 'base,depth:fraction,...' or one of {sorted(PRESETS)}")
    state = arm(owner.max_num_reqs, plan, device)
    stages = " ".join(f"depth{hi}:{fraction}" for hi, fraction in plan.stages)
    print(f"DRAFT_EARLY_STOP armed: base {plan.base} drafts on every lane, then {stages or 'no stage'} on the "
          f"highest-value lanes; select={state.mode} min_value={state.threshold}; every row of a stage is a real "
          f"lane (a stage that cannot fill its graph size with qualifying lanes drafts the next best); fused "
          f"multi-step draft decode is OFF while armed; the first window line follows the first speculative "
          f"step, then one every {REPORT_EVERY:.0f} s", flush=True)
    return plan, state


def held(device):
    return HELD.get(str(device))


def rows_bound(state, lanes):
    plan = state.plan
    sizes = state.last_sizes or stage_rows(plan, lanes)
    total, previous = int(lanes) * plan.base, plan.base
    for (hi, _), size in zip(plan.stages, sizes):
        total += min(int(size), int(lanes)) * (hi - previous)
        previous = hi
    return total


def reset(state, req_idx):
    state.reached[req_idx] = 0


def speculator_rows(owner):
    return SimpleNamespace(
        input_ids=owner.input_buffers.input_ids,
        positions=owner.input_buffers.positions,
        seq_lens=owner.input_buffers.seq_lens,
        sample_src_positions=owner.sample_src_positions,
        hidden_states=owner.hidden_states,
        idx_mapping=owner.idx_mapping,
        point_mass=speculator_point_mass(owner))


def speculator_point_mass(owner):
    cache = getattr(owner, "draft_logits", None)
    if cache is None:
        return None

    def apply(slots, reach, out, lanes, base):
        prev = vars(owner).get(NARROW_PREV)
        ids = None
        try:
            from vllm.taaf_block_verify import _state as verify_state
            ids = verify_state.get("ids")
        except ImportError:
            pass
        point_mass(cache, prev, ids, slots, reach, out, lanes, base)
        if not POINT_MASS_SAID:
            POINT_MASS_SAID.append(True)
            print(f"DRAFT_EARLY_STOP point mass: every column a lane did not reach is verified as a draft drawn from "
                  f"a distribution with all its mass on the repeated token (cache row -inf except 0, narrow table "
                  f"{'kept' if prev is not None else 'absent, full row wiped'}, verify id table "
                  f"{'kept' if ids is not None else 'absent'})", flush=True)

    return apply


if triton is not None:
    @triton.jit
    def _point_mass_kernel(out_ptr, out_stride, reach_ptr, slots_ptr, cache_ptr, cache_s0, cache_s1,
                           prev_ptr, prev_s0, prev_s1, ids_ptr, ids_s0, ids_s1, base, vocab, top,
                           HAS_PREV: tl.constexpr, HAS_IDS: tl.constexpr, PREV_W: tl.constexpr,
                           IDS_W: tl.constexpr, CLEAR_BLOCK: tl.constexpr):
        lane = tl.program_id(0).to(tl.int64)
        col = base + tl.program_id(1).to(tl.int64)
        reach = tl.load(reach_ptr + lane).to(tl.int64)
        slot = tl.load(slots_ptr + lane).to(tl.int64)
        token = tl.load(out_ptr + lane * out_stride + col).to(tl.int64)
        if (col >= reach) & (slot >= 0) & (token >= 0) & (token < vocab):
            row = cache_ptr + slot * cache_s0 + col * cache_s1
            wipe_all = True
            if HAS_PREV:
                prev_row = prev_ptr + slot * prev_s0 + col * prev_s1
                k = tl.arange(0, PREV_W)
                before = tl.load(prev_row + k)
                wipe_all = tl.load(prev_row) == -2
                if not wipe_all:
                    tl.store(row + before.to(tl.int64), tl.full((PREV_W,), float("-inf"), tl.float32)
                             .to(cache_ptr.dtype.element_ty), mask=before >= 0)
            if wipe_all:
                for start in range(0, vocab, CLEAR_BLOCK):
                    spot = start + tl.arange(0, CLEAR_BLOCK)
                    tl.store(row + spot, tl.full((CLEAR_BLOCK,), float("-inf"), tl.float32)
                             .to(cache_ptr.dtype.element_ty), mask=spot < vocab)
            tl.debug_barrier()
            tl.store(row + token, tl.zeros((), tl.float32).to(cache_ptr.dtype.element_ty))
            if HAS_PREV:
                k = tl.arange(0, PREV_W)
                tl.store(prev_ptr + slot * prev_s0 + col * prev_s1 + k, tl.where(k == 0, token, -1).to(tl.int32))
            if HAS_IDS:
                j = tl.arange(0, IDS_W)
                tl.store(ids_ptr + slot * ids_s0 + col * ids_s1 + j, tl.where(j == 0, token, -1).to(tl.int32),
                         mask=j < top)


def point_mass_reference(cache, prev, ids, slots, reach, out, lanes, base):
    depth = min(out.shape[1], cache.shape[1])
    for lane in range(int(lanes)):
        slot = int(slots[lane])
        for col in range(int(base), depth):
            token = int(out[lane, col])
            if col < int(reach[lane]) or slot < 0 or token < 0 or token >= cache.shape[-1]:
                continue
            if prev is not None and int(prev[slot, col, 0]) != -2:
                for old in prev[slot, col].tolist():
                    if old >= 0:
                        cache[slot, col, old] = float("-inf")
            else:
                cache[slot, col].fill_(float("-inf"))
            cache[slot, col, token] = 0.0
            if prev is not None:
                prev[slot, col].fill_(-1)
                prev[slot, col, 0] = token
            if ids is not None:
                ids[slot, col].fill_(-1)
                ids[slot, col, 0] = token


def point_mass(cache, prev, ids, slots, reach, out, lanes, base):
    lanes = int(lanes)
    depth = min(out.shape[1], cache.shape[1])
    if lanes == 0 or depth <= base:
        return
    if triton is None or not cache.is_cuda:
        return point_mass_reference(cache, prev, ids, slots, reach, out, lanes, base)
    if prev is not None and prev.shape[1] < depth:
        prev = None
    if ids is not None and ids.shape[1] < depth:
        ids = None
    dummy = slots
    _point_mass_kernel[(lanes, depth - base)](
        out, out.stride(0), reach.contiguous(), slots.contiguous(), cache, cache.stride(0), cache.stride(1),
        prev if prev is not None else dummy, prev.stride(0) if prev is not None else 0,
        prev.stride(1) if prev is not None else 0,
        ids if ids is not None else dummy, ids.stride(0) if ids is not None else 0,
        ids.stride(1) if ids is not None else 0,
        int(base), int(cache.shape[-1]), int(ids.shape[2]) if ids is not None else 0,
        HAS_PREV=prev is not None, HAS_IDS=ids is not None,
        PREV_W=int(prev.shape[2]) if prev is not None else 1,
        IDS_W=triton.next_power_of_2(int(ids.shape[2])) if ids is not None else 1,
        CLEAR_BLOCK=4096)


class RowIndexed:
    """Tensors that other kernels read by batch row for the drafter's rows (the hisparse
    request_state_indices, the gathered block tables). Each compacted stage presents them in the compacted row
    order and the original order is put back when the stages end."""

    def __init__(self, tensors, regather=None, pad=-1):
        self.tensors, self.regather, self.pad = list(tensors), regather, pad
        self.saved, self.idx, self.lanes = [], None, 0

    def begin(self, lanes, idx_mapping):
        self.lanes = int(lanes)
        self.saved = [t[: self.lanes].clone() for t in self.tensors]
        self.idx = idx_mapping[: self.lanes].clone()

    def stage(self, origin, size, idx_mapping, row_lane=None):
        size = int(size)
        for tensor, saved in zip(self.tensors, self.saved):
            if row_lane is None:
                tensor[:size].copy_(saved.index_select(0, origin))
                tensor[size: self.lanes].fill_(self.pad)
            else:
                tensor[: self.lanes].copy_(saved.index_select(0, row_lane))
        if self.regather is not None:
            self.regather(idx_mapping[:size], size)

    def end(self):
        for tensor, saved in zip(self.tensors, self.saved):
            tensor[: self.lanes].copy_(saved)
        if self.regather is not None:
            self.regather(self.idx, self.lanes)


def state_index_tensors(owner):
    found = owner.__dict__.get("_taaf_state_index_tensors")
    if found is not None:
        return found
    found, seen = [], set()
    config = getattr(owner.vllm_config, "compilation_config", None)
    context = getattr(config, "static_forward_context", None) or {}
    for layer in context.values():
        runtime = getattr(getattr(layer, "hisparse_cache", None), "runtime", None)
        tensor = getattr(runtime, "request_state_indices", None)
        if tensor is not None and tensor.data_ptr() not in seen:
            seen.add(tensor.data_ptr())
            found.append(tensor)
    owner.__dict__["_taaf_state_index_tensors"] = found
    print(f"DRAFT_EARLY_STOP row-indexed state: {len(found)} hisparse request_state_indices tensor(s) follow each "
          f"compacted stage, block tables are regathered per stage", flush=True)
    return found


def speculator_hooks(owner, num_reqs, batch_desc):
    padded = max(int(num_reqs), int(getattr(batch_desc, "num_reqs", 0) or 0))
    tables = owner.block_tables

    def regather(idx, size):
        tables.gather_block_tables(idx, num_reqs_padded=padded)

    return RowIndexed(state_index_tensors(owner), regather)


def run_stages(state, plan, rows, conf, out, num_reqs, step_fn, sizes=CAPTURE_SIZES, hooks=None):
    """After the base depths ran on all `num_reqs` lanes, run every later stage on a compacted batch.

    `rows` holds the per-lane working tensors (ROW_FIELDS), `conf[slot, depth]` the drafter's chosen
    probability, `out[lane, depth]` the persistent draft token matrix, `step_fn(depth, size, stage_out)` drafts
    one depth for the first `size` rows of the working tensors and writes tokens to `stage_out[row, depth-1]`.
    Every row of a stage is a real lane: the stage takes the highest-value qualifying lanes first and fills the
    rest of its graph size with the next best lanes, so no row carries a placeholder request slot.
    `hooks` (begin/stage/end) re-present row-indexed state to kernels that read it by row.
    No call below reads a device value on the host."""
    device = out.device
    lanes = int(num_reqs)
    state.last_sizes = stage_rows(plan, lanes, sizes)
    state.steps += 1
    slots = rows.idx_mapping[:lanes].long().clone()
    state.reached.index_put_((slots,), torch.full((lanes,), plan.base, dtype=torch.int32, device=device))
    out[:lanes, plan.base:] = NO_DRAFT
    if not plan.stages:
        return state.reached
    if hooks is not None:
        hooks.begin(lanes, rows.idx_mapping)
    alive = torch.ones(lanes, dtype=torch.bool, device=device)
    origin = torch.arange(lanes, device=device)
    row_lane = torch.arange(lanes, device=device)
    previous = plan.base
    held = lanes
    stage_out = torch.full_like(out, NO_DRAFT)
    for index, ((hi, _), size) in enumerate(zip(plan.stages, stage_rows(plan, lanes, sizes))):
        size = min(size, held)
        conf_rows = conf[rows.idx_mapping[:held].long(), :previous]
        value = state.calibration.value(conf_rows, hi - previous)
        noise = torch.rand(held, device=device) if state.mode == "random" else None
        real = torch.ones(size, dtype=torch.bool, device=device)
        if size < held:
            src, valid = select(value, alive[:held], size, state.mode, state.threshold, noise)
            compact_rows(rows, src, size, real, held)
            origin = origin[src]
            row_lane[:size] = row_lane.index_select(0, src)
            if hooks is not None:
                hooks.stage(origin, size, rows.idx_mapping, row_lane=row_lane)
        else:
            valid = alive[:held] & (value >= state.threshold)
        stage_out[:size] = NO_DRAFT
        for depth in range(previous + 1, hi + 1):
            step_fn(depth, size, stage_out)
        out[:lanes, previous:hi] = NO_DRAFT
        scatter_depths(out, stage_out, origin, real, previous, hi)
        state.reached.index_put_((rows.idx_mapping[:size].long(),),
                                 torch.full((size,), hi, dtype=torch.int32, device=device))
        state.stats[index, 0] += valid.sum()
        state.stats[index, 1] += size
        state.stats[index, 2] += lanes
        alive, held, previous = valid, size, hi
    if hooks is not None:
        hooks.end()
    rows.idx_mapping[:lanes].copy_(slots.to(rows.idx_mapping.dtype))
    reach = fill_unreached(state, plan, slots, conf, out, lanes)
    mass = getattr(rows, "point_mass", None)
    if mass is not None:
        mass(slots, reach, out, lanes, plan.base)
    return state.reached


def fill_unreached(state, plan, slots, conf, out, lanes):
    depth = plan.depth
    reach = state.reached[slots].long()
    cols = torch.arange(depth, device=out.device)
    unreached = cols.unsqueeze(0) >= reach.unsqueeze(1)
    last = out[:lanes, :depth].gather(1, (reach - 1).clamp(min=0).unsqueeze(1))
    block = out[:lanes, :depth]
    out[:lanes, :depth] = torch.where(unreached, last.expand_as(block), block)
    rows_conf = conf[slots, :depth]
    conf[slots, :depth] = torch.where(unreached, torch.zeros_like(rows_conf), rows_conf)
    return reach


def clamp_capacities(state, capacities, idx_mapping, num_reqs):
    caps = state.reached[idx_mapping[:num_reqs].long()]
    capacities[:num_reqs].copy_(torch.minimum(capacities[:num_reqs], caps.to(capacities.dtype)))
    return capacities


def report(state, every=None):
    every = REPORT_EVERY if every is None else every
    now = time.monotonic()
    first = state.steps == 1 and not getattr(state, "reported", 0.0)
    if not first and now - getattr(state, "reported", 0.0) < every:
        return
    state.reported = now or 1e-9
    rows = state.stats.tolist()
    state.stats.zero_()
    cells, saved, full, previous = [], 0, 0, state.plan.base
    for (hi, _), (qualified, ran, lanes) in zip(state.plan.stages, rows):
        cells.append(f"d{previous + 1}-{hi}:ran={ran}/{lanes} qualified={qualified} stopped={lanes - ran}")
        saved += (lanes - ran) * (hi - previous)
        full += lanes * (hi - previous)
        previous = hi
    pct = 100.0 * saved / full if full else 0.0
    print(f"DRAFT_EARLY_STOP window steps={state.steps} lane-depth rows per stage {' '.join(cells)} "
          f"rows_saved={saved}/{full} ({pct:.1f}%) base={state.plan.base} depth={state.plan.depth}", flush=True)
