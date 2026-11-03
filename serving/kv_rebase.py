import collections
import dataclasses
import os

import numpy as np
import torch

KNOB = "TAAF_VLLM_KV_REBASE"
RESYNC_KNOB = "TAAF_VLLM_KV_REBASE_RESYNC"
RESYNC_DEFAULT = 3
HEAD_DIM = 256
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
COMPRESS_RATIO = 4
RHT_SEED = 0x9E3779B9
LEVELS = 15.0
MISSING = (
    "registering a planned rebase as computed tokens for the new request across the hybrid KV groups "
    "(synthesized cache entries under the new request's hashes, block allocation and references)",
    "the scheduler-to-worker channel that runs the page ops before the request's first forward",
    "resolving the old request's blocks in the GPU cache or the RAM tier and pinning them for the copy",
    "the request token rewrite for pads with its block hashes rebuilt",
)


def enabled():
    return str(os.environ.get(KNOB) or "").strip() not in ("", "0")


def resync_every():
    said = str(os.environ.get(RESYNC_KNOB) or "").strip()
    every = int(said) if said else RESYNC_DEFAULT
    if every < 0:
        raise ValueError(f"{RESYNC_KNOB} must be 0 (never recompute) or a trim count >= 1, found {said!r}")
    return every


def action_for(trim_number, every):
    if every and trim_number % every == 0:
        return "recompute"
    return "rebase"


def refusal():
    return (f"{KNOB} asks to reuse a trimmed request's kept KV at shifted positions; the index, the plan and "
            f"the page ops exist, still missing: " + "; ".join(MISSING))


def rht_signs(dim=HEAD_DIM):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RHT_SEED)
    return 2.0 * torch.bernoulli(torch.full((dim,), 0.5), generator=generator) - 1.0


def hadamard(x):
    dim = x.shape[-1]
    if dim & (dim - 1):
        raise ValueError(f"the Hadamard transform needs a power-of-two width, got {dim}")
    out = x.reshape(-1, dim).clone()
    span = 1
    while span < dim:
        out = out.view(-1, dim // (2 * span), 2, span)
        left, right = out[:, :, 0, :], out[:, :, 1, :]
        out = torch.stack((left + right, left - right), dim=2).view(-1, dim)
        span *= 2
    return out.view(x.shape)


def rht(x, signs=None):
    signs = rht_signs(x.shape[-1]) if signs is None else signs
    return hadamard(x * signs.to(x.dtype))


def inverse_rht(y, signs=None):
    signs = rht_signs(y.shape[-1]) if signs is None else signs
    return hadamard(y) * signs.to(y.dtype) / y.shape[-1]


def inverse_frequencies(rotary_dim=ROTARY_DIM, theta=ROPE_THETA):
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))


def rotate(x, positions, rotary_dim=ROTARY_DIM, theta=ROPE_THETA):
    half = rotary_dim // 2
    angles = positions.to(torch.float64)[..., None] * inverse_frequencies(rotary_dim, theta)
    while angles.dim() < x.dim():
        angles = angles.unsqueeze(-2)
    cos, sin = torch.cos(angles).to(x.dtype), torch.sin(angles).to(x.dtype)
    first, second, rest = x[..., :half], x[..., half:rotary_dim], x[..., rotary_dim:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin, rest), dim=-1)


def shift_matrix(delta, dim=HEAD_DIM, rotary_dim=ROTARY_DIM, theta=ROPE_THETA, signs=None):
    eye = torch.eye(dim, dtype=torch.float64)
    signs = rht_signs(dim).to(torch.float64) if signs is None else signs.to(torch.float64)
    back = inverse_rht(eye, signs)
    moved = rotate(back, torch.full((dim,), float(delta), dtype=torch.float64), rotary_dim, theta)
    return rht(moved, signs).contiguous()


def quantize(x):
    flat = x.to(torch.float32)
    low = flat.amin(dim=-1, keepdim=True)
    high = flat.amax(dim=-1, keepdim=True)
    scale = torch.clamp((high - low) / LEVELS, min=1e-6)
    ratio = -low / scale
    zero = torch.clamp(torch.where(ratio >= 0, torch.floor(ratio + 0.5), torch.ceil(ratio - 0.5)), 0.0, LEVELS)
    scaled = flat / scale + zero
    codes = torch.clamp(torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)), 0.0, LEVELS)
    bits = scale.view(torch.int32) & -16 | zero.to(torch.int32)
    return codes.to(torch.uint8), bits.view(torch.float32)


def dequantize(codes, packed):
    bits = packed.view(torch.int32)
    zero = (bits & 0xF).to(torch.float32)
    scale = (bits & -16).view(torch.float32)
    return (codes.to(torch.float32) - zero) * scale


def rebase_keys(codes, packed, delta, matrix=None):
    matrix = shift_matrix(delta) if matrix is None else matrix
    values = dequantize(codes, packed).to(torch.float64)
    return quantize((values @ matrix).to(torch.float32))


def aligned_shift(kept_start_old, kept_start_new, ratio=COMPRESS_RATIO):
    return (kept_start_old - kept_start_new) % ratio == 0


def pad_for_alignment(kept_start_old, kept_start_new, ratio=COMPRESS_RATIO):
    return (kept_start_old - kept_start_new) % ratio


WINDOW = 64
STRIDE = 16
KEEP_REQUESTS = 256
HASH_SEED = 0x51ED27
STATE_UNIT = 128


def _weights(width):
    return np.random.default_rng(HASH_SEED).integers(1, 2 ** 63, size=width, dtype=np.int64).astype(np.uint64) | np.uint64(1)


_WEIGHTS = {}


def window_keys(tokens, starts, width=WINDOW):
    tokens = np.asarray(tokens, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    starts = starts[starts + width <= len(tokens)]
    if not len(starts):
        return starts, np.empty(0, dtype=np.uint64)
    weights = _WEIGHTS.get(width)
    if weights is None:
        weights = _WEIGHTS[width] = _weights(width)
    rows = tokens[starts[:, None] + np.arange(width)[None, :]].astype(np.uint64)
    with np.errstate(over="ignore"):
        keys = (rows * weights[None, :]).sum(axis=1, dtype=np.uint64)
    return starts, keys


def window_key(tokens, start, width=WINDOW):
    starts, keys = window_keys(tokens, [start], width)
    return int(keys[0]) if len(keys) else None


def common_run(a, b):
    count = min(len(a), len(b))
    if not count:
        return 0
    differ = np.flatnonzero(a[:count] != b[:count])
    return int(differ[0]) if differ.size else count


class TrimIndex:
    def __init__(self, keep=KEEP_REQUESTS, width=WINDOW, stride=STRIDE):
        self.keep = keep
        self.width = width
        self.stride = stride
        self.tokens = collections.OrderedDict()
        self.where = {}
        self.conversation = {}
        self.trims = collections.Counter()

    def register(self, request_id, tokens, conversation=None):
        tokens = np.asarray(list(tokens) if not isinstance(tokens, np.ndarray) else tokens, dtype=np.int64)
        if request_id in self.tokens:
            self.forget(request_id)
        self.conversation[request_id] = request_id if conversation is None else conversation
        self.tokens[request_id] = tokens
        starts, keys = window_keys(tokens, np.arange(0, max(len(tokens) - self.width + 1, 0), self.stride), self.width)
        for start, key in zip(starts.tolist(), keys.tolist()):
            self.where.setdefault(key, []).append((request_id, start))
        while len(self.tokens) > self.keep:
            self.forget(next(iter(self.tokens)))

    def forget(self, request_id):
        tokens = self.tokens.pop(request_id, None)
        if tokens is None:
            return
        self.conversation.pop(request_id, None)
        starts, keys = window_keys(tokens, np.arange(0, max(len(tokens) - self.width + 1, 0), self.stride), self.width)
        for key in set(keys.tolist()):
            kept = [one for one in self.where.get(key, []) if one[0] != request_id]
            if kept:
                self.where[key] = kept
            else:
                self.where.pop(key, None)

    def find(self, tokens, start, minimum=1024):
        tokens = np.asarray(tokens, dtype=np.int64)
        offsets, keys = window_keys(tokens, np.arange(start, min(start + self.stride, len(tokens) - self.width + 1)),
                                    self.width)
        best = None
        for offset, key in zip(offsets.tolist(), keys.tolist()):
            for request_id, old in self.where.get(key, []):
                begin_old = old - (offset - start)
                if begin_old < 0:
                    continue
                run = common_run(tokens[start:], self.tokens[request_id][begin_old:])
                if run >= minimum and (best is None or run > best[3]):
                    best = (request_id, begin_old, start, run)
        return best

    def segments(self, tokens, start, minimum=256):
        tokens = np.asarray(tokens, dtype=np.int64)
        found = []
        position = start
        while position < len(tokens) - self.width:
            hit = self.find(tokens, position, minimum)
            if hit is None:
                break
            found.append(hit)
            position = hit[2] + hit[3]
        return found

    def decide(self, old_request_id, every):
        conversation = self.conversation.get(old_request_id, old_request_id)
        self.trims[conversation] += 1
        number = self.trims[conversation]
        return action_for(number, every), conversation, number


@dataclasses.dataclass(frozen=True)
class Segment:
    request: object
    begin_old: int
    begin_new: int
    length: int

    @property
    def shift(self):
        return self.begin_new - self.begin_old


@dataclasses.dataclass(frozen=True)
class Plan:
    request: object
    segments: tuple
    pads: int
    boundary_new: int
    boundary_old: int
    junction_group: int
    action: str
    trim_number: int
    tokens: tuple

    @property
    def reused(self):
        return sum(one.length for one in self.segments)


def plan_trim(index, tokens, cached, every, unit=STATE_UNIT, ratio=COMPRESS_RATIO, minimum=1024):
    tokens = np.asarray(tokens, dtype=np.int64)
    found = index.segments(tokens, cached, minimum=min(minimum, 256))
    if not found:
        return None
    owner = found[0][0]
    chain = []
    position = cached
    for request_id, begin_old, begin_new, length in found:
        if request_id != owner or begin_new != position:
            break
        chain.append(Segment(request_id, begin_old, begin_new, length))
        position = begin_new + length
    moved = [one for one in chain if one.shift != 0]
    if not moved or moved[0].length < minimum:
        return None
    if len(moved) > 1 or chain[-1] is not moved[0]:
        chain = chain[:chain.index(moved[0]) + 1]
    run = chain[-1]
    pads = (-run.shift) % ratio
    source = index.tokens[owner]
    if run.begin_old - pads < 0:
        return None
    inserted = source[run.begin_old - pads:run.begin_old]
    rewritten = np.concatenate([tokens[:run.begin_new], inserted, tokens[run.begin_new:]])
    run = Segment(owner, run.begin_old - pads, run.begin_new, run.length + pads)
    chain = tuple(chain[:-1]) + (run,)
    end_old = run.begin_old + run.length
    boundary_old = end_old // unit * unit
    boundary_new = boundary_old + run.shift
    if boundary_new <= run.begin_new or (boundary_new % ratio):
        return None
    trimmed = Segment(owner, run.begin_old, run.begin_new, boundary_new - run.begin_new)
    chain = chain[:-1] + (trimmed,)
    junction = run.begin_new // ratio if run.begin_new % ratio else -1
    action, _, number = index.decide(owner, every)
    return Plan(owner, chain, pads, boundary_new, boundary_old, junction, action, number,
                tuple(int(one) for one in rewritten.tolist()))


def positions_are_token_index(model_cls, sample=8):
    reader = getattr(model_cls, "get_mrope_input_positions", None)
    if reader is None:
        return True
    try:
        positions, delta = reader(None, list(range(sample)), [])
    except Exception:
        return False
    expected = torch.arange(sample, dtype=positions.dtype).unsqueeze(0).expand(3, -1)
    return int(delta) == 0 and positions.shape == expected.shape and bool(torch.equal(positions, expected))


KEY_BYTES = 128
VALUE_BYTES = 128
SCALE_AT = KEY_BYTES + VALUE_BYTES


def unpack_nibbles(packed):
    low = packed & 0xF
    high = packed >> 4
    return torch.stack((low, high), dim=-1).flatten(-2)


def pack_nibbles(codes):
    pairs = codes.view(*codes.shape[:-1], -1, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def qsa_rows(cache):
    blocks, heads, tokens, content = cache.shape
    return cache.permute(0, 2, 1, 3), tokens


def gather_rows(cache, slots):
    rows, tokens = qsa_rows(cache)
    slots = torch.as_tensor(slots, dtype=torch.long, device=cache.device)
    return rows[slots // tokens, slots % tokens]


def scatter_rows(cache, slots, values):
    rows, tokens = qsa_rows(cache)
    slots = torch.as_tensor(slots, dtype=torch.long, device=cache.device)
    rows[slots // tokens, slots % tokens] = values


def rebase_rows(rows, matrix):
    keys = unpack_nibbles(rows[..., :KEY_BYTES])
    packed = rows[..., SCALE_AT:SCALE_AT + 4].contiguous().view(torch.float32)
    values = dequantize(keys, packed)
    moved = (values.to(matrix.dtype) @ matrix).to(torch.float32)
    codes, scales = quantize(moved)
    out = rows.clone()
    out[..., :KEY_BYTES] = pack_nibbles(codes)
    out[..., SCALE_AT:SCALE_AT + 4] = scales.contiguous().view(torch.uint8).view(*scales.shape[:-1], 4)
    return out


def rebase_qsa_pages(src_cache, src_slots, dst_cache, dst_slots, shift, matrix=None):
    rows = gather_rows(src_cache, src_slots)
    if shift:
        matrix = shift_matrix(shift).to(device=rows.device) if matrix is None else matrix
        rows = rebase_rows(rows, matrix)
    scatter_rows(dst_cache, dst_slots, rows)
    return rows.shape[0]


def rebase_compressed_rows(src_cache, src_rows, dst_cache, dst_rows, shift, rotary_dim=ROTARY_DIM, theta=ROPE_THETA,
                           stored=torch.float8_e4m3fn):
    width = src_cache.shape[-1]
    source = src_cache.view(torch.uint8).reshape(-1, width)
    target = dst_cache.view(torch.uint8).reshape(-1, width)
    src_rows = torch.as_tensor(src_rows, dtype=torch.long, device=src_cache.device)
    dst_rows = torch.as_tensor(dst_rows, dtype=torch.long, device=dst_cache.device)
    picked = source[src_rows]
    if shift:
        values = picked.view(stored).to(torch.float32).to(torch.float64)
        moved = rotate(values, torch.full(values.shape[:-1], float(shift), dtype=torch.float64), rotary_dim, theta)
        picked = moved.to(torch.float32).to(stored).view(torch.uint8)
    target[dst_rows] = picked
    return int(src_rows.numel())


def copy_state_pages(src_cache, src_blocks, dst_cache, dst_blocks):
    src_blocks = torch.as_tensor(src_blocks, dtype=torch.long, device=src_cache.device)
    dst_blocks = torch.as_tensor(dst_blocks, dtype=torch.long, device=dst_cache.device)
    dst_cache[dst_blocks] = src_cache[src_blocks]
    return int(src_blocks.numel())


def token_slots(block_ids, begin, length, block_size):
    positions = torch.arange(begin, begin + length, dtype=torch.long)
    blocks = torch.as_tensor(block_ids, dtype=torch.long)[positions // block_size]
    return blocks * block_size + positions % block_size


def group_rows(block_ids, begin, length, block_size, ratio=COMPRESS_RATIO):
    groups_per_block = block_size // ratio
    first, last = begin // ratio, (begin + length) // ratio
    groups = torch.arange(first, last, dtype=torch.long)
    blocks = torch.as_tensor(block_ids, dtype=torch.long)[groups // groups_per_block]
    return blocks * groups_per_block + groups % groups_per_block
