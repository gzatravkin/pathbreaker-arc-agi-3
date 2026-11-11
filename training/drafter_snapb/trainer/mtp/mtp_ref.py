from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

FP8_MAX = 448.0
PREFIX = "mtp."
EXPERT_FUSED = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
EXPERT_SPLIT = re.compile(
    r"^mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_inv)$")
INDEXER = re.compile(r"\.self_attn\.indexer\.")
SCALE_SUFFIXES = ("weight_scale", "weight_scale_inv")


@dataclass
class Shape:
    hidden: int
    hc: int
    lowrank: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float
    experts: int
    top_k: int
    inter: int
    shared_inter: int
    index_heads: int
    index_dim: int
    budget: int
    ratio: int
    norm_topk: bool = True
    layers: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Shape":
        text = config.get("text_config") or config
        mtp = text.get("mtp") or {}
        head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
        factor = float(text.get("partial_rotary_factor")
                       or (text.get("rope_parameters") or {}).get("partial_rotary_factor") or 1.0)
        theta = float(mtp.get("rope_theta") or (text.get("rope_parameters") or {}).get("rope_theta")
                      or text.get("rope_theta") or 10000.0)
        return cls(hidden=int(text["hidden_size"]), hc=int(text.get("hc_count") or 1),
                   lowrank=int(text.get("hc_lowrank") or 1), heads=int(text["num_attention_heads"]),
                   kv_heads=int(text["num_key_value_heads"]), head_dim=head_dim,
                   rotary_dim=int(head_dim * factor), rope_theta=theta,
                   eps=float(text.get("rms_norm_eps") or 1e-6), experts=int(text["num_experts"]),
                   top_k=int(text["num_experts_per_tok"]), inter=int(text["moe_intermediate_size"]),
                   shared_inter=int(text.get("shared_expert_intermediate_size") or 0),
                   index_heads=int(text.get("indexer_n_heads") or 0),
                   index_dim=int(text.get("indexer_head_dim") or 0),
                   budget=int(text.get("indexer_budget") or 0),
                   ratio=int(text.get("indexer_compress_ratio") or 1),
                   norm_topk=bool(text.get("norm_topk_prob", True)),
                   layers=int(text.get("mtp_num_hidden_layers") or mtp.get("num_hidden_layers") or 1))


def safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        head = json.loads(handle.read(size).decode("utf-8"))
    head.pop("__metadata__", None)
    return head, 8 + size


def mtp_inventory(model_dir: Path) -> dict[str, dict[str, Any]]:
    index = model_dir / "model.safetensors.index.json"
    files = (sorted({name for key, name in json.loads(index.read_text(encoding="utf-8"))["weight_map"].items()
                     if key.startswith(PREFIX)}) if index.is_file()
             else [one.name for one in sorted(model_dir.glob("*.safetensors"))])
    found: dict[str, dict[str, Any]] = {}
    for name in files:
        head, _ = safetensors_header(model_dir / name)
        for key, spec in head.items():
            if key.startswith(PREFIX):
                found[key] = {"file": name, "dtype": spec["dtype"], "shape": list(spec["shape"])}
    return found


def read_tensors(model_dir: Path, names: list[str], inventory: dict[str, dict[str, Any]],
                 device: str = "cpu") -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(inventory[name]["file"], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for file, wanted in by_file.items():
        with safe_open(str(model_dir / file), framework="pt", device=device) as handle:
            for name in wanted:
                out[name] = handle.get_tensor(name)
    return out


def expand_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scale = scale.float()
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(rows, cols)
    if scale.dim() == 1 or (scale.dim() == 2 and scale.shape[1] == 1 and scale.shape[0] == rows):
        return scale.reshape(rows, 1).expand(rows, cols)
    if scale.dim() != 2:
        raise ValueError(f"unsupported scale shape {tuple(scale.shape)}")
    block_rows = math.ceil(rows / scale.shape[0])
    block_cols = math.ceil(cols / scale.shape[1])
    full = scale.repeat_interleave(block_rows, 0).repeat_interleave(block_cols, 1)
    return full[:rows, :cols]


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, shape: list[int], group: int) -> torch.Tensor:
    rows, cols = int(shape[0]), int(shape[1])
    shifts = torch.arange(8, dtype=torch.int32, device=packed.device) * 4
    nibbles = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    values = nibbles.reshape(packed.shape[0], -1)[:rows, :cols].to(torch.float32) - 8.0
    scales = scale.float().repeat_interleave(int(group), dim=1)[:rows, :cols]
    return (values * scales).to(torch.bfloat16)


def pack_int4(weight: torch.Tensor, group: int,
              scale_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    group = int(group)
    if cols % group or cols % 8:
        raise ValueError(f"{cols} columns do not split into groups of {group} and words of 8 nibbles")
    grouped = weight.float().reshape(rows, cols // group, group)
    scale = (grouped.abs().amax(-1) / 7.0).clamp(min=1e-12).to(scale_dtype)
    levels = (grouped / scale.float()[..., None]).round().clamp(-8, 7).to(torch.int32) + 8
    nibbles = levels.reshape(rows, cols // 8, 8)
    shifts = torch.arange(8, dtype=torch.int32, device=weight.device) * 4
    words = (nibbles << shifts).sum(-1, dtype=torch.int64)
    packed = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return packed, scale, torch.tensor([rows, cols], dtype=torch.int64)


def int4_roundtrip(weight: torch.Tensor, group: int) -> torch.Tensor:
    packed, scale, shape = pack_int4(weight, group)
    return unpack_int4(packed, scale, shape.tolist(), group)


def dequantize(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is None:
        return weight.float()
    rows, cols = weight.shape
    return weight.float() * expand_scale(scale, rows, cols)


def quantize_like(weight: torch.Tensor, scale_like: torch.Tensor,
                  dtype: torch.dtype = torch.float8_e4m3fn) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float()
    rows, cols = weight.shape
    if scale_like.numel() == 1:
        amax = weight.abs().amax().reshape(scale_like.shape)
    elif scale_like.dim() == 1 or (scale_like.dim() == 2 and scale_like.shape[1] == 1
                                   and scale_like.shape[0] == rows):
        amax = weight.abs().amax(dim=1).reshape(scale_like.shape)
    else:
        grid_rows, grid_cols = scale_like.shape
        block_rows = math.ceil(rows / grid_rows)
        block_cols = math.ceil(cols / grid_cols)
        padded = F.pad(weight.abs(), (0, grid_cols * block_cols - cols, 0, grid_rows * block_rows - rows))
        amax = padded.reshape(grid_rows, block_rows, grid_cols, block_cols).amax(dim=(1, 3))
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    quantized = (weight / expand_scale(scale, rows, cols)).clamp(-FP8_MAX, FP8_MAX).to(dtype)
    return quantized, scale.to(scale_like.dtype)


def gemma_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def grouped_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int) -> torch.Tensor:
    shape = x.shape
    xf = x.float().unflatten(-1, (groups, shape[-1] // groups))
    out = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).flatten(-2)
    return (out * (1.0 + weight.float())).to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim))
    angles = positions.float()[:, None] * inv[None, :]
    shape = [angles.shape[0]] + [1] * (x.dim() - 2) + [half]
    cos = angles.cos().reshape(shape)
    sin = angles.sin().reshape(shape)
    xf = x.float()
    first, second = xf[..., :half], xf[..., half:rotary_dim]
    turned = torch.cat((first * cos - second * sin, second * cos + first * sin, xf[..., rotary_dim:]), -1)
    return turned.to(x.dtype)


KV_QUANTS = ("", "int4_per_token_head")
INDEX_FP8 = "fp8_index"
_RHT_SIGNS: dict[int, torch.Tensor] = {}


def rht_signs(d: int) -> torch.Tensor:
    if d not in _RHT_SIGNS:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x9E3779B9)
        _RHT_SIGNS[d] = 2.0 * torch.bernoulli(torch.full((d,), 0.5, device="cpu"), generator=gen) - 1.0
    return _RHT_SIGNS[d]


def hadamard(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    h = 1
    while h < d:
        pairs = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        x = torch.stack([pairs[..., 0, :] + pairs[..., 1, :], pairs[..., 0, :] - pairs[..., 1, :]], dim=-2)
        x = x.reshape(*x.shape[:-3], d)
        h <<= 1
    return x


def half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def int4_kv_roundtrip(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    signs = rht_signs(d).to(x.device)
    rotated = hadamard(x.float() * signs).to(x.dtype).float()
    low = rotated.amin(-1, keepdim=True)
    high = rotated.amax(-1, keepdim=True)
    scale = torch.clamp((high - low) / 15.0, min=1e-6)
    zero = torch.clamp(half_away(-low / scale), 0.0, 15.0)
    codes = torch.clamp(half_away(rotated * (1.0 / scale) + zero), 0.0, 15.0)
    stored = (scale.view(torch.int32) & -16).view(torch.float32)
    return (hadamard((codes - zero) * stored) * signs / d).to(x.dtype)


def index_fp8(mode: str) -> bool:
    return INDEX_FP8 in mode.split("+")


def index_view(x: torch.Tensor, mode: str) -> torch.Tensor:
    if not index_fp8(mode):
        return x
    return x.to(torch.bfloat16).to(torch.float8_e4m3fn).to(x.dtype)


def kv_view(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = "+".join(part for part in mode.split("+") if part and part != INDEX_FP8)
    if not mode:
        return x
    if mode == "int4_per_token_head":
        return int4_kv_roundtrip(x)
    raise ValueError(f"unknown KV view {mode!r}; one of {KV_QUANTS}")


def key(name: str) -> str:
    return name.replace(".", "__")


class MTPHead(nn.Module):
    def __init__(self, shape: Shape, dense: dict[str, torch.Tensor], gate_up: torch.Tensor,
                 down: torch.Tensor, train_experts: bool = False, layer: int = 0) -> None:
        super().__init__()
        self.shape = shape
        self.layer = layer
        self.kv_quant = ""
        self.allowed: torch.Tensor | None = None
        self.names = sorted(dense)
        self.w = nn.ParameterDict({key(name): nn.Parameter(tensor.float().clone())
                                   for name, tensor in dense.items()})
        for name in self.names:
            if INDEXER.search(name):
                self.w[key(name)].requires_grad_(False)
        self.trains_experts = bool(train_experts)
        if train_experts:
            self.gate_up_each = nn.ParameterList([nn.Parameter(one.float().clone()) for one in gate_up])
            self.down_each = nn.ParameterList([nn.Parameter(one.float().clone()) for one in down])
        else:
            self.register_buffer("gate_up", gate_up.to(torch.bfloat16).clone(), persistent=False)
            self.register_buffer("down", down.to(torch.bfloat16).clone(), persistent=False)

    def expert_weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.trains_experts:
            return self.gate_up_each[expert], self.down_each[expert]
        return self.gate_up[expert], self.down[expert]

    def experts_stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.trains_experts:
            with torch.no_grad():
                return (torch.stack([one.detach() for one in self.gate_up_each]),
                        torch.stack([one.detach() for one in self.down_each]))
        return self.gate_up.detach(), self.down.detach()

    def expert_parameters(self) -> list[nn.Parameter]:
        return [*self.gate_up_each, *self.down_each] if self.trains_experts else []

    def p(self, tail: str) -> torch.Tensor:
        return self.w[key(f"mtp.{tail}")]

    def l(self, tail: str) -> torch.Tensor:
        return self.w[key(f"mtp.layers.{self.layer}.{tail}")]

    def trainable(self) -> list[nn.Parameter]:
        return [one for one in self.parameters() if one.requires_grad]

    def dense_state(self) -> dict[str, torch.Tensor]:
        return {name: self.w[key(name)].detach() for name in self.names}

    def gate_mix(self, prefix: str, xn: torch.Tensor, inject: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        s = self.shape
        down = self.p(f"{prefix}.input_mix_weight_down.weight") if prefix == "hyper_connection_mixer" \
            else self.l(f"{prefix}.input_mix_weight_down.weight")
        up = self.p(f"{prefix}.input_mix_weight_up.weight") if prefix == "hyper_connection_mixer" \
            else self.l(f"{prefix}.input_mix_weight_up.weight")
        lora = F.linear(xn, down.to(xn.dtype)).float() / s.hc
        lora = (lora * torch.sigmoid(lora)).to(xn.dtype)
        gate = F.linear(lora, up.to(xn.dtype))
        mixed = (torch.sigmoid(gate.float()) * xn.float()).unflatten(-1, (s.hc, s.hidden)).mean(-2)
        injection = None
        if inject:
            injection = F.linear(xn, self.l(f"{prefix}.block_inject_weight.weight").to(xn.dtype))
        return mixed.to(xn.dtype), injection

    def combine(self, residual: torch.Tensor, block: torch.Tensor, injection: torch.Tensor | None) -> torch.Tensor:
        s = self.shape
        res = residual.float().unflatten(-1, (s.hc, s.hidden))
        if injection is None:
            out = res + block.float()[..., None, :]
        else:
            weight = 2.0 * torch.sigmoid(injection.float() / s.hc)
            out = res + block.float()[..., None, :] * weight[..., :, None]
        return out.flatten(-2).to(residual.dtype)

    def norm_of(self, prefix: str) -> torch.Tensor:
        return self.p(f"{prefix}.hc_norm.weight") if prefix == "hyper_connection_mixer" \
            else self.l(f"{prefix}.hc_norm.weight")

    def front(self, hidden: torch.Tensor, embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.shape
        dtype = embeds.dtype
        e = gemma_norm(embeds, self.p("pre_fc_norm_embedding.weight"), s.eps)
        e = F.linear(e, self.p("fc_embedding.weight").to(dtype))
        h = gemma_norm(hidden.to(dtype), self.p("pre_fc_norm_hidden.weight"), s.eps)
        h = F.linear(h.unflatten(-1, (s.hc, s.hidden)), self.p("fc_hidden.weight").to(dtype)).flatten(-2)
        h = self.combine(h, e, None)
        xn = grouped_norm(h, self.norm_of("attn_hyper_connection"), s.eps, s.hc)
        block, injection = self.gate_mix("attn_hyper_connection", xn, True)
        return {"h": h, "block": block, "inj": injection}

    def qkv(self, block: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.shape
        dtype = block.dtype
        qg = F.linear(block, self.l("self_attn.q_proj.weight").to(dtype)).unflatten(-1, (s.heads, 2 * s.head_dim))
        q, gate = qg[..., :s.head_dim], qg[..., s.head_dim:]
        k = F.linear(block, self.l("self_attn.k_proj.weight").to(dtype)).unflatten(-1, (s.kv_heads, s.head_dim))
        v = F.linear(block, self.l("self_attn.v_proj.weight").to(dtype)).unflatten(-1, (s.kv_heads, s.head_dim))
        q = rope(gemma_norm(q, self.l("self_attn.q_norm.weight"), s.eps), positions, s.rotary_dim, s.rope_theta)
        k = rope(gemma_norm(k, self.l("self_attn.k_norm.weight"), s.eps), positions, s.rotary_dim, s.rope_theta)
        return {"q": q, "k": k, "v": v, "gate": gate}

    def index_qk(self, block: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.shape
        with torch.no_grad():
            projected = F.linear(block.detach(), self.l("self_attn.indexer.index_qk_proj.weight").to(block.dtype))
            q = projected[..., :s.index_heads * s.index_dim].unflatten(-1, (s.index_heads, s.index_dim))
            raw = projected[..., s.index_heads * s.index_dim:]
            q = rope(gemma_norm(q, self.l("self_attn.indexer.q_layernorm.weight"), s.eps), positions,
                     min(s.rotary_dim, s.index_dim), s.rope_theta)
        return q, raw

    def compressed_keys(self, raw: torch.Tensor, first: int) -> torch.Tensor:
        s = self.shape
        with torch.no_grad():
            blocks = raw.shape[0] // s.ratio
            pooled = raw[:blocks * s.ratio].float().reshape(blocks, s.ratio, -1).mean(1).to(raw.dtype)
            pooled = gemma_norm(pooled, self.l("self_attn.indexer.k_layernorm.weight"), s.eps)
            starts = first + torch.arange(blocks, device=raw.device) * s.ratio
            return rope(pooled[:, None, :], starts, min(s.rotary_dim, s.index_dim), s.rope_theta)[:, 0, :]

    def selection(self, query_pos: torch.Tensor, index_q: torch.Tensor, ckeys: torch.Tensor,
                  key_pos: torch.Tensor, limit: torch.Tensor | None = None) -> torch.Tensor:
        s = self.shape
        visible = (query_pos + 1) // s.ratio
        if limit is not None:
            visible = torch.minimum(visible, (limit + 1) // s.ratio)
        allowed = key_pos[None, :] <= query_pos[:, None]
        if s.budget <= 0 or s.index_heads == 0 or int(visible.max()) <= s.budget // s.ratio:
            return allowed
        block_topk = s.budget // s.ratio
        with torch.no_grad():
            scores = torch.einsum("qhd,cd->qhc", index_view(index_q, self.kv_quant).float(),
                                  index_view(ckeys, self.kv_quant).float()).clamp(min=0).sum(1)
            first_block = int(key_pos[0]) // s.ratio
            block_ids = first_block + torch.arange(ckeys.shape[0], device=scores.device)
            scores = scores.masked_fill(block_ids[None, :] >= visible[:, None], float("-inf"))
            take = min(block_topk, scores.shape[1])
            picked = scores.topk(take, dim=1).indices
            chosen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, picked, True)
            chosen &= block_ids[None, :] < visible[:, None]
            key_block = key_pos // s.ratio - first_block
            in_block = torch.zeros_like(allowed)
            valid = (key_block >= 0) & (key_block < ckeys.shape[0])
            in_block[:, valid] = chosen[:, key_block[valid]]
            tail = key_pos[None, :] >= (visible * s.ratio)[:, None]
            few = visible <= block_topk
            return allowed & (in_block | tail | few[:, None])

    def attend(self, q: torch.Tensor, gate: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor,
               mask: torch.Tensor, own_k: list[torch.Tensor], own_v: list[torch.Tensor]) -> torch.Tensor:
        s = self.shape
        group = s.heads // s.kv_heads
        qh = q.unflatten(1, (s.kv_heads, group)).float()
        scale = s.head_dim ** -0.5
        scores = torch.einsum("qgnd,tgd->qgnt", qh, k0.float()) * scale
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        extra = [torch.einsum("qgnd,qgd->qgn", qh, one.float())[..., None] * scale for one in own_k]
        if extra:
            scores = torch.cat([scores, *extra], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("qgnt,tgd->qgnd", probs[..., :k0.shape[0]], v0.float())
        for index, one in enumerate(own_v):
            out = out + probs[..., k0.shape[0] + index][..., None] * one.float()[:, :, None, :]
        out = out.flatten(1, 2) * torch.sigmoid(gate.float())
        return out.flatten(-2).to(q.dtype)

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shape
        dtype = x.dtype
        logits = F.linear(x, self.l("mlp.gate.weight").to(dtype)).float()
        if self.allowed is not None:
            logits = logits.masked_fill(~self.allowed.to(logits.device), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        weights, chosen = probs.topk(s.top_k, dim=-1)
        if s.norm_topk:
            weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros(x.shape[0], s.hidden, dtype=torch.float32, device=x.device)
        flat = chosen.reshape(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=s.experts).tolist()
        rows = (torch.arange(x.shape[0], device=x.device)[:, None].expand_as(chosen)).reshape(-1)[order]
        scales = weights.reshape(-1)[order]
        start = 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            idx = rows[start:start + count]
            part = x[idx]
            gate_up, down = self.expert_weights(expert)
            gu = F.linear(part, gate_up.to(dtype))
            act = F.silu(gu[:, :s.inter].float()) * gu[:, s.inter:].float()
            y = F.linear(act.to(dtype), down.to(dtype)).float()
            out = out.index_add(0, idx, y * scales[start:start + count, None])
            start += count
        if s.shared_inter:
            g = F.linear(x, self.l("mlp.shared_expert.gate_proj.weight").to(dtype)).float()
            u = F.linear(x, self.l("mlp.shared_expert.up_proj.weight").to(dtype)).float()
            y = F.linear((F.silu(g) * u).to(dtype), self.l("mlp.shared_expert.down_proj.weight").to(dtype)).float()
            sg = torch.sigmoid(F.linear(x, self.l("mlp.shared_expert_gate.weight").to(dtype)).float())
            out = out + y * sg
        return out.to(dtype)

    def back(self, front: dict[str, torch.Tensor], attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.shape
        dtype = attn.dtype
        attn_out = F.linear(attn, self.l("self_attn.o_proj.weight").to(dtype))
        h2 = self.combine(front["h"], attn_out, front["inj"])
        xn2 = grouped_norm(h2, self.norm_of("mlp_hyper_connection"), s.eps, s.hc)
        block2, inj2 = self.gate_mix("mlp_hyper_connection", xn2, True)
        mlp_out = self.moe(block2)
        multi = self.combine(h2, mlp_out, inj2)
        xn3 = grouped_norm(multi, self.norm_of("hyper_connection_mixer"), s.eps, s.hc)
        sample, _ = self.gate_mix("hyper_connection_mixer", xn3, False)
        return sample, multi

    def window_keys(self, hidden: torch.Tensor, ids: torch.Tensor, positions: torch.Tensor, embed_ids,
                    start: int, stop: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ks, vs, raws = [], [], []
        for one in range(start, stop, max(1, chunk)):
            end = min(stop, one + max(1, chunk))
            front = self.front(hidden[one:end], embed_ids(ids[one + 1:end + 1]))
            proj = self.qkv(front["block"], positions[one:end])
            _, raw = self.index_qk(front["block"], positions[one:end])
            ks.append(proj["k"])
            vs.append(proj["v"])
            raws.append(raw)
        return torch.cat(ks, 0), torch.cat(vs, 0), torch.cat(raws, 0)

    def forward_steps(self, hidden: torch.Tensor, ids: torch.Tensor, embed: nn.Module | torch.Tensor,
                      bases: torch.Tensor, steps: int, first_pos: int = 0, chunk: int = 256,
                      share_index: bool = False, dtype: torch.dtype = torch.bfloat16, grad_from: int = 0,
                      far_chunk: int = 4096) -> list[torch.Tensor]:
        s = self.shape
        count = ids.shape[0]
        positions = torch.arange(count - 1, device=ids.device) + first_pos

        def embed_ids(tokens: torch.Tensor) -> torch.Tensor:
            if isinstance(embed, torch.Tensor):
                return F.embedding(tokens, embed).to(dtype)
            return embed(tokens).to(dtype)

        local = bases - first_pos
        hidden = hidden[:count].to(dtype)
        if grad_from > 0:
            cut = min(int(grad_from), count - 2)
            with torch.no_grad():
                far = self.window_keys(hidden, ids, positions, embed_ids, 0, cut, far_chunk)
            near = self.window_keys(hidden, ids, positions, embed_ids, cut, count - 1, count)
            k0, v0, raw0 = (torch.cat([one, two], 0) for one, two in zip(far, near))
            base_front = self.front(hidden[local], embed_ids(ids[local + 1]))
            base_proj = self.qkv(base_front["block"], positions[local])
            base_index_q, _ = self.index_qk(base_front["block"], positions[local])
        else:
            front0 = self.front(hidden[:count - 1], embed_ids(ids[1:]))
            proj0 = self.qkv(front0["block"], positions)
            k0, v0 = proj0["k"], proj0["v"]
            index_q0, raw0 = self.index_qk(front0["block"], positions)
            base_front = {name: value[local] for name, value in front0.items() if value is not None}
            base_proj = {"q": proj0["q"][local], "gate": proj0["gate"][local]}
            base_index_q = index_q0[local]
        base_front.setdefault("inj", None)
        k0, v0 = kv_view(k0, self.kv_quant), kv_view(v0, self.kv_quant)
        ckeys = self.compressed_keys(raw0, first_pos) if s.index_heads and s.budget else None
        outs: list[torch.Tensor] = []
        sel_cache: dict[int, torch.Tensor] = {}
        chain_h: torch.Tensor | None = None
        own_k: list[torch.Tensor] = []
        own_v: list[torch.Tensor] = []
        for step in range(steps):
            if step == 0:
                front = base_front
                q, gate = base_proj["q"], base_proj["gate"]
                index_q = base_index_q
            else:
                token_pos = (local + step + 1).clamp(max=count - 1)
                front = self.front(chain_h, embed_ids(ids[token_pos]))
                pos = positions[local] + step
                proj = self.qkv(front["block"], pos)
                q, gate = proj["q"], proj["gate"]
                own_k = own_k + [kv_view(proj["k"], self.kv_quant)]
                own_v = own_v + [kv_view(proj["v"], self.kv_quant)]
                index_q, _ = self.index_qk(front["block"], pos)
            attn_parts = []
            for start in range(0, local.shape[0], chunk):
                rows = slice(start, start + chunk)
                key_index = 0 if share_index else step
                if (start, key_index) not in sel_cache:
                    qpos = positions[local[rows]] + key_index
                    ref_q = base_index_q[rows] if share_index else index_q[rows]
                    sel = self.selection(qpos, ref_q, ckeys, positions, positions[local[rows]]) \
                        if ckeys is not None else positions[None, :] <= qpos[:, None]
                    sel &= positions[None, :] <= positions[local[rows]][:, None]
                    sel_cache[(start, key_index)] = sel
                mask = sel_cache[(start, key_index)]
                extra_k = [] if share_index else [one[rows] for one in own_k]
                extra_v = [] if share_index else [one[rows] for one in own_v]
                if torch.is_grad_enabled():
                    attn_parts.append(checkpoint(self.attend, q[rows], gate[rows], k0, v0, mask, extra_k,
                                                 extra_v, use_reentrant=False))
                else:
                    attn_parts.append(self.attend(q[rows], gate[rows], k0, v0, mask, extra_k, extra_v))
            attn = torch.cat(attn_parts, 0)
            sample, chain_h = self.back(front, attn)
            outs.append(sample)
        return outs


def load_head(model_dir: Path, config: dict[str, Any], device: str = "cpu",
              train_experts: bool = False) -> tuple[MTPHead, dict[str, Any]]:
    inventory = mtp_inventory(model_dir)
    if not inventory:
        raise RuntimeError(f"{model_dir} carries no mtp.* tensors")
    tensors = read_tensors(model_dir, sorted(inventory), inventory, device)
    head, report = build_head(tensors, Shape.from_config(config), train_experts)
    report["inventory"] = inventory
    return head, report


def build_head(tensors: dict[str, torch.Tensor], shape: Shape,
               train_experts: bool = False) -> tuple[MTPHead, dict[str, Any]]:
    dense: dict[str, torch.Tensor] = {}
    fused: dict[str, torch.Tensor] = {}
    split: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    formats: dict[str, str] = {}
    for name, tensor in tensors.items():
        found = EXPERT_FUSED.match(name)
        if found:
            fused[found.group(2)] = tensor
            formats[name] = str(tensor.dtype)
            continue
        found = EXPERT_SPLIT.match(name)
        if found:
            split.setdefault((int(found.group(2)), found.group(3)), {})[found.group(4)] = tensor
            continue
        if name.endswith(SCALE_SUFFIXES) or name.endswith("input_scale"):
            base = name.rsplit(".", 1)[0] + ".weight"
            if base in tensors:
                continue
        scale = next((tensors[name.rsplit(".", 1)[0] + "." + one] for one in SCALE_SUFFIXES
                      if name.endswith(".weight") and name.rsplit(".", 1)[0] + "." + one in tensors), None)
        dense[name] = dequantize(tensor, scale) if scale is not None else tensor.float()
    if fused:
        gate_up = fused["gate_up_proj"].float()
        down = fused["down_proj"].float()
        layout = "fused"
    elif split:
        experts = 1 + max(index for index, _ in split)
        gate_up = torch.empty(experts, 2 * shape.inter, shape.hidden)
        down = torch.empty(experts, shape.hidden, shape.inter)
        for (expert, proj), parts in split.items():
            scale = next((parts[one] for one in SCALE_SUFFIXES if one in parts), None)
            value = dequantize(parts["weight"], scale)
            if proj == "gate_proj":
                gate_up[expert, :shape.inter] = value
            elif proj == "up_proj":
                gate_up[expert, shape.inter:] = value
            else:
                down[expert] = value
        layout = "split"
    else:
        raise RuntimeError("no routed-expert tensors found under mtp.*")
    shape.experts = gate_up.shape[0]
    head = MTPHead(shape, dense, gate_up, down, train_experts)
    return head, {"experts_layout": layout, "experts": int(gate_up.shape[0]), "dense": len(dense),
                  "trainable": sum(one.numel() for one in head.trainable())}


def export_tensors(head: MTPHead, source: dict[str, torch.Tensor],
                   include_experts: bool = False) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    state = head.dense_state()
    for name, value in state.items():
        original = source[name]
        if not head.w[key(name)].requires_grad:
            continue
        scale_name = next((name.rsplit(".", 1)[0] + "." + one for one in SCALE_SUFFIXES
                           if name.rsplit(".", 1)[0] + "." + one in source), None)
        if scale_name is not None:
            quantized, scale = quantize_like(value, source[scale_name], original.dtype)
            out[name] = quantized.contiguous()
            out[scale_name] = scale.contiguous()
        else:
            out[name] = value.to(original.dtype).reshape(original.shape).contiguous()
    if include_experts:
        gate_up, down = (one.float() for one in head.experts_stacked())
        inter = head.shape.inter
        for name, original in source.items():
            found = EXPERT_FUSED.match(name)
            if found:
                value = gate_up if found.group(2) == "gate_up_proj" else down
                out[name] = value.to(original.dtype).contiguous()
                continue
            found = EXPERT_SPLIT.match(name)
            if not found or found.group(4) != "weight":
                continue
            expert, proj = int(found.group(2)), found.group(3)
            value = (gate_up[expert, :inter] if proj == "gate_proj" else
                     gate_up[expert, inter:] if proj == "up_proj" else down[expert])
            base = name.rsplit(".", 1)[0]
            scale_name = next((base + "." + one for one in SCALE_SUFFIXES if base + "." + one in source), None)
            if scale_name is None:
                out[name] = value.to(original.dtype).contiguous()
            else:
                quantized, scale = quantize_like(value, source[scale_name], original.dtype)
                out[name] = quantized.contiguous()
                out[scale_name] = scale.contiguous()
    return out


def game_distribution(logits: torch.Tensor, temperature: float, top_k: int,
                      top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
    values, ids = logits.float().topk(top_k, dim=-1)
    probs = torch.softmax(values / temperature, dim=-1)
    before = probs.cumsum(-1) - probs
    probs = torch.where(before < top_p, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(-1, keepdim=True)
    return probs, ids


def acceptance(draft_logits: torch.Tensor, target_probs: torch.Tensor, target_ids: torch.Tensor,
               vocab_ids: torch.Tensor | None = None) -> torch.Tensor:
    best = draft_logits.argmax(-1)
    if vocab_ids is not None:
        best = vocab_ids[best]
    return (target_probs * (target_ids == best[:, None])).sum(-1)


def sampled_acceptance(draft_logits: torch.Tensor, target_probs: torch.Tensor, target_cols: torch.Tensor,
                       temperature: float) -> torch.Tensor:
    q = torch.softmax(draft_logits.float() / temperature, dim=-1)
    valid = target_cols >= 0
    picked = q.gather(-1, target_cols.clamp(min=0)) * valid
    return torch.minimum(target_probs.float(), picked).sum(-1)


def soft_ce(draft_logits: torch.Tensor, target_probs: torch.Tensor, target_cols: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(draft_logits.float(), dim=-1)
    valid = target_cols >= 0
    picked = logp.gather(-1, target_cols.clamp(min=0))
    mass = (target_probs * valid).sum(-1, keepdim=True).clamp(min=1e-9)
    return -((target_probs * valid / mass) * picked).sum(-1)


FP8_MAX = 448.0


def draft_conf(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits.float() / max(float(temperature), 1e-6), dim=-1).amax(-1)


def head_view(rows: torch.Tensor, view: str) -> torch.Tensor:
    held = rows.float()
    if view == "fp8row":
        scale = held.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    elif view == "fp8tensor":
        scale = held.abs().amax().clamp(min=1e-12) / FP8_MAX
    elif view.startswith("fp8g") and view[4:].isdigit() and held.shape[1] % int(view[4:]) == 0:
        group = int(view[4:])
        blocks = held.view(held.shape[0], -1, group)
        scale = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).view_as(held).to(rows.dtype)
    else:
        raise ValueError(f"unknown lm_head view {view!r}; it takes one of fp8row, fp8tensor, fp8g<group>")
    return ((held / scale).to(torch.float8_e4m3fn).float() * scale).to(rows.dtype)


@dataclass
class Tally:
    steps: int
    conditional: list[float] = field(default_factory=list)
    reached: list[float] = field(default_factory=list)
    bases: int = 0
    ceiling: float = 0.0

    def __post_init__(self) -> None:
        self.conditional = [0.0] * self.steps
        self.reached = [0.0] * self.steps

    def add(self, rates: list[torch.Tensor], best: torch.Tensor | None = None) -> None:
        alive = torch.ones_like(rates[0], dtype=torch.float64)
        for step, rate in enumerate(rates):
            rate = rate.double()
            alive = alive * rate
            self.conditional[step] += float(rate.sum())
            self.reached[step] += float(alive.sum())
        self.bases += int(rates[0].numel())
        if best is not None:
            self.ceiling += float(best.double().sum())

    def summary(self) -> dict[str, Any]:
        if not self.bases:
            return {"bases": 0}
        reached = [self.reached[i] / self.bases for i in range(self.steps)]
        return {"bases": self.bases,
                "teacher_forced": [round(self.conditional[i] / self.bases, 4) for i in range(self.steps)],
                "per_position": [round(one, 4) for one in reached],
                "tokens_per_step": round(1.0 + sum(reached), 4),
                "ceiling_first": round(self.ceiling / self.bases, 4)}
