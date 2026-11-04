import dataclasses
import importlib
import os
import sys
import time

import torch

PACK_ENV = "TAAF_VLLM_GDN_STATE_Q8_PACK"
FUSED_MODULE = "vllm.taaf_gdn_fused"
HISPARSE_MODULE = "vllm.taaf_qsa_hisparse"
SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
SCHEDULER_CLASS = "Scheduler"
LIVE_MARK = "_taaf_q8_pack_live"
LIVE_EVERY_S = 60.0
LANES_AT = 110
MAMBA = "MambaSpec"
_PACK = {"said": set(), "layers": 0, "groups": 0, "block_bytes": 0, "began": None, "live_at": None}


def on():
    said = str(os.environ.get(PACK_ENV) or "").strip().lower()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{PACK_ENV}={said!r} is not one of '', '0', '1'")
    return said == "1"


def say(key, line):
    if key in _PACK["said"]:
        return
    _PACK["said"].add(key)
    print(line, flush=True)


def q8_bytes(temporal):
    heads, v_dim, k_dim = (int(one) for one in temporal[-3:])
    return heads * v_dim * k_dim + heads * v_dim * 4


def state_shape(layer, shapes):
    if not on():
        return shapes
    conv, temporal = shapes
    size = layer.get_state_dtype()[1].itemsize
    layer.taaf_q8_dims = tuple(int(one) for one in temporal)
    return conv, (-(-q8_bytes(temporal) // size),)


def spec(layer, made):
    if not on() or made is None:
        return made
    return dataclasses.replace(made, page_size_padded=None)


def full_view(container, heads, v_dim, k_dim):
    stride = int(container.stride(0))
    storage = container.untyped_storage().nbytes() // container.element_size()
    room = storage - container.storage_offset() - heads * v_dim * k_dim
    blocks = min(int(container.shape[0]), room // stride + 1) if room >= 0 else 0
    if blocks <= 0:
        raise RuntimeError(f"{PACK_ENV}: the KV pool leaves no room for a [{heads}, {v_dim}, {k_dim}] state view")
    return torch.as_strided(container, (blocks, heads, v_dim, k_dim), (stride, v_dim * k_dim, k_dim, 1),
                            container.storage_offset())


def bind(layer, fused=None):
    if not on():
        return None
    container = layer.kv_cache[1]
    heads, v_dim, k_dim = layer.taaf_q8_dims
    full = full_view(container, heads, v_dim, k_dim)
    layer.taaf_q8_full = full
    fused = fused or importlib.import_module(FUSED_MODULE)
    fused._Q8_FULL[(container.data_ptr(), int(container.shape[0]))] = full
    say("bind", f"GDN_STATE_Q8_PACK bound: {getattr(layer, 'prefix', '?')} keeps its int8 state in "
                f"{container[0].numel() * container.element_size()} bytes a block (codes, then row scales) next to "
                f"its convolution window; the cache copies and zeroing see only those bytes, the Q8 kernels read "
                f"them through a {list(full.shape)} view ({int(container.shape[0]) - int(full.shape[0])} trailing "
                f"pool block(s) reachable by pointer only)")
    return full


def group_page(group, name):
    held = getattr(group.kv_cache_spec, "kv_cache_specs", None)
    return int((held[name] if held is not None else group.kv_cache_spec).page_size_bytes)


def leaves(spec):
    held = getattr(spec, "kv_cache_specs", None)
    return list(held.values()) if held is not None else [spec]


def is_mamba(spec):
    return any(kind.__name__ == MAMBA for kind in type(spec).__mro__)


def holds_mamba(group):
    found = leaves(group.kv_cache_spec)
    return bool(found) and all(is_mamba(one) for one in found)


def same(left, right):
    return type(left.kv_cache_spec) is type(right.kv_cache_spec) and leaves(left.kv_cache_spec) == leaves(
        right.kv_cache_spec) and all(getattr(left, field.name) == getattr(right, field.name)
                                     for field in dataclasses.fields(left)
                                     if field.name not in ("layer_names", "kv_cache_spec"))


def joined(held, group):
    spec = held.kv_cache_spec
    if getattr(spec, "kv_cache_specs", None) is not None:
        spec = dataclasses.replace(spec, kv_cache_specs={**spec.kv_cache_specs, **group.kv_cache_spec.kv_cache_specs})
    return dataclasses.replace(held, layer_names=[*held.layer_names, *group.layer_names], kv_cache_spec=spec)


def single(group, name):
    spec = group.kv_cache_spec
    if getattr(spec, "kv_cache_specs", None) is not None:
        spec = dataclasses.replace(spec, kv_cache_specs={name: spec.kv_cache_specs[name]})
    return dataclasses.replace(group, layer_names=[name], kv_cache_spec=spec)


def scheduler_view(groups):
    return [dataclasses.replace(group, kv_cache_spec=leaves(group.kv_cache_spec)[0]) for group in groups]


def pair(groups):
    if not on():
        return groups
    out, held, layers = [], None, 0
    for group in groups:
        if not holds_mamba(group) or len(group.layer_names) != 1:
            if held is not None:
                out.append(held)
                held = None
            out.append(group)
            continue
        layers += 1
        if held is not None and same(held, group):
            out.append(joined(held, group))
            held = None
            continue
        if held is not None:
            out.append(held)
        held = group
    if held is not None:
        out.append(held)
    _PACK["layers"] = layers
    _PACK["groups"] = sum(1 for group in out if holds_mamba(group))
    kinds = sorted({type(group.kv_cache_spec).__name__ for group in out if holds_mamba(group)})
    say("pair", f"GDN_STATE_Q8_PACK groups: {layers} single-layer Mamba groups become {_PACK['groups']} "
                f"({', '.join(kinds) or 'none'}), two GDN layers of a lane sharing each state block")
    return out


def unpacked(groups):
    out = []
    for group in groups:
        if holds_mamba(group) and len(group.layer_names) > 1:
            out.extend(single(group, name) for name in group.layer_names)
        else:
            out.append(group)
    return out


def lanes_line(groups, num_blocks):
    try:
        capacity = importlib.import_module(HISPARSE_MODULE).admit_capacity
    except Exception as failure:
        return f"lanes that fit not computed ({type(failure).__name__}: {failure})"
    packed = capacity(scheduler_view(groups), num_blocks)
    plain = capacity(scheduler_view(unpacked(groups)), num_blocks)
    return "lanes that fit with pages released per chunk, unpacked -> packed: " + ", ".join(
        f"{one['tokens']} tokens {two['real_lanes']} -> {one['real_lanes']} "
        f"({two['real_lanes'] * one['tokens']:,} -> {one['real_lanes'] * one['tokens']:,} tokens)"
        for one, two in zip(packed, plain))


def check(groups, bytes_per_block, num_blocks):
    if not on():
        return None
    mamba = [group for group in groups if holds_mamba(group)]
    if not mamba:
        raise RuntimeError(f"{PACK_ENV}=1 found no Mamba cache group to pack among "
                           f"{sorted({type(group.kv_cache_spec).__name__ for group in groups})}")
    others = [sum(group_page(group, name) for name in group.layer_names) for group in groups
              if not holds_mamba(group)]
    widest = max(sum(group_page(group, name) for name in group.layer_names) for group in mamba)
    room = max(others) if others else 0
    if widest > room:
        raise RuntimeError(f"{PACK_ENV}=1: a packed state block needs {widest} bytes, more than the {room} the "
                           f"pool block holds for the other groups, so packing would widen every pool block")
    layers = sum(len(group.layer_names) for group in mamba)
    saved = layers - len(mamba)
    _PACK.update(layers=layers, groups=len(mamba), block_bytes=int(bytes_per_block))
    say("pool", f"GDN_STATE_Q8_PACK pool: {layers} GDN layers keep their int8 states in {len(mamba)} Mamba groups "
                f"({widest} of {bytes_per_block} bytes a pool block), so a lane holds {len(mamba)} state blocks "
                f"instead of {layers}: {saved} blocks ({saved * bytes_per_block / 2 ** 20:.1f} MiB) less a lane, "
                f"{saved * LANES_AT} of {num_blocks} pool blocks ({100.0 * saved * LANES_AT / max(num_blocks, 1):.1f}%) "
                f"at {LANES_AT} lanes; {lanes_line(groups, num_blocks)}")
    arm_live()
    return saved


def arm_live():
    try:
        scheduler_class = getattr(importlib.import_module(SCHEDULER_MODULE), SCHEDULER_CLASS)
    except Exception as failure:
        say("live-missing", f"GDN_STATE_Q8_PACK live line not armed ({type(failure).__name__}: {failure})")
        return False
    if getattr(scheduler_class, LIVE_MARK, False):
        return False
    build = scheduler_class._build_kv_connector_meta

    def counted_build(self, connector, scheduler_output):
        live(self)
        return build(self, connector, scheduler_output)

    scheduler_class._build_kv_connector_meta = counted_build
    setattr(scheduler_class, LIVE_MARK, True)
    return True


def packed_lanes(scheduler):
    managers = [manager for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
                if all(is_mamba(one) for one in leaves(manager.kv_cache_spec))]
    if not managers:
        return 0
    return sum(1 for blocks in managers[0].req_to_blocks.values() if any(not block.is_null for block in blocks))


def live(scheduler, now=None):
    now = time.monotonic() if now is None else now
    if _PACK["began"] is None:
        _PACK["began"] = now
    if _PACK["live_at"] is not None and now - _PACK["live_at"] < LIVE_EVERY_S:
        return None
    _PACK["live_at"] = now
    lanes = packed_lanes(scheduler)
    saved = lanes * (_PACK["layers"] - _PACK["groups"])
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    line = (f"GDN_STATE_Q8_PACK live [{(now - _PACK['began']) / 60.0:.1f} min]: {lanes} lanes hold their GDN states "
            f"in {_PACK['groups']} blocks each instead of {_PACK['layers']}; packing keeps {saved} pool blocks "
            f"({saved * _PACK['block_bytes'] / 2 ** 30:.2f} GiB) for KV pages; pool free {free} blocks")
    print(line, file=sys.stderr, flush=True)
    return line
