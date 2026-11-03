import json
import os
import re
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ENV = "TAAF_VLLM_EXPERT_HOT"
GO_ENV = "TAAF_VLLM_EXPERT_HOT_GO"
DIR_ENV = "TAAF_VLLM_EXPERT_HOT_DIR"
STALL_ENV = "TAAF_VLLM_EXPERT_HOT_STALL_S"
STALL_DEFAULT = 90.0
GO_DEFAULT = "/tmp/taaf_expert_hot_go"
DIR_DEFAULT = "/var/tmp/taaf_expert_hot"
PLAN_NAME = "plan.json"
DONE_NAME = "done.json"
ASIDE_NAME = "aside.json"
STATUS_NAME = "status.json"
CUT_PLAN = "expert_cut_plan.json"
ORDER_NAME = "expert_usage.json"
STAGE_PREFIX = "stage"
MASKED = -30000.0
ALIGN = 512
LAYER_NAME = re.compile(r"^(?!.*mtp)(?:.*\.)?layers\.(\d+)\.mlp\.experts$")
EXPERT_TENSOR = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$")
SHARDS = {"gate_proj": ("w13_", "w1"), "up_proj": ("w13_", "w3"), "down_proj": ("w2_", "w2")}
B_NAMES = ("w13", "w2", "w13_scale", "w2_scale", "w13_zeros", "w2_zeros", "g1", "g2")

WORKER = {"layers": {}, "bound": False, "said": set(), "aside": "", "store": None}
ENGINE = {"state": "idle", "checked": 0.0, "plan": None, "pool": None, "first": 0, "end": 0, "claimed": set(),
          "began": 0.0, "said": 0.0, "applied": None, "moved_at": 0.0, "held_last": 0, "preempted": 0,
          "set": set(), "rows_open": 0, "target": 0}


def on():
    return str(os.environ.get(ENV, "")).strip() == "1"


def say(line):
    print(f"EXPERT_HOT {line}", flush=True)


def once(key, line):
    if key not in WORKER["said"]:
        WORKER["said"].add(key)
        say(line)


def record_aside(stage):
    try:
        out = store_dir()
        out.mkdir(parents=True, exist_ok=True)
        import traceback
        trace = traceback.format_exc()
        (out / ASIDE_NAME).write_text(json.dumps({"stage": stage, "why": WORKER["aside"],
                                                  "trace": trace[-1800:]}), encoding="utf-8")
        say(f"stood aside at {stage}, trace: {trace[-1800:]!r}")
    except OSError:
        pass


def store_dir():
    return Path(os.environ.get(DIR_ENV, "") or DIR_DEFAULT)


def go_path():
    return Path(os.environ.get(GO_ENV, "") or GO_DEFAULT)


class LayerState:
    def __init__(self, index, kept, extra, gate_extra, mask, map_b, specs, config):
        self.index = index
        self.kept = kept
        self.extra = extra
        self.gate_extra = gate_extra
        self.mask = mask
        self.map_b = map_b
        self.specs = specs
        self.config = config
        self.views = {}
        self.pending = None


def _moe_sources(model_runner):
    sources = []
    model = getattr(model_runner, "model", None)
    if model is None and hasattr(model_runner, "get_model"):
        try:
            model = model_runner.get_model()
        except Exception:
            model = None
    if model is not None and hasattr(model, "named_modules"):
        sources.append(("model modules", list(model.named_modules())))
    for owner in (getattr(model_runner, "compilation_config", None), _current_compilation_config()):
        context = getattr(owner, "static_forward_context", None)
        if context:
            sources.append(("forward context", list(context.items())))
    return sources


def _current_compilation_config():
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config().compilation_config
    except Exception:
        return None


def _runners(model_runner):
    seen = {}
    WORKER["found_by"] = {}
    for label, items in _moe_sources(model_runner):
        hits = 0
        for name, layer in items:
            found = LAYER_NAME.match(str(name))
            if found and hasattr(layer, "router") and hasattr(layer, "routed_experts"):
                hits += 1
                seen.setdefault(id(layer), (int(found.group(1)), layer))
        WORKER["found_by"][label] = WORKER["found_by"].get(label, 0) + hits
        if not hits:
            WORKER.setdefault("names", {})[label] = [str(n) for n, _ in items if "expert" in str(n)][:6]
    return list(seen.values())


def _served_dir(model_runner):
    return Path(model_runner.model_config.model)


def _usage_order(rows):
    found = []
    roots = [Path(__file__).resolve().parent, Path("/kaggle/input")]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in (ORDER_NAME, f"*/{ORDER_NAME}", f"*/*/{ORDER_NAME}", f"*/*/*/{ORDER_NAME}"):
            found += sorted(root.glob(pattern))
    for path in found:
        try:
            order = json.loads(path.read_text(encoding="utf-8"))
            if all(set(int(x) for x in order[str(layer)][:len(kept)]) == set(kept) for layer, kept in rows.items()):
                return {int(k): [int(x) for x in v] for k, v in order.items() if str(k).isdigit()}, str(path)
        except Exception:
            continue
    return None, f"no {ORDER_NAME} of {len(found)} found ranks the kept rows first"


def _index(served):
    index = served / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError(f"no safetensors index in {served}")
    return json.loads(index.read_text(encoding="utf-8"))["weight_map"]


class Shards:
    def __init__(self, served, weight_map):
        self.served = served
        self.weight_map = weight_map
        self.open = {}

    def get(self, name):
        from safetensors import safe_open
        file = self.weight_map[name]
        if file not in self.open:
            self.open[file] = safe_open(str((self.served / file).resolve()), framework="pt", device="cpu")
        return self.open[file].get_tensor(name)

    def close(self):
        self.open.clear()


def _sizes(runner):
    routed = runner.routed_experts
    config = getattr(runner, "moe_config", None)
    inter = getattr(routed, "intermediate_size_per_partition", None)
    if inter is None and config is not None:
        inter = getattr(config, "intermediate_size_per_partition", None)
    hidden = getattr(routed, "hidden_size", None)
    if hidden is None and config is not None:
        hidden = getattr(config, "hidden_dim", None) or getattr(config, "hidden_size", None)
    if inter is None or hidden is None:
        raise RuntimeError("cannot read the expert sizes from the layer")
    return int(inter), int(hidden)


def _temp_layer(runner, count):
    routed = runner.routed_experts
    inter, hidden = _sizes(runner)
    holder = torch.nn.Module()
    holder._loaded_expert_biases = set()
    holder.intermediate_size_per_partition = inter
    holder.hidden_size = hidden
    method = routed.quant_method
    dtype = getattr(routed, "params_dtype", None) or torch.bfloat16
    with torch.device(routed.w13_qweight.device):
        method.create_weights(holder, num_experts=count, hidden_size=hidden, intermediate_size_per_partition=inter,
                              params_dtype=dtype, weight_loader=routed.weight_loader)
    holder.intermediate_size_per_partition = inter
    return holder


def _fill(routed, holder, shards, layer_index, experts):
    names = {}
    for name in shards.weight_map:
        found = EXPERT_TENSOR.match(name)
        if found and int(found.group(1)) == layer_index:
            names.setdefault(int(found.group(2)), []).append((found.group(3), found.group(4), name))
    loaded = skipped = 0
    missing = set()
    for local, expert in enumerate(experts):
        for proj, suffix, name in names.get(int(expert), []):
            prefix, shard = SHARDS[proj]
            attr = f"{prefix}{suffix}"
            param = getattr(holder, attr, None)
            if not isinstance(param, torch.nn.Parameter):
                skipped += 1
                missing.add(attr)
                continue
            tensor = shards.get(name)
            routed.weight_loader(param, tensor, f"model.language_model.layers.{layer_index}.mlp.experts.{attr}",
                                 shard, local)
            loaded += 1
    return loaded, skipped, sorted(missing)


def _convert(routed, holder):
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import convert_to_wna16_moe_kernel_format
    method = routed.quant_method
    made = convert_to_wna16_moe_kernel_format(
        backend=method.wna16_moe_backend, layer=holder, quant_config=method.quant_config,
        input_dtype=method.input_dtype, w13=holder.w13_qweight, w2=holder.w2_qweight,
        w13_scale=holder.w13_scales, w2_scale=holder.w2_scales,
        w13_qzeros=getattr(holder, "w13_qzeros", None), w2_qzeros=getattr(holder, "w2_qzeros", None))
    if made is None:
        raise RuntimeError("the WNA16 backend rewrote the layer in place; only the Marlin backends are supported")
    w13, w2, w13_scale, w2_scale, w13_zeros, w2_zeros, g1, g2, _, _ = made
    return {"w13": w13, "w2": w2, "w13_scale": w13_scale, "w2_scale": w2_scale, "w13_zeros": w13_zeros,
            "w2_zeros": w2_zeros, "g1": g1, "g2": g2}


def _premix(holder):
    if str(os.environ.get("TAAF_VLLM_MOE_SILU_EPI", "")).strip() != "1":
        return False
    try:
        from vllm import taaf_moe_block
    except ImportError:
        return False
    return bool(taaf_moe_block.silu_premix(holder))


def _marlin_experts(runner):
    seen = set()
    todo = [runner.routed_experts.quant_method]
    while todo:
        one = todo.pop()
        if one is None or id(one) in seen:
            continue
        seen.add(id(one))
        if all(hasattr(one, name) for name in ("w1_scale", "w2_scale", "quant_type_id", "a1_gscale")):
            return one
        for name in dir(one):
            if name.startswith("__"):
                continue
            try:
                value = getattr(one, name)
            except Exception:
                continue
            if hasattr(value, "__dict__") and not isinstance(value, (torch.Tensor, type)) and len(seen) < 200:
                todo.append(value)
    return None


def _call_check(runner, made, config):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    experts = _marlin_experts(runner)
    if experts is None:
        return "no Marlin experts object on the layer to compare the cold call against"
    routed = runner.routed_experts
    _, hidden_size = _sizes(runner)
    device = routed.w13_qweight.device
    torch.manual_seed(0)
    rows, top = 24, 10
    hidden = (torch.randn(rows, hidden_size, device=device) * 0.05).to(torch.bfloat16)
    weights = torch.full((rows, top), 1.0 / top, device=device, dtype=torch.float32)
    ids = torch.zeros((rows, top), device=device, dtype=torch.int32)
    kwargs = {"activation": config["activation"]} if config["activation"] is not None else {}
    want = fused_marlin_moe(
        hidden_states=hidden, w1=routed.w13_qweight, w2=routed.w2_qweight, bias1=None, bias2=None,
        w1_scale=experts.w1_scale, w2_scale=experts.w2_scale, topk_weights=weights, topk_ids=ids,
        quant_type_id=experts.quant_type_id, global_num_experts=routed.w13_qweight.shape[0],
        input_global_scale1=getattr(experts, "a1_gscale", None), input_global_scale2=getattr(experts, "a2_gscale", None),
        global_scale1=getattr(experts, "g1_alphas", None), global_scale2=getattr(experts, "g2_alphas", None),
        w1_zeros=getattr(experts, "w1_zp", None), w2_zeros=getattr(experts, "w2_zp", None),
        input_dtype=getattr(experts, "input_dtype", None), **kwargs)
    wide = torch.full((512,), -1, dtype=torch.int32, device=device)
    wide[500] = 0
    got = fused_marlin_moe(
        hidden_states=hidden, w1=made["w13"], w2=made["w2"], bias1=None, bias2=None,
        w1_scale=made["w13_scale"], w2_scale=made["w2_scale"], topk_weights=weights,
        topk_ids=torch.full_like(ids, 500), quant_type_id=config["quant_type_id"], global_num_experts=512,
        expert_map=wide, input_global_scale1=made.get("g1"), input_global_scale2=made.get("g2"),
        w1_zeros=made.get("w13_zeros") if config["zeros"] else None,
        w2_zeros=made.get("w2_zeros") if config["zeros"] else None, input_dtype=config["input_dtype"], **kwargs)
    gap = ((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)).item()
    say(f"self-check: the cold call (expert_map, stored scales) on a kept expert rebuilt from the checkpoint "
        f"against the served expert through the stock Marlin path: relative gap {gap:.2e}")
    return "" if gap < 2e-2 else f"the cold call differs from the served expert by {gap:.2e}"


def _probe(runner, shards, layer_index, kept):
    routed = runner.routed_experts
    holder = _temp_layer(runner, 1)
    loaded, _, _ = _fill(routed, holder, shards, layer_index, [kept[0]])
    if not loaded:
        return "the probe expert has no tensors in the checkpoint"
    plain = _temp_layer(runner, 1)
    _fill(routed, plain, shards, layer_index, [kept[0]])
    plain_made = _convert(routed, plain)
    del plain
    _premix(holder)
    made = _convert(routed, holder)
    want = routed.w13_qweight[0:1]
    got = made["w13"]
    if tuple(want.shape) != tuple(got.shape) or not torch.equal(want, got):
        return f"w13 codes of kept expert {kept[0]} differ from the served layer ({tuple(got.shape)} vs {tuple(want.shape)})"
    want2 = routed.w2_qweight[0:1]
    if tuple(want2.shape) != tuple(made["w2"].shape) or not torch.equal(want2, made["w2"]):
        return f"w2 codes of kept expert {kept[0]} differ from the served layer"
    return _call_check(runner, plain_made, _experts_config(runner))


def _experts_config(runner):
    routed = runner.routed_experts
    method = routed.quant_method
    activation = getattr(routed, "activation", None) or getattr(runner.moe_config, "activation", None)
    return {"quant_type_id": int(method.quant_config.quant_type.id), "input_dtype": method.input_dtype,
            "activation": activation, "zeros": not bool(getattr(method.quant_config, "is_sym", True))}


def prepare(model_runner):
    if not on():
        return
    began = time.time()
    try:
        served = _served_dir(model_runner)
        plan_path = served / CUT_PLAN
        if not plan_path.is_file():
            raise RuntimeError(f"{plan_path} is absent; the hot switch needs the cut-at-load view")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rows = {int(k): [int(x) for x in v] for k, v in plan["rows_by_layer"].items()}
        weight_map = _index(served)
        shards = Shards(served, weight_map)
        out = store_dir()
        out.mkdir(parents=True, exist_ok=True)
        import shutil
        free = shutil.disk_usage(out).free
        if free < 16 * 2**30:
            raise RuntimeError(f"{out} has {free / 2**30:.1f} GiB free, the cold experts need about 13")
        total = 0
        found = sorted(_runners(model_runner), key=lambda pair: pair[0])
        indices = [index for index, _ in found]
        mark("prepare began", served=str(served), moe_layers=len(indices), plan_layers=len(rows),
             found_by=WORKER.get("found_by"), expert_names=WORKER.get("names"))
        order, where = _usage_order(rows)
        WORKER["order"] = order
        say(f"cold experts ranked by {where}" if order else f"cold experts in id order: {where}")
        if sorted(set(indices)) != sorted(rows) or len(indices) != len(rows):
            raise RuntimeError(f"the forward context holds MoE layers {indices[:8]}... ({len(indices)}), the cut plan "
                               f"{len(rows)} layers")
        for index, runner in found:
            served_n = int(runner.routed_experts.w13_qweight.shape[0])
            if served_n != len(rows[index]):
                raise RuntimeError(f"layer {index} serves {served_n} experts, the cut plan keeps {len(rows[index])}")
            routed = runner.routed_experts
            kept = rows[index]
            keptset = set(kept)
            ranked = (WORKER.get("order") or {}).get(index)
            extra = ([e for e in ranked if e not in keptset] if ranked
                     else [e for e in range(512) if e not in keptset])
            if index == min(rows):
                why = _probe(runner, shards, index, kept)
                if why:
                    raise RuntimeError(f"self-check failed: {why}")
                say(f"self-check: kept expert {kept[0]} of layer {index} rebuilt from the checkpoint matches the "
                    f"served codes byte for byte")
            holder = _temp_layer(runner, len(extra))
            loaded, skipped, missing = _fill(routed, holder, shards, index, extra)
            made = _convert(routed, holder)
            del holder
            cpu = {name: (value.detach().to("cpu") if isinstance(value, torch.Tensor) else None)
                   for name, value in made.items()}
            del made
            torch.save(cpu, out / f"layer{index}.pt")
            specs = {name: (tuple(value.shape), value.dtype) for name, value in cpu.items() if value is not None}
            size = sum(_aligned(_nbytes(shape, dtype)) for shape, dtype in specs.values())
            total += size
            gate_name = f"model.language_model.layers.{index}.mlp.gate.weight"
            gate_full = shards.get(gate_name)
            if gate_full.shape[0] != 512:
                raise RuntimeError(f"{gate_name} has {gate_full.shape[0]} rows, wanted 512")
            device = routed.w13_qweight.device
            try:
                gate_dtype = runner.gate.weight.dtype
            except AttributeError:
                gate_dtype = torch.bfloat16
            gate_extra = gate_full[extra].to(device=device, dtype=gate_dtype).contiguous()
            mask = torch.full((len(extra),), MASKED, dtype=gate_dtype, device=device)
            map_b = torch.full((512,), -1, dtype=torch.int32, device=device)
            for local, expert in enumerate(extra):
                map_b[len(kept) + local] = local
            state = LayerState(index, kept, extra, gate_extra, mask, map_b, specs, _experts_config(runner))
            WORKER["layers"][id(runner)] = state
            _wrap_router(runner, state)
            once(("layer", index), f"layer {index}: {len(extra)} cold experts built ({loaded} tensors, {skipped} "
                                   f"skipped {missing}), {size / 2**20:.0f} MiB stored, router widened to 512 with "
                                   f"the cold rows masked")
        shards.close()
        torch.cuda.empty_cache()
        WORKER["store"] = str(out)
        WORKER["bytes"] = total
        mark("prepared", layers=len(WORKER["layers"]), gib=round(total / 2**30, 2), seconds=round(time.time() - began))
        say(f"prepared {len(WORKER['layers'])} layers in {time.time() - began:.0f}s: {total / 2**30:.2f} GiB of cold "
            f"experts on disk at {out}; they wait for the tail of the KV pool")
    except Exception as failure:
        WORKER["aside"] = f"{type(failure).__name__}: {str(failure)[:400]}"
        for runner_id, state in list(WORKER["layers"].items()):
            pass
        WORKER["layers"].clear()
        record_aside("load")
        say(f"stood aside at load: {WORKER['aside']}; the serve keeps its 400 experts and the router is unchanged")


def _nbytes(shape, dtype):
    count = 1
    for size in shape:
        count *= int(size)
    return count * torch.empty((), dtype=dtype).element_size()


def _aligned(size):
    return (size + ALIGN - 1) // ALIGN * ALIGN


def _wrap_router(runner, state):
    router = runner.router
    stock = router.select_experts

    def select_experts(hidden_states, router_logits, *args, **kwargs):
        extra = F.linear(hidden_states.to(state.gate_extra.dtype), state.gate_extra) + state.mask
        wide = torch.cat([router_logits, extra.to(router_logits.dtype)], dim=-1)
        weights, ids = stock(hidden_states, wide, *args, **kwargs)
        kept = len(state.kept)
        cold = ids >= kept
        state.pending = (hidden_states, weights, ids)
        return torch.where(cold, torch.zeros_like(weights), weights), torch.where(cold, torch.zeros_like(ids), ids)

    router.select_experts = select_experts


def cold_output(runner, state):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    hidden, weights, ids = state.pending
    state.pending = None
    v = state.views
    config = state.config
    kwargs = {}
    if config["activation"] is not None:
        kwargs["activation"] = config["activation"]
    return fused_marlin_moe(
        hidden_states=hidden.contiguous(), w1=v["w13"], w2=v["w2"], bias1=None, bias2=None,
        w1_scale=v["w13_scale"], w2_scale=v["w2_scale"], topk_weights=weights, topk_ids=ids,
        quant_type_id=config["quant_type_id"], global_num_experts=512, expert_map=state.map_b,
        input_global_scale1=v.get("g1"), input_global_scale2=v.get("g2"),
        w1_zeros=v.get("w13_zeros") if config["zeros"] else None,
        w2_zeros=v.get("w2_zeros") if config["zeros"] else None,
        input_dtype=config["input_dtype"], **kwargs)


def wrap_forward(cls):
    stock = cls._forward_impl
    if getattr(stock, "_taaf_expert_hot", False):
        return

    def _forward_impl(self, *args, **kwargs):
        made = stock(self, *args, **kwargs)
        state = WORKER["layers"].get(id(self))
        if state is None or not WORKER["bound"] or state.pending is None:
            if state is not None:
                state.pending = None
            return made
        cold = cold_output(self, state)
        if isinstance(made, tuple):
            try:
                from vllm import taaf_moe_block
                taaf_moe_block.settle()
            except ImportError:
                pass
            shared, fused = made
            return shared, fused + cold
        return made + cold

    _forward_impl._taaf_expert_hot = True
    cls._forward_impl = _forward_impl


def _storage_bytes(tensor):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def bind_tail(model_runner, kv_cache_config):
    if not on() or not WORKER["layers"]:
        return
    if ENGINE_SIDE_DONE():
        say("a KV re-allocation after the switch: the cold experts would lose their memory; refusing to rebind")
        return
    WORKER["bound"] = False
    try:
        caches = [t for t in model_runner.kv_caches if isinstance(t, torch.Tensor)]
        if not caches:
            raise RuntimeError("no KV tensors on the model runner")
        base, size = _storage_bytes(caches[0])
        for one in caches:
            if _storage_bytes(one) != (base, size):
                raise RuntimeError("KV tensors do not share one backing allocation")
        blocks = int(kv_cache_config.num_blocks)
        if size % blocks:
            raise RuntimeError(f"backing size {size} is not a multiple of {blocks} blocks")
        block_bytes = size // blocks
        for one in caches:
            if one.dim() >= 2 and one.shape[0] == blocks:
                stride = one.stride(0) * one.element_size()
                if stride != block_bytes:
                    raise RuntimeError(f"a KV view strides {stride} bytes a block, not {block_bytes}: the layout is "
                                       f"not block-outermost")
                offset = one.data_ptr() - base
                if offset < 0 or offset >= block_bytes:
                    raise RuntimeError(f"a KV view starts {offset} bytes into the allocation, past one block")
        need = sum(sum(_aligned(_nbytes(shape, dtype)) for shape, dtype in state.specs.values())
                   for state in WORKER["layers"].values())
        tail = -(-need // block_bytes) + 1
        if tail > blocks - max(blocks // 4, 1024):
            raise RuntimeError(f"the cold experts need {tail} of {blocks} blocks and would leave the pool less "
                               f"than a quarter of itself")
        first = blocks - tail
        raw = torch.empty(0, dtype=torch.int8, device=caches[0].device)
        raw.set_(caches[0].untyped_storage(), 0, (size,), (1,))
        at = first * block_bytes
        layout = []
        kepts = set()
        colds = set()
        for index_state in sorted(WORKER["layers"].values(), key=lambda s: s.index):
            cold = len(index_state.extra)
            kepts.add(len(index_state.kept))
            colds.add(cold)
            for name in B_NAMES:
                if name not in index_state.specs:
                    continue
                shape, dtype = index_state.specs[name]
                length = _nbytes(shape, dtype)
                index_state.views[name] = raw[at:at + length].view(dtype).view(shape)
                rowed = bool(shape) and int(shape[0]) == cold
                layout.append([at, length // cold if rowed else 0, length])
                at += _aligned(length)
        if len(kepts) != 1 or len(colds) != 1:
            raise RuntimeError(f"layers keep {sorted(kepts)} and hold {sorted(colds)} cold experts; stages need one count")
        if at > size:
            raise RuntimeError("the cold experts run past the KV allocation")
        plan = {"first": first, "count": tail, "blocks": blocks, "block_bytes": block_bytes, "bytes": need,
                "layers": len(WORKER["layers"]), "kept": kepts.pop(), "cold": colds.pop(), "views": layout}
        plan["targets"] = {str(plan["kept"] + rows): len(_stage_blocks(plan, rows))
                           for rows in range(0, plan["cold"] + 1, 5)}
        out = store_dir()
        (out / PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
        WORKER["bound"] = True
        WORKER["plan"] = plan
        say(f"bound: the cold experts of {plan['layers']} layers map onto KV blocks {first}..{blocks - 1} "
            f"({tail} blocks of {block_bytes / 2**20:.2f} MiB, {need / 2**30:.2f} GiB); until the switch no token "
            f"routes there and the blocks hold KV; switch with `touch {go_path()}`")
    except Exception as failure:
        WORKER["aside"] = f"{type(failure).__name__}: {str(failure)[:400]}"
        for runner_id, state in list(WORKER["layers"].items()):
            state.views = {}
        WORKER["bound"] = False
        record_aside("kv allocation")
        say(f"stood aside at KV allocation: {WORKER['aside']}; the cold experts stay masked for the whole run")


def _stage_blocks(plan, rows_open):
    first, blocks, size, cold = plan["first"], plan["blocks"], plan["block_bytes"], plan["cold"]
    if rows_open >= cold:
        return set(range(first, blocks))
    ids = set()
    if rows_open <= 0:
        return ids
    for start, row_bytes, length in plan["views"]:
        end = start + (rows_open * row_bytes if row_bytes else length)
        if end > start:
            ids.update(range(start // size, -(-end // size)))
    return {one for one in ids if first <= one < blocks}


def ENGINE_SIDE_DONE():
    return bool(WORKER.get("switched"))


def apply_switch(worker, rows_open=None):
    began = time.time()
    if not WORKER["bound"]:
        return {"ok": False, "why": WORKER["aside"] or "not bound"}
    torch.cuda.synchronize()
    out = store_dir()
    moved = 0
    start = int(WORKER.get("rows_open", 0))
    stop_all = start
    for state in sorted(WORKER["layers"].values(), key=lambda s: s.index):
        cold = len(state.extra)
        stop = cold if rows_open is None else max(0, min(cold, int(rows_open)))
        stop_all = max(stop_all, stop)
        if stop <= start:
            continue
        held = torch.load(out / f"layer{state.index}.pt", map_location="cpu")
        for name, view in state.views.items():
            source = held[name]
            if tuple(source.shape) != tuple(view.shape) or source.dtype != view.dtype:
                raise RuntimeError(f"layer {state.index} {name}: stored {tuple(source.shape)} {source.dtype} vs "
                                   f"view {tuple(view.shape)} {view.dtype}")
            if view.dim() >= 1 and int(view.shape[0]) == cold:
                view[start:stop].copy_(source[start:stop], non_blocking=False)
                moved += source[start:stop].numel() * source.element_size()
            elif start == 0:
                view.copy_(source, non_blocking=False)
                moved += source.numel() * source.element_size()
        del held
    torch.cuda.synchronize()
    for state in WORKER["layers"].values():
        cold = len(state.extra)
        stop = cold if rows_open is None else max(0, min(cold, int(rows_open)))
        if stop > start:
            state.mask[start:stop] = 0.0
    torch.cuda.synchronize()
    WORKER["rows_open"] = stop_all
    WORKER["switched"] = True
    seconds = time.time() - began
    first = next(iter(WORKER["layers"].values()))
    say(f"switched: {moved / 2**30:.2f} GiB of cold experts copied into the KV tail and unmasked in "
        f"{seconds:.1f}s; every layer now routes over {len(first.kept) + stop_all} experts")
    return {"ok": True, "bytes": moved, "seconds": seconds, "rows_open": stop_all}


def mark(stage, **extra):
    try:
        out = store_dir()
        out.mkdir(parents=True, exist_ok=True)
        path = out / STATUS_NAME
        rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        rows.append(dict(stage=stage, pid=os.getpid(), at=round(time.time(), 1), **extra))
        path.write_text(json.dumps(rows, default=str), encoding="utf-8")
    except Exception as failure:
        say(f"status record failed at {stage}: {failure!r}")


def install_worker():
    if not on():
        return
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as failure:
        say(f"worker install failed: {failure!r}")
        mark("install failed", why=repr(failure))
        return
    wrap_forward(MoERunner)
    stock_load = Worker.load_model
    if not getattr(stock_load, "_taaf_expert_hot", False):
        def load_model(self, *args, **kwargs):
            made = stock_load(self, *args, **kwargs)
            runner = self.model_runner
            mark("load hook", runner=f"{type(runner).__module__}.{type(runner).__name__}")
            try:
                from vllm.config import set_current_vllm_config
                with set_current_vllm_config(self.vllm_config):
                    prepare(runner)
            except Exception as failure:
                WORKER["aside"] = f"{type(failure).__name__}: {str(failure)[:400]}"
                WORKER["layers"].clear()
                record_aside("load hook")
            return made
        load_model._taaf_expert_hot = True
        Worker.load_model = load_model
    stock_init = Worker.initialize_from_config
    if not getattr(stock_init, "_taaf_expert_hot", False):
        def initialize_from_config(self, kv_cache_config, *args, **kwargs):
            made = stock_init(self, kv_cache_config, *args, **kwargs)
            mark("kv hook", layers=len(WORKER["layers"]), blocks=getattr(kv_cache_config, "num_blocks", None))
            try:
                bind_tail(self.model_runner, kv_cache_config)
            except Exception as failure:
                WORKER["aside"] = f"{type(failure).__name__}: {str(failure)[:400]}"
                record_aside("kv hook")
            return made
        initialize_from_config._taaf_expert_hot = True
        Worker.initialize_from_config = initialize_from_config
    Worker.taaf_expert_hot_apply = lambda self, rows_open=None: apply_switch(self, rows_open)
    mark("worker armed")
    say(f"worker armed in pid {os.getpid()}: Worker.load_model builds the cold experts after the model loads and "
        f"Worker.initialize_from_config binds them to the KV tail after the real allocation")


def _pool(core):
    manager = getattr(core.scheduler, "kv_cache_manager", None)
    pool = getattr(manager, "block_pool", None)
    if pool is None:
        coordinator = getattr(manager, "coordinator", None)
        pool = getattr(coordinator, "block_pool", None)
    if pool is None:
        raise RuntimeError("no GPU block pool on the scheduler")
    return pool


def _claim(pool, block):
    if block.block_id in ENGINE["claimed"] or block.is_null:
        return False
    pool._maybe_evict_cached_block(block)
    ENGINE["claimed"].add(block.block_id)
    return True


def claim_free(pool):
    taken = 0
    for block_id in sorted(ENGINE["set"] - ENGINE["claimed"]):
        block = pool.blocks[block_id]
        if block.ref_cnt == 0 and block.block_id not in ENGINE["claimed"]:
            pool.free_block_queue.remove(block)
            block.ref_cnt = 1
            if _claim(pool, block):
                taken += 1
    return taken


def wrap_free(cls):
    stock = cls.free_blocks
    if getattr(stock, "_taaf_expert_hot", False):
        return

    def free_blocks(self, ordered_blocks):
        if ENGINE["state"] != "draining" or self is not ENGINE["pool"]:
            return stock(self, ordered_blocks)
        keep = []
        for block in ordered_blocks:
            if (block.block_id in ENGINE["set"] and block.ref_cnt == 1
                    and (block.pool is None or block.pool is self) and _claim(self, block)):
                continue
            keep.append(block)
        return stock(self, keep)

    free_blocks._taaf_expert_hot = True
    cls.free_blocks = free_blocks


def stall_seconds():
    try:
        return float(os.environ.get(STALL_ENV, "") or STALL_DEFAULT)
    except ValueError:
        return STALL_DEFAULT


def preempt_holders(core):
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager
    left = ENGINE["set"] - ENGINE["claimed"]
    holders = []
    for request in list(scheduler.running):
        try:
            groups = manager.get_block_ids(request.request_id)
        except Exception:
            continue
        if any(int(block) in left for group in groups for block in group):
            holders.append(request)
    stamp = time.monotonic()
    taken = 0
    for request in holders:
        if request not in scheduler.running:
            continue
        scheduler.running.remove(request)
        scheduler._preempt_request(request, stamp, drop_stale_output=True)
        scheduler.prev_step_scheduled_req_ids.discard(request.request_id)
        taken += 1
    return taken, len(left)


def tick(core):
    now = time.time()
    state = ENGINE["state"]
    if state in ("done", "aside"):
        return
    if state == "idle":
        if now - ENGINE["checked"] < 2.0:
            return
        ENGINE["checked"] = now
        if not go_path().exists():
            return
        plan_path = store_dir() / PLAN_NAME
        if not plan_path.is_file():
            ENGINE["state"] = "aside"
            aside = store_dir() / ASIDE_NAME
            why = aside.read_text(encoding="utf-8") if aside.is_file() else "no reason was recorded"
            print(f"EXPERT_HOT the switch was asked but {plan_path} is absent (the worker stood aside: {why}); the "
                  f"serve keeps 400 experts", flush=True)
            return
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        try:
            said = go_path().read_text(encoding="utf-8")
        except OSError:
            said = ""
        found = re.search(r"target=(\d+)", said)
        cold = int(plan.get("cold", 10 ** 6))
        kept = int(plan.get("kept", 0))
        rows_open = min(cold, int(found.group(1)) - kept) if found and "views" in plan else cold
        if rows_open <= ENGINE["rows_open"]:
            return
        pool = _pool(core)
        if len(pool.blocks) != plan["blocks"]:
            ENGINE["state"] = "aside"
            print(f"EXPERT_HOT the scheduler holds {len(pool.blocks)} blocks, the worker planned {plan['blocks']}; "
                  f"the switch stands aside", flush=True)
            return
        wanted = _stage_blocks(plan, rows_open) if "views" in plan else set(range(int(plan["first"]), int(plan["blocks"])))
        ENGINE.update(plan=plan, pool=pool, first=int(plan["first"]), end=int(plan["first"]) + int(plan["count"]),
                      began=now, said=now, state="draining", moved_at=now, held_last=len(wanted & ENGINE["claimed"]),
                      set=wanted, target=kept + rows_open)
        taken = claim_free(pool)
        print(f"EXPERT_HOT switch asked to {kept + rows_open} experts: draining {len(wanted - ENGINE['claimed']) + taken} "
              f"KV blocks of {len(wanted)} the stage needs; {taken} were free and are held now, the rest are held as "
              f"their owners free them", flush=True)
        state = "draining"
    if state == "draining":
        pool = ENGINE["pool"]
        claim_free(pool)
        held = len(ENGINE["set"] & ENGINE["claimed"])
        count = len(ENGINE["set"])
        if held > ENGINE["held_last"]:
            ENGINE["held_last"], ENGINE["moved_at"] = held, now
        if held < count and now - ENGINE["moved_at"] >= stall_seconds():
            taken, left = preempt_holders(core)
            ENGINE["preempted"] += taken
            ENGINE["moved_at"] = now
            print(f"EXPERT_HOT drain stalled {stall_seconds():.0f}s at {held} of {count}: {taken} running request(s) "
                  f"held the {left} tail blocks still out and were preempted; blocks held by anything else wait",
                  flush=True)
            claim_free(pool)
            held = len(ENGINE["set"] & ENGINE["claimed"])
        if held < count:
            if now - ENGINE["said"] >= 30:
                ENGINE["said"] = now
                print(f"EXPERT_HOT draining: {held} of {count} tail blocks held after {now - ENGINE['began']:.0f}s; "
                      f"{pool.get_num_free_blocks()} blocks free in the pool", flush=True)
            return
        ENGINE["state"] = "applying"
        print(f"EXPERT_HOT drained {count} blocks in {now - ENGINE['began']:.0f}s; the worker copies the cold experts "
              f"in and unmasks them", flush=True)
        plan = ENGINE["plan"]
        rows_open = int(ENGINE["target"]) - int(plan.get("kept", 0))
        full = rows_open >= int(plan.get("cold", 10 ** 6)) or "views" not in plan
        try:
            results = core.model_executor.collective_rpc(
                "taaf_expert_hot_apply", args=(None if full else rows_open,))
        except Exception as failure:
            ENGINE["state"] = "aside"
            print(f"EXPERT_HOT apply failed: {failure!r}; the tail blocks stay held and the cold experts stay masked",
                  flush=True)
            return
        ENGINE["applied"] = results
        if not full:
            ENGINE["rows_open"] = rows_open
            ENGINE["state"] = "idle"
            ENGINE["checked"] = now
            try:
                (store_dir() / f"{STAGE_PREFIX}{ENGINE['target']}.json").write_text(
                    json.dumps({"at": now, "target": ENGINE["target"], "blocks": count, "held_total": len(ENGINE["claimed"]),
                                "results": results}, default=str), encoding="utf-8")
            except OSError:
                pass
            print(f"EXPERT_HOT stage done {time.time() - ENGINE['began']:.0f}s after the ask: {ENGINE['target']} experts "
                  f"route now, {len(ENGINE['claimed'])} tail blocks held: {results}", flush=True)
            return
        ENGINE["rows_open"] = int(plan.get("cold", 0))
        ENGINE["state"] = "done"
        try:
            (store_dir() / DONE_NAME).write_text(json.dumps({"at": now, "results": results}, default=str),
                                                 encoding="utf-8")
        except OSError:
            pass
        print(f"EXPERT_HOT done {time.time() - ENGINE['began']:.0f}s after the ask: {results}", flush=True)


def install_engine():
    if not on():
        return
    try:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.engine.core import EngineCore
    except Exception as failure:
        print(f"EXPERT_HOT engine install failed: {failure!r}", flush=True)
        return
    wrap_free(BlockPool)
    stock = EngineCore.post_step
    if getattr(stock, "_taaf_expert_hot", False):
        return

    def post_step(self, *args, **kwargs):
        made = stock(self, *args, **kwargs)
        try:
            tick(self)
        except Exception as failure:
            ENGINE["state"] = "aside"
            print(f"EXPERT_HOT tick raised {failure!r}; the switch stands aside", flush=True)
        return made

    post_step._taaf_expert_hot = True
    EngineCore.post_step = post_step
    print("EXPERT_HOT engine armed: the scheduler drains the KV tail when "
          f"{go_path()} appears", flush=True)
