import importlib
import os
import sys

import torch

ENV = "TAAF_VLLM_SPEC_INDEX"
EVERY_ENV = "TAAF_VLLM_SPEC_INDEX_EVERY"
VARIANTS_ENV = "TAAF_VLLM_SPEC_INDEX_VARIANTS"
ROWS_ENV = "TAAF_VLLM_SPEC_INDEX_ROWS"
LANES_ENV = "TAAF_VLLM_SPEC_INDEX_LANES"
PREFETCH_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC"
PREFETCH_LAYERS_ENV = "TAAF_VLLM_HOST_KV_PREFETCH_SPEC_LAYERS"
EVERY = 200
VARIANTS = ("self", "mlp", "attn", "prev")
FIELDS = ("lane_steps", "real_rows", "spec_rows", "overlap", "real_miss", "spec_miss", "covered")
STASH_ROWS = 128
LANES = 16
HISPARSE_MODULE = "vllm.taaf_qsa_hisparse"
INDEXER_MODULE = "vllm.models.qwen4_exp.nvidia.indexer_qsa"
OPS_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
WIDE = torch.iinfo(torch.int64).max
MARK = "_taaf_spec_index"
_STATE = {"tally": None}


def enabled():
    return str(os.environ.get(ENV) or "").strip() == "1"


def prefetch_on():
    return str(os.environ.get(PREFETCH_ENV) or "").strip() == "1"


def prefetch_layers():
    said = str(os.environ.get(PREFETCH_LAYERS_ENV) or "").strip()
    if not said:
        return None
    return {int(one) for one in said.split(",") if one.strip()}


def layer_number(name):
    numbers = [int(part) for part in str(name).split(".") if part.isdigit()]
    return numbers[0] if numbers else -1


def every():
    said = str(os.environ.get(EVERY_ENV) or "").strip()
    return int(said) if said.isdigit() and int(said) > 0 else EVERY


def variants():
    said = str(os.environ.get(VARIANTS_ENV) or "").strip()
    if not said:
        return VARIANTS
    chosen = tuple(one.strip() for one in said.split(",") if one.strip())
    unknown = [one for one in chosen if one not in VARIANTS]
    if unknown or not chosen:
        raise ValueError(f"{VARIANTS_ENV}={said!r} names {unknown or 'nothing'}; choose from {VARIANTS}")
    return chosen


def say(line):
    print(f"SPEC_INDEX {line}", flush=True)


def capturing():
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def lane_sets(picks, lanes):
    flat = picks.reshape(lanes, -1).to(torch.int64)
    ordered = torch.where(flat >= 0, flat, torch.full_like(flat, WIDE)).sort(1).values
    repeat = torch.zeros_like(ordered, dtype=torch.bool)
    repeat[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
    return torch.where(repeat, torch.full_like(ordered, WIDE), ordered).sort(1).values


def members(ordered, probe):
    at = torch.searchsorted(ordered, probe.contiguous()).clamp_max(ordered.shape[1] - 1)
    return (ordered.gather(1, at) == probe) & (probe != WIDE)


def host_rows(positions, table, request_ids, block_size):
    pages = table.index_select(0, request_ids.to(torch.long)).to(torch.long)
    valid = positions != WIDE
    wide = torch.where(valid, positions, torch.zeros_like(positions))
    page = wide // int(block_size)
    block = pages.gather(1, page.clamp_max(pages.shape[1] - 1))
    inside = valid & (page < pages.shape[1]) & (block >= 0)
    return torch.where(inside, block * int(block_size) + wide % int(block_size), torch.full_like(wide, WIDE)), inside


class Residency:
    def __init__(self, table, request_ids, block_size, resident, states):
        self.table = table
        self.request_ids = request_ids
        self.block_size = block_size
        lane_state = states.index_select(0, request_ids.to(torch.long))
        self.lane_ok = lane_state >= 0
        owned = resident.index_select(0, lane_state.clamp_min(0).to(torch.long)).to(torch.long)
        self.owned = torch.where(owned >= 0, owned, torch.full_like(owned, WIDE)).sort(1).values

    def misses(self, positions):
        rows, inside = host_rows(positions, self.table, self.request_ids, self.block_size)
        live = inside & self.lane_ok.view(-1, 1)
        return live & ~members(self.owned, torch.where(live, rows, torch.full_like(rows, WIDE)))


def counts(real, spec, real_miss, spec_miss, lane_ok):
    real_valid = real != WIDE
    spec_valid = spec != WIDE
    shared = members(real, spec)
    lanes = (real_valid.any(1) & lane_ok).sum(dtype=torch.int64)
    return torch.stack((lanes, real_valid.sum(dtype=torch.int64), spec_valid.sum(dtype=torch.int64),
                        shared.sum(dtype=torch.int64), real_miss.sum(dtype=torch.int64),
                        spec_miss.sum(dtype=torch.int64), (shared & spec_miss).sum(dtype=torch.int64)))


def decode_rows(indexer):
    held = indexer._metadata()
    if held is None:
        return None, "none"
    compressed = held[1]
    rows = int(getattr(compressed, "num_decode_tokens", 0) or 0)
    if rows <= 0:
        return None, "none"
    if getattr(compressed, "bucket", None) is not None:
        return None, "bucket"
    return compressed, "pure" if rows == int(compressed.num_actual_tokens) else "mixed"


def head_rows(compressed, cap):
    decodes = int(compressed.num_decodes)
    query_len = int(compressed.decode_query_len)
    lanes = min(decodes, int(cap))
    return lanes, lanes * query_len


def gemma_norm(values, weight, eps):
    wide = values.float()
    wide = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps)
    return (wide * (1.0 + weight.float())).to(values.dtype)


def select_blocks(indexer, projected, positions, compressed, lanes, rows):
    rope = importlib.import_module(INDEXER_MODULE).apply_qsa_rope
    ops = importlib.import_module(OPS_MODULE)
    heads, width = indexer.index_n_heads, indexer.index_head_dim
    query = projected[:rows, :heads * width].reshape(-1, heads, width)
    query = gemma_norm(query.reshape(-1, width), indexer.q_layernorm.weight,
                       indexer.q_layernorm.variance_epsilon).reshape_as(query)
    query = rope(indexer.rotary_emb, positions[..., :rows], query).to(indexer.indexer_dtype)
    blocks = torch.empty(rows, indexer.token_topk // indexer.compress_ratio, dtype=torch.int32, device=query.device)
    ops.qsa_select_paged_decode(query, indexer.compressed_key_cache.kv_cache,
                                compressed.block_table[:lanes], compressed.visible_blocks[:rows], indexer.token_topk,
                                indexer.compress_ratio, compressed.decode_query_len, blocks)
    return blocks


def select_from(indexer, projected, positions, compressed, lanes, rows):
    rope = importlib.import_module(INDEXER_MODULE).apply_qsa_rope
    ops = importlib.import_module(OPS_MODULE)
    heads, width = indexer.index_n_heads, indexer.index_head_dim
    query = projected[:rows, :heads * width].reshape(-1, heads, width)
    query = gemma_norm(query.reshape(-1, width), indexer.q_layernorm.weight,
                       indexer.q_layernorm.variance_epsilon).reshape_as(query)
    query = rope(indexer.rotary_emb, positions[..., :rows], query).to(indexer.indexer_dtype)
    visible = compressed.visible_blocks[:rows]
    blocks = torch.empty(rows, indexer.token_topk // indexer.compress_ratio, dtype=torch.int32, device=query.device)
    ops.qsa_select_paged_decode(query, indexer.compressed_key_cache.kv_cache,
                                compressed.block_table[:lanes], visible, indexer.token_topk,
                                indexer.compress_ratio, compressed.decode_query_len, blocks)
    out = torch.empty(rows, indexer.packed_output_width, dtype=torch.int32, device=query.device)
    ops.expand_qsa_block_indices(blocks, compressed.logical_positions[:rows], visible, indexer.compress_ratio,
                                 indexer.token_topk, out)
    return out


def projection(layer, state):
    mixer = layer.attn_hyper_connection
    block_input = type(mixer).mix(mixer, state)[1]
    proj = layer.self_attn.indexer.index_qk_proj
    said = type(proj).forward(proj, block_input)
    return said[0] if isinstance(said, tuple) else said


def decoder_pairs(model):
    layers = []
    seen = set()
    for _, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if hasattr(module, "layer_type") and hasattr(module, "attn_hyper_connection") and hasattr(
                module, "mlp_hyper_connection"):
            layers.append(module)
    return [(layers[at - 1], layers[at]) for at in range(1, len(layers))
            if layers[at].layer_type == "full_attention"
            and getattr(getattr(layers[at], "self_attn", None), "indexer", None) is not None]


def current(owner, name):
    held = owner.__dict__.get(name) if hasattr(owner, "__dict__") else None
    if held is not None:
        return held
    kind = type(owner)
    return lambda *args, **kwargs: getattr(kind, name)(owner, *args, **kwargs)


class Tally:
    def __init__(self, device, chosen, layers, every_steps=None, out=say, select=select_from, project=projection,
                 lanes=None):
        self.device = torch.device(device)
        self.variants = tuple(chosen)
        self.layers = layers
        self.every = every_steps or every()
        self.out = out
        self.select = select
        self.project = project
        self.lanes = lanes or lanes_cap()
        self.acc = torch.zeros(len(self.variants), len(FIELDS), dtype=torch.int64, device=self.device)
        self.host = self.acc.new_zeros(self.acc.shape, device="cpu")
        if self.device.type == "cuda":
            self.host = self.host.pin_memory()
        self.stash = {}
        self.stash_bytes = 0
        self.pending = {}
        self.waiting = None
        self.steps = 0
        self.total = 0
        self.window = 0
        self.lines = 0
        self.host_calls = 0
        self.observed = 0
        self.skipped = {}
        self.announced = set()
        self.last = None

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def reserve(self, name, variant, rows, width, dtype):
        held = torch.zeros(rows, width, dtype=dtype, device=self.device)
        self.stash[(name, variant)] = held
        self.stash_bytes += held.numel() * held.element_size()

    def keep(self, name, variant, state):
        held = self.stash.get((name, variant))
        if held is None or state is None or state.dim() != 2 or state.shape[1] != held.shape[1]:
            return self.skip(f"{variant}_keep")
        rows = min(int(state.shape[0]), int(held.shape[0]))
        held[:rows].copy_(state[:rows])

    def speculate(self, layer, projected, positions):
        attn = layer.self_attn
        name = attn.layer_name
        compressed, kind = decode_rows(attn.indexer)
        if compressed is None:
            self.pending.pop(name, None)
            if kind == "bucket":
                self.skip("bucket")
            return None
        lanes, rows = head_rows(compressed, self.lanes)
        made = {}
        for variant in self.variants:
            if variant == "prev":
                continue
            if variant == "self":
                source = projected
            else:
                state = self.stash.get((name, variant))
                if state is None or rows > state.shape[0]:
                    self.skip(f"{variant}_stash")
                    continue
                source = self.project(layer, state[:rows])
            made[variant] = self.select(attn.indexer, source, positions, compressed, lanes, rows)
        self.pending[name] = made
        self.skip(kind)
        self.host_calls += 1
        if "spec" not in self.announced:
            self.announced.add("spec")
            self.out(f"fired layer={name} lanes={lanes} rows={rows} kind={kind} variants={','.join(made)} capturing={capturing()}: layer L's "
                     f"own indexer scored on its real input (self), on the state before layer L-1's MLP (mlp) and "
                     f"before layer L-1's attention (attn), each through layer L's own mixer and projection; prev is "
                     f"the previous QSA layer's real picks of the same step")
        return made

    def observe(self, layer, logical, meta):
        name = getattr(layer, "layer_name", None)
        made = self.pending.pop(name, None)
        if made is None:
            return None
        if getattr(meta, "fixed", None) is not None:
            return self.skip("fixed")
        group = getattr(layer, "taaf_hisparse_group", None)
        query_len = int(meta.decode_max_query_len)
        if group is None or int(meta.num_decodes) <= 0 or int(meta.num_decode_tokens) != int(
                meta.num_decodes) * query_len or len(group.caches) != 1:
            return self.skip("shape")
        decodes = min(int(meta.num_decodes), self.lanes)
        rows = decodes * query_len
        handle = group.cache(0)
        runtime = handle.runtime
        table = getattr(handle, "source_block_table", None)
        states = getattr(runtime, "request_state_indices", None)
        index = getattr(runtime, "index_group", None)
        resident = getattr(index, "device_global_indices", None)
        if table is None or states is None or resident is None:
            return self.skip("residency")
        width = int(logical.shape[1]) - 1
        place = Residency(table, group.request_ids[:decodes], int(meta.block_size), resident, states)
        real = lane_sets(logical[:rows, :width], decodes)
        real_miss = place.misses(real)
        at = self.layers.index(name) if name in self.layers else -1
        if self.last is not None and self.last[:2] == (self.total, at - 1) and self.last[2].shape[0] >= rows:
            made = {**made, "prev": self.last[2]}
        if "prev" in self.variants:
            self.last = (self.total, at, logical[:rows].clone())
        for slot, variant in enumerate(self.variants):
            spec = made.get(variant)
            if spec is None or spec.shape[0] < rows:
                continue
            chosen = lane_sets(spec[:rows, :width], decodes)
            self.acc[slot].add_(counts(real, chosen, real_miss, place.misses(chosen), place.lane_ok))
        self.observed += 1
        if "observe" not in self.announced:
            self.announced.add("observe")
            self.out(f"observed layer={layer.layer_name} lanes={decodes} query_len={query_len} width={width} "
                     f"region={int(resident.shape[1])} capturing={capturing()}: each lane's distinct picks are "
                     f"tested against that layer's hot region before its own swap-in, and the spec picks against "
                     f"the real ones")
        return True

    def step(self):
        self.total += 1
        self.steps += 1
        if self.waiting is not None and self.waiting[0].query():
            self.report(self.waiting[1])
            self.waiting = None
        if self.waiting is None and self.total % self.every == 0:
            self.host.copy_(self.acc, non_blocking=True)
            self.acc.zero_()
            event = None
            if self.device.type == "cuda":
                event = torch.cuda.Event()
                event.record()
            self.waiting = (event or _Done(), self.steps)
            self.steps = 0

    def report(self, steps):
        lines = report_lines(self.variants, self.host.numpy().copy(), steps, self.window, self.host_calls,
                             self.observed, self.skipped)
        self.window += 1
        for line in lines:
            self.out(line)
        return lines


class _Done:
    def query(self):
        return True


def share(top, bottom):
    return top / bottom if bottom else 0.0


def report_lines(chosen, acc, steps, window, host_calls, observed, skipped):
    out = []
    for slot, variant in enumerate(chosen):
        values = dict(zip(FIELDS, (int(one) for one in acc[slot])))
        lanes = max(values["lane_steps"], 1)
        out.append(f"window={window} steps={steps} variant={variant} lane_layer_steps={values['lane_steps']} "
                   f"real_rows_per_lane={values['real_rows'] / lanes:.1f} "
                   f"spec_rows_per_lane={values['spec_rows'] / lanes:.1f} "
                   f"overlap={share(values['overlap'], values['real_rows']):.4f} "
                   f"real_miss_per_lane={values['real_miss'] / lanes:.2f} "
                   f"spec_miss_per_lane={values['spec_miss'] / lanes:.2f} "
                   f"miss_coverage={share(values['covered'], values['real_miss']):.4f} "
                   f"copy_precision={share(values['covered'], values['spec_miss']):.4f} "
                   f"copied_per_real_miss={share(values['spec_miss'], values['real_miss']):.3f}")
    out.append(f"window={window} host_calls={host_calls} host_observed={observed} "
               f"events={','.join(f'{key}:{value}' for key, value in sorted(skipped.items())) or 'none'}")
    return out


def wrap_stash(tally, owner, name, layer_name, variant):
    original = current(owner, name)

    def stashed(*args, **kwargs):
        out = original(*args, **kwargs)
        tally.keep(layer_name, variant, out[0] if isinstance(out, tuple) else None)
        return out

    setattr(owner, name, stashed)


def wrap_indexer(tally, layer, timed):
    indexer = layer.self_attn.indexer
    original = current(indexer, "forward")
    speculate = timed("spec_index")(tally.speculate)

    def forward(projected_qk, positions, out=None):
        selected = original(projected_qk, positions, out)
        speculate(layer, projected_qk, positions)
        return selected

    indexer.forward = forward


def wrap_attention(tally, module, timed):
    original = module.decode_attention
    if getattr(original, MARK, False):
        return False
    observe = timed("spec_tally")(tally.observe)

    def decode_attention(layer, impl, query, logical, meta, out, gate, before=None):
        observe(layer, logical, meta)
        return original(layer, impl, query, logical, meta, out, gate, before)

    setattr(decode_attention, MARK, True)
    module.decode_attention = decode_attention
    return True


def wrap_runner(tally, runner):
    execute = current(runner, "execute_model")

    def execute_model(scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if not dummy and getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            tally.step()
        return execute(scheduler_output, *args, **kwargs)

    runner.execute_model = execute_model


def stash_rows():
    said = str(os.environ.get(ROWS_ENV) or "").strip()
    return int(said) if said.isdigit() and int(said) > 0 else STASH_ROWS


def lanes_cap():
    said = str(os.environ.get(LANES_ENV) or "").strip()
    return int(said) if said.isdigit() and int(said) > 0 else LANES


def plain(name):
    return lambda function: function


class Prefetch:
    def __init__(self, module, names, out=say, select=select_blocks, project=projection):
        self.module = module
        self.order = {name: at for at, name in enumerate(names)}
        self.out = out
        self.select = select
        self.project = project
        self.positions = None
        self.at = None
        self.fired = 0
        self.skipped = {}
        self.announced = False

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def note(self, name, positions):
        self.at = self.order.get(name)
        self.positions = positions

    def fire(self, layer, state):
        name = layer.self_attn.layer_name
        index = self.order.get(name)
        if index is None or self.at is None or self.positions is None or self.at >= index:
            return self.skip("first")
        indexer = layer.self_attn.indexer
        compressed, kind = decode_rows(indexer)
        if kind != "pure":
            return self.skip(kind)
        decodes = int(compressed.num_decodes)
        query_len = int(compressed.decode_query_len)
        rows = decodes * query_len
        if state is None or state.dim() != 2 or int(state.shape[0]) < rows:
            return self.skip("shape")
        blocks = self.select(indexer, self.project(layer, state[:rows]), self.positions, compressed, decodes, rows)
        width = self.module.spec_issue(name, blocks, decodes, query_len, int(indexer.compress_ratio))
        self.fired += 1
        if not self.announced:
            self.announced = True
            self.out(f"prefetch fired layer={name} decodes={decodes} rows={rows} width={width} "
                     f"capturing={capturing()}: layer L's indexer scores the state entering layer L-1's MLP and the "
                     f"rows it names that L's hot region and resident pages lack are copied while that MLP runs")
        return width


def wrap_prefetch_indexer(prefetch, layer):
    indexer = layer.self_attn.indexer
    original = current(indexer, "forward")
    name = layer.self_attn.layer_name

    def forward(projected_qk, positions, out=None):
        prefetch.note(name, positions)
        return original(projected_qk, positions, out)

    indexer.forward = forward


def wrap_prefetch_mixer(prefetch, earlier, later, timed):
    mixer = earlier.mlp_hyper_connection
    original = current(mixer, "combine_and_mix")
    fire = timed("spec_index")(prefetch.fire)

    def combine_and_mix(*args, **kwargs):
        out = original(*args, **kwargs)
        fire(later, out[0] if isinstance(out, tuple) else None)
        return out

    mixer.combine_and_mix = combine_and_mix


def install_prefetch(runner, timed=None, module=None, select=select_blocks, project=projection, out=say):
    timed = timed or plain
    pairs = decoder_pairs(runner.model)
    module = module or sys.modules.get(HISPARSE_MODULE)
    if not pairs or module is None or not hasattr(module, "spec_issue"):
        out("prefetch found no QSA layer behind another decoder layer or no host-KV module; nothing is prefetched")
        return None
    names = [later.self_attn.layer_name for _, later in pairs]
    prefetch = Prefetch(module, names, out=out, select=select, project=project)
    chosen = prefetch_layers()
    for earlier, later in pairs:
        wrap_prefetch_indexer(prefetch, later)
    targets = [(earlier, later) for earlier, later in pairs
               if chosen is None or layer_number(later.self_attn.layer_name) in chosen]
    for earlier, later in targets:
        wrap_prefetch_mixer(prefetch, earlier, later, timed)
    _STATE["tally"] = prefetch
    out(f"prefetch armed on {len(targets)} of {len(pairs)} QSA layers ({PREFETCH_LAYERS_ENV} "
        f"{'all' if chosen is None else sorted(chosen)}): on pure decode forwards, after layer L-1's MLP mixer, "
        f"layer L's own indexer scores that state through L's mixer and projection without writing its caches, "
        f"and the rows its picks name that L's hot region and resident pages lack are copied on the prefetch "
        f"stream while the MLP runs; the first QSA layer of a forward has no earlier index layer and is skipped")
    return prefetch


def install(runner, timed=None, module=None, select=select_from, project=projection, out=say):
    if _STATE["tally"] is not None:
        return _STATE["tally"]
    if prefetch_on():
        return install_prefetch(runner, timed, module, out=out)
    timed = timed or plain
    chosen = variants()
    pairs = decoder_pairs(runner.model)
    if not pairs:
        out("found no attention layer with a QSA indexer behind another decoder layer; nothing is tallied")
        return None
    module = module or sys.modules.get(HISPARSE_MODULE)
    if module is None or not hasattr(module, "decode_attention"):
        out(f"{HISPARSE_MODULE} is not loaded, so no hot region is known; nothing is tallied")
        return None
    tally = Tally(runner.device, chosen, [later.self_attn.layer_name for _, later in pairs], out=out, select=select,
                  project=project)
    rows = max(stash_rows(), tally.lanes * 8)
    for earlier, later in pairs:
        name = later.self_attn.layer_name
        for variant, mixer, methods in (("mlp", earlier.mlp_hyper_connection, ("combine_and_mix",)),
                                        ("attn", earlier.attn_hyper_connection, ("combine_and_mix", "mix"))):
            if variant not in chosen:
                continue
            dtype = getattr(getattr(mixer, "config", None), "params_dtype", None) or torch.bfloat16
            tally.reserve(name, variant, rows, int(mixer.hyper_hidden_size), dtype)
            for method in methods:
                wrap_stash(tally, mixer, method, name, variant)
        wrap_indexer(tally, later, timed)
    wrap_attention(tally, module, timed)
    wrap_runner(tally, runner)
    _STATE["tally"] = tally
    out(f"armed on {len(pairs)} QSA layers, variants {','.join(chosen)}: on pure decode forwards each QSA layer's "
        f"indexer also selects from a speculative input (its own input for self, the multi-stream state before "
        f"the previous layer's MLP for mlp, before the previous layer's attention for attn) through its own "
        f"mixer and projection, without writing its caches, and prev takes the previous QSA layer's real picks; "
        f"before the layer's swap-in the real and the spec picks are tested against its hot region, and the counts "
        f"go into device accumulators read by a non-blocking copy and printed every {tally.every} steps; the "
        f"speculative states are copied into {len(tally.stash)} buffers of {rows} rows, "
        f"{tally.stash_bytes / 2**20:.1f} MiB in all, so a replayed graph refills the same memory; nothing is "
        f"prefetched")
    return tally
