import copy
import dataclasses
import math
import os
import sys
import time

import torch
import triton
import triton.language as tl

META_ONCE_ENV = "TAAF_VLLM_GDN_META_ONCE"
META_SLICE_ENV = "TAAF_VLLM_GDN_META_SLICE"
PREP_ENV = "TAAF_VLLM_PREP_GDN"
PREP_ROWS = 128
META_SHARED = ("spec_sequence_masks", "spec_token_indx", "non_spec_token_indx", "spec_query_start_loc",
               "non_spec_query_start_loc", "num_accepted_tokens")
_META = {"open": False, "building": False, "held": {}, "bases": 0, "derived": 0}
_META_BUFFERS = {}
_PREP = {"on": False, "dead": False, "warm": None, "step": 0, "static": {}, "memo": None, "ident": None, "lead": None, "regs": {}, "table": None,
         "ran": None, "stats": {"fused": 0, "launches": 0, "fallback": 0, "registered": 0}}
_REPLACE = {}
_SAID = set()
TALLY_S = 15.0
_TALLY = {"at": 0.0}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def tally():
    now = time.monotonic()
    if now - _TALLY["at"] < TALLY_S:
        return
    _TALLY["at"] = now
    counts = {"bases": _META["bases"], "derived": _META["derived"], "unshared": _META.get("unshared", 0)}
    if meta_slice():
        counts["sliced"] = _META.get("sliced", 0)
    if prep_on():
        counts.update(_PREP["stats"])
    print(f"GDN_META_ONCE tally {counts}", flush=True)


def meta_once():
    return str(os.environ.get(META_ONCE_ENV) or "").strip().lower() in ("1", "on", "true")


def meta_slice():
    return str(os.environ.get(META_SLICE_ENV) or "").strip().lower() in ("1", "on", "true")


def prep_on():
    return str(os.environ.get(PREP_ENV) or "").strip().lower() in ("1", "on", "true")


def lead_rows(mask):
    if mask.dim() != 1:
        return None
    rows = int(mask.sum())
    if not bool(mask[:rows].all()):
        return None
    return rows


def _rows_copy(rows):
    return rows.clone(memory_format=torch.contiguous_format)


def meta_open():
    _META["held"] = {}
    _META["building"] = False
    _META["open"] = meta_once()
    _PREP["on"] = bool(_META["open"] and prep_on())
    _PREP["step"] += 1
    _PREP["memo"] = {} if _PREP["on"] else None
    _PREP["ident"] = {} if _PREP["on"] else None
    _PREP["lead"] = None
    _PREP["ran"] = None


def meta_close():
    if _PREP["warm"] is not None:
        _prep_warm()
    _META["held"] = {}
    _META["building"] = False
    _META["open"] = False
    _PREP["on"] = False
    _PREP["memo"] = None
    _PREP["ident"] = None
    _PREP["lead"] = None
    _PREP["ran"] = None


def meta_share(builder):
    if not meta_once():
        return False
    parallel = getattr(builder.vllm_config, "parallel_config", None)
    if getattr(parallel, "use_ubatching", False):
        say("meta-ubatch", "GDN_META_ONCE: micro-batching keeps one builder per micro-batch, so the GDN "
                           "builders keep their own persistent buffers and every group builds its own metadata")
        return False
    tensors = {name: getattr(builder, name) for name in META_SHARED}
    key = (id(builder.vllm_config),
           tuple((name, tuple(t.shape), t.dtype, str(t.device)) for name, t in tensors.items()))
    held = _META_BUFFERS.get(key)
    if held is None or held[0] is not builder.vllm_config:
        _META_BUFFERS[key] = (builder.vllm_config, tensors)
        return False
    for name, tensor in held[1].items():
        setattr(builder, name, tensor)
    say("meta-share", "GDN_META_ONCE: every GDN builder reads one set of persistent step buffers (masks, token "
                      "indices, query starts, accepted counts); only the state indices stay per group")
    return True


def _ident(tensor):
    if tensor is None:
        return None
    cache = _PREP["ident"]
    if cache is not None:
        held = cache.get(id(tensor))
        if held is not None:
            return held[1]
    made = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, str(tensor.device))
    if cache is not None:
        cache[id(tensor)] = (tensor, made)
    return made


def meta_key(builder, common, num_accepted_tokens, draft_rows_cpu):
    if not _META["open"] or _META["building"]:
        return None
    mixed = sys.modules.get("vllm.taaf_gdn_mixed_full")
    if mixed is not None and mixed.claims(builder, common, num_accepted_tokens, draft_rows_cpu):
        return None
    if _PREP["on"]:
        return _prep_key(builder, common, num_accepted_tokens, draft_rows_cpu)
    spec = builder.kv_cache_spec
    return (type(builder), id(builder.vllm_config), builder.num_spec, builder.use_spec_decode,
            getattr(builder, "spec_state_slots", builder.num_spec + 1), getattr(builder, "gdn_accept", False),
            builder.use_full_cuda_graph, builder.decode_cudagraph_max_bs, builder.gdn_prefill_backend,
            spec.block_size, getattr(spec, "num_speculative_blocks", 0),
            builder.vllm_config.cache_config.mamba_cache_mode,
            tuple(id(getattr(builder, name)) for name in META_SHARED),
            _ident(common.query_start_loc), _ident(common.query_start_loc_cpu), _ident(common.seq_lens),
            _ident(getattr(common, "is_prefilling", None)), int(common.num_reqs), int(common.num_actual_tokens),
            _ident(num_accepted_tokens), _ident(draft_rows_cpu))


def _prep_key(builder, common, num_accepted_tokens, draft_rows_cpu):
    static = _PREP["static"].get(id(builder))
    if static is None or static[0] is not builder:
        spec = builder.kv_cache_spec
        static = (builder, (type(builder), id(builder.vllm_config), builder.num_spec, builder.use_spec_decode,
                            getattr(builder, "spec_state_slots", builder.num_spec + 1),
                            getattr(builder, "gdn_accept", False), builder.use_full_cuda_graph,
                            builder.decode_cudagraph_max_bs, builder.gdn_prefill_backend, spec.block_size,
                            getattr(spec, "num_speculative_blocks", 0),
                            builder.vllm_config.cache_config.mamba_cache_mode,
                            tuple(id(getattr(builder, name)) for name in META_SHARED)))
        _PREP["static"][id(builder)] = static
    cache = _PREP["ident"]
    prefilling = getattr(common, "is_prefilling", None)
    token = (id(common.query_start_loc), id(common.query_start_loc_cpu), id(common.seq_lens), id(prefilling),
             id(draft_rows_cpu))
    held = cache.get(token)
    if held is None:
        held = ((common.query_start_loc, common.query_start_loc_cpu, common.seq_lens, prefilling, draft_rows_cpu),
                (_ident(common.query_start_loc), _ident(common.query_start_loc_cpu), _ident(common.seq_lens),
                 _ident(prefilling), _ident(draft_rows_cpu)))
        cache[token] = held
    first, second, third, fourth, fifth = held[1]
    return static[1] + (first, second, third, fourth, int(common.num_reqs), int(common.num_actual_tokens),
                        _ident(num_accepted_tokens), fifth)


def meta_base(builder, key, build):
    _META["building"] = True
    builder.taaf_prep_memo = _PREP["memo"]
    try:
        made = build()
    finally:
        _META["building"] = False
        builder.taaf_prep_memo = None
    mask = getattr(builder, "taaf_meta_mask", None)
    builder.taaf_meta_mask = None
    graphs = builder.use_full_cuda_graph and made.num_prefills == 0
    graph_spec = (graphs and made.num_decodes == 0
                  and made.num_spec_decodes <= builder.decode_cudagraph_max_bs
                  and made.num_spec_decode_tokens <= builder.decode_cudagraph_max_bs)
    graph_plain = graphs and made.num_spec_decodes == 0 and made.num_decodes <= builder.decode_cudagraph_max_bs
    commit = getattr(made, "accept_commit", None)
    inner = commit.commit if hasattr(commit, "slots") and hasattr(commit, "commit") else commit
    if commit is not None and not hasattr(inner, "keywords"):
        _META["held"].pop(key, None)
        _META["unshared"] = _META.get("unshared", 0) + 1
        say("meta-once-unshared", f"GDN_META_ONCE stands aside: the step's accept commit is a {type(commit).__name__}, "
                                  "so every GDN group builds its own metadata")
        _META["bases"] += 1
        tally()
        return made
    rows = None if commit is None else inner.keywords.get("request_indices")
    _META["held"][key] = (made, mask, graph_spec, graph_plain, rows)
    _META["bases"] += 1
    tally()
    return made


@triton.jit(do_not_specialize=["count", "batch", "plain_rows", "block_size", "null_block"])
def _state_kernel(table_addrs, table_strides, out_addrs, fresh_spec, fresh_plain, seq_lens, count, batch,
                  plain_rows, block_size, null_block,
                  ROWS: tl.constexpr, SLOTS: tl.constexpr, FRESH: tl.constexpr, PLAIN: tl.constexpr):
    group = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    live = rows < batch
    table = tl.load(table_addrs + group).to(tl.pointer_type(tl.int32))
    stride = tl.load(table_strides + group)
    seq = tl.load(seq_lens + rows, mask=live, other=1)
    start = tl.maximum((seq - 1) // block_size, 0).to(tl.int64)
    take = rows < count
    source = table + rows.to(tl.int64) * stride + start
    if FRESH:
        spec = fresh_spec + group.to(tl.int64) * count * SLOTS
        for k in tl.static_range(SLOTS):
            value = tl.load(source + k, mask=live & take, other=0)
            tl.store(spec + rows * SLOTS + k, value, mask=live & take)
        if PLAIN:
            plain = fresh_plain + group.to(tl.int64) * plain_rows
            rest = live & (rows >= count)
            tl.store(plain + (rows - count), tl.load(source, mask=rest, other=0), mask=rest)
    else:
        out = tl.load(out_addrs + group).to(tl.pointer_type(tl.int32))
        for k in tl.static_range(SLOTS):
            value = tl.load(source + k, mask=live & take, other=0)
            tl.store(out + rows * SLOTS + k, tl.where(take, value, null_block), mask=live)


def _prep_lead(mask):
    held = _PREP["lead"]
    if held is not None and held[0] is mask:
        return held[1]
    lead = lead_rows(mask)
    _PREP["lead"] = (mask, lead)
    return lead


def _prep_register(builder, common, slots, null_block):
    table = common.block_table_tensor
    buffer = getattr(builder, "spec_state_indices_tensor", None)
    spec = builder.kv_cache_spec
    if (buffer is None or builder.vllm_config.cache_config.mamba_cache_mode != "align"
            or table.dtype != torch.int32 or buffer.dtype != torch.int32 or table.dim() != 2
            or table.stride(1) != 1 or buffer.dim() != 2 or not buffer.is_contiguous()
            or buffer.shape[1] != slots or table.device != buffer.device):
        return None
    identity = (table.data_ptr(), table.stride(0), buffer.data_ptr(), int(spec.block_size), int(slots),
                int(null_block))
    held = _PREP["regs"].get(id(builder))
    if held is not None and held["identity"] == identity:
        return held
    _PREP["regs"][id(builder)] = {"builder": builder, "identity": identity}
    _PREP["table"] = None
    _PREP["warm"] = (table.device, identity[3:])
    _PREP["stats"]["registered"] = len(_PREP["regs"])
    return None


def _prep_table(device, shape):
    held = _PREP["table"]
    if held is not None and held["shape"] == shape:
        return held
    chosen = [reg for reg in _PREP["regs"].values() if reg["identity"][3:] == shape]
    held = {
        "shape": shape,
        "ids": {id(reg["builder"]): reg["identity"] for reg in chosen},
        "order": {id(reg["builder"]): index for index, reg in enumerate(chosen)},
        "tables": torch.tensor([reg["identity"][0] for reg in chosen], dtype=torch.int64).to(device),
        "strides": torch.tensor([reg["identity"][1] for reg in chosen], dtype=torch.int64).to(device),
        "outs": torch.tensor([reg["identity"][2] for reg in chosen], dtype=torch.int64).to(device),
        "groups": len(chosen),
    }
    _PREP["table"] = held
    return held


def _prep_warm():
    pending = _PREP["warm"]
    _PREP["warm"] = None
    if pending is None or _PREP["dead"]:
        return
    device, shape = pending
    try:
        held = _prep_table(device, shape)
        block_size, slots, null_block = shape
        groups = held["groups"]
        rows = 2
        seq = torch.ones(rows, dtype=torch.int32, device=device)
        scratch = torch.zeros((groups, rows, slots), dtype=torch.int32, device=device)
        outs = torch.tensor([scratch[index].data_ptr() for index in range(groups)], dtype=torch.int64).to(device)
        spec = torch.zeros((groups, 1, slots), dtype=torch.int32, device=device)
        plain = torch.zeros((groups, 1), dtype=torch.int32, device=device)
        grid = (groups, 1)
        for fresh, with_plain in ((False, False), (True, False), (True, True)):
            _state_kernel[grid](held["tables"], held["strides"], outs, spec, plain, seq, 1, rows if not fresh else 2,
                                1, block_size, null_block, ROWS=PREP_ROWS, SLOTS=slots, FRESH=fresh,
                                PLAIN=with_plain)
        say("prep-gdn-warm", f"PREP_GDN: the state-index kernel is compiled for {groups} GDN cache groups")
    except Exception as failure:
        _PREP["dead"] = True
        selfcheck_fail(PREP_ENV, f"the state-index kernel did not run at warm ({type(failure).__name__}: {failure})")
        say("prep-gdn-dead", f"PREP_GDN stands aside: the state-index kernel did not run "
                             f"({type(failure).__name__}: {failure}); every group builds its own indices")


def _prep_launch(common, base, shape, fresh):
    seq_lens = common.seq_lens
    held = _prep_table(seq_lens.device, shape)
    batch = int(common.num_reqs)
    count = int(base.num_spec_decodes)
    block_size, slots, null_block = shape
    groups = held["groups"]
    spec_all = None
    plain_all = None
    plain_rows = 0
    if fresh:
        spec_all = torch.empty((groups, count, slots), dtype=torch.int32, device=seq_lens.device)
        if base.num_prefills or base.num_decodes:
            plain_rows = batch - count
            plain_all = torch.empty((groups, plain_rows), dtype=torch.int32, device=seq_lens.device)
    _state_kernel[(groups, max(1, math.ceil(batch / PREP_ROWS)))](
        held["tables"], held["strides"], held["outs"], seq_lens if spec_all is None else spec_all,
        seq_lens if plain_all is None else plain_all, seq_lens, count, batch, plain_rows, block_size, null_block,
        ROWS=PREP_ROWS, SLOTS=slots, FRESH=fresh, PLAIN=plain_all is not None)
    _PREP["stats"]["launches"] += 1
    _PREP["ran"] = {"ids": held["ids"], "order": held["order"], "batch": batch, "count": count, "shape": shape,
                    "fresh": fresh, "spec": spec_all, "plain": plain_all}
    say("prep-gdn-launch", f"PREP_GDN live: one kernel writes the speculative"
                           f"{' and plain' if plain_all is not None else ''} state indices of {groups} GDN cache "
                           f"groups for {batch} request rows ({count} speculative), in place of about fifteen "
                           f"small launches per group")


def _fused_states(builder, common, base, slots, null_block, fresh):
    if _PREP["dead"] or common.seq_lens.dtype != torch.int32 or not common.seq_lens.is_contiguous():
        return None
    reg = _prep_register(builder, common, slots, null_block)
    if reg is None:
        _PREP["stats"]["fallback"] += 1
        return None
    shape = reg["identity"][3:]
    if _PREP["ran"] is None:
        try:
            _prep_launch(common, base, shape, fresh)
        except Exception as failure:
            _PREP["dead"] = True
            selfcheck_fail(PREP_ENV, f"the state-index kernel did not run at first launch "
                                     f"({type(failure).__name__}: {failure})")
            say("prep-gdn-dead", f"PREP_GDN stands aside: the state-index kernel did not run "
                                 f"({type(failure).__name__}: {failure}); every group builds its own indices")
            return None
    ran = _PREP["ran"]
    if (ran["shape"] != shape or ran["batch"] != int(common.num_reqs) or ran["count"] != int(base.num_spec_decodes)
            or ran["fresh"] != fresh or ran["ids"].get(id(builder)) != reg["identity"]):
        _PREP["stats"]["fallback"] += 1
        return None
    _PREP["stats"]["fused"] += 1
    if not fresh:
        return builder.spec_state_indices_tensor[:int(common.num_reqs)], None
    index = ran["order"][id(builder)]
    return ran["spec"][index], None if ran["plain"] is None else ran["plain"][index]


def _replace(base, **changes):
    cls = type(base)
    simple = _REPLACE.get(cls)
    if simple is None:
        simple = (dataclasses.is_dataclass(cls) and not hasattr(cls, "__post_init__")
                  and not getattr(cls, "__slots__", None)
                  and all(field.init for field in dataclasses.fields(cls)))
        _REPLACE[cls] = simple
    if not simple:
        return dataclasses.replace(base, **changes)
    made = copy.copy(base)
    for name, value in changes.items():
        setattr(made, name, value)
    return made


def _share_computed(common, memo):
    state = getattr(common, "__dict__", None)
    if state is None:
        return None
    token = (id(common.seq_lens), id(common.query_start_loc))
    had = "compute_num_computed_tokens" in state
    original = common.compute_num_computed_tokens
    held = memo.get("computed")
    if held is not None and held[0] == token:
        value = held[1]
        state["compute_num_computed_tokens"] = lambda: value
    else:
        def record():
            value = original()
            memo["computed"] = (token, value)
            return value

        state["compute_num_computed_tokens"] = record
    return had, original


def _commit(builder, common, spec_state, base, rows, commit_for):
    memo = _PREP["memo"]
    if memo is None:
        return commit_for(builder, common, spec_state, base.spec_query_start_loc, base.num_spec_decodes, rows)
    builder.taaf_prep_memo = memo
    restore = _share_computed(common, memo)
    try:
        return commit_for(builder, common, spec_state, base.spec_query_start_loc, base.num_spec_decodes, rows)
    finally:
        builder.taaf_prep_memo = None
        if restore is not None:
            if restore[0]:
                common.__dict__["compute_num_computed_tokens"] = restore[1]
            else:
                common.__dict__.pop("compute_num_computed_tokens", None)


def meta_derive(builder, common, key, block_table_of, null_block, commit_for):
    held = _META["held"].get(key)
    if held is None:
        return None
    base, mask, graph_spec, graph_plain, rows = held
    prep = _PREP["on"]
    spec_state = None
    plain_state = None
    slots = getattr(builder, "spec_state_slots", builder.num_spec + 1)
    lead = None
    if mask is not None and meta_slice():
        lead = _prep_lead(mask) if prep else lead_rows(mask)
    fused = None
    if (prep and lead is not None and lead == base.num_spec_decodes
            and common.block_table_tensor.shape[0] == mask.shape[0] == common.num_reqs):
        if graph_spec and not (base.num_prefills or base.num_decodes):
            fused = _fused_states(builder, common, base, slots, null_block, False)
        elif not graph_spec and not graph_plain:
            fused = _fused_states(builder, common, base, slots, null_block, True)
    if fused is not None:
        spec_state, plain_state = fused
        _META["sliced"] = _META.get("sliced", 0) + 1
    else:
        table = block_table_of(common.block_table_tensor, common.seq_lens, builder.kv_cache_spec,
                               builder.vllm_config.cache_config.mamba_cache_mode)
        if mask is None:
            plain_state = table[:, 0]
        elif lead is not None and table.shape[0] == mask.shape[0]:
            spec_state = _rows_copy(table[:lead, :slots])
            if base.num_prefills or base.num_decodes:
                plain_state = _rows_copy(table[lead:, 0])
            _META["sliced"] = _META.get("sliced", 0) + 1
            say("meta-slice", "GDN_META_SLICE live: a derived GDN group whose speculative rows lead the batch "
                              "takes its state indices by row slices, not by a host mask")
        else:
            spec_state = table[mask, :slots]
            if base.num_prefills or base.num_decodes:
                plain_state = table[~mask, 0]
    prefill_state = None
    if base.num_prefills > 0:
        prefill_state = plain_state[base.num_decodes:] if mask is None and base.num_decodes > 0 else plain_state
    batch = common.num_reqs
    if graph_spec and fused is None:
        count = base.num_spec_decodes
        builder.spec_state_indices_tensor[:count].copy_(spec_state, non_blocking=True)
        spec_state = builder.spec_state_indices_tensor[:batch]
        spec_state[count:].fill_(null_block)
    if graph_plain:
        count = base.num_decodes
        builder.non_spec_state_indices_tensor[:count].copy_(plain_state, non_blocking=True)
        plain_state = builder.non_spec_state_indices_tensor[:batch]
        plain_state[count:].fill_(null_block)
    changes = dict(spec_state_indices_tensor=spec_state, non_spec_state_indices_tensor=plain_state,
                   prefill_state_indices=prefill_state)
    made = _replace(base, **changes) if prep else dataclasses.replace(base, **changes)
    if hasattr(made, "accept_commit"):
        made.accept_commit = None
        if getattr(builder, "gdn_accept", False) and base.num_spec_decodes > 0:
            made.accept_commit = _commit(builder, common, spec_state, base, rows, commit_for)
    _META["derived"] += 1
    say("meta-once", "GDN_META_ONCE live: a step's GDN attention metadata is built once and every other GDN "
                     "cache group takes it with only its own state indices gathered")
    return made
