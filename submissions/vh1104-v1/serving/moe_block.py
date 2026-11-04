import os
import re
import sys

import torch

ENV = "TAAF_VLLM_MOE_BLOCK_M"
OFF = ("", "0", "off", "false", "no")
ON = ("1", "on", "true", "yes")
DEFAULT_BANDS = ((1700, 2100, 32),)
BLOCKS = (8, 16, 32, 48, 64)
BAND = re.compile(r"(\d+)-(\d+):(\d+)")
CHECK_REL = 1e-2
_STATE = {"bands": None, "forced": None, "verified": {}, "failed": {}, "said": set(), "fired": {}}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def say(message):
    print(f"MOE_BLOCK_M {message}", file=sys.stderr, flush=True)


def parse(value):
    said = str(value or "").strip().lower()
    if said in OFF:
        return ()
    if said in ON:
        return DEFAULT_BANDS
    bands = []
    for part in said.split(","):
        found = BAND.fullmatch(part.strip())
        if not found:
            raise ValueError(f"{ENV} must be 1 or <first rows>-<last rows>:<block>[,...], found {value!r}")
        first, last, block = (int(found.group(index)) for index in (1, 2, 3))
        if first < 1 or last < first or block not in BLOCKS:
            raise ValueError(f"{ENV} band {part!r} needs 1 <= first <= last and a block in {list(BLOCKS)}")
        bands.append((first, last, block))
    ordered = sorted(bands)
    for (_, last, _), (first, _, _) in zip(ordered, ordered[1:]):
        if first <= last:
            raise ValueError(f"{ENV} bands overlap: {value!r}")
    return tuple(ordered)


def bands():
    if _STATE["bands"] is None:
        _STATE["bands"] = parse(os.environ.get(ENV))
    return _STATE["bands"]


def band(rows):
    for first, last, block in bands():
        if first <= rows <= last:
            return block
    return None


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _key(scope):
    return (tuple(scope["w1"].shape), tuple(scope["w2"].shape), str(scope["w1"].device))


def _arguments(scope, fn):
    code = fn.__code__
    names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
    return {name: scope[name] for name in names if name in scope}


def _run(fn, held, block):
    held = dict(held, output=torch.empty_like(held["hidden_states"]))
    _STATE["forced"] = block
    try:
        back = fn(**held)
    finally:
        _STATE["forced"] = None
    return held["output"] if back is None else back


def compare(stock, mine):
    if torch.equal(stock, mine):
        return True, "bit-exact"
    scale = float(stock.float().abs().max())
    gap = float((stock.float() - mine.float()).abs().max())
    rel = gap / max(scale, 1e-30)
    return rel <= CHECK_REL, f"max rel {rel:.3g} (limit {CHECK_REL})"


def check(scope, fn, stock, mine, key):
    held = _arguments(scope, fn)
    try:
        want = _run(fn, held, stock)
        got = _run(fn, held, mine)
        ok, detail = compare(want, got)
    except Exception as failure:
        ok, detail = False, f"{type(failure).__name__}: {str(failure)[:200]}"
    rows = int(scope["hidden_states"].shape[0])
    if ok:
        _STATE["verified"][key] = detail
        say(f"live: {rows} rows run Marlin MoE at block {mine} in place of vLLM's {stock} for experts {key[0]}; "
            f"first call checked against block {stock}: {detail}")
    else:
        selfcheck_fail(ENV, f"block {mine} against vLLM's {stock} at {rows} rows for experts {key[0]}: {detail}")
        _STATE["failed"][key] = detail
        say(f"stood aside for experts {key[0]}: block {mine} against vLLM's {stock} at {rows} rows: {detail}; "
            f"vLLM's block stays")
    return ok


def choose(rows, stock, scope, fn):
    if _STATE["forced"] is not None:
        return _STATE["forced"]
    want = band(int(rows))
    if want is None or want == stock:
        return stock
    key = _key(scope)
    if key in _STATE["failed"]:
        return stock
    if key not in _STATE["verified"]:
        if _capturing():
            if key not in _STATE["said"]:
                _STATE["said"].add(key)
                say(f"unchecked when first met inside a graph capture for experts {key[0]}; vLLM's block "
                    f"{stock} is captured")
            return stock
        if not check(scope, fn, stock, want, key):
            return stock
    _STATE["fired"][want] = _STATE["fired"].get(want, 0) + 1
    return want


def tally():
    return {"bands": [list(one) for one in bands()], "verified": len(_STATE["verified"]),
            "failed": len(_STATE["failed"]), "fired": dict(_STATE["fired"])}


SUM_ENV = "TAAF_VLLM_MOE_SUM_EPI"
SUM_CHECKS = 3
_SUM = {"on": None, "op": None, "op_int8": None, "failed": {}, "checked": {}, "stash": None, "said": set(), "fired": 0,
        "aside": "", "aside_int8": "", "zeroed": None, "prezeroed": 0, "reader": False, "deferred": None,
        "handed": 0, "hand_aside": ""}


def sum_say(key, message):
    if key in _SUM["said"]:
        return
    _SUM["said"].add(key)
    print(f"MOE_SUM_EPI {message}", file=sys.stderr, flush=True)


def sum_on():
    if _SUM["on"] is None:
        said = str(os.environ.get(SUM_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{SUM_ENV}={said!r} is not '', '0' or '1'")
        _SUM["on"] = said == "1"
    return _SUM["on"]


def _act(scope):
    return "int8" if scope.get("input_dtype") == torch.int8 else "bf16"


def sum_op(topk, act="bf16"):
    slot, aside = ("op", "aside") if act == "bf16" else ("op_int8", "aside_int8")
    if _SUM[slot] is None and not _SUM[aside]:
        try:
            from vllm import taaf_marlin_sum
            _SUM[slot] = taaf_marlin_sum.build(topk, say=lambda line: sum_say(("built", act), line), act=act)
        except Exception as failure:
            _SUM[aside] = f"the {act} Marlin build failed: {type(failure).__name__}: {str(failure)[-300:]}"
            selfcheck_fail(SUM_ENV, _SUM[aside])
            sum_say((aside,), f"stood aside: {_SUM[aside]}; Marlin MoE keeps the stock w2 and top-k sum for {act} "
                              f"input")
    return _SUM[slot]


def _sum_eligible(scope, depth=3, quantizing=False):
    caller = sys._getframe(depth).f_code.co_name
    if caller != "fused_marlin_moe":
        return False
    from vllm.scalar_type import scalar_types
    hidden = scope["hidden_states"]
    int8 = scope.get("input_dtype") == torch.int8 and (quantizing or scope.get("a_scales2") is not None)
    return ((scope.get("input_dtype") is None or int8) and scope.get("bias2") is None
            and scope.get("global_scale2") is None
            and scope.get("w2_zeros") is None and scope.get("expert_map") is None
            and not scope.get("apply_router_weight_on_input") and scope.get("quant_type") == scalar_types.uint4b8
            and hidden.dtype == torch.bfloat16 and int(scope["num_topk"]) > 1)


def _sum_call(op, scope, cache, stock):
    from vllm.scalar_type import scalar_types
    rows, hidden, inter, topk = int(scope["M"]), int(scope["K"]), int(scope["N"]), int(scope["num_topk"])
    a_scales = scope.get("a_scales2") if _act(scope) == "int8" else None
    args = [scope["intermediate_cache2"], cache, scope["w2"], None, scope["w2_scale"], a_scales, None, None,
            scope["workspace"], scope["sorted_token_ids"], scope["expert_ids"], scope["num_tokens_post_padded"],
            scope["topk_weights"], int(scope["block_size_m"]), 1, True, scalar_types.uint4b8.id, rows * topk, hidden,
            inter, not stock, True, False, -1, -1, -1]
    return args


def _stock_gemm():
    import vllm._custom_ops
    return torch.ops._moe_C.moe_wna16_marlin_gemm


def _int8_stock_gemm():
    try:
        import vllm._custom_ops
    except Exception:
        pass
    if not hasattr(torch.ops, "w4a8fix") or not hasattr(torch.ops.w4a8fix, "moe_wna16_marlin_gemm"):
        return None
    return torch.ops.w4a8fix.moe_wna16_marlin_gemm


def _sums_of(scope):
    rows, hidden = int(scope["M"]), int(scope["K"])
    return scope["intermediate_cache3"].view(-1).view(torch.float32)[:rows * hidden].view(rows, hidden)


def _apart(one, other):
    one_start, other_start = one.data_ptr(), other.data_ptr()
    one_end = one_start + one.numel() * one.element_size()
    other_end = other_start + other.numel() * other.element_size()
    return one_end <= other_start or other_end <= one_start


def _sum_target(scope, x):
    if scope is None or not sum_on() or x is not scope.get("intermediate_cache2"):
        return None
    if not _sum_eligible(scope, depth=4, quantizing=True):
        return None
    act = _act(scope)
    if act != "int8" or _SUM["aside_int8"] or _SUM["op_int8"] is None:
        return None
    key = (tuple(scope["w2"].shape), int(scope["num_topk"]), act)
    if key in _SUM["failed"] or _SUM["checked"].get(key, 0) < SUM_CHECKS:
        return None
    sums = _sums_of(scope)
    if not sums.is_contiguous() or not _apart(sums, x):
        return None
    return sums


def _zero(sums, held):
    if held is not None and held.data_ptr() == sums.data_ptr() and held.numel() == sums.numel():
        _SUM["prezeroed"] += 1
        if _SUM["prezeroed"] == 1:
            sum_say("prezeroed", "live: the w2 input quant also zeroes the fp32 sums, so the separate zero fill "
                                 "before the summing w2 is skipped")
        return
    sums.zero_()


def sum_epilogue(scope):
    held, _SUM["zeroed"] = _SUM["zeroed"], None
    if not sum_on() or not _sum_eligible(scope):
        return None
    act = _act(scope)
    if _SUM["aside" if act == "bf16" else "aside_int8"]:
        return None
    stock_gemm = _stock_gemm if act == "bf16" else _int8_stock_gemm
    if act == "int8" and stock_gemm() is None:
        sum_say(("no signed op",), "stood aside for int8 input: the signed-scale int8 Marlin op is not loaded, so "
                                   "there is no reference to check against; the stock w2 and top-k sum stay")
        return None
    key = (tuple(scope["w2"].shape), int(scope["num_topk"]), act)
    if key in _SUM["failed"]:
        return None
    capturing = _capturing()
    done = _SUM["checked"].get(key, 0)
    if capturing and (_SUM["op" if act == "bf16" else "op_int8"] is None or done < SUM_CHECKS):
        sum_say(("capture", key), f"a graph captured before the check keeps the stock w2 and top-k sum for experts "
                                  f"{key[0]}")
        return None
    op = sum_op(int(scope["num_topk"]), act)
    if op is None:
        return None
    rows, hidden, topk = int(scope["M"]), int(scope["K"]), int(scope["num_topk"])
    cache = scope["intermediate_cache3"]
    sums = _sums_of(scope)
    if not capturing and done < SUM_CHECKS:
        stock_out = torch.empty((rows * topk, hidden), dtype=cache.dtype, device=cache.device)
        held = _sum_call(None, scope, stock_out, True)
        want = stock_gemm()(*held)
        want = want.view(rows, topk, hidden).float().sum(dim=1)
        try:
            sums.zero_()
            op(*_sum_call(op, scope, cache, False))
            ok, detail = compare(want, sums)
        except Exception as failure:
            ok, detail = False, f"{type(failure).__name__}: {str(failure)[:200]}"
        if not ok:
            selfcheck_fail(SUM_ENV, f"experts {key[0]}: {detail}")
            _SUM["failed"][key] = detail
            sum_say(("failed", key), f"stood aside for experts {key[0]}: {detail}; the stock w2 and top-k sum stay")
            return None
        _SUM["checked"][key] = done + 1
        sum_say(("check", key, done + 1), f"check {done + 1}/{SUM_CHECKS}: {rows} rows, {act} input, experts {key[0]}: "
                                            f"{detail} against the stock w2 and an fp32 top-k sum")
    elif capturing:
        _zero(sums, held)
        op(*_sum_call(op, scope, cache, False))
    else:
        try:
            _zero(sums, held)
            op(*_sum_call(op, scope, cache, False))
        except Exception as failure:
            selfcheck_fail(SUM_ENV, f"experts {key[0]}: {type(failure).__name__}: {str(failure)[:200]}")
            _SUM["failed"][key] = f"{type(failure).__name__}: {str(failure)[:200]}"
            sum_say(("failed", key), f"stood aside for experts {key[0]}: {_SUM['failed'][key]}; the stock w2 and "
                                     f"top-k sum stay")
            return None
    _SUM["stash"] = sums
    _SUM["fired"] += 1
    return cache


def take_summed(output):
    held = _SUM["stash"]
    if held is None:
        return False
    _SUM["stash"] = None
    settle()
    reader, _SUM["reader"] = _SUM["reader"], False
    if reader and not _SUM["hand_aside"] and (_SUM["handed"] or not _capturing()) \
            and tuple(held.shape) == tuple(output.shape) and output.dtype == torch.bfloat16 \
            and output.is_contiguous():
        _SUM["deferred"] = (output, held)
        return True
    output.copy_(held)
    return True


def expect_reader(ready):
    _SUM["reader"] = bool(ready)


def settle():
    held, _SUM["deferred"] = _SUM["deferred"], None
    if held is not None:
        held[0].copy_(held[1])


def claim(fused):
    held, _SUM["deferred"] = _SUM["deferred"], None
    if held is None:
        return None
    output, sums = held
    if fused.data_ptr() == output.data_ptr() and tuple(fused.shape) == tuple(output.shape) \
            and fused.stride() == output.stride():
        _SUM["handed"] += 1
        if _SUM["handed"] == 1:
            sum_say("handed", "live: the shared-expert add reads the fp32 sums and rounds them to BF16 itself, so the "
                              "separate BF16 copy after the summing w2 is skipped")
        return sums
    output.copy_(sums)
    _SUM["hand_aside"] = (f"the add read a tensor other than the w2 output (shape {tuple(fused.shape)} against "
                          f"{tuple(output.shape)})")
    sum_say("hand aside", f"stood aside from handing the sums to the add: {_SUM['hand_aside']}; the BF16 copy "
                          f"stays")
    if fused.data_ptr() != output.data_ptr() and tuple(fused.shape) == tuple(sums.shape):
        return sums
    return None


SILU_ENV = "TAAF_VLLM_MOE_SILU_EPI"
SILU_CHECKS = 3
SILU_CHUNK = 64
SILU_SIGMA = 1.0
PREMIX_ENV = "TAAF_VLLM_MOE_SILU_PREMIX"
SILU_BATCH = 32
_SILU = {"on": None, "op": None, "op_int8": None, "aside": "", "aside_int8": "", "failed": {}, "checked": {},
         "mixed": {}, "said": set(), "fired": 0, "premixed": set(), "premix_count": 0, "premix_off": "",
         "fallback": 0}


def silu_say(key, message):
    if key in _SILU["said"]:
        return
    _SILU["said"].add(key)
    print(f"MOE_SILU_EPI {message}", file=sys.stderr, flush=True)


def silu_on():
    if _SILU["on"] is None:
        said = str(os.environ.get(SILU_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{SILU_ENV}={said!r} is not '', '0' or '1'")
        _SILU["on"] = said == "1"
    return _SILU["on"]


def silu_op(topk, act="bf16"):
    slot, aside = ("op", "aside") if act == "bf16" else ("op_int8", "aside_int8")
    if _SILU[slot] is None and not _SILU[aside]:
        try:
            from vllm import taaf_marlin_sum
            _SILU[slot] = taaf_marlin_sum.build(topk, say=lambda line: silu_say(("built", act), line), kind="silu",
                                                act=act)
        except Exception as failure:
            _SILU[aside] = f"the {act} Marlin build failed: {type(failure).__name__}: {str(failure)[-300:]}"
            selfcheck_fail(SILU_ENV, _SILU[aside])
            silu_say((aside,), f"stood aside: {_SILU[aside]}; Marlin MoE keeps the stock w1 and SiLU for {act} input")
    return _SILU[slot]


def _plain_activation(func):
    return func is None or getattr(func, "__name__", "") == "activation"


def _plain_config(config):
    return config is None or (getattr(config, "clamp_limit", None) is None and getattr(config, "alpha", 1.0) == 1.0
                              and getattr(config, "beta", 0.0) == 0.0
                              and getattr(config, "activation_situ_beta", None) is None
                              and getattr(config, "activation_situ_linear_beta", None) is None)


def _silu_refusal(scope):
    from vllm.scalar_type import scalar_types
    activation = scope.get("activation")
    hidden = scope["hidden_states"]
    checks = (
        ("an fp8 input", scope.get("input_dtype") in (None, torch.int8)),
        ("an int8 input without its activation scales",
         scope.get("input_dtype") != torch.int8 or (scope.get("a_scales1") is not None
                                                    and scope.get("gate_up_input") is not None)),
        ("a w1 bias", scope.get("bias1") is None),
        ("a w1 global scale", scope.get("global_scale1") is None),
        ("w1 zero points", scope.get("w1_zeros") is None),
        ("an expert map", scope.get("expert_map") is None),
        (f"activation_func {getattr(scope.get('activation_func'), '__name__', scope.get('activation_func'))!r}",
         _plain_activation(scope.get("activation_func"))),
        (f"activation_config {scope.get('activation_config')!r}", _plain_config(scope.get("activation_config"))),
        ("router weights on the input", not scope.get("apply_router_weight_on_input")),
        ("a quant type other than uint4b8", scope.get("quant_type") == scalar_types.uint4b8),
        (f"{hidden.dtype} activations", hidden.dtype == torch.bfloat16),
        (f"activation {getattr(activation, 'name', activation)}", getattr(activation, "name", str(activation)) == "SILU"),
        ("an ungated w1", int(scope.get("w13_num_shards", 2)) == 2),
        (f"2N = {2 * int(scope['N'])} not a multiple of {2 * SILU_CHUNK}", (2 * int(scope["N"])) % (2 * SILU_CHUNK) == 0),
    )
    return next((why for why, ok in checks if not ok), "")


def _silu_eligible(scope):
    why = _silu_refusal(scope)
    if why:
        silu_say(("ineligible", why), f"stood aside for a call with {why}; that call keeps the stock w1 and SiLU")
    return not why


def _order(chunks, device, inverse=False):
    half = chunks // 2
    order = torch.stack((torch.arange(half, device=device), torch.arange(half, chunks, device=device)), dim=1)
    order = order.reshape(-1)
    if not inverse:
        return order
    back = torch.empty_like(order)
    back[order] = torch.arange(chunks, device=device)
    return back


def mix(w1, w1_scale, width, inverse=False):
    chunks = width // SILU_CHUNK
    order = _order(chunks, w1.device, inverse)
    experts = w1.shape[0]
    with torch.no_grad():
        for begin in range(0, experts, SILU_BATCH):
            part = slice(begin, min(experts, begin + SILU_BATCH))
            words = w1[part]
            words.copy_(words.reshape(words.shape[0], words.shape[1], chunks, -1)[:, :, order]
                        .reshape(words.shape))
            scales = w1_scale[part]
            scales.copy_(scales.reshape(scales.shape[0], scales.shape[1], chunks, -1)[:, :, order]
                         .reshape(scales.shape))


def _gemm_args(scope, out, stock):
    from vllm.scalar_type import scalar_types
    rows, hidden, inter, topk = int(scope["M"]), int(scope["K"]), int(scope["N"]), int(scope["num_topk"])
    int8 = _act(scope) == "int8"
    return [scope["gate_up_input"] if int8 else scope["hidden_states"], out, scope["w1"], None, scope["w1_scale"],
            scope["a_scales1"] if int8 else None, None, None,
            scope["workspace"], scope["sorted_token_ids"], scope["expert_ids"], scope["num_tokens_post_padded"],
            scope["topk_weights"], int(scope["block_size_m"]), topk, False, scalar_types.uint4b8.id, rows,
            2 * inter, hidden, False, True, False, -1, -1, -1]


def _sane_scope(scope, seed):
    held = scope["gate_up_input"]
    scales = scope["a_scales1"]
    gen = torch.Generator(device=held.device).manual_seed(seed)
    hidden = (torch.randn(held.shape, generator=gen, device=held.device) * SILU_SIGMA).to(torch.bfloat16).float()
    step = (hidden.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
    quantized = torch.clamp(torch.round(hidden / step), -128, 127).to(held.dtype)
    probe = dict(scope, gate_up_input=quantized,
                 a_scales1=step.reshape(scales.shape).to(scales.dtype) if step.numel() == scales.numel()
                 else torch.full_like(scales, SILU_SIGMA * 4.0 / 127.0))
    weights = scope.get("topk_weights")
    if torch.is_tensor(weights) and not bool(torch.isfinite(weights).all()):
        probe["topk_weights"] = torch.full_like(weights, 1.0 / max(1, int(scope["num_topk"])))
    return probe


def _stock_silu(scope, mixed):
    rows, inter, topk = int(scope["M"]), int(scope["N"]), int(scope["num_topk"])
    made = torch.zeros((rows * topk, 2 * inter), dtype=scope["hidden_states"].dtype,
                       device=scope["hidden_states"].device)
    made = (_stock_gemm if _act(scope) == "bf16" else _int8_stock_gemm)()(*_gemm_args(scope, made, True))
    if mixed:
        chunks = 2 * inter // SILU_CHUNK
        made = made.view(rows * topk, chunks, SILU_CHUNK)[:, _order(chunks, made.device, inverse=True)]
        made = made.reshape(rows * topk, 2 * inter)
    return torch.nn.functional.silu(made[:, :inter].float()) * made[:, inter:].float()


def _put_back_kernel():
    if _SILU.get("put_back") is None:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        @triton.jit
        def put_back(made_ptr, out_ptr, stride_made, stride_out, INTER: tl.constexpr, CHUNK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            pair = tl.program_id(1)
            cols = tl.arange(0, CHUNK)
            gate = tl.load(made_ptr + row * stride_made + (2 * pair) * CHUNK + cols).to(tl.float32)
            up = tl.load(made_ptr + row * stride_made + (2 * pair + 1) * CHUNK + cols).to(tl.float32)
            act = (gate / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            tl.store(out_ptr + row * stride_out + pair * CHUNK + cols, (act * up).to(tl.bfloat16))

        _SILU["put_back"] = put_back
    return _SILU["put_back"]


def put_back_silu(made, out, inter):
    rows = made.shape[0]
    if (made.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or made.stride(1) != 1 or out.stride(1) != 1
            or inter % SILU_CHUNK or not (made.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")):
        chunks = 2 * inter // SILU_CHUNK
        plain = made.view(rows, chunks, SILU_CHUNK)[:, _order(chunks, made.device, inverse=True)]
        plain = plain.reshape(rows, 2 * inter)
        out.copy_(torch.nn.functional.silu(plain[:, :inter]) * plain[:, inter:])
        return out
    if rows:
        _put_back_kernel()[(rows, inter // SILU_CHUNK)](made, out, made.stride(0), out.stride(0), INTER=inter,
                                                         CHUNK=SILU_CHUNK, num_warps=1)
    return out


def _stock_put_back(scope, out):
    rows, inter, topk = int(scope["M"]), int(scope["N"]), int(scope["num_topk"])
    made = torch.empty((rows * topk, 2 * inter), dtype=scope["hidden_states"].dtype,
                       device=scope["hidden_states"].device)
    made = (_stock_gemm if _act(scope) == "bf16" else _int8_stock_gemm)()(*_gemm_args(scope, made, True))
    return put_back_silu(made, out, inter)


def _unmix(scope, key, detail, failed=True):
    if failed:
        selfcheck_fail(SILU_ENV, f"experts {key[1]}: {detail}")
    mix(scope["w1"], scope["w1_scale"], 2 * int(scope["N"]), inverse=True)
    _SILU["mixed"].pop(key, None)
    _SILU["failed"][key] = detail
    silu_say(("failed", key), f"stood aside for experts {key[1]}: {detail}; the w1 columns are put back and the "
                              f"stock w1 and SiLU stay")


def _permute_last(tensor, chunks):
    order = _order(chunks, tensor.device)
    with torch.no_grad():
        for begin in range(0, tensor.shape[0], SILU_BATCH):
            part = tensor[begin:begin + SILU_BATCH]
            part.copy_(part.reshape(*part.shape[:-1], chunks, -1)[..., order, :].reshape(part.shape))


def premix_on():
    said = str(os.environ.get(PREMIX_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise ValueError(f"{PREMIX_ENV}={said!r} is not '', '0' or '1'")
    return said != "0"


def silu_premix(layer):
    if not silu_on() or _SILU["premix_off"]:
        return False
    if not premix_on():
        _SILU["premix_off"] = f"{PREMIX_ENV}=0"
        silu_say(("premix off",), f"int8 w1 not interleaved at load ({PREMIX_ENV}=0); int8 calls keep the stock w1 "
                                  f"and SiLU, bf16 calls keep the fused w1")
        return False
    try:
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import marlin_moe_padded_intermediate
        weight, scales = layer.w13_qweight, layer.w13_scales
        zeros = getattr(layer, "w13_qzeros", None)
        inter = int(layer.intermediate_size_per_partition)
        width = int(weight.shape[-1])
        chunks = width // SILU_CHUNK
        checks = (
            (f"w13 width {width} is not 2 x {inter}", width == 2 * inter),
            (f"w13 width {width} is not a multiple of {2 * SILU_CHUNK}", width % (2 * SILU_CHUNK) == 0),
            ("the intermediate size is padded for Marlin", marlin_moe_padded_intermediate(inter, 128) == inter),
            ("w13 scales do not match the weight columns", int(scales.shape[-1]) == width),
            ("w13 zero points do not pack 8 columns a word", zeros is None or int(zeros.shape[-1]) * 8 == width),
        )
        why = next((text for text, ok in checks if not ok), "")
    except Exception as failure:
        why = f"{type(failure).__name__}: {str(failure)[:200]}"
    if why:
        if _SILU["premix_count"]:
            raise RuntimeError(f"MOE_SILU_EPI premix refused a layer after {_SILU['premix_count']} were interleaved: "
                               f"{why}")
        _SILU["premix_off"] = why
        silu_say(("premix off",), f"int8 w1 not interleaved at load: {why}; int8 calls keep the stock w1 and SiLU")
        return False
    _permute_last(weight.data, chunks)
    _permute_last(scales.data, chunks)
    if zeros is not None:
        _permute_last(zeros.data, chunks)
    _SILU["premix_count"] += 1
    return True


def silu_premixed(layer):
    _SILU["premixed"].add(tuple(layer.w13_qweight.shape))
    if _SILU["premix_count"] == 1:
        silu_say(("premixed",), f"int8 w1 gate and up columns interleaved in {SILU_CHUNK}-column chunks before the "
                                f"Marlin repack; converted experts {tuple(layer.w13_qweight.shape)}")


def _fallback(scope, out, why):
    _SILU["fallback"] += 1
    silu_say(("fallback", why), f"an interleaved int8 w1 runs the stock w1 with its columns put back and SiLU "
                                f"({why})")
    return _stock_put_back(scope, out)


def _premixed_w1(scope):
    key = (scope["w1"].data_ptr(), tuple(scope["w1"].shape))
    out = scope["intermediate_cache2"]
    why = _silu_refusal(scope)
    if why:
        return _fallback(scope, out, f"a call with {why}")
    if key in _SILU["failed"]:
        return _fallback(scope, out, "the fused w1 failed its check")
    op = silu_op(int(scope["num_topk"]), "int8")
    if op is None:
        return _fallback(scope, out, "no int8 build")
    capturing = _capturing()
    done = _SILU["checked"].get(key, 0)
    if capturing and done < SILU_CHECKS:
        return _fallback(scope, out, f"a graph captured after {done} of {SILU_CHECKS} checks, so this layer keeps "
                                     f"the stock w1 for the rest of the process")
    if not capturing and done < SILU_CHECKS:
        try:
            probe = _sane_scope(scope, 1000 * done + 17)
            probe_out = torch.zeros_like(out)
            want = _stock_silu(probe, True)
            op(*_gemm_args(probe, probe_out, False))
            ok, detail = compare(want, probe_out)
        except Exception as failure:
            ok, detail = False, f"{type(failure).__name__}: {str(failure)[:200]}"
        if not ok:
            selfcheck_fail(SILU_ENV, f"interleaved int8 w1, experts {key[1]}: {detail}")
            _SILU["failed"][key] = detail
            silu_say(("failed", key), f"stood aside for experts {key[1]}: {detail}; this interleaved int8 w1 runs "
                                      f"the stock w1 with its columns put back")
            return _fallback(scope, out, "the fused w1 failed its check")
        _SILU["checked"][key] = done + 1
        silu_say(("check", key, done + 1), f"check {done + 1}/{SILU_CHECKS}: {int(scope['M'])} rows, synthetic "
                                             f"N(0,{SILU_SIGMA}) int8 input quantized per token, w1 interleaved at load, on the call's own "
                                             f"routing, experts {key[1]}: {detail} against the stock w1 and SiLU")
    try:
        op(*_gemm_args(scope, out, False))
    except Exception as failure:
        if capturing:
            raise
        selfcheck_fail(SILU_ENV, f"interleaved int8 w1, experts {key[1]}: {type(failure).__name__}: "
                                 f"{str(failure)[:200]}")
        _SILU["failed"][key] = f"{type(failure).__name__}: {str(failure)[:200]}"
        return _fallback(scope, out, "the fused w1 raised")
    _SILU["fired"] += 1
    return out


def silu_w1(scope):
    if not silu_on():
        return None
    if _act(scope) == "int8" and tuple(scope["w1"].shape) in _SILU["premixed"]:
        return _premixed_w1(scope)
    if not _silu_eligible(scope):
        return None
    act = _act(scope)
    if act == "int8":
        silu_say(("int8 not premixed",), "stood aside for int8 input: its w1 was not interleaved at load, and the "
                                         "in-place interleave does not hold for the int8 Marlin layout; the stock w1 "
                                         "and SiLU stay")
        return None
    if _SILU["aside" if act == "bf16" else "aside_int8"]:
        return None
    if act == "int8" and _int8_stock_gemm() is None:
        silu_say(("no signed op",), "stood aside for int8 input: the signed-scale int8 Marlin op is not loaded, so "
                                    "there is no reference to check against; the stock w1 and SiLU stay")
        return None
    key = (scope["w1"].data_ptr(), tuple(scope["w1"].shape))
    if key in _SILU["failed"]:
        return None
    capturing = _capturing()
    mixed = key in _SILU["mixed"]
    silu_say("seen", f"first eligible w1 call: {int(scope['M'])} rows, experts {key[1]}, "
                     f"{'inside' if capturing else 'outside'} a graph capture")
    if capturing and not mixed:
        silu_say(("capture", key), f"a graph captured before the check keeps the stock w1 and SiLU for experts "
                                   f"{key[1]}")
        return None
    op = silu_op(int(scope["num_topk"]), act)
    if op is None:
        if mixed:
            _unmix(scope, key, "no build", failed=False)
        return None
    out = scope["intermediate_cache2"]
    done = _SILU["checked"].get(key, 0)
    if not capturing and done < SILU_CHECKS:
        try:
            want = _stock_silu(scope, mixed)
            if not mixed:
                mix(scope["w1"], scope["w1_scale"], 2 * int(scope["N"]))
                _SILU["mixed"][key] = True
            op(*_gemm_args(scope, out, False))
            ok, detail = compare(want, out)
        except Exception as failure:
            ok, detail = False, f"{type(failure).__name__}: {str(failure)[:200]}"
        if not ok:
            if key in _SILU["mixed"]:
                _unmix(scope, key, detail)
            else:
                _SILU["failed"][key] = detail
            return None
        _SILU["checked"][key] = done + 1
        silu_say(("check", key, done + 1), f"check {done + 1}/{SILU_CHECKS}: {int(scope['M'])} rows, {act} input, "
                                             f"experts {key[1]}: {detail} against the stock w1 and silu_and_mul")
    elif capturing and done < SILU_CHECKS:
        silu_say(("capture early", key), f"a graph captured after {done} of {SILU_CHECKS} checks runs the stock w1 "
                                         f"with its columns put back and SiLU for experts {key[1]}")
        return _stock_put_back(scope, out)
    else:
        try:
            op(*_gemm_args(scope, out, False))
        except Exception as failure:
            if capturing:
                raise
            _unmix(scope, key, f"{type(failure).__name__}: {str(failure)[:200]}")
            return None
    _SILU["fired"] += 1
    return out


def silu_gemm(done, stock):
    if done is None:
        return stock
    return lambda *args, **kwargs: None


QUANT_ENV = "TAAF_VLLM_MOE_QUANT_FUSED"
QUANT_CHECKS = 3
_QUANT = {"on": None, "kernel": None, "aside": "", "checked": 0, "said": set(), "fired": 0}


def quant_say(key, message):
    if key in _QUANT["said"]:
        return
    _QUANT["said"].add(key)
    print(f"MOE_QUANT_FUSED {message}", file=sys.stderr, flush=True)


def quant_on():
    if _QUANT["on"] is None:
        said = str(os.environ.get(QUANT_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{QUANT_ENV}={said!r} is not '', '0' or '1'")
        _QUANT["on"] = said == "1"
    return _QUANT["on"]


def _quant_kernel():
    if _QUANT["kernel"] is None:
        from vllm.triton_utils import tl, triton

        @triton.jit
        def quant_scaled(x_ptr, xq_ptr, s_ptr, g_ptr, z_ptr, stride_x, stride_xq, N, Z, HAS_G: tl.constexpr,
                         BLOCK: tl.constexpr, ZERO: tl.constexpr, PER: tl.constexpr):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
            scale = absmax / 127
            x_q = tl.extra.cuda.libdevice.round(x * (127 / absmax)).to(tl.int8)
            tl.store(xq_ptr + row * stride_xq + cols, x_q, mask=mask)
            if HAS_G:
                scale = scale * tl.load(g_ptr).to(tl.float32)
            tl.store(s_ptr + row, scale)
            if ZERO:
                spots = row.to(tl.int64) * PER + tl.arange(0, PER)
                tl.store(z_ptr + spots, tl.zeros((PER,), dtype=tl.float32), mask=spots < Z)
        _QUANT["kernel"] = (quant_scaled, triton.next_power_of_2)
    return _QUANT["kernel"]


def _quant_stock(x, scale):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_quant_input
    quantized, scales = marlin_quant_input(x, torch.int8)
    return quantized, (scales * scale if scale is not None else scales)


def _quant_fused(x, scale, zero=None):
    kernel, power = _quant_kernel()
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    rows, width = flat.shape
    quantized = torch.empty((rows, width), device=x.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    block = power(width)
    clear = zero is not None and rows > 0
    kernel[(rows,)](flat, quantized, scales, scales if scale is None else scale, zero if clear else scales,
                    flat.stride(-2), quantized.stride(-2), width, zero.numel() if clear else 0,
                    HAS_G=scale is not None, BLOCK=block, ZERO=clear,
                    PER=power(-(-zero.numel() // rows)) if clear else 1,
                    num_warps=min(max(block // 256, 1), 8), num_stages=1)
    return quantized.view(*shape), scales.view(*shape[:-1], 1)


ACT_QUANT_ENV = "TAAF_VLLM_MOE_ACT_QUANT"
ACT_QUANT_CHECKS = 3
ACT_QUANT_ROWS = 8
_ACT_QUANT = {"on": None, "kernel": None, "aside": "", "checked": 0, "said": set(), "fired": 0, "pending": None}


def act_quant_say(key, message):
    if key in _ACT_QUANT["said"]:
        return
    _ACT_QUANT["said"].add(key)
    print(f"MOE_ACT_QUANT {message}", file=sys.stderr, flush=True)


def act_quant_on():
    if _ACT_QUANT["on"] is None:
        said = str(os.environ.get(ACT_QUANT_ENV) or "").strip()
        if said not in ("", "0", "1"):
            raise ValueError(f"{ACT_QUANT_ENV}={said!r} is not '', '0' or '1'")
        _ACT_QUANT["on"] = said == "1"
    return _ACT_QUANT["on"]


def _act_quant_kernel():
    if _ACT_QUANT["kernel"] is None:
        from vllm.triton_utils import tl, triton
        from triton.language.extra import libdevice

        @triton.jit
        def silu_mul_quant(x_ptr, xq_ptr, s_ptr, g_ptr, stride_x, stride_xq, rows, N, HAS_G: tl.constexpr,
                           BLOCK: tl.constexpr, ROWS: tl.constexpr):
            row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
            cols = tl.arange(0, BLOCK)
            live = row < rows
            mask = live[:, None] & (cols[None, :] < N)
            base = x_ptr + row[:, None] * stride_x + cols[None, :]
            gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(base + N, mask=mask, other=0.0).to(tl.float32)
            act = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
            x = (act * up).to(tl.bfloat16).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
            scale = absmax / 127
            x_q = libdevice.round(x * (127 / absmax)[:, None]).to(tl.int8)
            tl.store(xq_ptr + row[:, None] * stride_xq + cols[None, :], x_q, mask=mask)
            if HAS_G:
                scale = scale * tl.load(g_ptr).to(tl.float32)
            tl.store(s_ptr + row, scale, mask=live)

        _ACT_QUANT["kernel"] = (silu_mul_quant, triton.next_power_of_2)
    return _ACT_QUANT["kernel"]


def _act_quant_fused(made, inter, scale):
    from vllm.triton_utils import triton

    kernel, power = _act_quant_kernel()
    rows = made.shape[0]
    codes = torch.empty((rows, inter), device=made.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=made.device, dtype=torch.float32)
    block = power(inter)
    per = max(1, min(ACT_QUANT_ROWS, 1024 // block))
    if rows:
        kernel[(triton.cdiv(rows, per),)](made, codes, scales, scales if scale is None else scale, made.stride(0),
                                          codes.stride(0), rows, inter, HAS_G=scale is not None, BLOCK=block,
                                          ROWS=per, num_warps=4, num_stages=1)
    return codes, scales


def _act_quant_stock(made, out, scale):
    torch.ops._C.silu_and_mul(out, made)
    return _quant_stock(out, scale)


def _act_quant_refusal(scope):
    if scope.get("input_dtype") != torch.int8:
        return "input other than int8"
    if scope.get("activation_func") is not None:
        return "a custom activation function"
    if str(getattr(scope.get("activation"), "value", scope.get("activation"))) != "silu":
        return f"activation {scope.get('activation')}"
    config = scope.get("activation_config")
    if config is not None and getattr(config, "clamp_limit", None) is not None:
        return "a clamped SiLU"
    if scope.get("_taaf_act") is not None:
        return "the SiLU epilogue in w1"
    scale = scope.get("input_global_scale2")
    if scale is not None and (scale.numel() != 1 or not scale.is_floating_point()):
        return f"a global scale of shape {tuple(scale.shape)}"
    made = scope.get("intermediate_cache1")
    if made is None or made.dtype != torch.bfloat16:
        return "a w1 output other than bf16"
    return ""


def act_quant_probe(made):
    generator = torch.Generator(device=made.device).manual_seed(int(made.shape[0]) * 7919 + int(made.shape[1]))
    rows = torch.randn(made.shape, generator=generator, device=made.device, dtype=torch.float32)
    return (rows * 4).to(made.dtype)


def act_quant_compare(made, out, inter, scale, label, want=None):
    try:
        want_q, want_s = _act_quant_stock(made, out, scale) if want is None else want
        got_q, got_s = _act_quant_fused(made, inter, scale)
        same = torch.equal(want_q, got_q) and torch.equal(want_s.float(), got_s)
        detail = "bit-exact" if same else (f"{label}: {int((want_q != got_q).sum())} codes and "
                                           f"{int((want_s.float() != got_s).sum())} scales differ")
    except Exception as failure:
        same, detail = False, f"{label}: {type(failure).__name__}: {str(failure)[:200]}"
    return same, detail


def act_quant(scope):
    _ACT_QUANT["pending"] = None
    if not act_quant_on() or _ACT_QUANT["aside"]:
        return False
    why = _act_quant_refusal(scope)
    if why:
        act_quant_say(("refused", why), f"stood aside for a call with {why}; that call keeps silu_and_mul and the "
                                        f"quant")
        return False
    out = scope["intermediate_cache2"]
    inter = int(out.shape[-1])
    made = scope["intermediate_cache1"].view(-1, 2 * inter)
    if made.stride(1) != 1 or made.shape[0] != out.shape[0]:
        act_quant_say(("layout",), f"stood aside for a w1 output of shape {tuple(made.shape)} against "
                                   f"{tuple(out.shape)}; that call keeps silu_and_mul and the quant")
        return False
    scale = scope.get("input_global_scale2")
    if _ACT_QUANT["checked"] < ACT_QUANT_CHECKS:
        if _capturing():
            act_quant_say("capture", "a graph captured before the checks keeps silu_and_mul and the quant")
            return False
        probe = act_quant_probe(made)
        same, detail = act_quant_compare(probe, torch.empty_like(out), inter, scale, "random rows")
        want_q, want_s = _act_quant_stock(made, out, scale)
        if same:
            same, detail = act_quant_compare(made, None, inter, scale, "the call's rows", (want_q, want_s))
            detail = f"{detail}; random rows bit-exact"
        if not same:
            selfcheck_fail(ACT_QUANT_ENV, detail)
            _ACT_QUANT["aside"] = detail
            act_quant_say("aside", f"stood aside: {detail} against silu_and_mul and the stock quant; they stay")
        else:
            _ACT_QUANT["checked"] += 1
            act_quant_say(("check", _ACT_QUANT["checked"]),
                          f"check {_ACT_QUANT['checked']}/{ACT_QUANT_CHECKS}: {int(made.shape[0])} rows x {inter}: "
                          f"{detail} against silu_and_mul and the stock quant")
        _ACT_QUANT["pending"] = (out, want_q, want_s)
        return True
    _ACT_QUANT["fired"] += 1
    if _ACT_QUANT["fired"] == 1:
        act_quant_say("live", "live: the w2 input is the SiLU of w1's gate times its up half, quantized to int8 with "
                              "its per-token scale in one Triton kernel that reads w1's output once; silu_and_mul's "
                              "BF16 buffer and the separate quant are skipped")
    got_q, got_s = _act_quant_fused(made, inter, scale)
    _ACT_QUANT["pending"] = (out, got_q, got_s)
    return True


def quant_int8(x, scale, scope=None):
    _SUM["zeroed"] = None
    pending, _ACT_QUANT["pending"] = _ACT_QUANT["pending"], None
    if pending is not None and pending[0] is x:
        return pending[1], pending[2]
    if not quant_on() or _QUANT["aside"]:
        return _quant_stock(x, scale)
    if scale is not None and (scale.numel() != 1 or not scale.is_floating_point()):
        _QUANT["aside"] = f"a global scale of shape {tuple(scale.shape)} and dtype {scale.dtype}"
        quant_say("aside", f"stood aside for {_QUANT['aside']}; the stock quant and scale multiply stay")
        return _quant_stock(x, scale)
    capturing = _capturing()
    if _QUANT["checked"] < QUANT_CHECKS:
        if capturing:
            quant_say("capture", "a graph captured before the checks keeps the stock quant and scale multiply")
            return _quant_stock(x, scale)
        want_q, want_s = _quant_stock(x, scale)
        try:
            got_q, got_s = _quant_fused(x, scale)
            same = torch.equal(want_q, got_q) and torch.equal(want_s.float(), got_s)
            detail = "bit-exact" if same else (f"{int((want_q != got_q).sum())} codes and "
                                               f"{int((want_s.float() != got_s).sum())} scales differ")
        except Exception as failure:
            same, detail = False, f"{type(failure).__name__}: {str(failure)[:200]}"
        if not same:
            selfcheck_fail(QUANT_ENV, detail)
            _QUANT["aside"] = detail
            quant_say("aside", f"stood aside: {detail} against the stock quant and scale multiply; they stay")
            return want_q, want_s
        _QUANT["checked"] += 1
        quant_say(("check", _QUANT["checked"]), f"check {_QUANT['checked']}/{QUANT_CHECKS}: {int(x.shape[0])} rows "
                                                 f"x {int(x.shape[-1])}: {detail} against the stock quant and "
                                                 f"scale multiply")
        return got_q, got_s
    _QUANT["fired"] += 1
    zero = _sum_target(scope, x)
    if zero is None:
        return _quant_fused(x, scale)
    made = _quant_fused(x, scale, zero)
    _SUM["zeroed"] = zero
    return made
