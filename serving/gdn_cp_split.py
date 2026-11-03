import importlib
import inspect
import os
import sys
import time

import torch

ENV = "TAAF_VLLM_GDN_CP_SPLIT"
STOCK = 512
FINE = 64
CHECKS = 3
SLACK = 2.0
FLOOR = 2.0 ** -6
LONE_TOLERANCE = 2.0e-2
TALLY_S = 15.0
CP_MODULE = "flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm120"
CHOOSER = "choose_cp_chunk_len_host"
_STATE = {"wanted": None, "seen": None, "aside": "", "checked": 0, "hooked": None, "at": 0.0,
          "granularity": None}
_TALLY = {"split": 0, "same": 0, "other": 0, "ctas_stock": 0, "ctas_split": 0}
_SAID = set()
_WRAPPED = {}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def granularity():
    if _STATE["granularity"] is None:
        said = str(os.environ.get(ENV) or "").strip().lower()
        if said in ("", "0"):
            value = 0
        elif said == "1":
            value = FINE
        elif said.isdigit() and int(said) % FINE == 0 and FINE <= int(said) < STOCK:
            value = int(said)
        else:
            raise ValueError(f"{ENV}={said!r} is not '', '0', '1' or a multiple of {FINE} below {STOCK}")
        _STATE["granularity"] = value
    return _STATE["granularity"]


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def stand_aside(reason, failed=True):
    _STATE["aside"] = reason
    if failed:
        selfcheck_fail(ENV, reason)
    say("aside", f"GDN_CP_SPLIT stands aside: {reason}; every GDN chunked prefill keeps the stock "
                 f"{STOCK}-token context-parallel split for the rest of the process")


def tally():
    now = time.monotonic()
    if now - _STATE["at"] < TALLY_S:
        return
    _STATE["at"] = now
    print(f"GDN_CP_SPLIT tally {dict(_TALLY)}", flush=True)


def _hook():
    if _STATE["hooked"] is not None:
        return _STATE["hooked"]
    try:
        module = importlib.import_module(CP_MODULE)
        original = getattr(module, CHOOSER)
        if "chunk_len_granularity" not in inspect.signature(original).parameters:
            raise TypeError(f"{CHOOSER} takes no chunk_len_granularity")
    except Exception as failure:
        _STATE["hooked"] = False
        stand_aside(f"hook {type(failure).__name__}: {failure}"[:300], failed=False)
        return False

    def choose(max_seqlen, num_heads, num_sms, *args, **kwargs):
        made = original(max_seqlen, num_heads, num_sms, *args, **kwargs)
        wanted = _STATE["wanted"]
        if wanted is None or args or kwargs.get("chunk_len_granularity", STOCK) != STOCK:
            return made
        split = original(max_seqlen, num_heads, num_sms, *args, **{**kwargs, "chunk_len_granularity": wanted})
        if -(-max_seqlen // split) <= 1:
            split = made
        _STATE["seen"] = (made, split, -(-max_seqlen // made) * num_heads, -(-max_seqlen // split) * num_heads)
        return split

    setattr(module, CHOOSER, choose)
    _STATE["hooked"] = True
    return True


def _capturing():
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _run(function, args, kwargs, wanted):
    _STATE["wanted"], _STATE["seen"] = wanted, None
    try:
        return function(*args, **kwargs), _STATE["seen"]
    finally:
        _STATE["wanted"] = None


def _parts(result):
    return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def _error(got, want):
    got, want = got.detach().float(), want.detach().float()
    if got.shape != want.shape or not bool(torch.isfinite(got).all()):
        return float("inf")
    if not got.numel():
        return 0.0
    return float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))


def _check(function, args, kwargs, made, seen):
    stock, _ = _run(function, args, kwargs, None)
    try:
        reference, _ = _run(function, args, {**kwargs, "use_cp": False}, None)
    except Exception as failure:
        reference = None
        say("lone", f"GDN_CP_SPLIT check has no sequential reference ({type(failure).__name__}: "
                    f"{str(failure)[:160]}); the split is held to {LONE_TOLERANCE:g} of the stock split")
    rows = []
    passed = True
    for name, got, want, base in zip(("out", "state"), _parts(made), _parts(stock),
                                     _parts(reference) if reference is not None else (None, None)):
        if base is None:
            split_error, stock_error = _error(got, want), 0.0
            bound = LONE_TOLERANCE
        else:
            split_error, stock_error = _error(got, base), _error(want, base)
            bound = max(SLACK * stock_error, FLOOR)
        passed &= split_error <= bound
        rows.append(f"{name}_split={split_error:.2e} {name}_stock={stock_error:.2e} {name}_bound={bound:.2e}")
    tokens = int(kwargs["q"].shape[0]) if "q" in kwargs else -1
    print(f"GDN_CP_SPLIT check tokens={tokens} chunk={seen[0]}->{seen[1]} ctas={seen[2]}->{seen[3]} "
          + " ".join(rows) + f" {'pass' if passed else 'FAIL'}", flush=True)
    if not passed:
        stand_aside(f"a check at {tokens} tokens left its bound")
        return stock
    _STATE["checked"] += 1
    return made


def call(function, args, kwargs):
    if _STATE["aside"] or not _hook():
        return function(*args, **kwargs)
    ready = _STATE["checked"] >= CHECKS
    if not ready and _capturing():
        _TALLY["other"] += 1
        return function(*args, **kwargs)
    made, seen = _run(function, args, kwargs, granularity())
    if seen is None:
        _TALLY["other"] += 1
    elif seen[0] == seen[1]:
        _TALLY["same"] += 1
    else:
        if not ready:
            made = _check(function, args, kwargs, made, seen)
            if _STATE["aside"]:
                return made
        _TALLY["split"] += 1
        _TALLY["ctas_stock"] += seen[2]
        _TALLY["ctas_split"] += seen[3]
        if _STATE["checked"] >= CHECKS:
            say("live", f"GDN_CP_SPLIT live: a single-sequence GDN chunked prefill picks its context-parallel "
                        f"chunk length on a {granularity()}-token grid instead of {STOCK}, so the chunks "
                        f"fill the SMs, after {CHECKS} checks against the sequential kernel")
    tally()
    return made


def wrap(function):
    if not granularity():
        return function
    held = _WRAPPED.get(id(function))
    if held is None or held[0] is not function:
        def run(*args, **kwargs):
            return call(function, args, kwargs)

        held = (function, run)
        _WRAPPED[id(function)] = held
    return held[1]
