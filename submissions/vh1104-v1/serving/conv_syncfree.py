import os
import sys
import time

import torch

SYNCFREE_ENV = "TAAF_VLLM_CONV_SYNCFREE"
CHECKS = 3
TALLY_S = 15.0
_SAID = set()
_STATE = {"aside": "", "checked": 0, "at": 0.0}
_TALLY = {"sized": 0, "checked": 0, "stock": 0}


SELFCHECK_FATAL_ENV = "TAAF_VLLM_SELFCHECK_FATAL"
SELFCHECK_EXIT = 70


def selfcheck_fail(knob, detail):
    print(f"SELFCHECK FAIL {knob}: {detail}", file=sys.stderr, flush=True)
    if str(os.environ.get(SELFCHECK_FATAL_ENV) or "1").strip().lower() not in ("0", "off", "false", "no"):
        os._exit(SELFCHECK_EXIT)


def on():
    return str(os.environ.get(SYNCFREE_ENV) or "").strip().lower() in ("1", "on", "true")


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def tally():
    now = time.monotonic()
    if now - _STATE["at"] < TALLY_S:
        return
    _STATE["at"] = now
    print(f"CONV_SYNCFREE tally {dict(_TALLY)}", flush=True)


def capturing(tensor):
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def token_group(req_group, query_lens, query_lens_cpu):
    if not on() or _STATE["aside"] or query_lens_cpu is None or query_lens_cpu.dim() != 1 \
            or query_lens_cpu.shape[0] != query_lens.shape[0]:
        _TALLY["stock"] += 1
        return torch.repeat_interleave(req_group, query_lens)
    if _STATE["checked"] < CHECKS and capturing(query_lens):
        _TALLY["stock"] += 1
        return torch.repeat_interleave(req_group, query_lens)
    total = int(query_lens_cpu.sum())
    if _STATE["checked"] >= CHECKS:
        _TALLY["sized"] += 1
        tally()
        return torch.repeat_interleave(req_group, query_lens, output_size=total)
    want = torch.repeat_interleave(req_group, query_lens)
    sized = torch.repeat_interleave(req_group, query_lens, output_size=total) if want.shape[0] == total else None
    if sized is None or not torch.equal(want, sized):
        _STATE["aside"] = f"host lengths sum to {total}, the device ones to {want.shape[0]}"
        selfcheck_fail(SYNCFREE_ENV, _STATE["aside"])
        print(f"CONV_SYNCFREE stands aside: the token groups sized from the host query lengths differ from "
              f"the stock ones ({_STATE['aside']}); every later step syncs as stock does", flush=True)
        return want
    _STATE["checked"] += 1
    _TALLY["checked"] += 1
    print(f"CONV_SYNCFREE check {_STATE['checked']} of {CHECKS}: {query_lens.shape[0]} requests, {total} tokens, "
          f"token groups equal to stock", flush=True)
    say("live", "CONV_SYNCFREE live: the PLE short-conv metadata of a mixed speculative step sizes its token "
                "groups from the host query lengths it already holds, so its repeat_interleave no longer waits "
                "for the GPU to drain")
    tally()
    return sized
