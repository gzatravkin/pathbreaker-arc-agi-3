import contextlib
import os

import torch

ENV = "TAAF_VLLM_GDN_COMMIT_STREAM"
_STATE = {"stream": None, "event": None, "open": None, "forked": False, "held": (), "said": False, "lane": None}


def mode():
    said = str(os.environ.get(ENV) or "").strip().lower()
    if said in ("", "0", "off", "false", "no"):
        return 0
    if said in ("1", "true", "yes", "on"):
        return 1
    if said in ("2", "3"):
        return int(said)
    raise ValueError(f"{ENV} must be 0, 1, 2 or 3, found {said!r}")


def on():
    return mode() > 0


def _held_of(owner, tensors):
    held = list(tensors)
    held.extend(vars(owner.recoverssm).values())
    return tuple(held)


def begin(owner, *tensors):
    if not on() or getattr(owner, "recoverssm", None) is None:
        return None
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return None
    main = torch.cuda.current_stream()
    if _STATE["stream"] is None:
        _STATE["stream"] = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[0])
        _STATE["event"] = torch.cuda.Event()
    side = _STATE["stream"]
    side.wait_stream(main)
    context = torch.cuda.stream(side)
    context.__enter__()
    return (context, _held_of(owner, tensors))


def end(token):
    if token is None:
        return
    context, held = token
    side = _STATE["stream"]
    context.__exit__(None, None, None)
    for item in held:
        if isinstance(item, torch.Tensor) and item.is_cuda:
            item.record_stream(side)
    _STATE["event"].record(side)
    _STATE["held"] = held
    _STATE["forked"] = True
    if not _STATE["said"]:
        _STATE["said"] = True
        beside = {1: "on the main stream beside them", 2: "on a higher-priority stream beside them",
                  3: "on the main stream beside them, the commit's grid capped"}[mode()]
        print(f"{ENV} first forked commit: the GDN state commit and the align postprocess run on a side stream "
              f"of priority {side.priority} (the main stream's is {torch.cuda.current_stream().priority}, "
              f"the device's range {torch.cuda.Stream.priority_range()}) after the sampler (mode {mode()}: "
              f"the drafter runs {beside}), the next execute_model waits on their event",
              flush=True)


@contextlib.contextmanager
def draft_lane(capture=False):
    if (mode() != 2 or not (capture or _STATE["forked"]) or not torch.cuda.is_available()
            or torch.cuda.is_current_stream_capturing()):
        yield
        return
    if _STATE["lane"] is None:
        _STATE["lane"] = torch.cuda.Stream(priority=-1)
    main = torch.cuda.current_stream()
    lane = _STATE["lane"]
    lane.wait_stream(main)
    with torch.cuda.stream(lane):
        yield
    main.wait_stream(lane)


def join():
    if not _STATE["forked"]:
        return
    torch.cuda.current_stream().wait_event(_STATE["event"])
    _STATE["forked"] = False
    _STATE["held"] = ()
