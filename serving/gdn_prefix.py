import os
import time

PREFIX_ENV = "TAAF_VLLM_GDN_SPEC_PREFIX"
FIELD = "taaf_spec_prefix"
TALLY_S = 15.0
_TALLY = {"at": 0.0, "marked": 0, "scattered": 0, "sliced": 0}
_SAID = set()


def on():
    return str(os.environ.get(PREFIX_ENV) or "").strip().lower() in ("1", "on", "true")


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
    counts = {name: _TALLY[name] for name in ("marked", "scattered", "sliced")}
    print(f"GDN_SPEC_PREFIX tally {counts}", flush=True)


def prefix_rows(mask_cpu):
    if mask_cpu is None or mask_cpu.dim() != 1:
        return None
    rows = int(mask_cpu.sum())
    if rows == 0 or not bool(mask_cpu[:rows].all()):
        return None
    return rows


def mark(spec_token_indx, mask_cpu, query_start_loc_cpu, spec_tokens):
    if spec_token_indx is None or not on():
        return
    rows = prefix_rows(mask_cpu)
    tokens = int(spec_tokens)
    if rows is None or query_start_loc_cpu.shape[0] <= rows or int(query_start_loc_cpu[rows]) != tokens:
        _TALLY["scattered"] += 1
        tally()
        return
    setattr(spec_token_indx, FIELD, tokens)
    _TALLY["marked"] += 1
    tally()


def prefix(spec_token_indx):
    if spec_token_indx is None:
        return None
    tokens = getattr(spec_token_indx, FIELD, None)
    if tokens is None:
        return None
    _TALLY["sliced"] += 1
    say("live", "GDN_SPEC_PREFIX live: in a mixed step whose speculative rows lead the batch, each GDN layer "
                "slices its speculative and prefill tokens and writes both outputs in place, with no "
                "index_select or index_copy")
    return tokens
