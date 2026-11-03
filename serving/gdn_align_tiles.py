import os

ENV = "TAAF_VLLM_GDN_ALIGN_TILES"
LIMIT = 64
_STATE = {"said": set()}


def requested():
    said = str(os.environ.get(ENV) or "").strip()
    if not said:
        return 0
    try:
        tiles = int(said)
    except ValueError:
        tiles = -1
    if not 0 <= tiles <= LIMIT:
        raise ValueError(f"{ENV} must be a whole number of tiles from 0 to {LIMIT}, found {said!r}")
    return tiles


def tiles(stock):
    asked = requested()
    if not asked:
        return stock
    if asked not in _STATE["said"]:
        _STATE["said"].add(asked)
        print(f"{ENV} first align copy with {asked} temporal tile(s) in place of {stock}: the pre-copy and the "
              f"post-copy of the Mamba align path launch {asked} CTA(s) per request and state instead of {stock}",
              flush=True)
    return asked
