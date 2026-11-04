import os

ENV = "TAAF_VLLM_PREFILL_FILL"
REPORT_EVERY = 2000
_TALLY = {"filled": 0, "tokens": 0}


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() not in ("", "0", "false", "no", "off")


def take(start, end):
    _TALLY["filled"] += 1
    _TALLY["tokens"] += end - start
    if _TALLY["filled"] == 1:
        print("PREFILL_FILL live: a prefill chunk whose token budget cannot reach the next Mamba block boundary "
              "takes the budget as a sub-block chunk inside its block, and the next chunk realigns at that "
              "boundary", flush=True)
    if _TALLY["filled"] % REPORT_EVERY == 1:
        print(f"PREFILL_FILL tally: sub-block chunks={_TALLY['filled']} tokens={_TALLY['tokens']}", flush=True)
    return True
