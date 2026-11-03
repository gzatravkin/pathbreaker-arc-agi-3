import os

SOFT_ENV = "ARC3_NOOP_SOFT"
SOFT_VERDICT = ("didn't change the board (the action itself may still work at a different position, in a different "
                "state or on a different level - don't discard it)")
SOFT_SHORT = "didn't change the board here (it may still work at a different position or level - don't discard it)"
SHORT_ENV = "ARC3_NOOP_SHORT"
SHORT_VERDICT = "no change here"


def soft() -> bool:
    return os.environ.get(SOFT_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def short() -> bool:
    return os.environ.get(SHORT_ENV, "0").strip().lower() in ("1", "true", "yes", "on")

def announce(letter: str, text: str) -> None:
    try:
        from inference.agent import knob_arms
        owner = getattr(knob_arms._local, "owner", None)
        stem = knob_arms.stem_of(owner) if owner is not None else ""
        knob_arms.say_once(f"{letter} {stem}", f"KNOB {letter} first used by {stem or 'a game'}: {text}")
    except Exception as exc:
        print(f"KNOB {letter} announce failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
