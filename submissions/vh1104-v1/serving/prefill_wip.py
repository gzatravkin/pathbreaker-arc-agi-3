import os

ENV = "TAAF_VLLM_PREFILL_WIP"
_REPORT_EVERY = 2000


def parse(said):
    parts = str(said or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{ENV}={said!r} is not 'requests:tokens'")
    requests, tokens = (int(part) for part in parts)
    if requests < 1 or tokens < 1:
        raise ValueError(f"{ENV}={said!r} needs at least one request and one token")
    return requests, tokens


class Limit:
    def __init__(self, requests, tokens):
        self.requests = requests
        self.tokens = tokens
        self.steps = 0
        self.clamped = 0
        self.refused = 0
        self.begin()

    def begin(self):
        self.used_requests = 0
        self.used_tokens = 0
        self.steps += 1
        if self.steps % _REPORT_EVERY == 0:
            print(f"PREFILL_WIP tally: steps={self.steps} clamped={self.clamped} refused={self.refused} "
                  f"requests={self.requests} tokens={self.tokens}", flush=True)

    def grant(self, wanted):
        room = self.tokens - self.used_tokens
        if self.used_requests >= self.requests or room <= 0:
            self.refused += 1
            return 0
        allowed = min(wanted, room)
        if allowed < wanted:
            self.clamped += 1
        self.used_requests += 1
        self.used_tokens += allowed
        return allowed


LIMIT = Limit(*parse(os.environ[ENV])) if str(os.environ.get(ENV) or "").strip() not in ("", "0") else None


def begin():
    if LIMIT is not None:
        LIMIT.begin()


def grant(wanted):
    return wanted if LIMIT is None else LIMIT.grant(wanted)


def step_tokens():
    return 1 << 30 if LIMIT is None else LIMIT.tokens
