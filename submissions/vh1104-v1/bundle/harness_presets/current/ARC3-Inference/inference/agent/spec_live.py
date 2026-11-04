import os
import queue
import sys
import threading
import time

ENV = "ARC3_SPEC_LIVE"
EVERY_ENV = "ARC3_SPEC_LIVE_S"
OFF = ("0", "off", "false", "no")
LABELS = ("player", "forum")
TOP = 5


class Tally:
    def __init__(self):
        self.hist = []
        self.turns = 0

    def add(self, hist):
        if len(self.hist) < len(hist):
            self.hist += [0] * (len(hist) - len(self.hist))
        for index, count in enumerate(hist):
            self.hist[index] += int(count)
        self.turns += 1

    def steps(self):
        return sum(self.hist)

    def conditional(self):
        reach = [sum(self.hist[at:]) for at in range(len(self.hist))]
        return [round(reach[at] / reach[at - 1], 3) if reach[at - 1] else None for at in range(1, len(reach))]

    def mean_len(self):
        steps = self.steps()
        return round(1 + sum(at * count for at, count in enumerate(self.hist)) / steps, 3) if steps else None


HOME_STDOUT = sys.stdout
LINES = queue.Queue(maxsize=max(1, int(os.environ.get("ARC3_SPEC_LIVE_QUEUE") or 1000)))
DROPPED = [0]
WRITER = []
WRITER_LOCK = threading.Lock()


def _writer():
    while True:
        text = LINES.get()
        try:
            stream = HOME_STDOUT or sys.__stdout__
            stream.write(text + chr(10))
            stream.flush()
        except Exception:
            DROPPED[0] += 1


def say_raw(text):
    if not WRITER:
        with WRITER_LOCK:
            if not WRITER:
                WRITER.append(threading.Thread(target=_writer, name="spec-live-log", daemon=True))
                WRITER[0].start()
    try:
        LINES.put_nowait(text)
    except queue.Full:
        DROPPED[0] += 1


class Live:
    def __init__(self, every=None, say=None, clock=time.monotonic):
        self.every = float(os.environ.get(EVERY_ENV) or 120) if every is None else every
        self.say = say or say_raw
        self.clock = clock
        self.started = clock()
        self.last = self.started
        self.lock = threading.Lock()
        self.total = {label: Tally() for label in LABELS}
        self.window = {label: Tally() for label in LABELS}
        self.games = {}

    def observe(self, label, game, spec):
        hist = (spec or {}).get("acceptance_histogram") if isinstance(spec, dict) else None
        if not hist:
            return None
        label = label if label in LABELS else "player"
        with self.lock:
            for book in (self.total, self.window):
                book[label].add(hist)
            if label == "player":
                self.games.setdefault(str(game), Tally()).add(hist)
            now = self.clock()
            if now - self.last < self.every:
                return None
            line = self.line(now)
            self.last = now
            self.window = {one: Tally() for one in LABELS}
        self.say(line)
        return line

    def part(self, label):
        window, total = self.window[label], self.total[label]
        return (f"{label} pos1..{len(total.hist) - 1} = {window.conditional()} mean_len {window.mean_len()} "
                f"{window.turns} turns (all {total.conditional()} {total.mean_len()} {total.turns})")

    def line(self, now):
        games = sorted(self.games.items(), key=lambda item: -item[1].turns)[:TOP]
        shown = ", ".join(f"{game} {tally.conditional()[0] if tally.conditional() else None} ({tally.turns})"
                          for game, tally in games)
        minutes = (now - self.started) / 60
        return f"SPEC_LIVE [{minutes:.0f} min]: {self.part('player')} | {self.part('forum')} | top games: {shown}"


_LIVE = None
_LOCK = threading.Lock()


def observe(label, game, spec):
    global _LIVE
    if str(os.environ.get(ENV) or "1").strip().lower() in OFF:
        return None
    with _LOCK:
        if _LIVE is None:
            _LIVE = Live()
    return _LIVE.observe(label, game, spec)
