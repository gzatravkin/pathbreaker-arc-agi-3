import json
import os
import re
import threading
from array import array

PHEASANT = 600.0
BIG_PROMPT = 3000
WALL_ENV = "ARC3_LIVE_STATS_WALL_TOKENS"
WALL_DEFAULT = 40000
WIDTH = 300
ANGLE = re.compile(r"[<>]{2,}")
GARBAGE_CODE = re.compile(r"(?:\bif\s+){3,}|(?::=\s*){3,}|\bfor\s+for\b|[<>]{2,}|</?[A-Za-z]+>")
FORMAT_ERRORS = ("empty_code", "unknown_tool", "syntax_error")
SIGNATURE = ("angle_text", "garbage_code")
GEN = "vllm:generation_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
READ = (GEN, DRAFTS, ACCEPTED, RUNNING, WAITING)
STEP_GROUP = re.compile(r"STEP_TIMERS \d+s (decode|mixed)/\S+ n=(\d+) .*? step ([\d.]+)/.*?\| lanes [\d.]+ \(\d+-\d+\) "
                        r"dec ([\d.]+)")
STEP_SPLIT = re.compile(r"STEP_TIMERS \d+s split: decode (\d+) steps ([\d.]+) ms, mixed (\d+) steps ([\d.]+) ms, "
                        r"prefill (\d+) steps ([\d.]+) ms")
STEP_KINDS = ("decode", "mixed", "prefill")
NO_RUN = "TRIM_HOLE no_run"
FRESH_MARK = "PAUSED cause fresh after_miss_no_run"
FRESH_NO_RUN = re.compile(r"PAUSED cause fresh after_miss_no_run: key \S+, (\d+) prompt tokens")
TALLY = "PAUSED tally {"
TALLY_REUSED = re.compile(r"'reused': (\d+)")
TALLY_FED = re.compile(r"'fed': (\d+)")
SPEC_LINE = re.compile(r"SpecDecoding metrics: .*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
                       r"Per-position acceptance rate: ([\d., ]+)")


def call_codes(tool_calls):
    codes = []
    for call in tool_calls or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            parsed = None
        codes.append(parsed.get("code") if isinstance(parsed, dict) else None)
    return codes


def text_events(reasoning, content):
    return ["angle_text"] if ANGLE.search((reasoning or "") + "\n" + (content or "")) else []


def result_events(result, codes):
    events = []
    if result.lstrip().startswith("{") and '"error"' in result[:30]:
        if "non-empty" in result:
            events.append("empty_code")
        elif "Unknown tool" in result:
            events.append("unknown_tool")
        elif "syntax error" in result:
            events.append("syntax_error")
            if any(code and GARBAGE_CODE.search(code) for code in codes):
                events.append("garbage_code")
    return events


def plain(value):
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def reported_tokens(analyzer):
    value = (getattr(analyzer, "generated_tokens", None) if hasattr(analyzer, "generated_tokens")
             else getattr(analyzer, "total_tokens", 0))
    return _int(value)


def game_label(session):
    key = str(getattr(session, "_rotation_key", "") or "")
    if key.startswith("arc3-rot-"):
        name = key[len("arc3-rot-"):]
    else:
        run = getattr(getattr(session, "game", None), "game_run", None)
        name = (f"{getattr(run, 'game_id', None) or getattr(session, 'game_index', '?')}"
                f"/p{getattr(session, 'pass_index', '?')}")
    head, _, tail = name.partition("/")
    return head.split("-")[0] + (f"/{tail}" if tail else "")


def arm_of(session):
    key = str(getattr(session, "_rotation_key", "") or "")
    name = key[len("arc3-rot-"):] if key.startswith("arc3-rot-") else ""
    head, _, tail = name.partition("/")
    return arm_of_stem(f"{head}_{tail}" if head and tail else "")


def arm_of_stem(stem):
    every = os.environ.get("ARC3_KNOB_ALL", "")
    try:
        plan = json.loads(os.environ.get("ARC3_KNOB_PLAN", "") or "{}")
    except ValueError:
        plan = {}
    if not isinstance(plan, dict):
        plan = {}
    return str(plan.get(stem, every) if stem else every) or "-"


def paired(rows):
    per = []
    for on, off in rows.values():
        if on and off:
            per.append(sum(on) / len(on) - sum(off) / len(off))
    return (sum(per) / len(per), len(per)) if per else (None, 0)


SPLIT = ("tool", "image", "user_text", "system", "recompute", "rewrite", "other")
SPLIT_NAMES = {"tool": "tool results", "image": "images", "user_text": "user text", "system": "system",
               "recompute": "lost cache", "rewrite": "rewritten history", "other": "template"}


def usage_numbers(usage):
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    details = usage.get("prompt_tokens_details") or {}
    media = (details.get("multimodal_tokens") or {}) if isinstance(details, dict) else {}
    return (_int(usage.get("prompt_tokens")), _int(details.get("cached_tokens")) if isinstance(details, dict) else 0,
            _int(usage.get("completion_tokens")), _int(media.get("image")) if isinstance(media, dict) else 0)


def message_parts(message):
    content = message.get("content")
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for part in content or []:
        if isinstance(part, dict):
            out.append(("image", "") if part.get("type") == "image_url" else ("text", str(part.get("text") or "")))
    return out


def split_chars(messages, with_assistant):
    held, images = {}, 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if with_assistant:
                text = str(message.get("reasoning") or message.get("reasoning_content") or "")
                text += "".join(piece for kind, piece in message_parts(message) if kind == "text")
                held["other"] = held.get("other", 0) + len(text)
            continue
        for kind, piece in message_parts(message):
            if kind == "image":
                images += 1
            else:
                name = "system" if role == "system" else "tool" if role == "tool" else "user_text"
                held[name] = held.get(name, 0) + len(piece)
    return held, images


def prefill_split(previous, messages, usage):
    prompt, cached, made, image_tokens = usage_numbers(usage)
    computed = max(0, prompt - cached)
    out = dict.fromkeys(SPLIT, 0.0)
    boundary = prompt + made
    if previous is None:
        fresh, new = messages, computed
    else:
        old, old_boundary = previous
        same = 0
        for a, b in zip(old, messages):
            if a != b:
                break
            same += 1
        if same < len(old):
            out["rewrite"] = float(computed)
            return out, boundary, computed, made
        fresh = messages[same:]
        new = min(computed, max(0, prompt - old_boundary))
        out["recompute"] = float(computed - new)
    held, images = split_chars(fresh, previous is None)
    every = sum(1 for message in messages for kind, _ in message_parts(message) if kind == "image")
    image_new = min(new, round(image_tokens / every * images)) if every else 0
    out["image"] = float(image_new)
    rest = new - image_new
    chars = sum(held.values())
    if chars:
        for name, count in held.items():
            out[name] += rest * count / chars
    else:
        out["other"] += rest
    return out, boundary, computed, made


def levels_of(session):
    return _int(getattr(session.game.current_state, "levels_completed", 0))


def run_complete(session):
    state = session.game.current_state
    raw = getattr(getattr(state, "raw", None), "state", None)
    return str(getattr(raw, "name", raw)) == "WIN" or getattr(state, "won", False) is True


def key_name(display):
    return str(display or "").split("(")[0].strip() or "?"


def tokens_text(value):
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


def share(part, whole):
    if not whole:
        return "n/a"
    value = 100.0 * part / whole
    return f"{value:.2f}%" if 0 < value < 1 else f"{value:.1f}%"


def quantile(held, q):
    return held[min(len(held) - 1, int(round(q * (len(held) - 1))))]


def p50_p90(values):
    if not len(values):
        return "n/a"
    held = sorted(values)
    return f"{tokens_text(quantile(held, 0.5))}/{tokens_text(quantile(held, 0.9))}"


def fit(head, items, width=WIDTH):
    for count in range(len(items), -1, -1):
        rest = len(items) - count
        text = head + ", ".join(items[:count]) + ((", " if count else "") + f"+{rest} more" if rest else "")
        if len(text) <= width or count == 0:
            return text
    return head


def number(value, spec):
    return "n/a" if value is None else format(value, spec)


def wrap(line, head, width=WIDTH):
    out = []
    while len(line) > width:
        cut = line.rfind(" | ", len(head), width)
        if cut < 0:
            break
        out.append(line[:cut])
        line = head + line[cut + 3:]
    return out + [line]


class Window:
    def __init__(self):
        self.n = {}
        self.prompts = array("q")
        self.replies = array("q")
        self.won = []
        self.lost = []
        self.stretch = (0, "")

    def add(self, name, value=1):
        self.n[name] = self.n.get(name, 0) + value

    def get(self, name):
        return self.n.get(name, 0)


class Game:
    def __init__(self, session, label, base):
        self.session = session
        self.label = label
        self.base = base
        self.level_start = 0
        self.last_press = 0
        self.index = 0
        self.last_change = 0
        self.keys = {}
        self.final = None
        self.arm = arm_of(session)
        self.level_presses = 0
        self.levels = 0

    def tokens(self):
        if self.final is not None:
            return self.final
        now = reported_tokens(getattr(self.session, "analyzer", None))
        if now < self.base:
            self.base = now
        return now - self.base

    def dropped(self):
        return sorted(key for key, (presses, noops, last) in self.keys.items()
                      if 1 <= presses <= 2 and noops == presses and self.last_change > last)


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.minute = Window()
        self.total = Window()
        self.games = {}
        self.seen = set()
        self.first = None
        self.previous = None
        self.lanes = [0, 0.0, 0.0]
        self.tally = {"raw": None, "last": None, "mark": (0, 0), "offset": (0, 0)}
        self.log_seen = False
        self.split_seen = False
        self.trim_seen = False
        self.steps_seen = False
        self.spec_log_seen = False
        self.dropped_mark = 0
        self.arms = {}
        self.arm_prefill = {}
        self.renders = 0
        self.errors = {}

    def books(self):
        return self.minute, self.total

    def request(self, requester, usage, finish=""):
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details")
        known = isinstance(details, dict) and details.get("cached_tokens") is not None
        prompt, made = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
        cached = _int(details.get("cached_tokens")) if known else 0
        with self.lock:
            first = requester not in self.seen
            self.seen.add(requester)
            for book in self.books():
                book.add("requests")
                book.add("prompt", prompt)
                book.prompts.append(prompt)
                book.replies.append(made)
                if known:
                    book.add("known")
                    book.add("known_prompt", prompt)
                    book.add("cached", cached)
                    if not first and cached == 0 and prompt >= BIG_PROMPT:
                        book.add("reprefill")
                        book.add("reprefill_tokens", prompt)

    def mark(self, token, event):
        held = token["events"]
        if event in held:
            return
        fresh_format = event in FORMAT_ERRORS and not any(one in held for one in FORMAT_ERRORS)
        fresh_stray = event in SIGNATURE and not any(one in held for one in SIGNATURE)
        flagged = not held
        held.add(event)
        for book in self.books():
            book.add(event)
            if fresh_format:
                book.add("format")
            if fresh_stray:
                book.add("stray")
            if flagged:
                book.add("flagged")

    def reply(self, agent, result, reasoning, content, tool_calls, malformed=0, messages=None):
        finish = str(getattr(result, "finish_reason", "") or "").strip()
        self.request(id(agent), getattr(result, "usage", None), finish)
        if isinstance(messages, list):
            split, boundary, computed, made = prefill_split(getattr(agent, "_live_prefill", None), messages,
                                                            getattr(result, "usage", None))
            agent._live_prefill = (list(messages), boundary)
            said = getattr(agent, "_burn_said", None)
            arm = arm_of_stem(str(said[0]) if said and said[0] else "")
            with self.lock:
                held = self.arm_prefill.setdefault(arm, {})
                held["computed"] = held.get("computed", 0) + computed
                held["made"] = held.get("made", 0) + made
                for name, value in split.items():
                    held[name] = held.get(name, 0) + value
                for book in self.books():
                    book.add("pf_computed", computed)
                    book.add("pf_made", made)
                    for name, value in split.items():
                        book.add("pf_" + name, value)
        token = {"events": set(), "codes": call_codes(tool_calls)}
        events = text_events(plain(reasoning), plain(content))
        with self.lock:
            for book in self.books():
                book.add("replies")
                if finish == "length":
                    book.add("length")
                if malformed:
                    book.add("bad_json")
            for event in events:
                self.mark(token, event)
        agent._live_reply = token
        return token

    def tool_result(self, agent, index, content):
        token = getattr(agent, "_live_reply", None)
        if index or token is None:
            return
        agent._live_reply = None
        events = result_events(plain(content), token["codes"])
        with self.lock:
            for event in events:
                self.mark(token, event)

    def game(self, session):
        found = self.games.get(id(session))
        if found is None:
            found = Game(session, game_label(session), reported_tokens(getattr(session, "analyzer", None)))
            self.games[id(session)] = found
            self.arms.setdefault(found.arm, {"won": [], "acts": [], "first": 0})
        return found

    def begin(self, session):
        with self.lock:
            self.games.pop(id(session), None)
            self.game(session)

    def press(self, session, display, changed, before, after, automatic=False):
        with self.lock:
            game = self.game(session)
            if automatic or game.final is not None:
                return
            tokens = game.tokens()
            stretch = tokens - game.last_press
            game.last_press = tokens
            game.index += 1
            game.level_presses += 1
            held = game.keys.setdefault(key_name(display), [0, 0, 0])
            held[0] += 1
            held[1] += 1 if changed is False else 0
            held[2] = game.index
            if changed is True:
                game.last_change = game.index
            for book in self.books():
                book.add("presses")
                if changed is False:
                    book.add("noops")
                if stretch > book.stretch[0]:
                    book.stretch = (stretch, game.label)
            if _int(after) > _int(before):
                for book in self.books():
                    book.won.append(tokens - game.level_start)
                arm = self.arms[game.arm]
                arm["won"].append(tokens - game.level_start)
                arm["acts"].append(game.level_presses)
                arm["first"] += 1 if _int(before) == 0 else 0
                game.levels += 1
                game.level_start = tokens
                game.level_presses = 0

    def end(self, session):
        complete = run_complete(session)
        with self.lock:
            game = self.games.get(id(session))
            if game is None or game.final is not None:
                return
            game.final = game.tokens()
            stretch = game.final - game.last_press
            for book in self.books():
                if stretch > book.stretch[0]:
                    book.stretch = (stretch, game.label)
                if not complete:
                    book.lost.append(game.final - game.level_start)

    def serve_lines(self, lines):
        with self.lock:
            self.log_seen = True
            for line in lines:
                if TALLY in line:
                    self.tally_line(line)
                elif "STEP_TIMERS" in line:
                    self.step_line(line)
                elif NO_RUN in line:
                    self.trim_seen = True
                    for book in self.books():
                        book.add("no_run")
                elif FRESH_MARK in line:
                    found = FRESH_NO_RUN.search(line)
                    for book in self.books() if found else ():
                        book.add("fresh_no_run")
                        book.add("fresh_no_run_tokens", int(found.group(1)))
                elif "SpecDecoding metrics" in line:
                    self.spec_line(line)

    def step_line(self, line):
        found = STEP_SPLIT.search(line)
        if found:
            self.split_seen = self.steps_seen = True
            for book in self.books():
                for at, kind in enumerate(STEP_KINDS):
                    book.add(f"ms_{kind}", float(found.group(2 * at + 2)))
            return
        found = None if self.split_seen else STEP_GROUP.search(line)
        if found is None:
            return
        self.steps_seen = True
        kind = "prefill" if found.group(1) == "mixed" and float(found.group(4)) == 0.0 else found.group(1)
        for book in self.books():
            book.add(f"ms_{kind}", int(found.group(2)) * float(found.group(3)))

    def tally_line(self, line):
        if "'trim_" in line:
            self.trim_seen = True
        reused, fed = TALLY_REUSED.search(line), TALLY_FED.search(line)
        if reused is None or fed is None:
            return
        now, tally = (int(reused.group(1)), int(fed.group(1))), self.tally
        if tally["raw"] is not None and (now[0] < tally["raw"][0] or now[1] < tally["raw"][1]):
            tally["offset"] = tally["last"]
        tally["raw"] = now
        tally["last"] = (now[0] + tally["offset"][0], now[1] + tally["offset"][1])

    def spec_line(self, line):
        found = SPEC_LINE.search(line)
        positions = [one for one in (found.group(3) if found else "").replace(" ", "").split(",") if one]
        if not positions:
            return
        self.spec_log_seen = True
        for book in self.books():
            book.add("spec_accepted", int(found.group(1)))
            book.add("spec_drafts", int(found.group(2)) / len(positions))

    def take_reading(self, reading, read_at):
        if not isinstance(reading, dict) or read_at is None:
            return None, None
        sample = (float(read_at), {name: reading.get(name) for name in READ})
        previous = self.previous
        if previous is not None and sample[1][GEN] is not None and previous[1][GEN] is not None \
                and sample[1][GEN] < previous[1][GEN]:
            self.first = previous = None
        if self.first is None:
            self.first = sample
        self.previous = sample
        if sample[1][RUNNING] is not None:
            self.lanes = [self.lanes[0] + 1, self.lanes[1] + sample[1][RUNNING],
                          self.lanes[2] + (sample[1][WAITING] or 0.0)]
        return sample, previous

    @staticmethod
    def rate(now, then, name):
        if now is None or then is None or now[0] <= then[0] or now[1][name] is None or then[1][name] is None:
            return None
        return (now[1][name] - then[1][name]) / (now[0] - then[0])

    @staticmethod
    def accept(now, then):
        if now is None or then is None:
            return None
        values = [side[1][name] for side in (now, then) for name in (DRAFTS, ACCEPTED)]
        if any(value is None for value in values) or values[0] - values[2] <= 0:
            return None
        return 1.0 + (values[1] - values[3]) / (values[0] - values[2])

    def spec_text(self, sample, previous):
        minute, total = self.accept(sample, previous), self.accept(sample, self.first)
        source = ""
        if minute is None and total is None and self.spec_log_seen:
            minute, total = [1.0 + book.get("spec_accepted") / book.get("spec_drafts") if book.get("spec_drafts")
                             else None for book in self.books()]
            source = ", serve log"
        if minute is None and total is None:
            return "spec accept len n/a"
        return f"spec accept len {number(minute, '.2f')} (all {number(total, '.2f')}{source})"

    def steps_text(self):
        if not self.log_seen:
            return "step time n/a (no serve log tail)"
        if not self.steps_seen:
            return "step time n/a (no STEP_TIMERS lines)"
        parts = []
        for book in self.books():
            spent = [book.get(f"ms_{kind}") for kind in STEP_KINDS]
            parts.append("n/a" if not sum(spent) else "/".join(f"{100.0 * one / sum(spent):.0f}" for one in spent))
        return (f"step time % decode/mixed/prefill {parts[0]} (all {parts[1]})"
                + ("" if self.split_seen else ", prefill = mixed groups with no decode lane"))

    def serve_line(self, head, sample, previous):
        gen, gen_all = self.rate(sample, previous, GEN), self.rate(sample, self.first, GEN)
        speed = "gen n/a" if gen is None and gen_all is None else (
            f"gen {number(gen, '.0f')} tok/s = {number(None if gen is None else gen / PHEASANT, '.2f')} фазана (all "
            f"{number(gen_all, '.0f')} = {number(None if gen_all is None else gen_all / PHEASANT, '.2f')})")
        if sample is not None and sample[1][RUNNING] is not None and self.lanes[0]:
            lanes = (f"lanes run {sample[1][RUNNING]:.0f} wait {sample[1][WAITING] or 0:.0f} (all mean "
                     f"{self.lanes[1] / self.lanes[0]:.1f}/{self.lanes[2] / self.lanes[0]:.1f})")
        else:
            lanes = "lanes n/a"
        return f"{head}serve: {speed} | {lanes} | {self.spec_text(sample, previous)} | {self.steps_text()}"

    def prefill_line(self, head):
        minute, total = self.books()
        if not total.get("requests"):
            return f"{head}prefill: n/a (no client request yet)"
        if total.get("known"):
            cache = (f"cached {share(minute.get('cached'), minute.get('known_prompt'))} (all "
                     f"{share(total.get('cached'), total.get('known_prompt'))} of {tokens_text(total.get('known_prompt'))}"
                     f"), computed {tokens_text(minute.get('known_prompt') - minute.get('cached'))} (all "
                     f"{tokens_text(total.get('known_prompt') - total.get('cached'))})")
        else:
            cache = "cached n/a (no cached_tokens in usage)"
        return (f"{head}prefill: {minute.get('requests')} requests (all {total.get('requests')}), prompt "
                f"{tokens_text(minute.get('prompt'))}, {cache} | prompt p50/p90 {p50_p90(minute.prompts)} (all "
                f"{p50_p90(total.prompts)}) | reply p50/p90 {p50_p90(minute.replies)} (all {p50_p90(total.replies)})")

    def split_line(self, head):
        minute, total = self.books()
        if not total.get("pf_computed"):
            return f"{head}prefill split: n/a (no player request with its messages yet)"

        def ratio(book):
            return f"{book.get('pf_computed') / max(1, book.get('pf_made')):.2f}"

        parts = [f"{SPLIT_NAMES[name]} {share(total.get('pf_' + name), total.get('pf_computed'))}" for name in SPLIT
                 if total.get("pf_" + name) >= 0.005 * total.get("pf_computed")]
        return (f"{head}prefill split: computed per generated token {ratio(minute)} (all {ratio(total)}, "
                f"{tokens_text(total.get('pf_computed'))} computed) | all: " + ", ".join(parts))

    def reprefill_line(self, head):
        minute, total = self.books()
        client = (f"later requests with cached 0 and prompt >= {tokens_text(BIG_PROMPT)} {minute.get('reprefill')} = "
                  f"{tokens_text(minute.get('reprefill_tokens'))} (all {total.get('reprefill')} = "
                  f"{tokens_text(total.get('reprefill_tokens'))})" if total.get("known")
                  else "cached 0 n/a (no cached_tokens in usage)")
        if not self.log_seen:
            return f"{head}re-prefill: {client} | TRIM_HOLE n/a, PAUSED tally n/a (no serve log tail)"
        trim = (f"TRIM_HOLE no_run {minute.get('no_run')} (all {total.get('no_run')}), fresh after no_run "
                f"{minute.get('fresh_no_run')} = {tokens_text(minute.get('fresh_no_run_tokens'))} (all "
                f"{total.get('fresh_no_run')} = {tokens_text(total.get('fresh_no_run_tokens'))})"
                if self.trim_seen else "TRIM_HOLE n/a (no TRIM_HOLE lines)")
        tally = self.tally
        if tally["last"] is None:
            paused = "PAUSED tally n/a"
        else:
            mark, last = tally["mark"], tally["last"]
            paused = (f"PAUSED fed/reused {tokens_text(last[1] - mark[1])}/{tokens_text(last[0] - mark[0])} (all "
                      f"{tokens_text(last[1])}/{tokens_text(last[0])})")
            tally["mark"] = last
        return f"{head}re-prefill: {client} | {trim} | {paused}"

    def replies_line(self, head):
        minute, total = self.books()
        if not total.get("replies"):
            return f"{head}replies: n/a (no player reply yet)"

        def pair(name):
            return (f"{minute.get(name)} = {share(minute.get(name), minute.get('replies'))} (all {total.get(name)} = "
                    f"{share(total.get(name), total.get('replies'))})")

        return (f"{head}replies: {minute.get('replies')} (all {total.get('replies')}) | malformed tool calls "
                f"{pair('format')} | stray angle tokens {pair('stray')} | length cuts {pair('length')} | any scan "
                f"event {pair('flagged')} | bad JSON args dropped {minute.get('bad_json')} (all {total.get('bad_json')})")

    def levels_lines(self, head, wall):
        minute, total = self.books()
        if not self.games:
            return [f"{head}levels: n/a (no game started)"]
        open_games = [(game.tokens() - game.level_start, game.label) for game in self.games.values()
                      if game.final is None]
        heavy = sorted((one for one in open_games if one[0] >= wall), reverse=True)
        text = (f"{head}levels: won {len(minute.won)} at p50/p90 {p50_p90(minute.won)} tokens (all {len(total.won)} at "
                f"{p50_p90(total.won)}) | not won: {len(open_games)} open at {p50_p90([one[0] for one in open_games])}"
                f", {len(total.lost)} ended at {p50_p90(total.lost)}")
        wall_text = f"{head}open >= {tokens_text(wall)} tokens without a win: {len(heavy)}"
        if not heavy:
            return [text, wall_text]
        return [text, fit(wall_text + ": ", [f"{label} {tokens_text(tokens)}" for tokens, label in heavy])]

    def arms_lines(self, head, wall):
        if len(self.arms) < 2:
            return []
        parts = []
        for letters in sorted(self.arms):
            arm = self.arms[letters]
            games = [game for game in self.games.values() if game.arm == letters]
            heavy = sum(1 for game in games if game.final is None and game.tokens() - game.level_start >= wall)
            acts = sorted(arm["acts"])
            parts.append(f"{letters} {len(games)} copies: won {len(arm['won'])} ({len(arm['won']) / max(1, len(games)):.2f} "
                         f"a copy, level 1 {arm['first']}), tokens/level p50/p90 {p50_p90(arm['won'])}, actions/level "
                         f"p50 {acts[len(acts) // 2] if acts else 'n/a'}, open >= {tokens_text(wall)} {heavy}")
        out = [f"{head}arms: " + " | ".join(parts)]
        prefill = []
        for letters in sorted(self.arm_prefill):
            held = self.arm_prefill[letters]
            if not held.get("computed"):
                continue
            top = ", ".join(f"{SPLIT_NAMES[name]} {share(held.get(name, 0), held['computed'])}" for name in
                            ("tool", "image", "user_text", "recompute") if held.get(name, 0) >= 0.005 * held["computed"])
            prefill.append(f"{letters} computed {tokens_text(held['computed'])}, "
                           f"{held['computed'] / max(1, held.get('made', 0)):.2f} per generated token ({top})")
        if len(prefill) >= 2:
            out.append(f"{head}arm prefill: " + " | ".join(prefill))
        names = sorted({letter for letters in self.arms for letter in letters if letter != "-"})
        varying = [letter for letter in names if not all(letter in letters for letters in self.arms)]
        effects = []
        for letter in varying:
            levels, tokens = {}, {}
            for game in self.games.values():
                base = game.label.split("/")[0]
                side = 0 if letter in game.arm else 1
                levels.setdefault(base, ([], []))[side].append(game.levels)
                tokens.setdefault(base, ([], []))[side].append(game.tokens())
            delta, count = paired(levels)
            spent, _ = paired(tokens)
            if delta is not None:
                effects.append(f"{letter} on-off levels a copy {delta:+.2f}, tokens a copy {spent / 1000:+.1f}k over {count} "
                               f"games paired")
        if effects:
            out.append(f"{head}arm letters: " + " | ".join(effects))
        return out

    def presses_lines(self, head):
        minute, total = self.books()
        if not total.get("presses"):
            return [f"{head}presses: n/a (no press yet)"]
        open_stretch = max(((game.tokens() - game.last_press, game.label) for game in self.games.values()
                            if game.final is None), default=(0, ""))
        longest_minute, longest_total = max(minute.stretch, open_stretch), max(total.stretch, open_stretch)
        dropped = [(game.label, game.dropped()) for game in sorted(self.games.values(), key=lambda one: one.label)]
        dropped = [(label, keys) for label, keys in dropped if keys]
        count = sum(len(keys) for _, keys in dropped)
        fresh, self.dropped_mark = count - self.dropped_mark, count
        text = (f"{head}presses: {minute.get('presses')}, NO-OP {share(minute.get('noops'), minute.get('presses'))} "
                f"(all {total.get('presses')}, {share(total.get('noops'), total.get('presses'))}) | longest stretch "
                f"without a press {tokens_text(longest_minute[0])} tokens {longest_minute[1] or '-'} (all "
                f"{tokens_text(longest_total[0])} {longest_total[1] or '-'})")
        keys = (f"{head}keys pressed 1-2 times, NO-OP each time, not pressed again after a later change: {count} in "
                f"{len(dropped)} games ({fresh:+d} this minute)")
        if not dropped:
            return [text, keys]
        return [text, fit(keys + ": ", [f"{label} {','.join(names)}" for label, names in dropped])]

    def lines(self, minutes, reading=None, read_at=None, final=False):
        with self.lock:
            sample, previous = self.take_reading(reading, read_at)
            self.renders += 1
            if self.renders == 1 and not final:
                self.minute = Window()
                return []
            head = f"STATS [{minutes:.1f} min{', final' if final else ''}] "
            wall = _int(os.environ.get(WALL_ENV) or WALL_DEFAULT)
            out = [self.serve_line(head, sample, previous), self.prefill_line(head), self.split_line(head),
                   self.reprefill_line(head),
                   self.replies_line(head), *self.levels_lines(head, wall), *self.arms_lines(head, wall),
                   *self.presses_lines(head)]
            if self.errors:
                out.append(f"{head}hook errors: " + ", ".join(f"{name} {count}" for name, count in
                                                              sorted(self.errors.items())))
            self.minute = Window()
        return [piece for line in out for piece in wrap(line, head)]


_LIVE = []
_LOCK = threading.Lock()
ARMED = []
SAID = set()


def live():
    if not _LIVE:
        with _LOCK:
            if not _LIVE:
                _LIVE.append(Live())
    return _LIVE[0]


def arm(say=print):
    if ARMED:
        return False
    ARMED.append(True)
    live()
    say(f"STATS armed: after each levels line a block of STATS lines, each value for the last minute (all = since "
        f"play start); speed from /metrics, prefill and replies from every client request, levels and presses from "
        f"every press, step time and TRIM_HOLE from the serve log tail; 1 фазан = {PHEASANT:.0f} tok/s", flush=True)
    return True


def call(name, *args):
    if not ARMED:
        return None
    held = live()
    try:
        return getattr(held, name)(*args)
    except Exception as exc:
        with _LOCK:
            held.errors[name] = held.errors.get(name, 0) + 1
            first = name not in SAID
            SAID.add(name)
        if first:
            print(f"STATS hook {name} raised {type(exc).__name__}: {str(exc)[:200]} (counted; said once)", flush=True)
        return None


def lines(minutes, reading=None, read_at=None, final=False):
    if not ARMED:
        return []
    return live().lines(minutes, reading, read_at, final)


def serve_lines(read):
    return call("serve_lines", list(read))
