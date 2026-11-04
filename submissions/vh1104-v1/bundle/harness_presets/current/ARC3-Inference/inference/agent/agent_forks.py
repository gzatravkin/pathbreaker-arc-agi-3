import ast
import json
import os
import threading

REFLECT_KNOB = "ARC3_FORK_REFLECT"
PYTHON_KNOB = "ARC3_FORK_PYTHON"
KINDS = ("reflect", "python")
INLINE_KNOB = "ARC3_REFLECT_INLINE"
SHOWN_CHARS = 500
INLINE_LEAD = "Note:"
INLINE_EVERY_KNOB = "ARC3_REFLECT_EVERY"
INLINE_EVERY = 4
INLINE_TOKENS = 1024
SESSION_HEADER = "X-Session-ID"
BRANCH_PREFIX = "taaf-branch"
FIRST_TURN = 2
PYTHON_EVERY = 2
PYTHON_ROUNDS = 4
MAX_TOKENS = 4096
LIST_LIMIT = 40
LINE_CHARS = 200
NOTE_PREFIX = "Note from yourself: "
REFLECT_REQUEST = (
    "Review what you are doing. Maybe you should change your approach, or you made a mistake somewhere. "
    "What would you tell yourself? Write the message as if you wanted to tell yourself something important."
)
REFLECT_ROUNDS = 3
REFLECT_STOP = (
    "Stop. Do not call python again and do not take any game action. Answer me: is everything you are doing "
    "right? Maybe you got stuck on something and should have tested another hypothesis, or something you are "
    "doing contains a mistake you did not notice? Write the message as if you wanted to tell yourself something "
    "important."
)
INLINE_REQUEST = (
    "Write me a message that someone else could send you to point out one contradiction between what you "
    "concluded and what the frames actually showed: an observation your current idea does not explain, a step "
    "you counted wrong, or an effect you claimed that did not happen. Name the frame or the action. Do not "
    "propose a new theory of the game. Reply with that message only, as one paragraph without blank lines."
)
PYTHON_REQUEST = (
    "Write the single most important and useful Python function that you will most likely need in this game. "
    "You can see the existing helper functions below; you may also fix or improve one of them, call them and "
    "test them."
)
NO_HELPERS = "(no helper functions yet)"
HELPERS_HEAD = "Helper functions available in this game:"


def switched_on(kind: str) -> bool:
    name = REFLECT_KNOB if kind == "reflect" else PYTHON_KNOB
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def inline_on() -> bool:
    return os.environ.get(INLINE_KNOB, "").strip().lower() in {"1", "true", "yes", "on"}


def inline_every() -> int:
    try:
        return max(1, int(os.environ.get(INLINE_EVERY_KNOB, "") or INLINE_EVERY))
    except ValueError:
        return INLINE_EVERY


def any_on() -> bool:
    return any(switched_on(kind) for kind in KINDS) or inline_on()


def describe(source: str) -> str:
    try:
        node = ast.parse(source).body[-1]
        head = node.name + "(" + ast.unparse(node.args) + ")"
        doc = ast.get_docstring(node)
        if doc and doc.strip():
            line = doc.strip().splitlines()[0].strip()
        else:
            segment = ast.get_source_segment(source, node.body[0]) or ""
            line = segment.strip().splitlines()[0].strip() if segment.strip() else ""
    except Exception:
        lines = [one.strip() for one in source.splitlines() if one.strip()]
        return (lines[0] if lines else "")[:LINE_CHARS]
    return (head + (": " + line if line else ""))[:LINE_CHARS]


def parse_calls(message: dict):
    calls = json.loads(json.dumps(message.get("tool_calls") or []))
    parsed = []
    for call in calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        raw = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (ValueError, TypeError):
            return calls, None
        if not isinstance(arguments, dict):
            return calls, None
        parsed.append((str(call.get("id", "") or ""), str(function.get("name", "") or "").strip(), arguments))
    return calls, parsed


def helpers_block(functions: dict) -> str:
    if not functions:
        return NO_HELPERS
    return "```python\n" + "\n\n".join(functions.values()) + "\n```"


def python_request(functions: dict) -> str:
    return PYTHON_REQUEST + "\n\n" + helpers_block(functions)


class Forks:
    def __init__(self, agent):
        self.agent = agent
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.turn = 0
        self.step = None
        self.launched = 0
        self.flying = {kind: None for kind in KINDS}
        self.ready = {kind: None for kind in KINDS}
        self.handed = {kind: None for kind in KINDS}
        self.said_no_functions = False
        self.inlined = 0
        self.inline_tally = {"asked": 0, "shown": 0, "empty": 0, "failed": 0, "prompt_tokens": 0,
                             "completion_tokens": 0}
        self.tally = {kind: {"launched": 0, "finished": 0, "failed": 0, "shown": 0, "requests": 0,
                             "prompt_tokens": 0, "completion_tokens": 0} for kind in KINDS}

    def label(self) -> str:
        return str(getattr(self.agent, "_diag_name", "?"))

    def begin_turn(self, step, resuming: bool) -> list:
        if not any_on():
            return []
        if step is None:
            if not resuming:
                self.turn += 1
        elif step != self.step:
            self.turn += 1
            self.step = step
        messages = []
        try:
            with self.lock:
                self.handed = {kind: None for kind in KINDS}
                taken = {kind: self.ready[kind] for kind in KINDS}
                for kind in KINDS:
                    if taken[kind] is not None:
                        self.ready[kind] = None
                        self.handed[kind] = taken[kind]
            if taken["reflect"] is not None:
                messages.append({"role": "user", "content": NOTE_PREFIX + taken["reflect"]["text"]})
                self.tally["reflect"]["shown"] += 1
                print(f"fork {self.label()} reflect: note shown at main turn {self.turn}, "
                      f"{len(taken['reflect']['text'])} chars", flush=True)
            if taken["python"] is not None:
                message = self.helpers_message(taken["python"])
                if message is not None:
                    messages.append(message)
                    self.tally["python"]["shown"] += 1
        except Exception as exc:
            print(f"fork {self.label()}: handing results to the main turn failed ({exc!r}); ignored", flush=True)
        return messages

    def restore(self) -> None:
        with self.lock:
            for kind in KINDS:
                if self.handed[kind] is not None and self.ready[kind] is None:
                    self.ready[kind] = self.handed[kind]
                self.handed[kind] = None

    def helpers_message(self, result: dict):
        agent = self.agent
        current = dict(agent._kept_functions)
        applied, held_back = [], []
        for name in result["added"] + result["changed"]:
            if current.get(name) in (result["before"].get(name), result["kept"][name]):
                current[name] = result["kept"][name]
                applied.append(name)
            else:
                held_back.append(name)
        agent._kept_functions = current
        added = [name for name in applied if name in result["added"]]
        changed = [name for name in applied if name in result["changed"]]
        if not added and not changed:
            print(f"fork {self.label()} python: nothing new to show"
                  + (f"; kept your own version of {', '.join(held_back)}" if held_back else ""), flush=True)
            return None
        lines = [HELPERS_HEAD]
        for name in sorted(current)[:LIST_LIMIT]:
            lines.append("- " + describe(current[name]))
        if len(current) > LIST_LIMIT:
            lines.append(f"- ... and {len(current) - LIST_LIMIT} more")
        if added:
            lines.append("Added by a background helper-writing pass: " + ", ".join(added) + ".")
        if changed:
            lines.append("Changed by it: " + ", ".join(changed) + ".")
        if held_back:
            lines.append("Not replaced, because you redefined them meanwhile: " + ", ".join(held_back) + ".")
        print(f"fork {self.label()} python: helper list shown, added {added}, changed {changed}", flush=True)
        return {"role": "user", "content": "\n".join(lines)}

    def launch(self, state_path, messages, tools, valid_actions) -> None:
        try:
            if self.stopped.is_set() or not any_on() or self.turn < FIRST_TURN or self.launched == self.turn:
                return
            self.launched = self.turn
            if switched_on("reflect"):
                self.start("reflect", state_path, messages, tools, valid_actions,
                           dict(getattr(self.agent, "_kept_functions", None) or {}))
            if switched_on("python") and (self.turn - FIRST_TURN) % PYTHON_EVERY == 0:
                if not self.agent._fork_functions_on():
                    if not self.said_no_functions:
                        self.said_no_functions = True
                        print(f"fork {self.label()} python: persistent functions are off, so there is no helper "
                              f"namespace to write into; the fork does not run", flush=True)
                else:
                    self.start("python", state_path, messages, tools, valid_actions, dict(self.agent._kept_functions))
        except Exception as exc:
            print(f"fork {self.label()}: launching failed ({exc!r}); ignored", flush=True)

    def start(self, kind, state_path, messages, tools, valid_actions, functions) -> None:
        with self.lock:
            if self.flying[kind] is not None:
                return
            thread = threading.Thread(target=self.run, name=f"fork-{kind}-{self.label()}", daemon=True,
                                      args=(kind, state_path, list(messages), tools, list(valid_actions or []), functions))
            self.flying[kind] = thread
            self.tally[kind]["launched"] += 1
        print(f"fork {self.label()} {kind}: launched at main turn {self.turn} on {len(messages)} messages", flush=True)
        thread.start()

    def run(self, kind, state_path, messages, tools, valid_actions, functions) -> None:
        result = None
        try:
            if kind == "reflect":
                result = self.reflect(state_path, messages, tools, valid_actions, dict(functions or {}))
            else:
                result = self.write_python(state_path, messages, tools, valid_actions, functions)
            self.tally[kind]["finished"] += 1
        except Exception as exc:
            self.tally[kind]["failed"] += 1
            print(f"fork {self.label()} {kind}: failed ({exc!r}); ignored", flush=True)
        finally:
            with self.lock:
                self.flying[kind] = None
                if result is not None and not self.stopped.is_set():
                    self.ready[kind] = result

    def inline(self, state_path, messages, tools):
        if not inline_on() or self.turn < inline_every() or self.turn % inline_every() or self.inlined == self.turn:
            return None
        self.inlined = self.turn
        self.inline_tally["asked"] += 1
        ask = [*messages, {"role": "user", "content": INLINE_REQUEST}]
        try:
            self.log(state_path, "inline", "request", ask, tools, None, 1)
            reply = self.agent._fork_complete("inline", ask, tools, max_tokens=INLINE_TOKENS)
            self.log(state_path, "inline", "response", ask, tools, reply, 1)
        except Exception as exc:
            self.inline_tally["failed"] += 1
            print(f"reflect {self.label()} inline: failed ({exc!r}); the turn goes on without it", flush=True)
            return None
        usage = reply.usage if isinstance(reply.usage, dict) else {}
        self.inline_tally["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.inline_tally["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        calls, parsed = parse_calls(reply.message)
        text = self.agent._fork_text(reply.message)
        if parsed or not text:
            self.inline_tally["empty"] += 1
            print(f"reflect {self.label()} inline: no message at main turn {self.turn}", flush=True)
            return None
        if not text.startswith(INLINE_LEAD):
            text = f"{INLINE_LEAD} {text}"
        self.inline_tally["shown"] += 1
        shown = " ".join(text.split())
        shown = shown if len(shown) <= SHOWN_CHARS else shown[:SHOWN_CHARS] + " ..."
        print(f"reflect {self.label()} inline at main turn {self.turn} ({len(text)} chars): {shown}", flush=True)
        return {"role": "user", "content": text}

    def call(self, kind, state_path, messages, tools, index):
        agent = self.agent
        self.log(state_path, kind, "request", messages, tools, None, index)
        reply = agent._fork_complete(kind, messages, tools)
        self.count(kind, reply.usage)
        self.log(state_path, kind, "response", messages, tools, reply, index)
        return reply

    def log(self, state_path, kind, event, messages, tools, reply, index) -> None:
        try:
            self.agent._fork_log(state_path, kind, event, messages, tools, reply, index)
        except Exception as exc:
            print(f"fork {self.label()} {kind}: the request log write failed ({exc!r}); ignored", flush=True)

    def count(self, kind, usage) -> None:
        usage = usage if isinstance(usage, dict) else {}
        row = self.tally[kind]
        row["requests"] += 1
        for field, keys in (("prompt_tokens", ("prompt_tokens", "input_tokens")),
                            ("completion_tokens", ("completion_tokens", "output_tokens", "generated_tokens"))):
            for key in keys:
                try:
                    row[field] += max(0, int(usage[key]))
                    break
                except (KeyError, TypeError, ValueError):
                    continue

    def reflect(self, state_path, messages, tools=None, valid_actions=None, functions=None):
        agent = self.agent
        convo = [*messages, {"role": "user", "content": REFLECT_REQUEST}]
        kept = dict(functions or {})
        index = 0
        for index in range(1, REFLECT_ROUNDS + 1):
            if self.stopped.is_set():
                return None
            reply = self.call("reflect", state_path, convo, tools, index)
            calls, parsed = parse_calls(reply.message)
            if not parsed:
                convo.append({"role": "assistant", "content": agent._fork_text(reply.message)})
                break
            convo.append(agent._fork_assistant(reply.message, calls))
            for call_id, name, arguments in parsed:
                if self.stopped.is_set():
                    return None
                if name != "python":
                    text = json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
                else:
                    text, kept = agent._fork_run(state_path, str(arguments.get("code", "")), kept,
                                                 list(valid_actions or []))
                convo.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if self.stopped.is_set():
            return None
        convo.append({"role": "user", "content": REFLECT_STOP})
        reply = self.call("reflect", state_path, convo, tools, index + 1)
        text = self.agent._fork_text(reply.message)
        if not text:
            print(f"fork {self.label()} reflect: the reply had no final text "
                  f"(finish {getattr(reply, 'finish_reason', '')}); no note", flush=True)
            return None
        return {"text": text}

    def write_python(self, state_path, messages, tools, valid_actions, functions):
        agent = self.agent
        before = dict(functions)
        kept = dict(before)
        convo = [*messages, {"role": "user", "content": python_request(before)}]
        for index in range(1, PYTHON_ROUNDS + 1):
            if self.stopped.is_set():
                return None
            reply = self.call("python", state_path, convo, tools, index)
            calls, parsed = parse_calls(reply.message)
            if not parsed:
                break
            convo.append(agent._fork_assistant(reply.message, calls))
            for call_id, name, arguments in parsed:
                if self.stopped.is_set():
                    return None
                if name != "python":
                    text = json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
                else:
                    text, kept = agent._fork_run(state_path, str(arguments.get("code", "")), kept, valid_actions)
                convo.append({"role": "tool", "tool_call_id": call_id, "content": text})
        added = sorted(name for name in kept if name not in before)
        changed = sorted(name for name in kept if name in before and kept[name] != before[name])
        return {"before": before, "kept": kept, "added": added, "changed": changed}

    def end(self) -> str:
        self.stopped.set()
        with self.lock:
            for kind in KINDS:
                self.ready[kind] = None
        if not any_on():
            return ""
        parts = []
        for kind in KINDS:
            if not switched_on(kind):
                continue
            row = self.tally[kind]
            parts.append(f"{kind}:launched={row['launched']},finished={row['finished']},failed={row['failed']},"
                         f"shown={row['shown']},requests={row['requests']},prompt={row['prompt_tokens']},"
                         f"completion={row['completion_tokens']}")
        if inline_on():
            row = self.inline_tally
            parts.append(f"inline:asked={row['asked']},shown={row['shown']},empty={row['empty']},failed={row['failed']},"
                         f"prompt={row['prompt_tokens']},completion={row['completion_tokens']}")
        note = " forks " + " ".join(parts)
        print(f"fork {self.label()}: game over;{note}", flush=True)
        return note
