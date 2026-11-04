import atexit
import json
import math
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ENV = "TAAF_VLLM_DRAFT_CAPTURE"
DIR_ENV = "TAAF_VLLM_DRAFT_CAPTURE_DIR"
GIB_ENV = "TAAF_VLLM_DRAFT_CAPTURE_GIB"
LANES_ENV = "TAAF_VLLM_DRAFT_CAPTURE_LANES"
IDLE_ENV = "TAAF_VLLM_DRAFT_CAPTURE_IDLE_S"
DRAFTER_ENV = "TAAF_VLLM_DRAFT_CAPTURE_DRAFTER"
HARD_ENV = "TAAF_VLLM_DRAFT_CAPTURE_HARD"
ROTATE_ENV = "TAAF_VLLM_DRAFT_CAPTURE_ROTATE"
GAMES_ENV = "TAAF_VLLM_DRAFT_CAPTURE_GAMES"
SESSIONS = "sessions.jsonl"
GAME_KEY = re.compile(r"(?:^|[-_/])([a-z0-9]{4})-[0-9a-f]{8}(?=$|[-_/#])")
BRANCH_PREFIX = "taaf-branch"
DEFAULT_DIR = os.path.join(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working", "draft-capture")
DEFAULT_GIB = 10.0
DEFAULT_LANES = 6
DEFAULT_IDLE_S = 900.0
TOP_K = 20
TOP_P = 0.95
TEMPERATURE = 0.6
GROUP = 128
QUEUE_DEPTH = 4
MAX_ROWS = 4096
INDEX_EVERY_S = 60.0
SAY_EVERY_S = 300.0
SAY_EVERY_V2_S = 60.0
GONE_S = 30.0
BREAK_SAYS = 20
INDEX = "snapshot.json"
PARTS = ("ids", "q", "s", "tp", "ti")
DR_PART = "dr"
DR_TYPE = np.dtype([("id", "<i4"), ("conf", "<f2"), ("acc", "u1")])
NO_DRAFT = 255
WINDOW_ROWS = 8
CONTEXT_ROWS = 8

_STATE = {"capture": None}


def say(text):
    print(f"DRAFT_CAPTURE {text}", flush=True)


def enabled():
    return str(os.environ.get(ENV) or "").strip().lower() in ("1", "on", "true")


def settings(environ=None):
    environ = os.environ if environ is None else environ
    folder = str(environ.get(DIR_ENV) or "").strip() or DEFAULT_DIR
    gib = float(str(environ.get(GIB_ENV) or "").strip() or DEFAULT_GIB)
    lanes = int(str(environ.get(LANES_ENV) or "").strip() or DEFAULT_LANES)
    idle = float(str(environ.get(IDLE_ENV) or "").strip() or DEFAULT_IDLE_S)
    if not 0 < gib <= 64 or not 0 < lanes <= 64 or idle <= 0:
        raise ValueError(f"{GIB_ENV} in (0, 64], {LANES_ENV} in 1..64 and {IDLE_ENV} above 0, found {gib}, {lanes}, "
                         f"{idle}")
    drafter = str(environ.get(DRAFTER_ENV) or "").strip().lower() in ("1", "on", "true")
    hard_said = str(environ.get(HARD_ENV) or "").strip()
    hard = float(hard_said) if hard_said else None
    rotate_said = str(environ.get(ROTATE_ENV) or "").strip()
    rotate = int(rotate_said) if rotate_said else None
    if hard is not None and not 0 < hard <= 1 or rotate is not None and rotate <= 0:
        raise ValueError(f"{HARD_ENV} in (0, 1] and {ROTATE_ENV} above 0, found {hard}, {rotate}")
    if hard is not None and not drafter:
        raise ValueError(f"{HARD_ENV} judges the drafter's confidence and needs {DRAFTER_ENV}=1")
    games = str(environ.get(GAMES_ENV) or "").strip().lower() in ("1", "on", "true")
    return {"folder": Path(folder), "budget": int(gib * 2 ** 30), "lanes": lanes, "idle": idle, "drafter": drafter,
            "hard": hard, "rotate": rotate, "games": games}


def external(rid):
    head, _, tail = str(rid).rpartition("-")
    return head if head and len(tail) == 8 else str(rid)


def game_key(key):
    key = str(key or "")
    if not key or key.startswith(BRANCH_PREFIX) or "#" in key:
        return None
    found = GAME_KEY.search(key)
    return found.group(1) if found else None


def group_for(width, hc):
    return math.gcd(width // hc, GROUP)


def row_bytes(width, group):
    return width // 2 + 2 * (width // group) + 4 + 6 * TOP_K


def quantize4(multi, group):
    rows = multi.shape[0]
    view = multi.float().reshape(rows, -1, group)
    scale = (view.abs().amax(-1) / 7.0).clamp(min=1e-8)
    q = (view / scale[..., None]).round().clamp(-7, 7).to(torch.int16) + 8
    flat = q.reshape(rows, -1)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.reshape(rows, -1).to(torch.float16)


def game_distribution(logits):
    values, ids = logits.float().topk(TOP_K, dim=-1)
    probs = torch.softmax(values / TEMPERATURE, dim=-1)
    before = probs.cumsum(-1) - probs
    probs = torch.where(before < TOP_P, probs, torch.zeros_like(probs))
    return (probs / probs.sum(-1, keepdim=True)).to(torch.float16), ids.to(torch.int32)


class Lane:
    def __init__(self, rid, seg, now):
        self.rid = rid
        self.seg = seg
        self.next_pos = 0
        self.turn_start = None
        self.turns = 0
        self.seen = now
        self.broken = False
        self.done = False
        self.captured = 0
        self.rejected = None
        self.buf = None
        self.run = None
        self.windows = 0
        self.ctx = None


class Capture:
    def __init__(self, folder, budget, lanes, idle, hc, width, multi_of, logits_of, computed_of, device,
                 clock=time.monotonic, max_rows=MAX_ROWS, drafter=False, hard=None, rotate=None, conf_of=None,
                 games=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.max_lanes = lanes
        self.idle = idle
        self.hc = hc
        self.width = width
        self.group = group_for(width, hc)
        self.drafter = bool(drafter)
        self.hard = hard
        self.rotate = rotate
        self.conf_of = conf_of
        self.bytes_per_row = row_bytes(width, self.group) + (DR_TYPE.itemsize if self.drafter else 0)
        self.multi_of = multi_of
        self.logits_of = logits_of
        self.computed_of = computed_of
        self.device = device
        self.clock = clock
        self.lanes = {}
        self.all_lanes = {}
        self.refused = set()
        self.segments = []
        self.entries = []
        self.counts = {"steps": 0, "rows": 0, "bytes": 0, "queued_bytes": 0, "dropped_steps": 0, "dropped_rows": 0,
                       "broken": 0, "lanes_opened": 0}
        if self.hard is not None:
            self.counts.update({"windows_kept": 0, "windows_dropped": 0})
        if self.rotate is not None:
            self.counts["rotated"] = 0
        if self.drafter:
            self.counts["draft_rows"] = 0
        self.games = bool(games)
        self.keys = {}
        self.keys_read = 0
        if self.games:
            self.counts["not_games"] = 0
        self.lock = threading.RLock()
        self.stopped = False
        self.kept = None
        self.queue = queue.Queue()
        self.max_rows = max_rows
        self.copy_stream = torch.cuda.Stream(device) if torch.cuda.is_available() and device.type == "cuda" else None
        pin = self.copy_stream is not None
        self.free = queue.Queue()
        for _ in range(QUEUE_DEPTH):
            self.free.put({"multi": torch.empty((max_rows, width), dtype=torch.bfloat16, pin_memory=pin),
                           "ids": torch.empty(max_rows, dtype=torch.int32, pin_memory=pin),
                           "tp": torch.empty((max_rows, TOP_K), dtype=torch.float16, pin_memory=pin),
                           "ti": torch.empty((max_rows, TOP_K), dtype=torch.int32, pin_memory=pin),
                           "rej": torch.empty(64, dtype=torch.int32, pin_memory=pin),
                           "end": torch.empty(64, dtype=torch.int64, pin_memory=pin),
                           **({"dr": torch.empty(max_rows, dtype=torch.float16, pin_memory=pin)} if self.drafter
                              else {})})
        self.last_index = clock()
        self.last_say = clock()
        self.break_says = 0
        self.writer = threading.Thread(target=self.write_loop, name="taaf-draft-capture", daemon=True)
        self.writer.start()

    def keep(self, hidden, batch):
        self.kept = (hidden, batch)

    def open_lane(self, rid, now):
        seg = None if self.hard is not None else self.new_segment(rid)
        self.lanes[rid] = self.all_lanes[rid] = Lane(rid, seg, now)
        self.counts["lanes_opened"] += 1

    def session_key(self, rid):
        name = external(rid)
        if name not in self.keys:
            path = self.folder / SESSIONS
            if path.is_file():
                with open(path, "rb") as source:
                    source.seek(self.keys_read)
                    data = source.read()
                whole = data.rfind(b"\n") + 1
                self.keys_read += whole
                for line in data[:whole].splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    self.keys[str(row.get("rid"))] = row.get("key")
        return self.keys.get(name)

    def takes(self, rid):
        if not self.games or game_key(self.session_key(rid)) is not None:
            return True
        self.refused.add(rid)
        self.counts["not_games"] += 1
        return False

    def new_segment(self, rid, **more):
        with self.lock:
            self.segments.append({"rows": 0, "bits": 4, "rid": rid, **more})
            return len(self.segments) - 1

    def choose(self, batch, now):
        chosen = []
        computed = batch.num_computed_tokens_np
        gone = self.idle if self.rotate is None else min(self.idle, GONE_S)
        present = set(list(batch.req_ids)[:batch.num_reqs])
        if self.rotate is not None:
            for rid, lane in list(self.lanes.items()):
                if lane.done or lane.broken or rid not in present and now - lane.seen > gone:
                    self.close_lane(rid)
        for index, rid in enumerate(list(batch.req_ids)[:batch.num_reqs]):
            lane = self.lanes.get(rid)
            if lane is None and rid not in self.refused and not self.stopped and len(self.lanes) < self.max_lanes \
                    and int(computed[index]) == 0 and self.takes(rid):
                self.open_lane(rid, now)
                lane = self.lanes[rid]
            elif lane is None and int(computed[index]) != 0:
                self.refused.add(rid)
            if lane is not None and self.rotate is not None and (lane.done or lane.broken):
                self.close_lane(rid)
                continue
            if lane is not None and not lane.broken:
                lane.seen = now
                chosen.append(index)
        for rid, lane in list(self.lanes.items()):
            if now - lane.seen > gone:
                self.close_lane(rid)
        return chosen

    def note_break(self, why):
        if self.break_says < BREAK_SAYS:
            self.break_says += 1
            say(f"lanes ended: {why}; " + self.line())

    def tick(self, now):
        every = SAY_EVERY_S if self.rotate is None and self.hard is None and not self.drafter else SAY_EVERY_V2_S
        if now - self.last_say > every:
            self.last_say = now
            say(self.line())

    def close_lane(self, rid):
        if self.lanes.pop(rid, None) is not None:
            self.refused.add(rid)

    def end_turn(self, lane):
        if self.hard is not None:
            return self.close_run(lane)
        if lane.turn_start is not None and lane.next_pos > lane.turn_start:
            self.entries.append({"key": f"{lane.rid}#t{lane.turns}", "split": "train", "game": "", "run": "capture",
                                 "step": 0, "call": lane.turns, "seg": lane.seg, "count": lane.next_pos,
                                 "start": lane.turn_start, "rid": lane.rid})
            lane.turns += 1
        lane.turn_start = None

    def step(self, num_rejected):
        kept, self.kept = self.kept, None
        if kept is None or self.stopped:
            return None
        hidden, batch = kept
        now = self.clock()
        chosen = self.choose(batch, now)
        self.counts["steps"] += 1
        self.tick(now)
        if not chosen:
            return None
        qsl = batch.query_start_loc_np
        drafts = batch.num_draft_tokens_per_req
        spans = [(index, int(qsl[index]), int(qsl[index + 1])) for index in chosen]
        rows = sum(stop - start for _, start, stop in spans)
        cost = rows * self.bytes_per_row
        if self.counts["bytes"] + self.counts["queued_bytes"] + cost > self.budget:
            self.stop("the byte budget is spent")
            return None
        slot = None if rows > self.max_rows or len(spans) > 64 else self.take_slot()
        if slot is None:
            self.counts["dropped_steps"] += 1
            self.counts["dropped_rows"] += rows
            self.note_break(f"step of {rows} rows over {len(spans)} sessions dropped ("
                            + (f"above the {self.max_rows}-row slot" if rows > self.max_rows else "no free slot") + ")")
            for index, _, _ in spans:
                lane = self.lanes.get(batch.req_ids[index])
                if lane is not None:
                    lane.broken = True
                    self.counts["broken"] += 1
            return None
        order = np.concatenate([np.arange(start, stop) for _, start, stop in spans])
        verify = np.concatenate([np.full(stop - start, drafts is not None and int(drafts[index]) > 0, dtype=bool)
                                 for index, start, stop in spans])
        where = torch.from_numpy(order).to(self.device, non_blocking=True)
        picked = torch.tensor(chosen, device=self.device)
        gpu = {"multi": self.multi_of()[where], "ids": batch.input_ids[where].to(torch.int32),
               "rej": num_rejected[picked].to(torch.int32),
               "end": self.computed_of(batch.idx_mapping[picked]).to(torch.int64)}
        pending = self.conf_of() if self.drafter and self.conf_of is not None else None
        if pending is not None:
            across = np.concatenate([np.full(stop - start, at) for at, (_, start, stop) in enumerate(spans)])
            along = np.concatenate([np.arange(stop - start) for _, start, stop in spans])
            slots = batch.idx_mapping[picked][torch.from_numpy(across).to(self.device, non_blocking=True)]
            steps = torch.from_numpy(np.maximum(along - 1, 0)).to(self.device, non_blocking=True)
            gpu["dr"] = pending[slots.long(), steps.clamp(max=pending.shape[1] - 1)].to(torch.float16)
        verify_rows = np.nonzero(verify)[0]
        if len(verify_rows) and hidden is not None:
            picks = torch.from_numpy(verify_rows).to(self.device, non_blocking=True)
            gpu["tp"], gpu["ti"] = game_distribution(self.logits_of(hidden[where[picks]]))
        event = None
        if self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                for name, tensor in gpu.items():
                    slot[name][:tensor.shape[0]].copy_(tensor, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.copy_stream)
        else:
            for name, tensor in gpu.items():
                slot[name][:tensor.shape[0]].copy_(tensor)
        meta = {"spans": [(batch.req_ids[index], stop - start, bool(drafts is not None and int(drafts[index]) > 0))
                          for index, start, stop in spans],
                "rows": rows, "verify_rows": verify_rows, "with_dist": "tp" in gpu, "with_dr": "dr" in gpu}
        self.counts["queued_bytes"] += cost
        self.queue.put_nowait((event, gpu, slot, meta, cost))
        return rows

    def take_slot(self):
        try:
            return self.free.get_nowait()
        except queue.Empty:
            return None

    def stop(self, why):
        if self.stopped:
            return
        self.stopped = True
        for rid in list(self.lanes):
            self.close_lane(rid)
        say(f"stopped: {why}; " + self.line())
        self.queue.put(None)

    def write_loop(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.write_index()
                return
            event, gpu, slot, meta, cost = item
            if event is not None:
                event.synchronize()
            del gpu
            try:
                self.write_item(slot, meta)
            finally:
                self.free.put(slot)
                self.counts["queued_bytes"] -= cost
            now = self.clock()
            if now - self.last_index > INDEX_EVERY_S:
                self.write_index()
                self.last_index = now
            self.tick(now)

    def write_item(self, slot, meta):
        rows = meta["rows"]
        packed, scale = quantize4(slot["multi"][:rows], self.group)
        tp = np.zeros((rows, TOP_K), dtype=np.float16)
        ti = np.zeros((rows, TOP_K), dtype=np.int32)
        if meta["with_dist"]:
            count = len(meta["verify_rows"])
            tp[meta["verify_rows"]] = slot["tp"][:count].numpy()
            ti[meta["verify_rows"]] = slot["ti"][:count].numpy()
        arrays = {"ids": slot["ids"][:rows].numpy().copy(), "q": packed.numpy(), "s": scale.numpy(), "tp": tp,
                  "ti": ti, "rej": slot["rej"].numpy().copy(), "end": slot["end"].numpy().copy()}
        conf = slot["dr"][:rows].numpy().copy() if meta["with_dr"] else None
        offset = 0
        for at, (rid, length, verify) in enumerate(meta["spans"]):
            lane = self.all_lanes.get(rid)
            take = length - int(arrays["rej"][at]) if verify else length
            start = int(arrays["end"][at]) - take
            if lane is None or lane.broken or lane.done or take <= 0:
                offset += length
                continue
            with self.lock:
                if start != lane.next_pos:
                    lane.broken = True
                    self.counts["broken"] += 1
                    self.note_break(f"session {rid} jumped to position {start} from {lane.next_pos}")
                    self.end_turn(lane)
                    offset += length
                    continue
                if self.hard is None:
                    if verify and lane.turn_start is None:
                        lane.turn_start = start
                    if not verify and lane.turn_start is not None:
                        self.end_turn(lane)
                part = slice(offset, offset + take)
                block = {name: np.ascontiguousarray(arrays[name][part]) for name in PARTS}
                if self.drafter:
                    whole = slice(offset, offset + length)
                    block[DR_PART] = self.draft_rows(lane, arrays["ids"][whole], None if conf is None
                                                     else conf[whole], take, length, verify)
                lane.next_pos = start + take
                if self.hard is None:
                    self.append(lane.seg, block, lane)
                    self.segments[lane.seg]["rows"] = lane.next_pos
                else:
                    self.feed_hard(lane, block)
                if self.rotate is not None and lane.captured >= self.rotate and not lane.done:
                    lane.done = True
                    self.end_turn(lane)
                    self.counts["rotated"] += 1
            offset += length

    def draft_rows(self, lane, ids, conf, take, length, verify):
        out = np.zeros(take, dtype=DR_TYPE)
        out["id"] = -1
        out["acc"] = NO_DRAFT
        known = np.full(length, -1.0, dtype=np.float16) if conf is None else conf
        if lane.rejected is not None:
            out[0] = (lane.rejected[0], lane.rejected[1], 0)
            lane.rejected = None
            self.counts["draft_rows"] += 1
        if verify:
            if take > 1:
                out["id"][1:] = ids[1:take]
                out["conf"][1:] = known[1:take]
                out["acc"][1:] = 1
                self.counts["draft_rows"] += take - 1
            if take < length:
                lane.rejected = (int(ids[take]), known[take])
        return out

    def append(self, seg, block, lane):
        rows = len(block["ids"])
        for name, array in block.items():
            with open(self.folder / f"seg{seg:05d}.{name}", "ab") as sink:
                sink.write(array.tobytes())
        lane.captured += rows
        if self.hard is not None:
            self.segments[seg]["rows"] += rows
        self.counts["rows"] += rows
        self.counts["bytes"] += rows * self.bytes_per_row

    def feed_hard(self, lane, block):
        lane.buf = block if lane.buf is None else {name: np.concatenate([lane.buf[name], block[name]])
                                                   for name in block}
        while len(lane.buf["ids"]) >= WINDOW_ROWS:
            window = {name: array[:WINDOW_ROWS] for name, array in lane.buf.items()}
            lane.buf = {name: array[WINDOW_ROWS:] for name, array in lane.buf.items()}
            self.settle(lane, window)
            lane.windows += 1

    def settle(self, lane, window):
        draft = window[DR_PART]
        seen = (draft["acc"] != NO_DRAFT) & (draft["conf"] >= 0)
        if not bool(np.any(seen & (draft["conf"] < np.float16(self.hard)))):
            self.close_run(lane)
            lane.ctx = window
            self.counts["windows_dropped"] += 1
            return
        if lane.run is None:
            before = lane.ctx
            first = (lane.windows - (1 if before is not None else 0)) * WINDOW_ROWS
            seg = self.new_segment(lane.rid, pos0=first)
            lane.run = {"seg": seg, "ctx": 0, "n": lane.turns}
            lane.turns += 1
            if before is not None:
                self.append(seg, before, lane)
                lane.run["ctx"] = len(before["ids"])
        lane.ctx = None
        self.append(lane.run["seg"], window, lane)
        self.counts["windows_kept"] += 1

    def run_entry(self, lane, held=False):
        run = lane.run
        entry = {"key": f"{lane.rid}#w{run['n']}", "split": "train", "game": "", "run": "capture", "step": 0,
                 "call": run["n"], "seg": run["seg"], "count": self.segments[run["seg"]]["rows"],
                 "start": run["ctx"], "rid": lane.rid}
        if held:
            entry["open"] = True
        return entry

    def close_run(self, lane):
        if lane.run is not None:
            self.entries.append(self.run_entry(lane))
            lane.run = None

    def index(self):
        with self.lock:
            entries = list(self.entries)
            for lane in self.all_lanes.values():
                if lane.run is not None:
                    entries.append(self.run_entry(lane, True))
                if lane.turn_start is not None and lane.next_pos > lane.turn_start:
                    entries.append({"key": f"{lane.rid}#t{lane.turns}", "split": "train", "game": "",
                                    "run": "capture", "step": 0, "call": lane.turns, "seg": lane.seg,
                                    "count": lane.next_pos, "start": lane.turn_start, "rid": lane.rid, "open": True})
            return {"hc": self.hc, "width": self.width, "topk": TOP_K, "segments": [dict(one) for one in self.segments],
                    "entries": entries, "bytes": self.counts["bytes"], "temperature": TEMPERATURE, "top_k": TOP_K,
                    "top_p": TOP_P, "source": "decode-capture", "counts": dict(self.counts)}

    def write_index(self):
        held = self.index()
        temporary = self.folder / f".{INDEX}.tmp"
        temporary.write_text(json.dumps(held), encoding="utf-8")
        os.replace(temporary, self.folder / INDEX)

    def line(self):
        held = self.counts
        return (f"lanes {len(self.lanes)} open, {held['lanes_opened']} opened; rows {held['rows']}; written "
                f"{held['bytes'] / 2 ** 30:.2f} of {self.budget / 2 ** 30:.1f} GiB; entries {len(self.entries)}; "
                f"dropped {held['dropped_steps']} steps ({held['dropped_rows']} rows); broken {held['broken']}; "
                f"steps seen {held['steps']}" + self.extra())

    def extra(self):
        held = self.counts
        said = ""
        if self.drafter:
            said += f"; draft rows {held['draft_rows']}"
        if self.hard is not None:
            said += f"; windows kept {held['windows_kept']} dropped {held['windows_dropped']} (conf < {self.hard})"
        if self.rotate is not None:
            said += f"; rotated {held['rotated']} (after {self.rotate} rows)"
        if self.games:
            said += f"; {held['not_games']} requests passed over as not a game session"
        return said

    def flush(self):
        self.stop("the server is exiting")
        self.writer.join(timeout=60)


def conf_source(device):
    def find():
        module = sys.modules.get("vllm.taaf_draft_conf_log")
        for state in list(getattr(module, "_states", {}).values()):
            if state["device"] == device:
                return state["pending_prob"]
        return None
    return find if find() is not None else None


def extras(held, capture):
    said = ""
    if capture.drafter:
        said += " (with the drafter's proposal, confidence and verdict per row)"
    if capture.hard is not None:
        said += f", only {WINDOW_ROWS}-row windows with a draft below confidence {capture.hard} and {WINDOW_ROWS} context rows"
    if capture.rotate is not None:
        said += f", a session released after {capture.rotate} rows"
    if capture.games:
        said += f", only game sessions (a session key naming a game id, no branch; read from {SESSIONS})"
    return said


def install(runner):
    if not enabled() or _STATE["capture"] is not None:
        return None
    model = getattr(runner, "model", None)
    multi_of = getattr(model, "get_mtp_target_hidden_states", None)
    logits_of = getattr(model, "compute_logits", None)
    sample_tokens = getattr(runner, "sample_tokens", None)
    postprocess = getattr(runner, "postprocess_sampled", None)
    if multi_of is None or multi_of() is None or logits_of is None or sample_tokens is None or postprocess is None:
        say("not armed: the runner lacks the MTP hidden buffer, compute_logits, sample_tokens or postprocess_sampled")
        return None
    held = settings()
    buffer = multi_of()
    text = getattr(model.config, "text_config", None) or getattr(model, "config", None)
    hc = int(getattr(text, "hc_count", 0) or getattr(model.config, "hc_count", 0) or 1)
    batched = int(getattr(getattr(runner, "scheduler_config", None), "max_num_batched_tokens", 0) or MAX_ROWS)
    capture = Capture(held["folder"], held["budget"], held["lanes"], held["idle"], hc, int(buffer.shape[-1]),
                      multi_of, logits_of, lambda index: runner.req_states.num_computed_tokens.gpu[index],
                      buffer.device, max_rows=max(256, min(batched, MAX_ROWS)), drafter=held["drafter"],
                      hard=held["hard"], rotate=held["rotate"], conf_of=conf_source(buffer.device),
                      games=held["games"])
    if held["drafter"] and capture.conf_of is None:
        capture.drafter = False
        capture.hard = None
        capture.bytes_per_row -= DR_TYPE.itemsize
        say(f"{DRAFTER_ENV} ignored: the draft confidence log is not armed on {buffer.device}")

    def wrapped_sample_tokens(*args, **kwargs):
        state = getattr(runner, "execute_model_state", None)
        if state is not None:
            capture.keep(getattr(state, "hidden_states", None), getattr(state, "input_batch", None))
        return sample_tokens(*args, **kwargs)

    def wrapped_postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs):
        out = postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs)
        try:
            capture.step(num_rejected)
        except Exception as failure:
            capture.stop(f"{type(failure).__name__}: {failure}")
        return out

    runner.sample_tokens = wrapped_sample_tokens
    runner.postprocess_sampled = wrapped_postprocess
    atexit.register(capture.flush)
    _STATE["capture"] = capture
    say(f"armed: up to {held['lanes']} sessions from their first token, every committed row's target multi-stream "
        f"hidden state (the MTP head's input, {buffer.shape[-1]} wide, int4 g{capture.group}) with its token, and on "
        f"verify rows the target's top-{TOP_K} distribution at T{TEMPERATURE} top-p {TOP_P}; {capture.bytes_per_row} "
        f"bytes a row{extras(held, capture)}, at most {held['budget'] / 2 ** 30:.1f} GiB ({held['budget'] // capture.bytes_per_row} rows) to "
        f"{held['folder']}; the rows go to {QUEUE_DEPTH} preallocated pinned slots of {capture.max_rows} rows by a side "
        f"stream and a background thread quantizes and writes them; a step with no free slot is dropped and its "
        f"sessions end")
    return capture
