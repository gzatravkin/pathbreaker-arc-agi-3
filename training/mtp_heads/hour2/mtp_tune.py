from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKER = HERE / "mtp_tune_worker.py"
AB_WORKER = HERE / "draft_ab_worker.py"
DFLASH_WORKER = HERE / "dflash_collect_worker.py"
NEEDED = ("mtp_ref.py", "mtp_tune_worker.py")
AB_NEEDED = (*NEEDED, "draft_ab_worker.py")
SNAP_NEEDED = (*AB_NEEDED, "mtp_snapshot.py")
DFLASH_NEEDED = (*NEEDED, "mtp_snapshot.py", "dflash_collect_worker.py", "dflash_tune.py", "capture_snapshot.py",
                 "dflash_eval.py")


def needed(mode: str, mtp: bool = False) -> tuple[str, ...]:
    if mode == "dflash":
        return (*DFLASH_NEEDED, "draft_ab_worker.py") if mtp else DFLASH_NEEDED
    return SNAP_NEEDED if mode == "snap" else AB_NEEDED if mode == "ab" else NEEDED


def companions(env: dict[str, str]) -> list[str]:
    return list(needed(str(env.get("ARC_MTP_TUNE_MODE") or "").strip(),
                       str(env.get("ARC_MTP_TUNE_DFLASH_MTP") or "").strip().lower() in ("1", "true", "yes", "on")))


IDENTITY = "vllm-server-identity.json"
PREPARED = "vllm-server-prepared.json"
HARD_KNOB = "ARC_CARRIED_HARD_MINUTES"
SALIENCY = "expert_saliency.json"
CORPUS_MARK = "MTP_CORPUS.json"
PROC = Path("/proc")
WARM_PIECE = 64 << 20
FOLLOW_AHEAD = 12
FOLLOW_THREADS = 6
FOLLOW_POLL_S = 0.5
DEFAULTS: dict[str, Any] = {
    "mode": "eval", "train_minutes": 60.0, "train_records": 0, "eval_records": 150, "steps": 3,
    "eval_steps": 5, "step_weights": [1.0, 0.7, 0.5, 0.4, 0.3], "lr": 2e-5, "warmup": 20, "accum": 4,
    "clip": 1.0, "context": 32768, "attn_chunk": 128, "train_bases": 3072, "eval_bases": 4096,
    "prefill_chunk": 16384, "kv_gib": 12.0, "share_index": False, "train_experts": False, "seed": 7,
    "log_every": 10, "init": "", "skip_records": 0, "target": "game", "kl": 0.0, "temperature": 0.6,
    "eval_temperatures": [], "grad_context": 0, "span": "reply", "late": 0.0, "vocab": "", "tuned": "",
    "ab_heads": "albucino,intel,tuned", "ab_vocabs": "draft_vocab.json,draft_vocab_franzen.json",
    "ab_tune_minutes": 0.0, "ab_tune_from": "best", "ab_drafts": 3, "schedule": "records", "chain": 0.0,
    "train_block": 0, "eval_block": 0, "ab_tuned_name": "mtp_tuned.safetensors", "expert_lr": 0.0,
    "snap_dir": "/dev/shm/mtp_snap", "snap_gib": 60.0, "snap_margin_gib": 14.0, "snap_extract_minutes": 20.0,
    "snap_parity": 10, "snap_arms": "dense,experts", "snap_minutes": 20.0, "snap_epochs": 50,
    "snap_save_gib": 0.0, "snap_from": "", "snap_train_bits": 8, "snap_select_every": 0,
    "snap_select_records": 40, "sampled_taus": [], "train_probe_records": 0, "kv_quant_views": "",
    "kv_only": False, "snap_ref": "", "snap_heldout": "", "focus": "", "focus_at": 0.95, "focus_keep": 0.0,
    "lm_head_ms": 0.0, "step_ms": 87.1, "dflash_calls": 1000, "dflash_block": 8, "dflash_exclude": "",
    "dflash_bits": 8, "dflash_window": 2048, "dflash_minutes": 32.0, "dflash_lr": 3e-5, "dflash_warmup": 50,
    "dflash_anchors": 128, "dflash_steps": 4, "dflash_drafter": "arc3-dflash2-incoai", "dflash_eval_reserve": 8.0,
    "dflash_mtp": False, "dflash_mtp_dir": "/tmp/dflash_mtp", "dflash_mtp_share": 0.5, "dflash_mtp_gib": 8.0,
    "dflash_mtp_free_gib": 25.0, "dflash_mtp_minutes": 20.0, "dflash_mtp_lr": 2e-5, "dflash_mtp_kl": 0.05,
    "dflash_mtp_eval_records": 120, "dflash_mtp_eval_reserve": 12.0, "dflash_mtp_select_share": 0.1,
    "dflash_mtp_ram_gib": 150.0, "dflash_target": "ce", "dflash_kl": 0.05, "dflash_select_minutes": 0.0,
    "dflash_select_records": 24, "dflash_select_share": 0.25, "dflash_eval_records": 120, "snap_free_gib": 0.0,
    "snap_floor_gib": 1.0, "dflash_patience": 0, "dflash_patience_gain": 0.001, "dflash_features": True,
    "dflash_phases": ""}
CGROUP = Path("/sys/fs/cgroup")
SKIP_PLE_ROWS = "TAAF_VLLM_PLE_Q6_SKIP_CHECKPOINT_ROWS"
PLE_MARKS = (".ple.", "ngram_embedding")
KNOB = "ARC_MTP_TUNE_"
TARGETS = ("game", "argmax", "tv")
MODES = ("eval", "train", "ab", "snap", "dflash")
SNAP_ARMS = ("dense", "experts")
AB_HEADS = ("albucino", "intel", "tuned", "tuned-int4", "tuned-k400")
SPANS = ("reply", "assistant")
KV_VIEWS = {"bf16": "", "int4_per_token_head": "int4_per_token_head", "fp8_index": "fp8_index"}
SCHEDULES = ("records", "time")
DFLASH_PHASES = ("extract", "train", "eval", "mtp_snap", "mtp_train", "mtp_eval")
FOCUS = ("", "target", "verdict", "drafter")
VOCAB_VIEWS = ("fp8row", "fp8tensor", "fp8g", "top")
INPUTS = Path("/kaggle/input")
SERVE_VOCAB = "draft_vocab.json"
VOCAB_FILE_ENV = "TAAF_VLLM_DRAFT_VOCAB_FILE"


def say(text: str) -> None:
    print(f"mtp_tune: {text}", flush=True)


def settings(environ: dict[str, str]) -> dict[str, Any]:
    out = dict(DEFAULTS)
    for name, default in DEFAULTS.items():
        said = environ.get(KNOB + name.upper())
        if said is None or not said.strip():
            continue
        if isinstance(default, bool):
            out[name] = said.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            out[name] = int(said)
        elif isinstance(default, float):
            out[name] = float(said)
        elif isinstance(default, list):
            out[name] = [float(one) for one in said.replace(";", ",").split(",") if one.strip()]
        else:
            out[name] = said.strip()
    out["snap_arms"] = "" if out["snap_arms"].strip().lower() == "none" else out["snap_arms"].replace(";", ",")
    views = [one.strip() for one in out["kv_quant_views"].replace(";", ",").split(",") if one.strip()]
    if any(part not in KV_VIEWS for one in views for part in one.split("+")):
        raise SystemExit(f"{KNOB}KV_QUANT_VIEWS takes some of {sorted(KV_VIEWS)}, joined by '+' within a view, found "
                         f"{out['kv_quant_views']!r}")
    out["kv_quant_views"] = ["+".join(KV_VIEWS[part] for part in one.split("+") if KV_VIEWS[part]) for one in views]
    if out["mode"] not in MODES:
        raise SystemExit(f"{KNOB}MODE must be one of {MODES}, found {out['mode']!r}")
    if out["focus"] not in FOCUS or not 0.0 < out["focus_at"] <= 1.0 or not 0.0 <= out["focus_keep"] <= 1.0:
        raise SystemExit(f"{KNOB}FOCUS takes one of {FOCUS} with FOCUS_AT in (0, 1] and FOCUS_KEEP in [0, 1], "
                         f"found {out['focus']!r} at {out['focus_at']} keep {out['focus_keep']}")
    for name in [one.strip() for one in out["ab_vocabs"].split(",") if one.strip()]:
        view = name.partition("+")[2]
        if view and not (view in ("fp8row", "fp8tensor") or (view.startswith("fp8g") and view[4:].isdigit())
                         or (view.startswith("top") and view[3:].isdigit())):
            raise SystemExit(f"{KNOB}AB_VOCABS view {name!r}: after '+' it takes fp8row, fp8tensor, fp8g<group> or "
                             f"top<count>")
    heads = [one.strip() for one in out["ab_heads"].split(",") if one.strip()]
    unknown = [one for one in heads if one not in AB_HEADS]
    if out["mode"] in ("ab", "snap") and (unknown or not heads):
        raise SystemExit(f"{KNOB}AB_HEADS names {unknown or 'nothing'}; it takes some of {AB_HEADS}")
    if out["mode"] in ("ab", "snap") and any(one.startswith("tuned") for one in heads) and not out["tuned"]:
        raise SystemExit(f"{KNOB}AB_HEADS asks for the tuned head but {KNOB}TUNED names no file")
    if out["mode"] == "snap":
        arms = [one.strip() for one in out["snap_arms"].split(",") if one.strip()]
        kinds = [one.partition("@")[0] for one in arms]
        rates = [one.partition("@")[2] for one in arms]
        try:
            bad_rate = any(rate and float(rate) <= 0 for rate in rates)
        except ValueError:
            bad_rate = True
        if (not arms and not out["snap_from"]) or any(one not in SNAP_ARMS for one in kinds) or len(set(arms)) != len(arms) or bad_rate:
            raise SystemExit(f"{KNOB}SNAP_ARMS takes arms of {SNAP_ARMS}, each once, with an optional positive "
                             f"@learning-rate, found {out['snap_arms']!r}")
        if "albucino" not in heads or any(one not in ("albucino", "tuned-int4") for one in heads):
            raise SystemExit(f"{KNOB}AB_HEADS in snap mode is albucino, optionally with tuned-int4: the arms train "
                             f"on albucino's experts as served")
        if out["eval_temperatures"]:
            raise SystemExit(f"{KNOB}EVAL_TEMPERATURES must be empty in snap mode: the snapshot stores the target "
                             f"distribution at one temperature")
        if out["snap_gib"] < 8 or out["snap_minutes"] <= 0 or out["snap_extract_minutes"] <= 0                 or out["snap_parity"] < 0 or out["snap_epochs"] < 1 or out["snap_margin_gib"] < 0:
            raise SystemExit(f"{KNOB}SNAP_GIB is at least 8, the SNAP minutes positive, SNAP_EPOCHS at least 1 and "
                             f"SNAP_PARITY and SNAP_MARGIN_GIB not negative")
        if out["snap_train_bits"] not in (4, 8) or out["snap_select_every"] < 0 or out["snap_select_records"] < 1:
            raise SystemExit(f"{KNOB}SNAP_TRAIN_BITS is 8 or 4, SNAP_SELECT_EVERY not negative and "
                             f"SNAP_SELECT_RECORDS at least 1")
        if not out["snap_dir"].startswith("/") or out["snap_dir"].rstrip("/") in ("", "/dev/shm", "/tmp"):
            raise SystemExit(f"{KNOB}SNAP_DIR is an absolute folder of its own (it is emptied first), found "
                             f"{out['snap_dir']!r}")
    if out["mode"] == "dflash":
        if out["dflash_calls"] < 1 or out["dflash_block"] < 1 or out["dflash_bits"] not in (4, 8)                 or out["dflash_window"] < 64 or not 1 <= out["dflash_steps"] <= 7 or out["dflash_minutes"] <= 0                 or out["dflash_lr"] <= 0 or out["dflash_anchors"] < 1 or out["dflash_eval_reserve"] < 0:
            raise SystemExit(f"{KNOB}DFLASH_CALLS and DFLASH_BLOCK are at least 1, DFLASH_BITS 4 or 8, DFLASH_WINDOW at "
                             f"least 64, DFLASH_STEPS 1..7, DFLASH_MINUTES and DFLASH_LR positive, DFLASH_ANCHORS at "
                             f"least 1 and DFLASH_EVAL_RESERVE not negative")
        if not out["snap_dir"].startswith("/") or out["snap_dir"].rstrip("/") in ("", "/dev/shm", "/tmp"):
            raise SystemExit(f"{KNOB}SNAP_DIR is an absolute folder of its own (it is emptied first), found "
                             f"{out['snap_dir']!r}")
        if out["dflash_target"] not in ("ce", "tv") or out["dflash_kl"] < 0 or out["dflash_select_minutes"] < 0 \
                or out["dflash_select_records"] < 1 or not 0 < out["dflash_select_share"] < 1 \
                or out["dflash_eval_records"] < 1 or out["snap_free_gib"] < 0 or out["snap_floor_gib"] < 0 \
                or out["dflash_patience"] < 0 or out["dflash_patience_gain"] < 0 \
                or (out["snap_free_gib"] and out["snap_free_gib"] <= out["snap_floor_gib"]):
            raise SystemExit(f"{KNOB}DFLASH_TARGET is ce or tv, DFLASH_KL, DFLASH_SELECT_MINUTES and SNAP_FREE_GIB are not "
                             f"negative, SNAP_FREE_GIB above SNAP_FLOOR_GIB, DFLASH_SELECT_RECORDS and DFLASH_EVAL_RECORDS "
                             f"at least 1, DFLASH_SELECT_SHARE in (0, 1)")
        if out["dflash_mtp"] and (not out["dflash_mtp_dir"].startswith("/")
                                  or out["dflash_mtp_dir"].rstrip("/") in ("", "/dev/shm", "/tmp")
                                  or not 0 < out["dflash_mtp_share"] <= 1 or out["dflash_mtp_gib"] <= 0
                                  or out["dflash_mtp_minutes"] <= 0 or out["dflash_mtp_lr"] <= 0
                                  or out["dflash_mtp_eval_records"] < 1 or not 0 < out["dflash_mtp_select_share"] < 1
                                  or out["dflash_mtp_free_gib"] < out["dflash_mtp_gib"]
                                  or out["dflash_mtp_ram_gib"] <= 0):
            raise SystemExit(f"{KNOB}DFLASH_MTP_DIR is an absolute folder of its own, DFLASH_MTP_SHARE in (0, 1], "
                             f"DFLASH_MTP_GIB, DFLASH_MTP_MINUTES, DFLASH_MTP_LR and DFLASH_MTP_RAM_GIB positive, "
                             f"DFLASH_MTP_FREE_GIB at least DFLASH_MTP_GIB, DFLASH_MTP_EVAL_RECORDS at least 1 and "
                             f"DFLASH_MTP_SELECT_SHARE in (0, 1)")
        named = [one.strip() for one in out["dflash_phases"].split(",") if one.strip()]
        if any(one not in DFLASH_PHASES for one in named) or (named and named[0] != "extract") \
                or (not out["dflash_features"] and (not named or {"train", "eval"} & set(named)
                                                    or not out["dflash_mtp"])):
            raise SystemExit(f"{KNOB}DFLASH_PHASES names some of {DFLASH_PHASES} starting with extract; with "
                             f"DFLASH_FEATURES off the DFlash rows are not written, so DFLASH_MTP must be on and "
                             f"the phases must be named without train and eval, found {out['dflash_phases']!r} "
                             f"(features {out['dflash_features']}, MTP {out['dflash_mtp']})")
    if out["expert_lr"] < 0:
        raise SystemExit(f"{KNOB}EXPERT_LR must not be negative")
    if out["ab_tune_from"] not in ("best", *AB_HEADS) or out["ab_tune_from"] == "tuned-k400":
        raise SystemExit(f"{KNOB}AB_TUNE_FROM is best or a head served with every expert, "
                         f"found {out['ab_tune_from']!r}")
    if out["target"] not in TARGETS:
        raise SystemExit(f"{KNOB}TARGET must be one of {TARGETS}, found {out['target']!r}")
    if out["span"] not in SPANS:
        raise SystemExit(f"{KNOB}SPAN must be one of {SPANS}, found {out['span']!r}")
    if out["kl"] and out["target"] not in ("argmax", "tv"):
        raise SystemExit(f"{KNOB}KL weighs a KL term beside the argmax or tv target; the {out['target']} target "
                         f"is already the target's distribution")
    if any(float(one) <= 0 for one in out["sampled_taus"]):
        raise SystemExit(f"{KNOB}SAMPLED_TAUS are positive multipliers of TEMPERATURE, found {out['sampled_taus']!r}")
    if out["schedule"] not in SCHEDULES:
        raise SystemExit(f"{KNOB}SCHEDULE must be one of {SCHEDULES}, found {out['schedule']!r}")
    if out["chain"] < 0 or out["train_block"] < 0 or out["eval_block"] < 0:
        raise SystemExit(f"{KNOB}CHAIN, TRAIN_BLOCK and EVAL_BLOCK must not be negative")
    if not out["ab_tuned_name"].endswith(".safetensors") or "/" in out["ab_tuned_name"]:
        raise SystemExit(f"{KNOB}AB_TUNED_NAME is a bare .safetensors file name, found {out['ab_tuned_name']!r}")
    if out["kl"] < 0 or out["late"] < 0 or out["grad_context"] < 0 or out["temperature"] <= 0:
        raise SystemExit(f"{KNOB}KL, LATE and GRAD_CONTEXT must not be negative and TEMPERATURE must be positive")
    return out


def input_file(said: str, search: Path = INPUTS) -> Path:
    path = Path(said)
    if path.is_absolute():
        if not path.is_file():
            raise SystemExit(f"{said} is not a file")
        return path
    found = sorted(search.rglob(said)) if search.is_dir() else []
    if len(found) != 1:
        raise SystemExit(f"{said} matches {len(found)} file(s) under {search}, not one")
    return found[0]


def vocabularies(env: dict[str, str], bundle: Path, said: str, search: Path = INPUTS) -> dict[str, str]:
    serve = ""
    if env.get("TAAF_VLLM_DRAFT_VOCAB") == "1":
        named = str(env.get(VOCAB_FILE_ENV) or "").strip()
        held = bundle / (named or SERVE_VOCAB)
        if named and not held.is_file():
            held = input_file(named, search)
        serve = str(held) if held.is_file() else ""
    if not said:
        return {"draft_vocab": serve, "serve_vocab": "", "vocab_file": ""}
    chosen = str(bundle / said) if not Path(said).is_absolute() and (bundle / said).is_file() \
        else str(input_file(said, search))
    return {"draft_vocab": chosen, "serve_vocab": serve if serve and serve != chosen else "", "vocab_file": chosen}


def serve_words(argv: list[str]) -> tuple[str, list[str]]:
    at = argv.index("serve")
    return argv[at + 1], argv[at + 2:]


def held_server(working: Path) -> tuple[dict[str, Any], bool]:
    prepared = working / PREPARED
    if prepared.is_file():
        return json.loads(prepared.read_text(encoding="utf-8")), True
    return json.loads((working / IDENTITY).read_text(encoding="utf-8")), False


def speculative_model(words: list[str]) -> str:
    if "--speculative-config" not in words:
        return ""
    return str(json.loads(words[words.index("--speculative-config") + 1]).get("model") or "")


def ab_sources(env: dict[str, str], bundle: Path, cfg: dict[str, Any], search: Path = INPUTS,
               words: list[str] = ()) -> dict[str, Any]:
    vocabs = {}
    for name in [one.strip() for one in cfg["ab_vocabs"].split(",") if one.strip()]:
        plain = name.partition("+")[0]
        held = bundle / plain
        vocabs[name] = str(held if held.is_file() else input_file(plain, search))
    return {"draft_model": speculative_model(list(words)) or str(env.get("TAAF_VLLM_DRAFT_MODEL") or ""),
            "tuned_file": str(input_file(cfg["tuned"], search)) if cfg["tuned"] else "",
            "saliency": str(bundle / SALIENCY) if (bundle / SALIENCY).is_file() else "",
            "ab_vocab_files": vocabs}


def snapshot_folder(said: str, search: Path = INPUTS) -> Path:
    path = Path(said)
    if path.is_absolute():
        if not (path / "snapshot.json").is_file():
            raise SystemExit(f"{said} holds no snapshot.json")
        return path
    found = sorted(one.parent for one in search.rglob("snapshot.json") if one.parent.name == said) \
        if search.is_dir() else []
    if not found:
        unpacked = unpack_folder(said, search)
        found = [unpacked] if unpacked is not None else []
    if len(found) != 1:
        raise SystemExit(f"{len(found)} snapshot folder(s) named {said} under {search}, not one")
    return found[0]


UNPACK_ROOT = Path("/kaggle/tmp/snap_unpacked")


def unpack_folder(said: str, search: Path, root: Path | None = None) -> Path | None:
    import tarfile
    root = UNPACK_ROOT if root is None else root
    archives = sorted(search.rglob("outputs.tar.gz")) + sorted(search.rglob("outputs.tar")) if search.is_dir() else []
    for archive in archives:
        began = time.monotonic()
        target = root / said
        names = []
        with tarfile.open(archive, "r|*") as held:
            for member in held:
                if not member.isfile() or Path(member.name).parent.name != said:
                    continue
                target.mkdir(parents=True, exist_ok=True)
                source = held.extractfile(member)
                with open(target / Path(member.name).name, "wb") as sink:
                    while True:
                        chunk = source.read(1 << 24)
                        if not chunk:
                            break
                        sink.write(chunk)
                names.append(Path(member.name).name)
        if "snapshot.json" not in names:
            continue
        say(f"unpacked {len(names)} file(s) of {said} from {archive} to {target} in "
            f"{time.monotonic() - began:.0f} s")
        return target
    return None


def from_capture(folder: Path, cfg: dict[str, Any], out: Path) -> Path:
    import importlib.util
    path = next((one for one in (HERE / "capture_snapshot.py", HERE / "mtp" / "capture_snapshot.py") if one.is_file()),
                None)
    if path is None:
        return folder
    spec = importlib.util.spec_from_file_location("capture_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.is_capture(folder):
        return folder
    if not cfg.get("snap_ref"):
        raise SystemExit(f"{folder} is a decode-path capture; {KNOB}SNAP_REF must name the snapshot whose lm_head, "
                         f"embedding and config it borrows")
    reference = snapshot_folder(cfg["snap_ref"])
    target = out / "snap_capture"
    named = [one.strip() for one in str(cfg.get("snap_heldout") or "").replace(";", ",").split(",") if one.strip()]
    said = module.convert(folder, reference, target, named, max_heldout=int(cfg.get("snap_select_records") or 40) + 80)
    say(f"decode-path capture {folder} -> {target} (lm_head, embedding and config from {reference}): " + json.dumps(said))
    return target


def read_env(pid: int) -> dict[str, str]:
    raw = (PROC / str(pid) / "environ").read_bytes()
    out: dict[str, str] = {}
    for chunk in raw.split(b"\x00"):
        if b"=" in chunk:
            name, _, value = chunk.partition(b"=")
            out[name.decode("utf-8", "ignore")] = value.decode("utf-8", "ignore")
    return out


def find_corpus(root: Path) -> Path:
    found = sorted(path.parent for path in root.rglob(CORPUS_MARK))
    if len(found) != 1:
        raise SystemExit(f"{len(found)} {CORPUS_MARK} under {root}, not one: {[str(one) for one in found][:4]}")
    return found[0]


def children(pid: int) -> set[int]:
    out: set[int] = set()
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == pid:
            out.add(int(entry.name))
    return out


def stop(pid: int, what: str = "the server") -> None:
    family = {pid}
    frontier = {pid}
    while frontier:
        frontier = set().union(*(children(one) for one in frontier)) - family
        family |= frontier
    for one in family:
        try:
            os.kill(one, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and any((PROC / str(one)).exists() for one in family):
        time.sleep(0.5)
    for one in family:
        try:
            os.kill(one, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(3)
    say(f"stopped {what} and {len(family) - 1} child process(es) so the GPU is free")


def read_through(path: Path) -> int:
    done = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            piece = handle.read(WARM_PIECE)
            if not piece:
                return done
            done += len(piece)


def shards(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix == ".safetensors")


def open_index(pid: int, order: dict[str, int], proc: Path = PROC) -> int:
    best = -1
    try:
        entries = list((proc / str(pid) / "fd").iterdir())
    except OSError:
        entries = []
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        best = max(best, order.get(target, -1))
    try:
        mapped = (proc / str(pid) / "maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return best
    for line in mapped.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            best = max(best, order.get(parts[5].strip(), -1))
    return best


def follow(child: subprocess.Popen, folder: Path, ahead: int = FOLLOW_AHEAD,
           threads: int = FOLLOW_THREADS, proc: Path = PROC) -> dict[str, Any]:
    files = shards(folder)
    order = {str(path.resolve()): index for index, path in enumerate(files)}
    order.update({str(path): index for index, path in enumerate(files)})
    issued: set[int] = set()
    futures = []
    began = time.monotonic()
    furthest = -1
    with ThreadPoolExecutor(max_workers=threads) as pool:
        while child.poll() is None and len(issued) < len(files):
            furthest = max(furthest, open_index(child.pid, order, proc))
            for index in range(furthest + 1, min(len(files), furthest + 1 + ahead)):
                if index not in issued:
                    issued.add(index)
                    futures.append(pool.submit(read_through, files[index]))
            time.sleep(FOLLOW_POLL_S)
        total = sum(future.result() for future in futures if not future.cancelled())
    return {"files": len(files), "read": len(issued), "bytes": total, "furthest": furthest,
            "seconds": round(time.monotonic() - began, 1)}


def ram_free_gib(base: Path = CGROUP) -> float | None:
    try:
        ceiling = (base / "memory.max").read_text(encoding="utf-8").strip()
        stat = dict(line.split() for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines()
                    if len(line.split()) == 2)
    except (OSError, ValueError):
        return None
    if not ceiling.isdigit():
        return None
    return (int(ceiling) - int(stat.get("anon", 0)) - int(stat.get("shmem", 0))) / 2**30


def ple_bytes(model_dir: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with open(path, "rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(size))
        held = sum(int(one["data_offsets"][1]) - int(one["data_offsets"][0]) for name, one in header.items()
                   if name != "__metadata__" and any(mark in name for mark in PLE_MARKS))
        if held:
            found[path.name] = held
    return found


def allocated(path: Path) -> int | None:
    blocks = getattr(os.stat(path), "st_blocks", None)
    return None if blocks is None else int(blocks) * 512


def ple_not_stubbed(model_dir: Path, env: dict[str, str]) -> None:
    if str(env.get(SKIP_PLE_ROWS) or "").strip() == "1":
        say(f"PLE_STUB ok: {SKIP_PLE_ROWS}=1, the serve takes the PLE table from its int6 parts, not the checkpoint")
        return
    files = ple_bytes(model_dir)
    stubs, unknown = [], []
    for name, held in files.items():
        path = Path(model_dir) / name
        have = allocated(path)
        if have is None:
            unknown.append(name)
        elif have < path.stat().st_size - held / 2:
            stubs.append(f"{name} {have / 2**30:.1f} of {path.stat().st_size / 2**30:.1f} GiB on disk")
    if stubs:
        raise SystemExit(f"PLE_STUB FAIL {len(stubs)} of {len(files)} checkpoint files that hold PLE rows are sparse "
                         f"stubs ({'; '.join(stubs[:3])}) and the serve reads PLE rows from the checkpoint "
                         f"({SKIP_PLE_ROWS} unset): the features would be built on zeros; nothing loaded")
    say(f"PLE_STUB ok: {len(files) - len(unknown)} checkpoint files with PLE rows are fully on disk"
        + (f", {len(unknown)} could not be checked" if unknown else ""))


def room_for_snapshot(cfg: dict[str, Any]) -> None:
    folder = Path(cfg["snap_dir"])
    probe = next(one for one in (folder, *folder.parents) if one.exists())
    free = shutil.disk_usage(probe).free / 2**30
    if free < float(cfg["snap_free_gib"]):
        raise SystemExit(f"SNAP_ROOM FAIL {probe} has {free:.1f} GiB free, below the {cfg['snap_free_gib']:g} GiB the "
                         f"DFlash snapshot needs ({cfg['snap_gib']:g} GiB budget); nothing loaded")
    room = min(float(cfg["snap_gib"]), free - float(cfg["snap_floor_gib"]))
    say(f"SNAP_ROOM ok: {probe} has {free:.1f} GiB free; the snapshot takes at most {cfg['snap_gib']:g} GiB and stops "
        f"with {cfg['snap_floor_gib']:g} GiB left, so about {room:.0f} GiB if nothing else on the disk changes")


def room_for_mtp(cfg: dict[str, Any], base: Path = CGROUP) -> None:
    folder = Path(cfg["dflash_mtp_dir"])
    probe = next(one for one in (folder, *folder.parents) if one.exists())
    free = shutil.disk_usage(probe).free / 2**30
    if free < float(cfg["dflash_mtp_free_gib"]):
        raise SystemExit(f"MTP_ROOM FAIL {probe} has {free:.1f} GiB free, below the {cfg['dflash_mtp_free_gib']:g} GiB "
                         f"the MTP head's input rows need ({cfg['dflash_mtp_gib']:g} GiB budget); nothing loaded")
    ram = ram_free_gib(base)
    if ram is not None and ram < float(cfg["dflash_mtp_ram_gib"]):
        raise SystemExit(f"MTP_ROOM FAIL the container has {ram:.1f} GiB of RAM free (memory.max less anon and shmem), "
                         f"below the {cfg['dflash_mtp_ram_gib']:g} GiB the target, the DFlash snapshot and the MTP rows "
                         f"need; nothing loaded")
    say(f"MTP_ROOM ok: {probe} has {free:.1f} GiB free for at most {cfg['dflash_mtp_gib']:g} GiB of MTP rows; RAM "
        + (f"{ram:.1f} GiB free" if ram is not None else "unknown (no cgroup memory files)"))


def main() -> int:
    started = time.time()
    cfg = settings(dict(os.environ))
    if cfg["mode"] == "dflash" and cfg["snap_free_gib"] > 0:
        room_for_snapshot(cfg)
    if cfg["mode"] == "dflash" and cfg["dflash_mtp"]:
        room_for_mtp(cfg)
    missing = [name for name in needed(cfg["mode"], bool(cfg.get("dflash_mtp"))) if not (HERE / name).is_file()]
    if missing:
        raise SystemExit(f"the bundle carries no {missing}; pass carried/mtp/{missing[0]} with --carry")
    working = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working")
    server, prepared = held_server(working)
    argv = [str(one) for one in server["argv"]]
    env = read_env(int(server["pid"]))
    if not env.get("PYTHONPATH"):
        raise SystemExit("the server's environment could not be read from /proc; nothing ran")
    model_dir, words = serve_words(argv)
    if cfg["mode"] == "dflash":
        ple_not_stubbed(Path(model_dir), env)
    corpus = find_corpus(Path(os.environ.get("ARC_MTP_TUNE_CORPUS_ROOT") or "/kaggle/input"))
    bundle = Path(env.get("TAAF_KAGGLE_BUNDLE_DIR") or "")
    out = working / "mtp_tune"
    cfg.update({"model_dir": model_dir, "serve_args": words, "corpus": str(corpus), "out": str(out),
                **vocabularies(env, bundle, cfg["vocab"]), "model_repo": env.get("TAAF_MODEL_REPO", "")})
    if cfg["mode"] in ("ab", "snap") or (cfg["mode"] == "dflash" and cfg["dflash_mtp"]):
        cfg.update(ab_sources(env, bundle, cfg, words=words))
    if cfg["mode"] == "dflash":
        cfg.update({"hard_minutes": float(os.environ.get(HARD_KNOB) or 0.0), "driver_started": started})
    if cfg["mode"] == "snap" and cfg["snap_from"]:
        cfg["snap_from"] = str(snapshot_folder(cfg["snap_from"]))
        cfg["snap_from"] = str(from_capture(Path(cfg["snap_from"]), cfg, out))
        cfg.update({"hard_minutes": float(os.environ.get(HARD_KNOB) or 0.0), "driver_started": started})
    if prepared:
        say(f"the setup prepared the serve and started no server (pid {server['pid']} only holds its "
            f"environment); the worker loads the weights once, in-process, with the flags the server would have run")
    else:
        endpoint = os.environ.get("LOCAL_ANALYZER_BASE_URL") or "unset"
        say(f"the run's endpoint {endpoint} goes dark: its server is stopped and the worker loads the same "
            f"weights in-process with the server's own flags")
    say(f"mode {cfg['mode']}; model {model_dir}; corpus {corpus}; draft vocab "
        f"{cfg['draft_vocab'] or 'off (full vocabulary)'}; settings "
        + json.dumps({k: v for k, v in cfg.items() if k not in ('serve_args',)}))
    stop(int(server["pid"]), "the environment holder" if prepared else "the server")
    child_env = {**env, "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "PYTHONUNBUFFERED": "1"}
    began = time.monotonic()
    worker = DFLASH_WORKER if cfg["mode"] == "dflash" else AB_WORKER if cfg["mode"] in ("ab", "snap") else WORKER
    phases = (["train"] if cfg["snap_from"] else ["extract", "train"]) if cfg["mode"] == "snap" else         (["extract", "train", "mtp_train", "eval", "mtp_eval"] if cfg["dflash_mtp"] else ["extract", "train", "eval"]) \
        if cfg["mode"] == "dflash" else [""]
    if cfg["mode"] == "dflash" and cfg["dflash_phases"]:
        phases = [one.strip() for one in cfg["dflash_phases"].split(",") if one.strip()]
    code = 0
    for phase in phases:
        sent = {**cfg, "phase": phase} if phase else cfg
        child = subprocess.Popen([argv[0], str(worker), json.dumps(sent)], env=child_env, cwd=str(working))
        if phase in ("", "extract"):
            try:
                followed = follow(child, Path(model_dir))
                say(f"read {followed['read']} of the model's {followed['files']} shards "
                    f"({followed['bytes'] / 2**30:.1f} GiB) {FOLLOW_AHEAD} ahead of the worker's loader with "
                    f"{FOLLOW_THREADS} threads in {followed['seconds']:.0f} s")
            except Exception as failure:
                say(f"the read-ahead stopped ({type(failure).__name__}: {failure}); the worker loads on its own")
        done = types.SimpleNamespace(returncode=child.wait())
        code = done.returncode
        say(f"worker{' phase ' + phase if phase else ''} exited {code} after "
            f"{(time.monotonic() - began) / 60:.1f} min; outputs under {out}")
        if code and cfg["mode"] == "dflash" and phase in ("train", "mtp_train"):
            say(f"phase {phase} failed; the evaluation phases still run")
            continue
        if code and cfg["mode"] == "dflash" and phase == "extract" and kept_rows(Path(cfg["snap_dir"])):
            say(f"the extract phase failed after writing {kept_rows(Path(cfg['snap_dir'])):,} rows; the train phase "
                f"runs on them")
            continue
        if code:
            break
    return code


def kept_rows(folder: Path) -> int:
    try:
        return int(json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))["counts"]["rows"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
