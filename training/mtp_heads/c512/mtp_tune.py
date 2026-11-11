from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKER = HERE / "mtp_tune_worker.py"
NEEDED = ("mtp_ref.py", "mtp_tune_worker.py")


def companions(env: dict[str, str]) -> list[str]:
    return list(NEEDED)


IDENTITY = "vllm-server-identity.json"
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
    "eval_temperatures": [], "grad_context": 0, "span": "reply", "late": 0.0, "vocab": ""}
KNOB = "ARC_MTP_TUNE_"
TARGETS = ("game", "argmax")
SPANS = ("reply", "assistant")
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
            out[name] = [float(one) for one in said.split(",") if one.strip()]
        else:
            out[name] = said.strip()
    if out["mode"] not in ("eval", "train"):
        raise SystemExit(f"{KNOB}MODE must be eval or train, found {out['mode']!r}")
    if out["target"] not in TARGETS:
        raise SystemExit(f"{KNOB}TARGET must be one of {TARGETS}, found {out['target']!r}")
    if out["span"] not in SPANS:
        raise SystemExit(f"{KNOB}SPAN must be one of {SPANS}, found {out['span']!r}")
    if out["kl"] and out["target"] != "argmax":
        raise SystemExit(f"{KNOB}KL weighs a KL term beside the argmax target; the {out['target']} target is "
                         f"already the target's distribution")
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


def stop(pid: int) -> None:
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
    say(f"stopped the server and {len(family) - 1} child process(es) so the GPU is free")


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


def main() -> int:
    missing = [name for name in NEEDED if not (HERE / name).is_file()]
    if missing:
        raise SystemExit(f"the bundle carries no {missing}; pass carried/mtp/{missing[0]} with --carry")
    working = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working")
    server = json.loads((working / IDENTITY).read_text(encoding="utf-8"))
    argv = [str(one) for one in server["argv"]]
    env = read_env(int(server["pid"]))
    if not env.get("PYTHONPATH"):
        raise SystemExit("the server's environment could not be read from /proc; nothing ran")
    model_dir, words = serve_words(argv)
    cfg = settings(dict(os.environ))
    corpus = find_corpus(Path(os.environ.get("ARC_MTP_TUNE_CORPUS_ROOT") or "/kaggle/input"))
    bundle = Path(env.get("TAAF_KAGGLE_BUNDLE_DIR") or "")
    out = working / "mtp_tune"
    cfg.update({"model_dir": model_dir, "serve_args": words, "corpus": str(corpus), "out": str(out),
                **vocabularies(env, bundle, cfg["vocab"]), "model_repo": env.get("TAAF_MODEL_REPO", "")})
    endpoint = os.environ.get("LOCAL_ANALYZER_BASE_URL") or "unset"
    say(f"the run's endpoint {endpoint} goes dark: its server is stopped and the worker loads the same "
        f"weights in-process with the server's own flags")
    say(f"mode {cfg['mode']}; model {model_dir}; corpus {corpus}; draft vocab "
        f"{cfg['draft_vocab'] or 'off (full vocabulary)'}; settings "
        + json.dumps({k: v for k, v in cfg.items() if k not in ('serve_args',)}))
    stop(int(server["pid"]))
    child_env = {**env, "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "PYTHONUNBUFFERED": "1"}
    began = time.monotonic()
    child = subprocess.Popen([argv[0], str(WORKER), json.dumps(cfg)], env=child_env, cwd=str(working))
    try:
        followed = follow(child, Path(model_dir))
        say(f"read {followed['read']} of the model's {followed['files']} shards ({followed['bytes'] / 2**30:.1f} GiB) "
            f"{FOLLOW_AHEAD} ahead of the worker's loader with {FOLLOW_THREADS} threads in {followed['seconds']:.0f} s")
    except Exception as failure:
        say(f"the read-ahead stopped ({type(failure).__name__}: {failure}); the worker loads on its own")
    done = types.SimpleNamespace(returncode=child.wait())
    say(f"worker exited {done.returncode} after {(time.monotonic() - began) / 60:.1f} min; "
        f"outputs under {out}")
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
